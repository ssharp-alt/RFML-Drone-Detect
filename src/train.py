"""
Streaming training entry point for the drone-RF classifier. Two mutually
exclusive strategies for never keeping the full dataset locally:

  --cache-mode shard (default for --source zip): the 70% train split is
    divided into --num-train-shards stratified shards, cached to an
    ephemeral local directory and visited in a boustrophedon (ping-pong)
    order across epochs so the shard downloaded last in one epoch is
    exactly the one needed first in the next (skipping that re-download),
    with the *following* shard prefetched in the background while the
    current one trains. The whole cache directory is deleted when the run
    ends (see ShardCache). Worth the bookkeeping when the source is slow/
    metered relative to local disk headroom (e.g. the open internet).

  --cache-mode streaming (default for --source ftp): no local disk cache
    at all. Every __getitem__ fetches that one sample's bytes straight
    from the source into memory, decodes it, and discards the bytes (see
    StreamingDataset). Simpler and has zero disk footprint, at the cost of
    re-fetching the full train split every epoch -- the right trade when
    the source is fast and local disk is scarce (e.g. a LAN FTP mirror).

Usage:
    .venv/Scripts/python.exe -m src.train --source zip --epochs 10 --num-train-shards 4
    .venv/Scripts/python.exe -m src.train --source ftp --epochs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import config
from .dataset import LocalShardDataset
from .ftp_source import FTPIndex
from .model import DroneRFClassifier
from .remote_zip import RemoteZipIndex, SampleEntry
from .shard_cache import ShardCache, boustrophedon_order
from .streaming_dataset import StreamingDataset

log = logging.getLogger("train")

QUICK_EVAL_SHARD_ID = "quick_eval"
QUICK_EVAL_SAMPLES_PER_CLASS = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source", choices=["zip", "ftp"], default="zip",
        help="'zip': remote Kaggle/GCS zip via HTTP range requests (config.get_zip_url()). "
             "'ftp': individual files over FTP (config.get_ftp_config()).",
    )
    p.add_argument(
        "--cache-mode", choices=["auto", "shard", "streaming"], default="auto",
        help="'auto' (default): shard for --source zip, streaming for --source ftp. "
             "See the module docstring for the trade-off.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-train-shards", type=int, default=config.DEFAULT_NUM_TRAIN_SHARDS,
                    help="shard mode only")
    p.add_argument("--num-test-shards", type=int, default=config.DEFAULT_NUM_TEST_SHARDS,
                    help="shard mode only")
    p.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker processes. In streaming mode against --source ftp, this is also "
             "the number of concurrent FTP connections held for the whole run -- stay under your "
             "server's per-user connection cap (the reference LAN mirror caps at 5).",
    )
    p.add_argument(
        "--download-threads", type=int, default=None,
        help="shard mode only. default: config.DOWNLOAD_THREADS_PER_SHARD for --source zip, "
             "config.FTP_DOWNLOAD_THREADS for --source ftp (FTP servers often cap "
             "concurrent connections per user)",
    )
    p.add_argument("--cache-dir", type=Path, default=config.DEFAULT_CACHE_DIR, help="shard mode only")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--eval-every", type=int, default=1, help="epochs between quick-eval passes")
    p.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    p.add_argument("--no-prefetch", action="store_true",
                    help="shard mode only: disable background next-shard prefetch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def class_weights(train_entries: list[SampleEntry], num_classes: int) -> torch.Tensor:
    counts = Counter(e.target for e in train_entries)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def quick_eval_subset(test_entries: list[SampleEntry], seed: int) -> list[SampleEntry]:
    import random

    by_class: dict[int, list[SampleEntry]] = {}
    for e in test_entries:
        by_class.setdefault(e.target, []).append(e)
    subset = []
    for target, group in by_class.items():
        rng = random.Random(f"{seed}-quick-eval-{target}")
        picked = group[:]
        rng.shuffle(picked)
        subset.extend(picked[:QUICK_EVAL_SAMPLES_PER_CLASS])
    return subset


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    correct_per_class = Counter()
    total_per_class = Counter()
    confusion = [[0] * num_classes for _ in range(num_classes)]  # rows=actual, cols=predicted
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        for pred, target in zip(preds.tolist(), y.tolist()):
            total_per_class[target] += 1
            confusion[target][pred] += 1
            if pred == target:
                correct_per_class[target] += 1
    per_class_acc = {
        c: correct_per_class[c] / total_per_class[c] for c in total_per_class if total_per_class[c] > 0
    }
    overall_acc = sum(correct_per_class.values()) / max(sum(total_per_class.values()), 1)
    macro_acc = sum(per_class_acc.values()) / max(len(per_class_acc), 1)
    model.train()
    return {
        "overall_acc": overall_acc, "macro_acc": macro_acc,
        "per_class_acc": per_class_acc, "confusion_matrix": confusion,
    }


def train_on_shard(
    model: nn.Module,
    shard_dir: Path,
    entries: list[SampleEntry],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    ds = LocalShardDataset(shard_dir, entries)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    model.train()
    running_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


def save_checkpoints(model, optimizer, idx, epoch, eval_result, best_macro_acc, history, metrics, checkpoint_dir):
    if eval_result is not None and eval_result["macro_acc"] > best_macro_acc:
        best_macro_acc = eval_result["macro_acc"]
        torch.save(
            {"model_state": model.state_dict(), "class_names": idx.class_names, "epoch": epoch},
            checkpoint_dir / "best.pt",
        )
    history.append(metrics)
    torch.save(
        {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
         "epoch": epoch, "class_names": idx.class_names},
        checkpoint_dir / "last.pt",
    )
    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
    return best_macro_acc


def log_final_results(final_result: dict, idx, checkpoint_dir: Path) -> None:
    log.info(
        "FINAL TEST RESULTS: overall_acc=%.4f macro_acc=%.4f",
        final_result["overall_acc"], final_result["macro_acc"],
    )
    for c, acc in sorted(final_result["per_class_acc"].items()):
        log.info("  class %d (%s): acc=%.4f", c, idx.class_names[c], acc)
    (checkpoint_dir / "final_test_results.json").write_text(json.dumps(
        {"overall_acc": final_result["overall_acc"], "macro_acc": final_result["macro_acc"],
         "per_class_acc": {idx.class_names[c]: acc for c, acc in final_result["per_class_acc"].items()},
         "class_names": idx.class_names, "confusion_matrix": final_result.get("confusion_matrix")},
        indent=2,
    ))


def run_shard_mode(idx, train_entries, test_entries, quick_entries, model, optimizer, criterion, args, device, download_threads) -> None:
    train_shards = idx.make_shards(train_entries, args.num_train_shards, seed=args.seed)
    test_shards = idx.make_shards(test_entries, args.num_test_shards, seed=args.seed)
    log.info(
        "Train shards: %s | Test shards: %s | quick-eval subset: %d samples",
        [len(s) for s in train_shards], [len(s) for s in test_shards], len(quick_entries),
    )

    flat_schedule = [
        i for epoch in range(args.epochs) for i in boustrophedon_order(len(train_shards), epoch)
    ]

    best_macro_acc = -1.0
    history = []

    with ShardCache(idx, args.cache_dir, num_threads=download_threads) as cache:
        log.info("Downloading fixed quick-eval subset once (kept cached for the whole run)...")
        cache.ensure_shard(QUICK_EVAL_SHARD_ID, quick_entries)

        for epoch in range(args.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            for pos in range(len(train_shards)):
                g = epoch * len(train_shards) + pos
                shard_idx = flat_schedule[g]
                shard_key = f"train_{shard_idx}"
                entries = train_shards[shard_idx]

                t0 = time.time()
                shard_dir = cache.ensure_shard(shard_key, entries)
                log.info("Epoch %d [%d/%d] shard %d ready in %.1fs (%d files)",
                          epoch, pos + 1, len(train_shards), shard_idx, time.time() - t0, len(entries))

                keep = {shard_key, QUICK_EVAL_SHARD_ID}
                if not args.no_prefetch and g + 1 < len(flat_schedule):
                    next_shard_idx = flat_schedule[g + 1]
                    next_key = f"train_{next_shard_idx}"
                    if next_key != shard_key:
                        cache.prefetch_async(next_key, train_shards[next_shard_idx])
                        keep.add(next_key)
                cache.evict_all_except(keep)

                shard_loss = train_on_shard(
                    model, shard_dir, entries, optimizer, criterion, device,
                    args.batch_size, args.num_workers,
                )
                epoch_loss += shard_loss
                log.info("Epoch %d shard %d loss %.4f", epoch, shard_idx, shard_loss)

            avg_loss = epoch_loss / len(train_shards)
            log.info("Epoch %d done in %.1fs, avg train loss %.4f", epoch, time.time() - epoch_start, avg_loss)

            metrics = {"epoch": epoch, "train_loss": avg_loss}
            eval_result = None
            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                quick_dir = cache.ensure_shard(QUICK_EVAL_SHARD_ID, quick_entries)
                quick_loader = DataLoader(
                    LocalShardDataset(quick_dir, quick_entries), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                )
                eval_result = evaluate(model, quick_loader, device, len(idx.class_names))
                metrics.update({k: v for k, v in eval_result.items() if k != "confusion_matrix"})
                log.info(
                    "Quick eval @ epoch %d: overall_acc=%.3f macro_acc=%.3f",
                    epoch, eval_result["overall_acc"], eval_result["macro_acc"],
                )

            best_macro_acc = save_checkpoints(
                model, optimizer, idx, epoch, eval_result, best_macro_acc, history, metrics, args.checkpoint_dir,
            )

        # Final full evaluation over the entire (previously untouched-this-run) test split.
        log.info("Running final full evaluation over all %d test shards...", len(test_shards))
        full_correct, full_total = Counter(), Counter()
        num_classes = len(idx.class_names)
        full_confusion = [[0] * num_classes for _ in range(num_classes)]
        for i, entries in enumerate(test_shards):
            shard_key = f"test_{i}"
            shard_dir = cache.ensure_shard(shard_key, entries)
            loader = DataLoader(
                LocalShardDataset(shard_dir, entries), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
            )
            result = evaluate(model, loader, device, num_classes)
            for c, acc in result["per_class_acc"].items():
                n = sum(1 for e in entries if e.target == c)
                full_correct[c] += round(acc * n)
                full_total[c] += n
            for r in range(num_classes):
                for c in range(num_classes):
                    full_confusion[r][c] += result["confusion_matrix"][r][c]
            cache.evict(shard_key)
            log.info("Test shard %d/%d done", i + 1, len(test_shards))

        final_per_class = {c: full_correct[c] / full_total[c] for c in full_total}
        final_result = {
            "overall_acc": sum(full_correct.values()) / max(sum(full_total.values()), 1),
            "macro_acc": sum(final_per_class.values()) / max(len(final_per_class), 1),
            "per_class_acc": final_per_class,
            "confusion_matrix": full_confusion,
        }
        log_final_results(final_result, idx, args.checkpoint_dir)


def run_streaming_mode(idx, train_entries, test_entries, quick_entries, model, optimizer, criterion, args, device) -> None:
    # Deliberately NOT persistent_workers=True: each DataLoader below is only
    # ever iterated, never held open across the others, so worker processes
    # (and the FTP/HTTP connections they hold) are fully torn down as soon as
    # each loader's iteration finishes. With persistent workers instead, the
    # train and quick-eval loaders' worker processes would stay alive
    # simultaneously and their connections would stack up -- verified live
    # against the FTP source, where that combination blew past the server's
    # 5-connections-per-user cap once the final test-set evaluation tried to
    # open its own workers on top of the two still-lingering loaders.
    train_loader = DataLoader(
        StreamingDataset(idx, train_entries), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    quick_loader = DataLoader(
        StreamingDataset(idx, quick_entries), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    best_macro_acc = -1.0
    history = []
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # StreamingDataset already retries+backs-off individual sample fetches, but
        # an outage that outlasts that (observed live: a LAN drop that persisted
        # past all per-item retries) still surfaces as the DataLoader worker
        # raising, which would otherwise kill the whole multi-hour run with zero
        # checkpoint for the epoch in progress. Re-running the epoch is safe/
        # idempotent (no state besides model/optimizer weights, which are
        # untouched by a failed attempt), so retry at this level too before
        # giving up for good.
        for epoch_attempt in range(3):
            try:
                avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                break
            except RuntimeError as exc:
                if epoch_attempt == 2:
                    log.error("Epoch %d failed 3 times, giving up. Last checkpoint is from the previous epoch.", epoch)
                    raise
                log.warning("Epoch %d attempt %d/3 failed (%s); waiting 30s and retrying the epoch", epoch, epoch_attempt + 1, exc)
                time.sleep(30)
        log.info("Epoch %d done in %.1fs, avg train loss %.4f (streaming mode)",
                  epoch, time.time() - epoch_start, avg_loss)

        metrics = {"epoch": epoch, "train_loss": avg_loss}
        eval_result = None
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_result = evaluate(model, quick_loader, device, len(idx.class_names))
            metrics.update({k: v for k, v in eval_result.items() if k != "confusion_matrix"})
            log.info(
                "Quick eval @ epoch %d: overall_acc=%.3f macro_acc=%.3f",
                epoch, eval_result["overall_acc"], eval_result["macro_acc"],
            )

        best_macro_acc = save_checkpoints(
            model, optimizer, idx, epoch, eval_result, best_macro_acc, history, metrics, args.checkpoint_dir,
        )

    log.info("Running final full evaluation over the entire test split (streaming)...")
    test_loader = DataLoader(
        StreamingDataset(idx, test_entries), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    final_result = evaluate(model, test_loader, device, len(idx.class_names))
    log_final_results(final_result, idx, args.checkpoint_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "ftp":
        log.info("Indexing FTP source (LIST only, no file downloads yet)...")
        idx = FTPIndex(**config.get_ftp_config())
        download_threads = args.download_threads or config.FTP_DOWNLOAD_THREADS
    else:
        log.info("Indexing remote archive (reads only the zip central directory)...")
        idx = RemoteZipIndex(config.get_zip_url())
        download_threads = args.download_threads or config.DOWNLOAD_THREADS_PER_SHARD
    log.info("Found %d samples across %d classes: %s", len(idx.samples), len(idx.class_names), idx.class_names)

    train_entries, test_entries = idx.split_train_test(seed=args.seed)
    log.info("Split: %d train / %d test (disjoint by construction)", len(train_entries), len(test_entries))
    quick_entries = quick_eval_subset(test_entries, args.seed)

    model = DroneRFClassifier(num_classes=len(idx.class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_entries, len(idx.class_names)).to(device))

    cache_mode = args.cache_mode
    if cache_mode == "auto":
        cache_mode = "streaming" if args.source == "ftp" else "shard"
    log.info("Using --cache-mode=%s (source=%s)", cache_mode, args.source)

    if cache_mode == "streaming":
        run_streaming_mode(idx, train_entries, test_entries, quick_entries, model, optimizer, criterion, args, device)
    else:
        run_shard_mode(idx, train_entries, test_entries, quick_entries, model, optimizer, criterion, args, device, download_threads)

    # Export for serving.
    model.eval()
    example_input = torch.zeros(1, 1, config.N_FFT, config.IQ_WINDOW_SAMPLES // config.HOP_LENGTH)
    scripted = torch.jit.trace(model.cpu(), example_input)
    scripted.save(str(args.checkpoint_dir / "model_scripted.pt"))
    (args.checkpoint_dir / "class_names.json").write_text(json.dumps(idx.class_names, indent=2))
    log.info("Saved TorchScript model to %s", args.checkpoint_dir / "model_scripted.pt")


if __name__ == "__main__":
    main()
