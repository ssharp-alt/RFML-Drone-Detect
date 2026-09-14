"""
Standalone post-hoc confusion-matrix computation.

train.py's evaluate() was extended to also return a confusion matrix, but a
job already running in another process keeps its old in-memory code and
won't pick that up. This script re-evaluates a saved checkpoint against the
same deterministic test split (same seed => same split, so this is not
"peeking" at different data) using the current evaluate(), and (re)writes
checkpoint_dir/final_test_results.json with the confusion matrix included --
the same file/shape an in-process run would have produced directly.

Usage:
    .venv/Scripts/python.exe -m dashboard.compute_confusion_matrix \
        --source ftp --checkpoint checkpoints/best.pt --checkpoint-dir checkpoints
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import config
from src.ftp_source import FTPIndex
from src.model import DroneRFClassifier
from src.remote_zip import RemoteZipIndex
from src.streaming_dataset import StreamingDataset
from src.train import evaluate, log_final_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["zip", "ftp"], default="ftp")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.source == "ftp":
        idx = FTPIndex(**config.get_ftp_config())
    else:
        idx = RemoteZipIndex(config.get_zip_url())

    _, test_entries = idx.split_train_test(seed=args.seed)
    print(f"Test set: {len(test_entries)} samples (same seed={args.seed} as the original run)")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DroneRFClassifier(num_classes=len(idx.class_names))
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    loader = DataLoader(
        StreamingDataset(idx, test_entries), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    result = evaluate(model, loader, device, len(idx.class_names))
    log_final_results(result, idx, args.checkpoint_dir)
    print("Wrote", args.checkpoint_dir / "final_test_results.json", "(now includes confusion_matrix)")


if __name__ == "__main__":
    main()
