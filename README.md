# Drone RF Classifier — Streaming Training + KServe Deployment

**License:** [Apache 2.0](LICENSE) — free to use, modify, and distribute,
commercially or otherwise. Third-party dataset attribution is in [`NOTICE`](NOTICE).

Classifies drone/RC control-link IQ captures (`DJI`, `FutabaT14`, `FutabaT7`,
`Graupner`, `Noise`, `Taranis`, `Turnigy`) from the
[Noisy Drone RF Signal Classification v2](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2)
dataset (© sgluege, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) —
see attribution in `NOTICE`),

**without ever storing the full ~125GB dataset locally**. Individual sample
files are read on demand straight out of the remote zip via HTTP range
requests, trained on, and discarded. This can be a road map for others to train models on large datasets without downloading everything..

`ProtoType.py` (project root) is the original local-file prototype: it reads
`.pt` files from disk and rasterizes each one to a PNG spectrogram image. It's
kept for reference but superseded by `src/` below, which trains directly on
spectrogram *tensors* (no PNG round-trip) and never requires the data to sit
on local disk.

## How it works

| Module | Responsibility |
|---|---|
| `src/remote_zip.py` | Reads only the zip's **central directory** (~2MB / 5 HTTP requests for this 125GB archive) to list all 17,744 sample files without downloading them. Produces a deterministic, **stratified 70/30 train/test split with zero overlap by construction** (each sample is assigned to exactly one output list), and stratified sharding of the train set. |
| `src/shard_cache.py` | Ephemeral, disk-backed cache implementing a **boustrophedon (ping-pong) shard schedule**: shards are visited `1→2→3→4` on odd epochs and `4→3→2→1` on even epochs, so the shard downloaded last in one epoch is exactly the one needed first in the next — skipping that re-download. A background thread also prefetches the *next* shard while the current one trains, if there's disk headroom for two shards at once. The whole cache directory is deleted at the end of the run (or on crash, via `atexit`). |
| `src/transforms.py` | IQ → normalized log-magnitude spectrogram (`torch.stft`), shared byte-for-byte between training and serving so preprocessing can't drift between the two. |
| `src/model.py` | Compact ResNet-style CNN with adaptive pooling, so the classifier head tolerates some variation in spectrogram time-frame count. |
| `src/dataset.py` | `Dataset` that reads `.pt` files from a local shard directory and yields `(spectrogram, label)`. |
| `src/train.py` | Wires it all together: boustrophedon epoch loop, per-epoch quick eval on a small fixed stratified subset, a final full evaluation over the entire held-out 30%, checkpointing, and TorchScript export for serving. |
| `serving/` | KServe custom-predictor artifacts (see below). |

This was validated end-to-end against the live archive: the 70/30 split has
**0 overlap** and the union of the two splits equals the full dataset;
sharding is class-stratified; the boustrophedon scheduler was confirmed to
skip a shard's re-download at an epoch boundary (0.0s "download" time on
reuse) in a live smoke run.

### Why not cache everything, or nothing?

Streaming every sample fresh every epoch means re-transferring close to the
full ~125GB archive *per epoch*. Caching the whole dataset to disk once
defeats the "don't keep the data locally" goal. Sharding is the middle
ground: only one (or two, with prefetch) shard's worth of disk is ever used
at a time, it's deleted the moment it's no longer needed, and the
boustrophedon order recovers a chunk of the naive re-download cost for free.

**Disk sizing**: with the default 4 train shards, each shard is roughly
`0.7 * 125GB / 4 ≈ 22GB` compressed (a bit more decompressed on disk, since
files are cached as their original `.pt` bytes). `train.py` checks free disk
space before every shard download and raises a clear error if there isn't
enough room — pass `--num-shards` higher to shrink shard size, or `--cache-dir`
to point at a bigger volume. With 32GB RAM this is a non-issue either way:
RAM only holds one batch at a time, never a whole shard.

## Setup

```
py -3 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

The Kaggle/GCS signed URL is **time-limited** (~3 days from issuance) and is
deliberately not committed to source control. Provide it either via:

```
$env:DRONE_RF_ZIP_URL = "https://storage.googleapis.com/..."
```

or by writing it to `zip_url.txt` in the project root (already
`.gitignore`d; this repo was bootstrapped with the URL provided, written
there for convenience — replace it's expired).

## Training

```
.venv/Scripts/python.exe -m src.train --epochs 10 --num-train-shards 4 --batch-size 16
```

Key flags (see `python -m src.train --help`):
- `--num-train-shards` / `--num-test-shards` — default 4 / 2, per the disk
  sizing note above.
- `--eval-every` — epochs between quick-eval passes on a small fixed
  stratified subset (default: every epoch). A full pass over the entire 30%
  test set only happens once, at the very end.
- `--no-prefetch` — disable the background next-shard download if you'd
  rather not risk two shards' worth of disk at once.
- `--cache-dir` — where the ephemeral shard cache lives (default
  `.shard_cache/`, auto-deleted after the run).

Outputs land in `checkpoints/`: `last.pt` / `best.pt` (state dicts, for
resuming/inspection), `history.json`, `final_test_results.json`, and — the
serving artifact — `model_scripted.pt` (TorchScript) + `class_names.json`.

Class weights are computed from the train split's class distribution and
passed to the loss (the archive is heavily imbalanced: `Noise` is ~50% of
samples, `FutabaT7`/`Graupner` are ~4.5% each), and evaluation reports both
raw and macro-averaged (per-class-mean) accuracy since raw accuracy is
misleading under this imbalance.

## Results (FTP-sourced run, 10 epochs, streaming mode)

Full held-out test set (5,324 samples, never touched during training):

| Metric | Value |
|---|---|
| Overall accuracy | **94.8%** |
| Macro accuracy (per-class mean) | **92.6%** |

Per-class: Noise 99.3%, Graupner 97.5%, Taranis 97.0%, Turnigy 92.2%,
FutabaT7 91.3%, FutabaT14 87.1%, DJI 83.9% (DJI is the hardest class —
plausibly because OcuSync/Lightbridge is spectrally the most complex signal
in the set). Confusion matrix available via the dashboard (see
`dashboard/`) once `checkpoints/final_test_results.json` includes it — a run
finishes with it directly as of this version of `train.py`; a run started
before that change needs `dashboard/compute_confusion_matrix.py` run once
against its checkpoint to backfill it (same deterministic test split, so
this isn't re-evaluating on different data).

Checkpoint used for serving: `checkpoints/best.pt` (epoch 9), exported to
`checkpoints/model_scripted.pt`.

## Deploying to KServe on k3s

`serving/` contains **artifacts only** — build/push/apply are left to you (future work)
since this work has no visibility into your cluster.

1. Train first (produces `checkpoints/model_scripted.pt` + `class_names.json`,
   which the image bakes in — no object storage / storageUri needed, the
   simplest option for a bare k3s cluster).
2. Build the image from the project root:
   ```
   docker build -f serving/Dockerfile -t drone-rf-classifier:latest .
   ```
3. Get the image onto your k3s nodes — either push to a registry your
   cluster can pull from, or for a local dev cluster:
   ```
   docker save drone-rf-classifier:latest | k3s ctr images import -
   ```
4. Edit `serving/inferenceservice.yaml`'s `image:` field to match, then:
   ```
   kubectl apply -f serving/inferenceservice.yaml
   ```
5. Once ready:
   ```
   kubectl get inferenceservice drone-rf-classifier
   ```

### Request format

```
POST http://<ingress>/v1/models/drone-rf-classifier:predict
{
  "instances": [
    {"iq_base64": "<base64 of a little-endian float32 buffer, interleaved I0,Q0,I1,Q1,...>"}
  ]
}
```

A flat JSON list `[I0, Q0, I1, Q1, ...]` is also accepted per-instance
(simpler for `curl`, ~4x heavier over the wire than base64). Response:

```json
{"predictions": [{"class": "DJI", "confidence": 0.87, "probabilities": {...}}]}
```

### The bandwidth assumption

I assume the IQ pipeline handles 20% more bandwidth than a
commercial drone's channel. The dataset's classes (`FutabaT14`/`T7`,
`Graupner`, `Taranis`, `Turnigy`, `DJI`) are all 2.4GHz ISM-band RC
control-link protocols — DJI's OcuSync/Lightbridge is the widest of the
group. In the absence of a specific receiver spec, `src/config.py` assumes a
**20MHz commercial channel bandwidth × 1.2 margin = 24MHz sample rate**.
This is a documented, overridable assumption
(`ASSUMED_CHANNEL_BANDWIDTH_HZ` / `BANDWIDTH_MARGIN` env vars) — **not** a
measured fact about any real radio. If you know your actual SDR/receiver's
sample rate, override it and re-train, or resample your captures to 24MHz
before sending them to the endpoint. The training data itself has no
embedded sample rate (each capture is a fixed 1,048,576-sample window,
verified against the live archive), so the model only ever sees a fixed
number of samples per window — at the assumed rate, ~43.7ms per capture.
