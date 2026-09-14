"""
Central configuration for the drone-RF streaming training pipeline.

The Kaggle/GCS signed URL for the dataset zip is time-limited (the one the
project was bootstrapped with expires ~2026-08-03T20:20Z, 3 days from
issuance) so it is deliberately kept OUT of source control. Provide a fresh
one via the DRONE_RF_ZIP_URL environment variable, or write it to a local
`zip_url.txt` file in the project root (already listed in .gitignore).

Similarly, FTP credentials for a LAN-hosted copy of the dataset are kept out
of source control -- see get_ftp_config() below.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_zip_url() -> str:
    url = os.environ.get("DRONE_RF_ZIP_URL")
    if url:
        return url.strip()
    url_file = PROJECT_ROOT / "zip_url.txt"
    if url_file.exists():
        url = url_file.read_text().strip()
        if url:
            return url
    raise RuntimeError(
        "No dataset URL configured. Set the DRONE_RF_ZIP_URL environment "
        "variable or write the (re-signed, if expired) URL to "
        f"{url_file}."
    )


def get_ftp_config() -> dict:
    """FTP source config as {host, user, password, remote_dir, port}.
    Env vars take priority; falls back to a local, gitignored
    ftp_credentials.json (plaintext -- fine for a LAN dev NAS, but don't
    reuse these credentials anywhere that matters)."""
    host = os.environ.get("DRONE_RF_FTP_HOST")
    user = os.environ.get("DRONE_RF_FTP_USER")
    password = os.environ.get("DRONE_RF_FTP_PASSWORD")
    remote_dir = os.environ.get("DRONE_RF_FTP_DIR")
    port = int(os.environ.get("DRONE_RF_FTP_PORT", 21))
    if host and user and password is not None and remote_dir:
        return {"host": host, "user": user, "password": password, "remote_dir": remote_dir, "port": port}

    cred_file = PROJECT_ROOT / "ftp_credentials.json"
    if cred_file.exists():
        data = json.loads(cred_file.read_text())
        return {
            "host": data["host"], "user": data["user"], "password": data["password"],
            "remote_dir": data["remote_dir"], "port": int(data.get("port", 21)),
        }
    raise RuntimeError(
        "No FTP source configured. Set DRONE_RF_FTP_HOST / _USER / _PASSWORD / "
        f"_DIR environment variables, or write them to {cred_file} as JSON "
        '(keys: host, user, password, remote_dir, port).'
    )


# --- Dataset layout -------------------------------------------------------
# Ordered class-index -> class-name mapping, as published in class_stats.csv
# inside the archive. train.py re-derives this from the live archive at
# startup and only falls back to this constant if that read fails.
DEFAULT_CLASS_NAMES = [
    "DJI",
    "FutabaT14",
    "FutabaT7",
    "Graupner",
    "Noise",
    "Taranis",
    "Turnigy",
]

FILENAME_PREFIX = "IQdata_sample"

# --- Split / sharding -------------------------------------------------------
TRAIN_FRACTION = 0.7  # remainder (0.3) is held out as the test set
SPLIT_SEED = 1234  # fixed seed -> deterministic, reproducible, disjoint split

DEFAULT_NUM_TRAIN_SHARDS = 4
DEFAULT_NUM_TEST_SHARDS = 2  # test set is touched far less often; fewer/larger shards is fine

# --- Cache / disk ------------------------------------------------------------
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".shard_cache"
# Safety margin required (on top of one shard's raw size) before we allow
# background prefetch of the *next* shard to run concurrently with the
# current shard's training pass.
PREFETCH_DISK_HEADROOM_FRACTION = 1.15
DOWNLOAD_THREADS_PER_SHARD = 8
# The reference LAN FTP mirror caps concurrent connections per user at 5;
# stay comfortably under that (also leaves room for the one-off LIST/
# class_stats connection FTPIndex opens and closes during __init__).
FTP_DOWNLOAD_THREADS = 4

# --- Model / signal processing ---------------------------------------------
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 1024
# Every training sample is a fixed-length raw IQ capture of this many
# complex samples (verified against the archive: x_iq.shape == (2, 1048576)).
IQ_WINDOW_SAMPLES = 1024 * 1024
NUM_CLASSES = len(DEFAULT_CLASS_NAMES)

# --- Serving / RF assumptions ------------------------------------------------
# The dataset does not encode a physical sample rate -- these are RC control-
# link protocols (FutabaT14/T7, Graupner, Taranis, Turnigy are all 2.4GHz
# ISM-band hobbyist control links; DJI's OcuSync/Lightbridge is the widest of
# the group). In the absence of a receiver spec, we assume a 20 MHz
# commercial-drone-link channel bandwidth (DJI OcuSync's widest mode is a
# reasonable worst case for this protocol family) and require the inference
# pipeline's IQ capture to be sampled 20% above that, per the user's
# instruction. THIS IS A DOCUMENTED ASSUMPTION, NOT A MEASURED FACT --
# override via env vars if you know your actual receiver's specs.
ASSUMED_CHANNEL_BANDWIDTH_HZ = float(os.environ.get("ASSUMED_CHANNEL_BANDWIDTH_HZ", 20e6))
BANDWIDTH_MARGIN = float(os.environ.get("BANDWIDTH_MARGIN", 1.2))
ASSUMED_SAMPLE_RATE_HZ = ASSUMED_CHANNEL_BANDWIDTH_HZ * BANDWIDTH_MARGIN  # 24 MHz by default

# The model's conv stack + adaptive pooling head make it tolerant to some
# variation in spectrogram time-frame count, but for best fidelity to
# training, serving keeps the same IQ_WINDOW_SAMPLES window. At the assumed
# sample rate this is a capture window of:
IQ_WINDOW_DURATION_SEC = IQ_WINDOW_SAMPLES / ASSUMED_SAMPLE_RATE_HZ  # ~43.7 ms by default
