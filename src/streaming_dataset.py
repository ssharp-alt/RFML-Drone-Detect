"""
Source-agnostic "fetch on demand, keep nothing" Dataset: every __getitem__
call pulls exactly one sample's bytes straight from the source (remote zip
via HTTP range, or FTP RETR) into memory, decodes it, and discards the raw
bytes -- nothing ever touches local disk. No shard scheduling, no eviction,
no boustrophedon bookkeeping: just index once, then grab each file when
it's needed.

This trades a per-epoch full re-fetch of the train split (cheap on a fast
LAN FTP mirror, expensive over the open internet against a 125GB zip) for
zero disk usage and much simpler code than shard_cache.py. Use --cache-mode
shard instead when the source is slow/metered relative to local disk
headroom (see train.py's --cache-mode auto default).

Works against ANY index exposing ._open_fresh() -> object with
.read(filename) -> bytes -- both RemoteZipIndex and FTPIndex already
satisfy this (it's the same contract ShardCache relies on).
"""
from __future__ import annotations

import io
import logging
import time
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .sample_index import SampleEntry
from .transforms import compute_log_magnitude_spectrogram

log = logging.getLogger(__name__)


class StreamingDataset(Dataset):
    def __init__(self, index, entries: Sequence[SampleEntry], retries: int = 6, backoff_base_sec: float = 1.5):
        self.index = index
        self.entries = list(entries)
        self.retries = retries
        self.backoff_base_sec = backoff_base_sec
        # Deliberately left unset here: with DataLoader num_workers>0 (and
        # Windows/spawn multiprocessing, which requires the Dataset to be
        # picklable), each worker process gets its own copy of this object
        # via pickling *before* any connection is made, then lazily opens
        # its own fetcher on its first __getitem__ call -- so pickling is
        # always cheap/safe and every worker ends up with its own
        # connection instead of illegally sharing one across processes.
        self._fetcher = None

    def __len__(self) -> int:
        return len(self.entries)

    def _get_fetcher(self):
        if self._fetcher is None:
            self._fetcher = self.index._open_fresh()
        return self._fetcher

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        last_exc = None
        for attempt in range(self.retries):
            try:
                fetcher = self._get_fetcher()
                data_bytes = fetcher.read(entry.filename)
                data = torch.load(io.BytesIO(data_bytes), weights_only=True, map_location="cpu")
                spec = compute_log_magnitude_spectrogram(data["x_iq"])
                return spec, entry.target
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure means "reconnect and retry"
                log.warning("Fetch failed for %s (attempt %d/%d): %s", entry.filename, attempt + 1, self.retries, exc)
                last_exc = exc
                self._fetcher = None  # drop the (possibly broken) connection; reconnect next attempt
                if attempt < self.retries - 1:
                    # Exponential backoff -- a transient LAN blip (connection reset,
                    # host briefly unreachable) needs time to clear, not an instant
                    # retry. Observed live: 3 immediate retries all failed against a
                    # multi-second network drop; backoff is required to ride it out.
                    time.sleep(self.backoff_base_sec * (2 ** attempt))
        raise RuntimeError(f"Failed to fetch {entry.filename} after {self.retries} attempts") from last_exc
