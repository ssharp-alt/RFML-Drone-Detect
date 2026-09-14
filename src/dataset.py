"""Dataset that reads IQ samples from a local shard directory (populated by
ShardCache) and yields normalized log-magnitude spectrograms + labels."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .remote_zip import SampleEntry
from .transforms import compute_log_magnitude_spectrogram


class LocalShardDataset(Dataset):
    def __init__(self, shard_dir: Path, entries: Sequence[SampleEntry]):
        self.shard_dir = Path(shard_dir)
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        data = torch.load(self.shard_dir / entry.filename, weights_only=True, map_location="cpu")
        spec = compute_log_magnitude_spectrogram(data["x_iq"])
        return spec, entry.target
