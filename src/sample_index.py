"""
Source-agnostic sample bookkeeping shared by every data source (remote zip
over HTTP range requests, LAN FTP directory, ...). Splitting and sharding
only depend on the file *listing*, never on how bytes are actually fetched,
so this logic lives once here and every concrete index subclasses it.

A concrete index only needs to set self.samples / self.class_names and
implement _open_fresh() -> object with a .read(filename) -> bytes method --
that's the entire contract ShardCache relies on.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

from . import config


@dataclass(frozen=True)
class SampleEntry:
    filename: str
    sample_id: int
    target: int
    snr: int
    file_size: int  # bytes, as reported by the source's own listing


def parse_iq_filename(name: str) -> tuple[int, int, int]:
    # IQdata_sample{ID}_target{T}_snr{SNR}.pt
    stem = name[: -len(".pt")]
    parts = stem.split("_")
    sample_id = int(parts[1][len("sample"):])
    target = int(parts[2][len("target"):])
    snr = int(parts[3][len("snr"):])
    return sample_id, target, snr


class BaseSampleIndex:
    samples: List[SampleEntry]
    class_names: List[str]

    def _open_fresh(self):
        """Return a fresh, thread-local-safe object exposing .read(filename)
        -> bytes. Subclasses must implement this."""
        raise NotImplementedError

    def split_train_test(
        self, train_fraction: float = config.TRAIN_FRACTION, seed: int = config.SPLIT_SEED
    ) -> tuple[List[SampleEntry], List[SampleEntry]]:
        """Stratified per-class split. Each class's samples are shuffled
        independently and cut at train_fraction, then the per-class train/
        test groups are concatenated. Because every sample is assigned to
        exactly one of the two output lists, train and test are disjoint by
        construction -- there is no way for a file to land in both."""
        by_class: Dict[int, List[SampleEntry]] = defaultdict(list)
        for s in self.samples:
            by_class[s.target].append(s)

        train: List[SampleEntry] = []
        test: List[SampleEntry] = []
        for target, group in by_class.items():
            rng = random.Random(f"{seed}-split-{target}")  # distinct, deterministic stream per class
            shuffled = group[:]
            rng.shuffle(shuffled)
            cut = round(len(shuffled) * train_fraction)
            train.extend(shuffled[:cut])
            test.extend(shuffled[cut:])
        return train, test

    def make_shards(
        self, entries: Sequence[SampleEntry], num_shards: int, seed: int = config.SPLIT_SEED
    ) -> List[List[SampleEntry]]:
        """Stratified round-robin sharding: each class's entries are
        shuffled then dealt out across shards, so every shard is
        approximately class-balanced like the full dataset rather than
        accidentally dominated by one class."""
        by_class: Dict[int, List[SampleEntry]] = defaultdict(list)
        for s in entries:
            by_class[s.target].append(s)

        shards: List[List[SampleEntry]] = [[] for _ in range(num_shards)]
        for target, group in by_class.items():
            rng = random.Random(f"{seed}-shard-{target}")
            shuffled = group[:]
            rng.shuffle(shuffled)
            for i, sample in enumerate(shuffled):
                shards[i % num_shards].append(sample)
        return shards
