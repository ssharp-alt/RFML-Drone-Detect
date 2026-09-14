"""
Ephemeral, disk-backed shard cache for streaming training data off the
remote zip without ever keeping the full dataset locally.

Strategy (generalized from a "4 virtual drives" idea): split the file list
into N shards. Train through them in a boustrophedon (ping-pong) order --
1,2,3,4 on odd epochs, 4,3,2,1 on even epochs -- so the shard downloaded
last in one epoch is exactly the shard needed first in the next, and that
one download is skipped. A background thread also prefetches the *next*
shard over the network while the current one is still training, overlapping
I/O with compute, as long as there's enough free disk headroom to hold two
shards at once. Every shard directory is deleted once it's no longer needed,
and the whole cache root is removed at the end of the run (or on crash, via
atexit) -- nothing from the dataset is left on disk afterward.
"""
from __future__ import annotations

import atexit
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

from tqdm import tqdm

from . import config
from .remote_zip import RemoteZipIndex, SampleEntry

log = logging.getLogger(__name__)


def boustrophedon_order(num_shards: int, epoch_idx: int) -> List[int]:
    """Shard visiting order for a given (0-based) epoch: forward on even
    epochs, reversed on odd epochs, so consecutive epochs share a boundary
    shard instead of wrapping from last-shard back to first-shard."""
    order = list(range(num_shards))
    return order if epoch_idx % 2 == 0 else list(reversed(order))


def shard_bytes(entries: Sequence[SampleEntry]) -> int:
    return sum(e.file_size for e in entries)


def check_disk_space(path: Path, required_bytes: int, headroom_fraction: float = 1.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    needed = int(required_bytes * headroom_fraction)
    if free < needed:
        raise RuntimeError(
            f"Not enough free disk space at {path}: need ~{needed / 1e9:.1f}GB "
            f"(shard size {required_bytes / 1e9:.1f}GB x {headroom_fraction:.2f} headroom), "
            f"only {free / 1e9:.1f}GB free. Increase --num-shards to shrink shard size, "
            f"free up disk space, or point --cache-dir at a larger volume."
        )


class ShardCache:
    def __init__(
        self,
        index: RemoteZipIndex,
        cache_root: Path,
        num_threads: int = config.DOWNLOAD_THREADS_PER_SHARD,
    ):
        self.index = index
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads

        self._cached_dirs: Dict[int, Path] = {}
        self._thread_local = threading.local()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-prefetch")
        self._prefetch_future = None
        self._prefetch_shard_id: int | None = None
        self._lock = threading.Lock()

        atexit.register(self._cleanup_all)

    # -- internal -----------------------------------------------------
    def _zf_for_thread(self):
        zf = getattr(self._thread_local, "zf", None)
        if zf is None:
            zf = self.index._open_fresh()
            self._thread_local.zf = zf
        return zf

    def _shard_dir(self, shard_id: int) -> Path:
        return self.cache_root / f"shard_{shard_id}"

    def _download_one(self, entry: SampleEntry, dest_dir: Path) -> None:
        dest_path = dest_dir / entry.filename
        if dest_path.exists() and dest_path.stat().st_size == entry.file_size:
            return  # already present (e.g. resumed run)
        zf = self._zf_for_thread()
        data = zf.read(entry.filename)
        tmp_path = dest_path.with_suffix(".part")
        tmp_path.write_bytes(data)
        tmp_path.replace(dest_path)

    def _download_shard(self, shard_id: int, entries: Sequence[SampleEntry], show_progress: bool = True) -> Path:
        dest_dir = self._shard_dir(shard_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        check_disk_space(self.cache_root, shard_bytes(entries), headroom_fraction=1.05)

        desc = f"downloading shard {shard_id} ({len(entries)} files)"
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            futures = [pool.submit(self._download_one, e, dest_dir) for e in entries]
            iterator = tqdm(futures, desc=desc, unit="file") if show_progress else futures
            for f in iterator:
                f.result()  # surfaces exceptions immediately
        return dest_dir

    # -- public API -----------------------------------------------------
    def ensure_shard(self, shard_id: int, entries: Sequence[SampleEntry]) -> Path:
        """Block until shard_id is fully present on disk, downloading it
        (or waiting on an in-flight prefetch of it) if necessary."""
        with self._lock:
            already_prefetching_this = self._prefetch_shard_id == shard_id and self._prefetch_future is not None
            future = self._prefetch_future if already_prefetching_this else None

        if future is not None:
            log.info("Waiting on in-flight prefetch of shard %s", shard_id)
            future.result()
            with self._lock:
                self._cached_dirs[shard_id] = self._shard_dir(shard_id)
                self._prefetch_future = None
                self._prefetch_shard_id = None
            return self._shard_dir(shard_id)

        if shard_id in self._cached_dirs:
            return self._cached_dirs[shard_id]

        dest_dir = self._download_shard(shard_id, entries)
        self._cached_dirs[shard_id] = dest_dir
        return dest_dir

    def prefetch_async(self, shard_id: int, entries: Sequence[SampleEntry]) -> bool:
        """Kick off a background download of shard_id if it isn't already
        cached or being prefetched. Returns False (no-op) if there isn't
        enough spare disk headroom to safely hold two shards at once --
        callers should fall back to a synchronous ensure_shard() later."""
        with self._lock:
            if shard_id in self._cached_dirs or self._prefetch_shard_id == shard_id:
                return True

            required = shard_bytes(entries)
            free = shutil.disk_usage(self.cache_root).free
            if free < required * config.PREFETCH_DISK_HEADROOM_FRACTION:
                log.info(
                    "Skipping background prefetch of shard %d: not enough spare disk headroom "
                    "(%.1fGB free, need %.1fGB). Will download synchronously when needed.",
                    shard_id, free / 1e9, required * config.PREFETCH_DISK_HEADROOM_FRACTION / 1e9,
                )
                return False

            self._prefetch_shard_id = shard_id
            self._prefetch_future = self._prefetch_executor.submit(
                self._download_shard, shard_id, entries, False
            )
            return True

    def evict(self, shard_id: int) -> None:
        with self._lock:
            d = self._cached_dirs.pop(shard_id, None)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def evict_all_except(self, keep_ids: Sequence[int]) -> None:
        keep = set(keep_ids)
        for shard_id in list(self._cached_dirs):
            if shard_id not in keep:
                self.evict(shard_id)

    def _cleanup_all(self) -> None:
        self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def close(self) -> None:
        self._cleanup_all()
        try:
            atexit.unregister(self._cleanup_all)
        except Exception:
            pass

    def __enter__(self) -> "ShardCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
