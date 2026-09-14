"""
Lists and splits the contents of a (possibly huge) remote zip archive
without downloading it, using HTTP Range requests to read only the zip
central directory. Verified against the actual dataset: a 125GB archive's
central directory is ~2MB / 5 requests to open.

Actual per-file bytes are fetched on demand elsewhere (shard_cache.py) using
a fresh ZipFile handle per download worker thread, since zipfile.ZipFile is
not safe to share across threads.

Split/shard logic lives in sample_index.py, shared with other sources
(e.g. ftp_source.py) since it only depends on the file listing.
"""
from __future__ import annotations

import io
import zipfile
from typing import Dict, List

import requests

from . import config
from .sample_index import BaseSampleIndex, SampleEntry, parse_iq_filename


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file-like object backed by HTTP Range requests."""

    def __init__(self, url: str, size: int, session: requests.Session | None = None, retries: int = 4):
        self.url = url
        self.size = size
        self.pos = 0
        self.session = session or requests.Session()
        self.retries = retries
        self.requests_made = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        headers = {"Range": f"bytes={self.pos}-{end}"}

        last_exc = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.url, headers=headers, timeout=60)
                r.raise_for_status()
                data = r.content
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self.retries - 1:
                    raise
                continue
        else:
            raise last_exc  # pragma: no cover

        self.requests_made += 1
        self.bytes_fetched += len(data)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class RemoteZipIndex(BaseSampleIndex):
    """Reads the central directory of the remote dataset zip once, then
    provides deterministic, disjoint train/test splitting and stratified
    sharding purely from the in-memory file listing (no data download)."""

    def __init__(self, url: str):
        self.url = url
        head = requests.head(url, timeout=30)
        head.raise_for_status()
        self.size = int(head.headers["content-length"])
        if head.headers.get("accept-ranges", "").lower() != "bytes":
            raise RuntimeError("Remote server does not advertise HTTP Range support; cannot stream this zip.")

        zf = self._open_fresh()
        self.samples: List[SampleEntry] = []
        for info in zf.infolist():
            if info.filename.startswith(config.FILENAME_PREFIX) and info.filename.endswith(".pt"):
                sample_id, target, snr = parse_iq_filename(info.filename)
                self.samples.append(SampleEntry(info.filename, sample_id, target, snr, info.file_size))

        self.class_names = self._read_class_names(zf)

    def _open_fresh(self) -> zipfile.ZipFile:
        """A new ZipFile with its own HTTP-backed reader. Cheap (central
        directory is only ~2MB for this archive) and required because
        zipfile.ZipFile / its file object are not thread-safe."""
        raw = HTTPRangeFile(self.url, self.size)
        buffered = io.BufferedReader(raw, buffer_size=1024 * 1024)
        return zipfile.ZipFile(buffered)

    def _read_class_names(self, zf: zipfile.ZipFile) -> List[str]:
        try:
            text = zf.read("class_stats.csv").decode()
        except KeyError:
            return list(config.DEFAULT_CLASS_NAMES)
        rows = [line.split(",") for line in text.strip().splitlines()[1:]]
        by_idx: Dict[int, str] = {int(r[2]): r[1] for r in rows}
        return [by_idx[i] for i in sorted(by_idx)]
