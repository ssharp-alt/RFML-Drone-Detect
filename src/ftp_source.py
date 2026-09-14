"""
Lists and splits the contents of a dataset served as individual files over
FTP (e.g. a LAN NAS mirror of the same Kaggle drone-RF set), instead of
packed into a single remote zip.

No zip central directory trick is needed here -- an FTP LIST already gives
the full file listing with sizes directly. ShardCache only requires
_open_fresh() to return an object with a .read(filename) -> bytes method
(zipfile.ZipFile already satisfies this for remote_zip.py), so the same
ephemeral shard-cache / boustrophedon scheduler works against this source
completely unmodified -- only the fetch mechanism differs.

ftplib.FTP (like zipfile.ZipFile) is NOT thread-safe: _open_fresh() opens a
brand-new control+data connection per call, and ShardCache caches one such
connection per download worker thread.
"""
from __future__ import annotations

import ftplib
import io
from typing import List

from . import config
from .sample_index import BaseSampleIndex, SampleEntry, parse_iq_filename


class FTPFetcher:
    def __init__(self, host: str, user: str, password: str, remote_dir: str, port: int = 21, timeout: int = 60):
        self.ftp = ftplib.FTP(timeout=timeout)
        self.ftp.connect(host, port)
        self.ftp.login(user, password)
        self.ftp.cwd(remote_dir)

    def read(self, filename: str) -> bytes:
        buf = io.BytesIO()
        self.ftp.retrbinary(f"RETR {filename}", buf.write)
        return buf.getvalue()

    def close(self) -> None:
        try:
            self.ftp.quit()
        except Exception:
            try:
                self.ftp.close()
            except Exception:
                pass


class FTPIndex(BaseSampleIndex):
    def __init__(self, host: str, user: str, password: str, remote_dir: str, port: int = 21):
        self.host = host
        self.user = user
        self.password = password
        self.remote_dir = remote_dir
        self.port = port

        ftp = ftplib.FTP(timeout=60)
        ftp.connect(host, port)
        ftp.login(user, password)
        ftp.cwd(remote_dir)

        lines: List[str] = []
        ftp.retrlines("LIST", lines.append)

        self.samples: List[SampleEntry] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue
            name = parts[-1]
            size = int(parts[4])
            if name.startswith(config.FILENAME_PREFIX) and name.endswith(".pt"):
                sample_id, target, snr = parse_iq_filename(name)
                self.samples.append(SampleEntry(name, sample_id, target, snr, size))

        self.class_names = self._read_class_names(ftp)
        ftp.quit()

    def _read_class_names(self, ftp: ftplib.FTP) -> List[str]:
        try:
            buf = io.BytesIO()
            ftp.retrbinary("RETR class_stats.csv", buf.write)
            text = buf.getvalue().decode()
        except ftplib.all_errors:
            return list(config.DEFAULT_CLASS_NAMES)
        rows = [line.split(",") for line in text.strip().splitlines()[1:]]
        by_idx = {int(r[2]): r[1] for r in rows}
        return [by_idx[i] for i in sorted(by_idx)]

    def _open_fresh(self) -> FTPFetcher:
        return FTPFetcher(self.host, self.user, self.password, self.remote_dir, self.port)
