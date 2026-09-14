"""
Local live-progress dashboard for a running (or finished) src.train job.

Serves a small static page plus a /status.json endpoint the page polls --
everything is read straight from the log file, the checkpoints/ directory,
and the live process tree via psutil, all on this machine. No published
Artifact is involved (a hosted Artifact page's CSP blocks it from reaching
localhost anyway) -- just open http://localhost:<port>/ in a browser.

Usage:
    .venv/Scripts/python.exe -m dashboard.server --log train_ftp_full.log \
        --checkpoint-dir checkpoints --target-epochs 10 --port 8787
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

FETCH_FAIL_RE = re.compile(r"Fetch failed for")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")

DASHBOARD_DIR = Path(__file__).parent


def find_training_processes():
    """The main `python -m src.train` process(es) plus their spawned
    DataLoader worker children (psutil.Process objects), deduplicated by
    PID. Windows' venv launcher shows up as its own process (parent of the
    real interpreter), so it independently matches the same cmdline filter
    as its child -- without dedup, each root's recursive children() call
    re-counts the other root's descendants too, inflating the count."""
    by_pid = {}
    roots = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any("src.train" in part for part in cmdline):
            roots.append(proc)
            by_pid[proc.pid] = proc

    for proc in roots:
        try:
            for child in proc.children(recursive=True):
                by_pid[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return list(by_pid.values())


def process_stats() -> dict:
    procs = find_training_processes()
    if not procs:
        return {"alive": False, "cpu_seconds": None, "thread_count": None, "process_count": 0}
    cpu_seconds = 0.0
    thread_count = 0
    for proc in procs:
        try:
            t = proc.cpu_times()
            cpu_seconds += t.user + t.system
            thread_count += proc.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "alive": True, "cpu_seconds": round(cpu_seconds, 1),
        "thread_count": thread_count, "process_count": len(procs),
    }


def parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"error_count": 0, "first_ts": None, "last_ts": None, "last_line": None}
    error_count = 0
    first_ts = None
    last_ts = None
    last_line = None
    with log_path.open("r", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            last_line = line
            if FETCH_FAIL_RE.search(line):
                error_count += 1
            m = TS_RE.match(line)
            if m:
                if first_ts is None:
                    first_ts = m.group(1)
                last_ts = m.group(1)
    return {"error_count": error_count, "first_ts": first_ts, "last_ts": last_ts, "last_line": last_line}


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def build_status(log_path: Path, checkpoint_dir: Path, target_epochs: int) -> dict:
    history = load_json(checkpoint_dir / "history.json") or []
    final_results = load_json(checkpoint_dir / "final_test_results.json")
    log_info = parse_log(log_path)
    proc_info = process_stats()

    elapsed_seconds = None
    if log_info["first_ts"] and log_info["last_ts"]:
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            elapsed_seconds = (
                datetime.strptime(log_info["last_ts"], fmt) - datetime.strptime(log_info["first_ts"], fmt)
            ).total_seconds()
        except ValueError:
            pass

    return {
        "target_epochs": target_epochs,
        "epochs_completed": len(history),
        "history": history,
        "final_results": final_results,
        "training_done": final_results is not None,
        "process": proc_info,
        "error_count": log_info["error_count"],
        "elapsed_seconds": elapsed_seconds,
        "last_log_line": log_info["last_line"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


class Handler(BaseHTTPRequestHandler):
    log_path: Path
    checkpoint_dir: Path
    target_epochs: int

    def log_message(self, format, *args):  # noqa: A002 -- quiet; avoid spamming this process's own stdout
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_file(DASHBOARD_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/status.json":
            status = build_status(self.log_path, self.checkpoint_dir, self.target_epochs)
            self._serve_bytes(json.dumps(status).encode(), "application/json", no_store=True)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        self._serve_bytes(path.read_bytes(), content_type)

    def _serve_bytes(self, body: bytes, content_type: str, no_store: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, default=Path("train_ftp_full.log"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--target-epochs", type=int, default=10)
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args()

    Handler.log_path = args.log
    Handler.checkpoint_dir = args.checkpoint_dir
    Handler.target_epochs = args.target_epochs

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard serving at http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
