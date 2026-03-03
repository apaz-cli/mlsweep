#!/usr/bin/env python3

"""File-based experiment logger.

Writes run_meta.json and metrics.jsonl co-located with training output.
If server is configured, also POSTs to exp_server.py for live visualization.

Local-first design: metrics are always written to local disk first. Server
POSTs are best-effort. On reconnect after an outage, missed records are
replayed to the server in chunks. No data is ever lost.

Constructor parameters:
    log_dir     Directory for this run's files (created if absent).
    experiment  Experiment name (groups runs together).
    run_name    Unique name for this run.
    tags        Dict of varied dimension values (str/int/float/bool).
    hparams     Full hyperparameter config dict (optional).
    server      http://host:port for exp_server.py (or set EXP_SERVER env var).
    tag         Optional prefix applied to every logged metric key.

Env vars:
    EXP_SERVER    http://host:port  (overridden by server= constructor arg)
    EXP_TAGS      comma-separated key=value pairs merged into tags
    MLSWEEP_TOKEN bearer token for server auth (overridden by token= constructor arg)
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

_logger = logging.getLogger(__name__)


def _git_head(path: str) -> str | None:
    """Return HEAD commit hash for the git repo containing path, or None."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_root(path: str) -> str | None:
    """Return the root directory of the git repo containing path, or None."""
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _collect_git_info() -> dict:
    """Return git commit hashes for mlsweep and the calling project (if distinct)."""
    mlsweep_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.getcwd()

    mlsweep_root = _git_root(mlsweep_dir)
    project_root = _git_root(project_dir)

    info = {}
    if mlsweep_root:
        commit = _git_head(mlsweep_dir)
        if commit:
            info["mlsweep_commit"] = commit
    if project_root and project_root != mlsweep_root:
        commit = _git_head(project_dir)
        if commit:
            info["project_commit"] = commit
    return info


_REPLAY_CHUNK = 500        # max records per /run/replay request
_HEARTBEAT_INTERVAL = 60.0  # seconds between heartbeat writes to run_meta.json
_STOP = object()           # sentinel that tells the flush thread to exit


class MLSweepLogger:
    """Local-first logger: always writes to disk, POSTs to server when reachable.

    On reconnect after a server outage, replays locally-buffered data so the
    server's metrics.jsonl is eventually consistent with the worker's copy.

    Non-blocking: log() enqueues immediately and returns.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        experiment: str,
        run_name: str,
        tags: dict[str, str | int | float | bool] | None = None,
        hparams: dict | None = None,
        server: str | None = None,
        token: str | None = None,
        tag: str | None = None,
    ):
        self.tag = tag
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.experiment = experiment
        self.run_name = run_name
        self.server = server or os.getenv("EXP_SERVER")
        self.token = token or os.getenv("MLSWEEP_TOKEN")

        # Merge EXP_TAGS env var into tags dict
        resolved_tags: dict[str, Any] = dict(tags or {})
        for pair in os.getenv("EXP_TAGS", "").split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                resolved_tags.setdefault(k.strip(), v.strip())

        # Server connectivity state
        self._server_ok = self.server is not None
        self._warned_server = False

        # File paths
        self._meta_path = os.path.join(log_dir, "run_meta.json")
        self._metrics_path = os.path.join(log_dir, "metrics.jsonl")
        self._start_time = time.time()

        # Persistent append file handle (line-buffered: flush on each newline)
        self._local_fh = open(self._metrics_path, "a", buffering=1)

        # _unsent_line: index of first line NOT yet confirmed on server.
        # Lines [0, _unsent_line) are on server; [_unsent_line, _total_lines) are local-only.
        self._unsent_line = 0
        self._total_lines = 0
        self._lines_lock = threading.Lock()

        meta = {
            "experiment": self.experiment,
            "run_name": self.run_name,
            "tags": resolved_tags,
            "hparams": hparams or {},
            "git": _collect_git_info(),
            "start_time": self._start_time,
            "last_heartbeat": None,
            "end_time": None,
            "status": "running",
        }
        self._write_meta_atomic(meta)

        # POST /run/start to server (best-effort; failure does not abort)
        if self.server:
            self._post_json("/run/start", meta)

        # Background flush thread
        self._queue: queue.Queue = queue.Queue()
        self._flush_thread = threading.Thread(
            target=self._flush_worker, daemon=True, name="exp-logger-flush"
        )
        self._flush_thread.start()

        _logger.info(
            f"MLSweepLogger enabled: experiment={self.experiment!r}, "
            f"run={self.run_name!r}, dir={log_dir}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_meta_atomic(self, meta: dict) -> None:
        """Write meta dict atomically via tmp file + os.replace."""
        tmp = self._meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, self._meta_path)

    def _update_heartbeat(self) -> None:
        """Stamp last_heartbeat in run_meta.json. Called every 60s by flush thread."""
        try:
            with open(self._meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["last_heartbeat"] = time.time()
        self._write_meta_atomic(meta)

    def _post_json(self, path: str, data: dict) -> bool:
        """POST JSON to server. Updates _server_ok. Returns True on success."""
        if not self.server:
            return False
        url = self.server.rstrip("/") + path
        body = json.dumps(data).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            url, data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
            if not self._server_ok:
                _logger.info(f"MLSweepLogger: server {self.server} back online")
            self._server_ok = True
            return True
        except (urllib.error.URLError, OSError) as e:
            if self._server_ok and not self._warned_server:
                _logger.warning(
                    f"MLSweepLogger: server {self.server} unreachable ({e}), "
                    "writing locally. Will replay on reconnect."
                )
                self._warned_server = True
            self._server_ok = False
            return False

    def _post_json_raw(self, path: str, data: dict, timeout: float = 30.0) -> bool:
        """POST JSON to server without updating _server_ok (used during replay)."""
        if not self.server:
            return False
        url = self.server.rstrip("/") + path
        body = json.dumps(data).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            url, data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                pass
            return True
        except (urllib.error.URLError, OSError):
            return False

    def _replay_to_server(self, until_line: int | None = None) -> bool:
        """Replay locally-written lines [_unsent_line, until_line) to server.

        Sends records in chunks of _REPLAY_CHUNK. Updates _unsent_line to
        reflect progress. Returns True if all pending records were sent.

        If until_line is None, replays up to current _total_lines.
        """
        with self._lines_lock:
            start = self._unsent_line
            end = until_line if until_line is not None else self._total_lines

        if start >= end:
            return True

        try:
            with open(self._metrics_path) as f:
                all_lines = f.readlines()
        except OSError:
            return False

        pending_lines = all_lines[start:end]
        if not pending_lines:
            return True

        records = []
        for line in pending_lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        sent = 0
        all_sent = True
        for i in range(0, len(records), _REPLAY_CHUNK):
            chunk = records[i:i + _REPLAY_CHUNK]
            ok = self._post_json_raw(
                "/run/replay",
                {
                    "experiment": self.experiment,
                    "run_name": self.run_name,
                    "records": chunk,
                },
            )
            if not ok:
                all_sent = False
                break
            sent += len(chunk)

        with self._lines_lock:
            self._unsent_line = start + sent

        return all_sent

    def _append_local(self, record: dict) -> None:
        """Append one JSONL record to local file and increment line count."""
        self._local_fh.write(json.dumps(record) + "\n")
        with self._lines_lock:
            self._total_lines += 1

    def _flush_worker(self) -> None:
        """Background thread: drain queue, write locally, POST to server."""
        last_heartbeat = time.time()

        while True:
            try:
                record = self._queue.get(timeout=_HEARTBEAT_INTERVAL)
            except queue.Empty:
                # Heartbeat tick (no records queued)
                self._update_heartbeat()
                last_heartbeat = time.time()
                continue

            if record is _STOP:
                break

            # Always write to local file first
            self._append_local(record)

            # Additionally POST to server if configured
            if self.server:
                prev_ok = self._server_ok
                ok = self._post_json(
                    "/metrics",
                    {
                        "run_name": self.run_name,
                        "experiment": self.experiment,
                        "record": record,
                    },
                )

                if ok:
                    if not prev_ok:
                        # Reconnect detected: replay records that server missed.
                        # The current record (at _total_lines - 1) was just POSTed,
                        # so replay only lines [_unsent_line, _total_lines - 1).
                        with self._lines_lock:
                            current_idx = self._total_lines - 1
                        replay_ok = self._replay_to_server(until_line=current_idx)
                        if replay_ok:
                            with self._lines_lock:
                                self._unsent_line = self._total_lines
                        # If replay partially failed: _unsent_line was advanced as far
                        # as it got. Next reconnect will replay from there.
                    else:
                        # Normal operation: this record was sent, advance pointer.
                        with self._lines_lock:
                            self._unsent_line = self._total_lines

            # Periodic heartbeat
            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                self._update_heartbeat()
                last_heartbeat = now

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Enqueue a metrics record. Never blocks the training loop."""
        record: dict[str, Any] = {"step": step, "t": time.time()}
        for k, v in metrics.items():
            key = k if self.tag is None else f"{self.tag}/{k}"
            record[key] = v
        self._queue.put(record)

    def close(self) -> None:
        """Drain queue, replay any unsent data, finalize run_meta.json."""
        # Signal flush thread to stop, wait for it to drain the queue
        self._queue.put(_STOP)
        self._flush_thread.join()

        # Replay any remaining locally-written records not yet on server
        if self.server and self._server_ok:
            with self._lines_lock:
                if self._unsent_line < self._total_lines:
                    self._replay_to_server()

        # Close persistent local file handle
        try:
            self._local_fh.close()
        except OSError:
            pass

        # Finalize run_meta.json
        end_time = time.time()
        try:
            with open(self._meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["end_time"] = end_time
        meta["status"] = "completed"
        self._write_meta_atomic(meta)

        # POST /run/end (best-effort)
        if self.server:
            self._post_json(
                "/run/end",
                {
                    "run_name": self.run_name,
                    "experiment": self.experiment,
                    "end_time": end_time,
                    "status": "completed",
                },
            )
