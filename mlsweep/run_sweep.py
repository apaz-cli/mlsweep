#!/usr/bin/env python3

"""Run experiment sweeps via mlsweep manager (HTTP/WebSocket client).

Usage:
    python -m mlsweep.run_sweep sweep.py --manager http://host:port [--stream] [--priority N]
    python -m mlsweep.run_sweep fetch --manager http://host:port --experiment EXP_ID
    python -m mlsweep.run_sweep watch EXP_ID --manager http://host:port
"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import secrets
import socket
import ssl
import struct
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from mlsweep._sweep import (
    _append_manifest_run,
    _write_manifest,
    count_expected,
    generate_variations,
    load_sweep_file,
    validate_options,
)
from mlsweep._writers import (
    MultiWriterFactory,
    WriterFactory,
)
from mlsweep._shared import _git_root

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _git_root(os.getcwd()) or os.getcwd()

# ANSI colors
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"
_DIM_COLORS = [_CYAN, _YELLOW, _MAGENTA, _BLUE]

_log_file = None


def sweep_print(msg: str, end: str = "\n") -> None:
    """Print to stdout (colored) and log file (plain)."""
    print(msg, end=end, flush=True)
    if _log_file is not None:
        _log_file.write(re.sub(r"\033\[[0-9;]*m", "", msg) + end)
        _log_file.flush()


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.0f}m"
    return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"


# ===============================================================================
# HTTP helpers
# ===============================================================================


def _http_request(
    method: str,
    url: str,
    token: str,
    *,
    json_data: Any = None,
    data: Any = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, Any]:
    """Make an HTTP request to the manager. Returns (status_code, parsed_response).

    Accepts JSON response and returns the parsed object or raw body for non-JSON.
    On error, prints a message and returns (status, None).
    """
    req_headers = {"Authorization": f"Bearer {token}"}
    if json_data is not None:
        data = json.dumps(json_data).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    if data is not None and "Content-Type" not in req_headers:
        req_headers["Content-Type"] = "application/octet-stream"
    if headers:
        req_headers.update(headers)

    req = Request(url, data=data, headers=req_headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except HTTPError as e:
        status = e.code
        raw = e.read()
    except URLError as e:
        sweep_print(f"{_RED}Error: cannot reach manager at {url}: {e.reason}{_RESET}")
        return (0, None)
    except Exception as e:
        sweep_print(f"{_RED}Error: HTTP request failed: {e}{_RESET}")
        return (0, None)

    # Parse JSON
    try:
        return (status, json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return (status, raw.decode("utf-8", errors="replace") if raw else None)


def _manager_url(manager: str, path: str) -> str:
    """Build a full URL from manager base and path."""
    base = manager.rstrip("/")
    return f"{base}{path}"


# ===============================================================================
# Minimal WebSocket client (stdlib only)
# ===============================================================================

_WS_OP_TEXT = 0x1
_WS_OP_CLOSE = 0x8
_WS_OP_PING = 0x9
_WS_OP_PONG = 0xA


class _WebSocket:
    """Minimal blocking WebSocket client using stdlib only.

    Handles connecting, upgrade handshake, reading text frames,
    sending ping/pong, and graceful close.
    """

    def __init__(self, ws_url: str, token: str, timeout: float = 15.0):
        p = urlparse(ws_url)
        self._host = p.hostname or "localhost"
        self._port = p.port or (443 if p.scheme == "wss" else 80)
        self._use_tls = p.scheme == "wss"
        self._path = p.path + ("?" + p.query if p.query else "")
        if not self._path.startswith("/"):
            self._path = "/" + self._path
        self._token = token
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._buf: bytes = b""

    def connect(self) -> None:
        """Establish WebSocket connection (TCP + TLS + upgrade handshake)."""
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        if self._use_tls:
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=self._host)

        # Build upgrade request
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        req = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
        )
        # Add auth via query param if not already present
        if "token=" not in self._path:
            req += f"Authorization: Bearer {self._token}\r\n"
        req += "\r\n"

        sock.sendall(req.encode())

        # Read HTTP response
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket handshake: no response")
            resp += chunk

        header, _ = resp.split(b"\r\n\r\n", 1)
        header_str = header.decode("utf-8", errors="replace")
        if "101" not in header_str.splitlines()[0] if header_str else "":
            raise ConnectionError(f"WebSocket upgrade rejected:\n{header_str}")

        # Store any leftover data after headers
        _, leftover = resp.split(b"\r\n\r\n", 1)
        self._buf = leftover
        self._sock = sock

    def recv_frame(self) -> tuple[int, bytes] | None:
        """Read one WebSocket frame. Returns (opcode, payload) or None on close/error."""
        if not self._sock:
            return None

        # Read 2-byte header
        while len(self._buf) < 2:
            chunk = self._recv_exact(2 - len(self._buf))
            if chunk is None:
                return None
            self._buf += chunk

        b0 = self._buf[0]
        b1 = self._buf[1]
        self._buf = self._buf[2:]

        opcode = b0 & 0x0F
        masked = (b1 & 0x80) != 0  # server frames are NOT masked
        length = b1 & 0x7F

        # Extended length
        if length == 126:
            while len(self._buf) < 2:
                chunk = self._recv_exact(2 - len(self._buf))
                if chunk is None:
                    return None
                self._buf += chunk
            length = struct.unpack("!H", self._buf[:2])[0]
            self._buf = self._buf[2:]
        elif length == 127:
            while len(self._buf) < 8:
                chunk = self._recv_exact(8 - len(self._buf))
                if chunk is None:
                    return None
                self._buf += chunk
            length = struct.unpack("!Q", self._buf[:8])[0]
            self._buf = self._buf[8:]

        # Masking key (should not be present on server frames, but be tolerant)
        if masked:
            while len(self._buf) < 4:
                chunk = self._recv_exact(4 - len(self._buf))
                if chunk is None:
                    return None
                self._buf += chunk
            mask_key = self._buf[:4]
            self._buf = self._buf[4:]
        else:
            mask_key = None

        # Payload
        while len(self._buf) < length:
            need = length - len(self._buf)
            chunk = self._recv_exact(need)
            if chunk is None:
                return None
            self._buf += chunk

        payload = self._buf[:length]
        self._buf = self._buf[length:]

        if mask_key:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))

        return (opcode, payload)

    def _recv_exact(self, n: int) -> bytes | None:
        """Receive exactly n bytes from socket, or None on EOF/error."""
        if not self._sock:
            return None
        try:
            self._sock.settimeout(self._timeout)
            data = self._sock.recv(n)
            if not data:
                return None
            return data
        except (socket.timeout, OSError):
            return None

    def send_frame(self, opcode: int, payload: bytes) -> None:
        """Send a masked WebSocket frame (client must mask)."""
        if not self._sock:
            return
        mask_key = secrets.token_bytes(4)
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))

        frame = bytearray()
        frame.append(0x80 | opcode)  # FIN + opcode
        length = len(payload)
        if length < 126:
            frame.append(0x80 | length)
        elif length < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack("!H", length))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack("!Q", length))
        frame.extend(mask_key)
        frame.extend(masked)

        try:
            self._sock.sendall(bytes(frame))
        except OSError:
            self._sock = None

    def send_ping(self) -> None:
        self.send_frame(_WS_OP_PING, b"")

    def send_close(self, code: int = 1000) -> None:
        payload = struct.pack("!H", code)
        self.send_frame(_WS_OP_CLOSE, payload)

    def close(self) -> None:
        if self._sock:
            try:
                self.send_close(1000)
            except Exception:
                pass
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ===============================================================================
# Manager operations
# ===============================================================================


def manager_create_experiment(
    manager: str,
    token: str,
    experiment_id: str,
    name: str,
    controller_id: str | None = None,
    note: str | None = None,
    expected_jobs: int = 0,
    singular_dims: list[str] | None = None,
) -> dict[str, Any] | None:
    """Create an experiment on the manager. Returns the experiment dict or None."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, "/api/experiments"),
        token,
        json_data={
            "experiment_id": experiment_id,
            "name": name,
            "controller_id": controller_id,
            "note": note,
            "status": "running",
            "expected_jobs": expected_jobs,
            "singular_dims": singular_dims or [],
        },
    )
    if status in (200, 201) and isinstance(resp, dict):
        sweep_print(f"  {_GREEN}OK{_RESET}    Experiment created: {experiment_id}")
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  Create experiment: {resp}")
    return None


def manager_register_artifact(
    manager: str,
    token: str,
    artifact_id: str,
    size_bytes: int,
    setup_command: str | None = None,
) -> dict[str, Any] | None:
    """Register an artifact on the manager."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, "/api/artifacts"),
        token,
        json_data={
            "artifact_id": artifact_id,
            "size_bytes": size_bytes,
            "setup_command": setup_command,
        },
    )
    if status in (200, 201) and isinstance(resp, dict):
        sweep_print(f"  {_GREEN}OK{_RESET}    Artifact registered: {artifact_id[:16]}...")
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  Register artifact: {resp}")
    return None


def manager_upload_artifact_data(
    manager: str,
    token: str,
    artifact_id: str,
    filepath: str | Path,
) -> bool:
    """Upload artifact tarball bytes to the manager.

    Uses PUT /api/artifacts/{artifact_id}/data with raw binary body.
    Streams the file in chunks to avoid loading the entire tarball into memory.
    """
    path = Path(filepath)
    if not path.exists():
        sweep_print(f"  {_RED}FAIL{_RESET}  Artifact file not found: {filepath}")
        return False

    file_size = path.stat().st_size

    status, resp = _http_request(
        "PUT",
        _manager_url(manager, f"/api/artifacts/{artifact_id}/data"),
        token,
        data=path.read_bytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(file_size),
        },
        timeout=120,
    )
    if status in (200, 201, 204):
        sweep_print(f"  {_GREEN}OK{_RESET}    Artifact uploaded ({file_size} bytes)")
        return True
    sweep_print(f"  {_RED}FAIL{_RESET}  Upload artifact: {resp}")
    return False


def manager_submit_jobs_bulk(
    manager: str,
    token: str,
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Submit multiple jobs in bulk. Returns list of created job records."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, "/api/jobs/bulk"),
        token,
        json_data=jobs,
    )
    if status in (200, 201) and isinstance(resp, list):
        sweep_print(f"  {_GREEN}OK{_RESET}    {len(resp)} jobs submitted")
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  Submit jobs: {resp}")
    return None


def manager_submit_job(
    manager: str,
    token: str,
    job: dict[str, Any],
) -> dict[str, Any] | None:
    """Submit a single job. Returns the created job record."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, "/api/jobs"),
        token,
        json_data=job,
    )
    if status in (200, 201) and isinstance(resp, dict):
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  Submit job: {resp}")
    return None


def manager_get_job_metrics(
    manager: str,
    token: str,
    experiment_id: str,
    run_id: str,
) -> list[dict[str, Any]] | None:
    """Fetch metrics.jsonl for a completed job."""
    status, resp = _http_request(
        "GET",
        _manager_url(manager, f"/api/experiments/{experiment_id}/jobs/{run_id}/metrics"),
        token,
    )
    if status == 200:
        if isinstance(resp, str):
            lines: list[dict[str, Any]] = []
            for line in resp.strip().splitlines():
                if line.strip():
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return lines
        if isinstance(resp, list):
            return resp
    return None


def manager_get_experiment_summary(
    manager: str,
    token: str,
    experiment_id: str,
) -> dict[str, Any] | None:
    """Get experiment summary from manager."""
    status, resp = _http_request(
        "GET",
        _manager_url(manager, f"/api/experiments/{experiment_id}/summary"),
        token,
    )
    if status == 200 and isinstance(resp, dict):
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  Get summary: {resp}")
    return None


def manager_list_experiment_jobs(
    manager: str,
    token: str,
    experiment_id: str,
    status_filter: str | None = None,
) -> list[dict[str, Any]] | None:
    """List jobs for an experiment."""
    path = f"/api/experiments/{experiment_id}/jobs"
    if status_filter:
        path += f"?status={status_filter}"
    status, resp = _http_request("GET", _manager_url(manager, path), token)
    if status == 200 and isinstance(resp, list):
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  List jobs: {resp}")
    return None


def manager_cancel_job(
    manager: str,
    token: str,
    run_id: str,
    experiment_id: str,
) -> dict[str, Any] | None:
    """Cancel a pending job."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, f"/api/jobs/{run_id}/cancel?experiment_id={experiment_id}"),
        token,
    )
    if status == 200 and isinstance(resp, dict):
        return resp
    return None


def manager_retry_job(
    manager: str,
    token: str,
    run_id: str,
    experiment_id: str,
) -> dict[str, Any] | None:
    """Retry a failed job."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, f"/api/jobs/{run_id}/retry?experiment_id={experiment_id}"),
        token,
    )
    if status == 200 and isinstance(resp, dict):
        return resp
    return None


def manager_check_artifact(
    manager: str,
    token: str,
    artifact_id: str,
) -> bool:
    """Check if an artifact already exists on the manager via HEAD request.

    Returns True if the artifact exists (HTTP 200), False otherwise.
    """
    url = _manager_url(manager, f"/api/artifacts/{artifact_id}")
    req = Request(url, method="HEAD")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.status == 200  # type: ignore[no-any-return]
    except HTTPError as e:
        if e.code == 404:
            return False
        sweep_print(f"  {_YELLOW}WARN{_RESET}  HEAD check for artifact failed: {e}")
        return False
    except URLError as e:
        sweep_print(f"  {_YELLOW}WARN{_RESET}  Cannot reach manager for artifact check: {e.reason}")
        return False
    except Exception as e:
        sweep_print(f"  {_YELLOW}WARN{_RESET}  Artifact check error: {e}")
        return False


def manager_download_experiment(
    manager: str,
    token: str,
    experiment_id: str,
    output_dir: str | Path,
) -> bool:
    """Download experiment results from the manager and extract to output_dir.

    Makes a GET request to /api/experiments/{experiment_id}/download
    and streams the tar.gz response, extracting it to output_dir.
    Returns True on success.
    """
    url = _manager_url(manager, f"/api/experiments/{experiment_id}/download")
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {token}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                sweep_print(f"  {_RED}FAIL{_RESET}  Download experiment: HTTP {resp.status}")
                return False
            sweep_print(f"  Downloading and extracting to {out}...")
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                tar.extractall(path=str(out))
        sweep_print(f"  {_GREEN}OK{_RESET}    Experiment downloaded to {out}")
        return True
    except HTTPError as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  Download experiment: HTTP {e.code}")
        return False
    except URLError as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  Cannot reach manager: {e.reason}")
        return False
    except tarfile.ReadError as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  Invalid tar stream: {e}")
        return False
    except Exception as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  Download error: {e}")
        return False


# ===============================================================================
# Artifact packer
# ===============================================================================


class _HashWriter:
    """Wrap a file object, copying all writes to a hasher for incremental hashing."""

    def __init__(self, f: Any, hasher: Any) -> None:
        self.f: Any = f
        self.hasher: Any = hasher

    def write(self, data: bytes) -> int:
        self.hasher.update(data)
        return self.f.write(data)  # type: ignore[no-any-return]

    def flush(self) -> None:
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _pack_project(
    project_root: str | Path,
    *,
    exclude_patterns: list[str] | None = None,
) -> tuple[str, str]:
    """Create a tar.gz of the project directory.

    Returns (tarball_path, sha256_hex).
    Skips common VCS and cache directories.
    """
    root = Path(project_root).resolve()
    if exclude_patterns is None:
        exclude_patterns = []

    excludes: set[str] = {
        ".git", ".svn", ".hg",
        "__pycache__", ".pyc", ".pyo",
        ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "node_modules",
        "outputs",
        ".venv", "venv", ".env",
        "*.egg-info", "*.dist-info",
        ".DS_Store",
        ".mlsweep",
    }
    for pat in exclude_patterns:
        excludes.add(pat)

    # Build tar in memory, then write to temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz", prefix="mlsweep_artifact_")
    os.close(tmp_fd)

    sha = hashlib.sha256()

    try:
        with open(tmp_path, "wb") as raw_f:
            hw = _HashWriter(raw_f, sha)
            with tarfile.open(fileobj=hw, mode="w:gz") as tar:  # type: ignore[call-overload]
                for entry in sorted(root.rglob("*")):
                    rel = entry.relative_to(root)
                    parts = rel.parts

                    skip = False
                    for part in parts:
                        if part in excludes:
                            skip = True
                            break

                    if skip:
                        continue
                    if entry.is_symlink() or entry.is_socket() or entry.is_fifo():
                        continue

                    try:
                        tar.add(str(entry), arcname=str(rel), recursive=False)
                    except (PermissionError, OSError):
                        continue

        # SHA-256 is computed incrementally during the write above

    except Exception:
        os.unlink(tmp_path)
        raise

    return (tmp_path, sha.hexdigest())


# ===============================================================================
# WebSocket status streaming
# ===============================================================================


def _ws_stream_url(manager: str, experiment_id: str, token: str) -> str:
    """Build WebSocket URL for experiment event stream."""
    http_url = manager.rstrip("/")
    if http_url.startswith("https://"):
        ws_url = "wss://" + http_url[8:]
    elif http_url.startswith("http://"):
        ws_url = "ws://" + http_url[7:]
    else:
        ws_url = "ws://" + http_url
    return f"{ws_url}/ws/experiments/{experiment_id}?token={token}"


def _stream_status_live(
    manager: str,
    token: str,
    experiment_id: str,
    *,
    max_idle: float = 120.0,
    on_event: Any = None,
    writer_factory: Any = None,
    variations: list[dict[str, Any]] | None = None,
    output_dir: str = "",
) -> None:
    """Connect to manager WebSocket and display live job status until idle timeout.

    If *on_event* is callable, it is invoked as ``on_event(event, manager, token, experiment_id)``
    for every received event.  This allows the Bayes controller to react to
    job completions (tell + suggest + submit) inline.

    If *writer_factory* is provided (e.g. MultiWriterFactory), run writers are
    created lazily for each run and fed metric / finish events.
    *variations* is used to look up combos when creating writers.
    """
    ws_url = _ws_stream_url(manager, experiment_id, token)

    ws = _WebSocket(ws_url, token, timeout=10.0)
    try:
        ws.connect()
    except Exception as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  WebSocket connection failed: {e}")
        sweep_print(f"  Check: {ws_url}")
        return

    sweep_print(f"  {_GREEN}OK{_RESET}    Streaming events from {experiment_id}\n")

    # Track job statuses
    job_status: dict[str, dict[str, Any]] = {}
    last_event = time.time()
    total_jobs = 0
    done_jobs = 0

    # Writer state
    run_writers: dict[str, Any] = {}
    if variations is None:
        variations = []

    if writer_factory is not None:
        try:
            dim_names = list(variations[0]["combo"].keys()) if variations else []
            run_ids = [v["name"] for v in variations]
            writer_factory.on_sweep_start(experiment_id, dim_names, run_ids)
        except Exception as e:
            sweep_print(f"  {_YELLOW}WARN{_RESET}  Writer on_sweep_start failed: {e}")

    # Helper: find combo for a run_id from variations
    def _combo_for(run_id: str) -> dict[str, Any]:
        for v in variations:
            if v["name"] == run_id:
                return v["combo"]  # type: ignore[no-any-return]
        return {}

    # Heartbeat thread
    stop_heartbeat = threading.Event()

    def _ping_loop() -> None:
        while not stop_heartbeat.is_set():
            try:
                ws.send_ping()
            except Exception:
                break
            stop_heartbeat.wait(20.0)

    heartbeat = threading.Thread(target=_ping_loop, daemon=True)
    heartbeat.start()

    try:
        while True:
            frame = ws.recv_frame()
            if frame is None:
                break

            opcode, payload = frame
            if opcode == _WS_OP_CLOSE:
                break
            elif opcode == _WS_OP_PONG:
                last_event = time.time()
                continue
            elif opcode == _WS_OP_PING:
                ws.send_frame(_WS_OP_PONG, payload)
                continue
            elif opcode != _WS_OP_TEXT:
                continue

            try:
                event = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            last_event = time.time()
            event_type = event.get("type", "unknown")
            run_id = event.get("run_id", "")

            if event_type in ("job_updated", "run_result", "job_done"):
                # run_result is the current manager event; job_done is the canonical name
                if event_type in ("run_result", "job_done"):
                    success = event.get("success", False)
                    status = "done" if success else "failed"
                else:
                    status = event.get("status", "")
                job_status[run_id] = {
                    "status": status,
                    "elapsed": event.get("elapsed"),
                    "exit_code": event.get("exit_code"),
                }
                if status in ("done", "failed", "cancelled"):
                    done_jobs += 1

                # ── Feed writer on_finish ──
                if writer_factory is not None and run_id and run_id in run_writers:
                    try:
                        elapsed = event.get("elapsed", 0.0)
                        run_writers[run_id].on_finish(status, elapsed)
                    except Exception as e:
                        sweep_print(f"  {_YELLOW}WARN{_RESET}  Writer on_finish failed for {run_id}: {e}")

            elif event_type == "metric":
                if writer_factory is not None and run_id:
                    try:
                        # Lazily create writer for this run
                        if run_id not in run_writers:
                            combo = _combo_for(run_id)
                            run_writers[run_id] = writer_factory.make(
                                run_id, combo, output_dir
                            )
                        data = event.get("data")
                        step = event.get("step", 0)
                        if data:
                            run_writers[run_id].on_metric(int(step), data)
                    except Exception as e:
                        sweep_print(f"  {_YELLOW}WARN{_RESET}  Writer on_metric failed for {run_id}: {e}")

            elif event_type in ("job_dispatched", "run_dispatched"):
                job_status[run_id] = {"status": "dispatched"}
                total_jobs = max(total_jobs, len(job_status))

            elif event_type in ("job_started", "run_started"):
                job_status[run_id] = {"status": "running"}

            elif event_type == "status_updated":
                sweep_print(f"\n  Experiment status: {event.get('status')}")
                if event.get("status") in ("completed", "aborted"):
                    break

            elif event_type == "experiment_done":
                sweep_print(f"\n  {_GREEN}Experiment complete!{_RESET}")
                break

            # ── Invoke callback for iterative Bayes / custom logic ──
            if callable(on_event):
                try:
                    on_event(event, manager, token, experiment_id)
                except Exception as e:
                    sweep_print(f"  {_RED}Error in event callback: {e}{_RESET}")

            # Print current status table
            _render_status_table(job_status, total_jobs)

            # Check idle timeout
            if time.time() - last_event > max_idle:
                sweep_print(f"\n  {_YELLOW}Idle for {max_idle:.0f}s — disconnecting{_RESET}")
                break

    except KeyboardInterrupt:
        sweep_print(f"\n  {_YELLOW}Interrupted{_RESET}")
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1.0)
        ws.close()

    # ── Finish writers ──
    if writer_factory is not None:
        # Finish any writers that haven't been finished yet
        for rid, w in run_writers.items():
            if rid in job_status:
                js = job_status[rid]
                if js["status"] in ("done", "failed", "cancelled"):
                    continue  # already called on_finish above
            try:
                w.on_finish("unknown", 0.0)
            except Exception:
                pass
        try:
            writer_factory.on_sweep_end()
        except Exception as e:
            sweep_print(f"  {_YELLOW}WARN{_RESET}  Writer on_sweep_end failed: {e}")

    # Final summary
    sweep_print(f"\n{'=' * 80}")
    ok = sum(1 for s in job_status.values() if s["status"] == "done")
    failed = sum(1 for s in job_status.values() if s["status"] == "failed")
    running = sum(1 for s in job_status.values() if s["status"] in ("pending", "dispatched", "running"))
    sweep_print(f"Final: {ok} OK, {failed} failed, {running} pending/running")
    sweep_print(f"{'=' * 80}")


def _render_status_table(
    job_status: dict[str, dict[str, Any]],
    total_jobs: int,
    max_display: int = 20,
) -> None:
    """Render a compact status table in-place (overwrite terminal lines)."""
    if not job_status:
        return

    # Sort: running first, then pending, then done
    priority_order = {"running": 0, "dispatched": 1, "pending": 2, "done": 3, "failed": 4, "cancelled": 5}

    sorted_jobs = sorted(job_status.items(), key=lambda x: priority_order.get(x[1]["status"], 99))

    # Build lines
    lines = []
    status_icons = {
        "done": f"{_GREEN}✓{_RESET}",
        "failed": f"{_RED}✗{_RESET}",
        "running": f"{_CYAN}▶{_RESET}",
        "dispatched": f"{_YELLOW}→{_RESET}",
        "pending": f"{_DIM_COLORS[0]}○{_RESET}",
        "cancelled": f"{_YELLOW}✕{_RESET}",
    }

    for run_id, info in sorted_jobs[:max_display]:
        status = info["status"]
        icon = status_icons.get(status, "?")
        elapsed = info.get("elapsed")
        time_str = f" {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
        lines.append(f"  {icon} {run_id}{time_str}")

    if len(job_status) > max_display:
        lines.append(f"  ... and {len(job_status) - max_display} more")

    # Count summary
    n_ok = sum(1 for s in job_status.values() if s["status"] == "done")
    n_fail = sum(1 for s in job_status.values() if s["status"] == "failed")
    n_run = sum(1 for s in job_status.values() if s["status"] in ("running", "dispatched"))
    n_pend = sum(1 for s in job_status.values() if s["status"] == "pending")

    summary = f"  [{n_ok} ok, {n_fail} fail, {n_run} running, {n_pend} pending]"
    lines.append(summary)

    # Clear and re-print (use \r\033[K for simple overwrite)
    # For multi-line, move cursor up
    output = "\n".join(lines)
    # Move cursor up by number of previous lines if we've printed before
    if hasattr(_render_status_table, "_prev_lines"):
        prev = _render_status_table._prev_lines  # pyright: ignore[reportFunctionMemberAccess]
        # Move up and clear
        sys.stdout.write(f"\033[{prev}A\033[J")
    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    _render_status_table._prev_lines = len(lines) + 1  # type: ignore[attr-defined]


_render_status_table._prev_lines = 0  # type: ignore[attr-defined]


# ===============================================================================
# Summary printer (from results, used by fetch)
# ===============================================================================


def print_jobs_summary(jobs: list[dict[str, Any]]) -> bool:
    """Print a summary of job records. Returns True if any failures."""
    if not jobs:
        sweep_print("  No jobs found.")
        return False

    n_ok = sum(1 for j in jobs if j["status"] == "done")
    n_fail = sum(1 for j in jobs if j["status"] == "failed")
    n_pending = sum(1 for j in jobs if j["status"] in ("pending", "dispatched", "running"))
    n_cancelled = sum(1 for j in jobs if j["status"] == "cancelled")
    total = len(jobs)

    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"SUMMARY — {total} jobs: {n_ok} OK, {n_fail} failed, "
                f"{n_pending} pending, {n_cancelled} cancelled")
    sweep_print(f"{'=' * 80}")

    for j in jobs:
        status = j["status"]
        run_id = j["run_id"]
        elapsed = j["elapsed"]
        elapsed_str = f" ({elapsed:.1f}s)" if isinstance(elapsed, (int, float)) else ""

        if status == "done":
            sweep_print(f"  {_GREEN}   OK{_RESET}  {run_id}{elapsed_str}")
        elif status == "failed":
            exit_code = j["exit_code"]
            sweep_print(f"  {_RED} FAIL{_RESET}  {run_id} (exit {exit_code}){elapsed_str}")
        elif status in ("pending", "dispatched", "running"):
            sweep_print(f"  {_YELLOW}{status.upper()}{_RESET}  {run_id}")
        elif status == "cancelled":
            sweep_print(f"  {_YELLOW}CANCEL{_RESET}  {run_id}")
        else:
            sweep_print(f"  {status}  {run_id}")

    return n_fail > 0


# ===============================================================================
# Job payload builder
# ===============================================================================


def _build_job_payloads(
    variations: list[dict[str, Any]],
    experiment_id: str,
    artifact_id: str,
    command: list[str],
    extra_flags: list[str],
    gpus_per_run: int,
    nodes_per_run: int,
    set_dist_env: bool,
    run_from: str | None,
    priority: int,
    max_retries: int,
    setup_command: str | None = None,
    jobs_per_gpu: int = 1,
) -> list[dict[str, Any]]:
    """Convert sweep variations into job payloads for the manager API.

    Each payload is a dict with keys matching the JobRecord fields expected
    by POST /api/jobs/bulk.
    """
    jobs = []
    for var in variations:
        full_command = list(command) + var["overrides"] + list(extra_flags)
        env_dict: dict[str, str] = {}
        tag_parts = [f"{k}={v}" for k, v in var["combo"].items() if v is not None]
        if tag_parts:
            env_dict["EXP_TAGS"] = ",".join(tag_parts)

        job = {
            "run_id": var["name"],
            "experiment_id": experiment_id,
            "priority": priority,
            "command": full_command,
            "combo": var["combo"],
            "env": env_dict,
            "status": "pending",
            "gpus_per_run": gpus_per_run,
            "nodes_per_run": nodes_per_run,
            "set_dist_env": set_dist_env,
            "run_from": run_from,
            "artifact_id": artifact_id,
            "max_retries": max_retries,
            "jobs_per_gpu": jobs_per_gpu,
            "return_files": [],  # could configure via sweep file
        }
        if setup_command:
            job["setup_command"] = setup_command
        jobs.append(job)
    return jobs


# ===============================================================================
# Watch subcommand
# ===============================================================================


def _watch_cmd(args: list[str]) -> None:
    """Watch an experiment's progress via WebSocket event stream."""
    parser = argparse.ArgumentParser(
        prog="run_sweep.py watch",
        description="Watch experiment progress via WebSocket",
    )
    parser.add_argument("experiment_id", help="Experiment ID to watch")
    parser.add_argument("--manager", required=True, help="Manager URL (http://host:port)")
    parser.add_argument("--token", default=None, help="Manager auth token (or set MLSWEEP_TOKEN env)")
    parsed = parser.parse_args(args)

    token = parsed.token or os.environ.get("MLSWEEP_TOKEN", "")
    if not token:
        sweep_print(f"{_RED}Error: --token is required (or set MLSWEEP_TOKEN env){_RESET}")
        sys.exit(1)

    manager = parsed.manager.rstrip("/")

    # Connect WebSocket with since=now to only get new events
    ws_url = _ws_stream_url(manager, parsed.experiment_id, token)
    # Append since parameter for new events only
    since = time.time()
    if "?" in ws_url:
        ws_url += f"&since={since}"
    else:
        ws_url += f"?since={since}"

    ws = _WebSocket(ws_url, token, timeout=10.0)
    try:
        ws.connect()
    except Exception as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  WebSocket connection failed: {e}")
        sys.exit(1)

    sweep_print(f"Watching experiment {_YELLOW}{parsed.experiment_id}{_RESET}")
    sweep_print(f"Manager: {manager}")
    sweep_print("")

    stop_heartbeat = threading.Event()

    def _ping_loop() -> None:
        while not stop_heartbeat.is_set():
            try:
                ws.send_ping()
            except Exception:
                break
            stop_heartbeat.wait(20.0)

    heartbeat = threading.Thread(target=_ping_loop, daemon=True)
    heartbeat.start()

    try:
        while True:
            frame = ws.recv_frame()
            if frame is None:
                break

            opcode, payload = frame
            if opcode == _WS_OP_CLOSE:
                break
            elif opcode == _WS_OP_PONG:
                continue
            elif opcode == _WS_OP_PING:
                ws.send_frame(_WS_OP_PONG, payload)
                continue
            elif opcode != _WS_OP_TEXT:
                continue

            try:
                event = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            event_type = event.get("type", "unknown")
            run_id = event.get("run_id", "")

            if event_type in ("job_started", "run_started"):
                sweep_print(f"  {_CYAN}▶ START {_RESET} {run_id}")
                worker = event.get("worker_id", "")
                if worker:
                    sweep_print(f"         on worker {worker}")

            elif event_type in ("job_done", "run_result"):
                success = event.get("success", False)
                elapsed = event.get("elapsed")
                elapsed_str = f" ({elapsed:.1f}s)" if isinstance(elapsed, (int, float)) else ""
                if success:
                    sweep_print(f"  {_GREEN}✓ DONE {_RESET} {run_id}{elapsed_str}")
                else:
                    sweep_print(f"  {_RED}✗ FAIL {_RESET} {run_id}{elapsed_str}")

            elif event_type in ("job_dispatched", "run_dispatched"):
                sweep_print(f"  {_YELLOW}→ DISPATCHED {_RESET} {run_id}")

            elif event_type == "metric":
                data = event.get("data", {})
                step = event.get("step", "")
                pairs = ", ".join(f"{k}={v}" for k, v in data.items())
                sweep_print(f"  {_MAGENTA}📊 METRIC {_RESET} {run_id}: {pairs}"
                            + (f" (step {step})" if step else ""))

            elif event_type == "experiment_done":
                sweep_print(f"\n  {_GREEN}Experiment complete!{_RESET}")
                break

            elif event_type == "status_updated":
                sweep_print(f"\n  Experiment status: {event.get('status')}")

    except KeyboardInterrupt:
        sweep_print(f"\n  {_YELLOW}Interrupted{_RESET}")
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1.0)
        ws.close()


# ===============================================================================
# Fetch subcommand
# ===============================================================================


def _fetch_cmd(args: list[str]) -> None:
    """Fetch experiment results from a manager and print summary."""
    parser = argparse.ArgumentParser(
        prog="run_sweep.py fetch",
        description="Fetch experiment results from mlsweep manager",
    )
    parser.add_argument("--manager", required=True, help="Manager URL (http://host:port)")
    parser.add_argument("--experiment", required=True, help="Experiment ID to fetch")
    parser.add_argument("--token", default=None, help="Manager auth token (or set MLSWEEP_TOKEN env)")
    parser.add_argument("--output-dir", default=None, help="Directory to download artifacts (optional)")
    parser.add_argument("--status", default=None, help="Filter jobs by status (done, failed, pending, etc.)")
    parsed = parser.parse_args(args)

    token = parsed.token or os.environ.get("MLSWEEP_TOKEN", "")
    if not token:
        sweep_print(f"{_RED}Error: --token is required (or set MLSWEEP_TOKEN env){_RESET}")
        sys.exit(1)

    manager = parsed.manager.rstrip("/")

    # Get experiment summary
    summary = manager_get_experiment_summary(manager, token, parsed.experiment)
    if summary:
        sweep_print(f"Experiment: {summary['name']}")
        sweep_print(f"Status:     {summary['status']}")
        counts = summary["job_counts"]
        if counts:
            sweep_print(f"Jobs:       {counts}")
        sweep_print("")

    # Get jobs
    jobs = manager_list_experiment_jobs(manager, token, parsed.experiment, status_filter=parsed.status)
    if jobs is not None:
        print_jobs_summary(jobs)
    else:
        sys.exit(1)

    # Download experiment artifacts
    output_dir = parsed.output_dir or os.path.join(os.getcwd(), "mlsweep_downloads", parsed.experiment)
    manager_download_experiment(manager, token, parsed.experiment, output_dir)


# ===============================================================================
# Main
# ===============================================================================


def main() -> None:
    global _log_file

    argv = sys.argv[1:]

    # ── subcommands ───────────────────────────────────────────────────────
    if argv and argv[0] == "fetch":
        _fetch_cmd(argv[1:])
        return
    if argv and argv[0] == "watch":
        _watch_cmd(argv[1:])
        return

    # ── argparse ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Run experiment sweeps via mlsweep manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Extra args after -- are passed to every training run.\n\n"
            "Environment variables:\n"
            "  MLSWEEP_TOKEN  Authentication token for manager\n"
        ),
    )
    parser.add_argument("sweep_file", help="Path to sweep .py file")
    parser.add_argument("--manager", default=None,
                        help="Manager URL (http://host:port)")
    parser.add_argument("--token", default=None,
                        help="Manager auth token (or set MLSWEEP_TOKEN env)")
    parser.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "outputs", "sweeps"),
                        help="Output directory for local artifacts")
    parser.add_argument("--experiment", default=None,
                        help="Experiment name (default: <sweep>_<timestamp>)")
    parser.add_argument("--resume", default=None,
                        help="Resume an existing experiment (experiment_id)")
    parser.add_argument("--note", default=None,
                        help="Human-readable note stored with the experiment")
    parser.add_argument("--priority", type=int, default=0,
                        help="Job priority (higher = earlier, default: 0)")
    parser.add_argument("--stream", action="store_true",
                        help="Subscribe to WebSocket event stream for live status")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch results after submission (if --stream not used)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without execution")
    parser.add_argument("--validate", action="store_true",
                        help="Validate sweep config, print all combinations, and exit")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries for failed jobs (default: 2)")
    parser.add_argument("--setup-command", default=None,
                        help="Shell command executed before training in the worker workspace")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (enables wandb logging)")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity/team")
    parser.add_argument("--tensorboard-dir", default=None,
                        help="TensorBoard output directory (enables TensorBoard logging)")
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {importlib.metadata.version('mlsweep')}")
    parser.add_argument(
        "-j", "--jobs-per-gpu", type=int, default=1, metavar="N",
        help="Max concurrent jobs per GPU for this sweep (default: 1)")

    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    # ── Writer factories ────────────────────────────────────────────────────
    writer_factory = None
    if args.wandb_project or args.tensorboard_dir:
        factories: list[WriterFactory] = []
        if args.wandb_project:
            from mlsweep._writer_wandb import WandbWriterFactory
            factories.append(WandbWriterFactory(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
            ))
        if args.tensorboard_dir:
            from mlsweep._writer_tensorboard import TensorBoardWriterFactory
            factories.append(TensorBoardWriterFactory(tb_dir=args.tensorboard_dir))
        writer_factory = MultiWriterFactory(factories)

    # ── Load sweep ─────────────────────────────────────────────────────────
    info = load_sweep_file(args.sweep_file)
    sweep_name = info["name"]
    options = info["options"]
    command = info["command"]
    exclude_fn = info["exclude"]
    extra_flags: list[str] = list(info.get("extra_flags", []))
    gpus_per_run: int = info.get("gpus_per_run", 1)
    nodes_per_run: int = info.get("nodes_per_run", 1)
    run_from: str | None = info.get("run_from")
    set_dist_env: bool = info.get("set_dist_env", False)
    method: str = info.get("method", "grid")
    optimize_cfg: dict[str, Any] = info.get("optimize") or {}

    validate_options(options, method=method)
    assert options is not None and command is not None

    # ── Validate mode ──────────────────────────────────────────────────────
    if args.validate:
        if method == "bayes":
            sweep_print(f"Sweep: {sweep_name} (bayes, budget={optimize_cfg['budget']})")
            sweep_print(f"Metric: {optimize_cfg['metric']} ({optimize_cfg['goal']})")
            sweep_print(f"Dimensions ({len(options)}):")
            for key, opt in options.items():
                dim = key[1:]
                if opt.get("_type") == "continuous":
                    sweep_print(f"  {dim}: {opt['distribution']} [{opt['min']}, {opt['max']}]"
                                + (" [singular]" if opt.get("singular") else ""))
                elif opt["_values"] != [None]:
                    sweep_print(f"  {dim}: {opt['_values']}"
                                + (" [singular]" if opt.get("singular") else ""))
            sys.exit(0)

        all_variations = generate_variations(sweep_name, options, exclude_fn, extra_flags)
        expected = count_expected(options)
        excluded = expected - len(all_variations)
        dim_names = [k[1:] for k in options]
        sweep_print(f"Sweep: {sweep_name}")
        sweep_print(f"Dimensions: {', '.join(dim_names) if dim_names else '(none)'}")
        for key in options:
            dim_name = key[1:]
            values = options[key].get("_values", [])
            if values != [None]:
                sweep_print(f"  {dim_name}: {values}")
        sweep_print(f"\nTotal combinations: {len(all_variations)}")
        if excluded:
            sweep_print(f"Excluded by EXCLUDE filter: {excluded}")
        sweep_print(f"\nRuns:")
        for var in all_variations:
            sweep_print(f"  {var['name']}: {var['combo']}")
        sys.exit(0)

    # ── Generate variations ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    rand_suffix = secrets.token_hex(2)  # 4 hex chars
    resume = args.resume is not None
    if resume and method != "bayes":
        sweep_print(f"{_RED}Error: --resume is only supported for method='bayes'{_RESET}")
        sys.exit(1)
    if resume:
        experiment_id = args.resume
    else:
        experiment_id = args.experiment or f"{sweep_name}_{timestamp}_{rand_suffix}"
    output_dir = os.path.abspath(args.output_dir)
    exp_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    if not args.dry_run:
        _log_file = open(os.path.join(exp_dir, "sweep.log"), "w")

    if method == "bayes":
        from mlsweep._bayes import BayesianOptimizer
        optimizer = BayesianOptimizer(
            sweep_name, options, optimize_cfg, extra_flags=list(extra_flags)
        )
        budget: int = optimize_cfg["budget"]
        expected = budget
        done_jobs: list[dict[str, Any]] = []
        variations: list[dict[str, Any]] = []
    else:
        optimizer = None
        variations = generate_variations(sweep_name, options, exclude_fn, extra_flags)
        expected = count_expected(options)
        budget = len(variations)
        done_jobs = []

    # ── Header (before network calls) ──────────────────────────────────────
    sweep_print(f"Command: {' '.join(command)}")
    if method == "bayes":
        sweep_print(f"Sweep: {sweep_name} (bayes, budget={expected})")
    else:
        n_probes = len(variations)
        n_expected = expected  # count_expected: ignores singular probes
        if n_expected < n_probes:
            sweep_print(f"Sweep: {sweep_name} ({n_expected}–{n_probes} runs, {n_probes - n_expected} singular probes)")
        else:
            sweep_print(f"Sweep: {sweep_name} ({n_probes} runs)")
    sweep_print(f"Experiment: {experiment_id}")
    if extra:
        sweep_print(f"Extra overrides: {' '.join(extra)}")

    if args.dry_run:
        if method == "bayes":
            assert optimizer is not None
            n_initial = optimizer.n_initial
            variations = optimizer.suggest(n=n_initial)
        for var in variations:
            colored = []
            for ci, (key, val) in enumerate(var["combo"].items()):
                flags = var["effective_options"].get(key, {}).get("_flags", {}).get(val, [])
                if flags:
                    color = _DIM_COLORS[ci % len(_DIM_COLORS)]
                    colored.append(f"{color}{' '.join(flags)}{_RESET}")
            sweep_print(f"{_GREEN}{var['name']}{_RESET}: {' '.join(colored)}")
            sweep_print(f"{' '.join(list(command) + var['overrides'] + list(extra))}\n")
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"DRY RUN — {len(variations)} runs would be submitted to manager")
        sweep_print(f"{'=' * 80}")
        return

    # ── Manager required ───────────────────────────────────────────────────
    if not args.manager:
        sweep_print(f"\n{_RED}Error: --manager URL is required.{_RESET}")
        sweep_print(f"Usage: run_sweep.py <sweep.py> --manager http://host:port [--stream]")
        sys.exit(1)

    manager = args.manager.rstrip("/")
    token = args.token or os.environ.get("MLSWEEP_TOKEN", "")
    if not token:
        # Auto-detect token from local manager file
        token_file = Path("~/.mlsweep/manager.token").expanduser()
        if token_file.exists():
            token = token_file.read_text().strip()
        if not token:
            sweep_print(f"\n{_RED}Error: --token is required (or set MLSWEEP_TOKEN env, "
                        f"or place token in ~/.mlsweep/manager.token){_RESET}")
            sys.exit(1)

    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"Connecting to manager: {manager}")
    sweep_print(f"{'=' * 80}\n")

    # ── Resume: fetch completed jobs and rebuild optimizer ─────────────────
    if resume:
        sweep_print("Resuming experiment...")
        summary = manager_get_experiment_summary(manager, token, experiment_id)
        if summary is None:
            sweep_print(f"  {_RED}FAIL{_RESET}  Cannot fetch experiment summary — is manager reachable?")
            sys.exit(1)
        sweep_print(f"  Experiment: {summary['name']}")
        sweep_print(f"  Status:     {summary['status']}")

        done_jobs = manager_list_experiment_jobs(manager, token, experiment_id, status_filter="done")  # type: ignore[assignment]
        if done_jobs is None:
            done_jobs = []
        sweep_print(f"  Completed jobs: {len(done_jobs)}")

        # Rebuild optimizer from completed jobs
        assert optimizer is not None
        metric_name = optimize_cfg["metric"]
        goal = optimize_cfg["goal"]
        told_count = 0
        for job in done_jobs:
            combo = job["combo"]
            if isinstance(combo, str):
                try:
                    combo = json.loads(combo)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(combo, dict):
                continue
            run_id = job["run_id"]
            metrics_list = manager_get_job_metrics(manager, token, experiment_id, run_id)
            if metrics_list is None:
                continue
            # Extract best metric value
            best: float | None = None
            for row in metrics_list:
                val = row.get(metric_name)
                if isinstance(val, (int, float)):
                    best_val = float(val)
                    if best is None or (goal == "minimize" and best_val < best) or (goal == "maximize" and best_val > best):
                        best = best_val
            if best is not None:
                optimizer.tell(combo, best)
                told_count += 1
            else:
                # No metric found — tell as failure so TPE can learn
                optimizer.tell(combo, None)
        sweep_print(f"  Replayed {told_count} metrics into optimizer")

        # Generate remaining suggestions
        remaining = budget - optimizer._told
        if remaining <= 0:
            sweep_print(f"  {_GREEN}Budget exhausted — all {budget} jobs already done.{_RESET}")
            if _log_file:
                _log_file.close()
            return
        variations = optimizer.suggest(n=remaining)
        sweep_print(f"  Submitting {len(variations)} new job(s) (budget {budget}, {optimizer._told} told)")
    elif method == "bayes":
        # Fresh Bayes: if streaming, submit first job only (iterative loop
        # in callback submits the rest).  Otherwise submit all budget upfront.
        assert optimizer is not None
        if args.stream:
            variations = optimizer.suggest(n=1)
        else:
            variations = optimizer.suggest(n=budget)

    n = len(variations)

    # ── List variations ────────────────────────────────────────────────────
    for var in variations:
        colored = []
        for ci, (key, val) in enumerate(var["combo"].items()):
            flags = var["effective_options"].get(key, {}).get("_flags", {}).get(val, [])
            if flags:
                color = _DIM_COLORS[ci % len(_DIM_COLORS)]
                colored.append(f"{color}{' '.join(flags)}{_RESET}")
        sweep_print(f"{_GREEN}{var['name']}{_RESET}: {' '.join(colored)}")
        sweep_print(f"{' '.join(list(command) + var['overrides'] + list(extra))}\n")

    # ── 1. Pack and upload artifact ────────────────────────────────────────
    if resume:
        sweep_print("Resuming — skipping artifact upload")
        # Fetch artifact_id from a completed job
        artifact_id = None
        if done_jobs:
            artifact_id = done_jobs[0]["artifact_id"]
        if not artifact_id:
            sweep_print(f"  {_YELLOW}WARN{_RESET}  Cannot determine artifact_id; jobs may fail without artifact")
    else:
        sweep_print("Packing project artifact...")
        try:
            tarball_path, artifact_hash = _pack_project(_PROJECT_ROOT)
            sweep_print(f"  Artifact created: {os.path.basename(tarball_path)} "
                        f"({os.path.getsize(tarball_path)} bytes, sha256={artifact_hash[:16]}...)")
        except Exception as e:
            sweep_print(f"  {_RED}FAIL{_RESET}  Cannot pack project: {e}")
            sys.exit(1)

        artifact_id = f"sha256:{artifact_hash}"

        # Check if artifact already exists on manager
        if manager_check_artifact(manager, token, artifact_id):
            sweep_print(f"  {_GREEN}OK{_RESET}    Artifact already on manager, skipping upload")
            try:
                os.unlink(tarball_path)
            except OSError:
                pass
        else:
            # Register artifact
            if not manager_register_artifact(
                manager, token, artifact_id,
                size_bytes=os.path.getsize(tarball_path),
                setup_command=getattr(args, "setup_command", None),
            ):
                os.unlink(tarball_path)
                sys.exit(1)

            # Upload artifact data
            if not manager_upload_artifact_data(manager, token, artifact_id, tarball_path):
                os.unlink(tarball_path)
                sys.exit(1)

            # Clean up tarball
            try:
                os.unlink(tarball_path)
            except OSError:
                pass

    # ── 2. Create experiment ───────────────────────────────────────────────
    if resume:
        sweep_print("Resuming — experiment already exists")
    else:
        singular_dim_names = [k[1:] for k, v in options.items() if v.get("singular")]
        if not manager_create_experiment(
            manager, token, experiment_id,
            name=sweep_name,
            note=args.note,
            expected_jobs=expected if method == "bayes" else 0,
            singular_dims=singular_dim_names,
        ):
            sys.exit(1)

    # ── 3. Build and submit jobs ───────────────────────────────────────────
    if n > 0:
        sweep_print("Submitting jobs...")
        job_payloads = _build_job_payloads(
            variations=variations,
            experiment_id=experiment_id,
            artifact_id=artifact_id or "",
            command=command,
            extra_flags=extra_flags,
            gpus_per_run=gpus_per_run,
            nodes_per_run=nodes_per_run,
            set_dist_env=set_dist_env,
            run_from=run_from,
            priority=args.priority,
            max_retries=args.max_retries,
            setup_command=args.setup_command,
            jobs_per_gpu=args.jobs_per_gpu,
        )

        records = manager_submit_jobs_bulk(manager, token, job_payloads)
        if records is None:
            sys.exit(1)
    else:
        sweep_print("No new jobs to submit.")

    # ── 4. Write local manifest ────────────────────────────────────────────
    _write_manifest(exp_dir, experiment_id, variations, note=args.note)
    for var in variations:
        _append_manifest_run(exp_dir, var)

    # ── 5. Stream or fetch ─────────────────────────────────────────────────
    if args.stream:
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"Streaming live status (Ctrl+C to stop)")
        sweep_print(f"{'=' * 80}\n")

        if method == "bayes" and optimizer is not None:
            # Build iterative Bayes callback
            metric_name = optimize_cfg["metric"]
            goal = optimize_cfg["goal"]

            # Track probes per lex combo so we only tell/suggest once per
            # completed lex evaluation (not once per singular probe).
            # singular probes all share the same lex_key.
            _sing_dim_names = frozenset(k[1:] for k in optimizer._singular_options)

            def _lex_key(c: dict[str, Any]) -> tuple[tuple[str, str], ...]:
                return tuple((k, str(c[k])) for k in sorted(c) if k not in _sing_dim_names)

            _lex_pending: dict[tuple[tuple[str, str], ...], int] = {}   # lex_key → outstanding probes
            _lex_done: set[tuple[tuple[str, str], ...]] = set()          # lex_keys already told

            def _register_vars(vs: list[dict[str, Any]]) -> None:
                for v in vs:
                    lk = _lex_key(v["combo"])
                    _lex_pending[lk] = _lex_pending.get(lk, 0) + 1

            _register_vars(variations)

            def _submit_new(eid: str) -> None:
                assert optimizer is not None
                if optimizer.exhausted:
                    return
                new_vars = optimizer.suggest(n=1)
                if not new_vars:
                    return
                new_job = _build_job_payloads(
                    variations=new_vars,
                    experiment_id=eid,
                    artifact_id=artifact_id or "",
                    command=command,
                    extra_flags=extra_flags,
                    gpus_per_run=gpus_per_run,
                    nodes_per_run=nodes_per_run,
                    set_dist_env=set_dist_env,
                    run_from=run_from,
                    priority=args.priority,
                    max_retries=args.max_retries,
                    setup_command=args.setup_command,
                    jobs_per_gpu=args.jobs_per_gpu,
                )
                if new_job:
                    submitted = manager_submit_jobs_bulk(manager, token, new_job)
                    if submitted:
                        variations.extend(new_vars)
                        _register_vars(new_vars)
                        names = ", ".join(v["name"] for v in new_vars)
                        sweep_print(f"  {_CYAN}→ SUBMITTED {_RESET} {names} "
                                    f"(told: {optimizer._told}/{budget})")

            def _bayes_on_event(
                event: dict[str, Any],
                mgr: str,
                tok: str,
                eid: str,
            ) -> None:
                nonlocal optimizer, variations
                assert optimizer is not None
                # Only act on job completions
                event_type = event.get("type", "")
                if event_type not in ("job_done", "run_result"):
                    return
                # If budget already exhausted, nothing to do
                if optimizer.exhausted:
                    return
                run_id = event.get("run_id", "")
                success = event.get("success", False)
                # Try to find the combo from local variations first
                combo = None
                for v in variations:
                    if v["name"] == run_id:
                        combo = v["combo"]
                        break
                if combo is None:
                    # Fetch job from manager to get combo
                    status_code, job_resp = _http_request(
                        "GET",
                        _manager_url(mgr, f"/api/jobs/{run_id}?experiment_id={eid}"),
                        tok,
                    )
                    if status_code == 200 and isinstance(job_resp, dict):
                        combo_raw = job_resp["combo"]
                        if isinstance(combo_raw, str):
                            try:
                                combo = json.loads(combo_raw)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        elif isinstance(combo_raw, dict):
                            combo = combo_raw

                if combo is None:
                    return

                lk = _lex_key(combo)

                # Decrement the outstanding probe count for this lex combo.
                _lex_pending[lk] = max(0, _lex_pending.get(lk, 1) - 1)

                # If this lex combo is already told (a sibling probe already
                # reported), do nothing — only suggest/tell once per lex combo.
                if lk in _lex_done:
                    return

                if success:
                    # First success for this lex combo: tell optimizer and
                    # immediately submit a replacement.
                    metrics_list = manager_get_job_metrics(mgr, tok, eid, run_id)
                    best: float | None = None
                    if metrics_list:
                        for row in metrics_list:
                            val = row.get(metric_name)
                            if isinstance(val, (int, float)):
                                best_val = float(val)
                                if best is None or (goal == "minimize" and best_val < best) or (goal == "maximize" and best_val > best):
                                    best = best_val
                    optimizer.tell(combo, best)
                    _lex_done.add(lk)
                    _submit_new(eid)
                elif _lex_pending.get(lk, 0) == 0:
                    # All probes for this lex combo failed — tell optimizer and
                    # submit a replacement.
                    optimizer.tell(combo, None)
                    _lex_done.add(lk)
                    _submit_new(eid)
                # else: more singular probes are still pending; wait.

            _stream_status_live(manager, token, experiment_id,
                                on_event=_bayes_on_event,
                                writer_factory=writer_factory,
                                variations=variations,
                                output_dir=exp_dir)
        else:
            _stream_status_live(manager, token, experiment_id,
                                writer_factory=writer_factory,
                                variations=variations,
                                output_dir=exp_dir)
    elif args.fetch:
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"Fetching results...")
        sweep_print(f"{'=' * 80}")
        jobs = manager_list_experiment_jobs(manager, token, experiment_id)
        if jobs is not None:
            print_jobs_summary(jobs)
        # Download experiment artifacts
        manager_download_experiment(manager, token, experiment_id, exp_dir)
    else:
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"{n} jobs submitted.")
        sweep_print(f"Monitor: {manager}/?token={token}")
        sweep_print(f"Or use: run_sweep.py fetch --manager {manager} --experiment {experiment_id}")
        sweep_print(f"{'=' * 80}")

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
