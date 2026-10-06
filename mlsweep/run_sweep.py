#!/usr/bin/env python3

"""Run experiment sweeps via mlsweep manager (HTTP/WebSocket client).

Usage:
    python -m mlsweep.run_sweep sweep.py --manager http://host:port [--stream] [--priority N]
    python -m mlsweep.run_sweep fetch --manager http://host:port --experiment EXP_ID
    python -m mlsweep.run_sweep watch EXP_ID --manager http://host:port
"""

import argparse
import base64
import collections
import functools
import hashlib
import importlib.metadata
import json
import math
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from mlsweep._sweep import (
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
from mlsweep._shared import (
    DEFAULT_CAMPAIGN, DEFAULT_MANAGER_URL, _BOLD, _GREEN, _RED, _YELLOW, _CYAN, _MAGENTA, _BLUE,
    _RESET, _git_root, _mlsweep_dir, set_color, strip_color_flag, validate_campaign,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _git_root(os.getcwd()) or os.getcwd()

_DIM_COLORS = [_CYAN, _YELLOW, _MAGENTA, _BLUE]

_log_file = None


def sweep_print(msg: str, end: str = "\n") -> None:
    """Print to stdout (colored) and log file (plain)."""
    print(msg, end=end, flush=True)
    if _log_file is not None:
        _log_file.write(re.sub(r"\033\[[0-9;]*m", "", msg) + end)
        _log_file.flush()


def _resolve_token(token_arg: str | None) -> str:
    """Resolve the manager token from explicit arg, env, or $MLSWEEP_DIR/manager.token."""
    token = token_arg if token_arg else os.environ.get("MLSWEEP_TOKEN", "")
    if not token:
        token_file = _mlsweep_dir() / "manager.token"
        if token_file.exists():
            token = token_file.read_text().strip()
    return token


def _require_token(token_arg: str | None) -> str:
    """Like _resolve_token, but exit with an error if no token is found."""
    token = _resolve_token(token_arg)
    if not token:
        sweep_print(f"{_RED}Error: --token is required (or set MLSWEEP_TOKEN env, "
                    f"or place token in {_mlsweep_dir() / 'manager.token'}){_RESET}")
        sys.exit(1)
    return token


def _manager_token(args: argparse.Namespace) -> tuple[str, str]:
    """Return (manager, token) resolved from parsed args."""
    return args.manager.rstrip("/"), _require_token(args.token)


def _add_manager_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared --manager / --token / --campaign / --all-campaigns flags."""
    parser.add_argument("--manager", default=os.environ.get("MLSWEEP_MANAGER", DEFAULT_MANAGER_URL),
                        help=f"Manager URL (env: MLSWEEP_MANAGER, default: {DEFAULT_MANAGER_URL})")
    parser.add_argument("--token", default=None,
                        help="Manager auth token (or set MLSWEEP_TOKEN env)")
    _add_campaign_args(parser)


def _add_campaign_args(parser: argparse.ArgumentParser) -> None:
    """Add --campaign NAME and --all-campaigns (also spelled --all_campaigns)."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--campaign", default=None, metavar="NAME",
                       help=f"Campaign to work in (env: MLSWEEP_CAMPAIGN, default: {DEFAULT_CAMPAIGN})")
    group.add_argument("--all-campaigns", "--all_campaigns", dest="all_campaigns",
                       action="store_true", help="Work across every campaign")


def _resolve_campaign(args: argparse.Namespace) -> str | None:
    """The campaign a command works in, or None for --all-campaigns.

    Order: --all-campaigns, --campaign, $MLSWEEP_CAMPAIGN, then the default.
    Exits with an error on an invalid name.
    """
    if getattr(args, "all_campaigns", False):
        return None
    name = getattr(args, "campaign", None) or os.environ.get("MLSWEEP_CAMPAIGN") or DEFAULT_CAMPAIGN
    try:
        return validate_campaign(name)
    except ValueError as e:
        sweep_print(f"{_RED}Error: {e}{_RESET}")
        sys.exit(1)


def _campaign_argv(campaign: str | None) -> list[str]:
    """Flags that select *campaign* (None = every campaign) on another command."""
    return ["--all-campaigns"] if campaign is None else ["--campaign", campaign]


def _with_campaign(path: str, campaign: str | None) -> str:
    """Append ``campaign=`` to an API *path* (unchanged for None = every campaign)."""
    if campaign is None:
        return path
    return f"{path}{'&' if '?' in path else '?'}campaign={quote(campaign)}"


def require_campaign(manager: str, token: str, experiment_id: str, campaign: str | None) -> None:
    """Exit with a hint when *experiment_id* is in a campaign other than *campaign*.

    An unknown experiment or an unreachable manager passes silently, so each
    command reports that its own way.  Every request the command makes
    afterwards also carries ``?campaign=``, so the manager refuses it if the
    experiment moves.
    """
    if campaign is None:
        return
    status, resp = _http_request(
        "GET", _manager_url(manager, _with_campaign(f"/api/experiments/{experiment_id}", campaign)),
        token, quiet=True,
    )
    if status == 404 and isinstance(resp, dict) and resp.get("campaign"):
        actual = resp["campaign"]
        sweep_print(f"{_RED}Error: experiment {experiment_id} is in campaign '{actual}', "
                    f"not '{campaign}'.{_RESET}")
        sweep_print(f"Pass --campaign {actual} or --all-campaigns.")
        sys.exit(1)


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
    quiet: bool = False,
) -> tuple[int, Any]:
    """Make an HTTP request to the manager. Returns (status_code, parsed_response).

    Accepts JSON response and returns the parsed object or raw body for non-JSON.
    On error, prints a message (unless *quiet*) and returns (status, None).
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
            content_type = resp.headers.get_content_type()
    except HTTPError as e:
        status = e.code
        raw = e.read()
        content_type = e.headers.get_content_type()
    except URLError as e:
        if not quiet:
            sweep_print(f"{_RED}Error: cannot reach manager at {url}: {e.reason}{_RESET}")
        return (0, None)
    except Exception as e:
        if not quiet:
            sweep_print(f"{_RED}Error: HTTP request failed: {e}{_RESET}")
        return (0, None)

    # Decode by declared type. Guessing breaks on text bodies that happen to
    # be valid JSON, such as a one-line metrics.jsonl.
    if content_type == "application/json":
        try:
            return (status, json.loads(raw))
        except json.JSONDecodeError:
            pass
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
        self._buf = bytearray()

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
        self._buf = bytearray(leftover)
        self._sock = sock

    def recv_frame(self) -> tuple[int, bytes] | None:
        """Read one frame. Returns (opcode, payload), or None once the connection is closed.

        Raises TimeoutError if no complete frame arrives within the timeout.
        Partial frames stay buffered, so calling again resumes where it left off.
        """
        while True:
            frame = self._pop_frame()
            if frame is not None:
                return frame
            if not self._sock:
                return None
            try:
                self._sock.settimeout(self._timeout)
                chunk = self._sock.recv(65536)
            except TimeoutError:
                raise
            except OSError:
                return None
            if not chunk:
                return None
            self._buf += chunk

    def _pop_frame(self) -> tuple[int, bytes] | None:
        """Remove and return one complete frame from the buffer, or None if incomplete."""
        buf = self._buf
        if len(buf) < 2:
            return None
        opcode = buf[0] & 0x0F
        masked = (buf[1] & 0x80) != 0  # server frames are NOT masked, but be tolerant
        length = buf[1] & 0x7F
        pos = 2
        if length == 126:
            if len(buf) < 4:
                return None
            length = struct.unpack_from("!H", buf, 2)[0]
            pos = 4
        elif length == 127:
            if len(buf) < 10:
                return None
            length = struct.unpack_from("!Q", buf, 2)[0]
            pos = 10
        mask_key = b""
        if masked:
            if len(buf) < pos + 4:
                return None
            mask_key = bytes(buf[pos:pos + 4])
            pos += 4
        if len(buf) < pos + length:
            return None
        payload = bytes(buf[pos:pos + length])
        del buf[:pos + length]
        if mask_key:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return (opcode, payload)

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
        sock = self._sock
        if sock:
            self.send_close(1000)  # clears self._sock if the peer is already gone
            self._sock = None
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def iter_events(self, *, max_idle: float | None = None) -> Iterator[dict[str, Any]]:
        """Yield decoded JSON events until the server closes the stream.

        Pings every 20s. A pong counts as activity, so *max_idle* trips only
        when the server stops answering.  A quiet stream while jobs wait for
        GPUs keeps the connection open.
        Closes the socket when the generator finishes or is closed.
        """
        stop = threading.Event()

        def _ping_loop() -> None:
            while not stop.is_set():
                self.send_ping()
                stop.wait(20.0)

        heartbeat = threading.Thread(target=_ping_loop, daemon=True)
        heartbeat.start()
        last_activity = time.time()
        try:
            while True:
                try:
                    frame = self.recv_frame()
                except TimeoutError:
                    if max_idle is not None and time.time() - last_activity > max_idle:
                        sweep_print(f"\n  {_YELLOW}No response from manager for {max_idle:.0f}s, disconnecting{_RESET}")
                        return
                    continue
                if frame is None:
                    return
                opcode, payload = frame
                if opcode == _WS_OP_CLOSE:
                    return
                last_activity = time.time()
                if opcode == _WS_OP_PING:
                    self.send_frame(_WS_OP_PONG, payload)
                if opcode != _WS_OP_TEXT:
                    continue
                try:
                    yield json.loads(payload.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        finally:
            stop.set()
            heartbeat.join(timeout=1.0)
            self.close()


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
    max_concurrent: int = 0,
    skip_rules: dict[str, Any] | None = None,
    metric: str | None = None,
    goal: str | None = None,
    campaign: str = DEFAULT_CAMPAIGN,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Create an experiment in *campaign* on the manager. Returns the response dict or None.

    With *jobs*, the experiment and its jobs are created in one atomic request:
    either all of it lands or none of it does.
    """
    path = f"/api/experiments/{experiment_id}/jobs/bulk" if jobs else "/api/experiments"
    status, resp = _http_request(
        "POST",
        _manager_url(manager, _with_campaign(path, campaign)),
        token,
        json_data={
            "experiment_id": experiment_id,
            "name": name,
            "campaign": campaign,
            "controller_id": controller_id,
            "note": note,
            "status": "running",
            "expected_jobs": expected_jobs,
            "singular_dims": singular_dims or [],
            "max_concurrent": max_concurrent,
            "skip_rules": skip_rules or {},
            "metric": metric,
            "goal": goal,
            **({"jobs": jobs} if jobs else {}),
        },
    )
    if status in (200, 201) and isinstance(resp, dict):
        with_jobs = f" with {len(jobs)} job(s)" if jobs else ""
        sweep_print(f"  {_GREEN}OK{_RESET}    Experiment created: {experiment_id} "
                    f"(campaign {campaign}){with_jobs}")
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
    campaign: str | None = None,
) -> list[dict[str, Any]] | None:
    """Submit multiple jobs in bulk. Returns list of created job records."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, _with_campaign("/api/jobs/bulk", campaign)),
        token,
        json_data=jobs,
    )
    if status in (200, 201) and isinstance(resp, list):
        sweep_print(f"  {_GREEN}OK{_RESET}    {len(resp)} jobs submitted")
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  Submit jobs: {resp}")
    return None


def manager_get_job_metrics(
    manager: str,
    token: str,
    experiment_id: str,
    run_id: str,
    campaign: str | None = None,
) -> list[dict[str, Any]] | None:
    """Fetch metrics.jsonl for a completed job."""
    status, resp = _http_request(
        "GET",
        _manager_url(manager, _with_campaign(
            f"/api/experiments/{experiment_id}/jobs/{run_id}/metrics", campaign)),
        token,
    )
    if status == 200 and isinstance(resp, str):
        lines: list[dict[str, Any]] = []
        for line in resp.strip().splitlines():
            if line.strip():
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return lines
    return None


def manager_get_experiment_summary(
    manager: str,
    token: str,
    experiment_id: str,
    *,
    quiet: bool = False,
    campaign: str | None = None,
) -> dict[str, Any] | None:
    """Get experiment summary from manager."""
    status, resp = _http_request(
        "GET",
        _manager_url(manager, _with_campaign(f"/api/experiments/{experiment_id}/summary", campaign)),
        token,
    )
    if status == 200 and isinstance(resp, dict):
        return resp
    if not quiet:
        sweep_print(f"  {_RED}FAIL{_RESET}  Get summary: {resp}")
    return None


def manager_list_experiments(
    manager: str,
    token: str,
    status_filter: str | None = None,
    campaign: str | None = None,
) -> list[dict[str, Any]] | None:
    """List experiments on the manager, only *campaign*'s unless it is None."""
    path = "/api/experiments"
    if status_filter:
        path += f"?status={status_filter}"
    path = _with_campaign(path, campaign)
    status, resp = _http_request("GET", _manager_url(manager, path), token, timeout=10)
    if status == 200 and isinstance(resp, list):
        return resp
    return None


def manager_list_workers(manager: str, token: str) -> list[dict[str, Any]] | None:
    """List workers on the manager, enriched with live GPU occupancy and health."""
    status, resp = _http_request("GET", _manager_url(manager, "/api/workers"), token, timeout=10)
    if status == 200 and isinstance(resp, list):
        return resp
    return None


def manager_list_campaigns(manager: str, token: str) -> list[dict[str, Any]] | None:
    """List campaigns with their experiment and job counts."""
    status, resp = _http_request("GET", _manager_url(manager, "/api/campaigns"), token, timeout=10)
    if status == 200 and isinstance(resp, list):
        return resp
    return None


def manager_move_experiment(
    manager: str,
    token: str,
    experiment_id: str,
    target: str,
    campaign: str | None = None,
) -> dict[str, Any] | None:
    """Move an experiment (it must be in *campaign*, unless None) to campaign *target*."""
    status, resp = _http_request(
        "PUT",
        _manager_url(manager, _with_campaign(f"/api/experiments/{experiment_id}/campaign", campaign)),
        token,
        json_data={"campaign": target},
    )
    if status == 200 and isinstance(resp, dict):
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  Move experiment: {resp}")
    return None


def manager_list_experiment_jobs(
    manager: str,
    token: str,
    experiment_id: str,
    status_filter: str | None = None,
    campaign: str | None = None,
) -> list[dict[str, Any]] | None:
    """List jobs for an experiment."""
    path = f"/api/experiments/{experiment_id}/jobs"
    if status_filter:
        path += f"?status={status_filter}"
    path = _with_campaign(path, campaign)
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
    campaign: str | None = None,
) -> dict[str, Any] | None:
    """Cancel a pending job."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, _with_campaign(
            f"/api/jobs/{run_id}/cancel?experiment_id={experiment_id}", campaign)),
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
    campaign: str | None = None,
) -> dict[str, Any] | None:
    """Retry a failed job."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, _with_campaign(
            f"/api/jobs/{run_id}/retry?experiment_id={experiment_id}", campaign)),
        token,
    )
    if status == 200 and isinstance(resp, dict):
        return resp
    return None


def manager_set_job_label(
    manager: str,
    token: str,
    run_id: str,
    experiment_id: str,
    label: str | None,
    campaign: str | None = None,
) -> dict[str, Any] | None:
    """Set a run's display name, or clear it with *label* None."""
    status, resp = _http_request(
        "PUT",
        _manager_url(manager, _with_campaign(f"/api/jobs/{run_id}/label", campaign)),
        token,
        json_data={"experiment_id": experiment_id, "label": label},
    )
    if status == 200 and isinstance(resp, dict):
        return resp
    return None


def manager_set_experiment_status(
    manager: str,
    token: str,
    experiment_id: str,
    status: str,
    campaign: str | None = None,
) -> dict[str, Any] | None:
    """Update an experiment's status (running | paused | completed | aborted)."""
    s, resp = _http_request(
        "PUT",
        _manager_url(manager, _with_campaign(f"/api/experiments/{experiment_id}/status", campaign)),
        token,
        json_data={"status": status},
    )
    if s == 200 and isinstance(resp, dict):
        return resp
    sweep_print(f"  {_RED}FAIL{_RESET}  set status {status}: {resp}")
    return None


def manager_get_job_logs(
    manager: str,
    token: str,
    experiment_id: str,
    run_id: str,
    campaign: str | None = None,
) -> str | None:
    """Return a run's training log as text."""
    status, resp = _http_request(
        "GET",
        _manager_url(manager, _with_campaign(
            f"/api/experiments/{experiment_id}/jobs/{run_id}/logs", campaign)),
        token,
    )
    if status == 200 and isinstance(resp, str):
        return resp
    return None


_ACTIVE_JOB_STATUSES = ("pending", "dispatched", "running")
_STATUS_ORDER = {"pending": 0, "dispatched": 1, "running": 2, "failed": 3, "cancelled": 4, "done": 5, "xfailed": 6}


def _parse_combo(raw: Any) -> dict[str, Any] | None:
    """Decode a job's combo, which the manager may send as a JSON string."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    return raw if isinstance(raw, dict) else None


def _combo_str(combo: Any) -> str:
    """Render a job combo dict as `k=v` pairs."""
    if not isinstance(combo, dict):
        return ""
    return "  ".join(f"{k}={v}" for k, v in combo.items())


def _run_str(run_id: Any, label: Any) -> str:
    """A run ID in green, followed by its display name if it has one."""
    return f"{_GREEN}{run_id}{_RESET}" + (f"  ({label})" if label else "")


def _metric_values(metrics: list[dict[str, Any]] | None, metric: str) -> list[float]:
    """Finite values of *metric* across metric rows, in step order."""
    return [
        float(r[metric])
        for r in (metrics or [])
        if isinstance(r, dict) and isinstance(r.get(metric), (int, float))
        and math.isfinite(float(r[metric]))
    ]


def _best_metric(metrics: list[dict[str, Any]] | None, metric: str, goal: str) -> float | None:
    """Best finite value of *metric* under *goal*, or None if it was never logged."""
    vals = _metric_values(metrics, metric)
    if not vals:
        return None
    return min(vals) if goal == "minimize" else max(vals)


def build_leaderboard(
    manager: str,
    token: str,
    experiment_id: str,
    metric: str = "loss",
    goal: str = "minimize",
    jobs: list[dict[str, Any]] | None = None,
    campaign: str | None = None,
) -> list[dict[str, Any]]:
    """Compute a ranked list of run results, best-first by *metric*.

    Each row: {run_id, label, status, combo, value, final, elapsed, exit_code}.
    """
    if jobs is None:
        jobs = manager_list_experiment_jobs(manager, token, experiment_id, campaign=campaign) or []

    # Metrics are per-run requests; fetch them concurrently.
    done_ids = [j.get("run_id") or "" for j in jobs if j.get("status") == "done"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        all_metrics = dict(zip(done_ids, pool.map(
            lambda rid: manager_get_job_metrics(manager, token, experiment_id, rid, campaign=campaign),
            done_ids)))
    return rank_leaderboard(jobs, all_metrics, metric, goal)


def rank_leaderboard(
    jobs: list[dict[str, Any]],
    all_metrics: dict[str, list[dict[str, Any]] | None],
    metric: str = "loss",
    goal: str = "minimize",
) -> list[dict[str, Any]]:
    """Rank *jobs* best-first by *metric*, given each run's metric rows by run_id."""
    rows: list[dict[str, Any]] = []
    for j in jobs:
        st = j.get("status") or "?"
        run_id = j.get("run_id") or ""
        value = final = None
        if st == "done":
            vals = _metric_values(all_metrics.get(run_id), metric)
            if vals:
                value = min(vals) if goal == "minimize" else max(vals)
                final = vals[-1]
        rows.append({
            "run_id": run_id,
            "label": j.get("label"),
            "status": st,
            "combo": _parse_combo(j.get("combo")),
            "value": value,
            "final": final,
            "elapsed": j.get("elapsed"),
            "exit_code": j.get("exit_code"),
        })

    def _key(r: dict[str, Any]) -> tuple[int, Any]:
        if r["value"] is not None:
            return (0, r["value"] if goal == "minimize" else -r["value"])
        return (1, _STATUS_ORDER.get(r["status"], 9))

    rows.sort(key=_key)
    return rows


def _leaderboard_header(metric: str, goal: str, summary: str) -> None:
    """Print the ruled ``LEADERBOARD:`` banner shared by the ranking printers."""
    rule = f"{_CYAN}{'=' * 80}{_RESET}"
    sweep_print(f"\n{rule}")
    sweep_print(f"{_BOLD}LEADERBOARD:{_RESET} {_MAGENTA}{metric}{_RESET} ({goal}), {summary}")
    sweep_print(rule)


def _leaderboard_row(i: int, r: dict[str, Any], prefix: str = "") -> None:
    """Print one ranked run; *prefix* goes just before the run name."""
    final = f"  final={r['final']:.6f}" if isinstance(r["final"], (int, float)) else ""
    sweep_print(f"  {i:>3}. {_MAGENTA}{r['value']:>12.6f}{_RESET}{final}  {prefix}"
                f"{_run_str(r['run_id'], r['label'])}  {_combo_str(r['combo'])}")


def print_leaderboard(
    rows: list[dict[str, Any]],
    metric: str = "loss",
    goal: str = "minimize",
    top: int = 10,
) -> None:
    """Print the ranked runs."""
    done = [r for r in rows if r["value"] is not None]
    _leaderboard_header(metric, goal, f"{_GREEN}{len(done)}{_RESET} completed runs")
    if not done:
        sweep_print(f"  {_YELLOW}(no completed runs with a metric value){_RESET}")
        return
    for i, r in enumerate(done[:top] if top else done, 1):
        _leaderboard_row(i, r)


def _wait_until_settled(
    manager: str,
    token: str,
    experiment_id: str,
    interval: int = 10,
    campaign: str | None = None,
) -> bool:
    """Block until an experiment has no pending/dispatched/running jobs.

    Returns True when at least one job finished as ``failed`` (so callers can
    exit non-zero), False for a clean settle.  Progress goes to stderr so that
    ``--json`` output on stdout stays parseable.
    """
    print(f"Waiting for {experiment_id} to settle (Ctrl+C to stop)...", file=sys.stderr, flush=True)
    while True:
        resp = manager_get_experiment_summary(manager, token, experiment_id, quiet=True,
                                              campaign=campaign)
        if resp:
            counts = resp.get("job_counts") or {}
            active = sum(int(counts.get(s, 0)) for s in _ACTIVE_JOB_STATUSES)
            if active == 0:
                failed = int(counts.get("failed", 0)) > 0
                if failed:
                    print("  experiment settled with failures.", file=sys.stderr, flush=True)
                else:
                    print("  experiment settled.", file=sys.stderr, flush=True)
                return failed
        time.sleep(interval)


def resolve_ranking(
    manager: str,
    token: str,
    experiment_id: str,
    metric: str | None = None,
    goal: str | None = None,
    campaign: str | None = None,
) -> tuple[str, str]:
    """Resolve the ``(metric, goal)`` used to rank an experiment's runs.

    An explicitly supplied value always wins; otherwise fall back to the
    metric/goal stored on the experiment (from ``METRIC``/``GOAL`` or
    ``OPTIMIZE``) and finally to ``loss``/``minimize``.
    """
    if metric is None or goal is None:
        summary = manager_get_experiment_summary(manager, token, experiment_id, quiet=True,
                                                 campaign=campaign) or {}
        if metric is None:
            stored_metric = summary.get("metric")
            metric = stored_metric if isinstance(stored_metric, str) and stored_metric else "loss"
        if goal is None:
            stored_goal = summary.get("goal")
            goal = stored_goal if stored_goal in ("minimize", "maximize") else "minimize"
    return metric, goal


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
    campaign: str | None = None,
) -> bool:
    """Download experiment results from the manager and extract to output_dir.

    Makes a GET request to /api/experiments/{experiment_id}/download
    and streams the tar.gz response, extracting it to output_dir.
    Returns True on success.
    """
    url = _manager_url(manager, _with_campaign(f"/api/experiments/{experiment_id}/download", campaign))
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
                _safe_tar_extract(tar, str(out))
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


def _safe_tar_extract(tar: tarfile.TarFile, dest: str) -> None:
    """Extract a tar archive with path-traversal protection."""
    try:
        tar.extractall(path=dest, filter="data")
    except TypeError:
        pass
    else:
        return

    resolved_dest = os.path.realpath(dest)
    for member in tar.getmembers():
        member_path = os.path.realpath(os.path.join(resolved_dest, member.name))
        if os.path.commonpath([member_path, resolved_dest]) != resolved_dest:
            raise ValueError(
                f"tar member {member.name!r} escapes destination"
            )
    tar.extractall(path=dest)


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


def _ws_stream_url(manager: str, experiment_id: str, token: str, campaign: str | None = None) -> str:
    """Build WebSocket URL for experiment event stream."""
    http_url = manager.rstrip("/")
    if http_url.startswith("https://"):
        ws_url = "wss://" + http_url[8:]
    elif http_url.startswith("http://"):
        ws_url = "ws://" + http_url[7:]
    else:
        ws_url = "ws://" + http_url
    return _with_campaign(f"{ws_url}/ws/experiments/{experiment_id}?token={token}", campaign)


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
    campaign: str | None = None,
) -> None:
    """Connect to manager WebSocket and display live job status until idle timeout.

    If *on_event* is callable, it is invoked as ``on_event(event, manager, token, experiment_id)``
    for every received event.  This allows the Bayes controller to react to
    job completions (tell + suggest + submit) inline.

    If *writer_factory* is provided (e.g. MultiWriterFactory), run writers are
    created lazily for each run and fed metric / finish events.
    *variations* is used to look up combos when creating writers.
    """
    ws_url = _ws_stream_url(manager, experiment_id, token, campaign)

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

    try:
        for event in ws.iter_events(max_idle=max_idle):
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

    except KeyboardInterrupt:
        sweep_print(f"\n  {_YELLOW}Interrupted{_RESET}")
    finally:
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
    sweep_print(f"\n{_CYAN}{'=' * 80}{_RESET}")
    ok = sum(1 for s in job_status.values() if s["status"] == "done")
    failed = sum(1 for s in job_status.values() if s["status"] == "failed")
    running = sum(1 for s in job_status.values() if s["status"] in _ACTIVE_JOB_STATUSES)
    sweep_print(f"{_BOLD}Final:{_RESET} {_GREEN}{ok} OK{_RESET}, {_RED}{failed} failed{_RESET}, "
                f"{_YELLOW}{running} pending/running{_RESET}")
    sweep_print(f"{_CYAN}{'=' * 80}{_RESET}")


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

    summary = (f"  [{_GREEN}{n_ok} ok{_RESET}, {_RED}{n_fail} fail{_RESET}, "
               f"{_CYAN}{n_run} running{_RESET}, {_YELLOW}{n_pend} pending{_RESET}]")
    lines.append(summary)

    # Clear and re-print (use \r\033[K for simple overwrite)
    # For multi-line, move cursor up
    output = "\n".join(lines)
    # Move the cursor up over the previous table, if one was printed, and clear it
    global _status_table_lines
    if _status_table_lines:
        sys.stdout.write(f"\033[{_status_table_lines}A\033[J")
    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    _status_table_lines = len(lines) + 1


# Lines the last _render_status_table printed (0 = none yet).
_status_table_lines = 0


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
    n_pending = sum(1 for j in jobs if j["status"] in _ACTIVE_JOB_STATUSES)
    n_cancelled = sum(1 for j in jobs if j["status"] == "cancelled")
    total = len(jobs)

    sweep_print(f"\n{_CYAN}{'=' * 80}{_RESET}")
    sweep_print(f"{_BOLD}SUMMARY{_RESET} — {total} jobs: {_GREEN}{n_ok} OK{_RESET}, "
                f"{_RED}{n_fail} failed{_RESET}, {_YELLOW}{n_pending} pending{_RESET}, "
                f"{n_cancelled} cancelled")
    sweep_print(f"{_CYAN}{'=' * 80}{_RESET}")

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
        elif status in _ACTIVE_JOB_STATUSES:
            sweep_print(f"  {_YELLOW}{status.upper()}{_RESET}  {run_id}")
        elif status == "cancelled":
            sweep_print(f"  {_YELLOW}CANCEL{_RESET}  {run_id}")
        else:
            sweep_print(f"  {status}  {run_id}")

    return n_fail > 0


# ===============================================================================
# Job payload builder
# ===============================================================================


def _skip_rules(options: dict[str, Any]) -> dict[str, Any]:
    """The manager's input to ``should_skip``: every dim (subdims included), by name.

    Empty when no dim is monotonic or singular.  Every dim is listed because
    ``should_skip`` holds all the others fixed when comparing two combos.
    """
    rules: dict[str, Any] = {}

    def walk(opts: dict[str, Any]) -> None:
        for key, opt in opts.items():
            rules[key[1:]] = {
                "monotonic": opt.get("monotonic"),
                "singular": bool(opt.get("singular")),
                "_values": opt.get("_values", []),
            }
            for sub in opt.get("_sub_opts_map", {}).values():
                walk(sub)

    walk(options)
    if not any(r["monotonic"] or r["singular"] for r in rules.values()):
        return {}
    return rules


def _check_duplicate_run_names(variations: list[dict[str, Any]]) -> None:
    """Exit if two runs would share a name (jobs require unique names per experiment)."""
    counts = collections.Counter(v["name"] for v in variations)
    dups = sorted(name for name, n in counts.items() if n > 1)
    if not dups:
        return
    for name in dups:
        sweep_print(f"  duplicate run name: {name}")
    sweep_print(
        f"{_RED}Error:{_RESET} {len(dups)} run name(s) collide. "
        "Give each dimension a 'name' (or distinct values) so every run name is unique."
    )
    sys.exit(1)


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
            "return_files": [],  # could configure via sweep file
        }
        if setup_command:
            job["setup_command"] = setup_command
        jobs.append(job)
    return jobs


# ===============================================================================
# Watch subcommand
# ===============================================================================


def _watch_cmd(args: list[str], prog: str = "mlsweep_run watch") -> None:
    """Watch an experiment's progress via WebSocket event stream."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Watch experiment progress via WebSocket",
    )
    parser.add_argument("experiment_id", help="Experiment ID to watch")
    _add_manager_args(parser)
    parser.add_argument("--color", action="store_true",
                        help="Enable ANSI color in human-readable output (default: off)")
    parser.add_argument("--events", action="store_true",
                        help="Emit one JSON event per line and nothing else (machine-readable)")
    parser.add_argument("--json", action="store_true",
                        help="Alias for --events; stream one JSON event per line")
    parsed = parser.parse_args(args)
    if parsed.color:
        set_color(True)
    emit_json = bool(parsed.events or parsed.json)

    manager, token = _manager_token(parsed)
    campaign = _resolve_campaign(parsed)
    require_campaign(manager, token, parsed.experiment_id, campaign)

    # Connect WebSocket with since=now to only get new events
    ws_url = _ws_stream_url(manager, parsed.experiment_id, token, campaign)
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

    if not emit_json:
        sweep_print(f"Watching experiment {_YELLOW}{parsed.experiment_id}{_RESET}")
        sweep_print(f"Manager: {manager}")
        sweep_print("")

    failed = False
    try:
        for event in ws.iter_events():
            event_type = event.get("type", "unknown")
            run_id = event.get("run_id", "")
            if event_type in ("job_done", "run_result") and not event.get("success", False):
                failed = True

            if emit_json:
                print(json.dumps(event), flush=True)
                if event_type == "experiment_done":
                    break
                continue

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
        if not emit_json:
            sweep_print(f"\n  {_YELLOW}Interrupted{_RESET}")
    finally:
        ws.close()

    if failed:
        sys.exit(1)


# ===============================================================================
# Fetch subcommand
# ===============================================================================


def _fetch_cmd(args: list[str], prog: str = "mlsweep_run fetch") -> None:
    """Fetch experiment results from a manager and print summary + leaderboard."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Fetch experiment results from mlsweep manager",
    )
    _add_manager_args(parser)
    parser.add_argument("--experiment", required=True, help="Experiment ID to fetch")
    parser.add_argument("--output-dir", default=None, help="Directory to download artifacts (optional)")
    parser.add_argument("--status", default=None, help="Filter jobs by status (done, failed, pending, etc.)")
    parser.add_argument("--metric", default=None, help="Metric to rank runs by (default: experiment's metric, else loss)")
    parser.add_argument("--goal", default=None, choices=["minimize", "maximize"],
                        help="Rank direction (default: experiment's goal, else minimize)")
    parser.add_argument("--top", type=int, default=10, help="Show top N runs in the leaderboard (0 = all)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON (skips download)")
    parser.add_argument("--wait", action="store_true", help="Block until the experiment settles")
    parser.add_argument("--wait-interval", type=int, default=10, help="Seconds between --wait polls")
    parser.add_argument("--color", action="store_true",
                        help="Enable ANSI color in human-readable output (default: off)")
    parsed = parser.parse_args(args)
    if parsed.color:
        set_color(True)

    manager, token = _manager_token(parsed)
    campaign = _resolve_campaign(parsed)
    require_campaign(manager, token, parsed.experiment, campaign)

    had_failure = False
    if parsed.wait:
        had_failure = _wait_until_settled(manager, token, parsed.experiment, parsed.wait_interval,
                                          campaign=campaign)

    metric, goal = resolve_ranking(manager, token, parsed.experiment, parsed.metric, parsed.goal,
                                   campaign=campaign)

    summary = manager_get_experiment_summary(manager, token, parsed.experiment,
                                             campaign=campaign) or {}
    jobs = manager_list_experiment_jobs(manager, token, parsed.experiment,
                                        status_filter=parsed.status, campaign=campaign)
    if jobs is None:
        sys.exit(1)

    rows = build_leaderboard(manager, token, parsed.experiment, metric, goal, jobs=jobs,
                             campaign=campaign)

    if parsed.json:
        out = {
            "experiment_id": parsed.experiment,
            "campaign": summary.get("campaign"),
            "name": summary.get("name"),
            "status": summary.get("status"),
            "job_counts": summary.get("job_counts"),
            "metric": metric,
            "goal": goal,
            "runs": rows,
        }
        print(json.dumps(out, indent=2))
        if had_failure:
            sys.exit(1)
        return

    if summary:
        sweep_print(f"Experiment: {summary['name']}")
        sweep_print(f"Campaign:   {summary.get('campaign')}")
        sweep_print(f"Status:     {summary['status']}")
        counts = summary["job_counts"]
        if counts:
            sweep_print(f"Jobs:       {counts}")

    if jobs:
        print_jobs_summary(jobs)
    print_leaderboard(rows, metric, goal, parsed.top)

    # Download experiment artifacts
    output_dir = parsed.output_dir or str(_mlsweep_dir() / "downloads" / parsed.experiment)
    manager_download_experiment(manager, token, parsed.experiment, output_dir, campaign=campaign)

    if had_failure:
        sys.exit(1)


# ===============================================================================
# Main
# ===============================================================================


def main() -> None:
    global _log_file

    # Standalone `mlsweep_run` also accepts the global --color flag; consume it
    # before subcommand dispatch so `mlsweep_run fetch --color ...` works too.
    argv = strip_color_flag(list(sys.argv[1:]))

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
            "Subcommands:\n"
            "  mlsweep_run fetch --manager URL --experiment EXP_ID   fetch results + summary\n"
            "  mlsweep_run watch EXP_ID --manager URL                live status\n\n"
            "Environment variables:\n"
            "  MLSWEEP_MANAGER  Manager URL for fetch/watch (default: http://localhost:7891)\n"
            "  MLSWEEP_TOKEN    Authentication token for manager\n"
            "  MLSWEEP_CAMPAIGN Campaign to submit under (default: default)\n"
            "  MLSWEEP_DIR      Local state dir holding manager.token (default: ~/.mlsweep)\n"
        ),
    )
    parser.add_argument("sweep_file", help="Path to sweep .py file")
    parser.add_argument("--manager", default=None,
                        help="Manager URL (http://host:port)")
    parser.add_argument("--token", default=None,
                        help="Manager auth token (or set MLSWEEP_TOKEN env)")
    _add_campaign_args(parser)
    parser.add_argument("--output-dir", default=str(_mlsweep_dir() / "submissions"),
                        help="Directory for local submit-side artifacts (submit log + manifest). "
                             "Results themselves live on the manager under ~/.mlsweep/experiments/.")
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
        "--max-concurrent", type=int, default=0, metavar="K",
        help="Cap on this sweep's simultaneously-running jobs across the whole "
             "cluster (0 = unlimited). Use it to take only a slice of a shared cluster.")
    parser.add_argument("--color", action="store_true",
                        help="Enable ANSI color in human-readable output (default: off)")

    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    campaign = _resolve_campaign(args)
    if campaign is None and args.resume is None and not args.validate:
        sweep_print(f"{_RED}Error: a new sweep needs one campaign. Pick it with --campaign NAME "
                    f"instead of --all-campaigns.{_RESET}")
        sys.exit(1)

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
    if not args.setup_command and info.get("setup_command"):
        args.setup_command = info["setup_command"]

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
        _check_duplicate_run_names(all_variations)
        expected = count_expected(options)
        excluded = expected - len(all_variations)
        dim_names = [k[1:] for k in options]
        sweep_print(f"{_BOLD}Sweep:{_RESET} {_CYAN}{sweep_name}{_RESET}")
        sweep_print(f"{_BOLD}Dimensions:{_RESET} {', '.join(dim_names) if dim_names else '(none)'}")
        for key in options:
            dim_name = key[1:]
            values = options[key].get("_values", [])
            if values != [None]:
                sweep_print(f"  {_CYAN}{dim_name}{_RESET}: {values}")
        sweep_print(f"\n{_BOLD}Total combinations:{_RESET} {_GREEN}{len(all_variations)}{_RESET}")
        if excluded:
            sweep_print(f"{_YELLOW}Excluded by EXCLUDE filter: {excluded}{_RESET}")
        sweep_print(f"\n{_BOLD}{_CYAN}Runs:{_RESET}")
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
    sweep_print(f"{_BOLD}Command:{_RESET} {' '.join(command)}")
    if method == "bayes":
        sweep_print(f"{_BOLD}Sweep:{_RESET} {_CYAN}{sweep_name}{_RESET} (bayes, budget={expected})")
    else:
        n_probes = len(variations)
        n_expected = expected  # count_expected: ignores singular probes
        if n_expected < n_probes:
            sweep_print(f"{_BOLD}Sweep:{_RESET} {_CYAN}{sweep_name}{_RESET} ({n_expected}–{n_probes} runs, {n_probes - n_expected} singular probes)")
        else:
            sweep_print(f"{_BOLD}Sweep:{_RESET} {_CYAN}{sweep_name}{_RESET} ({n_probes} runs)")
    sweep_print(f"{_BOLD}Experiment:{_RESET} {_CYAN}{experiment_id}{_RESET}")
    sweep_print(f"{_BOLD}Campaign:{_RESET} {_CYAN}{campaign or 'any (--all-campaigns)'}{_RESET}")
    if extra:
        sweep_print(f"{_BOLD}Extra overrides:{_RESET} {' '.join(extra)}")

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
        sweep_print(f"\n{_CYAN}{'=' * 80}{_RESET}")
        sweep_print(f"{_BOLD}DRY RUN{_RESET} — {_GREEN}{len(variations)}{_RESET} runs would be submitted to manager")
        sweep_print(f"{_CYAN}{'=' * 80}{_RESET}")
        return

    # ── Manager required ───────────────────────────────────────────────────
    if not args.manager:
        sweep_print(f"\n{_RED}Error: --manager URL is required.{_RESET}")
        sweep_print(f"Usage: mlsweep_run <sweep.py> --manager http://host:port [--stream]")
        sys.exit(1)

    manager, token = _manager_token(args)

    sweep_print(f"\n{_CYAN}{'=' * 80}{_RESET}")
    sweep_print(f"{_BOLD}Connecting to manager:{_RESET} {_CYAN}{manager}{_RESET}")
    sweep_print(f"{_CYAN}{'=' * 80}{_RESET}\n")

    # ── Resume: fetch completed jobs and rebuild optimizer ─────────────────
    if resume:
        sweep_print("Resuming experiment...")
        require_campaign(manager, token, experiment_id, campaign)
        summary = manager_get_experiment_summary(manager, token, experiment_id, campaign=campaign)
        if summary is None:
            sweep_print(f"  {_RED}FAIL{_RESET}  Cannot fetch experiment summary — is manager reachable?")
            sys.exit(1)
        sweep_print(f"  Experiment: {summary['name']}")
        sweep_print(f"  Status:     {summary['status']}")

        done_jobs = manager_list_experiment_jobs(manager, token, experiment_id, status_filter="done",  # type: ignore[assignment]
                                                 campaign=campaign)
        if done_jobs is None:
            done_jobs = []
        sweep_print(f"  Completed jobs: {len(done_jobs)}")

        # Rebuild optimizer from completed jobs
        assert optimizer is not None
        metric_name = optimize_cfg["metric"]
        goal = optimize_cfg["goal"]
        told_count = 0
        for job in done_jobs:
            combo = _parse_combo(job["combo"])
            if combo is None:
                continue
            run_id = job["run_id"]
            metrics_list = manager_get_job_metrics(manager, token, experiment_id, run_id,
                                                   campaign=campaign)
            if metrics_list is None:
                continue
            best = _best_metric(metrics_list, metric_name, goal)
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
    _check_duplicate_run_names(variations)

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
                setup_command=args.setup_command,
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

    # ── 2. Build job payloads ───────────────────────────────────────────────
    build_payloads = functools.partial(
        _build_job_payloads,
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
    )
    job_payloads = build_payloads(variations=variations, experiment_id=experiment_id)

    # ── 3. Create experiment and submit jobs atomically ────────────────────
    if resume:
        sweep_print("Resuming — experiment already exists")
    if job_payloads:
        sweep_print("Submitting jobs...")
    if resume:
        if job_payloads and manager_submit_jobs_bulk(
                manager, token, job_payloads, campaign=campaign) is None:
            sys.exit(1)
    elif not manager_create_experiment(
        manager, token, experiment_id,
        name=sweep_name,
        note=args.note,
        expected_jobs=expected if method == "bayes" else 0,
        singular_dims=[k[1:] for k, v in options.items() if v.get("singular")],
        max_concurrent=args.max_concurrent,
        # The Bayes controller handles singular probes itself.
        skip_rules=_skip_rules(options) if method == "grid" else None,
        metric=info.get("metric"),
        goal=info.get("goal"),
        campaign=campaign or DEFAULT_CAMPAIGN,
        jobs=job_payloads,
    ):
        sys.exit(1)

    # ── 4. Write local manifest ────────────────────────────────────────────
    _write_manifest(exp_dir, experiment_id, variations, note=args.note)

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
                new_job = build_payloads(variations=new_vars, experiment_id=eid)
                if new_job:
                    submitted = manager_submit_jobs_bulk(manager, token, new_job, campaign=campaign)
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
                        _manager_url(mgr, _with_campaign(f"/api/jobs/{run_id}?experiment_id={eid}", campaign)),
                        tok,
                    )
                    if status_code == 200 and isinstance(job_resp, dict):
                        combo = _parse_combo(job_resp["combo"])

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
                    metrics_list = manager_get_job_metrics(mgr, tok, eid, run_id, campaign=campaign)
                    optimizer.tell(combo, _best_metric(metrics_list, metric_name, goal))
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
                                output_dir=exp_dir,
                                campaign=campaign)
        else:
            _stream_status_live(manager, token, experiment_id,
                                writer_factory=writer_factory,
                                variations=variations,
                                output_dir=exp_dir,
                                campaign=campaign)
    elif args.fetch:
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"Fetching results...")
        sweep_print(f"{'=' * 80}")
        metric = info.get("metric") or optimize_cfg.get("metric", "loss")
        goal = info.get("goal") or optimize_cfg.get("goal", "minimize")
        jobs = manager_list_experiment_jobs(manager, token, experiment_id, campaign=campaign)
        if jobs is not None:
            print_jobs_summary(jobs)
            print_leaderboard(build_leaderboard(manager, token, experiment_id, metric, goal, jobs=jobs,
                                                campaign=campaign), metric, goal)
        # Download experiment artifacts
        manager_download_experiment(manager, token, experiment_id, exp_dir, campaign=campaign)
    else:
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"{n} jobs submitted.")
        scope = " ".join(_campaign_argv(campaign))
        sweep_print(f"Watch:   mlsweep watch {experiment_id} --manager {manager} {scope}")
        sweep_print(f"Fetch:   mlsweep fetch --experiment {experiment_id} --manager {manager} {scope} --wait")
        sweep_print(f"Results: ~/.mlsweep/experiments/{experiment_id}/")
        sweep_print(f"{'=' * 80}")

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
