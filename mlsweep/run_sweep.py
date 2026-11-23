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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from mlsweep._jsonout import emit, emit_line
from mlsweep._sweep import (
    _write_manifest,
    build_job_payloads as _build_job_payloads,
    count_expected,
    generate_variations,
    metric_values,
    options_to_wire,
    singular_dims,
    validate_optimize_spec,
    validate_options,
)
from mlsweep._provenance import submit_provenance, untracked_summary
from mlsweep._snapshot import ScannedTree, scan_tree, short_id
from mlsweep._writers import (
    MultiWriterFactory,
    WriterFactory,
)
from mlsweep._shared import (
    DEFAULT_MANAGER_URL, _BOLD, _GREEN, _RED, _YELLOW, _CYAN, _MAGENTA, _BLUE,
    _RESET, _git_root, _mlsweep_dir, mlsweep_dir_id, set_color, strip_color_flag, validate_campaign,
    validate_experiment_dir,
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


def capture_session_id() -> str | None:
    """The agent trace this command belongs to, for experiment attribution.

    Resolved from the environment the coding agent ran this command with
    (see ``_capture.resolve_capture_session``).  Import errors or a missing
    watcher only ever mean "no trace": a sweep is never held up by it.
    """
    try:
        from mlsweep._capture import resolve_capture_session
        return resolve_capture_session()
    except Exception:
        return None


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
    _add_connection_args(parser)
    _add_campaign_args(parser)


def _active_context() -> tuple[Any, bool]:
    """The context in force before any --context flag (see ``_context``).  Exits on a bad one."""
    from mlsweep._context import ContextError, active_context

    try:
        return active_context()
    except ContextError as e:
        sweep_print(f"{_RED}Error: {e}{_RESET}")
        sys.exit(2)


class _ContextAction(argparse.Action):
    """``--context NAME``: take the manager, token and campaign from a named context.

    Flags given on the command line still win, wherever they appear: a value
    that differs from its default was given explicitly (or comes after, and
    overwrites this).
    """

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        from mlsweep._context import ContextError, get_context

        try:
            ctx = get_context(values)
        except ContextError as e:
            parser.error(str(e))
        setattr(namespace, self.dest, values)
        token = ctx.token()
        # What the context doesn't set falls back as if no context were in force, never to
        # the values of the one this replaces.
        manager = ctx.manager or os.environ.get("MLSWEEP_MANAGER") or DEFAULT_MANAGER_URL
        for dest, value in (("manager", manager), ("token", token), ("campaign", ctx.campaign)):
            if getattr(namespace, dest, None) == parser.get_default(dest):
                setattr(namespace, dest, value)
        if token is None and (ctx.token_file or ctx.token_env) \
                and getattr(namespace, "token", None) == parser.get_default("token"):
            setattr(namespace, "token", "")  # the context's token is missing: don't fall back to another's
            where = ctx.token_file or f"${ctx.token_env}"
            print(f"mlsweep: warning: context {ctx.name!r} names a token in {where}, which is missing or empty",
                  file=sys.stderr)


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    """Add --manager, --token and --context."""
    ctx, explicit = _active_context()
    manager = os.environ.get("MLSWEEP_MANAGER") or DEFAULT_MANAGER_URL
    token = None
    if ctx is not None:
        manager = ctx.manager or manager
        token = ctx.token()
    parser.add_argument("--manager", default=manager,
                        help=f"Manager URL (env: MLSWEEP_MANAGER, default: {DEFAULT_MANAGER_URL})")
    parser.add_argument("--token", default=token,
                        help="Manager auth token (or set MLSWEEP_TOKEN env)")
    parser.add_argument("--context", default=ctx.name if ctx else None, metavar="NAME", action=_ContextAction,
                        help="Use a named context's manager, token and campaign (env: MLSWEEP_CONTEXT; "
                             "see `mlsweep context`)")


def _context_campaign() -> str | None:
    """The campaign the active context sets, where it outranks $MLSWEEP_CAMPAIGN."""
    ctx, explicit = _active_context()
    if ctx is None or (not explicit and os.environ.get("MLSWEEP_CAMPAIGN")):
        return None
    return str(ctx.campaign) if ctx.campaign else None


def _add_campaign_args(parser: argparse.ArgumentParser) -> None:
    """Add --campaign NAME and --all-campaigns (also spelled --all_campaigns)."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--campaign", default=_context_campaign(), metavar="NAME",
                       help="Campaign to work in (env: MLSWEEP_CAMPAIGN, or a context's campaign); required "
                            "unless --all-campaigns")
    group.add_argument("--all-campaigns", "--all_campaigns", dest="all_campaigns",
                       action="store_true", help="Work across every campaign")


def _selected_campaign(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> str | None:
    """The campaign named by --campaign (or a context), else $MLSWEEP_CAMPAIGN, else None.

    *env* defaults to the process environment.  Ignores --all-campaigns.
    Exits with an error on an invalid name.
    """
    env = os.environ if env is None else env
    name = getattr(args, "campaign", None) or env.get("MLSWEEP_CAMPAIGN")
    if not name:
        return None
    try:
        return validate_campaign(name)
    except ValueError as e:
        sweep_print(f"{_RED}Error: {e}{_RESET}")
        sys.exit(1)


def _resolve_campaign(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> str | None:
    """The campaign a command works in, or None for --all-campaigns.

    Order: --all-campaigns, --campaign (or the active context's campaign), then
    $MLSWEEP_CAMPAIGN (in *env*, by default the process environment).  Exits
    with an error when none of them names a campaign, or on an invalid name.
    """
    if getattr(args, "all_campaigns", False):
        return None
    name = _selected_campaign(args, env)
    if name is None:
        sweep_print(f"{_RED}Error: no campaign given. Pass --campaign NAME, set $MLSWEEP_CAMPAIGN, "
                    f"or set one in a context (`mlsweep context`). --all-campaigns covers every "
                    f"campaign.{_RESET}")
        sys.exit(1)
    return name


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
    raw: bool = False,
) -> tuple[int, Any]:
    """Make an HTTP request to the manager. Returns (status_code, parsed_response).

    Accepts JSON response and returns the parsed object or raw body for non-JSON.
    With *raw*, a successful response's body is returned as bytes, undecoded.
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
            raw_body = resp.read()
            status = resp.status
            content_type = resp.headers.get_content_type()
    except HTTPError as e:
        status = e.code
        raw_body = e.read()
        content_type = e.headers.get_content_type()
    except URLError as e:
        if not quiet:
            sweep_print(f"{_RED}Error: cannot reach manager at {url}: {e.reason}{_RESET}")
        return (0, None)
    except Exception as e:
        if not quiet:
            sweep_print(f"{_RED}Error: HTTP request failed: {e}{_RESET}")
        return (0, None)

    if raw and 200 <= status < 300:
        return (status, raw_body)
    # Decode by declared type. Guessing breaks on text bodies that happen to
    # be valid JSON, such as a one-line metrics.jsonl.
    if content_type == "application/json":
        try:
            return (status, json.loads(raw_body))
        except json.JSONDecodeError:
            pass
    return (status, raw_body.decode("utf-8", errors="replace") if raw_body else None)


def _ws_refusal(header: str, body: bytes) -> str:
    """One line saying why a WebSocket upgrade was refused: the status and the manager's error."""
    status = " ".join(header.splitlines()[0].split()[1:]) if header else "no status"
    try:
        detail = json.loads(body)["error"]
    except (ValueError, KeyError, TypeError):
        detail = body.decode("utf-8", errors="replace").strip()
    return f"HTTP {status}: {detail}" if detail else f"HTTP {status}"


def _report_failure(label: str, status: int, resp: Any) -> None:
    """Print a FAIL line with the manager's ``error`` message, else its body.

    Status 0 means the manager was unreachable, which ``_http_request``
    already reported, so nothing is printed.
    """
    if not status:
        return
    text = resp["error"] if isinstance(resp, dict) and "error" in resp else resp
    sweep_print(f"  {_RED}FAIL{_RESET}  {label}: {text}")


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

        header, body = resp.split(b"\r\n\r\n", 1)
        header_str = header.decode("utf-8", errors="replace")
        if "101" not in header_str.splitlines()[0] if header_str else "":
            raise ConnectionError(_ws_refusal(header_str, body))

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
    campaign: str,
    controller_id: str | None = None,
    note: str | None = None,
    expected_jobs: int = 0,
    singular_dims: list[str] | None = None,
    max_concurrent: int = 0,
    skip_rules: dict[str, Any] | None = None,
    metric: str | None = None,
    goal: str | None = None,
    jobs: list[dict[str, Any]] | None = None,
    optimize: dict[str, Any] | None = None,
    code: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    capture_session: str | None = None,
) -> dict[str, Any] | None:
    """Create an experiment in *campaign* on the manager. Returns the response dict or None.

    *code* is the provenance of the code snapshot the jobs use, recorded with the experiment.

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
            **({"optimize": optimize} if optimize else {}),
            **({"code": code} if code else {}),
            **({"controls": controls} if controls else {}),
            **({"capture_session": capture_session} if capture_session else {}),
        },
    )
    if status in (200, 201) and isinstance(resp, dict):
        with_jobs = f" with {len(jobs)} job(s)" if jobs else ""
        sweep_print(f"  {_GREEN}OK{_RESET}    Experiment created: {experiment_id} "
                    f"(campaign {campaign}){with_jobs}")
        return resp
    _report_failure("Create experiment", status, resp)
    return None




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
    _report_failure("Submit jobs", status, resp)
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
        _report_failure("Get summary", status, resp)
    return None


def manager_list_experiments(
    manager: str,
    token: str,
    status_filter: str | None = None,
    campaign: str | None = None,
    quiet: bool = False,
) -> list[dict[str, Any]] | None:
    """List experiments on the manager, only *campaign*'s unless it is None."""
    path = "/api/experiments"
    if status_filter:
        path += f"?status={status_filter}"
    path = _with_campaign(path, campaign)
    status, resp = _http_request("GET", _manager_url(manager, path), token, timeout=10, quiet=quiet)
    if status == 200 and isinstance(resp, list):
        return resp
    if not quiet:
        _report_failure("List experiments", status, resp)
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
    _report_failure("Move experiment", status, resp)
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
    _report_failure("List jobs", status, resp)
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
    _report_failure(f"cancel {run_id}", status, resp)
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
    _report_failure(f"retry {run_id}", status, resp)
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
    _report_failure(f"set status {status}", s, resp)
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
_STATUS_ORDER = {"pending": 0, "dispatched": 1, "running": 2, "failed": 3, "cancelled": 4, "done": 5, "xfailed": 6,
                 "pruned": 7}


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
            vals = metric_values(all_metrics.get(run_id), metric)
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


def _leaderboard_summary(rows: list[dict[str, Any]], ranked: int) -> str:
    """``N completed runs, best step shown``, plus every run the ranking leaves out."""
    counts = collections.Counter(r["status"] for r in rows if r["value"] is None)
    missing = counts.pop("done", 0)
    left_out = [f"{n} {status}" for status, n in sorted(counts.items())]
    if missing:
        left_out.append(f"{missing} without this metric")
    summary = f"{_GREEN}{ranked}{_RESET} completed runs, best step shown"
    return f"{summary}; not ranked: {_YELLOW}{', '.join(left_out)}{_RESET}" if left_out else summary


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
    _leaderboard_header(metric, goal, _leaderboard_summary(rows, len(done)))
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
        status, resp = _http_request(
            "GET", _manager_url(manager, _with_campaign(f"/api/experiments/{experiment_id}/summary", campaign)),
            token, quiet=True)
        if status == 404:
            print(f"Error: no experiment {experiment_id}. Check the ID with `mlsweep ls`.",
                  file=sys.stderr, flush=True)
            sys.exit(1)
        if status == 200 and isinstance(resp, dict):
            counts = resp.get("job_counts") or {}
            active = sum(int(counts.get(s, 0)) for s in _ACTIVE_JOB_STATUSES)
            # Runs added after a `stop`, or left by a stop from an older version, never start.
            if active == int(counts.get("pending", 0)) and resp.get("status") == "aborted":
                active = 0
            # A Bayesian sweep adds runs as results come in, so a gap between them isn't the end.
            if active == 0 and not awaits_more_runs(manager, token, experiment_id, campaign):
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


def _get_experiment(manager: str, token: str, experiment_id: str, campaign: str | None) -> dict[str, Any]:
    status, resp = _http_request(
        "GET", _manager_url(manager, _with_campaign(f"/api/experiments/{experiment_id}", campaign)),
        token, quiet=True)
    return resp if status == 200 and isinstance(resp, dict) else {}


def experiment_status(manager: str, token: str, experiment_id: str, campaign: str | None = None) -> str | None:
    """The experiment's status (running, paused, completed, aborted), or None if unknown."""
    return _get_experiment(manager, token, experiment_id, campaign).get("status")


def manager_note(exp: dict[str, Any]) -> str | None:
    """Why the manager paused or completed an experiment (a run control, or a Bayesian sweep's optimizer)."""
    for field in ("controls", "optimize"):
        raw = exp.get(field)
        try:
            spec = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            continue
        if isinstance(spec, dict) and spec.get("note"):
            return str(spec["note"])
    return None


def awaits_more_runs(manager: str, token: str, experiment_id: str, campaign: str | None = None) -> bool:
    """True if the manager still expects runs here, so no active runs doesn't mean finished.

    That's a running Bayesian sweep, or an experiment created with ``expected_jobs``.
    """
    exp = _get_experiment(manager, token, experiment_id, campaign)
    return exp.get("status") == "running" and bool(exp.get("optimize") or exp.get("expected_jobs"))


def metric_keys(
    manager: str,
    token: str,
    experiment_id: str,
    run_id: str,
    campaign: str | None = None,
) -> list[str]:
    """The metric names *run_id* logged."""
    rows = manager_get_job_metrics(manager, token, experiment_id, run_id, campaign=campaign) or []
    return sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and k != "step"})



def manager_local_experiment_dir(manager: str, token: str, experiment_id: str) -> Path | None:
    """The experiment's folder if the manager keeps it on this machine, else None.

    The manager reports its folder and the random ID stored there.  Reading the
    same ID at that path here means both see the same files.
    """
    status, health = _http_request("GET", _manager_url(manager, "/api/health"), token,
                                   timeout=10, quiet=True)
    if status != 200 or not isinstance(health, dict):
        return None
    return local_experiment_dir(health, experiment_id)


def local_experiment_dir(health: dict[str, Any], experiment_id: str) -> Path | None:
    """The experiment's folder, if the manager whose ``/api/health`` is *health* keeps it here."""
    remote_dir, remote_id = health.get("mlsweep_dir"), health.get("mlsweep_dir_id")
    if not isinstance(remote_dir, str) or not isinstance(remote_id, str) or not remote_id:
        return None
    if mlsweep_dir_id(remote_dir) != remote_id:
        return None
    try:
        exp_dir = Path(remote_dir) / "experiments" / validate_experiment_dir(experiment_id)
    except ValueError:
        return None
    return exp_dir if exp_dir.is_dir() else None


def fetch_or_point_local(
    manager: str,
    token: str,
    experiment_id: str,
    output_dir: str | Path,
    explicit_output_dir: bool,
    campaign: str | None = None,
) -> None:
    """Download an experiment's results to *output_dir*.

    When the manager runs on this machine and no output folder was asked for,
    point at the manager's copy instead of making another one.
    """
    local = None if explicit_output_dir else manager_local_experiment_dir(manager, token, experiment_id)
    if local is not None:
        sweep_print(f"  {_GREEN}OK{_RESET}    The manager runs on this machine. Results are in {local}")
        sweep_print("        (pass --output-dir to make a copy)")
    else:
        manager_download_experiment(manager, token, experiment_id, output_dir, campaign=campaign)


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
# Code snapshots
# ===============================================================================

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            break
        n /= 1024
    return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"


def _put_object(manager: str, token: str, h: str, path: Path) -> bool:
    """Upload one file's bytes as object *h*, streamed rather than read into memory."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as body:
            status, resp = _http_request("PUT", _manager_url(manager, f"/api/objects/{h}"), token,
                                         data=body, headers={"Content-Length": str(size)}, timeout=600)
    except OSError as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  Cannot read {path}: {e}")
        return False
    if status in (200, 201):
        return True
    _report_failure(f"Upload {path}", status, resp)
    return False


def upload_snapshot(
    manager: str,
    token: str,
    tree: ScannedTree,
) -> tuple[str, dict[str, int]] | None:
    """Send the manager the files of *tree* it does not have, then register the snapshot.

    Returns (snapshot ID, counts of new and reused files and bytes), or None
    after reporting why it failed.
    """
    manifest = tree.manifest
    status, resp = _http_request("POST", _manager_url(manager, "/api/objects/missing"), token,
                                 json_data={"hashes": sorted(manifest.hashes())}, timeout=120)
    if status == 404:
        sweep_print(f"  {_RED}FAIL{_RESET}  The manager does not take code snapshots; it is older than "
                    "this mlsweep. Restart it with this version.")
        return None
    if status != 200 or not isinstance(resp, dict):
        _report_failure("Check stored code", status, resp)
        return None
    missing: list[str] = resp.get("missing", [])
    cap = int(resp.get("max_new_bytes") or 0)
    sizes = {f.hash: f.size for f in manifest.files}
    new_bytes = sum(sizes[h] for h in missing)
    missing_set = set(missing)
    if cap and new_bytes > cap:
        biggest = sorted((f for f in manifest.files if f.hash in missing_set), key=lambda f: -f.size)[:8]
        sweep_print(f"  {_RED}FAIL{_RESET}  {_fmt_bytes(new_bytes)} of new files to upload; the manager "
                    f"takes at most {_fmt_bytes(cap)} per submission. Largest new files:")
        for f in biggest:
            sweep_print(f"          {_fmt_bytes(f.size):>9}  {f.path}")
        sweep_print("        Leave out what the runs don't need with UPLOAD_EXCLUDE in the sweep file "
                    "or --exclude. Files the manager already has don't count.")
        return None
    for h in missing:
        src = tree.sources[h]
        if sizes[h] >= 16 << 20:
            sweep_print(f"  Uploading {src.relative_to(tree.root)} ({_fmt_bytes(sizes[h])})")
        if not _put_object(manager, token, h, src):
            return None
    status, resp = _http_request("POST", _manager_url(manager, "/api/snapshots"), token,
                                 json_data={"manifest": manifest.to_json()},
                                 timeout=120)
    if status != 201 or not isinstance(resp, dict):
        _report_failure("Register code snapshot", status, resp)
        return None
    return resp["artifact_id"], {
        "files": len(manifest.files), "bytes": manifest.total_bytes,
        "new_files": sum(f.hash in missing_set for f in manifest.files), "new_bytes": new_bytes,
    }


def _print_snapshot_summary(snapshot_id: str, stats: dict[str, int], provenance: dict[str, Any],
                            tree: ScannedTree) -> None:
    sweep_print(f"  {_GREEN}OK{_RESET}    Code {short_id(snapshot_id)}: {stats['files']} files, "
                f"{_fmt_bytes(stats['bytes'])}")
    reused_files = stats["files"] - stats["new_files"]
    sweep_print(f"          new: {stats['new_files']} files, {_fmt_bytes(stats['new_bytes'])} · "
                f"reused: {reused_files} files, {_fmt_bytes(stats['bytes'] - stats['new_bytes'])}")
    vcs = provenance.get("vcs")
    if vcs and vcs.get("commit"):
        dirty = f" +{len(vcs.get('changed_files') or [])} uncommitted" if vcs.get("dirty") else ""
        sweep_print(f"          git: {vcs['commit'][:10]} {vcs.get('branch') or '(detached)'}{dirty}")
    big = [(p, b) for p, b in untracked_summary(vcs, tree.manifest) if b >= 1 << 20]
    if big:
        sweep_print("          not tracked by git: " + ", ".join(f"{p} ({_fmt_bytes(b)})" for p, b in big))


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
    writer_factory: Any = None,
    variations: list[dict[str, Any]] | None = None,
    output_dir: str = "",
    campaign: str | None = None,
) -> None:
    """Connect to manager WebSocket and display live job status until idle timeout.

    If *writer_factory* is provided (e.g. MultiWriterFactory), run writers are
    created lazily for each run and fed metric / finish events.
    *variations* is used to look up combos when creating writers.  A Bayesian sweep
    has none up front, so a run not listed there is looked up on the manager.
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
        status, job = _http_request(
            "GET",
            _manager_url(manager, _with_campaign(f"/api/jobs/{run_id}?experiment_id={experiment_id}", campaign)),
            token,
            quiet=True,
        )
        combo = _parse_combo(job.get("combo")) if status == 200 and isinstance(job, dict) else None
        return combo or {}

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


def describe_controls(controls: dict[str, Any] | None) -> list[str]:
    """One line per run control the manager will enforce."""
    from mlsweep._sweep import format_duration

    if not controls:
        return []
    out = []
    if controls.get("max_runtime"):
        out.append(f"runs that take longer than {format_duration(controls['max_runtime'])} are stopped "
                   "and marked failed (MAX_RUNTIME)")
    prune = controls.get("prune")
    if prune:
        if prune["strategy"] == "asha":
            at = f"steps {prune['min_steps']}, x{prune['reduction_factor']} after"
            rule = f"outside the best 1/{prune['reduction_factor']}"
        else:
            at = f"step {prune['min_steps']}, then every {prune['every']}"
            rule = "worse than the median"
        out.append(f"PRUNE ({prune['strategy']}): at {at}, a run whose best {prune['metric']} is {rule} "
                   f"of the runs that reached that step is stopped, once {prune['n_warmup']} have")
    check = controls.get("metric_check")
    if check:
        out.append(f"the sweep pauses if its first {check['runs']} finished run(s) all log no "
                   f"{check['metric']!r} (METRIC_CHECK)")
    return out


def _load_and_lint(sweep_file: str, as_json: bool = False) -> dict[str, Any]:
    """Load a sweep file, print its lint findings, and exit on an error.

    With *as_json*, an error exits with ``{"error", "exit_code", "findings"}``.
    """
    from mlsweep._lint import Finding, lint_module
    from mlsweep._sweep import _load_module, sweep_info

    try:
        mod = _load_module(sweep_file)
    except Exception as e:
        sweep_print(f"{_RED}Error: cannot load {sweep_file}: {e!r}{_RESET}")
        sys.exit(1)
    findings: list[Finding] = lint_module(mod, sweep_file)
    for f in findings:
        color = _RED if f.severity == "error" else _YELLOW
        sweep_print(f"{color}{f.severity}:{_RESET} {f.message}")
    if any(f.severity == "error" for f in findings):
        sweep_print(f"{_RED}{sweep_file} has errors; nothing was submitted.{_RESET} "
                    "(`mlsweep lint` checks a sweep file on its own.)")
        if as_json:
            emit({"error": f"{sweep_file} has errors; nothing was submitted", "exit_code": 1,
                  "findings": [{"severity": f.severity, "message": f.message} for f in findings]})
        sys.exit(1)
    info = sweep_info(mod, sweep_file)
    info["findings"] = [{"severity": f.severity, "message": f.message} for f in findings]
    return info


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


# ===============================================================================
# Watch subcommand
# ===============================================================================


def _watch_cmd(args: list[str], prog: str = "mlsweep_run watch", connect: Any = None) -> None:
    """Watch an experiment's progress via WebSocket event stream.

    *connect* makes the event stream from ``(url, token, timeout=)``; by default a ``_WebSocket``.
    """
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

    ws = (connect or _WebSocket)(ws_url, token, timeout=10.0)
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
                emit_line(event)
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
        emit(out)
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

    output_dir = parsed.output_dir or str(_mlsweep_dir() / "downloads" / parsed.experiment)
    fetch_or_point_local(manager, token, parsed.experiment, output_dir,
                         explicit_output_dir=parsed.output_dir is not None, campaign=campaign)

    if had_failure:
        sys.exit(1)


# ===============================================================================
# Main
# ===============================================================================


def main() -> None:
    from mlsweep._jsonout import contract

    with contract(sys.argv[1:]):
        _main()


def _main() -> None:
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
            "  MLSWEEP_MANAGER  Manager URL (default: http://localhost:7891)\n"
            "  MLSWEEP_TOKEN    Authentication token for manager\n"
            "  MLSWEEP_CAMPAIGN Campaign to submit under (required unless --campaign or a context sets one)\n"
            "  MLSWEEP_DIR      Local state dir holding manager.token (default: ~/.mlsweep)\n"
        ),
    )
    parser.add_argument("sweep_file", help="Path to sweep .py file")
    _add_connection_args(parser)
    _add_campaign_args(parser)
    parser.add_argument("--output-dir", default=None,
                        help="Directory for local submit-side artifacts (submit log + manifest; "
                             "default: ~/.mlsweep/submissions). "
                             "Results themselves live on the manager under ~/.mlsweep/experiments/.")
    parser.add_argument("--experiment", default=None,
                        help="Experiment name (default: <sweep>_<timestamp>)")
    parser.add_argument("--append", action="store_true",
                        help="Add this sweep's runs to the experiment named by --experiment, if it exists")
    parser.add_argument("--note", default=None,
                        help="Human-readable note stored with the experiment")
    parser.add_argument("--priority", type=int, default=0,
                        help="Job priority (higher = earlier, default: 0)")
    parser.add_argument("--stream", action="store_true",
                        help="Watch live status over the manager's WebSocket stream. Only watches: "
                             "the sweep carries on without it, Bayesian sweeps included.")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch results after submission (if --stream not used)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without execution")
    parser.add_argument("--validate", action="store_true",
                        help="Validate sweep config, print all combinations, and exit")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max automatic retries of a run lost with its worker (default: 2)")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="Leave matching files out of the uploaded project (repeatable; adds to the "
                             "sweep file's UPLOAD_EXCLUDE). A pattern without '/' matches a name at any "
                             "depth, one with '/' matches the path from the project root, e.g. runs or data/*.bin.")
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
             "cluster (0 = unlimited). Use it to take only a slice of a shared cluster. "
             "In a Bayesian sweep it is also how many trials run at once (default: n_initial).")
    parser.add_argument("--color", action="store_true",
                        help="Enable ANSI color in human-readable output (default: off)")
    parser.add_argument("--json", action="store_true",
                        help="Print the result as JSON: the submitted experiment, or with --validate "
                             "and --dry-run the runs it would make. Not with --stream or --fetch.")

    args, extra = parser.parse_known_args(argv)
    if args.json and (args.stream or args.fetch):
        parser.error("--json prints one result; it can't be used with --stream or --fetch")
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.validate:  # checks the sweep locally, so no campaign is needed
        campaign = None if args.all_campaigns else _selected_campaign(args)
    else:
        campaign = _resolve_campaign(args)
    if campaign is None and not args.validate:
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

    # ── Load and lint the sweep ─────────────────────────────────────────────
    info = _load_and_lint(args.sweep_file, as_json=args.json)
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

    spec: dict[str, Any] | None = None
    if method == "bayes":
        try:
            spec = validate_optimize_spec({
                **optimize_cfg, "sweep_name": sweep_name, "options": options_to_wire(options),
                "extra_flags": list(extra_flags), "job": {
                    "command": command, "cli_overrides": extra, "gpus_per_run": gpus_per_run,
                    "nodes_per_run": nodes_per_run, "set_dist_env": set_dist_env, "run_from": run_from,
                    "priority": args.priority, "max_retries": args.max_retries,
                    "setup_command": args.setup_command,
                },
            })
        except ValueError as e:
            sweep_print(f"{_RED}Error: {e}{_RESET}")
            sys.exit(1)

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
            if args.json:
                emit({"sweep": sweep_name, "method": "bayes", "budget": optimize_cfg["budget"],
                      "metric": optimize_cfg["metric"], "goal": optimize_cfg["goal"],
                      "dimensions": {k[1:]: ({"distribution": o["distribution"], "min": o["min"], "max": o["max"]}
                                             if o.get("_type") == "continuous" else o["_values"])
                                     for k, o in options.items() if o["_values"] != [None]
                                     or o.get("_type") == "continuous"},
                      "controls": info.get("controls"), "findings": info["findings"]})
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
        if args.json:
            emit({"sweep": sweep_name, "method": "grid", "combinations": len(all_variations),
                  "excluded": excluded, "runs": [{"name": v["name"], "combo": v["combo"]} for v in all_variations],
                  "controls": info.get("controls"), "findings": info["findings"]})
        sys.exit(0)

    # ── Generate variations ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    rand_suffix = secrets.token_hex(2)  # 4 hex chars
    experiment_id = args.experiment or f"{sweep_name}_{timestamp}_{rand_suffix}"
    output_dir = os.path.abspath(args.output_dir or str(_mlsweep_dir() / "submissions"))
    exp_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    if not args.dry_run:
        _log_file = open(os.path.join(exp_dir, "sweep.log"), "w")

    if spec is not None:
        if args.append:
            sweep_print(f"{_RED}Error: --append can't add a Bayesian sweep to an existing experiment.{_RESET}")
            sys.exit(1)
        expected = spec["budget"]
        variations: list[dict[str, Any]] = []
    else:
        variations = generate_variations(sweep_name, options, exclude_fn, extra_flags)
        expected = count_expected(options)

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
    for line in describe_controls(info.get("controls")):
        sweep_print(f"{_BOLD}Control:{_RESET} {line}")
    if extra:
        sweep_print(f"{_BOLD}Extra overrides:{_RESET} {' '.join(extra)}")

    if args.dry_run:
        if spec is not None:
            from mlsweep._bayes import suggest
            variations = suggest(spec, [], min(spec["n_initial"], spec["budget"]))
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
        if args.json:
            emit({"experiment_id": experiment_id, "campaign": campaign, "dry_run": True,
                  "runs": [{"name": v["name"], "combo": v["combo"],
                            "command": list(command) + v["overrides"] + list(extra)} for v in variations],
                  "findings": info["findings"]})
        return

    manager, token = _manager_token(args)

    if args.experiment and not args.append and _get_experiment(manager, token, args.experiment, campaign):
        sweep_print(f"{_RED}Error: experiment {args.experiment} already exists.{_RESET} "
                    "Pass --append to add these runs to it, or choose another name.")
        sys.exit(1)

    sweep_print(f"\n{_CYAN}{'=' * 80}{_RESET}")
    sweep_print(f"{_BOLD}Connecting to manager:{_RESET} {_CYAN}{manager}{_RESET}")
    sweep_print(f"{_CYAN}{'=' * 80}{_RESET}\n")

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

    # ── 1. Snapshot the code, uploading only what the manager lacks ────────
    sweep_print("Snapshotting project code...")
    upload_exclude = info["upload_exclude"] + args.exclude
    if upload_exclude:
        sweep_print(f"  Leaving out: {', '.join(upload_exclude)}")
    try:
        tree = scan_tree(_PROJECT_ROOT, exclude_patterns=upload_exclude,
                         cache_path=_mlsweep_dir() / "hashcache.db")
    except OSError as e:
        sweep_print(f"  {_RED}FAIL{_RESET}  Cannot read the project: {e}")
        sys.exit(1)
    uploaded = upload_snapshot(manager, token, tree)
    if uploaded is None:
        sys.exit(1)
    artifact_id, snap_stats = uploaded
    provenance = submit_provenance(root=Path(_PROJECT_ROOT), manifest=tree.manifest,
                                   snapshot_id=artifact_id, sweep_file=args.sweep_file)
    _print_snapshot_summary(artifact_id, snap_stats, provenance, tree)

    # ── 2. Build job payloads ───────────────────────────────────────────────
    job_payloads = _build_job_payloads(
        variations=variations,
        experiment_id=experiment_id,
        artifact_id=artifact_id,
        command=command,
        cli_overrides=extra,
        gpus_per_run=gpus_per_run,
        nodes_per_run=nodes_per_run,
        set_dist_env=set_dist_env,
        run_from=run_from,
        priority=args.priority,
        max_retries=args.max_retries,
        setup_command=args.setup_command,
    )

    # ── 3. Create experiment and submit jobs atomically ────────────────────
    if job_payloads:
        sweep_print("Submitting jobs...")
    if spec is not None:
        spec["job"]["artifact_id"] = artifact_id
    assert campaign is not None  # resolved above; only --validate runs without one, and it returned
    if not manager_create_experiment(
        manager, token, experiment_id,
        name=sweep_name,
        note=args.note,
        singular_dims=singular_dims(options),
        max_concurrent=args.max_concurrent,
        skip_rules=_skip_rules(options) if method == "grid" else None,
        metric=info.get("metric"),
        goal=info.get("goal"),
        campaign=campaign,
        jobs=job_payloads,
        optimize=spec,
        code=provenance,
        controls=info.get("controls"),
        capture_session=(capture_session := capture_session_id()),
    ):
        sys.exit(1)
    # The session that submitted a sweep is followed, so its trace uploads
    # (only while capture is on for this machine).
    from mlsweep._capture import follow_if_on
    follow_if_on(capture_session, why=f"submitted {experiment_id}", manager=manager)

    # ── 4. Write local manifest ────────────────────────────────────────────
    _write_manifest(exp_dir, experiment_id, variations, note=args.note)
    if args.json:
        emit({"experiment_id": experiment_id, "campaign": campaign, "method": method,
              "runs": len(variations), "budget": spec["budget"] if spec else None,
              "snapshot": artifact_id, "controls": info.get("controls"), "findings": info["findings"]})
        if _log_file:
            _log_file.close()
        return

    # ── 5. Stream or fetch ─────────────────────────────────────────────────
    if args.stream:
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"Streaming live status (Ctrl+C to stop)")
        sweep_print(f"{'=' * 80}\n")

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
        fetch_or_point_local(manager, token, experiment_id, exp_dir,
                             explicit_output_dir=args.output_dir is not None, campaign=campaign)
    else:
        sweep_print(f"\n{'=' * 80}")
        scope = " ".join(_campaign_argv(campaign))
        if spec is not None:
            sweep_print(f"Bayesian sweep submitted. The manager chooses the runs, up to {spec['budget']} "
                        f"trials with results.")
            sweep_print(f"Control: mlsweep optimize {experiment_id} --manager {manager} {scope}")
        else:
            sweep_print(f"{n} jobs submitted.")
        sweep_print(f"Watch:   mlsweep watch {experiment_id} --manager {manager} {scope}")
        sweep_print(f"Fetch:   mlsweep fetch --experiment {experiment_id} --manager {manager} {scope} --wait")
        sweep_print(f"Results: ~/.mlsweep/experiments/{experiment_id}/")
        sweep_print(f"{'=' * 80}")

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
