"""Shared utilities and wire protocol for mlsweep worker ↔ controller communication."""

import asyncio
import json
import os
import socket
import struct
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from secrets import token_hex as _token_hex
from typing import Any


# ── Utilities ──────────────────────────────────────────────────────────────────

# ANSI escape codes
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


def _resolve_safe_subpath(base: str | Path, sub: str | None) -> str:
    """Join *base* and relative *sub*; raise ValueError if *sub* escapes *base*."""
    base = str(base)
    if not sub:
        return base
    resolved = os.path.realpath(os.path.join(base, sub))
    base_resolved = os.path.realpath(base)
    if os.path.commonpath([resolved, base_resolved]) != base_resolved:
        raise ValueError(f"path {sub!r} escapes base directory {base!r}")
    return resolved


def _git_root(path: str) -> str | None:
    """Return the root directory of the git repo containing path, or None."""
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _parse_tag_value(s: str) -> bool | int | float | str:
    """Convert a tag value string to a typed Python value."""
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _val_sort_key(v: Any) -> tuple[int, Any]:
    """Sort key for dim values: bools first, then numbers, then strings."""
    if isinstance(v, bool):
        return (0, str(v))
    if isinstance(v, (int, float)):
        return (1, v)
    return (2, str(v))


def _detect_sub_dims(
    runs: list[dict[str, Any]],
    dims: dict[str, list[Any]],
) -> dict[str, dict[str, Any]]:
    """Detect dims that only appear when a parent dim has a specific value.

    runs:  list of dicts with "hash" and "combo" keys.
    dims:  {dim_name: [sorted values]}.
    Returns {child_dim: {"parentDim": ..., "parentValue": ...}}.
    """
    all_names = {r["hash"] for r in runs}
    names_with = {dim: {r["hash"] for r in runs if dim in r["combo"]} for dim in dims}
    sub_dims: dict[str, dict[str, Any]] = {}
    for dim in dims:
        if names_with[dim] == all_names:
            continue  # universal dim — not a subdim
        for parent_dim in dims:
            if parent_dim == dim:
                continue
            for parent_val in dims[parent_dim]:
                names_with_parent = {
                    r["hash"] for r in runs if r["combo"].get(parent_dim) == parent_val
                }
                if names_with_parent == names_with[dim]:
                    sub_dims[dim] = {"parentDim": parent_dim, "parentValue": parent_val}
                    break
            if dim in sub_dims:
                break
    return sub_dims


# ── Protocol messages ──────────────────────────────────────────────────────────
# One JSON object per line over TCP, terminated by \n.  All messages have a "t" field.
# Controller → Worker messages use t in {"hello","run","cancel","cleanup","replay","bye","shutdown","ping"}.
# Worker → Controller messages use t in {"whello","started","log","metric","syncreq","result","cleaned","pong"}.

# ── Controller → Worker ────────────────────────────────────────────────────────

@dataclass
class MsgHello:
    token: str
    controller_id: str
    t: str = "hello"


@dataclass
class MsgRun:
    command: list[str]
    run_id: str = field(default_factory=lambda: _token_hex(8))
    experiment: str = "pool"
    env: dict[str, str] = field(default_factory=dict)
    gpu_ids: list[int] = field(default_factory=list)
    # Filled by WorkerPool from WorkerConfig; set explicitly only when
    # sending MsgRun directly to a worker without a pool.
    remote_dir: str = ""
    scratch: str = ""
    run_from: str | None = None
    set_dist_env: bool = False
    files: dict[str, str] = field(default_factory=dict)
    # {workspace-relative path: text content}. Worker creates an isolated
    # workspace directory and writes these files into it.
    # Sets MLSWEEP_WORKSPACE; cwd becomes workspace instead of remote_dir.
    return_files: list[str] = field(default_factory=list)
    # Workspace-relative paths copied into artifacts/ after the run,
    # before the normal artifact rsync.
    artifact_id: str = ""
    # Opaque identifier for the artifact tarball (used as download subpath).
    artifact_url: str = ""
    # Base URL of the artifact manager (e.g. "http://host:port").
    # Worker fetches {artifact_url}/{artifact_id}.tar.gz if both are set.
    setup_command: list[str] = field(default_factory=list)
    # Command list executed in the workspace after artifact extraction
    # and before training. Run without shell for safety.
    t: str = "run"


@dataclass
class MsgCancel:
    run_id: str
    t: str = "cancel"


@dataclass
class MsgCleanup:
    run_id: str
    t: str = "cleanup"


@dataclass
class MsgReplay:
    run_id: str
    log_seq: int
    metric_seq: int
    t: str = "replay"


@dataclass
class MsgShutdown:
    t: str = "shutdown"


@dataclass
class MsgPing:
    t: str = "ping"


# ── Worker → Controller ────────────────────────────────────────────────────────

@dataclass
class MsgWorkerHello:
    gpus: list[int]
    topo: dict[str, int]          # "{gpu_a},{gpu_b}" → score (JSON requires string keys)
    resuming: list[dict[str, Any]]  # [{run_id, log_seq, metric_seq, pid}]
    scratch_dir: str
    t: str = "whello"


@dataclass
class MsgStarted:
    run_id: str
    pid: int
    t: str = "started"


@dataclass
class MsgLog:
    run_id: str
    seq: int
    data: str
    t: str = "log"


@dataclass
class MsgMetric:
    run_id: str
    step: int
    data: dict[str, Any]
    t: str = "metric"


@dataclass
class MsgSyncReq:
    run_id: str
    t: str = "syncreq"


@dataclass
class MsgResult:
    run_id: str
    success: bool
    elapsed: float
    exit_code: int
    t: str = "result"


@dataclass
class MsgCleaned:
    run_id: str
    t: str = "cleaned"


@dataclass
class MsgPong:
    t: str = "pong"


@dataclass
class MsgGpuStats:
    stats: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"gpu": int, "util_pct": int, "mem_used_mb": int, "mem_total_mb": int}
    t: str = "gpu_stats"


_MSG_TYPES: dict[str, type] = {
    "hello": MsgHello,
    "run": MsgRun,
    "cancel": MsgCancel,
    "cleanup": MsgCleanup,
    "replay": MsgReplay,
    "shutdown": MsgShutdown,
    "ping": MsgPing,
    "whello": MsgWorkerHello,
    "started": MsgStarted,
    "log": MsgLog,
    "metric": MsgMetric,
    "syncreq": MsgSyncReq,
    "result": MsgResult,
    "cleaned": MsgCleaned,
    "pong": MsgPong,
    "gpu_stats": MsgGpuStats,
}


def encode(msg: Any) -> bytes:
    """Encode a protocol message to a length-prefixed frame: 4-byte big-endian length + JSON payload."""
    payload = json.dumps(asdict(msg)).encode()
    return struct.pack(">I", len(payload)) + payload


def decode(payload: bytes) -> Any:
    """Decode a JSON payload bytes to the appropriate protocol message dataclass."""
    obj: dict[str, Any] = json.loads(payload)
    t = obj.get("t")
    cls = _MSG_TYPES.get(t)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"Unknown message type: {t!r}")
    return cls(**obj)


async def aread_msg(reader: asyncio.StreamReader) -> bytes:
    """Read one length-prefixed message from an asyncio StreamReader."""
    hdr = await reader.readexactly(4)
    (n,) = struct.unpack(">I", hdr)
    return await reader.readexactly(n)


def read_msg(sock: socket.socket) -> bytes | None:
    """Read one length-prefixed message from a blocking socket. Returns None on EOF/error."""
    def _recv_exactly(n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    hdr = _recv_exactly(4)
    if hdr is None:
        return None
    (n,) = struct.unpack(">I", hdr)
    return _recv_exactly(n)
