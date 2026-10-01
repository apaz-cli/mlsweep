"""Shared utilities and wire protocol for mlsweep worker ↔ controller communication."""

import asyncio
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from secrets import token_hex as _token_hex
from typing import Any


# ── Utilities ──────────────────────────────────────────────────────────────────

# ANSI color support.  The constants are sentinels whose rendering depends on
# the global toggle in mlsweep._colors; import them from here for backwards
# compatibility.  Color is off unless an entry point receives --color.
from mlsweep._colors import (  # noqa: F401
    _BLUE as _BLUE,
    _BOLD as _BOLD,
    _BRIGHT_BLUE as _BRIGHT_BLUE,
    _BRIGHT_GREEN as _BRIGHT_GREEN,
    _CYAN as _CYAN,
    _DIM as _DIM,
    _GREEN as _GREEN,
    _MAGENTA as _MAGENTA,
    _RED as _RED,
    _RESET as _RESET,
    _YELLOW as _YELLOW,
    color_enabled as color_enabled,
    set_color as set_color,
    strip_color_flag as strip_color_flag,
)

DEFAULT_MANAGER_URL = "http://localhost:7891"

# Every experiment belongs to one campaign; this one when none is given.
DEFAULT_CAMPAIGN = "default"
_CAMPAIGN_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def validate_campaign(name: str) -> str:
    """Return *name* if it is a valid campaign name; raise ValueError otherwise."""
    if not isinstance(name, str) or not _CAMPAIGN_RE.match(name):
        raise ValueError(f"campaign must be 1-128 chars of [a-zA-Z0-9_-], got {name!r}")
    return name


def _mlsweep_dir() -> Path:
    """mlsweep state directory ($MLSWEEP_DIR, default ~/.mlsweep)."""
    return Path(os.environ.get("MLSWEEP_DIR", "~/.mlsweep")).expanduser()


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


def _val_sort_key(v: Any) -> tuple[int, Any]:
    """Sort key for dim values: bools first, then numbers, then strings."""
    if isinstance(v, bool):
        return (0, str(v))
    if isinstance(v, (int, float)):
        return (1, v)
    return (2, str(v))


def dist_master_port(experiment: str, run_id: str) -> int:
    """Deterministic torch.distributed master port for a run, in [20000, 30000)."""
    return 20000 + int(hashlib.md5(f"{experiment}/{run_id}".encode()).hexdigest()[:4], 16) % 10000


# ── Run logs ──────────────────────────────────────────────────────────────────

# A stored or sent log chunk is at most this many bytes of whole lines.
LOG_CHUNK_BYTES = 64 * 1024


def line_chunks(data: bytes) -> list[bytes]:
    """Split *data* into pieces of at most LOG_CHUNK_BYTES, at line ends where possible."""
    view = memoryview(data)
    chunks = []
    start = 0
    while start < len(data):
        end = min(start + LOG_CHUNK_BYTES, len(data))
        if end < len(data):
            end = data.rfind(b"\n", start, end) + 1 or end
        chunks.append(bytes(view[start:end]))
        start = end
    return chunks


# ── Protocol messages ──────────────────────────────────────────────────────────
# Each message is a length-prefixed JSON object with a "t" field.  A worker
# whose PROTOCOL_VERSION differs from the manager's is refused at hello.
PROTOCOL_VERSION = 2
# A run is identified by (experiment, run_id).  Run names are derived from the
# sweep, so two experiments of the same sweep use the same run_ids.
# Controller → Worker messages use t in {"hello","run","cancel","cleanup","replay","shutdown","ping"}.
# Worker → Controller messages use t in {"whello","started","log","metric","syncreq","result","cleaned","pong","gpu_stats"}.

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
    experiment: str
    t: str = "cancel"


@dataclass
class MsgCleanup:
    run_id: str
    experiment: str
    final: bool = False     # True = run finished and artifacts synced; safe to delete scratch
    t: str = "cleanup"


@dataclass
class MsgReplay:
    """Re-send the run's log from byte *log_seq* on, and all of its metrics."""
    run_id: str
    experiment: str
    log_seq: int
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
    resuming: list[dict[str, Any]]  # [{run_id, experiment, pid, gpu_ids}]
    scratch_dir: str
    max_jobs_per_gpu: int = 1     # worker's per-GPU packing cap (0 = unlimited)
    # Results of runs that ended but that no manager has acknowledged yet:
    # [{run_id, success, elapsed, exit_code, experiment}].  Re-sent on every hello so a
    # result produced while the manager was disconnected (or restarting) is not lost.
    completed: list[dict[str, Any]] = field(default_factory=list)
    protocol: int = 0             # PROTOCOL_VERSION of the worker
    t: str = "whello"


@dataclass
class MsgStarted:
    run_id: str
    pid: int
    experiment: str
    t: str = "started"


@dataclass
class MsgLog:
    run_id: str
    seq: int                # byte offset in training.log just past this chunk
    data: str               # whole lines
    start: int              # byte offset in training.log where this chunk begins
    experiment: str
    t: str = "log"


@dataclass
class MsgMetric:
    run_id: str
    step: int
    data: dict[str, Any]
    experiment: str
    t: str = "metric"


@dataclass
class MsgSyncReq:
    run_id: str
    experiment: str
    t: str = "syncreq"


@dataclass
class MsgResult:
    run_id: str
    success: bool
    elapsed: float
    exit_code: int
    experiment: str
    t: str = "result"


@dataclass
class MsgCleaned:
    run_id: str
    experiment: str
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
_MSG_FIELDS: dict[type, frozenset[str]] = {
    cls: frozenset(f.name for f in fields(cls)) for cls in _MSG_TYPES.values()
}


def encode(msg: Any) -> bytes:
    """Encode a protocol message to a length-prefixed frame: 4-byte big-endian length + JSON payload."""
    payload = json.dumps(asdict(msg)).encode()
    return struct.pack(">I", len(payload)) + payload


def decode(payload: bytes) -> Any:
    """Decode a JSON payload bytes to the appropriate protocol message dataclass."""
    return from_obj(json.loads(payload))


def from_obj(obj: dict[str, Any]) -> Any:
    """Build the protocol message dataclass for a decoded JSON object."""
    t = obj.get("t")
    cls = _MSG_TYPES.get(t)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"Unknown message type: {t!r}")
    # Ignore fields this version doesn't know, so a newer peer that adds a field
    # (e.g. MsgWorkerHello.completed) can still talk to an older one.
    known = _MSG_FIELDS[cls]
    return cls(**{k: v for k, v in obj.items() if k in known})


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
