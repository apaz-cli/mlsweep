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

DEFAULT_SCRATCH_DIR = "/tmp/mlsweep"

# Experiment statuses whose pending jobs the scheduler must NOT dispatch.
# 'paused' is a temporary hold (resumable); 'aborted' is a permanent stop.
# 'running' and 'completed' remain schedulable so that retrying a job in a
# finished experiment works without a separate status flip.
NON_SCHEDULABLE_EXPERIMENT_STATUSES: tuple[str, ...] = ("paused", "aborted")
_CAMPAIGN_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def validate_campaign(name: str) -> str:
    """Return *name* if it is a valid campaign name; raise ValueError otherwise."""
    if not isinstance(name, str) or not _CAMPAIGN_RE.match(name):
        raise ValueError(f"campaign must be 1-128 chars of [a-zA-Z0-9_-], got {name!r}")
    return name


_ARTIFACT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:+=\-]{0,199}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def validate_artifact_id(artifact_id: Any) -> str:
    """Return *artifact_id* if it is safe in a file name; raise ValueError otherwise."""
    if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError("artifact_id must be 1-200 chars of [a-zA-Z0-9._:+=-], starting with a letter or digit")
    return artifact_id


def validate_run_id(run_id: Any) -> str:
    """Return *run_id* if it stays inside its experiment's directory; raise ValueError otherwise.

    Run names come from sweep option values, so a ``/`` is allowed and makes a
    nested directory.  Empty, ``.`` and ``..`` components are not.
    """
    if not isinstance(run_id, str) or not run_id or len(run_id) > 1024 or _CONTROL_CHARS_RE.search(run_id):
        raise ValueError("run_id must be 1-1024 chars with no control characters")
    if any(part in ("", ".", "..") for part in run_id.split("/")):
        raise ValueError(f"run_id must not have empty, '.' or '..' path components: {run_id!r}")
    return run_id


def validate_experiment_dir(experiment_id: Any) -> str:
    """Return *experiment_id* if it names one directory; raise ValueError otherwise.

    New experiments are held to the stricter ``_sanitize_experiment_id``.  This
    is the check for looking one up, so an older name still resolves.
    """
    validate_run_id(experiment_id)
    if "/" in experiment_id:
        raise ValueError(f"experiment_id must not contain '/': {experiment_id!r}")
    return str(experiment_id)


def _mlsweep_dir() -> Path:
    """mlsweep state directory ($MLSWEEP_DIR, default ~/.mlsweep)."""
    return Path(os.environ.get("MLSWEEP_DIR", "~/.mlsweep")).expanduser()


DIR_ID_FILE = "manager.id"


def mlsweep_dir_id(mlsweep_dir: str | Path, *, create: bool = False) -> str:
    """Random ID stored in *mlsweep_dir*, or "" when there is none.

    A client that reads the same ID from the path the manager reports is looking
    at the manager's own folder, so it can use the files there without copying.
    """
    path = Path(mlsweep_dir) / DIR_ID_FILE
    try:
        return path.read_text().strip()
    except OSError:
        if not create:
            return ""
    new = _token_hex(16)
    try:
        with open(path, "x") as f:
            f.write(new)
        return new
    except FileExistsError:
        return mlsweep_dir_id(mlsweep_dir)
    except OSError:
        return ""


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


def same_attempt(a: int, b: int) -> bool:
    """Whether two attempt numbers name the same attempt.  0 is unknown (from a
    peer predating attempt numbers) and matches any attempt."""
    return not a or not b or a == b


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


# The modules a worker runs: its import closure, plus the logger its runs use
# to reach it.  tests/test_worker_update.py checks this against what importing
# the worker loads.  Edits to other modules (manager, notes, CLI) leave the
# fingerprint alone, so they do not mark workers outdated.
WORKER_MODULES = ("__init__.py", "_colors.py", "_env.py", "_shared.py", "_snapshot.py", "_topology.py",
                  "logger.py", "worker.py")


def code_fingerprint(package_dir: Path | None = None) -> str:
    """Hash of the worker modules in *package_dir* (default: this package) as on disk now.

    A source checkout changes code without changing its version.  The manager
    compares each worker's ``CODE_FINGERPRINT`` with this, taken from its own
    package directory, which is what updating a worker would install.  A
    wheel carries the same ``.py`` files as the tree it was built from.
    """
    root = package_dir or Path(__file__).resolve().parent
    stats: list[tuple[int, int] | None] = []
    for name in WORKER_MODULES:
        try:
            st = (root / name).stat()
            stats.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stats.append(None)
    key = (str(root), tuple(stats))
    cached = _fingerprint_cache.get(key)
    if cached is None:
        h = hashlib.sha256()
        for name in WORKER_MODULES:
            try:
                h.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
            except OSError:
                h.update(name.encode() + b"\0missing\0")
        cached = _fingerprint_cache[key] = h.hexdigest()[:16]
    return cached


# (package dir, per-module (mtime, size)) -> fingerprint.  The manager asks on
# every worker listing; unchanged files then cost a stat each, not a rehash.
_fingerprint_cache: dict[tuple[str, tuple[Any, ...]], str] = {}


# ── Protocol messages ──────────────────────────────────────────────────────────
# Each message is a length-prefixed JSON object with a "t" field.  A worker
# whose PROTOCOL_VERSION differs from the manager's is refused at hello.
PROTOCOL_VERSION = 4  # 3: code arrives as a snapshot (manifest + objects), see mlsweep._snapshot
#                      4: GPU health arrives after the hello too (MsgGpuHealth, pending_gpus)
# A run is identified by (experiment, run_id).  Run names are derived from the
# sweep, so two experiments of the same sweep use the same run_ids.
# Each dispatch of a run is a new attempt, numbered from 1.  MsgRun carries it
# and the worker's messages about the run echo it, so a late message from an
# attempt the manager has taken back (an evicted run's result) is not mistaken
# for the current attempt's.  0 means unknown, from a peer predating the field.
# Controller → Worker messages use t in {"hello","run","cancel","cleanup","replay","shutdown","ping"}.
# Worker → Controller messages use t in {"whello","started","log","metric","syncreq","result","cleaned","pong","gpu_stats","gpu_health"}.

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
    # The run's code: a snapshot ("snap:...") or a legacy tarball ("sha256:...").
    artifact_url: str = ""
    # Where to get it: for a snapshot, its manifest URL (objects come from
    # /api/objects/ on the same manager); for a tarball, the tarball itself.
    setup_command: list[str] = field(default_factory=list)
    # Command list executed in the workspace after artifact extraction
    # and before training. Run without shell for safety.
    attempt: int = 0
    t: str = "run"


@dataclass
class MsgCancel:
    run_id: str
    experiment: str
    # Seconds after the SIGTERM to SIGKILL what is still running (0 = never).
    # An older worker ignores it and only sends the SIGTERM.
    kill_after: float = 0.0
    t: str = "cancel"


@dataclass
class MsgCleanup:
    run_id: str
    experiment: str
    final: bool = False     # True = run finished and artifacts synced; safe to delete scratch
    attempt: int = 0
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
    resuming: list[dict[str, Any]]  # [{run_id, experiment, attempt, pid, gpu_ids}]
    scratch_dir: str
    max_jobs_per_gpu: int = 1     # worker's per-GPU packing cap (0 = unlimited)
    # Results of runs that ended but that no manager has acknowledged yet:
    # [{run_id, success, elapsed, exit_code, experiment, attempt}].  Re-sent on every hello so a
    # result produced while the manager was disconnected (or restarting) is not lost.
    completed: list[dict[str, Any]] = field(default_factory=list)
    # GPUs the worker enumerated but excluded because CUDA context creation failed
    # (e.g. a faulted device pending reset).  Informational; the scheduler only sees
    # ``gpus``.  Defaulted, so an older peer that omits it stays compatible.
    unhealthy_gpus: list[int] = field(default_factory=list)
    # Why each unhealthy GPU was excluded, keyed by str(gpu) (JSON keys are strings):
    # "cuInit failed: CUDA_ERROR_... (n)" or "CUDA probe timed out after 60s".
    unhealthy_reasons: dict[str, str] = field(default_factory=dict)
    # GPUs whose probe had not finished when the hello was sent; a MsgGpuHealth
    # follows when it does.  Excluded from ``gpus`` until then.
    pending_gpus: list[int] = field(default_factory=list)
    protocol: int = 0             # PROTOCOL_VERSION of the worker
    code: str = ""                # the worker's CODE_FINGERPRINT; "" from older workers
    t: str = "whello"


@dataclass
class MsgStarted:
    run_id: str
    pid: int
    experiment: str
    attempt: int = 0
    t: str = "started"


@dataclass
class MsgLog:
    run_id: str
    seq: int                # byte offset in training.log just past this chunk
    data: str               # whole lines
    start: int              # byte offset in training.log where this chunk begins
    experiment: str
    attempt: int = 0
    t: str = "log"


@dataclass
class MsgMetric:
    run_id: str
    step: int
    data: dict[str, Any]
    experiment: str
    attempt: int = 0
    step_seconds: float | None = None  # the logger's wall seconds per step, if known
    total_steps: int | None = None  # the step the run ends at, if the logger was told
    t: str = "metric"


@dataclass
class MsgHeartbeat:
    """The script is alive and working, though it logged nothing (``MLSweepLogger.heartbeat``)."""

    run_id: str
    experiment: str
    attempt: int = 0
    t: str = "heartbeat"


@dataclass
class MsgSyncReq:
    run_id: str
    experiment: str
    attempt: int = 0
    t: str = "syncreq"


@dataclass
class MsgResult:
    run_id: str
    success: bool
    elapsed: float
    exit_code: int
    experiment: str
    attempt: int = 0
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
    # Each entry: {"gpu": int, "util_pct": int, "mem_used_mb": int, "mem_total_mb": int,
    # "name": str}.  Older workers send no "name".
    t: str = "gpu_stats"


@dataclass
class MsgGpuHealth:
    """The worker's GPU health changed after its hello: a pending probe finished,
    or a GPU that had failed passed a later probe.  Same fields as the hello's."""
    gpus: list[int] = field(default_factory=list)
    unhealthy_gpus: list[int] = field(default_factory=list)
    unhealthy_reasons: dict[str, str] = field(default_factory=dict)
    pending_gpus: list[int] = field(default_factory=list)
    t: str = "gpu_health"


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
    "heartbeat": MsgHeartbeat,
    "result": MsgResult,
    "cleaned": MsgCleaned,
    "pong": MsgPong,
    "gpu_stats": MsgGpuStats,
    "gpu_health": MsgGpuHealth,
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


def read_msg(sock: socket.socket, limit: int | None = None) -> bytes | None:
    """Read one length-prefixed message from a blocking socket.

    Returns None on EOF, on error, or if the message is longer than *limit*.
    """
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
    if limit is not None and n > limit:
        return None
    return _recv_exactly(n)


def _dir_size(path: Path) -> int:
    """Total bytes of files under *path* (symlinks counted as links, not followed)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total
