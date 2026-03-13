"""Shared utilities and wire protocol for mlsweep worker ↔ controller communication."""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


# ── Utilities (from _utils.py) ─────────────────────────────────────────────────


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
# Controller → Worker messages use t in {"hello","run","cancel","cleanup","replay","bye","shutdown"}.
# Worker → Controller messages use t in {"whello","started","log","metric","syncreq","result","cleaned"}.

# ── Controller → Worker ────────────────────────────────────────────────────────

@dataclass
class MsgHello:
    token: str
    controller_id: str
    t: str = "hello"


@dataclass
class MsgRun:
    run_id: str
    experiment: str
    command: list[str]
    env: dict[str, str]
    gpu_ids: list[int]
    remote_dir: str
    scratch: str
    run_from: str | None = None
    set_dist_env: bool = False
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


_MSG_TYPES: dict[str, type] = {
    "hello": MsgHello,
    "run": MsgRun,
    "cancel": MsgCancel,
    "cleanup": MsgCleanup,
    "replay": MsgReplay,
    "shutdown": MsgShutdown,
    "whello": MsgWorkerHello,
    "started": MsgStarted,
    "log": MsgLog,
    "metric": MsgMetric,
    "syncreq": MsgSyncReq,
    "result": MsgResult,
    "cleaned": MsgCleaned,
}


def encode(msg: Any) -> bytes:
    """Encode a protocol message dataclass to a wire line (JSON + newline)."""
    return (json.dumps(asdict(msg)) + "\n").encode()


def decode(line: bytes) -> Any:
    """Decode a wire line to the appropriate protocol message dataclass."""
    obj: dict[str, Any] = json.loads(line)
    t = obj.get("t")
    cls = _MSG_TYPES.get(t)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"Unknown message type: {t!r}")
    return cls(**obj)
