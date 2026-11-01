"""Export an experiment's stored metrics to TensorBoard or Weights & Biases.

The live writers (``_writer_tensorboard``, ``_writer_wandb``) stream metrics
while ``mlsweep run`` is attached.  This module works after the fact, from the
metrics the manager already stored, so any experiment can be exported from the
web UI at any time.

TensorBoard export needs no extra packages.  Event files are encoded here by
hand: TFRecord framing around a few protobuf messages (scalar summaries plus
the hparams plugin's session start/end records, so the HParams dashboard shows
the sweep as a table).  W&B export needs ``wandb`` installed in the manager's
environment and uploads from the manager process.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from mlsweep._manager_db import FINISHED_JOB_STATUSES

# ===============================================================================
# TensorBoard event files
# ===============================================================================


def _make_crc32c_table() -> list[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    return table


_CRC32C_TABLE = _make_crc32c_table()


def _crc32c(data: bytes) -> int:
    table = _CRC32C_TABLE
    crc = 0xFFFFFFFF
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _record(payload: bytes) -> bytes:
    """Frame *payload* as one TFRecord (length, crc, data, crc)."""
    header = struct.pack("<Q", len(payload))
    return (header + struct.pack("<I", _masked_crc(header))
            + payload + struct.pack("<I", _masked_crc(payload)))


# Protobuf wire encoding, only the pieces the messages below use.

def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _f_varint(num: int, n: int) -> bytes:
    return _varint(num << 3) + _varint(n)


def _f_double(num: int, x: float) -> bytes:
    return _varint(num << 3 | 1) + struct.pack("<d", x)


def _f_float(num: int, x: float) -> bytes:
    return _varint(num << 3 | 5) + struct.pack("<f", x)


def _f_bytes(num: int, data: bytes) -> bytes:
    return _varint(num << 3 | 2) + _varint(len(data)) + data


def _f_str(num: int, s: str) -> bytes:
    return _f_bytes(num, s.encode())


def _event(wall_time: float, step: int, *, summary: bytes | None = None,
           file_version: str | None = None) -> bytes:
    msg = _f_double(1, wall_time) + _f_varint(2, step)
    if file_version is not None:
        msg += _f_str(3, file_version)
    if summary is not None:
        msg += _f_bytes(5, summary)
    return _record(msg)


def _scalar_value(tag: str, x: float) -> bytes:
    """A ``Summary.Value`` holding a ``simple_value``."""
    return _f_bytes(1, _f_str(1, tag) + _f_float(2, x))


# The hparams plugin reads its records from summaries tagged like this, with
# the plugin data in the metadata and a placeholder scalar tensor.
_HPARAMS_START_TAG = "_hparams_/session_start_info"
_HPARAMS_END_TAG = "_hparams_/session_end_info"
_NULL_TENSOR = _f_varint(1, 1) + _f_bytes(2, _f_bytes(2, b""))  # DT_FLOAT, shape [0]
_STATUS_SUCCESS, _STATUS_FAILURE = 1, 2


def _hparams_value(tag: str, plugin_data: bytes) -> bytes:
    metadata = _f_bytes(1, _f_str(1, "hparams") + _f_bytes(2, plugin_data))
    return _f_bytes(1, _f_str(1, tag) + _f_bytes(9, metadata) + _f_bytes(8, _NULL_TENSOR))


def _hparam(value: Any) -> bytes:
    """A ``google.protobuf.Value`` for one hyperparameter."""
    if isinstance(value, bool):
        return _f_varint(4, int(value))
    if isinstance(value, (int, float)):
        return _f_double(2, float(value))
    return _f_str(3, str(value))


def _session_start(hparams: dict[str, Any], start: float) -> bytes:
    info = b"".join(
        _f_bytes(1, _f_str(1, k) + _f_bytes(2, _hparam(v))) for k, v in hparams.items()
    )
    # No group_name: the dashboard would merge every session in a group into one row.
    info += _f_double(5, start)
    return _hparams_value(_HPARAMS_START_TAG, _f_varint(1, 0) + _f_bytes(3, info))


def _session_end(status: int, end: float) -> bytes:
    info = _f_varint(1, status) + _f_double(2, end)
    return _hparams_value(_HPARAMS_END_TAG, _f_varint(1, 0) + _f_bytes(4, info))


def tfevents_for_run(
    rows: list[dict[str, Any]],
    *,
    hparams: dict[str, Any],
    status: str,
    start_time: float,
    end_time: float | None,
) -> bytes:
    """Encode one run's metric *rows* as the contents of a tfevents file.

    *rows* are ``{"step": n, metric: value, ...}`` dicts as the manager stores
    them.  Non-numeric values are skipped.  Metrics carry no timestamps, so
    every event gets *start_time* as its wall time.
    """
    out = [_event(start_time, 0, file_version="brain.Event:2"),
           _event(start_time, 0, summary=_session_start(hparams, start_time))]
    last_step = 0
    for row in rows:
        step = int(row.get("step", 0))
        last_step = max(last_step, step)
        values = b"".join(
            _scalar_value(k, float(v)) for k, v in row.items()
            if k != "step" and isinstance(v, (int, float))
        )
        if values:
            out.append(_event(start_time, step, summary=values))
    if status in FINISHED_JOB_STATUSES:
        code = _STATUS_SUCCESS if status == "done" else _STATUS_FAILURE
        end = end_time if end_time is not None else start_time
        out.append(_event(end, last_step, summary=_session_end(code, end)))
    return b"".join(out)


def tfevents_name(start_time: float) -> str:
    return f"events.out.tfevents.{int(start_time)}.mlsweep"


# ===============================================================================
# Weights & Biases
# ===============================================================================


def wandb_info() -> dict[str, Any]:
    """Whether wandb is importable by the manager, without importing it."""
    if importlib.util.find_spec("wandb") is None:
        return {"available": False, "version": None, "env_key": False}
    try:
        version: str | None = importlib.metadata.version("wandb")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "available": True,
        "version": version,
        "env_key": bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")),
    }


@dataclass
class ExportScope:
    """What an export covers: one experiment, one campaign, or everything.

    *experiments* maps each covered experiment id to its label, the name used
    for its folder in a TensorBoard zip and its group in W&B.  *campaign_of*
    maps each covered experiment id to its campaign.
    """

    kind: str  # experiment | campaign | all
    campaign: str | None = None
    experiment_id: str | None = None
    experiments: dict[str, str] = field(default_factory=dict)
    campaign_of: dict[str, str] = field(default_factory=dict)

    @property
    def campaigns(self) -> set[str]:
        return set(self.campaign_of.values())

    @property
    def label(self) -> str:
        if self.kind == "experiment":
            return f"experiment {self.experiments.get(self.experiment_id or '', self.experiment_id)}"
        if self.kind == "campaign":
            return f"campaign {self.campaign}"
        return "all campaigns"


def experiment_labels(exps: list[tuple[str, str]]) -> dict[str, str]:
    """Labels for ``(experiment_id, name)`` pairs, unique within the list.

    The name is used when no other experiment shares it (renames make that
    possible), else the id is appended.
    """
    counts: dict[str, int] = {}
    for _, name in exps:
        counts[name] = counts.get(name, 0) + 1
    return {eid: name if counts[name] == 1 or name == eid else f"{name}_{eid}"
            for eid, name in exps}


@dataclass
class ExportRun:
    experiment_id: str
    run_id: str
    experiment: str = ""  # the experiment's label
    status: str = "pending"  # pending | uploading | done | failed | skipped
    url: str | None = None
    error: str | None = None


@dataclass
class WandbExport:
    """Progress of one W&B upload, as reported by ``/api/exports``."""

    scope: ExportScope
    project: str
    entity: str | None
    export_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    target: str = "wandb"
    status: str = "running"  # running | done | failed | cancelled
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    runs: list[ExportRun] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_id": self.export_id,
            "scope": self.scope.kind,
            "scope_label": self.scope.label,
            "campaign": self.scope.campaign,
            "experiment_id": self.scope.experiment_id,
            "experiments": len(self.scope.experiments),
            "campaigns": sorted(self.scope.campaigns),
            "target": self.target,
            "project": self.project,
            "entity": self.entity,
            "status": self.status,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "total": len(self.runs),
            "done": sum(r.status in ("done", "failed", "skipped") for r in self.runs),
            "runs": [vars(r) for r in self.runs],
        }


def _wandb_settings(wandb: Any, api_key: str | None) -> Any:
    """Settings that keep the manager's own machine out of the uploaded run.

    Without them wandb would record the manager's CPU/GPU stats, console,
    git state and code as if they belonged to the training run.
    """
    wanted: dict[str, Any] = {
        "api_key": api_key,
        "console": "off",
        "silent": True,
        "save_code": False,
        "disable_git": True,
        "disable_code": True,
        "disable_job_creation": True,
        "x_disable_stats": True,
        "x_disable_meta": True,
        "x_disable_machine_info": True,
        "x_save_requirements": False,
    }
    fields = getattr(wandb.Settings, "model_fields", None)
    kwargs = {k: v for k, v in wanted.items()
              if v is not None and (fields is None or k in fields)}
    return wandb.Settings(**kwargs)


def upload_wandb_run(
    wandb: Any,
    *,
    export: WandbExport,
    run_id: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    summary: dict[str, Any],
    group: str,
    tags: list[str],
    api_key: str | None,
    work_dir: str,
) -> str | None:
    """Create one W&B run from stored *rows* and return its URL (blocking)."""
    run = wandb.init(
        project=export.project,
        entity=export.entity,
        name=run_id,
        group=group,
        tags=tags,
        config=config,
        dir=work_dir,
        # Honour WANDB_MODE, so ``WANDB_MODE=offline`` (or ``disabled``) works
        # for testing without an account; otherwise wandb's default is online.
        mode=os.environ.get("WANDB_MODE") or "online",
        reinit="create_new",
        settings=_wandb_settings(wandb, api_key),
    )
    ok = False
    try:
        # The manager may have cancelled while wandb.init was blocking.
        if not export.cancel.is_set():
            for row in rows:
                if export.cancel.is_set():
                    break
                data = {k: v for k, v in row.items() if k != "step"}
                if data:
                    run.log(data, step=int(row.get("step", 0)))
            if not export.cancel.is_set():
                for k, v in summary.items():
                    if v is not None:
                        run.summary[k] = v
                ok = True
    finally:
        run.finish(exit_code=0 if ok else 1)
    return getattr(run, "url", None)

