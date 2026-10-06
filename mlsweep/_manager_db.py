"""SQLite database layer for mlsweep manager.

Provides:
  - Schema initialisation (tables, indexes)
  - Data-class row types: JobRecord, ExperimentRecord, WorkerRecord, ArtifactRecord
  - Full CRUD async functions for every entity
  - Bulk / query helpers used by the manager scheduler and HTTP API

Write convention
----------------
All mutating statements that use ``RETURNING *`` must go through ``_exec_one``
or ``_exec_all`` rather than the bare ``cursor = await db.execute(...)`` form.
See the comment block above those helpers for the full explanation.  The caller
is always responsible for ``await db.commit()`` so that multiple writes can share
one transaction (batching).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import dataclasses
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Literal, NamedTuple, Sequence, TypeVar

import aiosqlite

from mlsweep._shared import DEFAULT_CAMPAIGN
from mlsweep._sweep import SkipIndex

_T = TypeVar("_T")

JobStatus = Literal["pending", "dispatched", "running", "done", "failed", "xfailed", "cancelled"]
ExperimentStatus = Literal["running", "paused", "completed", "aborted"]

ACTIVE_JOB_STATUSES: tuple[JobStatus, ...] = ("dispatched", "running")
FINISHED_JOB_STATUSES: tuple[JobStatus, ...] = ("done", "failed", "xfailed", "cancelled")
# Experiment statuses whose pending jobs the scheduler must NOT dispatch.
# 'paused' is a temporary hold (resumable); 'aborted' is a permanent stop.
# 'running' and 'completed' remain schedulable so that retrying a job in a
# finished experiment works without a separate status flip.
NON_SCHEDULABLE_EXPERIMENT_STATUSES: tuple[str, ...] = ("paused", "aborted")


def _sql_in(statuses: Sequence[str]) -> str:
    """``status IN (...)`` for a fixed tuple of status constants (never user input)."""
    return "status IN (" + ", ".join(f"'{s}'" for s in statuses) + ")"


# Restricts a jobs query to the experiments of one campaign (one bound parameter).
_IN_CAMPAIGN = "experiment_id IN (SELECT experiment_id FROM experiments WHERE campaign = ?)"

_ACTIVE_IN = _sql_in(ACTIVE_JOB_STATUSES)
_UNFINISHED_IN = _sql_in(("pending", *ACTIVE_JOB_STATUSES))
_FINISHED_IN = _sql_in(FINISHED_JOB_STATUSES)
WorkerStatus = Literal["offline", "connected", "reconnecting", "dead"]


# ===============================================================================
# Row types
# ===============================================================================


@dataclass(order=False)
class ExperimentRecord:
    """A registered sweep experiment."""

    experiment_id: str
    name: str
    submit_time: datetime
    campaign: str = DEFAULT_CAMPAIGN
    controller_id: str | None = None
    note: str | None = None
    status: ExperimentStatus = "running"
    expected_jobs: int = 0
    singular_dims: str = "[]"  # JSON list of dim names that are singular probes
    max_concurrent: int = 0  # max simultaneously-running jobs for this exp; 0 = unlimited
    skip_rules: str = "{}"  # JSON {dim: {monotonic, singular, _values}} for should_skip; {} = none
    metric: str | None = None  # ranking metric for this sweep (e.g. "val_loss")
    goal: str | None = None  # ranking direction: "minimize" | "maximize"


@dataclass(order=False)
class WorkerRecord:
    """A worker that has connected at least once."""

    worker_id: str
    host: str
    remote_dir: str
    status: WorkerStatus = "offline"
    last_seen: datetime | None = None
    scratch_dir: str | None = None
    port: int = 7890
    ssh_key: str | None = None
    venv: str | None = None
    devices: str | None = None  # JSON list of ints
    unhealthy_devices: str | None = None  # JSON list of ints excluded by the CUDA probe
    last_error: str | None = None  # human-readable reason for the last failure


@dataclass(order=False)
class JobRecord:
    """A single run / job tracked in the database.

    This is the canonical row for both pending and completed jobs.  The
    in-memory pending list holds a subset of these rows (status='pending').
    """

    run_id: str
    experiment_id: str
    priority: int
    submit_time: datetime
    command: str  # JSON list of strings
    status: JobStatus = "pending"
    dispatch_time: datetime | None = None
    start_time: datetime | None = None
    finish_time: datetime | None = None
    elapsed: float | None = None
    exit_code: int | None = None
    worker_id: str | None = None
    env: str = "{}"  # JSON object
    artifact_id: str | None = None
    setup_command: str | None = None
    gpus_per_run: int = 1
    nodes_per_run: int = 1
    set_dist_env: bool = False
    run_from: str | None = None
    return_files: str = "[]"  # JSON list of strings
    files: str = "{}"  # JSON object: {rel_path: text_content}
    retry_count: int = 0
    max_retries: int = 2
    combo: str = "{}"  # JSON object
    dispatched_gpu_ids: str | None = None  # JSON list of ints, set on dispatch
    label: str | None = None
    job_key: int = 0  # integer id used by the logs and metrics tables
    attempt: int = 0  # number of times the job has been dispatched

    @property
    def key(self) -> tuple[str, str]:
        """(experiment_id, run_id), how the manager identifies a run."""
        return (self.experiment_id, self.run_id)


@dataclass(order=False)
class ArtifactRecord:
    """A content-addressed artifact stored on the manager node."""

    artifact_id: str
    size_bytes: int | None = None
    stored_at: datetime | None = None
    ref_count: int = 0
    setup_command: str | None = None


@dataclass(order=False)
class JobNodeRecord:
    """One node of a multi-node job (durable placement + per-node result)."""

    run_id: str
    experiment_id: str
    node_rank: int
    worker_id: str | None = None
    gpu_ids: str = "[]"  # JSON list of ints
    status: JobStatus = "dispatched"
    success: bool | None = None
    elapsed: float | None = None


# ===============================================================================
# Helpers – row → dict / dict → row
# ===============================================================================


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    """Map a database row to a JobRecord."""
    return JobRecord(
        run_id=row["run_id"],
        experiment_id=row["experiment_id"],
        priority=row["priority"],
        submit_time=_utc(row["submit_time"]),
        command=row["command"],
        status=row["status"],
        dispatch_time=_maybe_utc(row["dispatch_time"]),
        start_time=_maybe_utc(row["start_time"]),
        finish_time=_maybe_utc(row["finish_time"]),
        elapsed=row["elapsed"],
        exit_code=row["exit_code"],
        worker_id=row["worker_id"],
        env=row["env"],
        artifact_id=row["artifact_id"],
        setup_command=row["setup_command"],
        gpus_per_run=row["gpus_per_run"],
        nodes_per_run=row["nodes_per_run"],
        set_dist_env=bool(row["set_dist_env"]),
        run_from=row["run_from"],
        return_files=row["return_files"],
        files=row["files"],
        retry_count=row["retry_count"],
        max_retries=row["max_retries"],
        combo=row["combo"],
        dispatched_gpu_ids=row["dispatched_gpu_ids"],
        label=row["label"],
        job_key=row["job_key"],
        attempt=row["attempt"],
    )


def _row_to_experiment(row: sqlite3.Row) -> ExperimentRecord:
    """Map a database row to an ExperimentRecord."""
    return ExperimentRecord(
        experiment_id=row["experiment_id"],
        name=row["name"],
        submit_time=_utc(row["submit_time"]),
        campaign=row["campaign"],
        controller_id=row["controller_id"],
        note=row["note"],
        status=row["status"],
        expected_jobs=row["expected_jobs"],
        singular_dims=row["singular_dims"],
        max_concurrent=row["max_concurrent"],
        skip_rules=row["skip_rules"],
        metric=row["metric"],
        goal=row["goal"],
    )


def _row_to_worker(row: sqlite3.Row) -> WorkerRecord:
    """Map a database row to a WorkerRecord."""
    return WorkerRecord(
        worker_id=row["worker_id"],
        host=row["host"],
        remote_dir=row["remote_dir"],
        status=row["status"],
        last_seen=_maybe_utc(row["last_seen"]),
        scratch_dir=row["scratch_dir"],
        port=row["port"],
        ssh_key=row["ssh_key"],
        venv=row["venv"],
        devices=row["devices"],
        unhealthy_devices=row["unhealthy_devices"],
        last_error=row["last_error"],
    )


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRecord:
    """Map a database row to an ArtifactRecord."""
    return ArtifactRecord(
        artifact_id=row["artifact_id"],
        size_bytes=row["size_bytes"],
        stored_at=_maybe_utc(row["stored_at"]),
        ref_count=row["ref_count"],
        setup_command=row["setup_command"],
    )


def _utc(ts: float) -> datetime:
    """A REAL epoch-seconds column as a UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _maybe_utc(ts: float | None) -> datetime | None:
    return None if ts is None else _utc(ts)


def _now_epoch() -> float:
    """Return current time as a Unix timestamp (float)."""
    return datetime.now(timezone.utc).timestamp()


# ===============================================================================
# Write helpers
# ===============================================================================
#
# aiosqlite's Connection.execute() returns a Result object (aiosqlite/context.py)
# that is both awaitable and an async context manager.  Its __aexit__ closes the
# cursor, which resets the underlying SQLite statement to "not busy".  Without
# that close, Python 3.12+ commit() raises "cannot commit transaction - SQL
# statements in progress" when a RETURNING * statement was stepped but not yet
# finished (e.g. the last fetchone() left rows on the wire, or the cursor was
# never explicitly closed before commit).
#
# Rule for write functions: use _exec_one / _exec_all for any statement that
# produces rows (RETURNING *).  Plain `await db.execute(sql)` is safe for DML
# with no RETURNING clause because those statements step to completion during
# execute() itself.  Never write `cursor = await db.execute(sql)` on a
# RETURNING statement followed by a commit.
#
# The caller is responsible for `await db.commit()` (and rollback on error).
# Keeping commit at the call site preserves the ability to batch multiple
# _exec_* calls into a single transaction.


async def _exec_one(
    db: aiosqlite.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
) -> sqlite3.Row | None:
    """Execute *sql* and return the first row, closing the cursor immediately.

    The caller must ``await db.commit()`` (or rollback) after this returns.
    """
    async with db.execute(sql, params) as cursor:
        return await cursor.fetchone()


async def _exec_all(
    db: aiosqlite.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[sqlite3.Row]:
    """Execute *sql* and return all rows, closing the cursor immediately.

    The caller must ``await db.commit()`` (or rollback) after this returns.
    """
    async with db.execute(sql, params) as cursor:
        return list(await cursor.fetchall())


# ===============================================================================
# Schema initialisation
# ===============================================================================


# Jobs are addressed by (experiment_id, run_id) everywhere except in the logs
# and metrics tables, which hold almost all of the data and so refer to a job
# by its integer job_key instead of repeating both ids on every row.  Each
# dispatch of a job is a new attempt; logs and metrics are kept per attempt so
# a retried run never collides with (or reads back) an earlier attempt's rows.
_JOBS_TABLE = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_key            INTEGER PRIMARY KEY,
        run_id             TEXT NOT NULL,
        experiment_id      TEXT NOT NULL REFERENCES experiments(experiment_id),
        priority           INTEGER NOT NULL DEFAULT 0,
        status             TEXT NOT NULL DEFAULT 'pending',
        submit_time        REAL NOT NULL,
        dispatch_time      REAL,
        start_time         REAL,
        finish_time        REAL,
        elapsed            REAL,
        exit_code          INTEGER,
        worker_id          TEXT REFERENCES workers(worker_id),
        command            TEXT NOT NULL,
        env                TEXT NOT NULL DEFAULT '{}',
        artifact_id        TEXT REFERENCES artifacts(artifact_id),
        setup_command      TEXT,
        gpus_per_run       INTEGER NOT NULL DEFAULT 1,
        nodes_per_run      INTEGER NOT NULL DEFAULT 1,
        set_dist_env       INTEGER NOT NULL DEFAULT 0,
        run_from           TEXT,
        return_files       TEXT NOT NULL DEFAULT '[]',
        files              TEXT NOT NULL DEFAULT '{}',
        retry_count        INTEGER NOT NULL DEFAULT 0,
        max_retries        INTEGER NOT NULL DEFAULT 2,
        combo              TEXT NOT NULL DEFAULT '{}',
        dispatched_gpu_ids TEXT,
        label              TEXT,
        attempt            INTEGER NOT NULL DEFAULT 0,
        UNIQUE (experiment_id, run_id)
    )
"""

# While a run is live each metric step is its own row (data: JSON TEXT).
# When the run finishes, its rows are packed into one row (data: a zlib BLOB of
# "step<TAB>json" lines, keyed by the first step), which is several times smaller.
_METRICS_TABLE = """
    CREATE TABLE IF NOT EXISTS metrics (
        job_key  INTEGER NOT NULL,
        attempt  INTEGER NOT NULL,
        step     INTEGER NOT NULL,
        data     NOT NULL,
        PRIMARY KEY (job_key, attempt, step)
    ) WITHOUT ROWID
"""

# A log row is a chunk of whole lines; seq is the byte offset in the run's
# training.log just past the chunk.  data is TEXT, or zlib-compressed UTF-8
# as a BLOB when that is smaller.
_LOGS_TABLE = """
    CREATE TABLE IF NOT EXISTS logs (
        job_key  INTEGER NOT NULL,
        attempt  INTEGER NOT NULL,
        seq      INTEGER NOT NULL,
        data     NOT NULL,
        PRIMARY KEY (job_key, attempt, seq)
    ) WITHOUT ROWID
"""

def _pack_log(text: str) -> str | bytes:
    """Compress a log chunk if that saves at least 10%."""
    raw = text.encode("utf-8")
    if len(raw) >= 256:
        packed = zlib.compress(raw, 6)
        if len(packed) < len(raw) * 0.9:
            return packed
    return text


def _unpack_log(data: str | bytes) -> str:
    if isinstance(data, bytes):
        return zlib.decompress(data).decode("utf-8", errors="replace")
    return data


def _merge_metric_rows(rows: Sequence[tuple[int, str | bytes]]) -> dict[int, str]:
    """``{step: json}`` from per-step and packed metric rows.

    Per-step rows are already merged by ``insert_metric``, so the first row for
    a step is the whole dict.  A packed row (zlib BLOB) overwrites per-step rows
    for the same step.
    """
    by_step: dict[int, str] = {}
    for step, data in rows:
        if isinstance(data, str):
            by_step.setdefault(step, data)
    for _, data in rows:
        if isinstance(data, bytes):
            for line in zlib.decompress(data).decode("utf-8").split("\n"):
                s, _, d = line.partition("\t")
                by_step[int(s)] = d
    return by_step


def _merge_metric_json(datas: Sequence[str]) -> str | None:
    """One step's JSON dicts merged into one (later dicts win per key).

    ``None`` if none of them parses as a dict.
    """
    merged: dict[str, Any] | None = None
    for data in datas:
        try:
            d = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict):
            merged = merged or {}
            merged.update(d)
    return None if merged is None else json.dumps(merged, separators=(",", ":"))


def _pack_metric_rows(
    rows: Sequence[tuple[int, str | bytes]],
    extra: Sequence[tuple[int, str]],
) -> tuple[int, bytes]:
    """``(first_step, zlib blob)`` for ``pack_metrics``; CPU-bound, run off the loop.

    Folds same-step dicts from the synced metrics.jsonl (which may have several
    ``log()`` calls at one step), then overlays the stored rows (which include
    any already-packed BLOB).  Live rows win on conflicting keys, and extra-only
    keys and steps are preserved.  Only steps with more than one source are
    parsed; the rest pass through as stored.
    """
    sources: dict[int, list[str]] = {}
    for step, data in [*extra, *_merge_metric_rows(rows).items()]:
        sources.setdefault(step, []).append(data)
    by_step: dict[int, str] = {}
    for step, datas in sources.items():
        merged = datas[0] if len(datas) == 1 else _merge_metric_json(datas)
        if merged is not None:
            by_step[step] = merged
    text = "\n".join(f"{s}\t{by_step[s]}" for s in sorted(by_step)).encode("utf-8")
    return min(by_step), zlib.compress(text, 6)


async def pack_metrics(
    db: aiosqlite.Connection,
    job_key: int,
    attempt: int,
    extra: Sequence[tuple[int, str]] = (),
) -> None:
    """Pack a finished attempt's metrics into one compressed row.

    *extra* holds ``(step, json)`` rows from the run's synced metrics.jsonl;
    they fill in steps the live stream missed.
    """
    rows = [(r[0], r[1]) for r in await _exec_all(
        db, "SELECT step, data FROM metrics WHERE job_key = ? AND attempt = ?", (job_key, attempt),
    )]
    if len(rows) + len(extra) < 2:
        return  # nothing to merge
    first_step, packed = await asyncio.to_thread(_pack_metric_rows, rows, extra)
    await db.execute("DELETE FROM metrics WHERE job_key = ? AND attempt = ?", (job_key, attempt))
    await db.execute(
        "INSERT INTO metrics (job_key, attempt, step, data) VALUES (?, ?, ?, ?)",
        (job_key, attempt, first_step, packed),
    )
    await db.commit()


async def _add_missing_columns(
    db: aiosqlite.Connection, table: str, columns: dict[str, str],
) -> None:
    """Add any of *columns* that *table* is missing (idempotent migration).

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so the current column names are
    read from ``PRAGMA table_info`` first.  The table/column names come from
    module constants, never user input.
    """
    async with db.execute(f"PRAGMA table_info({table})") as cursor:
        existing = {row["name"] for row in await cursor.fetchall()}
    for name, decl in columns.items():
        if name not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


async def init_db(db: aiosqlite.Connection) -> None:
    """Create tables and indexes if they do not exist (idempotent).

    Enables WAL mode and foreign key enforcement on the connection.
    synchronous=NORMAL is safe under WAL and avoids an fsync on every
    commit, which matters because every log line and metric is a commit.
    """
    db.row_factory = sqlite3.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA foreign_keys=ON")

    # ── experiments ─────────────────────────────────────────────────
    await db.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id  TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            campaign       TEXT NOT NULL DEFAULT 'default',
            submit_time    REAL NOT NULL,
            controller_id  TEXT,
            note           TEXT,
            status         TEXT NOT NULL DEFAULT 'running',
            expected_jobs  INTEGER NOT NULL DEFAULT 0,
            singular_dims  TEXT NOT NULL DEFAULT '[]',
            max_concurrent INTEGER NOT NULL DEFAULT 0,
            skip_rules     TEXT NOT NULL DEFAULT '{}',
            metric         TEXT,
            goal           TEXT
        );
    """)
    # Idempotent migration for databases created before metric/goal/campaign
    # existed.  Experiments from before campaigns land in the default one.
    await _add_missing_columns(db, "experiments", {
        "metric": "TEXT", "goal": "TEXT", "campaign": "TEXT NOT NULL DEFAULT 'default'",
    })

    # ── workers ─────────────────────────────────────────────────────
    await db.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            worker_id     TEXT PRIMARY KEY,
            host          TEXT NOT NULL,
            remote_dir    TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'offline',
            last_seen     REAL,
            scratch_dir   TEXT,
            port          INTEGER NOT NULL DEFAULT 7890,
            ssh_key       TEXT,
            venv          TEXT,
            devices       TEXT,
            unhealthy_devices TEXT,
            last_error    TEXT
        );
    """)
    # Idempotent migration for databases created before the device-health probe.
    await _add_missing_columns(db, "workers", {
        "unhealthy_devices": "TEXT",
    })

    # ── artifacts ───────────────────────────────────────────────────
    await db.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id   TEXT PRIMARY KEY,
            size_bytes    INTEGER,
            stored_at     REAL NOT NULL,
            ref_count     INTEGER NOT NULL DEFAULT 0,
            setup_command TEXT
        );
    """)

    await db.execute(_JOBS_TABLE)
    await db.execute(_METRICS_TABLE)
    await db.execute(_LOGS_TABLE)

    # ── job_nodes ────────────────────────────────────────────────────
    # One row per node of a multi-node job (nodes_per_run > 1).  This is the
    # durable, restart-safe placement + aggregation state: each node records
    # which worker/GPUs it runs on and its individual result.  A multi-node job
    # is complete once none of its node rows are still non-terminal — a query,
    # not an in-memory counter that a manager restart would lose.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS job_nodes (
            run_id        TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            node_rank     INTEGER NOT NULL,
            worker_id     TEXT,
            gpu_ids       TEXT,
            status        TEXT NOT NULL DEFAULT 'dispatched',
            success       INTEGER,
            elapsed       REAL,
            PRIMARY KEY (run_id, experiment_id, node_rank)
        );
    """)

    # ── indexes ─────────────────────────────────────────────────────
    await db.executescript("""
        CREATE INDEX IF NOT EXISTS idx_experiments_campaign
            ON experiments(campaign);
        CREATE INDEX IF NOT EXISTS idx_jobs_dispatch
            ON jobs(status, priority DESC, submit_time ASC);
        CREATE INDEX IF NOT EXISTS idx_jobs_worker
            ON jobs(worker_id);
    """)
    await db.commit()


# ===============================================================================
# Experiments
# ===============================================================================


async def create_experiment(
    db: aiosqlite.Connection,
    *,
    experiment_id: str,
    name: str,
    campaign: str = DEFAULT_CAMPAIGN,
    controller_id: str | None = None,
    note: str | None = None,
    status: ExperimentStatus = "running",
    expected_jobs: int = 0,
    singular_dims: list[str] | None = None,
    max_concurrent: int = 0,
    skip_rules: dict[str, Any] | None = None,
    metric: str | None = None,
    goal: str | None = None,
    commit: bool = True,
) -> ExperimentRecord:
    """Insert a new experiment and return the row.

    Re-creating an existing experiment updates it in place but keeps its
    campaign; ``update_experiment_campaign`` is the only way to move it.
    With ``commit=False`` the caller owns the transaction.
    """
    now = _now_epoch()
    singular_dims_json = json.dumps(singular_dims or [])
    row = await _exec_one(
        db,
        """
        INSERT INTO experiments (experiment_id, name, campaign, submit_time, controller_id, note,
                                 status, expected_jobs, singular_dims, max_concurrent,
                                 skip_rules, metric, goal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (experiment_id) DO UPDATE SET
            name = EXCLUDED.name,
            controller_id = EXCLUDED.controller_id,
            note = EXCLUDED.note,
            status = EXCLUDED.status,
            expected_jobs = EXCLUDED.expected_jobs,
            singular_dims = EXCLUDED.singular_dims,
            max_concurrent = EXCLUDED.max_concurrent,
            skip_rules = EXCLUDED.skip_rules,
            metric = COALESCE(EXCLUDED.metric, experiments.metric),
            goal = COALESCE(EXCLUDED.goal, experiments.goal)
        RETURNING *;
        """,
        (experiment_id, name, campaign, now, controller_id, note, status, expected_jobs,
         singular_dims_json, max_concurrent, json.dumps(skip_rules or {}), metric, goal),
    )
    if commit:
        await db.commit()
    assert row is not None
    return _row_to_experiment(row)


async def get_experiment(
    db: aiosqlite.Connection,
    experiment_id: str,
) -> ExperimentRecord | None:
    """Return a single experiment or None."""
    cursor = await db.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_experiment(row)


async def update_experiment_status(
    db: aiosqlite.Connection,
    experiment_id: str,
    status: ExperimentStatus,
) -> ExperimentRecord | None:
    """Update an experiment's status. Returns updated row or None."""
    row = await _exec_one(
        db,
        "UPDATE experiments SET status = ? WHERE experiment_id = ? RETURNING *",
        (status, experiment_id),
    )
    await db.commit()
    return _row_to_experiment(row) if row else None


async def update_experiment_name(
    db: aiosqlite.Connection,
    experiment_id: str,
    name: str,
) -> ExperimentRecord | None:
    """Update an experiment's display name. Returns updated row or None."""
    row = await _exec_one(
        db,
        "UPDATE experiments SET name = ? WHERE experiment_id = ? RETURNING *",
        (name, experiment_id),
    )
    await db.commit()
    return _row_to_experiment(row) if row else None


async def update_experiment_campaign(
    db: aiosqlite.Connection,
    experiment_id: str,
    campaign: str,
) -> ExperimentRecord | None:
    """Move an experiment, with all its runs, to *campaign*. Returns updated row or None."""
    row = await _exec_one(
        db,
        "UPDATE experiments SET campaign = ? WHERE experiment_id = ? RETURNING *",
        (campaign, experiment_id),
    )
    await db.commit()
    return _row_to_experiment(row) if row else None


# Statuses broken out in ``job_counts``, plus "total".
_JOB_COUNT_STATUSES = ("done", "failed", "xfailed", "running", "pending", "dispatched")
_JOB_COUNT_COLUMNS = ",\n               ".join(
    ["COUNT(j.run_id) AS total_jobs"]
    + [f"SUM(CASE WHEN j.status = '{st}' THEN 1 ELSE 0 END) AS {st}_jobs" for st in _JOB_COUNT_STATUSES]
)


def _job_counts(row: Any) -> dict[str, int]:
    """``job_counts`` from a row selected with ``_JOB_COUNT_COLUMNS`` (None = all zero)."""
    keys = ("total", *_JOB_COUNT_STATUSES)
    return {k: (row[f"{k}_jobs"] or 0) if row is not None else 0 for k in keys}


async def list_campaigns(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Every campaign with its experiment and per-status job counts, by name.

    A campaign exists while some experiment is in it.  The default campaign
    is always listed so clients have somewhere to start.
    """
    rows = await _exec_all(
        db,
        f"""
        SELECT e.campaign AS campaign,
               COUNT(DISTINCT e.experiment_id) AS experiments,
               {_JOB_COUNT_COLUMNS},
               MAX(e.submit_time) AS last_submit
        FROM experiments e
        LEFT JOIN jobs j ON j.experiment_id = e.experiment_id
        GROUP BY e.campaign
        ORDER BY e.campaign
        """,
    )
    by_name: dict[str, dict[str, Any]] = {DEFAULT_CAMPAIGN: {
        "campaign": DEFAULT_CAMPAIGN, "experiments": 0, "last_submit": None,
        "job_counts": _job_counts(None),
    }}
    for r in rows:
        by_name[r["campaign"]] = {
            "campaign": r["campaign"],
            "experiments": r["experiments"],
            "job_counts": _job_counts(r),
            "last_submit": _utc(r["last_submit"]).isoformat(),
        }
    return sorted(by_name.values(), key=lambda c: c["campaign"])


# ===============================================================================
# Workers
# ===============================================================================


async def upsert_worker(
    db: aiosqlite.Connection,
    *,
    worker_id: str,
    host: str,
    remote_dir: str,
    scratch_dir: str | None = None,
    port: int = 7890,
    ssh_key: str | None = None,
    venv: str | None = None,
    devices: str | None = None,
    unhealthy_devices: str | None = None,
    status: WorkerStatus = "connected",
    last_error: str | None = None,
) -> WorkerRecord:
    """Insert or update a worker row; set last_seen = now."""
    now = _now_epoch()
    row = await _exec_one(
        db,
        """
        INSERT INTO workers (worker_id, host, remote_dir, status, last_seen,
                             scratch_dir, port, ssh_key, venv, devices,
                             unhealthy_devices, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (worker_id) DO UPDATE SET
            host = EXCLUDED.host,
            remote_dir = EXCLUDED.remote_dir,
            status = EXCLUDED.status,
            last_seen = EXCLUDED.last_seen,
            scratch_dir = EXCLUDED.scratch_dir,
            port = EXCLUDED.port,
            ssh_key = EXCLUDED.ssh_key,
            venv = EXCLUDED.venv,
            devices = EXCLUDED.devices,
            unhealthy_devices = EXCLUDED.unhealthy_devices,
            last_error = EXCLUDED.last_error
        RETURNING *;
        """,
        (worker_id, host, remote_dir, status, now,
         scratch_dir, port, ssh_key, venv, devices, unhealthy_devices, last_error),
    )
    await db.commit()
    assert row is not None
    return _row_to_worker(row)


async def update_worker_devices(
    db: aiosqlite.Connection,
    worker_id: str,
    devices: str,
) -> None:
    """Persist an updated GPU device list for a worker."""
    await db.execute(
        "UPDATE workers SET devices = ? WHERE worker_id = ?",
        (devices, worker_id),
    )
    await db.commit()


async def update_worker_status(
    db: aiosqlite.Connection,
    worker_id: str,
    status: WorkerStatus,
    last_error: str | None = None,
) -> WorkerRecord | None:
    """Update a worker's status and last_seen. Returns updated row or None.

    ``last_error`` records a human-readable failure reason when moving to a
    failed state.  Passing ``None`` (the default) clears any previous error.
    """
    now = _now_epoch()
    row = await _exec_one(
        db,
        "UPDATE workers SET status = ?, last_seen = ?, last_error = ? "
        "WHERE worker_id = ? RETURNING *",
        (status, now, last_error, worker_id),
    )
    await db.commit()
    return _row_to_worker(row) if row else None


async def touch_worker(
    db: aiosqlite.Connection,
    worker_id: str,
) -> None:
    """Update last_seen to now without changing any other fields."""
    await db.execute(
        "UPDATE workers SET last_seen = ? WHERE worker_id = ?",
        (_now_epoch(), worker_id),
    )
    await db.commit()


async def get_worker(
    db: aiosqlite.Connection,
    worker_id: str,
) -> WorkerRecord | None:
    """Return a single worker or None."""
    cursor = await db.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_worker(row)


async def list_workers(
    db: aiosqlite.Connection,
    status: WorkerStatus | None = None,
) -> list[WorkerRecord]:
    """List workers, optionally filtered by status."""
    if status is not None:
        cursor = await db.execute(
            "SELECT * FROM workers WHERE status = ? ORDER BY worker_id", (status,)
        )
    else:
        cursor = await db.execute("SELECT * FROM workers ORDER BY worker_id")
    rows = await cursor.fetchall()
    return [_row_to_worker(r) for r in rows]


# ===============================================================================
# Jobs
# ===============================================================================


def _serialize_job_fields(
    command: "Sequence[str] | str",
    combo: "dict[str, Any] | None",
    env: "dict[str, str] | None",
    return_files: "Sequence[str] | None",
    files: "dict[str, str] | None",
) -> tuple[str, str, str, str, str]:
    command_json = json.dumps(command if isinstance(command, list) else [command])
    combo_json = json.dumps(combo or {})
    env_json = json.dumps(env or {})
    return_files_json = json.dumps(list(return_files or []))
    files_json = json.dumps(files or {})
    return command_json, combo_json, env_json, return_files_json, files_json


async def insert_job(
    db: aiosqlite.Connection,
    *,
    run_id: str,
    experiment_id: str,
    priority: int = 0,
    command: Sequence[str] | str,
    combo: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    status: JobStatus = "pending",
    gpus_per_run: int = 1,
    nodes_per_run: int = 1,
    set_dist_env: bool = False,
    run_from: str | None = None,
    return_files: Sequence[str] | None = None,
    files: dict[str, str] | None = None,
    max_retries: int = 2,
    artifact_id: str | None = None,
    setup_command: str | None = None,
) -> JobRecord:
    """Insert a new job row. Returns the created JobRecord."""
    [job] = await insert_jobs_bulk(db, [dict(
        run_id=run_id, experiment_id=experiment_id, priority=priority,
        command=command, combo=combo, env=env, status=status,
        gpus_per_run=gpus_per_run, nodes_per_run=nodes_per_run,
        set_dist_env=set_dist_env, run_from=run_from, return_files=return_files,
        files=files, max_retries=max_retries, artifact_id=artifact_id,
        setup_command=setup_command,
    )])
    return job


async def _insert_jobs_and_reopen(
    db: aiosqlite.Connection,
    jobs: list[dict[str, Any]],
) -> list[JobRecord]:
    """Insert job rows and reopen experiments that gained pending jobs.

    Does not commit (the caller owns the transaction).
    """
    now = _now_epoch()
    records: list[JobRecord] = []
    for j in jobs:
        command_json, combo_json, env_json, return_files_json, files_json = _serialize_job_fields(
            j["command"], j.get("combo"), j.get("env"), j.get("return_files"), j.get("files")
        )

        row = await _exec_one(
            db,
            """
            INSERT INTO jobs (
                run_id, experiment_id, priority, status, submit_time,
                command, combo, env, gpus_per_run, nodes_per_run,
                set_dist_env, run_from, return_files, files, max_retries,
                artifact_id, setup_command
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?
            )
            RETURNING *;
            """,
            (j["run_id"], j["experiment_id"], j.get("priority", 0),
             j.get("status", "pending"), now,
             command_json, combo_json, env_json,
             j.get("gpus_per_run", 1), j.get("nodes_per_run", 1),
             int(j.get("set_dist_env", False)), j.get("run_from"),
             return_files_json, files_json, j.get("max_retries", 2),
             j.get("artifact_id"), j.get("setup_command")),
        )
        assert row is not None
        records.append(_row_to_job(row))
    await _reopen_experiments(db, {r.experiment_id for r in records if r.status == "pending"})
    return records


async def insert_jobs_bulk(
    db: aiosqlite.Connection,
    jobs: list[dict[str, Any]],
) -> list[JobRecord]:
    """Insert many jobs in a single transaction.  Each dict must contain the
    same keyword arguments as `insert_job` (camelCase keys matching the
    parameter names except run_id / experiment_id / priority / command /
    combo / env etc.).

    Returns the inserted JobRecord list.
    """
    try:
        records = await _insert_jobs_and_reopen(db, jobs)
        await db.commit()
        return records
    except Exception:
        await db.rollback()
        raise


async def create_experiment_with_jobs(
    db: aiosqlite.Connection,
    *,
    jobs: list[dict[str, Any]],
    **fields: Any,
) -> tuple[ExperimentRecord, list[JobRecord]]:
    """Create (or update) an experiment and insert its jobs atomically.

    *fields* are ``create_experiment``'s keyword arguments.  Either both the
    experiment row and every job land, or nothing does, so a job-insert
    failure never leaves a half-created experiment behind.
    """
    try:
        exp = await create_experiment(db, commit=False, **fields)
        records = await _insert_jobs_and_reopen(db, jobs)
        await db.commit()
        return exp, records
    except Exception:
        await db.rollback()
        raise


async def get_job(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> JobRecord | None:
    """Return a single job or None."""
    cursor = await db.execute(
        "SELECT * FROM jobs WHERE run_id = ? AND experiment_id = ?",
        (run_id, experiment_id),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_job(row)


async def list_pending_jobs(
    db: aiosqlite.Connection,
    experiment_id: str | None = None,
    limit: int | None = None,
    campaign: str | None = None,
) -> list[JobRecord]:
    """Return pending jobs ordered by priority DESC, submit_time ASC.

    This matches the composite index ``idx_jobs_dispatch`` for efficient
    index-only scans.  *campaign* keeps only jobs of experiments in it.
    """
    where = ["status = 'pending'"]
    params: list[Any] = []
    if experiment_id is not None:
        where.append("experiment_id = ?")
        params.append(experiment_id)
    if campaign is not None:
        where.append(_IN_CAMPAIGN)
        params.append(campaign)
    sql = f"SELECT * FROM jobs WHERE {' AND '.join(where)} ORDER BY priority DESC, submit_time ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [_row_to_job(r) for r in await _exec_all(db, sql, tuple(params))]


class SchedulableJob(NamedTuple):
    experiment_id: str
    run_id: str
    gpus_per_run: int
    nodes_per_run: int


async def list_schedulable_jobs(
    db: aiosqlite.Connection,
    *,
    cpu_only: bool = False,
) -> list[SchedulableJob]:
    """Return pending jobs eligible for dispatch, in scheduling order.

    A job is schedulable iff its status is ``pending`` and its experiment is
    not paused or aborted (see ``NON_SCHEDULABLE_EXPERIMENT_STATUSES``).  This
    is the scheduler's sole input — the in-memory pending mirror is gone, so a
    paused/aborted experiment simply stops producing schedulable jobs.

    With *cpu_only*, only jobs that need no GPU (the scheduler asks for just
    these when every GPU is full).

    Ordered by priority DESC, submit_time ASC to match ``idx_jobs_dispatch``.
    """
    placeholders = ",".join("?" * len(NON_SCHEDULABLE_EXPERIMENT_STATUSES))
    cursor = await db.execute(
        f"""
        SELECT j.experiment_id, j.run_id, j.gpus_per_run, j.nodes_per_run FROM jobs j
        JOIN experiments e ON j.experiment_id = e.experiment_id
        WHERE j.status = 'pending'
          AND e.status NOT IN ({placeholders})
          {"AND j.gpus_per_run = 0" if cpu_only else ""}
        ORDER BY j.priority DESC, j.submit_time ASC
        """,
        NON_SCHEDULABLE_EXPERIMENT_STATUSES,
    )
    return [SchedulableJob(*r) for r in await cursor.fetchall()]


async def experiment_concurrency_caps(
    db: aiosqlite.Connection,
) -> dict[str, int]:
    """Return ``{experiment_id: max_concurrent}`` for every experiment.

    ``max_concurrent`` of 0 means unlimited.  Used by the scheduler to bound
    how many of an experiment's jobs run at once.
    """
    cursor = await db.execute("SELECT experiment_id, max_concurrent FROM experiments")
    rows = await cursor.fetchall()
    return {r["experiment_id"]: r["max_concurrent"] for r in rows}


async def count_pending_jobs(db: aiosqlite.Connection) -> int:
    """Return the number of jobs in 'pending' status (for health/status)."""
    cursor = await db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'pending'")
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def update_experiment_max_concurrent(
    db: aiosqlite.Connection,
    experiment_id: str,
    max_concurrent: int,
) -> ExperimentRecord | None:
    """Set an experiment's max_concurrent cap. Returns updated row or None."""
    row = await _exec_one(
        db,
        "UPDATE experiments SET max_concurrent = ? WHERE experiment_id = ? RETURNING *",
        (max_concurrent, experiment_id),
    )
    await db.commit()
    return _row_to_experiment(row) if row else None


# ===============================================================================
# Job nodes (multi-node placement + per-node aggregation)
# ===============================================================================


def _row_to_job_node(row: sqlite3.Row) -> JobNodeRecord:
    success = row["success"]
    return JobNodeRecord(
        run_id=row["run_id"],
        experiment_id=row["experiment_id"],
        node_rank=row["node_rank"],
        worker_id=row["worker_id"],
        gpu_ids=row["gpu_ids"],
        status=row["status"],
        success=None if success is None else bool(success),
        elapsed=row["elapsed"],
    )


async def insert_job_nodes(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    placements: list[tuple[int, str, list[int]]],
) -> None:
    """Record the placement of every node of a multi-node job.

    *placements* is a list of ``(node_rank, worker_id, gpu_ids)``.  Replaces any
    existing rows for this run (e.g. on re-dispatch).  Each node starts
    'dispatched'.
    """
    await db.execute(
        "DELETE FROM job_nodes WHERE run_id = ? AND experiment_id = ?",
        (run_id, experiment_id),
    )
    for node_rank, worker_id, gpu_ids in placements:
        await db.execute(
            """
            INSERT INTO job_nodes (run_id, experiment_id, node_rank, worker_id, gpu_ids, status)
            VALUES (?, ?, ?, ?, ?, 'dispatched')
            """,
            (run_id, experiment_id, node_rank, worker_id, json.dumps(gpu_ids)),
        )
    await db.commit()


async def mark_job_node_result(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    worker_id: str,
    success: bool,
    elapsed: float,
) -> None:
    """Record one node's terminal result, keyed by the worker that ran it.

    A multi-node run places at most one node per worker, so (run_id, worker_id)
    identifies the node without needing a node_rank on the wire.
    """
    await db.execute(
        """
        UPDATE job_nodes
        SET status = ?, success = ?, elapsed = ?
        WHERE run_id = ? AND experiment_id = ? AND worker_id = ?
        """,
        ("done" if success else "failed", int(success), elapsed,
         run_id, experiment_id, worker_id),
    )
    await db.commit()


async def list_job_nodes(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> list[JobNodeRecord]:
    """Return all node rows for a run, ordered by node_rank."""
    cursor = await db.execute(
        "SELECT * FROM job_nodes WHERE run_id = ? AND experiment_id = ? ORDER BY node_rank",
        (run_id, experiment_id),
    )
    return [_row_to_job_node(r) for r in await cursor.fetchall()]


async def multinode_progress(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> tuple[int, bool, float]:
    """Aggregate a multi-node run's node results from the DB.

    Returns ``(remaining, all_success, max_elapsed)`` where *remaining* is the
    number of nodes not yet in a terminal state.  When *remaining* is 0 the run
    is finished: *all_success* is the AND of every node's success and
    *max_elapsed* is the slowest node's wall time.  Because this is derived from
    durable rows, it is correct even after a manager restart.
    """
    cursor = await db.execute(
        "SELECT status, success, elapsed FROM job_nodes "
        "WHERE run_id = ? AND experiment_id = ?",
        (run_id, experiment_id),
    )
    rows = await cursor.fetchall()
    remaining = 0
    all_success = True
    max_elapsed = 0.0
    for r in rows:
        if r["status"] not in ("done", "failed"):
            remaining += 1
            continue
        if not r["success"]:
            all_success = False
        if r["elapsed"] is not None and r["elapsed"] > max_elapsed:
            max_elapsed = r["elapsed"]
    return remaining, all_success, max_elapsed


async def delete_job_nodes(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> None:
    """Remove a run's node rows (on finalization or cancellation)."""
    await db.execute(
        "DELETE FROM job_nodes WHERE run_id = ? AND experiment_id = ?",
        (run_id, experiment_id),
    )
    await db.commit()


async def list_jobs_by_experiment(
    db: aiosqlite.Connection,
    experiment_id: str,
    status: JobStatus | None = None,
) -> list[JobRecord]:
    """Return all jobs for an experiment, optionally filtered by status."""
    if status is not None:
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE experiment_id = ? AND status = ? ORDER BY submit_time ASC",
            (experiment_id, status),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE experiment_id = ? ORDER BY submit_time ASC",
            (experiment_id,),
        )
    rows = await cursor.fetchall()
    return [_row_to_job(r) for r in rows]


async def update_job_status(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    status: JobStatus,
    *,
    only_from: Sequence[str] | None = None,
    **kwargs: Any,
) -> JobRecord | None:
    """Generic job status update.  Extra keyword arguments are set as columns
    (e.g. ``exit_code=0``, ``elapsed=12.3``).  With *only_from*, the row is
    updated only if its current status is one of those.

    Returns the updated row or None.
    """
    set_clauses = ["status = ?"]
    values: list[Any] = [status]
    for col, val in kwargs.items():
        set_clauses.append(f"{col} = ?")
        values.append(val)
    values.append(run_id)
    values.append(experiment_id)
    guard = ""
    if only_from is not None:
        guard = f" AND status IN ({','.join('?' * len(only_from))})"
        values.extend(only_from)
    row = await _exec_one(
        db,
        f"UPDATE jobs SET {', '.join(set_clauses)} WHERE run_id = ? AND experiment_id = ?{guard} RETURNING *",
        tuple(values),
    )
    if row is not None and status == "pending":
        await _reopen_experiments(db, {experiment_id})
    await db.commit()
    return _row_to_job(row) if row else None


async def dispatch_job(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    worker_id: str,
    dispatched_gpu_ids: list[int] | None = None,
) -> JobRecord | None:
    """Atomically move a job from pending → dispatched.

    Uses ``UPDATE ... WHERE status = 'pending'`` as a lightweight lock so
    two schedulers cannot grab the same job.
    """
    now = _now_epoch()
    gpu_json = json.dumps(dispatched_gpu_ids) if dispatched_gpu_ids is not None else None
    row = await _exec_one(
        db,
        """
        UPDATE jobs
        SET status = 'dispatched',
            dispatch_time = ?,
            worker_id = ?,
            dispatched_gpu_ids = ?,
            attempt = attempt + 1
        WHERE run_id = ? AND experiment_id = ? AND status = 'pending'
        RETURNING *;
        """,
        (now, worker_id, gpu_json, run_id, experiment_id),
    )
    await db.commit()
    return _row_to_job(row) if row else None


async def mark_job_running(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> JobRecord | None:
    """Atomically move a job from dispatched → running. Sets start_time."""
    now = _now_epoch()
    row = await _exec_one(
        db,
        """
        UPDATE jobs
        SET status = 'running', start_time = ?
        WHERE run_id = ? AND experiment_id = ? AND status = 'dispatched'
        RETURNING *;
        """,
        (now, run_id, experiment_id),
    )
    await db.commit()
    return _row_to_job(row) if row else None


async def finish_job(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    *,
    success: bool,
    exit_code: int,
    elapsed: float,
) -> JobRecord | None:
    """Mark a job as done or failed."""
    status = "done" if success else "failed"
    now = _now_epoch()
    row = await _exec_one(
        db,
        f"""
        UPDATE jobs
        SET status = ?, finish_time = ?, exit_code = ?, elapsed = ?
        WHERE run_id = ? AND experiment_id = ? AND {_ACTIVE_IN}
        RETURNING *;
        """,
        (status, now, exit_code, elapsed, run_id, experiment_id),
    )
    await db.commit()
    return _row_to_job(row) if row else None


# Columns cleared whenever a job goes back to 'pending'.
_CLEAR_DISPATCH = """
    dispatch_time = NULL,
    start_time = NULL,
    finish_time = NULL,
    elapsed = NULL,
    exit_code = NULL,
    worker_id = NULL,
    dispatched_gpu_ids = NULL
"""


def _keys_clause(keys: Sequence[tuple[str, str]]) -> tuple[str, tuple[str, ...]]:
    """WHERE clause matching (experiment_id, run_id) *keys*."""
    where = " OR ".join("(experiment_id = ? AND run_id = ?)" for _ in keys)
    return f"({where})", tuple(x for key in keys for x in key)


async def _reopen_experiments(db: aiosqlite.Connection, experiment_ids: set[str]) -> None:
    """A completed experiment that gets pending work again is running again."""
    for eid in experiment_ids:
        await db.execute(
            "UPDATE experiments SET status = 'running' WHERE experiment_id = ? AND status = 'completed'",
            (eid,),
        )


async def cancel_jobs(
    db: aiosqlite.Connection,
    keys: Sequence[tuple[str, str]],
) -> list[JobRecord]:
    """Cancel jobs that have not finished yet; finished jobs are left alone.

    *keys* are (experiment_id, run_id).  Returns the rows that were cancelled.
    Also drops their multi-node rows.
    """
    if not keys:
        return []
    where, params = _keys_clause(keys)
    rows = await _exec_all(
        db,
        f"""
        UPDATE jobs SET status = 'cancelled'
        WHERE {where} AND {_UNFINISHED_IN}
        RETURNING *
        """,
        params,
    )
    await db.execute(f"DELETE FROM job_nodes WHERE {where}", params)
    await db.commit()
    return [_row_to_job(r) for r in rows]


async def requeue_jobs(
    db: aiosqlite.Connection,
    keys: Sequence[tuple[str, str]],
    *,
    spend_retry: bool,
) -> tuple[list[JobRecord], list[JobRecord]]:
    """Move dispatched/running jobs back to 'pending'; other rows are left alone.

    With *spend_retry* (the run was lost), each requeue uses one retry and a
    job with none left is marked failed instead.  Without it (the manager took
    the run away, e.g. an eviction), no retry is spent.

    *keys* are (experiment_id, run_id).  Returns ``(requeued, failed)``.
    Also drops their multi-node rows.
    """
    if not keys:
        return [], []
    where, params = _keys_clause(keys)
    spend = ", retry_count = retry_count + 1" if spend_retry else ""
    has_retry = " AND retry_count < max_retries" if spend_retry else ""
    requeued = await _exec_all(
        db,
        f"""
        UPDATE jobs SET status = 'pending'{spend}, {_CLEAR_DISPATCH}
        WHERE {where} AND {_ACTIVE_IN}{has_retry}
        RETURNING *
        """,
        params,
    )
    failed = []
    if spend_retry:
        failed = await _exec_all(
            db,
            f"""
            UPDATE jobs SET status = 'failed', exit_code = -1, elapsed = 0.0, finish_time = ?
            WHERE {where} AND {_ACTIVE_IN}
            RETURNING *
            """,
            (_now_epoch(), *params),
        )
    await db.execute(f"DELETE FROM job_nodes WHERE {where}", params)
    await db.commit()
    return [_row_to_job(r) for r in requeued], [_row_to_job(r) for r in failed]


async def retry_job(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> JobRecord | None:
    """Re-queue a finished job, using one retry.

    Returns None if the job is not finished or has no retries left.
    """
    row = await _exec_one(
        db,
        f"""
        UPDATE jobs SET status = 'pending', retry_count = retry_count + 1, {_CLEAR_DISPATCH}
        WHERE run_id = ? AND experiment_id = ?
          AND {_FINISHED_IN}
          AND retry_count < max_retries
        RETURNING *
        """,
        (run_id, experiment_id),
    )
    if row is not None:
        await _reopen_experiments(db, {experiment_id})
    await db.commit()
    return _row_to_job(row) if row else None


async def list_active_jobs(
    db: aiosqlite.Connection,
    worker_id: str | None = None,
) -> list[JobRecord]:
    """Return dispatched/running jobs, optionally only those with a node on *worker_id*."""
    rows = await _exec_all(
        db,
        f"""
        SELECT * FROM jobs j
        WHERE j.{_ACTIVE_IN}
          AND (? IS NULL OR j.worker_id = ? OR EXISTS (
                SELECT 1 FROM job_nodes n
                WHERE n.run_id = j.run_id AND n.experiment_id = j.experiment_id
                  AND n.worker_id = ?))
        """,
        (worker_id, worker_id, worker_id),
    )
    return [_row_to_job(r) for r in rows]


async def apply_result_rules(
    db: aiosqlite.Connection,
    experiment_id: str,
    run_id: str,
    success: bool,
) -> list[str]:
    """After *run_id* finishes, mark the jobs its result makes moot as xfailed.

    * A success settles its singular dims, so failed probes of the same
      non-singular combo at other singular values become xfailed.
    * The experiment's skip rules (``monotonic`` / ``singular``) then mark
      pending jobs that no longer need to run as xfailed.

    Returns the run_ids changed.
    """
    exp = await get_experiment(db, experiment_id)
    if exp is None:
        return []
    singular_dims: list[str] = json.loads(exp.singular_dims)
    rules: dict[str, Any] = json.loads(exp.skip_rules)
    # A failure can only trigger monotonic skips; a success only singular ones.
    relevant = any(r["singular"] if success else r["monotonic"] for r in rules.values())
    if not relevant and not (success and singular_dims):
        return []

    rows = [(r["run_id"], r["status"], r["exit_code"], json.loads(r["combo"]))
            for r in await _exec_all(
                db,
                "SELECT run_id, status, exit_code, combo FROM jobs WHERE experiment_id = ? "
                "AND status IN ('pending', 'done', 'failed', 'xfailed')",
                (experiment_id,),
            )]

    xfail: list[str] = []
    if success and singular_dims:
        this = next((c for rid, _, _, c in rows if rid == run_id), None)
        if this is not None:
            singular = set(singular_dims)
            lex = {k: v for k, v in this.items() if k not in singular}
            xfail += [
                rid for rid, status, _, combo in rows
                if status == "failed"
                and {k: v for k, v in combo.items() if k not in singular} == lex
                and any(combo.get(d) != this.get(d) for d in singular_dims)
            ]
    if relevant:
        # A job skipped earlier (xfailed with no exit code) never ran, so it is not a failure.
        failed = [c for _, status, code, c in rows
                  if status == "failed" or (status == "xfailed" and code is not None)]
        succeeded = [c for _, status, _, c in rows if status == "done"]
        index = SkipIndex(failed, succeeded, rules)
        xfail += [rid for rid, status, _, combo in rows if status == "pending" and index.skips(combo)]
    if not xfail:
        return []
    await db.execute(
        f"UPDATE jobs SET status = 'xfailed' WHERE experiment_id = ? "
        f"AND status IN ('pending', 'failed') AND run_id IN ({','.join('?' * len(xfail))})",
        (experiment_id, *xfail),
    )
    await db.commit()
    return xfail


async def update_job_priority(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    priority: int,
) -> JobRecord | None:
    """Update a job's priority."""
    # Try pending first (lightweight lock)
    row = await _exec_one(
        db,
        "UPDATE jobs SET priority = ? WHERE run_id = ? AND experiment_id = ? AND status = 'pending' RETURNING *",
        (priority, run_id, experiment_id),
    )
    if row is None:
        # Fallback: update regardless of status
        row = await _exec_one(
            db,
            "UPDATE jobs SET priority = ? WHERE run_id = ? AND experiment_id = ? RETURNING *",
            (priority, run_id, experiment_id),
        )
    await db.commit()
    return _row_to_job(row) if row else None


async def update_job_label(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    label: str | None,
) -> JobRecord | None:
    """Set or clear a job's human-readable label."""
    row = await _exec_one(
        db,
        "UPDATE jobs SET label = ? WHERE run_id = ? AND experiment_id = ? RETURNING *",
        (label, run_id, experiment_id),
    )
    await db.commit()
    return _row_to_job(row) if row else None


async def list_jobs_by_status(
    db: aiosqlite.Connection,
    status: JobStatus,
    limit: int | None = None,
    campaign: str | None = None,
) -> list[JobRecord]:
    """Return jobs with the given *status* across all experiments (or only
    those in *campaign*), newest first.  If *limit* is provided, only the
    first N rows are returned.
    """
    sql = "SELECT * FROM jobs WHERE status = ?"
    params: list[Any] = [status]
    if campaign is not None:
        sql += f" AND {_IN_CAMPAIGN}"
        params.append(campaign)
    sql += " ORDER BY submit_time DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [_row_to_job(r) for r in await _exec_all(db, sql, tuple(params))]


# Whitelist of column names that can be used with list_jobs_since.
_ALLOWED_SINCE_COLS = frozenset({"start_time", "finish_time"})


async def list_jobs_since(
    db: aiosqlite.Connection,
    experiment_id: str,
    statuses: list[JobStatus],
    since_col: str,
    since_ts: float | datetime,
) -> list[JobRecord]:
    """Return jobs for *experiment_id* with a status in *statuses* whose
    *since_col* is >= *since_ts*, ordered by that column ascending.

    *since_col* must be one of ``'start_time'`` or ``'finish_time'``
    (validated server-side to prevent SQL injection).
    """
    if since_col not in _ALLOWED_SINCE_COLS:
        raise ValueError(
            f"since_col must be one of {sorted(_ALLOWED_SINCE_COLS)}, got {since_col!r}"
        )

    placeholders = ", ".join("?" for _ in statuses)
    cursor = await db.execute(
        f"SELECT * FROM jobs WHERE experiment_id = ? AND status IN ({placeholders}) "
        f"AND {since_col} >= ? ORDER BY {since_col} ASC",
        (experiment_id, *statuses, since_ts),
    )
    rows = await cursor.fetchall()
    return [_row_to_job(r) for r in rows]


# ===============================================================================
# Artifacts
# ===============================================================================


async def register_artifact(
    db: aiosqlite.Connection,
    *,
    artifact_id: str,
    size_bytes: int | None = None,
    setup_command: str | None = None,
) -> ArtifactRecord:
    """Insert or update an artifact row; bump ref_count by 1."""
    now = _now_epoch()
    row = await _exec_one(
        db,
        """
        INSERT INTO artifacts (artifact_id, size_bytes, stored_at, ref_count, setup_command)
        VALUES (?, ?, ?, 1, ?)
        ON CONFLICT (artifact_id) DO UPDATE SET
            size_bytes = EXCLUDED.size_bytes,
            ref_count = artifacts.ref_count + 1,
            setup_command = EXCLUDED.setup_command
        RETURNING *;
        """,
        (artifact_id, size_bytes, now, setup_command),
    )
    await db.commit()
    assert row is not None
    return _row_to_artifact(row)


async def get_artifact(
    db: aiosqlite.Connection,
    artifact_id: str,
) -> ArtifactRecord | None:
    """Return a single artifact or None."""
    cursor = await db.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_artifact(row)


async def increment_artifact_ref(
    db: aiosqlite.Connection,
    artifact_id: str,
    delta: int = 1,
) -> ArtifactRecord | None:
    """Increment (or decrement) an artifact's ref_count."""
    row = await _exec_one(
        db,
        "UPDATE artifacts SET ref_count = ref_count + ? WHERE artifact_id = ? RETURNING *",
        (delta, artifact_id),
    )
    await db.commit()
    return _row_to_artifact(row) if row else None


# ===============================================================================
# Bulk / aggregation helpers
# ===============================================================================


async def count_jobs_by_status(
    db: aiosqlite.Connection,
    experiment_id: str,
) -> dict[JobStatus, int]:
    """Return ``{status: count}`` for a given experiment."""
    cursor = await db.execute(
        """
        SELECT status, COUNT(*) AS cnt
        FROM jobs
        WHERE experiment_id = ?
        GROUP BY status
        """,
        (experiment_id,),
    )
    rows = await cursor.fetchall()
    return {r["status"]: r["cnt"] for r in rows}


async def count_active_jobs(db: aiosqlite.Connection, experiment_id: str) -> int:
    """Return the number of jobs in active (non-terminal) states for an experiment."""
    row = await _exec_one(
        db, f"SELECT COUNT(*) FROM jobs WHERE experiment_id = ? AND {_UNFINISHED_IN}", (experiment_id,),
    )
    return row[0] if row else 0


async def experiment_summary(
    db: aiosqlite.Connection,
    experiment_id: str,
) -> dict[str, Any]:
    """Return a summary dict of an experiment: metadata + job counts."""
    exp = await get_experiment(db, experiment_id)
    counts = await count_jobs_by_status(db, experiment_id)
    return {
        "experiment_id": experiment_id,
        "name": exp.name if exp else None,
        "campaign": exp.campaign if exp else None,
        "status": exp.status if exp else None,
        "note": exp.note if exp else None,
        "submit_time": exp.submit_time.isoformat() if exp else None,
        "metric": exp.metric if exp else None,
        "goal": exp.goal if exp else None,
        "job_counts": counts,
    }


async def list_experiments_with_counts(
    db: aiosqlite.Connection,
    status: "ExperimentStatus | None" = None,
    campaign: str | None = None,
) -> list[dict[str, Any]]:
    """List experiments with per-status job counts in a single query.

    *status* and *campaign* each narrow the list when given.
    """
    conds: list[str] = []
    params: tuple[str, ...] = ()
    if status is not None:
        conds.append("e.status = ?")
        params += (status,)
    if campaign is not None:
        conds.append("e.campaign = ?")
        params += (campaign,)
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    cursor = await db.execute(
        f"""
        SELECT e.*,
               {_JOB_COUNT_COLUMNS}
        FROM experiments e
        LEFT JOIN jobs j ON j.experiment_id = e.experiment_id
        {where}
        GROUP BY e.experiment_id
        ORDER BY e.submit_time DESC
        """,
        params,
    )
    rows = await cursor.fetchall()
    result = []
    for r in rows:
        exp = _row_to_experiment(r)
        d = dataclasses.asdict(exp)
        d["submit_time"] = exp.submit_time.isoformat()
        d["job_counts"] = _job_counts(r)
        result.append(d)
    return result


async def delete_experiment(
    db: aiosqlite.Connection,
    experiment_id: str,
) -> bool:
    """Delete an experiment and all its jobs. Returns True if it existed."""
    keys = "SELECT job_key FROM jobs WHERE experiment_id = ?"
    await db.execute(f"DELETE FROM metrics WHERE job_key IN ({keys})", (experiment_id,))
    await db.execute(f"DELETE FROM logs WHERE job_key IN ({keys})", (experiment_id,))
    await db.execute("DELETE FROM jobs WHERE experiment_id = ?", (experiment_id,))
    await db.execute("DELETE FROM job_nodes WHERE experiment_id = ?", (experiment_id,))
    row = await _exec_one(
        db,
        "DELETE FROM experiments WHERE experiment_id = ? RETURNING experiment_id",
        (experiment_id,),
    )
    await db.commit()
    return row is not None


# ===============================================================================
# Metrics
# ===============================================================================


async def insert_metric(
    db: aiosqlite.Connection,
    job_key: int,
    attempt: int,
    step: int,
    data: dict[str, Any],
) -> None:
    """Persist one metric row, merging into any row already at this step.

    A second ``log()`` at the same step is merged (not dropped): keys already
    stored are preserved, and keys in *data* are added or overwrite them.
    """
    await db.execute(
        """
        INSERT INTO metrics (job_key, attempt, step, data)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (job_key, attempt, step) DO UPDATE SET
            data = CASE WHEN typeof(data) = 'text'
                        THEN json_patch(data, excluded.data)
                        ELSE excluded.data
                   END
        """,
        (job_key, attempt, step, json.dumps(data, separators=(",", ":"))),
    )
    await db.commit()


async def get_metrics_for_run(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Return the latest attempt's metric rows for a run as dicts, ordered by step."""
    rows = [(r[0], r[1]) for r in await _exec_all(
        db,
        """
        SELECT m.step, m.data FROM metrics m JOIN jobs j ON m.job_key = j.job_key
        WHERE j.experiment_id = ? AND j.run_id = ?
          AND m.attempt = (SELECT MAX(attempt) FROM metrics WHERE job_key = j.job_key)
        """,
        (experiment_id, run_id),
    )]
    by_step = _merge_metric_rows(rows)
    return [{**json.loads(by_step[step]), "step": step} for step in sorted(by_step)]


# ===============================================================================
# Logs
# ===============================================================================


async def insert_log(
    db: aiosqlite.Connection,
    job_key: int,
    attempt: int,
    seq: int,
    data: str,
) -> None:
    """Persist one log chunk.  A duplicate (job_key, attempt, seq) is ignored."""
    await db.execute(
        "INSERT OR IGNORE INTO logs (job_key, attempt, seq, data) VALUES (?, ?, ?, ?)",
        (job_key, attempt, seq, _pack_log(data)),
    )
    await db.commit()


async def last_log_seq(db: aiosqlite.Connection, job_key: int, attempt: int) -> int:
    """Byte offset in the run's training.log up to which the log is stored."""
    row = await _exec_one(
        db, "SELECT MAX(seq) FROM logs WHERE job_key = ? AND attempt = ?", (job_key, attempt),
    )
    return int(row[0]) if row and row[0] is not None else 0


async def get_logs_for_run(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> str:
    """Return a run's log text, every attempt in order, each after a header line
    when there was more than one."""
    rows = await _exec_all(
        db,
        """
        SELECT l.attempt, l.data FROM logs l JOIN jobs j ON l.job_key = j.job_key
        WHERE j.experiment_id = ? AND j.run_id = ?
        ORDER BY l.attempt, l.seq
        """,
        (experiment_id, run_id),
    )
    multiple = len({row["attempt"] for row in rows}) > 1
    parts: list[str] = []
    attempt = None
    for row in rows:
        if multiple and row["attempt"] != attempt:
            attempt = row["attempt"]
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")
            parts.append(f"[mlsweep] ── attempt {attempt} ──\n")
        parts.append(_unpack_log(row["data"]))
    return "".join(parts)


# ===============================================================================
# DB Writer Actor
# ===============================================================================


class DbWriter:
    """Serial write actor for the SQLite database.

    Owns an exclusive write connection.  All mutations are submitted through
    an asyncio.Queue and processed one at a time, eliminating the
    "cannot commit transaction - SQL statements in progress" race that occurs
    when multiple coroutines share a single aiosqlite connection.

    Usage::

        writer = DbWriter(write_db)
        asyncio.create_task(writer.run())   # start the actor loop
        await writer.insert_metric(...)     # submit a write from any coroutine
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._q: asyncio.Queue[tuple[Callable[[], Coroutine[Any, Any, Any]], asyncio.Future[Any]]] = asyncio.Queue()

    async def run(self) -> None:
        """Actor loop — run as a long-lived asyncio task."""
        while True:
            fn, fut = await self._q.get()
            try:
                fut.set_result(await fn())
            except Exception as exc:
                try:
                    await self._db.rollback()
                except Exception:
                    pass
                fut.set_exception(exc)

    async def _enqueue(self, fn: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[_T] = loop.create_future()
        await self._q.put((fn, fut))
        return await fut

    # ── Experiments ───────────────────────────────────────────────────────────

    async def create_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        campaign: str = DEFAULT_CAMPAIGN,
        controller_id: str | None = None,
        note: str | None = None,
        status: ExperimentStatus = "running",
        expected_jobs: int = 0,
        singular_dims: list[str] | None = None,
        max_concurrent: int = 0,
        skip_rules: dict[str, Any] | None = None,
        metric: str | None = None,
        goal: str | None = None,
    ) -> ExperimentRecord:
        db = self._db
        return await self._enqueue(lambda: create_experiment(
            db, experiment_id=experiment_id, name=name, campaign=campaign,
            controller_id=controller_id, note=note,
            status=status, expected_jobs=expected_jobs,
            singular_dims=singular_dims, max_concurrent=max_concurrent,
            skip_rules=skip_rules, metric=metric, goal=goal,
        ))

    async def create_experiment_with_jobs(
        self, *, jobs: list[dict[str, Any]], **fields: Any,
    ) -> tuple[ExperimentRecord, list[JobRecord]]:
        db = self._db
        return await self._enqueue(lambda: create_experiment_with_jobs(db, jobs=jobs, **fields))

    async def update_experiment_status(
        self, experiment_id: str, status: ExperimentStatus
    ) -> ExperimentRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_experiment_status(db, experiment_id, status))

    async def update_experiment_max_concurrent(
        self, experiment_id: str, max_concurrent: int
    ) -> ExperimentRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_experiment_max_concurrent(db, experiment_id, max_concurrent))

    async def update_experiment_name(
        self, experiment_id: str, name: str
    ) -> ExperimentRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_experiment_name(db, experiment_id, name))

    async def update_experiment_campaign(
        self, experiment_id: str, campaign: str
    ) -> ExperimentRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_experiment_campaign(db, experiment_id, campaign))

    async def delete_experiment(self, experiment_id: str) -> bool:
        db = self._db
        return await self._enqueue(lambda: delete_experiment(db, experiment_id))

    # ── Workers ───────────────────────────────────────────────────────────────

    async def upsert_worker(
        self,
        *,
        worker_id: str,
        host: str,
        remote_dir: str,
        scratch_dir: str | None = None,
        port: int = 7890,
        ssh_key: str | None = None,
        venv: str | None = None,
        devices: str | None = None,
        unhealthy_devices: str | None = None,
        status: WorkerStatus = "connected",
        last_error: str | None = None,
    ) -> WorkerRecord:
        db = self._db
        return await self._enqueue(lambda: upsert_worker(
            db, worker_id=worker_id, host=host, remote_dir=remote_dir,
            scratch_dir=scratch_dir, port=port, ssh_key=ssh_key,
            venv=venv, devices=devices, unhealthy_devices=unhealthy_devices,
            status=status, last_error=last_error,
        ))

    async def update_worker_status(
        self, worker_id: str, status: WorkerStatus, last_error: str | None = None
    ) -> WorkerRecord | None:
        db = self._db
        return await self._enqueue(
            lambda: update_worker_status(db, worker_id, status, last_error)
        )

    async def touch_worker(self, worker_id: str) -> None:
        db = self._db
        await self._enqueue(lambda: touch_worker(db, worker_id))

    async def update_worker_devices(self, worker_id: str, devices: str) -> None:
        db = self._db
        await self._enqueue(lambda: update_worker_devices(db, worker_id, devices))

    # ── Jobs ──────────────────────────────────────────────────────────────────

    async def insert_job(
        self,
        *,
        run_id: str,
        experiment_id: str,
        priority: int = 0,
        command: Sequence[str] | str,
        combo: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
        status: JobStatus = "pending",
        gpus_per_run: int = 1,
        nodes_per_run: int = 1,
        set_dist_env: bool = False,
        run_from: str | None = None,
        return_files: Sequence[str] | None = None,
        files: dict[str, str] | None = None,
        max_retries: int = 2,
        artifact_id: str | None = None,
        setup_command: str | None = None,
    ) -> JobRecord:
        db = self._db
        return await self._enqueue(lambda: insert_job(
            db, run_id=run_id, experiment_id=experiment_id, priority=priority,
            command=command, combo=combo, env=env, status=status,
            gpus_per_run=gpus_per_run, nodes_per_run=nodes_per_run,
            set_dist_env=set_dist_env, run_from=run_from, return_files=return_files,
            files=files, max_retries=max_retries, artifact_id=artifact_id,
            setup_command=setup_command,
        ))

    async def insert_jobs_bulk(self, jobs: list[dict[str, Any]]) -> list[JobRecord]:
        db = self._db
        return await self._enqueue(lambda: insert_jobs_bulk(db, jobs))

    async def update_job_status(
        self,
        run_id: str,
        experiment_id: str,
        status: JobStatus,
        *,
        only_from: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_job_status(
            db, run_id, experiment_id, status, only_from=only_from, **kwargs))

    async def dispatch_job(
        self,
        run_id: str,
        experiment_id: str,
        worker_id: str,
        dispatched_gpu_ids: list[int] | None = None,
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: dispatch_job(db, run_id, experiment_id, worker_id, dispatched_gpu_ids))

    async def mark_job_running(
        self, run_id: str, experiment_id: str
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: mark_job_running(db, run_id, experiment_id))

    async def finish_job(
        self,
        run_id: str,
        experiment_id: str,
        *,
        success: bool,
        exit_code: int,
        elapsed: float,
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: finish_job(
            db, run_id, experiment_id, success=success, exit_code=exit_code, elapsed=elapsed
        ))

    async def apply_result_rules(
        self, experiment_id: str, run_id: str, success: bool,
    ) -> list[str]:
        db = self._db
        return await self._enqueue(lambda: apply_result_rules(db, experiment_id, run_id, success))

    async def update_job_priority(
        self, run_id: str, experiment_id: str, priority: int
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_job_priority(db, run_id, experiment_id, priority))

    async def update_job_label(
        self, run_id: str, experiment_id: str, label: str | None
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_job_label(db, run_id, experiment_id, label))

    async def cancel_jobs(self, keys: Sequence[tuple[str, str]]) -> list[JobRecord]:
        db = self._db
        return await self._enqueue(lambda: cancel_jobs(db, keys))

    async def requeue_jobs(
        self, keys: Sequence[tuple[str, str]], *, spend_retry: bool
    ) -> tuple[list[JobRecord], list[JobRecord]]:
        db = self._db
        return await self._enqueue(lambda: requeue_jobs(db, keys, spend_retry=spend_retry))

    async def retry_job(self, run_id: str, experiment_id: str) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: retry_job(db, run_id, experiment_id))

    async def insert_job_nodes(
        self, run_id: str, experiment_id: str,
        placements: list[tuple[int, str, list[int]]],
    ) -> None:
        db = self._db
        await self._enqueue(lambda: insert_job_nodes(db, run_id, experiment_id, placements))

    async def mark_job_node_result(
        self, run_id: str, experiment_id: str, worker_id: str,
        success: bool, elapsed: float,
    ) -> None:
        db = self._db
        await self._enqueue(lambda: mark_job_node_result(
            db, run_id, experiment_id, worker_id, success, elapsed))

    async def delete_job_nodes(
        self, run_id: str, experiment_id: str
    ) -> None:
        db = self._db
        await self._enqueue(lambda: delete_job_nodes(db, run_id, experiment_id))

    # ── Artifacts ─────────────────────────────────────────────────────────────

    async def register_artifact(
        self,
        *,
        artifact_id: str,
        size_bytes: int | None = None,
        setup_command: str | None = None,
    ) -> ArtifactRecord:
        db = self._db
        return await self._enqueue(lambda: register_artifact(
            db, artifact_id=artifact_id, size_bytes=size_bytes, setup_command=setup_command
        ))

    async def increment_artifact_ref(
        self, artifact_id: str, delta: int = 1
    ) -> ArtifactRecord | None:
        db = self._db
        return await self._enqueue(lambda: increment_artifact_ref(db, artifact_id, delta))

    # ── Metrics and logs ──────────────────────────────────────────────────────

    async def insert_metric(
        self, job_key: int, attempt: int, step: int, data: dict[str, Any],
    ) -> None:
        db = self._db
        await self._enqueue(lambda: insert_metric(db, job_key, attempt, step, data))

    async def pack_metrics(
        self, job_key: int, attempt: int, extra: Sequence[tuple[int, str]] = (),
    ) -> None:
        db = self._db
        await self._enqueue(lambda: pack_metrics(db, job_key, attempt, extra))

    async def insert_log(self, job_key: int, attempt: int, seq: int, data: str) -> None:
        db = self._db
        await self._enqueue(lambda: insert_log(db, job_key, attempt, seq, data))
