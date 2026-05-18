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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Literal, Sequence, TypeVar

import aiosqlite

_T = TypeVar("_T")

JobStatus = Literal["pending", "dispatched", "running", "done", "failed", "xfailed", "cancelled"]
ExperimentStatus = Literal["running", "completed", "aborted"]
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
    controller_id: str | None = None
    note: str | None = None
    status: ExperimentStatus = "running"
    expected_jobs: int = 0
    singular_dims: str = "[]"  # JSON list of dim names that are singular probes


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
    jobs_per_gpu: int = 1


@dataclass(order=False)
class ArtifactRecord:
    """A content-addressed artifact stored on the manager node."""

    artifact_id: str
    size_bytes: int | None = None
    stored_at: datetime | None = None
    ref_count: int = 0
    setup_command: str | None = None


# ===============================================================================
# Helpers – row → dict / dict → row
# ===============================================================================


def _col(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Get a column value from a sqlite3.Row, returning *default* if NULL or missing.

    Only use for columns that have a NOT NULL DEFAULT constraint in the schema.
    For columns that legitimately hold NULL, access them directly — this helper
    would mask intentional NULLs by substituting *default*.
    """
    try:
        v = row[key]
    except (IndexError, KeyError):
        return default
    return v if v is not None else default


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    """Map a database row to a JobRecord."""
    return JobRecord(
        run_id=row["run_id"],
        experiment_id=row["experiment_id"],
        priority=row["priority"],
        submit_time=_ensure_utc(row["submit_time"]),
        command=row["command"],
        status=row["status"],
        dispatch_time=_maybe_utc(row["dispatch_time"]),
        start_time=_maybe_utc(row["start_time"]),
        finish_time=_maybe_utc(row["finish_time"]),
        elapsed=row["elapsed"],
        exit_code=row["exit_code"],
        worker_id=row["worker_id"],
        env=_col(row, "env", "{}"),
        artifact_id=row["artifact_id"],
        setup_command=row["setup_command"],
        gpus_per_run=_col(row, "gpus_per_run", 1),
        nodes_per_run=_col(row, "nodes_per_run", 1),
        set_dist_env=bool(row["set_dist_env"]),
        run_from=row["run_from"],
        return_files=_col(row, "return_files", "[]"),
        files=_col(row, "files", "{}"),
        retry_count=_col(row, "retry_count", 0),
        max_retries=_col(row, "max_retries", 2),
        combo=_col(row, "combo", "{}"),
        dispatched_gpu_ids=row["dispatched_gpu_ids"],
        jobs_per_gpu=_col(row, "jobs_per_gpu", 1),
    )


def _row_to_experiment(row: sqlite3.Row) -> ExperimentRecord:
    """Map a database row to an ExperimentRecord."""
    return ExperimentRecord(
        experiment_id=row["experiment_id"],
        name=row["name"],
        submit_time=_ensure_utc(row["submit_time"]),
        controller_id=row["controller_id"],
        note=row["note"],
        status=_col(row, "status", "running"),
        expected_jobs=_col(row, "expected_jobs", 0),
        singular_dims=_col(row, "singular_dims", "[]"),
    )


def _row_to_worker(row: sqlite3.Row) -> WorkerRecord:
    """Map a database row to a WorkerRecord."""
    return WorkerRecord(
        worker_id=row["worker_id"],
        host=row["host"],
        remote_dir=row["remote_dir"],
        status=_col(row, "status", "offline"),
        last_seen=_maybe_utc(row["last_seen"]),
        scratch_dir=row["scratch_dir"],
        port=_col(row, "port", 7890),
        ssh_key=row["ssh_key"],
        venv=row["venv"],
        devices=row["devices"],
    )


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRecord:
    """Map a database row to an ArtifactRecord."""
    return ArtifactRecord(
        artifact_id=row["artifact_id"],
        size_bytes=row["size_bytes"],
        stored_at=_maybe_utc(row["stored_at"]),
        ref_count=_col(row, "ref_count", 0),
        setup_command=row["setup_command"],
    )


def _ensure_utc(dt: float | int | datetime | None) -> datetime:
    """Return *dt* with tzinfo=UTC.  Assumes naive datetimes are UTC.

    Accepts float/int (epoch seconds) from the DB REAL columns
    and converts them to timezone-aware datetimes.
    """
    if dt is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(dt, (float, int)):
        dt = datetime.fromtimestamp(dt, tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _maybe_utc(dt: float | int | datetime | None) -> datetime | None:
    """Return *dt* with tzinfo=UTC or None.

    Accepts float/int (epoch seconds) from the DB REAL columns
    and converts them to timezone-aware datetimes.
    """
    if dt is None:
        return None
    if isinstance(dt, (float, int)):
        dt = datetime.fromtimestamp(dt, tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


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


async def init_db(db: aiosqlite.Connection) -> None:
    """Create tables and indexes if they do not exist (idempotent).

    Enables WAL mode and foreign key enforcement on the connection.
    """
    db.row_factory = sqlite3.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")

    # ── experiments ─────────────────────────────────────────────────
    await db.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id  TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            submit_time    REAL NOT NULL,
            controller_id  TEXT,
            note           TEXT,
            status         TEXT NOT NULL DEFAULT 'running',
            expected_jobs  INTEGER NOT NULL DEFAULT 0,
            singular_dims  TEXT NOT NULL DEFAULT '[]'
        );
    """)

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
            jobs_per_gpu  INTEGER NOT NULL DEFAULT 1
        );
    """)

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

    # ── jobs ────────────────────────────────────────────────────────
    await db.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
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
            jobs_per_gpu       INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (run_id, experiment_id)
        );
    """)

    # ── metrics ─────────────────────────────────────────────────────
    await db.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            run_id        TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            step          INTEGER NOT NULL,
            data          TEXT NOT NULL,
            PRIMARY KEY (run_id, experiment_id, step)
        );
    """)

    # ── logs ─────────────────────────────────────────────────────────
    await db.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            run_id        TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            seq           INTEGER NOT NULL,
            data          TEXT NOT NULL,
            PRIMARY KEY (run_id, experiment_id, seq)
        );
    """)

    # ── migrations for existing databases ───────────────────────────
    try:
        await db.execute("ALTER TABLE jobs ADD COLUMN jobs_per_gpu INTEGER DEFAULT 1")
        await db.commit()
    except Exception:
        pass  # column already exists
    try:
        await db.execute("ALTER TABLE experiments ADD COLUMN singular_dims TEXT NOT NULL DEFAULT '[]'")
        await db.commit()
    except Exception:
        pass  # column already exists

    # ── indexes ─────────────────────────────────────────────────────
    await db.executescript("""
        CREATE INDEX IF NOT EXISTS idx_jobs_experiment
            ON jobs(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_dispatch
            ON jobs(status, priority DESC, submit_time ASC);
        CREATE INDEX IF NOT EXISTS idx_jobs_worker
            ON jobs(worker_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_experiment
            ON metrics(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_logs_experiment
            ON logs(experiment_id);
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
    controller_id: str | None = None,
    note: str | None = None,
    status: ExperimentStatus = "running",
    expected_jobs: int = 0,
    singular_dims: list[str] | None = None,
) -> ExperimentRecord:
    """Insert a new experiment and return the row."""
    now = _now_epoch()
    singular_dims_json = json.dumps(singular_dims or [])
    row = await _exec_one(
        db,
        """
        INSERT INTO experiments (experiment_id, name, submit_time, controller_id, note, status, expected_jobs, singular_dims)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (experiment_id) DO UPDATE SET
            name = EXCLUDED.name,
            controller_id = EXCLUDED.controller_id,
            note = EXCLUDED.note,
            status = EXCLUDED.status,
            expected_jobs = EXCLUDED.expected_jobs,
            singular_dims = EXCLUDED.singular_dims
        RETURNING *;
        """,
        (experiment_id, name, now, controller_id, note, status, expected_jobs, singular_dims_json),
    )
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


async def list_experiments(
    db: aiosqlite.Connection,
    status: ExperimentStatus | None = None,
) -> list[ExperimentRecord]:
    """List experiments, optionally filtered by status."""
    if status is not None:
        cursor = await db.execute(
            "SELECT * FROM experiments WHERE status = ? ORDER BY submit_time DESC", (status,)
        )
    else:
        cursor = await db.execute("SELECT * FROM experiments ORDER BY submit_time DESC")
    rows = await cursor.fetchall()
    return [_row_to_experiment(r) for r in rows]


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
    status: WorkerStatus = "connected",
) -> WorkerRecord:
    """Insert or update a worker row; set last_seen = now."""
    now = _now_epoch()
    row = await _exec_one(
        db,
        """
        INSERT INTO workers (worker_id, host, remote_dir, status, last_seen,
                             scratch_dir, port, ssh_key, venv, devices)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (worker_id) DO UPDATE SET
            host = EXCLUDED.host,
            remote_dir = EXCLUDED.remote_dir,
            status = EXCLUDED.status,
            last_seen = EXCLUDED.last_seen,
            scratch_dir = EXCLUDED.scratch_dir,
            port = EXCLUDED.port,
            ssh_key = EXCLUDED.ssh_key,
            venv = EXCLUDED.venv,
            devices = EXCLUDED.devices
        RETURNING *;
        """,
        (worker_id, host, remote_dir, status, now,
         scratch_dir, port, ssh_key, venv, devices),
    )
    await db.commit()
    assert row is not None
    return _row_to_worker(row)


async def update_worker_status(
    db: aiosqlite.Connection,
    worker_id: str,
    status: WorkerStatus,
) -> WorkerRecord | None:
    """Update a worker's status and last_seen. Returns updated row or None."""
    now = _now_epoch()
    row = await _exec_one(
        db,
        "UPDATE workers SET status = ?, last_seen = ? WHERE worker_id = ? RETURNING *",
        (status, now, worker_id),
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
    jobs_per_gpu: int = 1,
) -> JobRecord:
    """Insert a new job row. Returns the created JobRecord."""
    now = _now_epoch()
    command_json, combo_json, env_json, return_files_json, files_json = _serialize_job_fields(
        command, combo, env, return_files, files
    )

    row = await _exec_one(
        db,
        """
        INSERT INTO jobs (
            run_id, experiment_id, priority, status, submit_time,
            command, combo, env, gpus_per_run, nodes_per_run,
            set_dist_env, run_from, return_files, files, max_retries,
            artifact_id, setup_command, jobs_per_gpu
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?
        )
        RETURNING *;
        """,
        (run_id, experiment_id, priority, status, now,
         command_json, combo_json, env_json, gpus_per_run, nodes_per_run,
         int(set_dist_env), run_from, return_files_json, files_json, max_retries,
         artifact_id, setup_command, jobs_per_gpu),
    )
    await db.commit()
    assert row is not None
    return _row_to_job(row)


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
    now = _now_epoch()
    try:
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
                    artifact_id, setup_command, jobs_per_gpu
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                RETURNING *;
                """,
                (j["run_id"], j["experiment_id"], j.get("priority", 0),
                 j.get("status", "pending"), now,
                 command_json, combo_json, env_json,
                 j.get("gpus_per_run", 1), j.get("nodes_per_run", 1),
                 int(j.get("set_dist_env", False)), j.get("run_from"),
                 return_files_json, files_json, j.get("max_retries", 2),
                 j.get("artifact_id"), j.get("setup_command"),
                 j.get("jobs_per_gpu", 1)),
            )
            assert row is not None
            records.append(_row_to_job(row))
        await db.commit()
        return records
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
) -> list[JobRecord]:
    """Return pending jobs ordered by priority DESC, submit_time ASC.

    This matches the composite index ``idx_jobs_dispatch`` for efficient
    index-only scans.
    """
    if experiment_id is not None:
        if limit is not None:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending' AND experiment_id = ?
                ORDER BY priority DESC, submit_time ASC
                LIMIT ?
                """,
                (experiment_id, limit),
            )
        else:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending' AND experiment_id = ?
                ORDER BY priority DESC, submit_time ASC
                """,
                (experiment_id,),
            )
    else:
        if limit is not None:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending'
                ORDER BY priority DESC, submit_time ASC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending'
                ORDER BY priority DESC, submit_time ASC
                """
            )
    rows = await cursor.fetchall()
    return [_row_to_job(r) for r in rows]


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
    **kwargs: Any,
) -> JobRecord | None:
    """Generic job status update.  Extra keyword arguments are set as columns
    (e.g. ``exit_code=0``, ``elapsed=12.3``).

    Returns the updated row or None.
    """
    set_clauses = ["status = ?"]
    values: list[Any] = [status]
    for col, val in kwargs.items():
        set_clauses.append(f"{col} = ?")
        values.append(val)
    values.append(run_id)
    values.append(experiment_id)
    row = await _exec_one(
        db,
        f"UPDATE jobs SET {', '.join(set_clauses)} WHERE run_id = ? AND experiment_id = ? RETURNING *",
        tuple(values),
    )
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
            dispatched_gpu_ids = ?
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
        """
        UPDATE jobs
        SET status = ?, finish_time = ?, exit_code = ?, elapsed = ?
        WHERE run_id = ? AND experiment_id = ? AND status IN ('running', 'dispatched')
        RETURNING *;
        """,
        (status, now, exit_code, elapsed, run_id, experiment_id),
    )
    await db.commit()
    return _row_to_job(row) if row else None


async def reclassify_singular_xfails(
    db: aiosqlite.Connection,
    experiment_id: str,
    succeeded_combo: dict[str, Any],
) -> list[str]:
    """After a job succeeds, reclassify failed singular-probe siblings as xfailed.

    A failed job is xfailed if it shares the same lex (non-singular) dim values as the
    succeeded job but has a different value on at least one singular dim — meaning it was
    probing above the threshold that the success established.  Returns run_ids reclassified.
    """
    exp = await get_experiment(db, experiment_id)
    if exp is None:
        return []
    singular_dims: list[str] = json.loads(exp.singular_dims)
    if not singular_dims:
        return []

    singular_set = set(singular_dims)
    lex_combo = {k: v for k, v in succeeded_combo.items() if k not in singular_set}

    cursor = await db.execute(
        "SELECT run_id, combo FROM jobs WHERE experiment_id = ? AND status = 'failed'",
        (experiment_id,),
    )
    rows = await cursor.fetchall()

    to_reclassify: list[str] = []
    for row in rows:
        try:
            combo: dict[str, Any] = json.loads(row["combo"])
        except (json.JSONDecodeError, TypeError):
            continue
        row_lex = {k: v for k, v in combo.items() if k not in singular_set}
        if row_lex != lex_combo:
            continue
        if any(combo.get(d) != succeeded_combo.get(d) for d in singular_dims):
            to_reclassify.append(row["run_id"])

    if not to_reclassify:
        return []

    placeholders = ",".join("?" * len(to_reclassify))
    await db.execute(
        f"UPDATE jobs SET status = 'xfailed' WHERE run_id IN ({placeholders}) AND experiment_id = ?",
        (*to_reclassify, experiment_id),
    )
    await db.commit()
    return to_reclassify


async def cancel_job(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> JobRecord | None:
    """Mark a pending job as cancelled (does not affect running jobs)."""
    row = await _exec_one(
        db,
        """
        UPDATE jobs SET status = 'cancelled'
        WHERE run_id = ? AND experiment_id = ? AND status = 'pending'
        RETURNING *;
        """,
        (run_id, experiment_id),
    )
    await db.commit()
    return _row_to_job(row) if row else None


async def reset_dispatched_running_to_pending(
    db: aiosqlite.Connection,
) -> int:
    """On manager restart, move any dispatched/running jobs back to pending.

    Returns the number of jobs reset.
    """
    async with db.execute(
        """
        UPDATE jobs
        SET status = 'pending',
            dispatch_time = NULL,
            start_time = NULL,
            worker_id = NULL,
            dispatched_gpu_ids = NULL
        WHERE status IN ('dispatched', 'running');
        """,
    ) as cursor:
        rowcount = cursor.rowcount
    await db.commit()
    return rowcount


async def increment_retry(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> JobRecord | None:
    """Increment retry_count and reset status to pending. Returns None if
    max_retries already reached.
    """
    row = await _exec_one(
        db,
        """
        UPDATE jobs
        SET retry_count = retry_count + 1,
            status = 'pending',
            dispatch_time = NULL,
            start_time = NULL,
            finish_time = NULL,
            elapsed = NULL,
            exit_code = NULL,
            worker_id = NULL,
            dispatched_gpu_ids = NULL
        WHERE run_id = ? AND experiment_id = ? AND retry_count < max_retries
        RETURNING *;
        """,
        (run_id, experiment_id),
    )
    await db.commit()
    return _row_to_job(row) if row else None


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


async def list_jobs_by_status(
    db: aiosqlite.Connection,
    status: JobStatus,
    limit: int | None = None,
) -> list[JobRecord]:
    """Return jobs with the given *status* across all experiments,
    newest first.  If *limit* is provided, only the first N rows are
    returned.
    """
    if limit is not None:
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY submit_time DESC LIMIT ?",
            (status, limit),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY submit_time DESC",
            (status,),
        )
    rows = await cursor.fetchall()
    return [_row_to_job(r) for r in rows]


async def reset_worker_jobs(
    db: aiosqlite.Connection,
    worker_id: str,
) -> list[tuple[str, str]]:
    """Reset any dispatched/running jobs belonging to *worker_id* back to
    pending and clear their dispatch metadata.  Returns a list of
    ``(run_id, experiment_id)`` tuples for the affected jobs.
    """
    rows = await _exec_all(
        db,
        """
        UPDATE jobs
        SET status = 'pending',
            dispatch_time = NULL,
            worker_id = NULL,
            start_time = NULL,
            dispatched_gpu_ids = NULL
        WHERE worker_id = ? AND status IN ('dispatched', 'running')
        RETURNING run_id, experiment_id
        """,
        (worker_id,),
    )
    await db.commit()
    return [(r["run_id"], r["experiment_id"]) for r in rows]


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
    cursor = await db.execute(
        "SELECT COUNT(*) FROM jobs WHERE experiment_id = ? "
        "AND status IN ('pending', 'dispatched', 'running')",
        (experiment_id,),
    )
    row = await cursor.fetchone()
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
        "status": exp.status if exp else None,
        "note": exp.note if exp else None,
        "submit_time": exp.submit_time.isoformat() if exp else None,
        "job_counts": counts,
    }


async def list_experiments_with_counts(
    db: aiosqlite.Connection,
    status: "ExperimentStatus | None" = None,
) -> list[dict[str, Any]]:
    """List experiments with per-status job counts in a single query."""
    where = "WHERE e.status = ?" if status is not None else ""
    params = (status,) if status is not None else ()
    cursor = await db.execute(
        f"""
        SELECT e.*,
               COUNT(j.run_id) AS total_jobs,
               SUM(CASE WHEN j.status = 'done'       THEN 1 ELSE 0 END) AS done_jobs,
               SUM(CASE WHEN j.status = 'failed'     THEN 1 ELSE 0 END) AS failed_jobs,
               SUM(CASE WHEN j.status = 'xfailed'    THEN 1 ELSE 0 END) AS xfailed_jobs,
               SUM(CASE WHEN j.status = 'running'    THEN 1 ELSE 0 END) AS running_jobs,
               SUM(CASE WHEN j.status = 'pending'    THEN 1 ELSE 0 END) AS pending_jobs,
               SUM(CASE WHEN j.status = 'dispatched' THEN 1 ELSE 0 END) AS dispatched_jobs
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
        d["job_counts"] = {
            "total":      r["total_jobs"]      or 0,
            "done":       r["done_jobs"]       or 0,
            "failed":     r["failed_jobs"]     or 0,
            "xfailed":    r["xfailed_jobs"]    or 0,
            "running":    r["running_jobs"]    or 0,
            "pending":    r["pending_jobs"]    or 0,
            "dispatched": r["dispatched_jobs"] or 0,
        }
        result.append(d)
    return result


async def delete_experiment(
    db: aiosqlite.Connection,
    experiment_id: str,
) -> bool:
    """Delete an experiment and all its jobs. Returns True if it existed."""
    await db.execute("DELETE FROM jobs WHERE experiment_id = ?", (experiment_id,))
    await db.execute("DELETE FROM metrics WHERE experiment_id = ?", (experiment_id,))
    await db.execute("DELETE FROM logs WHERE experiment_id = ?", (experiment_id,))
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
    run_id: str,
    experiment_id: str,
    step: int,
    data: dict[str, Any],
) -> None:
    """Persist a single metric row. Silently ignores duplicate (run_id, experiment_id, step)."""
    await db.execute(
        "INSERT OR IGNORE INTO metrics (run_id, experiment_id, step, data) VALUES (?, ?, ?, ?)",
        (run_id, experiment_id, step, json.dumps(data)),
    )
    await db.commit()


async def get_metrics_for_run(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Return all metric rows for a run as dicts, ordered by step."""
    async with db.execute(
        "SELECT step, data FROM metrics WHERE run_id = ? AND experiment_id = ? ORDER BY step",
        (run_id, experiment_id),
    ) as cursor:
        rows = await cursor.fetchall()
    result = []
    for row in rows:
        d = json.loads(row["data"])
        d["step"] = row["step"]
        result.append(d)
    return result


# ===============================================================================
# Logs
# ===============================================================================


async def insert_log(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
    seq: int,
    data: str,
) -> None:
    """Persist a single log chunk. Silently ignores duplicate (run_id, experiment_id, seq)."""
    await db.execute(
        "INSERT OR IGNORE INTO logs (run_id, experiment_id, seq, data) VALUES (?, ?, ?, ?)",
        (run_id, experiment_id, seq, data),
    )
    await db.commit()


async def get_logs_for_run(
    db: aiosqlite.Connection,
    run_id: str,
    experiment_id: str,
) -> str:
    """Return the full log text for a run, ordered by seq."""
    async with db.execute(
        "SELECT data FROM logs WHERE run_id = ? AND experiment_id = ? ORDER BY seq",
        (run_id, experiment_id),
    ) as cursor:
        rows = await cursor.fetchall()
    return "".join(row["data"] for row in rows)


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
        controller_id: str | None = None,
        note: str | None = None,
        status: ExperimentStatus = "running",
        expected_jobs: int = 0,
        singular_dims: list[str] | None = None,
    ) -> ExperimentRecord:
        db = self._db
        return await self._enqueue(lambda: create_experiment(
            db, experiment_id=experiment_id, name=name,
            controller_id=controller_id, note=note,
            status=status, expected_jobs=expected_jobs,
            singular_dims=singular_dims,
        ))

    async def update_experiment_status(
        self, experiment_id: str, status: ExperimentStatus
    ) -> ExperimentRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_experiment_status(db, experiment_id, status))

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
        status: WorkerStatus = "connected",
    ) -> WorkerRecord:
        db = self._db
        return await self._enqueue(lambda: upsert_worker(
            db, worker_id=worker_id, host=host, remote_dir=remote_dir,
            scratch_dir=scratch_dir, port=port, ssh_key=ssh_key,
            venv=venv, devices=devices, status=status,
        ))

    async def update_worker_status(
        self, worker_id: str, status: WorkerStatus
    ) -> WorkerRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_worker_status(db, worker_id, status))

    async def touch_worker(self, worker_id: str) -> None:
        db = self._db
        await self._enqueue(lambda: touch_worker(db, worker_id))

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
        jobs_per_gpu: int = 1,
    ) -> JobRecord:
        db = self._db
        return await self._enqueue(lambda: insert_job(
            db, run_id=run_id, experiment_id=experiment_id, priority=priority,
            command=command, combo=combo, env=env, status=status,
            gpus_per_run=gpus_per_run, nodes_per_run=nodes_per_run,
            set_dist_env=set_dist_env, run_from=run_from, return_files=return_files,
            files=files, max_retries=max_retries, artifact_id=artifact_id,
            setup_command=setup_command, jobs_per_gpu=jobs_per_gpu,
        ))

    async def insert_jobs_bulk(self, jobs: list[dict[str, Any]]) -> list[JobRecord]:
        db = self._db
        return await self._enqueue(lambda: insert_jobs_bulk(db, jobs))

    async def update_job_status(
        self,
        run_id: str,
        experiment_id: str,
        status: JobStatus,
        **kwargs: Any,
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_job_status(db, run_id, experiment_id, status, **kwargs))

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

    async def reclassify_singular_xfails(
        self, experiment_id: str, succeeded_combo: dict[str, Any]
    ) -> list[str]:
        db = self._db
        return await self._enqueue(lambda: reclassify_singular_xfails(db, experiment_id, succeeded_combo))

    async def cancel_job(
        self, run_id: str, experiment_id: str
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: cancel_job(db, run_id, experiment_id))

    async def increment_retry(
        self, run_id: str, experiment_id: str
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: increment_retry(db, run_id, experiment_id))

    async def update_job_priority(
        self, run_id: str, experiment_id: str, priority: int
    ) -> JobRecord | None:
        db = self._db
        return await self._enqueue(lambda: update_job_priority(db, run_id, experiment_id, priority))

    async def reset_worker_jobs(
        self, worker_id: str
    ) -> list[tuple[str, str]]:
        db = self._db
        return await self._enqueue(lambda: reset_worker_jobs(db, worker_id))

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
        self,
        run_id: str,
        experiment_id: str,
        step: int,
        data: dict[str, Any],
    ) -> None:
        db = self._db
        await self._enqueue(lambda: insert_metric(db, run_id, experiment_id, step, data))

    async def insert_log(
        self,
        run_id: str,
        experiment_id: str,
        seq: int,
        data: str,
    ) -> None:
        db = self._db
        await self._enqueue(lambda: insert_log(db, run_id, experiment_id, seq, data))
