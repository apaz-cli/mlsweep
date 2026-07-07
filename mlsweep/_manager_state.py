"""In-memory state dataclasses for mlsweep manager.

Provides:
  - ``InFlightJob`` — a job dispatched to workers, tracked in memory
  - ``WorkerConn`` — live connection to a worker (occupancy derived from in_flight)
  - ``ManagerState`` — central scheduling state with thread-safe helpers

Multi-node aggregation state is **not** held here; it lives in the ``job_nodes``
table so it survives a manager restart.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from mlsweep._manager_db import DbWriter


# ── In-flight job ─────────────────────────────────────────────────────────────


@dataclass
class InFlightJob:
    """A job dispatched to one or more workers and currently tracked in memory.

    For single-node jobs, ``worker_id`` and ``worker_ids`` contain the same
    single element.  For multi-node jobs, ``worker_id`` is the primary
    (coordinating) worker while ``worker_ids`` lists all participating workers.
    """

    run_id: str
    worker_id: str
    experiment_id: str
    dispatch_time: datetime
    start_time: datetime | None = None
    gpu_ids: list[int] = field(default_factory=list)
    worker_ids: list[str] = field(default_factory=list)
    combo: dict[str, Any] = field(default_factory=dict)
    log_seq: int = 0
    metric_seq: int = 0


# ── Worker connection ─────────────────────────────────────────────────────────


@dataclass
class WorkerConn:
    """Live connection to a worker.

    GPU occupancy is **not** stored as a counter here; it is derived on demand
    from ``in_flight`` (each entry carries this worker's ``gpu_ids``).  This
    keeps occupancy impossible to leak — there is no decrement to forget.
    """

    worker_id: str
    host: str
    port: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    gpus: list[int] = field(default_factory=list)
    topo: dict[str, int] = field(default_factory=dict)
    gpu_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    max_jobs_per_gpu: int = 1
    send_queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    in_flight: dict[str, InFlightJob] = field(default_factory=dict)
    status: str = "connected"
    connected_at: datetime | None = None
    scratch_dir: str = "/tmp/mlsweep"
    remote_dir: str = ""
    password: str | None = None
    ssh_key: str | None = None
    venv: str | None = None
    tunnel_proc: Any = None  # asyncio.subprocess.Process keeping the reverse tunnel alive


# ── Manager state ──────────────────────────────────────────────────────────────


class ManagerState:
    """Central in-memory state for the sweep manager.

    Owns all scheduling data structures and provides thread-safe methods
    for manipulating them.  Uses asyncio.Lock to serialise mutations.
    """

    def __init__(
        self,
        output_dir: str = "",
        artifact_base_url: str = "",
        token: str = "",
    ) -> None:
        self.output_dir: str = output_dir
        self.artifact_base_url: str = artifact_base_url
        self.token: str = token
        self.dispatch_callback: Any = None
        self.db_writer: DbWriter = cast(DbWriter, None)
        self.workers: dict[str, WorkerConn] = {}
        # NOTE: there is no in-memory ``pending`` list.  Pending jobs live in
        # the database; the scheduler reads them fresh each pass via
        # ``list_schedulable_jobs``.  This makes the DB the single source of
        # truth, so control verbs (cancel, abort, delete, retry) take effect by
        # writing the DB and cannot desync an in-memory mirror.
        self.in_flight: dict[str, InFlightJob] = {}
        self.subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self.scheduler_lock: asyncio.Lock = asyncio.Lock()
        # Coalescing guard for schedule_pending: prevents concurrent scheduling
        # passes and folds rapid re-triggers into a single follow-up pass.
        self._scheduling: bool = False
        self._reschedule: bool = False

    # ── In-flight helpers ─────────────────────────────────────────────────

    def add_in_flight(self, job: InFlightJob) -> None:
        self.in_flight[job.run_id] = job

    def remove_in_flight(self, run_id: str) -> InFlightJob | None:
        return self.in_flight.pop(run_id, None)

    def get_in_flight(self, run_id: str) -> InFlightJob | None:
        return self.in_flight.get(run_id)

    # ── Subscriber helpers ────────────────────────────────────────────────

    def add_subscriber(
        self, experiment_id: str, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        self.subscribers.setdefault(experiment_id, []).append(queue)

    def remove_subscriber(
        self, experiment_id: str, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        queues = self.subscribers.get(experiment_id)
        if queues is not None:
            try:
                queues.remove(queue)
            except ValueError:
                pass
            if not queues:
                del self.subscribers[experiment_id]

    def broadcast(self, experiment_id: str, event: dict[str, Any]) -> None:
        queues = self.subscribers.get(experiment_id)
        if not queues:
            return
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.remove_subscriber(experiment_id, q)


__all__ = [
    "InFlightJob",
    "ManagerState",
    "WorkerConn",
]
