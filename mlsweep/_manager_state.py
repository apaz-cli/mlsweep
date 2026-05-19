"""In-memory state dataclasses for mlsweep manager.

Provides:
  - ``MultiNodeState`` — typed dict for multi-node job aggregation
  - ``InFlightJob`` — a job dispatched to workers, tracked in memory
  - ``WorkerConn`` — live connection to a worker with GPU occupancy
  - ``ManagerState`` — central scheduling state with thread-safe helpers
"""

from __future__ import annotations

import asyncio
import bisect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict, cast

from mlsweep._manager_db import DbWriter, JobRecord


# ── Multi-node aggregation state ──────────────────────────────────────────────


class MultiNodeState(TypedDict):
    """Aggregation state for a multi-node job tracked in ``multinode_pending``."""

    remaining: int
    success: bool
    elapsed: float


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
    """Live connection to a worker, with GPU occupancy and message queue."""

    worker_id: str
    host: str
    port: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    gpus: list[int] = field(default_factory=list)
    topo: dict[str, int] = field(default_factory=dict)
    gpu_occupancy: dict[int, int] = field(default_factory=dict)
    gpu_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    max_jobs_per_gpu: int = 1
    send_queue: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)
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
        self.pending: list[JobRecord] = []
        self.in_flight: dict[str, InFlightJob] = {}
        self.subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self.scheduler_lock: asyncio.Lock = asyncio.Lock()
        self.multinode_pending: dict[str, dict[str, Any]] = {}

    # ── Pending list helpers ──────────────────────────────────────────────

    def _pending_sort_key(self, job: JobRecord) -> tuple[int, datetime]:
        return (-job.priority, job.submit_time)

    def insert_pending(self, job: JobRecord) -> None:
        bisect.insort(
            self.pending, job, key=lambda j: self._pending_sort_key(j)
        )

    def remove_pending(self, run_id: str) -> JobRecord | None:
        for i, job in enumerate(self.pending):
            if job.run_id == run_id:
                return self.pending.pop(i)
        return None

    def update_priority(self, run_id: str, new_priority: int) -> bool:
        job = self.remove_pending(run_id)
        if job is None:
            return False
        job.priority = new_priority
        self.insert_pending(job)
        return True

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
    "MultiNodeState",
    "WorkerConn",
]
