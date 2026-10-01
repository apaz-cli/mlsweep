"""In-memory state for the mlsweep manager.

Provides:
  - ``InFlightRun`` — a dispatched run and the worker node(s) it occupies
  - ``WorkerConn`` — live connection to a worker
  - ``ManagerState`` — the control lock, the in-flight runs, and the workers

Concurrency model
-----------------
The database is the source of truth for job status.  The in-memory state
tracks which runs occupy which worker GPUs.  Every change to either (job
status transitions, in-flight tracking, worker membership) happens inside one
``async with state.lock`` block, so no coroutine ever observes the two out of
step.  Slow I/O (rsync, SSH, launching workers) never runs under the lock.

Jobs are dispatched only by the scheduler task (``scheduler_loop``).  Anything
that may free capacity or add work calls ``state.request_schedule()``, which
just wakes that task; the task also runs periodically, so a missed wake-up
delays scheduling by seconds instead of stalling it.

Multi-node aggregation state lives in the ``job_nodes`` table so it survives a
manager restart.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, cast

from mlsweep._manager_db import DbWriter, JobRecord

# A run is identified by (experiment_id, run_id); run names repeat across experiments.
RunKey = tuple[str, str]


# ── In-flight run ─────────────────────────────────────────────────────────────


@dataclass
class InFlightRun:
    """A run dispatched to one or more workers (one node per worker)."""

    experiment_id: str
    run_id: str
    job_key: int  # logs and metrics are stored under (job_key, attempt)
    attempt: int
    primary: str  # worker running node 0; only its log is stored
    multinode: bool = False  # nodes are also recorded in the job_nodes table
    nodes: dict[str, list[int]] = field(default_factory=dict)  # worker_id → GPU ids
    log_end: int = 0  # training.log is stored up to this byte offset
    replay_requested: bool = False
    last_progress: float = 0.0  # epoch seconds of the last log/metric/start event

    @classmethod
    def from_job(cls, job: JobRecord, primary: str, **kwargs: Any) -> InFlightRun:
        # A freshly adopted/dispatched run starts its stall clock now.
        kwargs.setdefault("last_progress", time.time())
        return cls(
            experiment_id=job.experiment_id, run_id=job.run_id, job_key=job.job_key,
            attempt=job.attempt, primary=primary, multinode=job.nodes_per_run > 1, **kwargs,
        )

    @property
    def key(self) -> RunKey:
        return (self.experiment_id, self.run_id)


# ── Worker connection ─────────────────────────────────────────────────────────


@dataclass
class WorkerConn:
    """Live connection to a worker.

    GPU occupancy is not stored here; ``ManagerState.occupancy`` derives it
    from the in-flight runs that have a node on this worker.
    """

    worker_id: str
    host: str
    port: int
    writer: asyncio.StreamWriter | None = None  # the current connection's stream
    gpus: list[int] = field(default_factory=list)
    topo: dict[str, int] = field(default_factory=dict)
    gpu_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    max_jobs_per_gpu: int = 1
    send_queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    status: str = "connecting"  # connecting → connected ⇄ reconnecting → dead
    conn_gen: int = 0  # bumped per TCP connection; tasks of older connections stand down
    reconnect_attempts: int = 0
    hello_seen: bool = False  # later hellos keep GPU/concurrency settings changed via the API
    scratch_dir: str = "/tmp/mlsweep"
    remote_dir: str = ""
    password: str | None = None
    ssh_key: str | None = None
    venv: str | None = None
    tunnel_proc: Any = None  # asyncio.subprocess.Process keeping the reverse tunnel alive


# ── Manager state ──────────────────────────────────────────────────────────────


class ManagerState:
    """Central in-memory state for the manager.  See the module docstring."""

    def __init__(
        self,
        output_dir: str = "",
        artifact_base_url: str = "",
        token: str = "",
    ) -> None:
        self.output_dir: str = output_dir
        self.artifact_base_url: str = artifact_base_url
        self.token: str = token
        self.manager_port: int = 0
        self.db_writer: DbWriter = cast(DbWriter, None)
        self.workers: dict[str, WorkerConn] = {}
        self.launching: set[str] = set()  # worker ids with a launch in progress
        self.runs: dict[RunKey, InFlightRun] = {}
        self.subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self.lock: asyncio.Lock = asyncio.Lock()
        self.schedule_event: asyncio.Event = asyncio.Event()
        self.shutdown_event: asyncio.Event = asyncio.Event()

    def request_schedule(self) -> None:
        """Ask the scheduler task for a pass.  Safe to call from anywhere, any time."""
        self.schedule_event.set()

    # ── In-flight helpers ─────────────────────────────────────────────────

    def runs_on(self, worker_id: str) -> list[InFlightRun]:
        """In-flight runs with a node on *worker_id*."""
        return [r for r in self.runs.values() if worker_id in r.nodes]

    def runs_of(self, experiment_id: str) -> list[RunKey]:
        """Keys of the in-flight runs of *experiment_id*."""
        return [r.key for r in self.runs.values() if r.experiment_id == experiment_id]

    def reserve_worker_id(self, worker_id: str) -> bool:
        """Claim *worker_id* for a launch.  False if it is live or already launching.

        The caller holds ``self.lock``; the launch releases the claim when done.
        """
        existing = self.workers.get(worker_id)
        if worker_id in self.launching or (existing is not None and existing.status != "dead"):
            return False
        self.launching.add(worker_id)
        return True

    def occupancy(self, wc: WorkerConn) -> dict[int, int]:
        """Number of in-flight runs on each of *wc*'s GPUs."""
        occ = {g: 0 for g in wc.gpus}
        for run in self.runs.values():
            for g in run.nodes.get(wc.worker_id, ()):
                if g in occ:
                    occ[g] += 1
        return occ

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
    "InFlightRun",
    "ManagerState",
    "RunKey",
    "WorkerConn",
]
