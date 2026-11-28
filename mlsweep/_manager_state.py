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

Jobs are dispatched only by the scheduler task (``scheduler_loop``), which also
adds the next trials of Bayesian sweeps (``_manager_optimizer``).  Anything
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
from typing import Any, Literal, cast

from mlsweep._manager_db import DbWriter, JobRecord, JobStatus
from mlsweep._shared import DEFAULT_SCRATCH_DIR

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
    sync_inflight: bool = False  # an artifact rsync for a mid-run sync is running
    last_progress: float = 0.0  # epoch seconds of the last log/metric/start event
    last_step: int | None = None  # newest metric step this session
    last_step_time: float = 0.0  # epoch seconds when last_step first arrived here
    step_seconds: float | None = None  # the logger's wall seconds per step, if it sent one
    total_steps: int | None = None  # the step the run ends at, if its logger said
    first_step: int | None = None  # first metric step this session, and when it arrived
    first_step_time: float = 0.0
    metric_seen: bool = False  # it logged its sweep's METRIC_CHECK metric this session
    # Set when the manager is stopping the run (pruned, timed out): the status and
    # reason its job gets once the worker's result arrives, and when the stop began.
    ending: tuple[JobStatus, str] | None = None
    ending_since: float = 0.0

    def note_step(self, step: int, step_seconds: float | None, now: float,
                  total_steps: int | None = None) -> None:
        """Record a metric for *step* that arrived at *now*.

        Only a higher step restarts the clock, so several log() calls at one
        step count once.  *step_seconds* is measured by the logger, and
        *total_steps* is what the script told its logger, if anything.
        """
        if step_seconds is not None:
            self.step_seconds = step_seconds
        if total_steps is not None:
            self.total_steps = total_steps
        if self.last_step is None or step > self.last_step:
            self.last_step = step
            self.last_step_time = now
        if self.first_step is None:
            self.first_step, self.first_step_time = step, now

    @property
    def avg_step_seconds(self) -> float | None:
        """Wall seconds per step since the first step this session, once it has advanced."""
        if self.first_step is None or self.last_step is None or self.last_step <= self.first_step:
            return None
        return (self.last_step_time - self.first_step_time) / (self.last_step - self.first_step)

    def eta_seconds(self, now: float) -> float | None:
        """Seconds until the run reaches ``total_steps``, at its average pace so far.

        None until the logger has said where the run ends and the step has
        advanced (or the logger has timed one).  Counted from when the last
        step arrived, so it keeps falling between steps, and never below 0.
        """
        per = self.avg_step_seconds or self.step_seconds
        if self.total_steps is None or self.last_step is None or not per:
            return None
        left = max(0, self.total_steps - self.last_step) * per - (now - self.last_step_time)
        return max(0.0, left)

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


# How the scheduler places jobs on GPUs.  "pack" fills the busiest GPUs and
# workers that still have room, keeping idle GPUs and whole machines free for
# multi-GPU and multi-node jobs.  "spread" puts each job on the least busy
# GPU, so every GPU gets one job before any gets a second.
Placement = Literal["pack", "spread"]
PLACEMENTS: tuple[Placement, ...] = ("pack", "spread")
DEFAULT_PLACEMENT: Placement = "pack"


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
    # The pool: GPUs the scheduler may use on this worker, editable through the API.
    # Health is kept apart from it, so a GPU that recovers needs no pool edit and
    # a health report never undoes one; placement uses ``usable_gpus``.
    gpus: list[int] = field(default_factory=list)
    unhealthy_gpus: list[int] = field(default_factory=list)  # failed the worker's CUDA probe
    unhealthy_reasons: dict[str, str] = field(default_factory=dict)  # str(gpu) -> why
    pending_gpus: list[int] = field(default_factory=list)  # probe not finished yet
    topo: dict[str, int] = field(default_factory=dict)
    gpu_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    max_jobs_per_gpu: int = 1
    send_queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    status: str = "connecting"  # connecting → connected ⇄ reconnecting → dead
    conn_gen: int = 0  # bumped per TCP connection; tasks of older connections stand down
    reconnect_attempts: int = 0
    hello_seen: bool = False  # later hellos keep GPU/concurrency settings changed via the API
    scratch_dir: str = DEFAULT_SCRATCH_DIR
    code: str = ""  # fingerprint of the worker's mlsweep code, from its hello
    remote_dir: str = ""
    password: str | None = None
    ssh_key: str | None = None
    venv: str | None = None
    tunnel_proc: Any = None  # asyncio.subprocess.Process keeping the reverse tunnel alive
    forward: Any = None  # PortForward the manager reaches the worker through, for a --tunnel worker
    tunnel_task: Any = None  # asyncio.Task of _tunnel_monitor_task, cancelled on retirement

    @property
    def usable_gpus(self) -> list[int]:
        """Pool GPUs that passed the worker's health probe."""
        bad = set(self.unhealthy_gpus) | set(self.pending_gpus)
        return [g for g in self.gpus if g not in bad]


@dataclass
class Pin:
    """Where the next dispatch of a job must go, chosen by hand (System page or
    ``mlsweep reschedule --to``).  The scheduler uses it once and drops it."""

    worker_id: str
    gpus: list[int]
    made: float = field(default_factory=time.time)


# A pin the scheduler could not use for this long is forgotten, so a job that
# was cancelled or paused meanwhile does not jump to that GPU much later.
PIN_TTL_SECONDS = 600.0


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
        self.public_url: str = ""  # set once the manager has reached itself at its public IP
        self.scratch_dir: str = DEFAULT_SCRATCH_DIR  # for workers that do not set their own
        self.db_writer: DbWriter = cast(DbWriter, None)
        self.workers: dict[str, WorkerConn] = {}
        self.launching: set[str] = set()  # worker ids with a launch in progress
        # Worker ids refused at their hello for speaking an older protocol.
        # The System page offers to update them (see ``update_worker``).
        self.outdated: set[str] = set()
        self.runs: dict[RunKey, InFlightRun] = {}
        self.pins: dict[RunKey, Pin] = {}  # hand-chosen placements for pending jobs
        self.placement: Placement = DEFAULT_PLACEMENT
        # Scheduling pauses, saved in the settings table.  While ``paused`` is
        # set nothing starts anywhere; a worker in ``paused_workers`` starts
        # nothing new, pinned jobs included.  Running jobs carry on either way.
        self.paused: bool = False
        self.paused_workers: set[str] = set()
        # Metric values of finished Bayesian runs, keyed by
        # (experiment_id, job_key, attempt, metric, goal).  See _manager_optimizer.
        self.run_values: dict[tuple[str, int, int, str, str], float | None] = {}
        self.pruned_points: dict[tuple[str, int, int, str, str], tuple[int, float | None]] = {}
        # Run controls of running and paused experiments, by experiment id, reloaded by
        # each scheduler pass, and the pruning state of those that prune.  See _manager_controls.
        self.controls: dict[str, dict[str, Any]] = {}
        self.prune: dict[str, Any] = {}  # experiment id → PruneTracker
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

    def runs_by_gpu(self, worker_id: str) -> dict[int, list[InFlightRun]]:
        """In-flight runs on each GPU of *worker_id*, in or out of its pool."""
        out: dict[int, list[InFlightRun]] = {}
        for run in self.runs.values():
            for g in run.nodes.get(worker_id, ()):
                out.setdefault(g, []).append(run)
        return out

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
    "Pin",
    "PIN_TTL_SECONDS",
    "WorkerConn",
]
