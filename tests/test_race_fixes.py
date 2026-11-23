"""Small regression tests for the control-plane races fixed alongside the UI."""

import asyncio

import aiosqlite

from mlsweep._manager_db import (
    create_experiment,
    get_job,
    init_db,
    insert_job,
    move_jobs,
    purge_jobs,
    update_job_status,
)
from mlsweep._sessions import SessionService


async def _db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    return db


async def _job(db: aiosqlite.Connection, experiment_id: str, run_id: str, status: str) -> None:
    await insert_job(db, run_id=run_id, experiment_id=experiment_id, command=["true"])
    if status != "pending":
        await update_job_status(db, run_id, experiment_id, status, exit_code=0)


def test_move_jobs_skips_a_job_that_is_not_finished():
    async def run() -> None:
        db = await _db()
        try:
            for eid in ("src", "dst"):
                await create_experiment(db, experiment_id=eid, name=eid, campaign="test")
            await _job(db, "src", "pending", "pending")
            await _job(db, "src", "done", "done")

            moved = await move_jobs(db, "src", "dst", ["pending", "done"])

            assert moved == ["done"]
            pending = await get_job(db, "pending", "src")
            done = await get_job(db, "done", "dst")
            assert pending is not None and pending.status == "pending"
            assert done is not None and done.status == "done"
        finally:
            await db.close()

    asyncio.run(run())


def test_purge_jobs_skips_a_job_that_is_not_finished():
    async def run() -> None:
        db = await _db()
        try:
            await create_experiment(db, experiment_id="e", name="e", campaign="test")
            await _job(db, "e", "pending", "pending")
            await _job(db, "e", "done", "done")

            purged = await purge_jobs(db, "e", ["pending", "done"])

            assert purged == ["done"]
            pending = await get_job(db, "pending", "e")
            done = await get_job(db, "done", "e")
            assert pending is not None and pending.status == "pending"
            assert done is None
        finally:
            await db.close()

    asyncio.run(run())


class _CountingSessions(SessionService):
    """A SessionService whose create step only records how many run at once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = 0
        self.peak = 0

    async def _create_locked(self, name, folder=""):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        return object()


def test_terminal_create_is_serialized(tmp_path):
    """Two simultaneous create requests must not both pass the limit check."""
    svc = _CountingSessions(tmp_path, 1, "token", notes=False)

    async def run() -> None:
        await asyncio.gather(svc.create("one"), svc.create("two"))

    asyncio.run(run())
    assert svc.peak == 1


def test_capture_watcher_lock_admits_one_holder(tmp_path):
    """A second watcher must see the lock taken and stand down."""
    from mlsweep._capture import _flocked

    lock = tmp_path / "watcher.lock"
    with _flocked(lock, block=False) as first:
        assert first
        with _flocked(lock, block=False) as second:
            assert not second
    with _flocked(lock, block=False) as again:
        assert again


def test_ipc_from_a_replaced_attempt_is_dropped(tmp_path):
    """A logger bound to attempt 1 must not write into attempt 2's metrics."""
    from mlsweep import worker

    def run_state(attempt: int) -> "worker.RunState":
        scratch = tmp_path / f"a{attempt}"
        scratch.mkdir()
        return worker.RunState(run_id="r", scratch_path=str(scratch), gpu_ids=[],
                               experiment="e", attempt=attempt)

    old, new = run_state(1), run_state(2)
    key = ("e", "r")
    with worker._lock:
        worker._in_flight[key] = new
    try:
        worker._handle_ipc_msg({"type": "metric", "run_id": "r", "experiment": "e",
                                "step": 1, "data": {"loss": 1.0}}, old)
    finally:
        with worker._lock:
            worker._in_flight.pop(key, None)
    assert not (tmp_path / "a1" / "metrics.jsonl").exists()
