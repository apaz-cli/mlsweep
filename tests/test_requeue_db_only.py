"""Jobs the database has running, but the manager holds no in-flight state for.

After a manager restart, a run the database still calls running is known only
from its row until its worker's hello adopts it.  When that worker never comes
back, is removed from the configuration, or comes back without the run, the
job must go back to pending (or fail once out of retries), not stay
``running`` for ever.
"""

import asyncio

import aiosqlite

import mlsweep._manager_workers as mw
from mlsweep._manager_db import (
    DbWriter,
    create_experiment,
    dispatch_job,
    get_job,
    init_db,
    insert_job,
    list_run_events,
    mark_job_running,
    upsert_worker,
)
from mlsweep._manager_state import ManagerState


def _with_db(tmp_path, body):
    async def run():
        db = await aiosqlite.connect(tmp_path / "manager.db")
        await init_db(db)
        writer = DbWriter(db)
        task = asyncio.create_task(writer.run())
        try:
            await create_experiment(db, experiment_id="e", name="e", campaign="test")
            state = ManagerState(output_dir=str(tmp_path / "out"))
            state.db_writer = writer
            return await body(db, state)
        finally:
            task.cancel()
            await db.close()

    return asyncio.run(run())


async def _running_on(db, worker_id, run_id, max_retries=2):
    await upsert_worker(db, worker_id=worker_id, host="gone.invalid", remote_dir="")
    await insert_job(db, run_id=run_id, experiment_id="e", command=["true"], max_retries=max_retries)
    await dispatch_job(db, run_id, "e", worker_id, [0])
    await mark_job_running(db, run_id, "e")


def test_jobs_on_a_worker_no_longer_configured_are_requeued(tmp_path):
    async def body(db, state):
        await _running_on(db, "gone:7890", "r1")
        await mw.requeue_jobs_of_unknown_workers(db, state)
        job = await get_job(db, "r1", "e")
        events = await list_run_events(db, "e", "r1")
        return job, events

    job, events = _with_db(tmp_path, body)
    assert job.status == "pending" and job.retry_count == 1
    assert events[-1]["action"] == "requeued" and "no longer configured" in events[-1]["detail"]


def test_a_dead_workers_jobs_from_before_a_restart_are_requeued(tmp_path):
    async def body(db, state):
        await _running_on(db, "w:7890", "r1")
        await _running_on(db, "w:7890", "last", max_retries=0)
        await mw.declare_worker_dead(db, state, "w:7890", "unreachable after 10 reconnect attempts")
        return await get_job(db, "r1", "e"), await get_job(db, "last", "e")

    r1, last = _with_db(tmp_path, body)
    assert r1.status == "pending"
    assert last.status == "failed"  # no retries left
