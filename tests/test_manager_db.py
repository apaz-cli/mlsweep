"""Unit tests for the mlsweep manager DB layer.

Tests CRUD functions against an in-memory SQLite database — no manager
process needed.  All async DB calls are wrapped with ``asyncio.run()`` so
pytest-asyncio is not required.
"""

import asyncio

import aiosqlite
import pytest

from mlsweep._manager_db import (
    JobStatus,
    init_db,
    create_experiment,
    delete_experiment,
    get_experiment,
    update_experiment_status,
    experiment_summary,
    insert_job,
    insert_jobs_bulk,
    get_job,
    list_jobs_by_experiment,
    list_pending_jobs,
    list_schedulable_jobs,
    experiment_concurrency_caps,
    update_experiment_max_concurrent,
    is_multinode_run,
    insert_job_nodes,
    mark_job_node_result,
    multinode_progress,
    delete_job_nodes,
    update_job_status,
    update_job_priority,
    cancel_job,
    increment_retry,
    count_active_jobs,
    count_jobs_by_status,
    insert_metric,
    get_metrics_for_run,
    insert_log,
    get_logs_for_run,
    register_artifact,
    get_artifact,
    increment_artifact_ref,
)


async def _init_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await init_db(conn)
    return conn


# ── Metrics CRUD ────────────────────────────────────────────────────────────────


def test_insert_and_get_metrics():
    async def run():
        db = await _init_db()
        try:
            await insert_metric(db, "run1", "exp1", 1, {"loss": 0.5})
            await insert_metric(db, "run1", "exp1", 2, {"loss": 0.3, "acc": 0.9})
            await insert_metric(db, "run1", "exp1", 3, {"loss": 0.1})
            rows = await get_metrics_for_run(db, "run1", "exp1")
            assert len(rows) == 3
            assert rows[0]["step"] == 1
            assert rows[0]["loss"] == 0.5
            assert rows[1]["step"] == 2
            assert rows[1]["acc"] == 0.9
            assert rows[2]["step"] == 3
        finally:
            await db.close()

    asyncio.run(run())


def test_insert_metric_duplicate():
    async def run():
        db = await _init_db()
        try:
            await insert_metric(db, "run1", "exp1", 1, {"loss": 0.5})
            await insert_metric(db, "run1", "exp1", 1, {"loss": 999.0})
            rows = await get_metrics_for_run(db, "run1", "exp1")
            assert len(rows) == 1
            assert rows[0]["loss"] == 0.5
        finally:
            await db.close()

    asyncio.run(run())


def test_get_metrics_empty_run():
    async def run():
        db = await _init_db()
        try:
            rows = await get_metrics_for_run(db, "run_nonexist", "exp_nonexist")
            assert rows == []
        finally:
            await db.close()

    asyncio.run(run())


def test_get_metrics_wrong_experiment():
    async def run():
        db = await _init_db()
        try:
            await insert_metric(db, "run1", "exp1", 1, {"loss": 0.5})
            rows = await get_metrics_for_run(db, "run1", "exp_wrong")
            assert rows == []
        finally:
            await db.close()

    asyncio.run(run())


def test_insert_and_get_logs():
    async def run():
        db = await _init_db()
        try:
            await insert_log(db, "run1", "exp1", 1, "hello\n")
            await insert_log(db, "run1", "exp1", 2, "world\n")
            text = await get_logs_for_run(db, "run1", "exp1")
            assert text == "hello\nworld\n"
            assert await get_logs_for_run(db, "run99", "exp99") == ""
        finally:
            await db.close()

    asyncio.run(run())


# ── Experiments ─────────────────────────────────────────────────────────────────


def test_create_and_get_experiment():
    async def run():
        db = await _init_db()
        try:
            exp = await create_experiment(db, experiment_id="exp1", name="test_exp")
            assert exp.experiment_id == "exp1"
            assert exp.name == "test_exp"
            assert exp.status == "running"
            fetched = await get_experiment(db, "exp1")
            assert fetched is not None
            assert fetched.experiment_id == "exp1"
        finally:
            await db.close()

    asyncio.run(run())


def test_get_experiment_not_found():
    async def run():
        db = await _init_db()
        try:
            assert await get_experiment(db, "nonexist") is None
        finally:
            await db.close()

    asyncio.run(run())


def test_update_experiment_status():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            updated = await update_experiment_status(db, "exp1", "completed")
            assert updated is not None
            assert updated.status == "completed"
        finally:
            await db.close()

    asyncio.run(run())


def test_delete_experiment_cascades():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="run1", experiment_id="exp1",
                             command=["echo", "hi"])
            await insert_metric(db, "run1", "exp1", 1, {"x": 1})
            await insert_log(db, "run1", "exp1", 1, "log\n")

            assert await get_experiment(db, "exp1") is not None
            assert await get_job(db, "run1", "exp1") is not None
            assert len(await get_metrics_for_run(db, "run1", "exp1")) == 1
            assert await get_logs_for_run(db, "run1", "exp1") != ""

            existed = await delete_experiment(db, "exp1")
            assert existed is True

            assert await get_experiment(db, "exp1") is None
            assert await get_job(db, "run1", "exp1") is None
            assert await get_metrics_for_run(db, "run1", "exp1") == []
            assert await get_logs_for_run(db, "run1", "exp1") == ""
        finally:
            await db.close()

    asyncio.run(run())


# ── Jobs ────────────────────────────────────────────────────────────────────────


def test_insert_job_basic():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            job = await insert_job(db, run_id="run1", experiment_id="exp1",
                                   command=["echo", "hello"])
            assert job.run_id == "run1"
            assert job.experiment_id == "exp1"
            assert job.status == "pending"
            assert job.retry_count == 0
            assert job.max_retries == 2
        finally:
            await db.close()

    asyncio.run(run())


def test_insert_jobs_bulk():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            payloads = [
                {"run_id": "run_a", "experiment_id": "exp1",
                 "command": ["echo", "a"]},
                {"run_id": "run_b", "experiment_id": "exp1",
                 "command": ["echo", "b"]},
            ]
            records = await insert_jobs_bulk(db, payloads)
            assert len(records) == 2
            assert {r.run_id for r in records} == {"run_a", "run_b"}
        finally:
            await db.close()

    asyncio.run(run())


def test_list_pending_jobs_ordering():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r3", experiment_id="exp1", priority=0,
                             command=["echo"])
            await insert_job(db, run_id="r1", experiment_id="exp1", priority=10,
                             command=["echo"])
            await insert_job(db, run_id="r2", experiment_id="exp1", priority=5,
                             command=["echo"])
            pending = await list_pending_jobs(db)
            assert [j.run_id for j in pending] == ["r1", "r2", "r3"]
        finally:
            await db.close()

    asyncio.run(run())


def test_list_schedulable_jobs_excludes_paused_and_aborted():
    """The scheduler's candidate query must skip jobs whose experiment is
    paused or aborted, and order the rest by priority then submit time.

    This is the DB half of the fix for the 'abort/pause does nothing' bug:
    the scheduler reads only from here, so a paused/aborted experiment simply
    stops producing schedulable work.
    """
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="run_exp", name="t", status="running")
            await create_experiment(db, experiment_id="pause_exp", name="t", status="paused")
            await create_experiment(db, experiment_id="abort_exp", name="t", status="aborted")
            await create_experiment(db, experiment_id="done_exp", name="t", status="completed")

            await insert_job(db, run_id="a", experiment_id="run_exp", priority=1, command=["echo"])
            await insert_job(db, run_id="b", experiment_id="run_exp", priority=9, command=["echo"])
            await insert_job(db, run_id="p", experiment_id="pause_exp", priority=5, command=["echo"])
            await insert_job(db, run_id="x", experiment_id="abort_exp", priority=5, command=["echo"])
            # 'completed' experiments stay schedulable (so retries still run).
            await insert_job(db, run_id="d", experiment_id="done_exp", priority=5, command=["echo"])

            schedulable = await list_schedulable_jobs(db)
            ids = [j.run_id for j in schedulable]
            assert "p" not in ids  # paused excluded
            assert "x" not in ids  # aborted excluded
            assert set(ids) == {"a", "b", "d"}
            # Highest priority first.
            assert ids[0] == "b"
        finally:
            await db.close()

    asyncio.run(run())


def test_multinode_aggregation_via_db():
    """Multi-node result aggregation is derived from durable job_nodes rows, so
    it is correct and restart-safe (no in-memory counter). A run completes only
    when every node is terminal; success is the AND across nodes, elapsed the max.
    """
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp", name="t")
            await insert_job(db, run_id="r", experiment_id="exp",
                             command=["echo"], nodes_per_run=2)

            assert not await is_multinode_run(db, "r", "exp")
            await insert_job_nodes(db, "r", "exp", [(0, "w0", [0]), (1, "w1", [0])])
            assert await is_multinode_run(db, "r", "exp")

            remaining, all_ok, elapsed = await multinode_progress(db, "r", "exp")
            assert remaining == 2 and all_ok and elapsed == 0.0

            await mark_job_node_result(db, "r", "exp", "w0", True, 1.5)
            remaining, all_ok, elapsed = await multinode_progress(db, "r", "exp")
            assert remaining == 1  # still waiting on w1

            await mark_job_node_result(db, "r", "exp", "w1", True, 3.0)
            remaining, all_ok, elapsed = await multinode_progress(db, "r", "exp")
            assert remaining == 0
            assert all_ok is True
            assert elapsed == 3.0  # slowest node

            await delete_job_nodes(db, "r", "exp")
            assert not await is_multinode_run(db, "r", "exp")
        finally:
            await db.close()

    asyncio.run(run())


def test_multinode_aggregation_failure():
    """One failed node makes the aggregated result a failure."""
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp", name="t")
            await insert_job(db, run_id="r", experiment_id="exp",
                             command=["echo"], nodes_per_run=2)
            await insert_job_nodes(db, "r", "exp", [(0, "w0", [0]), (1, "w1", [0])])

            await mark_job_node_result(db, "r", "exp", "w0", True, 1.0)
            await mark_job_node_result(db, "r", "exp", "w1", False, 2.0)
            remaining, all_ok, elapsed = await multinode_progress(db, "r", "exp")
            assert remaining == 0
            assert all_ok is False
            assert elapsed == 2.0
        finally:
            await db.close()

    asyncio.run(run())


def test_experiment_concurrency_caps_roundtrip():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="e1", name="t", max_concurrent=3)
            await create_experiment(db, experiment_id="e2", name="t")  # default 0

            caps = await experiment_concurrency_caps(db)
            assert caps["e1"] == 3
            assert caps["e2"] == 0

            updated = await update_experiment_max_concurrent(db, "e1", 7)
            assert updated is not None and updated.max_concurrent == 7
            caps = await experiment_concurrency_caps(db)
            assert caps["e1"] == 7
        finally:
            await db.close()

    asyncio.run(run())


def test_list_jobs_by_experiment_filter():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"], status="pending")
            await insert_job(db, run_id="r2", experiment_id="exp1",
                             command=["echo"], status="done")
            pending = await list_jobs_by_experiment(db, "exp1", status="pending")
            assert len(pending) == 1
            assert pending[0].run_id == "r1"
            done = await list_jobs_by_experiment(db, "exp1", status="done")
            assert len(done) == 1
            assert done[0].run_id == "r2"
        finally:
            await db.close()

    asyncio.run(run())


def test_update_job_status():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"])
            job = await update_job_status(db, "r1", "exp1", "done",
                                          exit_code=0, elapsed=1.5)
            assert job is not None
            assert job.status == "done"
            assert job.exit_code == 0
            assert job.elapsed == 1.5
        finally:
            await db.close()

    asyncio.run(run())


def test_update_job_priority():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"])
            job = await update_job_priority(db, "r1", "exp1", 100)
            assert job.priority == 100
        finally:
            await db.close()

    asyncio.run(run())


def test_cancel_job():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"], status="pending")
            job = await cancel_job(db, "r1", "exp1")
            assert job is not None
            assert job.status == "cancelled"
            # Cancel an already-cancelled job → None (only pending)
            assert await cancel_job(db, "r1", "exp1") is None
        finally:
            await db.close()

    asyncio.run(run())


def test_increment_retry():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"], status="failed")
            job = await increment_retry(db, "r1", "exp1")
            assert job is not None
            assert job.retry_count == 1
            assert job.status == "pending"
            job = await increment_retry(db, "r1", "exp1")
            assert job is not None
            assert job.retry_count == 2
            assert job.status == "pending"
        finally:
            await db.close()

    asyncio.run(run())


def test_increment_retry_at_max_retries():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"], status="failed")
            await increment_retry(db, "r1", "exp1")  # 1
            await increment_retry(db, "r1", "exp1")  # 2
            # retry_count (2) < max_retries (2) is false → None
            job = await increment_retry(db, "r1", "exp1")
            assert job is None
        finally:
            await db.close()

    asyncio.run(run())


# ── Experiment summary ──────────────────────────────────────────────────────────


def test_experiment_summary_counts():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="test")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"], status="done")
            await insert_job(db, run_id="r2", experiment_id="exp1",
                             command=["echo"], status="failed")
            await insert_job(db, run_id="r3", experiment_id="exp1",
                             command=["echo"], status="pending")
            summary = await experiment_summary(db, "exp1")
            assert summary["name"] == "test"
            counts = summary["job_counts"]
            assert counts.get("done", 0) == 1
            assert counts.get("failed", 0) == 1
            assert counts.get("pending", 0) == 1
        finally:
            await db.close()

    asyncio.run(run())


def test_count_active_jobs():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"], status="pending")
            await insert_job(db, run_id="r2", experiment_id="exp1",
                             command=["echo"], status="dispatched")
            await insert_job(db, run_id="r3", experiment_id="exp1",
                             command=["echo"], status="running")
            await insert_job(db, run_id="r4", experiment_id="exp1",
                             command=["echo"], status="done")
            await insert_job(db, run_id="r5", experiment_id="exp1",
                             command=["echo"], status="failed")
            await insert_job(db, run_id="r6", experiment_id="exp1",
                             command=["echo"], status="cancelled")
            assert await count_active_jobs(db, "exp1") == 3

            await update_job_status(db, "r1", "exp1", "done")
            await update_job_status(db, "r2", "exp1", "failed")
            await update_job_status(db, "r3", "exp1", "cancelled")
            assert await count_active_jobs(db, "exp1") == 0
        finally:
            await db.close()

    asyncio.run(run())


# ── Artifacts ───────────────────────────────────────────────────────────────────


def test_register_and_get_artifact():
    async def run():
        db = await _init_db()
        try:
            art = await register_artifact(db, artifact_id="sha256:abc",
                                          size_bytes=1024)
            assert art.artifact_id == "sha256:abc"
            assert art.size_bytes == 1024
            assert art.ref_count == 1
            art2 = await register_artifact(db, artifact_id="sha256:abc",
                                           size_bytes=2048)
            assert art2.ref_count == 2
            assert art2.size_bytes == 2048
            fetched = await get_artifact(db, "sha256:abc")
            assert fetched is not None
            assert fetched.ref_count == 2
        finally:
            await db.close()

    asyncio.run(run())


def test_increment_artifact_ref():
    async def run():
        db = await _init_db()
        try:
            await register_artifact(db, artifact_id="sha256:abc")
            art = await increment_artifact_ref(db, "sha256:abc", delta=1)
            assert art.ref_count == 2
            art = await increment_artifact_ref(db, "sha256:abc", delta=-1)
            assert art.ref_count == 1
            assert await increment_artifact_ref(db, "sha256:nope") is None
        finally:
            await db.close()

    asyncio.run(run())
