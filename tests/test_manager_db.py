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
    insert_job_nodes,
    list_job_nodes,
    mark_job_node_result,
    multinode_progress,
    delete_job_nodes,
    update_job_status,
    update_job_priority,
    cancel_jobs,
    requeue_jobs,
    retry_job,
    dispatch_job,
    list_active_jobs,
    upsert_worker,
    get_worker,
    pack_metrics,
    apply_result_rules,
    count_active_jobs,
    count_jobs_by_status,
    insert_metric,
    get_metrics_for_run,
    insert_log,
    get_logs_for_run,
    last_log_seq,
    uncompacted_log_attempts,
    compact_log_attempt,
    replace_log_rows,
    finish_job,
    DbWriter,
    register_artifact,
    get_artifact,
    increment_artifact_ref,
    get_setting,
    set_setting,
)


async def _init_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await init_db(conn)
    return conn


async def _dispatched(db, run_id="run1", experiment_id="exp1"):
    """A job dispatched once (attempt 1), as a worker would be running it."""
    if await get_experiment(db, experiment_id) is None:
        await create_experiment(db, experiment_id=experiment_id, name="t")
        await upsert_worker(db, worker_id="w1", host="h", remote_dir="/")
    await insert_job(db, run_id=run_id, experiment_id=experiment_id, command=["echo"])
    return await dispatch_job(db, run_id, experiment_id, "w1", [0])


# ── Metrics and logs ────────────────────────────────────────────────────────────


def test_settings_roundtrip():
    async def run():
        db = await _init_db()
        try:
            assert await get_setting(db, "placement") is None
            await set_setting(db, "placement", "spread")
            await set_setting(db, "placement", "pack")
            assert await get_setting(db, "placement") == "pack"
        finally:
            await db.close()

    asyncio.run(run())


def test_setup_command_may_be_a_string_or_an_argument_list():
    """Dispatch splits the stored command with shlex, so a list is stored joined."""
    import shlex

    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="e", name="e")
            jobs = await insert_jobs_bulk(db, [
                {"run_id": "s", "experiment_id": "e", "command": ["true"], "setup_command": "make all"},
                {"run_id": "l", "experiment_id": "e", "command": ["true"],
                 "setup_command": ["sh", "-c", "echo 'a b'"]},
            ])
            assert jobs[0].setup_command == "make all"
            assert shlex.split(jobs[1].setup_command) == ["sh", "-c", "echo 'a b'"]
            with pytest.raises(ValueError):
                await insert_jobs_bulk(db, [{"run_id": "x", "experiment_id": "e",
                                             "command": ["true"], "setup_command": 3}])
        finally:
            await db.close()

    asyncio.run(run())


def test_worker_unhealthy_devices_roundtrip():
    async def run():
        db = await _init_db()
        try:
            await upsert_worker(db, worker_id="w1", host="h", remote_dir="/",
                                devices="[0, 1, 3]", unhealthy_devices="[2]")
            w = await get_worker(db, "w1")
            assert w.devices == "[0, 1, 3]"
            assert w.unhealthy_devices == "[2]"
            # A later upsert without health clears it.
            await upsert_worker(db, worker_id="w1", host="h", remote_dir="/",
                                devices="[0, 1]")
            w = await get_worker(db, "w1")
            assert w.devices == "[0, 1]"
            assert w.unhealthy_devices is None
        finally:
            await db.close()

    asyncio.run(run())


def test_insert_and_get_metrics():
    async def run():
        db = await _init_db()
        try:
            job = await _dispatched(db)
            await insert_metric(db, job.job_key, job.attempt, 1, {"loss": 0.5})
            await insert_metric(db, job.job_key, job.attempt, 2, {"loss": 0.3, "acc": 0.9})
            await insert_metric(db, job.job_key, job.attempt, 3, {"loss": 0.1})
            rows = await get_metrics_for_run(db, "run1", "exp1")
            assert [r["step"] for r in rows] == [1, 2, 3]
            assert rows[0]["loss"] == 0.5
            assert rows[1]["acc"] == 0.9
            assert await get_metrics_for_run(db, "run1", "exp_wrong") == []
            assert await get_metrics_for_run(db, "run_nonexist", "exp1") == []
        finally:
            await db.close()

    asyncio.run(run())


def test_insert_metric_merges_same_step():
    async def run():
        db = await _init_db()
        try:
            job = await _dispatched(db)
            await insert_metric(db, job.job_key, job.attempt, 1, {"loss": 0.5, "acc": 0.9})
            await insert_metric(db, job.job_key, job.attempt, 1, {"loss": 999.0, "final": 0.8})
            rows = await get_metrics_for_run(db, "run1", "exp1")
            assert len(rows) == 1
            # Same-key values are overwritten; other keys are preserved.
            assert rows[0]["loss"] == 999.0
            assert rows[0]["acc"] == 0.9
            assert rows[0]["final"] == 0.8
        finally:
            await db.close()

    asyncio.run(run())


def test_retried_job_reports_only_the_latest_attempts_metrics():
    async def run():
        db = await _init_db()
        try:
            first = await _dispatched(db)
            await insert_metric(db, first.job_key, first.attempt, 1, {"loss": 9.0})
            await insert_metric(db, first.job_key, first.attempt, 2, {"loss": 8.0})
            await update_job_status(db, "run1", "exp1", "failed")
            await retry_job(db, "run1", "exp1")
            second = await dispatch_job(db, "run1", "exp1", "w1", [0])
            assert second.attempt == first.attempt + 1
            # These reuse attempt 1's steps and must not be dropped as duplicates.
            await insert_metric(db, second.job_key, second.attempt, 1, {"loss": 0.5})
            rows = await get_metrics_for_run(db, "run1", "exp1")
            assert [(r["step"], r["loss"]) for r in rows] == [(1, 0.5)]
        finally:
            await db.close()

    asyncio.run(run())


def test_packed_metrics_read_back_the_same():
    async def run():
        db = await _init_db()
        try:
            job = await _dispatched(db)
            for step in range(1, 51):
                await insert_metric(db, job.job_key, job.attempt, step, {"loss": 1.0 / step})
            await pack_metrics(db, job.job_key, job.attempt)
            async with db.execute("SELECT count(*), typeof(data) FROM metrics") as cur:
                assert tuple(await cur.fetchone()) == (1, "blob")
            # A late duplicate does not override; a genuinely new step is kept.
            await insert_metric(db, job.job_key, job.attempt, 7, {"loss": 999.0})
            await insert_metric(db, job.job_key, job.attempt, 51, {"loss": 0.0})
            rows = await get_metrics_for_run(db, "run1", "exp1")
            assert [r["step"] for r in rows] == list(range(1, 52))
            assert rows[6]["loss"] == 1.0 / 7 and rows[50]["loss"] == 0.0
            await pack_metrics(db, job.job_key, job.attempt)  # repacking merges too
            assert await get_metrics_for_run(db, "run1", "exp1") == rows
        finally:
            await db.close()

    asyncio.run(run())


def test_insert_and_get_logs():
    async def run():
        db = await _init_db()
        try:
            job = await _dispatched(db)
            await insert_log(db, job.job_key, job.attempt, 6, "hello\n")
            big = "".join(f"step {i} loss 0.5\n" for i in range(2000))  # compressed on disk
            await insert_log(db, job.job_key, job.attempt, 6 + len(big), big)
            assert await get_logs_for_run(db, "run1", "exp1") == "hello\n" + big
            assert await get_logs_for_run(db, "run99", "exp99") == ""
            async with db.execute("SELECT typeof(data) FROM logs ORDER BY seq") as cur:
                assert [r[0] for r in await cur.fetchall()] == ["text", "blob"]
        finally:
            await db.close()

    asyncio.run(run())


def test_logs_of_each_attempt_are_kept_apart():
    async def run():
        db = await _init_db()
        try:
            first = await _dispatched(db)
            await insert_log(db, first.job_key, first.attempt, 4, "one\n")
            await requeue_jobs(db, [("exp1", "run1")], spend_retry=True)
            second = await dispatch_job(db, "run1", "exp1", "w1", [0])
            await insert_log(db, second.job_key, second.attempt, 4, "two\n")
            text = await get_logs_for_run(db, "run1", "exp1")
            assert "one\n" in text and "two\n" in text
            assert text.index("one") < text.index("two")
        finally:
            await db.close()

    asyncio.run(run())



async def _log_rows(db, job_key):
    async with db.execute(
        "SELECT attempt, seq, data FROM logs WHERE job_key = ? ORDER BY attempt, seq", (job_key,)
    ) as cur:
        return [tuple(r) for r in await cur.fetchall()]


async def _write_chunks(db, job, chunks):
    seq = await last_log_seq(db, job.job_key, job.attempt)
    for chunk in chunks:
        seq += len(chunk.encode())
        await insert_log(db, job.job_key, job.attempt, seq, chunk)
    return seq


_CHUNKS = ["hello\n", "".join(f"step {i} loss 0.5\n" for i in range(500)), "bye\n"]


def test_old_without_rowid_logs_table_is_rebuilt(tmp_path):
    async def run():
        path = str(tmp_path / "m.db")
        db = await aiosqlite.connect(path)
        await db.execute(
            "CREATE TABLE logs (job_key INTEGER NOT NULL, attempt INTEGER NOT NULL, "
            "seq INTEGER NOT NULL, data NOT NULL, PRIMARY KEY (job_key, attempt, seq)) WITHOUT ROWID"
        )
        await db.executemany("INSERT INTO logs VALUES (?, ?, ?, ?)",
                             [(1, 1, 4, "one\n"), (1, 1, 8, b"x"), (2, 1, 4, "two\n")])
        await db.commit()
        await db.close()

        db = await aiosqlite.connect(path)
        try:
            await init_db(db)
            async with db.execute("SELECT sql FROM sqlite_master WHERE name = 'logs'") as cur:
                assert "WITHOUT ROWID" not in (await cur.fetchone())[0].upper()
            async with db.execute("PRAGMA auto_vacuum") as cur:
                assert (await cur.fetchone())[0] == 2  # incremental, set by the rebuild's VACUUM
            async with db.execute("SELECT * FROM logs ORDER BY job_key, seq") as cur:
                assert [tuple(r) for r in await cur.fetchall()] == [
                    (1, 1, 4, "one\n"), (1, 1, 8, b"x"), (2, 1, 4, "two\n")]
            await init_db(db)  # idempotent
            await insert_log(db, 2, 1, 4, "dup\n")  # duplicate seq still ignored
            assert len(await _log_rows(db, 2)) == 1
        finally:
            await db.close()

    asyncio.run(run())


def test_finished_attempt_logs_compact_to_one_row():
    async def run():
        db = await _init_db()
        writer = DbWriter(db)
        task = asyncio.create_task(writer.run())
        try:
            job = await _dispatched(db)
            end = await _write_chunks(db, job, _CHUNKS)
            # Still running, so left alone.
            assert await uncompacted_log_attempts(db) == []

            await finish_job(db, "run1", "exp1", success=True, exit_code=0, elapsed=1.0)
            assert await uncompacted_log_attempts(db) == [(job.job_key, job.attempt)]
            assert await compact_log_attempt(db, writer, job.job_key, job.attempt)

            rows = await _log_rows(db, job.job_key)
            assert [(a, s) for a, s, _ in rows] == [(job.attempt, end)]
            assert isinstance(rows[0][2], bytes)
            assert await get_logs_for_run(db, "run1", "exp1") == "".join(_CHUNKS)
            assert await last_log_seq(db, job.job_key, job.attempt) == end
            assert await uncompacted_log_attempts(db) == []
        finally:
            task.cancel()
            await db.close()

    asyncio.run(run())


def test_earlier_attempt_logs_compact_while_the_retry_runs():
    async def run():
        db = await _init_db()
        writer = DbWriter(db)
        task = asyncio.create_task(writer.run())
        try:
            first = await _dispatched(db)
            await _write_chunks(db, first, _CHUNKS)
            await requeue_jobs(db, [("exp1", "run1")], spend_retry=True)
            second = await dispatch_job(db, "run1", "exp1", "w1", [0])
            await _write_chunks(db, second, ["a\n", "b\n"])

            assert await uncompacted_log_attempts(db) == [(first.job_key, first.attempt)]
            before = await get_logs_for_run(db, "run1", "exp1")
            assert await compact_log_attempt(db, writer, first.job_key, first.attempt)
            assert await get_logs_for_run(db, "run1", "exp1") == before
        finally:
            task.cancel()
            await db.close()

    asyncio.run(run())


def test_log_swap_is_refused_when_rows_changed_and_keeps_later_rows():
    async def run():
        db = await _init_db()
        try:
            job = await _dispatched(db)
            end = await _write_chunks(db, job, _CHUNKS)
            rows = await _log_rows(db, job.job_key)
            # Read 3 rows, but a 4th arrived after them: the 3 are swapped, the 4th kept.
            await _write_chunks(db, job, ["late\n"])
            assert await replace_log_rows(db, job.job_key, job.attempt, 3, end, "".join(_CHUNKS))
            assert await get_logs_for_run(db, "run1", "exp1") == "".join(_CHUNKS) + "late\n"
            # A count that no longer matches writes nothing.
            after = await _log_rows(db, job.job_key)
            assert not await replace_log_rows(db, job.job_key, job.attempt, len(rows), end, "x")
            assert await _log_rows(db, job.job_key) == after
        finally:
            await db.close()

    asyncio.run(run())


def test_failed_log_swap_rolls_back(tmp_path):
    async def run():
        db = await aiosqlite.connect(str(tmp_path / "m.db"))
        await init_db(db)
        writer = DbWriter(db)
        task = asyncio.create_task(writer.run())
        try:
            job = await _dispatched(db)
            end = await _write_chunks(db, job, _CHUNKS)
            before = await _log_rows(db, job.job_key)
            # The delete runs, then the insert fails (data NOT NULL).
            with pytest.raises(Exception):
                await writer.replace_log_rows(job.job_key, job.attempt, 3, end, None)
            assert await _log_rows(db, job.job_key) == before
        finally:
            task.cancel()
            await db.close()

    asyncio.run(run())


def test_interrupted_log_swap_leaves_nothing(tmp_path):
    async def run():
        path = str(tmp_path / "m.db")
        db = await aiosqlite.connect(path)
        await init_db(db)
        job = await _dispatched(db)
        end = await _write_chunks(db, job, _CHUNKS)
        before = await _log_rows(db, job.job_key)
        # Die between the delete and the commit.
        await db.execute("DELETE FROM logs WHERE job_key = ? AND seq <= ?", (job.job_key, end))
        await db.close()

        db = await aiosqlite.connect(path)
        try:
            await init_db(db)
            assert await _log_rows(db, job.job_key) == before
        finally:
            await db.close()

    asyncio.run(run())

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


def test_create_experiment_metric_and_goal():
    async def run():
        db = await _init_db()
        try:
            exp = await create_experiment(
                db, experiment_id="exp1", name="test_exp",
                metric="val_loss", goal="maximize",
            )
            assert exp.metric == "val_loss"
            assert exp.goal == "maximize"
            fetched = await get_experiment(db, "exp1")
            assert fetched is not None
            assert (fetched.metric, fetched.goal) == ("val_loss", "maximize")
            summary = await experiment_summary(db, "exp1")
            assert summary["metric"] == "val_loss"
            assert summary["goal"] == "maximize"
            # Re-creating without metric/goal keeps the stored values.
            again = await create_experiment(db, experiment_id="exp1", name="test_exp")
            assert (again.metric, again.goal) == ("val_loss", "maximize")
        finally:
            await db.close()

    asyncio.run(run())


def test_init_db_migrates_metric_and_goal_columns():
    async def run():
        db = await aiosqlite.connect(":memory:")
        try:
            # An experiments table from before metric/goal existed.
            await db.execute("""
                CREATE TABLE experiments (
                    experiment_id  TEXT PRIMARY KEY,
                    name           TEXT NOT NULL,
                    submit_time    REAL NOT NULL,
                    controller_id  TEXT,
                    note           TEXT,
                    status         TEXT NOT NULL DEFAULT 'running',
                    expected_jobs  INTEGER NOT NULL DEFAULT 0,
                    singular_dims  TEXT NOT NULL DEFAULT '[]',
                    max_concurrent INTEGER NOT NULL DEFAULT 0,
                    skip_rules     TEXT NOT NULL DEFAULT '{}'
                )
            """)
            await db.commit()
            await init_db(db)
            async with db.execute("PRAGMA table_info(experiments)") as cur:
                cols = {r["name"] for r in await cur.fetchall()}
            assert {"metric", "goal"} <= cols
            exp = await create_experiment(db, experiment_id="e1", name="e1",
                                          metric="acc", goal="maximize")
            assert (exp.metric, exp.goal) == ("acc", "maximize")
            await init_db(db)  # idempotent
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
            job = await get_job(db, "run1", "exp1")
            await insert_metric(db, job.job_key, job.attempt, 1, {"x": 1})
            await insert_log(db, job.job_key, job.attempt, 4, "log\n")

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

            await insert_job_nodes(db, "r", "exp", [(0, "w0", [0]), (1, "w1", [0])])
            assert [n.worker_id for n in await list_job_nodes(db, "r", "exp")] == ["w0", "w1"]

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
            assert await list_job_nodes(db, "r", "exp") == []
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


def test_cancel_jobs_leaves_finished_jobs_alone():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1", command=["echo"])
            await insert_job(db, run_id="r2", experiment_id="exp1", command=["echo"], status="done")
            cancelled = await cancel_jobs(db, [("exp1", "r1"), ("exp1", "r2")])
            assert [j.run_id for j in cancelled] == ["r1"]
            assert (await get_job(db, "r2", "exp1")).status == "done"
            assert await cancel_jobs(db, [("exp1", "r1")]) == []
        finally:
            await db.close()

    asyncio.run(run())


def test_requeue_only_touches_jobs_in_flight():
    async def run():
        db = await _init_db()
        try:
            await _dispatched(db, "r1")
            await insert_job(db, run_id="r2", experiment_id="exp1", command=["echo"], status="cancelled")
            requeued, failed = await requeue_jobs(db, [("exp1", "r1"), ("exp1", "r2")], spend_retry=False)
            assert [j.run_id for j in requeued] == ["r1"] and failed == []
            assert requeued[0].status == "pending" and requeued[0].retry_count == 0
            assert (await get_job(db, "r2", "exp1")).status == "cancelled"
            assert await list_active_jobs(db) == []
        finally:
            await db.close()

    asyncio.run(run())


def test_requeue_of_a_lost_run_spends_retries_then_fails():
    async def run():
        db = await _init_db()
        try:
            await _dispatched(db, "r1")
            for expected_retries in (1, 2):
                requeued, _ = await requeue_jobs(db, [("exp1", "r1")], spend_retry=True)
                assert requeued[0].retry_count == expected_retries
                await dispatch_job(db, "r1", "exp1", "w1", [0])
            requeued, failed = await requeue_jobs(db, [("exp1", "r1")], spend_retry=True)
            assert requeued == [] and [j.status for j in failed] == ["failed"]
        finally:
            await db.close()

    asyncio.run(run())


def test_manual_retry_restores_the_automatic_retry_budget():
    async def run():
        db = await _init_db()
        try:
            await _dispatched(db, "r1")
            for _ in range(3):  # two retries spent, then the third loss fails it
                await requeue_jobs(db, [("exp1", "r1")], spend_retry=True)
                await dispatch_job(db, "r1", "exp1", "w1", [0])
            assert (await get_job(db, "r1", "exp1")).status == "failed"

            assert (await retry_job(db, "r1", "exp1")).retry_count == 0
            job = await dispatch_job(db, "r1", "exp1", "w1", [0])
            assert job.attempt == 4
            requeued, failed = await requeue_jobs(db, [("exp1", "r1")], spend_retry=True)
            assert [j.retry_count for j in requeued] == [1] and failed == []
        finally:
            await db.close()

    asyncio.run(run())


def test_result_rules_apply_monotonic_and_singular():
    async def run():
        db = await _init_db()
        try:
            rules = {
                "bs": {"monotonic": "increasing", "singular": False, "_values": [8, 16, 32]},
                "lr": {"monotonic": None, "singular": False, "_values": [1, 2]},
            }
            await create_experiment(db, experiment_id="m", name="t", skip_rules=rules)
            for bs in (8, 16, 32):
                for lr in (1, 2):
                    await insert_job(db, run_id=f"b{bs}_l{lr}", experiment_id="m",
                                     command=["x"], combo={"bs": bs, "lr": lr})
            await update_job_status(db, "b8_l1", "m", "done", exit_code=0)
            assert await apply_result_rules(db, "m", "b8_l1", True) == []
            await update_job_status(db, "b16_l1", "m", "failed", exit_code=1)
            # 32 is worse than the failed 16 at the same lr; lr=2 is unaffected.
            assert await apply_result_rules(db, "m", "b16_l1", False) == ["b32_l1"]
            assert (await get_job(db, "b32_l1", "m")).status == "xfailed"
            assert (await get_job(db, "b32_l2", "m")).status == "pending"

            await create_experiment(db, experiment_id="s", name="t", skip_rules={
                "bs": {"monotonic": None, "singular": True, "_values": [8, 16, 32]},
            })
            for bs in (8, 16, 32):
                await insert_job(db, run_id=f"b{bs}", experiment_id="s", command=["x"], combo={"bs": bs})
            await update_job_status(db, "b8", "s", "done", exit_code=0)
            assert sorted(await apply_result_rules(db, "s", "b8", True)) == ["b16", "b32"]

            # With only singular_dims set (as Bayesian sweeps do), a success turns
            # failed probes of the same non-singular combo into xfailed.
            await create_experiment(db, experiment_id="b", name="t", singular_dims=["bs"])
            for bs, status in ((8, "failed"), (16, "done")):
                await insert_job(db, run_id=f"b{bs}", experiment_id="b", command=["x"],
                                 combo={"bs": bs}, status=status)
            assert await apply_result_rules(db, "b", "b16", True) == ["b8"]
        finally:
            await db.close()

    asyncio.run(run())


def test_retry_job():
    async def run():
        db = await _init_db()
        try:
            await create_experiment(db, experiment_id="exp1", name="t")
            await insert_job(db, run_id="r1", experiment_id="exp1",
                             command=["echo"], status="pending")
            assert await retry_job(db, "r1", "exp1") is None  # not finished
            await update_job_status(db, "r1", "exp1", "failed")
            await update_experiment_status(db, "exp1", "completed")
            for _ in range(4):  # max_retries caps automatic retries only
                job = await retry_job(db, "r1", "exp1")
                assert (job.status, job.retry_count) == ("pending", 0)
                await update_job_status(db, "r1", "exp1", "failed")
            assert (await get_experiment(db, "exp1")).status == "running"
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
