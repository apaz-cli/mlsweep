"""Regressions for the 2026-11-28 cleanup: history written with the change it
describes, removed workers hidden, trace reads by prefix and kind."""

import asyncio

import aiosqlite

from mlsweep import _manager_db as mdb


async def _db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await mdb.init_db(db)
    return db


def _run(coro_fn):
    async def main():
        db = await _db()
        try:
            await coro_fn(db)
        finally:
            await db.close()
    asyncio.run(main())


async def _workers(db, *ids):
    for wid in ids:
        await mdb.upsert_worker(db, worker_id=wid, host="h", remote_dir="/r")


async def _history(db, eid):
    return [(e["run_id"], e["action"], e["source"], e["detail"])
            for e in await mdb.list_run_events(db, eid, None, 100)]


def test_cancel_records_only_what_it_cancelled_with_the_status_it_was_in():
    async def body(db):
        await mdb.create_experiment(db, experiment_id="e", name="e", campaign="c")
        await _workers(db, "w1")
        for rid in ("queued", "busy", "over"):
            await mdb.insert_job(db, run_id=rid, experiment_id="e", command=["true"])
        await mdb.dispatch_job(db, "busy", "e", "w1")
        await mdb.dispatch_job(db, "over", "e", "w1")
        await mdb.finish_job(db, "over", "e", success=True, exit_code=0, elapsed=1.0)

        cancelled = await mdb.cancel_jobs(db, [("e", "queued"), ("e", "busy"), ("e", "over")], source="cli")
        assert {j.run_id for j in cancelled} == {"queued", "busy"}
        # The finished run has no "cancelled" entry: it wasn't cancelled.
        assert sorted(h for h in await _history(db, "e") if h[1] == "cancelled") == [
            ("busy", "cancelled", "cli", "while dispatched"),
            ("queued", "cancelled", "cli", "while pending"),
        ]

        await mdb.insert_job(db, run_id="later", experiment_id="e", command=["true"])
        await mdb.cancel_jobs(db, [("e", "later")], why="the experiment was stopped")
        assert ("later", "cancelled", "manager", "the experiment was stopped") in await _history(db, "e")
    _run(body)


def test_dispatch_and_finish_write_their_history_in_the_same_commit():
    async def body(db):
        writer = mdb.DbWriter(db)
        task = asyncio.create_task(writer.run())
        try:
            await mdb.create_experiment(db, experiment_id="e", name="e", campaign="c")
            await _workers(db, "w1", "w2")
            await mdb.insert_job(db, run_id="r", experiment_id="e", command=["true"], nodes_per_run=2)
            claimed = await writer.dispatch_job(
                "r", "e", "w1", [0], nodes=[(0, "w1", [0]), (1, "w2", [0])],
                history=lambda c: mdb.RunEvent("e", "r", "dispatched", "manager", "to w1", c.attempt))
            assert claimed is not None and claimed.attempt == 1
            nodes = await mdb._exec_all(db, "SELECT worker_id FROM job_nodes WHERE run_id = 'r' ORDER BY node_rank")
            assert [n[0] for n in nodes] == ["w1", "w2"]
            # A job no longer pending is not claimed, and gets no entry.
            assert await writer.dispatch_job(
                "r", "e", "w1", [0], history=lambda c: mdb.RunEvent("e", "r", "dispatched", "manager")) is None
            await writer.finish_job("r", "e", success=False, exit_code=3, elapsed=2.0,
                                    history=lambda j: mdb.RunEvent("e", "r", "finished", "manager", j.status))
            assert [(a, d) for _, a, _, d in await _history(db, "e")] == [
                ("dispatched", "to w1"), ("finished", "failed")]
        finally:
            task.cancel()
    _run(body)


def test_a_removed_worker_is_gone_unless_asked_for():
    async def body(db):
        await mdb.upsert_worker(db, worker_id="w", host="h", remote_dir="/r", status="removed")
        assert await mdb.get_worker(db, "w") is None
        assert (await mdb.get_worker(db, "w", include_removed=True)).status == "removed"
    _run(body)


def test_trace_events_by_prefix_and_kind():
    async def body(db):
        sid = "9f6c0e22-aaaa-4000-8000-000000000001"
        events = [
            {"seq": 1, "ts": 1.0, "kind": "user", "text": "hi"},
            {"seq": 2, "ts": 2.0, "kind": "thinking", "text": "hmm"},
            {"seq": 3, "ts": 3.0, "kind": "tool", "name": "Bash", "tool_id": "t", "text": "ls"},
            {"seq": 4, "ts": 4.0, "kind": "tool_result", "tool_id": "t", "ok": True},
            {"seq": 5, "ts": 5.0, "kind": "assistant", "text": "done"},
        ]
        await mdb.store_capture_batch(db, {"session_id": sid, "agent": "pi", "state": "active"}, events)
        found, error = await mdb.find_capture_session(db, sid[:10])
        assert error is None and found["session_id"] == sid
        assert (await mdb.find_capture_session(db, "9f6c0e23"))[0] is None

        from mlsweep._trace_view import stored_kinds
        tools = await mdb.get_capture_events(db, sid, kinds=stored_kinds("tools"))
        assert [e["seq"] for e in tools] == [3, 4]   # the result comes with its call
        users = await mdb.get_capture_events(db, sid, kinds=stored_kinds("user"))
        assert [e["seq"] for e in users] == [1]
        assert len(await mdb.get_capture_events(db, sid)) == 5
    _run(body)


def test_listing_traces_by_search_works_with_or_without_fts5():
    async def body(db):
        sid = "9f6c0e22-aaaa-4000-8000-000000000002"
        await mdb.store_capture_batch(
            db, {"session_id": sid, "agent": "pi", "state": "active", "title": "tune the optimizer"},
            [{"seq": 1, "ts": 1.0, "kind": "user", "text": "tune the optimizer please"}])
        got = await mdb.list_capture_sessions(db, q="optimizer")
        assert [s["session_id"] for s in got] == [sid]
    _run(body)
