"""Tests for how a multi-node node's result is synced and cleaned up.

Each test drives ``_handle_result`` against a real SQLite database and real
localhost workers' scratch directories, which the manager copies on sync.
"""

import asyncio
import os

import aiosqlite

from mlsweep import _manager_workers as mw
from mlsweep._manager_db import (
    DbWriter,
    create_experiment,
    dispatch_job,
    get_job,
    init_db,
    insert_job,
    insert_job_nodes,
    upsert_worker,
)
from mlsweep._manager_state import InFlightRun, ManagerState, WorkerConn
from mlsweep._shared import MsgCancel, MsgCleanup, MsgResult, decode


def _sent(wc):
    out = []
    while not wc.send_queue.empty():
        out.append(decode(wc.send_queue.get_nowait()[4:]))
    return out


def _result(success=True):
    return MsgResult(run_id="run1", experiment="exp1", success=success, elapsed=1.0,
                     exit_code=0 if success else 1)


def _node_result(tmp_path, *, success=True, in_flight=True, before=None):
    """Run exp1/run1 on w0 (rank 0) and w1 (rank 1), then deliver w1's result.

    Each worker's scratch holds the run's training.log and an artifact.
    *before(out_dir)* runs just before the result is handled.
    Returns ``(state, w0, w1, out_dir)``.
    """
    out_dir = tmp_path / "out"

    async def body():
        db = await aiosqlite.connect(tmp_path / "manager.db")
        await init_db(db)
        writer = DbWriter(db)
        writer_task = asyncio.create_task(writer.run())
        try:
            for wid in ("w0", "w1"):
                await upsert_worker(db, worker_id=wid, host="localhost", remote_dir=str(tmp_path))
            await create_experiment(db, experiment_id="exp1", name="exp1")
            await insert_job(db, run_id="run1", experiment_id="exp1", command=["true"],
                             nodes_per_run=2)
            await dispatch_job(db, "run1", "exp1", "w0", [0])
            await insert_job_nodes(db, "run1", "exp1", [(0, "w0", [0]), (1, "w1", [0])])
            job = await get_job(db, "run1", "exp1")

            state = ManagerState(output_dir=str(out_dir))
            state.db_writer = writer
            workers = {}
            for wid in ("w0", "w1"):
                scratch = tmp_path / f"scratch_{wid}"
                run_dir = scratch / "exp1" / "run1"
                (run_dir / "artifacts").mkdir(parents=True)
                (run_dir / "artifacts" / "model.pt").write_text(wid)
                (run_dir / "training.log").write_text(f"{wid} log\n")
                workers[wid] = WorkerConn(worker_id=wid, host="localhost", port=0,
                                          scratch_dir=str(scratch))
            state.workers = workers
            if in_flight:
                state.runs[("exp1", "run1")] = InFlightRun.from_job(
                    job, "w0", nodes={"w0": [0], "w1": [0]})
            if before is not None:
                before(out_dir)
            await mw._handle_result(db, state, workers["w1"], _result(success))
            return state, workers["w0"], workers["w1"], out_dir
        finally:
            writer_task.cancel()
            await db.close()

    return asyncio.run(body())


def test_node_result_syncs_into_its_rank_dir_and_cleans(tmp_path):
    state, w0, w1, out_dir = _node_result(tmp_path)

    node_dir = out_dir / "exp1" / "run1" / "node1"
    assert (node_dir / "training.log").read_text() == "w1 log\n"
    assert (node_dir / "artifacts" / "model.pt").read_text() == "w1"
    [cleanup] = _sent(w1)
    assert isinstance(cleanup, MsgCleanup)
    assert (cleanup.final, cleanup.experiment, cleanup.run_id) == (True, "exp1", "run1")
    # Its GPUs are free; the run is still in flight on the other node.
    assert state.runs[("exp1", "run1")].nodes == {"w0": [0]}


def test_node_result_skips_cleanup_when_sync_fails(tmp_path):
    def block_artifacts(out_dir):
        # A file where the artifacts directory must go makes the copy fail.
        node_dir = out_dir / "exp1" / "run1" / "node1"
        node_dir.mkdir(parents=True)
        (node_dir / "artifacts").write_text("in the way")

    _, _, w1, _ = _node_result(tmp_path, before=block_artifacts)
    # Sync failed, so the worker must NOT be told to delete its scratch.
    assert _sent(w1) == []


def test_failed_node_stops_its_peers(tmp_path):
    _, w0, _, _ = _node_result(tmp_path, success=False)
    assert [type(m) for m in _sent(w0)] == [MsgCancel]


def test_result_for_a_run_not_in_flight_is_only_acknowledged(tmp_path):
    _, _, w1, out_dir = _node_result(tmp_path, in_flight=False)
    [ack] = _sent(w1)
    assert isinstance(ack, MsgCleanup) and ack.final is False
    assert not os.path.exists(out_dir)  # nothing synced
