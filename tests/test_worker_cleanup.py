"""Unit tests for the worker's MsgCleanup handler."""

import os
import queue

from mlsweep import worker
from mlsweep._shared import MsgCleanup, MsgCleaned, decode


class _FakeConn:
    def __init__(self):
        self.send_queue = queue.Queue()
        self.closed = False


def _make_run_dir(scratch: str, experiment: str, run_id: str) -> str:
    d = os.path.join(scratch, experiment, run_id)
    os.makedirs(os.path.join(d, "artifacts"))
    open(os.path.join(d, "training.log"), "w").close()
    return d


def test_cleanup_final_deletes_scratch(tmp_path, monkeypatch):
    scratch = str(tmp_path / "scratch")
    monkeypatch.setattr(worker, "_scratch_dir", scratch)
    run_dir = _make_run_dir(scratch, "exp_1", "run_1")
    conn = _FakeConn()

    worker._handle_cleanup(
        MsgCleanup(run_id="run_1", experiment="exp_1", final=True), conn
    )

    assert not os.path.exists(run_dir)
    ack = decode(conn.send_queue.get_nowait()[4:])
    assert isinstance(ack, MsgCleaned) and ack.run_id == "run_1"


def test_cleanup_midrun_keeps_scratch(tmp_path, monkeypatch):
    scratch = str(tmp_path / "scratch")
    monkeypatch.setattr(worker, "_scratch_dir", scratch)
    run_dir = _make_run_dir(scratch, "exp_1", "run_1")
    conn = _FakeConn()

    worker._handle_cleanup(
        MsgCleanup(run_id="run_1", experiment="exp_1", final=False), conn
    )

    assert os.path.isdir(run_dir)
    ack = decode(conn.send_queue.get_nowait()[4:])
    assert isinstance(ack, MsgCleaned)


def test_cleanup_rejects_traversal(tmp_path, monkeypatch):
    scratch = str(tmp_path / "scratch")
    os.makedirs(scratch)
    outside = str(tmp_path / "outside")
    os.makedirs(outside)
    monkeypatch.setattr(worker, "_scratch_dir", scratch)
    conn = _FakeConn()

    worker._handle_cleanup(
        MsgCleanup(run_id="run_1", experiment="../outside", final=True), conn
    )

    assert os.path.isdir(outside)
    ack = decode(conn.send_queue.get_nowait()[4:])
    assert isinstance(ack, MsgCleaned)
