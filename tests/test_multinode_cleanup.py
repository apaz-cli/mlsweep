"""Unit tests for multi-node scratch sync + cleanup."""

import asyncio
from types import SimpleNamespace

from mlsweep import _manager_workers as mw
from mlsweep._shared import MsgCleanup, decode


def _wc(worker_id="w1"):
    return SimpleNamespace(
        scratch_dir="/scratch",
        host="localhost",
        password=None,
        ssh_key=None,
        worker_id=worker_id,
    )


def _state(output_dir="/out"):
    return SimpleNamespace(output_dir=output_dir)


def _run(coro):
    return asyncio.run(coro)


def test_sync_node_scratch_syncs_per_rank_and_cleans(monkeypatch):
    captured = {}

    async def fake_list_job_nodes(db, run_id, experiment_id):
        return [
            SimpleNamespace(node_rank=0, worker_id="w0"),
            SimpleNamespace(node_rank=1, worker_id="w1"),
        ]

    def fake_rsync_sync(worker_host, remote_scratch, local_run_dir, run_id,
                        password=None, ssh_key=None):
        captured["remote_scratch"] = remote_scratch
        captured["local_run_dir"] = local_run_dir
        return True

    sent = []

    async def fake_send(wc, data):
        sent.append(decode(data[4:]))

    monkeypatch.setattr(mw, "list_job_nodes", fake_list_job_nodes)
    monkeypatch.setattr(mw, "_rsync_sync", fake_rsync_sync)
    monkeypatch.setattr(mw, "_send_to_worker", fake_send)

    _run(mw._sync_node_scratch(None, _state(), _wc("w1"), "run1", "exp1"))

    assert captured["remote_scratch"] == "/scratch/exp1/run1"
    assert captured["local_run_dir"] == "/out/exp1/run1/node1"
    assert len(sent) == 1
    assert isinstance(sent[0], MsgCleanup)
    assert sent[0].final is True
    assert sent[0].experiment == "exp1"
    assert sent[0].run_id == "run1"


def test_sync_node_scratch_skips_cleanup_on_sync_failure(monkeypatch):
    async def fake_list_job_nodes(db, run_id, experiment_id):
        return [SimpleNamespace(node_rank=0, worker_id="w0")]

    def fake_rsync_sync(*args, **kwargs):
        return False

    sent = []

    async def fake_send(wc, data):
        sent.append(decode(data[4:]))

    monkeypatch.setattr(mw, "list_job_nodes", fake_list_job_nodes)
    monkeypatch.setattr(mw, "_rsync_sync", fake_rsync_sync)
    monkeypatch.setattr(mw, "_send_to_worker", fake_send)

    _run(mw._sync_node_scratch(None, _state(), _wc("w0"), "run1", "exp1"))

    # Sync failed, so the worker must NOT be told to delete its scratch.
    assert sent == []
