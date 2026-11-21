"""Taking GPUs out of the pool and moving runs by hand (the System page's GPU tiles).

Draining a GPU leaves its runs alone but places nothing new there.  Rescheduling
a run restarts it, optionally pinned to a worker's GPUs, and a pin may exceed
the worker's jobs-per-GPU cap.  Each test drives a real manager and workers
(see cluster_harness.py) and checks from the job journal what executed.
"""

import time

import pytest

from cluster_harness import Cluster, wait_until


@pytest.fixture
def cluster(tmp_path):
    made = []

    def make(workers):
        c = Cluster(tmp_path, workers)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def _running(c, eid, rid):
    return c.job(eid, rid)["status"] == "running"


def test_draining_a_gpu_keeps_its_run_and_places_nothing_new(cluster):
    c = cluster([{"devices": [0], "jobs": 0}])
    wid = c.worker_ids[0]
    eid = c.experiment("drain")
    c.submit(eid, "long", 4, gpus=1)
    wait_until(lambda: _running(c, eid, "long"), 15, "long to start")

    resp = c.patch(f"/api/workers/{wid}/devices", {"drain": [0]})
    assert resp["gpus"] == [] and resp["draining"] == ["long"] and resp["evicted"] == []
    worker = next(w for w in c.get("/api/workers") if w["worker_id"] == wid)
    assert [r["run_id"] for r in worker["gpu_runs"]["0"]] == ["long"]

    c.submit(eid, "next", 0.2, gpus=1)
    time.sleep(1.5)
    assert c.job(eid, "next")["status"] == "pending"
    c.wait_statuses(eid, {"done": 1, "pending": 1}, timeout=20)
    assert c.starts(eid) == {"long": 1}   # it was never interrupted

    c.patch(f"/api/workers/{wid}/devices", {"add": [0]})
    c.wait_statuses(eid, {"done": 2}, timeout=20)


def test_reschedule_moves_a_run_to_the_chosen_worker(cluster):
    c = cluster([{"devices": [0], "jobs": 1}, {"devices": [0], "jobs": 1}])
    a, b = c.worker_ids
    eid = c.experiment("move")
    c.submit(eid, "r", 30, gpus=1)
    wait_until(lambda: _running(c, eid, "r"), 15, "r to start")
    here = c.job(eid, "r")["worker_id"]
    there = b if here == a else a

    resp = c.post(f"/api/jobs/r/reschedule?experiment_id={eid}", {"worker_id": there, "gpus": [0]})
    assert resp["requeued"] and resp["pinned"] == {"worker_id": there, "gpus": [0]}
    wait_until(lambda: _running(c, eid, "r") and c.job(eid, "r")["worker_id"] == there, 15,
               "r to restart on the other worker")
    assert c.job(eid, "r")["retry_count"] == 0   # a move is not a retry
    wait_until(lambda: c.starts(eid)["r"] == 2, 10, "the second start to be journaled")
    first, second = sorted(c.spans(eid), key=lambda s: s.start)
    assert first.stop is not None and first.stop <= second.start + 1
    c.cancel(eid, "r")


def test_a_pin_shares_a_gpu_past_the_cap(cluster):
    c = cluster([{"devices": [0], "jobs": 1}])
    wid = c.worker_ids[0]
    eid = c.experiment("share")
    c.submit(eid, "a", 30, gpus=1)
    wait_until(lambda: _running(c, eid, "a"), 15, "a to start")
    c.submit(eid, "b", 30, gpus=1)
    time.sleep(1)
    assert c.job(eid, "b")["status"] == "pending"   # the cap holds b back

    c.post(f"/api/jobs/b/reschedule?experiment_id={eid}", {"worker_id": wid})
    wait_until(lambda: _running(c, eid, "b"), 15, "b to start beside a")
    assert c.starts(eid) == {"a": 1, "b": 1}
    for rid in "ab":
        c.cancel(eid, rid)


def test_reschedule_refuses_gpus_outside_the_pool(cluster):
    c = cluster([{"devices": [0], "jobs": 1}])
    wid = c.worker_ids[0]
    eid = c.experiment("refuse")
    c.submit(eid, "r", 0.2, gpus=1)
    c.patch(f"/api/workers/{wid}/devices", {"drain": [0]})
    assert c.status_code("POST", f"/api/jobs/r/reschedule?experiment_id={eid}",
                         {"worker_id": wid, "gpus": [0]}) == 400
    assert c.status_code("POST", f"/api/jobs/r/reschedule?experiment_id={eid}",
                         {"worker_id": "nobody:1"}) == 409
    c.patch(f"/api/workers/{wid}/devices", {"add": [0]})
    c.wait_statuses(eid, {"done": 1}, timeout=20)
    assert c.status_code("POST", f"/api/jobs/r/reschedule?experiment_id={eid}", {}) == 409


def test_a_pin_that_no_longer_fits_is_dropped():
    from mlsweep._manager_db import SchedulableJob
    from mlsweep._manager_state import ManagerState, Pin, WorkerConn
    from mlsweep._manager_workers import _pinned_placement

    state = ManagerState()
    wc = WorkerConn(worker_id="w", host="h", port=0, status="connected", gpus=[0, 1])
    state.workers["w"] = wc  # type: ignore[assignment]
    job = SchedulableJob("e", "r", gpus_per_run=1, nodes_per_run=1)
    assert _pinned_placement(job, Pin("w", [1]), state) == [(wc, [1])]
    wc.gpus = [0]                                   # GPU 1 left the pool
    assert _pinned_placement(job, Pin("w", [1]), state) is None
    wc.status = "reconnecting"
    assert _pinned_placement(job, Pin("w", [0]), state) is None
    assert _pinned_placement(job, Pin("gone", [0]), state) is None
