"""Pausing the scheduler, everywhere or on one worker (``mlsweep scheduler --pause``).

A pause stops new jobs from starting so someone can place runs by hand.
Running jobs carry on, and a job pinned while paused waits for the unpause.
Each test drives a real manager and workers through cluster_harness.
"""

import json

import pytest

from cluster_harness import Cluster, stays, wait_until


@pytest.fixture
def cluster_factory(tmp_path):
    made = []

    def make(workers, sub="c"):
        d = tmp_path / sub
        d.mkdir()
        c = Cluster(d, workers)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def test_pausing_everywhere_starts_nothing_but_lets_running_jobs_finish(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("pause_all")
    c.submit(eid, "running", 1.5)
    wait_until(lambda: c.spans(eid), 10, "the first job to start")
    assert c.post("/api/scheduler/pause")["paused"] is True
    c.submit_many(eid, ["a", "b"], 0.1)
    c.wait_statuses(eid, {"done": 1, "pending": 2})  # the running job was not stopped
    stays(lambda: c.statuses(eid)["pending"] == 2, 1.0, "queued jobs to stay pending")

    settings = c.post("/api/scheduler/unpause")
    assert settings["paused"] is False and settings["paused_workers"] == []
    c.wait_statuses(eid, {"done": 3})


def test_a_paused_worker_gets_no_new_jobs(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}, {"devices": [0], "jobs": 0}])
    paused, other = c.worker_ids
    c.post("/api/scheduler/pause", {"worker_id": paused})
    assert c.get("/api/scheduler")["paused_workers"] == [paused]
    assert {w["worker_id"]: w["paused"] for w in c.get("/api/workers")} == {paused: True, other: False}

    eid = c.experiment("pause_one")
    c.submit_many(eid, [f"r{i}" for i in range(6)], 0.1)
    c.wait_statuses(eid, {"done": 6})
    assert {j["worker_id"] for j in c.jobs(eid).values()} == {other}


def test_a_job_pinned_to_a_paused_worker_waits_for_it(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}, {"devices": [0], "jobs": 0}])
    paused, _ = c.worker_ids
    c.post("/api/scheduler/pause")
    c.post("/api/scheduler/pause", {"worker_id": paused})
    eid = c.experiment("pin")
    c.submit(eid, "r", 0.1)
    c.post(f"/api/jobs/r/reschedule?experiment_id={eid}", {"worker_id": paused})

    # Unpausing one worker leaves the cluster-wide pause in force.
    assert c.post("/api/scheduler/unpause", {"worker_id": paused})["paused"] is True
    c.post("/api/scheduler/pause", {"worker_id": paused})
    c.post("/api/scheduler/unpause")  # lifts both pauses at once
    c.wait_statuses(eid, {"done": 1})
    assert c.job(eid, "r")["worker_id"] == paused


def test_a_pin_waits_on_a_paused_worker_while_the_others_are_free(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}, {"devices": [0], "jobs": 0}])
    paused, _ = c.worker_ids
    c.post("/api/scheduler/pause", {"worker_id": paused})
    eid = c.experiment("pin_one")
    c.set_exp_status(eid, "paused")  # hold the job until it is pinned
    c.submit(eid, "r", 0.1)
    c.post(f"/api/jobs/r/reschedule?experiment_id={eid}", {"worker_id": paused})
    c.set_exp_status(eid, "running")
    stays(lambda: c.job(eid, "r")["status"] == "pending", 1.0, "the pinned job to wait")
    c.post("/api/scheduler/unpause", {"worker_id": paused})
    c.wait_statuses(eid, {"done": 1})
    assert c.job(eid, "r")["worker_id"] == paused


def test_pauses_survive_a_manager_restart(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    (wid,) = c.worker_ids
    c.post("/api/scheduler/pause")
    c.post("/api/scheduler/pause", {"worker_id": wid})
    c.kill_manager()
    c.start_manager()
    assert c.get("/api/scheduler") == {"placement": "pack", "paused": True, "paused_workers": [wid]}
    eid = c.experiment("restart")
    c.submit(eid, "r", 0.1)
    stays(lambda: c.job(eid, "r")["status"] == "pending", 1.0, "the job to stay pending")
    c.post("/api/scheduler/unpause")
    c.wait_statuses(eid, {"done": 1})


def test_pausing_an_unknown_worker_is_refused(cluster_factory):
    c = cluster_factory([{"devices": [0]}])
    assert c.status_code("POST", "/api/scheduler/pause", {"worker_id": "nope"}) == 404
    assert c.status_code("POST", "/api/scheduler/pause", {"worker_id": 3}) == 400
    assert c.get("/api/scheduler")["paused_workers"] == []


def test_the_cli_pauses_and_unpauses(cluster_factory, capsys):
    from cluster_harness import TOKEN
    from mlsweep.cli import main

    c = cluster_factory([{"devices": [0]}])
    (wid,) = c.worker_ids
    conn = ["--manager", c.url, "--token", TOKEN]

    def scheduler(*flags):
        main(["scheduler", *flags, *conn])
        return capsys.readouterr().out

    assert "paused everywhere" in scheduler("--pause")
    assert c.get("/api/scheduler")["paused"] is True
    out = scheduler("--pause", wid)
    assert f"paused workers: {wid}" in out
    scheduler("--unpause", wid)
    assert c.get("/api/scheduler") == {"placement": "pack", "paused": True, "paused_workers": []}
    scheduler("--pause", wid, "--placement", "spread")
    assert json.loads(scheduler("--json")) == {"placement": "spread", "paused": True, "paused_workers": [wid]}
    scheduler("--unpause")
    assert c.get("/api/scheduler") == {"placement": "spread", "paused": False, "paused_workers": []}
    assert "scheduling: running" in scheduler()
    with pytest.raises(SystemExit):
        scheduler("--pause", "nope")
