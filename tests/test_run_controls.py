"""MAX_RUNTIME, PRUNE and METRIC_CHECK enforced by a real manager on real runs."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cluster_harness import TOKEN, Cluster, wait_until

# Logs --key with each of --values at steps 1, 2, ..., then sleeps --sleep seconds.
# On SIGTERM it saves a checkpoint to its artifacts dir, as a training script would.
LOGS = """
import argparse, os, signal, sys, time
from mlsweep.logger import MLSweepLogger
def save(*_):
    with open(os.path.join(os.environ["MLSWEEP_RUN_DIR"], "ckpt.txt"), "w") as f:
        f.write("saved")
    sys.exit(143)
signal.signal(signal.SIGTERM, save)
p = argparse.ArgumentParser()
p.add_argument("--key", default="loss")
p.add_argument("--values", default="")
p.add_argument("--sleep", type=float, default=0.0)
a = p.parse_args()
with MLSweepLogger() as logger:
    for step, v in enumerate(filter(None, a.values.split(",")), start=1):
        logger.log({a.key: float(v)}, step=step)
        logger.sync()
    time.sleep(a.sleep)
"""


@pytest.fixture
def cluster(tmp_path):
    c = Cluster(tmp_path, [{"devices": [0], "jobs": 0}])
    yield c
    c.close()


def _job(eid: str, rid: str, *args: str) -> dict:
    return {"run_id": rid, "experiment_id": eid, "gpus_per_run": 0,
            "command": [sys.executable, "-c", LOGS, *args]}


def _checkpoint(c: Cluster, eid: str, rid: str):
    """The checkpoint the run saved on SIGTERM, as synced to the manager."""
    return c.tmp / "out" / "experiments" / eid / rid / "artifacts" / "ckpt.txt"


def _wait_job(c: Cluster, eid: str, rid: str, status: str, timeout: float = 60) -> dict:
    wait_until(lambda: c.job(eid, rid)["status"] == status, timeout,
               f"{rid} to be {status} (now {c.job(eid, rid)['status']})")
    return c.job(eid, rid)


def test_a_run_past_max_runtime_is_stopped_and_failed(cluster):
    eid = cluster.experiment("timeout", controls={"max_runtime": 2})
    cluster.post("/api/jobs/bulk", [_job(eid, "slow", "--sleep", "120"), _job(eid, "quick")])
    slow = _wait_job(cluster, eid, "slow", "failed", timeout=30)
    assert slow["reason"] == "timed out: ran longer than MAX_RUNTIME (2s)"
    assert 2 <= slow["elapsed"] < 30
    assert slow["exit_code"] == 143  # it got SIGTERM and exited by itself
    assert _checkpoint(cluster, eid, "slow").read_text() == "saved"
    assert cluster.job(eid, "quick")["status"] == "done"
    assert not cluster.live_pids(eid)


def test_a_run_behind_the_others_is_pruned_and_keeps_its_metrics(cluster):
    prune = {"metric": "loss", "goal": "minimize", "strategy": "median", "min_steps": 2, "every": 2,
             "n_warmup": 2}
    eid = cluster.experiment("prune", controls={"prune": prune})
    cluster.post("/api/jobs/bulk", [_job(eid, f"good{i}", "--values", "1.0,0.5,0.4") for i in range(2)])
    wait_until(lambda: cluster.statuses(eid)["done"] == 2, 60, "the good runs to finish")
    cluster.post("/api/jobs", _job(eid, "bad", "--values", "3.0,2.0", "--sleep", "120"))
    bad = _wait_job(cluster, eid, "bad", "pruned")
    assert bad["reason"].startswith("pruned at step 2: best loss 2 is worse than the median 0.5")
    rows = [json.loads(line) for line in
            cluster.get_text(f"/api/experiments/{eid}/jobs/bad/metrics").splitlines() if line]
    assert [r["loss"] for r in rows] == [3.0, 2.0]
    assert _checkpoint(cluster, eid, "bad").read_text() == "saved"
    # A pruned run counts as finished: the experiment completes.
    wait_until(lambda: cluster.get(f"/api/experiments/{eid}")["status"] == "completed", 30, "completion")


def test_runs_already_behind_when_the_manager_restarts_are_pruned(cluster):
    prune = {"metric": "loss", "goal": "minimize", "strategy": "median", "min_steps": 1, "every": 1,
             "n_warmup": 1}
    eid = cluster.experiment("prune_restart", controls={"prune": prune})
    cluster.post("/api/jobs", _job(eid, "good", "--values", "0.1"))
    _wait_job(cluster, eid, "good", "done")
    cluster.kill_manager()
    cluster.start_manager()
    cluster.post("/api/jobs", _job(eid, "bad", "--values", "5.0", "--sleep", "120"))
    assert _wait_job(cluster, eid, "bad", "pruned")["reason"].startswith("pruned at step 1")


def test_metric_check_pauses_a_sweep_whose_runs_log_another_name(cluster):
    eid = cluster.experiment("mcheck", max_concurrent=1, controls={"metric_check": {"metric": "val_loss", "runs": 2}})
    cluster.post("/api/jobs/bulk", [_job(eid, f"r{i}", "--key", "val/loss", "--values", "0.5") for i in range(4)])
    wait_until(lambda: cluster.get(f"/api/experiments/{eid}")["status"] == "paused", 60, "the pause")
    jobs = cluster.jobs(eid)
    done = [j for j in jobs.values() if j["status"] == "done"]
    assert len(done) == 2
    assert all("did you mean 'val/loss'" in j["reason"] for j in done)
    assert [j["status"] for j in jobs.values()].count("pending") == 2
    note = json.loads(cluster.get(f"/api/experiments/{eid}")["controls"])["note"]
    assert note.startswith("paused by METRIC_CHECK: the first 2 finished run(s) logged no 'val_loss'")
    # The Experiments page shows it from the listing.
    listed = next(e for e in cluster.get("/api/experiments") if e["experiment_id"] == eid)
    assert listed["manager_note"] == note

    # `mlsweep wait` reports the pause instead of waiting forever.
    r = subprocess.run([sys.executable, "-m", "mlsweep", "wait", eid, "--manager", cluster.url,
                        "--token", TOKEN, "--interval", "0.2", "--timeout", "30"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "PAUSED" in r.stdout and "METRIC_CHECK" in r.stdout

    # Unpausing carries on, and the check does not fire again.
    cluster.set_exp_status(eid, "running")
    wait_until(lambda: cluster.statuses(eid)["done"] == 4, 60, "the rest to run")
    exp = cluster.get(f"/api/experiments/{eid}")
    assert "note" not in json.loads(exp["controls"]) and exp["status"] == "completed"


def test_metric_check_stays_quiet_once_a_run_logs_the_metric(cluster):
    eid = cluster.experiment("mcheck_ok", max_concurrent=1,
                             controls={"metric_check": {"metric": "loss", "runs": 2}})
    cluster.post("/api/jobs/bulk", [_job(eid, "a", "--values", "0.5"),
                                    _job(eid, "b", "--key", "other", "--values", "1"),
                                    _job(eid, "c", "--key", "other", "--values", "1")])
    wait_until(lambda: cluster.statuses(eid)["done"] == 3, 60, "all runs")
    assert cluster.get(f"/api/experiments/{eid}")["status"] == "completed"
    assert cluster.job(eid, "a")["reason"] is None
    assert cluster.job(eid, "b")["reason"].startswith("missing metric 'loss'")


def test_bad_controls_are_refused(cluster):
    code = cluster.status_code("POST", "/api/experiments",
                               {"experiment_id": f"bad_{cluster.tag}", "controls": {"max_runtime": "soon"}})
    assert code == 400


def _controls_cli(c: Cluster, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "mlsweep", "controls", *args, "--manager", c.url,
                           "--token", TOKEN, "--campaign", "default"],
                          capture_output=True, text=True, timeout=60)


def test_controls_can_be_changed_while_a_sweep_runs(cluster):
    eid = cluster.experiment("edit", metric="loss", goal="minimize")
    cluster.post("/api/jobs", _job(eid, "slow", "--sleep", "120"))
    _wait_job(cluster, eid, "slow", "running")
    assert cluster.get(f"/api/experiments/{eid}/controls") == {}

    r = _controls_cli(cluster, eid, "--max-runtime", "2s", "--prune", '{"min_steps": 100}', "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    controls = json.loads(r.stdout)
    assert controls["max_runtime"] == 2.0
    # PRUNE's metric and goal come from the experiment.
    assert controls["prune"]["metric"] == "loss" and controls["prune"]["goal"] == "minimize"
    # It applies to the run already going.
    assert _wait_job(cluster, eid, "slow", "failed", timeout=30)["reason"].startswith("timed out")

    r = _controls_cli(cluster, eid, "--max-runtime", "off", "--prune", "off", "--metric-check", "off")
    assert r.returncode == 0 and "no run controls" in r.stdout
    assert cluster.get(f"/api/experiments/{eid}/controls") == {}

    r = _controls_cli(cluster, eid, "--prune", '{"min_step": 10}')
    assert r.returncode == 1 and "unknown key" in r.stdout
    assert cluster.status_code("PATCH", f"/api/experiments/{eid}/controls", {"budget": 3}) == 400
