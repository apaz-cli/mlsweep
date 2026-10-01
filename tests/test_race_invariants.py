"""Invariants of the manager's control plane under concurrent and badly timed actions.

These complement tests/test_races.py.  Each test drives a real manager and
real workers (see cluster_harness.py), does things at awkward moments, and
then checks from the job journal what actually executed:

  * a GPU slot is never double-booked, and an experiment's cap is never exceeded;
  * nothing starts in a paused or aborted experiment, and nothing is stranded
    after the controls are fiddled with;
  * cancel, abort, delete and eviction stop the processes they should (and only
    those), and a cancelled job stays cancelled;
  * retry runs a job exactly once more, however many retries race;
  * worker loss, worker removal and manager crashes neither lose nor duplicate runs.
"""

import os
import random
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cluster_harness import Cluster, max_overlap, stays, wait_until
from test_reconnect import _Proxy, proxied  # noqa: F401  (proxied is a fixture)


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


def _no_live(c, eid, timeout=10):
    wait_until(lambda: not c.live_pids(eid), timeout, f"{eid}'s processes to exit")


# ── Capacity ──────────────────────────────────────────────────────────────────


def test_a_gpu_is_never_double_booked(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 1}, {"devices": [0], "jobs": 1}])
    eid = c.experiment("gpu")
    rids = [f"r{i}" for i in range(10)]
    c.submit_many(eid, rids, 1, gpus=1)
    c.wait_statuses(eid, {"done": 10})

    spans = c.spans(eid)
    assert c.starts(eid) == {r: 1 for r in rids}
    per_slot = max_overlap(spans, key=lambda s: (s.worker, s.gpu))
    assert set(per_slot.values()) == {1}, per_slot
    assert len(per_slot) == 2  # both workers were used
    assert max_overlap(spans)[None] == 2


def test_max_concurrent_is_never_exceeded(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("cap", max_concurrent=2)
    c.submit_many(eid, [f"r{i}" for i in range(10)], 0.7)
    c.wait_statuses(eid, {"done": 10})
    assert max_overlap(c.spans(eid))[None] == 2
    assert set(c.starts(eid).values()) == {1}


def test_lowering_the_cap_mid_run_holds_back_new_starts(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("lower", max_concurrent=3)
    c.submit_many(eid, [f"r{i}" for i in range(9)], 1.5)
    wait_until(lambda: len([s for s in c.spans(eid) if s.stop is None]) == 3, 20, "3 running")
    c.put(f"/api/experiments/{eid}/max_concurrent", {"max_concurrent": 1})
    lowered = time.time()
    c.wait_statuses(eid, {"done": 9}, timeout=90)

    spans = c.spans(eid)
    for s in spans:
        if s.start > lowered:
            others = [o for o in spans if o is not s and o.start <= s.start
                      and (o.stop is None or o.stop > s.start)]
            assert not others, f"{s.run} started beside {[o.run for o in others]}"


def test_pending_jobs_start_in_priority_order(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 1}])
    eid = c.experiment("prio")
    c.set_exp_status(eid, "paused")
    for rid in "abcde":
        c.submit(eid, rid, 0.3, gpus=1)
    for rid, prio in (("c", 10), ("a", 5), ("e", -1)):
        c.patch(f"/api/experiments/{eid}/jobs/{rid}", {"priority": prio})
    time.sleep(1)
    assert not c.spans(eid)  # paused: nothing ran
    c.set_exp_status(eid, "running")
    c.wait_statuses(eid, {"done": 5})
    assert [s.run for s in c.spans(eid)] == ["c", "a", "b", "d", "e"]


def test_new_capacity_is_used_promptly(cluster_factory):
    """Adding a device wakes the scheduler; it does not wait for the periodic pass."""
    c = cluster_factory([{"devices": [0], "jobs": 1}])
    wid = c.worker_ids[0]
    c.patch(f"/api/workers/{wid}/devices", {"remove": [0]})
    eid = c.experiment("wake")
    c.submit(eid, "r", 0.2, gpus=1)
    time.sleep(1)
    assert c.job(eid, "r")["status"] == "pending"
    added = time.time()
    c.patch(f"/api/workers/{wid}/devices", {"add": [0]})
    wait_until(lambda: c.spans(eid), 10, "r to start")
    assert c.spans(eid)[0].start - added < 3


def test_nothing_is_stranded_or_run_twice_after_fiddling(cluster_factory):
    """Hammer every scheduling control from several threads, then restore them:
    every job must still run, exactly once."""
    c = cluster_factory([{"devices": [0], "jobs": 0}, {"devices": [0], "jobs": 0}])
    eid = c.experiment("fiddle", max_concurrent=3)
    rids = [f"r{i}" for i in range(30)]
    c.submit_many(eid, rids, 0.4)
    stop = time.time() + 5
    rng = random.Random(1234)

    def fiddle(seed):
        r = random.Random(seed)
        while time.time() < stop:
            action = r.randrange(6)
            wid = r.choice(c.worker_ids)
            if action == 0:
                c.set_exp_status(eid, r.choice(["paused", "running"]))
            elif action == 1:
                c.put(f"/api/experiments/{eid}/max_concurrent", {"max_concurrent": r.randrange(4)})
            elif action == 2:
                c.patch(f"/api/experiments/{eid}/jobs/{r.choice(rids)}", {"priority": r.randrange(-5, 5)})
            elif action == 3:
                c.patch(f"/api/workers/{wid}/devices", {r.choice(["add", "remove"]): [0]})
            elif action == 4:
                c.patch(f"/api/workers/{wid}/concurrency", {"max_jobs_per_gpu": r.randrange(3)})
            else:
                c.get(f"/api/experiments/{eid}/summary")
            time.sleep(r.random() * 0.05)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(fiddle, [rng.random() for _ in range(6)]))
    c.set_exp_status(eid, "running")
    c.put(f"/api/experiments/{eid}/max_concurrent", {"max_concurrent": 0})
    for wid in c.worker_ids:
        c.patch(f"/api/workers/{wid}/devices", {"add": [0]})
        c.patch(f"/api/workers/{wid}/concurrency", {"max_jobs_per_gpu": 0})

    c.wait_statuses(eid, {"done": 30}, timeout=90)
    assert c.starts(eid) == {r: 1 for r in rids}
    assert c.get(f"/api/experiments/{eid}")["status"] == "completed"


# ── Cancel, pause, abort, delete ──────────────────────────────────────────────


def test_cancel_racing_dispatch_leaves_nothing_running(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("cxl")
    rids = [f"r{i}" for i in range(12)]
    c.submit_many(eid, rids, 30)
    with ThreadPoolExecutor(12) as pool:
        list(pool.map(lambda r: c.cancel(eid, r), rids))
    assert c.statuses(eid) == {"cancelled": 12}
    _no_live(c, eid)
    stays(lambda: c.statuses(eid) == {"cancelled": 12}, 3, "all jobs cancelled")
    assert not c.live_pids(eid)
    assert all(s.how == "term" for s in c.spans(eid))
    assert c.get(f"/api/experiments/{eid}")["status"] == "completed"


def test_cancelling_a_running_job_kills_it_and_it_stays_cancelled(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 1}])
    eid = c.experiment("kill")
    c.submit(eid, "r", 60, gpus=1)
    wait_until(lambda: c.job(eid, "r")["status"] == "running", 20, "r to run")
    # Many concurrent cancels of the same job are all fine and act once.
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda _: c.status_code(
            "POST", f"/api/jobs/r/cancel?experiment_id={eid}"), range(8)))
    assert codes == [200] * 8
    _no_live(c, eid)
    [span] = c.spans(eid)
    assert span.how == "term"
    stays(lambda: c.job(eid, "r")["status"] == "cancelled", 3, "r cancelled")
    assert c.job(eid, "r")["retry_count"] == 0
    # The slot is free again.
    c.submit(eid, "next", 0.2, gpus=1)
    wait_until(lambda: c.job(eid, "next")["status"] == "done", 20, "next to finish")


def test_cancel_only_touches_its_own_experiment(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    a, b = c.experiment("twin_a"), c.experiment("twin_b")
    c.submit(a, "same", 30)
    c.submit(b, "same", 3)
    wait_until(lambda: len([s for s in c.spans() if s.stop is None]) == 2, 20, "both to run")
    c.cancel(a, "same")
    _no_live(c, a)
    wait_until(lambda: c.job(b, "same")["status"] == "done", 20, "b to finish")
    assert c.job(a, "same")["status"] == "cancelled"
    [sb] = c.spans(b)
    assert sb.how == "end"


def test_pause_holds_pending_jobs_and_lets_running_ones_finish(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 1}])
    eid = c.experiment("pause")
    c.submit_many(eid, ["r0", "r1", "r2", "r3"], 1.5, gpus=1)
    wait_until(lambda: c.spans(eid), 20, "first start")
    c.set_exp_status(eid, "paused")
    paused = time.time()
    wait_until(lambda: c.statuses(eid).get("done") == 1, 20, "running job to finish")
    stays(lambda: c.statuses(eid) == {"done": 1, "pending": 3}, 3, "others held")
    assert all(s.start < paused for s in c.spans(eid))
    c.set_exp_status(eid, "running")
    c.wait_statuses(eid, {"done": 4})
    assert set(c.starts(eid).values()) == {1}


def test_experiment_whose_last_job_finished_while_paused_completes_on_resume(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("resume")
    c.submit(eid, "r", 1)
    wait_until(lambda: c.job(eid, "r")["status"] == "running", 20, "r to run")
    c.set_exp_status(eid, "paused")
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 20, "r to finish")
    assert c.get(f"/api/experiments/{eid}")["status"] == "paused"
    c.set_exp_status(eid, "running")
    assert c.get(f"/api/experiments/{eid}")["status"] == "completed"


def test_abort_kills_in_flight_runs_and_starts_nothing_more(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("abort", max_concurrent=2)
    c.submit_many(eid, [f"r{i}" for i in range(5)], 30)
    wait_until(lambda: len(c.live_pids(eid)) == 2, 20, "2 running")
    c.set_exp_status(eid, "aborted")
    aborted = time.time()
    _no_live(c, eid)
    assert c.statuses(eid) == {"cancelled": 2, "pending": 3}
    c.submit(eid, "late", 1)  # added after the abort
    stays(lambda: not c.live_pids(eid), 4, "nothing runs in an aborted experiment")
    assert all(s.start < aborted for s in c.spans(eid))
    assert c.job(eid, "late")["status"] == "pending"


def test_deleting_an_experiment_kills_its_runs_for_good(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid, keep = c.experiment("del"), c.experiment("keep")
    c.submit_many(eid, ["r0", "r1"], 30)
    c.submit(keep, "k", 30)
    wait_until(lambda: len(c.live_pids()) == 3, 20, "3 running")
    c.delete(f"/api/experiments/{eid}")
    _no_live(c, eid)
    time.sleep(2)  # the workers' results for the killed runs arrive meanwhile
    assert c.status_code("GET", f"/api/experiments/{eid}") == 404
    assert c.status_code("GET", f"/api/jobs/r0?experiment_id={eid}") == 404
    assert len(c.live_pids(keep)) == 1
    assert c.job(keep, "k")["status"] == "running"


# ── Retry ─────────────────────────────────────────────────────────────────────


def test_racing_retries_rerun_a_job_exactly_once(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("retry")
    c.submit(eid, "r", 0.3)
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 20, "first attempt")
    path = f"/api/jobs/r/retry?experiment_id={eid}"
    with ThreadPoolExecutor(8) as pool:
        codes = sorted(pool.map(lambda _: c.status_code("POST", path), range(8)))
    assert codes[0] == 200 and codes.count(200) == 1, codes
    assert set(codes[1:]) == {409}
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 20, "second attempt")
    assert c.starts(eid) == {"r": 2}
    job = c.job(eid, "r")
    assert job["retry_count"] == 1
    lines = c.get_text(f"/api/experiments/{eid}/jobs/r/logs").splitlines()
    assert [ln for ln in lines if not ln.startswith("[mlsweep]")] == ["attempt 1", "attempt 2"]


def test_retry_of_a_job_in_flight_is_refused(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("busy")
    c.submit(eid, "r", 3)
    wait_until(lambda: c.job(eid, "r")["status"] == "running", 20, "r to run")
    assert c.status_code("POST", f"/api/jobs/r/retry?experiment_id={eid}") == 409
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 20, "r to finish")
    assert c.starts(eid) == {"r": 1}


def test_jobs_added_to_a_completed_experiment_run(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("reopen")
    c.submit(eid, "first", 0.2)
    wait_until(lambda: c.get(f"/api/experiments/{eid}")["status"] == "completed", 20, "completion")
    c.submit(eid, "second", 0.2)
    wait_until(lambda: c.job(eid, "second")["status"] == "done", 20, "second to run")
    wait_until(lambda: c.get(f"/api/experiments/{eid}")["status"] == "completed", 10,
               "completion again")


# ── Workers come and go ───────────────────────────────────────────────────────


def test_evicted_run_is_requeued_without_a_retry_and_never_overlaps(cluster_factory):
    """Removing the GPU a run is on and adding it straight back: the run is
    stopped, requeued without spending a retry, and runs again only after the
    evicted process is gone."""
    c = cluster_factory([{"devices": [0], "jobs": 1}])
    eid = c.experiment("evict")
    wid = c.worker_ids[0]
    c.submit(eid, "r", 60, 0.5, gpus=1)  # the first attempt is long, the second short
    wait_until(lambda: c.job(eid, "r")["status"] == "running", 20, "r to run")
    out = c.patch(f"/api/workers/{wid}/devices", {"remove": [0]})
    assert out["evicted"] == ["r"]
    c.patch(f"/api/workers/{wid}/devices", {"add": [0]})
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 30, "r to rerun and finish")
    first, second = c.spans(eid)
    assert first.how == "term" and second.how == "end"
    assert second.start >= first.stop
    assert c.job(eid, "r")["retry_count"] == 0


def test_removing_a_worker_moves_its_runs_elsewhere(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 1}, {"devices": [0], "jobs": 1}])
    wa, wb = c.worker_ids
    eid = c.experiment("rm")
    c.patch(f"/api/workers/{wb}/devices", {"remove": [0]})
    c.submit(eid, "r", 60, 0.5, gpus=1)
    wait_until(lambda: c.job(eid, "r")["status"] == "running", 20, "r to run")
    assert c.job(eid, "r")["worker_id"] == wa
    c.patch(f"/api/workers/{wb}/devices", {"add": [0]})
    c.delete(f"/api/workers/{wa}")
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 30, "r to finish on B")
    first, second = c.spans(eid)
    assert first.how == "term"
    assert first.worker != second.worker
    job = c.job(eid, "r")
    assert (job["worker_id"], job["retry_count"]) == (wb, 0)
    wait_until(lambda: not c.worker_pids(c.ports[0]), 15, "removed worker to exit")


def test_a_worker_that_loses_its_runs_gets_them_requeued(cluster_factory):
    """The worker dies with its run and comes back empty on the same port:
    its hello does not report the run, which is requeued (using a retry)."""
    c = cluster_factory([{"devices": [0], "jobs": 1}])
    eid = c.experiment("lost")
    c.submit(eid, "r", 60, 0.5, gpus=1)
    wait_until(lambda: c.job(eid, "r")["status"] == "running", 20, "r to run")
    for pid in c.worker_pids(c.ports[0]) + c.live_pids(eid):
        os.kill(pid, signal.SIGKILL)
    replacement = subprocess.Popen(
        [sys.executable, "-m", "mlsweep.worker", "--port", str(c.ports[0]), "-g", "0",
         "--remote-dir", str(c.proj), "--scratch-dir", str(c.tmp / "scratch2")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_until(lambda: c.job(eid, "r")["status"] == "done", 60, "r to rerun")
        assert c.starts(eid) == {"r": 2}
        assert c.job(eid, "r")["retry_count"] == 1
    finally:
        replacement.terminate()
        replacement.wait()


def test_concurrent_adds_of_the_same_worker_launch_it_once(cluster_factory):
    from conftest import _find_free_port
    c = cluster_factory([{"devices": [0], "jobs": 1}])
    port = _find_free_port()
    body = {"host": "localhost", "port": port, "remote_dir": str(c.proj), "devices": [0]}
    with ThreadPoolExecutor(6) as pool:
        codes = sorted(pool.map(lambda _: c.status_code("POST", "/api/workers", body), range(6)))
    assert codes == [200] + [409] * 5, codes
    wait_until(lambda: c.connected() == 2, 30, "the new worker to connect")
    assert len(c.worker_pids(port)) == 1
    c.delete(f"/api/workers/localhost:{port}")
    wait_until(lambda: not c.worker_pids(port), 15, "the new worker to exit")


# ── Disconnects and manager crashes ───────────────────────────────────────────


def test_cancel_sent_while_disconnected_reaches_the_worker_on_reconnect(proxied, tmp_path):
    url, proxy = proxied
    c = Cluster.__new__(Cluster)  # reuse the harness' journal helpers on the proxied manager
    c.url, c.journal = url, tmp_path / "journal.txt"
    c.post("/api/jobs", c.job_body("rc", "r", 60))
    wait_until(lambda: c.job("rc", "r")["status"] == "running", 20, "r to run")
    proxy.refuse_for(5)
    proxy.sever()
    c.cancel("rc", "r")
    assert c.job("rc", "r")["status"] == "cancelled"
    assert c.live_pids("rc")  # the worker has not heard yet
    wait_until(lambda: not c.live_pids("rc"), 40, "the cancel to reach the worker")
    [span] = c.spans("rc")
    assert span.how == "term"
    stays(lambda: c.job("rc", "r")["status"] == "cancelled", 3, "r cancelled")


def test_manager_crashes_during_a_dispatch_burst_run_every_job_once(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}, {"devices": [0], "jobs": 0}])
    eid = c.experiment("burst", max_concurrent=3)
    rids = [f"r{i}" for i in range(12)]
    c.submit_many(eid, rids, 1.5)
    for n in (2, 6):  # crash twice, at different points of the sweep
        wait_until(lambda: len(c.spans(eid)) >= n, 30, f"{n} starts")
        c.kill_manager()
        c.start_manager()
    c.wait_statuses(eid, {"done": 12}, timeout=120)
    assert c.starts(eid) == {r: 1 for r in rids}
    assert max_overlap(c.spans(eid))[None] <= 3


def test_manager_crash_during_setup_runs_the_job_once(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}])
    eid = c.experiment("setup")
    c.submit(eid, "r", 0.5, setup_command="sleep 3")
    wait_until(lambda: c.job(eid, "r")["status"] == "dispatched", 20, "dispatch")
    time.sleep(1)  # the worker is in setup
    c.kill_manager()
    c.start_manager()
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 60, "r to finish")
    assert c.starts(eid) == {"r": 1}
    assert c.job(eid, "r")["retry_count"] == 0


def test_manager_restart_keeps_many_running_jobs_in_place(cluster_factory):
    c = cluster_factory([{"devices": [0], "jobs": 0}] * 3)
    eid = c.experiment("many")
    rids = [f"r{i}" for i in range(15)]
    c.submit_many(eid, rids, 12)
    wait_until(lambda: len(c.live_pids(eid)) == 15, 30, "15 running")
    before = {r: j["worker_id"] for r, j in c.jobs(eid).items()}
    c.kill_manager()
    restarted = time.time()
    c.start_manager()
    wait_until(lambda: c.statuses(eid) == {"running": 15}, 15, "all 15 adopted")
    assert time.time() - restarted < 20
    assert {r: j["worker_id"] for r, j in c.jobs(eid).items()} == before
    c.wait_statuses(eid, {"done": 15}, timeout=60)
    assert c.starts(eid) == {r: 1 for r in rids}
