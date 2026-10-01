"""Regression tests for races in the manager's control plane.

Each test drives a real manager and real workers and checks an invariant that
used to break when things happened concurrently:
  - a run is never executed twice across a manager restart, even when its
    worker is slow to come back;
  - cancelling a run during its setup stops it before the training command;
  - two experiments can run jobs with the same run_id without mixing them up;
  - a run's stored log is complete and duplicate-free across a dropped connection;
  - control actions keep job and worker state consistent.
"""

import fcntl
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from conftest import _api_request, _find_free_port
from test_reconnect import (  # noqa: F401  (proxied is a fixture)
    TOKEN,
    _Proxy,
    _api,
    _start_manager,
    _wait,
    proxied,
)


def _put(url, path, body, token=TOKEN):
    return _api_request(url, token, "PUT", path, body)


def _patch(url, path, body, token=TOKEN):
    return _api_request(url, token, "PATCH", path, body)


def _log_text(url, eid, rid, token=TOKEN):
    req = urllib.request.Request(f"{url}/api/experiments/{eid}/jobs/{rid}/logs",
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode()


def _job(url, eid, rid, token=TOKEN):
    return _api(url, f"/api/jobs/{rid}?experiment_id={eid}", token=token)


def _marked(marker, seconds):
    """A command that appends a line to *marker* when it starts, then runs *seconds*."""
    return [sys.executable, "-c",
            f"open({str(marker)!r}, 'a').write('start\\n')\n"
            f"import time\nfor i in range({seconds}): print(i, flush=True); time.sleep(1)"]


# ── Manager restart while a worker is unreachable ──────────────────────────────


def test_restart_does_not_rerun_jobs_of_a_worker_slow_to_return(tmp_path):
    """After a manager restart, a job stays with its worker while that worker is
    unreachable, even when another worker has room for it."""
    PA, PB, QB, M1, M2 = (_find_free_port() for _ in range(5))
    proj = tmp_path / "proj"
    proj.mkdir()
    # Worker B sits behind a proxy the test can cut; the flock makes the
    # manager's own launch for port PB defer to it.
    lock = open(f"/tmp/.mlsweep_worker_port_{PB}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_SH)
    worker_b = subprocess.Popen(
        [sys.executable, "-m", "mlsweep.worker", "--token", TOKEN, "--port", str(QB),
         "--remote-dir", str(proj), "--scratch-dir", str(tmp_path / "scratch_b")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert worker_b.stdout.readline().startswith(b"PORT=")
    proxy = _Proxy(PB, QB)
    wf = tmp_path / "workers.toml"
    wf.write_text(
        f'[[workers]]\nhost = "localhost"\nremote_dir = "{proj}"\nport = {PA}\ndevices = [0]\n'
        f'[[workers]]\nhost = "localhost"\nremote_dir = "{proj}"\nport = {PB}\n'
    )
    managers = []
    try:
        managers.append(_start_manager(tmp_path, M1, str(wf)))
        url = f"http://127.0.0.1:{M1}"
        _wait(lambda: len([w for w in _api(url, "/api/workers") if w["status"] == "connected"]) == 2,
              30, "both workers to connect")
        _api(url, "/api/experiments", {"experiment_id": "rs"})
        wid_a, wid_b = f"localhost:{PA}", f"localhost:{PB}"

        # Put the long job on B (A has no GPU for a moment), the short one on A.
        _patch(url, f"/api/workers/{wid_a}/devices", {"remove": [0]})
        long_marker, short_marker = tmp_path / "long.txt", tmp_path / "short.txt"
        _api(url, "/api/jobs", {"run_id": "long", "experiment_id": "rs",
                                "command": _marked(long_marker, 25)})
        _wait(lambda: _job(url, "rs", "long")["status"] == "running", 20, "long to start")
        assert _job(url, "rs", "long")["worker_id"] == wid_b
        _patch(url, f"/api/workers/{wid_a}/devices", {"add": [0]})
        _api(url, "/api/jobs", {"run_id": "short", "experiment_id": "rs",
                                "command": _marked(short_marker, 3)})
        _wait(lambda: _job(url, "rs", "short")["status"] == "running", 20, "short to start")

        # Crash the manager; bring it back while B is unreachable.
        managers[-1].send_signal(signal.SIGKILL)
        managers[-1].wait()
        proxy.refuse_for(15)
        proxy.sever()
        managers.append(_start_manager(tmp_path, M2, str(wf)))
        url = f"http://127.0.0.1:{M2}"

        # A comes back and finishes short; B's long job must stay put meanwhile.
        _wait(lambda: _job(url, "rs", "short")["status"] == "done", 30, "short to finish")
        deadline = time.time() + 8
        while time.time() < deadline:
            job = _job(url, "rs", "long")
            assert job["status"] in ("dispatched", "running"), job
            assert job["worker_id"] == wid_b
            time.sleep(0.5)

        _wait(lambda: _job(url, "rs", "long")["status"] in ("done", "failed"), 60, "long to finish")
        long_job = _job(url, "rs", "long")
        assert (long_job["status"], long_job["retry_count"]) == ("done", 0)
        assert long_marker.read_text().count("start") == 1
        assert short_marker.read_text().count("start") == 1
    finally:
        for m in managers:
            if m.poll() is None:
                m.terminate()
                m.wait()
        worker_b.terminate()
        worker_b.wait()
        subprocess.run(["pkill", "-f", f"mlsweep.worker .*--port {PA}"], check=False)
        proxy.close()
        lock.close()


# ── Cancel during setup ─────────────────────────────────────────────────────────


def test_cancel_during_setup_stops_the_run_before_it_starts(manager_with_worker, tmp_path):
    server, url = manager_with_worker
    tok = server.token
    _api(url, "/api/experiments", {"experiment_id": "cs"}, token=tok)
    marker = tmp_path / "started.txt"
    _api(url, "/api/jobs", {
        "run_id": "slow_setup", "experiment_id": "cs", "gpus_per_run": 0,
        "setup_command": "sleep 4",
        "command": _marked(marker, 1),
    }, token=tok)
    _wait(lambda: _job(url, "cs", "slow_setup", tok)["status"] == "dispatched", 20, "dispatch")
    time.sleep(1)  # the worker is now in setup
    _api(url, "/api/jobs/slow_setup/cancel?experiment_id=cs", {}, token=tok)
    assert _job(url, "cs", "slow_setup", tok)["status"] == "cancelled"
    time.sleep(6)  # past the end of setup
    assert not marker.exists()
    assert _job(url, "cs", "slow_setup", tok)["status"] == "cancelled"


# ── Same run_id in two experiments ─────────────────────────────────────────────


def test_same_run_id_in_two_experiments_stays_separate(manager_with_worker):
    server, url = manager_with_worker
    tok = server.token
    cmd = [sys.executable, "-c",
           "import os, time\nprint('exp=' + os.environ['EXP_EXPERIMENT'], flush=True)\ntime.sleep(3)"]
    for eid in ("twin_a", "twin_b"):
        _api(url, "/api/experiments", {"experiment_id": eid}, token=tok)
        _api(url, "/api/jobs", {"run_id": "same", "experiment_id": eid,
                                "gpus_per_run": 0, "command": cmd}, token=tok)
    for eid in ("twin_a", "twin_b"):
        _wait(lambda: _job(url, eid, "same", tok)["status"] == "done", 30, f"{eid} to finish")
    for eid, other in (("twin_a", "twin_b"), ("twin_b", "twin_a")):
        text = _log_text(url, eid, "same", tok)
        assert f"exp={eid}" in text and f"exp={other}" not in text
        assert _api(url, f"/api/experiments/{eid}", token=tok)["status"] == "completed"


# ── Log integrity across a dropped connection ───────────────────────────────────


def test_log_is_complete_and_unduplicated_across_a_dropped_connection(proxied):
    url, proxy = proxied
    n = 60
    cmd = [sys.executable, "-c",
           f"import time\nfor i in range({n}): print(f'line {{i}}', flush=True); time.sleep(0.1)"]
    _api(url, "/api/jobs", {"run_id": "chatty", "experiment_id": "rc", "gpus_per_run": 0,
                            "command": cmd})
    _wait(lambda: _job(url, "rc", "chatty")["status"] == "running", 20, "chatty to start")
    time.sleep(2)
    proxy.sever()
    _wait(lambda: _job(url, "rc", "chatty")["status"] == "done", 60, "chatty to finish")
    lines = _log_text(url, "rc", "chatty").splitlines()
    assert lines == [f"line {i}" for i in range(n)]


# ── Control actions ─────────────────────────────────────────────────────────────


def test_readding_a_connected_worker_is_refused(manager_with_worker):
    server, url = manager_with_worker
    tok = server.token
    [worker] = [w for w in _api(url, "/api/workers", token=tok) if w["status"] == "connected"]
    with pytest.raises(urllib.error.HTTPError) as e:
        _api(url, "/api/workers", {"host": worker["host"], "worker_id": worker["worker_id"]},
             token=tok)
    assert e.value.code == 409


def test_experiment_completes_when_its_last_job_is_cancelled(manager_server):
    server, url = manager_server
    tok = server.token
    _api(url, "/api/experiments", {"experiment_id": "cc"}, token=tok)
    _api(url, "/api/jobs", {"run_id": "only", "experiment_id": "cc", "command": ["true"]},
         token=tok)
    _api(url, "/api/jobs/only/cancel?experiment_id=cc", {}, token=tok)
    assert _api(url, "/api/experiments/cc", token=tok)["status"] == "completed"


def test_status_route_refuses_to_touch_a_job_in_flight(manager_with_worker):
    server, url = manager_with_worker
    tok = server.token
    _api(url, "/api/experiments", {"experiment_id": "st"}, token=tok)
    _api(url, "/api/jobs", {"run_id": "busy", "experiment_id": "st", "gpus_per_run": 0,
                            "command": [sys.executable, "-c", "import time; time.sleep(5)"]},
         token=tok)
    _wait(lambda: _job(url, "st", "busy", tok)["status"] == "running", 20, "busy to start")
    with pytest.raises(urllib.error.HTTPError) as e:
        _put(url, "/api/jobs/busy/status", {"experiment_id": "st", "status": "pending"}, token=tok)
    assert e.value.code == 409
    _wait(lambda: _job(url, "st", "busy", tok)["status"] == "done", 20, "busy to finish")
