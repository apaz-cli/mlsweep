"""Shared fixtures for mlsweep tests.

Provides a ``manager_server`` fixture that starts a real mlsweep manager
process backed by a temporary SQLite database, plus HTTP helper functions
for integration tests.
"""

import subprocess
import time
import socket
import json
import os
import signal
import sys
import urllib.request
import urllib.error

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Ports for test servers come from a block per xdist worker, below the kernel's
# ephemeral range and the [20000, 30000) range of hashed dist master ports.
# Asking the kernel instead (bind to port 0, close, hand out the number) lets
# two test processes get the same port before either binds it, and a manager
# then adopts the other test's worker on its "fixed" port.
_PORT_BASE = 10000
_PORT_BLOCK = 400  # 25 blocks fit below 20000; more xdist workers wrap around
_ports_handed_out = 0


def _find_free_port():
    """Return a TCP port on localhost that is free and that no other test
    process will be handed."""
    global _ports_handed_out
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    n = int(worker[2:]) if worker[2:].isdigit() else 0
    start = _PORT_BASE + (n % 25) * _PORT_BLOCK
    for _ in range(_PORT_BLOCK):
        port = start + _ports_handed_out % _PORT_BLOCK
        _ports_handed_out += 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
            except OSError:
                continue  # in use by something outside the tests
        return port
    raise RuntimeError(f"no free port in [{start}, {start + _PORT_BLOCK})")


def _api_get(url, token, path):
    """GET *path* from the manager at *url* with *token*, return parsed JSON."""
    req = urllib.request.Request(
        f"{url}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _api_request(url, token, method, path, data=None):
    """Send *data* as JSON with *method* to *path*, return the parsed JSON response."""
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{url}{path}", data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _api_post(url, token, path, data=None):
    """POST *data* (JSON-serialisable) to *path*, return parsed JSON response."""
    return _api_request(url, token, "POST", path, data)


def _experiment_jobs(url, token, experiment_id):
    """Return the list of jobs for *experiment_id* from the manager API."""
    return _api_get(url, token, f"/api/experiments/{experiment_id}/jobs")


def _wait_for_job(url, token, run_id, experiment_id, timeout=60):
    """Poll until *run_id* reaches a terminal status.  Returns the job dict."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            job = _api_get(url, token, f"/api/jobs/{run_id}?experiment_id={experiment_id}")
        except Exception:
            time.sleep(0.5)
            continue
        if job and job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    return None


def _wait_for_experiment_complete(url, token, experiment_id,
                                  expected_jobs=0, expected_success=0, timeout=120):
    """Poll until stopping condition is met.  Returns True on success.

    - expected_jobs: wait until at least this many jobs are in any terminal state
    - expected_success: wait until at least this many jobs have status done/finished
    - If neither is given, wait until no jobs are active (pending/dispatched/running)
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            jobs = _api_get(url, token,
                            f"/api/experiments/{experiment_id}/jobs")
        except Exception:
            time.sleep(1.0)
            continue
        terminal = [j for j in jobs
                    if j["status"] in ("done", "failed",
                                       "cancelled")]
        success = [j for j in jobs if j["status"] == "done"]
        if expected_success and len(success) >= expected_success:
            return True
        if expected_jobs and len(terminal) >= expected_jobs:
            return True
        # If no expected count given, wait until no pending/dispatched/running
        if not expected_jobs and not expected_success:
            active = [j for j in jobs
                      if j["status"] in ("pending", "dispatched", "running")]
            if not active and jobs:
                return True
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _log_tail(tmp_path, lines=40):
    """The end of a manager's log, for a failure message."""
    try:
        text = (tmp_path / "manager.log").read_text(errors="replace")
    except OSError:
        return ""
    return "manager.log:\n" + "\n".join(text.splitlines()[-lines:])


def _start_manager(tmp_path, *, with_worker: bool = False, n_local_workers: int = 0):
    """Start a real mlsweep manager process and return (proc, server, url).

    with_worker=False (default): passes an empty workers file so no local
    worker is spawned.  Use for tests that inspect job state directly via the
    API without needing jobs to execute.

    with_worker=True: omits --workers so the manager spawns a local worker
    automatically.  Use for tests that submit jobs and wait for them to run.

    n_local_workers>0: writes a workers.toml with that many distinct localhost
    worker entries (each pinned to device 0) and waits for all to connect.  Used
    to exercise multi-node scheduling on a single machine — each worker process
    is treated as a separate node.

    Workers keep their scratch under ``tmp_path/scratch`` (the second of two
    workers in a subdirectory it claims), so tests running at the same time
    never share run directories.  The manager's output goes to
    ``tmp_path/manager.log``.
    """
    db_path = str(tmp_path / "manager.db")
    port = _find_free_port()
    mlsweep_dir = tmp_path / "mlsweep"
    mlsweep_dir.mkdir()
    token = "test-token"

    cmd = [
        sys.executable, "-m", "mlsweep.manager",
        "--port", str(port),
        "--db", db_path,
        "--mlsweep-dir", str(mlsweep_dir),
        "--token", token,
        "--scratch-dir", str(tmp_path / "scratch"),
    ]

    if n_local_workers > 0:
        proj = tmp_path / "proj"
        proj.mkdir(exist_ok=True)
        entries = "\n".join(
            "[[workers]]\n"
            'host = "localhost"\n'
            f'remote_dir = "{proj}"\n'
            "devices = [0]\n"
            "port = 0\n"
            for _ in range(n_local_workers)
        )
        workers_file = tmp_path / "workers.toml"
        workers_file.write_text(entries)
        cmd += ["--workers", str(workers_file)]
    elif not with_worker:
        # Empty workers file prevents manager from spawning a local worker,
        # avoiding races with tests that check or manipulate job status directly.
        workers_file = tmp_path / "workers.toml"
        workers_file.write_text("")
        cmd += ["--workers", str(workers_file)]

    # A file rather than a pipe.  Nothing reads the output, and a full pipe
    # would block the manager.
    log = open(tmp_path / "manager.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.close()

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    started = False
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{url}/api/health",
                headers={"Authorization": f"Bearer {token}"},
            )
            if urllib.request.urlopen(req, timeout=2).status == 200:
                started = True
                break
        except Exception:
            time.sleep(0.1)

    if not started:
        proc.terminate()
        proc.wait()
        pytest.fail(f"Manager did not start within 30 seconds.\n{_log_tail(tmp_path)}")

    need_workers = n_local_workers if n_local_workers > 0 else (1 if with_worker else 0)
    if need_workers:
        # Wait for the expected number of workers to connect before yielding.
        deadline = time.time() + 30
        ready = False
        while time.time() < deadline:
            try:
                workers = _api_get(url, token, "/api/workers")
                connected = sum(1 for w in workers if w.get("status") == "connected")
                if connected >= need_workers:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.1)
        if not ready:
            proc.terminate()
            proc.wait()
            pytest.fail(f"Expected {need_workers} worker(s) to connect within 30 seconds.\n"
                        f"{_log_tail(tmp_path)}")

    class Server:
        pass

    server = Server()
    server.url = url
    server.token = token
    server.proc = proc
    server.mlsweep_dir = mlsweep_dir

    return proc, server, url


def _teardown_manager(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager_server(tmp_path):
    """Manager with no worker — for API/status tests that don't execute jobs."""
    proc, server, url = _start_manager(tmp_path, with_worker=False)
    yield server, url
    _teardown_manager(proc)


@pytest.fixture
def manager_with_worker(tmp_path):
    """Manager with a local worker — for tests that submit and run jobs."""
    proc, server, url = _start_manager(tmp_path, with_worker=True)
    yield server, url
    _teardown_manager(proc)


@pytest.fixture
def manager_with_two_workers(tmp_path):
    """Manager with two distinct local workers — for multi-node scheduling tests.

    Each worker process is a separate 'node' on the same machine.
    """
    proc, server, url = _start_manager(tmp_path, n_local_workers=2)
    yield server, url
    _teardown_manager(proc)
