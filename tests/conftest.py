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

def _find_free_port():
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _api_get(url, token, path):
    """GET *path* from the manager at *url* with *token*, return parsed JSON."""
    req = urllib.request.Request(
        f"{url}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _api_post(url, token, path, data=None):
    """POST *data* (JSON-serialisable) to *path*, return parsed JSON response."""
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{url}{path}",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


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
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def manager_server(tmp_path):
    """Start a real mlsweep manager, yield ``(server, url)``, tear down."""
    db_path = str(tmp_path / "manager.db")
    port = _find_free_port()
    mlsweep_dir = tmp_path / "mlsweep"
    mlsweep_dir.mkdir()
    token = "test-token"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlsweep.manager",
            "--port", str(port),
            "--db", db_path,
            "--mlsweep-dir", str(mlsweep_dir),
            "--token", token,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

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
            time.sleep(0.5)

    if not started:
        proc.terminate()
        proc.wait()
        stdout, stderr = proc.communicate()
        pytest.fail(
            f"Manager did not start within 30 seconds.\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

    class Server:
        pass

    server = Server()
    server.url = url
    server.token = token
    server.proc = proc

    yield server, url

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
