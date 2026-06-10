"""Integration tests for mlsweep controller HTTP client (run_sweep.py).

Tests the HTTP client functions against a real mlsweep manager (fixture from
conftest.py).  Jobs are dispatched to a local worker and executed
asynchronously — tests poll for completion via the HTTP API.
"""

import json
import time
from pathlib import Path

import pytest

from mlsweep.run_sweep import (
    _http_request,
    _manager_url,
    _WebSocket,
    manager_create_experiment,
    manager_get_experiment_summary,
    manager_list_experiment_jobs,
    manager_register_artifact,
    manager_submit_jobs_bulk,
    manager_upload_artifact_data,
)

# Import helpers from conftest
from conftest import (
    _api_get,
    _wait_for_job,
)


_TOKEN = "test-token"


# ===============================================================================
# Helpers
# ===============================================================================


def _submit_and_wait(url: str, experiment_id: str, run_id: str,
                     command: list[str],
                     files: dict[str, str] | None = None,
                     return_files: list[str] | None = None,
                     timeout: int = 60) -> dict:
    """Submit a single job and poll until it reaches a terminal status."""
    jobs = [{
        "run_id": run_id,
        "experiment_id": experiment_id,
        "command": command,
        "files": files or {},
        "return_files": return_files or [],
    }]
    results = manager_submit_jobs_bulk(url, _TOKEN, jobs)
    assert results is not None, "manager_submit_jobs_bulk returned None"
    assert len(results) == 1
    # The returned record is "pending" — wait for completion
    job = _wait_for_job(url, _TOKEN, run_id, experiment_id, timeout=timeout)
    assert job is not None, f"Job {run_id} did not complete within {timeout}s"
    return job


# ===============================================================================
# Tests
# ===============================================================================


def test_simple_command(manager_with_worker):
    """Submit a job that echoes hello; verify success."""
    server, url = manager_with_worker

    exp = manager_create_experiment(url, _TOKEN, "exp_simple", "simple_test")
    assert exp is not None

    job = _submit_and_wait(url, "exp_simple", "run1", ["echo", "hello"])
    assert job["status"] == "done", f"unexpected status: {job['status']}"
    assert job["exit_code"] == 0


def test_failed_command(manager_with_worker):
    """Submit a job that exits with code 42; verify failure."""
    server, url = manager_with_worker

    manager_create_experiment(url, _TOKEN, "exp_fail", "fail_test")

    job = _submit_and_wait(url, "exp_fail", "run1", ["sh", "-c", "exit 42"])
    assert job["status"] == "failed"
    assert job["exit_code"] == 42


def test_files_workspace(manager_with_worker, tmp_path):
    """Submit a job with file injection; verify the command can read the file
    and the original project directory is unmodified."""
    server, url = manager_with_worker

    # Create a "project" directory with an existing file
    remote_dir = tmp_path / "project"
    remote_dir.mkdir()
    (remote_dir / "untouched.txt").write_text("original")

    manager_create_experiment(url, _TOKEN, "exp_files", "files_test")

    # Write a script that reads an injected file and exits 0 if successful
    script = (
        "import pathlib, sys; "
        "content = pathlib.Path('a.py').read_text(); "
        "sys.exit(0 if 'workspace ok' in content else 1)"
    )
    job = _submit_and_wait(
        url, "exp_files", "run1",
        command=["python", "-c", script],
        files={"a.py": "print('workspace ok')"},
    )

    assert job["status"] == "done"
    assert job["exit_code"] == 0

    # remote_dir must be unmodified
    assert (remote_dir / "untouched.txt").read_text() == "original"
    assert not (remote_dir / "a.py").exists()


def _job_status(url: str, experiment_id: str, run_id: str) -> str | None:
    """Return a single job's status, or None if not found."""
    status, resp = _http_request(
        "GET",
        _manager_url(url, f"/api/jobs/{run_id}?experiment_id={experiment_id}"),
        _TOKEN,
    )
    if status == 200 and isinstance(resp, dict):
        return resp.get("status")
    return None


def test_abort_experiment_stops_dispatch(manager_with_worker):
    """Aborting an experiment must immediately stop the scheduler from
    dispatching its remaining pending jobs (regression for D1: abort used to be
    a no-op because the scheduler ignored experiment status).

    With a single-slot worker and many short jobs, the buggy version would churn
    through and complete all of them; the fixed scheduler reads experiment
    status from the DB each pass, so once aborted, untouched jobs stay pending.
    """
    server, url = manager_with_worker
    exp = "exp_abort"
    manager_create_experiment(url, _TOKEN, exp, "abort_test")

    n = 6
    jobs = [
        {
            "run_id": f"run{i}",
            "experiment_id": exp,
            "command": ["sh", "-c", "sleep 1"],
            "files": {},
            "return_files": [],
        }
        for i in range(n)
    ]
    assert manager_submit_jobs_bulk(url, _TOKEN, jobs) is not None

    # Abort right away — before the single slot can chew through all jobs.
    status, _ = _http_request(
        "PUT", _manager_url(url, f"/api/experiments/{exp}/status"),
        _TOKEN, json_data={"status": "aborted"},
    )
    assert status == 200

    # Give the buggy version ample time to (wrongly) run everything serially.
    time.sleep(n + 3)

    all_jobs = manager_list_experiment_jobs(url, _TOKEN, exp)
    assert all_jobs is not None
    by_status: dict[str, int] = {}
    for j in all_jobs:
        by_status[j["status"]] = by_status.get(j["status"], 0) + 1

    done = by_status.get("done", 0)
    pending = by_status.get("pending", 0)
    # The sweep was halted: not everything ran, and untouched jobs remain pending.
    assert done < n, f"abort did not stop dispatch; statuses={by_status}"
    assert pending >= 1, f"expected held pending jobs; statuses={by_status}"


def test_cancel_running_job_frees_slot(manager_with_worker):
    """Cancelling a running job must free its slot so new work can run
    (regression for D2: cancel used to leak GPU occupancy permanently, so the
    worker would look 'full' forever and never accept new jobs).

    Every slot is filled first, so a new job can only run if the cancelled
    job's slot is actually released.
    """
    server, url = manager_with_worker
    exp = "exp_cancel_slot"
    manager_create_experiment(url, _TOKEN, exp, "cancel_slot_test")

    # Total slots = sum of GPUs across connected workers (one job per GPU).
    workers = _api_get(url, _TOKEN, "/api/workers")
    slots = sum(len(w.get("gpus") or []) for w in workers if w.get("status") == "connected")
    slots = max(slots, 1)

    # Fill every slot with a long-running hog.
    hogs = [{
        "run_id": f"hog{i}", "experiment_id": exp,
        "command": ["sh", "-c", "sleep 30"], "files": {}, "return_files": [],
    } for i in range(slots)]
    assert manager_submit_jobs_bulk(url, _TOKEN, hogs) is not None

    deadline = time.time() + 30
    while time.time() < deadline:
        if all(_job_status(url, exp, f"hog{i}") in ("dispatched", "running")
               for i in range(slots)):
            break
        time.sleep(0.5)
    else:
        pytest.fail("hogs did not all start")

    # A new job submitted now stays pending — the cluster is full.
    assert manager_submit_jobs_bulk(url, _TOKEN, [{
        "run_id": "follow", "experiment_id": exp,
        "command": ["echo", "ok"], "files": {}, "return_files": [],
    }]) is not None
    time.sleep(1.5)
    assert _job_status(url, exp, "follow") == "pending"

    # Cancelling one hog must free its slot so the follow-up can run.
    status, _ = _http_request(
        "DELETE", _manager_url(url, f"/api/experiments/{exp}/jobs/hog0"), _TOKEN,
    )
    assert status == 200

    follow = _wait_for_job(url, _TOKEN, "follow", exp, timeout=30)
    assert follow is not None and follow["status"] == "done", (
        "follow-up job did not run — the cancelled job's slot leaked"
    )


def test_concurrent_slots(manager_with_worker):
    """Submit two jobs in one bulk call; verify both complete."""
    server, url = manager_with_worker

    manager_create_experiment(url, _TOKEN, "exp_concurrent", "concurrent_test")

    jobs = [
        {
            "run_id": "run_alpha",
            "experiment_id": "exp_concurrent",
            "command": ["sh", "-c", "echo alpha"],
            "files": {},
            "return_files": [],
        },
        {
            "run_id": "run_beta",
            "experiment_id": "exp_concurrent",
            "command": ["sh", "-c", "echo beta"],
            "files": {},
            "return_files": [],
        },
    ]
    results = manager_submit_jobs_bulk(url, _TOKEN, jobs)
    assert results is not None
    assert len(results) == 2

    # Wait for both to complete
    job_a = _wait_for_job(url, _TOKEN, "run_alpha", "exp_concurrent", timeout=60)
    job_b = _wait_for_job(url, _TOKEN, "run_beta", "exp_concurrent", timeout=60)
    assert job_a is not None, "run_alpha did not complete"
    assert job_b is not None, "run_beta did not complete"
    assert job_a["status"] == "done"
    assert job_b["status"] == "done"
    assert job_a["exit_code"] == 0
    assert job_b["exit_code"] == 0
