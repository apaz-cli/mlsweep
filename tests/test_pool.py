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


def test_simple_command(manager_server):
    """Submit a job that echoes hello; verify success."""
    server, url = manager_server

    exp = manager_create_experiment(url, _TOKEN, "exp_simple", "simple_test")
    assert exp is not None

    job = _submit_and_wait(url, "exp_simple", "run1", ["echo", "hello"])
    assert job["status"] == "done", f"unexpected status: {job['status']}"
    assert job.get("exit_code") == 0


def test_failed_command(manager_server):
    """Submit a job that exits with code 42; verify failure."""
    server, url = manager_server

    manager_create_experiment(url, _TOKEN, "exp_fail", "fail_test")

    job = _submit_and_wait(url, "exp_fail", "run1", ["sh", "-c", "exit 42"])
    assert job["status"] == "failed"
    assert job.get("exit_code") == 42


def test_files_workspace(manager_server, tmp_path):
    """Submit a job with file injection; verify the command can read the file
    and the original project directory is unmodified."""
    server, url = manager_server

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
    assert job.get("exit_code") == 0

    # remote_dir must be unmodified
    assert (remote_dir / "untouched.txt").read_text() == "original"
    assert not (remote_dir / "a.py").exists()


def test_concurrent_slots(manager_server):
    """Submit two jobs in one bulk call; verify both complete."""
    server, url = manager_server

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
    assert job_a.get("exit_code") == 0
    assert job_b.get("exit_code") == 0
