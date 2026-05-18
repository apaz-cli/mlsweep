"""Integration tests for mlsweep_run CLI.

Runs ``mlsweep_run`` as a subprocess against the real **mlsweep manager**
(fixture defined in conftest.py).  Jobs are dispatched to a local worker and
executed asynchronously — tests poll for completion via the HTTP API.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# Import helpers from conftest
from conftest import (
    _api_get,
    _api_post,
    _experiment_jobs,
    _wait_for_experiment_complete,
    _wait_for_job,
)

REPO_ROOT = Path(__file__).parent.parent
EXP = "test_exp"
_TOKEN = "test-token"

# mlsweep_run is installed as ``mlsweep.run_sweep:main`` console_script.
MLSWEEP_RUN = [sys.executable, "-m", "mlsweep.run_sweep"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(
    sweep_file: str,
    output_dir: Path,
    *extra_args: str,
    manager_url: str | None = None,
    expect_failure: bool = False,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run ``mlsweep_run`` as a subprocess.

    Parameters
    ----------
    manager_url:
        When given, ``--manager`` and ``--token`` are appended.
    expect_failure:
        When True, a non-zero return code does *not* raise an assertion.
    """
    cmd = [*MLSWEEP_RUN, sweep_file,
           "--output-dir", str(output_dir),
           "--experiment", EXP]
    if manager_url is not None:
        cmd += ["--manager", manager_url, "--token", _TOKEN]
    cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if not expect_failure and result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        print(f"STDOUT:\n{result.stdout}", file=sys.stderr)
    return result


def _exp_dir(output_dir: Path) -> Path:
    return output_dir / EXP


def _manifest(output_dir: Path) -> dict:
    return json.loads((_exp_dir(output_dir) / "sweep_manifest.json").read_text())


def _gpu_count() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=5
        )
        return len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


# ── Tests: dry-run / validate (no manager needed) ──────────────────────────────

def test_dry_run_creates_no_output(tmp_path):
    """``--dry-run`` prints commands and creates no output files."""
    result = _run("tests/sweeps/integration_grid.py", tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not (_exp_dir(tmp_path) / "sweep_manifest.json").exists()
    assert "python" in result.stdout


def test_validate_prints_combos(tmp_path):
    """``--validate`` prints all combinations and exits without a manager."""
    result = _run("tests/sweeps/integration_grid.py", tmp_path, "--validate")
    assert result.returncode == 0, result.stderr
    assert "lr" in result.stdout
    assert "bs" in result.stdout
    assert "Total combinations" in result.stdout


# ── Tests: end-to-end via real manager ─────────────────────────────────────────

def test_grid_end_to_end(manager_with_worker, tmp_path):
    """Grid sweep: submit 4 jobs, verify all finish successfully."""
    server, url = manager_with_worker
    token = server.token

    result = _run("tests/sweeps/integration_grid.py", tmp_path,
                  "--fetch", manager_url=url)
    assert result.returncode == 0, result.stderr

    # Check manifest written locally
    manifest = _manifest(tmp_path)
    assert set(manifest["dims"].keys()) == {"lr", "bs"}
    assert len(manifest["runs"]) == 4

    # Wait for all 4 jobs to finish
    ok = _wait_for_experiment_complete(url, token, EXP, expected_jobs=4, timeout=120)
    assert ok, "Timed out waiting for grid sweep jobs to complete"

    # Check via HTTP API: 4 jobs, all finished
    exp_jobs = _experiment_jobs(url, token, EXP)
    assert len(exp_jobs) >= 4
    finished = [j for j in exp_jobs if j["status"] == "done"]
    assert len(finished) == 4, f"Expected 4 done, got {len(finished)}: {exp_jobs}"


def test_bayes_end_to_end(manager_with_worker, tmp_path):
    """Bayesian sweep: submit budget=12 jobs, verify behaviour."""
    server, url = manager_with_worker
    token = server.token

    result = _run("tests/sweeps/bayes_sweep.py", tmp_path,
                  "--fetch", manager_url=url)
    assert result.returncode == 0, result.stderr

    # Wait for at least 12 successful jobs (budget=12 lex combos, each with 2
    # successful batch_size probes: 64 and 32).
    ok = _wait_for_experiment_complete(url, token, EXP, expected_success=12, timeout=180)
    assert ok, "Timed out waiting for bayes sweep jobs"

    exp_jobs = _experiment_jobs(url, token, EXP)
    completed = [j for j in exp_jobs if j["status"] == "done"]
    # batch_size=256 and =128 fail then get reclassified as xfailed when =64 succeeds
    xfailed = [j for j in exp_jobs if j["status"] == "xfailed"]

    # budget=12: at least 12 successful evaluations
    assert len(completed) >= 12

    # Run names follow bayes_sweep_bayes_NNNN
    assert all(j["run_id"].startswith("bayes_sweep_bayes_") for j in completed)

    # All completions have exit_code 0
    assert all(j["exit_code"] == 0 for j in completed)
    # Singular probes that OOMed are reclassified to xfailed (not failed)
    assert len(xfailed) > 0, "expected at least one xfailed singular probe"
    assert all(j["exit_code"] != 0 for j in xfailed)


def test_experiment_created(manager_with_worker, tmp_path):
    """Verify the manager records the experiment metadata."""
    server, url = manager_with_worker
    token = server.token

    _run("tests/sweeps/integration_grid.py", tmp_path,
         "--fetch", manager_url=url)

    # Check experiment via HTTP API
    exp = _api_get(url, token, f"/api/experiments/{EXP}")
    assert exp is not None, f"experiment {EXP} not found via API"
    assert exp["status"] in ("running", "completed")
    assert exp["name"] == "integration_grid"


# ── Tests: rank-zero logging / dist env (no torch) ────────────────────────────

def test_rank_zero_logging(manager_with_worker, tmp_path):
    """Torchrun sweep with SET_DIST_ENV / GPUS_PER_RUN=2."""
    server, url = manager_with_worker
    token = server.token

    result = _run("tests/sweeps/torchrun_sweep.py", tmp_path,
                  "--fetch", manager_url=url)
    assert result.returncode == 0, result.stderr

    # Poll for the single job
    ok = _wait_for_experiment_complete(url, token, EXP, expected_jobs=1, timeout=60)
    assert ok, "Timed out waiting for torchrun sweep job"

    exp_jobs = _experiment_jobs(url, token, EXP)
    assert len(exp_jobs) >= 1
    finished = [j for j in exp_jobs if j["status"] == "done"]
    assert len(finished) == 1
    assert finished[0]["exit_code"] == 0


def test_set_dist_env(manager_with_worker, tmp_path):
    """Sweep with SET_DIST_ENV=True."""
    server, url = manager_with_worker
    token = server.token

    result = _run("tests/sweeps/set_dist_env_sweep.py", tmp_path,
                  "--fetch", manager_url=url)
    assert result.returncode == 0, result.stderr

    # Poll for the single job
    ok = _wait_for_experiment_complete(url, token, EXP, expected_jobs=1, timeout=60)
    assert ok, "Timed out waiting for set_dist_env sweep job"

    exp_jobs = _experiment_jobs(url, token, EXP)
    assert len(exp_jobs) >= 1
    finished = [j for j in exp_jobs if j["status"] == "done"]
    assert len(finished) == 1
    assert finished[0]["exit_code"] == 0


# ── Tests: GPU (require multiple GPUs) ─────────────────────────────────────────

@pytest.mark.skipif(_gpu_count() < 2, reason="requires at least 2 GPUs")
def test_gpus_per_run(manager_with_worker, tmp_path):
    """Sweep with GPUS_PER_RUN=2 in the sweep file."""
    server, url = manager_with_worker
    token = server.token

    result = _run("tests/sweeps/multigpu_sweep.py", tmp_path,
                  "--fetch", manager_url=url)
    assert result.returncode == 0, result.stderr

    # Wait for both jobs
    ok = _wait_for_experiment_complete(url, token, EXP, expected_jobs=2, timeout=60)
    assert ok, "Timed out waiting for multi-GPU sweep jobs"

    exp_jobs = _experiment_jobs(url, token, EXP)
    assert len(exp_jobs) >= 2
    finished = [j for j in exp_jobs if j["status"] == "done"]
    assert len(finished) == 2
    assert all(j["exit_code"] == 0 for j in finished)


# ── Tests: missing manager (error path) ────────────────────────────────────────

def test_missing_manager_fails(tmp_path):
    """Running without --manager should exit non-zero with an error message."""
    result = _run("tests/sweeps/integration_grid.py", tmp_path,
                  expect_failure=True)
    assert result.returncode != 0
    assert "--manager" in result.stderr or "--manager" in result.stdout


# ── Tests: subcommands (watch / fetch) ─────────────────────────────────────────

def test_watch_subcommand(manager_with_worker, tmp_path):
    """``mlsweep_run watch {experiment_id}`` connects, receives events, and exits cleanly."""
    server, url = manager_with_worker

    # Submit sweep WITHOUT --fetch so it returns as fast as possible.
    # Then immediately start watching before all jobs finish.
    _run("tests/sweeps/integration_grid.py", tmp_path, manager_url=url)

    watch = subprocess.run(
        MLSWEEP_RUN + ["watch", EXP, "--manager", url, "--token", server.token],
        env={**os.environ, "MLSWEEP_MANAGER": url, "MLSWEEP_TOKEN": server.token},
        capture_output=True, text=True, timeout=30,
    )
    assert watch.returncode == 0, watch.stderr
    assert "Experiment complete" in watch.stdout or "done" in watch.stdout.lower()


def test_fetch_subcommand(manager_with_worker, tmp_path):
    """``mlsweep_run fetch`` downloads experiment results after a sweep finishes."""
    server, url = manager_with_worker

    _run("tests/sweeps/integration_grid.py", tmp_path, "--fetch", manager_url=url)

    # Wait for jobs so there is something to fetch
    ok = _wait_for_experiment_complete(url, server.token, EXP, expected_jobs=4, timeout=120)
    assert ok, "Timed out waiting for grid sweep jobs"

    fetch_dir = tmp_path / "fetched"
    fetch_dir.mkdir()

    fetch = subprocess.run(
        MLSWEEP_RUN + [
            "fetch",
            "--manager", url,
            "--token", server.token,
            "--experiment", EXP,
            "--output-dir", str(fetch_dir),
        ],
        env={**os.environ, "MLSWEEP_MANAGER": url, "MLSWEEP_TOKEN": server.token},
        capture_output=True, text=True, timeout=30,
    )
    assert fetch.returncode == 0, fetch.stderr
    # fetch should print job summary lines
    assert EXP in fetch.stdout or "done" in fetch.stdout.lower() or "Job" in fetch.stdout


# ── Tests: authentication ─────────────────────────────────────────────────────

def test_auth_required(manager_server):
    """Requests without valid token receive HTTP 401."""
    server, url = manager_server

    # No token at all
    req = urllib.request.Request(f"{url}/api/experiments")
    try:
        urllib.request.urlopen(req, timeout=5)
        pytest.fail("Should have raised HTTPError for missing token")
    except urllib.error.HTTPError as e:
        assert e.code == 401

    # Wrong token
    req = urllib.request.Request(
        f"{url}/api/experiments",
        headers={"Authorization": "Bearer wrong-token"},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        pytest.fail("Should have raised HTTPError for wrong token")
    except urllib.error.HTTPError as e:
        assert e.code == 401


# ── Tests: error paths ────────────────────────────────────────────────────────

def test_bad_sweep_file(manager_server, tmp_path):
    """Malformed sweep file produces non-zero exit and an error message."""
    server, url = manager_server

    bad_sweep = tmp_path / "bad_sweep.py"
    bad_sweep.write_text(
        "COMMAND = [sys.executable, 'tests/scripts/fast_train.py']\n"
        "# Missing SWEEP_NAME\n"
    )

    result = _run(str(bad_sweep), tmp_path, manager_url=url, expect_failure=True)
    assert result.returncode != 0
    assert "error" in (result.stdout + result.stderr).lower()


# ── Tests: artifact sweep ─────────────────────────────────────────────────────


def test_artifact_sweep_end_to_end(manager_with_worker, tmp_path):
    """Artifact sweep: jobs download a project tarball, write output files, and
    those files are synced back to the manager's experiment output directory.

    This exercises:
    - Artifact upload to the manager via the HTTP API (run_sweep.py)
    - MsgRun dispatched via a worker thread (worker.py threading change)
    - Artifact download from the manager's HTTP server (localhost tunnel path)
    - Output file rsync from worker scratch to manager output dir
    """
    server, url = manager_with_worker
    token = server.token

    # Run 2×2 = 4 jobs (subset of the full 3×3 grid — fast enough for CI)
    result = _run(
        "tests/sweeps/artifacts.py", tmp_path,
        "--fetch", "--experiment", EXP,
        manager_url=url, timeout=180,
    )
    assert result.returncode == 0, result.stderr

    ok = _wait_for_experiment_complete(url, token, EXP, expected_jobs=9, timeout=150)
    assert ok, "Timed out waiting for artifact sweep to complete"

    jobs = _experiment_jobs(url, token, EXP)
    done = [j for j in jobs if j["status"] == "done"]
    assert len(done) == 9, f"Expected 9 done jobs, got {len(done)}: {[j['status'] for j in jobs]}"

    # Verify artifact files were synced back for every completed run.
    # Output path: {mlsweep_dir}/experiments/{experiment_id}/{run_id}/artifacts/
    exp_output = server.mlsweep_dir / "experiments" / EXP
    for job in done:
        run_dir = exp_output / job["run_id"] / "artifacts"
        assert (run_dir / "plot.png").exists(), f"Missing plot.png for {job['run_id']}"
        assert (run_dir / "results.json").exists(), f"Missing results.json for {job['run_id']}"
        assert (run_dir / "training.csv").exists(), f"Missing training.csv for {job['run_id']}"


# ── Tests: logs sweep ─────────────────────────────────────────────────────────


def test_logs_sweep_end_to_end(manager_with_worker, tmp_path):
    """Logs sweep: runs log_train.py with varied lr/bs, verifies all jobs
    complete and the log endpoint returns data for each run.

    Passes ``-- --epochs 1 --bs 512`` to cap runtime to a few seconds per run.
    """
    server, url = manager_with_worker
    token = server.token

    result = _run(
        "tests/sweeps/logs.py", tmp_path,
        "--fetch", "--experiment", EXP,
        "--", "--epochs", "1", "--bs", "512",
        manager_url=url, timeout=120,
    )
    assert result.returncode == 0, result.stderr

    ok = _wait_for_experiment_complete(url, token, EXP, expected_jobs=4, timeout=90)
    assert ok, "Timed out waiting for logs sweep to complete"

    jobs = _experiment_jobs(url, token, EXP)
    done = [j for j in jobs if j["status"] == "done"]
    assert len(done) == 4, f"Expected 4 done jobs, got {len(done)}"

    # Every run should have at least one log line streamed back.
    for job in done:
        try:
            req = urllib.request.Request(
                f"{url}/api/experiments/{EXP}/jobs/{job['run_id']}/logs",
                headers={"Authorization": f"Bearer {token}"},
            )
            text = urllib.request.urlopen(req, timeout=10).read().decode()
            assert text.strip(), f"No logs for {job['run_id']}"
        except urllib.error.HTTPError as e:
            pytest.fail(f"Log endpoint returned {e.code} for {job['run_id']}")
