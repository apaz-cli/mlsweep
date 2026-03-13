"""Integration tests for mlsweep_run.

Runs mlsweep_run as a subprocess against fast CPU-only training scripts
and asserts on the output files. No GPU computation required; CUDA_VISIBLE_DEVICES
is set by the runner but the scripts ignore it.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MLSWEEP_RUN = str(Path(sys.executable).parent / "mlsweep_run")
EXP = "test_exp"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(sweep_file: str, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [MLSWEEP_RUN, sweep_file, "--output_dir", str(output_dir), "--experiment", EXP, *extra_args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _exp_dir(output_dir: Path) -> Path:
    return output_dir / EXP


def _status(output_dir: Path) -> dict:
    return json.loads((_exp_dir(output_dir) / "sweep_status.json").read_text())


def _manifest(output_dir: Path) -> dict:
    return json.loads((_exp_dir(output_dir) / "sweep_manifest.json").read_text())


def _gpu_count() -> int:
    try:
        r = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=5)
        return len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_grid_end_to_end(tmp_path):
    result = _run("tests/sweeps/integration_grid.py", tmp_path)
    assert result.returncode == 0, result.stderr

    status = _status(tmp_path)
    assert len(status) == 4
    assert all(s["status"] == "ok" for s in status.values())

    manifest = _manifest(tmp_path)
    assert set(manifest["dims"].keys()) == {"lr", "bs"}
    assert len(manifest["runs"]) == 4

    exp_dir = _exp_dir(tmp_path)
    for run_name in status:
        metrics_path = exp_dir / run_name / "metrics.jsonl"
        assert metrics_path.exists()
        rows = [json.loads(l) for l in metrics_path.read_text().splitlines()]
        assert len(rows) == 1
        assert "loss" in rows[0]


def test_bayes_end_to_end(tmp_path):
    result = _run("tests/sweeps/bayes_sweep.py", tmp_path)
    assert result.returncode == 0, result.stderr

    status = _status(tmp_path)
    completed = {k: v for k, v in status.items() if v["status"] == "ok"}
    failed = {k: v for k, v in status.items() if v["status"] == "failed"}

    # budget=12: at least 12 successful lex evaluations (may slightly exceed due to in-flight runs)
    assert len(completed) >= 12

    # singular dim: all completions are at batch_size=64 (the largest that fits)
    assert all(v["combo"]["batch_size"] == 64 for v in completed.values())

    # all failures are at batch_size > 64
    assert all(v["combo"]["batch_size"] > 64 for v in failed.values())

    # run names follow bayes_sweep_bayes_NNNN
    assert all(k.startswith("bayes_sweep_bayes_") for k in completed)

    # metrics logged for every completed run
    exp_dir = _exp_dir(tmp_path)
    for run_name in completed:
        metrics_path = exp_dir / run_name / "metrics.jsonl"
        assert metrics_path.exists()
        rows = [json.loads(l) for l in metrics_path.read_text().splitlines()]
        assert any("val_loss" in r for r in rows)


def test_resume_skips_completed(tmp_path):
    r1 = _run("tests/sweeps/integration_grid.py", tmp_path)
    assert r1.returncode == 0, r1.stderr
    status_before = _status(tmp_path)
    assert len(status_before) == 4

    r2 = _run("tests/sweeps/integration_grid.py", tmp_path, "--resume")
    assert r2.returncode == 0, r2.stderr
    status_after = _status(tmp_path)

    assert set(status_after.keys()) == set(status_before.keys())


def test_dry_run_creates_no_output(tmp_path):
    result = _run("tests/sweeps/integration_grid.py", tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not (_exp_dir(tmp_path) / "sweep_status.json").exists()
    assert "python" in result.stdout


def test_torchrun_rank_zero_logging(tmp_path):
    pytest.importorskip("torch")

    result = _run("tests/sweeps/torchrun_sweep.py", tmp_path)
    assert result.returncode == 0, result.stderr

    status = _status(tmp_path)
    assert len(status) == 1
    assert all(s["status"] == "ok" for s in status.values())

    exp_dir = _exp_dir(tmp_path)
    for run_name in status:
        metrics_path = exp_dir / run_name / "metrics.jsonl"
        rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
        assert len(rows) == 1, f"expected 1 metrics row from rank 0, got {len(rows)}"
        assert rows[0].get("world_size") == 2.0


def test_set_dist_env(tmp_path):
    pytest.importorskip("torch")

    result = _run("tests/sweeps/set_dist_env_sweep.py", tmp_path)
    assert result.returncode == 0, result.stderr

    status = _status(tmp_path)
    assert len(status) == 1
    assert all(s["status"] == "ok" for s in status.values())

    exp_dir = _exp_dir(tmp_path)
    for run_name in status:
        metrics_path = exp_dir / run_name / "metrics.jsonl"
        rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
        # Only rank 0 logs; world_size == GPUS_PER_RUN == 2
        assert len(rows) == 1, f"expected 1 metrics row from rank 0, got {len(rows)}"
        assert rows[0].get("world_size") == 2.0


@pytest.mark.skipif(_gpu_count() < 2, reason="requires at least 2 GPUs")
def test_gpus_per_run(tmp_path):
    result = _run("tests/sweeps/multigpu_sweep.py", tmp_path, "-g", "2")
    assert result.returncode == 0, result.stderr

    status = _status(tmp_path)
    assert len(status) == 2
    assert all(s["status"] == "ok" for s in status.values())

    # verify CUDA_VISIBLE_DEVICES was set to a 2-GPU group for each run
    exp_dir = _exp_dir(tmp_path)
    for run_name in status:
        log = (exp_dir / run_name / "training.log").read_text()
        cvd_line = next(l for l in log.splitlines() if "CUDA_VISIBLE_DEVICES" in l)
        devices = cvd_line.split("=", 1)[1].strip()
        assert "," in devices, f"expected 2 devices in CUDA_VISIBLE_DEVICES, got {devices!r}"
