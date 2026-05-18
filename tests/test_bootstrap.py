"""Tests for mlsweep worker wheel bootstrap logic.

Verifies that ``_ensure_worker_wheels`` handles its build pipeline
correctly (success or graceful failure), that ``_worker_candidates``
returns sensible defaults, and that ``_worker_shell_cmd`` generates
syntactically valid bash.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mlsweep._manager_workers import _worker_shell_cmd, _worker_candidates


@pytest.fixture(autouse=True)
def _remove_sentinel():
    """Remove .complete sentinel so tests always trigger a build check."""
    wheels_dir = Path(__file__).resolve().parent.parent / "mlsweep" / "_wheels"
    sentinel = wheels_dir / ".complete"
    sentinel_exists = sentinel.exists()
    if sentinel_exists:
        sentinel.unlink()
    yield
    # Restore sentinel if we removed it
    if sentinel_exists:
        sentinel.touch()


def test_ensure_worker_wheels_handles_outcome():
    """Calling _ensure_worker_wheels should either build wheels or fail
    gracefully (cleaning up orphaned wheels)."""
    from mlsweep._manager_workers import _ensure_worker_wheels

    wheels_dir = Path(__file__).resolve().parent.parent / "mlsweep" / "_wheels"
    (wheels_dir / ".complete").unlink(missing_ok=True)

    try:
        _ensure_worker_wheels()
    except Exception as exc:
        pytest.fail(f"_ensure_worker_wheels raised unexpectedly: {exc}")

    # After the call, either the sentinel + wheels exist (success),
    # or the sentinel is absent and no orphaned mlsweep wheels remain
    # (pip download failure handled gracefully).
    if (wheels_dir / ".complete").exists():
        whls = list(wheels_dir.glob("mlsweep-*.whl"))
        assert len(whls) >= 1, f"sentinel present but no mlsweep wheel in {wheels_dir}"
    else:
        # pip download may have failed (e.g. Python version mismatch);
        # verify orphaned mlsweep wheel was cleaned up.
        orphaned = list(wheels_dir.glob("mlsweep-*.whl"))
        assert len(orphaned) == 0, (
            f"orphaned mlsweep wheels left behind in {wheels_dir}: {orphaned}"
        )


def test_worker_shell_cmd_valid_bash():
    """The generated worker shell command should be syntactically valid bash."""
    # _worker_shell_cmd takes (candidates, worker_args)
    candidates = _worker_candidates(venv=None)
    worker_args = [
        "--manager", "http://127.0.0.1:9999",
        "--token", "test",
        "--remote-dir", "/tmp/test",
    ]
    cmd = _worker_shell_cmd(candidates, worker_args)

    assert "exec" in cmd
    assert "--remote-dir" in cmd

    try:
        r = subprocess.run(
            ["bash", "-n", "-c", cmd],
            capture_output=True, text=True, timeout=5,
        )
        assert r.returncode == 0, f"bash syntax error:\n{r.stderr}"
    except FileNotFoundError:
        pytest.skip("bash not found — cannot validate shell syntax")


def test_worker_candidates_defaults():
    """_worker_candidates returns sensible default search paths."""
    candidates = _worker_candidates(venv=None)
    assert any("mlsweep_worker" in c for c in candidates)
    assert "/tmp/mlsweep_venv/bin/mlsweep_worker" in candidates
