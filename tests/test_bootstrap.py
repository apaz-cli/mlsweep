"""Tests for mlsweep worker wheel bootstrap logic.

Covers ``_ensure_worker_wheels`` (reusing, building and failing to build the
worker wheel), the defaults of ``_worker_candidates``, and that
``_worker_shell_cmd`` generates valid bash.

The wheel tests point the wheels directory at tmp_path and stand in for pip,
so they never touch the checkout.  Other tests pack the checkout into
artifacts while they run, and a manager running from it ships its wheel to
remote workers.
"""

import importlib.metadata
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from mlsweep import _manager_workers
from mlsweep._manager_workers import _ensure_worker_wheels, _worker_candidates, _worker_shell_cmd

VERSION = importlib.metadata.version("mlsweep")


@pytest.fixture
def wheels(tmp_path, monkeypatch):
    """A temporary wheels dir; ``pip(ok)`` installs a fake pip that builds into it
    (or fails), and ``calls`` collects the pip command lines."""
    d = tmp_path / "_wheels"
    monkeypatch.setattr(_manager_workers, "_WHEELS_DIR", d)
    calls: list[list[str]] = []

    def pip(ok: bool, seconds: float = 0.0) -> None:
        def run(cmd, **kwargs):
            calls.append(cmd)
            time.sleep(seconds)
            if ok:
                (d / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, b"", b"" if ok else b"no")
        monkeypatch.setattr(_manager_workers, "subprocess", SimpleNamespace(run=run))

    return SimpleNamespace(dir=d, calls=calls, pip=pip)


def test_ensure_worker_wheels_builds_and_records_the_version(wheels):
    wheels.dir.mkdir()
    (wheels.dir / "mlsweep-0.0.1-py3-none-any.whl").write_bytes(b"old")
    wheels.pip(ok=True)
    _ensure_worker_wheels()
    assert len(wheels.calls) == 1 and "wheel" in wheels.calls[0]
    assert [p.name for p in wheels.dir.glob("*.whl")] == [f"mlsweep-{VERSION}-py3-none-any.whl"]
    assert (wheels.dir / ".complete").read_text() == VERSION


def test_ensure_worker_wheels_reuses_a_current_wheel(wheels):
    wheels.dir.mkdir()
    (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
    (wheels.dir / ".complete").write_text(VERSION)
    wheels.pip(ok=True)
    _ensure_worker_wheels()
    assert wheels.calls == []


def test_ensure_worker_wheels_rebuilds_after_a_version_change(wheels):
    wheels.dir.mkdir()
    (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
    (wheels.dir / ".complete").write_text("0.0.1")
    wheels.pip(ok=True)
    _ensure_worker_wheels()
    assert len(wheels.calls) == 1
    assert (wheels.dir / ".complete").read_text() == VERSION


def test_managers_starting_together_build_the_wheel_once(wheels):
    wheels.pip(ok=True, seconds=0.5)
    threads = [threading.Thread(target=_ensure_worker_wheels) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wheels.calls) == 1
    assert (wheels.dir / ".complete").read_text() == VERSION


def test_ensure_worker_wheels_survives_a_failed_build(wheels):
    wheels.pip(ok=False)
    _ensure_worker_wheels()
    assert not (wheels.dir / ".complete").exists()
    assert list(wheels.dir.glob("mlsweep-*.whl")) == []


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
