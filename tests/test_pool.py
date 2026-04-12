"""Integration tests for mlsweep WorkerPool (pool.py).

All tests use a local CPU-only worker — no GPU, no SSH, no network.
"""

import threading
from pathlib import Path

import pytest

from mlsweep._shared import MsgRun
from mlsweep.pool import WorkerConfig, WorkerPool


def _local_pool(tmp_path: Path, remote_dir: str = "", jobs: int = 1) -> WorkerPool:
    return WorkerPool(
        [WorkerConfig(host=None, remote_dir=remote_dir, devices=[], jobs=jobs)],
        output_dir=str(tmp_path / "pool_output"),
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_simple_command(tmp_path):
    with _local_pool(tmp_path) as pool:
        result = pool.run(MsgRun(command=["echo", "hello"]))
    assert result.success
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"


def test_failed_command(tmp_path):
    with _local_pool(tmp_path) as pool:
        result = pool.run(MsgRun(command=["sh", "-c", "exit 42"]))
    assert not result.success
    assert result.exit_code == 42


def test_files_workspace(tmp_path):
    remote_dir = tmp_path / "project"
    remote_dir.mkdir()
    (remote_dir / "untouched.txt").write_text("original")

    script = "import pathlib; print(pathlib.Path('a.py').read_text())"
    with _local_pool(tmp_path, remote_dir=str(remote_dir)) as pool:
        result = pool.run(MsgRun(
            command=["python", "-c", script],
            files={"a.py": "print('workspace ok')"},
        ))

    assert result.success
    assert "workspace ok" in result.stdout
    # remote_dir must be unmodified
    assert (remote_dir / "untouched.txt").read_text() == "original"
    assert not (remote_dir / "a.py").exists()


def test_return_files_with_workspace(tmp_path):
    with _local_pool(tmp_path) as pool:
        result = pool.run(MsgRun(
            command=["sh", "-c", "echo modified > f.txt"],
            files={"f.txt": "original"},
            return_files=["f.txt"],
        ))
    assert result.success
    assert result.files["f.txt"].strip() == "modified"


def test_return_files_without_workspace(tmp_path):
    remote_dir = tmp_path / "project"
    remote_dir.mkdir()

    with _local_pool(tmp_path, remote_dir=str(remote_dir)) as pool:
        result = pool.run(MsgRun(
            command=["sh", "-c", "echo noworkspace > out.txt"],
            return_files=["out.txt"],
        ))
    assert result.success
    assert result.files["out.txt"].strip() == "noworkspace"


def test_concurrent_slots(tmp_path):
    """Two-slot pool: both jobs run concurrently and outputs don't mix."""
    results = {}
    errors = []

    def run_job(pool, tag):
        try:
            r = pool.run(MsgRun(command=["sh", "-c", f"echo {tag}"]))
            results[tag] = r
        except Exception as e:
            errors.append(e)

    with _local_pool(tmp_path, jobs=2) as pool:
        threads = [
            threading.Thread(target=run_job, args=(pool, "alpha")),
            threading.Thread(target=run_job, args=(pool, "beta")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors
    assert results["alpha"].stdout.strip() == "alpha"
    assert results["beta"].stdout.strip() == "beta"
