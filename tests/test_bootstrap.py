"""Tests for mlsweep worker wheel bootstrap logic.

Covers ``WheelCache.ensure`` (reusing, building and failing to build the
worker wheel), the defaults of ``_worker_candidates``, and that
``_worker_shell_cmd`` generates valid bash.

The wheel tests point the wheels directory at tmp_path and stand in for pip,
so they never touch the checkout.  Other tests pack the checkout into
artifacts while they run, and a manager running from it ships its wheel to
remote workers.
"""

import asyncio
import importlib.metadata
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from mlsweep import _manager_workers
from mlsweep._shared import code_fingerprint
from mlsweep._manager_workers import WheelCache, _worker_candidates, _worker_shell_cmd

VERSION = importlib.metadata.version("mlsweep")
STAMP = f"{VERSION} {code_fingerprint()}"  # the code on disk, which a build ships


@pytest.fixture
def wheels(tmp_path):
    """A wheel cache in a temporary dir.  ``pip(ok)`` makes its pip a stand-in that
    builds into it (or fails), and ``calls`` collects the pip command lines."""
    d = tmp_path / "_wheels"
    calls: list[list[str]] = []
    cache = WheelCache(d, run=lambda cmd, **kw: pytest.fail("pip was not expected to run"))

    def pip(ok: bool, seconds: float = 0.0) -> None:
        def run(cmd, **kwargs):
            calls.append(cmd)
            time.sleep(seconds)
            if ok:
                (d / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, b"", b"" if ok else b"no")
        cache.run = run

    return SimpleNamespace(dir=d, calls=calls, pip=pip, cache=cache)


def test_ensure_worker_wheels_builds_and_records_the_version(wheels):
    wheels.dir.mkdir()
    (wheels.dir / "mlsweep-0.0.1-py3-none-any.whl").write_bytes(b"old")
    wheels.pip(ok=True)
    wheels.cache.ensure()
    assert len(wheels.calls) == 1 and "wheel" in wheels.calls[0]
    assert [p.name for p in wheels.dir.glob("*.whl")] == [f"mlsweep-{VERSION}-py3-none-any.whl"]
    assert (wheels.dir / ".complete").read_text() == STAMP


def test_ensure_worker_wheels_reuses_a_current_wheel(wheels):
    wheels.dir.mkdir()
    (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
    (wheels.dir / ".complete").write_text(STAMP)
    wheels.pip(ok=True)
    wheels.cache.ensure()
    assert wheels.calls == []


def test_ensure_worker_wheels_rebuilds_after_a_version_change(wheels):
    wheels.dir.mkdir()
    (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
    (wheels.dir / ".complete").write_text("0.0.1")
    wheels.pip(ok=True)
    wheels.cache.ensure()
    assert len(wheels.calls) == 1
    assert (wheels.dir / ".complete").read_text() == STAMP


def test_managers_starting_together_build_the_wheel_once(wheels):
    wheels.pip(ok=True, seconds=0.5)
    threads = [threading.Thread(target=wheels.cache.ensure) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wheels.calls) == 1
    assert (wheels.dir / ".complete").read_text() == STAMP


def test_ensure_worker_wheels_survives_a_failed_build(wheels):
    wheels.pip(ok=False)
    wheels.cache.ensure()
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


def test_a_code_change_rebuilds_a_wheel_of_the_same_version(wheels):
    """A source checkout changes code without changing its version, so the
    sentinel also records the code fingerprint and a mismatch rebuilds."""
    wheels.dir.mkdir()
    (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"old code")
    (wheels.dir / ".complete").write_text(f"{VERSION} 0000000000000000")
    wheels.pip(ok=True)
    wheels.cache.ensure()
    assert len(wheels.calls) == 1
    assert (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").read_bytes() == b"wheel"


def _fake_ssh(version_output: bytes, pip_stderr: bytes | None = None):
    """A stand-in for spawning ssh/scp: ``(spawn, remote)``, where *remote* collects
    the remote commands run, in order.  With *pip_stderr*, the install step fails
    and prints it."""
    remote: list[str] = []

    async def exec_(*argv, **kwargs):
        remote.append(argv[-1])
        out = version_output if "--version" in argv[-1] else b""
        fail = pip_stderr is not None and "pip install" in argv[-1]
        proc = SimpleNamespace(returncode=1 if fail else 0)

        async def communicate():
            return out, pip_stderr if fail else b""
        proc.communicate = communicate
        return proc
    return exec_, remote


def _current_wheel(wheels) -> None:
    """A built wheel whose stamp matches the code on disk, so a bootstrap ships it as is."""
    wheels.dir.mkdir()
    (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
    (wheels.dir / ".complete").write_text(STAMP)


def test_bootstrap_reuses_a_venv_with_the_same_code(wheels):
    _current_wheel(wheels)
    spawn, remote = _fake_ssh(f"mlsweep_worker {STAMP}".encode())
    asyncio.run(_manager_workers._bootstrap_worker_venv("gpu1", wheels=wheels.cache, spawn=spawn))
    assert len(remote) == 1                # only the version check


def test_bootstrap_reinstalls_a_venv_of_the_same_version_with_other_code(wheels):
    """A source checkout changes code without changing its version."""
    _current_wheel(wheels)
    for reported in (f"mlsweep_worker {VERSION}", f"mlsweep_worker {VERSION} 0000000000000000"):
        spawn, remote = _fake_ssh(reported.encode())
        asyncio.run(_manager_workers._bootstrap_worker_venv("gpu1", wheels=wheels.cache, spawn=spawn))
        assert any(c.startswith("rm -rf /tmp/mlsweep_venv") for c in remote)


def test_bootstrap_rebuilds_a_stale_wheel_before_shipping_it(wheels):
    """Worker code edited after the manager started reaches the next launch."""
    wheels.dir.mkdir()
    (wheels.dir / f"mlsweep-{VERSION}-py3-none-any.whl").write_bytes(b"old code")
    (wheels.dir / ".complete").write_text(f"{VERSION} 0000000000000000")
    wheels.pip(ok=True)
    spawn, _ = _fake_ssh(b"MISSING")
    asyncio.run(_manager_workers._bootstrap_worker_venv("gpu1", wheels=wheels.cache, spawn=spawn))
    assert len(wheels.calls) == 1
    assert (wheels.dir / ".complete").read_text() == STAMP


def test_bootstrap_installs_only_the_wheel_it_ships(wheels):
    """A wheel an older manager left in /tmp/mlsweep_wheels made pip refuse to
    install two versions of mlsweep at once, so every bootstrap failed."""
    _current_wheel(wheels)
    spawn, remote = _fake_ssh(b"MISSING")
    asyncio.run(_manager_workers._bootstrap_worker_venv("gpu1", wheels=wheels.cache, spawn=spawn))
    assert "rm -rf /tmp/mlsweep_wheels && mkdir -p /tmp/mlsweep_wheels" in remote
    [install] = [c for c in remote if "pip install" in c]
    assert install.endswith(f"pip install /tmp/mlsweep_wheels/mlsweep-{VERSION}-py3-none-any.whl")


def test_a_failed_install_reports_pips_error(wheels):
    _current_wheel(wheels)
    spawn, _ = _fake_ssh(b"MISSING", pip_stderr=(
        b"Processing /tmp/mlsweep_wheels/x.whl\n"
        b"ERROR: Could not find a version that satisfies the requirement aiohttp\n"))
    with pytest.raises(_manager_workers.BootstrapError) as e:
        asyncio.run(_manager_workers._bootstrap_worker_venv("gpu1", wheels=wheels.cache, spawn=spawn))
    assert str(e.value) == ("installing mlsweep failed on gpu1: "
                            "ERROR: Could not find a version that satisfies the requirement aiohttp")
