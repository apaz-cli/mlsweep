"""Tests for the default run environment (mlsweep._env) and the worker's use of it."""

import os
import socket
import subprocess
import sys
import threading

import pytest

from mlsweep import _env
from mlsweep._shared import (
    MsgHello, MsgLog, MsgResult, MsgRun, MsgWorkerHello, decode, encode, read_msg,
)

_TOKEN = "env-token"

_PYPROJECT = """\
[project]
name = "demo"
version = "0"
dependencies = []
"""


def _write(d, name, text):
    with open(os.path.join(d, name), "w") as f:
        f.write(text)


# ── detect ────────────────────────────────────────────────────────────────────


def test_detect_pyproject_static_deps_is_cached(tmp_path):
    _write(tmp_path, "pyproject.toml",
           '[project]\nname = "x"\nversion = "0"\ndependencies = ["b", "a"]\n')
    spec = _env.detect(str(tmp_path))
    assert spec is not None and spec.install == ["a", "b"] and spec.cache_key


def test_detect_key_ignores_dependency_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write(a, "pyproject.toml", '[project]\nname = "x"\nversion = "0"\ndependencies = ["p", "q"]\n')
    _write(b, "pyproject.toml", '[project]\nname = "y"\nversion = "1"\ndependencies = ["q", "p"]\n')
    assert _env.detect(str(a)).cache_key == _env.detect(str(b)).cache_key


def test_detect_dynamic_deps_installs_project_per_run(tmp_path):
    _write(tmp_path, "pyproject.toml",
           '[project]\nname = "x"\nversion = "0"\ndynamic = ["dependencies"]\n')
    spec = _env.detect(str(tmp_path))
    assert spec is not None and spec.install == [str(tmp_path)] and spec.cache_key is None


def test_detect_pyproject_without_project_table_falls_through(tmp_path):
    _write(tmp_path, "pyproject.toml", "[tool.ruff]\nline-length = 100\n")
    _write(tmp_path, "requirements.txt", "numpy\n")
    spec = _env.detect(str(tmp_path))
    assert spec is not None and spec.source == "requirements.txt" and spec.cache_key


@pytest.mark.parametrize("line", ["-e .", "-r other.txt", "./pkg", "file:///x"])
def test_detect_requirements_with_local_refs_not_cached(tmp_path, line):
    _write(tmp_path, "requirements.txt", f"numpy\n{line}\n")
    assert _env.detect(str(tmp_path)).cache_key is None


def test_detect_setup_py(tmp_path):
    _write(tmp_path, "setup.py", "")
    spec = _env.detect(str(tmp_path))
    assert spec is not None and spec.source == "setup.py" and spec.cache_key is None


def test_detect_nothing(tmp_path):
    assert _env.detect(str(tmp_path)) is None


# ── ensure ────────────────────────────────────────────────────────────────────


def _run(calls):
    def run(cmd, cwd):
        calls.append(cmd)
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)
    return run


def test_ensure_builds_once_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSWEEP_ENV_CACHE", str(tmp_path / "cache"))
    _write(tmp_path, "pyproject.toml", _PYPROJECT)
    spec = _env.detect(str(tmp_path))
    calls, logs = [], []
    first = _env.ensure(spec, str(tmp_path), str(tmp_path), _run(calls), logs.append)
    second = _env.ensure(spec, str(tmp_path), str(tmp_path), _run(calls), logs.append)
    assert first == second and first.startswith(str(tmp_path / "cache"))
    assert os.path.isfile(os.path.join(first, "bin", "python"))
    assert len(calls) == 1  # venv creation only; nothing to pip install
    assert "building cached env" in logs[0] and "using cached env" in logs[1]


def test_ensure_concurrent_runs_build_once(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSWEEP_ENV_CACHE", str(tmp_path / "cache"))
    _write(tmp_path, "pyproject.toml", _PYPROJECT)
    spec = _env.detect(str(tmp_path))
    calls, results = [], []
    threads = [
        threading.Thread(target=lambda: results.append(
            _env.ensure(spec, str(tmp_path), str(tmp_path), _run(calls), lambda _: None)))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1 and len(calls) == 1


def test_ensure_discards_partial_build(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSWEEP_ENV_CACHE", str(tmp_path / "cache"))
    _write(tmp_path, "pyproject.toml", _PYPROJECT)
    spec = _env.detect(str(tmp_path))
    partial = tmp_path / "cache" / spec.cache_key
    partial.mkdir(parents=True)
    (partial / "junk").write_text("left by a build that died")
    venv = _env.ensure(spec, str(tmp_path), str(tmp_path), _run([]), lambda _: None)
    assert not os.path.exists(os.path.join(venv, "junk"))
    assert os.path.isfile(os.path.join(venv, _env.MARKER))


def test_ensure_uncacheable_builds_in_run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSWEEP_ENV_CACHE", str(tmp_path / "cache"))
    _write(tmp_path, "requirements.txt", "-r other.txt\n")
    _write(tmp_path, "other.txt", "")
    spec = _env.detect(str(tmp_path))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    venv = _env.ensure(spec, str(tmp_path), str(run_dir), _run([]), lambda _: None)
    assert venv == str(run_dir / "venv")
    assert not (tmp_path / "cache").exists()


# ── worker ────────────────────────────────────────────────────────────────────


@pytest.fixture
def worker(tmp_path):
    """A worker started outside any venv, with its env cache under tmp_path."""
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    env["MLSWEEP_ENV_CACHE"] = str(tmp_path / "cache")
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlsweep.worker", "--port", "0",
         "--scratch-dir", str(tmp_path / "scratch"), "--token", _TOKEN,
         "--remote-dir", str(remote_dir)],
        stdout=subprocess.PIPE, text=True, env=env,
    )
    try:
        line = proc.stdout.readline()
        assert line.startswith("PORT="), line
        sock = socket.create_connection(("127.0.0.1", int(line[5:])), timeout=60)
        sock.sendall(encode(MsgHello(token=_TOKEN, controller_id="test")))
        _recv_until(sock, lambda m: isinstance(m, MsgWorkerHello))
        yield sock
        sock.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _recv_until(sock, done):
    """Read messages until one satisfies *done*; return it."""
    while True:
        payload = read_msg(sock)
        assert payload is not None, "worker closed the connection"
        m = decode(payload)
        if done(m):
            return m


def _run_to_result(sock, msg):
    """Send *msg*; return (MsgResult, the run's log text)."""
    sock.sendall(encode(msg))
    log = []

    def done(m):
        if isinstance(m, MsgLog) and m.run_id == msg.run_id:
            log.append(m.data)
        return isinstance(m, MsgResult) and m.run_id == msg.run_id

    return _recv_until(sock, done), "".join(log)


def test_worker_builds_default_env_then_reuses_it(worker, tmp_path):
    files = {"pyproject.toml": _PYPROJECT,
             "main.py": "import sys; print('PREFIX=' + sys.prefix)\n"}
    cache = str(tmp_path / "cache")
    for i, expect in enumerate(["building cached env", "using cached env"]):
        res, log = _run_to_result(worker, MsgRun(
            command=["python", "main.py"], run_id=f"r{i}", experiment="e", files=files))
        assert res.success, log
        assert expect in log
        prefix = log.split("PREFIX=", 1)[1].split()[0]
        assert prefix.startswith(cache), log


def test_worker_logs_spawn_failure(worker):
    res, log = _run_to_result(worker, MsgRun(
        command=["mlsweep-no-such-binary"], run_id="bad", experiment="e",
        files={"main.py": ""}))
    assert not res.success and res.exit_code == -1
    assert "could not start 'mlsweep-no-such-binary'" in log
