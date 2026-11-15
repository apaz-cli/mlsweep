"""The SSH tunnel record/reap helpers in ``_manager_workers``.

The remote command records the per-connection sshd pid, so a later manager can
kill a reverse forward left behind by one that died without closing it.  These
tests run the generated shell scripts locally with a fake ``sshd`` binary.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

import pytest

from mlsweep._manager_workers import (
    _manager_identity,
    _reap_tunnels_script,
    _tunnel_remote_cmd,
)

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"),
                       reason="relies on Linux ps output and process names"),
    pytest.mark.xdist_group("tunnel_records"),
]

PORT = 7891


def _env(home):
    return {**os.environ, "HOME": str(home)}


def _shell(script, home, timeout=20):
    return subprocess.run(["sh", "-c", script], env=_env(home),
                          capture_output=True, text=True, timeout=timeout)


def _lstart(pid):
    # Matches the writer, which does `echo $(ps -o lstart= -p $PPID)`.
    out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                         capture_output=True, text=True).stdout
    return " ".join(out.split())


def _record(home, pid, machine, instance, host="worker-host"):
    path = os.path.join(home, ".mlsweep", "tunnels", f"{PORT}.{pid}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{machine} {instance} {host}\n{_lstart(pid)}\n")
    return path


@pytest.fixture
def spawned():
    procs = []
    yield procs
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()


@pytest.fixture
def fake_sshd(tmp_path, spawned):
    sleep = shutil.which("sleep")
    if sleep is None:
        pytest.skip("no sleep binary")
    path = tmp_path / "bin" / "sshd"
    path.parent.mkdir()
    shutil.copy(sleep, path)
    path.chmod(0o755)

    def spawn():
        p = subprocess.Popen([str(path), "60"])
        spawned.append(p)
        for _ in range(100):
            comm = subprocess.run(["ps", "-o", "comm=", "-p", str(p.pid)],
                                  capture_output=True, text=True).stdout.strip()
            if comm == "sshd":
                return p
            time.sleep(0.01)
        p.kill()
        p.wait()
        pytest.skip("copied sleep binary is not named sshd")

    return spawn


def test_remote_command_records_and_removes_its_tunnel(tmp_path):
    proc = subprocess.Popen(["sh", "-c", _tunnel_remote_cmd(PORT)], env=_env(tmp_path),
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        found = []
        lines = []
        for _ in range(200):
            found = glob.glob(os.path.join(tmp_path, ".mlsweep", "tunnels", f"{PORT}.*"))
            if found:
                lines = open(found[0]).read().splitlines()
                if len(lines) >= 2:
                    break
            time.sleep(0.01)
        assert found, "the remote command never wrote its record"
        assert len(lines) >= 2, lines
        assert lines[0].split()[:2] == list(_manager_identity())
        assert lines[1].strip()
        proc.stdin.close()
        proc.wait(timeout=10)
        assert not os.path.exists(found[0])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_reaper_kills_a_stale_sshd_from_this_machine(tmp_path, fake_sshd):
    victim = fake_sshd()
    machine, _ = _manager_identity()
    path = _record(tmp_path, victim.pid, machine, "old-boot:123")
    r = _shell(_reap_tunnels_script(PORT), tmp_path)
    victim.wait(timeout=10)
    assert f"killed {victim.pid}" in r.stdout
    assert not os.path.exists(path)


def test_reaper_spares_another_machines_tunnel(tmp_path, fake_sshd):
    holder = fake_sshd()
    path = _record(tmp_path, holder.pid, "other-machine", "x", host="laptop")
    r = _shell(_reap_tunnels_script(PORT), tmp_path)
    assert "held laptop" in r.stdout
    assert holder.poll() is None
    assert os.path.exists(path)
    os.remove(path)


def test_reaper_drops_records_for_reused_or_dead_pids(tmp_path, spawned):
    sleep = shutil.which("sleep")
    if sleep is None:
        pytest.skip("no sleep binary")
    sleeper = subprocess.Popen([sleep, "60"])
    spawned.append(sleeper)
    machine, _ = _manager_identity()
    path = _record(tmp_path, sleeper.pid, machine, "old-boot")
    r = _shell(_reap_tunnels_script(PORT), tmp_path)
    assert not os.path.exists(path)
    assert sleeper.poll() is None
    assert "killed" not in r.stdout

    missing = _record(tmp_path, 99999999, machine, "old-boot")
    r = _shell(_reap_tunnels_script(PORT), tmp_path)
    assert not os.path.exists(missing)
    assert r.returncode == 0


def test_reaper_is_a_no_op_without_a_record_dir(tmp_path):
    r = _shell(_reap_tunnels_script(PORT), tmp_path)
    assert r.returncode == 0
    assert r.stdout == ""


def test_reaper_reports_this_managers_own_tunnel(tmp_path, fake_sshd):
    holder = fake_sshd()
    path = _record(tmp_path, holder.pid, *_manager_identity())
    r = _shell(_reap_tunnels_script(PORT), tmp_path)
    assert f"own {holder.pid}" in r.stdout
    assert holder.poll() is None
    os.remove(path)
