"""Device-health probe and worker-health reporting.

A GPU can enumerate in nvidia-smi (so it is advertised by a naive worker) while
being unusable for compute -- e.g. after a fault that sets ``GPU Recovery Action:
Reset``.  Every CUDA context creation on it then fails, so each job dispatched
there dies in its first ``.cuda()`` call.  These tests cover the probe that keeps
such a device out of the scheduler and the reporting that surfaces it in status.
"""
import subprocess

from mlsweep import _topology
from mlsweep._manager_db import WorkerRecord
from mlsweep._manager_http import _enrich_worker
from mlsweep._manager_state import ManagerState, WorkerConn
from mlsweep._shared import MsgWorkerHello, from_obj
from mlsweep._topology import probe_usable_devices


# ── probe_usable_devices ──────────────────────────────────────────────────────

def test_probe_script_is_valid_python():
    # The probe source is passed to `python -c`; a stray escape (e.g. an interpreted
    # newline) makes it a SyntaxError, and then every device looks unhealthy.
    compile(_topology._CUDA_PROBE, "<cuda-probe>", "exec")


class _FakeProc:
    def __init__(self, rc: int | None) -> None:
        self.rc = rc  # None = hangs past the deadline
        self.killed = False

    def wait(self, timeout=None):
        if self.rc is None and not self.killed:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        return self.rc

    def kill(self) -> None:
        self.killed = True


def _fake_popen(monkeypatch, rc_for):
    """Patch Popen; *rc_for* maps the probe env to an exit code (None = hang)."""
    envs: list[dict[str, str]] = []

    def fake(cmd, env=None, **kwargs):
        envs.append(env)
        return _FakeProc(rc_for(env))

    monkeypatch.setattr(_topology.subprocess, "Popen", fake)
    return envs


def test_probe_splits_on_exit_code(monkeypatch):
    envs = _fake_popen(monkeypatch, lambda env: 0 if env["CUDA_VISIBLE_DEVICES"] in ("0", "2") else 1)
    assert probe_usable_devices([0, 1, 2, 3]) == ([0, 2], [1, 3])
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["0", "1", "2", "3"]


def test_probe_timeout_counts_as_unhealthy(monkeypatch):
    _fake_popen(monkeypatch, lambda env: None if env["CUDA_VISIBLE_DEVICES"] == "1" else 0)
    assert probe_usable_devices([0, 1], timeout=0.01) == ([0], [1])


def test_probe_sets_cvd_and_clears_hip(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    envs = _fake_popen(monkeypatch, lambda env: 0)
    assert probe_usable_devices([3]) == ([3], [])
    assert envs[0]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "HIP_VISIBLE_DEVICES" not in envs[0]


# ── protocol compatibility ────────────────────────────────────────────────────

def test_worker_hello_unhealthy_gpus_defaults_to_empty():
    # Old workers omit the field; from_obj must still build the message.
    msg = from_obj({
        "t": "whello", "gpus": [0, 1], "topo": {}, "resuming": [],
        "scratch_dir": "/tmp", "protocol": 2,
    })
    assert isinstance(msg, MsgWorkerHello)
    assert msg.unhealthy_gpus == []


def test_worker_hello_roundtrips_unhealthy_gpus():
    msg = MsgWorkerHello(gpus=[0, 1, 3], topo={}, resuming=[], scratch_dir="/tmp",
                         unhealthy_gpus=[2], protocol=2)
    back = from_obj({**msg.__dict__, "t": "whello"})
    assert back.unhealthy_gpus == [2]


# ── _enrich_worker reporting ──────────────────────────────────────────────────

def test_enrich_worker_reports_live_health():
    state = ManagerState()
    state.workers["w1"] = WorkerConn(worker_id="w1", host="h", port=0,
                                     gpus=[0, 1, 3], unhealthy_gpus=[2])
    rec = WorkerRecord(worker_id="w1", host="h", remote_dir="/",
                       devices="[0, 1, 3]", unhealthy_devices="[2]")
    d = _enrich_worker(rec, state)
    assert d["gpus"] == [0, 1, 3]
    assert d["unhealthy_gpus"] == [2]


def test_enrich_worker_offline_parses_db_health():
    state = ManagerState()
    rec = WorkerRecord(worker_id="w2", host="h", remote_dir="/",
                       devices="[0, 1]", unhealthy_devices="[2, 3]")
    d = _enrich_worker(rec, state)
    assert d["gpus"] == [0, 1]
    assert d["unhealthy_gpus"] == [2, 3]


def test_enrich_worker_offline_without_health():
    state = ManagerState()
    rec = WorkerRecord(worker_id="w3", host="h", remote_dir="/", devices="[0]")
    d = _enrich_worker(rec, state)
    assert d["unhealthy_gpus"] == []
