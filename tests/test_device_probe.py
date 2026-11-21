"""Device-health probe and worker-health reporting.

A GPU can enumerate in nvidia-smi (so it is advertised by a naive worker) while
being unusable for compute -- e.g. after a fault that sets ``GPU Recovery Action:
Reset``.  Every CUDA context creation on it then fails, so each job dispatched
there dies in its first ``.cuda()`` call.  These tests cover the probe that keeps
such a device out of the scheduler and the reporting that surfaces it in status.
"""
import asyncio
import subprocess

from mlsweep import _topology, worker
from mlsweep._manager_db import WorkerRecord
from mlsweep._manager_http import _enrich_worker
from mlsweep._manager_state import ManagerState, WorkerConn
from mlsweep._manager_workers import _find_gpu_group, _handle_gpu_health
from mlsweep._shared import MsgGpuHealth, MsgWorkerHello, from_obj
from mlsweep._topology import probe_devices


# ── probe_devices ──────────────────────────────────────────────────────

def test_probe_script_is_valid_python():
    # The probe source is passed to `python -c`; a stray escape (e.g. an interpreted
    # newline) makes it a SyntaxError, and then every device looks unhealthy.
    compile(_topology._CUDA_PROBE, "<cuda-probe>", "exec")


class _FakeProc:
    """rc None = hangs past the deadline; *out* is what the probe printed."""

    def __init__(self, rc: int | None, out: bytes = b"", err: bytes = b"") -> None:
        self.rc, self.out, self.err = rc, out, err
        self.returncode: int | None = None
        self.killed = False

    def communicate(self, timeout=None):
        if self.rc is None and not self.killed:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        self.returncode = -9 if self.killed else self.rc
        return self.out, self.err

    def kill(self) -> None:
        self.killed = True


def _fake_popen(proc_for):
    """A stand-in Popen: ``(popen, envs)``.  *proc_for* maps the probe's device (str)
    to a _FakeProc, and *envs* collects each probe's environment."""
    envs: list[dict[str, str]] = []

    def fake(cmd, env=None, **kwargs):
        envs.append(env)
        return proc_for(env["CUDA_VISIBLE_DEVICES"])

    return fake, envs


def test_probe_reports_cuda_error():
    bad = b"cuInit failed: CUDA_ERROR_SYSTEM_NOT_READY (802)\n"
    popen, envs = _fake_popen(lambda d: _FakeProc(0) if d in ("0", "2") else _FakeProc(1, bad))
    assert probe_devices([0, 1, 2, 3], popen=popen) == {
        0: None, 1: "cuInit failed: CUDA_ERROR_SYSTEM_NOT_READY (802)",
        2: None, 3: "cuInit failed: CUDA_ERROR_SYSTEM_NOT_READY (802)",
    }
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["0", "1", "2", "3"]


def test_probe_timeout_is_reported_as_timeout():
    popen, _ = _fake_popen(lambda d: _FakeProc(None) if d == "1" else _FakeProc(0))
    assert probe_devices([0, 1], timeout=0.01, popen=popen) == {0: None, 1: "CUDA probe timed out after 0.01s"}


def test_probe_crash_reports_stderr_tail():
    popen, _ = _fake_popen(lambda d: _FakeProc(1, err=b"Traceback ...\nMemoryError\n"))
    assert probe_devices([0], popen=popen) == {0: "CUDA probe exited with code 1: MemoryError"}


def test_probe_that_cannot_start():
    def fake(cmd, env=None, **kwargs):
        raise OSError("no such interpreter")
    assert probe_devices([0], popen=fake) == {0: "probe could not start: no such interpreter"}


def test_probe_sets_cvd_and_clears_hip():
    popen, envs = _fake_popen(lambda d: _FakeProc(0))
    env = {"CUDA_VISIBLE_DEVICES": "7", "HIP_VISIBLE_DEVICES": "0", "PATH": "/bin"}
    assert probe_devices([3], popen=popen, env=env) == {3: None}
    assert envs[0]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "HIP_VISIBLE_DEVICES" not in envs[0]
    assert envs[0]["PATH"] == "/bin"


# ── worker health bookkeeping ─────────────────────────────────────────────────

def test_health_split_pending_usable_unhealthy():
    health = worker.GpuHealth()
    health.health = {0: None, 1: "cuInit failed: X (1)"}
    with health.cond:
        assert health.split_locked([0, 1, 2]) == ([0], [1], {"1": "cuInit failed: X (1)"}, [2])


def test_health_split_without_probe_trusts_every_gpu():
    health = worker.GpuHealth(enabled=False)
    with health.cond:
        assert health.split_locked([0, 1]) == ([0, 1], [], {}, [])


# ── manager placement ─────────────────────────────────────────────────────────

def test_placement_skips_unhealthy_and_pending_but_keeps_them_in_pool():
    wc = WorkerConn(worker_id="w", host="h", port=0, gpus=[0, 1, 2, 3],
                    unhealthy_gpus=[1], pending_gpus=[2])
    assert wc.usable_gpus == [0, 3]
    occ = {g: 0 for g in wc.gpus}
    assert _find_gpu_group(wc, 2, {}, occupancy=occ, spread=True) == [0, 3]
    assert _find_gpu_group(wc, 3, {}, occupancy=occ, spread=True) is None


def test_gpu_health_message_returns_recovered_gpu():
    state = ManagerState()
    writes: list[tuple[str, str]] = []

    class _Db:
        async def update_worker_health(self, worker_id, unhealthy):
            writes.append((worker_id, unhealthy))

    state.db_writer = _Db()  # type: ignore[assignment]
    wc = WorkerConn(worker_id="w", host="h", port=0, gpus=[0, 1], unhealthy_gpus=[1],
                    unhealthy_reasons={"1": "CUDA probe timed out after 60s"})
    asyncio.run(_handle_gpu_health(None, state, wc, MsgGpuHealth(gpus=[0, 1])))  # type: ignore[arg-type]
    assert wc.usable_gpus == [0, 1] and wc.unhealthy_reasons == {}
    assert writes == [("w", "[]")]


# ── protocol compatibility ────────────────────────────────────────────────────

def test_worker_hello_unhealthy_gpus_defaults_to_empty():
    # Old workers omit the field; from_obj must still build the message.
    msg = from_obj({
        "t": "whello", "gpus": [0, 1], "topo": {}, "resuming": [],
        "scratch_dir": "/tmp", "protocol": 2,
    })
    assert isinstance(msg, MsgWorkerHello)
    assert msg.unhealthy_gpus == []
    assert msg.pending_gpus == [] and msg.unhealthy_reasons == {}


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


def test_enrich_worker_live_lists_only_usable_pool_gpus():
    state = ManagerState()
    state.workers["w1"] = WorkerConn(worker_id="w1", host="h", port=0, gpus=[0, 1, 2, 3],
                                     unhealthy_gpus=[2], unhealthy_reasons={"2": "why"},
                                     pending_gpus=[3])
    rec = WorkerRecord(worker_id="w1", host="h", remote_dir="/")
    d = _enrich_worker(rec, state)
    assert d["gpus"] == [0, 1]
    assert d["unhealthy_reasons"] == {"2": "why"} and d["pending_gpus"] == [3]


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


def test_slow_probe_reports_pending_then_sends_health():
    """The hello does not wait out a slow probe; the result follows as MsgGpuHealth."""
    import queue
    import threading

    from mlsweep._shared import decode

    release = threading.Event()

    def slow_probe(devices):
        release.wait(10)
        return {0: None, 1: "CUDA probe timed out after 60s"}

    stop = threading.Event()
    health = worker.GpuHealth(candidates=lambda: [0, 1], probe=slow_probe)
    conn = worker.ConnState(sock=None, send_queue=queue.Queue(), ready=True)  # type: ignore[arg-type]

    t = threading.Thread(target=health.run, args=(stop, lambda: [conn]), daemon=True)
    t.start()
    with health.cond:
        assert health.split_locked([0, 1]) == ([], [], {}, [0, 1])
    release.set()
    msg = decode(conn.send_queue.get(timeout=10)[4:])
    assert isinstance(msg, MsgGpuHealth)
    assert (msg.gpus, msg.unhealthy_gpus, msg.pending_gpus) == ([0], [1], [])
    assert msg.unhealthy_reasons == {"1": "CUDA probe timed out after 60s"}
    stop.set()
    health.wake.set()
    t.join(5)
    assert not t.is_alive()
