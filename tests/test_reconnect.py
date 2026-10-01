"""End-to-end tests for the manager↔worker reconnect and resume paths.

These run a real manager and a real worker.  To drop the TCP connection between them
without root, the worker is reached through a small TCP proxy: the test holds a shared
flock on the worker's port lock file, so the worker the manager launches for that port
takes the "port already served" path and the manager connects to the proxy instead.

Covered:
  - one dropped connection leads to exactly one reconnect (no reconnect loop), and the
    abandoned socket is closed;
  - a run that finishes after the reconnect reports its result on the new connection;
  - a run that finishes while no manager is connected is not re-run: its result is kept
    by the worker and reported on the next hello;
  - after a manager restart, a resumed run keeps the GPU it is actually running on.
"""

import asyncio
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from conftest import _api_get, _api_post, _experiment_jobs, _find_free_port
from mlsweep._shared import MsgWorkerHello, decode, encode

TOKEN = "reconnect-test-token"


# ── helpers ────────────────────────────────────────────────────────────────────


def _api(url: str, path: str, body: dict | None = None, token: str = TOKEN):
    if body is not None:
        return _api_post(url, token, path, body)
    return _api_get(url, token, path)


def _jobs(url: str) -> dict[str, dict]:
    return {j["run_id"]: j for j in _experiment_jobs(url, TOKEN, "rc")}


def _wait(pred, timeout: float, what: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            v = pred()
            if v:
                return v
        except Exception:
            pass
        time.sleep(0.25)
    pytest.fail(f"timed out waiting for {what}")


def _sleeper(seconds: int) -> list[str]:
    return [sys.executable, "-c",
            f"import time\nfor i in range({seconds}): print(i, flush=True); time.sleep(1)"]


def _start_manager(tmp_path, port: int, workers_file: str, gpus: str = "0") -> subprocess.Popen:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": gpus}
    return subprocess.Popen(
        [sys.executable, "-m", "mlsweep.manager", "--port", str(port),
         "--db", str(tmp_path / "m.db"), "--mlsweep-dir", str(tmp_path / "out"),
         "--token", TOKEN, "--workers", workers_file],
        stdout=open(tmp_path / f"manager_{port}.log", "w"), stderr=subprocess.STDOUT, env=env,
    )


def _connected_worker(url: str):
    ws = [w for w in _api(url, "/api/workers") if w["status"] == "connected"]
    return ws[0] if ws else None


def _process_gpu(run_id: str) -> str | None:
    """CUDA_VISIBLE_DEVICES of the live process running *run_id*."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        if f"MLSWEEP_RUN_NAME={run_id}".encode() in env:
            for kv in env:
                if kv.startswith(b"CUDA_VISIBLE_DEVICES="):
                    return kv.split(b"=", 1)[1].decode()
    return None


class _Proxy:
    """TCP proxy on its own event-loop thread.  ``sever()`` drops every connection;
    ``refuse_for(s)`` rejects new connections for *s* seconds."""

    def __init__(self, listen_port: int, target_port: int):
        self.listen_port, self.target_port = listen_port, target_port
        self.accepted = 0
        self.open = 0
        self._writers: list = []
        self._refuse_until = 0.0
        self.loop = asyncio.new_event_loop()
        started = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(started,), daemon=True)
        self._thread.start()
        started.wait(5)

    def _run(self, started):
        asyncio.set_event_loop(self.loop)
        self.server = self.loop.run_until_complete(
            asyncio.start_server(self._handle, "127.0.0.1", self.listen_port))
        started.set()
        self.loop.run_forever()

    async def _handle(self, cr, cw):
        if time.time() < self._refuse_until:
            cw.transport.abort()
            return
        self.accepted += 1
        ur, uw = await asyncio.open_connection("127.0.0.1", self.target_port)
        self._writers += [cw, uw]
        self.open += 1

        async def pipe(r, w):
            try:
                while data := await r.read(65536):
                    w.write(data)
                    await w.drain()
            except Exception:
                pass
            finally:
                try:
                    w.close()
                except Exception:
                    pass

        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
        self.open -= 1

    def sever(self):
        def _abort():
            for w in self._writers:
                try:
                    w.transport.abort()
                except Exception:
                    pass
            self._writers.clear()
        self.loop.call_soon_threadsafe(_abort)

    def refuse_for(self, seconds: float):
        self._refuse_until = time.time() + seconds

    def close(self):
        self.loop.call_soon_threadsafe(self.server.close)
        self.loop.call_soon_threadsafe(self.loop.stop)


@pytest.fixture
def proxied(tmp_path):
    """Manager → proxy → real worker.  Yields (url, proxy)."""
    P, Q, M = _find_free_port(), _find_free_port(), _find_free_port()
    proj = tmp_path / "proj"
    proj.mkdir()
    lock = open(f"/tmp/.mlsweep_worker_port_{P}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_SH)      # the manager's own worker launch for P defers to us
    worker = subprocess.Popen(
        [sys.executable, "-m", "mlsweep.worker", "--token", TOKEN, "--port", str(Q),
         "--remote-dir", str(proj), "--scratch-dir", str(tmp_path / "scratch")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert worker.stdout.readline().startswith(b"PORT=")
    proxy = _Proxy(P, Q)
    wf = tmp_path / "workers.toml"
    wf.write_text(f'[[workers]]\nhost = "localhost"\nremote_dir = "{proj}"\nport = {P}\n')
    mgr = _start_manager(tmp_path, M, str(wf))
    url = f"http://127.0.0.1:{M}"
    try:
        _wait(lambda: _connected_worker(url), 30, "worker to connect")
        _api(url, "/api/experiments", {"experiment_id": "rc"})
        yield url, proxy
    finally:
        mgr.terminate()
        mgr.wait()
        worker.terminate()
        worker.wait()
        proxy.close()
        lock.close()


# ── protocol ───────────────────────────────────────────────────────────────────


def test_decode_ignores_unknown_fields_and_defaults_missing_ones():
    hello = MsgWorkerHello(gpus=[0], topo={}, resuming=[], scratch_dir="/s",
                           completed=[{"run_id": "r"}])
    obj = json.loads(encode(hello)[4:])
    obj["added_in_a_future_version"] = 1
    assert decode(json.dumps(obj).encode()).completed == [{"run_id": "r"}]
    obj.pop("completed")                  # a worker from before the field existed
    assert decode(json.dumps(obj).encode()).completed == []


# ── reconnect ──────────────────────────────────────────────────────────────────


def test_dropped_connection_reconnects_once_and_keeps_result(proxied):
    url, proxy = proxied
    _api(url, "/api/jobs", {"run_id": "j1", "experiment_id": "rc", "gpus_per_run": 0,
                            "command": _sleeper(12)})
    _wait(lambda: _jobs(url)["j1"]["status"] == "running", 20, "j1 to start")
    time.sleep(2)
    proxy.sever()

    _wait(lambda: _jobs(url)["j1"]["status"] in ("done", "failed"), 40, "j1 to finish")
    j1 = _jobs(url)["j1"]
    assert (j1["status"], j1["exit_code"], j1["retry_count"]) == ("done", 0, 0)

    time.sleep(8)                         # a reconnect loop would keep opening connections
    assert _connected_worker(url) is not None
    assert proxy.accepted == 2            # the original connection plus one reconnect
    assert proxy.open == 1                # the abandoned connection was closed


def test_result_produced_while_disconnected_is_not_rerun(proxied):
    url, proxy = proxied
    _api(url, "/api/jobs", {"run_id": "j2", "experiment_id": "rc", "gpus_per_run": 0,
                            "command": _sleeper(4)})
    _wait(lambda: _jobs(url)["j2"]["status"] == "running", 20, "j2 to start")
    proxy.refuse_for(9)                   # keep the manager out until after j2 has ended
    proxy.sever()

    _wait(lambda: _jobs(url)["j2"]["status"] in ("done", "failed"), 60, "j2 to finish")
    j2 = _jobs(url)["j2"]
    assert (j2["status"], j2["exit_code"], j2["retry_count"]) == ("done", 0, 0)
    assert _connected_worker(url) is not None


# ── manager restart ────────────────────────────────────────────────────────────


@pytest.fixture
def restartable(tmp_path):
    """Manager with a local 2-GPU worker on a fixed port, so a new manager reattaches."""
    Q = _find_free_port()
    proj = tmp_path / "proj"
    proj.mkdir()
    wf = tmp_path / "workers.toml"
    wf.write_text(f'[[workers]]\nhost = "localhost"\nremote_dir = "{proj}"\nport = {Q}\n'
                  "devices = [0, 1]\n")
    managers: list[subprocess.Popen] = []

    def start():
        M = _find_free_port()
        managers.append(_start_manager(tmp_path, M, str(wf), gpus="0,1"))
        url = f"http://127.0.0.1:{M}"
        _wait(lambda: _connected_worker(url), 30, "worker to connect")
        return url

    try:
        yield start, managers
    finally:
        for m in managers:
            if m.poll() is None:
                m.terminate()
                m.wait()
        subprocess.run(["pkill", "-f", f"mlsweep.worker .*--port {Q}"], check=False)


def test_manager_restart_keeps_the_gpu_a_run_is_actually_on(restartable):
    start, managers = restartable
    url = start()
    _api(url, "/api/experiments", {"experiment_id": "rc"})
    _api(url, "/api/jobs", {"run_id": "A", "experiment_id": "rc", "command": _sleeper(3)})
    time.sleep(1)
    _api(url, "/api/jobs", {"run_id": "B", "experiment_id": "rc", "command": _sleeper(60)})
    _wait(lambda: _process_gpu("B") is not None, 20, "B to start")
    _wait(lambda: _jobs(url)["A"]["status"] == "done", 30, "A to finish")
    b_gpu = _process_gpu("B")

    managers[-1].send_signal(signal.SIGKILL)   # crash; the worker keeps B running
    managers[-1].wait()
    url = start()

    booked = _wait(lambda: _jobs(url)["B"].get("dispatched_gpu_ids"), 20, "B to be restored")
    assert json.loads(booked) == [int(b_gpu)]
    _api(url, "/api/jobs", {"run_id": "C", "experiment_id": "rc", "command": _sleeper(10)})
    c_gpu = _wait(lambda: _process_gpu("C"), 20, "C to start")
    assert c_gpu != b_gpu                      # C must not be stacked onto B's GPU


def test_run_that_ends_while_manager_is_down_is_not_rerun(restartable, tmp_path):
    start, managers = restartable
    url = start()
    _api(url, "/api/experiments", {"experiment_id": "rc"})
    starts = tmp_path / "d_starts.txt"          # one line per time D is launched
    cmd = [sys.executable, "-c",
           f"open({str(starts)!r}, 'a').write('start\\n')\n"
           "import time\nfor i in range(4): print(i, flush=True); time.sleep(1)"]
    _api(url, "/api/jobs", {"run_id": "D", "experiment_id": "rc", "command": cmd})
    _wait(lambda: _jobs(url)["D"]["status"] == "running", 20, "D to start")

    managers[-1].send_signal(signal.SIGKILL)
    managers[-1].wait()
    time.sleep(6)                              # D ends with no manager connected
    url = start()

    _wait(lambda: _jobs(url)["D"]["status"] in ("done", "failed"), 30, "D to be finished")
    time.sleep(6)                              # long enough for a wrongful re-run to show up
    d = _jobs(url)["D"]
    assert (d["status"], d["exit_code"]) == ("done", 0)
    assert starts.read_text().count("start") == 1   # D ran exactly once
