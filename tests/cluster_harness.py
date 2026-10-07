"""A real manager with real local workers, for race and scalability tests.

Every job runs ``journaled()``: a small script that appends ``start``, ``end``
or ``term`` (killed by SIGTERM) lines to a journal file.  From the journal the
tests check what really executed: how often each run started, which ran at
the same time, and on which worker and GPU.

Cleanup only touches processes whose command line names this cluster's own
temporary paths, never other mlsweep processes on the machine.
"""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from conftest import _api_get, _api_post, _api_request, _find_free_port
from test_reconnect import _start_manager, _wait

TOKEN = "reconnect-test-token"  # the token test_reconnect._start_manager passes

_JOB = r"""
import os, signal, sys, time
journal, schedule = sys.argv[1], [float(s) for s in sys.argv[2].split(",")]
term_delay = float(sys.argv[3])
exp, run = os.environ["EXP_EXPERIMENT"], os.environ["MLSWEEP_RUN_NAME"]
worker = os.path.basename(os.environ.get("MLSWEEP_WORKER_SOCKET", "-"))
gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or "-"
def write(ev):
    fd = os.open(journal, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, f"{ev} {exp} {run} {os.getpid()} {worker} {gpu} {time.time()}\n".encode())
    os.close(fd)
def term(*_):
    time.sleep(term_delay)
    write("term")
    os._exit(143)
signal.signal(signal.SIGTERM, term)
try:
    prior = sum(1 for l in open(journal) if l.split()[:3] == ["start", exp, run])
except OSError:
    prior = 0
write("start")
print(f"attempt {prior + 1}", flush=True)
end = time.time() + schedule[min(prior, len(schedule) - 1)]
while time.time() < end:
    time.sleep(0.05)
write("end")
"""


def journaled(journal: Path, *seconds: float, term_delay: float = 0.0) -> list[str]:
    """Command for a journaled job.  Attempt *n* runs ``seconds[n]`` seconds
    (the last value repeats).  On SIGTERM it lingers *term_delay* seconds, then
    journals ``term`` and exits."""
    return [sys.executable, "-c", _JOB, str(journal), ",".join(str(s) for s in seconds),
            str(term_delay)]


@dataclass(frozen=True)
class Span:
    """One execution of a run's training process."""

    experiment: str
    run: str
    pid: int
    worker: str
    gpu: str
    start: float
    stop: float | None  # None while still running
    how: str  # "end", "term" or "" (still running, or killed without SIGTERM)


def max_overlap(spans, key=lambda s: None) -> dict:
    """Largest number of *spans* running at the same instant, per ``key(span)``."""
    events: dict = {}
    for s in spans:
        k = key(s)
        events.setdefault(k, []).append((s.start, 1))
        events[k].append((s.stop if s.stop is not None else float("inf"), -1))
    out = {}
    for k, evs in events.items():
        cur = best = 0
        for _, d in sorted(evs, key=lambda e: (e[0], e[1])):  # stops sort before starts
            cur += d
            best = max(best, cur)
        out[k] = best
    return out


class Cluster:
    """A manager plus local workers on fixed ports, so a restarted manager reconnects to them.

    *workers* is one dict per worker with ``devices`` (list of GPU ids) and
    optionally ``jobs`` (max jobs per GPU, 0 = unlimited).
    """

    def __init__(self, tmp_path: Path, workers: list[dict]):
        self.tmp = tmp_path
        self.proj = tmp_path / "proj"
        self.proj.mkdir()
        self.journal = tmp_path / "journal.txt"
        self.tag = uuid.uuid4().hex[:8]
        self.ports = [_find_free_port() for _ in workers]
        entries = []
        for port, w in zip(self.ports, workers):
            entry = (f'[[workers]]\nhost = "localhost"\nremote_dir = "{self.proj}"\n'
                     f'port = {port}\ndevices = {list(w.get("devices", [0]))}\n')
            if "jobs" in w:
                entry += f'jobs = {w["jobs"]}\n'
            entries.append(entry)
        self.workers_file = tmp_path / "workers.toml"
        self.workers_file.write_text("".join(entries))
        self.managers: list[subprocess.Popen] = []
        self.url = ""
        try:
            self.start_manager()
        except BaseException:
            self.close()
            raise

    # ── lifecycle ──────────────────────────────────────────────────────────

    @property
    def worker_ids(self) -> list[str]:
        return [f"localhost:{p}" for p in self.ports]

    def start_manager(self, wait_workers: int | None = None) -> None:
        port = _find_free_port()
        self.managers.append(_start_manager(self.tmp, port, str(self.workers_file), gpus="0,1"))
        self.url = f"http://127.0.0.1:{port}"
        n = len(self.ports) if wait_workers is None else wait_workers
        _wait(lambda: self.connected() >= n, 60, f"{n} worker(s) to connect")

    def kill_manager(self) -> None:
        """SIGKILL the manager: no chance to clean up, like a crash."""
        m = self.managers[-1]
        m.send_signal(signal.SIGKILL)
        m.wait()

    def connected(self) -> int:
        return sum(w["status"] == "connected" for w in self.get("/api/workers"))

    def worker_pids(self, port: int) -> list[int]:
        return _pids_with(str(self.proj), "mlsweep.worker", f"--port {port}")

    def close(self) -> None:
        for m in self.managers:
            if m.poll() is None:
                m.terminate()
                try:
                    m.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    m.kill()
                    m.wait()
        for pid in _pids_with(str(self.proj)) + _pids_with(str(self.journal)):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    # ── API ────────────────────────────────────────────────────────────────

    def get(self, path):
        return _api_get(self.url, TOKEN, path)

    def get_text(self, path) -> str:
        req = urllib.request.Request(f"{self.url}{path}", headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()

    def post(self, path, body=None):
        return _api_post(self.url, TOKEN, path, body if body is not None else {})

    def put(self, path, body):
        return _api_request(self.url, TOKEN, "PUT", path, body)

    def patch(self, path, body):
        return _api_request(self.url, TOKEN, "PATCH", path, body)

    def delete(self, path):
        return _api_request(self.url, TOKEN, "DELETE", path)

    def status_code(self, method, path, body=None) -> int:
        """HTTP status of a request (instead of raising on 4xx/5xx)."""
        try:
            _api_request(self.url, TOKEN, method, path, body)
            return 200
        except urllib.error.HTTPError as e:
            return e.code

    def experiment(self, name: str, **fields) -> str:
        """Create an experiment with a name unique to this cluster; return its id.

        Workers this cluster's manager launches keep their scratch under its
        tmp_path.  Ids stay unique anyway, so a run directory is never shared
        with a worker started some other way.
        """
        eid = f"{name}_{self.tag}"
        self.post("/api/experiments", {"experiment_id": eid, **fields})
        return eid

    def job_body(self, eid, rid, *seconds, gpus=0, **fields) -> dict:
        return {"run_id": rid, "experiment_id": eid, "gpus_per_run": gpus,
                "command": journaled(self.journal, *seconds), **fields}

    def submit(self, eid, rid, *seconds, gpus=0, **fields):
        return self.post("/api/jobs", self.job_body(eid, rid, *seconds, gpus=gpus, **fields))

    def submit_many(self, eid, rids, *seconds, gpus=0, **fields):
        return self.post("/api/jobs/bulk", [self.job_body(eid, r, *seconds, gpus=gpus, **fields)
                                            for r in rids])

    def job(self, eid, rid) -> dict:
        return self.get(f"/api/jobs/{rid}?experiment_id={eid}")

    def jobs(self, eid) -> dict[str, dict]:
        return {j["run_id"]: j for j in self.get(f"/api/experiments/{eid}/jobs")}

    def statuses(self, eid) -> Counter:
        return Counter(j["status"] for j in self.jobs(eid).values())

    def wait_statuses(self, eid, want: dict, timeout: float = 60):
        _wait(lambda: self.statuses(eid) == Counter(want), timeout,
              f"{eid} statuses to be {want} (now {dict(self.statuses(eid))})")

    def cancel(self, eid, rid):
        return self.post(f"/api/jobs/{rid}/cancel?experiment_id={eid}")

    def set_exp_status(self, eid, status):
        return self.put(f"/api/experiments/{eid}/status", {"status": status})

    # ── what actually ran ──────────────────────────────────────────────────

    def spans(self, eid: str | None = None) -> list[Span]:
        try:
            lines = self.journal.read_text().splitlines()
        except OSError:
            return []
        open_: dict[int, tuple] = {}
        spans = []
        for line in lines:
            ev, exp, run, pid, worker, gpu, t = line.split()
            pid, t = int(pid), float(t)
            if ev == "start":
                open_[pid] = (exp, run, worker, gpu, t)
            elif pid in open_:
                exp, run, worker, gpu, t0 = open_.pop(pid)
                spans.append(Span(exp, run, pid, worker, gpu, t0, t, ev))
        spans += [Span(exp, run, pid, worker, gpu, t0, None, "")
                  for pid, (exp, run, worker, gpu, t0) in open_.items()]
        spans.sort(key=lambda s: s.start)
        return [s for s in spans if eid is None or s.experiment == eid]

    def starts(self, eid) -> Counter:
        return Counter(s.run for s in self.spans(eid))

    def live_pids(self, eid: str | None = None) -> list[int]:
        """Training processes of this cluster (of *eid*) that are still alive."""
        pids = _pids_with(str(self.journal))
        if eid is None:
            return pids
        return [p for p in pids if f"EXP_EXPERIMENT={eid}".encode() in _environ(p)]


def _cmdline(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _environ(pid: int) -> list[bytes]:
    try:
        return open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except OSError:
        return []


def _pids_with(*needles: str) -> list[int]:
    """Live pids (not zombies) whose command line contains every one of *needles*."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        cmd = _cmdline(int(d))
        if cmd and all(n in cmd for n in needles):
            out.append(int(d))
    return out


def wait_until(pred, timeout: float, what: str):
    return _wait(pred, timeout, what)


def stays(pred, seconds: float, what: str, every: float = 0.25) -> None:
    """Assert *pred* holds continuously for *seconds*."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        assert pred(), f"stopped holding: {what}"
        time.sleep(every)
