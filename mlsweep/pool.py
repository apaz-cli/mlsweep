"""Programmatic WorkerPool for mlsweep.

A lightweight alternative to run_sweep.py for callers that want to submit
individual MsgRun jobs and get back structured results — without sweep files,
variation grids, or logger integration.

Typical usage::

    from mlsweep.pool import WorkerPool, WorkerConfig, RunResult
    from mlsweep._shared import MsgRun

    pool = WorkerPool([
        WorkerConfig(host="user@gpu-box", remote_dir="/home/user/project",
                     devices=[0, 1, 2, 3]),
        WorkerConfig(host=None, remote_dir=""),   # local, CPU-only
    ])
    pool.start()

    result: RunResult = pool.run(MsgRun(
        run_id="my-run",
        experiment="my-exp",
        command=["python", "train.py"],
        env={},
        gpu_ids=[0],
        remote_dir="",
        scratch="/tmp/mlsweep",
        files={"train.py": "..."},
        return_files=["train.py"],
    ))
    print(result.stdout)
    print(result.files["train.py"])

    pool.shutdown()
"""

from __future__ import annotations

import dataclasses
import os
import queue
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import threading
from typing import Any

from mlsweep._parsync import parsync_bin
from mlsweep._shared import (
    MsgHello,
    MsgLog,
    MsgResult,
    MsgRun,
    MsgShutdown,
    MsgWorkerHello,
    decode,
    encode,
)
from mlsweep._topology import _best_gpu_groups
from mlsweep.worker import _LineReader


@dataclasses.dataclass
class WorkerConfig:
    host:         str | None  # None = local; "user@host" = SSH
    remote_dir:   str         # project root on worker (cwd when files={})
    devices:      list[int] = dataclasses.field(default_factory=list)
    # GPU device indices.  Empty list = CPU-only worker.
    gpus_per_run: int = 1
    jobs:         int = 1     # concurrent runs per device group (or total for CPU workers)
    scratch_dir:  str = "/tmp/mlsweep"
    port:         int = 7890  # fixed worker port; 0 = ephemeral (no reuse)
    ssh_key:      str | None = None
    password:     str | None = None
    venv:         str | None = None


@dataclasses.dataclass
class RunResult:
    run_id:    str
    success:   bool
    exit_code: int
    elapsed:   float
    stdout:    str             # concatenated rank-0 stdout (no logger required)
    files:     dict[str, str]  # return_files contents keyed by relative path


@dataclasses.dataclass
class _EvLog:
    run_id: str
    data:   str


@dataclasses.dataclass
class _EvResult:
    run_id:    str
    success:   bool
    exit_code: int
    elapsed:   float


@dataclasses.dataclass
class _EvArtifactSynced:
    run_id: str


@dataclasses.dataclass
class _WorkerState:
    cfg:         WorkerConfig
    host:        str           # "user@host" or "localhost"
    scratch_dir: str           # from MsgWorkerHello
    send_queue:  "queue.Queue[bytes | None]"


_sshpass_available: bool | None = None


def _sshpass_prefix(password: str | None) -> list[str]:
    global _sshpass_available
    if not password:
        return []
    if _sshpass_available is None:
        _sshpass_available = shutil.which("sshpass") is not None
    if not _sshpass_available:
        raise RuntimeError("sshpass is not installed but a password was specified")
    return ["sshpass", "-p", password]


def _worker_candidates(venv: str | None) -> list[str]:
    candidates: list[str] = []
    if venv:
        p = venv.rstrip("/")
        bn = os.path.basename(p)
        if bn == "mlsweep_worker":
            candidates.append(p)
        elif bn in ("python", "python3", "activate"):
            candidates.append(os.path.join(os.path.dirname(p), "mlsweep_worker"))
        elif bn == "bin":
            candidates.append(os.path.join(p, "mlsweep_worker"))
        else:
            candidates += [
                os.path.join(p, "bin", "mlsweep_worker"),
                os.path.join(p, ".venv", "bin", "mlsweep_worker"),
                os.path.join(p, "venv", "bin", "mlsweep_worker"),
            ]
    candidates.append("mlsweep_worker")
    return candidates


def _worker_shell_cmd(candidates: list[str], worker_args: list[str]) -> str:
    args_str = shlex.join(worker_args)
    paths_str = " ".join(shlex.quote(c) for c in candidates)
    return (
        f"for _p in {paths_str}; do\n"
        f"    [ -x \"$_p\" ] && exec \"$_p\" {args_str}\n"
        f"done\n"
        f"echo 'mlsweep: mlsweep_worker not found (tried: {paths_str})' >&2; exit 1"
    )


def _launch_worker(
    cfg: WorkerConfig,
    token: str,
) -> tuple[_WorkerState, socket.socket, _LineReader, MsgWorkerHello]:
    """Connect to an existing worker or start a fresh one.

    When ``cfg.port != 0``, first attempts to connect to an already-running
    worker at that port (e.g. one started by ``mlsweep_run``).  Workers started
    without ``--token`` accept any connection, so they can be shared across
    controllers.  If the connect or handshake fails, a fresh worker is launched.

    The full MsgHello / MsgWorkerHello handshake is performed here so that
    ``start()`` can populate GPU slots immediately without an async wait loop.
    Returns ``(state, sock, reader, hello)`` where ``reader`` is already past
    the hello exchange so the read thread processes run messages directly.
    """
    host = cfg.host or "localhost"
    connect_host = "localhost" if host == "localhost" else host.split("@")[-1]

    def _handshake(
        sock: socket.socket,
    ) -> tuple[_WorkerState, socket.socket, _LineReader, MsgWorkerHello] | None:
        try:
            sock.sendall(encode(MsgHello(token=token, controller_id="pool")))
            reader = _LineReader(sock)
            line = reader.readline()
            if not line:
                return None
            msg = decode(line)
            if not isinstance(msg, MsgWorkerHello):
                return None
            ws = _WorkerState(
                cfg=cfg,
                host=host,
                scratch_dir=msg.scratch_dir,
                send_queue=queue.Queue(),
            )
            return ws, sock, reader, msg
        except (OSError, ValueError):
            return None

    # Try to reuse an existing worker at the fixed port
    if cfg.port != 0:
        try:
            sock = socket.create_connection((connect_host, cfg.port), timeout=2)
            result = _handshake(sock)
            if result is not None:
                return result
            sock.close()
        except OSError:
            pass

    # Launch a fresh worker
    key_args = ["-i", cfg.ssh_key] if cfg.ssh_key else []
    worker_args = [
        "--token", token,
        "--remote-dir", cfg.remote_dir,
        "--scratch-dir", cfg.scratch_dir,
        "--port", str(cfg.port),
        *(["--devices", ",".join(str(d) for d in cfg.devices)] if cfg.devices else []),
    ]

    if host == "localhost":
        bin_dir = os.path.dirname(sys.executable)
        worker_bin = os.path.join(bin_dir, "mlsweep_worker")
        cmd: list[str] = (
            [worker_bin] if os.path.isfile(worker_bin)
            else [sys.executable, "-m", "mlsweep.worker"]
        )
        proc = subprocess.Popen(
            cmd + worker_args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    else:
        shell_cmd = _worker_shell_cmd(_worker_candidates(cfg.venv), worker_args)
        proc = subprocess.Popen(
            [*_sshpass_prefix(cfg.password), "ssh", "-o", "ConnectTimeout=10",
             *key_args, host, shell_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line.startswith("PORT="):
        stderr_out = proc.stderr.read().strip() if proc.stderr else ""
        proc.wait()
        raise RuntimeError(f"worker on {host} failed to start: {stderr_out or line or '(no output)'}")

    port = int(line.split("=")[1])
    sock = socket.create_connection((connect_host, port), timeout=15)
    result = _handshake(sock)
    if result is None:
        raise RuntimeError(f"worker on {host} failed handshake after launch")
    return result



def _worker_write_thread(ws: _WorkerState, sock: socket.socket) -> None:
    while True:
        item = ws.send_queue.get()
        if item is None:
            break
        try:
            sock.sendall(item)
        except OSError:
            break


def _worker_read_thread(
    ws: _WorkerState,
    reader: _LineReader,
    ev_q: "queue.Queue[Any]",
) -> None:
    # MsgWorkerHello already consumed by _launch_worker; process run messages.
    try:
        while True:
            line = reader.readline()
            if line is None:
                return
            try:
                msg = decode(line)
            except (ValueError, KeyError):
                continue

            if isinstance(msg, MsgLog):
                ev_q.put(_EvLog(run_id=msg.run_id, data=msg.data))
            elif isinstance(msg, MsgResult):
                ev_q.put(_EvResult(
                    run_id=msg.run_id,
                    success=msg.success,
                    exit_code=msg.exit_code,
                    elapsed=msg.elapsed,
                ))
            # MsgStarted, MsgMetric, MsgSyncReq, MsgCleaned: ignored by pool
    except OSError:
        pass



def _rsync_thread(
    host: str,
    remote_scratch: str,
    local_run_dir: str,
    run_id: str,
    ev_q: "queue.Queue[Any]",
    password: str | None,
    ssh_key: str | None,
) -> None:
    if host == "localhost":
        src = os.path.join(remote_scratch, "artifacts")
        dst = os.path.join(local_run_dir, "artifacts")
        if src != dst and os.path.isdir(src):
            try:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            except OSError:
                pass
    else:
        env = os.environ.copy()
        if password:
            env["PARSYNC_SSH_PASSWORD"] = password
        subprocess.run(
            [parsync_bin(), "-rlu", f"{host}:{remote_scratch}/", f"{local_run_dir}/"],
            capture_output=True, env=env,
        )
    ev_q.put(_EvArtifactSynced(run_id=run_id))



class WorkerPool:
    """Manages a pool of mlsweep workers and dispatches MsgRun jobs to them.

    Workers are started on ``start()`` and accept jobs via ``submit()`` /
    ``run()``.  ``run()`` is a blocking convenience wrapper around
    ``submit()`` + ``wait()``.
    """

    def __init__(
        self,
        workers: list[WorkerConfig],
        output_dir: str = "/tmp/mlsweep_pool",
    ) -> None:
        self._cfgs = workers
        self._output_dir = output_dir
        self._token = secrets.token_hex(16)
        self._ev_q: queue.Queue[Any] = queue.Queue()

        self._workers: list[_WorkerState] = []

        # Available (worker, gpu_ids) slots; submit() blocks on .get() until one is free.
        self._slots_q: queue.Queue[tuple[_WorkerState, list[int]]] = queue.Queue()

        # Per-run state
        self._log_buf:          dict[str, list[str]]                           = {}
        self._run_done:         dict[str, threading.Event]                     = {}
        self._run_result:       dict[str, RunResult]                           = {}
        self._run_return_files: dict[str, list[str]]                           = {}
        # run_id → (worker, gpu_ids, remote_scratch_path) — cleared on _EvResult
        self._run_slot:         dict[str, tuple[_WorkerState, list[int], str]] = {}

        self._started = False

    def start(self) -> None:
        """Connect to or launch all workers; block until all are ready."""
        if self._started:
            raise RuntimeError("WorkerPool.start() called twice")
        self._started = True

        for cfg in self._cfgs:
            ws, sock, reader, hello = _launch_worker(cfg, self._token)
            self._workers.append(ws)
            self._populate_slots(ws, hello)
            threading.Thread(target=_worker_write_thread, args=(ws, sock), daemon=True).start()
            threading.Thread(target=_worker_read_thread, args=(ws, reader, self._ev_q), daemon=True).start()

        threading.Thread(target=self._event_loop, daemon=True).start()

    def shutdown(self) -> None:
        for ws in self._workers:
            try:
                ws.send_queue.put(encode(MsgShutdown()))
                ws.send_queue.put(None)
            except Exception:
                pass

    def __enter__(self) -> "WorkerPool":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def submit(self, msg: MsgRun) -> str:
        """Dispatch a run to the first free slot; blocks until one is free."""
        self._log_buf[msg.run_id] = []
        self._run_done[msg.run_id] = threading.Event()
        self._run_return_files[msg.run_id] = list(msg.return_files)
        os.makedirs(os.path.join(self._output_dir, msg.run_id), exist_ok=True)

        ws, gpu_ids = self._slots_q.get()
        run_scratch = os.path.join(ws.scratch_dir, msg.experiment, msg.run_id)
        self._run_slot[msg.run_id] = (ws, gpu_ids, run_scratch)

        patched = dataclasses.replace(
            msg,
            gpu_ids=gpu_ids if not msg.gpu_ids else msg.gpu_ids,
            scratch=run_scratch,
            remote_dir=msg.remote_dir or ws.cfg.remote_dir,
        )
        ws.send_queue.put(encode(patched))
        return msg.run_id

    def wait(self, run_id: str) -> RunResult:
        """Block until the run completes and return its result (with files)."""
        self._run_done[run_id].wait()
        return self._run_result[run_id]

    def run(self, msg: MsgRun) -> RunResult:
        """Submit a run and block until it completes."""
        return self.wait(self.submit(msg))

    def _populate_slots(self, ws: _WorkerState, hello: MsgWorkerHello) -> None:
        """Add all GPU slots for a freshly connected worker to the slot queue."""
        topo = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in hello.topo.items()
        }
        if hello.gpus:
            gpu_groups = _best_gpu_groups(
                hello.gpus, ws.cfg.gpus_per_run, ws.cfg.jobs, topo=topo,
            )
        else:
            gpu_groups = [[] for _ in range(ws.cfg.jobs)]
        for gpu_ids in gpu_groups:
            self._slots_q.put((ws, gpu_ids))

    def _event_loop(self) -> None:
        """Drain the event queue in a background thread."""
        while True:
            try:
                ev = self._ev_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(ev, _EvLog):
                buf = self._log_buf.get(ev.run_id)
                if buf is not None:
                    buf.append(ev.data)
            elif isinstance(ev, _EvResult):
                self._on_result(ev)
            elif isinstance(ev, _EvArtifactSynced):
                self._on_synced(ev)

    def _on_result(self, ev: _EvResult) -> None:
        ws, gpu_ids, run_scratch = self._run_slot.pop(ev.run_id)
        self._slots_q.put((ws, gpu_ids))  # free the slot

        run_dir = os.path.join(self._output_dir, ev.run_id)
        os.makedirs(run_dir, exist_ok=True)
        threading.Thread(
            target=_rsync_thread,
            args=(ws.host, run_scratch, run_dir, ev.run_id, self._ev_q,
                  ws.cfg.password, ws.cfg.ssh_key),
            daemon=True,
        ).start()
        self._run_result[ev.run_id] = RunResult(
            run_id=ev.run_id,
            success=ev.success,
            exit_code=ev.exit_code,
            elapsed=ev.elapsed,
            stdout="".join(self._log_buf.pop(ev.run_id, [])),
            files={},
        )

    def _on_synced(self, ev: _EvArtifactSynced) -> None:
        run_id = ev.run_id
        result = self._run_result.get(run_id)
        if result is None:
            return
        artifacts_dir = os.path.join(self._output_dir, run_id, "artifacts")
        return_files = self._run_return_files.pop(run_id, [])
        files: dict[str, str] = {}
        for rel_path in return_files:
            abs_path = os.path.join(artifacts_dir, rel_path)
            try:
                files[rel_path] = open(abs_path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                pass
        self._run_result[run_id] = dataclasses.replace(result, files=files)
        done = self._run_done.get(run_id)
        if done is not None:
            done.set()
