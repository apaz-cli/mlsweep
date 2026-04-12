#!/usr/bin/env python3
"""mlsweep worker process.

Executes training subprocesses, streams logs/metrics over a persistent TCP
connection to the controller, and manages a scratch directory.

Lifetime
--------
The worker is started by mlsweep_run at sweep start and receives MsgShutdown
when the controller exits cleanly.  If the controller crashes or is killed
before sending MsgShutdown, the worker detects the TCP connection close:

  - If no runs are in flight, it exits immediately.
  - If runs are in flight, it keeps them running to completion and exits
    after the last run finishes.  This preserves work already in progress
    even when the controller dies unexpectedly.

SIGHUP is ignored so brief SSH disconnects do not kill the worker.

Invoked by the controller via SSH (or directly for local mode):
    python -m mlsweep.worker --token TOKEN --remote-dir /path/to/project

Startup behaviour:
  - Binds an ephemeral TCP port, prints PORT=N to stdout, and enters the
    accept loop.  The controller reads PORT=N and connects.
"""

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, IO

from mlsweep._shared import (
    MsgCancel,
    MsgCleaned,
    MsgCleanup,
    MsgHello,
    MsgLog,
    MsgMetric,
    MsgPing,
    MsgPong,
    MsgReplay,
    MsgResult,
    MsgRun,
    MsgShutdown,
    MsgStarted,
    MsgSyncReq,
    MsgWorkerHello,
    decode,
    encode,
)
from mlsweep._topology import _gpu_topology, _visible_devices

# ── Run state ──────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class RunState:
    run_id: str
    pids: list[int]         # per-GPU PIDs; pids[0] is rank-0
    scratch_path: str       # {scratch_dir}/{experiment}/{run_id}/
    gpu_ids: list[int]
    experiment: str
    log_seq: int = 0        # byte offset into training.log after last write
    metric_seq: int = 0     # byte offset into metrics.jsonl after last write
    log_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


# ── Connection state ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ConnState:
    sock: socket.socket
    send_queue: "queue.Queue[bytes | None]"
    closed: bool = False


# ── Global worker state (protected by _lock) ───────────────────────────────────

_lock = threading.Lock()
_in_flight: dict[str, RunState] = {}        # run_id → RunState
_busy_gpus: set[int] = set()
_connections: list[ConnState] = []
_shutdown_event = threading.Event()

# Set by main() from CLI args
_scratch_dir: str = "/tmp/mlsweep"
_remote_dir: str = ""
_token: str = ""
_device_override: list[int] | None = None  # None = use all visible


# ── Wire I/O helpers ───────────────────────────────────────────────────────────


class _LineReader:
    """Buffer TCP data and yield complete newline-terminated lines."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b""

    def readline(self) -> bytes | None:
        """Return the next complete line (including \\n), or None on EOF/error."""
        while True:
            if b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                return line + b"\n"
            try:
                chunk = self._sock.recv(4096)
            except OSError:
                return None
            if not chunk:
                return None
            self._buf += chunk


# ── Write thread (one per connection) ─────────────────────────────────────────


def _write_thread(conn: ConnState) -> None:
    """Drain send_queue and write bytes to socket.  None sentinel = stop."""
    while True:
        item = conn.send_queue.get()
        if item is None:
            break
        try:
            conn.sock.sendall(item)
        except OSError:
            break
    conn.closed = True
    with _lock:
        try:
            _connections.remove(conn)
        except ValueError:
            pass
    try:
        conn.sock.close()
    except OSError:
        pass


# ── Read thread (one per connection) ──────────────────────────────────────────


def _read_thread(conn: ConnState) -> None:
    """Read protocol messages from a controller connection and dispatch them."""
    reader = _LineReader(conn.sock)

    # First message must be MsgHello
    line = reader.readline()
    if not line:
        conn.send_queue.put(None)
        return
    try:
        msg = decode(line)
    except (ValueError, json.JSONDecodeError, TypeError):
        conn.send_queue.put(None)
        return
    if not isinstance(msg, MsgHello):
        conn.send_queue.put(None)
        return
    if _token and msg.token != _token:
        conn.send_queue.put(None)
        return

    # Build and send MsgWorkerHello
    gpus = _visible_devices()
    if _device_override is not None:
        gpus = [g for g in _device_override if g in gpus]
    topo_internal = _gpu_topology()
    topo_wire: dict[str, int] = {
        f"{a},{b}": score for (a, b), score in topo_internal.items()
    }
    with _lock:
        resuming = [
            {
                "run_id": rs.run_id,
                "log_seq": rs.log_seq,
                "metric_seq": rs.metric_seq,
                "pid": rs.pids[0],
            }
            for rs in _in_flight.values()
        ]
    hello_resp = MsgWorkerHello(
        gpus=gpus,
        topo=topo_wire,
        resuming=resuming,
        scratch_dir=_scratch_dir,
    )
    conn.send_queue.put(encode(hello_resp))

    # Main message loop
    while not _shutdown_event.is_set():
        line = reader.readline()
        if not line:
            break
        try:
            msg = decode(line)
        except (ValueError, json.JSONDecodeError, TypeError):
            continue
        _handle_msg(msg, conn)

    # Connection closed — signal write thread to drain and exit
    conn.send_queue.put(None)
    # If no runs are in flight, shut down so the worker exits when the controller disconnects
    with _lock:
        no_work = not _in_flight
    if no_work:
        _shutdown_event.set()


# ── Message handlers ───────────────────────────────────────────────────────────


def _handle_msg(msg: Any, conn: ConnState) -> None:
    if isinstance(msg, MsgRun):
        _handle_run(msg, conn)
    elif isinstance(msg, MsgCancel):
        _handle_cancel(msg)
    elif isinstance(msg, MsgCleanup):
        _handle_cleanup(msg, conn)
    elif isinstance(msg, MsgReplay):
        _handle_replay(msg, conn)
    elif isinstance(msg, MsgShutdown):
        _shutdown_event.set()
    elif isinstance(msg, MsgPing):
        if not conn.closed:
            conn.send_queue.put(encode(MsgPong()))


def _handle_run(msg: MsgRun, conn: ConnState) -> None:
    """Spawn one training subprocess per GPU in the run's GPU group."""
    scratch_path = os.path.join(_scratch_dir, msg.experiment, msg.run_id)
    log_path = os.path.join(scratch_path, "training.log")
    metrics_path = os.path.join(scratch_path, "metrics.jsonl")
    artifacts_path = os.path.join(scratch_path, "artifacts")
    os.makedirs(artifacts_path, exist_ok=True)

    # Create buffer files
    open(log_path, "w").close()
    open(metrics_path, "w").close()

    # Workspace creation from file payload
    if msg.files:
        workspace = os.path.join(scratch_path, "workspace")
        base_dir = msg.remote_dir or _remote_dir
        if base_dir and os.path.isdir(base_dir):
            # Hard-link copy: large staged files cost no extra disk space
            subprocess.run(
                ["rsync", "-a", "--link-dest", base_dir + "/", base_dir + "/", workspace + "/"],
                check=True,
            )
        else:
            os.makedirs(workspace, exist_ok=True)
        for rel_path, content in msg.files.items():
            abs_path = os.path.join(workspace, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            Path(abs_path).write_text(content, encoding="utf-8")
        cwd = workspace
    else:
        workspace = None
        remote_dir = msg.remote_dir or _remote_dir
        cwd = os.path.join(remote_dir, msg.run_from) if msg.run_from else remote_dir

    # Build base env shared by all ranks
    device_str = ",".join(str(g) for g in msg.gpu_ids)
    base_env = {**os.environ, **msg.env}
    base_env["CUDA_VISIBLE_DEVICES"] = device_str
    base_env["HIP_VISIBLE_DEVICES"] = device_str
    base_env["MLSWEEP_RUN_DIR"] = artifacts_path
    base_env["MLSWEEP_RUN_NAME"] = msg.run_id
    base_env["EXP_EXPERIMENT"] = msg.experiment
    base_env["MLSWEEP_WORKER_SOCKET"] = os.path.join(_scratch_dir, ".worker.sock")
    if workspace is not None:
        base_env["MLSWEEP_WORKSPACE"] = workspace
    base_env.pop("EXP_SERVER", None)

    # Pre-compute dist env values if SET_DIST_ENV is requested
    _dist_base: dict[str, str] = {}
    _dist_node_rank = 0
    _dist_gpus_per_node = len(msg.gpu_ids)
    if msg.set_dist_env:
        nnodes = int(base_env.get("MLSWEEP_NNODES", "1"))
        _dist_node_rank = int(base_env.get("MLSWEEP_NODE_RANK", "0"))
        world_size = nnodes * _dist_gpus_per_node
        if nnodes > 1:
            master_addr = base_env["MLSWEEP_MASTER_ADDR"]
            master_port = base_env["MLSWEEP_MASTER_PORT"]
        else:
            master_addr = "localhost"
            master_port = str(
                20000 + int(hashlib.md5(msg.run_id.encode()).hexdigest()[:4], 16) % 10000
            )
        _dist_base = {
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        }

    # Spawn one process per GPU rank (or one process for CPU-only runs)
    n_ranks = max(1, len(msg.gpu_ids))
    procs: list[subprocess.Popen[bytes]] = []
    pids: list[int] = []
    try:
        for rank in range(n_ranks):
            rank_env = {**base_env, "MLSWEEP_GPU_RANK": str(rank)}
            if msg.set_dist_env:
                rank_env["RANK"] = str(_dist_node_rank * _dist_gpus_per_node + rank)
                rank_env["LOCAL_RANK"] = str(rank)
                rank_env.update(_dist_base)
            proc = subprocess.Popen(
                msg.command,
                stdout=subprocess.PIPE if rank == 0 else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if rank == 0 else subprocess.DEVNULL,
                env=rank_env,
                cwd=cwd,
            )
            procs.append(proc)
            pids.append(proc.pid)
    except OSError:
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass
        conn.send_queue.put(encode(MsgResult(
            run_id=msg.run_id, success=False, elapsed=0.0, exit_code=-1
        )))
        return

    with _lock:
        state = RunState(
            run_id=msg.run_id,
            pids=pids,
            scratch_path=scratch_path,
            gpu_ids=list(msg.gpu_ids),
            experiment=msg.experiment,
        )
        _in_flight[msg.run_id] = state
        _busy_gpus.update(msg.gpu_ids)

    conn.send_queue.put(encode(MsgStarted(run_id=msg.run_id, pid=pids[0])))

    t = threading.Thread(
        target=_run_thread,
        args=(procs, state, log_path, artifacts_path, workspace or cwd, msg.return_files, conn),
        daemon=True,
        name=f"run-{msg.run_id}",
    )
    t.start()


def _run_thread(
    procs: "list[subprocess.Popen[bytes]]",
    state: RunState,
    log_path: str,
    artifacts_path: str,
    run_dir: str,
    return_files: list[str],
    conn: ConnState,
) -> None:
    """Monitor all per-GPU subprocesses: stream rank-0 logs, send MsgResult when all exit."""
    t0 = time.time()

    # Stream rank-0 stdout to the log
    with open(log_path, "a", buffering=1) as log_fh:
        assert procs[0].stdout is not None
        for raw_line in procs[0].stdout:
            line_str = raw_line.decode("utf-8", errors="replace")
            with state.log_lock:
                log_fh.write(line_str)
                log_fh.flush()
                state.log_seq = log_fh.tell()
            if not conn.closed:
                conn.send_queue.put(encode(MsgLog(
                    run_id=state.run_id,
                    seq=state.log_seq,
                    data=line_str,
                )))

    # Wait for all ranks to finish
    rcs = [procs[0].wait()] + [p.wait() for p in procs[1:]]
    elapsed = time.time() - t0
    exit_code = next((rc for rc in rcs if rc != 0), 0)

    # Copy return_files into artifacts/ before rsync.
    # run_dir is the workspace (when files={...}) or the cwd (when files={}).
    if return_files:
        for rel_path in return_files:
            src = os.path.join(run_dir, rel_path)
            if os.path.isfile(src):
                dst = os.path.join(artifacts_path, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    with _lock:
        _in_flight.pop(state.run_id, None)
        _busy_gpus.difference_update(state.gpu_ids)

    if not conn.closed:
        conn.send_queue.put(encode(MsgResult(
            run_id=state.run_id,
            success=(exit_code == 0),
            elapsed=elapsed,
            exit_code=exit_code,
        )))


def _handle_cleanup(msg: MsgCleanup, conn: ConnState) -> None:
    """Acknowledge a cleanup request (artifacts already rsynced by controller)."""
    if not conn.closed:
        conn.send_queue.put(encode(MsgCleaned(run_id=msg.run_id)))


def _handle_cancel(msg: MsgCancel) -> None:
    """SIGTERM all per-GPU processes for the run."""
    with _lock:
        state = _in_flight.get(msg.run_id)
    if state is not None:
        for pid in state.pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def _handle_replay(msg: MsgReplay, conn: ConnState) -> None:
    """Replay buffered log and metric lines from given byte offsets."""
    with _lock:
        state = _in_flight.get(msg.run_id)
    if state is None:
        return
    t = threading.Thread(
        target=_replay_thread,
        args=(msg.run_id, state, msg.log_seq, msg.metric_seq, conn),
        daemon=True,
        name=f"replay-{msg.run_id}",
    )
    t.start()


def _replay_thread(
    run_id: str,
    state: RunState,
    log_seq: int,
    metric_seq: int,
    conn: ConnState,
) -> None:
    """Re-send log/metric lines the controller missed during a disconnect."""
    log_path = os.path.join(state.scratch_path, "training.log")
    metrics_path = os.path.join(state.scratch_path, "metrics.jsonl")

    with state.log_lock:
        # Replay log lines from log_seq offset
        try:
            with open(log_path, "rb") as f:
                f.seek(log_seq)
                while True:
                    raw = f.readline()
                    if not raw:
                        break
                    if not conn.closed:
                        conn.send_queue.put(encode(MsgLog(
                            run_id=run_id,
                            seq=f.tell(),
                            data=raw.decode("utf-8", errors="replace"),
                        )))
        except OSError:
            pass

        # Replay metric lines from metric_seq offset
        try:
            with open(metrics_path, "rb") as f:
                f.seek(metric_seq)
                while True:
                    raw = f.readline()
                    if not raw:
                        break
                    try:
                        rec: dict[str, Any] = json.loads(raw)
                        step = rec.get("step", 0)
                        data = {k: v for k, v in rec.items() if k != "step"}
                        if not conn.closed:
                            conn.send_queue.put(encode(MsgMetric(
                                run_id=run_id, step=step, data=data
                            )))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass


# ── IPC thread (unix socket for logger.py) ─────────────────────────────────────


def _ipc_thread(sock_path: str) -> None:
    """Accept connections from training scripts and route metric/sync messages."""
    try:
        os.unlink(sock_path)
    except OSError:
        pass

    ipc_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        ipc_sock.bind(sock_path)
        ipc_sock.listen(50)
        ipc_sock.settimeout(1.0)
        while not _shutdown_event.is_set():
            try:
                client, _ = ipc_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(
                target=_ipc_conn_thread, args=(client,), daemon=True
            )
            t.start()
    finally:
        ipc_sock.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass


def _ipc_conn_thread(sock: socket.socket) -> None:
    """Handle one IPC connection from a training script."""
    buf = b""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line_bytes = line_bytes.strip()
                if not line_bytes:
                    continue
                try:
                    msg: dict[str, Any] = json.loads(line_bytes)
                except json.JSONDecodeError:
                    continue
                _handle_ipc_msg(msg)
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _handle_ipc_msg(msg: dict[str, Any]) -> None:
    """Route an IPC message (metric or sync) from a training script."""
    run_id = msg.get("run_id", "")
    msg_type = msg.get("type")

    with _lock:
        state = _in_flight.get(run_id)
        conns = [c for c in _connections if not c.closed]

    if state is None:
        return

    metrics_path = os.path.join(state.scratch_path, "metrics.jsonl")

    if msg_type == "metric":
        step = int(msg.get("step", 0))
        data: dict[str, Any] = msg.get("data", {})
        record: dict[str, Any] = {"step": step, **data}
        line = json.dumps(record) + "\n"
        try:
            with open(metrics_path, "a", buffering=1) as f:
                f.write(line)
                f.flush()
                state.metric_seq = f.tell()
        except OSError:
            pass
        wire = encode(MsgMetric(run_id=run_id, step=step, data=data))
        for conn in conns:
            conn.send_queue.put(wire)

    elif msg_type == "sync":
        wire = encode(MsgSyncReq(run_id=run_id))
        for conn in conns:
            conn.send_queue.put(wire)


# ── Accept loop ────────────────────────────────────────────────────────────────


def _accept_loop(server_sock: socket.socket) -> None:
    """Accept incoming controller connections."""
    server_sock.settimeout(1.0)
    while not _shutdown_event.is_set():
        try:
            client_sock, _ = server_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        conn = ConnState(sock=client_sock, send_queue=queue.Queue())
        with _lock:
            _connections.append(conn)

        read_t = threading.Thread(target=_read_thread, args=(conn,), daemon=True)
        write_t = threading.Thread(target=_write_thread, args=(conn,), daemon=True)
        read_t.start()
        write_t.start()

    server_sock.close()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        global _scratch_dir, _remote_dir, _token, _device_override

        parser = argparse.ArgumentParser(description="mlsweep worker daemon")
        parser.add_argument("--token", default="", help="Authentication token")
        parser.add_argument("--scratch-dir", default="/tmp/mlsweep",
                            help="Base scratch directory for run buffers (default: /tmp/mlsweep)")
        parser.add_argument("--remote-dir", default="",
                            help="Project directory on this machine (cwd for training scripts)")
        parser.add_argument("--devices", default=None,
                            help="Comma-separated GPU device IDs to expose, e.g. 4,5,6,7")
        parser.add_argument("--port", type=int, default=7890,
                            help="TCP port to bind (0 = ephemeral, default: 7890)")
        args = parser.parse_args()

        _scratch_dir = args.scratch_dir
        _remote_dir = args.remote_dir or os.getcwd()
        _token = args.token
        if args.devices:
            _device_override = [int(x) for x in args.devices.split(",")]

        os.makedirs(_scratch_dir, exist_ok=True)

        # For a fixed port, use an flock to prevent two workers from racing to
        # bind the same port.  The lock file is a pure synchronization token —
        # no data is stored in it.
        #
        #   Winner:  LOCK_EX | LOCK_NB → bind → listen → downgrade to LOCK_SH.
        #   Loser:   fail LOCK_EX | LOCK_NB → block on LOCK_SH (unblocks only
        #            after winner downgrades, i.e. is already listening) →
        #            print PORT={args.port} and exit.
        #
        # The controller always sees a normal "PORT=N" line and needs no
        # special handling.  The winner holds LOCK_SH for its lifetime so
        # late arrivals also wait correctly.
        _lock_file = None
        if args.port != 0:
            lock_path = f"/tmp/.mlsweep_worker_port_{args.port}.lock"
            _lock_file = open(lock_path, "w")
            try:
                fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fcntl.flock(_lock_file, fcntl.LOCK_SH)  # blocks until winner is listening
                print(f"PORT={args.port}", flush=True)
                sys.exit(0)

        # Bind TCP port (fixed or ephemeral)
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("", args.port))
        port = server_sock.getsockname()[1]
        server_sock.listen(10)

        # Downgrade to LOCK_SH — unblocks any losers waiting to relay the port.
        if _lock_file is not None:
            fcntl.flock(_lock_file, fcntl.LOCK_SH)

        # Ignore SIGHUP so brief SSH disconnects don't kill the worker
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

        # Print port so the controller can read it and connect
        print(f"PORT={port}", flush=True)

        # Start IPC thread for logger.py connections
        sock_path = os.path.join(_scratch_dir, ".worker.sock")
        ipc_t = threading.Thread(target=_ipc_thread, args=(sock_path,), daemon=True)
        ipc_t.start()

        # Enter accept loop (blocks until _shutdown_event is set)
        _accept_loop(server_sock)


    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
