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
import importlib.metadata
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
from typing import Any
from urllib.request import urlopen

from mlsweep._shared import (
    MsgCancel,
    MsgCleaned,
    MsgCleanup,
    MsgGpuStats,
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
    LOG_CHUNK_BYTES,
    PROTOCOL_VERSION,
    _resolve_safe_subpath,
    decode,
    dist_master_port,
    encode,
    line_chunks,
    read_msg,
    set_color,
)
from mlsweep._topology import _gpu_topology, visible_devices
from mlsweep import _env

# ── Run state ──────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class RunState:
    """A run this worker has accepted, from MsgRun until its result is recorded.

    Registered before setup (artifact download, setup_command) so that a cancel
    or a reconnect during setup sees the run.  ``pids`` stays empty until the
    training processes are spawned.
    """
    run_id: str
    scratch_path: str       # {scratch_dir}/{experiment}/{run_id}/
    gpu_ids: list[int]
    experiment: str
    pids: list[int] = dataclasses.field(default_factory=list)  # per-GPU; pids[0] is rank-0
    setup_proc: "subprocess.Popen[bytes] | None" = None
    cancelled: bool = False
    # Bytes of training.log written and sent to the manager, and the lines in between.
    log_seq: int = 0
    sent_seq: int = 0
    pending_log: list[bytes] = dataclasses.field(default_factory=list)
    log_fh: Any = None
    log_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    # Open logger (IPC) connections of this run; its result waits for them to drain.
    ipc_conns: int = 0


# ── Connection state ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ConnState:
    sock: socket.socket
    send_queue: "queue.Queue[bytes | None]"
    closed: bool = False


# ── Global worker state (protected by _lock) ───────────────────────────────────

_lock = threading.Lock()
# Runs are keyed by (experiment, run_id) because run names repeat across experiments.
RunKey = tuple[str, str]
_in_flight: dict[RunKey, RunState] = {}
_connections: list[ConnState] = []
# Results the manager has not acknowledged yet (it acks with MsgCleanup).  A result
# sent while the manager is disconnected, or on a connection it has abandoned, would
# otherwise be lost; these are re-reported in every MsgWorkerHello as ``completed``.
_unacked_results: dict[RunKey, MsgResult] = {}
_shutdown_event = threading.Event()
# Signalled when an IPC connection is attributed to its run or closes.
_ipc_cond = threading.Condition(_lock)
# IPC connections accepted but not yet attributed to a run (no message read yet).
_ipc_unclaimed = 0
# How long a finished run's result waits for its loggers' last messages.
_IPC_DRAIN_TIMEOUT = 10.0

# Set by main() from CLI args
_scratch_dir: str = "/tmp/mlsweep"
_remote_dir: str = ""
_token: str = ""
_device_override: list[int] | None = None  # None = use all visible
_max_jobs_per_gpu: int = 1  # per-GPU packing cap reported to the manager (0 = unlimited)
_ipc_sock_path: str = "/tmp/mlsweep/.worker.sock"  # set to port-specific path at startup


# ── Artifact download serialisation ──────────────────────────────────────────

_artifact_locks: dict[str, threading.Lock] = {}
_artifact_lock_refs: dict[str, int] = {}

def _artifact_lock_for(artifact_id: str) -> threading.Lock:
    """Return a per-artifact lock so concurrent runs do not clobber downloads."""
    with _lock:
        lock = _artifact_locks.get(artifact_id)
        if lock is None:
            lock = threading.Lock()
            _artifact_locks[artifact_id] = lock
            _artifact_lock_refs[artifact_id] = 0
        _artifact_lock_refs[artifact_id] += 1
        return lock

def _artifact_lock_done(artifact_id: str) -> None:
    with _lock:
        refs = _artifact_lock_refs.get(artifact_id, 0)
        if refs <= 1:
            _artifact_locks.pop(artifact_id, None)
            _artifact_lock_refs.pop(artifact_id, None)
        else:
            _artifact_lock_refs[artifact_id] = refs - 1


# ── Wire I/O helpers ───────────────────────────────────────────────────────────




# ── Sending run traffic ───────────────────────────────────────────────────────


def _current_connection() -> "ConnState | None":
    """The most recently accepted connection that is still open.

    Run messages (started / log / result) go here rather than to the connection that
    dispatched the run: after a reconnect, the dispatching connection is one the
    manager no longer reads, and anything sent on it is silently lost."""
    with _lock:
        for conn in reversed(_connections):
            if not conn.closed:
                return conn
    return None


def _send_run_msg(data: bytes) -> None:
    conn = _current_connection()
    if conn is not None:
        conn.send_queue.put(data)


def _report_result(result: MsgResult) -> None:
    """Record *result* as unacknowledged, then send it.  It is re-sent in the next
    MsgWorkerHello until the manager acknowledges it with MsgCleanup."""
    with _lock:
        _unacked_results[(result.experiment, result.run_id)] = result
    _send_run_msg(encode(result))


# ── Run logs ─────────────────────────────────────────────────────────────────
#
# A run's output is appended to its training.log and sent as chunks of whole
# lines, each carrying its exact byte range, so the manager can tell a gap (a
# chunk lost with a dropped connection) from a duplicate.  It asks for a
# replay from where its copy ends when it sees a gap.

_LOG_FLUSH_INTERVAL = 0.25


def _append_log(rs: RunState, raw: bytes) -> None:
    """Append whole lines to training.log; they are sent within _LOG_FLUSH_INTERVAL."""
    with rs.log_lock:
        if rs.log_fh is None:
            rs.log_fh = open(os.path.join(rs.scratch_path, "training.log"), "ab", buffering=0)
        rs.log_fh.write(raw)
        rs.log_seq += len(raw)
        rs.pending_log.append(raw)
        if rs.log_seq - rs.sent_seq >= LOG_CHUNK_BYTES:
            _flush_log_locked(rs)


def _log_msg(rs: RunState, start: int, raw: bytes) -> bytes:
    return encode(MsgLog(run_id=rs.run_id, experiment=rs.experiment, start=start,
                         seq=start + len(raw), data=raw.decode("utf-8", errors="replace")))


def _flush_log_locked(rs: RunState) -> None:
    """Send the lines written since the last send.  Caller holds ``rs.log_lock``."""
    if rs.pending_log:
        _send_run_msg(_log_msg(rs, rs.sent_seq, b"".join(rs.pending_log)))
        rs.pending_log.clear()
        rs.sent_seq = rs.log_seq


def _close_log(rs: RunState) -> None:
    """Send what is left of the run's log and close training.log."""
    with rs.log_lock:
        _flush_log_locked(rs)
        if rs.log_fh is not None:
            rs.log_fh.close()
            rs.log_fh = None


def _log_flush_thread() -> None:
    """Send every run's buffered log lines at least every _LOG_FLUSH_INTERVAL."""
    while not _shutdown_event.wait(_LOG_FLUSH_INTERVAL):
        with _lock:
            runs = list(_in_flight.values())
        for rs in runs:
            with rs.log_lock:
                _flush_log_locked(rs)


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
    # First message must be MsgHello
    payload = read_msg(conn.sock)
    if not payload:
        conn.send_queue.put(None)
        return
    try:
        msg = decode(payload)
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
    gpus = visible_devices()
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
                "pid": rs.pids[0] if rs.pids else 0,
                "gpu_ids": list(rs.gpu_ids),
                "experiment": rs.experiment,
            }
            for rs in _in_flight.values()
        ]
        completed = [dataclasses.asdict(res) for res in _unacked_results.values()]
    hello_resp = MsgWorkerHello(
        gpus=gpus,
        topo=topo_wire,
        resuming=resuming,
        scratch_dir=_scratch_dir,
        max_jobs_per_gpu=_max_jobs_per_gpu,
        completed=completed,
        protocol=PROTOCOL_VERSION,
    )
    conn.send_queue.put(encode(hello_resp))

    # Main message loop
    while not _shutdown_event.is_set():
        payload = read_msg(conn.sock)
        if not payload:
            break
        try:
            msg = decode(payload)
        except (ValueError, json.JSONDecodeError, TypeError):
            continue
        _handle_msg(msg, conn)

    # Connection closed — signal write thread to drain and exit
    conn.send_queue.put(None)
    # If no runs are in flight and no other controller connection is open, shut down so
    # the worker exits when the controller disconnects.  (A manager that reconnected
    # closes its stale connection afterwards; that must not stop the worker.)
    with _lock:
        others = any(c is not conn and not c.closed for c in _connections)
        no_work = not _in_flight and not others
    if no_work:
        _shutdown_event.set()


# ── Message handlers ───────────────────────────────────────────────────────────


def _handle_msg(msg: Any, conn: ConnState) -> None:
    if isinstance(msg, MsgRun):
        key = (msg.experiment, msg.run_id)
        with _lock:
            if key in _in_flight:
                return  # duplicate dispatch of a run we are already executing
            _unacked_results.pop(key, None)
            rs = _in_flight[key] = RunState(
                run_id=msg.run_id,
                scratch_path=os.path.join(_scratch_dir, msg.experiment, msg.run_id),
                gpu_ids=list(msg.gpu_ids),
                experiment=msg.experiment,
            )
        t = threading.Thread(
            target=_handle_run, args=(msg, rs),
            daemon=True,
            name=f"setup-{msg.run_id}",
        )
        t.start()
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


def _download_file(url: str, dest: str, timeout: int = 300) -> None:
    """Download a file from *url* to *dest* via HTTP GET."""
    with urlopen(url, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)



def _handle_run(msg: MsgRun, rs: RunState) -> None:
    """Set up the run, then spawn one training subprocess per GPU in its group."""
    try:
        _setup_and_spawn(msg, rs)
    except _Cancelled:
        _finish_unstarted(msg, rs, exit_code=-int(signal.SIGTERM))
    except Exception as exc:
        print(f"[worker] ERROR in run {msg.run_id}: {exc}", file=sys.stderr, flush=True)
        _append_log(rs, f"[mlsweep] run setup failed: {exc}\n".encode())
        _finish_unstarted(msg, rs, exit_code=-1)


class _Cancelled(Exception):
    """The run was cancelled before its training processes started."""


def _run_setup_proc(rs: RunState, cmd: "list[str] | str", cwd: str) -> None:
    """Run a setup step, streaming its output to training.log. Cancellable: a cancel
    signals its process group. Raises CalledProcessError on a nonzero exit."""
    with _lock:
        _check_cancelled_locked(rs)
        proc = rs.setup_proc = subprocess.Popen(
            cmd,
            shell=isinstance(cmd, str),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so a cancel can signal the whole group
        )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_log(rs, line)
    proc.wait()
    with _lock:
        rs.setup_proc = None
    _check_cancelled(rs)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _find_venv(cwd: str) -> str | None:
    """A ``.venv`` in the run's cwd, else in remote_dir."""
    for base in (cwd, _remote_dir):
        if base and os.path.isdir(os.path.join(base, ".venv", "bin")):
            return os.path.join(base, ".venv")
    return None


def _auto_env_enabled() -> bool:
    """Whether the worker should build a default env.

    Off with MLSWEEP_AUTO_ENV=0. Also off when the worker was started inside a
    venv or conda env other than mlsweep's own: runs inherit that environment."""
    if os.environ.get("MLSWEEP_AUTO_ENV", "1").lower() in ("0", "false", "no", "off"):
        return False
    return not any(
        prefix and os.path.realpath(prefix) != os.path.realpath(sys.prefix)
        for prefix in (os.environ.get("VIRTUAL_ENV"), os.environ.get("CONDA_PREFIX"))
    )


def _default_env(rs: RunState, project_dir: str) -> str | None:
    """Build or reuse a venv from the project's dependency files (see mlsweep._env)."""
    if not _auto_env_enabled():
        return None
    spec = _env.detect(project_dir)
    if spec is None:
        _append_log(rs, b"[mlsweep] no .venv, pyproject.toml, requirements.txt or setup.py "
                        b"found; running with the worker's PATH\n")
        return None
    return _env.ensure(
        spec, project_dir, rs.scratch_path,
        run=lambda cmd, cwd: _run_setup_proc(rs, cmd, cwd),
        log=lambda text: _append_log(rs, text.encode()),
    )


def _finish_unstarted(msg: MsgRun, rs: RunState, exit_code: int) -> None:
    """Report a failed result for a run that never spawned its training processes."""
    _close_log(rs)
    result = MsgResult(run_id=msg.run_id, experiment=msg.experiment,
                       success=False, elapsed=0.0, exit_code=exit_code)
    with _lock:
        _in_flight.pop((msg.experiment, msg.run_id), None)
    _report_result(result)


def _check_cancelled_locked(rs: RunState) -> None:
    if rs.cancelled:
        raise _Cancelled


def _check_cancelled(rs: RunState) -> None:
    with _lock:
        _check_cancelled_locked(rs)


def _setup_and_spawn(msg: MsgRun, rs: RunState) -> None:
    scratch_path = rs.scratch_path
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
        os.makedirs(workspace, exist_ok=True)
        for rel_path, content in msg.files.items():
            abs_path = _resolve_safe_subpath(workspace, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            Path(abs_path).write_text(content, encoding="utf-8")
        cwd = workspace
    else:
        workspace = None
        remote_dir = msg.remote_dir or _remote_dir
        cwd = remote_dir

    # ── Artifact download & extraction ───────────────────────────────────────
    if msg.artifact_id and msg.artifact_url:
        try:
            with _artifact_lock_for(msg.artifact_id):
                if workspace is None:
                    workspace = os.path.join(scratch_path, "workspace")
                    os.makedirs(workspace, exist_ok=True)
                tarball_url = msg.artifact_url
                tarball_path = os.path.join(scratch_path, "artifact.tar.gz")
                _download_file(tarball_url, tarball_path)
                try:
                    subprocess.run(
                        ["tar", "-xzf", tarball_path, "-C", workspace],
                        check=True,
                    )
                finally:
                    try:
                        os.unlink(tarball_path)
                    except OSError:
                        pass
                cwd = workspace
        finally:
            _artifact_lock_done(msg.artifact_id)
    _check_cancelled(rs)

    # ── Optional setup command ───────────────────────────────────────────────
    if msg.setup_command:
        if workspace is None:
            workspace = os.path.join(scratch_path, "workspace")
            os.makedirs(workspace, exist_ok=True)
            cwd = workspace
        _run_setup_proc(rs, msg.setup_command, workspace)

    # ── RUN_FROM: working directory for the run ────────────────────────────
    # Resolve against the run's base cwd (the extracted workspace when a file
    # payload or artifact was shipped, otherwise remote_dir).  Relative paths
    # must stay within the base; absolute paths are used as-is.
    if msg.run_from:
        if os.path.isabs(msg.run_from):
            cwd = msg.run_from
        else:
            cwd = _resolve_safe_subpath(cwd, msg.run_from)

    # Build base env shared by all ranks
    device_str = ",".join(str(g) for g in msg.gpu_ids)
    base_env = {**os.environ, **msg.env}
    base_env["CUDA_VISIBLE_DEVICES"] = device_str
    base_env["HIP_VISIBLE_DEVICES"] = device_str
    base_env["MLSWEEP_RUN_DIR"] = artifacts_path
    base_env["MLSWEEP_RUN_NAME"] = msg.run_id
    base_env["EXP_EXPERIMENT"] = msg.experiment
    base_env["MLSWEEP_WORKER_SOCKET"] = _ipc_sock_path
    if workspace is not None:
        base_env["MLSWEEP_WORKSPACE"] = workspace
        existing = base_env.get("PYTHONPATH", "")
        base_env["PYTHONPATH"] = workspace + (os.pathsep + existing if existing else "")
    base_env.pop("EXP_SERVER", None)

    # Activate a venv: .venv in the cwd, then in remote_dir, else a default env
    # built from the project's dependency files.
    venv_dir = _find_venv(cwd) or _default_env(rs, workspace or cwd)
    if venv_dir is not None:
        _old_path = base_env.get("PATH", os.environ.get("PATH", ""))
        base_env["PATH"] = os.path.join(venv_dir, "bin") + os.pathsep + _old_path
        base_env["VIRTUAL_ENV"] = venv_dir
        base_env.pop("PYTHONHOME", None)

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
            master_port = str(dist_master_port(msg.experiment, msg.run_id))
        _dist_base = {
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        }

    _check_cancelled(rs)

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
    except OSError as exc:
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass
        raise RuntimeError(f"could not start {msg.command[0]!r}: {exc}") from exc

    # A cancel that raced with the spawn found no pids to signal; honour it now.
    with _lock:
        rs.pids = pids
        cancelled = rs.cancelled
    if cancelled:
        _signal_pids(pids)

    _send_run_msg(encode(MsgStarted(run_id=msg.run_id, experiment=msg.experiment, pid=pids[0])))

    t = threading.Thread(
        target=_run_thread,
        args=(procs, rs, artifacts_path, workspace or cwd, msg.return_files),
        daemon=True,
        name=f"run-{msg.run_id}",
    )
    t.start()


def _run_thread(
    procs: "list[subprocess.Popen[bytes]]",
    state: RunState,
    artifacts_path: str,
    run_dir: str,
    return_files: list[str],
) -> None:
    """Monitor all per-GPU subprocesses: stream rank-0 logs, send MsgResult when all exit."""
    t0 = time.time()

    # Stream rank-0 stdout to the log
    assert procs[0].stdout is not None
    for raw_line in procs[0].stdout:
        _append_log(state, raw_line)
    _close_log(state)

    # Wait for all ranks to finish
    rcs = [procs[0].wait()] + [p.wait() for p in procs[1:]]
    elapsed = time.time() - t0
    exit_code = next((rc for rc in rcs if rc != 0), 0)

    # Copy return_files into artifacts/ before rsync.
    # run_dir is the workspace (when files={...}) or the cwd (when files={}).
    if return_files:
        for rel_path in return_files:
            src = _resolve_safe_subpath(run_dir, rel_path)
            if os.path.isfile(src):
                dst = _resolve_safe_subpath(artifacts_path, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    result = MsgResult(
        run_id=state.run_id,
        experiment=state.experiment,
        success=(exit_code == 0),
        elapsed=elapsed,
        exit_code=exit_code,
    )
    # The processes are gone, but their loggers' last metrics may still be unread
    # in the IPC sockets.  Record them before the result, which ends the run.
    with _ipc_cond:
        _ipc_cond.wait_for(lambda: state.ipc_conns == 0 and _ipc_unclaimed == 0,
                           timeout=_IPC_DRAIN_TIMEOUT)
    # Move the run from in-flight to unacknowledged in one step, so a hello snapshot
    # taken at any moment reports it as either resuming or completed.
    with _lock:
        _in_flight.pop((state.experiment, state.run_id), None)
        _unacked_results[(state.experiment, state.run_id)] = result

    # Free the workspace (large extracted artifact copy); keep logs and output artifacts.
    workspace_dir = os.path.join(state.scratch_path, "workspace")
    if os.path.isdir(workspace_dir):
        try:
            shutil.rmtree(workspace_dir)
        except OSError:
            pass

    _send_run_msg(encode(result))


def _handle_cleanup(msg: MsgCleanup, conn: ConnState) -> None:
    """Handle a cleanup request.

    The controller sends ``final=True`` only after it has rsynced the run's
    artifacts, logs, and metrics into the persistent output directory, so the
    scratch directory is safe to delete at that point.  Mid-run syncs arrive
    with ``final=False`` and must leave the scratch intact.
    """
    # Any MsgCleanup for a run means the manager has processed its result.
    with _lock:
        _unacked_results.pop((msg.experiment, msg.run_id), None)
    if msg.final and msg.experiment:
        try:
            exp_dir = _resolve_safe_subpath(_scratch_dir, msg.experiment)
            run_dir = _resolve_safe_subpath(exp_dir, msg.run_id)
        except ValueError:
            run_dir = ""
        else:
            # Only remove when run_dir is a single component directly under
            # exp_dir (guards against path traversal in the message fields).
            if os.path.dirname(run_dir) != exp_dir or not os.path.isdir(run_dir):
                run_dir = ""
        if run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)
    if not conn.closed:
        conn.send_queue.put(encode(MsgCleaned(run_id=msg.run_id, experiment=msg.experiment)))


def _signal_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _handle_cancel(msg: MsgCancel) -> None:
    """Cancel a run: SIGTERM its processes, or stop it before they start."""
    with _lock:
        state = _in_flight.get((msg.experiment, msg.run_id))
        if state is None:
            return
        state.cancelled = True
        pids = list(state.pids)
        setup_proc = state.setup_proc
    if setup_proc is not None:
        try:
            os.killpg(setup_proc.pid, signal.SIGTERM)
        except OSError:
            pass
    _signal_pids(pids)


def _handle_replay(msg: MsgReplay, conn: ConnState) -> None:
    """Re-send a running run's log from the requested offset, and its metrics."""
    with _lock:
        state = _in_flight.get((msg.experiment, msg.run_id))
    if state is None:
        return
    t = threading.Thread(
        target=_replay_thread,
        args=(msg.run_id, state, msg.log_seq, conn),
        daemon=True,
        name=f"replay-{msg.run_id}",
    )
    t.start()


def _replay_thread(
    run_id: str,
    state: RunState,
    log_seq: int,
    conn: ConnState,
) -> None:
    """Re-send the log from *log_seq* on, and every metric line.

    The log is re-sent up to what was written so far, which supersedes the
    lines still waiting to be sent; later lines continue from there.
    """
    log_path = os.path.join(state.scratch_path, "training.log")
    metrics_path = os.path.join(state.scratch_path, "metrics.jsonl")

    with state.log_lock:
        try:
            with open(log_path, "rb") as f:
                f.seek(log_seq)
                missed = f.read(max(0, state.log_seq - log_seq))
        except OSError:
            missed = b""
        start = log_seq
        for chunk in line_chunks(missed):
            if not conn.closed:
                conn.send_queue.put(_log_msg(state, start, chunk))
            start += len(chunk)
        state.pending_log.clear()
        state.sent_seq = state.log_seq

        # Replay every metric line; the manager ignores steps it already has.
        try:
            with open(metrics_path, "rb") as f:
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
                                run_id=run_id, experiment=state.experiment,
                                step=step, data=data,
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
    """Handle one IPC connection from a training script.

    The connection counts toward its run's ``ipc_conns`` from its first message
    until it closes, so the run's result is only sent once it has drained.
    """
    global _ipc_unclaimed
    with _lock:
        _ipc_unclaimed += 1
    claimed = False
    rs: RunState | None = None
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
                if not claimed:
                    with _ipc_cond:
                        claimed = True
                        _ipc_unclaimed -= 1
                        rs = _ipc_run_state_locked(msg)
                        if rs is not None:
                            rs.ipc_conns += 1
                        _ipc_cond.notify_all()
                _handle_ipc_msg(msg)
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass
        with _ipc_cond:
            if not claimed:
                _ipc_unclaimed -= 1
            elif rs is not None:
                rs.ipc_conns -= 1
            _ipc_cond.notify_all()


def _ipc_run_state_locked(msg: dict[str, Any]) -> RunState | None:
    """The in-flight run an IPC message belongs to.  The caller holds ``_lock``."""
    run_id = msg.get("run_id", "")
    experiment = msg.get("experiment", "")
    if experiment:
        return _in_flight.get((experiment, run_id))
    # Loggers from older mlsweep versions send only the run name.
    return next((rs for rs in _in_flight.values() if rs.run_id == run_id), None)


def _handle_ipc_msg(msg: dict[str, Any]) -> None:
    """Route an IPC message (metric or sync) from a training script."""
    run_id = msg.get("run_id", "")
    msg_type = msg.get("type")

    with _lock:
        state = _ipc_run_state_locked(msg)

    if state is None:
        return

    metrics_path = os.path.join(state.scratch_path, "metrics.jsonl")

    if msg_type == "metric":
        step = int(msg.get("step", 0))
        data: dict[str, Any] = msg.get("data", {})
        record: dict[str, Any] = {"step": step, **data}
        line = json.dumps(record) + "\n"
        try:
            with open(metrics_path, "a") as f:
                f.write(line)
        except OSError:
            pass
        _send_run_msg(encode(MsgMetric(
            run_id=run_id, experiment=state.experiment, step=step, data=data)))

    elif msg_type == "sync":
        _send_run_msg(encode(MsgSyncReq(run_id=run_id, experiment=state.experiment)))


# ── GPU stats polling ─────────────────────────────────────────────────────────


def _query_gpu_stats() -> list[dict[str, Any]]:
    """Return per-GPU utilization stats from nvidia-smi or rocm-smi."""
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                stats: list[dict[str, Any]] = []
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 4:
                        continue
                    try:
                        stats.append({
                            "gpu": int(parts[0]),
                            "util_pct": int(parts[1]),
                            "mem_used_mb": int(parts[2]),
                            "mem_total_mb": int(parts[3]),
                        })
                    except ValueError:
                        continue
                return stats
        except Exception:
            pass

    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                data = json.loads(r.stdout)
                stats = []
                for key, val in sorted(data.items()):
                    if not isinstance(val, dict):
                        continue
                    try:
                        gpu_id = int(key.lstrip("card"))
                        used_str = val.get("VRAM Total Used Memory (B)", "0")
                        total_str = val.get("VRAM Total Memory (B)", "0")
                        util_str = val.get("GPU use (%)", "0")
                        stats.append({
                            "gpu": gpu_id,
                            "util_pct": int(float(util_str)),
                            "mem_used_mb": int(int(used_str) / 1024 / 1024),
                            "mem_total_mb": int(int(total_str) / 1024 / 1024),
                        })
                    except (ValueError, KeyError):
                        continue
                return stats
        except Exception:
            pass

    return []


def _gpu_stats_thread() -> None:
    """Periodically poll GPU stats and broadcast to all manager connections."""
    while not _shutdown_event.wait(5.0):
        stats = _query_gpu_stats()
        if not stats:
            continue
        # Filter to our assigned GPUs only
        if _device_override is not None:
            our_gpus = set(_device_override)
            stats = [s for s in stats if s["gpu"] in our_gpus]
        if not stats:
            continue
        wire = encode(MsgGpuStats(stats=stats))
        with _lock:
            conns = list(_connections)
        for conn in conns:
            if not conn.closed:
                try:
                    conn.send_queue.put_nowait(wire)
                except Exception:
                    pass


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
        global _scratch_dir, _remote_dir, _token, _device_override, _max_jobs_per_gpu

        parser = argparse.ArgumentParser(description="mlsweep worker daemon")
        parser.add_argument("--token", default="", help="Authentication token")
        parser.add_argument("--scratch-dir", default="/tmp/mlsweep",
                            help="Base scratch directory for run buffers (default: /tmp/mlsweep)")
        parser.add_argument("--remote-dir", default="",
                            help="Project directory on this machine (cwd for training scripts)")
        parser.add_argument("-g", "--devices", default=None,
                            help="Comma-separated GPU device IDs to expose, e.g. 4,5,6,7 "
                                 "(default: all visible)")
        parser.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                            help="Max concurrent jobs per GPU on this worker "
                                 "(0 = unlimited, default: 1)")
        parser.add_argument("--port", type=int, default=7890,
                            help="TCP port to bind (0 = ephemeral, default: 7890)")
        parser.add_argument("--color", action="store_true",
                            help="Enable ANSI color in human-readable output (default: off)")
        parser.add_argument("--version", action="version",
                            version=f"%(prog)s {importlib.metadata.version('mlsweep')}")
        args = parser.parse_args()

        if args.color:
            set_color(True)

        _scratch_dir = args.scratch_dir
        _remote_dir = args.remote_dir or os.getcwd()
        _token = args.token
        _max_jobs_per_gpu = args.jobs
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

        # Start IPC thread for logger.py connections.
        # Use a port-specific socket name so multiple workers on the same host
        # (e.g., concurrent test workers) don't clobber each other's sockets.
        global _ipc_sock_path
        _ipc_sock_path = os.path.join(_scratch_dir, f".worker-{port}.sock")
        ipc_t = threading.Thread(target=_ipc_thread, args=(_ipc_sock_path,), daemon=True)
        ipc_t.start()

        gpu_t = threading.Thread(target=_gpu_stats_thread, daemon=True)
        gpu_t.start()
        threading.Thread(target=_log_flush_thread, daemon=True).start()

        # Enter accept loop (blocks until _shutdown_event is set)
        _accept_loop(server_sock)


    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
