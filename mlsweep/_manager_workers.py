"""Async worker connection management for mlsweep manager.

Ports the worker lifecycle logic from ``run_sweep.py`` (sync threads) to
asyncio tasks.  Provides:

  * Worker launch (local subprocess or remote SSH)
  * Per-connection read/write/heartbeat tasks
  * Reconnect with exponential backoff
  * Handlers for every Worker → Controller protocol message
  * Integration with ``ManagerState`` and the SQLite database
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any

import aiosqlite

from mlsweep._manager_db import (
    JobRecord,
    WorkerRecord,
    dispatch_job,
    finish_job,
    get_experiment,
    increment_retry,
    mark_job_running,
    reclassify_singular_xfails,
    update_experiment_status,
    update_job_status,
    update_worker_status,
    upsert_worker,
)
from mlsweep._manager_state import InFlightJob, ManagerState, WorkerConn
from mlsweep._parsync import parsync_bin
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
    _git_root,
    decode,
    encode,
)
from mlsweep._topology import _best_gpu_groups, _gpu_topology, _parse_topo_wire, visible_devices

# ===============================================================================
# ANSI helpers (kept minimal — the manager may want its own later)
# ===============================================================================

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


# ===============================================================================
# Worker configuration parsing (ported from run_sweep._parse_workers)
# ===============================================================================


def _parse_workers_file(
    path: str,
) -> list[dict[str, Any]]:
    """Parse a TOML workers file.

    Each ``[[workers]]`` entry requires ``host`` and ``remote_dir``.
    Optional fields: gpus, jobs, devices, pass, ssh_key, venv, port.

    Returns a list of dicts suitable for ``launch_worker()``.
    """
    try:
        import tomllib  # type: ignore[import-not-found]  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    with open(path, "rb") as f:
        data = tomllib.load(f)

    global_pass = os.environ.get("MLSWEEP_SSH_PASS")
    result: list[dict[str, Any]] = []

    for i, entry in enumerate(data.get("workers", [])):
        host = entry.get("host")
        remote_dir = entry.get("remote_dir")
        if not host or not remote_dir:
            raise ValueError(
                f"{path}: workers entry {i + 1} missing required field 'host' or 'remote_dir'"
            )
        result.append(
            {
                "host": host,
                "remote_dir": remote_dir,
                "gpus": entry.get("gpus"),
                "jobs": entry.get("jobs"),
                "devices": entry.get("devices"),
                "password": entry.get("pass") or global_pass,
                "ssh_key": entry.get("ssh_key"),
                "venv": entry.get("venv") or remote_dir,
                "port": entry.get("port", 7890),
            }
        )
    return result


# ===============================================================================
# Worker launch helpers (ported from run_sweep.py)
# ===============================================================================


def _worker_candidates(venv: str | None) -> list[str]:
    """Return candidate ``mlsweep_worker`` binary paths, given a venv specifier.

    The bootstrapped ``/tmp/mlsweep_venv/bin/mlsweep_worker`` is always
    the highest-priority candidate.  After that the configured *venv* is
    tried, and finally ``mlsweep_worker`` on PATH.
    """
    candidates: list[str] = ["/tmp/mlsweep_venv/bin/mlsweep_worker"]
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
    """Return a self-contained shell command that execs the first available worker binary."""
    args_str = shlex.join(worker_args)
    paths_str = " ".join(shlex.quote(c) for c in candidates)
    return (
        f"for _p in {paths_str}; do\n"
        f"    [ -x \"$_p\" ] && exec \"$_p\" {args_str}\n"
        f"done\n"
        f"echo 'mlsweep: mlsweep_worker not found (tried: {paths_str})' >&2; exit 1"
    )


async def _bootstrap_worker_venv(
    host: str,
    ssh_key: str | None = None,
    password: str | None = None,
) -> bool:
    """Ensure ``/tmp/mlsweep_venv/bin/mlsweep_worker`` exists on *host*.

    If the binary already exists this returns ``True`` immediately.
    Otherwise it SCPs the bundled wheels to ``/tmp/mlsweep_wheels/``,
    creates the venv, and pip-installs ``mlsweep`` into it.

    Returns ``True`` on success, ``False`` if any step fails.
    """
    key_args = ["-i", ssh_key] if ssh_key else []

    # 1. Quick check: is the binary already present?
    try:
        proc = await asyncio.create_subprocess_exec(
            *_sshpass_prefix(password),
            "ssh", "-o", "ConnectTimeout=10",
            *key_args,
            host,
            "test -x /tmp/mlsweep_venv/bin/mlsweep_worker && echo OK || echo MISSING",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        if b"OK" in stdout:
            return True
    except (OSError, asyncio.TimeoutError):
        pass  # fall through to bootstrap

    # 2. SCP the bundled wheels to /tmp/mlsweep_wheels/
    wheels_src = str(Path(__file__).resolve().parent / "_wheels") + "/"
    try:
        proc = await asyncio.create_subprocess_exec(
            *_sshpass_prefix(password),
            "scp", "-r", "-o", "ConnectTimeout=10",
            *key_args,
            wheels_src,
            f"{host}:/tmp/mlsweep_wheels/",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            print(
                f"[bootstrap] scp failed for {host}: "
                f"{stderr.decode(errors='replace')[:200]}",
                file=sys.stderr,
            )
            return False
    except (OSError, asyncio.TimeoutError) as e:
        print(f"[bootstrap] scp failed for {host}: {e}", file=sys.stderr)
        return False

    # 3. Create venv and install mlsweep from local wheels
    try:
        proc = await asyncio.create_subprocess_exec(
            *_sshpass_prefix(password),
            "ssh", "-o", "ConnectTimeout=10",
            *key_args,
            host,
            (
                "python3 -m venv /tmp/mlsweep_venv && "
                "/tmp/mlsweep_venv/bin/pip install --no-index "
                "--find-links=/tmp/mlsweep_wheels/ mlsweep"
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        if proc.returncode != 0:
            print(
                f"[bootstrap] install failed for {host}: "
                f"{stderr.decode(errors='replace')[:200]}",
                file=sys.stderr,
            )
            return False
        return True
    except (OSError, asyncio.TimeoutError) as e:
        print(f"[bootstrap] install failed for {host}: {e}", file=sys.stderr)
        return False


_sshpass_available: bool | None = None


def _sshpass_prefix(password: str | None) -> list[str]:
    """Return ``['sshpass', '-p', password]`` if a password is given, else ``[]``."""
    global _sshpass_available
    if not password:
        return []
    if _sshpass_available is None:
        _sshpass_available = shutil.which("sshpass") is not None
    if not _sshpass_available:
        raise RuntimeError("sshpass is not installed but a password was specified")
    return ["sshpass", "-p", password]


# ===============================================================================
# Worker launch (async)
# ===============================================================================


async def launch_worker(
    host: str,
    remote_dir: str,
    token: str,
    scratch_dir: str = "/tmp/mlsweep",
    devices: list[int] | None = None,
    password: str | None = None,
    ssh_key: str | None = None,
    venv: str | None = None,
    port: int = 0,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, int]:
    """Launch a worker process and return connected streams + port.

    For ``host == "localhost"``, spawns ``python -m mlsweep.worker`` as a
    local subprocess.  For remote hosts, connects via SSH and runs the
    ``mlsweep_worker`` binary.

    Returns ``(reader, writer, port)`` where *reader* and *writer* are
    asyncio stream objects connected to the worker's TCP port.
    """
    connect_host = "localhost" if host == "localhost" else host.split("@")[-1]
    devices_args = (
        ["--devices", ",".join(str(d) for d in devices)] if devices else []
    )
    key_args = ["-i", ssh_key] if ssh_key else []
    bind_port = port

    # ── Try to reuse an existing worker at the fixed port ────────────────
    if bind_port != 0:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(connect_host, bind_port),
                timeout=2.0,
            )
            return reader, writer, bind_port
        except (OSError, asyncio.TimeoutError):
            pass

    # ── Launch a fresh worker ────────────────────────────────────────────
    if host == "localhost":
        # Local: spawn python -m mlsweep.worker
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mlsweep.worker",
            "--token", token,
            "--remote-dir", remote_dir,
            "--scratch-dir", scratch_dir,
            "--port", str(bind_port),
            *devices_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        # Remote: bootstrap venv if needed, then SSH and run worker binary
        ok = await _bootstrap_worker_venv(
            host, ssh_key=ssh_key, password=password,
        )
        if not ok:
            raise RuntimeError(
                f"failed to bootstrap /tmp/mlsweep_venv on {host}"
            )
        worker_args = [
            "--token", token,
            "--remote-dir", remote_dir,
            "--port", str(bind_port),
            *devices_args,
        ]
        shell_cmd = _worker_shell_cmd(_worker_candidates(venv), worker_args)
        ssh_cmd = [
            *_sshpass_prefix(password),
            "ssh", "-o", "ConnectTimeout=10",
            *key_args,
            host, shell_cmd,
        ]
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    assert proc.stdout is not None and proc.stderr is not None

    # Read the PORT= line from stdout
    try:
        line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"worker on {host} timed out waiting for PORT= line")

    line = line_bytes.decode().strip()

    if not line.startswith("PORT="):
        # Worker failed — read stderr and diagnose
        stderr_bytes = await proc.stderr.read()
        stderr_out = stderr_bytes.decode(errors="replace").strip()
        await proc.wait()

        hint = ""
        if "Permission denied" in stderr_out or "Authentication failed" in stderr_out:
            hint = "\n  hint: authentication failed — check ssh_key / pass / MLSWEEP_SSH_PASS"
        elif "Host key verification failed" in stderr_out:
            hint = "\n  hint: host key not in known_hosts — ssh to the machine manually first"
        elif "Connection refused" in stderr_out:
            hint = "\n  hint: connection refused — is the host reachable on port 22?"
        elif "Connection timed out" in stderr_out or "Operation timed out" in stderr_out:
            hint = "\n  hint: connection timed out — check the hostname/IP and firewall"
        elif "Could not resolve" in stderr_out or "Name or service not known" in stderr_out:
            hint = "\n  hint: hostname not found — check the host field in workers.toml"
        elif "No module named mlsweep" in stderr_out:
            hint = "\n  hint: mlsweep is not installed on the remote machine"
        elif "python: command not found" in stderr_out or "python3: command not found" in stderr_out:
            hint = "\n  hint: python not found on remote — is it in PATH?"
        elif "UNPROTECTED PRIVATE KEY" in stderr_out:
            hint = "\n  hint: ssh_key permissions are too open — run: chmod 600 <key>"
        last_line = stderr_out.splitlines()[-1] if stderr_out else (line or "(no output)")
        raise RuntimeError(f"worker failed to start on {host}: {last_line}{hint}")

    worker_port = int(line.split("=")[1])

    # Connect to the worker's TCP port
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(connect_host, worker_port),
        timeout=10.0,
    )
    return reader, writer, worker_port


# ===============================================================================
# Per-connection tasks (read / write / heartbeat)
# ===============================================================================


async def _worker_write_task(
    wc: WorkerConn,
    shutdown_event: asyncio.Event,
) -> None:
    """Drain *wc.send_queue* and write every message to the socket.

    A ``None`` sentinel or an ``OSError`` causes the task to exit.
    """
    while True:
        if shutdown_event.is_set():
            return
        try:
            item = await wc.send_queue.get()
        except RuntimeError:
            # Queue closed
            return
        if item is None:
            return
        try:
            wc.writer.write(item)
            await wc.writer.drain()
        except OSError:
            return


async def _worker_heartbeat_task(
    wc: WorkerConn,
    shutdown_event: asyncio.Event,
    interval: float = 10.0,
) -> None:
    """Send ``MsgPing`` every *interval* seconds to keep the connection alive."""
    while True:
        await asyncio.sleep(interval)
        if wc.status not in ("connected", "connecting"):
            return
        if shutdown_event.is_set():
            return
        try:
            wc.send_queue.put_nowait(encode(MsgPing()))
        except asyncio.QueueFull:
            pass


# ── Message handlers ──────────────────────────────────────────────────────────
#
# Each handler is an async function that receives:
#   db: aiosqlite.Connection     – database handle
#   state: ManagerState    – in-memory state
#   wc: WorkerConn         – the worker connection that sent the message
#   msg: ...               – the decoded protocol message
#   Each handler takes (db, state, wc, msg) with no extra context.
#
# They are called by the read task in sequence (no concurrency per worker).
# ────────────────────────────────────────────────────────────────────────────────


def _check_all_workers_ready(
    state: ManagerState,
    workers_ready: asyncio.Event,
) -> None:
    """Set *workers_ready* if every worker in state has finished its hello handshake."""
    if all(
        wc.status in ("connected", "dead")
        for wc in state.workers.values()
    ):
        workers_ready.set()


async def _handle_worker_hello(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgWorkerHello,
    *,
    workers_ready: asyncio.Event | None = None,
) -> None:
    """Handle ``MsgWorkerHello``: initialise GPU occupancy, mark connected."""
    async with state.scheduler_lock:
        wc.gpus = msg.gpus
        wc.topo = msg.topo
        wc.scratch_dir = msg.scratch_dir

        # Initialise GPU occupancy tracking
        wc.gpu_occupancy = {g: 0 for g in wc.gpus}

        # Mark connected
        wc.status = "connected"
        wc.connected_at = datetime.now(timezone.utc)

        # Persist to DB
        await upsert_worker(
            db,
            worker_id=wc.worker_id,
            host=wc.host,
            remote_dir=wc.remote_dir,
            scratch_dir=msg.scratch_dir,
            port=wc.port,
            devices=json.dumps(msg.gpus),
            status="connected",
        )

        n_gpus = len(msg.gpus)
        gpu_plural = "s" if n_gpus != 1 else ""
        print(
            f"  {_GREEN}OK{_RESET}    {wc.host}: {n_gpus} GPU{gpu_plural} available"
        )

        # If all workers have completed hello, signal the manager
        if workers_ready is not None:
            _check_all_workers_ready(state, workers_ready)

    # Broadcast worker-connected event
    state.broadcast(
        "*",  # global broadcast
        {
            "type": "worker_connected",
            "worker_id": wc.worker_id,
            "host": wc.host,
            "gpus": msg.gpus,
        },
    )

    # Trigger scheduling so any jobs submitted before this worker connected
    # get dispatched now that we have capacity.
    if state.dispatch_callback is not None:
        asyncio.create_task(state.dispatch_callback())


async def _handle_started(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgStarted,
) -> None:
    """Handle ``MsgStarted``: create run directory, open log handles, mark job running."""
    async with state.scheduler_lock:
        job = state.get_in_flight(msg.run_id)
        if job is not None:
            job.start_time = datetime.now(timezone.utc)

    if job is not None and state.output_dir:
        run_dir = os.path.join(state.output_dir, job.experiment_id, msg.run_id)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "training.log")
        try:
            fh = open(log_path, "a")
            wc.log_handles[msg.run_id] = fh
        except OSError:
            pass

    await mark_job_running(db, msg.run_id, job.experiment_id if job else "")

    state.broadcast(
        job.experiment_id if job else "*",
        {
            "type": "job_started",
            "run_id": msg.run_id,
            "worker_id": wc.worker_id,
            "pid": msg.pid,
        },
    )


async def _handle_log(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgLog,
) -> None:
    """Handle ``MsgLog``: write to log file and broadcast."""
    job = state.get_in_flight(msg.run_id)
    fh = wc.log_handles.get(msg.run_id)
    if fh is not None:
        fh.write(msg.data)
        fh.flush()

    if job:
        state.broadcast(
            job.experiment_id,
            {
                "type": "log",
                "run_id": msg.run_id,
                "seq": msg.seq,
                "data": msg.data,
            },
        )


async def _handle_metric(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgMetric,
) -> None:
    """Handle ``MsgMetric``: broadcast to WebSocket subscribers."""
    job = state.get_in_flight(msg.run_id)

    if job:
        state.broadcast(
            job.experiment_id,
            {
                "type": "metric",
                "run_id": msg.run_id,
                "step": msg.step,
                "data": msg.data,
            },
        )


async def _handle_sync_req(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgSyncReq,
) -> None:
    """Handle ``MsgSyncReq``: start artifact sync."""
    job = state.get_in_flight(msg.run_id)
    if job is None:
        return

    run_scratch = os.path.join(wc.scratch_dir, job.experiment_id, msg.run_id)
    run_dir = os.path.join(state.output_dir, job.experiment_id, msg.run_id)

    # Run rsync in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        _rsync_sync,
        wc.host,
        run_scratch,
        run_dir,
        msg.run_id,
        wc.password,
        wc.ssh_key,
    )

    # After sync, send MsgCleanup
    await _send_to_worker(wc, encode(MsgCleanup(run_id=msg.run_id)))


def _rsync_sync(
    worker_host: str,
    remote_scratch: str,
    local_run_dir: str,
    run_id: str,
    password: str | None = None,
    ssh_key: str | None = None,
) -> None:
    """Synchronous artifact sync (runs in executor thread)."""
    if worker_host == "localhost":
        os.makedirs(local_run_dir, exist_ok=True)
        src_artifacts = os.path.join(remote_scratch, "artifacts")
        dst_artifacts = os.path.join(local_run_dir, "artifacts")
        if src_artifacts != dst_artifacts and os.path.isdir(src_artifacts):
            try:
                if os.path.exists(dst_artifacts):
                    shutil.rmtree(dst_artifacts)
                shutil.copytree(src_artifacts, dst_artifacts)
            except OSError as e:
                print(f"  {_YELLOW}WARN{_RESET}  parsync failed for {run_id}: {e}")
        for fname in ("metrics.jsonl", "training.log"):
            src_f = os.path.join(remote_scratch, fname)
            dst_f = os.path.join(local_run_dir, fname)
            if src_f != dst_f and os.path.isfile(src_f):
                try:
                    shutil.copy2(src_f, dst_f)
                except OSError as e:
                    print(f"  {_YELLOW}WARN{_RESET}  copy {fname} failed for {run_id}: {e}")
    else:
        env = os.environ.copy()
        if password:
            env["PARSYNC_SSH_PASSWORD"] = password
        result = subprocess.run(
            [
                parsync_bin(),
                "-rlu",
                f"{worker_host}:{remote_scratch}/",
                f"{local_run_dir}/",
            ],
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            print(
                f"  {_YELLOW}WARN{_RESET}  parsync failed for {run_id}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )


async def _handle_result(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgResult,
) -> None:
    """Handle ``MsgResult``: finish job, free GPU occupancy, trigger re-schedule.

    For multi-node jobs (``nodes_per_run > 1``), aggregates results from
    individual workers before finalising.  Only the last arriving result
    triggers the DB update, broadcast, artifact sync, and dispatch.
    """
    async with state.scheduler_lock:
        job = state.get_in_flight(msg.run_id)

        # Free GPU occupancy for this worker regardless
        if job is not None:
            for gpu_id in job.gpu_ids:
                wc.gpu_occupancy[gpu_id] = max(0, wc.gpu_occupancy[gpu_id] - 1)

        wc.in_flight.pop(msg.run_id, None)

        is_multinode: bool
        if job is not None:
            is_multinode = len(job.worker_ids) > 1
        else:
            is_multinode = False

        if not is_multinode:
            job = state.remove_in_flight(msg.run_id)

    # ── Multi-node aggregation ─────────────────────────────────────────

    if is_multinode:
        async with state.scheduler_lock:
            assert job is not None
            agg = state.multinode_pending.get(msg.run_id)
            if agg is None:
                # First result for this run — initialise remaining count
                agg = {
                    "remaining": len(job.worker_ids),
                    "success": True,
                    "elapsed": 0.0,
                }
                state.multinode_pending[msg.run_id] = agg

            agg["remaining"] -= 1
            if not msg.success:
                agg["success"] = False
            if msg.elapsed > agg["elapsed"]:
                agg["elapsed"] = msg.elapsed

            if agg["remaining"] > 0:
                return

            del state.multinode_pending[msg.run_id]
            job = state.remove_in_flight(msg.run_id)
            final_success = agg["success"]
            final_elapsed = agg["elapsed"]
    else:
        final_success = msg.success
        final_elapsed = msg.elapsed

    # Persist to DB
    await finish_job(
        db,
        msg.run_id,
        job.experiment_id if job else "",
        success=final_success,
        exit_code=msg.exit_code,
        elapsed=final_elapsed,
    )

    # Retroactively reclassify singular-probe failures as xfailed
    xfailed_ids: list[str] = []
    if final_success and job is not None:
        xfailed_ids = await reclassify_singular_xfails(db, job.experiment_id, job.combo)

    # Close log file handle
    fh = wc.log_handles.pop(msg.run_id, None)
    if fh is not None:
        try:
            fh.close()
        except OSError:
            pass

    # Broadcast
    if job:
        state.broadcast(
            job.experiment_id,
            {
                "type": "job_done",
                "run_id": msg.run_id,
                "worker_id": wc.worker_id,
                "success": final_success,
                "elapsed": final_elapsed,
                "exit_code": msg.exit_code,
                "xfailed": xfailed_ids,
            },
        )

    # Sync artifacts after run completes
    if job is not None:
        run_scratch = os.path.join(wc.scratch_dir, job.experiment_id, msg.run_id)
        run_dir = os.path.join(state.output_dir, job.experiment_id, msg.run_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _rsync_sync,
            wc.host,
            run_scratch,
            run_dir,
            msg.run_id,
            wc.password,
            wc.ssh_key,
        )

    # Send cleanup after sync
    await _send_to_worker(wc, encode(MsgCleanup(run_id=msg.run_id)))

    # Trigger dispatch
    if state.dispatch_callback is not None:
        await state.dispatch_callback()

    # ── Check if experiment is complete ─────────────────────────────────
    if job is not None:
        experiment_id = job.experiment_id
        async with state.scheduler_lock:
            has_active = any(
                j.experiment_id == experiment_id
                for j in state.in_flight.values()
            )
            has_pending = any(
                j.experiment_id == experiment_id
                for j in state.pending
            )
            if has_active or has_pending:
                return

            exp = await get_experiment(db, experiment_id)
            if exp is None:
                return

            cursor = await db.execute(
                "SELECT COUNT(*) FROM jobs WHERE experiment_id = ? AND status = 'done'",
                (experiment_id,),
            )
            row = await cursor.fetchone()
            submitted_count: int = row[0] if row else 0
            expected = exp.expected_jobs
            if expected != 0 and submitted_count < expected:
                return

            await update_experiment_status(db, experiment_id, "completed")

        state.broadcast(
            experiment_id,
            {
                "type": "experiment_done",
                "experiment_id": experiment_id,
                "submitted_count": submitted_count,
            },
        )


async def _handle_cleaned(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgCleaned,
) -> None:
    """Handle ``MsgCleaned`` — nothing to do currently."""
    pass


async def _handle_pong(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgPong,
) -> None:
    """Handle ``MsgPong`` — update last_seen so the UI reflects a live worker."""
    from mlsweep._manager_db import touch_worker
    await touch_worker(db, wc.worker_id)


# ── Message dispatch table ─────────────────────────────────────────────────────

_HANDLERS: dict[type, Any] = {
    MsgWorkerHello: _handle_worker_hello,
    MsgStarted: _handle_started,
    MsgLog: _handle_log,
    MsgMetric: _handle_metric,
    MsgSyncReq: _handle_sync_req,
    MsgResult: _handle_result,
    MsgCleaned: _handle_cleaned,
    MsgPong: _handle_pong,
}


async def _send_to_worker(wc: WorkerConn, data: bytes) -> bool:
    """Try to enqueue *data* to the worker's send queue. Returns ``True`` on success."""
    try:
        wc.send_queue.put_nowait(data)
        return True
    except asyncio.QueueFull:
        return False


# ===============================================================================
# Main read task — runs per worker connection
# ===============================================================================


async def _worker_read_task(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    *,
    workers_ready: asyncio.Event | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Read lines from *wc.reader*, decode, and dispatch to handlers.

    This task runs until the connection is lost or the shutdown event is set.
    On disconnect, it triggers the reconnect logic.
    """
    # First message MUST be MsgWorkerHello
    try:
        line = await asyncio.wait_for(wc.reader.readline(), timeout=30.0)
    except (asyncio.TimeoutError, OSError):
        await _on_worker_lost(db, state, wc, shutdown_event)
        return

    if not line:
        await _on_worker_lost(db, state, wc, shutdown_event)
        return

    try:
        msg = decode(line)
    except (ValueError, json.JSONDecodeError, TypeError):
        await _on_worker_lost(db, state, wc, shutdown_event)
        return

    if not isinstance(msg, MsgWorkerHello):
        await _on_worker_lost(db, state, wc, shutdown_event)
        return

    await _handle_worker_hello(db, state, wc, msg, workers_ready=workers_ready)

    # Main message loop
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            return
        if wc.status not in ("connected", "connecting"):
            return

        try:
            line = await asyncio.wait_for(wc.reader.readline(), timeout=60.0)
        except asyncio.TimeoutError:
            continue
        except OSError:
            break

        if not line:
            break

        try:
            msg = decode(line)
        except (ValueError, json.JSONDecodeError, TypeError):
            continue

        handler = _HANDLERS.get(type(msg))
        if handler is not None:
            try:
                await handler(db, state, wc, msg)
            except Exception as e:
                print(f"  {_RED}ERROR{_RESET} handling {type(msg).__name__} "
                      f"from {wc.host}: {e}")

    # Connection lost — trigger reconnect
    await _on_worker_lost(db, state, wc, shutdown_event)


async def _on_worker_lost(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Called when the TCP connection to a worker is lost.

    Marks the worker as disconnected and starts a reconnect task.
    """
    if wc.status not in ("connected", "connecting"):
        return

    async with state.scheduler_lock:
        wc.status = "reconnecting"

    await update_worker_status(db, wc.worker_id, "reconnecting")

    print(f"  {_YELLOW}WARN{_RESET}  Worker {wc.host} disconnected; reconnecting...")

    # Start reconnect task (fire-and-forget)
    asyncio.create_task(
        _reconnect_worker(db, state, wc, shutdown_event)
    )


# ===============================================================================
# Reconnect logic
# ===============================================================================


async def _reconnect_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    shutdown_event: asyncio.Event | None = None,
    max_attempts: int = 10,
) -> None:
    """Try to reconnect to a worker with exponential backoff.

    On success, sends ``MsgReplay`` for resuming runs and re-queues orphaned
    runs (with retry logic).
    """
    backoff = 1.0
    for _ in range(max_attempts):
        if shutdown_event is not None and shutdown_event.is_set():
            return

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(wc.host, wc.port),
                timeout=5.0,
            )

            writer.write(encode(MsgHello(token=state.token, controller_id="manager")))
            await writer.drain()

            # Read MsgWorkerHello
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not line:
                writer.close()
                continue

            msg = decode(line)
            if not isinstance(msg, MsgWorkerHello):
                writer.close()
                continue

            # Success — attach new streams
            async with state.scheduler_lock:
                wc.reader = reader
                wc.writer = writer
                wc.status = "connected"
                wc.connected_at = datetime.now(timezone.utc)

            await update_worker_status(db, wc.worker_id, "connected")

            print(f"  {_GREEN}OK{_RESET}    Worker {wc.host} reconnected")

            # Reconstruct gpu_occupancy from DB for resuming runs (single batch query)
            if msg.resuming:
                resuming_ids = [r["run_id"] for r in msg.resuming]
                placeholders = ",".join("?" * len(resuming_ids))
                cursor = await db.execute(
                    f"SELECT run_id, dispatched_gpu_ids FROM jobs WHERE run_id IN ({placeholders})",
                    resuming_ids,
                )
                rows = await cursor.fetchall()
                async with state.scheduler_lock:
                    for row in rows:
                        if row["dispatched_gpu_ids"] is not None:
                            gpu_ids = json.loads(row["dispatched_gpu_ids"])
                            for gpu_id in gpu_ids:
                                wc.gpu_occupancy[gpu_id] += 1

            # Send MsgReplay for resuming runs
            for rinfo in msg.resuming:
                await _send_to_worker(
                    wc,
                    encode(MsgReplay(
                        run_id=rinfo["run_id"],
                        log_seq=rinfo.get("log_seq", 0),
                        metric_seq=rinfo.get("metric_seq", 0),
                    )),
                )

            # Handle orphaned runs (were in-flight but worker isn't resuming them)
            orphaned: list[str] = []
            async with state.scheduler_lock:
                for run_id in list(wc.in_flight.keys()):
                    if run_id not in {r["run_id"] for r in msg.resuming}:
                        orphaned.append(run_id)

            for run_id in orphaned:
                in_flight = wc.in_flight.get(run_id)
                await _handle_orphaned_run(db, state, wc, run_id,
                                           experiment_id=in_flight.experiment_id if in_flight else "")

            # Restart read/write/heartbeat tasks for the reconnected worker
            _start_worker_tasks(db, state, wc, shutdown_event)

            return

        except (OSError, ValueError, json.JSONDecodeError, asyncio.TimeoutError):
            continue

    # All attempts exhausted
    async with state.scheduler_lock:
        wc.status = "dead"
        wc.gpu_occupancy.clear()

    await update_worker_status(db, wc.worker_id, "dead")

    print(f"  {_RED}FAIL{_RESET}  Worker {wc.host} unreachable; re-queuing runs")

    # Re-queue all in-flight runs
    async with state.scheduler_lock:
        orphaned = list(wc.in_flight.keys())

    for run_id in orphaned:
        in_flight = wc.in_flight.get(run_id)
        await _handle_orphaned_run(db, state, wc, run_id,
                                   experiment_id=in_flight.experiment_id if in_flight else "")


async def _handle_orphaned_run(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    run_id: str,
    experiment_id: str = "",
) -> None:
    """Re-queue or fail an orphaned run after a worker disconnect."""
    job = await increment_retry(db, run_id, experiment_id)

    async with state.scheduler_lock:
        # Free GPU occupancy for the orphaned run
        in_flight_job = wc.in_flight.get(run_id)
        if in_flight_job is not None:
            for gpu_id in in_flight_job.gpu_ids:
                wc.gpu_occupancy[gpu_id] = max(0, wc.gpu_occupancy[gpu_id] - 1)

        wc.in_flight.pop(run_id, None)
        in_flight_job = state.remove_in_flight(run_id)

    if job is not None:
        # Successfully retried — re-insert into pending
        async with state.scheduler_lock:
            state.insert_pending(job)
        print(f"  {_YELLOW}RETRY{_RESET} {run_id} (attempt {job.retry_count}/{job.max_retries})")
    else:
        # Max retries exceeded — mark as failed
        await finish_job(
            db, run_id, experiment_id,
            success=False, exit_code=-1, elapsed=0.0,
        )
        print(f"  {_RED}FAIL{_RESET}  {run_id}: max retries exceeded")

        if in_flight_job:
            state.broadcast(
                in_flight_job.experiment_id,
                {
                    "type": "job_done",
                    "run_id": run_id,
                    "worker_id": wc.worker_id,
                    "success": False,
                    "elapsed": 0.0,
                    "exit_code": -1,
                    "orphaned": True,
                },
            )

    if state.dispatch_callback:
        await state.dispatch_callback()


# ===============================================================================
# Start all tasks for a worker connection
# ===============================================================================


def _start_worker_tasks(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    shutdown_event: asyncio.Event | None = None,
    *,
    workers_ready: asyncio.Event | None = None,
) -> None:
    """Spawn read, write, and heartbeat tasks for *wc*."""
    asyncio.create_task(
        _worker_read_task(db, state, wc, shutdown_event=shutdown_event, workers_ready=workers_ready)
    )
    asyncio.create_task(
        _worker_write_task(wc, shutdown_event or asyncio.Event())
    )
    asyncio.create_task(
        _worker_heartbeat_task(wc, shutdown_event or asyncio.Event())
    )


# ===============================================================================
# Top-level: connect to all workers
# ===============================================================================


async def connect_workers(
    db: aiosqlite.Connection,
    state: ManagerState,
    *,
    workers_file: str | None = None,
    scratch_dir: str = "/tmp/mlsweep",
    max_gpus: int | None = None,
    shutdown_event: asyncio.Event | None = None,
    workers_ready: asyncio.Event | None = None,
) -> list[WorkerConn]:
    """Connect to all workers and return a list of ``WorkerConn`` objects.

    If *workers_file* is ``None``, launches a single local worker using
    ``max_gpus`` (or all visible GPUs).
    """
    if workers_file:
        configs = _parse_workers_file(workers_file)
    else:
        # Local mode — single worker
        local_gpus = max_gpus
        if local_gpus is None:
            visible = visible_devices()
            local_gpus = len(visible) if visible else 1
        configs = [
            {
                "host": "localhost",
                "remote_dir": _git_root(os.getcwd()) or os.getcwd(),
                "gpus": local_gpus,
                "devices": None,
                "password": None,
                "ssh_key": None,
                "venv": None,
                "port": 0,
            }
        ]

    workers: list[WorkerConn] = []

    for idx, cfg in enumerate(configs):
        host = cfg["host"]
        remote_dir = cfg["remote_dir"]
        w_devices = cfg.get("devices")
        w_pass = cfg.get("password")
        w_key = cfg.get("ssh_key")
        w_venv = cfg.get("venv")
        w_port = cfg.get("port", 0)

        worker_id = f"{host}:{w_port or 'ephemeral'}:{idx}"

        wc = await connect_single_worker(
            db, state,
            host=host,
            remote_dir=remote_dir,
            worker_id=worker_id,
            scratch_dir=scratch_dir,
            password=w_pass,
            ssh_key=w_key,
            venv=w_venv,
            port=w_port,
            devices=w_devices,
            shutdown_event=shutdown_event,
            workers_ready=workers_ready,
        )
        if wc is not None:
            workers.append(wc)

    if not workers and workers_ready is not None:
        workers_ready.set()

    return workers


# ===============================================================================
# Single-worker connection (shared by connect_workers and _connect_worker)
# ===============================================================================


async def connect_single_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    host: str,
    remote_dir: str,
    *,
    worker_id: str | None = None,
    scratch_dir: str = "/tmp/mlsweep",
    password: str | None = None,
    ssh_key: str | None = None,
    venv: str | None = None,
    port: int = 0,
    devices: list[int] | None = None,
    shutdown_event: asyncio.Event | None = None,
    workers_ready: asyncio.Event | None = None,
) -> WorkerConn | None:
    """Launch and connect to a single worker, register in state, and start tasks.

    Returns the ``WorkerConn`` on success, or ``None`` if the launch failed.
    Callers should handle the ``None`` case (log, mark dead, etc.).
    """
    try:
        reader, writer, actual_port = await launch_worker(
            host=host,
            remote_dir=remote_dir,
            token=state.token,
            scratch_dir=scratch_dir,
            devices=devices,
            password=password,
            ssh_key=ssh_key,
            venv=venv,
            port=port,
        )
    except Exception as e:
        print(f"  {_RED}WARN{_RESET}  Cannot start worker on {host}: {e}")
        return None

    # Determine worker_id if not provided
    if worker_id is None:
        worker_id = f"{host}:{actual_port}"

    # Create WorkerConn and register
    wc = WorkerConn(
        worker_id=worker_id,
        host=host,
        port=actual_port,
        reader=reader,
        writer=writer,
        status="connecting",
        remote_dir=remote_dir,
        scratch_dir=scratch_dir,
        password=password,
        ssh_key=ssh_key,
    )

    async with state.scheduler_lock:
        state.workers[worker_id] = wc

    writer.write(encode(MsgHello(token=state.token, controller_id="manager")))
    await writer.drain()

    _start_worker_tasks(db, state, wc, shutdown_event, workers_ready=workers_ready)

    print(f"  {_CYAN}START{_RESET} Worker {host}:{actual_port}")
    return wc


# ===============================================================================
# Utility: send MsgRun to a worker
def _parse_job_fields(dispatched: JobRecord) -> tuple[list[str], dict[str, str], dict[str, str], list[str]]:
    """Decode JSON-encoded command/env/files/return_files from a dispatched JobRecord."""
    command = json.loads(dispatched.command)
    if isinstance(command, str):
        command = [command]
    env = json.loads(dispatched.env) if isinstance(dispatched.env, str) else (dispatched.env or {})
    files: dict[str, str] = json.loads(dispatched.files) if isinstance(dispatched.files, str) else (dispatched.files or {})
    return_files: list[str] = json.loads(dispatched.return_files) if isinstance(dispatched.return_files, str) else (dispatched.return_files or [])
    return command, env, files, return_files


# ===============================================================================


async def dispatch_to_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    job: JobRecord,
    *,
    gpu_ids: list[int],
) -> bool:
    """Dispatch *job* to *wc* on the given *gpu_ids*.

    Updates the DB (dispatch_job), creates an InFlightJob, and sends ``MsgRun``.
    Returns ``True`` on success, ``False`` if the worker is not connected.
    """
    if wc.status != "connected":
        return False

    # Hold the scheduler lock through DB write so that another concurrent
    # scheduling pass cannot see the GPU occupancy as still free and
    # double-assign them.
    async with state.scheduler_lock:
        # Atomically claim the job in DB
        dispatched = await dispatch_job(
            db, job.run_id, job.experiment_id, wc.worker_id, gpu_ids
        )
        if dispatched is None:
            return False  # job already taken by another scheduler

        command, job_env, job_files, job_return_files = _parse_job_fields(dispatched)

        if state.artifact_base_url and job.artifact_id:
            artifact_url = f"{state.artifact_base_url}/api/artifacts/{job.artifact_id}"
            if state.token:
                artifact_url += f"?token={state.token}"
        else:
            artifact_url = ""

        run_scratch = os.path.join(wc.scratch_dir, job.experiment_id, job.run_id)

        run_msg = MsgRun(
            run_id=job.run_id,
            experiment=job.experiment_id,
            command=command,
            env=job_env,
            gpu_ids=gpu_ids,
            remote_dir="",
            scratch=run_scratch,
            run_from=job.run_from,
            set_dist_env=job.set_dist_env,
            files=job_files,
            return_files=job_return_files,
            artifact_id=job.artifact_id or "",
            artifact_url=artifact_url,
            setup_command=shlex.split(job.setup_command) if job.setup_command else [],
        )

    # Delegate in-memory update + send + broadcast to the shared core
    return await _dispatch_core(
        state, wc, job,
        gpu_ids=gpu_ids,
        run_msg=run_msg,
    )


# ===============================================================================
# Internal dispatch core — shared between single-node and multi-node paths
# ===============================================================================


async def _dispatch_core(
    state: ManagerState,
    wc: WorkerConn,
    job: JobRecord,
    *,
    gpu_ids: list[int],
    run_msg: MsgRun,
    worker_ids: list[str] | None = None,
    combo: dict[str, Any] | None = None,
) -> bool:
    """Update in-memory state for a dispatch and send ``MsgRun`` to the worker.

    Does **not** write to the database — the caller must have already
    called ``dispatch_job`` (or the multi-node equivalent).

    Returns ``True`` on success, ``False`` if the worker is not connected
    or the send queue is full.
    """
    if wc.status != "connected":
        return False

    if combo is None:
        combo = json.loads(job.combo) if isinstance(job.combo, str) else job.combo

    async with state.scheduler_lock:
        in_flight = InFlightJob(
            run_id=job.run_id,
            worker_id=wc.worker_id,
            experiment_id=job.experiment_id,
            dispatch_time=datetime.now(timezone.utc),
            gpu_ids=gpu_ids,
            worker_ids=worker_ids or [wc.worker_id],
            combo=combo or {},
        )
        state.add_in_flight(in_flight)
        wc.in_flight[job.run_id] = in_flight

        for gpu_id in gpu_ids:
            wc.gpu_occupancy[gpu_id] += 1

    # Send MsgRun (outside the lock — I/O)
    success = await _send_to_worker(wc, encode(run_msg))
    if not success:
        return False

    # Broadcast
    state.broadcast(
        job.experiment_id,
        {
            "type": "job_dispatched",
            "run_id": job.run_id,
            "worker_id": wc.worker_id,
            "host": wc.host,
            "gpu_ids": gpu_ids,
        },
    )
    return True


# ===============================================================================
# GPU group finder (scheduler helper)
# ===============================================================================


def _find_gpu_group(
    wc: WorkerConn,
    gpus_needed: int,
    topo: dict[tuple[int, int], int] | None = None,
    *,
    occupancy: dict[int, int] | None = None,
    jobs_per_gpu: int = 1,
) -> list[int] | None:
    """Find an available GPU group of size *gpus_needed* on this worker.

    Uses *occupancy* if given (for tentative planning), otherwise reads
    ``wc.gpu_occupancy`` directly.

    *jobs_per_gpu* is the per-job concurrency limit (from the job record).

    Returns a list of GPU device indices, or ``None`` if unavailable.
    CPU-only jobs (``gpus_needed == 0``) return ``[]``.
    """
    if gpus_needed == 0:
        return []

    occ = occupancy if occupancy is not None else wc.gpu_occupancy
    capacity = jobs_per_gpu

    # Filter GPUs that have room for another job
    available = [
        g for g in wc.gpus
        if occ[g] < capacity
    ]
    if len(available) < gpus_needed:
        return None

    # Use topology-aware grouping if we have topology data
    if topo and len(available) >= gpus_needed:
        groups = _best_gpu_groups(available, gpus_needed, 1, topo=topo)
        if groups:
            return groups[0]

    # Fallback: first N available GPUs
    return available[:gpus_needed]


# ===============================================================================
# Main scheduling entry point
# ===============================================================================


async def schedule_pending(
    db: aiosqlite.Connection,
    state: ManagerState,
) -> int:
    """Try to dispatch as many pending jobs as possible.

    Acquires ``state.scheduler_lock`` to scan the pending list and find
    available GPUs, then releases it before dispatching.  The dispatch
    helpers (``dispatch_to_worker`` and ``_execute_assignment``) acquire
    the lock internally for their DB write + in-memory state update.

    Returns the number of jobs successfully dispatched.
    """
    # ── Phase 1: find assignments under lock ───────────────────────────
    async with state.scheduler_lock:
        connected = [
            wc
            for wc in state.workers.values()
            if wc.status == "connected"
        ]
        if not connected or not state.pending:
            return 0

        # Build tentative occupancy maps per worker for planning.
        # We copy the current occupancy so that assignments within this
        # planning pass do not double-book the same GPUs.
        tentative_occ: dict[str, dict[int, int]] = {}
        for wc in connected:
            tentative_occ[wc.worker_id] = dict(wc.gpu_occupancy)

        # Plan: list of (job, [(worker, gpu_ids), ...])
        plan: list[tuple[JobRecord, list[tuple[WorkerConn, list[int]]]]] = []
        remaining: list[JobRecord] = []

        for job in state.pending:
            if job.nodes_per_run <= 1:
                # ── Single-node ────────────────────────────────────────
                assigned = False
                for wc in connected:
                    gpus = _find_gpu_group(
                        wc, job.gpus_per_run,
                        topo=_parse_topo_wire(wc.topo),
                        occupancy=tentative_occ[wc.worker_id],
                        jobs_per_gpu=job.jobs_per_gpu,
                    )
                    if gpus is not None:
                        plan.append((job, [(wc, gpus)]))
                        # Tentatively reserve these GPUs
                        occ = tentative_occ[wc.worker_id]
                        for g in gpus:
                            occ[g] += 1
                        assigned = True
                        break
                if not assigned:
                    remaining.append(job)
            else:
                # ── Multi-node ─────────────────────────────────────────
                assignments: list[tuple[WorkerConn, list[int]]] = []
                used_workers: set[str] = set()
                for wc in connected:
                    if wc.worker_id in used_workers:
                        continue
                    gpus = _find_gpu_group(
                        wc, job.gpus_per_run,
                        topo=_parse_topo_wire(wc.topo),
                        occupancy=tentative_occ[wc.worker_id],
                        jobs_per_gpu=job.jobs_per_gpu,
                    )
                    if gpus is not None:
                        assignments.append((wc, gpus))
                        used_workers.add(wc.worker_id)
                        occ = tentative_occ[wc.worker_id]
                        for g in gpus:
                            occ[g] += 1
                        if len(assignments) >= job.nodes_per_run:
                            break
                if len(assignments) >= job.nodes_per_run:
                    plan.append((job, assignments))
                else:
                    remaining.append(job)

        # Replace pending list with only the jobs we couldn't assign
        state.pending = remaining

    # ── Phase 2: dispatch outside the lock ─────────────────────────────
    dispatched_count = 0

    for job, assignments in plan:
        ok = await _execute_assignment(db, state, job, assignments)
        if ok:
            dispatched_count += 1
        else:
            # Dispatch failed — re-insert into pending
            async with state.scheduler_lock:
                state.insert_pending(job)

    return dispatched_count


# ===============================================================================
# Assignment execution
# ===============================================================================


async def _execute_assignment(
    db: aiosqlite.Connection,
    state: ManagerState,
    job: JobRecord,
    assignments: list[tuple[WorkerConn, list[int]]],
) -> bool:
    """Execute a dispatch plan for one job across one or more workers.

    For single-node (``len(assignments) == 1``), delegates to
    ``dispatch_to_worker``.

    For multi-node, dispatches to the first worker via the DB, then
    sends ``MsgRun`` directly to the remaining workers without additional
    DB updates (the run_id is shared across nodes).
    """
    if len(assignments) == 1:
        wc, gpu_ids = assignments[0]
        return await dispatch_to_worker(db, state, wc, job, gpu_ids=gpu_ids)

    # ── Multi-node path ────────────────────────────────────────────────
    if not assignments:
        return False

    primary_wc, primary_gpus = assignments[0]

    # Atomically claim the job in DB via the primary worker
    async with state.scheduler_lock:
        dispatched = await dispatch_job(
            db, job.run_id, job.experiment_id, primary_wc.worker_id, primary_gpus,
        )
        if dispatched is None:
            return False  # job already taken by another scheduler

        command, job_env, job_files, job_return_files = _parse_job_fields(dispatched)

    all_worker_ids = [wc.worker_id for wc, _ in assignments]

    # Determine master address / port for distributed setup
    master_host = primary_wc.host.split("@")[-1]
    master_port = 29500  # default torch distributed port; could be parameterised

    if state.artifact_base_url and job.artifact_id:
        artifact_url = f"{state.artifact_base_url}/api/artifacts/{job.artifact_id}"
        if state.token:
            artifact_url += f"?token={state.token}"
    else:
        artifact_url = ""

    all_ok = True

    for node_rank, (wc, gpu_ids) in enumerate(assignments):
        run_scratch = os.path.join(wc.scratch_dir, job.experiment_id, job.run_id)

        node_env: dict[str, str] = {
            **job_env,
            "MLSWEEP_NNODES": str(len(assignments)),
            "MLSWEEP_NODE_RANK": str(node_rank),
            "MLSWEEP_MASTER_ADDR": master_host,
            "MLSWEEP_MASTER_PORT": str(master_port),
        }

        run_msg = MsgRun(
            run_id=job.run_id,
            experiment=job.experiment_id,
            command=command,
            env=node_env,
            gpu_ids=gpu_ids,
            remote_dir="",
            scratch=run_scratch,
            run_from=job.run_from,
            set_dist_env=job.set_dist_env,
            files=job_files,
            return_files=job_return_files,
            artifact_id=job.artifact_id or "",
            artifact_url=artifact_url,
            setup_command=shlex.split(job.setup_command) if job.setup_command else [],
        )

        ok = await _dispatch_core(
            state, wc, job,
            gpu_ids=gpu_ids,
            run_msg=run_msg,
            worker_ids=all_worker_ids,
        )
        if not ok:
            all_ok = False
            break

    if not all_ok:
        # Partial failure — the primary DB dispatch already happened.
        # The job will be treated as orphaned and retried via the normal
        # reconnect / lost-worker path.
        return False

    return True


