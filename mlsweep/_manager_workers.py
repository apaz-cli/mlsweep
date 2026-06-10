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
    experiment_concurrency_caps,
    get_experiment,
    is_multinode_run,
    list_jobs_by_run_ids,
    list_schedulable_jobs,
    multinode_progress,
    reset_jobs_to_pending_batch,
    update_job_status,
)
from mlsweep._manager_state import InFlightJob, ManagerState, WorkerConn
from mlsweep._parsync import parsync_bin
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
    _GREEN,
    _RED,
    _YELLOW,
    _CYAN,
    _MAGENTA,
    _BLUE,
    _RESET,
    _git_root,
    aread_msg,
    decode,
    encode,
)
from mlsweep._topology import _best_gpu_groups, _gpu_topology, _parse_topo_wire, visible_devices


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
        import tomli as tomllib  # Python < 3.11

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


def _ensure_worker_wheels() -> None:
    """Build _wheels/ if it doesn't already contain a mlsweep wheel.

    Runs synchronously at manager startup (before the event loop).
    Two steps:
      1. pip wheel --no-deps  → builds the local mlsweep wheel
      2. pip download         → fetches abi3 deps from PyPI, seeded by the
                                wheel from step 1 so mlsweep itself is never
                                pulled from the public index
    """
    wheels_dir = Path(__file__).resolve().parent / "_wheels"
    if wheels_dir.exists() and (wheels_dir / ".complete").exists():
        return

    print("[wheels] Building worker wheels...", flush=True)
    wheels_dir.mkdir(exist_ok=True)

    # Step 1: build the local mlsweep wheel.
    repo_root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--wheel-dir", str(wheels_dir), str(repo_root)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(
            f"[wheels] pip wheel failed:\n{r.stderr.decode(errors='replace')}",
            file=sys.stderr,
        )
        return

    # Step 2: download abi3 deps from PyPI for the mlsweep wheel we just built.
    # --find-links points at our wheels dir so pip reads mlsweep's metadata
    # from the local wheel and never fetches mlsweep itself from PyPI.
    r = subprocess.run(
        [sys.executable, "-m", "pip", "download",
         "--dest", str(wheels_dir),
         "--find-links", str(wheels_dir),
         "--platform", "manylinux_2_17_x86_64",
         "--implementation", "cp",
         "--abi", "abi3",
         "--python-version", "3.10",
         "--only-binary", ":all:",
         "mlsweep"],
        capture_output=True,
    )
    if r.returncode != 0:
        print(
            f"[wheels] pip download failed:\n{r.stderr.decode(errors='replace')}",
            file=sys.stderr,
        )
        # Step 1 left an orphaned mlsweep wheel — remove it so the
        # guard on next startup doesn't skip the full rebuild.
        for w in wheels_dir.glob("mlsweep-*.whl"):
            w.unlink(missing_ok=True)
        return

    # Both steps succeeded — write a sentinel so we never reuse a
    # partial result.
    (wheels_dir / ".complete").touch()

    wheels = [p.name for p in wheels_dir.iterdir()]
    print(f"[wheels] Ready ({len(wheels)} wheels)", flush=True)


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
    ssh_opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    sshpass_args, sshpass_env = _sshpass_args(password)

    # 1. Quick check: is the binary already present?
    try:
        proc = await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh", *ssh_opts,
            *key_args,
            host,
            "test -x /tmp/mlsweep_venv/bin/mlsweep_worker && echo OK || echo MISSING",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        if b"OK" in stdout:
            return True
    except (OSError, asyncio.TimeoutError):
        pass  # fall through to bootstrap

    # 2. mkdir + SCP bundled wheels to remote.
    wheels_dir = Path(__file__).resolve().parent / "_wheels"
    wheel_files = [str(p) for p in wheels_dir.iterdir()]
    try:
        proc = await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh", *ssh_opts,
            *key_args,
            host,
            "mkdir -p /tmp/mlsweep_wheels",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode != 0:
            print(
                f"[bootstrap] mkdir failed for {host}: "
                f"{stderr.decode(errors='replace')[:200]}",
                file=sys.stderr,
            )
            return False
    except (OSError, asyncio.TimeoutError) as e:
        print(f"[bootstrap] mkdir failed for {host}: {e}", file=sys.stderr)
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            *sshpass_args,
            "scp", *ssh_opts,
            *key_args,
            *wheel_files,
            f"{host}:/tmp/mlsweep_wheels/",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
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

    # 3. Create venv and install mlsweep fully offline.
    try:
        proc = await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh", *ssh_opts,
            *key_args,
            host,
            (
                "python3 -m venv /tmp/mlsweep_venv && "
                "/tmp/mlsweep_venv/bin/pip install --no-index "
                "--find-links=/tmp/mlsweep_wheels/ mlsweep"
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
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


def _sshpass_args(password: str | None) -> tuple[list[str], dict[str, str] | None]:
    """Return ``(sshpass_prefix, env_dict)`` for subprocess calls.

    Uses ``sshpass -e`` to pass *password* via the ``SSHPASS`` env var
    rather than exposing it on the command line via ``-p``.
    """
    global _sshpass_available
    if not password:
        return [], None
    if _sshpass_available is None:
        _sshpass_available = shutil.which("sshpass") is not None
    if not _sshpass_available:
        raise RuntimeError("sshpass is not installed but a password was specified")
    return ["sshpass", "-e"], {**os.environ, "SSHPASS": password}


# ===============================================================================
# Worker launch (async)
# ===============================================================================


async def launch_worker(
    host: str,
    remote_dir: str,
    token: str,
    scratch_dir: str = "/tmp/mlsweep",
    devices: list[int] | None = None,
    max_jobs_per_gpu: int = 1,
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
    jobs_args = ["--jobs", str(max_jobs_per_gpu)]
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
            *jobs_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        # Remote: bootstrap venv if needed, then SSH and run worker binary
        sshpass_args, sshpass_env = _sshpass_args(password)
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
            *jobs_args,
        ]
        shell_cmd = _worker_shell_cmd(_worker_candidates(venv), worker_args)
        ssh_cmd = [
            *sshpass_args,
            "ssh", "-o", "ConnectTimeout=10",
            *key_args,
            host, shell_cmd,
        ]
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
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
    """Handle ``MsgWorkerHello``: register the worker's GPUs, mark connected.

    If the worker reports ``resuming`` runs (jobs it kept executing across a
    manager restart), those jobs are restored to in-flight state so the
    scheduler does not re-dispatch them.
    """
    resuming_map: dict[str, dict[str, Any]] = {r["run_id"]: r for r in msg.resuming}
    resumed_jobs: list[tuple[JobRecord, list[int]]] = []

    # Look up the DB rows for any resuming runs before taking the lock.  On
    # manager restart, reset_dispatched_running_to_pending moved these jobs back
    # to 'pending'; we pull them by run_id (the worker's resume payload has no
    # experiment_id) and restore in-flight state so the scheduler skips them.
    resume_records: dict[str, JobRecord] = {}
    if resuming_map:
        for jr in await list_jobs_by_run_ids(db, list(resuming_map)):
            resume_records.setdefault(jr.run_id, jr)

    async with state.scheduler_lock:
        wc.gpus = msg.gpus
        wc.topo = msg.topo
        wc.max_jobs_per_gpu = msg.max_jobs_per_gpu
        wc.scratch_dir = msg.scratch_dir

        # Mark connected
        wc.status = "connected"
        wc.connected_at = datetime.now(timezone.utc)

        # Restore in-flight state for jobs the worker is already running.  The
        # exact GPUs are unknown post-restart; pick the first N free (occupancy
        # is derived from wc.in_flight, so adding the entry reserves them).
        for run_id, job in resume_records.items():
            if run_id in state.in_flight:
                continue
            occ = _worker_occupancy(wc)
            cap = wc.max_jobs_per_gpu  # 0 = unlimited
            gpu_ids: list[int] = []
            for g in wc.gpus:
                if len(gpu_ids) >= job.gpus_per_run:
                    break
                if cap <= 0 or occ.get(g, 0) < cap:
                    gpu_ids.append(g)
                    occ[g] = occ.get(g, 0) + 1

            combo = json.loads(job.combo) if isinstance(job.combo, str) else (job.combo or {})
            in_flight_job = InFlightJob(
                run_id=job.run_id,
                worker_id=wc.worker_id,
                experiment_id=job.experiment_id,
                dispatch_time=datetime.now(timezone.utc),
                gpu_ids=gpu_ids,
                worker_ids=[wc.worker_id],
                combo=combo,
            )
            state.add_in_flight(in_flight_job)
            wc.in_flight[job.run_id] = in_flight_job
            resumed_jobs.append((job, gpu_ids))

        # Persist to DB
        await state.db_writer.upsert_worker(
            worker_id=wc.worker_id,
            host=wc.host,
            remote_dir=wc.remote_dir,
            scratch_dir=msg.scratch_dir,
            port=wc.port,
            ssh_key=wc.ssh_key,
            venv=wc.venv,
            devices=json.dumps(msg.gpus),
            status="connected",
        )

        n_gpus = len(msg.gpus)
        gpu_plural = "s" if n_gpus != 1 else ""
        print(
            f"  {_GREEN}OK{_RESET}    {wc.host}: {n_gpus} GPU{gpu_plural} available"
        )
        if resumed_jobs:
            print(f"  {_GREEN}RESUME{_RESET} {wc.host}: {len(resumed_jobs)} run(s) still active")

        # If all workers have completed hello, signal the manager
        if workers_ready is not None:
            _check_all_workers_ready(state, workers_ready)

    # Re-mark resumed jobs as running in DB and send MsgReplay so the worker
    # replays any logs/metrics the manager missed while it was down.
    # Done outside the scheduler lock to avoid holding it during I/O.
    for job, gpu_ids in resumed_jobs:
        rinfo = resuming_map[job.run_id]
        dispatched = await state.db_writer.dispatch_job(
            job.run_id, job.experiment_id, wc.worker_id, gpu_ids
        )
        if dispatched is not None:
            await state.db_writer.mark_job_running(job.run_id, job.experiment_id)
        await _send_to_worker(wc, encode(MsgReplay(
            run_id=job.run_id,
            log_seq=rinfo.get("log_seq", 0),
            metric_seq=rinfo.get("metric_seq", 0),
        )))

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
    """Handle ``MsgStarted``: mark job running."""
    async with state.scheduler_lock:
        job = state.get_in_flight(msg.run_id)
        if job is not None:
            job.start_time = datetime.now(timezone.utc)

    await state.db_writer.mark_job_running(msg.run_id, job.experiment_id if job else "")

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
    """Handle ``MsgLog``: persist to DB and broadcast."""
    job = state.get_in_flight(msg.run_id)

    if job:
        await state.db_writer.insert_log(msg.run_id, job.experiment_id, msg.seq, msg.data)
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
    """Handle ``MsgMetric``: persist to DB and broadcast to WebSocket subscribers."""
    job = state.get_in_flight(msg.run_id)

    if job:
        await state.db_writer.insert_metric(msg.run_id, job.experiment_id, msg.step, msg.data)
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
        in_flight = state.get_in_flight(msg.run_id)

        # Job was pre-evicted/cancelled and re-queued; this result is stale.
        if in_flight is None and msg.run_id not in wc.in_flight:
            return

        # Identify the experiment before we drop tracking.
        local = wc.in_flight.get(msg.run_id)
        experiment_id = (
            in_flight.experiment_id if in_flight is not None
            else (local.experiment_id if local is not None else "")
        )

        # Drop this worker's record of the run; occupancy is derived from
        # wc.in_flight, so this is what frees the GPUs (no counter to decrement).
        wc.in_flight.pop(msg.run_id, None)

    # ── Multi-node aggregation (durable, restart-safe) ─────────────────
    # Multi-node-ness is decided by the presence of job_nodes rows, not by
    # len(worker_ids): after a manager restart each worker resumes its own node
    # as a separate in-flight entry, but the node rows still tie them together.
    multinode = await is_multinode_run(db, msg.run_id, experiment_id)

    if multinode:
        # Record this node's result in the DB and free this worker's local run.
        await state.db_writer.mark_job_node_result(
            msg.run_id, experiment_id, wc.worker_id, msg.success, msg.elapsed,
        )
        await _send_to_worker(wc, encode(MsgCleanup(run_id=msg.run_id)))

        remaining, all_success, max_elapsed = await multinode_progress(
            db, msg.run_id, experiment_id,
        )
        if remaining > 0:
            # Other nodes are still running; this worker is now free for work.
            if state.dispatch_callback is not None:
                await state.dispatch_callback()
            return

        # Last node in — finalise exactly once.
        async with state.scheduler_lock:
            job = state.remove_in_flight(msg.run_id)
        await state.db_writer.delete_job_nodes(msg.run_id, experiment_id)
        final_success = all_success
        final_elapsed = max_elapsed
    else:
        async with state.scheduler_lock:
            job = state.remove_in_flight(msg.run_id)
        final_success = msg.success
        final_elapsed = msg.elapsed

    # Persist to DB
    await state.db_writer.finish_job(
        msg.run_id,
        experiment_id,
        success=final_success,
        exit_code=msg.exit_code,
        elapsed=final_elapsed,
    )

    # Retroactively reclassify singular-probe failures as xfailed (needs combo).
    xfailed_ids: list[str] = []
    if final_success and job is not None:
        xfailed_ids = await state.db_writer.reclassify_singular_xfails(experiment_id, job.combo)

    # Broadcast
    if experiment_id:
        state.broadcast(
            experiment_id,
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
    if experiment_id:
        run_scratch = os.path.join(wc.scratch_dir, experiment_id, msg.run_id)
        run_dir = os.path.join(state.output_dir, experiment_id, msg.run_id)
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

    # Send cleanup after sync. For multi-node, each node already received
    # MsgCleanup as its result arrived, so don't double-send to this worker.
    if not multinode:
        await _send_to_worker(wc, encode(MsgCleanup(run_id=msg.run_id)))

    # Trigger dispatch
    if state.dispatch_callback is not None:
        await state.dispatch_callback()

    # ── Check if experiment is complete ─────────────────────────────────
    # Authoritative DB check: an experiment is complete once it has no jobs in a
    # non-terminal state (pending / dispatched / running).  Reading from the DB
    # (rather than an in-memory mirror) means a deleted/cancelled job can never
    # leave a phantom that blocks completion.  update_experiment_status is
    # idempotent, so any TOCTOU is harmless.
    if experiment_id:
        exp = await get_experiment(db, experiment_id)
        if exp is None:
            return

        cursor = await db.execute(
            "SELECT COUNT(*) FROM jobs WHERE experiment_id = ? "
            "AND status IN ('pending', 'dispatched', 'running')",
            (experiment_id,),
        )
        row = await cursor.fetchone()
        active_count: int = row[0] if row else 0
        if active_count > 0:
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

        await state.db_writer.update_experiment_status(experiment_id, "completed")

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
    await state.db_writer.touch_worker(wc.worker_id)


async def _handle_gpu_stats(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    msg: MsgGpuStats,
) -> None:
    """Handle ``MsgGpuStats`` — store latest GPU utilization data for the UI."""
    wc.gpu_stats = {s["gpu"]: s for s in msg.stats if "gpu" in s}


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
    MsgGpuStats: _handle_gpu_stats,
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
        payload = await asyncio.wait_for(aread_msg(wc.reader), timeout=30.0)
    except (asyncio.TimeoutError, OSError, asyncio.IncompleteReadError):
        await _on_worker_lost(db, state, wc, shutdown_event)
        return

    try:
        msg = decode(payload)
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
            payload = await asyncio.wait_for(aread_msg(wc.reader), timeout=60.0)
        except asyncio.TimeoutError:
            continue
        except (OSError, asyncio.IncompleteReadError):
            break

        try:
            msg = decode(payload)
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

    # Signal the write task to stop draining the now-dead connection.
    # The write task returns when it dequeues this sentinel.
    wc.send_queue.put_nowait(None)

    await state.db_writer.update_worker_status(wc.worker_id, "reconnecting")

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
        # If the worker was explicitly deleted while we were backing off, stop
        # trying — otherwise a successful reconnect would resurrect it.
        if wc.status == "dead":
            return

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)

        try:
            connect_host = wc.host.split("@")[-1]
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(connect_host, wc.port),
                timeout=5.0,
            )

            writer.write(encode(MsgHello(token=state.token, controller_id="manager")))
            await writer.drain()

            # Read MsgWorkerHello
            try:
                payload = await asyncio.wait_for(aread_msg(reader), timeout=10.0)
            except (asyncio.TimeoutError, OSError, asyncio.IncompleteReadError):
                writer.close()
                continue

            msg = decode(payload)
            if not isinstance(msg, MsgWorkerHello):
                writer.close()
                continue

            # Success — attach new streams
            async with state.scheduler_lock:
                wc.reader = reader
                wc.writer = writer
                wc.status = "connected"
                wc.connected_at = datetime.now(timezone.utc)

            await state.db_writer.update_worker_status(wc.worker_id, "connected")

            print(f"  {_GREEN}OK{_RESET}    Worker {wc.host} reconnected")

            # Occupancy needs no reconstruction: wc.in_flight is preserved across
            # the reconnect and occupancy is derived from it.

            # Replace the send queue so messages queued for the dead connection
            # (stale MsgRun, old heartbeats, etc.) cannot bleed onto the new one.
            # The old write task already exited on the None sentinel from _on_worker_lost.
            wc.send_queue = asyncio.Queue()

            # Send MsgReplay for resuming runs (into the fresh queue)
            for rinfo in msg.resuming:
                await _send_to_worker(
                    wc,
                    encode(MsgReplay(
                        run_id=rinfo["run_id"],
                        log_seq=rinfo.get("log_seq", 0),
                        metric_seq=rinfo.get("metric_seq", 0),
                    )),
                )

            # Handle orphaned runs (were in-flight but worker isn't resuming them).
            # Collect both keys and values under the lock to avoid reading stale data
            # between iterations.
            orphaned: list[tuple[str, str]] = []
            async with state.scheduler_lock:
                for run_id, ifj in list(wc.in_flight.items()):
                    if run_id not in {r["run_id"] for r in msg.resuming}:
                        orphaned.append((run_id, ifj.experiment_id))

            for run_id, exp_id in orphaned:
                await _handle_orphaned_run(db, state, wc, run_id, experiment_id=exp_id)

            # Restart read/write/heartbeat tasks for the reconnected worker
            _start_worker_tasks(db, state, wc, shutdown_event)

            return

        except (OSError, ValueError, json.JSONDecodeError, asyncio.TimeoutError):
            continue

    # All attempts exhausted
    async with state.scheduler_lock:
        wc.status = "dead"

    await state.db_writer.update_worker_status(wc.worker_id, "dead")

    print(f"  {_RED}FAIL{_RESET}  Worker {wc.host} unreachable; re-queuing runs")

    # Re-queue all in-flight runs — collect run_id + experiment_id together
    # under the lock so we don't read stale data between iterations.
    dead_orphaned: list[tuple[str, str]] = []
    async with state.scheduler_lock:
        for run_id, ifj in list(wc.in_flight.items()):
            dead_orphaned.append((run_id, ifj.experiment_id))

    for run_id, exp_id in dead_orphaned:
        await _handle_orphaned_run(db, state, wc, run_id, experiment_id=exp_id)


async def _handle_orphaned_run(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    run_id: str,
    experiment_id: str = "",
) -> None:
    """Re-queue or fail an orphaned run after a worker disconnect."""
    job = await state.db_writer.increment_retry(run_id, experiment_id)

    async with state.scheduler_lock:
        # Drop in-flight tracking (frees derived occupancy for this run).
        wc.in_flight.pop(run_id, None)
        in_flight_job = state.remove_in_flight(run_id)

    if job is not None:
        # Successfully retried — increment_retry already reset the DB row to
        # 'pending', so the next scheduling pass will pick it up.
        print(f"  {_YELLOW}RETRY{_RESET} {run_id} (attempt {job.retry_count}/{job.max_retries})")
    else:
        # Max retries exceeded — mark as failed
        await state.db_writer.finish_job(
            run_id, experiment_id,
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
# SSH reverse tunnel (artifact delivery for remote workers)
# ===============================================================================


async def _launch_tunnel(
    host: str,
    manager_port: int,
    ssh_key: str | None = None,
    password: str | None = None,
) -> "asyncio.subprocess.Process | None":
    """Spawn ssh -N -R {port}:localhost:{port} so the worker can reach the manager's HTTP server."""
    key_args = ["-i", ssh_key] if ssh_key else []
    sshpass_args, sshpass_env = _sshpass_args(password)
    try:
        return await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh", "-N",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"{manager_port}:localhost:{manager_port}",
            *key_args,
            host,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=sshpass_env,
        )
    except OSError as e:
        print(f"  {_YELLOW}WARN{_RESET}  Could not start tunnel to {host}: {e}", file=sys.stderr)
        return None


async def _tunnel_monitor_task(
    wc: WorkerConn,
    manager_port: int,
    shutdown_event: asyncio.Event,
) -> None:
    """Keep the SSH reverse tunnel alive, restarting with backoff if it dies."""
    backoff = 2.0
    while True:
        proc = wc.tunnel_proc
        if proc is None or shutdown_event.is_set():
            return

        wait_proc = asyncio.create_task(proc.wait())
        wait_shut = asyncio.create_task(shutdown_event.wait())
        done, pending = await asyncio.wait(
            {wait_proc, wait_shut}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()

        if shutdown_event.is_set() or wc.status == "dead":
            if proc.returncode is None:
                proc.terminate()
            return

        print(
            f"  {_YELLOW}WARN{_RESET}  SSH tunnel to {wc.host} lost; "
            f"reconnecting in {backoff:.0f}s"
        )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)

        new_proc = await _launch_tunnel(
            wc.host, manager_port, ssh_key=wc.ssh_key, password=wc.password
        )
        if new_proc is not None:
            wc.tunnel_proc = new_proc
            backoff = 2.0
            print(f"  {_GREEN}OK{_RESET}    SSH tunnel to {wc.host} re-established")


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
    manager_port: int = 0,
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
        w_max_jobs = cfg.get("jobs")
        if w_max_jobs is None:
            w_max_jobs = 1  # default 1 job/GPU; 0 means unlimited

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
            max_jobs_per_gpu=w_max_jobs,
            manager_port=manager_port,
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
    max_jobs_per_gpu: int = 1,
    manager_port: int = 0,
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
            max_jobs_per_gpu=max_jobs_per_gpu,
            password=password,
            ssh_key=ssh_key,
            venv=venv,
            port=port,
        )
    except Exception as e:
        print(f"  {_RED}WARN{_RESET}  Cannot start worker on {host}: {e}")
        return None

    # Determine worker_id if not provided.
    # NOTE: all current callers pass an explicit worker_id, so this branch
    # is never reached.  Kept as a defensive fallback; if called without an
    # id, the format matches none of the other id-generation sites.
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
        venv=venv,
        max_jobs_per_gpu=max_jobs_per_gpu,
    )

    if host != "localhost" and manager_port:
        tunnel_proc = await _launch_tunnel(
            host, manager_port, ssh_key=ssh_key, password=password
        )
        wc.tunnel_proc = tunnel_proc
        asyncio.create_task(
            _tunnel_monitor_task(wc, manager_port, shutdown_event or asyncio.Event())
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

    # Pre-compute stable values — no lock needed, these fields are immutable.
    if state.artifact_base_url and job.artifact_id:
        artifact_url = f"{state.artifact_base_url}/api/artifacts/{job.artifact_id}"
        if state.token:
            artifact_url += f"?token={state.token}"
    else:
        artifact_url = ""
    run_scratch = os.path.join(wc.scratch_dir, job.experiment_id, job.run_id)

    # Hold the scheduler lock only for the DB write (the CAS that prevents
    # double-dispatch) and the immediate field extraction — nothing else.
    async with state.scheduler_lock:
        dispatched = await state.db_writer.dispatch_job(
            job.run_id, job.experiment_id, wc.worker_id, gpu_ids
        )
        if dispatched is None:
            return False  # job already taken by another scheduler

        command, job_env, job_files, job_return_files = _parse_job_fields(dispatched)

    # Build MsgRun outside the lock.
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

    # Update in-memory state and send.
    ok = await _dispatch_core(
        state, wc, job,
        gpu_ids=gpu_ids,
        run_msg=run_msg,
    )
    if not ok:
        # We claimed the row (dispatch_job succeeded) but couldn't hand it to
        # the worker; return it to pending so the next pass retries it.  Without
        # this the row would be stuck in 'dispatched' with nothing tracking it.
        await state.db_writer.reset_job_to_pending(job.run_id, job.experiment_id)
    return ok


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
        # Occupancy is derived from wc.in_flight (see _worker_occupancy); adding
        # the entry above is what marks these GPUs busy. No counter to bump.

    # Send MsgRun (outside the lock — I/O)
    success = await _send_to_worker(wc, encode(run_msg))
    if not success:
        async with state.scheduler_lock:
            wc.in_flight.pop(job.run_id, None)
            state.remove_in_flight(job.run_id)
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


def _worker_occupancy(wc: WorkerConn) -> dict[int, int]:
    """Derive per-GPU job counts for *wc* from its in-flight jobs.

    Occupancy is never stored as a mutable counter; it is recomputed from
    ``wc.in_flight`` (each entry carries this worker's ``gpu_ids`` for one run).
    Removing a run from ``wc.in_flight`` therefore frees its GPUs automatically,
    with no decrement to forget — this is what eliminates the occupancy-leak
    bug class (a cancelled/deleted run can no longer permanently consume a slot).
    """
    occ = {g: 0 for g in wc.gpus}
    for ifj in wc.in_flight.values():
        for g in ifj.gpu_ids:
            if g in occ:
                occ[g] += 1
    return occ


def _find_gpu_group(
    wc: WorkerConn,
    gpus_needed: int,
    topo: dict[tuple[int, int], int] | None = None,
    *,
    occupancy: dict[int, int] | None = None,
) -> list[int] | None:
    """Find an available GPU group of size *gpus_needed* on this worker.

    Uses *occupancy* if given (for tentative planning), otherwise derives it
    from ``wc.in_flight`` via ``_worker_occupancy``.

    Per-GPU packing is bounded by the worker's ``max_jobs_per_gpu`` cap
    (0 = unlimited).

    Returns a list of GPU device indices, or ``None`` if unavailable.
    CPU-only jobs (``gpus_needed == 0``) return ``[]``.
    """
    if gpus_needed == 0:
        return []

    occ = occupancy if occupancy is not None else _worker_occupancy(wc)
    cap = wc.max_jobs_per_gpu  # 0 = unlimited

    # Filter GPUs that have room for another job
    available = [
        g for g in wc.gpus
        if cap <= 0 or occ[g] < cap
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
# Cancellation / eviction — one path for taking a run off the workers
# ===============================================================================


async def _detach_in_flight(
    state: ManagerState,
    run_ids: list[str],
) -> tuple[list[InFlightJob], list[tuple[WorkerConn, str]]]:
    """Remove in-flight tracking for *run_ids*; return (detached, cancel_targets).

    Occupancy is derived from ``wc.in_flight``, so popping the entry frees the
    run's GPUs with nothing to decrement.  Pre-removing also means any stale
    ``MsgResult`` that arrives afterwards hits the discard guard in
    ``_handle_result``.  Caller is responsible for sending ``MsgCancel`` to the
    returned targets and for the DB write (reset-to-pending or cancel).
    """
    detached: list[InFlightJob] = []
    cancel_targets: list[tuple[WorkerConn, str]] = []
    async with state.scheduler_lock:
        for run_id in run_ids:
            in_flight = state.get_in_flight(run_id)
            if in_flight is None:
                continue
            for wid in in_flight.worker_ids:
                wc = state.workers.get(wid)
                if wc is None:
                    continue
                wc.in_flight.pop(run_id, None)
                cancel_targets.append((wc, run_id))
            state.remove_in_flight(run_id)
            detached.append(in_flight)
    return detached, cancel_targets


async def evict_jobs(
    db: aiosqlite.Connection,
    state: ManagerState,
    run_ids: list[str],
) -> None:
    """Take in-flight runs off their workers and re-queue them (no retry spent).

    Detaches in-memory (frees derived occupancy), resets the DB rows to
    'pending', and sends ``MsgCancel`` so the workers SIGTERM the processes.
    The next scheduling pass picks the jobs back up from the DB.
    """
    detached, cancel_targets = await _detach_in_flight(state, run_ids)
    if not detached:
        return

    # One transaction resets all evicted jobs to pending in the DB.
    pairs = [(inf.run_id, inf.experiment_id) for inf in detached]
    await state.db_writer.reset_jobs_to_pending_batch(pairs)

    # Drop any multi-node placement rows; a re-dispatch re-records them. (No-op
    # for single-node runs.)
    for run_id, experiment_id in pairs:
        await state.db_writer.delete_job_nodes(run_id, experiment_id)

    for wc, run_id in cancel_targets:
        await _send_to_worker(wc, encode(MsgCancel(run_id=run_id)))

    if state.dispatch_callback is not None:
        await state.dispatch_callback()


async def cancel_runs(
    db: aiosqlite.Connection,
    state: ManagerState,
    pairs: list[tuple[str, str]],
) -> None:
    """Cancel jobs — the single cancellation path for every route.

    Works uniformly whether a job is pending or running: any in-flight run is
    stopped on its worker(s) (``MsgCancel`` → SIGTERM), every job row is marked
    'cancelled', and ``job_done`` is broadcast.  Because the run leaves
    ``wc.in_flight``, its GPUs are freed automatically — there is no separate
    occupancy counter to leak.  *pairs* is a list of ``(run_id, experiment_id)``.
    """
    run_ids = [run_id for run_id, _ in pairs]
    _, cancel_targets = await _detach_in_flight(state, run_ids)

    for wc, run_id in cancel_targets:
        await _send_to_worker(wc, encode(MsgCancel(run_id=run_id)))

    for run_id, experiment_id in pairs:
        await state.db_writer.update_job_status(run_id, experiment_id, "cancelled")
        # Remove any multi-node placement rows for the cancelled run (no-op for
        # single-node).
        await state.db_writer.delete_job_nodes(run_id, experiment_id)
        state.broadcast(
            experiment_id,
            {
                "type": "job_done",
                "run_id": run_id,
                "status": "cancelled",
                "success": False,
            },
        )

    # Freed capacity — let other work fill it.
    if cancel_targets and state.dispatch_callback is not None:
        await state.dispatch_callback()


# ===============================================================================
# Main scheduling entry point
# ===============================================================================


async def schedule_pending(
    db: aiosqlite.Connection,
    state: ManagerState,
) -> int:
    """Coalesced entry point: run scheduling passes until nothing can be dispatched.

    If called while a pass is already in progress, sets a flag so that the
    running pass executes one more iteration after it finishes.  This collapses
    any number of rapid concurrent triggers into at most two passes, preventing
    redundant planning phases and the job-visibility gap they create.

    Acquires ``state.scheduler_lock`` to scan the pending list and find
    available GPUs, then releases it before dispatching.  The dispatch
    helpers (``dispatch_to_worker`` and ``_execute_assignment``) acquire
    the lock internally for their DB write + in-memory state update.

    Returns the number of jobs successfully dispatched.
    """
    if state._scheduling:
        state._reschedule = True
        return 0
    state._scheduling = True
    try:
        total = 0
        while True:
            state._reschedule = False
            n = await _do_schedule_pending(db, state)
            total += n
            if not state._reschedule:
                break
        return total
    finally:
        state._scheduling = False


async def _do_schedule_pending(
    db: aiosqlite.Connection,
    state: ManagerState,
) -> int:
    """Single scheduling pass — called only by ``schedule_pending``.

    The candidate set comes straight from the database
    (``list_schedulable_jobs``), not an in-memory mirror.  This makes the DB the
    single source of truth: a job that was cancelled, whose experiment was
    paused/aborted, or that was already claimed simply does not appear here (or
    its ``dispatch_job`` CAS fails), so control verbs take effect with no
    in-memory reconciliation to get wrong.
    """
    # ── Phase 0: read candidates + caps from the DB (before the lock) ──
    # The dispatch_job CAS makes any TOCTOU safe: a job that changes out from
    # under us fails to claim and is skipped this pass.
    pending = await list_schedulable_jobs(db)
    if not pending:
        return 0
    caps = await experiment_concurrency_caps(db)

    # ── Phase 1: find assignments under lock ───────────────────────────
    async with state.scheduler_lock:
        connected = [
            wc
            for wc in state.workers.values()
            if wc.status == "connected"
        ]
        if not connected:
            return 0

        # Derive occupancy per worker from in-flight jobs; tentative bumps
        # within this pass prevent double-booking the same GPUs.
        tentative_occ: dict[str, dict[int, int]] = {
            wc.worker_id: _worker_occupancy(wc) for wc in connected
        }

        # Count running jobs per experiment so we can honour max_concurrent.
        exp_running: dict[str, int] = {}
        for ifj in state.in_flight.values():
            exp_running[ifj.experiment_id] = exp_running.get(ifj.experiment_id, 0) + 1

        # Plan: list of (job, [(worker, gpu_ids), ...])
        plan: list[tuple[JobRecord, list[tuple[WorkerConn, list[int]]]]] = []

        for job in pending:
            # Per-experiment concurrency cap (0 = unlimited).
            cap = caps.get(job.experiment_id, 0)
            if cap and exp_running.get(job.experiment_id, 0) >= cap:
                continue

            if job.nodes_per_run <= 1:
                # ── Single-node ────────────────────────────────────────
                for wc in connected:
                    gpus = _find_gpu_group(
                        wc, job.gpus_per_run,
                        topo=_parse_topo_wire(wc.topo),
                        occupancy=tentative_occ[wc.worker_id],
                    )
                    if gpus is not None:
                        plan.append((job, [(wc, gpus)]))
                        occ = tentative_occ[wc.worker_id]
                        for g in gpus:
                            occ[g] += 1
                        exp_running[job.experiment_id] = (
                            exp_running.get(job.experiment_id, 0) + 1
                        )
                        break
                # If unassigned, the job just stays 'pending' in the DB and is
                # reconsidered on the next pass — there is no in-memory list.
            else:
                # ── Multi-node ─────────────────────────────────────────
                node_assignments: list[tuple[WorkerConn, list[int]]] = []
                used_workers: set[str] = set()
                for wc in connected:
                    if wc.worker_id in used_workers:
                        continue
                    gpus = _find_gpu_group(
                        wc, job.gpus_per_run,
                        topo=_parse_topo_wire(wc.topo),
                        occupancy=tentative_occ[wc.worker_id],
                    )
                    if gpus is not None:
                        node_assignments.append((wc, gpus))
                        used_workers.add(wc.worker_id)
                        occ = tentative_occ[wc.worker_id]
                        for g in gpus:
                            occ[g] += 1
                        if len(node_assignments) >= job.nodes_per_run:
                            break
                if len(node_assignments) >= job.nodes_per_run:
                    plan.append((job, node_assignments))
                    exp_running[job.experiment_id] = (
                        exp_running.get(job.experiment_id, 0) + 1
                    )
                else:
                    # Couldn't place all nodes — roll back the tentative GPU
                    # reservations so the partial plan doesn't block other jobs
                    # later in this same pass.
                    for wc, gpus in node_assignments:
                        occ = tentative_occ[wc.worker_id]
                        for g in gpus:
                            occ[g] = max(0, occ[g] - 1)

    # ── Phase 2: dispatch outside the lock ─────────────────────────────
    # A failed dispatch leaves (or resets) the job 'pending' in the DB, so it is
    # naturally retried on the next pass; there is nothing to re-insert.
    dispatched_count = 0
    for job, assignments in plan:
        ok = await _execute_assignment(db, state, job, assignments)
        if ok:
            dispatched_count += 1

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

    # Pre-compute stable values before the DB write.
    all_worker_ids = [wc.worker_id for wc, _ in assignments]
    master_host = primary_wc.host.split("@")[-1]
    master_port = 29500  # default torch distributed port; could be parameterised

    if state.artifact_base_url and job.artifact_id:
        artifact_url = f"{state.artifact_base_url}/api/artifacts/{job.artifact_id}"
        if state.token:
            artifact_url += f"?token={state.token}"
    else:
        artifact_url = ""

    # Atomically claim the job in DB — hold lock only for the CAS + field extraction.
    async with state.scheduler_lock:
        dispatched = await state.db_writer.dispatch_job(
            job.run_id, job.experiment_id, primary_wc.worker_id, primary_gpus,
        )
        if dispatched is None:
            return False  # job already taken by another scheduler

        command, job_env, job_files, job_return_files = _parse_job_fields(dispatched)

    # Record durable per-node placement BEFORE dispatching, so a fast node's
    # MsgResult can never arrive before the rows exist (which would make
    # _handle_result mistake the run for single-node and finalise it early).
    placements = [
        (rank, wc.worker_id, gpu_ids)
        for rank, (wc, gpu_ids) in enumerate(assignments)
    ]
    await state.db_writer.insert_job_nodes(job.run_id, job.experiment_id, placements)

    # Dispatch each node outside the lock, tracking which nodes succeeded so
    # we can roll back cleanly on partial failure.
    dispatched_nodes: list[tuple[WorkerConn, list[int]]] = []

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
        if ok:
            dispatched_nodes.append((wc, gpu_ids))
        else:
            # Partial failure: roll back every node that already received MsgRun,
            # cancel them, and reset the DB — otherwise the job is permanently stuck
            # in in_flight on the successful nodes with no chance of a full result.
            async with state.scheduler_lock:
                for prev_wc, prev_gpu_ids in dispatched_nodes:
                    # Dropping the in-flight entry frees this node's derived
                    # occupancy; no counter to decrement.
                    prev_wc.in_flight.pop(job.run_id, None)
                state.remove_in_flight(job.run_id)

            for prev_wc, _ in dispatched_nodes:
                await _send_to_worker(prev_wc, encode(MsgCancel(run_id=job.run_id)))

            await state.db_writer.delete_job_nodes(job.run_id, job.experiment_id)
            await state.db_writer.reset_job_to_pending(job.run_id, job.experiment_id)
            return False

    return True


