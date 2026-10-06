"""Worker connections and the scheduling control plane of the mlsweep manager.

Provides:

  * Worker launch (local subprocess or remote SSH) and SSH reverse tunnels
  * Per-connection read/write/heartbeat tasks, and reconnect with backoff
  * Handlers for every Worker → Controller protocol message
  * Taking runs off workers (cancel, requeue) and the scheduler task

See ``_manager_state`` for the concurrency model.  Job status transitions and
in-flight tracking change together under ``state.lock``, and only
``scheduler_loop`` dispatches.  Functions named ``*_locked`` expect the caller
to hold ``state.lock``.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import aiosqlite

from mlsweep._manager_db import (
    JobRecord,
    SchedulableJob,
    count_jobs_by_status,
    experiment_concurrency_caps,
    get_experiment,
    last_log_seq,
    list_active_jobs,
    list_job_nodes,
    list_schedulable_jobs,
    list_workers,
    multinode_progress,
)
from mlsweep._manager_state import InFlightRun, ManagerState, RunKey, WorkerConn
from mlsweep._parsync import parsync_bin
from mlsweep._shared import (
    MsgCancel,
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
    PROTOCOL_VERSION,
    _GREEN,
    _RED,
    _YELLOW,
    _CYAN,
    _RESET,
    _git_root,
    aread_msg,
    decode,
    encode,
    from_obj,
    dist_master_port,
    line_chunks,
)
from mlsweep._topology import _best_gpu_groups, _parse_topo_wire

_HELLO_TIMEOUT = 30.0
_HEARTBEAT_INTERVAL = 10.0
# The manager pings every _HEARTBEAT_INTERVAL and the worker answers, so a
# connection silent for this long is dead even if TCP has not noticed.
_READ_TIMEOUT = 45.0
_MAX_RECONNECT_ATTEMPTS = 10
_SCHEDULE_INTERVAL = 5.0


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
    """Build or fetch the local mlsweep wheel into ``_wheels/`` at startup.

    Runs synchronously at manager startup (before the event loop).  Remote
    workers are bootstrapped by SCPing this wheel and pip-installing it there;
    pip resolves mlsweep's dependencies on the remote, against the remote's
    own Python and platform.  We deliberately do not pre-download dependency
    wheels: a wheel for the manager's Python may not fit the remote's Python.

    The ``.complete`` sentinel stores the mlsweep version the wheel was built
    from, so a version bump (or a sentinel left by an older checkout)
    invalidates the cache.  Without this, a stale wheel would be shipped to
    remote workers and install an ``mlsweep_worker`` that lacks flags the
    manager passes (e.g. ``--jobs``).
    """
    wheels_dir = Path(__file__).resolve().parent / "_wheels"
    local_version = importlib.metadata.version("mlsweep")
    complete = wheels_dir / ".complete"

    # Reuse the cache only when the sentinel matches the version we are about
    # to ship *and* a wheel for that version is actually present.
    if complete.exists():
        try:
            cached_version = complete.read_text(encoding="utf-8").strip()
        except OSError:
            cached_version = ""
        if cached_version == local_version and list(
            wheels_dir.glob(f"mlsweep-{local_version}-*.whl")
        ):
            return

    print("[wheels] Building worker wheel...", flush=True)
    wheels_dir.mkdir(exist_ok=True)

    # Drop stale mlsweep wheels from older versions.  We install the wheel
    # file directly on the remote, so leave exactly one candidate behind.
    for w in wheels_dir.glob("mlsweep-*.whl"):
        w.unlink(missing_ok=True)

    # From a source checkout (plain or editable), build the wheel from the
    # tree so local changes ship to workers.  From a regular install the
    # package's parent is site-packages, which pip cannot build, so fetch the
    # published wheel for the exact installed version instead.
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "pyproject.toml").exists():
        action = "wheel"
        args = ["--wheel-dir", str(wheels_dir), str(repo_root)]
    else:
        action = "download"
        args = ["--only-binary=:all:", "--dest", str(wheels_dir),
                f"mlsweep=={local_version}"]
    r = subprocess.run([sys.executable, "-m", "pip", action, "--no-deps", *args],
                       capture_output=True)
    if r.returncode != 0:
        print(
            f"[wheels] pip {action} failed:\n{r.stderr.decode(errors='replace')}",
            file=sys.stderr,
        )
        return

    # Record the version so the sentinel self-invalidates the moment the
    # local mlsweep version changes.
    try:
        complete.write_text(local_version, encoding="utf-8")
    except OSError:
        pass

    wheels = [p.name for p in wheels_dir.glob("*.whl")]
    print(f"[wheels] Ready ({len(wheels)} wheel)", flush=True)


async def _bootstrap_worker_venv(
    host: str,
    ssh_key: str | None = None,
    password: str | None = None,
) -> bool:
    """Ensure ``/tmp/mlsweep_venv/bin/mlsweep_worker`` exists on *host*.

    If the binary is already present and its version matches the local
    mlsweep, returns ``True`` immediately.  If it is present but outdated,
    the old venv is removed and a fresh one installed (``/tmp/mlsweep_venv``
    is manager-owned ephemeral state, so reinstalling it is safe).
    Otherwise it SCPs the bundled wheels to ``/tmp/mlsweep_wheels/``,
    creates the venv, and pip-installs ``mlsweep`` into it.

    Returns ``True`` on success, ``False`` if any install step fails.
    """
    key_args = ["-i", ssh_key] if ssh_key else []
    ssh_opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    sshpass_args, sshpass_env = _sshpass_args(password)

    # 1. Quick check: is the binary already present and up to date?
    local_version = importlib.metadata.version("mlsweep")
    check_cmd = (
        "if [ -x /tmp/mlsweep_venv/bin/mlsweep_worker ]; then "
        "/tmp/mlsweep_venv/bin/mlsweep_worker --version 2>/dev/null; "
        "else echo MISSING; fi"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh", *ssh_opts,
            *key_args,
            host,
            check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        out = stdout.decode(errors="replace").strip()
        if out == "MISSING":
            pass  # not installed — fall through to a fresh bootstrap
        elif local_version in out:
            return True  # present and matching — reuse it
        else:
            remote_ver = out.split()[-1] if out else "older (no --version flag)"
            print(
                f"  {_YELLOW}WARN{_RESET}  worker on {host} is outdated "
                f"(remote: {remote_ver}, expected: {local_version}); "
                f"reinstalling /tmp/mlsweep_venv",
                flush=True,
            )
            # fall through to a fresh bootstrap below
    except (OSError, asyncio.TimeoutError):
        pass  # fall through to bootstrap

    # 2. mkdir + SCP bundled wheels to remote.
    wheels_dir = Path(__file__).resolve().parent / "_wheels"
    wheel_files = [str(p) for p in wheels_dir.glob("*.whl")]
    if not wheel_files:
        print(
            f"[bootstrap] no mlsweep wheel in {wheels_dir} — "
            f"run the manager once to build it",
            file=sys.stderr,
        )
        return False
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

    # 3. Remove any stale venv, then create a fresh one and install the
    # bundled mlsweep wheel.  Dependencies are resolved by the remote's pip
    # against its own Python/platform.  The rm -rf matters: python3 -m venv
    # on an existing dir does not purge an old mlsweep install, and pip would
    # leave the stale worker's console script in place alongside the new one.
    try:
        proc = await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh", *ssh_opts,
            *key_args,
            host,
            (
                "rm -rf /tmp/mlsweep_venv && "
                "python3 -m venv /tmp/mlsweep_venv && "
                "/tmp/mlsweep_venv/bin/pip install "
                "/tmp/mlsweep_wheels/mlsweep-*.whl"
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)
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
    connect_host = _bare_host(host)
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
        elif "unrecognized arguments" in stderr_out:
            hint = "\n  hint: the worker on the remote machine is outdated — update mlsweep there"
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
# Worker identity and connection lifecycle
# ===============================================================================


def _bare_host(host: str) -> str:
    """*host* without an ``user@`` prefix, for connecting to it directly."""
    return host.split("@")[-1]


def worker_id_for(host: str, port: int, index: int) -> str:
    """Stable id for a worker.

    A worker on a fixed port is one long-lived process, so every path that
    reaches it (workers file, API) must agree on its id.  A worker on an
    ephemeral port is a fresh process per launch, told apart by *index*.
    """
    return f"{host}:{port}" if port else f"{host}:ephemeral:{index}"


def _send(wc: WorkerConn, msg: Any) -> None:
    """Queue *msg* on *wc*'s current connection (the queue is unbounded)."""
    wc.send_queue.put_nowait(encode(msg))


def _close_writer(writer: Any) -> None:
    try:
        writer.close()
    except Exception:
        pass


def _disconnect(wc: WorkerConn, *, shutdown: bool = False) -> None:
    """End *wc*'s current connection; with *shutdown*, tell the worker to exit first."""
    if shutdown:
        _send(wc, MsgShutdown())
    wc.send_queue.put_nowait(None)
    if not shutdown:
        _close_writer(wc.writer)


def _start_connection_locked(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Attach a new TCP connection to *wc*, send ``MsgHello``, start its tasks.

    Each connection has its own send queue and generation number, so tasks of
    an older connection can never act on this one.
    """
    wc.writer = writer
    wc.conn_gen += 1
    wc.status = "connecting"
    wc.send_queue = asyncio.Queue()
    _send(wc, MsgHello(token=state.token, controller_id="manager"))
    gen = wc.conn_gen
    asyncio.create_task(_worker_write_task(wc.send_queue, writer))
    asyncio.create_task(_worker_heartbeat_task(wc, gen))
    asyncio.create_task(_worker_read_task(db, state, wc, gen, reader))


async def _worker_write_task(queue: asyncio.Queue[bytes | None], writer: asyncio.StreamWriter) -> None:
    """Write one connection's queued messages until a ``None`` sentinel or an error."""
    try:
        while (item := await queue.get()) is not None:
            writer.write(item)
            await writer.drain()
    except OSError:
        pass
    finally:
        _close_writer(writer)


async def _worker_heartbeat_task(wc: WorkerConn, gen: int) -> None:
    """Ping the worker periodically; its pongs keep the read timeout from firing."""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        if wc.conn_gen != gen or wc.status not in ("connecting", "connected"):
            return
        _send(wc, MsgPing())


async def _worker_read_task(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    gen: int,
    reader: asyncio.StreamReader,
) -> None:
    """Read one connection: the hello first, then every other message in order."""
    try:
        hello = decode(await asyncio.wait_for(aread_msg(reader), timeout=_HELLO_TIMEOUT))
        if isinstance(hello, MsgWorkerHello):
            await _handle_worker_hello(db, state, wc, gen, hello)
        else:
            print(f"  {_YELLOW}WARN{_RESET}  Worker {wc.host} sent {type(hello).__name__} "
                  "before its hello; dropping the connection")
            hello = None
    except Exception as e:
        if not isinstance(e, (asyncio.TimeoutError, OSError, asyncio.IncompleteReadError, ValueError)):
            print(f"  {_RED}ERROR{_RESET} handling hello from {wc.host}: {e!r}")
            traceback.print_exc()
        hello = None

    while hello is not None and wc.conn_gen == gen:
        try:
            payload = await asyncio.wait_for(aread_msg(reader), timeout=_READ_TIMEOUT)
        except (asyncio.TimeoutError, OSError, asyncio.IncompleteReadError):
            break
        try:
            msg = decode(payload)
        except (ValueError, TypeError):
            continue
        handler = _HANDLERS.get(type(msg))
        if handler is None:
            continue
        try:
            await handler(db, state, wc, msg)
        except Exception as e:
            print(f"  {_RED}ERROR{_RESET} handling {type(msg).__name__} from {wc.host}: {e!r}")
            traceback.print_exc()

    await _on_connection_lost(db, state, wc, gen)


async def _on_connection_lost(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    gen: int,
) -> None:
    """Start reconnecting.  In-flight runs stay put: the worker keeps running them."""
    async with state.lock:
        if wc.conn_gen != gen or wc.status not in ("connecting", "connected"):
            return
        wc.status = "reconnecting"
        _disconnect(wc)
        if state.shutdown_event.is_set():
            return
        await state.db_writer.update_worker_status(wc.worker_id, "reconnecting")
    print(f"  {_YELLOW}WARN{_RESET}  Worker {wc.host} disconnected; reconnecting...")
    asyncio.create_task(_reconnect_worker(db, state, wc))


async def _reconnect_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
) -> None:
    """Reconnect with exponential backoff; declare the worker dead after too many tries.

    Each attempt goes through ``launch_worker``, which connects to a worker
    still serving the port and starts a fresh one if none is.  An idle worker
    exits when its manager goes away, so after a manager restart the worker
    may be gone (or on its way out); its runs, if any, died with it and the
    fresh worker's hello requeues them.

    The attempt count (and with it the backoff) is only reset by a successful
    hello, so a worker that accepts connections but fails the handshake backs
    off and eventually runs out like one that refuses them.
    """
    while wc.status == "reconnecting" and not state.shutdown_event.is_set():
        if wc.reconnect_attempts >= _MAX_RECONNECT_ATTEMPTS:
            await declare_worker_dead(
                db, state, wc.worker_id,
                f"unreachable after {_MAX_RECONNECT_ATTEMPTS} reconnect attempts",
            )
            return
        await asyncio.sleep(min(2.0 ** wc.reconnect_attempts, 30.0))
        wc.reconnect_attempts += 1
        try:
            reader, writer, _ = await launch_worker(
                host=wc.host, remote_dir=wc.remote_dir, token=state.token,
                scratch_dir=wc.scratch_dir, devices=sorted(wc.gpus + wc.unhealthy_gpus), max_jobs_per_gpu=wc.max_jobs_per_gpu,
                password=wc.password, ssh_key=wc.ssh_key, venv=wc.venv, port=wc.port,
            )
        except Exception:
            continue
        async with state.lock:
            if wc.status != "reconnecting":
                _close_writer(writer)
                return
            _start_connection_locked(db, state, wc, reader, writer)
        return


async def declare_worker_dead(
    db: aiosqlite.Connection,
    state: ManagerState,
    worker_id: str,
    reason: str,
    *,
    shutdown: bool = False,
) -> None:
    """Retire a worker and requeue everything the database has running on it.

    With *shutdown*, the worker is being removed on purpose.  Its runs are
    cancelled and requeued without spending retries, and it is told to exit.
    """
    async with state.lock:
        wc = state.workers.get(worker_id)
        keys = {run.key for run in state.runs_on(worker_id)}
        keys |= {j.key for j in await list_active_jobs(db, worker_id)}
        # Cancels go out before the worker is marked dead (dead workers get no messages).
        await requeue_runs_locked(db, state, sorted(keys), lost=not shutdown)
        if wc is not None and wc.status != "dead":
            wc.status = "dead"
            _disconnect(wc, shutdown=shutdown)
        await state.db_writer.update_worker_status(worker_id, "dead", last_error=reason or None)
    if not shutdown:
        print(f"  {_RED}FAIL{_RESET}  Worker {worker_id}: {reason}")


# ===============================================================================
# Worker hello, which reconciles what the worker runs with what the database expects
# ===============================================================================


async def _handle_worker_hello(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    gen: int,
    msg: MsgWorkerHello,
) -> None:
    """Adopt the worker's report of its runs, then make it schedulable.

    The same handling applies to every connection, including the first one
    after a manager restart.
    The worker reports runs it is executing (``resuming``) and results no
    manager has acknowledged (``completed``).  Against the jobs the database
    has in flight on this worker:

      * a run both sides know stays in flight (missing logs are replayed on demand);
      * a run the worker executes that the database does not expect here
        (cancelled, requeued or deleted meanwhile) is cancelled;
      * a run the database expects that the worker does not report was lost
        and is requeued (spending a retry only if it had started);
      * a result for an expected run is processed like any result; any other
        result is just acknowledged.
    """
    if msg.protocol != PROTOCOL_VERSION:
        await declare_worker_dead(
            db, state, wc.worker_id,
            f"worker speaks protocol {msg.protocol}, this manager {PROTOCOL_VERSION}; "
            "restart the worker with this mlsweep version",
            shutdown=True,
        )
        return
    results: list[MsgResult] = []
    async with state.lock:
        if wc.conn_gen != gen:
            return
        if not wc.hello_seen:
            wc.gpus = msg.gpus
            wc.max_jobs_per_gpu = msg.max_jobs_per_gpu
            wc.hello_seen = True
        wc.unhealthy_gpus = msg.unhealthy_gpus
        wc.topo = msg.topo
        wc.scratch_dir = msg.scratch_dir

        expected = {j.key: j for j in await list_active_jobs(db, wc.worker_id)}
        resuming = {(r["experiment"], r["run_id"]): r for r in msg.resuming}
        completed = {(r["experiment"], r["run_id"]): r for r in msg.completed}

        lost: list[JobRecord] = []
        for key, job in expected.items():
            report = resuming.get(key) or completed.get(key)
            if report is None:
                lost.append(job)
                continue
            await _adopt_run_locked(db, state, wc, job, report, running=key in resuming)
            if key in completed:
                results.append(from_obj(report))
        for exp_id, run_id in resuming.keys() - expected.keys():
            _send(wc, MsgCancel(run_id=run_id, experiment=exp_id))
        for exp_id, run_id in completed.keys() - expected.keys():
            _send(wc, MsgCleanup(run_id=run_id, experiment=exp_id, final=False))
        # A run that never started has not really been tried; one that started has.
        await requeue_runs_locked(db, state, [j.key for j in lost if j.status == "dispatched"], lost=False)
        await requeue_runs_locked(db, state, [j.key for j in lost if j.status == "running"], lost=True)

        wc.status = "connected"
        wc.reconnect_attempts = 0
        await state.db_writer.upsert_worker(
            worker_id=wc.worker_id,
            host=wc.host,
            remote_dir=wc.remote_dir,
            scratch_dir=msg.scratch_dir,
            port=wc.port,
            ssh_key=wc.ssh_key,
            venv=wc.venv,
            # Persist probe-excluded GPUs too: a manager restart relaunches the
            # worker with --devices from this column, and a GPU that is reset
            # later must still be a candidate for the next probe.
            devices=json.dumps(sorted(wc.gpus + wc.unhealthy_gpus)),
            unhealthy_devices=json.dumps(wc.unhealthy_gpus),
            status="connected",
        )
        n_gpus = len(wc.gpus)
        extra = (f"  {_YELLOW}{len(wc.unhealthy_gpus)} unhealthy GPU(s) "
                 f"excluded: {wc.unhealthy_gpus}{_RESET}") if wc.unhealthy_gpus else ""
        print(f"  {_GREEN}OK{_RESET}    {wc.host}: {n_gpus} GPU{'s' if n_gpus != 1 else ''} available{extra}")
        n_resumed = len(resuming.keys() & expected.keys())
        if n_resumed:
            print(f"  {_GREEN}RESUME{_RESET} {wc.host}: {n_resumed} run(s) still active")

    for result in results:
        await _handle_result(db, state, wc, result)
    state.request_schedule()


async def _adopt_run_locked(
    db: aiosqlite.Connection,
    state: ManagerState,
    wc: WorkerConn,
    job: JobRecord,
    report: dict[str, Any],
    *,
    running: bool,
) -> None:
    """Track *wc*'s node of *job* as in flight, as reported by the worker."""
    run = state.runs.get(job.key)
    if run is None:
        run = state.runs[job.key] = InFlightRun.from_job(
            job, primary=job.worker_id or wc.worker_id,
            log_end=await last_log_seq(db, job.job_key, job.attempt),
        )
    run.replay_requested = False
    # Book the GPUs the run is actually on.  A result has none to book.
    run.nodes[wc.worker_id] = [int(g) for g in report.get("gpu_ids", [])]
    if running and report.get("pid") and job.status == "dispatched":
        await state.db_writer.mark_job_running(job.run_id, job.experiment_id)


# ===============================================================================
# Message handlers
# ===============================================================================


def _find_run(state: ManagerState, wc: WorkerConn, experiment: str, run_id: str) -> InFlightRun | None:
    """The in-flight run a message from *wc* refers to, if *wc* holds a node of it."""
    run = state.runs.get((experiment, run_id))
    return run if run is not None and wc.worker_id in run.nodes else None


async def _handle_started(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgStarted,
) -> None:
    async with state.lock:
        run = _find_run(state, wc, msg.experiment, msg.run_id)
        if run is None:
            return
        run.last_progress = time.time()
        await state.db_writer.mark_job_running(run.run_id, run.experiment_id)
    state.broadcast(run.experiment_id, {
        "type": "job_started", "run_id": run.run_id, "worker_id": wc.worker_id, "pid": msg.pid,
    })


async def _handle_log(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgLog,
) -> None:
    """Store a log chunk if it continues the stored log exactly.

    Chunks sent on a connection that dropped are lost, so the next chunk
    leaves a gap.  The manager then asks once for a replay from where its copy
    ends and drops chunks until the replay arrives, which covers them.
    """
    run = _find_run(state, wc, msg.experiment, msg.run_id)
    if run is None or wc.worker_id != run.primary or msg.seq <= run.log_end:
        return
    if msg.start != run.log_end:
        if not run.replay_requested:
            run.replay_requested = True
            _send(wc, MsgReplay(run_id=run.run_id, experiment=run.experiment_id, log_seq=run.log_end))
        return
    run.log_end = msg.seq
    run.replay_requested = False
    run.last_progress = time.time()
    await state.db_writer.insert_log(run.job_key, run.attempt, msg.seq, msg.data)
    state.broadcast(run.experiment_id, {
        "type": "log", "run_id": run.run_id, "seq": msg.seq, "data": msg.data,
    })


async def _handle_metric(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgMetric,
) -> None:
    run = _find_run(state, wc, msg.experiment, msg.run_id)
    if run is None:
        return
    run.last_progress = time.time()
    await state.db_writer.insert_metric(run.job_key, run.attempt, msg.step, msg.data)
    state.broadcast(run.experiment_id, {
        "type": "metric", "run_id": run.run_id, "step": msg.step, "data": msg.data,
    })


async def _handle_sync_req(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgSyncReq,
) -> None:
    """Copy a running run's artifacts to the manager, off the read loop."""
    run = _find_run(state, wc, msg.experiment, msg.run_id)
    if run is None:
        return

    async def sync() -> None:
        await _run_rsync(state, wc, run.experiment_id, run.run_id)
        # final=False because the run is still executing; the worker keeps its scratch.
        _send(wc, MsgCleanup(run_id=run.run_id, experiment=run.experiment_id, final=False))

    asyncio.create_task(sync())


async def _handle_result(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgResult,
) -> None:
    """Record a node's result; the run finishes when its last node reports.

    The node's outputs are synced and its log and metrics completed before the
    result is recorded, so a job is never seen finished with a partial log or
    missing metrics.  Its GPUs are freed at once, since its process has exited.

    A result for a run not in flight on *wc* (cancelled, requeued or deleted,
    before or during the sync) is only acknowledged.
    """
    async with state.lock:
        run = _find_run(state, wc, msg.experiment, msg.run_id)
        if run is None:
            _send(wc, MsgCleanup(run_id=msg.run_id, experiment=msg.experiment, final=False))
            return
        eid, rid = run.experiment_id, run.run_id
        run.nodes[wc.worker_id] = []  # frees this node's GPUs
        if run.multinode and not msg.success:
            # The other nodes would wait on this one forever; stop them.
            for wid in run.nodes:
                if wid != wc.worker_id and (peer := state.workers.get(wid)) is not None:
                    _send(peer, MsgCancel(run_id=rid, experiment=eid))
        nodes = await list_job_nodes(db, rid, eid) if run.multinode else []
    state.request_schedule()

    # Copy the node's outputs to the manager and fill in what the live stream missed.
    subdir = None
    if nodes:
        rank = next((n.node_rank for n in nodes if n.worker_id == wc.worker_id), 0)
        subdir = f"node{rank}"
    synced = await _run_rsync(state, wc, eid, rid, subdir)
    if synced and wc.worker_id == run.primary:
        await _store_log_tail(state, run, subdir)
    primary_dir = _run_output_dir(state, eid, rid, "node0" if run.multinode else None)
    extra = await asyncio.to_thread(_read_metric_rows, primary_dir / "metrics.jsonl")

    async with state.lock:
        if _find_run(state, wc, eid, rid) is not run:
            _send(wc, MsgCleanup(run_id=rid, experiment=eid, final=synced))
            return
        if nodes:
            await state.db_writer.mark_job_node_result(rid, eid, wc.worker_id, msg.success, msg.elapsed)
            del run.nodes[wc.worker_id]
            remaining, success, elapsed = await multinode_progress(db, rid, eid)
            finished = remaining == 0
        else:
            finished, success, elapsed = True, msg.success, msg.elapsed

        if finished:
            del state.runs[run.key]
            await state.db_writer.pack_metrics(run.job_key, run.attempt, extra)
            await state.db_writer.finish_job(
                rid, eid, success=success, exit_code=msg.exit_code, elapsed=elapsed,
            )
            if nodes:
                await state.db_writer.delete_job_nodes(rid, eid)
            xfailed = await state.db_writer.apply_result_rules(eid, rid, success)
            state.broadcast(eid, {
                "type": "job_done", "run_id": rid, "worker_id": wc.worker_id,
                "success": success, "elapsed": elapsed, "exit_code": msg.exit_code,
                "xfailed": xfailed,
            })
            await _check_experiments_complete_locked(db, state, {eid})
    state.request_schedule()
    # The outputs are safe on the manager; the worker may delete its scratch.
    if synced:
        _send(wc, MsgCleanup(run_id=rid, experiment=eid, final=True))


def _run_output_dir(state: ManagerState, experiment_id: str, run_id: str, subdir: str | None = None) -> Path:
    """Where a run's (or one node's) outputs are synced to on the manager."""
    return Path(state.output_dir, experiment_id, run_id, subdir or "")


async def _store_log_tail(state: ManagerState, run: InFlightRun, subdir: str | None) -> None:
    """Store the part of the synced training.log that the live stream missed."""
    def read_tail() -> bytes:
        try:
            with open(_run_output_dir(state, run.experiment_id, run.run_id, subdir) / "training.log", "rb") as f:
                f.seek(run.log_end)
                return f.read()
        except OSError:
            return b""

    start = run.log_end
    for chunk in line_chunks(await asyncio.to_thread(read_tail)):
        start += len(chunk)
        await state.db_writer.insert_log(
            run.job_key, run.attempt, start, chunk.decode("utf-8", errors="replace"))


def _read_metric_rows(path: Path) -> list[tuple[int, str]]:
    """``(step, json)`` rows of a metrics.jsonl file."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rec = json.loads(line)
            step = int(rec.pop("step"))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        rows.append((step, json.dumps(rec, separators=(",", ":"))))
    return rows


async def _handle_pong(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgPong,
) -> None:
    """Update last_seen so the UI reflects a live worker."""
    await state.db_writer.touch_worker(wc.worker_id)


async def _handle_gpu_stats(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgGpuStats,
) -> None:
    """Keep the latest GPU utilization for the UI."""
    wc.gpu_stats = {s["gpu"]: s for s in msg.stats if "gpu" in s}


_HANDLERS: dict[type, Any] = {
    MsgStarted: _handle_started,
    MsgLog: _handle_log,
    MsgMetric: _handle_metric,
    MsgSyncReq: _handle_sync_req,
    MsgResult: _handle_result,
    MsgPong: _handle_pong,
    MsgGpuStats: _handle_gpu_stats,
}


async def _run_rsync(
    state: ManagerState, wc: WorkerConn, experiment_id: str, run_id: str, subdir: str | None = None,
) -> bool:
    """Copy *wc*'s scratch for a run into the manager's output dir (in a thread)."""
    return await asyncio.to_thread(
        _rsync_sync, wc.host, os.path.join(wc.scratch_dir, experiment_id, run_id),
        str(_run_output_dir(state, experiment_id, run_id, subdir)), run_id, wc.password, wc.ssh_key,
    )


# ===============================================================================
# Taking runs off workers
# ===============================================================================


def _detach_locked(state: ManagerState, keys: Sequence[RunKey]) -> None:
    """Stop tracking runs and tell every worker holding one of their nodes to kill it.

    A worker that is reconnecting misses the cancel; its next hello reports the
    run, which the database no longer expects there, so it is cancelled then.
    """
    for key in keys:
        run = state.runs.pop(key, None)
        if run is None:
            continue
        for wid in run.nodes:
            wc = state.workers.get(wid)
            if wc is not None and wc.status != "dead":
                _send(wc, MsgCancel(run_id=run.run_id, experiment=run.experiment_id))


async def cancel_runs_locked(
    db: aiosqlite.Connection, state: ManagerState, keys: Sequence[RunKey],
) -> list[JobRecord]:
    """Cancel jobs, pending or in flight.  Finished jobs are left alone."""
    _detach_locked(state, keys)
    cancelled = await state.db_writer.cancel_jobs(keys)
    for job in cancelled:
        state.broadcast(job.experiment_id, {
            "type": "job_done", "run_id": job.run_id, "status": "cancelled", "success": False,
        })
    await _check_experiments_complete_locked(db, state, {j.experiment_id for j in cancelled})
    state.request_schedule()
    return cancelled


async def requeue_runs_locked(
    db: aiosqlite.Connection, state: ManagerState, keys: Sequence[RunKey], *, lost: bool,
) -> None:
    """Take in-flight runs off their workers and put the jobs back to pending.

    *lost*: the run died with its worker, which spends a retry (a job with none
    left fails).  Otherwise the manager took the run away, e.g. to free a GPU,
    and no retry is spent.
    """
    if not keys:
        return
    _detach_locked(state, keys)
    requeued, failed = await state.db_writer.requeue_jobs(keys, spend_retry=lost)
    for job in requeued:
        if lost:
            print(f"  {_YELLOW}RETRY{_RESET} {job.run_id} (attempt {job.retry_count}/{job.max_retries})")
        state.broadcast(job.experiment_id, {"type": "job_updated", "run_id": job.run_id, "status": "pending"})
    for job in failed:
        print(f"  {_RED}FAIL{_RESET}  {job.run_id}: max retries exceeded")
        state.broadcast(job.experiment_id, {
            "type": "job_done", "run_id": job.run_id, "success": False,
            "elapsed": 0.0, "exit_code": -1, "orphaned": True,
        })
    await _check_experiments_complete_locked(db, state, {j.experiment_id for j in failed})
    state.request_schedule()


async def _check_experiments_complete_locked(
    db: aiosqlite.Connection, state: ManagerState, experiment_ids: set[str],
) -> None:
    """Mark running experiments completed once none of their jobs can still run."""
    for eid in experiment_ids:
        exp = await get_experiment(db, eid)
        if exp is None or exp.status != "running":
            continue
        counts = await count_jobs_by_status(db, eid)
        if not counts:
            # No jobs yet (e.g. a freshly created or un-paused experiment):
            # there is nothing to complete, so leave it running for submissions.
            continue
        if counts.get("pending", 0) + counts.get("dispatched", 0) + counts.get("running", 0):
            continue
        done = counts.get("done", 0)
        if exp.expected_jobs and done < exp.expected_jobs:
            continue
        await state.db_writer.update_experiment_status(eid, "completed")
        state.broadcast(eid, {"type": "experiment_done", "experiment_id": eid, "submitted_count": done})


async def requeue_jobs_of_unknown_workers(db: aiosqlite.Connection, state: ManagerState) -> None:
    """After startup, requeue jobs the database has running on workers no longer configured."""
    async with state.lock:
        keys = [
            j.key for j in await list_active_jobs(db)
            if j.worker_id not in state.workers and j.worker_id not in state.launching
        ]
        await requeue_runs_locked(db, state, keys, lost=True)


# ===============================================================================
# Connecting workers
# ===============================================================================


async def connect_workers(
    db: aiosqlite.Connection,
    state: ManagerState,
    *,
    workers_file: str | None = None,
    scratch_dir: str = "/tmp/mlsweep",
    manager_port: int = 0,
) -> list[WorkerConn]:
    """Launch and connect every configured worker concurrently.

    If *workers_file* is ``None``, launches a single local worker.
    """
    if workers_file:
        configs = _parse_workers_file(workers_file)
    else:
        # Local mode uses a single worker, which picks its own GPUs.
        configs = [{
            "host": "localhost",
            "remote_dir": _git_root(os.getcwd()) or os.getcwd(),
            "port": 0,
        }]

    ids = [worker_id_for(cfg["host"], cfg.get("port", 0), idx) for idx, cfg in enumerate(configs)]
    async with state.lock:
        reserved = [state.reserve_worker_id(wid) for wid in ids]
    results = await asyncio.gather(*(
        connect_single_worker(
            db, state,
            host=cfg["host"],
            remote_dir=cfg["remote_dir"],
            worker_id=wid,
            scratch_dir=scratch_dir,
            password=cfg.get("password"),
            ssh_key=cfg.get("ssh_key"),
            venv=cfg.get("venv"),
            port=cfg.get("port", 0),
            devices=cfg.get("devices"),
            max_jobs_per_gpu=cfg["jobs"] if cfg.get("jobs") is not None else 1,
            manager_port=manager_port,
        )
        for wid, cfg, ok in zip(ids, configs, reserved) if ok
    ))
    return [wc for wc in results if wc is not None]


async def reconnect_known_workers(
    db: aiosqlite.Connection,
    state: ManagerState,
    *,
    manager_port: int = 0,
) -> int:
    """Re-attempt remote workers recorded in the DB but not launched this session.

    Workers added dynamically (via the API) persist in the database across
    manager restarts, while ``connect_workers`` only starts the workers file
    (or a single local worker).  Without this, a restart leaves those workers
    dead with a stale ``last_error`` from the previous session.  Each eligible
    worker is scheduled for a background ``connect_single_worker``; success
    clears the error and a failure records a fresh reason.

    A worker is skipped when its host is already covered by a worker launched
    this session (the workers file is authoritative for that host), or when its
    worker id is already claimed.  Returns the number of workers scheduled for
    reconnect.
    """
    known = await list_workers(db)

    async with state.lock:
        covered = {wc.host for wc in state.workers.values()}

    scheduled = 0
    for wr in known:
        if wr.host == "localhost":
            continue
        if wr.host in covered:
            continue
        async with state.lock:
            if not state.reserve_worker_id(wr.worker_id):
                continue
        covered.add(wr.host)

        devices = json.loads(wr.devices) if wr.devices else None
        # Show the retry in the UI immediately instead of the previous
        # session's error; the connect task below replaces this with
        # "connected" or a fresh failure reason.
        await state.db_writer.update_worker_status(
            wr.worker_id, "reconnecting", last_error=None,
        )

        asyncio.create_task(connect_single_worker(
            db, state,
            host=wr.host,
            remote_dir=wr.remote_dir,
            worker_id=wr.worker_id,
            scratch_dir=wr.scratch_dir or "/tmp/mlsweep",
            ssh_key=wr.ssh_key,
            venv=wr.venv,
            port=wr.port,
            devices=devices,
            max_jobs_per_gpu=1,
            manager_port=manager_port,
        ))
        scheduled += 1

    return scheduled


async def connect_single_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    host: str,
    remote_dir: str,
    *,
    worker_id: str,
    scratch_dir: str = "/tmp/mlsweep",
    password: str | None = None,
    ssh_key: str | None = None,
    venv: str | None = None,
    port: int = 0,
    devices: list[int] | None = None,
    max_jobs_per_gpu: int = 1,
    manager_port: int = 0,
) -> WorkerConn | None:
    """Launch and connect to a worker, register it, and start its tasks.

    The caller has claimed *worker_id* with ``state.reserve_worker_id``; the
    claim is released here.  Returns ``None`` if the launch failed (the worker
    is then marked dead with the reason, and jobs the database had running on
    it are requeued).
    """
    reader: asyncio.StreamReader | None
    writer: asyncio.StreamWriter | None
    reason = ""
    try:
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
            reason = f"cannot start on {host}: {e}"
            if not (port and await list_active_jobs(db, worker_id)):
                await declare_worker_dead(db, state, worker_id, reason)
                return None
            # A worker on a fixed port outlives the manager and may still be
            # running our jobs; keep trying to reach it before giving them up.
            print(f"  {_YELLOW}WARN{_RESET}  {reason}; retrying, it has jobs in flight")
            reader = writer = None
            actual_port = port

        wc = WorkerConn(
            worker_id=worker_id,
            host=host,
            port=actual_port,
            remote_dir=remote_dir,
            scratch_dir=scratch_dir,
            password=password,
            ssh_key=ssh_key,
            venv=venv,
            max_jobs_per_gpu=max_jobs_per_gpu,
        )
        if host != "localhost" and manager_port:
            wc.tunnel_proc = await _launch_tunnel(host, manager_port, ssh_key=ssh_key, password=password)
            asyncio.create_task(_tunnel_monitor_task(wc, manager_port, state.shutdown_event))

        async with state.lock:
            state.workers[worker_id] = wc
            if reader is None or writer is None:
                wc.status = "reconnecting"
                await state.db_writer.update_worker_status(worker_id, "reconnecting", last_error=reason)
                asyncio.create_task(_reconnect_worker(db, state, wc))
                return wc
            _start_connection_locked(db, state, wc, reader, writer)
    finally:
        state.launching.discard(worker_id)

    print(f"  {_CYAN}START{_RESET} Worker {host}:{actual_port}")
    return wc


# ===============================================================================
# Scheduling
# ===============================================================================


async def scheduler_loop(db: aiosqlite.Connection, state: ManagerState) -> None:
    """The only place jobs are dispatched.

    Runs a pass whenever ``state.request_schedule()`` is called, and every
    ``_SCHEDULE_INTERVAL`` seconds regardless, so a missed wake-up can delay
    work but never strand it.  A pass holds ``state.lock`` from reading the
    pending jobs until the last dispatch, so nothing it planned on can change
    underneath it.
    """
    while True:
        try:
            await asyncio.wait_for(state.schedule_event.wait(), timeout=_SCHEDULE_INTERVAL)
        except asyncio.TimeoutError:
            pass
        state.schedule_event.clear()
        try:
            async with state.lock:
                await _schedule_pass_locked(db, state)
        except Exception as e:
            print(f"  {_RED}ERROR{_RESET} scheduling pass failed: {e!r}")
            traceback.print_exc()


async def _schedule_pass_locked(db: aiosqlite.Connection, state: ManagerState) -> None:
    """Dispatch every pending job that fits, in priority order."""
    connected = [wc for wc in state.workers.values() if wc.status == "connected"]
    if not connected:
        return
    occupancy = {wc.worker_id: state.occupancy(wc) for wc in connected}
    gpu_free = any(
        wc.max_jobs_per_gpu <= 0 or n < wc.max_jobs_per_gpu
        for wc in connected for n in occupancy[wc.worker_id].values()
    )
    pending = await list_schedulable_jobs(db, cpu_only=not gpu_free)
    if not pending:
        return
    caps = await experiment_concurrency_caps(db)
    topos = {wc.worker_id: _parse_topo_wire(wc.topo) for wc in connected}
    running: dict[str, int] = {}
    for run in state.runs.values():
        running[run.experiment_id] = running.get(run.experiment_id, 0) + 1

    # Occupancy only grows during a pass, so a shape that did not fit never will.
    no_fit: set[tuple[int, int]] = set()
    for job in pending:
        cap = caps.get(job.experiment_id, 0)
        if cap and running.get(job.experiment_id, 0) >= cap:
            continue
        shape = (job.gpus_per_run, job.nodes_per_run)
        if shape in no_fit:
            continue
        placements: list[tuple[WorkerConn, list[int]]] = []
        for wc in connected:
            gpus = _find_gpu_group(wc, job.gpus_per_run, topos[wc.worker_id],
                                   occupancy=occupancy[wc.worker_id])
            if gpus is not None:
                placements.append((wc, gpus))
                if len(placements) == max(1, job.nodes_per_run):
                    break
        if len(placements) < max(1, job.nodes_per_run):
            no_fit.add(shape)
            continue
        if not await _dispatch_locked(state, job, placements):
            continue
        for wc, gpus in placements:
            for g in gpus:
                occupancy[wc.worker_id][g] += 1
        running[job.experiment_id] = running.get(job.experiment_id, 0) + 1


async def _dispatch_locked(
    state: ManagerState,
    job: SchedulableJob,
    placements: list[tuple[WorkerConn, list[int]]],
) -> bool:
    """Claim *job* and send it to one worker per node.  Node 0 is the primary."""
    primary, primary_gpus = placements[0]
    claimed = await state.db_writer.dispatch_job(
        job.run_id, job.experiment_id, primary.worker_id, primary_gpus,
    )
    if claimed is None:
        return False  # no longer pending
    multinode = len(placements) > 1
    if multinode:
        # Recorded before any node starts, so no result can arrive first.
        await state.db_writer.insert_job_nodes(job.run_id, job.experiment_id, [
            (rank, wc.worker_id, gpus) for rank, (wc, gpus) in enumerate(placements)
        ])
    run = InFlightRun.from_job(
        claimed, primary=primary.worker_id,
        nodes={wc.worker_id: gpus for wc, gpus in placements},
    )
    state.runs[run.key] = run

    command = json.loads(claimed.command)
    if isinstance(command, str):
        command = [command]
    env: dict[str, str] = json.loads(claimed.env)
    artifact_url = ""
    if state.artifact_base_url and claimed.artifact_id:
        artifact_url = f"{state.artifact_base_url}/api/artifacts/{claimed.artifact_id}"
        if state.token:
            artifact_url += f"?token={state.token}"
    master_port = dist_master_port(job.experiment_id, job.run_id)
    for rank, (wc, gpus) in enumerate(placements):
        node_env = env
        if multinode:
            node_env = {
                **env,
                "MLSWEEP_NNODES": str(len(placements)),
                "MLSWEEP_NODE_RANK": str(rank),
                "MLSWEEP_MASTER_ADDR": _bare_host(primary.host),
                "MLSWEEP_MASTER_PORT": str(master_port),
            }
        _send(wc, MsgRun(
            run_id=job.run_id,
            experiment=job.experiment_id,
            command=command,
            env=node_env,
            gpu_ids=gpus,
            scratch=os.path.join(wc.scratch_dir, job.experiment_id, job.run_id),
            run_from=claimed.run_from,
            set_dist_env=claimed.set_dist_env,
            files=json.loads(claimed.files),
            return_files=json.loads(claimed.return_files),
            artifact_id=claimed.artifact_id or "",
            artifact_url=artifact_url,
            setup_command=shlex.split(claimed.setup_command) if claimed.setup_command else [],
        ))
        state.broadcast(job.experiment_id, {
            "type": "job_dispatched", "run_id": job.run_id,
            "worker_id": wc.worker_id, "host": wc.host, "gpu_ids": gpus,
        })
    return True


def _find_gpu_group(
    wc: WorkerConn,
    gpus_needed: int,
    topo: dict[tuple[int, int], int],
    *,
    occupancy: dict[int, int],
) -> list[int] | None:
    """Find *gpus_needed* GPUs on *wc* with room for another job, or ``None``.

    Per-GPU packing is bounded by the worker's ``max_jobs_per_gpu`` (0 =
    unlimited).  Prefers topologically close GPUs.  CPU-only jobs
    (``gpus_needed == 0``) get ``[]``.
    """
    if gpus_needed == 0:
        return []
    cap = wc.max_jobs_per_gpu
    available = [g for g in wc.gpus if cap <= 0 or occupancy[g] < cap]
    if len(available) < gpus_needed:
        return None
    if topo:
        groups = _best_gpu_groups(available, gpus_needed, 1, topo=topo)
        if groups:
            return groups[0]
    return available[:gpus_needed]


# ===============================================================================
# Artifact sync and SSH reverse tunnels
# ===============================================================================


def _rsync_sync(
    worker_host: str,
    remote_scratch: str,
    local_run_dir: str,
    run_id: str,
    password: str | None = None,
    ssh_key: str | None = None,
) -> bool:
    """Synchronous artifact sync (runs in executor thread).  Returns True on success."""
    ok = True
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
                ok = False
                print(f"  {_YELLOW}WARN{_RESET}  parsync failed for {run_id}: {e}")
        for fname in ("metrics.jsonl", "training.log"):
            src_f = os.path.join(remote_scratch, fname)
            dst_f = os.path.join(local_run_dir, fname)
            if src_f != dst_f and os.path.isfile(src_f):
                try:
                    shutil.copy2(src_f, dst_f)
                except OSError as e:
                    ok = False
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
            ok = False
            print(
                f"  {_YELLOW}WARN{_RESET}  parsync failed for {run_id}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
    return ok


def _load_prctl() -> Callable[..., int] | None:
    """libc's ``prctl``, resolved once in the parent (None where unavailable)."""
    try:
        import ctypes

        return ctypes.CDLL(None, use_errno=True).prctl
    except Exception:
        return None


_PRCTL = _load_prctl()


def _die_with_parent() -> None:
    """Arrange for a child process to get SIGTERM when its parent (the manager) dies.

    ``prctl(PR_SET_PDEATHSIG, ...)`` asks the kernel to deliver SIGTERM to this
    process the moment its parent exits — covering SIGKILL and crashes, which a
    shutdown hook can't.  This runs as ``preexec_fn`` in the child, so libc is
    resolved beforehand and the child only makes the syscall.  Errors are
    ignored so tunnel setup never fails because of it.
    """
    if _PRCTL is not None:
        _PRCTL(1, signal.SIGTERM)  # PR_SET_PDEATHSIG == 1


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
            preexec_fn=_die_with_parent,
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
