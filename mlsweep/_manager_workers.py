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
import contextlib
import fcntl
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import aiosqlite

from mlsweep._manager_controls import PruneTracker, metric_value, missing_metric_reason
from mlsweep._manager_optimizer import advance_optimizers_locked, give_up_locked
from mlsweep._manager_db import (
    MISSING_METRIC_REASON,
    JobRecord,
    JobStatus,
    SchedulableJob,
    WorkerRecord,
    count_jobs_by_status,
    experiment_concurrency_caps,
    get_experiment,
    get_metrics_for_run,
    get_worker,
    last_log_seq,
    latest_metric_rows,
    list_active_jobs,
    list_controlled_experiments,
    list_job_nodes,
    list_schedulable_jobs,
    list_workers,
    missing_metric_counts,
    multinode_progress,
    parse_metric_rows,
    running_jobs_started_before,
)
from mlsweep._sweep import format_duration
from mlsweep._manager_state import (
    PIN_TTL_SECONDS, InFlightRun, ManagerState, Pin, Placement, RunKey, WorkerConn,
)
from mlsweep._parsync import parsync_bin
from mlsweep._snapshot import is_snapshot_id
from mlsweep._shared import (
    DEFAULT_SCRATCH_DIR,
    MsgCancel,
    MsgCleanup,
    MsgGpuHealth,
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
    code_fingerprint,
    decode,
    encode,
    from_obj,
    dist_master_port,
    line_chunks,
    same_attempt,
)
from mlsweep._topology import _parse_topo_wire, closest_gpu_group

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
    Optional fields: gpus, jobs, devices, pass, ssh_key, venv, port, scratch_dir.

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
        validate_worker_host(host)
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
                "scratch_dir": entry.get("scratch_dir"),
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


class WheelCache:
    """The worker wheel shipped to remote workers, built or fetched at manager startup.

    *dir* holds the wheel and a ``.complete`` sentinel; *run* runs pip (like
    ``subprocess.run``).
    """

    def __init__(self, dir: Path, run: Callable[..., "subprocess.CompletedProcess[bytes]"] = subprocess.run):
        self.dir = dir
        self.run = run

    def ensure(self) -> None:
        """Build or fetch the local mlsweep wheel into *dir*, unless it is current.

        Remote workers are bootstrapped by SCPing this wheel and pip-installing
        it there; pip resolves mlsweep's dependencies on the remote, against the
        remote's own Python and platform.  We deliberately do not pre-download
        dependency wheels: a wheel for the manager's Python may not fit the
        remote's Python.

        The ``.complete`` sentinel stores the mlsweep version and the worker
        code's ``code_fingerprint()`` the wheel was built from, so a version bump
        or a worker-code change in a source checkout (which keeps its version)
        invalidates the cache.  Without this, a stale wheel would be shipped to
        remote workers and install an ``mlsweep_worker`` that lacks flags the
        manager passes (e.g. ``--jobs``).

        Managers starting at the same time from one install build one at a time,
        under a lock in the wheels directory; the others then reuse the result.
        """
        local_version = importlib.metadata.version("mlsweep")
        if self.is_current(local_version):
            return
        self.dir.mkdir(exist_ok=True)
        with open(self.dir / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.is_current(local_version):  # not built while we waited
                self.build(local_version)

    def read_stamp(self) -> str | None:
        """The stamp the last finished wheel build left, or None if there is none."""
        try:
            return (self.dir / ".complete").read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def is_current(self, local_version: str) -> bool:
        """Whether the sentinel matches the version and code we are about to ship
        *and* a wheel for that version is actually present."""
        return self.read_stamp() == _wheel_stamp(local_version) and bool(
            list(self.dir.glob(f"mlsweep-{local_version}-*.whl")))

    def shipped_stamp(self, local_version: str) -> str:
        """Version and code fingerprint of the wheel a bootstrap would install.

        An installed worker prints the same pair after its name for ``--version``.
        """
        return self.read_stamp() or _wheel_stamp(local_version)

    def wheels(self) -> list[Path]:
        return sorted(self.dir.glob("*.whl"))

    def build(self, local_version: str) -> None:
        """Replace the cached wheel with one for *local_version*.  Caller holds the lock."""
        complete = self.dir / ".complete"
        stamp = _wheel_stamp(local_version)  # before the build, so a later edit invalidates it
        print("[wheels] Building worker wheel...", flush=True)

        # Drop stale mlsweep wheels from older versions.  We install the wheel
        # file directly on the remote, so leave exactly one candidate behind.
        for w in self.dir.glob("mlsweep-*.whl"):
            w.unlink(missing_ok=True)

        # From a source checkout (plain or editable), build the wheel from the
        # tree so local changes ship to workers.  From a regular install the
        # package's parent is site-packages, which pip cannot build, so fetch the
        # published wheel for the exact installed version instead.
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / "pyproject.toml").exists():
            action = "wheel"
            args = ["--wheel-dir", str(self.dir), str(repo_root)]
        else:
            action = "download"
            args = ["--only-binary=:all:", "--dest", str(self.dir),
                    f"mlsweep=={local_version}"]
        r = self.run([sys.executable, "-m", "pip", action, "--no-deps", *args], capture_output=True)
        if r.returncode != 0:
            print(
                f"[wheels] pip {action} failed:\n{r.stderr.decode(errors='replace')}",
                file=sys.stderr,
            )
            return

        # Record version and code so the sentinel self-invalidates the moment
        # either changes.
        try:
            complete.write_text(stamp, encoding="utf-8")
        except OSError:
            pass

        print(f"[wheels] Ready ({len(self.wheels())} wheel)", flush=True)


def _wheel_stamp(local_version: str) -> str:
    # The code on disk, which is what a build ships, not what this process loaded.
    return f"{local_version} {code_fingerprint()}"


# The manager's own wheel cache, next to the package.
WHEELS = WheelCache(Path(__file__).resolve().parent / "_wheels")


def _ensure_worker_wheels() -> None:
    """Build or fetch the worker wheel at startup (see ``WheelCache.ensure``)."""
    WHEELS.ensure()


# How the manager starts ssh, scp and local workers: ``asyncio.create_subprocess_exec``.
Spawn = Callable[..., Any]


async def _bootstrap_worker_venv(
    host: str,
    ssh_key: str | None = None,
    password: str | None = None,
    *,
    wheels: WheelCache | None = None,
    spawn: Spawn | None = None,
) -> None:
    """Ensure ``/tmp/mlsweep_venv/bin/mlsweep_worker`` on *host* runs this manager's worker code.

    The wheel is first rebuilt if the worker code here changed since it was
    built.  If the remote binary reports the wheel's version and code
    fingerprint, returns immediately.  Otherwise the old venv, if any, is
    removed and a fresh one installed (``/tmp/mlsweep_venv`` is
    manager-owned ephemeral state, so reinstalling it is safe): the wheel is
    copied to ``/tmp/mlsweep_wheels/``, the venv created and ``mlsweep``
    pip-installed into it.  Comparing the fingerprint, not just the version,
    matters for a source checkout, which changes code without changing its
    version.

    *wheels* is the wheel cache to ship from (default ``WHEELS``), and *spawn*
    starts ssh and scp (default ``asyncio.create_subprocess_exec``).

    Raises ``BootstrapError`` naming the step that failed and why.
    """
    wheels = wheels or WHEELS
    spawn = spawn or asyncio.create_subprocess_exec
    key_args = ["-i", ssh_key] if ssh_key else []
    ssh_opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    sshpass_args, sshpass_env = _sshpass_args(password)

    ssh = ["ssh", *ssh_opts, *key_args, "--", host]

    async def run(step: str, argv: list[str], timeout: float) -> str:
        try:
            proc = await spawn(
                *sshpass_args, *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=sshpass_env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (OSError, asyncio.TimeoutError) as e:
            raise BootstrapError(f"{step} failed on {host}: {e or type(e).__name__}") from None
        if proc.returncode != 0:
            err = stderr.decode(errors="replace")
            print(f"[bootstrap] {step} failed for {host}:\n{err}", file=sys.stderr)
            raise BootstrapError(f"{step} failed on {host}: {_error_summary(err)}")
        return str(stdout.decode(errors="replace")).strip()

    # 1. Quick check: is the binary already present and up to date?
    await asyncio.to_thread(wheels.ensure)
    local_version = importlib.metadata.version("mlsweep")
    expected = wheels.shipped_stamp(local_version)
    check_cmd = (
        "if [ -x /tmp/mlsweep_venv/bin/mlsweep_worker ]; then "
        "/tmp/mlsweep_venv/bin/mlsweep_worker --version 2>/dev/null; "
        "else echo MISSING; fi"
    )
    out = "MISSING"
    with contextlib.suppress(BootstrapError):
        out = await run("checking the venv", [*ssh, check_cmd], 15.0)
    if out != "MISSING":
        remote = " ".join(out.split()[1:])  # drop the program name
        if remote == expected:
            return  # present and matching — reuse it
        print(
            f"  {_YELLOW}WARN{_RESET}  worker on {host} is outdated "
            f"(remote: {remote or 'older, no --version flag'}, expected: {expected}); "
            f"reinstalling /tmp/mlsweep_venv",
            flush=True,
        )

    # 2. Ship the bundled wheel into an emptied /tmp/mlsweep_wheels, so wheels
    # from older managers do not pile up there.  Step 3 names the wheels it
    # installs, since pip refuses two versions of mlsweep at once.
    shipped = wheels.wheels()
    if not shipped:
        raise BootstrapError(f"no mlsweep wheel in {wheels.dir}; the manager failed to build it "
                             "(its log has pip's error)")
    await run("preparing /tmp/mlsweep_wheels",
              [*ssh, "rm -rf /tmp/mlsweep_wheels && mkdir -p /tmp/mlsweep_wheels"], 30.0)
    await run("copying the wheel",
              ["scp", *ssh_opts, *key_args, "--", *map(str, shipped), f"{host}:/tmp/mlsweep_wheels/"],
              120.0)

    # 3. Remove any stale venv, then create a fresh one and install the
    # bundled mlsweep wheel.  Dependencies are resolved by the remote's pip
    # against its own Python/platform.  The rm -rf matters: python3 -m venv
    # on an existing dir does not purge an old mlsweep install, and pip would
    # leave the stale worker's console script in place alongside the new one.
    remote_wheels = " ".join(shlex.quote(f"/tmp/mlsweep_wheels/{w.name}") for w in shipped)
    await run("installing mlsweep", [*ssh, (
        "rm -rf /tmp/mlsweep_venv && "
        "python3 -m venv /tmp/mlsweep_venv && "
        f"/tmp/mlsweep_venv/bin/pip install {remote_wheels}"
    )], 300.0)


class BootstrapError(RuntimeError):
    """Installing mlsweep on a remote worker host failed; the message says why."""


def _error_summary(stderr: str, limit: int = 300) -> str:
    """The lines of a failed command's stderr that say what went wrong.

    pip prints progress before its error, so prefer its ``ERROR:`` lines and
    fall back to the last few lines.
    """
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.startswith("ERROR")]
    text = " ".join(errors or lines[-3:]) or "no output"
    return text if len(text) <= limit else text[:limit] + "…"


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


# Tasks copying launched workers' output; held so they are not garbage collected.
_output_relays: set["asyncio.Task[None]"] = set()


async def _relay_stream(stream: asyncio.StreamReader, label: str) -> None:
    """Copy *stream* to the manager's stderr, one prefixed line at a time."""
    buf = b""
    while chunk := await stream.read(65536):
        *lines, buf = (buf + chunk).split(b"\n")
        if len(buf) > 65536:  # no newline in sight; do not buffer without bound
            lines.append(buf)
            buf = b""
        for ln in lines:
            print(f"[worker {label}] {ln.decode(errors='replace')}", file=sys.stderr, flush=True)
    if buf:
        print(f"[worker {label}] {buf.decode(errors='replace')}", file=sys.stderr, flush=True)


def _relay_worker_output(proc: asyncio.subprocess.Process, label: str) -> None:
    """Keep reading a launched worker's stdout and stderr for as long as it runs.

    Left unread, a pipe fills up after about 64 KiB, and the worker then blocks
    on its next write of a traceback, a warning or a child's error output.
    """
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            task = asyncio.create_task(_relay_stream(stream, label))
            _output_relays.add(task)
            task.add_done_callback(_output_relays.discard)


async def launch_worker(
    host: str,
    remote_dir: str,
    token: str,
    scratch_dir: str = DEFAULT_SCRATCH_DIR,
    devices: list[int] | None = None,
    max_jobs_per_gpu: int = 1,
    password: str | None = None,
    ssh_key: str | None = None,
    venv: str | None = None,
    port: int = 0,
    wheels: WheelCache | None = None,
    spawn: Spawn | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, int]:
    """Launch a worker process and return connected streams + port.

    For ``host == "localhost"``, spawns ``python -m mlsweep.worker`` as a
    local subprocess.  For remote hosts, connects via SSH and runs the
    ``mlsweep_worker`` binary.

    Returns ``(reader, writer, port)`` where *reader* and *writer* are
    asyncio stream objects connected to the worker's TCP port.  *wheels* and
    *spawn* are as for ``_bootstrap_worker_venv``.
    """
    spawn = spawn or asyncio.create_subprocess_exec
    validate_worker_host(host)
    connect_host = _bare_host(host)
    devices_args = (
        ["--devices", ",".join(str(d) for d in devices)] if devices else []
    )
    jobs_args = ["--jobs", str(max_jobs_per_gpu)]
    key_args = ["-i", ssh_key] if ssh_key else []
    bind_port = port

    # ── Try to reuse an existing worker at the fixed port ────────────────
    if bind_port != 0:
        conn = await _try_connect(connect_host, bind_port)
        if conn is not None:
            return conn[0], conn[1], bind_port

    # ── Launch a fresh worker ────────────────────────────────────────────
    if host == "localhost":
        # Local: spawn python -m mlsweep.worker
        proc = await spawn(
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
        await _bootstrap_worker_venv(host, ssh_key=ssh_key, password=password, wheels=wheels, spawn=spawn)
        worker_args = [
            "--token", token,
            "--remote-dir", remote_dir,
            "--scratch-dir", scratch_dir,
            "--port", str(bind_port),
            *devices_args,
            *jobs_args,
        ]
        shell_cmd = _worker_shell_cmd(_worker_candidates(venv), worker_args)
        ssh_cmd = [
            *sshpass_args,
            "ssh", "-o", "ConnectTimeout=10",
            *key_args,
            "--", host, shell_cmd,
        ]
        proc = await spawn(
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
    _relay_worker_output(proc, f"{host}:{worker_port}")

    # Connect to the worker's TCP port
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(connect_host, worker_port),
        timeout=10.0,
    )
    return reader, writer, worker_port


# ===============================================================================
# Worker identity and connection lifecycle
# ===============================================================================


def validate_worker_host(host: Any) -> str:
    """Return *host* if it can only be read as an SSH destination; raise ValueError otherwise."""
    if not isinstance(host, str) or not host or host.startswith("-") or re.search(r"[\s\x00-\x1f\x7f]", host):
        raise ValueError(f"invalid worker host: {host!r}")
    return host


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
    except asyncio.IncompleteReadError:
        if wc.conn_gen == gen and not state.shutdown_event.is_set():
            print(f"  {_YELLOW}WARN{_RESET}  Worker {wc.host} closed the connection before its hello; "
                  "if this repeats, its --token may not match the manager's")
        hello = None
    except Exception as e:
        if not isinstance(e, (asyncio.TimeoutError, OSError, ValueError)):
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
                scratch_dir=wc.scratch_dir, devices=wc.gpus, max_jobs_per_gpu=wc.max_jobs_per_gpu,
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
            _close_tunnel(wc)
        await state.db_writer.update_worker_status(worker_id, "dead", last_error=reason or None)
    if not shutdown:
        print(f"  {_RED}FAIL{_RESET}  Worker {worker_id}: {reason}")


def _close_tunnel(wc: WorkerConn) -> None:
    """End *wc*'s reverse tunnel and the task that restarts it.  A relaunch of
    the worker opens a new one on the same remote port, which an open old
    tunnel would hold."""
    task, wc.tunnel_task = wc.tunnel_task, None
    if task is not None and task is not asyncio.current_task():
        task.cancel()
    proc, wc.tunnel_proc = wc.tunnel_proc, None
    _terminate_proc(proc)


def _terminate_proc(proc: "asyncio.subprocess.Process | None") -> None:
    """Ask *proc* to exit if it is still running."""
    if proc is not None and proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()


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
        state.outdated.add(wc.worker_id)
        await declare_worker_dead(
            db, state, wc.worker_id,
            f"worker speaks protocol {msg.protocol}, this manager {PROTOCOL_VERSION}; "
            "update the worker to this mlsweep version",
            shutdown=True,
        )
        return
    results: list[MsgResult] = []
    async with state.lock:
        if wc.conn_gen != gen:
            return
        if not wc.hello_seen:
            # Every GPU the worker offers joins the pool; the unhealthy and pending
            # ones are skipped by placement until the worker reports them usable.
            wc.gpus = sorted(set(msg.gpus) | set(msg.unhealthy_gpus) | set(msg.pending_gpus))
            wc.max_jobs_per_gpu = msg.max_jobs_per_gpu
            wc.hello_seen = True
        _set_health(wc, msg)
        wc.topo = msg.topo
        wc.scratch_dir = msg.scratch_dir

        expected = {j.key: j for j in await list_active_jobs(db, wc.worker_id)}

        def current(r: dict[str, Any]) -> bool:
            """Whether a reported run is the attempt the database has on this worker."""
            job = expected.get((r["experiment"], r["run_id"]))
            return job is not None and same_attempt(r.get("attempt", 0), job.attempt)

        # Reports of other attempts (or of runs not expected here) are ghosts.
        resuming = {(r["experiment"], r["run_id"]): r for r in msg.resuming if current(r)}
        completed = {(r["experiment"], r["run_id"]): r for r in msg.completed if current(r)}
        for r in msg.resuming:
            if not current(r):
                _send(wc, MsgCancel(run_id=r["run_id"], experiment=r["experiment"]))
        for r in msg.completed:
            if not current(r):
                _send(wc, MsgCleanup(run_id=r["run_id"], experiment=r["experiment"],
                                     attempt=r.get("attempt", 0), final=False))

        lost: list[JobRecord] = []
        for key, job in expected.items():
            report = resuming.get(key) or completed.get(key)
            if report is None:
                lost.append(job)
                continue
            await _adopt_run_locked(db, state, wc, job, report, running=key in resuming)
            if key in completed:
                results.append(from_obj(report))
        # A run that never started has not really been tried; one that started has.
        await requeue_runs_locked(db, state, [j.key for j in lost if j.status == "dispatched"], lost=False)
        await requeue_runs_locked(db, state, [j.key for j in lost if j.status == "running"], lost=True)

        wc.status = "connected"
        wc.reconnect_attempts = 0
        # Compared with the code on disk when the worker is listed (see
        # ``worker_outdated``), so an update clears it without a manager restart.
        wc.code = msg.code
        state.outdated.discard(wc.worker_id)
        await state.db_writer.upsert_worker(
            worker_id=wc.worker_id,
            host=wc.host,
            remote_dir=wc.remote_dir,
            scratch_dir=msg.scratch_dir,
            port=wc.port,
            ssh_key=wc.ssh_key,
            venv=wc.venv,
            # The pool includes probe-excluded GPUs: a manager restart relaunches
            # the worker with --devices from this column, and a GPU that is reset
            # later must still be a candidate for the next probe.
            devices=json.dumps(wc.gpus),
            unhealthy_devices=json.dumps(wc.unhealthy_gpus),
            status="connected",
        )
        print(f"  {_GREEN}OK{_RESET}    {wc.host}: {_health_summary(wc)}")
        n_resumed = len(resuming.keys() & expected.keys())
        if n_resumed:
            print(f"  {_GREEN}RESUME{_RESET} {wc.host}: {n_resumed} run(s) still active")

    for result in results:
        await _handle_result(db, state, wc, result)
    state.request_schedule()


def _set_health(wc: WorkerConn, msg: MsgWorkerHello | MsgGpuHealth) -> None:
    wc.unhealthy_gpus = list(msg.unhealthy_gpus)
    wc.unhealthy_reasons = dict(msg.unhealthy_reasons)
    wc.pending_gpus = list(msg.pending_gpus)


def _health_summary(wc: WorkerConn) -> str:
    """"3 GPUs available  5 unhealthy GPU(s) excluded: [0, 1] cuInit failed: ..." for logs."""
    n = len(wc.usable_gpus)
    out = f"{n} GPU{'s' if n != 1 else ''} available"
    if wc.unhealthy_gpus:
        by_reason: dict[str, list[int]] = {}
        for g in wc.unhealthy_gpus:
            by_reason.setdefault(wc.unhealthy_reasons.get(str(g), "failed the CUDA probe"), []).append(g)
        why = "; ".join(f"{gs} {r}" for r, gs in by_reason.items())
        out += f"  {_YELLOW}{len(wc.unhealthy_gpus)} unhealthy GPU(s) excluded: {why}{_RESET}"
    if wc.pending_gpus:
        out += f"  {len(wc.pending_gpus)} GPU(s) still probing: {wc.pending_gpus}"
    return out


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


def _find_run(
    state: ManagerState, wc: WorkerConn, experiment: str, run_id: str, attempt: int,
) -> InFlightRun | None:
    """The in-flight run a message from *wc* refers to, if *wc* holds a node of it
    and the message is about the attempt in flight, not one taken back earlier."""
    run = state.runs.get((experiment, run_id))
    if run is None or wc.worker_id not in run.nodes or not same_attempt(attempt, run.attempt):
        return None
    return run


async def _handle_started(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgStarted,
) -> None:
    async with state.lock:
        run = _find_run(state, wc, msg.experiment, msg.run_id, msg.attempt)
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
    run = _find_run(state, wc, msg.experiment, msg.run_id, msg.attempt)
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
    run = _find_run(state, wc, msg.experiment, msg.run_id, msg.attempt)
    if run is None:
        return
    run.last_progress = time.time()
    run.note_step(msg.step, msg.step_seconds, run.last_progress)
    await state.db_writer.insert_metric(run.job_key, run.attempt, msg.step, msg.data)
    state.broadcast(run.experiment_id, {
        "type": "metric", "run_id": run.run_id, "step": msg.step, "data": msg.data,
        "step_seconds": msg.step_seconds,
    })
    why = _apply_metric_controls(state, run, msg.step, msg.data)
    if why is not None:
        async with state.lock:
            if state.runs.get(run.key) is run:
                stop_runs_locked(state, [run.key], "pruned", why)


async def _handle_sync_req(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgSyncReq,
) -> None:
    """Copy a running run's artifacts to the manager, off the read loop."""
    async with state.lock:
        run = _find_run(state, wc, msg.experiment, msg.run_id, msg.attempt)
        # Two sync calls for one run would rsync into the same directory at once.
        if run is None or run.sync_inflight:
            return
        run.sync_inflight = True

    async def sync() -> None:
        try:
            await _run_rsync(state, wc, run.experiment_id, run.run_id)
            # final=False because the run is still executing; the worker keeps its scratch.
            _send(wc, MsgCleanup(run_id=run.run_id, experiment=run.experiment_id,
                                 attempt=run.attempt, final=False))
        finally:
            run.sync_inflight = False

    asyncio.create_task(sync())


async def _handle_result(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgResult,
) -> None:
    """Record a node's result; the run finishes when its last node reports.

    The node's outputs are synced and its log and metrics completed before the
    result is recorded, so a job is never seen finished with a partial log or
    missing metrics.  Its GPUs are freed at once, since its process has exited.

    A result for a run not in flight on *wc* (cancelled, requeued or deleted,
    before or during the sync), or for an earlier attempt of it, is only
    acknowledged.
    """
    async with state.lock:
        run = _find_run(state, wc, msg.experiment, msg.run_id, msg.attempt)
        if run is None:
            _send(wc, MsgCleanup(run_id=msg.run_id, experiment=msg.experiment,
                                 attempt=msg.attempt, final=False))
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
        if _find_run(state, wc, eid, rid, msg.attempt) is not run:
            _send(wc, MsgCleanup(run_id=rid, experiment=eid, attempt=run.attempt, final=synced))
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
            # A run the manager stopped ends as what it stopped it for, however it exited.
            status, reason = run.ending if run.ending is not None else (None, None)
            if status is not None:
                success = False
            await state.db_writer.finish_job(
                rid, eid, success=success, exit_code=msg.exit_code, elapsed=elapsed,
                status=status, reason=reason,
            )
            if nodes:
                await state.db_writer.delete_job_nodes(rid, eid)
            # A pruned run ran fine, so for skip rules it is a success.
            xfailed = await state.db_writer.apply_result_rules(eid, rid, success or status == "pruned")
            state.broadcast(eid, {
                "type": "job_done", "run_id": rid, "worker_id": wc.worker_id,
                "success": success, "elapsed": elapsed, "exit_code": msg.exit_code,
                "xfailed": xfailed, **({"status": status, "reason": reason} if status else {}),
            })
            check = state.controls.get(eid, {}).get("metric_check")
            if success and check and not run.metric_seen:
                await _check_run_metric_locked(db, state, eid, rid, check)
            await _check_experiments_complete_locked(db, state, {eid})
    state.request_schedule()
    # The outputs are safe on the manager; the worker may delete its scratch.
    if synced:
        _send(wc, MsgCleanup(run_id=rid, experiment=eid, attempt=run.attempt, final=True))


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


async def _handle_gpu_health(
    db: aiosqlite.Connection, state: ManagerState, wc: WorkerConn, msg: MsgGpuHealth,
) -> None:
    """A probe finished after the hello, or a failed GPU recovered."""
    async with state.lock:
        _set_health(wc, msg)
        await state.db_writer.update_worker_health(wc.worker_id, json.dumps(wc.unhealthy_gpus))
    print(f"  {_GREEN}GPUS{_RESET}  {wc.host}: {_health_summary(wc)}")
    state.request_schedule()


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
    MsgGpuHealth: _handle_gpu_health,
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
    for key in keys:
        state.pins.pop(key, None)
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
    # A run being stopped is not run again: it ends as what it was stopped for.
    for key in [k for k in keys if (r := state.runs.get(k)) is not None and r.ending is not None]:
        ending = state.runs[key].ending
        assert ending is not None
        await end_runs_locked(db, state, [key], *ending)
    keys = [k for k in keys if k in state.runs]
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
        if exp.optimize is not None:
            continue  # it decides when it's done; see _manager_optimizer
        counts = await count_jobs_by_status(db, eid)
        if not counts:
            # No jobs yet (e.g. a freshly created or un-paused experiment):
            # there is nothing to complete, so leave it running for submissions.
            continue
        if counts.get("pending", 0) + counts.get("dispatched", 0) + counts.get("running", 0):
            continue
        done = counts.get("done", 0) + counts.get("pruned", 0)
        if exp.expected_jobs and done < exp.expected_jobs:
            continue
        await state.db_writer.update_experiment_status(eid, "completed")
        state.broadcast(eid, {"type": "experiment_done", "experiment_id": eid, "submitted_count": done})


# ── Run controls: MAX_RUNTIME, PRUNE, METRIC_CHECK (see _manager_controls) ──────


async def end_runs_locked(
    db: aiosqlite.Connection, state: ManagerState, keys: Sequence[RunKey], status: JobStatus, reason: str,
) -> list[JobRecord]:
    """Finish in-flight runs' jobs as *status* now, without waiting for their results.

    For a stopped run whose result never came, or whose worker was lost.  The
    worker kills the process; its late result is only acknowledged, as for a
    cancel, so its outputs are not synced.  The metrics already stored are kept
    and packed.  ``stop_runs_locked`` is the graceful way.
    """
    for key in keys:
        state.pins.pop(key, None)
    runs = [r for k in keys if (r := state.runs.get(k)) is not None]
    _detach_locked(state, keys)
    ended = await state.db_writer.end_jobs(keys, status, reason)
    for run in runs:
        await state.db_writer.pack_metrics(run.job_key, run.attempt)
    for job in ended:
        print(f"  {_YELLOW}END  {_RESET} {job.experiment_id}/{job.run_id} without its result: {reason}")
        # A pruned run ran fine, so for skip rules it is a success.
        xfailed = await state.db_writer.apply_result_rules(job.experiment_id, job.run_id, status == "pruned")
        state.broadcast(job.experiment_id, {
            "type": "job_done", "run_id": job.run_id, "status": status, "success": False,
            "elapsed": job.elapsed, "exit_code": None, "reason": reason, "xfailed": xfailed,
        })
    await _check_experiments_complete_locked(db, state, {j.experiment_id for j in ended})
    state.request_schedule()
    return ended


# How long a stopped run has between its SIGTERM and SIGKILL, to save a checkpoint.
STOP_GRACE_SECONDS = 30.0
# How long past that the manager waits for its result before ending the job without it.
_STOP_GIVE_UP_SECONDS = STOP_GRACE_SECONDS + 60.0


def stop_runs_locked(
    state: ManagerState, keys: Sequence[RunKey], status: JobStatus, reason: str,
) -> list[InFlightRun]:
    """Stop in-flight runs gracefully, to finish their jobs as *status* ("pruned", or
    "failed" for a timeout) once their results arrive.

    The worker sends SIGTERM, and SIGKILL after ``STOP_GRACE_SECONDS``.  The run
    stays in flight until then, so its outputs are synced, its log and metrics
    completed and its GPUs freed exactly as for a run that exited by itself.
    ``enforce_controls_locked`` ends any whose result never comes.  Returns the
    runs it began stopping (a run already being stopped is left alone).
    """
    stopping = []
    for key in keys:
        run = state.runs.get(key)
        if run is None or run.ending is not None:
            continue
        run.ending = (status, reason)
        run.ending_since = time.time()
        word = "PRUNE" if status == "pruned" else "STOP "
        print(f"  {_YELLOW}{word}{_RESET} {run.experiment_id}/{run.run_id}: {reason}")
        for wid in run.nodes:
            wc = state.workers.get(wid)
            if wc is not None and wc.status != "dead":
                _send(wc, MsgCancel(run_id=run.run_id, experiment=run.experiment_id,
                                    kill_after=STOP_GRACE_SECONDS))
        state.broadcast(run.experiment_id, {"type": "job_updated", "run_id": run.run_id, "reason": reason})
        stopping.append(run)
    return stopping


def _apply_metric_controls(state: ManagerState, run: InFlightRun, step: int, data: dict[str, Any]) -> str | None:
    """Note a metric row against its sweep's controls.  Returns why to prune the run, or None."""
    controls = state.controls.get(run.experiment_id)
    if not controls:
        return None
    check = controls.get("metric_check")
    if check and metric_value(data, check["metric"]) is not None:
        run.metric_seen = True
    tracker: PruneTracker | None = state.prune.get(run.experiment_id)
    if tracker is None:
        return None
    value = metric_value(data, tracker.spec["metric"])
    if value is None:
        return None
    if not tracker.loaded:
        tracker.buffered.append((run.run_id, run.attempt, step, value))
        return None
    return tracker.observe(run.run_id, run.attempt, step, value)


async def _load_prune_tracker_locked(
    db: aiosqlite.Connection, state: ManagerState, experiment_id: str, tracker: PruneTracker,
) -> list[tuple[RunKey, str]]:
    """Fill *tracker* from the stored metrics.  Returns the in-flight runs to prune, with why."""
    stored = await latest_metric_rows(db, experiment_id)
    parsed = await asyncio.to_thread(
        lambda: {rid: (attempt, parse_metric_rows(rows)) for rid, (attempt, rows) in stored.items()})
    in_flight = {r.run_id: r for r in state.runs.values() if r.experiment_id == experiment_id}
    metric = tracker.spec["metric"]
    stop: dict[RunKey, str] = {}
    # Finished runs first, so a running one is judged against all of them.
    for rid in sorted(parsed, key=lambda rid: rid in in_flight):
        attempt, rows = parsed[rid]
        run = in_flight.get(rid)
        live = run is not None and run.attempt == attempt
        for row in rows:
            value = metric_value(row, metric)
            if value is not None:
                why = tracker.observe(rid, run.attempt if live and run else -attempt, row["step"], value,
                                      judge=live)
                if why is not None and run is not None:
                    stop.setdefault(run.key, why)
    for rid, attempt, step, value in tracker.buffered:
        run = in_flight.get(rid)
        why = tracker.observe(rid, attempt, step, value)
        if why is not None and run is not None and run.attempt == attempt:
            stop.setdefault(run.key, why)
    tracker.buffered.clear()
    tracker.loaded = True
    return list(stop.items())


async def enforce_controls_locked(db: aiosqlite.Connection, state: ManagerState) -> None:
    """Reload the run controls, then stop runs past MAX_RUNTIME and prune what is behind."""
    state.controls = {e.experiment_id: json.loads(e.controls) for e in await list_controlled_experiments(db)
                      if e.controls}
    for eid in list(state.prune):
        if state.controls.get(eid, {}).get("prune") != state.prune[eid].spec:
            del state.prune[eid]
    now = time.time()
    for eid, controls in state.controls.items():
        if controls.get("prune") and eid not in state.prune:
            state.prune[eid] = PruneTracker(controls["prune"])
        tracker = state.prune.get(eid)
        if tracker is not None and not tracker.loaded:
            for key, why in await _load_prune_tracker_locked(db, state, eid, tracker):
                stop_runs_locked(state, [key], "pruned", why)
        limit = controls.get("max_runtime")
        if limit:
            late = [j.key for j in await running_jobs_started_before(db, eid, now - limit) if j.key in state.runs]
            stop_runs_locked(state, late, "failed",
                             f"timed out: ran longer than MAX_RUNTIME ({format_duration(limit)})")
    # A stopped run whose result never came (an older worker ignores the SIGKILL
    # deadline, or the process will not die): end its job without it.
    for run in [r for r in state.runs.values() if r.ending and now - r.ending_since > _STOP_GIVE_UP_SECONDS]:
        assert run.ending is not None
        await end_runs_locked(db, state, [run.key], *run.ending)


async def _check_run_metric_locked(
    db: aiosqlite.Connection, state: ManagerState, experiment_id: str, run_id: str, check: dict[str, Any],
) -> None:
    """A run finished without (live) logging its sweep's metric: record that, and pause a sweep
    whose first finished runs all did the same."""
    metric = check["metric"]
    rows = await get_metrics_for_run(db, run_id, experiment_id)
    if any(metric_value(r, metric) is not None for r in rows):
        return  # it was logged, just not while this manager was watching (e.g. before a restart)
    logged = {k for r in rows for k in r}
    if metric in logged:
        reason = f"{MISSING_METRIC_REASON} {metric!r}: every value it logged was NaN or infinite"
    else:
        reason = missing_metric_reason(metric, logged)
    print(f"  {_YELLOW}WARN{_RESET}  {experiment_id}/{run_id}: {reason}")
    await state.db_writer.set_job_reason(run_id, experiment_id, reason)
    state.broadcast(experiment_id, {"type": "job_updated", "run_id": run_id, "reason": reason})
    missing, done = await missing_metric_counts(db, experiment_id)
    if missing >= check["runs"] and missing == done:
        await _pause_for_controls_locked(
            db, state, experiment_id,
            f"paused by METRIC_CHECK: the first {done} finished run(s) logged no {metric!r} "
            f"(last one: {reason.removeprefix(MISSING_METRIC_REASON + ' ' + repr(metric) + ': ')}). "
            "Fix METRIC or the training script; unpause to carry on regardless.",
        )


async def _pause_for_controls_locked(
    db: aiosqlite.Connection, state: ManagerState, experiment_id: str, why: str,
) -> None:
    """Pause a running sweep for a run control, noting why.  The check fires once, so unpausing carries on."""
    exp = await get_experiment(db, experiment_id)
    if exp is None or exp.status != "running":
        return
    controls = json.loads(exp.controls or "{}")
    controls.pop("metric_check", None)
    controls["note"] = why
    await state.db_writer.update_experiment_controls(experiment_id, controls)
    state.controls[experiment_id] = controls
    if exp.optimize is not None:
        await give_up_locked(db, state, exp, json.loads(exp.optimize), why)
        return
    print(f"  {_YELLOW}PAUSE{_RESET} {experiment_id}: {why}")
    await state.db_writer.update_experiment_status(experiment_id, "paused")
    state.broadcast(experiment_id, {"type": "status_updated", "status": "paused", "note": why})


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
            scratch_dir=cfg.get("scratch_dir") or state.scratch_dir,
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

        # Show the retry in the UI immediately instead of the previous
        # session's error; the connect task below replaces this with
        # "connected" or a fresh failure reason.
        await state.db_writer.update_worker_status(
            wr.worker_id, "reconnecting", last_error=None,
        )

        asyncio.create_task(connect_single_worker(
            db, state, **_relaunch_args(state, wr), manager_port=manager_port))
        scheduled += 1

    return scheduled


async def connect_single_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    host: str,
    remote_dir: str,
    *,
    worker_id: str,
    scratch_dir: str = DEFAULT_SCRATCH_DIR,
    password: str | None = None,
    ssh_key: str | None = None,
    venv: str | None = None,
    port: int = 0,
    devices: list[int] | None = None,
    max_jobs_per_gpu: int = 1,
    manager_port: int = 0,
    launch: Callable[..., Any] | None = None,
) -> WorkerConn | None:
    """Launch and connect to a worker, register it, and start its tasks.

    The caller has claimed *worker_id* with ``state.reserve_worker_id``; the
    claim is released here.  Returns ``None`` if the launch failed (the worker
    is then marked dead with the reason, and jobs the database had running on
    it are requeued).  *launch* starts the worker (default ``launch_worker``).
    """
    reader: asyncio.StreamReader | None
    writer: asyncio.StreamWriter | None
    reason = ""
    try:
        # Every worker has a row from its first launch on, so the System page
        # lists one that fails to start (and its retry or update has a target).
        if await get_worker(db, worker_id) is None:
            await state.db_writer.upsert_worker(
                worker_id=worker_id, host=host, remote_dir=remote_dir, scratch_dir=scratch_dir,
                port=port, ssh_key=ssh_key, venv=venv,
                devices=json.dumps(devices) if devices else None, status="offline",
            )
        try:
            reader, writer, actual_port = await (launch or launch_worker)(
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
            wc.tunnel_task = asyncio.create_task(_tunnel_monitor_task(wc, manager_port, state.shutdown_event))

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


def worker_outdated(state: ManagerState, worker_id: str, on_disk: str | None = None) -> bool:
    """Whether *worker_id* runs other worker code than updating it would install.

    True for a worker refused at its hello for speaking another protocol (a
    long-lived worker on a fixed port from an older mlsweep), and for a
    connected remote worker whose fingerprint differs from the worker modules
    on disk here (an older worker sends none).  A local worker runs this
    manager's own interpreter, so a restart of the manager, not an update, is
    what changes its code.  *on_disk* is the fingerprint of the worker code
    on disk, read now if not given.
    """
    if worker_id in state.outdated:
        return True
    wc = state.workers.get(worker_id)
    return (wc is not None and wc.status == "connected" and wc.host != "localhost"
            and wc.code != (on_disk if on_disk is not None else code_fingerprint()))


async def _try_connect(
    host: str, port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    """Connect to *host*:*port* within two seconds, or None if nothing answers."""
    try:
        return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2.0)
    except (OSError, asyncio.TimeoutError):
        return None


async def _wait_until_closed(host: str, port: int, timeout: float = 20.0) -> None:
    """Wait until nothing accepts connections on *host*:*port*, up to *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        conn = await _try_connect(host, port)
        if conn is None:
            return
        _close_writer(conn[1])
        await asyncio.sleep(0.5)


def _relaunch_args(state: ManagerState, record: WorkerRecord) -> dict[str, Any]:
    """``connect_single_worker`` arguments that relaunch a stored worker.

    Settings the database does not keep (the ssh password and jobs per GPU)
    come from the live connection when there is one.
    """
    wc = state.workers.get(record.worker_id)
    return dict(
        host=record.host,
        remote_dir=record.remote_dir,
        worker_id=record.worker_id,
        scratch_dir=record.scratch_dir or state.scratch_dir,
        password=wc.password if wc is not None else None,
        ssh_key=record.ssh_key,
        venv=record.venv,
        port=record.port,
        devices=json.loads(record.devices) if record.devices else None,
        max_jobs_per_gpu=wc.max_jobs_per_gpu if wc is not None else 1,
    )


async def update_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    worker_id: str,
) -> None:
    """Relaunch a retired worker, so it runs this manager's worker code.

    The launch's bootstrap reinstalls ``/tmp/mlsweep_venv`` on a remote host
    whose venv has other code (see ``_bootstrap_worker_venv``).  A local
    worker runs the manager's own interpreter and code on disk.

    The caller has retired the old worker, claimed *worker_id* with
    ``state.reserve_worker_id`` and set its status to ``updating``; the claim
    is released here.  A failure marks the worker dead with the reason.
    """
    handed_off = False  # connect_single_worker releases the claim itself
    try:
        record = await get_worker(db, worker_id)
        if record is None:
            return
        args = _relaunch_args(state, record)
        print(f"  {_CYAN}UPDATE{_RESET} {record.host}: relaunching the worker", flush=True)
        if record.port:
            # The launch reuses a worker it finds on a fixed port, so let the old
            # one finish shutting down first.
            await _wait_until_closed(_bare_host(record.host), record.port)
        state.outdated.discard(worker_id)
        handed_off = True
        await connect_single_worker(db, state, **args, manager_port=state.manager_port)
    finally:
        if not handed_off:
            state.launching.discard(worker_id)


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
                await enforce_controls_locked(db, state)
                await advance_optimizers_locked(db, state)
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
    now = time.time()
    for key in [k for k, p in state.pins.items() if now - p.made > PIN_TTL_SECONDS]:
        del state.pins[key]
    # A pinned job goes to its GPUs even when every GPU is at its cap.
    pending = await list_schedulable_jobs(db, cpu_only=not gpu_free and not state.pins)
    if not pending:
        return
    caps = await experiment_concurrency_caps(db)
    topos = {wc.worker_id: _parse_topo_wire(wc.topo) for wc in connected}
    running: dict[str, int] = {}
    on_worker = {wc.worker_id: 0 for wc in connected}
    for run in state.runs.values():
        running[run.experiment_id] = running.get(run.experiment_id, 0) + 1
        for wid in run.nodes:
            if wid in on_worker:
                on_worker[wid] += 1

    # Occupancy only grows during a pass, so a shape that did not fit never will.
    no_fit: set[tuple[int, int]] = set()
    for job in pending:
        cap = caps.get(job.experiment_id, 0)
        if cap and running.get(job.experiment_id, 0) >= cap:
            continue
        pin = state.pins.get((job.experiment_id, job.run_id))
        placements = _pinned_placement(job, pin, state) if pin is not None else None
        if pin is not None and placements is None:
            # The worker left or the GPUs left its pool: place it as usual.
            del state.pins[(job.experiment_id, job.run_id)]
        if placements is not None:
            if not await _dispatch_locked(state, job, placements):
                continue
            del state.pins[(job.experiment_id, job.run_id)]
            _book(placements, occupancy, on_worker)
            running[job.experiment_id] = running.get(job.experiment_id, 0) + 1
            continue
        shape = (job.gpus_per_run, job.nodes_per_run)
        if shape in no_fit:
            continue
        placements = _place(job, connected, occupancy, on_worker, topos, state.placement)
        if placements is None:
            no_fit.add(shape)
            continue
        if not await _dispatch_locked(state, job, placements):
            continue
        _book(placements, occupancy, on_worker)
        running[job.experiment_id] = running.get(job.experiment_id, 0) + 1


def _pinned_placement(
    job: SchedulableJob, pin: Pin, state: ManagerState,
) -> list[tuple[WorkerConn, list[int]]] | None:
    """*pin* as a placement for *job*, or ``None`` once it can no longer be used.

    A pin ignores ``max_jobs_per_gpu``: whoever chose the GPU chose to share it.
    """
    wc = state.workers.get(pin.worker_id)
    if wc is None or wc.status != "connected" or job.nodes_per_run > 1:
        return None
    if len(pin.gpus) != job.gpus_per_run or not set(pin.gpus) <= set(wc.usable_gpus):
        return None
    return [(wc, list(pin.gpus))]


def _book(
    placements: list[tuple[WorkerConn, list[int]]],
    occupancy: dict[str, dict[int, int]],
    on_worker: dict[str, int],
) -> None:
    """Count a dispatched job's *placements* in the pass's occupancy tallies."""
    for wc, gpus in placements:
        on_worker[wc.worker_id] += 1
        for g in gpus:
            occupancy[wc.worker_id][g] += 1


def _place(
    job: SchedulableJob,
    workers: list[WorkerConn],
    occupancy: dict[str, dict[int, int]],
    on_worker: dict[str, int],
    topos: dict[str, dict[tuple[int, int], int]],
    placement: Placement,
) -> list[tuple[WorkerConn, list[int]]] | None:
    """Choose a worker and GPUs for each node of *job*, or ``None`` if it does not fit.

    Each worker offers its best GPU group.  The offers are ranked by how busy
    their GPUs are, then by how busy the whole worker is.  "spread" takes the
    least busy and "pack" the busiest.  Ties keep connection order.
    """
    spread = placement == "spread"
    offers: list[tuple[tuple[int, float], WorkerConn, list[int]]] = []
    for wc in workers:
        occ = occupancy[wc.worker_id]
        gpus = _find_gpu_group(wc, job.gpus_per_run, topos[wc.worker_id],
                               occupancy=occ, spread=spread)
        if gpus is None:
            continue
        gpu_load = max((occ[g] for g in gpus), default=0)
        worker_load = on_worker[wc.worker_id] / max(1, len(wc.usable_gpus))
        offers.append(((gpu_load, worker_load), wc, gpus))
    n_nodes = max(1, job.nodes_per_run)
    if len(offers) < n_nodes:
        return None
    # Sorting is stable, also when reversed, so ties keep connection order.
    offers.sort(key=lambda o: o[0], reverse=not spread)
    return [(wc, gpus) for _, wc, gpus in offers[:n_nodes]]


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
        # A snapshot's URL gives its manifest; the worker fetches the objects
        # it lacks from the same manager.  A legacy ID's gives the tarball.
        kind = "snapshots" if is_snapshot_id(claimed.artifact_id) else "artifacts"
        artifact_url = f"{state.artifact_base_url}/api/{kind}/{claimed.artifact_id}"
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
            attempt=claimed.attempt,
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
    spread: bool,
) -> list[int] | None:
    """Find *gpus_needed* GPUs on *wc* with room for another job, or ``None``.

    Per-GPU packing is bounded by the worker's ``max_jobs_per_gpu`` (0 =
    unlimited).  Takes the least busy GPUs when *spread*, otherwise the
    busiest, and among GPUs equally suitable prefers topologically close ones.
    CPU-only jobs (``gpus_needed == 0``) get ``[]``.
    """
    if gpus_needed == 0:
        return []
    cap = wc.max_jobs_per_gpu
    free = [g for g in wc.usable_gpus if cap <= 0 or occupancy[g] < cap]
    if len(free) < gpus_needed:
        return None
    # Sorting is stable, so GPUs at the same load stay in device order.
    free.sort(key=lambda g: occupancy[g], reverse=not spread)
    # Go only as far past the best load level as the group needs.
    level = occupancy[free[gpus_needed - 1]]
    pool = free[:gpus_needed] + [g for g in free[gpus_needed:] if occupancy[g] == level]
    return closest_gpu_group(pool, gpus_needed, topo)


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


# How long a relaunched tunnel must stay up before it counts as established.
# Longer than ssh's ConnectTimeout so a failed connect or forward is caught.
_TUNNEL_SETTLE_SECS = 15.0

# Where each tunnel records the remote sshd process holding its forward, as
# ``<port>.<sshd pid>``.  The file holds this manager's identity on line 1 and
# the sshd's start time on line 2 (so a reused pid is never mistaken for it).
_TUNNEL_RECORD_DIR = "$HOME/.mlsweep/tunnels"


def _read_id_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _manager_identity() -> tuple[str, str]:
    """``(machine, instance)`` naming this host and this manager process.

    The instance includes the boot id, so a manager from before a reboot never
    matches a new one that happens to get the same pid.
    """
    machine = _read_id_file("/etc/machine-id") or socket.gethostname()
    boot = _read_id_file("/proc/sys/kernel/random/boot_id") or "noboot"
    return machine, f"{boot}:{os.getpid()}"


def _tunnel_remote_cmd(manager_port: int) -> str:
    """Remote command that records the forward's sshd pid, then idles on stdin.

    ``$PPID`` of the exec'd ``sh`` is the per-connection sshd process, which
    owns the ``-R`` listener.  The record lets a later manager kill that sshd
    if this one dies without closing the connection (crash, reboot, network
    loss), since the remote sshd otherwise holds the port for hours.  ``cat``
    ends when the connection closes, and the record is removed with it.
    """
    machine, instance = _manager_identity()
    tag = shlex.quote(f"{machine} {instance} {socket.gethostname()}")
    script = (
        f'f="{_TUNNEL_RECORD_DIR}/{manager_port}.$PPID"; '
        f'mkdir -p "${{f%/*}}" && {{ echo {tag}; echo $(ps -o lstart= -p $PPID); }} > "$f"; '
        'cat; rm -f "$f"'
    )
    return f"exec sh -c {shlex.quote(script)}"


def _reap_tunnels_script(manager_port: int) -> str:
    """Remote script that kills sshd sessions left by earlier managers on this machine.

    Prints ``killed <pid>`` for each one killed, ``held <hostname>`` for a
    live tunnel recorded by another machine's manager and ``own <pid>`` for
    one of this manager's own; both of those are left alone.  Records of
    exited processes are removed.
    """
    machine, instance = _manager_identity()
    return f"""
cd "{_TUNNEL_RECORD_DIR}" 2>/dev/null || exit 0
for f in {manager_port}.*; do
  [ -e "$f" ] || continue
  pid=${{f#{manager_port}.}}
  {{ read m i h; read st; }} < "$f"
  if [ "$(echo $(ps -o lstart= -p "$pid" 2>/dev/null))" != "$st" ]; then rm -f "$f"; continue; fi
  case "$(ps -o comm= -p "$pid")" in sshd*) ;; *) rm -f "$f"; continue ;; esac
  if [ "$m" = {shlex.quote(machine)} ]; then
    [ "$i" = {shlex.quote(instance)} ] && {{ echo "own $pid"; continue; }}
    kill "$pid" && rm -f "$f" && echo "killed $pid"
  else
    echo "held $h"
  fi
done
"""


async def _reap_stale_tunnels(wc: WorkerConn, manager_port: int) -> str:
    """Free *manager_port* on the worker host from a dead earlier manager's tunnel.

    Returns a short note for the log describing what was found.
    """
    key_args = ["-i", wc.ssh_key] if wc.ssh_key else []
    try:
        sshpass_args, sshpass_env = _sshpass_args(wc.password)
        proc = await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh", "-o", "ConnectTimeout=10", *key_args, "--", wc.host,
            f"exec sh -c {shlex.quote(_reap_tunnels_script(manager_port))}",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=sshpass_env,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (OSError, RuntimeError, asyncio.TimeoutError) as e:
        return f"could not check for stale tunnels: {e}"
    lines = out.decode(errors="replace").split()
    found: dict[str, list[str]] = {"killed": [], "held": [], "own": []}
    for word, value in zip(lines, lines[1:]):
        if word in found:
            found[word].append(value)
    killed, holders, own = found["killed"], found["held"], found["own"]
    if killed:
        return f"killed stale sshd session(s) {', '.join(killed)} from an earlier manager"
    if holders:
        return f"port {manager_port} is used by the manager on {', '.join(sorted(set(holders)))}"
    if own:
        return (f"port {manager_port} is held by another tunnel of this manager "
                f"(sshd {', '.join(own)} on the worker host)")
    return (
        f"port {manager_port} is held by an unrecorded process, likely a stale ssh session "
        "from an older mlsweep; it frees within ~2h, or kill it on the worker host"
    )


async def _launch_tunnel(
    host: str,
    manager_port: int,
    ssh_key: str | None = None,
    password: str | None = None,
) -> "asyncio.subprocess.Process | None":
    """Spawn ssh -R {port}:localhost:{port} so the worker can reach the manager's HTTP server."""
    key_args = ["-i", ssh_key] if ssh_key else []
    sshpass_args, sshpass_env = _sshpass_args(password)
    try:
        return await asyncio.create_subprocess_exec(
            *sshpass_args,
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"{manager_port}:localhost:{manager_port}",
            *key_args,
            "--",
            host,
            _tunnel_remote_cmd(manager_port),
            # Never written; the remote ``cat`` reads it until the connection ends.
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=sshpass_env,
            preexec_fn=_die_with_parent,
        )
    except OSError as e:
        print(f"  {_YELLOW}WARN{_RESET}  Could not start tunnel to {host}: {e}", file=sys.stderr)
        return None


async def _tunnel_exit_reason(proc: "asyncio.subprocess.Process") -> str:
    """Last line ssh wrote to stderr before exiting, or its exit code."""
    err = b""
    if proc.stderr is not None:
        try:
            err = await asyncio.wait_for(proc.stderr.read(), timeout=1.0)
        except (asyncio.TimeoutError, OSError):
            pass
    lines = [ln.strip() for ln in err.decode(errors="replace").splitlines() if ln.strip()]
    return lines[-1] if lines else f"ssh exited with code {proc.returncode}"


def _terminate_launched(launch: "asyncio.Future[Any]") -> None:
    """End a tunnel whose monitor was cancelled while it was being spawned."""
    _terminate_proc(None if launch.cancelled() or launch.exception() else launch.result())


async def _tunnel_monitor_task(
    wc: WorkerConn,
    manager_port: int,
    shutdown_event: asyncio.Event,
    launch: Callable[..., Any] | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> None:
    """Keep the SSH reverse tunnel alive, restarting with backoff if it dies.

    A tunnel only counts as established once ssh has stayed up past its
    connect timeout.  ssh with ExitOnForwardFailure exits within seconds when
    the remote port is taken, typically by a stale sshd session left over from
    a manager that died without closing its connection.  Those are reaped via
    the records written by ``_tunnel_remote_cmd`` before retrying.

    *launch* starts a tunnel (default ``_launch_tunnel``), and *sleep* waits out
    the backoff between attempts.

    Retiring the worker cancels this task (see ``_close_tunnel``).
    """
    backoff = 2.0
    healthy = False  # becomes True once a tunnel survives the settle window
    failed = False   # a failure was reported, so announce the recovery
    while True:
        proc = wc.tunnel_proc
        if proc is None or shutdown_event.is_set():
            return

        wait_proc = asyncio.create_task(proc.wait())
        wait_shut = asyncio.create_task(shutdown_event.wait())
        done, pending = await asyncio.wait(
            {wait_proc, wait_shut},
            timeout=None if healthy else _TUNNEL_SETTLE_SECS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if shutdown_event.is_set():
            if proc.returncode is None:
                proc.terminate()
            return

        if not done:
            # Survived the settle window: the forward is up.
            healthy = True
            backoff = 2.0
            if failed:
                print(f"  {_GREEN}OK{_RESET}    SSH tunnel to {wc.host} re-established")
                failed = False
            continue

        reason = await _tunnel_exit_reason(proc)
        if "remote port forwarding failed" in reason:
            reason = await _reap_stale_tunnels(wc, manager_port)
        verb = "lost" if healthy else "failed"
        print(
            f"  {_YELLOW}WARN{_RESET}  SSH tunnel to {wc.host} {verb} ({reason}); "
            f"retrying in {backoff:.0f}s"
        )
        healthy = False
        failed = True
        await sleep(backoff)
        backoff = min(backoff * 2, 60.0)
        if shutdown_event.is_set():
            return

        # Shielded: a cancel mid-spawn would otherwise lose the new ssh process.
        launching = asyncio.ensure_future((launch or _launch_tunnel)(
            wc.host, manager_port, ssh_key=wc.ssh_key, password=wc.password))
        try:
            new_proc = await asyncio.shield(launching)
        except asyncio.CancelledError:
            launching.add_done_callback(_terminate_launched)
            raise
        if new_proc is not None:
            wc.tunnel_proc = new_proc
