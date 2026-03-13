#!/usr/bin/env python3

"""Run experiment sweeps. See docs/sweep_configuration.md for format and usage."""

import argparse
import dataclasses
import importlib.metadata
import json
import os
import queue
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, IO

from mlsweep._writers import MlsweepWriterFactory, MultiWriterFactory, RunWriter, WriterFactory

from mlsweep._parsync import parsync_bin
from mlsweep._shared import (
    MsgCleanup,
    MsgHello,
    MsgReplay,
    MsgRun,
    MsgShutdown,
    MsgWorkerHello,
    MsgStarted,
    MsgLog,
    MsgMetric,
    MsgSyncReq,
    MsgResult,
    MsgCleaned,
    decode,
    encode,
)
from mlsweep._sweep import (
    _append_manifest_run,
    _load_sweep_status,
    _singular_desc,
    _treatment_key,
    _update_sweep_status,
    _write_manifest,
    count_expected,
    extract_objective_metric,
    generate_variations,
    load_sweep_file,
    should_skip,
    validate_options,
)
from mlsweep._topology import _best_gpu_groups, _visible_devices
from mlsweep._shared import _git_root

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_PROJECT_ROOT = _git_root(os.getcwd()) or os.getcwd()

# ANSI colors
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"
_DIM_COLORS = [_CYAN, _YELLOW, _MAGENTA, _BLUE]

_log_file: IO[str] | None = None


def sweep_print(msg: str, end: str = "\n") -> None:
    """Print to stdout (colored) and log file (plain)."""
    print(msg, end=end, flush=True)
    if _log_file is not None:
        _log_file.write(re.sub(r"\033\[[0-9;]*m", "", msg) + end)
        _log_file.flush()


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.0f}m"
    return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"


# ── Worker connection state ────────────────────────────────────────────────────


@dataclasses.dataclass
class WorkerState:
    worker_id: int
    host: str                           # "localhost" for local
    port: int
    remote_dir: str
    scratch_dir: str                    # from MsgWorkerHello
    gpus: list[int]
    topo: dict[tuple[int, int], int]   # internal format {(a,b): score}
    slots: list[list[int]]             # GPU groups computed from topo
    busy_slots: set[int]               # indices into slots currently in use
    send_queue: "queue.Queue[bytes | None]"
    status: str                        # CONNECTING | CONNECTED | RECONNECTING | DEAD
    in_flight: dict[str, dict[str, Any]]  # run_id → var dict
    run_slots: dict[str, int] = dataclasses.field(default_factory=dict)  # run_id → slot_idx
    jobs_per_slot: int = 1
    password: str | None = None
    ssh_key: str | None = None


def _parse_workers(
    path: str,
) -> list[tuple[str, str, int | None, int | None, list[int] | None, str | None, str | None, str | None]]:
    """Parse a TOML workers file.

    Each [[workers]] entry requires 'host' and 'remote_dir'. Optional fields:
      gpus     (int)       total GPU count to use (default: all visible)
      jobs     (int)       concurrent jobs per GPU slot (default: 1)
      devices  (list[int]) specific GPU device IDs to use
      pass     (str)       SSH password; falls back to MLSWEEP_SSH_PASS env var
      ssh_key  (str)       path to SSH identity file (passed as -i)
      venv     (str)       path to venv (or project root); activate is sourced before running

    Returns [(host, remote_dir, gpus, jobs, devices, password, ssh_key, venv)].
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    global_pass = os.environ.get("MLSWEEP_SSH_PASS")
    result = []
    for i, entry in enumerate(data.get("workers", [])):
        host = entry.get("host")
        remote_dir = entry.get("remote_dir")
        if not host or not remote_dir:
            raise ValueError(f"{path}: workers entry {i + 1} missing required field 'host' or 'remote_dir'")
        gpus: int | None = entry.get("gpus")
        jobs: int | None = entry.get("jobs")
        devices: list[int] | None = entry.get("devices")
        password: str | None = entry.get("pass") or global_pass
        ssh_key: str | None = entry.get("ssh_key")
        venv: str | None = entry.get("venv") or remote_dir
        result.append((host, remote_dir, gpus, jobs, devices, password, ssh_key, venv))
    return result


# ── Event types (controller-internal) ─────────────────────────────────────────


@dataclasses.dataclass
class EvWorkerConnected:
    worker_id: int
    gpus: list[int]
    topo: dict[str, int]
    resuming: list[dict[str, Any]]
    scratch_dir: str


@dataclasses.dataclass
class EvWorkerDisconnected:
    worker_id: int


@dataclasses.dataclass
class EvRunStarted:
    worker_id: int
    run_id: str
    pid: int


@dataclasses.dataclass
class EvLogLine:
    run_id: str
    seq: int
    data: str


@dataclasses.dataclass
class EvMetricLine:
    run_id: str
    step: int
    data: dict[str, Any]


@dataclasses.dataclass
class EvSyncRequest:
    run_id: str
    worker_id: int


@dataclasses.dataclass
class EvRunResult:
    worker_id: int
    run_id: str
    success: bool
    elapsed: float
    exit_code: int


@dataclasses.dataclass
class EvArtifactSynced:
    run_id: str


@dataclasses.dataclass
class EvWorkerCleaned:
    run_id: str


@dataclasses.dataclass
class EvReconnectWorker:
    worker_id: int
    success: bool
    resuming: list[dict[str, Any]]


@dataclasses.dataclass
class EvInteractiveCommand:
    cmd: str
    args: list[str]


# ── Worker write thread ────────────────────────────────────────────────────────


def _worker_write_thread(ws: WorkerState) -> None:
    """Drain the worker's send_queue and write to socket."""
    while True:
        item = ws.send_queue.get()
        if item is None:
            break
        try:
            ws.sock.sendall(item)  # type: ignore[attr-defined]
        except OSError:
            break


# ── Worker read thread ─────────────────────────────────────────────────────────


class _LineReader:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b""

    def readline(self) -> bytes | None:
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


def _worker_read_thread(
    ws: WorkerState,
    sock: socket.socket,
    event_queue: "queue.Queue[Any]",
) -> None:
    """Read messages from worker and post events to the controller's event queue."""
    reader = _LineReader(sock)

    # First message: MsgWorkerHello
    line = reader.readline()
    if not line:
        event_queue.put(EvWorkerDisconnected(ws.worker_id))
        return
    try:
        msg = decode(line)
    except (ValueError, json.JSONDecodeError, TypeError):
        event_queue.put(EvWorkerDisconnected(ws.worker_id))
        return
    if not isinstance(msg, MsgWorkerHello):
        event_queue.put(EvWorkerDisconnected(ws.worker_id))
        return

    event_queue.put(EvWorkerConnected(
        worker_id=ws.worker_id,
        gpus=msg.gpus,
        topo=msg.topo,
        resuming=msg.resuming,
        scratch_dir=msg.scratch_dir,
    ))

    # Main loop
    while True:
        line = reader.readline()
        if not line:
            break
        try:
            msg = decode(line)
        except (ValueError, json.JSONDecodeError, TypeError):
            continue

        if isinstance(msg, MsgStarted):
            event_queue.put(EvRunStarted(
                worker_id=ws.worker_id, run_id=msg.run_id, pid=msg.pid
            ))
        elif isinstance(msg, MsgLog):
            event_queue.put(EvLogLine(run_id=msg.run_id, seq=msg.seq, data=msg.data))
        elif isinstance(msg, MsgMetric):
            event_queue.put(EvMetricLine(
                run_id=msg.run_id, step=msg.step, data=msg.data
            ))
        elif isinstance(msg, MsgSyncReq):
            event_queue.put(EvSyncRequest(run_id=msg.run_id, worker_id=ws.worker_id))
        elif isinstance(msg, MsgResult):
            event_queue.put(EvRunResult(
                worker_id=ws.worker_id,
                run_id=msg.run_id,
                success=msg.success,
                elapsed=msg.elapsed,
                exit_code=msg.exit_code,
            ))
        elif isinstance(msg, MsgCleaned):
            event_queue.put(EvWorkerCleaned(run_id=msg.run_id))

    event_queue.put(EvWorkerDisconnected(ws.worker_id))


# ── Reconnect thread ───────────────────────────────────────────────────────────


def _reconnect_thread(
    ws: WorkerState,
    event_queue: "queue.Queue[Any]",
    token: str,
    max_attempts: int = 10,
) -> None:
    """Try to reconnect to a worker after disconnect; post EvReconnectWorker."""
    backoff = 1.0
    for _ in range(max_attempts):
        time.sleep(backoff)
        backoff = min(backoff * 2, 30.0)
        try:
            sock = socket.create_connection((ws.host, ws.port), timeout=5)
            sock.sendall(encode(MsgHello(token=token, controller_id="controller")))
            reader = _LineReader(sock)
            line = reader.readline()
            if not line:
                sock.close()
                continue
            msg = decode(line)
            if not isinstance(msg, MsgWorkerHello):
                sock.close()
                continue
            # Reconnected — attach new socket to WorkerState
            ws.sock = sock  # type: ignore[attr-defined]
            ws.send_queue = queue.Queue()
            ws.status = "CONNECTED"
            event_queue.put(EvReconnectWorker(
                worker_id=ws.worker_id, success=True, resuming=msg.resuming
            ))
            # Restart read/write threads
            threading.Thread(
                target=_worker_write_thread, args=(ws,), daemon=True
            ).start()
            threading.Thread(
                target=_worker_read_thread, args=(ws, sock, event_queue), daemon=True
            ).start()
            return
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    ws.status = "DEAD"
    event_queue.put(EvReconnectWorker(worker_id=ws.worker_id, success=False, resuming=[]))


# ── Sync thread ────────────────────────────────────────────────────────────────


def _rsync_thread(
    worker_host: str,
    remote_scratch: str,
    local_run_dir: str,
    run_id: str,
    event_queue: "queue.Queue[Any]",
    password: str | None = None,
    ssh_key: str | None = None,
) -> None:
    """Sync artifacts from worker scratch to local output dir."""
    if worker_host == "localhost":
        # Local worker: copy artifacts dir if scratch != output
        src = os.path.join(remote_scratch, "artifacts")
        dst = os.path.join(local_run_dir, "artifacts")
        if src != dst and os.path.isdir(src):
            try:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            except OSError as e:
                sweep_print(f"  {_YELLOW}WARN{_RESET}  parsync failed for {run_id}: {e}")
    else:
        # parsync authenticates via SSH agent, ~/.ssh/config, or PARSYNC_SSH_PASSWORD.
        # ssh_key must be registered with the agent or listed in ~/.ssh/config.
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
            sweep_print(
                f"  {_YELLOW}WARN{_RESET}  parsync failed for {run_id}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
    event_queue.put(EvArtifactSynced(run_id=run_id))


# ── SSH helpers ────────────────────────────────────────────────────────────────

_sshpass_available: bool | None = None


def _worker_candidates(venv: str | None) -> list[str]:
    """Return candidate mlsweep_worker binary paths to try, given a venv specifier.

    Interprets the venv value and produces an ordered list; the caller tries
    each in sequence and execs the first one that exists and is executable.
    Falls back to mlsweep_worker on PATH if nothing else matches.
    """
    candidates: list[str] = []
    if venv:
        p = venv.rstrip("/")
        bn = os.path.basename(p)
        if bn == "mlsweep_worker":
            candidates.append(p)
        elif bn in ("python", "python3", "activate"):
            # Points at a file inside bin/ — use the same bin/
            candidates.append(os.path.join(os.path.dirname(p), "mlsweep_worker"))
        elif bn == "bin":
            # Points at a bin/ directory directly
            candidates.append(os.path.join(p, "mlsweep_worker"))
        else:
            # venv root or project root — try common layouts in order
            candidates += [
                os.path.join(p, "bin", "mlsweep_worker"),          # venv root
                os.path.join(p, ".venv", "bin", "mlsweep_worker"),  # project/.venv
                os.path.join(p, "venv", "bin", "mlsweep_worker"),   # project/venv
            ]
    candidates.append("mlsweep_worker")  # last resort: whatever is on PATH
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


def _sshpass_prefix(password: str | None) -> list[str]:
    """Return ['sshpass', '-p', password] if a password is given, else []."""
    global _sshpass_available
    if not password:
        return []
    if _sshpass_available is None:
        _sshpass_available = shutil.which("sshpass") is not None
    if not _sshpass_available:
        raise RuntimeError("sshpass is not installed but a password was specified")
    return ["sshpass", "-p", password]




# ── Stdin thread ───────────────────────────────────────────────────────────────


def _stdin_thread(event_queue: "queue.Queue[Any]") -> None:
    """Read interactive commands from stdin and post them as events."""
    try:
        for line in sys.stdin:
            parts = line.strip().split()
            if parts:
                event_queue.put(EvInteractiveCommand(cmd=parts[0], args=parts[1:]))
    except (OSError, EOFError):
        pass


# ── Worker connection setup ────────────────────────────────────────────────────


def _start_worker(
    host: str,
    remote_dir: str,
    token: str,
    scratch_dir: str,
    devices: list[int] | None = None,
    password: str | None = None,
    ssh_key: str | None = None,
    venv: str | None = None,
) -> tuple[socket.socket, int]:
    """SSH to start (or re-use) a worker daemon and return (socket, port).

    For local mode, host should be "localhost" and we use subprocess directly.
    """
    devices_args = ["--devices", ",".join(str(d) for d in devices)] if devices else []
    key_args = ["-i", ssh_key] if ssh_key else []
    if host == "localhost":
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "mlsweep.worker",
                "--token", token,
                "--remote-dir", remote_dir,
                "--scratch-dir", scratch_dir,
                *devices_args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        worker_args = [
            "--token", token,
            "--remote-dir", remote_dir,
            *devices_args,
        ]
        shell_cmd = _worker_shell_cmd(_worker_candidates(venv), worker_args)
        proc = subprocess.Popen(
            [
                *_sshpass_prefix(password),
                "ssh", "-o", "ConnectTimeout=10", *key_args,
                host, shell_cmd,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    assert proc.stdout is not None and proc.stderr is not None
    line = proc.stdout.readline().strip()
    if not line.startswith("PORT="):
        stderr_out = proc.stderr.read().strip()
        proc.wait()
        # Diagnose common SSH/auth failures
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
        raise RuntimeError(f"worker failed to start: {last_line}{hint}")
    port = int(line.split("=")[1])

    connect_host = "localhost" if host == "localhost" else host
    sock = socket.create_connection((connect_host, port), timeout=10)
    return sock, port


def _connect_workers(
    workers_file: str | None,
    gpus_per_run: int,
    token: str,
    event_queue: "queue.Queue[Any]",
    scratch_dir: str = "/tmp/mlsweep",
    max_gpus: int | None = None,
    jobs_per_slot: int = 1,
) -> list[WorkerState]:
    """Start worker daemons and set up connections.

    Returns a list of WorkerState objects (status=CONNECTING until
    EvWorkerConnected event is received from the read thread).
    """
    if workers_file:
        worker_configs = _parse_workers(workers_file)
    else:
        # Local mode
        worker_configs = [("localhost", _PROJECT_ROOT, max_gpus, jobs_per_slot, None, None, None, None)]

    workers: list[WorkerState] = []

    for worker_id, (host, remote_dir, w_gpus, w_jobs, w_devices, w_pass, w_key, w_venv) in enumerate(worker_configs):
        try:
            sock, port = _start_worker(host, remote_dir, token, scratch_dir, w_devices, w_pass, w_key, w_venv)
        except Exception as e:
            sweep_print(f"  {_RED}WARN{_RESET}  Cannot start worker on {host}: {e}")
            continue

        # Send MsgHello — worker will respond with MsgWorkerHello (handled by read thread)
        ws = WorkerState(
            worker_id=worker_id,
            host=host,
            port=port,
            remote_dir=remote_dir,
            scratch_dir=scratch_dir,
            gpus=[],
            topo={},
            slots=[],
            busy_slots=set(),
            send_queue=queue.Queue(),
            status="CONNECTING",
            in_flight={},
            jobs_per_slot=w_jobs if w_jobs is not None else jobs_per_slot,
            password=w_pass,
            ssh_key=w_key,
        )
        ws.sock = sock  # type: ignore[attr-defined]

        sock.sendall(encode(MsgHello(token=token, controller_id="controller")))

        # Start read/write threads
        threading.Thread(
            target=_worker_write_thread, args=(ws,), daemon=True
        ).start()
        threading.Thread(
            target=_worker_read_thread, args=(ws, sock, event_queue), daemon=True
        ).start()

        workers.append(ws)

    return workers


# ── Summary ────────────────────────────────────────────────────────────────────


def print_summary(
    results: list[tuple[Any, ...]],
    skipped: int,
    elapsed: float,
) -> bool:
    """Print per-treatment summary. Returns True if any treatment failed."""
    has_singular = any(
        o.get("singular")
        for var, _, _, _ in results
        for o in var["effective_options"].values()
    )

    # Group by treatment
    treatments: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for var, ok, el, log in results:
        tk = _treatment_key(var["combo"], var["effective_options"])
        treatments.setdefault(tk, []).append((var, ok, el, log))

    ok_count = sum(1 for runs in treatments.values() if any(s for _, s, _, _ in runs))
    total_treatments = len(treatments)
    total_runs = len(results)
    probe_fails = sum(
        sum(1 for _, s, _, _ in runs if not s)
        for runs in treatments.values() if any(s for _, s, _, _ in runs)
    ) if has_singular else 0

    parts = [f"{total_runs} runs"]
    if probe_fails:
        parts.append(f"{probe_fails} probe failures")
    if skipped:
        parts.append(f"{skipped} skipped")

    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"SUMMARY — {ok_count}/{total_treatments} OK in {fmt_time(elapsed)}"
                f" ({', '.join(parts)})")
    sweep_print(f"{'=' * 80}")

    for tk in sorted(treatments, key=lambda t: tuple('' if v is None else str(v) for v in t)):
        runs = treatments[tk]
        successes = [(v, e, lf) for v, s, e, lf in runs if s]
        if successes:
            best_var, best_el, best_log = min(successes, key=lambda x: x[1])
            sdesc = _singular_desc(best_var["combo"], best_var["effective_options"]) if has_singular else ""
            n_probes = sum(1 for _, s, _, _ in runs if not s)
            pnote = f", {n_probes} probes" if n_probes else ""
            sweep_print(f"  {_GREEN}   OK{_RESET}  {best_var['name']:<40}"
                        f" {sdesc:<20} ({best_el:.1f}s{pnote})")
            sweep_print(f"         {_BLUE}{best_log}{_RESET}")
        else:
            last_var, _, _, last_log = runs[-1]
            sweep_print(f"  {_RED} FAIL{_RESET}  {last_var['name']:<40}"
                        f" (all {len(runs)} combos failed)")
            sweep_print(f"         {_BLUE}{last_log}{_RESET}")

    return ok_count < total_treatments


# ── Scheduling helpers ─────────────────────────────────────────────────────────


def _dispatch_pending(
    workers: list[WorkerState],
    pending: list[dict[str, Any]],
    in_flight: dict[str, dict[str, Any]],
    failed: list[dict[str, Any]],
    succeeded: list[dict[str, Any]],
    output_dir: str,
    experiment: str,
    exp_dir: str,
    token: str,
    command: list[str],
    extra: list[str],
    run_from: str | None,
    gpus_per_run: int,
    has_singular: bool,
    wandb_env: dict[str, str],
    nodes_per_run: int = 1,
    multinode_state: dict[str, Any] | None = None,
    next_port: list[int] | None = None,
    set_dist_env: bool = False,
) -> list[dict[str, Any]]:
    """Try to assign pending variations to free slots. Returns still-pending list."""
    _MASTER_PORT_BASE = 29500
    if multinode_state is None:
        multinode_state = {}
    if next_port is None:
        next_port = [0]

    # Build set of treatment keys currently in-flight (for deferral)
    inflight_treatments: set[tuple[Any, ...]] = set()
    for run_id, var in in_flight.items():
        tk = _treatment_key(var["combo"], var["effective_options"])
        inflight_treatments.add(tk)

    deferred = []
    remaining = list(pending)

    for var in remaining:
        opts = var["effective_options"]
        tk = _treatment_key(var["combo"], opts)

        # Skip if monotonic/singular logic says to skip
        if should_skip(var["combo"], failed, succeeded, opts):
            # Don't add to deferred — permanently skip
            continue

        # Defer if same treatment is in-flight
        if tk in inflight_treatments:
            deferred.append(var)
            continue

        if nodes_per_run == 1:
            # Find a free slot on any CONNECTED worker
            slot_found = False
            for ws in workers:
                if ws.status != "CONNECTED":
                    continue
                for slot_idx, gpu_group in enumerate(ws.slots):
                    if slot_idx not in ws.busy_slots:
                        # Dispatch this run
                        ws.busy_slots.add(slot_idx)
                        run_id = var["name"]
                        in_flight[run_id] = var

                        run_dir = os.path.join(output_dir, experiment, run_id)
                        os.makedirs(run_dir, exist_ok=True)

                        run_scratch = os.path.join(ws.scratch_dir, experiment, run_id)
                        tag_str = ",".join(f"{k}={v}" for k, v in var["combo"].items() if v is not None)
                        env: dict[str, str] = {}
                        if tag_str:
                            env["EXP_TAGS"] = tag_str
                        env.update(wandb_env)

                        run_msg = MsgRun(
                            run_id=run_id,
                            experiment=experiment,
                            command=list(command) + var["overrides"] + list(extra),
                            env=env,
                            gpu_ids=gpu_group,
                            remote_dir=ws.remote_dir,
                            scratch=run_scratch,
                            run_from=run_from,
                            set_dist_env=set_dist_env,
                        )
                        ws.send_queue.put(encode(run_msg))
                        ws.in_flight[run_id] = var
                        ws.run_slots[run_id] = slot_idx

                        _append_manifest_run(exp_dir, var)

                        # Print dispatch line
                        nm = run_id.ljust(max(len(v["name"]) for v in remaining + deferred + list(in_flight.values())) if remaining or deferred or in_flight else len(run_id))
                        sdesc = _singular_desc(var["combo"], opts) if has_singular else ""
                        gpu_label = f"gpu{'s' if len(gpu_group) > 1 else ''} {','.join(str(g) for g in gpu_group)}"
                        loc = (f"{_MAGENTA}{ws.host}{_RESET} {gpu_label}"
                               if ws.host != "localhost" else gpu_label)
                        log_path = os.path.join(run_dir, "training.log")
                        if sdesc:
                            sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} ({sdesc}) {loc} {_BLUE}{log_path}{_RESET}")
                        else:
                            sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} {loc} {_BLUE}{log_path}{_RESET}")

                        inflight_treatments.add(tk)
                        slot_found = True
                        break
                if slot_found:
                    break

            if not slot_found:
                deferred.append(var)

        else:
            # Multi-node: collect one free slot per worker (distinct machines)
            free: list[tuple[WorkerState, int, list[int]]] = []
            for ws in workers:
                if ws.status != "CONNECTED":
                    continue
                for slot_idx, gpu_group in enumerate(ws.slots):
                    if slot_idx not in ws.busy_slots:
                        free.append((ws, slot_idx, gpu_group))
                        break  # one slot per worker

            if len(free) < nodes_per_run:
                deferred.append(var)
                continue

            assigned = free[:nodes_per_run]
            master_host = assigned[0][0].host.split("@")[-1]
            master_port = _MASTER_PORT_BASE + (next_port[0] % 100)
            next_port[0] += 1

            run_id = var["name"]
            in_flight[run_id] = var
            multinode_state[run_id] = {"pending": nodes_per_run, "success": True, "elapsed": 0.0}

            run_dir = os.path.join(output_dir, experiment, run_id)
            os.makedirs(run_dir, exist_ok=True)

            _append_manifest_run(exp_dir, var)
            inflight_treatments.add(tk)

            tag_str = ",".join(f"{k}={v}" for k, v in var["combo"].items() if v is not None)
            var_env: dict[str, str] = {}
            if tag_str:
                var_env["EXP_TAGS"] = tag_str
            var_env.update(wandb_env)

            loc_parts = []
            for node_rank, (ws, slot_idx, gpu_group) in enumerate(assigned):
                ws.busy_slots.add(slot_idx)
                run_scratch = os.path.join(ws.scratch_dir, experiment, run_id)
                node_env = {
                    **var_env,
                    "MLSWEEP_NNODES": str(nodes_per_run),
                    "MLSWEEP_NODE_RANK": str(node_rank),
                    "MLSWEEP_MASTER_ADDR": master_host,
                    "MLSWEEP_MASTER_PORT": str(master_port),
                }
                run_msg = MsgRun(
                    run_id=run_id,
                    experiment=experiment,
                    command=list(command) + var["overrides"] + list(extra),
                    env=node_env,
                    gpu_ids=gpu_group,
                    remote_dir=ws.remote_dir,
                    scratch=run_scratch,
                    run_from=run_from,
                    set_dist_env=set_dist_env,
                )
                ws.send_queue.put(encode(run_msg))
                ws.in_flight[run_id] = var
                ws.run_slots[run_id] = slot_idx
                gpu_label = f"gpu{'s' if len(gpu_group) > 1 else ''} {','.join(str(g) for g in gpu_group)}"
                loc_part = (f"{_MAGENTA}{ws.host}{_RESET} {gpu_label}"
                            if ws.host != "localhost" else gpu_label)
                loc_parts.append(f"[{loc_part}]")

            nm = run_id.ljust(max(len(v["name"]) for v in remaining + deferred + list(in_flight.values())) if remaining or deferred or in_flight else len(run_id))
            sdesc = _singular_desc(var["combo"], opts) if has_singular else ""
            loc = " ".join(loc_parts)
            log_path = os.path.join(run_dir, "training.log")
            if sdesc:
                sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} ({sdesc}) {loc} {_BLUE}{log_path}{_RESET}")
            else:
                sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} {loc} {_BLUE}{log_path}{_RESET}")

    return deferred


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        global _log_file

        argv = sys.argv[1:]
        extra_flags: list[str] = []
        gpus_per_run = 1
        run_from = None

        parser = argparse.ArgumentParser(
            description="Run experiment sweeps",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Extra args after -- are passed to every training run.\n\n"
                "Environment variables:\n"
                "  CUDA_VISIBLE_DEVICES  Nvidia GPU indices available for scheduling\n"
                "                        (used when --gpus is not set)\n"
                "  HIP_VISIBLE_DEVICES   AMD GPU indices available for scheduling\n"
                "                        (used when --gpus is not set)\n"
            ))
        parser.add_argument("sweep_file", help="Path to sweep .py file")
        parser.add_argument("--output_dir", default=os.path.join(_PROJECT_ROOT, "outputs", "sweeps"),
                            help="Output directory")
        parser.add_argument("--experiment", default=None,
                            help="Experiment name (default: <sweep>_<timestamp>)")
        parser.add_argument("--gpus", "-g", type=int, nargs="?", const=0, default=None,
                            help="Total GPUs to use (0 = all visible; default = GPUS_PER_RUN or 1)")
        parser.add_argument("--jobs-per-gpu", "-j", type=int, default=None,
                            help="Concurrent jobs per GPU (default: 1)")
        parser.add_argument("--workers", default=None,
                            help="Path to workers file for remote dispatch (one '<host> <remote_dir>' per line)")
        parser.add_argument("--resume", action="store_true",
                            help="Skip runs already recorded as completed in sweep_status.json")
        parser.add_argument("--dry-run", action="store_true",
                            help="Print commands without execution")
        parser.add_argument("--validate", action="store_true",
                            help="Validate sweep config, print all combinations, and exit (no jobs launched)")
        parser.add_argument("--note", default=None,
                            help="Human-readable note stored in sweep_manifest.json")
        parser.add_argument("--max-retries", type=int, default=2,
                            help="Max retries for orphaned runs (default: 2)")
        parser.add_argument("--scratch-dir", default="/tmp/mlsweep",
                            help="Worker scratch directory (default: /tmp/mlsweep)")
        parser.add_argument("--wandb-project", default=None,
                            help="W&B project name (enables wandb logging)")
        parser.add_argument("--wandb-entity", default=None,
                            help="W&B entity/team")
        parser.add_argument("--tensorboard-dir", default=None,
                            help="TensorBoard output directory (enables TensorBoard logging)")
        parser.add_argument(
            "--version", action="version",
            version=f"%(prog)s {importlib.metadata.version('mlsweep')}")

        args, extra = parser.parse_known_args(argv)
        if extra and extra[0] == "--":
            extra = extra[1:]

        if args.workers and (args.gpus is not None or args.jobs_per_gpu is not None):
            conflicting = []
            if args.gpus is not None:
                conflicting.append("-g/--gpus")
            if args.jobs_per_gpu is not None:
                conflicting.append("-j/--jobs-per-gpu")
            sweep_print(
                f"Error: {' and '.join(conflicting)} cannot be used with --workers. "
                f"Specify -g and -j per machine in the workers file instead."
            )
            sys.exit(1)

        info = load_sweep_file(args.sweep_file)
        sweep_name, options, command, exclude_fn, extra_flags = (
            info["name"], info["options"], info["command"],
            info["exclude"], info["extra_flags"])
        gpus_per_run = info.get("gpus_per_run", 1)
        nodes_per_run = info.get("nodes_per_run", 1)
        run_from = info.get("run_from")
        set_dist_env: bool = info.get("set_dist_env", False)
        method: str = info.get("method", "grid")
        optimize_cfg: dict[str, Any] = info.get("optimize") or {}
        validate_options(options, method=method)

        assert options is not None and command is not None

        # --validate: check config, print all combinations, exit (no jobs, no files)
        if args.validate:
            if method == "bayes":
                sweep_print(f"Sweep: {sweep_name} (bayes, budget={optimize_cfg['budget']})")
                sweep_print(f"Metric: {optimize_cfg['metric']} ({optimize_cfg['goal']})")
                sweep_print(f"Dimensions ({len(options)}):")
                for key, opt in options.items():
                    dim = key[1:]
                    if opt.get("_type") == "continuous":
                        sweep_print(f"  {dim}: {opt['distribution']} [{opt['min']}, {opt['max']}]"
                                    + (" [singular]" if opt.get("singular") else ""))
                    elif opt["_values"] != [None]:
                        sweep_print(f"  {dim}: {opt['_values']}"
                                    + (" [singular]" if opt.get("singular") else ""))
                sys.exit(0)

            all_variations = generate_variations(sweep_name, options, exclude_fn, extra_flags)
            expected = count_expected(options)
            excluded = expected - len(all_variations)
            dim_names = [k[1:] for k in options]
            sweep_print(f"Sweep: {sweep_name}")
            sweep_print(f"Dimensions: {', '.join(dim_names) if dim_names else '(none)'}")
            for key in options:
                dim_name = key[1:]
                values = options[key].get("_values", [])
                if values != [None]:
                    sweep_print(f"  {dim_name}: {values}")
            sweep_print(f"\nTotal combinations: {len(all_variations)}")
            if excluded:
                sweep_print(f"Excluded by EXCLUDE filter: {excluded}")
            sweep_print(f"\nRuns:")
            for var in all_variations:
                sweep_print(f"  {var['name']}: {var['combo']}")
            sys.exit(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment = args.experiment or f"{sweep_name}_{timestamp}"
        output_dir = os.path.abspath(args.output_dir)
        exp_dir = os.path.join(output_dir, experiment)
        os.makedirs(exp_dir, exist_ok=True)

        if not args.dry_run:
            _log_file = open(os.path.join(exp_dir, "sweep.log"), "w")

        if method == "bayes":
            from mlsweep._bayes import BayesianOptimizer
            optimizer: BayesianOptimizer | None = BayesianOptimizer(
                sweep_name, options, optimize_cfg, extra_flags=list(extra_flags)
            )
            variations: list[dict[str, Any]] = []
            expected = optimize_cfg["budget"]
            n = expected
        else:
            optimizer = None
            all_variations = generate_variations(sweep_name, options, exclude_fn, extra_flags)
            expected = count_expected(options)

            # Resume: skip variations already completed
            resumed_count = 0
            if args.resume:
                prior_status = _load_sweep_status(exp_dir)
                if prior_status:
                    completed = {name for name, s in prior_status.items() if s.get("status") == "ok"}
                    original_n = len(all_variations)
                    all_variations = [v for v in all_variations if v["name"] not in completed]
                    resumed_count = original_n - len(all_variations)
                    if resumed_count:
                        sweep_print(f"Resume: skipping {resumed_count} already-completed runs")

            variations = all_variations
            n = len(variations)

        # Header
        sweep_print(f"Command: {' '.join(command)}")
        if method == "bayes":
            sweep_print(f"Sweep: {sweep_name} (bayes, budget={expected})")
        elif expected < n:
            sweep_print(f"Sweep: {sweep_name} ({expected} expected, {n} worst case)")
        else:
            sweep_print(f"Sweep: {sweep_name} ({n} runs)")
        sweep_print(f"Experiment: {experiment}")
        if extra:
            sweep_print(f"Extra overrides: {' '.join(extra)}")

        # List all variations with colored flags (grid mode only — bayes has none yet)
        if method != "bayes":
            for var in variations:
                colored = []
                for ci, (key, val) in enumerate(var["combo"].items()):
                    flags = var["effective_options"].get(key, {}).get("_flags", {}).get(val, [])
                    if flags:
                        color = _DIM_COLORS[ci % len(_DIM_COLORS)]
                        colored.append(f"{color}{' '.join(flags)}{_RESET}")
                sweep_print(f"{_GREEN}{var['name']}{_RESET}: {' '.join(colored)}")
                sweep_print(f"{' '.join(list(command) + var['overrides'] + list(extra))}\n")

        if args.dry_run:
            sweep_print(f"\n{'=' * 80}")
            if method == "bayes":
                sweep_print(f"DRY RUN — {expected} bayes budget (runs generated at runtime)")
            elif expected < n:
                sweep_print(f"DRY RUN — {expected} expected ({n} worst case)")
            else:
                sweep_print(f"DRY RUN — {n} runs would be executed")
            sweep_print(f"{'=' * 80}")
            return

        # Write sweep_manifest.json
        _write_manifest(exp_dir, experiment, variations if method != "bayes" else [], note=args.note)

        # Build writer factory
        _writer_factories: list[WriterFactory] = [MlsweepWriterFactory()]
        if args.wandb_project:
            from mlsweep._writer_wandb import WandbWriterFactory
            _writer_factories.append(WandbWriterFactory(
                project=args.wandb_project,
                entity=args.wandb_entity,
            ))
        if args.tensorboard_dir:
            from mlsweep._writer_tensorboard import TensorBoardWriterFactory
            _writer_factories.append(TensorBoardWriterFactory(
                tb_dir=args.tensorboard_dir,
            ))
        factory: WriterFactory = MultiWriterFactory(_writer_factories)
        dims = list(variations[0]["combo"].keys()) if variations else []
        factory.on_sweep_start(experiment, dims, [v["name"] for v in variations])

        _wandb_skip = {"WANDB_RUN_ID", "WANDB_RESUME"}
        _wandb_env: dict[str, str] = {
            k: v for k, v in os.environ.items()
            if k.startswith("WANDB_") and k not in _wandb_skip
        }

        if method == "bayes":
            run_desc = f"bayes budget={expected}"
        else:
            run_desc = f"{expected} runs, {n} possible" if expected < n else f"{n} runs"
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"Starting sweep - ({run_desc})")
        sweep_print(f"{'=' * 80}\n")

        if not args.dry_run and not args.validate:
            sweep_print(f"Sweep log: {os.path.join(exp_dir, 'sweep.log')}")
        sweep_print(f"Output: {output_dir}\n")

        # Generate auth token
        token = secrets.token_hex(16)

        # Determine GPU configuration for local mode
        local_gpus: int | None = args.gpus
        jobs_per_slot = args.jobs_per_gpu if args.jobs_per_gpu is not None else 1

        if not args.workers:
            visible = _visible_devices()
            if not visible:
                if args.gpus is not None or os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES"):
                    sweep_print(f"{_RED}Error: GPUs requested but no GPU management tool found "
                                f"(tried nvidia-smi, amd-smi){_RESET}")
                    sys.exit(1)
                visible = [0]
            if args.gpus is None:
                num_gpus = gpus_per_run  # default: one slot
            elif args.gpus == 0:
                num_gpus = len(visible)  # all visible
            else:
                num_gpus = args.gpus
            if num_gpus > len(visible):
                sweep_print(f"Error: need {num_gpus} GPUs but only {len(visible)} visible")
                sys.exit(1)
            if num_gpus < gpus_per_run:
                sweep_print(f"Error: need at least {gpus_per_run} GPUs (GPUS_PER_RUN) "
                            f"but only {num_gpus} requested")
                sys.exit(1)
            if num_gpus % gpus_per_run != 0:
                sweep_print(f"Warning: {num_gpus} GPUs is not a multiple of "
                            f"GPUS_PER_RUN={gpus_per_run}; "
                            f"using {(num_gpus // gpus_per_run) * gpus_per_run}")
            local_gpus = num_gpus

        # Set up event queue and connect workers
        event_queue: queue.Queue[Any] = queue.Queue()

        sweep_print(f"Connecting to workers...")
        workers = _connect_workers(
            workers_file=args.workers,
            gpus_per_run=gpus_per_run,
            token=token,
            event_queue=event_queue,
            scratch_dir=args.scratch_dir,
            max_gpus=local_gpus,
            jobs_per_slot=jobs_per_slot,
        )

        if not workers:
            sweep_print(f"{_RED}Error: no workers could be started{_RESET}")
            sys.exit(1)

        # Wait for all workers to send MsgWorkerHello (with timeout)
        connected_count = 0
        connect_deadline = time.time() + 30.0
        while connected_count < len(workers) and time.time() < connect_deadline:
            try:
                ev = event_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(ev, EvWorkerConnected):
                ws = workers[ev.worker_id]
                ws.gpus = ev.gpus
                ws.scratch_dir = ev.scratch_dir
                # Convert topo from wire format to internal format
                ws.topo = {
                    tuple(int(x) for x in k.split(",")): v  # type: ignore[misc]
                    for k, v in ev.topo.items()
                }
                # Limit GPUs if requested (local mode)
                if ws.host == "localhost" and local_gpus is not None:
                    ws.gpus = ws.gpus[:local_gpus]
                n_slots = len(ws.gpus) // gpus_per_run
                if n_slots == 0:
                    sweep_print(f"  {_YELLOW}WARN{_RESET}  {ws.host}: only {len(ws.gpus)} GPU(s), "
                                f"need {gpus_per_run} per run — skipping")
                    ws.status = "DEAD"
                    connected_count += 1
                    continue
                base_slots = _best_gpu_groups(ws.gpus, gpus_per_run, n_slots, topo=ws.topo)
                # Expand by jobs_per_slot: each physical GPU group gets N slot entries
                ws.slots = [grp for grp in base_slots for _ in range(ws.jobs_per_slot)]
                ws.status = "CONNECTED"
                connected_count += 1

                phys_slots = n_slots
                total_jobs = len(ws.slots)  # already expanded by jobs_per_slot
                slot_word = "slot" if phys_slots == 1 else "slots"
                jobs_note = f" × {ws.jobs_per_slot} jobs" if ws.jobs_per_slot > 1 else ""
                sweep_print(f"  {_GREEN}OK{_RESET}    {ws.host}: {phys_slots} {slot_word} ({phys_slots * gpus_per_run} GPUs){jobs_note} = {total_jobs} concurrent jobs")
            elif isinstance(ev, EvWorkerDisconnected):
                ws = workers[ev.worker_id]
                ws.status = "DEAD"
                connected_count += 1
                sweep_print(f"  {_RED}FAIL{_RESET}  {ws.host}: failed to connect")

        active_workers = [ws for ws in workers if ws.status == "CONNECTED"]
        if not active_workers:
            sweep_print(f"{_RED}Error: no workers available{_RESET}")
            sys.exit(1)

        total_slots = sum(len(ws.slots) for ws in active_workers)
        sweep_print(f"Total slots: {total_slots}\n")

        # Start stdin thread for interactive commands
        stdin_t = threading.Thread(target=_stdin_thread, args=(event_queue,), daemon=True)
        stdin_t.start()

        # ── Scheduling loop ────────────────────────────────────────────────────────

        if method == "bayes":
            has_singular = any(v.get("singular") for v in options.values())
        else:
            has_singular = any(
                o.get("singular")
                for var in variations
                for o in var["effective_options"].values()
            )

        if method == "bayes":
            assert optimizer is not None
            initial_vars = optimizer.suggest(n=optimizer.n_initial)
            variations.extend(initial_vars)
            pending: list[dict[str, Any]] = list(initial_vars)
        else:
            pending = list(variations)

        told_treatments: set[tuple[Any, ...]] = set()
        in_flight: dict[str, dict[str, Any]] = {}   # run_id → var
        _MASTER_PORT_BASE = 29500
        next_port: list[int] = [0]
        multinode_state: dict[str, dict[str, Any]] = {}  # run_id → {pending, success, elapsed}
        results: list[tuple[Any, ...]] = []          # (var, success, elapsed, log_file)
        failed: list[dict[str, Any]] = []
        succeeded: list[dict[str, Any]] = []
        skipped_count = 0
        retry_counts: dict[str, int] = {}           # run_id → retry count
        log_handles: dict[str, IO[str]] = {}        # run_id → training.log file handle
        run_writers: dict[str, RunWriter] = {}      # run_id → metric writer
        t0 = time.time()

        # Initial dispatch
        pending = _dispatch_pending(
            workers, pending, in_flight, failed, succeeded,
            output_dir, experiment, exp_dir, token, command, extra, run_from,
            gpus_per_run, has_singular, _wandb_env,
            nodes_per_run, multinode_state, next_port,
            set_dist_env,
        )

        while pending or in_flight:
            try:
                ev = event_queue.get(timeout=0.5)
            except queue.Empty:
                # Re-try dispatch in case a slot freed up
                if pending:
                    pending = _dispatch_pending(
                        workers, pending, in_flight, failed, succeeded,
                        output_dir, experiment, exp_dir, token, command, extra, run_from,
                        gpus_per_run, has_singular, _wandb_env,
                        nodes_per_run, multinode_state, next_port,
                        set_dist_env,
                    )
                continue

            if isinstance(ev, EvRunStarted):
                # Open output files for this run
                run_dir = os.path.join(output_dir, experiment, ev.run_id)
                os.makedirs(run_dir, exist_ok=True)
                log_handles[ev.run_id] = open(os.path.join(run_dir, "training.log"), "w", buffering=1)
                _var = in_flight.get(ev.run_id)
                _combo: dict[str, Any] = _var.get("combo", {}) if _var is not None else {}
                run_writers[ev.run_id] = factory.make(ev.run_id, _combo, run_dir)

            elif isinstance(ev, EvLogLine):
                _log_fh = log_handles.get(ev.run_id)
                if _log_fh is not None:
                    _log_fh.write(ev.data)
                    _log_fh.flush()

            elif isinstance(ev, EvMetricLine):
                _writer = run_writers.get(ev.run_id)
                if _writer is not None:
                    _writer.on_metric(ev.step, ev.data)

            elif isinstance(ev, EvSyncRequest):
                var_sync = in_flight.get(ev.run_id)
                if var_sync:
                    ws = workers[ev.worker_id]  # noqa: F841 — ws used below
                    run_scratch = os.path.join(ws.scratch_dir, experiment, ev.run_id)
                    run_dir = os.path.join(output_dir, experiment, ev.run_id)
                    threading.Thread(
                        target=_rsync_thread,
                        args=(ws.host, run_scratch, run_dir, ev.run_id, event_queue,
                              ws.password, ws.ssh_key),
                        daemon=True,
                    ).start()

            elif isinstance(ev, EvRunResult):
                ws = workers[ev.worker_id]
                var = in_flight.pop(ev.run_id, {})
                ws.in_flight.pop(ev.run_id, {})

                # Free the exact slot that was used for this run
                used_slot_idx = ws.run_slots.pop(ev.run_id, None)
                if used_slot_idx is not None:
                    ws.busy_slots.discard(used_slot_idx)
                    used_gpu_group = ws.slots[used_slot_idx] if used_slot_idx < len(ws.slots) else []
                else:
                    used_gpu_group = []

                # Multi-node aggregation: wait for all nodes to complete
                if ev.run_id in multinode_state:
                    ms = multinode_state[ev.run_id]
                    ms["pending"] -= 1
                    ms["elapsed"] = max(ms["elapsed"], ev.elapsed)
                    if not ev.success:
                        ms["success"] = False
                    if ms["pending"] > 0:
                        # Not all nodes done yet — put var back and re-dispatch freed slot
                        if var:
                            in_flight[ev.run_id] = var
                        pending = _dispatch_pending(
                            workers, pending, in_flight, failed, succeeded,
                            output_dir, experiment, exp_dir, token, command, extra, run_from,
                            gpus_per_run, has_singular, _wandb_env,
                            nodes_per_run, multinode_state, next_port,
                            set_dist_env,
                        )
                        continue
                    # All nodes done — use aggregated values for result recording
                    del multinode_state[ev.run_id]
                    ev = EvRunResult(
                        worker_id=ev.worker_id,
                        run_id=ev.run_id,
                        success=ms["success"],
                        elapsed=ms["elapsed"],
                        exit_code=ev.exit_code,
                    )

                # Close log file handle and finish metric writer
                _log_fh = log_handles.pop(ev.run_id, None)
                if _log_fh is not None:
                    try:
                        _log_fh.close()
                    except OSError:
                        pass
                _writer = run_writers.pop(ev.run_id, None)
                if _writer is not None:
                    _writer.on_finish("ok" if ev.success else "failed", ev.elapsed)

                if var:
                    log_path = os.path.join(output_dir, experiment, ev.run_id, "training.log")
                    results.append((var, ev.success, ev.elapsed, log_path))
                    opts = var["effective_options"]
                    tk = _treatment_key(var["combo"], opts)

                    (succeeded if ev.success else failed).append(var["combo"])

                    _update_sweep_status(exp_dir, ev.run_id,
                                         "ok" if ev.success else "failed",
                                         ev.elapsed, var["combo"])

                    # Bayes: report result to optimizer and queue next suggestion
                    if optimizer is not None and ev.success and tk not in told_treatments:
                        told_treatments.add(tk)
                        mpath = os.path.join(output_dir, experiment, ev.run_id, "metrics.jsonl")
                        mval = extract_objective_metric(mpath, optimize_cfg["metric"], optimize_cfg["goal"])
                        optimizer.tell(var["combo"], mval)
                        if not optimizer.exhausted:
                            new_vars = optimizer.suggest(n=1)
                            pending.extend(new_vars)
                            variations.extend(new_vars)

                    pad = max((len(v["name"]) for v in variations), default=0)
                    nm = ev.run_id.ljust(pad)
                    sdesc = _singular_desc(var["combo"], opts) if has_singular else ""
                    log_path_str = log_path

                    if ev.success:
                        tag = f"{_GREEN}   OK{_RESET}"
                    elif has_singular and tk not in {_treatment_key(r[0]["combo"], r[0]["effective_options"]) for r in results if r[1]}:
                        tag = f"{_YELLOW}PROBE{_RESET}"
                    else:
                        tag = f"{_RED} FAIL{_RESET}"

                    gpu_label = (f"gpu{'s' if len(used_gpu_group) > 1 else ''} "
                                 f"{','.join(str(g) for g in used_gpu_group)}"
                                 if used_gpu_group else "")
                    loc = (f"{_MAGENTA}{ws.host}{_RESET} {gpu_label}"
                           if ws.host != "localhost" and gpu_label else gpu_label)

                    if method == "bayes":
                        told = len(told_treatments)
                        progress = f"[{told}/{expected} evaluated]"
                    else:
                        resolved = len({_treatment_key(r[0]["combo"], r[0]["effective_options"])
                                         for r in results if r[1]})
                        progress = f"[{resolved}/{expected} resolved]"

                    if sdesc:
                        sweep_print(f"  {tag}  {_GREEN}{nm}{_RESET} ({sdesc}) {loc} {_BLUE}{log_path_str}{_RESET} {ev.elapsed:.1f}s {progress}")
                    else:
                        sweep_print(f"  {tag}  {_GREEN}{nm}{_RESET} {loc} {_BLUE}{log_path_str}{_RESET} {ev.elapsed:.1f}s {progress}")

                    # Sync artifacts after run completes
                    run_scratch = os.path.join(ws.scratch_dir, experiment, ev.run_id)
                    run_dir = os.path.join(output_dir, experiment, ev.run_id)
                    threading.Thread(
                        target=_rsync_thread,
                        args=(ws.host, run_scratch, run_dir, ev.run_id, event_queue,
                              ws.password, ws.ssh_key),
                        daemon=True,
                    ).start()

                # Re-dispatch
                pending = _dispatch_pending(
                    workers, pending, in_flight, failed, succeeded,
                    output_dir, experiment, exp_dir, token, command, extra, run_from,
                    gpus_per_run, has_singular, _wandb_env,
                    nodes_per_run, multinode_state, next_port,
                    set_dist_env,
                )

            elif isinstance(ev, EvArtifactSynced):
                # Send MsgCleanup to the worker that ran this job
                for ws in workers:
                    if ev.run_id in ws.in_flight and ws.status == "CONNECTED":
                        ws.send_queue.put(encode(MsgCleanup(run_id=ev.run_id)))
                        break

            elif isinstance(ev, EvWorkerCleaned):
                pass  # nothing to do

            elif isinstance(ev, EvWorkerDisconnected):
                ws = workers[ev.worker_id]
                if ws.status == "CONNECTED":
                    sweep_print(f"  {_YELLOW}WARN{_RESET}  Worker {ws.host} disconnected; reconnecting...")
                    ws.status = "RECONNECTING"
                    # Mark in-flight runs as orphaned
                    for run_id in list(ws.in_flight.keys()):
                        _update_sweep_status(exp_dir, run_id, "orphaned", 0.0,
                                             ws.in_flight[run_id].get("combo", {}))
                    threading.Thread(
                        target=_reconnect_thread,
                        args=(ws, event_queue, token),
                        daemon=True,
                    ).start()

            elif isinstance(ev, EvReconnectWorker):
                ws = workers[ev.worker_id]
                if ev.success:
                    sweep_print(f"  {_GREEN}OK{_RESET}    Worker {ws.host} reconnected")
                    # Send MsgReplay for each resuming run
                    for rinfo in ev.resuming:
                        ws.send_queue.put(encode(MsgReplay(
                            run_id=rinfo["run_id"],
                            log_seq=rinfo["log_seq"],
                            metric_seq=rinfo["metric_seq"],
                        )))
                    # Re-queue orphaned runs that this worker was handling
                    orphaned = [run_id for run_id, var in ws.in_flight.items()
                                if run_id not in {r["run_id"] for r in ev.resuming}]
                    for run_id in orphaned:
                        var = ws.in_flight.pop(run_id, {})
                        if var:
                            rc = retry_counts.get(run_id, 0) + 1
                            retry_counts[run_id] = rc
                            if rc <= args.max_retries:
                                pending.append(var)
                            else:
                                sweep_print(f"  {_RED}FAIL{_RESET}  {run_id}: max retries exceeded")
                                results.append((var, False, 0.0,
                                                os.path.join(output_dir, experiment, run_id, "training.log")))
                                failed.append(var["combo"])
                                _update_sweep_status(exp_dir, run_id, "failed", 0.0, var["combo"])
                else:
                    sweep_print(f"  {_RED}FAIL{_RESET}  Worker {ws.host} unreachable; re-queuing runs")
                    for run_id, var in ws.in_flight.items():
                        rc = retry_counts.get(run_id, 0) + 1
                        retry_counts[run_id] = rc
                        if rc <= args.max_retries:
                            pending.append(var)
                        else:
                            results.append((var, False, 0.0,
                                            os.path.join(output_dir, experiment, run_id, "training.log")))
                            failed.append(var["combo"])
                            _update_sweep_status(exp_dir, run_id, "failed", 0.0, var["combo"])
                    ws.in_flight.clear()
                    ws.busy_slots.clear()

                # Try dispatch now that slots may be free
                if ev.success:
                    pending = _dispatch_pending(
                        workers, pending, in_flight, failed, succeeded,
                        output_dir, experiment, exp_dir, token, command, extra, run_from,
                        gpus_per_run, has_singular, _wandb_env,
                        nodes_per_run, multinode_state, next_port,
                        set_dist_env,
                    )

            elif isinstance(ev, EvInteractiveCommand):
                if ev.cmd == "status":
                    n_pending = len(pending)
                    n_inflight = len(in_flight)
                    n_done = len(results)
                    sweep_print(f"Status: {n_inflight} running, {n_pending} pending, {n_done} done")
                elif ev.cmd == "pause":
                    sweep_print("Pause not yet implemented")
                elif ev.cmd == "resume":
                    sweep_print("Resume not yet implemented")

        # Shut down all workers
        for ws in workers:
            if ws.status == "CONNECTED":
                ws.send_queue.put(encode(MsgShutdown()))
                ws.send_queue.put(None)  # drain and close write thread

        elapsed = time.time() - t0
        has_failures = print_summary(results, skipped_count, elapsed)
        sweep_print(f"\nOutput: {output_dir}")
        if _log_file:
            _log_file.close()
            print(f"Sweep log: {os.path.join(exp_dir, 'sweep.log')}")

        factory.on_sweep_end()

        if has_failures:
            sys.exit(1)


    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
