#!/usr/bin/env python3
"""mlsweep manager — central controller with HTTP API and SQLite backend.

Usage:
    python -m mlsweep.manager [--port PORT] [--db PATH] [--mlsweep-dir DIR]
                            [--workers FILE]

The manager is the persistent brain of an mlsweep cluster.  It owns the
SQLite database, accepts worker connections, schedules jobs, and
serves a REST/SSE HTTP API for dashboards and programmatic clients.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
import sys
from pathlib import Path
from secrets import token_hex

import aiosqlite
from aiohttp import web

from mlsweep._manager_state import PLACEMENTS, ManagerState
from mlsweep._shared import DEFAULT_SCRATCH_DIR


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="mlsweep manager — central controller"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Externally-reachable hostname for artifact URLs (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7891,
        help="HTTP/WebSocket port (default: 7891)",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("MLSWEEP_DB_PATH", ""),
        help="SQLite database path (env: MLSWEEP_DB_PATH)",
    )
    parser.add_argument(
        "--mlsweep-dir",
        default=os.environ.get("MLSWEEP_DIR", "~/.mlsweep"),
        help="mlsweep state directory (env: MLSWEEP_DIR, default: ~/.mlsweep)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Authentication token (auto-generated if not provided)",
    )
    parser.add_argument(
        "--workers",
        default=None,
        help="Path to workers config file (default: local worker with visible GPUs)",
    )
    parser.add_argument(
        "--scratch-dir",
        default=os.environ.get("MLSWEEP_SCRATCH_DIR", DEFAULT_SCRATCH_DIR),
        help="Workers' scratch directory, unless a workers file entry sets scratch_dir "
             f"(env: MLSWEEP_SCRATCH_DIR, default: {DEFAULT_SCRATCH_DIR})",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Enable ANSI color in human-readable output (default: off)",
    )
    return parser.parse_args(argv)


def _check_pid_file(pid_file: Path) -> None:
    """Ensure only one manager is running.  Exit if another is alive."""
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pass
        else:
            try:
                os.kill(old_pid, 0)  # signal 0 just checks existence
                print(f"Error: manager already running with PID {old_pid}", file=sys.stderr)
                sys.exit(1)
            except OSError:
                # Process not alive — stale pid file
                pid_file.unlink(missing_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


async def _find_reachable_urls(port: int, token: str) -> list[tuple[str, str]]:
    """Return list of (dashboard_url, label) for each IP that can reach us."""
    import aiohttp

    candidates: list[tuple[str, str]] = [("localhost", "local")]
    lan_ip: str = ""

    # LAN IP via UDP trick (no actual I/O)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
        if lan_ip and lan_ip != "127.0.0.1":
            candidates.append((lan_ip, "LAN"))
    except OSError:
        pass

    # Public IP via ipify
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.ipify.org?format=json", timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                data = await resp.json()
                pub_ip = data.get("ip", "")
                if pub_ip and pub_ip != lan_ip and pub_ip != "127.0.0.1":
                    candidates.append((pub_ip, "public"))
    except Exception:
        pass

    # Probe each candidate
    reachable: list[tuple[str, str]] = []
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession() as session:
        for host, label in candidates:
            try:
                async with session.get(
                    f"http://{host}:{port}/api/health",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=4),
                ) as resp:
                    if resp.status < 500:
                        reachable.append((f"http://{host}:{port}/?token={token}", label))
            except Exception:
                pass

    # Always include localhost even if probe fails (firewall may block self-connect)
    if not reachable:
        reachable.append((f"http://localhost:{port}/?token={token}", ""))

    return reachable


async def _async_main(args: argparse.Namespace) -> None:

    mlsweep_dir = Path(args.mlsweep_dir).expanduser().resolve()

    # ── PID file ──────────────────────────────────────────────────────────
    _check_pid_file(mlsweep_dir / "manager.pid")

    # ── Directories ───────────────────────────────────────────────────────
    (mlsweep_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (mlsweep_dir / "experiments").mkdir(parents=True, exist_ok=True)

    # ── Authentication token ──────────────────────────────────────────────
    token_file = mlsweep_dir / "manager.token"
    if args.token:
        token = args.token
    elif token_file.exists():
        token = token_file.read_text().strip()
    else:
        token = token_hex(16)
    import tempfile as _tempfile
    _tmp_fd, _tmp_path = _tempfile.mkstemp(dir=token_file.parent)
    try:
        os.write(_tmp_fd, token.encode())
        os.close(_tmp_fd)
        os.chmod(_tmp_path, 0o600)
        os.replace(_tmp_path, str(token_file))
    except Exception:
        os.unlink(_tmp_path)
        raise

    # ── Database ──────────────────────────────────────────────────────────
    db_path = args.db or os.path.join(str(mlsweep_dir), "manager.db")

    # write_db is owned exclusively by the DbWriter actor (serial writes).
    # read_db is used by all other coroutines for SELECT queries; WAL mode
    # allows concurrent reads alongside the writer without blocking.
    from mlsweep._manager_db import DbWriter, init_db, log_compactor
    import sqlite3 as _sqlite3

    write_db = await aiosqlite.connect(db_path)
    await init_db(write_db)
    print(f"Connected to database: {db_path}")
    print("Database schema initialized")

    async def _open_reader() -> aiosqlite.Connection:
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = _sqlite3.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        return conn

    read_db = await _open_reader()
    # The compactor reads whole logs; its own connection keeps those reads
    # from queueing ahead of the scheduler's and the HTTP handlers'.
    compact_db = await _open_reader()

    # ── DB writer actor ──────────────────────────────────────────────────
    writer = DbWriter(write_db)
    writer_task = asyncio.create_task(writer.run(), name="db-writer")
    compactor_task = asyncio.create_task(log_compactor(compact_db, writer), name="log-compactor")

    # ── State ────────────────────────────────────────────────────────────
    # Jobs a previous manager left dispatched/running stay that way.  Each
    # worker's hello reports what it is still running, and the jobs of
    # workers that do not come back are requeued (see _handle_worker_hello).
    from mlsweep._manager_db import count_pending_jobs, get_setting

    state = ManagerState()
    state.db_writer = writer
    placement = await get_setting(read_db, "placement")
    for p in PLACEMENTS:
        if p == placement:
            state.placement = p
    print(f"{await count_pending_jobs(read_db)} pending job(s) in database")

    # ── Shutdown coordination ────────────────────────────────────────────
    loop = asyncio.get_running_loop()

    def _shutdown_signal() -> None:
        print("\nShutting down...")
        state.shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown_signal)
        except NotImplementedError:
            # Windows does not support add_signal_handler
            pass

    # ── HTTP server ──────────────────────────────────────────────────────
    from mlsweep._manager_http import create_app
    from mlsweep._manager_workers import (
        connect_workers,
        reconnect_known_workers,
        requeue_jobs_of_unknown_workers,
        scheduler_loop,
    )

    state.output_dir = os.path.join(str(mlsweep_dir), "experiments")
    state.artifact_base_url = f"http://{args.host}:{args.port}"
    state.token = token
    state.manager_port = args.port
    state.scratch_dir = args.scratch_dir
    scheduler_task = asyncio.create_task(scheduler_loop(read_db, state), name="scheduler")

    app = create_app(read_db, state, token, mlsweep_dir=mlsweep_dir)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    # ── Startup message ──────────────────────────────────────────────────
    print(f"mlsweep manager ready — dir {mlsweep_dir}")

    async def _print_dashboard() -> None:
        reachable = await _find_reachable_urls(args.port, token)
        if len(reachable) == 1:
            print(f"Dashboard: {reachable[0][0]}")
        else:
            print("Dashboard:")
            for url, label in reachable:
                suffix = f"  ({label})" if label else ""
                print(f"  {url}{suffix}")

    # Probing addresses takes seconds; do it while the workers start.
    dashboard_task = asyncio.create_task(_print_dashboard())

    # ── Worker connections ───────────────────────────────────────────────
    # Each worker becomes schedulable once its hello is handled; the
    # scheduler task is already running and picks them up as they arrive.
    workers = await connect_workers(
        read_db, state,
        workers_file=args.workers,
        manager_port=args.port,
    )
    print(f"Started {len(workers)} worker(s)")
    n_reconnect = await reconnect_known_workers(read_db, state, manager_port=args.port)
    if n_reconnect:
        print(f"Reconnecting {n_reconnect} known worker(s)")
    await requeue_jobs_of_unknown_workers(read_db, state)

    await state.shutdown_event.wait()

    # ── Cleanup ──────────────────────────────────────────────────────────
    print("Shutting down HTTP server...")
    scheduler_task.cancel()
    compactor_task.cancel()
    dashboard_task.cancel()
    await runner.cleanup()
    writer_task.cancel()
    print("Closing database...")
    await read_db.close()
    await compact_db.close()
    await write_db.close()
    pid_file = mlsweep_dir / "manager.pid"
    pid_file.unlink(missing_ok=True)
    print("Manager stopped.")


def main() -> None:
    """Synchronous entry point for console_scripts."""
    args = _parse_args()
    from mlsweep._colors import set_color
    if args.color:
        set_color(True)
    from mlsweep._manager_workers import _ensure_worker_wheels
    _ensure_worker_wheels()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
