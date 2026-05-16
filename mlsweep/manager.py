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
import sys
from pathlib import Path
from secrets import token_hex

import aiosqlite
from aiohttp import web

from mlsweep._manager_state import ManagerState


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


async def _rebuild_state(
    db: aiosqlite.Connection,
) -> "ManagerState":  # noqa: F821
    """Rebuild in-memory state from the database on startup."""
    from mlsweep._manager_db import (
        list_pending_jobs,
        list_workers,
        reset_dispatched_running_to_pending,
    )

    state = ManagerState()

    # Reset any jobs that were left in dispatched/running state by a
    # previous manager crash.
    n = await reset_dispatched_running_to_pending(db)
    if n:
        print(f"Reset {n} dispatched/running jobs to pending")

    # Load pending jobs into sorted list.
    pending_jobs = await list_pending_jobs(db)
    for job in pending_jobs:
        state.insert_pending(job)
    print(f"Loaded {len(pending_jobs)} pending jobs")

    # Workers are loaded from DB for reference; WorkerConn objects are
    # created on-demand when workers actually connect over TCP.
    db_workers = await list_workers(db)
    print(f"Found {len(db_workers)} known workers in database")

    return state


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

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
    token_file.write_text(token)
    token_file.chmod(0o600)

    # ── Database ──────────────────────────────────────────────────────────
    db_path = args.db or os.path.join(str(mlsweep_dir), "manager.db")
    db = await aiosqlite.connect(db_path)
    print(f"Connected to database: {db_path}")

    from mlsweep._manager_db import init_db
    await init_db(db)
    print("Database schema initialized")

    # ── Rebuild state ────────────────────────────────────────────────────
    state = await _rebuild_state(db)

    # ── Shutdown coordination ────────────────────────────────────────────
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown_signal() -> None:
        print("\nShutting down...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown_signal)
        except NotImplementedError:
            # Windows does not support add_signal_handler
            pass

    # ── HTTP server ──────────────────────────────────────────────────────
    from mlsweep._manager_http import create_app
    from mlsweep._manager_workers import connect_workers, schedule_pending

    state.output_dir = os.path.join(str(mlsweep_dir), "experiments")
    state.artifact_base_url = f"http://{args.host}:{args.port}"
    state.token = token
    state.dispatch_callback = lambda: schedule_pending(db, state)

    app = create_app(db, state, token, mlsweep_dir=mlsweep_dir)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    # ── Startup message ──────────────────────────────────────────────────
    dashboard_url = f"{state.artifact_base_url}/?token={token}"
    print(f"mlsweep manager ready — dir {mlsweep_dir}")
    print(f"Dashboard: {dashboard_url}")

    # ── Worker connections ───────────────────────────────────────────────
    # Connect to workers (local or via workers file).  Workers register
    # their GPUs asynchronously; we use an event to wait for all hello
    # handshakes before the initial scheduling pass.
    workers_ready = asyncio.Event()

    workers = await connect_workers(
        db, state,
        workers_file=args.workers,
        shutdown_event=shutdown_event,
        workers_ready=workers_ready,
    )
    print(f"Connected to {len(workers)} worker(s)")

    # Wait for all workers to complete their hello handshake (GPU registration)
    # before the initial scheduling pass, with a generous timeout.
    try:
        await asyncio.wait_for(workers_ready.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("Warning: timed out waiting for worker hello handshakes")

    # Initial scheduling pass — dispatch any pending jobs loaded at startup.
    n = await schedule_pending(db, state)
    if n:
        print(f"Initial dispatch: {n} job(s) started")

    # Wait for shutdown
    await shutdown_event.wait()

    # ── Cleanup ──────────────────────────────────────────────────────────
    print("Shutting down HTTP server...")
    await runner.cleanup()
    print("Closing database...")
    await db.close()
    pid_file = mlsweep_dir / "manager.pid"
    pid_file.unlink(missing_ok=True)
    print("Manager stopped.")


def main() -> None:
    """Synchronous entry point for console_scripts."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
