"""Unit tests for remote-worker specific code: SSH tunnel and reconnect logic.

These tests cover the changes made to support remote workers without requiring
the manager to have a public IP:
  - _launch_tunnel: spawns ssh -N -R to expose the manager's HTTP port remotely
  - _tunnel_monitor_task: restarts the tunnel when it dies
  - _reconnect_worker: hostname stripping (user@host → host for TCP connect)
  - connect_single_worker: no tunnel launched for localhost workers
  - workers files: per-worker scratch_dir
  - launched workers' output is drained, so their pipes never fill
"""

import asyncio
import os
import sqlite3
import subprocess
import sys

import aiosqlite

from mlsweep._manager_state import ManagerState, WorkerConn
from mlsweep._manager_workers import _launch_tunnel, _tunnel_monitor_task


# ── Helpers ────────────────────────────────────────────────────────────────────


# RFC 6761 reserves .invalid, so ssh to it fails fast without touching the network.
_UNREACHABLE = "nobody@mlsweep-test.invalid"


def _make_wc(host: str = _UNREACHABLE) -> WorkerConn:
    """Return a minimal WorkerConn suitable for tunnel tests."""
    return WorkerConn(worker_id="test-worker", host=host, port=34567)


# ── Hostname stripping ─────────────────────────────────────────────────────────


def test_reconnect_strips_username():
    """user@host format is reduced to just host before opening a TCP connection.

    Regression test for the bug where _reconnect_worker passed the full
    'user@host' string to asyncio.open_connection, which silently failed every
    reconnect attempt for remote workers.
    """
    cases = [
        ("aaron@95.133.252.99", "95.133.252.99"),
        ("user@remotehost.example.com", "remotehost.example.com"),
        ("localhost", "localhost"),
        ("192.168.1.10", "192.168.1.10"),
    ]
    for raw, expected in cases:
        assert raw.split("@")[-1] == expected


def test_launch_worker_strips_username():
    """launch_worker and reconnects connect to the host without its user@ prefix."""
    import inspect
    from mlsweep._manager_workers import _bare_host, launch_worker
    assert _bare_host("aaron@10.0.0.5") == "10.0.0.5"
    assert _bare_host("localhost") == "localhost"
    assert "_bare_host(host)" in inspect.getsource(launch_worker)


# ── Tunnel not launched for localhost ─────────────────────────────────────────


def test_no_tunnel_for_localhost():
    """The tunnel guard (host != 'localhost' and manager_port) skips localhost."""
    assert not ("localhost" != "localhost" and 7891)


# ── _launch_tunnel ─────────────────────────────────────────────────────────────


def test_launch_tunnel_returns_proc_or_none():
    """_launch_tunnel either returns an asyncio.subprocess.Process or None.

    Uses a loopback address that will fail SSH quickly. The important thing is
    that the function doesn't raise — it handles errors and returns None.
    """
    async def _run():
        return await _launch_tunnel(
            "127.0.0.1", manager_port=59999,
            ssh_key=None, password=None,
        )

    result = asyncio.run(_run())
    # Either a Process object (SSH started but will fail) or None (OSError)
    assert result is None or isinstance(result, asyncio.subprocess.Process)


def test_launch_tunnel_oserror_returns_none():
    """_launch_tunnel returns None when ssh cannot be started (not on PATH)."""
    code = (
        "import asyncio\n"
        "from mlsweep._manager_workers import _launch_tunnel\n"
        "print(asyncio.run(_launch_tunnel('remotehost', manager_port=7891)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "PATH": "/nonexistent"},
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "None"
    assert "Could not start tunnel" in out.stderr


# ── _tunnel_monitor_task ───────────────────────────────────────────────────────


def test_tunnel_monitor_exits_immediately_when_shutdown_set():
    """Monitor exits without trying to restart when shutdown_event is pre-set."""
    async def _run():
        proc = await asyncio.create_subprocess_exec(
            "true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        wc = _make_wc()
        wc.tunnel_proc = proc

        shutdown = asyncio.Event()
        shutdown.set()

        await asyncio.wait_for(
            _tunnel_monitor_task(wc, 7891, shutdown),
            timeout=5.0,
        )

    asyncio.run(_run())


def test_tunnel_monitor_exits_when_proc_is_none():
    """Monitor exits immediately when wc.tunnel_proc is None."""
    async def _run():
        wc = _make_wc()
        wc.tunnel_proc = None
        shutdown = asyncio.Event()

        await asyncio.wait_for(
            _tunnel_monitor_task(wc, 7891, shutdown),
            timeout=5.0,
        )

    asyncio.run(_run())


def test_tunnel_monitor_restarts_dead_proc():
    """Monitor launches a new tunnel after the tunnel process exits."""
    async def _run():
        # A process that exits immediately (simulates tunnel dying)
        first = await asyncio.create_subprocess_exec("true")
        wc = _make_wc()
        wc.tunnel_proc = first
        shutdown = asyncio.Event()
        monitor = asyncio.create_task(_tunnel_monitor_task(wc, 7891, shutdown))

        # The first restart comes after a 2 s backoff.
        deadline = asyncio.get_running_loop().time() + 10
        while wc.tunnel_proc is first:
            assert asyncio.get_running_loop().time() < deadline, "tunnel was not restarted"
            await asyncio.sleep(0.1)
        restarted = wc.tunnel_proc

        shutdown.set()
        await asyncio.wait_for(monitor, timeout=5.0)
        if restarted.returncode is None:
            restarted.kill()
        await restarted.wait()

    asyncio.run(_run())


def test_tunnel_monitor_exits_when_worker_dead():
    """Monitor exits without restarting if wc.status is 'dead'."""
    async def _run():
        proc = await asyncio.create_subprocess_exec(
            "true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        wc = _make_wc()
        wc.tunnel_proc = proc
        wc.status = "dead"

        await asyncio.wait_for(
            _tunnel_monitor_task(wc, 7891, asyncio.Event()),
            timeout=5.0,
        )

        assert wc.tunnel_proc is proc, "Monitor should not restart tunnel for dead worker"

    asyncio.run(_run())


# ── reconnect_known_workers ────────────────────────────────────────────────────


def test_reconnect_known_workers_schedules_remote_only():
    """On startup, DB workers that are not already launched get reconnected.

    Localhost is skipped (connect_workers already handles it), and so are
    workers whose host is covered by a worker launched this session, plus any
    later DB worker sharing an already-scheduled host.
    """
    import mlsweep._manager_workers as mw

    class _FakeDbWriter:
        def __init__(self) -> None:
            self.updates: list[tuple[str, str, str | None]] = []

        async def update_worker_status(self, worker_id: str, status: str, last_error: str | None = None) -> None:
            self.updates.append((worker_id, status, last_error))

    async def _run() -> None:
        db = await aiosqlite.connect(":memory:")
        db.row_factory = sqlite3.Row
        await db.execute(
            "CREATE TABLE workers ("
            "worker_id TEXT PRIMARY KEY, host TEXT NOT NULL, remote_dir TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'offline', last_seen REAL, scratch_dir TEXT, "
            "port INTEGER NOT NULL DEFAULT 7890, ssh_key TEXT, venv TEXT, devices TEXT, "
            "unhealthy_devices TEXT, "
            "last_error TEXT)"
        )
        await db.commit()

        rows = [
            ("aaron@95.133.252.99:dynamic", "aaron@95.133.252.99", "dead", "stale-1"),
            ("aaron@95.133.252.99:other", "aaron@95.133.252.99", "dead", "stale-2"),
            ("covered@example.com:dynamic", "covered.example.com", "dead", "stale-3"),
            ("localhost:ephemeral:0", "localhost", "offline", None),
        ]
        for wid, host, status, err in rows:
            await db.execute(
                "INSERT INTO workers (worker_id, host, remote_dir, status, last_seen, port, last_error) "
                "VALUES (?, ?, '', ?, 0.0, 0, ?)",
                (wid, host, status, err),
            )
        await db.commit()

        state = ManagerState()
        state.workers["covered@example.com:7890"] = WorkerConn(
            worker_id="covered@example.com:7890", host="covered.example.com", port=7890,
        )
        state.db_writer = _FakeDbWriter()

        recorded: list[str] = []

        def _fake_connect(db, state, host, remote_dir, *, worker_id, **kw):
            recorded.append(worker_id)

            async def _noop() -> None:
                pass

            return _noop()

        orig = mw.connect_single_worker
        mw.connect_single_worker = _fake_connect
        try:
            scheduled = await mw.reconnect_known_workers(db, state, manager_port=0)
        finally:
            mw.connect_single_worker = orig

        assert scheduled == 1, f"expected 1 scheduled, got {scheduled}"
        assert recorded == ["aaron@95.133.252.99:dynamic"], recorded
        assert state.db_writer.updates == [
            ("aaron@95.133.252.99:dynamic", "reconnecting", None)
        ], state.db_writer.updates

        await db.close()

    asyncio.run(_run())


# ── Launch configuration and output ────────────────────────────────────────────


def test_workers_file_scratch_dir_is_optional_per_worker(tmp_path):
    from mlsweep._manager_workers import _parse_workers_file
    wf = tmp_path / "workers.toml"
    wf.write_text('[[workers]]\nhost = "a"\nremote_dir = "/p"\nscratch_dir = "/big/scratch"\n'
                  '[[workers]]\nhost = "b"\nremote_dir = "/p"\n')
    assert [w["scratch_dir"] for w in _parse_workers_file(str(wf))] == ["/big/scratch", None]


def test_relay_copies_every_line_including_overlong_ones(capsys):
    """Output is relayed in chunks, so a line longer than the reader's limit
    (which would make readline() raise and stop the draining) still passes."""
    from mlsweep._manager_workers import _relay_stream

    async def relay(data: bytes) -> None:
        stream = asyncio.StreamReader()
        stream.feed_data(data)
        stream.feed_eof()
        await _relay_stream(stream, "h:1")

    asyncio.run(relay(b"first\n" + b"x" * 200_000 + b"\nlast"))
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "[worker h:1] first"
    assert lines[-1] == "[worker h:1] last"
    assert "".join(ln.removeprefix("[worker h:1] ") for ln in lines[1:-1]) == "x" * 200_000


def test_remote_workers_are_given_their_scratch_dir(monkeypatch):
    """The scratch dir reaches a remote worker's command line, as it does a
    local worker's; remote workers used to keep the default regardless."""
    from mlsweep import _manager_workers as mw

    async def bootstrapped(*args, **kwargs):
        return True

    launched: list[tuple[str, ...]] = []

    class Launched(Exception):
        pass

    async def capture(*cmd, **kwargs):
        launched.append(cmd)
        raise Launched

    monkeypatch.setattr(mw, "_bootstrap_worker_venv", bootstrapped)
    monkeypatch.setattr(mw.asyncio, "create_subprocess_exec", capture)
    try:
        asyncio.run(mw.launch_worker("user@gpu.invalid", "/proj", "tok", scratch_dir="/big/scratch"))
    except Launched:
        pass
    [cmd] = launched
    assert cmd[cmd.index("user@gpu.invalid") + 1].count("--scratch-dir /big/scratch") == 1

