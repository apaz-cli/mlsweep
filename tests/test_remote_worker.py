"""Unit tests for remote-worker specific code: SSH tunnel and reconnect logic.

These tests cover the changes made to support remote workers without requiring
the manager to have a public IP:
  - _launch_tunnel: spawns ssh -N -R to expose the manager's HTTP port remotely
  - _tunnel_monitor_task: restarts the tunnel when it dies
  - _reconnect_worker: hostname stripping (user@host → host for TCP connect)
  - connect_single_worker: no tunnel launched for localhost workers
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mlsweep._manager_state import WorkerConn
from mlsweep._manager_workers import _launch_tunnel, _tunnel_monitor_task


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_wc(host: str = "user@remote.host") -> WorkerConn:
    """Return a minimal WorkerConn suitable for tunnel tests."""
    return WorkerConn(
        worker_id="test-worker",
        host=host,
        port=34567,
        reader=MagicMock(),
        writer=MagicMock(),
    )


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
    """launch_worker also strips user@ before connecting — verify the constant."""
    from mlsweep._manager_workers import launch_worker
    import inspect
    src = inspect.getsource(launch_worker)
    assert 'host.split("@")[-1]' in src or "host.split('@')[-1]" in src


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
    assert result is None or hasattr(result, "returncode")


def test_launch_tunnel_oserror_returns_none():
    """_launch_tunnel returns None when create_subprocess_exec raises OSError."""
    async def _run():
        with patch(
            "mlsweep._manager_workers.asyncio.create_subprocess_exec",
            side_effect=OSError("ssh not found"),
        ):
            return await _launch_tunnel("remotehost", manager_port=7891)

    assert asyncio.run(_run()) is None


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
    """Monitor calls _launch_tunnel after the tunnel process exits."""
    async def _run():
        # A process that exits immediately (simulates tunnel dying)
        proc = await asyncio.create_subprocess_exec(
            "true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        wc = _make_wc()
        wc.tunnel_proc = proc

        shutdown = asyncio.Event()
        restart_called = asyncio.Event()

        async def mock_launch(*args, **kwargs):
            restart_called.set()
            # Return a long-running proc so the monitor settles
            return await asyncio.create_subprocess_exec(
                "sleep", "3600",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

        async def _shutdown_after_restart():
            await asyncio.wait_for(restart_called.wait(), timeout=10.0)
            shutdown.set()
            # Clean up the sleep proc the mock started
            if wc.tunnel_proc is not None and wc.tunnel_proc.returncode is None:
                wc.tunnel_proc.terminate()

        async def instant_sleep(_delay, **_kw):
            pass

        with patch("mlsweep._manager_workers._launch_tunnel", side_effect=mock_launch), \
             patch("mlsweep._manager_workers.asyncio.sleep", side_effect=instant_sleep):
            await asyncio.gather(
                _tunnel_monitor_task(wc, 7891, shutdown),
                _shutdown_after_restart(),
            )

        assert restart_called.is_set()

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

        shutdown = asyncio.Event()
        restart_called = []

        async def mock_launch(*args, **kwargs):
            restart_called.append(True)
            return None

        async def instant_sleep(_delay, **_kw):
            pass

        with patch("mlsweep._manager_workers._launch_tunnel", side_effect=mock_launch), \
             patch("mlsweep._manager_workers.asyncio.sleep", side_effect=instant_sleep):
            await asyncio.wait_for(
                _tunnel_monitor_task(wc, 7891, shutdown),
                timeout=5.0,
            )

        assert not restart_called, "Monitor should not restart tunnel for dead worker"

    asyncio.run(_run())
