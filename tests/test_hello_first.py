"""The worker's first message on a connection must be its MsgWorkerHello.

The manager drops a connection whose first message is anything else.  Building
the hello can take seconds (nvidia-smi topology, device probes), and the GPU
stats thread and run traffic used to reach a connection during that window, so
on slow hosts every connection was dropped and the worker relaunched forever.
"""
import queue
import socket
import threading

from mlsweep import worker
from mlsweep._shared import MsgGpuStats, MsgHello, MsgWorkerHello, decode, encode, read_msg


class _FastEvent:
    """Stand-in for ``_shutdown_event``: ``wait`` returns at once, ``n`` times."""

    def __init__(self, n: int) -> None:
        self.n = n

    def wait(self, timeout: float | None = None) -> bool:
        self.n -= 1
        return self.n < 0


def _broadcast_stats_once(monkeypatch):
    """Run one iteration of the GPU stats thread."""
    real = worker._shutdown_event
    monkeypatch.setattr(worker, "_shutdown_event", _FastEvent(1))
    worker._gpu_stats_thread()
    monkeypatch.setattr(worker, "_shutdown_event", real)


def test_stats_and_run_traffic_wait_for_hello(monkeypatch):
    topo_started = threading.Event()
    stats_sent = threading.Event()

    def slow_topology():
        topo_started.set()
        assert stats_sent.wait(5.0)
        return {}

    monkeypatch.setattr(worker, "visible_devices", lambda: [0])
    monkeypatch.setattr(worker, "_device_probe", False)
    monkeypatch.setattr(worker, "_gpu_topology", slow_topology)
    monkeypatch.setattr(worker, "_query_gpu_stats",
                        lambda: [{"gpu": 0, "util_pct": 0, "mem_used_mb": 0, "mem_total_mb": 1}])
    monkeypatch.setattr(worker, "_token", "tok")
    monkeypatch.setattr(worker, "_connections", [])
    monkeypatch.setattr(worker, "_shutdown_event", threading.Event())

    mgr, wrk = socket.socketpair()
    conn = worker.ConnState(sock=wrk, send_queue=queue.Queue())
    worker._connections.append(conn)
    threading.Thread(target=worker._write_thread, args=(conn,), daemon=True).start()
    mgr.sendall(encode(MsgHello(token="tok", controller_id="manager")))
    reader = threading.Thread(target=worker._read_thread, args=(conn,), daemon=True)
    reader.start()

    # Mid-handshake: a stats broadcast and run traffic must not reach this connection.
    assert topo_started.wait(5.0)
    _broadcast_stats_once(monkeypatch)
    assert worker._current_connection() is None
    stats_sent.set()

    mgr.settimeout(5.0)
    assert isinstance(decode(read_msg(mgr)), MsgWorkerHello)
    assert worker._current_connection() is conn

    # After the hello, stats flow as before.
    _broadcast_stats_once(monkeypatch)
    assert isinstance(decode(read_msg(mgr)), MsgGpuStats)

    mgr.close()
    reader.join(5.0)
