"""The stdlib WebSocket client sends one frame at a time.

The CLI's ``iter_events`` heartbeat pings from its own thread while the reader
answers pongs and the caller may close.  Without serializing sendall(), two
frames can interleave and corrupt the connection.  This drives those two
threads into a fake socket that would notice overlap.
"""

import threading
import time

from mlsweep.run_sweep import _WS_OP_PING, _WS_OP_PONG, _WebSocket


class _OverlapDetectingSocket:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.frames = []
        self.first_entered = threading.Event()
        self.release_first = threading.Event()

    def sendall(self, data):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.frames.append(data)
            first = self.active == 1
        if first:
            self.first_entered.set()
            self.release_first.wait(2)
        with self._lock:
            self.active -= 1


def test_concurrent_ping_and_pong_do_not_overlap_in_sendall():
    ws = _WebSocket("ws://localhost/", token="")
    sock = _OverlapDetectingSocket()
    ws._sock = sock  # type: ignore[assignment]

    first = threading.Thread(target=ws.send_frame, args=(_WS_OP_PING, b""))
    first.start()
    try:
        assert sock.first_entered.wait(1), "first frame never reached the socket"
        second = threading.Thread(target=ws.send_frame, args=(_WS_OP_PONG, b""))
        second.start()
        time.sleep(0.2)
        assert sock.max_active == 1, "two frames were inside sendall() at once"
        sock.release_first.set()
        first.join(2)
        second.join(2)
    finally:
        sock.release_first.set()
        first.join(2)

    assert not first.is_alive()
    assert len(sock.frames) == 2
