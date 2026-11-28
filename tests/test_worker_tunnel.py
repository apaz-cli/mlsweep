"""Workers the manager reaches over SSH (``workers add --tunnel``).

The manager normally dials the worker's port.  A cloud machine whose firewall
lets in only SSH needs the connection to go through ``ssh -L`` instead, which
``PortForward`` runs.  These tests stand in a small Python forwarder for ssh
(the same local-to-remote copy, with no SSH server needed), so the probing and
connecting logic runs for real.
"""

import asyncio
import json
import socket
import sys
import threading
import time

import pytest

from conftest import _api_get, _api_post
from mlsweep import ctl
from mlsweep._manager_workers import PortForward, launch_worker

_TOKEN = "test-token"

# Copies each connection to the -L target, as ssh does: when nothing listens
# there, the local connection is accepted and then closed.
_FORWARDER = r"""
import socket, sys, threading
spec = next(a for i, a in enumerate(sys.argv) if sys.argv[i - 1] == "-L")
_, lport, _, rport = spec.split(":")
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", int(lport))); srv.listen(8)
def pipe(a, b):
    try:
        while data := a.recv(65536):
            b.sendall(data)
    except OSError:
        pass
    for s in (a, b):
        try: s.shutdown(socket.SHUT_RDWR)
        except OSError: pass
while True:
    c, _ = srv.accept()
    try:
        r = socket.create_connection(("127.0.0.1", int(rport)))
    except OSError:
        c.close(); continue
    threading.Thread(target=pipe, args=(c, r), daemon=True).start()
    threading.Thread(target=pipe, args=(r, c), daemon=True).start()
"""


async def _fake_ssh(*cmd, **kwargs):
    """Start the forwarder with ssh's arguments in place of ssh."""
    kwargs.pop("preexec_fn", None)
    return await asyncio.create_subprocess_exec(sys.executable, "-c", _FORWARDER, *cmd, **kwargs)


async def _failing_ssh(*cmd, **kwargs):
    kwargs.pop("preexec_fn", None)
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import sys; sys.stderr.write('ubuntu@host: Permission denied (publickey).\\n'); "
                              "sys.exit(255)", **kwargs)


class _QuietServer:
    """A TCP server that, like a worker, waits for the client to speak first and then echoes."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._echo, args=(c,), daemon=True).start()

    @staticmethod
    def _echo(c):
        with c:
            while data := c.recv(65536):
                c.sendall(data)

    def close(self):
        self.sock.close()


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_ssh_command_forwards_a_loopback_port_to_the_worker():
    fwd = PortForward("ubuntu@10.0.0.5", ssh_key="~/k")
    cmd = fwd.command(40001, 7890)
    assert cmd[:2] == ["ssh", "-N"]
    assert cmd[cmd.index("-L") + 1] == "127.0.0.1:40001:127.0.0.1:7890"
    assert cmd[cmd.index("-i") + 1] == "~/k"
    assert cmd[-2:] == ["--", "ubuntu@10.0.0.5"]
    assert "ExitOnForwardFailure=yes" in cmd


def test_connect_reaches_a_worker_through_the_forward():
    server = _QuietServer()

    async def run():
        fwd = PortForward("h", spawn=_fake_ssh)
        try:
            await fwd.open(server.port)
            proc = fwd.proc
            await fwd.open(server.port)  # already pointing there: same ssh
            assert fwd.proc is proc
            reader, writer = await fwd.connect(timeout=10)
            writer.write(b"hello")
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(5), 5) == b"hello"
            writer.close()
        finally:
            fwd.close()

    try:
        asyncio.run(run())
    finally:
        server.close()


def test_connect_tells_a_closed_port_from_a_worker():
    async def run():
        fwd = PortForward("h", spawn=_fake_ssh)
        try:
            await fwd.open(_closed_port())
            assert await fwd.connect(timeout=3, required=False) is None
            with pytest.raises(RuntimeError, match="nothing answered on the worker's port"):
                await fwd.connect(timeout=2)
        finally:
            fwd.close()

    asyncio.run(run())


def test_connect_says_why_ssh_gave_up():
    async def run():
        fwd = PortForward("ubuntu@host", spawn=_failing_ssh)
        await fwd.open(7890)
        await fwd.proc.wait()
        with pytest.raises(RuntimeError, match=r"Permission denied \(publickey\)"):
            await fwd.connect(timeout=5)

    asyncio.run(run())


def test_launch_worker_connects_through_the_tunnel_to_the_port_it_prints():
    server = _QuietServer()

    async def fake_worker(*cmd, **kwargs):
        return await asyncio.create_subprocess_exec(
            sys.executable, "-c", f"import time; print('PORT={server.port}', flush=True); time.sleep(30)", **kwargs)

    async def run():
        fwd = PortForward("h", spawn=_fake_ssh)
        try:
            reader, writer, port = await launch_worker(
                host="localhost", remote_dir="", token="t", spawn=fake_worker, tunnel=fwd)
            assert port == server.port and fwd.remote_port == server.port
            writer.write(b"x")
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(1), 5) == b"x"
            writer.close()
        finally:
            fwd.close()

    try:
        asyncio.run(run())
    finally:
        server.close()


def test_a_worker_the_manager_cannot_dial_says_so():
    port = _closed_port()

    async def fake_worker(*cmd, **kwargs):
        return await asyncio.create_subprocess_exec(
            sys.executable, "-c", f"import time; print('PORT={port}', flush=True); time.sleep(30)", **kwargs)

    with pytest.raises(RuntimeError, match=f"listens on port {port}, but the manager could not connect"):
        asyncio.run(launch_worker(host="localhost", remote_dir="", token="t", spawn=fake_worker))


def test_the_api_refuses_a_tunnel_to_localhost(manager_server):
    _, url = manager_server
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as exc:
        _api_post(url, _TOKEN, "/api/workers", {"host": "localhost", "tunnel": True})
    assert exc.value.code == 400


def test_workers_add_reports_why_a_worker_did_not_connect(manager_server, capsys):
    server, url = manager_server
    # RFC 6761 reserves .invalid, so ssh to it fails fast without touching the network.
    with pytest.raises(SystemExit) as exc:
        ctl.workers_cmd(["add", "nobody@mlsweep-test.invalid", "--tunnel", "--port", "7890", "--wait", "60",
                         "--json", "--manager", url, "--token", server.token])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "did not connect: cannot start on nobody@mlsweep-test.invalid" in out
    row = json.loads(out[out.index("\n{") + 1:])
    assert row["status"] == "dead" and row["tunnel"] is True and row["port"] == 7890
    [stored] = [w for w in _api_get(url, _TOKEN, "/api/workers") if w["worker_id"] == row["worker_id"]]
    assert stored["tunnel"] is True


def test_the_system_page_connects_a_worker_over_ssh(tmp_path_factory):
    from browser import Browser, find_chromium
    from campaign_helpers import call, shared_manager, static_url

    chrome = find_chromium()
    if chrome is None:
        pytest.skip("no Chromium-family browser installed")
    managers = shared_manager(tmp_path_factory, "tunnel_webui")
    mgr = next(managers)
    page = Browser(chrome)
    try:
        page.goto(static_url(mgr, "system.html"))
        page.wait_for("!!document.getElementById('btn-connect-worker')")
        page.eval("document.getElementById('btn-connect-worker').click()")
        page.wait_for("!document.getElementById('fw-tunnel').checked")
        page.eval("document.getElementById('fw-host').value = 'nobody@mlsweep-ui.invalid';"
                  "document.getElementById('fw-tunnel').checked = true;"
                  "document.getElementById('connect-submit').click()")
        page.wait_for("document.getElementById('connect-modal').classList.contains('hidden')")

        def added():
            _, workers = call(mgr, "GET", "/api/workers")
            return [w for w in workers if w["host"] == "nobody@mlsweep-ui.invalid"]

        deadline = time.monotonic() + 15
        while not added():
            assert time.monotonic() < deadline, "the worker was never added"
            time.sleep(0.1)
        assert added()[0]["tunnel"] is True
    finally:
        page.close()
        managers.close()
