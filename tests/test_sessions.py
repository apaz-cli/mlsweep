"""Terminals on the manager (on by default, ``--no-terminals`` turns them off):
the pty over a WebSocket, and the bubblewrap sandbox around it.

Runs a real manager.  Skipped where bubblewrap cannot make a sandbox (not
installed, or unprivileged user namespaces disabled).
"""

import asyncio
import json
import os
import re
import shutil
import time
import uuid

import aiohttp
import pytest

from campaign_helpers import TOKEN, call
from conftest import _start_manager, _teardown_manager

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    proc, server, _ = _start_manager(tmp_path_factory.mktemp("sessions"))
    status, body = call(server, "GET", "/api/sessions")
    assert status == 200 and body["enabled"]
    if body["error"]:
        _teardown_manager(proc)
        pytest.skip(body["error"])
    yield server
    _teardown_manager(proc)


def ws_url(server, sid, **params):
    query = "&".join(f"{k}={v}" for k, v in {"token": TOKEN, **params}.items())
    return server.url.replace("http://", "ws://") + f"/ws/sessions/{sid}?{query}"


class Term:
    """A test's view of one session: what it printed so far, and a way to type."""

    def __init__(self, ws):
        self.ws = ws
        self.out = b""
        self.control = []

    async def read_until(self, pattern, timeout=20):
        rx = re.compile(pattern.encode() if isinstance(pattern, str) else pattern)
        deadline = asyncio.get_running_loop().time() + timeout
        while not rx.search(self.out):
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                raise AssertionError(f"timed out waiting for {pattern!r}; got {self.out[-2000:]!r}")
            msg = await self.ws.receive(timeout=left)
            if msg.type == aiohttp.WSMsgType.BINARY:
                self.out += msg.data
            elif msg.type == aiohttp.WSMsgType.TEXT:
                self.control.append(json.loads(msg.data))
            else:
                raise AssertionError(f"socket closed: {msg.type}; got {self.out[-2000:]!r}")
        return rx.search(self.out)

    async def run(self, cmd, timeout=20):
        """Type *cmd* and return what it printed (the marker is computed so the echo can't match)."""
        tag = uuid.uuid4().hex[:6]
        start = len(self.out)
        await self.ws.send_bytes(f"{cmd}; echo \"@@{tag}-$((40+2))@@\"\r".encode())
        m = await self.read_until(rf"@@{tag}-42@@", timeout)
        return self.out[start:m.start()].decode(errors="replace")

    async def wait_control(self, kind, timeout=20):
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            for c in self.control:
                if c.get("type") == kind:
                    return c
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                raise AssertionError(f"no {kind} message; got {self.control}")
            msg = await self.ws.receive(timeout=left)
            if msg.type == aiohttp.WSMsgType.TEXT:
                self.control.append(json.loads(msg.data))
            elif msg.type == aiohttp.WSMsgType.BINARY:
                self.out += msg.data
            else:
                raise AssertionError(f"socket closed before {kind}")


def new_session(server, **body):
    status, info = call(server, "POST", "/api/sessions", body)
    assert status == 201, info
    return info["id"]


async def attach(server, sid, http, **params):
    ws = await http.ws_connect(ws_url(server, sid, **params))
    t = Term(ws)
    await t.read_until(r"sandbox\S*\$ ")   # the prompt
    return t


def run_async(coro):
    return asyncio.run(coro)


def test_off_with_no_terminals(tmp_path) -> None:
    proc, server, _ = _start_manager(tmp_path, extra_args=("--no-terminals",))
    try:
        assert call(server, "GET", "/api/sessions") == (200, {"enabled": False})
        status, _ = call(server, "POST", "/api/sessions", {})
        assert status == 404
    finally:
        _teardown_manager(proc)


def test_the_sandbox_hides_gpus_and_the_host(mgr) -> None:
    sid = new_session(mgr)

    async def go():
        async with aiohttp.ClientSession() as http:
            t = await attach(mgr, sid, http)
            assert b"CPU only, no GPUs" in t.out
            assert (await t.run("ls /dev | tr '\\n' ' '")).split()[-1:] != []
            assert "nvidia" not in await t.run("ls /dev")
            assert (await t.run("echo \"[$CUDA_VISIBLE_DEVICES]\"")).strip().endswith("[]")
            # The real home path, backed by the workspace.
            assert (await t.run("echo $HOME")).strip().endswith(os.path.expanduser("~"))
            assert ".bashrc" in await t.run("ls -a ~")
            # It starts in the notes clone.
            assert (await t.run("pwd")).strip().endswith(os.path.expanduser("~/notes"))
            # Neither the manager's state (token file, database) nor the user's home is there.
            assert "gone" in await t.run(f"test -e {mgr.mlsweep_dir}/manager.token || echo gone")
            assert "gone" in await t.run(f"test -e {os.path.expanduser('~/.ssh')} || echo gone")
            assert "gone" in await t.run("test -e ~/.mlsweep/manager.token || echo gone")
            # /usr is read-only.
            assert "refused" in await t.run("touch /usr/x 2>/dev/null || echo refused")
            await t.ws.close()

    run_async(go())
    call(mgr, "DELETE", f"/api/sessions/{sid}")


def test_mlsweep_and_the_notes_work_inside(mgr) -> None:
    sid = new_session(mgr)

    async def go():
        async with aiohttp.ClientSession() as http:
            t = await attach(mgr, sid, http)
            out = await t.run("mlsweep ls >/dev/null 2>&1 && echo talks-to-manager")
            assert "talks-to-manager" in out
            out = await t.run("cd ~/notes && echo 'from the sandbox' > sandbox-note.md && git add -A"
                              " && git commit -qm 'Sandbox note' && git push -q 2>&1; echo pushed=$?")
            assert "pushed=0" in out, out
            await t.ws.close()

    run_async(go())
    assert (mgr.mlsweep_dir / "notes" / "sandbox-note.md").read_text() == "from the sandbox\n"
    call(mgr, "DELETE", f"/api/sessions/{sid}")


def test_resize_and_the_workspace_persists(mgr) -> None:
    sid = new_session(mgr)

    async def go():
        async with aiohttp.ClientSession() as http:
            t = await attach(mgr, sid, http)
            await t.ws.send_str(json.dumps({"type": "resize", "cols": 101, "rows": 33}))
            assert "33 101" in await t.run("stty size")
            await t.run("echo kept > ~/kept.txt")
            await t.ws.close()

    run_async(go())
    call(mgr, "DELETE", f"/api/sessions/{sid}")
    assert (mgr.mlsweep_dir / "workspace" / "kept.txt").read_text() == "kept\n"
    sid2 = new_session(mgr)

    async def again():
        async with aiohttp.ClientSession() as http:
            t = await attach(mgr, sid2, http)
            assert "kept" in await t.run("cat ~/kept.txt")
            await t.ws.close()

    run_async(again())
    call(mgr, "DELETE", f"/api/sessions/{sid2}")


def test_a_session_outlives_its_socket_and_resumes(mgr) -> None:
    sid = new_session(mgr)

    async def go():
        async with aiohttp.ClientSession() as http:
            t = await attach(mgr, sid, http)
            await t.run("echo first-part")
            seen = len(t.out)
            await t.ws.close()
            # Output while nobody is attached is kept.
            status, body = call(mgr, "GET", "/api/sessions")
            assert sid in [s["id"] for s in body["sessions"]]
            ws = await http.ws_connect(ws_url(mgr, sid))
            t2 = Term(ws)
            await ws.send_bytes(b"echo second-$((1+1))\r")
            await t2.read_until("second-2")
            # A fresh attach replays from the start, banner included.
            assert b"CPU only" in t2.out and b"first-part" in t2.out
            await ws.close()
            # Resuming from an offset sends only what came after it.
            ws = await http.ws_connect(ws_url(mgr, sid, since=seen))
            t3 = Term(ws)
            await ws.send_bytes(b"echo third-$((1+2))\r")
            await t3.read_until("third-3")
            assert b"first-part" not in t3.out and b"second-2" in t3.out
            hello = next(c for c in t3.control if c["type"] == "hello")
            assert hello["offset"] == seen
            await ws.close()

    run_async(go())
    call(mgr, "DELETE", f"/api/sessions/{sid}")


def test_exit_and_kill(mgr) -> None:
    sid = new_session(mgr)

    async def go():
        async with aiohttp.ClientSession() as http:
            t = await attach(mgr, sid, http)
            await t.ws.send_bytes(b"exit 3\r")
            assert (await t.wait_control("exit"))["code"] == 3
            await t.ws.close()

    run_async(go())
    # Exited with no one attached: gone from the list.
    _, body = call(mgr, "GET", "/api/sessions")
    assert sid not in [s["id"] for s in body["sessions"]]

    sid = new_session(mgr)
    status, _ = call(mgr, "DELETE", f"/api/sessions/{sid}")
    assert status == 200
    _, body = call(mgr, "GET", "/api/sessions")
    assert sid not in [s["id"] for s in body["sessions"]]
    assert call(mgr, "DELETE", f"/api/sessions/{sid}")[0] == 404


def write_config(mgr, cfg):
    (mgr.mlsweep_dir / "terminals.json").write_text(json.dumps(cfg) if not isinstance(cfg, str) else cfg)


def test_config_tools_mounts_env_and_commands(mgr, tmp_path) -> None:
    # Under /tmp on purpose: the sandbox's own /tmp must not hide what is bound there.
    tools = tmp_path / "tools"
    elsewhere = tmp_path / "elsewhere"
    tools.mkdir()
    elsewhere.mkdir()
    (tools / "hello-tool").write_text("#!/bin/sh\necho hello-from-tool\n")
    (tools / "hello-tool").chmod(0o755)
    (elsewhere / "real").write_text("#!/bin/sh\necho via-symlink\n")
    (elsewhere / "real").chmod(0o755)
    (tools / "linked").symlink_to(elsewhere / "real")
    (elsewhere / "unrelated").write_text("not bound\n")
    data = tmp_path / "data.txt"
    data.write_text("mounted-data\n")
    write_config(mgr, {"tools": [str(tools), "~/no/such/tools"], "mounts": [str(data)],
                       "env": ["USER", "MLSWEEP_TEST_NOT_SET", "MLSWEEP_TEST_SET=a=b c"], "commands": {"greet": "echo greeted-$((2+3))"}})
    try:
        _, body = call(mgr, "GET", "/api/sessions")
        assert body["commands"] == ["greet"] and body["config_error"] is None
        sid = new_session(mgr, command="greet")

        async def go():
            async with aiohttp.ClientSession() as http:
                t = await attach(mgr, sid, http)
                # The command ran, then the shell stayed.
                assert b"greeted-5" in t.out
                # What could not be honoured is said, not silently dropped.
                assert b"tools: ~/no/such/tools not found; skipped" in t.out
                assert b"env: MLSWEEP_TEST_NOT_SET is not set" in t.out
                assert "hello-from-tool" in await t.run("hello-tool")
                assert "via-symlink" in await t.run("linked")
                assert "gone" in await t.run(f"test -e {elsewhere / 'unrelated'} || echo gone")
                assert "mounted-data" in await t.run(f"cat {data}")
                assert "refused" in await t.run(f"echo x > {data} 2>/dev/null || echo refused")
                assert os.environ["USER"] in await t.run("echo user=$USER")
                assert "[a=b c]" in await t.run("echo \"[$MLSWEEP_TEST_SET]\"")
                await t.ws.close()

        run_async(go())
        call(mgr, "DELETE", f"/api/sessions/{sid}")
    finally:
        (mgr.mlsweep_dir / "terminals.json").unlink()


def test_a_bad_config_is_reported_without_a_restart(mgr) -> None:
    for cfg, says in [("{not json", "not valid JSON"), ({"tool": []}, "unknown key tool"),
                      ({"limits": {"memory": 5}}, "limits.memory"), ({"commands": {"x": 1}}, "commands"), ({"env": ["1BAD=x"]}, "NAME or NAME=value")]:
        write_config(mgr, cfg)
        _, body = call(mgr, "GET", "/api/sessions")
        assert says in body["config_error"], body
        status, err = call(mgr, "POST", "/api/sessions", {})
        assert status == 409 and says in err["error"]
    # Fixed: the next terminal opens, with no restart.
    write_config(mgr, {"limits": {"terminals": 2, "memory": None}})
    try:
        assert call(mgr, "GET", "/api/sessions")[1]["config_error"] is None
        a, b = new_session(mgr), new_session(mgr)
        status, err = call(mgr, "POST", "/api/sessions", {})
        assert status == 409 and "limits.terminals" in err["error"]
        for sid in (a, b):
            call(mgr, "DELETE", f"/api/sessions/{sid}")
    finally:
        (mgr.mlsweep_dir / "terminals.json").unlink()


def test_limits_reach_the_scope(mgr) -> None:
    if call(mgr, "GET", "/api/sessions")[1]["limits"] != "systemd":
        pytest.skip("systemd-run --user is not available here")
    write_config(mgr, {"limits": {"memory": "1G", "processes": 77}})
    try:
        sid = new_session(mgr)

        async def go():
            async with aiohttp.ClientSession() as http:
                t = await attach(mgr, sid, http)
                tag = uuid.uuid4().hex[:8]
                await t.ws.send_bytes(f"exec -a zz{tag} sleep 300 &\r".encode())
                await t.run("true")
                await t.ws.close()
                return tag

        tag = run_async(go())
        import subprocess
        pid = subprocess.run(["pgrep", "-f", f"zz{tag}"], capture_output=True, text=True).stdout.split()[0]
        cg = "/sys/fs/cgroup" + open(f"/proc/{pid}/cgroup").read().strip().split("::")[-1]
        assert open(cg + "/pids.max").read().strip() == "77"
        assert open(cg + "/memory.max").read().strip() == str(1024 ** 3)
        call(mgr, "DELETE", f"/api/sessions/{sid}")
    finally:
        (mgr.mlsweep_dir / "terminals.json").unlink()


def test_unknown_command_and_cross_origin_are_refused(mgr) -> None:
    status, body = call(mgr, "POST", "/api/sessions", {"command": "nope"})
    assert status == 409 and "no command named" in body["error"]
    sid = new_session(mgr)

    async def go():
        async with aiohttp.ClientSession() as http:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
                await http.ws_connect(ws_url(mgr, sid), headers={"Origin": "http://evil.example"})
            assert exc.value.status == 403

    run_async(go())
    call(mgr, "DELETE", f"/api/sessions/{sid}")


# ── The terminal tab in the Notes page ─────────────────────────────────────────
from browser import Browser, find_chromium  # noqa: E402
from campaign_helpers import static_url  # noqa: E402

CHROME = find_chromium()


@pytest.fixture
def page(mgr):
    if CHROME is None:
        pytest.skip("no Chromium-family browser installed")
    b = Browser(CHROME)
    b.goto(static_url(mgr, "notes.html"))
    b.eval("localStorage.clear()")
    b.goto(static_url(mgr, "notes.html"))
    b.wait_for("S.ready && S.term.enabled && Groups.size === 1", timeout=20)
    yield b
    b.close()


SCREEN = ("(() => { const t = activeTab(); if (!t || !t.term || !t.term.x) return ''; const b = t.term.x.buffer.active;"
          " let s = ''; for (let i = 0; i < b.length; i++) s += b.getLine(i).translateToString(true) + '\\n'; return s; })()")


def type_line(page, text):
    page.eval("activeTab().term.x.focus()")
    page.call("Input.insertText", text=text)
    for kind in ("keyDown", "keyUp"):
        page.call("Input.dispatchKeyEvent", type=kind, key="Enter", code="Enter", windowsVirtualKeyCode=13,
                  **({"text": "\r"} if kind == "keyDown" else {}))


def live_ids(mgr):
    return [s["id"] for s in call(mgr, "GET", "/api/sessions")[1]["sessions"] if not s["exited"]]


def test_terminal_tab_types_survives_reload_and_moves(mgr, page) -> None:
    page.eval("newTerminal()")
    page.wait_for("activeTab() && activeTab().type === 'term'")
    page.wait_for(f"{SCREEN}.includes('sandbox:')", timeout=20)
    sid = page.eval("activeTab().sid")
    type_line(page, "echo typed-$((20+3))")
    page.wait_for(f"{SCREEN}.includes('typed-23')")
    # The terminal fills its pane: far more than xterm's default 80x24 at this window size.
    cols, rows = page.eval("[activeTab().term.x.cols, activeTab().term.x.rows]")
    assert cols > 80 and rows > 24

    # A reload brings the tab back, attached to the same shell, with its output.
    page.goto(static_url(mgr, "notes.html"))
    page.wait_for("!!(activeTab() && activeTab().type === 'term' && activeTab().term && activeTab().term.x)", timeout=20)
    assert page.eval("activeTab().sid") == sid
    page.wait_for(f"{SCREEN}.includes('typed-23')")

    # Beside a note, in its own group, still the same live shell.
    (mgr.mlsweep_dir / "notes" / "term-side.md").write_text("# Side\n")
    page.eval("loadTree().then(() => openFile('term-side.md', { newTab: true }))")
    page.wait_for("activeTab() && activeTab().path === 'term-side.md'")
    page.eval("splitTab(S.tabs.find(t => t.type === 'term'), 'right')")
    page.wait_for("Groups.size === 2 && activeTab().type === 'term'")
    type_line(page, "echo moved-$((1+1))")
    page.wait_for(f"{SCREEN}.includes('moved-2')")
    assert sid in live_ids(mgr)

    # Closing the tab ends the shell.
    page.eval("closeTab(activeTab().id)")
    page.wait_for("Groups.size === 1")
    deadline = time.time() + 10
    while sid in live_ids(mgr) and time.time() < deadline:
        time.sleep(0.2)
    assert sid not in live_ids(mgr)


def test_exit_shows_and_any_key_closes(mgr, page) -> None:
    page.eval("newTerminal()")
    page.wait_for(f"{SCREEN}.includes('sandbox:')", timeout=20)
    type_line(page, "exit 0")
    page.wait_for(f"{SCREEN}.includes('exited with code 0')")
    assert page.eval("activeTab().title").endswith("(ended)")
    type_line(page, "")
    page.wait_for("!S.tabs.some(t => t.type === 'term')")


def test_links_in_tools_bring_their_install(tmp_path) -> None:
    from mlsweep._sessions import Config, plan

    tools = tmp_path / "bin-dir"
    tools.mkdir()
    pkg = tmp_path / "lib" / "node_modules" / "agent"
    (pkg / "dist").mkdir(parents=True)
    (pkg / "package.json").write_text("{}")
    (pkg / "dist" / "cli.js").write_text("")
    (tools / "agent").symlink_to(pkg / "dist" / "cli.js")
    prefix = tmp_path / "python-3.12"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "bin" / "python3.12").write_text("")
    (tools / "python3.12").symlink_to(prefix / "bin" / "python3.12")
    versions = tmp_path / "claude" / "versions"
    versions.mkdir(parents=True)
    (versions / "2.1").write_text("")
    (tools / "claude").symlink_to(versions / "2.1")
    (tools / "local").symlink_to(tools / "claude")   # a link within the folder brings nothing more
    cfg = Config(tools=[str(tools)])
    binds = set(plan(cfg).binds)
    assert {str(tools), str(pkg), str(prefix), str(versions)} <= binds
    assert str(tmp_path) not in binds and str(pkg / "dist") not in binds


def test_ctrl_c_stops_the_manager_with_a_terminal_attached(tmp_path) -> None:
    import signal

    proc, server, _ = _start_manager(tmp_path)
    try:
        if call(server, "GET", "/api/sessions")[1]["error"]:
            pytest.skip("bubblewrap cannot make a sandbox here")
        sid = new_session(server)

        async def go():
            async with aiohttp.ClientSession() as http:
                t = await attach(server, sid, http)
                os.kill(proc.pid, signal.SIGINT)
                start = time.time()
                while proc.poll() is None and time.time() - start < 15:
                    await asyncio.sleep(0.1)
                # The open socket is closed for the browser, not waited on.
                msg = await t.ws.receive(timeout=5)
                assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING)
                return time.time() - start

        took = run_async(go())
        assert proc.poll() is not None and took < 10, f"manager took {took:.1f}s to stop"
    finally:
        _teardown_manager(proc)


def test_a_terminal_can_start_in_a_notes_folder(mgr) -> None:
    status, body = call(mgr, "POST", "/api/sessions", {"folder": "../etc"})
    assert status == 400 and "folder" in body["error"]
    # sources/ is in the starter notes, so the clone has it.
    sid = new_session(mgr, folder="sources")
    # A folder the clone does not have (yet).
    sid2 = new_session(mgr, folder="sources/not-pulled/yet")

    async def go():
        async with aiohttp.ClientSession() as http:
            t = await attach(mgr, sid, http)
            assert (await t.run("pwd")).strip().endswith(os.path.expanduser("~/notes/sources"))
            await t.ws.close()
            t = await attach(mgr, sid2, http)
            assert b"~/notes/sources/not-pulled/yet is not in this clone yet" in t.out
            assert (await t.run("pwd")).strip().endswith(os.path.expanduser("~/notes/sources"))
            await t.ws.close()

    run_async(go())
    for s in (sid, sid2):
        call(mgr, "DELETE", f"/api/sessions/{s}")


def test_open_terminal_here_from_the_file_tree(mgr, page) -> None:
    page.wait_for("!!document.querySelector(\"#tree .tnode[data-path='sources']\")")
    page.eval("""(() => { const r = document.querySelector("#tree .tnode[data-path='sources']").getBoundingClientRect();
      document.querySelector("#tree .tnode[data-path='sources']").dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, clientX: r.left + 5, clientY: r.top + 5 })); })()""")
    page.eval("[...document.querySelectorAll('#ctx-menu .cm-item')].find(i => i.textContent === 'Open terminal here').click()")
    page.wait_for("activeTab() && activeTab().type === 'term'")
    assert page.eval("activeTab().title") == "Terminal: sources"
    page.wait_for(f"{SCREEN}.includes('sandbox:')", timeout=20)
    type_line(page, "pwd")
    page.wait_for(f"{SCREEN}.includes({json.dumps(os.path.expanduser('~/notes/sources'))})")
    page.eval("closeTab(activeTab().id)")
