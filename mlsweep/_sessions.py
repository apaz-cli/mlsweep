"""Terminals in the dashboard: shells on the manager's machine, in a sandbox.

Terminals are on by default; ``mlsweep manager --no-terminals`` turns them off.
Each terminal is a shell in a pty that the manager owns, so it outlives the
browser tab that opened it and ends when the shell exits, when it is closed, or
when the manager stops.  The browser talks to it over ``/ws/sessions/{id}``:
binary frames carry terminal bytes both ways, text frames carry JSON control
messages (resize, exit).

The sandbox is bubblewrap, with no container image.  Inside are read-only
``/usr`` and ``/etc``, the manager's own Python environment (so ``mlsweep`` is
the manager's version), and one persistent workspace,
``<mlsweep-dir>/workspace``, shown at the user's real home path, so scripts
that use ``~/...`` find what is bound there.  It holds a clone of the notes at
``~/notes``.  Nothing else from the host is mounted: not the rest of the home
(the token file, ``~/.ssh``), not experiments or the database, and no GPU
device nodes, so a sandboxed process cannot open a GPU whatever it sets.  The
network is shared, so the shell reaches the manager on loopback and the internet.

The shell gets ``MLSWEEP_TOKEN``.  Whoever uses the terminal already holds the
token in their browser, so this gives an agent in it no more than its user has.

``<mlsweep-dir>/terminals.json`` adds to the sandbox (see ``Config``).  It is
read for every new terminal, so a change applies to the next one opened.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import json
import logging
import os
import secrets
import shutil
import struct
import sys
import termios
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import WSCloseCode, WSMsgType, web

import mlsweep
from mlsweep import _notes, _notes_git
from mlsweep._manager_http import (
    TOKEN_KEY, _bad_request, _error_response, _json_body, _json_response, _not_found_exc,
)

logger = logging.getLogger(__name__)

sessions_routes = web.RouteTableDef()
SESSIONS_KEY: web.AppKey["SessionService | None"] = web.AppKey("mlsweep_sessions", object)  # type: ignore[arg-type]

CONFIG_NAME = "terminals.json"
_SCROLLBACK = 512 * 1024        # bytes of output kept for a browser that attaches later


def _exit_msg(code: int | None) -> str:
    """The socket message telling a browser the shell exited with *code*."""
    return json.dumps({"type": "exit", "code": code})
_CLIENT_BACKLOG = 4 * 1024 * 1024  # a browser this far behind is dropped; it can reattach
_READ = 64 * 1024
_SYMLINKS_FOLLOWED = 1000       # per tools directory
_BASHRC = """\
# Written once by mlsweep for the sandbox's shell. Edit it freely.
PS1='\\[\\e[2m\\]sandbox\\[\\e[0m\\]:\\w\\$ '
"""
_BANNER = ("\x1b[2mmlsweep sandbox on the manager: CPU only, no GPUs. "
           "~ is the workspace and persists; ~/notes is a clone of the notes.\x1b[0m\r\n")


def _sandbox_home() -> str:
    """The home path inside: the manager's own, so ``~/...`` in scripts means the same place."""
    home = os.path.expanduser("~")
    if home in ("", "/") or home.startswith(("/usr", "/etc", "/proc", "/dev", "/tmp", "/run")):
        return "/home/sandbox"
    return home


class SessionError(Exception):
    """A terminal could not be started; the message says why."""


@dataclass
class Limits:
    memory: str | None = "50%"   # systemd MemoryMax for each terminal; null for none
    processes: int = 1024        # systemd TasksMax for each terminal
    terminals: int = 8           # open at once


@dataclass
class Config:
    """``terminals.json``.  Every key is optional::

        {
          "tools": ["~/git/Scripts", "~/.local/bin"],
          "mounts": ["~/data/papers"],
          "env": ["OPENAI_API_KEY", "EDITOR=vim"],
          "commands": {"claude": "claude"},
          "shell": "/bin/bash",
          "limits": {"memory": "50%", "processes": 1024, "terminals": 8}
        }

    ``tools`` are directories made visible read-only at the same path and put on
    ``$PATH``, in order.  A symlink in one brings what it points into: the
    target's directory, the npm package it is part of, or the prefix around a
    ``bin/`` it is in.  ``mounts`` are files or directories made
    visible read-only and not put on ``$PATH``.  ``env`` entries are ``NAME``,
    passed through from the manager's environment, or ``NAME=value``, set to
    that value.  ``commands`` are what a new
    terminal can start with, from the terminal icon's menu; the shell stays
    open afterwards.  ``~`` in a path is the manager's home.
    """

    tools: list[str] = field(default_factory=list)
    mounts: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    commands: dict[str, str] = field(default_factory=dict)
    shell: str = "/bin/bash"
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            return cls()
        name = path.name
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}: not valid JSON ({exc})") from None
        if not isinstance(raw, dict):
            raise ValueError(f"{name}: expected a JSON object")
        _no_unknown(name, raw, cls.__dataclass_fields__)
        for key in ("tools", "mounts", "env"):
            v = raw.get(key, [])
            if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                raise ValueError(f"{name}: {key} must be a list of strings")
        for entry in raw.get("env", []):
            var = entry.split("=", 1)[0]
            if not (var.replace("_", "a").isalnum() and not var[0].isdigit()):
                raise ValueError(f"{name}: env entry {entry!r} must be NAME or NAME=value")
        commands = raw.get("commands", {})
        if not isinstance(commands, dict) or not all(isinstance(k, str) and k and isinstance(v, str) and v
                                                     for k, v in commands.items()):
            raise ValueError(f"{name}: commands must map names to command lines")
        shell = raw.get("shell", cls.shell)
        if not isinstance(shell, str) or not shell.startswith("/"):
            raise ValueError(f"{name}: shell must be an absolute path")
        lim = raw.get("limits", {})
        if not isinstance(lim, dict):
            raise ValueError(f"{name}: limits must be an object")
        _no_unknown(f"{name}: limits", lim, Limits.__dataclass_fields__)
        limits = Limits(**lim)
        if limits.memory is not None and not (isinstance(limits.memory, str) and limits.memory):
            raise ValueError(f"{name}: limits.memory must be a size such as \"8G\" or \"50%\", or null")
        for key in ("processes", "terminals"):
            v = getattr(limits, key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{name}: limits.{key} must be a positive whole number")
        return cls(tools=raw.get("tools", []), mounts=raw.get("mounts", []), env=raw.get("env", []),
                   commands=commands, shell=shell, limits=limits)


def _no_unknown(where: str, raw: dict[str, Any], known: Any) -> None:
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError(f"{where}: unknown {'key' if len(unknown) == 1 else 'keys'} "
                         f"{', '.join(unknown)} (known: {', '.join(known)})")


# Shown read-only in every sandbox by _system_dirs, so never bound again.
_SYSTEM_LINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")  # into /usr when merged
_SYSTEM_DIRS = ("/usr", "/etc") + _SYSTEM_LINKS


def _system_dirs() -> list[str]:
    """bwrap arguments recreating the top-level system directories, read-only.

    On merged-/usr systems /bin, /lib and friends are symlinks into /usr, and
    are recreated as the same symlinks.
    """
    args: list[str] = ["--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
    for d in _SYSTEM_LINKS:
        if os.path.islink(d):
            args += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            args += ["--ro-bind", d, d]
    # resolv.conf is often a link into /run (systemd-resolved).
    resolv = os.path.realpath("/etc/resolv.conf")
    if not resolv.startswith(("/etc/", "/usr/")) and os.path.exists(resolv):
        args += ["--ro-bind", resolv, resolv]
    return args



def _python_paths() -> list[str]:
    """The manager's Python and mlsweep itself."""
    return [sys.prefix, sys.base_prefix, str(Path(mlsweep.__file__).resolve().parent)]


@dataclass
class Plan:
    """What a terminal's sandbox shows, worked out from the config."""

    binds: list[str]       # host paths, made visible read-only at the same path
    path: list[str]        # $PATH inside
    env: dict[str, str]    # passed through
    problems: list[str]    # entries that could not be honoured, for the banner


def _install_root(target: str) -> str:
    """What a link into *target* needs: the directory it is in, or the package it is part of.

    A launcher often links to one file of a larger install, such as an npm
    package's ``dist/cli.js``, which loads its neighbours and its package's
    dependencies.  So the nearest enclosing directory with a ``package.json``
    comes along whole, if there is one a few levels up, and a program in a
    ``bin/`` brings the prefix around it (``lib/`` and the rest).
    """
    d = target if os.path.isdir(target) else os.path.dirname(target)
    home = os.path.expanduser("~")
    a = d
    for _ in range(6):
        if os.path.exists(os.path.join(a, "package.json")):
            return a
        parent = os.path.dirname(a)
        if parent in (a, "/", home):
            break
        a = parent
    # A program in some prefix's bin/ (an interpreter, a toolchain) needs the rest of that prefix.
    if os.path.basename(d) == "bin" and os.path.dirname(d) not in ("/", home):
        return os.path.dirname(d)
    return d


def plan(cfg: Config) -> Plan:
    binds: list[str] = []
    path: list[str] = [str(Path(sys.executable).parent)]
    problems: list[str] = []

    def bind(p: str) -> None:
        if not any(p == d or p.startswith(d + "/") for d in _SYSTEM_DIRS):
            binds.append(p)

    for p in _python_paths():
        if p and os.path.exists(p):
            bind(p)
    for entry in cfg.tools:
        p = os.path.expanduser(entry)
        if not os.path.isdir(p):
            problems.append(f"tools: {entry} {'is not a directory' if os.path.exists(p) else 'not found'}")
            continue
        bind(p)
        path.append(p)
        # A link to somewhere else (~/.local/bin/claude -> ~/.local/share/claude/...)
        # brings the place it points to.
        with contextlib.suppress(OSError), os.scandir(p) as it:
            for n, e in enumerate(it):
                if n >= _SYMLINKS_FOLLOWED:
                    break
                if e.is_symlink():
                    target = os.path.realpath(e.path)
                    if os.path.exists(target) and os.path.dirname(target) != p:
                        bind(_install_root(target))
    for entry in cfg.mounts:
        p = os.path.expanduser(entry)
        if os.path.exists(p):
            bind(p)
        else:
            problems.append(f"mounts: {entry} not found")
    env: dict[str, str] = {}
    for entry in cfg.env:
        name, eq, value = entry.partition("=")
        if eq:
            env[name] = value
        elif name in os.environ:
            env[name] = os.environ[name]
        else:
            problems.append(f"env: {name} is not set in the manager's environment")
    if not os.path.exists(cfg.shell):
        problems.append(f"shell: {cfg.shell} not found")
    # Bind each place once, and nothing already inside another bound directory.
    uniq = sorted(set(binds))
    uniq = [b for b in uniq if not any(b != o and b.startswith(o.rstrip("/") + "/") for o in uniq)]
    path += ["/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin"]
    return Plan(binds=uniq, path=list(dict.fromkeys(path)), env=env, problems=problems)


class Session:
    def __init__(self, sid: str, title: str) -> None:
        self.id = sid
        self.title = title
        self.created = time.time()
        self.proc: asyncio.subprocess.Process | None = None
        self.master = -1
        self.buf = bytearray()      # at least the last _SCROLLBACK bytes of output
        self.total = 0              # bytes of output ever; buf holds [total - len(buf), total)
        self.clients: set[_Client] = set()
        self.exit_code: int | None = None
        self.pending_in = bytearray()

    @property
    def start(self) -> int:
        return self.total - len(self.buf)

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "created": self.created,
                "exited": self.exit_code is not None, "exit_code": self.exit_code,
                "clients": len(self.clients)}

    def output(self, data: bytes) -> None:
        self.total += len(data)
        self.buf += data
        # Trim only once the buffer holds twice the scrollback, so heavy output
        # does not move the whole buffer on every read.
        if len(self.buf) > 2 * _SCROLLBACK:
            cut = len(self.buf) - _SCROLLBACK
            # Start the kept part on a line, so a late reader gets whole lines.
            nl = self.buf.find(b"\n", cut, cut + 4096)
            del self.buf[: (nl + 1 if nl >= 0 else cut)]
        for c in list(self.clients):
            c.send(data)

    def resize(self, cols: int, rows: int, width: int = 0, height: int = 0) -> None:
        """Set the pty's size in cells and, when known, pixels.  Programs that
        draw images (chafa, yazi's previews) size them from the pixels, and give
        up when those are zero."""
        if self.master < 0:
            return
        cols = max(2, min(int(cols), 1000))
        rows = max(1, min(int(rows), 500))
        width = max(0, min(int(width), 65535))
        height = max(0, min(int(height), 65535))
        with contextlib.suppress(OSError):
            fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, width, height))

    def write(self, data: bytes) -> None:
        if self.master < 0 or self.exit_code is not None:
            return
        self.pending_in += data
        self._flush_in()

    def _flush_in(self) -> None:
        loop = asyncio.get_running_loop()
        while self.pending_in:
            try:
                n = os.write(self.master, self.pending_in)
            except BlockingIOError:
                loop.add_writer(self.master, self._flush_in)
                return
            except OSError:
                self.pending_in.clear()
                return
            del self.pending_in[:n]
        with contextlib.suppress(Exception):
            loop.remove_writer(self.master)


class _Client:
    """One browser attached to a session; its own queue, so a slow one holds no one up."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self.ws = ws
        self.queue: asyncio.Queue[bytes | str | None] = asyncio.Queue()
        self.backlog = 0
        self.dropped = False  # fell too far behind; the pump closes it

    def send(self, data: bytes | str) -> None:
        if self.dropped:
            return
        self.backlog += len(data)
        if self.backlog > _CLIENT_BACKLOG:
            # Too far behind: close it rather than queue output it will never read.
            self.dropped = True
            self.queue.put_nowait(None)
            return
        self.queue.put_nowait(data)

    async def pump(self) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                await self.ws.close()
                return
            self.backlog -= len(item)
            if isinstance(item, bytes):
                await self.ws.send_bytes(item)
            else:
                await self.ws.send_str(item)


class SessionService:
    def __init__(self, mlsweep_dir: Path, manager_port: int, token: str, notes: bool) -> None:
        self.config_path = mlsweep_dir / CONFIG_NAME
        self.home = mlsweep_dir / "workspace"
        self.inside_home = _sandbox_home()
        self.port = manager_port
        self.token = token
        self.notes = notes
        self.sessions: dict[str, Session] = {}
        self.error: str | None = None     # terminals cannot run here at all
        self.limits = "nice"
        self.bwrap = shutil.which("bwrap")
        self.setsid = shutil.which("setsid")
        self.systemd_run = shutil.which("systemd-run")
        self.nice = shutil.which("nice")
        self._probed: asyncio.Task[None] | None = None
        # One create at a time: the terminal limit and the notes clone are
        # shared state, and plan()/the clone are slow enough to interleave.
        self._create_lock = asyncio.Lock()

    def config(self) -> Config:
        """``terminals.json`` as it is now; a ValueError says what is wrong with it."""
        return Config.load(self.config_path)

    # ── startup ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Set up the workspace and probe the sandbox in the background.

        The probes spawn bwrap and a systemd scope, which can take seconds on a
        slow host, so they must not hold up the manager binding its port or
        reaching its workers.  Anything that needs their answer awaits ``ready``.
        """
        self.home.mkdir(parents=True, exist_ok=True)
        rc = self.home / ".bashrc"
        if not rc.exists():
            rc.write_text(_BASHRC)
        self._probed = asyncio.create_task(self._check())

    async def ready(self) -> None:
        """Wait for the startup probes, so ``error`` and ``limits`` are final."""
        if self._probed is not None:
            await asyncio.shield(self._probed)

    async def _check(self) -> None:
        if not self.bwrap:
            self.error = "bubblewrap (bwrap) is not installed on the manager's machine, so terminals are off"
        elif not self.setsid:
            self.error = "setsid (util-linux) is not installed on the manager's machine, so terminals are off"
        else:
            try:
                self.error, systemd = await asyncio.gather(self._probe(), self._systemd_works())
            except OSError as exc:
                self.error, systemd = f"could not run the sandbox check, so terminals are off: {exc}", False
            if not self.error and systemd:
                self.limits = "systemd"
        if self.error:
            logger.warning("terminals: %s", self.error)

    async def _run(self, *argv: str, timeout: float = 10) -> tuple[int, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=self._outer_env())
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, b"timed out"
        return proc.returncode or 0, out

    async def _probe(self) -> str | None:
        """Can bwrap make a sandbox here?  Some systems forbid unprivileged user namespaces."""
        assert self.bwrap
        code, out = await self._run(self.bwrap, *self._sandbox_args(plan(Config())), "true")
        if code:
            return ("bubblewrap could not start a sandbox (unprivileged user namespaces may be "
                    f"disabled), so terminals are off: {out.decode(errors='replace').strip()[:300]}")
        return None

    async def _systemd_works(self) -> bool:
        if not self.systemd_run:
            return False
        code, _ = await self._run(self.systemd_run, "--user", "--scope", "--quiet", "--collect",
                                  "-p", "TasksMax=16", "--", "true", timeout=5)
        return code == 0

    # ── the sandbox ────────────────────────────────────────────────────────
    def _outer_env(self) -> dict[str, str]:
        keep = ("PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "LANG", "LC_ALL")
        return {k: os.environ[k] for k in keep if k in os.environ}

    def _sandbox_args(self, pl: Plan, cwd: str | None = None) -> list[str]:
        # Mounts apply in order, and a later one hides what is under it: the empty
        # /dev, /tmp and /run first, then the system, the workspace as home, and
        # last what is bound on top of it.
        args = ["--die-with-parent", "--unshare-all", "--share-net", "--hostname", "sandbox"]
        # Only the standard nodes: no /dev/nvidia*, so no GPU, whatever the environment says.
        args += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/run"]
        args += _system_dirs()
        args += ["--bind", str(self.home), self.inside_home]
        for b in pl.binds:
            args += ["--ro-bind", b, b]
        args += ["--chdir", cwd or self.inside_home]
        return args

    def _inner_env(self, cfg: Config, pl: Plan) -> dict[str, str]:
        user = os.environ.get("USER", "sandbox")
        env = {
            "HOME": self.inside_home, "USER": user, "LOGNAME": os.environ.get("LOGNAME", user),
            "SHELL": cfg.shell, "PATH": ":".join(pl.path),
            "TERM": "xterm-256color", "COLORTERM": "truecolor",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "MLSWEEP_MANAGER": f"http://127.0.0.1:{self.port}", "MLSWEEP_TOKEN": self.token,
            "MLSWEEP_SANDBOX": "1", "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        }
        if self.notes:
            env["MLSWEEP_NOTES"] = f"{self.inside_home}/notes"
        env.update(pl.env)
        return env

    def argv(self, cfg: Config, pl: Plan, command: str | None, cwd: str | None = None) -> list[str]:
        assert self.bwrap and self.setsid
        # setsid --ctty: a new session whose controlling terminal is the pty, so
        # job control and Ctrl+C work in the shell.
        out = [self.setsid, "--ctty"]
        if self.limits == "systemd" and self.systemd_run:
            out += [self.systemd_run, "--user", "--scope", "--quiet", "--collect",
                    "-p", f"TasksMax={cfg.limits.processes}", "-p", "CPUWeight=20"]
            if cfg.limits.memory:
                out += ["-p", f"MemoryMax={cfg.limits.memory}"]
            out += ["--"]
        elif self.nice:
            out += [self.nice, "-n", "10"]
        out += [self.bwrap, *self._sandbox_args(pl, cwd), "--clearenv"]
        for k, v in self._inner_env(cfg, pl).items():
            out += ["--setenv", k, v]
        shell = cfg.shell
        if command:
            # The command runs first; the shell stays afterwards, like a terminal you typed it into.
            out += ["--", shell, "-i", "-c", f"{command}\nexec {shell} -i"]
        else:
            out += ["--", shell, "-i"]
        return out

    async def _sync_notes(self) -> str | None:
        """Clone the notes into the workspace, or refresh the clone's token."""
        if not self.notes:
            return None
        url = f"http://127.0.0.1:{self.port}/notes.git"
        dest = self.home / "notes"

        def _do() -> None:
            if _notes_git.update_clone(dest, url, token=self.token):
                return
            if dest.exists() and any(dest.iterdir()):
                raise _notes_git.GitError(f"~/notes exists and is not a clone of {url}; left alone")
            _notes_git.clone_repo(url, dest, token=self.token, name="sandbox")

        try:
            await asyncio.to_thread(_do)
        except Exception as exc:
            return str(exc)
        return None

    # ── terminals ──────────────────────────────────────────────────────────
    async def create(self, name: str | None, folder: str = "") -> Session:
        """Start a terminal, running command *name* from the config if given,
        in *folder* of the notes clone (a path relative to it; "" is its top)."""
        await self.ready()
        async with self._create_lock:
            return await self._create_locked(name, folder)

    async def _create_locked(self, name: str | None, folder: str = "") -> Session:
        if self.error:
            raise SessionError(self.error)
        try:
            cfg = self.config()
        except ValueError as exc:
            raise SessionError(str(exc)) from None
        live = [s for s in self.sessions.values() if s.exit_code is None]
        if len(live) >= cfg.limits.terminals:
            raise SessionError(f"{len(live)} terminals are open, the most {CONFIG_NAME} allows (limits.terminals)")
        command = None
        if name:
            if name not in cfg.commands:
                raise SessionError(f"no command named {name!r} in {CONFIG_NAME}")
            command = cfg.commands[name]
        # plan() walks the tool dirs on disk, so keep it off the event loop.
        pl, note_err = await asyncio.gather(asyncio.to_thread(plan, cfg), self._sync_notes())
        # A new terminal starts in the notes, when there is a clone to start in.
        has_notes = self.notes and not note_err and (self.home / "notes" / ".git").exists()
        cwd = f"{self.inside_home}/notes" if has_notes else self.inside_home
        missing = None
        if has_notes and folder:
            # The clone can be behind the manager's folder: start in the deepest part it has.
            parts = folder.split("/")
            have = parts
            while have and not (self.home / "notes" / Path(*have)).is_dir():
                have = have[:-1]
            if have != parts:
                missing = folder
            if have:
                cwd += "/" + "/".join(have)
        sid = secrets.token_hex(6)
        s = Session(sid, name or "Terminal")
        s.output(_BANNER.encode())
        for p in pl.problems:
            s.output(f"\x1b[33m{CONFIG_NAME}: {p}; skipped\x1b[0m\r\n".encode())
        if note_err:
            s.output(f"\x1b[33mCould not set up ~/notes: {note_err}\x1b[0m\r\n".encode())
        if missing:
            s.output(f"\x1b[33m~/notes/{missing} is not in this clone yet; `git pull` brings it.\x1b[0m\r\n".encode())
        master, slave = os.openpty()
        try:
            os.set_blocking(master, False)
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
            # Spawned from the event loop's thread: bwrap's --die-with-parent is
            # tied to the thread that started it, and this one lives as long as the manager.
            s.proc = await asyncio.create_subprocess_exec(
                *self.argv(cfg, pl, command, cwd), stdin=slave, stdout=slave, stderr=slave,
                env=self._outer_env(), cwd=str(self.home), close_fds=True)
        except Exception:
            os.close(master)
            raise
        finally:
            os.close(slave)
        s.master = master
        self.sessions[sid] = s
        asyncio.get_running_loop().add_reader(master, self._readable, s)
        asyncio.create_task(self._reap(s))
        logger.info("terminals: started %s (%s)", sid, s.title)
        return s

    def _readable(self, s: Session) -> bool:
        """Pass what the pty has to the session; False once nothing more is ready."""
        try:
            data = os.read(s.master, _READ)
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno != errno.EIO:   # EIO: every process holding the pty has exited
                logger.debug("terminals: read %s: %s", s.id, exc)
            data = b""
        if data:
            s.output(data)
            return True
        self._close_pty(s)
        return False

    def _close_pty(self, s: Session) -> None:
        if s.master < 0:
            return
        loop = asyncio.get_running_loop()
        with contextlib.suppress(Exception):
            loop.remove_reader(s.master)
        with contextlib.suppress(Exception):
            loop.remove_writer(s.master)
        with contextlib.suppress(OSError):
            os.close(s.master)
        s.master = -1

    async def _reap(self, s: Session) -> None:
        assert s.proc
        code = await s.proc.wait()
        # Whatever the shell printed last is still in the pty; read it before closing.
        while s.master >= 0 and self._readable(s):
            pass
        self._close_pty(s)
        s.exit_code = code
        msg = _exit_msg(code)
        for c in list(s.clients):
            c.send(msg)
        if not s.clients:
            self.sessions.pop(s.id, None)
        logger.info("terminals: %s exited with %s", s.id, code)

    async def kill(self, s: Session) -> None:
        if s.exit_code is None and s.proc:
            # SIGHUP is what a closed terminal sends; bwrap's children go with it.
            with contextlib.suppress(ProcessLookupError):
                s.proc.send_signal(1)
            try:
                await asyncio.wait_for(asyncio.shield(s.proc.wait()), 3)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    s.proc.kill()
        self.sessions.pop(s.id, None)

    async def stop(self) -> None:
        """End every terminal and close the browsers' sockets to them.

        This runs when the manager starts shutting down: the server waits for
        open requests to finish, and a terminal's socket never does by itself.
        """
        if self._probed is not None:
            self._probed.cancel()
        clients = [c for s in self.sessions.values() for c in s.clients]
        await asyncio.gather(*(c.ws.close(code=WSCloseCode.GOING_AWAY, message=b"the manager is stopping")
                               for c in clients), return_exceptions=True)
        await asyncio.gather(*(self.kill(s) for s in list(self.sessions.values())), return_exceptions=True)


# ── routes ─────────────────────────────────────────────────────────────────────
def _service(request: web.Request) -> SessionService:
    svc = request.app[SESSIONS_KEY]
    if svc is None:
        raise _not_found_exc("terminals are off; the manager was started with --no-terminals")
    return svc


@sessions_routes.get("/api/sessions")
async def list_sessions(request: web.Request) -> web.Response:
    svc = request.app[SESSIONS_KEY]
    if svc is None:
        return _json_response({"enabled": False})
    await svc.ready()
    try:
        commands, config_error = sorted(svc.config().commands), None
    except ValueError as exc:
        commands, config_error = [], str(exc)
    return _json_response({
        "enabled": True, "error": svc.error, "config_error": config_error, "limits": svc.limits,
        "commands": commands, "sessions": [s.info() for s in svc.sessions.values()],
    })


@sessions_routes.post("/api/sessions")
async def create_session(request: web.Request) -> web.Response:
    svc = _service(request)
    body = await _json_body(request) if request.can_read_body else {}
    name = body.get("command")
    if name is not None and not isinstance(name, str):
        raise _bad_request("command must be a string")
    try:
        folder = _notes.clean_rel(body.get("folder") or "", allow_root=True)
    except _notes.PathError as exc:
        raise _bad_request(f"folder: {exc}") from None
    try:
        s = await svc.create(name or None, folder)
    except SessionError as exc:
        return _error_response(str(exc), status=409)
    return _json_response(s.info(), status=201)


@sessions_routes.delete("/api/sessions/{session_id}")
async def delete_session(request: web.Request) -> web.Response:
    svc = _service(request)
    s = svc.sessions.get(request.match_info["session_id"])
    if s is None:
        return _error_response("no such terminal", status=404)
    await svc.kill(s)
    return _json_response({"ok": True})


def _same_origin(request: web.Request) -> bool:
    """Refuse a page on another site opening the socket (it would need the token too)."""
    origin = request.headers.get("Origin")
    if not origin:
        return True
    return origin.split("://", 1)[-1].rstrip("/") == request.host


@sessions_routes.get("/ws/sessions/{session_id}")
async def session_ws(request: web.Request) -> web.StreamResponse:
    svc = _service(request)
    if not _same_origin(request):
        return _error_response("cross-origin terminal connections are refused", status=403)
    s = svc.sessions.get(request.match_info["session_id"])
    if s is None:
        return _error_response("no such terminal; it may have exited or the manager restarted", status=404)
    try:
        since = int(request.query.get("since", "0"))
    except ValueError:
        raise _bad_request("since must be an integer")
    ws = web.WebSocketResponse(max_msg_size=1024 * 1024, heartbeat=30)
    await ws.prepare(request)
    client = _Client(ws)
    # Catch up from *since* if that is still kept; otherwise from the oldest byte kept.
    if since < s.start or since > s.total:
        client.send(json.dumps({"type": "reset"}))
        since = s.start
    client.send(json.dumps({"type": "hello", "id": s.id, "title": s.title, "offset": since}))
    if since < s.total:
        client.send(bytes(memoryview(s.buf)[since - s.start:]))
    if s.exit_code is not None:
        client.send(_exit_msg(s.exit_code))
    s.clients.add(client)
    pump = asyncio.create_task(client.pump())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                s.write(msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    ctl = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if isinstance(ctl, dict) and ctl.get("type") == "resize":
                    with contextlib.suppress(TypeError, ValueError):
                        s.resize(ctl.get("cols", 80), ctl.get("rows", 24), ctl.get("width", 0), ctl.get("height", 0))
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        s.clients.discard(client)
        pump.cancel()
        if s.exit_code is not None and not s.clients:
            svc.sessions.pop(s.id, None)
    return ws


def setup_sessions(app: web.Application, mlsweep_dir: Path, port: int, *, enabled: bool, notes: bool) -> None:
    """Register the terminal routes; start the service when *enabled*."""
    app.add_routes(sessions_routes)
    if not enabled:
        app[SESSIONS_KEY] = None
        return
    svc = SessionService(mlsweep_dir, port, app[TOKEN_KEY], notes)
    app[SESSIONS_KEY] = svc

    async def _start(_app: web.Application) -> None:
        svc.start()

    app.on_startup.append(_start)
    # on_shutdown, not on_cleanup: the server waits for open sockets between the two.
    app.on_shutdown.append(lambda _app: svc.stop())
