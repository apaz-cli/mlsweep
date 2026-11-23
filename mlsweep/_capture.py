"""Session capture: coding-agent traces, in the manager's dashboard.

Every coding agent writes a transcript of its session on disk.  The capture
watcher tails the transcripts of the sessions it follows, normalizes them to a
small event schema, and uploads each one to the manager, which serves it on
the dashboard's Traces tab next to the notes.  A trace answers "why was this
experiment run": the conversation that produced it, searchable later.

Which sessions
--------------
Only followed sessions are ever read.  A session is followed when it asks to
be (``mlsweep capture this``, run inside it) or, while capture is on, when it
submits or links an experiment (``mlsweep run``, ``mlsweep capture link``).
Every other transcript on the machine is never opened.  A followed session is
uploaded from its first line, so its trace holds the whole conversation.
Follow requests are files in ``<capture-dir>/follow/``, one per session, so a
client command never writes the watcher's own state.

What is captured (per event):

  * the user's prompts
  * the assistant's replies
  * thinking blocks
  * tool calls, as one summary line: the shell command for bash-like tools,
    paths and scalar arguments elsewhere — never a file's contents
  * tool results as status only (ok / error / duration), never their output

What is not captured: tool output, file contents, the system prompt.

Nothing is redacted and nothing is streamed: the watcher uploads a session's
new events in one batch right after it is followed, then whenever its
transcript has been quiet for a while, and again when the session closes.

Attribution
-----------
`mlsweep run` and friends want to link the experiment they submit to the agent
session they were typed into.  The harnesses tell their children who they are:
pi exports ``PI_SESSION_ID`` / ``PI_SESSION_FILE`` and Claude Code exports
``CLAUDE_CODE_SESSION_ID`` into every child process.  The client reads its own
environment, then its ``/proc`` ancestors' (for a command whose environment
was scrubbed).  Nothing is guessed: no variable means no session.
``MLSWEEP_SESSION_ID`` overrides everything.

Consent
-------
Capture is off until ``mlsweep capture on`` (or the first ``capture this``)
turns it on for this machine.  It writes the manager URL and token into
``<capture-dir>/on.json`` and starts the watcher, which follows nothing yet.  ``mlsweep capture off`` removes the credential, stops the watcher and
writes an ``off.json`` marker (who, when) that a running watcher checks every
loop — so "off" holds even if stopping the process failed.  Turning capture
back on replaces the marker; what was never uploaded stays local.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsweep._shared import _mlsweep_dir

# ── event schema ──────────────────────────────────────────────────────────────
#
# One normalized event, one line of JSON.  `seq` numbers events within a
# session, so an upload is idempotent on the manager: (session_id, seq).
#
#   {"seq": 12, "ts": 1730000000.5, "kind": "user", "text": "..."}
#   {"seq": 13, ..., "kind": "tool", "name": "bash", "text": "ls -la",
#    "tool_id": "call_00"}
#   {"seq": 14, ..., "kind": "tool_result", "name": "bash", "ok": true,
#    "duration_ms": 24}
#   {"seq": 15, ..., "kind": "meta", "text": "model → claude-sonnet-5"}
#
# `meta` is a note the agent's transcript carries (a model switch, say), shown
# as its own line in the ledger.
#
# A trace's id is the agent's own native session id (a UUID for both pi and
# Claude Code).  The agent is a column, not part of the id.


def _iso_to_epoch(ts: Any) -> float | None:
    """Parse an ISO-8601 timestamp (the agents' format) to epoch seconds."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        from datetime import datetime, timezone

        text = ts.strip().rstrip("Z")
        if "." in text:
            base, frac = text.split(".", 1)
            frac = re.sub(r"\D", "", frac)[:6]
            text = f"{base}.{frac}"
        dt = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


def _summarize_args(name: str, args: Any) -> str:
    """A tool call's arguments as one short line.

    Bash-like tools keep their full command (it is a shell command, the thing
    the design captures).  Everything else keeps scalar arguments, with long or
    multi-line strings — file contents, diffs — replaced by their length.
    """
    if not isinstance(args, dict):
        return str(args)[:200] if args is not None else ""
    for key in ("command", "cmd", "script"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= 400 else value[:399] + "…"
    parts: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, str):
            if len(value) <= 120 and "\n" not in value:
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}<{len(value)} chars>")
        elif isinstance(value, (bool, int, float)):
            parts.append(f"{key}={value}")
        elif isinstance(value, (list, dict)):
            parts.append(f"{key}<{len(value)}>")
    return " ".join(parts)[:400]


def _event(seq: int, ts: float | None, kind: str, **kw: Any) -> dict[str, Any]:
    event: dict[str, Any] = {"seq": seq, "ts": ts if ts is not None else time.time(), "kind": kind}
    for key, value in kw.items():
        if value is not None:
            event[key] = value
    return event


# ── pi ────────────────────────────────────────────────────────────────────────
#
# ~/.pi/agent/sessions/<encoded-cwd>/<ts>_<uuid>.jsonl.  Version 3: a
# `session` header line (id, cwd, timestamp), then `message` lines whose
# message.role is user / assistant / toolResult / system, with content blocks
# of type text / thinking / toolCall.  model_change lines carry the model.


def _normalize_pi_line(line: dict[str, Any], seq: int, meta: dict[str, Any]) -> list[dict[str, Any]]:
    kind = line.get("type")
    if kind == "session":
        meta["cwd"] = line.get("cwd") or meta.get("cwd")
        meta["started_at"] = _iso_to_epoch(line.get("timestamp")) or meta.get("started_at")
        if line.get("id"):
            meta["native_id"] = str(line["id"])
        return []
    if kind == "model_change":
        model = line.get("modelId") or meta.get("model")
        meta["model"] = model
        # A switch mid-session explains why later events name another model.
        return [_event(seq, _iso_to_epoch(line.get("timestamp")), "meta",
                       text=f"model → {model}")] if model else []
    if kind != "message":
        return []
    message = line.get("message") or {}
    role = message.get("role")
    ts = _iso_to_epoch(line.get("timestamp"))
    content = message.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    if role == "user":
        text = " ".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        if text.strip():
            return [_event(seq, ts, "user", text=text.strip())]
        return []
    if role == "assistant":
        events: list[dict[str, Any]] = []
        usage = message.get("usage") or {}
        common: dict[str, Any] = {
            "model": message.get("model") or meta.get("model"),
        }
        if usage.get("input") or usage.get("output"):
            common["tokens"] = {"input": usage.get("input"), "output": usage.get("output")}
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "thinking":
                events.append(_event(seq, ts, "thinking", text=block.get("thinking") or "", **common))
                seq += 1
            elif btype == "text":
                if (block.get("text") or "").strip():
                    events.append(_event(seq, ts, "assistant", text=block["text"], **common))
                    seq += 1
            elif btype == "toolCall":
                name = block.get("name") or "tool"
                events.append(_event(
                    seq, ts, "tool", name=name, tool_id=block.get("id"),
                    text=_summarize_args(name, block.get("arguments")), **common))
                seq += 1
        return events
    if role == "toolResult":
        # Status only: the result's text never leaves the machine.
        return [_event(
            seq, ts, "tool_result", name=message.get("toolName"),
            tool_id=message.get("toolCallId"), ok=not message.get("isError", False),
            duration_ms=message.get("durationMs"))]
    return []  # the system preamble, and anything else


# ── Claude Code ───────────────────────────────────────────────────────────────
#
# ~/.claude/projects/<encoded-cwd>/<uuid>.jsonl.  Lines of many types; user and
# assistant lines carry an Anthropic-API-shaped message, summary lines carry a
# title, every line a sessionId (and most a cwd).


_CLAUDE_COMMAND_RE = re.compile(r"<command-name>\s*(.*?)\s*</command-name>", re.DOTALL)
_CLAUDE_ARGS_RE = re.compile(r"<command-args>\s*(.*?)\s*</command-args>", re.DOTALL)


def _claude_prompt_text(text: str, line: dict[str, Any]) -> str:
    """What the user typed, from a Claude Code user text block.

    A slash command arrives wrapped in tags and becomes ``/name args``.  Its
    local output (``<local-command-stdout>``) and the harness's own tagged
    notes are not prompts: they are dropped, like other tool output.
    """
    if (line.get("origin") or {}).get("kind") == "human":
        return text
    command = _CLAUDE_COMMAND_RE.search(text)
    if command:
        args = _CLAUDE_ARGS_RE.search(text)
        return " ".join(p for p in (command.group(1), args.group(1) if args else "") if p)
    if text.startswith("<") and re.match(r"<[a-z][a-z0-9-]*>", text):
        return ""
    return text


def _normalize_claude_line(line: dict[str, Any], seq: int, meta: dict[str, Any]) -> list[dict[str, Any]]:
    kind = line.get("type")
    if line.get("sessionId"):
        meta["native_id"] = str(line["sessionId"])
    if line.get("cwd") and not meta.get("cwd"):
        meta["cwd"] = str(line["cwd"])
    ts = _iso_to_epoch(line.get("timestamp"))
    if kind == "summary":
        meta["title"] = line.get("summary") or meta.get("title")
        return []
    if kind == "attachment":
        snapshot = (line.get("attachment") or {}).get("snapshot") or {}
        if snapshot.get("workingDirectory") and not meta.get("cwd"):
            meta["cwd"] = str(snapshot["workingDirectory"])
        return []
    if kind not in ("user", "assistant"):
        return []
    message = line.get("message") or {}
    content = message.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    if kind == "user":
        events = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                events.append(_event(
                    seq, ts, "tool_result", tool_id=block.get("tool_use_id"),
                    ok=not block.get("is_error", False)))
                seq += 1
            elif block.get("type") == "text" and (block.get("text") or "").strip():
                if (line.get("origin") or {}).get("kind") != "human" and line.get("isMeta") is True:
                    continue  # tool results replayed as user messages
                text = _claude_prompt_text(block["text"].strip(), line)
                if text:
                    events.append(_event(seq, ts, "user", text=text))
                    seq += 1
        return events
    events = []
    usage = message.get("usage") or {}
    common: dict[str, Any] = {"model": message.get("model") or meta.get("model")}
    if usage.get("input_tokens") or usage.get("output_tokens"):
        common["tokens"] = {"input": usage.get("input_tokens"), "output": usage.get("output_tokens")}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            events.append(_event(seq, ts, "thinking", text=block.get("thinking") or "", **common))
            seq += 1
        elif btype == "text" and (block.get("text") or "").strip():
            events.append(_event(seq, ts, "assistant", text=block["text"], **common))
            seq += 1
        elif btype == "tool_use":
            name = block.get("name") or "tool"
            events.append(_event(
                seq, ts, "tool", name=name, tool_id=block.get("id"),
                text=_summarize_args(name, block.get("input")), **common))
            seq += 1
    return events


# ── agents ────────────────────────────────────────────────────────────────────

AGENT_NAMES = ("pi", "claude_code")
_NORMALIZERS: dict[str, Callable[[dict[str, Any], int, dict[str, Any]], list[dict[str, Any]]]] = {
    "pi": _normalize_pi_line,
    "claude_code": _normalize_claude_line,
}


@dataclass
class AgentSpec:
    """Where an agent writes transcripts, and how to read them."""

    name: str          # canonical agent name: pi, claude_code
    directory: Path    # sessions live under this, in per-project folders
    normalize: Any     # (line, seq, meta) -> events; meta is mutated in place

    def sessions(self) -> list[Path]:
        return sorted(self.directory.glob("*/*.jsonl")) if self.directory.is_dir() else []


def _sessions_root(agent: str) -> Path:
    home = Path.home()
    if agent == "pi":
        return home / ".pi" / "agent" / "sessions"
    return home / ".claude" / "projects"


def agent_specs(agents: list[str] | None = None,
                root: Path | None = None) -> list[AgentSpec]:
    """The agents this machine's capture watches, with their normalizers.

    *root* replaces where every agent's transcripts live (tests use it to
    point the scan at a scratch directory).
    """
    normalizers: list[AgentSpec] = []
    for name in agents or AGENT_NAMES:
        if name in _NORMALIZERS:
            normalizers.append(AgentSpec(
                name=name,
                directory=root if root is not None else _sessions_root(name),
                normalize=_NORMALIZERS[name],
            ))
    return normalizers


def session_id_for(agent: str, native_id: str) -> str:
    """A trace's id: the agent's own native session id, unchanged."""
    return str(native_id)


# ── state ─────────────────────────────────────────────────────────────────────


def capture_dir() -> Path:
    return _mlsweep_dir() / "capture"


@dataclass
class SessionState:
    """One transcript this machine knows about."""

    session_id: str
    agent: str
    path: str
    native_id: str
    cwd: str = ""
    title: str = ""
    model: str = ""
    started_at: float = 0.0
    last_seen_at: float = 0.0
    offset: int = 0          # transcript bytes read so far
    uploaded_seq: int = 0    # highest seq the manager has
    last_upload_at: float = 0.0
    closed: bool = False


def _native_id_from_path(agent: str, path: Path) -> str:
    stem = path.stem
    if agent == "pi":
        return stem.split("_", 1)[-1] if "_" in stem else stem
    return stem


def find_transcript(specs: list[AgentSpec], session_id: str) -> tuple[AgentSpec, Path] | None:
    """The transcript file a session id names, and the agent that wrote it."""
    if not session_id or "/" in session_id or session_id.startswith("."):
        return None
    for spec in specs:
        if not spec.directory.is_dir():
            continue
        pattern = f"*/*_{session_id}.jsonl" if spec.name == "pi" else f"*/{session_id}.jsonl"
        for path in sorted(spec.directory.glob(pattern)):
            return spec, path
    return None


def follow_dir() -> Path:
    return capture_dir() / "follow"


def request_follow(session_id: str, directory: Path | None = None, *, why: str = "") -> Path:
    """Ask the watcher to follow a session: one small file per session.

    Client commands write here and nowhere else, so they never race the
    watcher's own state file.
    """
    root = directory or follow_dir()
    root.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in session_id)[:150]
    path = root / f"{safe}.json"
    if not path.exists():
        path.write_text(json.dumps({
            "session_id": session_id, "why": why, "at": time.time(),
            "by": f"{os.getenv('USER', 'user')}@{socket.gethostname()}",
        }))
    return path


def followed_requests(directory: Path | None = None) -> list[dict[str, Any]]:
    root = directory or follow_dir()
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict) and raw.get("session_id"):
            out.append(raw)
    return out


class CaptureClient:
    """The watcher's state: the sessions it follows, and what was uploaded.

    Everything survives a restart in ``followed.json``; event batches not yet
    accepted sit in ``pending/<session_id>.jsonl``.  Only sessions with a
    follow request (``<dir>/follow/``) are ever read.
    """

    IDLE_SECONDS = 60.0        # a quiet transcript is ready to upload
    CLOSE_SECONDS = 30 * 60.0  # quiet this long, a session is finished
    POLL_SECONDS = 2.0
    FORCE_SECONDS = 600.0      # upload at least this often while a session lives

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or capture_dir()
        self.pending_dir = self.dir / "pending"
        self.follow_dir = self.dir / "follow"
        self.state_path = self.dir / "followed.json"
        self.sessions: dict[str, SessionState] = {}

    # ── persistence ──────────────────────────────────────────────────

    def load(self) -> None:
        try:
            raw = json.loads(self.state_path.read_text())
            self.sessions = {s["session_id"]: SessionState(**s) for s in raw.get("sessions", [])}
        except (OSError, ValueError, TypeError, KeyError):
            self.sessions = {}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"sessions": [vars(s) for s in self.sessions.values()]}))
        os.replace(tmp, self.state_path)

    def pending_path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in session_id)[:150]
        return self.pending_dir / f"{safe}.jsonl"

    def pending_count(self, session_id: str) -> int:
        try:
            with self.pending_path(session_id).open() as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    # ── following ────────────────────────────────────────────────────

    def follow(self, specs: list[AgentSpec], session_id: str) -> SessionState | None:
        """Start following a session, if its transcript can be found."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        found = find_transcript(specs, session_id)
        if found is None:
            return None  # not written yet; the request stays and is retried
        spec, path = found
        stat = path.stat()
        state = SessionState(
            session_id=session_id, agent=spec.name, path=str(path),
            native_id=_native_id_from_path(spec.name, path),
            started_at=stat.st_ctime, last_seen_at=stat.st_mtime,
        )
        self.sessions[session_id] = state
        return state

    def follow_requested(self, specs: list[AgentSpec]) -> None:
        for request in followed_requests(self.follow_dir):
            self.follow(specs, str(request["session_id"]))

    # ── transcript reading ───────────────────────────────────────────

    def scan(self, specs: list[AgentSpec]) -> None:
        """Read what followed transcripts gained since the last scan."""
        by_name = {spec.name: spec for spec in specs}
        for state in self.sessions.values():
            spec = by_name.get(state.agent)
            if spec is None or state.closed:
                continue
            path = Path(state.path)
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size < state.offset:  # truncated or replaced: read it again
                state.offset = 0
            state.last_seen_at = stat.st_mtime
            if stat.st_size <= state.offset:
                continue
            with path.open("rb") as f:
                f.seek(state.offset)
                raw = f.read(stat.st_size - state.offset)
            # Only whole lines: a line the agent is still writing waits.
            cut = raw.rfind(b"\n") + 1
            if not cut:
                continue
            state.offset += cut
            self._consume(state, spec, raw[:cut])

    def _consume(self, state: SessionState, spec: AgentSpec, raw: bytes) -> None:
        meta: dict[str, Any] = {
            "cwd": state.cwd, "title": state.title, "model": state.model,
            "started_at": state.started_at, "native_id": state.native_id,
        }
        events: list[dict[str, Any]] = []
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            events.extend(spec.normalize(parsed, 0, meta))
        self._apply_meta(state, meta)
        if not state.model:
            state.model = next((str(e["model"]) for e in events if e.get("model")), "")
        if not state.title:
            state.title = title_from_events(events)
        if not events:
            return
        # Renumber across reads: seq continues from uploaded + still-pending.
        seq = state.uploaded_seq + self.pending_count(state.session_id) + 1
        for event in events:
            event["seq"] = seq
            seq += 1
        self._spool(state, events)

    def _apply_meta(self, state: SessionState, meta: dict[str, Any]) -> None:
        for key in ("cwd", "title", "model", "native_id"):
            value = meta.get(key)
            if value and not getattr(state, key):
                setattr(state, key, str(value))
        if meta.get("started_at"):
            state.started_at = state.started_at or meta["started_at"]

    def _spool(self, state: SessionState, events: list[dict[str, Any]]) -> None:
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        with self.pending_path(state.session_id).open("a") as f:
            for event in events:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")

    # ── upload scheduling ─────────────────────────────────────────────

    def due_sessions(self, now: float) -> list[SessionState]:
        """Sessions with pending events that are quiet enough to upload.

        A session never uploaded is due at once, so following one shows its
        trace within a poll or two.
        """
        out = []
        for state in self.sessions.values():
            if state.closed:
                continue
            if not self.pending_count(state.session_id):
                continue
            idle = now - state.last_seen_at
            if idle >= self.IDLE_SECONDS or now - state.last_upload_at >= self.FORCE_SECONDS:
                out.append(state)
        return out

    def mark_uploaded(self, state: SessionState, count: int, now: float | None = None) -> None:
        state.uploaded_seq += count
        state.last_upload_at = now if now is not None else time.time()
        try:
            self.pending_path(state.session_id).unlink()
        except OSError:
            pass

    def mark_closed(self, state: SessionState) -> None:
        state.closed = True


def title_from_events(events: list[dict[str, Any]]) -> str:
    """A trace's title: its first prompt that is not a slash command, one line."""
    for event in events:
        text = event.get("text") or ""
        if event.get("kind") == "user" and text and not text.startswith("/"):
            return " ".join(text.split())[:80]
    return ""


# ── attribution: which agent session is this command running in? ─────────────


_SESSION_ENV: dict[str, tuple[str, ...]] = {
    "pi": ("PI_SESSION_ID", "PI_SESSION_FILE"),
    "claude_code": ("CLAUDE_CODE_SESSION_ID",),
}


def _native_from_env(agent: str, key: str, value: str) -> str:
    if agent == "pi" and key == "PI_SESSION_FILE":
        return _native_id_from_path("pi", Path(value))
    return value


def _from_env(env: dict[str, str]) -> str | None:
    """The session id the agent's own environment names, if any."""
    if env.get("MLSWEEP_SESSION_ID"):
        return env["MLSWEEP_SESSION_ID"]
    for agent, keys in _SESSION_ENV.items():
        for key in keys:
            value = env.get(key)
            if value:
                return session_id_for(agent, _native_from_env(agent, key, value))
    return None


def _read_proc(pid: int, path: str, limit: int = 65536) -> bytes | None:
    try:
        with open(f"/proc/{pid}/{path}", "rb") as f:
            return f.read(limit)
    except OSError:
        return None


def _ancestors() -> list[int]:
    """This process's ancestor pids, nearest first."""
    out: list[int] = []
    pid = os.getpid()
    for _ in range(64):
        stat = _read_proc(pid, "stat")
        if stat is None:
            break
        try:
            parent = int(stat.rpartition(b")")[2].split()[1])
        except (IndexError, ValueError):
            break
        if parent <= 1:
            break
        out.append(parent)
        pid = parent
    return out


def _ancestor_environ_session(ancestors: list[int] | None = None) -> str | None:
    """Walk up the process tree; an agent ancestor names itself in its environ."""
    for pid in _ancestors() if ancestors is None else ancestors:
        raw = _read_proc(pid, "environ")
        if raw is None:
            continue
        env: dict[str, str] = {}
        for entry in raw.split(b"\0"):
            if b"=" in entry:
                key, _, value = entry.partition(b"=")
                env[key.decode(errors="replace")] = value.decode(errors="replace")
        env.pop("MLSWEEP_SESSION_ID", None)
        found = _from_env(env)
        if found:
            return found
    return None


def resolve_capture_session(env: dict[str, str] | None = None, *,
                            ancestors: list[int] | None = None) -> str | None:
    """The agent session the current command runs in, or None.

    In order: the MLSWEEP_SESSION_ID override, then the session variables the
    agent harnesses export (our own environment, then our ancestors', for a
    command whose environment was scrubbed).  Exact or nothing: two agents in
    one directory are two sessions, and a human's shell names none.
    *ancestors* overrides the /proc walk (tests use it to stay off the real
    machine).
    """
    env = env if env is not None else dict(os.environ)
    direct = _from_env(env)
    if direct:
        return direct
    return _ancestor_environ_session(ancestors)


def capture_is_on() -> bool:
    return _on_path().exists() and not _off_path().exists()


def _same_manager(a: str, b: str) -> bool:
    """Whether two manager URLs name the same manager (localhost spellings agree)."""
    from urllib.parse import urlsplit

    def key(url: str) -> tuple[str, str, int | None]:
        parts = urlsplit(url.strip().rstrip("/"))
        host = (parts.hostname or "").lower()
        if host in ("127.0.0.1", "::1", "0.0.0.0"):
            host = "localhost"
        return parts.scheme or "http", host, parts.port
    return key(a) == key(b)


def follow_if_on(session_id: str | None, *, why: str, manager: str) -> bool:
    """Follow a session that submitted or linked something, if capture is on.

    Only for the manager capture uploads to: a sweep sent elsewhere (a test
    manager, another cluster) has no business pulling its session onto this
    one.  Best effort, like attribution: a submission never fails over capture.
    """
    if not session_id or not capture_is_on():
        return False
    try:
        on_file = json.loads(_on_path().read_text())
        if not _same_manager(str(on_file.get("manager", "")), manager):
            return False
        request_follow(session_id, why=why)
        ensure_watcher()
        return True
    except Exception:
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────


def _on_path() -> Path:
    return capture_dir() / "on.json"


def _off_path() -> Path:
    return capture_dir() / "off.json"


def watcher_pid() -> int | None:
    """The running watcher's pid, or None (including a stale record)."""
    try:
        pid = int(json.loads((capture_dir() / "watcher.json").read_text())["pid"])
    except (OSError, ValueError, KeyError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return pid


def _read_off_marker() -> dict[str, Any] | None:
    try:
        raw = json.loads(_off_path().read_text())
        return raw if isinstance(raw, dict) else None
    except (OSError, ValueError):
        return None


def _cmd(argv: list[str]) -> None:
    """`mlsweep capture`: this / on / off / watch / status / ls / link / import."""
    import argparse
    import sys


    parser = argparse.ArgumentParser(
        prog="mlsweep capture",
        description="Session capture: upload the coding-agent sessions you choose to the "
                    "manager's Traces tab. Off by default; nothing else is read.",
    )
    sub = parser.add_subparsers(dest="action")

    this = sub.add_parser("this", help="Upload the agent session this command runs in")
    this.add_argument("--session", help="Trace id (default: the session this command runs in)")
    this.add_argument("--manager", help="Manager URL, if this turns capture on "
                                        "(default: http://localhost:7891)")
    this.add_argument("--token", help="Manager token, if this turns capture on")

    on = sub.add_parser("on", help="Turn capture on: sessions that submit sweeps are uploaded")
    on.add_argument("--manager", help="Manager URL (default: http://localhost:7891)")
    on.add_argument("--token", help="Manager token (default: ~/.mlsweep/manager.token)")
    on.add_argument("--agents", default=",".join(AGENT_NAMES),
                    help="Agents to follow, comma-separated (default: pi,claude_code)")
    on.add_argument("--foreground", action="store_true",
                    help="Run the watcher in this terminal instead of the background")

    sub.add_parser("off", help="Turn capture off and stop the watcher")
    sub.add_parser("watch", help="Run the watcher in the foreground (used by `on`)")

    status = sub.add_parser("status", help="Whether capture is on, and what it follows")
    status.add_argument("--json", action="store_true", dest="as_json")

    ls = sub.add_parser("ls", aliases=["list"], help="Traces on the manager, newest first")
    ls.add_argument("--agent")
    ls.add_argument("--state", choices=["active", "closed", "imported", "all"], default="all")
    ls.add_argument("--limit", type=int, default=20)
    ls.add_argument("--manager")
    ls.add_argument("--token")
    ls.add_argument("--json", action="store_true", dest="as_json")

    link = sub.add_parser("link", help="Link a trace to an experiment")
    link.add_argument("experiment_id")
    link.add_argument("--session", help="Trace id (default: the session this command runs in)")
    link.add_argument("--manager")
    link.add_argument("--token")

    imp = sub.add_parser("import", help="Upload transcripts you already have (backfill)")
    imp.add_argument("--days", type=int, default=30, help="Look this many days back (default: 30)")
    imp.add_argument("--all", action="store_true", help="Every transcript, however old")
    imp.add_argument("--agents", default=",".join(AGENT_NAMES))
    imp.add_argument("--dry-run", action="store_true", help="List what would be uploaded")
    imp.add_argument("--manager")
    imp.add_argument("--token")

    args = parser.parse_args(argv)
    action = args.action or "status"
    if action == "this":
        _cmd_this(args, sys)
    elif action == "on":
        _cmd_on(args, sys)
    elif action == "off":
        _cmd_off(sys)
    elif action == "watch":
        _cmd_watch()
    elif action == "status":
        _cmd_status(args, sys)
    elif action in ("ls", "list"):
        _cmd_ls(args, sys)
    elif action == "link":
        _cmd_link(args, sys)
    elif action == "import":
        _cmd_import(args, sys)


def _cmd_status(args: Any, sys: Any) -> None:
    from mlsweep._colors import _DIM, _GREEN, _RESET, _YELLOW
    from mlsweep._jsonout import emit

    on_file = _on_path()
    pid = watcher_pid()
    followed = followed_requests()
    if args.as_json:
        emit({
            "state": "on" if capture_is_on() else "off",
            "off_marker": _read_off_marker(),
            "watcher_pid": pid,
            "credential": str(_on_path()) if on_file.exists() else None,
            "followed": followed,
        })
        return
    if capture_is_on():
        print(f"{_GREEN}on{_RESET} — following {len(followed)} session(s), "
              f"watcher pid {pid or '- (not running)'}")
        for entry in followed[-10:]:
            print(f"  {_DIM}{entry.get('session_id', '?')}  {entry.get('why', '')}{_RESET}")
    else:
        print(f"{_YELLOW}off{_RESET} — traces are not captured on this machine")
    marker = _read_off_marker()
    if marker:
        print(f"  {_DIM}turned off by {marker.get('by', '?')} at "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(marker.get('at', 0)))}{_RESET}")


def _turn_on(manager: str | None, token_arg: str | None, agents: list[str], sys: Any) -> str:
    """Write the credential that makes capture on; the manager URL it names."""
    from mlsweep._colors import _RED, _RESET
    from mlsweep.run_sweep import _manager_url, _resolve_token

    url = _manager_url(manager or "http://localhost:7891", "")
    token = _resolve_token(token_arg)
    if not token:
        print(f"{_RED}No token: pass --token, or start the manager first "
              f"(~/.mlsweep/manager.token).{_RESET}", file=sys.stderr)
        raise SystemExit(2)
    # The credential answers "is this machine's capture on".  Its absence,
    # with an off marker, is what the watcher checks every loop.
    capture_dir().mkdir(parents=True, exist_ok=True)
    _on_path().write_text(json.dumps({
        "manager": url, "token": token, "agents": agents,
        "by": f"{os.getenv('USER', 'user')}@{socket.gethostname()}", "at": time.time(),
    }))
    os.chmod(_on_path(), 0o600)
    _off_path().unlink(missing_ok=True)
    return url


def ensure_watcher() -> int:
    """The watcher's pid, starting it in the background if it is not running."""
    import subprocess

    pid = watcher_pid()
    if pid:
        return pid
    log = capture_dir() / "watcher.log"
    with log.open("ab") as f:
        proc = subprocess.Popen(
            ["mlsweep", "capture", "watch"], stdout=f, stderr=f,
            stdin=subprocess.DEVNULL, start_new_session=True)
    (capture_dir() / "watcher.json").write_text(json.dumps({"pid": proc.pid}))
    return proc.pid


def _cmd_on(args: Any, sys: Any) -> None:
    from mlsweep._colors import _DIM, _GREEN, _RESET

    url = _turn_on(args.manager, args.token, args.agents.split(","), sys)
    print(f"{_GREEN}on{_RESET} — uploading to {url} the sessions that submit sweeps "
          "or run `mlsweep capture this`")
    print(f"  {_DIM}Nothing else is read. mlsweep capture off turns it back off.{_RESET}")
    if args.foreground:
        _cmd_watch()
    else:
        print(f"  watcher pid {ensure_watcher()}")


def _cmd_this(args: Any, sys: Any) -> None:
    from mlsweep._colors import _DIM, _GREEN, _RED, _RESET

    session_id = args.session or resolve_capture_session()
    if not session_id:
        print(f"{_RED}No agent session here: run this inside the agent, or pass "
              f"--session.{_RESET}", file=sys.stderr)
        raise SystemExit(2)
    marker = _read_off_marker()
    if marker and not _on_path().exists():
        print(f"{_RED}Capture was turned off on this machine by {marker.get('by', '?')}. "
              f"`mlsweep capture on` turns it back on.{_RESET}", file=sys.stderr)
        raise SystemExit(2)
    if not _on_path().exists():
        url = _turn_on(args.manager, args.token, list(AGENT_NAMES), sys)
        print(f"{_GREEN}on{_RESET} — capture is now on for this machine ({url})")
    if find_transcript(agent_specs(), session_id) is None:
        print(f"{_DIM}No transcript for {session_id} yet; it uploads once the agent "
              f"writes one.{_RESET}")
    request_follow(session_id, why="capture this")
    pid = ensure_watcher()
    print(f"Following {session_id}: the whole session uploads within a few seconds, "
          f"and new turns after each quiet minute (watcher pid {pid}).")


def _cmd_off(sys: Any) -> None:
    import signal

    from mlsweep._colors import _RESET, _YELLOW

    capture_dir().mkdir(parents=True, exist_ok=True)
    _off_path().write_text(json.dumps({
        "by": f"{os.getenv('USER', 'user')}@{socket.gethostname()}", "at": time.time(),
    }))
    _on_path().unlink(missing_ok=True)
    pid = watcher_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    print(f"{_YELLOW}off{_RESET} — capture stopped on this machine. Traces already "
          "uploaded stay on the manager.")


def _cmd_watch() -> None:
    """The watcher loop: follow requested sessions, spool events, upload when quiet."""
    import signal


    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    client = CaptureClient()
    client.load()
    last_sweep = 0.0

    while True:
        if _off_path().exists() or not _on_path().exists():
            return  # off wins over on, whoever wrote it
        try:
            on_file = json.loads(_on_path().read_text())
        except (OSError, ValueError):
            return
        manager, token = on_file["manager"], on_file["token"]
        now = time.time()
        specs = agent_specs(list(on_file.get("agents") or AGENT_NAMES))
        client.follow_requested(specs)
        client.scan(specs)
        for state in client.due_sessions(now):
            _upload(client, state, manager, token)
        if now - last_sweep > 300:
            last_sweep = now
            # A transcript quiet this long is finished: upload what is left
            # and tell the manager, once.
            for state in list(client.sessions.values()):
                if not state.closed and now - state.last_seen_at > client.CLOSE_SECONDS:
                    state.closed = True
                if state.closed and client.pending_count(state.session_id):
                    _upload(client, state, manager, token)
        client.save()
        time.sleep(client.POLL_SECONDS)


def _upload(client: CaptureClient, state: SessionState, manager: str, token: str) -> None:
    from mlsweep.run_sweep import _http_request

    try:
        events: list[dict[str, Any]] = []
        path = client.pending_path(state.session_id)
        if path.exists():
            with path.open() as f:
                events = [json.loads(line) for line in f if line.strip()]
        meta = {
            "session_id": state.session_id, "agent": state.agent,
            "host": socket.gethostname(), "user": os.getenv("USER", "user"),
            "cwd": state.cwd, "title": state.title, "model": state.model,
            "started_at": state.started_at, "last_seen_at": state.last_seen_at,
            "state": "closed" if state.closed else "active",
        }
        status, resp = _http_request(
            "POST", f"{manager.rstrip('/')}/api/capture/sessions", token,
            json_data={"session": meta, "events": events})
        if status in (200, 201):
            client.mark_uploaded(state, len(events))
            if state.closed:
                _http_request(
                    "POST", f"{manager.rstrip('/')}/api/capture/sessions/{state.session_id}/close",
                    token, json_data={})
        else:
            print(f"capture: upload failed for {state.session_id}: {status} {resp}", flush=True)
    except Exception as exc:  # a manager that is down is not a crash
        print(f"capture: upload failed for {state.session_id}: {exc}", flush=True)


# ── ls / link / import ────────────────────────────────────────────────────────


def _cmd_ls(args: Any, sys: Any) -> None:
    from mlsweep._jsonout import emit
    from mlsweep.run_sweep import _http_request, _manager_url, _resolve_token

    url = _manager_url(args.manager or "http://localhost:7891", "/api/capture")
    token = _resolve_token(args.token)
    if not token:
        print("No token: start the manager or pass --token.", file=sys.stderr)
        raise SystemExit(2)
    params = f"?limit={args.limit}&state={args.state}" + (f"&agent={args.agent}" if args.agent else "")
    status, resp = _http_request("GET", f"{url}/sessions{params}", token)
    if status != 200:
        print(f"Manager answered {status}: {resp}", file=sys.stderr)
        raise SystemExit(1)
    sessions = resp if isinstance(resp, list) else resp.get("sessions", [])
    if args.as_json:
        emit({"sessions": sessions})
        return
    for s in sessions:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("last_seen_at") or 0))
        title = " ".join(str(s.get("title") or s.get("cwd") or "?").split())
        links = ", ".join(s.get("experiments") or [])
        print(f"{when}  {s.get('agent', '?'):12}  {s.get('state', '?'):8}  "
              f"{s.get('session_id', ''):40}  {str(title)[:60]}"
              + (f"  → {links}" if links else ""))


def _cmd_link(args: Any, sys: Any) -> None:
    from mlsweep.run_sweep import _http_request, _manager_url, _resolve_token

    url = _manager_url(args.manager or "http://localhost:7891", "/api/capture")
    token = _resolve_token(args.token)
    if not token:
        print("No token: start the manager or pass --token.", file=sys.stderr)
        raise SystemExit(2)
    session_id = args.session or resolve_capture_session()
    if not session_id:
        print("No agent session here: run this inside the agent, or pass --session.",
              file=sys.stderr)
        raise SystemExit(2)
    status, resp = _http_request("POST", f"{url}/link", token, json_data={
        "session_id": session_id, "experiment_id": args.experiment_id, "source": "manual"})
    if status in (200, 201):
        print(f"Linked {session_id} → {args.experiment_id}")
        follow_if_on(session_id, why=f"linked {args.experiment_id}", manager=url)
    else:
        print(f"Manager answered {status}: {resp}", file=sys.stderr)
        raise SystemExit(1)


def read_transcript(spec: AgentSpec, path: Path) -> tuple[str, list[dict[str, Any]]]:
    """A whole transcript file, normalized and numbered: (session_id, events)."""
    meta: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    try:
        with path.open("rb") as f:
            raw = f.read(256 * 1024 * 1024)
    except OSError:
        return "", []
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        events.extend(spec.normalize(parsed, 0, meta))
    for seq, event in enumerate(events, start=1):
        event["seq"] = seq
    native_id = str(meta.get("native_id") or _native_id_from_path(spec.name, path))
    return session_id_for(spec.name, native_id), events


def session_meta(spec: AgentSpec, path: Path, session_id: str,
                 events: list[dict[str, Any]]) -> dict[str, Any]:
    """The session header the manager stores for a whole-transcript import."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = time.time()
    first_ts = events[0]["ts"] if events else mtime
    last_ts = events[-1]["ts"] if events else first_ts
    title = next((e["text"] for e in events if e["kind"] == "user" and e.get("text")), "")[:80]
    model = next((e.get("model") for e in events if e.get("model")), "")
    cwd = ""
    return {
        "session_id": session_id, "agent": spec.name, "host": socket.gethostname(),
        "user": os.getenv("USER", "user"), "cwd": cwd, "title": title, "model": model,
        "started_at": first_ts, "last_seen_at": last_ts,
    }


def _cmd_import(args: Any, sys: Any) -> None:
    """Convert existing transcripts into traces on the manager, marked imported."""
    from mlsweep._colors import _RED, _RESET
    from mlsweep.run_sweep import _http_request, _manager_url, _resolve_token

    url = _manager_url(args.manager or "http://localhost:7891", "/api/capture")
    token = _resolve_token(args.token)
    if not token:
        print("No token: start the manager or pass --token.", file=sys.stderr)
        raise SystemExit(2)
    cutoff = time.time() - (args.days * 86400) if not args.all else 0
    directory = capture_dir()
    directory.mkdir(parents=True, exist_ok=True)
    imported_path = directory / "imported.json"
    try:
        imported: dict[str, float] = json.loads(imported_path.read_text())
    except (OSError, ValueError):
        imported = {}
    count = 0
    for spec in agent_specs(args.agents.split(",")):
        for path in spec.sessions():
            key = str(path)
            if key in imported:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    continue
            except OSError:
                continue
            if args.dry_run:
                print(f"would import {path}")
                continue
            session_id, events = read_transcript(spec, path)
            if not events:
                imported[key] = time.time()  # empty: never look again
                continue
            meta = session_meta(spec, path, session_id, events)
            meta["state"] = "imported"
            status, resp = _http_request("POST", f"{url}/sessions", token, json_data={
                "session": meta, "events": events})
            if status in (200, 201):
                count += 1
                imported[key] = time.time()
                print(f"imported {path} → {session_id} ({len(events)} events)")
            else:
                print(f"{_RED}failed {path}: {status} {resp}{_RESET}", file=sys.stderr)
    if not args.dry_run:
        imported_path.write_text(json.dumps(imported))
        print(f"Imported {count} transcript(s).")
