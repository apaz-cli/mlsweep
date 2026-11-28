"""Reading traces as text: references, event kinds, the ledger, ATIF.

This is the terminal's view of what the Traces tab shows.  ``mlsweep capture
show`` prints a trace as a compact ledger, one line per event, within a size
budget so an agent can read a long session in pages.  The same kinds filter it
(user, assistant, thinking, tools, mlsweep) and the same references name a trace
and an event (``mlsweep:trace/<id>?kind=<kind>#E<seq>``), so a link copied from
a note works here unchanged.

``to_atif`` converts a trace to ATIF, the Agent Trajectory Interchange Format
(https://docs.harborframework.com/agents/atif), so tools outside mlsweep can
read it.

Nothing here talks to the manager or imports the server: the CLI and the
manager's routes both use it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

# What a trace link may filter its trace to, as the Traces tab's kind buttons do.
TRACE_LINK_KINDS = ("user", "assistant", "thinking", "tools", "mlsweep")
_TRACE_EVENT_RE = re.compile(r"E(\d{1,12})")

# A tool call that runs the mlsweep CLI.  The same pattern, character for
# character, as MLSWEEP_RE in webui/page-traces.js (a test keeps them equal), so
# the terminal and the Traces tab agree on what counts as an mlsweep command.
MLSWEEP_PATTERN = (
    r"(?:^|[;&|(`\n]\s*|\$\(\s*)(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    r"(?:(?:sudo|env|nohup|uv\s+run|timeout\s+\S+)\s+)*(?:[^\s=]*\/)?"
    r"(?:mlsweep(?=\s|$)|python[0-9.]*\s+-m\s+mlsweep\b)"
)
MLSWEEP_RE = re.compile(MLSWEEP_PATTERN)
_STAGE_END_RE = re.compile(r"\s*(?:;|&&|\|\||\||\n|\)|`)")
_REDIRECT_RE = re.compile(r"\s+\d*(?:>>?|<)&?\s*\S+")

# A budget is counted in tokens, estimated at four characters each.
CHARS_PER_TOKEN = 4
DEFAULT_BUDGET = 2000


# ── references ───────────────────────────────────────────────────────────────


def parse_trace_ref(ref: str) -> tuple[str, str, int | None]:
    """``[mlsweep:]trace/<id>[?kind=<kind>][#E<seq>]`` -> (id, kind, seq).

    A bare id works too, as does ``<id>#E<seq>``.  The id may be a unique
    prefix.  ``kind`` is "" when the link shows every event, and ``seq`` None
    when it opens no particular event.  Raises ValueError, worded for whoever
    wrote the reference, on anything else.
    """
    from mlsweep._manager_db import validate_capture_id

    body = ref.strip()
    if body.startswith("mlsweep:"):
        body = body[len("mlsweep:"):]
    if body.startswith("trace/"):
        body = body[len("trace/"):]
    body, _, fragment = body.partition("#")
    session_id, _, query = body.partition("?")
    kind = ""
    for part in filter(None, query.split("&")):
        key, _, value = part.partition("=")
        if key != "kind" or value not in TRACE_LINK_KINDS:
            raise ValueError(f"`?{part}` is not a trace option; write ?kind= with one of "
                             + ", ".join(TRACE_LINK_KINDS))
        kind = value
    seq = None
    if fragment:
        m = _TRACE_EVENT_RE.fullmatch(fragment)
        if m is None:
            raise ValueError(f"`#{fragment}` does not name an event; write #E and its number, "
                             "as the Traces tab's Copy note link does")
        seq = int(m.group(1))
    validate_capture_id(session_id)
    return session_id, kind, seq


def trace_ref(session_id: str, seq: int | None = None, kind: str = "") -> str:
    """The note reference for a trace, one kind of its events, or one event."""
    return f"mlsweep:trace/{session_id}" + (f"?kind={kind}" if kind else "") + (
        f"#E{seq}" if seq is not None else "")


# ── kinds ────────────────────────────────────────────────────────────────────


def is_mlsweep(event: dict[str, Any]) -> bool:
    """Whether *event* is a tool call that runs mlsweep."""
    return event.get("kind") == "tool" and MLSWEEP_RE.search(str(event.get("text") or "")) is not None


def mlsweep_calls(text: str, most: int = 4) -> list[str]:
    """The mlsweep invocations in a command, each to the end of its pipeline
    stage, redirections dropped: what an mlsweep row shows."""
    out: list[str] = []
    for m in MLSWEEP_RE.finditer(text):
        start = m.end() - len("mlsweep")
        rest = text[start:]
        stop = _STAGE_END_RE.search(rest)
        call = _REDIRECT_RE.sub("", rest[:stop.start()] if stop else rest).strip()
        out.append(call if len(call) <= 160 else call[:159] + "…")
        if len(out) >= most:
            break
    return out


def matches_kind(event: dict[str, Any], kind: str) -> bool:
    """Whether *event* is shown under the kind button *kind* ("" is every event)."""
    if not kind:
        return True
    if kind == "mlsweep":
        return is_mlsweep(event)
    if kind == "tools":
        return event.get("kind") in ("tool", "tool_result")
    return event.get("kind") == kind


def stored_kinds(kind: str) -> tuple[str, ...]:
    """The stored event kinds a *kind* filter can show (empty: every kind).
    Results come with tool calls, since they fold into the call's row."""
    if not kind:
        return ()
    if kind in ("mlsweep", "tools"):
        return ("tool", "tool_result")
    return (kind,)


def kind_label(event: dict[str, Any]) -> str:
    """The tag a row carries: the tool's name for a call, else the kind."""
    kind = str(event.get("kind") or "?")
    if kind == "tool":
        return "mlsweep" if is_mlsweep(event) else str(event.get("name") or "tool")
    if kind == "tool_result":
        return "result"
    return kind


# ── the ledger ───────────────────────────────────────────────────────────────


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _clip(text: str, width: int) -> str:
    return text if width <= 0 or len(text) <= width else text[:max(width - 1, 1)] + "…"


def fmt_ms(ms: Any) -> str:
    if not isinstance(ms, (int, float)):
        return ""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s" if ms < 60000 else f"{ms / 60000:.1f} min"


def result_status(result: dict[str, Any] | None) -> str:
    """``✓ 24 ms`` or ``✗``: how a tool call ended, as its row shows it."""
    if result is None:
        return ""
    mark = "✗" if result.get("ok") is False else "✓"
    took = fmt_ms(result.get("duration_ms"))
    return f"{mark} {took}" if took else mark


def fmt_clock(ts: Any) -> str:
    if not isinstance(ts, (int, float)) or not ts:
        return "--:--:--"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def iso_time(ts: Any) -> str | None:
    if not isinstance(ts, (int, float)) or not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Ledger:
    """A trace's events as rows, one per event, as the Traces tab lists them.

    Results fold into the row of their call; empty thinking (a redacted block)
    is left out.  Feed events in seq order, a page at a time, with ``add``.
    """

    def __init__(self, kind: str = "") -> None:
        self.kind = kind
        self._calls: dict[str, dict[str, Any]] = {}   # tool_id -> its call's row
        self.rows: list[dict[str, Any]] = []

    def add(self, events: Iterable[dict[str, Any]]) -> None:
        for e in events:
            k = e.get("kind")
            if k == "tool_result" and e.get("tool_id") in self._calls:
                row = self._calls[str(e["tool_id"])]
                row["ok"] = e.get("ok") is not False
                if e.get("duration_ms") is not None:
                    row["duration_ms"] = e.get("duration_ms")
                continue
            if k == "thinking" and not str(e.get("text") or "").strip():
                continue
            row = {"seq": e.get("seq"), "ts": e.get("ts"), "kind": k, "label": kind_label(e),
                   "text": str(e.get("text") or "")}
            if k == "tool":
                row["name"] = e.get("name")
                if e.get("tool_id"):
                    self._calls[str(e["tool_id"])] = row
                if is_mlsweep(e):
                    row["mlsweep"] = mlsweep_calls(row["text"])
            if k == "tool_result":
                row["ok"] = e.get("ok") is not False
                row["duration_ms"] = e.get("duration_ms")
            if matches_kind(e, self.kind):
                self.rows.append(row)


def row_line(row: dict[str, Any], width: int = 160) -> str:
    """One ledger row as one line: ``E412  12:03:04  bash      ls -la  ✓ 24 ms``."""
    if row.get("mlsweep"):
        text = " · ".join(row["mlsweep"])
    elif row["kind"] == "tool_result":
        text = ""
    else:
        text = _one_line(row["text"])
    status = result_status(row) if row["kind"] in ("tool", "tool_result") and "ok" in row else ""
    body = _clip(text, width - len(status) - 2 if width > 0 and status else width)
    tail = f"  {status}" if status else ""
    return f"E{row['seq']:<6} {fmt_clock(row.get('ts'))}  {row['label'][:12]:<12} {body}{tail}".rstrip()


def take_within_budget(lines: Sequence[str], budget_tokens: int) -> int:
    """How many of *lines* fit in *budget_tokens* (at least one, so a page always moves)."""
    room = max(budget_tokens, 1) * CHARS_PER_TOKEN
    used = 0
    for i, line in enumerate(lines):
        used += len(line) + 1
        if used > room and i > 0:
            return i
    return len(lines)


# ── search snippets ──────────────────────────────────────────────────────────


def search_terms(q: str) -> list[str]:
    """*q* as lowercase terms: words, or "quoted phrases" kept whole."""
    terms = [a or b for a, b in re.findall(r'"([^"]+)"|(\S+)', q)]
    return [t.lower() for t in terms if t.strip()]


def event_haystack(event: dict[str, Any]) -> str:
    return f"{event.get('text') or ''} {event.get('name') or ''}"


def event_matches(event: dict[str, Any], terms: Sequence[str]) -> bool:
    hay = event_haystack(event).lower()
    return all(t in hay for t in terms)


def snippet(text: str, terms: Sequence[str], width: int = 140) -> str:
    """The part of *text* around its first matching term, on one line."""
    flat = _one_line(text)
    low = flat.lower()
    hits = [low.find(t) for t in terms if low.find(t) >= 0]
    if not hits or len(flat) <= width:
        return _clip(flat, width)
    start = max(min(hits) - width // 3, 0)
    piece = flat[start:start + width]
    return ("…" if start else "") + piece + ("…" if start + width < len(flat) else "")


# ── ATIF ─────────────────────────────────────────────────────────────────────

ATIF_VERSION = "ATIF-v1.8"
_AGENT_NAMES = {"claude_code": "claude-code", "pi": "pi"}


def to_atif(session: dict[str, Any], events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A trace as one ATIF trajectory document.

    Prompts become user steps, and each stretch of the agent's thinking,
    replies and tool calls between them becomes agent steps, one per response.
    A tool call's result joins its step as an observation.  Capture keeps only
    whether a call succeeded and how long it took, never its output, so an
    observation's content is that status, and a call's arguments are the
    one-line summary capture kept (under ``extra``).  Each step's ``extra``
    lists the mlsweep event numbers it came from, so a step can be cited back.
    """
    steps: list[dict[str, Any]] = []
    owner: dict[str, dict[str, Any]] = {}        # tool_call_id -> the step holding the call
    current: dict[str, Any] | None = None        # the agent step being built
    after_result = False                         # a result since current began: next reply is a new step

    def seqs(step: dict[str, Any], seq: Any) -> None:
        step["extra"]["mlsweep_seqs"].append(seq)

    def new_step(source: str, e: dict[str, Any], message: str = "") -> dict[str, Any]:
        step: dict[str, Any] = {"step_id": len(steps) + 1, "source": source, "message": message,
                                "extra": {"mlsweep_seqs": []}}
        ts = iso_time(e.get("ts"))
        if ts:
            step["timestamp"] = ts
        steps.append(step)
        return step

    for e in events:
        k = e.get("kind")
        text = str(e.get("text") or "")
        if k == "user":
            current = None
            seqs(new_step("user", e, text), e.get("seq"))
            continue
        if k == "meta":
            current = None
            seqs(new_step("system", e, text), e.get("seq"))
            continue
        if k == "tool_result":
            tool_id = str(e.get("tool_id") or "")
            status = ("error" if e.get("ok") is False else "ok") + (
                f" ({fmt_ms(e.get('duration_ms'))})" if e.get("duration_ms") is not None else "")
            result: dict[str, Any] = {"content": status,
                                      "extra": {"ok": e.get("ok") is not False,
                                                "duration_ms": e.get("duration_ms")}}
            step = owner.get(tool_id)
            if step is not None:
                result["source_call_id"] = tool_id
            else:
                step = current or new_step("agent", e)
                current = step
            step.setdefault("observation", {"results": []})["results"].append(result)
            seqs(step, e.get("seq"))
            after_result = True
            continue
        if k not in ("assistant", "thinking", "tool") or (k == "thinking" and not text.strip()):
            continue       # redacted thinking is an empty block: nothing to carry
        if current is None or after_result or (k != "tool" and current.get("tool_calls")):
            current = new_step("agent", e)
            after_result = False
        if e.get("model"):
            current.setdefault("model_name", str(e["model"]))
        tokens = e.get("tokens")
        if isinstance(tokens, dict):
            metrics = {key: int(tokens[src]) for key, src in (("prompt_tokens", "input"), ("completion_tokens", "output"))
                       if isinstance(tokens.get(src), int)}
            if metrics:
                current["metrics"] = metrics      # one response's usage, repeated on each of its blocks
        if k == "assistant":
            current["message"] = (current["message"] + "\n\n" + text) if current["message"] else text
        elif k == "thinking":
            prev = current.get("reasoning_content")
            current["reasoning_content"] = f"{prev}\n\n{text}" if prev else text
        else:
            call_id = str(e.get("tool_id") or f"mlsweep-E{e.get('seq')}")
            current.setdefault("tool_calls", []).append({
                "tool_call_id": call_id, "function_name": str(e.get("name") or "tool"),
                "arguments": {}, "extra": {"summary": text}})
            owner[call_id] = current
        seqs(current, e.get("seq"))

    prompt = sum(s.get("metrics", {}).get("prompt_tokens", 0) for s in steps)
    completion = sum(s.get("metrics", {}).get("completion_tokens", 0) for s in steps)
    final: dict[str, Any] = {"total_steps": len(steps)}
    if prompt or completion:
        final.update(total_prompt_tokens=prompt, total_completion_tokens=completion)
    agent: dict[str, Any] = {"name": _AGENT_NAMES.get(str(session.get("agent")), str(session.get("agent") or "agent")),
                             "version": "unknown"}
    if session.get("model"):
        agent["model_name"] = str(session["model"])
    return {
        "schema_version": ATIF_VERSION,
        "session_id": str(session.get("session_id") or ""),
        "agent": agent,
        "steps": steps,
        "notes": ("Exported by mlsweep. Capture records prompts, replies, thinking and each tool call's "
                  "one-line summary, but not tool output or file contents, so observations hold only "
                  "whether a call succeeded and how long it took."),
        "final_metrics": final,
        "extra": {"mlsweep": {
            "ref": trace_ref(str(session.get("session_id") or "")),
            "title": session.get("title") or "", "cwd": session.get("cwd") or "",
            "host": session.get("host") or "", "state": session.get("state"),
            "experiments": list(session.get("experiments") or []),
        }},
    }
