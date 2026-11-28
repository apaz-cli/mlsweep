"""Reading traces for agents: event-level search and ATIF export.

``GET /api/capture/search/events`` finds the events that match, not just the
traces, so a hit can be cited as ``mlsweep:trace/<id>#E<seq>`` straight away.
The full-text index has one row per uploaded batch, so it picks the candidate
traces and their events are then matched one by one.

``GET /api/capture/sessions/{id}/atif`` gives a trace as one ATIF trajectory
document (see ``_trace_view.to_atif``).

Both sit under ``/api/capture`` with the other capture routes, so
``--no-capture`` removes them too.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web

from mlsweep._manager_capture import find_trace_or_raise
from mlsweep._manager_db import (
    HAS_FTS5,
    _attach_experiment_links,
    _exec_all,
    _fts_query,
    _like_pattern,
    _marks,
)
from mlsweep._manager_http import (
    DB_KEY,
    _bad_request,
    _json_response,
    _limit_param,
)
from mlsweep._trace_view import (
    TRACE_LINK_KINDS,
    event_matches,
    kind_label,
    matches_kind,
    search_terms,
    snippet,
    to_atif,
    trace_ref,
)

trace_read_routes = web.RouteTableDef()

# Traces searched for matching events, best-ranked first.
_CANDIDATE_TRACES = 40
_MAX_HITS = 200


def _json_like_pattern(term: str) -> str | None:
    """A LIKE pattern for *term* in an event's stored JSON, or None when the
    JSON spells it differently (non-ASCII and escaped characters)."""
    if not term.isascii() or any(c in term for c in '"\\') or any(ord(c) < 32 for c in term):
        return None
    return _like_pattern(term)


async def _candidate_traces(db: Any, q: str) -> list[str]:
    """Trace ids whose text matches *q*, best first (newest first without FTS5)."""
    if HAS_FTS5:
        rows = await _exec_all(
            db, "SELECT session_id FROM capture_fts WHERE capture_fts MATCH ? ORDER BY rank LIMIT ?",
            (_fts_query(q), _CANDIDATE_TRACES * 20))
    else:
        rows = await _exec_all(db, "SELECT session_id FROM capture_sessions "
                                   "ORDER BY last_seen_at DESC LIMIT ?", (_CANDIDATE_TRACES,))
    out: list[str] = []
    for r in rows:
        if r[0] not in out:
            out.append(str(r[0]))
        if len(out) >= _CANDIDATE_TRACES:
            break
    return out


def _match_rows(
    sid: str, rows: list[Any], terms: list[str], kind: str, room: int,
) -> list[dict[str, Any]]:
    """Up to *room* hits among one trace's (seq, data) rows."""
    hits: list[dict[str, Any]] = []
    for seq, data in rows:
        event = json.loads(data)
        if not event_matches(event, terms) or not matches_kind(event, kind):
            continue
        hits.append({
            "session_id": sid, "seq": int(seq), "ref": trace_ref(sid, int(seq)),
            "kind": event.get("kind"), "label": kind_label(event), "ts": event.get("ts"),
            "snippet": snippet(str(event.get("text") or event.get("name") or ""), terms),
        })
        if len(hits) >= room:
            break
    return hits


async def search_capture_events(
    db: Any, q: str, *, session_id: str | None = None, kind: str = "", limit: int = 20,
) -> list[dict[str, Any]]:
    """Events matching every term of *q*: in one trace, or across the best-ranked
    traces.  Each hit names its trace, its seq and its reference."""
    terms = search_terms(q)
    if not terms:
        return []
    limit = min(max(limit, 1), _MAX_HITS)
    traces = [session_id] if session_id else await _candidate_traces(db, q)
    likes = [p for p in (_json_like_pattern(t) for t in terms) if p]
    where = "".join(" AND data LIKE ? ESCAPE '\\'" for _ in likes)
    hits: list[dict[str, Any]] = []
    for sid in traces:
        rows = await _exec_all(
            db, f"SELECT seq, data FROM capture_events WHERE session_id = ?{where} ORDER BY seq",
            (sid, *likes))
        # Parsing and matching a whole trace's events would stall the loop.
        hits += await asyncio.to_thread(_match_rows, sid, rows, terms, kind, limit - len(hits))
        if len(hits) >= limit:
            break
    if hits:
        ids = sorted({h["session_id"] for h in hits})
        meta = {str(r["session_id"]): dict(r) for r in await _exec_all(
            db, "SELECT session_id, agent, title, cwd, last_seen_at FROM capture_sessions "
                f"WHERE session_id IN ({_marks(ids)})", tuple(ids))}
        for h in hits:
            m = meta.get(h["session_id"], {})
            h.update(agent=m.get("agent"), title=m.get("title") or "", cwd=m.get("cwd") or "")
    return hits


@trace_read_routes.get("/api/capture/search/events")
async def handle_search_events(request: web.Request) -> web.Response:
    """``?q=WORDS[&session=ID][&kind=K][&limit=N]``: matching events, best traces first."""
    db = request.config_dict[DB_KEY]
    q = request.query.get("q", "").strip()
    kind = request.query.get("kind", "")
    if kind and kind not in TRACE_LINK_KINDS:
        return _bad_request(f"'kind' must be one of {list(TRACE_LINK_KINDS)}")
    session_id = None
    if request.query.get("session"):
        session = await find_trace_or_raise(db, request.query["session"])
        session_id = str(session["session_id"])
    if not q:
        return _json_response([])
    return _json_response(await search_capture_events(
        db, q, session_id=session_id, kind=kind, limit=_limit_param(request) or 20))


@trace_read_routes.get("/api/capture/sessions/{session_id}/atif")
async def handle_capture_atif(request: web.Request) -> web.Response:
    """The trace as one ATIF trajectory document, for tools outside mlsweep."""
    db = request.config_dict[DB_KEY]
    session = await find_trace_or_raise(db, request.match_info["session_id"])
    await _attach_experiment_links(db, [session])
    rows = await _exec_all(db, "SELECT data FROM capture_events WHERE session_id = ? ORDER BY seq",
                           (session["session_id"],))
    doc = await asyncio.to_thread(lambda: to_atif(session, [json.loads(r[0]) for r in rows]))
    headers = {}
    if request.query.get("download"):
        headers["Content-Disposition"] = f'attachment; filename="{session["session_id"]}.atif.json"'
    return web.json_response(doc, headers=headers)
