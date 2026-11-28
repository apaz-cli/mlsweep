"""Session traces on the manager: the HTTP side of capture.

The capture watcher on the machine the coding agents run on uploads each
session in batches (see ``_capture.py``); these routes store them and serve
them to the dashboard's Traces tab and to the notes' ``mlsweep:trace/…``
links.  ``mlsweep manager --no-capture`` turns the whole thing off.

Everything under ``/api/capture`` needs the manager token like any other
route.  Writes go through the ``DbWriter`` actor like every other write.
"""

from __future__ import annotations

from typing import Any

from aiohttp import web

from mlsweep._manager_db import (
    _MAX_EVENTS_PER_SESSION,
    CAPTURE_STATES,
    DbWriter,
    _attach_experiment_links,
    capture_counts,
    capture_sessions_for_experiment,
    capture_transcript_lines,
    find_capture_session,
    get_capture_events,
    get_capture_events_at,
    get_capture_session,
    list_capture_sessions,
    search_capture,
    validate_capture_id,
)
from mlsweep._manager_http import (
    DB_KEY,
    STATE_KEY,
    _bad_request,
    _error_response,
    _json_body,
    _json_response,
    _limit_param,
    _not_found_exc,
    _sanitize_experiment_id,
)
# Trace references (``trace/<id>[?kind=…][#E<seq>]``) are parsed where the
# terminal can use them too.
from mlsweep._trace_view import TRACE_LINK_KINDS, parse_trace_ref  # noqa: F401  (re-exported)

capture_routes = web.RouteTableDef()

# app key: whether the manager accepts capture at all
CAPTURE_KEY: web.AppKey[bool] = web.AppKey("mlsweep_capture", object)  # type: ignore[arg-type]

def capture_enabled(app: Any) -> bool:
    """Whether this manager takes traces: off with --no-capture (the routes are not there)."""
    return bool(app.get(CAPTURE_KEY, False))


# ── upload ────────────────────────────────────────────────────────────────────


@capture_routes.post("/api/capture/sessions")
async def handle_capture_session(request: web.Request) -> web.Response:
    """Upsert one session's header and append its events, in one idempotent call."""
    state: Any = request.config_dict[STATE_KEY]
    try:
        body = await _json_body(request)
    except web.HTTPException:
        return _bad_request("body must be a JSON object")
    meta = body.get("session")
    if not isinstance(meta, dict):
        return _bad_request("'session' is required")
    if not isinstance(meta.get("session_id"), str):
        return _bad_request("'session_id' must be a string")
    try:
        validate_capture_id(meta["session_id"])
    except ValueError as exc:
        return _bad_request(str(exc))
    events = body.get("events") or []
    if not isinstance(events, list) or len(events) > _MAX_EVENTS_PER_SESSION:
        return _bad_request(f"'events' must be a list of at most {_MAX_EVENTS_PER_SESSION}")

    writer: DbWriter = state.db_writer
    row, stored, error = await writer.store_capture_batch(meta, events)
    if error:
        return _error_response(error, status=413)
    row["events_stored"] = stored
    return _json_response(row, status=201)


# ── state and links ───────────────────────────────────────────────────────────


@capture_routes.delete("/api/capture/link")
async def handle_capture_unlink(request: web.Request) -> web.Response:
    state: Any = request.config_dict[STATE_KEY]
    try:
        body = await _json_body(request)
    except web.HTTPException:
        return _bad_request("body must be a JSON object")
    session_id, experiment_id = body.get("session_id"), body.get("experiment_id")
    if not isinstance(session_id, str) or not isinstance(experiment_id, str):
        return _bad_request("'session_id' and 'experiment_id' are required strings")
    try:
        validate_capture_id(session_id)
        _sanitize_experiment_id(experiment_id)
    except ValueError as exc:
        return _bad_request(str(exc))
    removed = await state.db_writer.unlink_capture(session_id, experiment_id)
    return _json_response({"session_id": session_id, "experiment_id": experiment_id,
                           "unlinked": removed})


# ── reads ─────────────────────────────────────────────────────────────────────


@capture_routes.get("/api/capture/sessions")
async def handle_capture_list(request: web.Request) -> web.Response:
    db = request.config_dict[DB_KEY]
    state_filter = request.query.get("state", "all")
    if state_filter not in ("all", *CAPTURE_STATES):
        return _bad_request(f"'state' must be one of {list(CAPTURE_STATES) + ['all']}")
    limit = _limit_param(request) or 50
    sessions = await list_capture_sessions(
        db, agent=request.query.get("agent") or None, state=state_filter,
        q=request.query.get("q") or None, experiment_id=request.query.get("experiment") or None,
        limit=limit)
    return _json_response(sessions)


@capture_routes.get("/api/capture/search")
async def handle_capture_search(request: web.Request) -> web.Response:
    db = request.config_dict[DB_KEY]
    q = request.query.get("q", "").strip()
    if not q:
        return _json_response([])
    return _json_response(await search_capture(db, q, limit=_limit_param(request) or 20))


@capture_routes.get("/api/capture/counts")
async def handle_capture_counts(request: web.Request) -> web.Response:
    """How many traces there are, in total and per experiment (for the row menus)."""
    return _json_response(await capture_counts(request.config_dict[DB_KEY]))


@capture_routes.get("/api/capture/sessions/{session_id}")
async def handle_capture_get(request: web.Request) -> web.Response:
    db = request.config_dict[DB_KEY]
    session_id = request.match_info["session_id"]
    try:
        validate_capture_id(session_id)
    except ValueError as exc:
        return _bad_request(str(exc))
    # A note's link may give only the start of the id; the answer has all of it.
    session, error = await find_capture_session(db, session_id)
    if session is None:
        if error and error != "no such trace":
            return _bad_request(error)
        raise _not_found_exc("trace")
    await _attach_experiment_links(db, [session])
    # Traces that continue this one: their transcripts start with its events.
    cursor = await db.execute("SELECT session_id FROM capture_sessions WHERE continues = ? ORDER BY started_at",
                              (session["session_id"],))
    session["continued_by"] = [r[0] for r in await cursor.fetchall()]
    return _json_response(session)


@capture_routes.get("/api/capture/sessions/{session_id}/events")
async def handle_capture_events(request: web.Request) -> web.Response:
    """One page of a trace's events, addressed by the first seq wanted."""
    db = request.config_dict[DB_KEY]
    session_id = request.match_info["session_id"]
    try:
        validate_capture_id(session_id)
        from_seq = max(int(request.query.get("from_seq", "0")), 0)
    except ValueError as exc:
        return _bad_request(str(exc))
    limit = _limit_param(request) or 500
    if not await get_capture_session(db, session_id):
        raise _not_found_exc("trace")
    events = await get_capture_events(db, session_id, from_seq=from_seq, limit=limit)
    return _json_response({
        "session_id": session_id, "from_seq": from_seq, "events": events,
        "next_from_seq": events[-1]["seq"] + 1 if events else from_seq,
    })


@capture_routes.get("/api/capture/sessions/{session_id}/transcript")
async def handle_capture_transcript(request: web.Request) -> web.Response:
    """The whole normalized session as JSONL, the form the watcher sent."""
    db = request.config_dict[DB_KEY]
    session_id = request.match_info["session_id"]
    try:
        validate_capture_id(session_id)
    except ValueError as exc:
        return _bad_request(str(exc))
    session = await get_capture_session(db, session_id)
    if session is None:
        raise _not_found_exc("trace")
    # Rows are stored as compact JSON already: no need to parse them again.
    body = "".join(line + "\n" for line in await capture_transcript_lines(db, session_id))
    return web.Response(
        text=body, content_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.jsonl"'})


@capture_routes.delete("/api/capture/sessions/{session_id}")
async def handle_capture_delete(request: web.Request) -> web.Response:
    state: Any = request.config_dict[STATE_KEY]
    session_id = request.match_info["session_id"]
    try:
        validate_capture_id(session_id)
    except ValueError as exc:
        return _bad_request(str(exc))
    deleted = await state.db_writer.delete_capture_session(session_id)
    if not deleted:
        raise _not_found_exc("trace")
    return _json_response({"deleted": session_id})


@capture_routes.get("/api/experiments/{experiment_id}/capture")
async def handle_experiment_capture(request: web.Request) -> web.Response:
    """The traces linked to one experiment, newest activity first."""
    db = request.config_dict[DB_KEY]
    sessions = await capture_sessions_for_experiment(db, request.match_info["experiment_id"])
    return _json_response(sessions)


# ── plumbing ──────────────────────────────────────────────────────────────────


async def resolve_trace_refs(db: Any, refs: list[str]) -> dict[str, dict[str, Any]]:
    """``trace/<session_id>[?kind=…][#E<seq>]`` references from notes: what the
    trace names, and the event when the link opens one."""
    out: dict[str, dict[str, Any]] = {}
    found: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
    parsed: dict[str, tuple[dict[str, Any], str, int | None]] = {}
    for ref in refs:
        try:
            session_id, kind, seq = parse_trace_ref(ref)
        except ValueError as exc:
            out[ref] = {"ok": False, "ref": ref, "error": str(exc)}
            continue
        if session_id not in found:
            found[session_id] = await find_capture_session(db, session_id)
        session, error = found[session_id]
        if session is None:
            out[ref] = {"ok": False, "ref": ref, "error": error or "no such trace"}
            continue
        parsed[ref] = (session, kind, seq)
    await _attach_experiment_links(db, [s for s, _, _ in parsed.values()])
    wanted = sorted({(s["session_id"], q) for s, _, q in parsed.values() if q is not None})
    events = await get_capture_events_at(db, wanted)
    for ref, (session, kind, seq) in parsed.items():
        session_id = session["session_id"]
        result: dict[str, Any] = {
            "ok": True, "ref": ref, "session_id": session_id, "kind": kind,
            "agent": session["agent"], "title": session.get("title") or "",
            "state": session.get("state"), "event_count": session.get("event_count"),
            "experiments": session["experiments"],
        }
        if seq is not None:
            event = events.get((session_id, seq))
            if event is None:
                out[ref] = {"ok": False, "ref": ref, "session_id": session_id,
                            "error": f"trace {session_id[:8]} has no event {seq}"}
                continue
            text = str(event.get("text") or "")
            result["event"] = {
                "seq": seq, "kind": event.get("kind"), "name": event.get("name"),
                "ts": event.get("ts"), "text": text[:200] + ("…" if len(text) > 200 else ""),
            }
        out[ref] = result
    return out


def setup_capture(app: web.Application, enabled: bool = True) -> None:
    """Register the capture routes.  ``enabled`` is False for --no-capture."""
    app[CAPTURE_KEY] = enabled
    if enabled:
        from mlsweep._manager_trace_read import trace_read_routes

        app.add_routes(capture_routes)
        app.add_routes(trace_read_routes)
