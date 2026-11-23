"""Session traces on the manager: the HTTP side of capture.

The capture watcher on the machine the coding agents run on uploads each
session in batches (see ``_capture.py``); these routes store them and serve
them to the dashboard's Traces tab and to the notes' ``mlsweep:trace/…``
links.  ``mlsweep manager --no-capture`` turns the whole thing off.

Everything under ``/api/capture`` needs the manager token like any other
route.  Writes go through the ``DbWriter`` actor like every other write.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aiohttp import web

from mlsweep._manager_db import (
    _MAX_EVENTS_PER_SESSION,
    CAPTURE_STATES,
    DbWriter,
    capture_counts,
    capture_sessions_for_experiment,
    get_capture_events,
    find_capture_session,
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
    _not_found_exc,
    _sanitize_experiment_id,
)

capture_routes = web.RouteTableDef()

# app key: whether the manager accepts capture at all
CAPTURE_KEY: web.AppKey[bool] = web.AppKey("mlsweep_capture", object)  # type: ignore[arg-type]

# Transcripts are tiny; one batch is bounded well under the app's body limit.
_MAX_BATCH_BYTES = 8 * 1024 * 1024


def _enabled(request: web.Request) -> bool:
    return bool(request.config_dict[CAPTURE_KEY])


def _refuse() -> web.Response:
    return _error_response("capture is off on this manager (--no-capture)", status=404)


# ── upload ────────────────────────────────────────────────────────────────────


@capture_routes.post("/api/capture/sessions")
async def handle_capture_session(request: web.Request) -> web.Response:
    """Upsert one session's header and append its events, in one idempotent call."""
    if not _enabled(request):
        return _refuse()
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
    row = await writer.upsert_capture_session(meta)
    stored, error = 0, None
    if events:
        stored, error = await writer.insert_capture_events(
            meta["session_id"], events, meta.get("last_seen_at"))
    if error:
        return _error_response(error, status=404)
    row["events_stored"] = stored
    return _json_response(row, status=201)


# ── state and links ───────────────────────────────────────────────────────────


@capture_routes.post("/api/capture/sessions/{session_id}/close")
async def handle_capture_close(request: web.Request) -> web.Response:
    if not _enabled(request):
        return _refuse()
    state: Any = request.config_dict[STATE_KEY]
    session_id = request.match_info["session_id"]
    try:
        validate_capture_id(session_id)
    except ValueError as exc:
        return _bad_request(str(exc))
    row = await state.db_writer.close_capture_session(session_id, "closed")
    if row is None:
        raise _not_found_exc("trace")
    return _json_response(row)


@capture_routes.delete("/api/capture/link")
async def handle_capture_unlink(request: web.Request) -> web.Response:
    if not _enabled(request):
        return _refuse()
    state: Any = request.config_dict[STATE_KEY]
    body = await _json_body(request)
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
    if not _enabled(request):
        return _refuse()
    db = request.config_dict[DB_KEY]
    state_filter = request.query.get("state", "all")
    if state_filter not in ("all", *CAPTURE_STATES):
        return _bad_request(f"'state' must be one of {list(CAPTURE_STATES) + ['all']}")
    try:
        limit = int(request.query.get("limit", "50"))
    except ValueError:
        return _bad_request("limit must be an integer")
    sessions = await list_capture_sessions(
        db, agent=request.query.get("agent") or None, state=state_filter,
        q=request.query.get("q") or None, experiment_id=request.query.get("experiment") or None,
        limit=limit)
    return _json_response(sessions)


@capture_routes.get("/api/capture/search")
async def handle_capture_search(request: web.Request) -> web.Response:
    if not _enabled(request):
        return _refuse()
    db = request.config_dict[DB_KEY]
    q = request.query.get("q", "").strip()
    if not q:
        return _json_response([])
    try:
        limit = int(request.query.get("limit", "20"))
    except ValueError:
        return _bad_request("limit must be an integer")
    return _json_response(await search_capture(db, q, limit=limit))


@capture_routes.get("/api/capture/counts")
async def handle_capture_counts(request: web.Request) -> web.Response:
    """How many traces there are, in total and per experiment (for the row menus)."""
    if not _enabled(request):
        return _refuse()
    return _json_response(await capture_counts(request.config_dict[DB_KEY]))


@capture_routes.get("/api/capture/sessions/{session_id}")
async def handle_capture_get(request: web.Request) -> web.Response:
    if not _enabled(request):
        return _refuse()
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
    session["experiments"] = [
        row[0] for row in await _simple_query(
            db, "SELECT experiment_id FROM capture_links WHERE session_id = ?",
            (session["session_id"],))]
    return _json_response(session)


@capture_routes.get("/api/capture/sessions/{session_id}/events")
async def handle_capture_events(request: web.Request) -> web.Response:
    """One page of a trace's events, addressed by the first seq wanted."""
    if not _enabled(request):
        return _refuse()
    db = request.config_dict[DB_KEY]
    session_id = request.match_info["session_id"]
    try:
        validate_capture_id(session_id)
        from_seq = max(int(request.query.get("from_seq", "0")), 0)
        limit = int(request.query.get("limit", "500"))
    except ValueError as exc:
        return _bad_request(str(exc))
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
    if not _enabled(request):
        return _refuse()
    db = request.config_dict[DB_KEY]
    session_id = request.match_info["session_id"]
    try:
        validate_capture_id(session_id)
    except ValueError as exc:
        return _bad_request(str(exc))
    session = await get_capture_session(db, session_id)
    if session is None:
        raise _not_found_exc("trace")
    events = await get_capture_events(db, session_id, from_seq=0, limit=_MAX_EVENTS_PER_SESSION)
    body = "\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n"
    return web.Response(
        text=body, content_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.jsonl"'})


@capture_routes.delete("/api/capture/sessions/{session_id}")
async def handle_capture_delete(request: web.Request) -> web.Response:
    if not _enabled(request):
        return _refuse()
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
    if not _enabled(request):
        return _refuse()
    db = request.config_dict[DB_KEY]
    sessions = await capture_sessions_for_experiment(db, request.match_info["experiment_id"])
    return _json_response(sessions)


# ── plumbing ──────────────────────────────────────────────────────────────────


async def link_experiment_to_capture(state: Any, session_id: str, experiment_id: str) -> None:
    """The auto-link: an experiment names the trace that submitted it.

    Best effort — a capture the manager never saw is not the submission's
    business to refuse.
    """
    try:
        validate_capture_id(session_id)
        _sanitize_experiment_id(experiment_id)
    except (ValueError, TypeError):
        return
    writer: DbWriter = state.db_writer
    await writer.link_capture(session_id, experiment_id, "auto")


# What a trace link may filter its trace to, as the Traces tab's kind buttons do.
TRACE_LINK_KINDS = ("user", "assistant", "thinking", "tools", "mlsweep")
_TRACE_EVENT_RE = re.compile(r"E(\d{1,12})")


def parse_trace_ref(ref: str) -> tuple[str, str, int | None]:
    """``trace/<id>[?kind=<kind>][#E<seq>]`` -> (id, kind, seq).

    The id may be a unique prefix.  ``kind`` is "" when the link shows every
    event, and ``seq`` None when it opens no particular event.  Raises
    ValueError, worded for the note's author, on anything else.
    """
    body = ref[len("trace/"):] if ref.startswith("trace/") else ref
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


async def resolve_trace_refs(db: Any, refs: list[str]) -> dict[str, dict[str, Any]]:
    """``trace/<session_id>[?kind=…][#E<seq>]`` references from notes: what the
    trace names, and the event when the link opens one."""
    out: dict[str, dict[str, Any]] = {}
    for ref in refs:
        try:
            session_id, kind, seq = parse_trace_ref(ref)
        except ValueError as exc:
            out[ref] = {"ok": False, "ref": ref, "error": str(exc)}
            continue
        session, error = await find_capture_session(db, session_id)
        if session is None:
            out[ref] = {"ok": False, "ref": ref, "error": error or "no such trace"}
            continue
        session_id = session["session_id"]
        result: dict[str, Any] = {
            "ok": True, "ref": ref, "session_id": session_id, "kind": kind,
            "agent": session["agent"], "title": session.get("title") or "",
            "state": session.get("state"), "event_count": session.get("event_count"),
            "experiments": [r[0] for r in await _simple_query(
                db, "SELECT experiment_id FROM capture_links WHERE session_id = ?", (session_id,))],
        }
        if seq is not None:
            rows = await _simple_query(
                db, "SELECT data FROM capture_events WHERE session_id = ? AND seq = ?", (session_id, seq))
            if not rows:
                out[ref] = {"ok": False, "ref": ref, "session_id": session_id,
                            "error": f"trace {session_id[:8]} has no event {seq}"}
                continue
            event = json.loads(rows[0][0])
            text = str(event.get("text") or "")
            result["event"] = {
                "seq": seq, "kind": event.get("kind"), "name": event.get("name"),
                "ts": event.get("ts"), "text": text[:200] + ("…" if len(text) > 200 else ""),
            }
        out[ref] = result
    return out


async def _simple_query(db: Any, sql: str, params: tuple[Any, ...]) -> list[Any]:
    async with db.execute(sql, params) as cursor:
        return list(await cursor.fetchall())


def setup_capture(app: web.Application, mlsweep_dir: Path, enabled: bool = True) -> None:
    """Register the capture routes.  ``enabled`` is False for --no-capture."""
    app[CAPTURE_KEY] = enabled
    if enabled:
        (Path(mlsweep_dir) / "capture").mkdir(parents=True, exist_ok=True)
        app.add_routes(capture_routes)
