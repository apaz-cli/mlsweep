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
    get_capture_session,
    get_experiment,
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


@capture_routes.post("/api/capture/link")
async def handle_capture_link(request: web.Request) -> web.Response:
    if not _enabled(request):
        return _refuse()
    db = request.config_dict[DB_KEY]
    state: Any = request.config_dict[STATE_KEY]
    body = await _json_body(request)
    session_id, experiment_id, source = (
        body.get("session_id"), body.get("experiment_id"), body.get("source", "manual"))
    if not isinstance(session_id, str) or not isinstance(experiment_id, str):
        return _bad_request("'session_id' and 'experiment_id' are required strings")
    try:
        validate_capture_id(session_id)
        _sanitize_experiment_id(experiment_id)
    except ValueError as exc:
        return _bad_request(str(exc))
    if source not in ("auto", "manual"):
        return _bad_request("'source' must be 'auto' or 'manual'")
    if not await get_capture_session(db, session_id):
        raise _not_found_exc("trace")
    if not await get_experiment(db, experiment_id):
        raise _not_found_exc("experiment")
    created = await state.db_writer.link_capture(session_id, experiment_id, source)
    return _json_response({"session_id": session_id, "experiment_id": experiment_id,
                           "linked": created}, status=201)


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
    session = await get_capture_session(db, session_id)
    if session is None:
        raise _not_found_exc("trace")
    session["experiments"] = [
        row[0] for row in await _simple_query(
            db, "SELECT experiment_id FROM capture_links WHERE session_id = ?", (session_id,))]
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


async def resolve_trace_refs(db: Any, refs: list[str]) -> dict[str, dict[str, Any]]:
    """``trace/<session_id>`` references from notes: what the trace names."""
    out: dict[str, dict[str, Any]] = {}
    for ref in refs:
        session_id = ref[len("trace/"):]
        try:
            validate_capture_id(session_id)
        except ValueError as exc:
            out[ref] = {"ok": False, "ref": ref, "error": str(exc)}
            continue
        session = await get_capture_session(db, session_id)
        if session is None:
            out[ref] = {"ok": False, "ref": ref, "error": "no such trace"}
            continue
        out[ref] = {
            "ok": True, "ref": ref, "session_id": session_id,
            "agent": session["agent"], "title": session.get("title") or "",
            "state": session.get("state"), "event_count": session.get("event_count"),
            "experiments": [r[0] for r in await _simple_query(
                db, "SELECT experiment_id FROM capture_links WHERE session_id = ?", (session_id,))],
        }
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
