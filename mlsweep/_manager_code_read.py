"""Searching the code experiments ran: ``POST /api/code/search``.

``mlsweep code search`` sends a pattern and either some experiments or
snapshots, or a campaign.  Each experiment stands for its latest code, the
version the code page and note links resolve to.  Files are content-addressed,
so a file several experiments share is read and matched once, and its hits name
the newest experiment that has it and count the rest.

Each hit carries its note reference (``mlsweep:code/EXP/path#L12``), so it can
be cited as it is.  Binary files and files over ``MAX_TEXT_BYTES`` are skipped,
like everywhere code is shown as text.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from aiohttp import web

from mlsweep._manager_db import list_experiments_with_counts
from mlsweep._manager_http import (
    DB_KEY,
    DIR_KEY,
    _bad_request,
    _json_body,
    _json_response,
)
from mlsweep._notes_code import _Snapshots
from mlsweep._snapshot import is_snapshot_id, short_id, under_prefix

code_read_routes = web.RouteTableDef()

DEFAULT_HITS = 50
MAX_HITS = 500
MAX_TARGETS = 200          # experiments one campaign-wide search looks at
_MAX_LINE = 2000           # a longer line is matched and shown cut short
_MAX_HITS_PER_FILE = 50


def _matcher(q: str, regex: bool, ignore_case: bool) -> Any:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(q if regex else re.escape(q), flags)


async def search_code(
    db: Any, mlsweep_dir: Any, q: str, *, targets: list[str], regex: bool = False,
    ignore_case: bool = False, path: str | None = None, context: int = 0, limit: int = DEFAULT_HITS,
) -> dict[str, Any]:
    """Lines matching *q* in the code of *targets* (experiment or snapshot ids, newest first)."""
    pattern = _matcher(q, regex, ignore_case)
    snaps = _Snapshots(db, mlsweep_dir)
    errors: dict[str, str] = {}
    seen: dict[tuple[str, str], dict[str, Any]] = {}   # (path, hash) -> where it is
    order: list[tuple[str, str]] = []
    for target in targets:
        hit = await snaps.resolve(target)
        if isinstance(hit, str):
            errors[target] = hit
            continue
        cite = target if not is_snapshot_id(target) else short_id(hit.snapshot)
        for f in hit.manifest.files:
            if not under_prefix(f.path, path):
                continue
            key = (f.path, f.hash)
            if key in seen:
                seen[key]["also_in"].append(cite)
                continue
            seen[key] = {"target": cite, "snapshot": hit.snapshot, "size": f.size, "also_in": []}
            order.append(key)

    hits: list[dict[str, Any]] = []
    files_searched = skipped = 0
    truncated = False
    for fpath, h in order:
        if len(hits) >= limit:
            truncated = True          # files left unsearched
            break
        where = seen[(fpath, h)]
        text, _ = await snaps.text(h, where["size"])
        if text is None:
            skipped += 1
            continue
        files_searched += 1
        lines = text.splitlines()
        found = await asyncio.to_thread(
            lambda: [i for i, line in enumerate(lines) if pattern.search(line[:_MAX_LINE])])
        room = min(limit - len(hits), _MAX_HITS_PER_FILE)
        truncated = truncated or len(found) > room
        for i in found[:room]:
            row: dict[str, Any] = {"target": where["target"], "snapshot": where["snapshot"], "path": fpath, "line": i + 1,
                   "text": lines[i][:_MAX_LINE], "ref": f"mlsweep:code/{where['target']}/{fpath}#L{i + 1}",
                   "also_in": where["also_in"]}
            if context:
                row["before"] = [s[:_MAX_LINE] for s in lines[max(i - context, 0):i]]
                row["after"] = [s[:_MAX_LINE] for s in lines[i + 1:i + 1 + context]]
            hits.append(row)
    return {"hits": hits, "truncated": truncated, "files_searched": files_searched,
            "files_skipped": skipped, "errors": errors}


@code_read_routes.post("/api/code/search")
async def handle_code_search(request: web.Request) -> web.Response:
    """``{"q", "targets"?, "campaign"?, "regex"?, "ignore_case"?, "path"?, "context"?, "limit"?}``.

    With no ``targets``, every experiment of ``campaign`` (all campaigns when
    that is null too), newest first, up to ``MAX_TARGETS``.
    """
    body = await _json_body(request)
    q = body.get("q")
    if not isinstance(q, str) or not q:
        return _bad_request("'q' must be a non-empty string")
    targets = body.get("targets")
    if targets is not None and (not isinstance(targets, list) or not all(isinstance(t, str) for t in targets)):
        return _bad_request("'targets' must be a list of experiment or snapshot ids")
    try:
        context = min(max(int(body.get("context", 0)), 0), 10)
        limit = min(max(int(body.get("limit", DEFAULT_HITS)), 1), MAX_HITS)
    except (TypeError, ValueError):
        return _bad_request("'context' and 'limit' must be numbers")
    try:
        _matcher(q, bool(body.get("regex")), bool(body.get("ignore_case")))
    except re.error as exc:
        return _bad_request(f"bad pattern: {exc}")
    db = request.config_dict[DB_KEY]
    campaign = body.get("campaign")
    if not targets:
        exps = await list_experiments_with_counts(db, campaign=campaign if isinstance(campaign, str) else None)
        targets = [e["experiment_id"] for e in exps[:MAX_TARGETS]]   # newest first
    path = body.get("path") if isinstance(body.get("path"), str) else None
    result = await search_code(
        db, request.config_dict[DIR_KEY], q, targets=targets, regex=bool(body.get("regex")),
        ignore_case=bool(body.get("ignore_case")), path=(path or "").strip("/") or None,
        context=context, limit=limit)
    if body.get("targets") is None:
        result["errors"] = {}       # a campaign's experiments without code are not errors
    return _json_response(result)
