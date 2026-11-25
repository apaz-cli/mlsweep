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
import json
import re
import sys
import weakref
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
REGEX_SECONDS = 20.0       # a -E search stops (and its matcher is killed) after this
_REGEX_PROCS = 4           # matcher processes running at once
_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = \
    weakref.WeakKeyDictionary()


def _regex_slots() -> asyncio.Semaphore:
    """This event loop's limit on matcher processes (tests run several loops)."""
    loop = asyncio.get_running_loop()
    if loop not in _slots:
        _slots[loop] = asyncio.Semaphore(_REGEX_PROCS)
    return _slots[loop]
_PIPE_LIMIT = 64 << 20     # longest line read back from the matcher


class RegexTimeout(Exception):
    pass


class _RegexProcess:
    """A ``_regex_worker`` child matching one search's pattern, file by file.

    A pattern that backtracks without end only costs a process, which is
    killed at the deadline, never one of the manager's threads.
    """

    def __init__(self, q: str, flags: int, deadline: float) -> None:
        self.q, self.flags, self.deadline = q, flags, deadline
        self.proc: asyncio.subprocess.Process | None = None
        self.slots = _regex_slots()

    async def __aenter__(self) -> _RegexProcess:  # noqa: PYI034 (Self needs 3.11)
        await self.slots.acquire()
        try:
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mlsweep._regex_worker",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, limit=_PIPE_LIMIT)
            await self._send({"q": self.q, "flags": self.flags, "max_line": _MAX_LINE})
        except BaseException:
            await self._stop()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stop()

    async def _stop(self) -> None:
        try:
            if self.proc is not None and self.proc.returncode is None:
                self.proc.kill()
                await self.proc.wait()
        finally:
            self.slots.release()

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj).encode() + b"\n")
        await self.proc.stdin.drain()

    async def find(self, lines: list[str]) -> list[int]:
        """Indices of the lines that match.  RegexTimeout once the deadline passes."""
        assert self.proc is not None and self.proc.stdout is not None
        remaining = self.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RegexTimeout
        try:
            await asyncio.wait_for(self._send({"lines": lines}), remaining)
            remaining = self.deadline - asyncio.get_running_loop().time()
            raw = await asyncio.wait_for(self.proc.stdout.readline(), max(remaining, 0.001))
        except (asyncio.TimeoutError, TimeoutError):
            raise RegexTimeout from None
        if not raw:
            raise RuntimeError("the regex matcher exited")
        return [int(i) for i in json.loads(raw)["found"]]


def _matcher(q: str, regex: bool, ignore_case: bool) -> Any:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(q if regex else re.escape(q), flags)


async def search_code(
    db: Any, mlsweep_dir: Any, q: str, *, targets: list[str], regex: bool = False,
    ignore_case: bool = False, path: str | None = None, context: int = 0, limit: int = DEFAULT_HITS,
    regex_seconds: float = REGEX_SECONDS,
) -> dict[str, Any]:
    """Lines matching *q* in the code of *targets* (experiment or snapshot ids, newest first).

    A literal *q* is matched here, in a thread: it cannot backtrack.  A regex
    is matched in a child process and given ``REGEX_SECONDS`` in all; past
    that the search stops, keeps what it found, and says ``timed_out``.
    """
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

    if not regex:
        async def find(lines: list[str]) -> list[int]:
            return await asyncio.to_thread(
                lambda: [i for i, line in enumerate(lines) if pattern.search(line[:_MAX_LINE])])
        return await _collect(snaps, seen, order, find, errors, context=context, limit=limit)
    deadline = asyncio.get_running_loop().time() + regex_seconds
    async with _RegexProcess(pattern.pattern, pattern.flags, deadline) as matcher:
        return await _collect(snaps, seen, order, matcher.find, errors, context=context, limit=limit)


async def _collect(
    snaps: Any, seen: dict[tuple[str, str], dict[str, Any]], order: list[tuple[str, str]],
    find: Any, errors: dict[str, str], *, context: int, limit: int,
) -> dict[str, Any]:
    """Match the files in *order* with *find* and gather hits, up to *limit*."""
    hits: list[dict[str, Any]] = []
    files_searched = skipped = 0
    truncated = timed_out = False
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
        try:
            found = await find(lines)
        except RegexTimeout:
            timed_out = truncated = True
            files_searched -= 1
            break
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
    return {"hits": hits, "truncated": truncated, "timed_out": timed_out,
            "files_searched": files_searched, "files_skipped": skipped, "errors": errors}


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
