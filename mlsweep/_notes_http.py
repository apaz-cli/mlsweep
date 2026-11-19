"""HTTP API for the dashboard's Notes tab.

The manager serves one notes folder, a git repo given by ``--notes`` (see
``_notes_git``).  ``NotesService`` owns it.  The same repo is the git remote,
at ``/notes.git``: ``git http-backend`` answers there, and a push updates the
folder in place.  Saves are committed a few seconds after they happen.  Every
write from the API and every push holds one lock (``tree_lock``), and a push
first commits pending saves, because git refuses to update a dirty folder.

Routes go through the manager's token authentication like every other route.
Paths in requests are validated by ``_notes.clean_rel`` and resolved inside the
folder; ``.git`` is never served as files.

Saves are compare-and-swap: the client sends the hash of the version it
edited, and a save over a file that changed since (an agent's push) is refused
with 409 and the current contents, so nothing is overwritten silently.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from os import environ
from pathlib import Path, PurePosixPath
from typing import Any

from aiohttp import BodyPartReader, web

from mlsweep import _notes, _notes_blocks, _notes_git, _notes_lint, _notes_rules
from mlsweep._manager_db import list_experiment_meta, list_run_ids
from mlsweep._manager_http import (
    DB_KEY,
    _bad_request,
    _error_response,
    _json_body,
    _json_dumps,
    _json_response,
    _not_found,
    untrusted_file_headers,
)

logger = logging.getLogger(__name__)

notes_routes = web.RouteTableDef()

SERVICE_KEY: web.AppKey["NotesService | None"] = web.AppKey("mlsweep_notes", object)  # type: ignore[arg-type]


@dataclass
class _KnownExperiments:
    """The experiment and run IDs, kept for a few seconds between requests."""

    at: float = -math.inf
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs_at: float = -math.inf
    runs: dict[str, list[str]] = field(default_factory=dict)


KNOWN_CACHE_KEY: web.AppKey[_KnownExperiments] = web.AppKey("mlsweep_notes_known", _KnownExperiments)

_REV_RE = re.compile(r"^[0-9a-f]{4,40}$")
_SAVE_COMMIT_DELAY = 4.0
_MAX_UPLOAD_FILES = 1000
_MAX_GIT_BODY = 2 * 1024 ** 3
_GIT_CHUNK = 1 << 20
# The only requests the git remote answers: smart HTTP fetch and push.
_GIT_REQUESTS = {("GET", "info/refs"), ("POST", "git-upload-pack"), ("POST", "git-receive-pack")}
_GIT_SERVICES = {"git-upload-pack", "git-receive-pack"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NotesService:
    """The manager's notes repo: locking, committing saves, cached index."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        # tree_lock guards the working tree: file writes from the API, commits and
        # pushes.  It is held inside worker threads.
        self.tree_lock = threading.RLock()
        self.lock = asyncio.Lock()          # one commit at a time
        self.last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._kick: asyncio.Event = asyncio.Event()
        self._kick_at: float = 0.0
        self._fp: tuple[float, str] = (0.0, "")
        self._index: tuple[str, frozenset[str], _notes.Index] | None = None
        self._run_scan: tuple[str, frozenset[str], dict[str, dict[str, list[int]]]] | None = None
        self._index_lock = asyncio.Lock()   # concurrent requests share one build
        self._parse_cache = _notes.ParseCache()
        self._git_info: dict[str, Any] | None = None

    # ── lifecycle ──
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="notes-commit")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.commit()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - logged, loop keeps going
                logger.exception("notes commit failed")
            self._kick.clear()
            if not self._kick_at:
                await self._kick.wait()
            # After a kick, wait out the debounce (later kicks push it back).
            while (wait := self._kick_at - time.monotonic()) > 0:
                await asyncio.sleep(wait)
            self._kick_at = 0.0

    def kick(self, delay: float = _SAVE_COMMIT_DELAY) -> None:
        """Commit *delay* seconds from now (later saves push the deadline back).

        Also drops the cached fingerprint, since every caller just wrote.
        """
        self.invalidate()
        self._kick_at = time.monotonic() + delay
        self._kick.set()

    async def commit(self) -> int:
        """Commit what is in the folder.  Returns the files committed; a failure is in ``last_error``."""
        def _commit() -> int:
            with self.tree_lock:
                return _notes_git.commit_all(self.root)
        async with self.lock:
            try:
                count = await asyncio.to_thread(_commit)
                self.last_error = None
            except (_notes_git.GitError, OSError) as exc:
                count = 0
                self.last_error = str(exc)
                logger.warning("notes commit: %s", exc)
        self._git_info = None
        return count

    # ── change detection ──
    @staticmethod
    def _fingerprint_of(files: list[_notes.Entry], dirs: list[str]) -> str:
        h = hashlib.sha1()
        for f in files:
            h.update(f"{f.path}\0{f.size}\0{f.mtime}\n".encode())
        for d in dirs:
            h.update(f"{d}/\n".encode())
        return h.hexdigest()[:16]

    async def fingerprint(self, max_age: float = 1.5) -> str:
        now = time.monotonic()
        if now - self._fp[0] > max_age:
            self._fp = (now, await asyncio.to_thread(lambda: self._fingerprint_of(*_notes.walk(self.root))))
        return self._fp[1]

    def remember_walk(self, files: list[_notes.Entry], dirs: list[str]) -> str:
        """Fingerprint a fresh walk and cache it, so callers need not walk twice."""
        fp = self._fingerprint_of(files, dirs)
        self._fp = (time.monotonic(), fp)
        return fp

    def invalidate(self) -> None:
        self._fp = (0.0, "")
        self._run_scan = None

    async def index(self, known: set[str]) -> _notes.Index:
        key = frozenset(known)
        async with self._index_lock:
            fp = await self.fingerprint()
            if self._index and self._index[0] == fp and self._index[1] == key:
                return self._index[2]
            idx = await asyncio.to_thread(_notes.build_index, self.root, known, self._parse_cache)
            self._index = (fp, key, idx)
            return idx

    async def run_citations(self, index: _notes.Index,
                            known_runs: set[str]) -> dict[str, dict[str, list[int]]]:
        """Run ID -> {note: lines}, cached until the notes or the run IDs move."""
        key = frozenset(known_runs)
        async with self._index_lock:
            fp = await self.fingerprint()
            if self._run_scan and self._run_scan[0] == fp and self._run_scan[1] == key:
                return self._run_scan[2]
            found = await asyncio.to_thread(_notes.run_mentions, index.texts, known_runs)
            self._run_scan = (fp, key, found)
            return found

    # ── status ──
    async def status(self, clone_url: str) -> dict[str, Any]:
        # Only a commit or a push moves these, and tabs poll status every few seconds.
        if self._git_info is None:
            self._git_info = await asyncio.to_thread(_notes_git.repo_info, self.root)
        info = self._git_info
        cfg = await asyncio.to_thread(_notes_git.read_config, self.root)
        return {
            "enabled": True,
            "name": self.root.name,
            "root": str(self.root),
            "version": await self.fingerprint(),
            "last_error": self.last_error,
            "protected": cfg["protected"],
            "max_file_mb": cfg["max_file_mb"],
            "branch": info["branch"],
            "remote": clone_url,
            "head": info["head"][:12],
        }


# ── helpers ────────────────────────────────────────────────────────────────────


def _service(request: web.Request) -> NotesService:
    svc = request.config_dict.get(SERVICE_KEY)
    if svc is None:
        raise web.HTTPNotFound(text=json.dumps({
            "error": "notes are not enabled on this manager; start it with --notes DIR",
            "enabled": False}), content_type="application/json")
    return svc


async def _known_experiments(request: web.Request) -> dict[str, dict[str, Any]]:
    """Every experiment on this manager, by ID (cached for a few seconds)."""
    cache = request.config_dict[KNOWN_CACHE_KEY]
    now = time.monotonic()
    if now - cache.at >= 5.0:
        cache.meta = await list_experiment_meta(request.config_dict[DB_KEY])
        cache.at = now
    return cache.meta


# Run IDs move with every submission, and the counts they feed are not urgent,
# so they are cached for longer than the experiments.
_RUNS_TTL = 60.0


async def _known_runs(request: web.Request) -> dict[str, list[str]]:
    """Every run ID on this manager -> the experiments that have it (cached)."""
    cache = request.config_dict[KNOWN_CACHE_KEY]
    now = time.monotonic()
    if now - cache.runs_at >= _RUNS_TTL:
        cache.runs = await list_run_ids(request.config_dict[DB_KEY])
        cache.runs_at = now
    return cache.runs


def _clone_url(request: web.Request) -> str:
    return f"{request.scheme}://{request.host}/notes.git"


def _rel_param(request: web.Request, name: str = "path", *, allow_root: bool = False) -> str:
    raw = request.query.get(name, "")
    try:
        return _notes.clean_rel(raw, allow_root=allow_root)
    except _notes.PathError as exc:
        raise _bad_request(str(exc)) from None


def _existing_param(request: web.Request, svc: NotesService) -> tuple[str, Path]:
    """The ``?path=`` of something that exists, and where it is; a 400 or 404 if not."""
    rel = _rel_param(request)
    try:
        return rel, _notes.resolve(svc.root, rel, must_exist=True)
    except FileNotFoundError:
        raise _file_not_found() from None
    except _notes.PathError as exc:
        raise _bad_request(str(exc)) from None


def _body_path(svc: NotesService, body: dict[str, Any]) -> tuple[str, Path]:
    """The ``path`` a JSON body names, and where it would be; a 400 if it is not allowed."""
    try:
        rel = _notes.clean_rel(str(body.get("path", "")))
        return rel, _notes.resolve(svc.root, rel)
    except _notes.PathError as exc:
        raise _bad_request(str(exc)) from None


def _file_not_found() -> web.HTTPNotFound:
    return web.HTTPNotFound(text=json.dumps({"error": "file not found"}), content_type="application/json")


def _int_query(request: web.Request, name: str, default: int) -> int:
    """Integer query parameter *name*; a 400 if it is not one."""
    try:
        return int(request.query.get(name) or default)
    except ValueError:
        raise _bad_request(f"{name} must be an integer") from None


def _file_meta(root: Path, rel: str, data: bytes) -> dict[str, Any]:
    st = _notes.resolve(root, rel).stat()
    return {**_notes.Entry.of(rel, st).as_dict(), "hash": _sha(data)}


def _unique_path(root: Path, rel: str) -> str:
    """*rel*, or ``name 1.ext``, ``name 2.ext``... if it exists."""
    p = PurePosixPath(rel)
    candidate = rel
    n = 1
    while (root / candidate).exists():
        candidate = str(p.with_name(f"{p.stem} {n}{p.suffix}"))
        n += 1
    return candidate


# ── routes ─────────────────────────────────────────────────────────────────────


@notes_routes.get("/api/notes/status")
async def notes_status(request: web.Request) -> web.Response:
    svc = request.config_dict.get(SERVICE_KEY)
    if svc is None:
        return _json_response({"enabled": False})
    return _json_response(await svc.status(_clone_url(request)))


@notes_routes.get("/api/notes/tree")
async def notes_tree(request: web.Request) -> web.Response:
    svc = _service(request)
    files, dirs = await asyncio.to_thread(_notes.walk, svc.root)
    version = svc.remember_walk(files, dirs)
    patterns = (await asyncio.to_thread(_notes_git.read_config, svc.root))["protected"]
    return _json_response({"version": version, "files": [f.as_dict() for f in files], "dirs": dirs,
                           "protected_dirs": [d for d in dirs if _notes_rules.is_protected(f"{d}/", patterns)]})


@notes_routes.get("/api/notes/file")
async def notes_file(request: web.Request) -> web.Response:
    svc = _service(request)
    rel, p = _existing_param(request, svc)
    if not p.is_file():
        return _error_response("not a file", status=400)
    # ?history=0: the caller only wants the contents (a note embedded in another).
    with_history = request.query.get("history") != "0"

    def _read() -> dict[str, Any]:
        st = p.stat()
        meta = _notes.Entry.of(rel, st).as_dict()
        if st.st_size <= _notes.MAX_WRITE_BYTES:
            data = p.read_bytes()
            meta["hash"] = _sha(data)
            if meta["kind"] in ("md", "text") or b"\x00" not in data[:8192]:
                try:
                    meta["content"] = data.decode("utf-8")
                    if meta["kind"] == "binary":
                        meta["kind"] = "text"
                except UnicodeDecodeError:
                    pass
        else:
            meta["hash"] = None
            meta["too_large"] = True
        if with_history:
            meta["last_commit"] = _notes_git.last_commit(svc.root, rel)
        cfg = _notes_git.read_config(svc.root)
        meta["protected"] = _notes_rules.is_protected(rel, cfg["protected"])
        return meta
    return _json_response(await asyncio.to_thread(_read))


@notes_routes.put("/api/notes/file")
async def notes_save(request: web.Request) -> web.Response:
    svc = _service(request)
    body = await _json_body(request)
    if not isinstance(body.get("content"), str):
        return _error_response("expected {path, content, base_hash}")
    rel, p = _body_path(svc, body)
    data = body["content"].encode("utf-8")
    if len(data) > _notes.MAX_WRITE_BYTES:
        return _error_response(f"file is larger than {_notes.MAX_WRITE_BYTES // 1048576} MB", status=413)
    base = body.get("base_hash")

    def _write() -> tuple[int, dict[str, Any]]:
        with svc.tree_lock:
            if p.exists():
                if not p.is_file():
                    return 409, {"error": "a folder with that name exists"}
                current = p.read_bytes()
                if base is None:
                    return 409, {"error": "file already exists", "exists": True,
                                 "current_hash": _sha(current)}
                if base != _sha(current):
                    try:
                        text = current.decode("utf-8")
                    except UnicodeDecodeError:
                        text = None
                    return 409, {"error": "the file changed since you opened it",
                                 "current_hash": _sha(current), "current_content": text}
                if current == data:
                    return 200, _file_meta(svc.root, rel, data)
            elif base is not None and not body.get("create", False):
                return 409, {"error": "the file was deleted since you opened it", "deleted": True}
            _notes.atomic_write(p, data)
            return 200, _file_meta(svc.root, rel, data)
    status, payload = await asyncio.to_thread(_write)
    if status == 200:
        svc.kick()
    return _json_response(payload, status=status)


@notes_routes.post("/api/notes/mkdir")
async def notes_mkdir(request: web.Request) -> web.Response:
    svc = _service(request)
    rel, p = _body_path(svc, await _json_body(request))

    def _mkdir() -> bool:
        with svc.tree_lock:
            if p.exists():
                return False
            p.mkdir(parents=True)
            return True
    if not await asyncio.to_thread(_mkdir):
        return _error_response("already exists", status=409)
    svc.invalidate()
    return _json_response({"path": rel})


@notes_routes.post("/api/notes/move")
async def notes_move(request: web.Request) -> web.Response:
    """Rename or move a file or folder; links to moved notes are updated."""
    svc = _service(request)
    body = await _json_body(request)
    src, dst = str(body.get("from", "")), str(body.get("to", ""))
    update_links = bool(body.get("update_links", True))

    def _move() -> dict[str, Any]:
        with svc.tree_lock:
            return _notes.move_path(svc.root, src, dst, update_links=update_links)
    try:
        result = await asyncio.to_thread(_move)
    except FileNotFoundError:
        return _not_found("source")
    except FileExistsError:
        return _error_response(f"{dst} already exists", status=409)
    except _notes.PathError as exc:
        return _error_response(str(exc))
    svc.kick()
    return _json_response(result)


@notes_routes.delete("/api/notes/file")
async def notes_delete(request: web.Request) -> web.Response:
    svc = _service(request)
    rel, p = _existing_param(request, svc)
    base = request.query.get("base_hash")

    def _delete() -> tuple[int, dict[str, Any]]:
        with svc.tree_lock:
            if p.is_dir() and not p.is_symlink():
                if any(p.iterdir()) and request.query.get("recursive") != "1":
                    return 409, {"error": "folder is not empty"}
                shutil.rmtree(p)
                return 200, {"deleted": rel}
            if base and p.is_file() and _sha(p.read_bytes()) != base:
                return 409, {"error": "the file changed since you opened it"}
            p.unlink()
            return 200, {"deleted": rel}
    status, payload = await asyncio.to_thread(_delete)
    if status == 200:
        svc.kick()
    return _json_response(payload, status=status)


@notes_routes.post("/api/notes/upload")
async def notes_upload(request: web.Request) -> web.Response:
    """Upload files (multipart field ``file``) into folder ``?dir=``."""
    svc = _service(request)
    folder = _rel_param(request, "dir", allow_root=True)
    try:
        _notes.resolve(svc.root, folder)
    except _notes.PathError as exc:
        return _error_response(str(exc))
    cfg = await asyncio.to_thread(_notes_git.read_config, svc.root)
    limit = min(_notes.MAX_UPLOAD_BYTES, int(cfg["max_file_mb"] * 1024 * 1024))
    if not request.content_type.startswith("multipart/"):
        return _error_response("expected a multipart upload with a 'file' field")
    reader = await request.multipart()
    saved: list[str] = []
    while True:
        part = await reader.next()
        if part is None:
            break
        if not isinstance(part, BodyPartReader) or part.name != "file" or not part.filename:
            continue
        if len(saved) >= _MAX_UPLOAD_FILES:
            if saved:
                svc.kick()
            return _error_response(f"at most {_MAX_UPLOAD_FILES} files per upload", status=413)
        name = PurePosixPath(str(part.filename).replace("\\", "/")).name.strip()
        try:
            name = _notes.clean_rel(name)
        except _notes.PathError:
            name = "upload"
        size = 0
        chunks: list[bytes] = []
        while True:
            chunk = await part.read_chunk(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                return _error_response(f"{name} is larger than the {limit / 1048576:g} MB limit for synced files", status=413)
            chunks.append(chunk)
        payload = b"".join(chunks)

        def _store(name: str = name, payload: bytes = payload) -> str:
            with svc.tree_lock:
                rel = _unique_path(svc.root, f"{folder}/{name}" if folder else name)
                _notes.atomic_write(_notes.resolve(svc.root, rel), payload)
                return rel
        saved.append(await asyncio.to_thread(_store))
    if saved:
        svc.kick()
    return _json_response({"saved": saved})


@notes_routes.get("/api/notes/raw")
async def notes_raw(request: web.Request) -> web.StreamResponse:
    """A file's bytes.  Only images and PDFs display inline; text is served as
    text/plain and everything else as a download, so no file in the notes can
    run script on the dashboard's origin."""
    svc = _service(request)
    rel, p = _existing_param(request, svc)
    if not p.is_file():
        raise _file_not_found()
    headers = untrusted_file_headers(rel, as_text=_notes.kind_of(rel) in ("md", "text"),
                                     download=request.query.get("download") == "1")
    return web.FileResponse(p, headers=headers)


@notes_routes.get("/api/notes/search")
async def notes_search(request: web.Request) -> web.Response:
    svc = _service(request)
    q = request.query.get("q", "").strip()[:200]
    if not q:
        return _json_response({"results": []})
    limit = min(_int_query(request, "limit", 100), 500)
    files, _ = await asyncio.to_thread(_notes.walk, svc.root)
    results = await asyncio.to_thread(_notes.search, svc.root, files, q, limit=limit)
    return _json_response({"query": q, "results": results})


@notes_routes.get("/api/notes/index")
async def notes_index(request: web.Request) -> web.Response:
    """Links between notes, and experiment mentions."""
    svc = _service(request)
    meta = await _known_experiments(request)
    idx = await svc.index(set(meta))
    version = await svc.fingerprint()

    def _encode() -> str:  # a large folder takes long enough to hold up the event loop
        data = idx.as_dict()
        data["version"] = version
        data["experiments"] = {e: meta[e] for e in data["mentions"] if e in meta}
        return _json_dumps(data)
    return web.Response(text=await asyncio.to_thread(_encode), content_type="application/json")


@notes_routes.get("/api/notes/mentions")
async def notes_mentions(request: web.Request) -> web.Response:
    """What cites each experiment: ``mentions`` (documents naming the experiment or
    any of its runs) and ``runs`` (documents naming that run, per experiment).

    Empty when notes are off.
    """
    svc = request.config_dict.get(SERVICE_KEY)
    if svc is None:
        return _json_response({"enabled": False, "mentions": {}, "runs": {}})
    meta = await _known_experiments(request)
    runs_by_id = await _known_runs(request)
    idx = await svc.index(set(meta))
    mentions = idx.exp_mentions()
    runs: dict[str, dict[str, list[str]]] = {}
    if runs_by_id:
        cited = await svc.run_citations(idx, set(runs_by_id))
        for run_id, per_src in cited.items():
            candidates = runs_by_id[run_id]
            for src in per_src:
                # A run ID two experiments share ("seed0") would otherwise credit
                # both; it counts only for the experiment the note also names.
                targets = candidates if len(candidates) == 1 else [
                    e for e in candidates if e in idx.mentions.get(src, {})]
                for exp in targets:
                    runs.setdefault(exp, {}).setdefault(run_id, []).append(src)
        for exp, by_run in runs.items():
            docs = set(mentions.get(exp, ()))
            for run_id, srcs in by_run.items():
                by_run[run_id] = sorted(set(srcs))
                docs.update(by_run[run_id])
            mentions[exp] = sorted(docs)
    only = request.query.get("experiment_id")
    if only:
        mentions = {only: mentions.get(only, [])}
        runs = {only: runs.get(only, {})}
    return _json_response({"enabled": True, "mentions": mentions, "runs": runs})


@notes_routes.get("/api/notes/history")
async def notes_history(request: web.Request) -> web.Response:
    """Commits, newest first; ``?path=`` limits them to one file."""
    svc = _service(request)
    path = request.query.get("path")
    rel = _rel_param(request) if path else None
    limit = min(_int_query(request, "limit", 50), 500)
    skip = max(_int_query(request, "skip", 0), 0)
    entries = await asyncio.to_thread(_notes_git.log_entries, svc.root, rel, limit=limit,
                                      skip=skip, with_files=rel is None)
    return _json_response({"commits": entries})


@notes_routes.get("/api/notes/show")
async def notes_show(request: web.Request) -> web.Response:
    """A file's contents at a commit."""
    svc = _service(request)
    rel = _rel_param(request)
    rev = request.query.get("rev", "")
    if not _REV_RE.match(rev):
        return _error_response("bad revision")
    data = await asyncio.to_thread(_notes_git.show_file, svc.root, rev, rel)
    if data is None:
        return _error_response("not in that commit", status=404)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _json_response({"path": rel, "rev": rev, "binary": True, "size": len(data)})
    return _json_response({"path": rel, "rev": rev, "content": text})


@notes_routes.get("/api/notes/diff")
async def notes_diff(request: web.Request) -> web.Response:
    """The diff a commit made, optionally for one file."""
    svc = _service(request)
    rev = request.query.get("rev", "")
    if not _REV_RE.match(rev):
        return _error_response("bad revision")
    rel = _rel_param(request) if request.query.get("path") else None
    diff = await asyncio.to_thread(_notes_git.commit_diff, svc.root, rev, rel)
    return _json_response({"rev": rev, "path": rel, "diff": diff[:2_000_000]})


@notes_routes.post("/api/notes/commit")
async def notes_commit(request: web.Request) -> web.Response:
    """Commit pending saves now, rather than a few seconds from now."""
    svc = _service(request)
    count = await svc.commit()
    svc.invalidate()
    return _json_response({"committed": count, "error": svc.last_error,
                           "status": await svc.status(_clone_url(request))})


async def _run_lint(request: web.Request, svc: NotesService) -> dict[str, Any]:
    known = set(await _known_experiments(request))
    runs = await _known_runs(request)
    idx = await svc.index(known)
    # Resolve evidence links first, in this event loop, so the thread that walks
    # the folder can compare each claimed number with the logged metric.
    refs = await asyncio.to_thread(_notes_lint.evidence_refs, idx, svc.root)
    evidence = await _notes_blocks.resolve_evidence_many(request.config_dict[DB_KEY], refs)
    return await asyncio.to_thread(_notes_lint.lint, svc.root, known_experiments=known,
                                   known_runs=runs, evidence=evidence, index=idx)


@notes_routes.get("/api/notes/lint")
async def notes_lint(request: web.Request) -> web.Response:
    svc = _service(request)
    result = await _run_lint(request, svc)
    result["version"] = await svc.fingerprint()
    return _json_response(result)


@notes_routes.post("/api/notes/lint/fix")
async def notes_lint_fix(request: web.Request) -> web.Response:
    """Apply the safe (tier 1) fixes found by a fresh lint."""
    svc = _service(request)
    result = await _run_lint(request, svc)
    def _fix() -> list[str]:
        with svc.tree_lock:
            return _notes_lint.apply_fixes(svc.root, result["findings"])
    changed = await asyncio.to_thread(_fix)
    if changed:
        svc.kick(1.0)
    return _json_response({"changed": changed})


@notes_routes.get("/api/notes/review-prompt")
async def notes_review_prompt(request: web.Request) -> web.Response:
    svc = _service(request)
    result = await _run_lint(request, svc)
    text = _notes_lint.review_prompt(svc.root, result, time.strftime("%Y-%m-%d"))
    return _json_response({"prompt": text})


@notes_routes.get("/api/notes/experiment-ids")
async def notes_experiment_ids(request: web.Request) -> web.Response:
    """Every experiment ID, for ``mlsweep notes`` commands on other machines."""
    return _json_response({"known": sorted(await _known_experiments(request))})


@notes_routes.get("/api/notes/run-ids")
async def notes_run_ids(request: web.Request) -> web.Response:
    """Every run ID and the experiments that use it, for ``mlsweep notes lint``."""
    return _json_response({"runs": await _known_runs(request)})


# ── data-backed markdown blocks ───────────────────────────────────────────────


@notes_routes.post("/api/notes/block")
async def notes_block(request: web.Request) -> web.Response:
    """Evaluate a ``mlsweep-chart`` or ``mlsweep-table`` block against the metrics.

    The note sends the block's spec exactly as it is written, so the dashboard
    and ``mlsweep notes lint`` share this parser.
    """
    _service(request)
    body = await _json_body(request)
    kind = body.get("kind")
    raw = body.get("spec")
    if kind not in ("chart", "table"):
        raise _bad_request("kind must be 'chart' or 'table'")
    if not isinstance(raw, str):
        raise _bad_request("spec must be the block's text")
    if len(raw.encode("utf-8", "replace")) > _notes_blocks.MAX_SPEC_BYTES:
        raise _bad_request("spec is too large")
    db = request.config_dict[DB_KEY]
    try:
        result = (await _notes_blocks.build_chart(db, raw) if kind == "chart"
                  else await _notes_blocks.build_table(db, raw))
    except _notes_blocks.SpecError as exc:
        raise _bad_request(str(exc)) from None
    return _json_response(result)


@notes_routes.post("/api/notes/evidence")
async def notes_evidence(request: web.Request) -> web.Response:
    """Resolve ``[number](mlsweep:EXP/RUN/METRIC@STEP)`` links for a note."""
    _service(request)
    body = await _json_body(request)
    refs = body.get("refs")
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise _bad_request("refs must be a list of strings")
    results = await _notes_blocks.resolve_evidence_many(request.config_dict[DB_KEY], refs)
    return _json_response({"results": results})


# ── the git remote ─────────────────────────────────────────────────────────────


def _run_git_backend(svc: NotesService, env: dict[str, str], body: Any, out: Any, *,
                     push: bool) -> subprocess.CompletedProcess[bytes]:
    """Run ``git http-backend``.  A push holds the tree lock and commits pending saves first."""
    def run() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", "http-backend"], stdin=body, stdout=out, stderr=subprocess.PIPE,
                              env=env, cwd=svc.root, timeout=_notes_git.GIT_TIMEOUT)
    if not push:
        return run()
    with svc.tree_lock:
        _notes_git.commit_all(svc.root)
        return run()


def _parse_cgi_head(head: bytes) -> tuple[int, dict[str, str]]:
    status, headers = 200, {}
    for line in head.decode("latin-1").split("\r\n"):
        key, _, value = line.partition(":")
        if key.lower() == "status":
            status = int(value.split()[0])
        elif key.lower() in ("content-type", "cache-control", "expires", "pragma"):
            headers[key] = value.strip()
    return status, headers


@notes_routes.get("/notes.git/{tail:.*}")
@notes_routes.post("/notes.git/{tail:.*}")
async def notes_git_remote(request: web.Request) -> web.StreamResponse:
    """The notes repo as a git remote, over smart HTTP."""
    svc = _service(request)
    tail = request.match_info["tail"]
    service = request.query.get("service", "")
    if (request.method, tail) not in _GIT_REQUESTS or (tail == "info/refs" and service not in _GIT_SERVICES):
        return _not_found("git path")
    # aiohttp has already undone any Content-Encoding on the body.
    env = {k: environ[k] for k in ("PATH", "HOME", "LANG") if k in environ}
    env.update({
        "GIT_PROJECT_ROOT": str(svc.root), "GIT_HTTP_EXPORT_ALL": "1", "PATH_INFO": f"/.git/{tail}",
        "REQUEST_METHOD": request.method, "QUERY_STRING": f"service={service}" if service else "",
        "CONTENT_TYPE": request.headers.get("Content-Type", ""), "REMOTE_USER": "mlsweep",
        "REMOTE_ADDR": request.remote or "",
    })
    if "Git-Protocol" in request.headers:
        env["GIT_PROTOCOL"] = request.headers["Git-Protocol"]
    push = tail == "git-receive-pack"
    with tempfile.TemporaryFile() as body, tempfile.TemporaryFile() as out:
        size = 0
        async for chunk in request.content.iter_chunked(_GIT_CHUNK):
            size += len(chunk)
            if size > _MAX_GIT_BODY:
                return _error_response("push is too large", status=413)
            await asyncio.to_thread(body.write, chunk)
        await asyncio.to_thread(body.seek, 0)
        env["CONTENT_LENGTH"] = str(size)
        try:
            done = await asyncio.to_thread(_run_git_backend, svc, env, body, out, push=push)
        except (_notes_git.GitError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("notes remote: %s", exc)
            return _error_response("git failed on the manager", status=500)
        if push:
            svc.invalidate()
            svc._git_info = None
        await asyncio.to_thread(out.seek, 0)
        first = await asyncio.to_thread(out.read, _GIT_CHUNK)
        head, sep, rest = first.partition(b"\r\n\r\n")
        if not sep:
            logger.warning("notes remote: %s", done.stderr.decode(errors="replace").strip())
            return _error_response("git failed on the manager", status=500)
        status, headers = _parse_cgi_head(head)
        response = web.StreamResponse(status=status, headers=headers)
        await response.prepare(request)
        while rest:
            await response.write(rest)
            rest = await asyncio.to_thread(out.read, _GIT_CHUNK)
        await response.write_eof()
        return response


def setup_notes(app: web.Application, notes_dir: Path | None) -> None:
    """Register the notes routes on *app*; start the service if *notes_dir* is set."""
    app.add_routes(notes_routes)
    app[KNOWN_CACHE_KEY] = _KnownExperiments()
    if not notes_dir:
        app[SERVICE_KEY] = None
        return
    svc = NotesService(notes_dir)
    app[SERVICE_KEY] = svc

    async def _start(_app: web.Application) -> None:
        await asyncio.to_thread(_notes_git.configure_served, svc.root)
        svc.start()

    async def _stop(_app: web.Application) -> None:
        await svc.stop()
        # Commit what was saved in the last few seconds, if it is quick.
        try:
            await asyncio.wait_for(svc.commit(), timeout=20)
        except Exception:
            logger.warning("notes: final commit on shutdown did not finish")

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
