"""Manager HTTP routes for code snapshots (see ``mlsweep._snapshot``).

* ``POST /api/objects/missing`` — which of these content hashes the store lacks
* ``PUT  /api/objects/{hash}`` — upload one object (checked against its hash)
* ``GET  /api/objects/{hash}`` — an object's bytes (workers fetch through this)
* ``POST /api/snapshots`` — register a manifest whose objects are all stored
* ``GET  /api/snapshots/{id}`` — a snapshot's manifest
* ``GET  /api/snapshots/{id}/files/{path}`` — one file of a snapshot
* ``GET  /api/snapshots/{id}/usage`` — the experiments and runs that used a snapshot
* ``GET  /api/snapshots/{id}/archive.zip`` — every file of a snapshot
* ``GET  /api/experiments/{id}/code`` — the code an experiment's runs use, with provenance
* ``GET  /api/code/submissions/{n}`` — one submission's full provenance (diff, sweep source)
* ``GET  /api/code/diff?from=&to=`` — what changed between two experiments' or snapshots' code

The store lives under ``<mlsweep-dir>/store/objects``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import aiosqlite
from aiohttp import web

from mlsweep._manager_db import (
    all_snapshot_manifests,
    experiment_code_versions,
    get_experiment,
    get_code_submission,
    get_snapshot_manifest,
    list_code_submissions,
    snapshot_usage,
    resolve_snapshot_prefix,
    snapshot_stats,
)
from mlsweep._manager_http import (
    _MAX_ARTIFACT_BYTES,
    DB_KEY,
    DIR_KEY,
    STATE_KEY,
    _bad_request,
    _error_response,
    _json_body,
    _json_dumps,
    _json_response,
    _not_found,
    _not_found_exc,
)
from mlsweep._inspect_model import InspectError, inspect_file
from mlsweep._manager_state import ManagerState
from mlsweep._snapshot import (
    SNAP_PREFIX,
    HashMismatch,
    Manifest,
    ObjectStore,
    is_snapshot_id,
    parse_manifest,
    snapshot_id_of,
    store_root,
    validate_object_hash,
)

code_routes = web.RouteTableDef()

# The most new bytes one submission may upload; also the largest single object.
MAX_NEW_BYTES = _MAX_ARTIFACT_BYTES
_MAX_HASHES_PER_QUERY = 200_000

_TOO_LARGE = f"object is larger than {MAX_NEW_BYTES // (1024 * 1024)} MB"

def _store(request: web.Request) -> ObjectStore:
    return ObjectStore(store_root(request.config_dict[DIR_KEY]))


def _hash_param(request: web.Request) -> str:
    h = request.match_info["object_hash"]
    try:
        return validate_object_hash(h)
    except ValueError as exc:
        raise _bad_request(str(exc)) from None


# Content named by its hash never changes, so a browser may keep it.  Private:
# it is fetched with the manager's token.
_IMMUTABLE = {"Cache-Control": "private, max-age=31536000, immutable"}


async def _stream_object(request: web.Request, store: ObjectStore, h: str) -> web.StreamResponse:
    if not await asyncio.to_thread(store.has, h):
        return _not_found("object")
    if request.headers.get("If-None-Match") == f'"{h}"':
        return web.Response(status=304, headers={**_IMMUTABLE, "ETag": f'"{h}"'})
    response = web.StreamResponse(headers={"Content-Type": "application/octet-stream",
                                           "ETag": f'"{h}"', **_IMMUTABLE})
    await response.prepare(request)
    chunks = store.iter_chunks(h)
    while True:
        chunk = await asyncio.to_thread(next, chunks, None)
        if chunk is None:
            break
        await response.write(chunk)
    await response.write_eof()
    return response


# ── Objects ────────────────────────────────────────────────────────────────────


@code_routes.post("/api/objects/missing")
async def handle_missing_objects(request: web.Request) -> web.Response:
    """Which of ``{"hashes": [...]}`` the store lacks.

    Each one it has is marked used, so garbage collection leaves it alone
    while the client goes on to register a snapshot that needs it.
    """
    body = await _json_body(request)
    hashes = body.get("hashes")
    if not isinstance(hashes, list) or len(hashes) > _MAX_HASHES_PER_QUERY:
        return _error_response(f"'hashes' must be a list of at most {_MAX_HASHES_PER_QUERY}")
    try:
        wanted = sorted({validate_object_hash(h) for h in hashes})
    except ValueError as exc:
        return _error_response(str(exc))
    store = _store(request)
    missing = await asyncio.to_thread(lambda: [h for h in wanted if not store.touch(h)])
    return _json_response({"missing": missing, "max_new_bytes": MAX_NEW_BYTES})


@code_routes.put("/api/objects/{object_hash}")
async def handle_put_object(request: web.Request) -> web.Response:
    """Store the request body as the object with this hash (refused if it hashes otherwise)."""
    h = _hash_param(request)
    if request.content_length is not None and request.content_length > MAX_NEW_BYTES:
        return _error_response(_TOO_LARGE, status=413)
    store = _store(request)
    if await asyncio.to_thread(store.touch, h):
        return _json_response({"hash": h, "stored": False})
    writer = store.writer(h)
    try:
        async for chunk in request.content.iter_chunked(1 << 20):
            if writer.size + len(chunk) > MAX_NEW_BYTES:
                return _error_response(_TOO_LARGE, status=413)
            await asyncio.to_thread(writer.write, chunk)
        await asyncio.to_thread(writer.commit)
    except HashMismatch as exc:
        return _error_response(f"upload does not match its hash ({exc}); the file may have "
                               "changed while it was being sent")
    finally:
        await asyncio.to_thread(writer.abort)
    return _json_response({"hash": h, "stored": True, "size_bytes": writer.size}, status=201)


@code_routes.get("/api/objects/{object_hash}")
async def handle_get_object(request: web.Request) -> web.StreamResponse:
    return await _stream_object(request, _store(request), _hash_param(request))


# ── Snapshots ──────────────────────────────────────────────────────────────────


@code_routes.post("/api/snapshots")
async def handle_create_snapshot(request: web.Request) -> web.Response:
    """Register ``{"manifest": {...}}``.  Refused (409, with ``missing``) until every object is stored."""
    state: ManagerState = request.config_dict[STATE_KEY]
    body = await _json_body(request)

    def parse(doc: Any) -> tuple[Manifest, bytes]:
        m = parse_manifest(doc)
        return m, m.canonical()

    try:
        manifest, canonical = await asyncio.to_thread(parse, body.get("manifest"))
    except (ValueError, TypeError) as exc:
        return _error_response(f"bad manifest: {exc}")
    store = _store(request)
    missing = await asyncio.to_thread(lambda: sorted(h for h in manifest.hashes() if not store.touch(h)))
    if missing:
        return _json_response({"error": f"{len(missing)} object(s) not uploaded", "missing": missing},
                              status=409)
    snapshot_id = snapshot_id_of(canonical)
    await state.db_writer.register_snapshot(
        artifact_id=snapshot_id, manifest=canonical.decode("utf-8"), size_bytes=manifest.total_bytes)
    return _json_response({"artifact_id": snapshot_id, "files": len(manifest.files),
                           "size_bytes": manifest.total_bytes}, status=201)


_MIN_PREFIX = len(SNAP_PREFIX) + 8


async def _resolve(db: aiosqlite.Connection, artifact_id: str) -> tuple[str, str | None]:
    """(full ID, manifest JSON) for a snapshot ID or a unique prefix of one at least 8 hex digits long."""
    if not is_snapshot_id(artifact_id):
        return artifact_id, None
    raw = await get_snapshot_manifest(db, artifact_id)
    if raw is None and len(artifact_id) >= _MIN_PREFIX:
        matches = await resolve_snapshot_prefix(db, artifact_id)
        if len(matches) > 1:
            raise web.HTTPConflict(text=_json_dumps({"error": f"{artifact_id} matches several snapshots"}),
                                   content_type="application/json")
        if matches:
            artifact_id, raw = matches[0]
    return artifact_id, raw


# Parsed manifests by ID (their content never changes), most recent last.
_PARSED_CACHE_SIZE = 16
_parsed: OrderedDict[str, Manifest] = OrderedDict()


async def _raw_manifest_or_404(db: aiosqlite.Connection, artifact_id: str) -> tuple[str, str]:
    artifact_id, raw = await _resolve(db, artifact_id)
    if raw is None:
        hint = ("; it is a code archive from before snapshots, which can't be shown"
                if artifact_id.startswith("sha256:") else "")
        raise _not_found_exc(f"snapshot {artifact_id} not found{hint}")
    return artifact_id, raw


async def _manifest_or_404(db: aiosqlite.Connection, artifact_id: str) -> tuple[str, Manifest]:
    artifact_id, raw = await _raw_manifest_or_404(db, artifact_id)
    manifest = _parsed.get(artifact_id)
    if manifest is None:
        manifest = await asyncio.to_thread(parse_manifest, raw)
        _parsed[artifact_id] = manifest
        if len(_parsed) > _PARSED_CACHE_SIZE:
            _parsed.popitem(last=False)
    else:
        _parsed.move_to_end(artifact_id)
    return artifact_id, manifest


@code_routes.get("/api/snapshots/{artifact_id}")
async def handle_get_snapshot(request: web.Request) -> web.Response:
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    artifact_id, raw = await _raw_manifest_or_404(db, request.match_info["artifact_id"])
    files, size_bytes = await snapshot_stats(db, artifact_id)
    # The stored manifest is canonical JSON, validated when it was registered,
    # so it goes out as stored rather than parsed and serialized again.
    body = (f'{{"artifact_id":{_json_dumps(artifact_id)},"manifest":{raw},'
            f'"files":{files},"size_bytes":{size_bytes}}}')
    # Asked for by its full ID, a snapshot never changes; a prefix could come to match another.
    exact = request.match_info["artifact_id"] == artifact_id
    return web.Response(text=body, content_type="application/json", headers=_IMMUTABLE if exact else None)


@code_routes.get("/api/snapshots/{artifact_id}/usage")
async def handle_snapshot_usage(request: web.Request) -> web.Response:
    """The experiments whose runs used this snapshot, or that it was submitted to."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    artifact_id, _ = await _raw_manifest_or_404(db, request.match_info["artifact_id"])
    usage = await snapshot_usage(db, artifact_id)
    usage["submissions"] = [_light(s) for s in usage["submissions"]]
    return _json_response(usage)


def _zip_snapshot(tmp_path: str, manifest: Manifest, store: ObjectStore) -> None:
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for d in manifest.dirs:
            zf.writestr(zipfile.ZipInfo(d.rstrip("/") + "/"), b"")
        for f in manifest.files:
            info = zipfile.ZipInfo(f.path, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (0o100000 | (f.mode or 0o644)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=f.size > (1 << 31)) as out:
                for chunk in store.iter_chunks(f.hash):
                    out.write(chunk)


@code_routes.get("/api/snapshots/{artifact_id}/archive.zip")
async def handle_snapshot_zip(request: web.Request) -> web.StreamResponse:
    """Every file of a snapshot, as a zip."""
    from mlsweep._manager_http import _serve_temp_zip
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    artifact_id, manifest = await _manifest_or_404(db, request.match_info["artifact_id"])
    store = _store(request)
    name = request.query.get("name") or artifact_id.replace(":", "-")[:17]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:100] or "code"
    return await _serve_temp_zip(
        lambda tmp: asyncio.to_thread(_zip_snapshot, tmp, manifest, store), f"{safe}.zip")


@code_routes.get("/api/snapshots/{artifact_id}/files/{path:.+}")
async def handle_get_snapshot_file(request: web.Request) -> web.StreamResponse:
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    _, manifest = await _manifest_or_404(db, request.match_info["artifact_id"])
    entry = next((f for f in manifest.files if f.path == request.match_info["path"]), None)
    if entry is None:
        return _not_found("file")
    return await _stream_object(request, _store(request), entry.hash)


# Model summaries keyed by object hash: content named by its hash never changes.
_INSPECT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_INSPECT_CACHE_SIZE = 64


def _inspect_object(store: ObjectStore, h: str, name: str) -> dict[str, Any]:
    """``inspect_file`` on object *h*, seen under its file *name*.

    The reader goes by the extension too, so it gets a path with the file's
    name: a link to the object, or a decompressed copy of a compressed one.
    """
    src = store.locate(h)
    if src is None:
        raise FileNotFoundError(h)
    with tempfile.TemporaryDirectory(prefix="mlsweep-inspect-") as tmp:
        dest = Path(tmp) / (Path(name).name or "file")
        if src.suffix == ".z":
            store.copy_to(h, dest)
        else:
            os.symlink(src, dest)
        return inspect_file(dest)


@code_routes.get("/api/snapshots/{artifact_id}/file-info/{path:.+}")
async def handle_inspect_snapshot_file(request: web.Request) -> web.Response:
    """Describe a model or array file in a snapshot, as ``artifact-info`` does
    for a run's artifacts.  A 422 means the format is not one it reads."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    _, manifest = await _manifest_or_404(db, request.match_info["artifact_id"])
    entry = next((f for f in manifest.files if f.path == request.match_info["path"]), None)
    if entry is None:
        return _not_found("file")
    info = _INSPECT_CACHE.get(entry.hash)
    if info is None:
        try:
            info = await asyncio.to_thread(_inspect_object, _store(request), entry.hash, entry.path)
        except InspectError as exc:
            return _error_response(str(exc), status=422)
        except FileNotFoundError:
            return _not_found("object")
        except OSError as exc:
            return _error_response(f"could not read file: {exc.strerror or exc}", status=500)
        _INSPECT_CACHE[entry.hash] = info
        while len(_INSPECT_CACHE) > _INSPECT_CACHE_SIZE:
            _INSPECT_CACHE.popitem(last=False)
    else:
        _INSPECT_CACHE.move_to_end(entry.hash)
    return _json_response(info)


@code_routes.get("/api/code/diff")
async def handle_code_diff(request: web.Request) -> web.Response:
    """What changed between two experiments' (or snapshots') code: ``?from=A&to=B[&path=P][&context=N]``.

    The same result a note's ``mlsweep-diff`` block shows, plus ``changes``:
    every changed path with its status, beyond the diffs' size limits.
    """
    from mlsweep._notes_blocks import SpecError
    from mlsweep._notes_code import build_diff_block
    q = request.query
    spec = {k: q[k] for k in ("from", "to", "path", "context") if q.get(k)}
    try:
        result = await build_diff_block(request.config_dict[DB_KEY], request.config_dict[DIR_KEY], spec)
    except SpecError as exc:
        return _error_response(str(exc))
    return _json_response(result)


@code_routes.get("/api/code/resolve")
async def handle_resolve_code(request: web.Request) -> web.Response:
    """Resolve ``?ref=code/EXP[/PATH][#Lstart-end]`` to a snapshot, as a note's code link does."""
    from mlsweep._notes_code import resolve_code_refs
    ref = request.query.get("ref", "")
    if not ref or len(ref) > 8192:
        return _error_response("'ref' is required")
    result = (await resolve_code_refs(request.config_dict[DB_KEY], request.config_dict[DIR_KEY], [ref]))[ref]
    return _json_response(result)


# ── Experiments ────────────────────────────────────────────────────────────────


@code_routes.get("/api/experiments/{experiment_id}/code")
async def handle_experiment_code(request: web.Request) -> web.Response:
    """The code an experiment's runs use, and the provenance of each submission."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    experiment_id = request.match_info["experiment_id"]
    if await get_experiment(db, experiment_id) is None:
        return _not_found("experiment")
    versions = await experiment_code_versions(db, experiment_id)
    submissions = [_light(s) for s in await list_code_submissions(db, experiment_id)]
    return _json_response({"experiment_id": experiment_id, "versions": versions, "submissions": submissions})


def _light(sub: dict[str, Any]) -> dict[str, Any]:
    """A submission without its bulky parts (the diff and the sweep file's source), saying they exist."""
    meta = dict(sub.get("meta") or {})
    if "sweep_source" in meta:
        meta["sweep_source_bytes"] = len(str(meta.pop("sweep_source")).encode())
    vcs = meta.get("vcs")
    if isinstance(vcs, dict) and vcs.get("diff"):
        vcs = dict(vcs)
        vcs["diff_bytes"] = len(str(vcs.pop("diff")).encode())
        meta["vcs"] = vcs
    return {**sub, "meta": meta}


@code_routes.get("/api/code/submissions/{submission_id}")
async def handle_code_submission(request: web.Request) -> web.Response:
    """One code submission with all its provenance, including the uncommitted diff and sweep source."""
    try:
        sid = int(request.match_info["submission_id"])
    except ValueError:
        return _error_response("submission_id must be a number")
    sub = await get_code_submission(request.config_dict[DB_KEY], sid)
    return _json_response(sub) if sub is not None else _not_found("submission")


# ── Garbage collection (called from /api/gc) ───────────────────────────────────


def _referenced_hashes(manifests: list[str]) -> set[str]:
    out: set[str] = set()
    for raw in manifests:
        try:
            out.update(item[1] for item in json.loads(raw).get("files", []))
        except (ValueError, TypeError, AttributeError, IndexError):
            continue
    return out


def unreferenced_objects(
    mlsweep_dir: Path, manifests: list[str], *, min_age: float, now: float,
) -> list[tuple[str, int]]:
    """Objects no manifest lists, last used more than *min_age* seconds ago."""
    store = ObjectStore(store_root(mlsweep_dir))
    live = _referenced_hashes(manifests)
    out: list[tuple[str, int]] = []
    for h, p in store.iter_objects():
        if h in live:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if now - st.st_mtime > min_age:
            out.append((h, st.st_size))
    return out


async def object_gc_plan(
    db: aiosqlite.Connection, mlsweep_dir: Path, dropping: set[str], *, min_age: float, now: float,
) -> list[tuple[str, int]]:
    """Objects that would be unreferenced once the snapshots in *dropping* are gone."""
    manifests = [m for aid, m in await all_snapshot_manifests(db) if aid not in dropping]
    return await asyncio.to_thread(unreferenced_objects, mlsweep_dir, manifests, min_age=min_age, now=now)


def remove_objects(mlsweep_dir: Path, hashes: list[str], *, min_age: float) -> int:
    """Delete these objects, except any used again since they were planned for deletion."""
    store = ObjectStore(store_root(mlsweep_dir))
    freed = 0
    now = time.time()
    for h in hashes:
        p = store.locate(h)
        try:
            if p is None or now - p.stat().st_mtime <= min_age:
                continue
        except OSError:
            continue
        freed += store.remove(h)
    return freed
