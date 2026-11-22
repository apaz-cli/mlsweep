"""HTTP REST API and WebSocket event stream for mlsweep manager.

Provides:
  - REST endpoints for campaigns, experiments, jobs, workers, artifacts
  - WebSocket event stream at ``/ws/experiments/{id}``
  - Static file serving for a web dashboard

All endpoints accept authentication via ``?token=...`` query parameter or
``Authorization: Bearer <token>`` header.  The token is the manager token
stored in ``manager.token``.  Requests that change anything, and refused
logins, are recorded in ``audit.log`` (see ``audit_middleware``).

Any endpoint also accepts ``?campaign=NAME``.  Listings return only that
campaign's experiments and jobs, a new experiment is created in it, and a
request naming an experiment from another campaign is refused with 404 (see
``campaign_middleware``).  Without the parameter nothing is restricted.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import hmac
import importlib.metadata
import ipaddress
import itertools
import json
import logging
import logging.handlers
import math
import mimetypes
import os
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, AsyncContextManager, Callable, Awaitable, TypeGuard

import aiosqlite
from aiohttp import WSMsgType, web

from mlsweep._export import (
    ExportRun,
    ExportScope,
    WandbExport,
    experiment_labels,
    tfevents_for_run,
    tfevents_name,
    upload_wandb_run,
    WandbSource,
    wandb_mode,
)
from mlsweep._inspect_model import InspectError, inspect_file
from mlsweep._manager_db import (
    ACTIVE_JOB_STATUSES,
    FINISHED_JOB_STATUSES,
    JOB_STATUSES,
    WorkerRecord,
    count_pending_jobs,
    experiment_summary,
    find_orphans,
    get_artifact,
    get_experiment,
    get_job,
    job_statuses,
    get_logs_for_run,
    get_metrics_for_run,
    get_worker,
    list_campaigns,
    list_experiments_with_counts,
    list_jobs_by_experiment,
    list_jobs_by_status,
    list_jobs_since,
    list_pending_jobs,
    list_workers,
)
from mlsweep._manager_optimizer import (
    advance_optimizer_locked,
    check_submitted_spec,
    cleared_for_running,
    read_progress,
    status_view,
)
from mlsweep._manager_state import PLACEMENTS, InFlightRun, ManagerState, Pin
from mlsweep._manager_workers import (
    _check_experiments_complete_locked,
    _detach_locked,
    cancel_runs_locked,
    connect_single_worker,
    declare_worker_dead,
    requeue_runs_locked,
    update_worker,
    worker_outdated,
    validate_worker_host,
    worker_id_for,
)
from mlsweep._sweep import build_controls, spec_singular_dims, validate_controls
from mlsweep._shared import (
    _dir_size,
    _resolve_safe_subpath,
    mlsweep_dir_id,
    validate_artifact_id,
    validate_campaign,
    validate_experiment_dir,
    validate_run_id,
)

logger = logging.getLogger(__name__)

_VERSION = importlib.metadata.version("mlsweep")

# aiohttp warns when application config uses plain strings; AppKey also gives
# each lookup a type instead of Any.
DB_KEY: web.AppKey[aiosqlite.Connection] = web.AppKey("mlsweep_db", aiosqlite.Connection)
WANDB_KEY: web.AppKey[WandbSource] = web.AppKey("mlsweep_wandb", WandbSource)
STATE_KEY: web.AppKey[ManagerState] = web.AppKey("mlsweep_state", ManagerState)
TOKEN_KEY: web.AppKey[str] = web.AppKey("mlsweep_token", str)
DIR_KEY: web.AppKey[Path] = web.AppKey("mlsweep_dir", Path)
EXPORTS_KEY: web.AppKey[dict[str, WandbExport]] = web.AppKey("mlsweep_exports")
EXPORT_TASKS_KEY: web.AppKey[set[asyncio.Task[Any]]] = web.AppKey("mlsweep_export_tasks")
THROTTLE_KEY: web.AppKey["_AuthThrottle"] = web.AppKey("mlsweep_auth_throttle", object)  # type: ignore[arg-type]

_MAX_JSON_BODY_BYTES = 128 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
# Provenance recorded with a code submission (diff, sweep source and so on).
_MAX_CODE_META_BYTES = 4 * 1024 * 1024
_MAX_WS_MESSAGE_BYTES = 1024 * 1024
# Zip exports build a temp file first; more than this at once wait their turn.
_ZIP_SLOTS = asyncio.Semaphore(4)

_audit = logging.getLogger("mlsweep.audit")

# ===============================================================================
# JSON helpers
# ===============================================================================


def _json_dumps(obj: Any) -> str:
    """JSON-serialise an object, handling dataclasses and datetimes."""

    def _default(o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=_default)


def _json_response(data: Any, *, status: int = 200) -> web.Response:
    """Return a JSON ``Response`` with the given *data* and *status*."""
    return web.Response(
        text=_json_dumps(data),
        content_type="application/json",
        status=status,
    )


def _error_response(message: str, *, status: int = 400) -> web.Response:
    """Return a JSON error ``Response``."""
    return _json_response({"error": message}, status=status)


def _internal_error(exc: Exception) -> web.Response:
    """A 500 that keeps the details (paths, SQL) in the manager's log."""
    logger.error("request failed: %s", exc, exc_info=exc)
    return _error_response("internal error; see the manager's log", status=500)


def _not_found(entity: str) -> web.Response:
    """Return a 404 JSON error."""
    return _error_response(f"{entity} not found", status=404)


def _schedule_cleanup(path: str, *, delay: float = 300) -> None:
    """Unlink *path* after *delay* seconds (best-effort, non-blocking).

    Intended for temporary zip files served via ``FileResponse``.
    """

    def _do() -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to unlink temp zip: %s", path)

    asyncio.get_event_loop().call_later(delay, _do)


# Default seconds without log/metric progress before a running job is "stalled".
# Clients may pass their own threshold (``mlsweep wait --stalled-after``); this
# is what the REST API reports in the job/summary ``stalled`` fields.
MANAGER_STALL_THRESHOLD_SECONDS = 900.0


def _stall_seconds(run: InFlightRun, now: float) -> float:
    """Seconds since *run* last made progress (0.0 when unknown)."""
    if run.last_progress <= 0.0:
        return 0.0
    return max(0.0, now - run.last_progress)


def _run_dir(mlsweep_dir: Path, experiment_id: str, run_id: str) -> Path:
    """A run's directory under *mlsweep_dir*; ValueError for an ID that could leave it."""
    validate_run_id(run_id)
    return Path(_resolve_safe_subpath(mlsweep_dir / "experiments", f"{experiment_id}/{run_id}"))


def _run_dir_exists(mlsweep_dir: Path, experiment_id: str, run_id: str) -> bool:
    """Whether *run_id* has files in *experiment_id*, treating an unsafe ID as having none."""
    try:
        return _run_dir(mlsweep_dir, experiment_id, run_id).exists()
    except ValueError:
        return False


def _experiment_runs(state: ManagerState, experiment_id: str) -> list[Any]:
    """In-memory runs belonging to *experiment_id*."""
    return [r for r in state.runs.values() if r.experiment_id == experiment_id]


async def _serve_temp_zip(
    fill: Callable[[str], Awaitable[Any]], dl_name: str,
) -> web.FileResponse:
    """Create a temp zip, let *fill* write it, then serve it as an attachment."""
    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="mlsweep_")
    os.close(fd)
    try:
        async with _ZIP_SLOTS:
            await fill(tmp_path)
    except BaseException:
        os.unlink(tmp_path)
        raise
    _schedule_cleanup(tmp_path)
    return web.FileResponse(
        tmp_path,
        headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
    )


def _zip_directory(tmp_path: str, source_dir: Path) -> None:
    """Write all files under *source_dir* into a new zip at *tmp_path* (sync, for executor)."""
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(source_dir))


def _zip_experiment_artifacts(tmp_path: str, exp_dir: Path) -> None:
    """Write artifacts from every run under *exp_dir* into a zip (sync, for executor)."""
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for run_dir in sorted(exp_dir.iterdir()):
            artifacts_dir = run_dir / "artifacts"
            if not artifacts_dir.is_dir():
                continue
            for p in sorted(artifacts_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(run_dir.name) / p.relative_to(artifacts_dir))


# ===============================================================================
# Input validation
# ===============================================================================

_EXPERIMENT_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,128}$')


def _sanitize_experiment_id(experiment_id: str) -> str:
    """Validate *experiment_id* against the allowed character set.

    Returns *experiment_id* unchanged if valid; raises ``ValueError`` otherwise.
    """
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_RE.match(experiment_id):
        raise ValueError(
            "experiment_id must be 1-128 chars of [a-zA-Z0-9_\\-]"
        )
    return experiment_id


# ===============================================================================
# Authentication
# ===============================================================================


def _check_auth(request: web.Request, token: str) -> bool:
    """Return True if *request* carries the correct *token*."""
    # If no token is configured, reject all requests.
    if not token:
        return False
    # Query parameter
    if _same_token(request.query.get("token", ""), token):
        return True
    # Authorization header
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and _same_token(auth[7:], token):
        return True
    # git sends the token as a Basic password; the username does not matter.
    if auth.startswith("Basic "):
        try:
            password = base64.b64decode(auth[6:], validate=True).partition(b":")[2].decode()
        except ValueError:
            return False
        return _same_token(password, token)
    return False


def _same_token(given: str, token: str) -> bool:
    """Compare in constant time, so the response time reveals nothing about *token*."""
    return hmac.compare_digest(given.encode(), token.encode())


_MAX_AUTH_FAILURES = 30
_AUTH_FAILURE_WINDOW = 300.0
_AUTH_BLOCK_SECS = 60.0
_MAX_TRACKED_ADDRESSES = 4096


class _AuthThrottle:
    """Refuses, per address, clients that keep presenting bad tokens.

    Loopback is exempt.  That keeps a mistyped token on the manager's own
    machine from locking out its CLI, and a reverse proxy, whose clients would
    all share its address, from locking out everyone.
    """

    def __init__(self) -> None:
        # address -> (failures, window start, blocked until)
        self._seen: dict[str, tuple[int, float, float]] = {}

    @staticmethod
    def _exempt(address: str | None) -> bool:
        try:
            return address is None or ipaddress.ip_address(address).is_loopback
        except ValueError:
            return False

    def retry_after(self, address: str | None) -> float:
        """Seconds *address* must wait, or 0 if it may try now."""
        if self._exempt(address):
            return 0.0
        entry = self._seen.get(str(address))
        return max(0.0, entry[2] - time.monotonic()) if entry else 0.0

    def failed(self, address: str | None) -> None:
        if self._exempt(address):
            return
        now = time.monotonic()
        count, start, _ = self._seen.get(str(address), (0, now, 0.0))
        if now - start > _AUTH_FAILURE_WINDOW:
            count, start = 0, now
        count += 1
        if str(address) not in self._seen and len(self._seen) >= _MAX_TRACKED_ADDRESSES:
            self._seen = {a: e for a, e in self._seen.items() if now - e[1] <= _AUTH_FAILURE_WINDOW}
            if len(self._seen) >= _MAX_TRACKED_ADDRESSES:
                return
        self._seen[str(address)] = (count, start, now + _AUTH_BLOCK_SECS if count >= _MAX_AUTH_FAILURES else 0.0)


def _job_insert_refused(exc: sqlite3.IntegrityError) -> web.Response:
    """What a constraint failure on inserting jobs means to the client."""
    if "FOREIGN KEY" in str(exc):
        return _not_found("experiment")
    return _error_response(
        "duplicate run_id in this experiment (run names must be unique per experiment); nothing was created",
        status=409,
    )


def _is_number(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _limit_param(request: web.Request) -> int | None:
    """The ``?limit=`` query parameter, or None without one; a 400 if it is not a count."""
    raw = request.query.get("limit")
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError:
        raise _bad_request("limit must be a non-negative integer") from None
    if limit < 0:
        raise _bad_request("limit must be a non-negative integer")
    return limit


def _status_param(request: web.Request, default: str | None = None) -> str | None:
    """The ``?status=`` query parameter; a 400 if no job can have it."""
    status = request.query.get("status", default)
    if status is not None and status not in JOB_STATUSES:
        raise _bad_request(f"status must be one of {list(JOB_STATUSES)}")
    return status


def _is_static(request: web.Request) -> bool:
    return request.path.startswith("/static/")


@web.middleware
async def error_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Turn an unexpected exception into a JSON 500 that hides its details.

    Once a response has started there is nothing left to replace, and a client
    going away is not an error, so those go on to aiohttp.
    """
    try:
        return await handler(request)
    except (web.HTTPException, ConnectionError):
        raise
    except Exception as exc:
        if request.writer.output_size > 0:
            raise
        return _internal_error(exc)


@web.middleware
async def auth_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Middleware that enforces token authentication on all routes.

    Skips static file routes (see ``_is_static``) so the web UI can load
    without a token in every asset request.  Static files are marked
    ``no-cache`` so a browser revalidates them (a cheap 304) on every load
    instead of pairing fresh HTML with a stale cached ``dialog.js``.
    """
    # Allow static files without auth
    if _is_static(request):
        response = await handler(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    throttle = request.config_dict[THROTTLE_KEY]
    wait = throttle.retry_after(request.remote)
    if wait:
        response = _error_response("too many failed logins; try again later", status=429)
        response.headers["Retry-After"] = str(math.ceil(wait))
        return response

    token: str = request.config_dict[TOKEN_KEY]
    if not _check_auth(request, token):
        git_remote = request.path.startswith("/notes.git/")
        # git's first request never carries credentials, so that is not a failed login.
        if not git_remote or "Authorization" in request.headers:
            throttle.failed(request.remote)
        response = _error_response(
            "wrong or missing token: open the link the manager printed at startup, "
            "or send Authorization: Bearer <token>",
            status=401,
        )
        if git_remote:
            response.headers["WWW-Authenticate"] = 'Basic realm="mlsweep notes"'
        return response

    return await handler(request)


@web.middleware
async def id_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Refuse IDs that could leave their directory, on every route that takes one.

    Route patterns do not protect these.  aiohttp decodes ``%2F`` after matching,
    so ``{run_id}`` can hold ``../..``, and handlers join them into paths.
    """
    info = request.match_info
    try:
        if "experiment_id" in info:
            validate_experiment_dir(info["experiment_id"])
        if "run_id" in info:
            validate_run_id(info["run_id"])
        if "artifact_id" in info:
            validate_artifact_id(info["artifact_id"])
    except ValueError as exc:
        return _error_response(str(exc))
    return await handler(request)


@web.middleware
async def audit_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Record every request that can change something, and every refused login.

    Logs the path but never the query string, which can hold the token.
    """
    status: int | None = None
    try:
        response = await handler(request)
        status = response.status
        return response
    except web.HTTPException as exc:
        status = exc.status
        raise
    finally:
        if request.method not in ("GET", "HEAD", "OPTIONS") or status in (401, 429):
            _audit.info(json.dumps({
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "remote": request.remote,
                "method": request.method,
                "path": request.path,
                "status": status,
            }))


def setup_audit_log(mlsweep_dir: Path) -> None:
    """Send the audit log to ``<mlsweep_dir>/audit.log``, keeping a few old files."""
    for old in list(_audit.handlers):
        _audit.removeHandler(old)
        old.close()
    handler = logging.handlers.RotatingFileHandler(
        mlsweep_dir / "audit.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    _audit.addHandler(handler)
    _audit.setLevel(logging.INFO)
    _audit.propagate = False


# Pages load scripts only from this server, and inline script does not run.
# Inline styles stay allowed: the pages set them throughout.
_PAGE_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
# A note can show an image from anywhere, so the notes page allows those too.
_NOTES_PAGE_CSP = _PAGE_CSP.replace("img-src 'self' data: blob:", "img-src 'self' data: blob: https: http:")
_API_CSP = "default-src 'none'; frame-ancestors 'self'"


_FONT_TYPES = {"ttf": "font/ttf", "otf": "font/otf", "woff": "font/woff", "woff2": "font/woff2"}


async def _add_security_headers(request: web.Request, response: web.StreamResponse) -> None:
    """Headers for every response.  A handler's own Content-Security-Policy wins."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if "Content-Security-Policy" in response.headers:
        return
    if _is_static(request):
        # aiohttp's own type table does not know fonts, and with nosniff some browsers refuse them.
        font = _FONT_TYPES.get(request.path.rsplit(".", 1)[-1].lower())
        if font:
            response.headers["Content-Type"] = font
        if request.path.endswith(".html"):
            notes = request.path.endswith("/notes.html")
            response.headers["Content-Security-Policy"] = _NOTES_PAGE_CSP if notes else _PAGE_CSP
    elif response.content_type != "application/pdf":
        # PDFs are left out: Chrome's viewer will not run under this policy.
        response.headers["Content-Security-Policy"] = _API_CSP


_INLINE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".avif": "image/avif", ".bmp": "image/bmp", ".ico": "image/x-icon",
    ".svg": "image/svg+xml", ".pdf": "application/pdf",
}


def untrusted_file_headers(name: str, *, as_text: bool, download: bool = False) -> dict[str, str]:
    """Response headers for a file whose contents the server does not trust.

    Only images and PDFs display inline, *as_text* files are served as
    text/plain and everything else as a download, so no such file can run
    script on the dashboard's origin.
    """
    ext = PurePosixPath(name).suffix.lower()
    headers = {"Cache-Control": "no-cache"}   # _add_security_headers adds the rest
    safe_name = PurePosixPath(name).name.replace('"', "")
    if ext in _INLINE_MIME and not download:
        ctype = _INLINE_MIME[ext]
        headers["Content-Disposition"] = f'inline; filename="{safe_name}"'
        if ext != ".pdf":
            # Chrome's PDF viewer will not run under a sandbox CSP; PDFs are
            # rendered by the browser's own viewer, not on this origin.
            headers["Content-Security-Policy"] = "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:"
    else:
        headers["Content-Security-Policy"] = "sandbox; default-src 'none'"
        if as_text and not download:
            ctype = "text/plain; charset=utf-8"
        else:
            ctype = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
            headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'
            if ctype.startswith(("text/html", "application/xhtml", "image/svg")):
                ctype = "application/octet-stream"
    headers["Content-Type"] = ctype
    return headers


async def _request_json(request: web.Request) -> Any:
    """The request's JSON body, reusing the copy ``campaign_middleware`` parsed."""
    if "json_body" in request:
        return request["json_body"]
    return await request.json()


def _bad_request(message: str) -> web.HTTPBadRequest:
    """A JSON 400 to raise from a helper."""
    return web.HTTPBadRequest(text=_json_dumps({"error": message}), content_type="application/json")


def _not_found_exc(message: str) -> web.HTTPNotFound:
    """A JSON 404 to raise from a helper."""
    return web.HTTPNotFound(text=_json_dumps({"error": message}), content_type="application/json")


async def _json_body(request: web.Request) -> dict[str, Any]:
    """The request's JSON object; a 400 for anything else."""
    try:
        body = await _request_json(request)
    except ValueError:
        raise _bad_request("invalid JSON body") from None
    if not isinstance(body, dict):
        raise _bad_request("body must be a JSON object")
    return body


async def _named_experiments(request: web.Request) -> set[str]:
    """Experiment ids a request names in its path, query, or job JSON body."""
    ids = {request.match_info.get("experiment_id"), request.query.get("experiment_id")}
    if request.method in ("POST", "PUT", "PATCH") and request.path.startswith("/api/jobs"):
        try:
            body = request["json_body"] = await request.json()
        except Exception:
            body = None
        items: list[Any] = []
        if isinstance(body, list):
            items = body
        elif isinstance(body, dict):
            jobs = body.get("jobs")
            items = jobs if isinstance(jobs, list) else [body]
        ids |= {i.get("experiment_id") for i in items if isinstance(i, dict)}
    return {i for i in ids if isinstance(i, str) and i}


def _wrong_campaign(experiment_id: str, actual: str, wanted: str, *, status: int = 404) -> web.Response:
    """Refuse a request for an experiment that is not in the requested campaign."""
    return _json_response({
        "error": f"experiment {experiment_id!r} is in campaign {actual!r}, not {wanted!r}",
        "campaign": actual,
    }, status=status)


@web.middleware
async def campaign_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Enforce ``?campaign=``. Every experiment the request names must be in it.

    The check runs before the handler, so it covers every route that takes an
    experiment id, the WebSocket stream included.  An unknown experiment is
    left to the handler (usually a 404 of its own).
    """
    campaign = request.query.get("campaign")
    if campaign is None or _is_static(request):
        return await handler(request)
    try:
        validate_campaign(campaign)
    except ValueError as exc:
        return _error_response(str(exc))
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    for experiment_id in sorted(await _named_experiments(request)):
        exp = await get_experiment(db, experiment_id)
        if exp is not None and exp.campaign != campaign:
            return _wrong_campaign(experiment_id, exp.campaign, campaign)
    return await handler(request)


# ===============================================================================
# Route table
# ===============================================================================

routes = web.RouteTableDef()


# ── Campaigns ──────────────────────────────────────────────────────────────────


@routes.get("/api/campaigns")
async def handle_list_campaigns(request: web.Request) -> web.Response:
    """List campaigns with experiment and job counts."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    return _json_response(await list_campaigns(db))


@routes.put("/api/experiments/{experiment_id}/campaign")
async def handle_update_experiment_campaign(request: web.Request) -> web.Response:
    """Move an experiment, with all its runs, to another campaign.

    Body: ``{"campaign": NAME}``.  With ``?campaign=`` the experiment must
    currently be in that campaign (enforced by ``campaign_middleware``).
    """
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    body = await _json_body(request)
    target = body.get("campaign") if isinstance(body, dict) else None
    if not target:
        return _error_response("'campaign' is required")
    try:
        validate_campaign(target)
    except ValueError as exc:
        return _error_response(str(exc))
    exp = await state.db_writer.update_experiment_campaign(experiment_id, target)
    if exp is None:
        return _not_found("experiment")
    _broadcast_experiment_event(request, experiment_id, "campaign_updated", campaign=target)
    return _json_response(exp)


# ── Experiments ────────────────────────────────────────────────────────────────


@routes.get("/api/experiments")
async def handle_list_experiments(request: web.Request) -> web.Response:
    """List experiments with job counts, optionally filtered by status and campaign."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    status_filter = request.query.get("status")
    experiments = await list_experiments_with_counts(
        db, status=status_filter, campaign=request.query.get("campaign"),  # type: ignore[arg-type]
    )
    return _json_response(experiments)


async def _experiment_fields(
    request: web.Request,
    db: aiosqlite.Connection,
    body: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any] | web.Response:
    """Experiment-create kwargs from a request body, or an error response.

    The campaign comes from the body's ``campaign``, else ``?campaign=``.
    One of them is required.  An experiment that already exists in another
    campaign is refused with 409.
    """
    goal = body.get("goal")
    if goal is not None and goal not in ("minimize", "maximize"):
        return _error_response("'goal' must be 'minimize' or 'maximize'")
    query_campaign = request.query.get("campaign")
    campaign = body.get("campaign") or query_campaign
    if not campaign:
        return _error_response("'campaign' is required (in the body or as ?campaign=)")
    if query_campaign is not None and campaign != query_campaign:
        return _error_response(
            f"body campaign {campaign!r} does not match ?campaign={query_campaign}")
    try:
        validate_campaign(campaign)
    except ValueError as exc:
        return _error_response(str(exc))
    existing = await get_experiment(db, experiment_id)
    if existing is not None and existing.campaign != campaign:
        return _wrong_campaign(experiment_id, existing.campaign, campaign, status=409)
    code = body.get("code")
    if code is not None:
        if not isinstance(code, dict) or not isinstance(code.get("snapshot"), str):
            return _error_response("'code' must be an object naming its 'snapshot'")
        if len(_json_dumps(code)) > _MAX_CODE_META_BYTES:
            return _error_response(f"'code' is larger than {_MAX_CODE_META_BYTES // (1024 * 1024)} MB")
    try:
        controls = validate_controls(body.get("controls"))
    except ValueError as exc:
        return _error_response(f"bad run controls: {exc}")
    optimizer: dict[str, Any] = {}
    if body.get("optimize") is not None:
        if existing is not None:
            return _error_response(f"experiment {experiment_id} already exists; "
                                   "a Bayesian sweep needs a new one", status=409)
        try:
            spec = await asyncio.to_thread(check_submitted_spec, body["optimize"])
        except Exception as exc:
            return _error_response(f"bad Bayesian sweep: {exc}")
        # The spec decides these, so they can't disagree with it.
        optimizer = {
            "optimize": spec, "metric": spec["metric"], "goal": spec["goal"], "expected_jobs": 0,
            "singular_dims": spec_singular_dims(spec), "skip_rules": {},
        }
    return {
        "experiment_id": experiment_id,
        "name": body.get("name") or experiment_id,
        "campaign": campaign,
        "controller_id": body.get("controller_id"),
        "note": body.get("note"),
        "status": body.get("status", "running"),
        "expected_jobs": body.get("expected_jobs", 0),
        "singular_dims": body.get("singular_dims") or [],
        "max_concurrent": body.get("max_concurrent", 0),
        "skip_rules": body.get("skip_rules") or {},
        "metric": body.get("metric"),
        "goal": goal,
        "code": code,
        "controls": controls,
        **optimizer,
    }


@routes.post("/api/experiments")
async def handle_create_experiment(request: web.Request) -> web.Response:
    """Create a new experiment.

    Its campaign comes from the body's ``campaign``, else ``?campaign=``, and
    one of them is required.  Re-creating an experiment that already exists in
    another campaign is refused with 409.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    body = await _json_body(request)

    experiment_id = body.get("experiment_id") or body.get("id")
    if not experiment_id:
        return _error_response("'experiment_id' is required")

    # Sanitize experiment_id
    try:
        experiment_id = _sanitize_experiment_id(experiment_id)
    except ValueError as exc:
        return _error_response(str(exc), status=400)

    fields = await _experiment_fields(request, db, body, experiment_id)
    if isinstance(fields, web.Response):
        return fields
    if fields.get("optimize") is None:
        exp = await state.db_writer.create_experiment(**fields)
        await _maybe_link_capture(request, state, body, exp)
        return _json_response(exp, status=201)

    # The first trials exist by the time this returns, so a client never sees a sweep with no runs.
    async with state.lock:
        exp = await state.db_writer.create_experiment(**fields)
        await advance_optimizer_locked(db, state, exp)
    await _maybe_link_capture(request, state, body, exp)
    state.request_schedule()
    return _json_response(exp, status=201)


async def _maybe_link_capture(
    request: web.Request, state: ManagerState, body: dict[str, Any], exp: Any,
) -> None:
    """Link a newly created experiment to the trace that submitted it.

    The body's ``capture_session`` is the agent session the sweep was typed
    into (resolved by the client from its own agent's environment).  A trace
    the manager has not seen, or capture being off, changes nothing.
    """
    session_id = body.get("capture_session") if isinstance(body, dict) else None
    if not session_id:
        return
    experiment_id = (exp.get("experiment_id") if isinstance(exp, dict)
                     else getattr(exp, "experiment_id", None))
    try:
        from mlsweep._manager_capture import link_experiment_to_capture
        await link_experiment_to_capture(state, str(session_id), str(experiment_id))
    except Exception:  # capture is best effort: never fail the submission
        pass


@routes.get("/api/experiments/{experiment_id}")
async def handle_get_experiment(request: web.Request) -> web.Response:
    """Get a single experiment by ID."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    experiment_id = request.match_info["experiment_id"]
    exp = await get_experiment(db, experiment_id)
    if exp is None:
        return _not_found("experiment")
    return _json_response(exp)


_VALID_EXPERIMENT_STATUSES = ("running", "paused", "completed", "aborted")


@routes.put("/api/experiments/{experiment_id}/status")
async def handle_update_experiment_status(request: web.Request) -> web.Response:
    """Update an experiment's status.

    ``paused`` and ``aborted`` cause the scheduler to stop dispatching this
    experiment's pending jobs (the scheduler reads experiment status from the
    DB).  ``aborted`` also cancels all its unfinished jobs, so a stopped sweep
    has one label for them.  Moving back to ``running`` resumes scheduling.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    body = await _json_body(request)
    status = body.get("status")
    if not status:
        return _error_response("'status' is required")
    if status not in _VALID_EXPERIMENT_STATUSES:
        return _error_response(
            f"status must be one of {', '.join(_VALID_EXPERIMENT_STATUSES)}"
        )
    async with state.lock:
        exp = await state.db_writer.update_experiment_status(experiment_id, status)
        if exp is None:
            return _not_found("experiment")
        if status == "running" and exp.optimize is not None:
            await state.db_writer.update_experiment_optimize(
                experiment_id, cleared_for_running(json.loads(exp.optimize)))
        if status == "running" and exp.controls is not None:
            controls = json.loads(exp.controls)
            if controls.pop("note", None) is not None:  # why the manager paused it
                await state.db_writer.update_experiment_controls(experiment_id, controls)
        # Aborting stops the sweep for good, so cancel everything unfinished.
        if status == "aborted":
            pending = await list_jobs_by_experiment(db, experiment_id, status="pending")
            await cancel_runs_locked(db, state, [*state.runs_of(experiment_id), *(j.key for j in pending)])
        # Its last job may have finished while it was paused.
        await _check_experiments_complete_locked(db, state, {experiment_id})
    state.request_schedule()

    # Broadcast event
    _broadcast_experiment_event(request, experiment_id, "status_updated", status=status)
    return _json_response(exp)


async def _optimizer_status(
    db: aiosqlite.Connection, state: ManagerState, experiment_id: str,
) -> dict[str, Any] | web.Response:
    exp = await get_experiment(db, experiment_id)
    if exp is None:
        return _not_found("experiment")
    if exp.optimize is None:
        return _error_response(f"{experiment_id} is not a Bayesian sweep", status=404)
    spec = json.loads(exp.optimize)
    return status_view(exp, spec, await read_progress(db, state, experiment_id, spec))


@routes.get("/api/experiments/{experiment_id}/optimize")
async def handle_get_optimizer(request: web.Request) -> web.Response:
    """A Bayesian sweep's progress: budget, trials with results, running, failed, best so far."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    async with state.lock:
        view = await _optimizer_status(db, state, request.match_info["experiment_id"])
    return view if isinstance(view, web.Response) else _json_response(view)


@routes.patch("/api/experiments/{experiment_id}/optimize")
async def handle_patch_optimizer(request: web.Request) -> web.Response:
    """Change a Bayesian sweep while it runs.

    Body fields, all optional:

    * ``budget`` (int >= 1): how many trials need results.  Setting it also undoes
      ``finish``, and a completed sweep with budget left runs again.
    * ``max_failures`` (int >= 1): failed trials in a row before the sweep pauses.
    * ``finish`` (bool): start no new trials, and complete once the running ones end.

    Runs that already exist are never touched.  To stop them, use ``cancel`` or ``stop``.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    body = await _json_body(request)
    if not isinstance(body, dict):
        return _error_response("body must be an object")
    for field in ("budget", "max_failures"):
        value = body.get(field)
        if field in body and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            return _error_response(f"'{field}' must be a positive int")
    if "finish" in body and not isinstance(body["finish"], bool):
        return _error_response("'finish' must be a bool")

    async with state.lock:
        exp = await get_experiment(db, experiment_id)
        if exp is None:
            return _not_found("experiment")
        if exp.optimize is None:
            return _error_response(f"{experiment_id} is not a Bayesian sweep", status=404)
        spec = json.loads(exp.optimize)
        if "max_failures" in body:
            spec["max_failures"] = body["max_failures"]
        if "budget" in body:
            spec["budget"] = body["budget"]
            spec["finishing"] = False
        if "finish" in body:
            spec["finishing"] = body["finish"]
        await state.db_writer.update_experiment_optimize(experiment_id, spec)
        if exp.status == "completed" and "budget" in body:
            # Completes again on the next pass if there is still nothing to do.
            await state.db_writer.update_experiment_status(experiment_id, "running")
            await state.db_writer.update_experiment_optimize(experiment_id, cleared_for_running(spec))
        view = await _optimizer_status(db, state, experiment_id)
    state.request_schedule()
    return view if isinstance(view, web.Response) else _json_response(view)


@routes.get("/api/experiments/{experiment_id}/controls")
async def handle_get_controls(request: web.Request) -> web.Response:
    """An experiment's run controls (MAX_RUNTIME, PRUNE, METRIC_CHECK), ``{}`` if it has none."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    exp = await get_experiment(db, request.match_info["experiment_id"])
    if exp is None:
        return _not_found("experiment")
    return _json_response(json.loads(exp.controls) if exp.controls else {})


@routes.patch("/api/experiments/{experiment_id}/controls")
async def handle_patch_controls(request: web.Request) -> web.Response:
    """Change an experiment's run controls while it runs.

    Body fields, all optional; ``null`` (or ``false`` for ``metric_check``) turns one off:

    * ``max_runtime``: seconds, or a duration like ``"4h"``.  Applies to runs already running.
    * ``prune``: a PRUNE dict.  Its metric and goal default to the experiment's.
      A new spec compares runs afresh from their stored metrics.
    * ``metric_check``: true, or how many finished runs to judge by.  Turning it on
      again after it fired checks again.

    Returns the controls as they now are.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    body = await _json_body(request)
    if not isinstance(body, dict):
        return _error_response("body must be an object")
    unknown = sorted(set(body) - {"max_runtime", "prune", "metric_check"})
    if unknown:
        return _error_response(f"unknown field(s) {unknown}")
    async with state.lock:
        exp = await get_experiment(db, experiment_id)
        if exp is None:
            return _not_found("experiment")
        current: dict[str, Any] = json.loads(exp.controls) if exp.controls else {}
        merged = {k: current.get(k) for k in ("max_runtime", "prune")}
        merged.update({k: v for k, v in body.items() if k in merged})
        check = current.get("metric_check")
        metric_check: Any = check["runs"] if check else False
        if "metric_check" in body:
            metric_check = body["metric_check"] if body["metric_check"] is not None else False
        try:
            controls = build_controls(
                max_runtime=merged["max_runtime"], prune=merged["prune"], metric_check=metric_check,
                metric=exp.metric, goal=exp.goal, where="controls",
            ) or {}
        except ValueError as exc:
            return _error_response(str(exc))
        if current.get("note"):
            controls["note"] = current["note"]
        await state.db_writer.update_experiment_controls(experiment_id, controls)
        if exp.status in ("running", "paused"):
            state.controls[experiment_id] = controls
    state.request_schedule()
    _broadcast_experiment_event(request, experiment_id, "controls_updated", controls=controls)
    return _json_response(controls)


@routes.put("/api/experiments/{experiment_id}/name")
async def handle_update_experiment_name(request: web.Request) -> web.Response:
    """Update an experiment's display name."""
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    body = await _json_body(request)
    name = body.get("name")
    if not name or not isinstance(name, str):
        return _error_response("'name' is required")
    name = name.strip()
    exp = await state.db_writer.update_experiment_name(experiment_id, name)
    if exp is None:
        return _not_found("experiment")
    _broadcast_experiment_event(request, experiment_id, "name_updated", name=name)
    return _json_response(exp)


@routes.put("/api/experiments/{experiment_id}/max_concurrent")
async def handle_update_experiment_max_concurrent(request: web.Request) -> web.Response:
    """Set an experiment's max concurrent running jobs (0 = unlimited).

    The cap is enforced by the scheduler each pass, so lowering it does not kill
    already-running jobs; it just stops new ones from starting until the running
    count drops below the cap.
    """
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    body = await _json_body(request)
    value = body.get("max_concurrent")
    if value is None or not isinstance(value, int) or value < 0:
        return _error_response("'max_concurrent' (int >= 0) is required")
    exp = await state.db_writer.update_experiment_max_concurrent(experiment_id, value)
    if exp is None:
        return _not_found("experiment")
    state.request_schedule()
    return _json_response(exp)


@routes.delete("/api/experiments/{experiment_id}")
async def handle_delete_experiment(request: web.Request) -> web.Response:
    """Delete an experiment and all its jobs.

    Any jobs still in-flight are first stopped on their workers (so we don't
    leave orphaned processes running for a deleted experiment), then the rows
    are removed.  Pending jobs simply vanish from the scheduler's DB query.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]

    async with state.lock:
        # Stop in-flight runs for this experiment before deleting their rows.
        _detach_locked(state, state.runs_of(experiment_id))
        existed = await state.db_writer.delete_experiment(experiment_id)
    state.request_schedule()
    if not existed:
        return _not_found("experiment")
    return _json_response({"deleted": experiment_id})


@routes.get("/api/experiments/{experiment_id}/summary")
async def handle_experiment_summary(request: web.Request) -> web.Response:
    """Get a summary of an experiment (metadata + job counts)."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    summary = await experiment_summary(db, experiment_id)
    if summary["name"] is None:
        return _not_found("experiment")
    # Fold the in-memory stall signal in so clients (e.g. `mlsweep wait
    # --until stalled`) get it without a second round trip.  Only jobs the
    # database considers running can be stalled.
    now = time.time()
    running_ids = {
        j.run_id for j in await list_jobs_by_experiment(db, experiment_id, status="running")
    }
    stalled_runs = [
        run.run_id for run in _experiment_runs(state, experiment_id)
        if run.run_id in running_ids
        and _stall_seconds(run, now) >= MANAGER_STALL_THRESHOLD_SECONDS
    ]
    summary["stalled_runs"] = sorted(stalled_runs)
    summary["stalled_jobs"] = len(stalled_runs)
    return _json_response(summary)


@routes.get("/api/experiments/{experiment_id}/jobs")
async def handle_list_experiment_jobs(request: web.Request) -> web.Response:
    """List jobs for an experiment, optionally filtered by status.

    Running jobs additionally carry ``stall_seconds`` (seconds since the last
    log/metric progress) and ``stalled`` (whether that exceeds the manager's
    default threshold), merged from in-memory run state.  Once a metric has
    arrived this session they also carry ``last_step``, ``step_age_seconds``
    (seconds since that step first arrived), ``step_seconds`` (wall seconds
    per step as the run's logger measured it, or null until it has one) and
    ``avg_step_seconds`` (wall seconds per step since the first step this
    session, or null until the step has advanced).
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    status_filter = _status_param(request)
    jobs = await list_jobs_by_experiment(db, experiment_id, status=status_filter)  # type: ignore[arg-type]
    now = time.time()
    out: list[dict[str, Any]] = []
    for job in jobs:
        d = dataclasses.asdict(job)
        if job.status == "running":
            run = state.runs.get((experiment_id, job.run_id))
            if run is not None:
                stall = _stall_seconds(run, now)
                d["stall_seconds"] = stall
                d["stalled"] = stall >= MANAGER_STALL_THRESHOLD_SECONDS
                if run.last_step is not None:
                    d["last_step"] = run.last_step
                    d["step_age_seconds"] = max(0.0, now - run.last_step_time)
                    d["step_seconds"] = run.step_seconds
                    d["avg_step_seconds"] = run.avg_step_seconds
        out.append(d)
    return _json_response(out)


@routes.get("/api/experiments/{experiment_id}/download")
async def handle_download_experiment(request: web.Request) -> web.StreamResponse:
    """Stream experiment directory as a ``.tar.gz`` download.

    Returns 404 if the experiment is not found or its directory does not
    exist on disk.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    experiment_id = request.match_info["experiment_id"]

    # Verify experiment exists in DB
    exp = await get_experiment(db, experiment_id)
    if exp is None:
        return _not_found("experiment")

    # Locate experiment directory on disk
    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    exp_dir = mlsweep_dir / "experiments" / experiment_id

    if not exp_dir.is_dir():
        return _not_found("experiment")

    # Stream tar.gz via subprocess to avoid blocking the event loop
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "application/gzip",
            "Content-Disposition": (
                f'attachment; filename="{experiment_id}.tar.gz"'
            ),
        },
    )
    await response.prepare(request)

    proc = await asyncio.create_subprocess_exec(
        "tar", "czf", "-", "-C", str(exp_dir), ".",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdout is not None
    try:
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            await response.write(chunk)
    finally:
        # Ensure the subprocess is cleaned up
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        await response.write_eof()

    return response


# ── Jobs ───────────────────────────────────────────────────────────────────────


@routes.get("/api/jobs")
async def handle_list_jobs(request: web.Request) -> web.Response:
    """List jobs.

    Query params:
      - experiment_id: filter by experiment
      - campaign: only jobs of experiments in this campaign
      - status: filter by status (default: 'pending')
      - limit: max number of results
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    experiment_id = request.query.get("experiment_id")
    campaign = request.query.get("campaign")
    status = _status_param(request, "pending")
    limit = _limit_param(request)

    if experiment_id:
        jobs = await list_jobs_by_experiment(db, experiment_id, status=status)  # type: ignore[arg-type]
        if limit is not None:
            jobs = jobs[:limit]
    elif status == "pending":
        jobs = await list_pending_jobs(db, limit=limit, campaign=campaign)
    else:
        jobs = await list_jobs_by_status(db, status, limit=limit, campaign=campaign)  # type: ignore[arg-type]

    return _json_response(jobs)


@routes.post("/api/jobs")
async def handle_insert_job(request: web.Request) -> web.Response:
    """Insert a single job."""
    state: ManagerState = request.config_dict[STATE_KEY]

    body = await _json_body(request)

    run_id = body.get("run_id")
    experiment_id = body.get("experiment_id")
    if not run_id or not experiment_id:
        return _error_response("'run_id' and 'experiment_id' are required")

    try:
        job = await state.db_writer.insert_job(
            run_id=run_id,
            experiment_id=experiment_id,
            priority=body.get("priority", 0),
            command=body.get("command", []),
            combo=body.get("combo"),
            env=body.get("env"),
            status=body.get("status", "pending"),
            gpus_per_run=body.get("gpus_per_run", 1),
            nodes_per_run=body.get("nodes_per_run", 1),
            set_dist_env=body.get("set_dist_env", False),
            run_from=body.get("run_from"),
            return_files=body.get("return_files"),
            files=body.get("files"),
            max_retries=body.get("max_retries", 2),
            artifact_id=body.get("artifact_id"),
            setup_command=body.get("setup_command"),
        )
    except ValueError as exc:
        return _error_response(str(exc))
    except sqlite3.IntegrityError as exc:
        return _job_insert_refused(exc)

    state.request_schedule()

    return _json_response(job, status=201)


@routes.post("/api/jobs/bulk")
async def handle_insert_jobs_bulk(request: web.Request) -> web.Response:
    """Insert multiple jobs in a single transaction."""
    state: ManagerState = request.config_dict[STATE_KEY]

    try:
        body = await _request_json(request)
    except ValueError:
        return _error_response("invalid JSON body")

    jobs_data = body if isinstance(body, list) else body.get("jobs") if isinstance(body, dict) else None

    if not jobs_data or not isinstance(jobs_data, list):
        return _error_response("provide a JSON array of job objects")

    try:
        records = await state.db_writer.insert_jobs_bulk(jobs_data)
    except ValueError as exc:
        return _error_response(str(exc))
    except sqlite3.IntegrityError as exc:
        return _job_insert_refused(exc)

    state.request_schedule()

    return _json_response(records, status=201)


@routes.post("/api/experiments/{experiment_id}/jobs/bulk")
async def handle_create_experiment_with_jobs(request: web.Request) -> web.Response:
    """Create (or update) an experiment and insert its jobs atomically.

    Either the experiment row and every job land, or nothing does.  A duplicate
    run_id inside the experiment aborts the whole submission with 409.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    try:
        experiment_id = _sanitize_experiment_id(experiment_id)
    except ValueError as exc:
        return _error_response(str(exc), status=400)

    body = await _json_body(request)

    jobs = body.get("jobs")
    if not isinstance(jobs, list) or not jobs or not all(isinstance(job, dict) for job in jobs):
        return _error_response("'jobs' must be a non-empty array of job objects")
    for job in jobs:
        job["experiment_id"] = experiment_id

    if body.get("optimize") is not None:
        return _error_response("a Bayesian sweep is created without jobs; the manager chooses them")
    fields = await _experiment_fields(request, db, body, experiment_id)
    if isinstance(fields, web.Response):
        return fields
    try:
        exp, records = await state.db_writer.create_experiment_with_jobs(jobs=jobs, **fields)
    except ValueError as exc:
        return _error_response(str(exc))
    except sqlite3.IntegrityError:
        return _error_response(
            "duplicate run_id in this experiment (run names must be unique per "
            "experiment); nothing was created — fix the sweep's names and resubmit",
            status=409,
        )

    await _maybe_link_capture(request, state, body, exp)
    state.request_schedule()
    return _json_response({"experiment": exp, "jobs": records}, status=201)


@routes.get("/api/jobs/pending")
async def handle_list_pending_jobs(request: web.Request) -> web.Response:
    """List pending jobs, optionally filtered by experiment and campaign."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    experiment_id = request.query.get("experiment_id")
    limit = _limit_param(request)
    jobs = await list_pending_jobs(
        db, experiment_id=experiment_id, limit=limit, campaign=request.query.get("campaign"),
    )
    return _json_response(jobs)


@routes.get("/api/jobs/{run_id}")
async def handle_get_job(request: web.Request) -> web.Response:
    """Get a single job by run_id and experiment_id."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    run_id = request.match_info["run_id"]
    experiment_id = request.query.get("experiment_id", "")
    job = await get_job(db, run_id, experiment_id)
    if job is None:
        return _not_found("job")
    return _json_response(job)


@routes.put("/api/jobs/{run_id}/status")
async def handle_update_job_status(request: web.Request) -> web.Response:
    """Update a job's status (and optionally other fields)."""
    state: ManagerState = request.config_dict[STATE_KEY]
    run_id = request.match_info["run_id"]
    body = await _json_body(request)

    status = body.get("status")
    if not status:
        return _error_response("'status' is required")
    experiment_id = body.get("experiment_id", "")
    if not experiment_id:
        return _error_response("'experiment_id' is required")

    if status not in JOB_STATUSES:
        return _error_response(f"'status' must be one of {list(JOB_STATUSES)}")
    if status in ACTIVE_JOB_STATUSES:
        return _error_response("only the scheduler dispatches jobs", status=400)

    # Only result columns may be set alongside the status.
    kwargs = {k: v for k, v in body.items() if k in ("exit_code", "elapsed")}
    if any(v is not None and not _is_number(v) for v in kwargs.values()):
        return _error_response("'exit_code' and 'elapsed' must be numbers")

    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    async with state.lock:
        # A job in flight must be cancelled instead, so its worker stops it.
        job = await state.db_writer.update_job_status(
            run_id, experiment_id, status,
            only_from=("pending", *FINISHED_JOB_STATUSES), **kwargs,
        )
        if job is None:
            if await get_job(db, run_id, experiment_id) is None:
                return _not_found("job")
            return _error_response("job is in flight; cancel it instead", status=409)
        await _check_experiments_complete_locked(db, state, {experiment_id})
    state.request_schedule()

    # Broadcast event
    _broadcast_experiment_event(
        request, job.experiment_id, "job_updated",
        run_id=run_id, status=status,
    )

    return _json_response(job)


@routes.put("/api/jobs/{run_id}/priority")
async def handle_update_job_priority(request: web.Request) -> web.Response:
    """Update a pending job's priority."""
    state: ManagerState = request.config_dict[STATE_KEY]
    run_id = request.match_info["run_id"]
    body = await _json_body(request)

    priority = body.get("priority")
    if not _is_number(priority):
        return _error_response("'priority' (a number) is required")
    experiment_id = body.get("experiment_id", "")
    if not experiment_id:
        return _error_response("'experiment_id' is required")

    # Update in DB — the scheduler reads pending jobs (ordered by priority)
    # straight from the DB, so this takes effect on the next pass.
    job = await state.db_writer.update_job_priority(run_id, experiment_id, priority)
    if job is None:
        return _not_found("job")

    state.request_schedule()

    # Broadcast event
    _broadcast_experiment_event(
        request, job.experiment_id, "priority_changed",
        run_id=run_id, priority=priority,
    )

    return _json_response(job)


@routes.put("/api/jobs/{run_id}/label")
async def handle_update_job_label(request: web.Request) -> web.Response:
    """Set or clear a job's human-readable label."""
    state: ManagerState = request.config_dict[STATE_KEY]
    run_id = request.match_info["run_id"]
    body = await _json_body(request)
    experiment_id = body.get("experiment_id", "")
    if not experiment_id:
        return _error_response("'experiment_id' is required")
    label = body.get("label")
    if label is not None:
        if not isinstance(label, str):
            return _error_response("'label' must be a string or null")
        label = label.strip() or None
    job = await state.db_writer.update_job_label(run_id, experiment_id, label)
    if job is None:
        return _not_found("job")
    _broadcast_experiment_event(
        request, job.experiment_id, "job_updated",
        run_id=run_id, label=label,
    )
    return _json_response(job)


@routes.post("/api/jobs/{run_id}/cancel")
async def handle_cancel_job(request: web.Request) -> web.Response:
    """Cancel a job (pending or running) via the unified cancel path."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    run_id = request.match_info["run_id"]
    experiment_id = request.query.get("experiment_id", "")

    job = await get_job(db, run_id, experiment_id)
    if job is None:
        return _not_found("job")

    # One path stops any in-flight run on its worker, marks the row cancelled,
    # and broadcasts job_done. Works whether the job is pending or running.
    async with state.lock:
        await cancel_runs_locked(db, state, [(experiment_id, run_id)])

    updated = await get_job(db, run_id, experiment_id)
    return _json_response(updated if updated is not None else job)


@routes.post("/api/jobs/{run_id}/reschedule")
async def handle_reschedule_job(request: web.Request) -> web.Response:
    """Restart an in-flight job elsewhere, or choose where a pending one starts.

    Accepts JSON: {"worker_id": str, "gpus": [int, ...]}, both optional.  An
    in-flight run is stopped and put back to pending without spending a retry
    (it restarts from scratch, or from its checkpoint if the script resumes).
    With *worker_id* the job is pinned there: its next dispatch takes those
    GPUs from the worker's pool, filling up to ``gpus_per_run`` with the least
    busy others, and ignores the worker's jobs-per-GPU cap.  Without it, the
    scheduler places it as usual.  Multi-node jobs cannot be pinned.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    run_id = request.match_info["run_id"]
    experiment_id = request.query.get("experiment_id", "")
    body = await _json_body(request)

    job = await get_job(db, run_id, experiment_id)
    if job is None:
        return _not_found("job")
    if job.status not in ("pending", "dispatched", "running"):
        return _error_response(f"job is {job.status}; only pending and in-flight jobs can be rescheduled", status=409)
    worker_id = body.get("worker_id")
    gpus = body.get("gpus") or []
    if worker_id is not None and not isinstance(worker_id, str):
        return _error_response("'worker_id' must be a string")
    if not (isinstance(gpus, list) and all(isinstance(g, int) and not isinstance(g, bool) for g in gpus)):
        return _error_response("'gpus' must be a list of GPU indices")
    if gpus and worker_id is None:
        return _error_response("'gpus' needs a 'worker_id'")

    key = (job.experiment_id, run_id)
    async with state.lock:
        pin: Pin | None = None
        if worker_id is not None:
            if job.nodes_per_run > 1:
                return _error_response("a multi-node job cannot be pinned to one worker")
            wc = state.workers.get(worker_id)
            if wc is None or wc.status != "connected":
                return _error_response(f"worker {worker_id} is not connected", status=409)
            if not set(gpus) <= set(wc.gpus):
                return _error_response(f"GPUs {sorted(set(gpus) - set(wc.gpus))} are not in {worker_id}'s pool")
            if not_usable := sorted(set(gpus) - set(wc.usable_gpus)):
                return _error_response(f"GPUs {not_usable} on {worker_id} have not passed the health probe",
                                       status=409)
            if len(set(gpus)) > job.gpus_per_run:
                return _error_response(f"the job uses {job.gpus_per_run} GPU(s)")
            occ = state.occupancy(wc)
            # The run's own GPUs free up when it is requeued below.
            for g in (state.runs[key].nodes.get(worker_id, ()) if key in state.runs else ()):
                if g in occ:
                    occ[g] -= 1
            chosen = sorted(set(gpus))
            others = sorted((g for g in wc.usable_gpus if g not in chosen), key=lambda g: occ[g])
            chosen += others[:job.gpus_per_run - len(chosen)]
            if len(chosen) < job.gpus_per_run:
                return _error_response(f"{worker_id} has only {len(wc.usable_gpus)} usable GPU(s) in its pool")
            pin = Pin(worker_id=worker_id, gpus=sorted(chosen))
        requeued = key in state.runs
        if requeued:
            await requeue_runs_locked(db, state, [key], lost=False)
        if pin is not None:
            state.pins[key] = pin
        else:
            state.pins.pop(key, None)
        state.request_schedule()
    return _json_response({
        "run_id": run_id, "experiment_id": job.experiment_id, "requeued": requeued,
        "pinned": {"worker_id": pin.worker_id, "gpus": pin.gpus} if pin else None,
    })


@routes.post("/api/jobs/{run_id}/retry")
async def handle_retry_job(request: web.Request) -> web.Response:
    """Retry a finished job: reset it to pending with a fresh retry budget.

    Only jobs in a terminal state (``failed``, ``cancelled``, ``done``)
    can be retried.  Running or pending jobs return 409 Conflict.
    ``max_retries`` does not apply here.  It only caps automatic retries.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    run_id = request.match_info["run_id"]
    experiment_id = request.query.get("experiment_id", "")

    job = await state.db_writer.retry_job(run_id, experiment_id)
    if job is None:
        current = await get_job(db, run_id, experiment_id)
        if current is None:
            return _not_found("job")
        return _error_response(
            f"job is {current.status}; only finished jobs can be retried", status=409,
        )

    # Broadcast event
    _broadcast_experiment_event(
        request, job.experiment_id, "job_retried", run_id=run_id,
    )

    state.request_schedule()
    return _json_response(job)


@routes.delete("/api/experiments/{experiment_id}/jobs/{run_id}")
async def handle_delete_job(request: web.Request) -> web.Response:
    """Cancel a job by experiment_id and run_id via the unified cancel path.

    Handles pending and running jobs identically: any in-flight run is stopped
    on its worker (freeing its GPUs), the row is marked 'cancelled', and
    job_done is broadcast.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]

    job = await get_job(db, run_id, experiment_id)
    if job is None or job.experiment_id != experiment_id:
        return _not_found("job")

    async with state.lock:
        await cancel_runs_locked(db, state, [(experiment_id, run_id)])

    updated = await get_job(db, run_id, experiment_id)
    return _json_response(updated if updated is not None else job)


def _run_ids_from(body: dict[str, Any]) -> list[str] | None:
    ids = body.get("run_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
        return None
    return list(dict.fromkeys(ids))


async def _not_finished(
    db: aiosqlite.Connection, state: ManagerState, experiment_id: str, run_ids: list[str],
) -> dict[str, str]:
    """``{run_id: why}`` for runs that do not exist or could still change.  Hold ``state.lock``."""
    statuses = await job_statuses(db, experiment_id, run_ids)
    problems: dict[str, str] = {}
    for rid in run_ids:
        status = statuses.get(rid)
        if status is None:
            problems[rid] = f"{rid}: no such run in {experiment_id}"
        elif status not in FINISHED_JOB_STATUSES or (experiment_id, rid) in state.runs:
            problems[rid] = f"{rid}: still {status}; cancel it first"
    return problems


def _run_conflict(problems: dict[str, str]) -> web.Response:
    return _json_response({"error": "; ".join(problems.values()), "problems": list(problems.values())}, status=409)


@routes.post("/api/experiments/{experiment_id}/jobs/move")
async def handle_move_jobs(request: web.Request) -> web.Response:
    """Move finished runs to another experiment, with their metrics, logs and files.

    Body: ``{"run_ids": [...], "to": TARGET_EXPERIMENT}``.  Refused (409) if a
    run is still pending or running, or if TARGET already has a run with the
    same ID.  With ``?campaign=``, both experiments must be in that campaign.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    source = request.match_info["experiment_id"]
    body = await _json_body(request)
    run_ids = _run_ids_from(body)
    target = body.get("to")
    if run_ids is None or not isinstance(target, str) or not target:
        return _error_response("expected {run_ids: [...], to: EXPERIMENT_ID}")
    if target == source:
        return _error_response("source and target are the same experiment")
    src_exp, dst_exp = await get_experiment(db, source), await get_experiment(db, target)
    if src_exp is None:
        return _not_found("experiment")
    if dst_exp is None:
        return _not_found(f"target experiment {target!r}")
    campaign = request.query.get("campaign")
    if campaign is not None and dst_exp.campaign != campaign:
        return _wrong_campaign(target, dst_exp.campaign, campaign)
    try:
        _sanitize_experiment_id(target)
    except ValueError as exc:
        return _error_response(str(exc))

    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    async with state.lock:
        problems = await _not_finished(db, state, source, run_ids)
        taken = await job_statuses(db, target, run_ids)

        def _dirs_taken() -> set[str]:
            return {rid for rid in run_ids if _run_dir_exists(mlsweep_dir, target, rid)}
        dirs_taken = await asyncio.to_thread(_dirs_taken)
        for rid in run_ids:
            if rid in problems:
                continue
            if rid in taken:
                problems[rid] = f"{rid}: {target} already has a run with this ID"
            elif rid in dirs_taken:
                problems[rid] = f"{rid}: {target} already has files for a run with this ID"
        if problems:
            return _run_conflict(problems)
        moved = await state.db_writer.move_jobs(source, target, run_ids)

        def _move_dirs() -> list[str]:
            failed = []
            for rid in moved:
                try:
                    src_dir = _run_dir(mlsweep_dir, source, rid)
                    if not src_dir.exists():
                        continue
                    dst_dir = _run_dir(mlsweep_dir, target, rid)
                    dst_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_dir), str(dst_dir))
                except (OSError, ValueError) as exc:
                    logger.warning("moving files of %s/%s: %s", source, rid, exc)
                    failed.append(rid)
            return failed
        failed = await asyncio.to_thread(_move_dirs)
    for eid in (source, target):
        _broadcast_experiment_event(request, eid, "jobs_moved", run_ids=moved, source=source, target=target)
    result: dict[str, Any] = {"moved": moved, "from": source, "to": target}
    if failed:
        result["files_not_moved"] = failed
    return _json_response(result)


@routes.post("/api/experiments/{experiment_id}/jobs/purge")
async def handle_purge_jobs(request: web.Request) -> web.Response:
    """Delete finished runs for good: rows, metrics, logs and their files.

    Body: ``{"run_ids": [...]}``.  Refused (409) for runs still pending or
    running; cancel them first.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    run_ids = _run_ids_from(await _json_body(request))
    if run_ids is None:
        return _error_response("expected {run_ids: [...]}")
    if await get_experiment(db, experiment_id) is None:
        return _not_found("experiment")
    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    async with state.lock:
        problems = await _not_finished(db, state, experiment_id, run_ids)
        if problems:
            return _run_conflict(problems)
        purged = await state.db_writer.purge_jobs(experiment_id, run_ids)

    def _rm() -> int:
        freed = 0
        for rid in purged:
            try:
                d = _run_dir(mlsweep_dir, experiment_id, rid)
            except ValueError:  # an old row with a bad ID: its row is gone, leave the disk alone
                continue
            if d.is_dir():
                freed += _dir_size(d)
                shutil.rmtree(d, ignore_errors=True)
        return freed
    freed = await asyncio.to_thread(_rm)
    _broadcast_experiment_event(request, experiment_id, "jobs_purged", run_ids=purged)
    return _json_response({"purged": purged, "bytes_freed": freed})


_GC_MIN_AGE = 3600.0     # younger files may be about to be used


def _orphan_dirs(root: Path, orphans: dict[str, Any], active: set[Any], now: float) -> list[dict[str, Any]]:
    """Run directories under *root* whose experiment or run no longer exists."""
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.is_symlink():
            continue
        if exp_dir.name not in orphans["experiments"]:
            if now - exp_dir.stat().st_mtime > _GC_MIN_AGE:
                found.append({"path": f"experiments/{exp_dir.name}", "reason": "experiment deleted",
                              "bytes": _dir_size(exp_dir)})
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            key = (exp_dir.name, run_dir.name)
            if key not in orphans["runs"] and key not in active and run_dir.is_dir() \
                    and not run_dir.is_symlink() and now - run_dir.stat().st_mtime > _GC_MIN_AGE:
                found.append({"path": f"experiments/{exp_dir.name}/{run_dir.name}",
                              "reason": "run deleted", "bytes": _dir_size(run_dir)})
    return found


def _orphan_archives(art_dir: Path, orphans: dict[str, Any], now: float) -> list[dict[str, Any]]:
    """Code archives under *art_dir* that no run uses."""
    if not art_dir.is_dir():
        return []
    unreferenced = {a["artifact_id"] for a in orphans["artifacts"]}
    found: list[dict[str, Any]] = []
    for f in sorted(art_dir.iterdir()):
        if not f.name.endswith(".tar.gz") or not f.is_file():
            continue
        aid = f.name[: -len(".tar.gz")]
        untracked = aid not in orphans["known_artifacts"] and aid not in orphans["referenced_artifacts"]
        if aid in unreferenced or (untracked and now - f.stat().st_mtime > _GC_MIN_AGE):
            found.append({"path": f"artifacts/{f.name}", "reason": "code archive no run uses",
                          "bytes": f.stat().st_size})
    return found


def _gc_scan(mlsweep_dir: Path, orphans: dict[str, Any], active: set[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now = time.time()
    return (_orphan_dirs(mlsweep_dir / "experiments", orphans, active, now),
            _orphan_archives(mlsweep_dir / "artifacts", orphans, now))


@routes.post("/api/gc")
async def handle_gc(request: web.Request) -> web.Response:
    """Find (and with ``{"dry_run": false}``, remove) what nothing refers to.

    * run directories of experiments or runs that no longer exist;
    * database rows (metrics, logs, run nodes) whose run is gone;
    * uploaded code archives and snapshots no run uses, older than an hour, and
      stored file contents no remaining snapshot lists.

    ``{"vacuum": true}`` also compacts the database file afterwards.  The
    default is a dry run that changes nothing.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    try:
        body = await request.json() if request.can_read_body else {}
    except Exception:
        return _error_response("invalid JSON body")
    if not isinstance(body, dict):
        body = {}
    dry_run = bool(body.get("dry_run", True))
    vacuum = bool(body.get("vacuum", False))
    mlsweep_dir: Path = request.config_dict[DIR_KEY]

    # A dry run changes nothing, so it does not hold up the scheduler.  A real
    # one keeps the lock until it has deleted: a run moved in meanwhile would
    # look like an orphan, since moving keeps a directory's old mtime.
    from mlsweep._manager_code import object_gc_plan, remove_objects

    guard: AsyncContextManager[Any] = contextlib.nullcontext() if dry_run else state.lock
    async with guard:
        orphans = await find_orphans(db, artifact_min_age=_GC_MIN_AGE)
        active = set(state.runs)
        dirs, files = await asyncio.to_thread(_gc_scan, mlsweep_dir, orphans, active)
        objects = await object_gc_plan(db, mlsweep_dir, {a["artifact_id"] for a in orphans["artifacts"]},
                                       min_age=_GC_MIN_AGE, now=time.time())
        object_bytes = sum(size for _, size in objects)
        plan = {
            "dry_run": dry_run,
            "directories": dirs,
            "artifact_files": files,
            "objects": {"count": len(objects), "bytes": object_bytes},
            "database": orphans["rows"],
            "artifact_rows": len(orphans["artifacts"]),
            "bytes": sum(item["bytes"] for item in dirs + files) + object_bytes,
        }
        if dry_run:
            return _json_response(plan)
        deleted_rows = await state.db_writer.delete_orphans([a["artifact_id"] for a in orphans["artifacts"]])
        await asyncio.to_thread(remove_objects, mlsweep_dir, [h for h, _ in objects], min_age=_GC_MIN_AGE)

        def _delete() -> int:
            n = 0
            for item in dirs + files:
                try:
                    p = Path(_resolve_safe_subpath(mlsweep_dir, item["path"]))
                except ValueError:
                    continue
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    n += 1
                elif p.is_file():
                    p.unlink(missing_ok=True)
                    n += 1
            return n
        removed = await asyncio.to_thread(_delete)
    if vacuum:
        await state.db_writer.vacuum()
    plan["removed_paths"] = removed
    plan["deleted_rows"] = deleted_rows
    plan["vacuumed"] = vacuum
    return _json_response(plan)


@routes.patch("/api/experiments/{experiment_id}/jobs/{run_id}")
async def handle_patch_job(request: web.Request) -> web.Response:
    """Update a job's priority (reorder).

    Accepts JSON: {priority: int}.  Updates the DB and re-sorts the in-memory
    pending list if the job is pending.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]

    body = await _json_body(request)

    priority = body.get("priority")
    if not _is_number(priority):
        return _error_response("'priority' (a number) is required")

    # Verify job exists and belongs to experiment
    job = await get_job(db, run_id, experiment_id)
    if job is None:
        return _not_found("job")
    if job.experiment_id != experiment_id:
        return _not_found("job")

    # Update priority in DB (works for any status). The scheduler reads pending
    # jobs ordered by priority from the DB, so this re-orders the next pass.
    job = await state.db_writer.update_job_priority(run_id, experiment_id, priority)
    if job is None:
        return _error_response("failed to update priority", status=500)

    state.request_schedule()

    # Broadcast event
    _broadcast_experiment_event(
        request, experiment_id, "priority_changed",
        run_id=run_id, priority=priority,
    )

    return _json_response(job)


# ── Job sub-resources ──────────────────────────────────────────────────────────


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/metrics")
async def handle_get_job_metrics(request: web.Request) -> web.Response:
    """Return all logged metrics for a job as JSONL (one row per step)."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]
    rows = await get_metrics_for_run(db, run_id, experiment_id)
    if not rows:
        return _not_found("metrics")
    jsonl = "\n".join(json.dumps(row) for row in rows)
    return web.Response(text=jsonl, content_type="text/plain")


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/logs")
async def handle_get_job_log(request: web.Request) -> web.Response:
    """Return training log for a job from the database."""
    db = request.config_dict[DB_KEY]
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]

    text = await get_logs_for_run(db, run_id, experiment_id)
    if not text:
        return _not_found("log")

    return web.Response(text=text, content_type="text/plain")


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/artifacts")
async def handle_list_job_artifacts(request: web.Request) -> web.Response:
    """List files in a job's artifacts/ directory, recursively.

    Returns a JSON array of ``{path, size, modified}`` objects sorted by path.
    Returns an empty array if the artifacts directory does not exist.
    """
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]

    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    artifacts_dir = mlsweep_dir / "experiments" / experiment_id / run_id / "artifacts"

    if not artifacts_dir.is_dir():
        return _json_response([])

    files = []
    for p in sorted(artifacts_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(artifacts_dir)).replace("\\", "/")
            stat = p.stat()
            files.append({
                "path": rel,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })

    return _json_response(files)


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/artifacts.zip")
async def handle_zip_job_artifacts(request: web.Request) -> web.StreamResponse:
    """Serve a zip of all artifact files for a single run."""
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]
    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    artifacts_dir = mlsweep_dir / "experiments" / experiment_id / run_id / "artifacts"
    if not artifacts_dir.is_dir():
        return _error_response("no artifacts", status=404)
    loop = asyncio.get_running_loop()
    return await _serve_temp_zip(
        lambda tmp_path: loop.run_in_executor(None, _zip_directory, tmp_path, artifacts_dir),
        f"{run_id[:12]}-artifacts.zip",
    )


@routes.get("/api/experiments/{experiment_id}/artifacts.zip")
async def handle_zip_experiment_artifacts(request: web.Request) -> web.StreamResponse:
    """Serve a zip of all artifact files for every run in an experiment."""
    experiment_id = request.match_info["experiment_id"]
    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    exp_dir = mlsweep_dir / "experiments" / experiment_id
    if not exp_dir.is_dir():
        return _error_response("no experiment artifacts", status=404)
    loop = asyncio.get_running_loop()
    return await _serve_temp_zip(
        lambda tmp_path: loop.run_in_executor(None, _zip_experiment_artifacts, tmp_path, exp_dir),
        f"{experiment_id[:16]}-artifacts.zip",
    )


@routes.get("/api/experiments/{experiment_id}/metrics.zip")
async def handle_zip_experiment_metrics(request: web.Request) -> web.StreamResponse:
    """Serve a zip of all metrics files (JSONL) for every run in an experiment."""
    experiment_id = request.match_info["experiment_id"]
    db = request.config_dict[DB_KEY]

    jobs = await list_jobs_by_experiment(db, experiment_id)
    if not jobs:
        return _error_response(f"no jobs in experiment {experiment_id!r}", status=404)

    async def fill(tmp_path: str) -> None:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for job in jobs:
                rows = await get_metrics_for_run(db, job.run_id, experiment_id)
                if rows:
                    jsonl = "\n".join(json.dumps(row) for row in rows)
                    zf.writestr(f"{job.run_id}.jsonl", jsonl)

    return await _serve_temp_zip(fill, f"{experiment_id[:16]}-metrics.zip")


@routes.get("/api/experiments/{experiment_id}/logs.zip")
async def handle_zip_experiment_logs(request: web.Request) -> web.StreamResponse:
    """Serve a zip of all log files for every run in an experiment."""
    experiment_id = request.match_info["experiment_id"]
    db = request.config_dict[DB_KEY]

    jobs = await list_jobs_by_experiment(db, experiment_id)
    if not jobs:
        return _error_response(f"no jobs in experiment {experiment_id!r}", status=404)

    async def fill(tmp_path: str) -> None:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for job in jobs:
                text = await get_logs_for_run(db, job.run_id, experiment_id)
                if text:
                    zf.writestr(f"{job.run_id}.log", text)

    return await _serve_temp_zip(fill, f"{experiment_id[:16]}-logs.zip")


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/artifacts/{path:.*}")
async def handle_get_job_artifact(request: web.Request) -> web.StreamResponse:
    """Serve a file from a job's artifacts/ directory."""
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]
    artifact_path = request.match_info["path"]

    mlsweep_dir: Path = request.config_dict[DIR_KEY]

    try:
        artifacts_root = mlsweep_dir / "experiments" / experiment_id / run_id / "artifacts"
        file_path = Path(_resolve_safe_subpath(artifacts_root, artifact_path))
    except ValueError:
        return _error_response("path traversal denied", status=403)
    except (OSError, TypeError):
        return _error_response("invalid path", status=400)

    if not file_path.is_file():
        return _not_found("artifact file")

    guessed = mimetypes.guess_type(file_path.name)[0] or ""
    as_text = (guessed.startswith("text/") and guessed != "text/html") or guessed == "application/json"
    return web.FileResponse(file_path, headers=untrusted_file_headers(file_path.name, as_text=as_text))


# Summaries keyed by (path, size, mtime): the Compare grid asks for the same
# checkpoint once per run card, and a big pickle takes a moment to walk.
_INSPECT_CACHE: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
_INSPECT_CACHE_SIZE = 64


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/artifact-info/{path:.*}")
async def handle_inspect_job_artifact(request: web.Request) -> web.Response:
    """Describe a model or array artifact: tensor names, dtypes, shapes and metadata.

    The file is parsed without torch or numpy and never unpickled for real
    (see ``_inspect_model``).  A 422 means the format is not one it reads.
    """
    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    try:
        root = mlsweep_dir / "experiments" / request.match_info["experiment_id"] / request.match_info["run_id"] / "artifacts"
        file_path = Path(_resolve_safe_subpath(root, request.match_info["path"]))
    except ValueError:
        return _error_response("path traversal denied", status=403)
    except (OSError, TypeError):
        return _error_response("invalid path", status=400)
    try:
        st = file_path.stat()
    except OSError:
        return _not_found("artifact file")
    if not file_path.is_file():
        return _not_found("artifact file")
    key = (str(file_path), st.st_size, st.st_mtime_ns)
    info = _INSPECT_CACHE.get(key)
    if info is None:
        try:
            info = await asyncio.to_thread(inspect_file, file_path)
        except InspectError as exc:
            return _error_response(str(exc), status=422)
        except OSError as exc:
            return _error_response(f"could not read file: {exc.strerror or exc}", status=500)
        _INSPECT_CACHE[key] = info
        while len(_INSPECT_CACHE) > _INSPECT_CACHE_SIZE:
            _INSPECT_CACHE.popitem(last=False)
    else:
        _INSPECT_CACHE.move_to_end(key)
    return _json_response(info)


# ── Export (TensorBoard / W&B) ─────────────────────────────────────────────────
#
# Each export covers a scope: one experiment (``/api/experiments/{id}/...`` or
# ``?experiment_id=``), one campaign (``?campaign=``), or every experiment
# (neither).  ``campaign_middleware`` already refuses an experiment id from
# another campaign, so the scope needs no checks of its own.

_MAX_FINISHED_EXPORTS = 20
# How often an in-flight upload checks whether the export was cancelled.
_EXPORT_CANCEL_POLL_SECS = 0.25


class _NoMetrics(Exception):
    """No run in the export scope stored any metrics."""


def _path_part(s: str) -> str:
    """*s* made safe to use as one path component inside a zip."""
    return re.sub(r"[/\\\x00]", "_", s).lstrip(".") or "_"


def _unique_path_parts(items: list[tuple[str, str]]) -> dict[str, str]:
    """Sanitized, unique path components for ``(key, label)`` pairs.

    Distinct labels can sanitize to the same component (``a/b`` and ``a_b``),
    which would merge their folders and silently drop runs.  Colliding later
    ones get a short hash of their key.  Sorted by key, so names are stable.
    """
    used: set[str] = set()
    out: dict[str, str] = {}
    for key, label in sorted(items):
        base = _path_part(label)
        candidate = base
        if candidate in used:
            suffix = hashlib.blake2b(key.encode(), digest_size=4).hexdigest()
            candidate = f"{base}_{suffix}"
            n = 2
            while candidate in used:
                candidate = f"{base}_{suffix}_{n}"
                n += 1
        used.add(candidate)
        out[key] = candidate
    return out


def _job_combo(job: Any) -> dict[str, Any]:
    try:
        combo = json.loads(job.combo or "{}")
    except json.JSONDecodeError:
        return {}
    return combo if isinstance(combo, dict) else {}


async def _export_scope(request: web.Request) -> ExportScope | web.Response:
    """The request's scope, or a 404."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    experiment_id = request.match_info.get("experiment_id") or request.query.get("experiment_id")
    campaign = request.query.get("campaign")
    if experiment_id:
        exp = await get_experiment(db, experiment_id)
        if exp is None or (campaign is not None and exp.campaign != campaign):
            return _not_found("experiment")
        exps = [(exp.experiment_id, exp.name, exp.campaign)]
        kind = "experiment"
    else:
        exps = [(e["experiment_id"], e["name"], e["campaign"])
                for e in await list_experiments_with_counts(db, campaign=campaign)]
        kind = "campaign" if campaign else "all"
    labels = experiment_labels([(i, name or i) for i, name, _ in exps])
    return ExportScope(kind=kind, campaign=campaign, experiment_id=experiment_id,
                       experiments=labels, campaign_of={i: c for i, _, c in exps})


async def _scope_jobs(request: web.Request, scope: ExportScope, only_done: bool) -> list[Any]:
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    jobs: list[Any] = []
    for experiment_id in scope.experiments:
        jobs += await list_jobs_by_experiment(db, experiment_id, "done" if only_done else None)
    return jobs


def _only_done(request: web.Request, body: Any = None) -> bool:
    """``?only_done=1`` or ``"only_done": true`` in the JSON body keeps succeeded runs."""
    return (request.query.get("only_done") in ("1", "true")
            or bool(isinstance(body, dict) and body.get("only_done")))


@routes.get("/api/experiments/{experiment_id}/tensorboard.zip")
@routes.get("/api/export/tensorboard.zip")
async def handle_export_tensorboard(request: web.Request) -> web.StreamResponse:
    """Serve metrics as TensorBoard event files, one directory per run.

    Runs sit under ``EXPERIMENT/RUN/``, with a ``CAMPAIGN/`` level on top when
    the export spans campaigns.  Each run's combo goes to the HParams dashboard.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    scope = await _export_scope(request)
    if isinstance(scope, web.Response):
        return scope
    jobs = await _scope_jobs(request, scope, _only_done(request))
    if not jobs:
        return _error_response("no jobs", status=404)
    loop = asyncio.get_running_loop()
    campaign_parts = (_unique_path_parts([(c, c) for c in scope.campaigns])
                      if scope.kind == "all" else {})
    experiment_parts = _unique_path_parts(list(scope.experiments.items()))
    runs_by_experiment: dict[str, list[str]] = {}
    for job in jobs:
        runs_by_experiment.setdefault(job.experiment_id, []).append(job.run_id)
    run_parts = {eid: _unique_path_parts([(r, r) for r in rids])
                 for eid, rids in runs_by_experiment.items()}

    def folder(job: Any) -> str:
        parts = [experiment_parts[job.experiment_id], run_parts[job.experiment_id][job.run_id]]
        if scope.kind == "all":
            parts.insert(0, campaign_parts[scope.campaign_of[job.experiment_id]])
        return "/".join(parts)

    async def fill(tmp_path: str) -> None:
        wrote = False
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for job in jobs:
                rows = await get_metrics_for_run(db, job.run_id, job.experiment_id)
                if not rows:
                    continue
                wrote = True
                start = (job.start_time or job.submit_time).timestamp()
                end = job.finish_time.timestamp() if job.finish_time else None
                path = f"{folder(job)}/{tfevents_name(start)}"
                # Encoding and deflating are CPU-bound, so both stay off the loop.
                await loop.run_in_executor(None, lambda: zf.writestr(path, tfevents_for_run(
                    rows, hparams=_job_combo(job), status=job.status,
                    start_time=start, end_time=end,
                )))
        if not wrote:
            raise _NoMetrics

    name = {"experiment": (scope.experiment_id or "")[:16], "campaign": scope.campaign or ""}
    try:
        return await _serve_temp_zip(fill, f"{name.get(scope.kind, 'mlsweep')}-tensorboard.zip")
    except _NoMetrics:
        return _error_response("no metrics", status=404)


@routes.get("/api/export/wandb")
async def handle_wandb_info(request: web.Request) -> web.Response:
    """Whether the manager can upload to W&B (wandb installed, key in env)."""
    return _json_response(request.config_dict[WANDB_KEY].info())


def _exports(request: web.Request) -> dict[str, WandbExport]:
    return request.config_dict[EXPORTS_KEY]


def _forget_old_exports(exports: dict[str, WandbExport]) -> None:
    finished = [e for e in exports.values() if e.status != "running"]
    finished.sort(key=lambda e: e.finished or 0.0)
    for e in finished[:-_MAX_FINISHED_EXPORTS]:
        del exports[e.export_id]


async def _cleanup_abandoned(futures: list[asyncio.Future[Any]], path: str) -> None:
    """Remove *path* after uploads that outlived a cancelled export finish."""
    await asyncio.gather(*futures, return_exceptions=True)
    shutil.rmtree(path, ignore_errors=True)


def _consume_future(fut: asyncio.Future[Any]) -> None:
    """Mark an abandoned future's result as retrieved, so asyncio does not warn."""
    if not fut.cancelled():
        fut.exception()


def _finish_pending(export: WandbExport) -> None:
    """Give every run a terminal status, even if the export stopped early."""
    for run in export.runs:
        if run.status == "pending":
            run.status = "skipped"
            run.error = "cancelled" if export.cancel.is_set() else "not attempted"


async def _run_wandb_export(
    db: aiosqlite.Connection,
    export: WandbExport,
    jobs: list[Any],
    api_key: str | None,
    background: set[asyncio.Task[Any]],
    source: WandbSource,
) -> None:
    """Upload *jobs* one at a time, recording progress on *export*."""
    loop = asyncio.get_running_loop()
    work_dir = tempfile.mkdtemp(prefix="mlsweep_wandb_")
    abandoned: list[asyncio.Future[Any]] = []
    try:
        wandb = await loop.run_in_executor(None, source.load)
        mode = wandb_mode(os.environ)
        for job, run in zip(jobs, export.runs):
            if export.cancel.is_set():
                break
            rows = await get_metrics_for_run(db, job.run_id, job.experiment_id)
            if not rows:
                run.status = "skipped"
                run.error = "no metrics"
                continue
            run.status = "uploading"
            summary = {"mlsweep/status": job.status, "mlsweep/elapsed": job.elapsed,
                       "mlsweep/exit_code": job.exit_code, "mlsweep/label": job.label,
                       "mlsweep/experiment_id": job.experiment_id}
            fut = loop.run_in_executor(None, lambda: upload_wandb_run(
                wandb, export=export, run_id=job.run_id, rows=rows,
                config=_job_combo(job), summary=summary, group=run.experiment,
                tags=[export.scope.campaign_of[job.experiment_id]], api_key=api_key,
                work_dir=work_dir, mode=mode,
            ))
            try:
                while True:
                    done, _ = await asyncio.wait({fut}, timeout=_EXPORT_CANCEL_POLL_SECS)
                    if done:
                        run.url = fut.result()
                        break
                    if export.cancel.is_set():
                        # wandb.init may be blocked on the network.  Stop
                        # waiting; the thread sees the cancel flag when init
                        # returns and finishes its run as failed.  Keep its
                        # work dir until then.
                        fut.add_done_callback(_consume_future)
                        abandoned.append(fut)
                        break
                if export.cancel.is_set():
                    run.status, run.error = "failed", "cancelled"
                else:
                    run.status = "done"
            except Exception as exc:
                logger.warning("W&B export of %s/%s failed: %s", job.experiment_id, job.run_id, exc)
                run.status = "failed"
                run.error = str(exc) or type(exc).__name__
        if export.cancel.is_set():
            export.status = "cancelled"
        elif any(r.status == "failed" for r in export.runs):
            export.status = "failed"
            export.error = "some runs failed to upload"
        else:
            export.status = "done"
    except Exception as exc:
        logger.exception("W&B export %s failed", export.export_id)
        export.status = "failed"
        export.error = str(exc) or type(exc).__name__
    finally:
        _finish_pending(export)
        export.finished = time.time()
        if abandoned:
            cleanup = asyncio.create_task(_cleanup_abandoned(abandoned, work_dir))
            background.add(cleanup)
            cleanup.add_done_callback(background.discard)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


@routes.post("/api/experiments/{experiment_id}/export/wandb")
@routes.post("/api/export/wandb")
async def handle_export_wandb(request: web.Request) -> web.Response:
    """Start uploading the scope's runs to W&B from the manager.

    Body: ``{project, entity?, api_key?, only_done?}``.  Without ``api_key``,
    wandb uses ``WANDB_API_KEY`` or the login stored for the manager's user.
    Each run's group is its experiment and its tag is its campaign.  Returns
    202 with the export; poll ``/api/exports/{export_id}`` for progress.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    body = await _json_body(request)
    if not isinstance(body, dict):
        return _error_response("body must be a JSON object")
    project = str(body.get("project") or "").strip()
    if not project:
        return _error_response("project is required")
    entity = str(body.get("entity") or "").strip() or None
    api_key = str(body.get("api_key") or "").strip() or None
    scope = await _export_scope(request)
    if isinstance(scope, web.Response):
        return scope
    source = request.config_dict[WANDB_KEY]
    if not source.info()["available"]:
        return _error_response(
            "wandb is not installed in the manager's environment "
            "(pip install 'mlsweep[wandb]' there and restart the manager)",
            status=501,
        )
    jobs = await _scope_jobs(request, scope, _only_done(request, body))
    if not jobs:
        return _error_response("no jobs to export", status=404)

    exports = _exports(request)
    for other in exports.values():
        if other.status == "running" and other.scope.experiments.keys() & scope.experiments.keys():
            return _error_response(
                f"an export of {other.scope.label} is already running and overlaps this one",
                status=409)
    export = WandbExport(scope=scope, project=project, entity=entity, runs=[
        ExportRun(j.experiment_id, j.run_id, scope.experiments[j.experiment_id]) for j in jobs])
    _forget_old_exports(exports)
    exports[export.export_id] = export
    tasks = request.config_dict[EXPORT_TASKS_KEY]
    task = asyncio.create_task(_run_wandb_export(db, export, jobs, api_key, tasks, source))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return _json_response(export.to_dict(), status=202)


@routes.get("/api/exports")
async def handle_list_exports(request: web.Request) -> web.Response:
    """Recent W&B exports, newest first.

    ``?experiment_id=`` keeps the ones covering that experiment, and
    ``?campaign=`` the ones whose experiments are all in that campaign.
    """
    experiment_id = request.query.get("experiment_id")
    campaign = request.query.get("campaign")
    out = [e.to_dict() for e in _exports(request).values()
           if (experiment_id is None or experiment_id in e.scope.experiments)
           and (campaign is None or e.scope.campaigns <= {campaign})]
    out.sort(key=lambda d: d["started"], reverse=True)
    return _json_response(out)


@routes.get("/api/exports/{export_id}")
async def handle_get_export(request: web.Request) -> web.Response:
    export = _exports(request).get(request.match_info["export_id"])
    if export is None:
        return _not_found("export")
    return _json_response(export.to_dict())


@routes.post("/api/exports/{export_id}/cancel")
async def handle_cancel_export(request: web.Request) -> web.Response:
    """Stop an export.  The run being uploaded is finished early and marked failed in W&B."""
    export = _exports(request).get(request.match_info["export_id"])
    if export is None:
        return _not_found("export")
    export.cancel.set()
    return _json_response(export.to_dict())


# ── Workers ────────────────────────────────────────────────────────────────────


def _json_list(s: str | None) -> list[Any]:
    """Decode a JSON-list DB column, treating NULL or garbage as empty."""
    try:
        return json.loads(s) if s else []
    except (json.JSONDecodeError, TypeError):
        return []


def _enrich_worker(wr: WorkerRecord, state: ManagerState) -> dict[str, Any]:
    """Merge a DB WorkerRecord with live WorkerConn data.

    Returns a dict suitable for JSON serialisation, containing all DB
    fields plus ``gpus`` (list[int]) and ``gpu_occupancy`` (dict[int,int])
    from the live connection when available.
    """
    d: dict[str, Any] = dataclasses.asdict(wr)

    wc = state.workers.get(wr.worker_id)
    if wc is not None:
        # "gpus" is what the scheduler can use now; pool GPUs held back by the
        # health probe are listed under unhealthy_gpus / pending_gpus instead.
        d["gpus"] = wc.usable_gpus
        d["unhealthy_gpus"] = wc.unhealthy_gpus
        d["unhealthy_reasons"] = wc.unhealthy_reasons
        d["pending_gpus"] = wc.pending_gpus
        d["gpu_occupancy"] = state.occupancy(wc)
        d["gpu_stats"] = wc.gpu_stats
        d["max_jobs_per_gpu"] = wc.max_jobs_per_gpu
    else:
        # devices holds every GPU the worker considered; the probe-excluded
        # ones are reported separately, as for a live worker.
        unhealthy = _json_list(d["unhealthy_devices"])
        d["gpus"] = [g for g in _json_list(d["devices"]) if g not in unhealthy]
        d["unhealthy_gpus"] = unhealthy
        d["unhealthy_reasons"] = {}
        d["pending_gpus"] = []
        d["gpu_occupancy"] = {}
        d["gpu_stats"] = {}
        d["max_jobs_per_gpu"] = 1
    # Every GPU's runs, including GPUs out of the pool whose runs are finishing,
    # and the jobs pinned to start here next.
    d["gpu_runs"] = {
        str(g): [{"run_id": r.run_id, "experiment_id": r.experiment_id} for r in runs]
        for g, runs in state.runs_by_gpu(wr.worker_id).items()
    }
    d["pins"] = [
        {"experiment_id": key[0], "run_id": key[1], "gpus": pin.gpus}
        for key, pin in state.pins.items() if pin.worker_id == wr.worker_id
    ]
    d["outdated"] = worker_outdated(state, wr.worker_id)
    return d


@routes.get("/api/workers")
async def handle_list_workers(request: web.Request) -> web.Response:
    """List all workers, optionally filtered by status.

    Returns DB records enriched with live GPU occupancy from connected
    workers.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    status_filter = request.query.get("status")
    workers = await list_workers(db, status=status_filter)  # type: ignore[arg-type]
    enriched = [_enrich_worker(wr, state) for wr in workers]
    return _json_response(enriched)


@routes.get("/api/workers/{worker_id}")
async def handle_get_worker(request: web.Request) -> web.Response:
    """Get a single worker by ID, enriched with live GPU occupancy."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    worker_id = request.match_info["worker_id"]
    worker = await get_worker(db, worker_id)
    if worker is None:
        return _not_found("worker")
    return _json_response(_enrich_worker(worker, state))


@routes.post("/api/workers")
async def handle_add_worker(request: web.Request) -> web.Response:
    """Add a new worker dynamically.

    Accepts JSON body: {host (required), remote_dir?, ssh_key?, venv?,
    port?, devices?, scratch_dir?}.  Generates a worker_id from the host, upserts into the DB,
    and spawns a background task to connect to the worker.

    ``remote_dir`` is optional: the training code is shipped as an artifact and
    the worker falls back to its own working directory when it is empty.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]

    body = await _json_body(request)

    host = body.get("host")
    remote_dir = body.get("remote_dir") or ""
    if not host:
        return _error_response("'host' is required")
    try:
        validate_worker_host(host)
    except ValueError as exc:
        return _error_response(str(exc))

    ssh_key = body.get("ssh_key")
    venv = body.get("venv")
    port = body.get("port", 0)
    devices = body.get("devices")
    scratch_dir = body.get("scratch_dir") or state.scratch_dir
    for field, value in (("remote_dir", remote_dir), ("ssh_key", ssh_key), ("venv", venv),
                         ("scratch_dir", scratch_dir), ("worker_id", body.get("worker_id"))):
        if value is not None and not isinstance(value, str):
            return _error_response(f"'{field}' must be a string")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        return _error_response("'port' must be an integer from 0 to 65535")
    if devices is not None and not (isinstance(devices, list) and all(
            isinstance(d, int) and not isinstance(d, bool) for d in devices)):
        return _error_response("'devices' must be a list of GPU indices")

    # Use explicit worker_id (reconnect) or derive from host/port (new worker),
    # and claim it so a concurrent add of the same worker is refused.
    async with state.lock:
        worker_id = body.get("worker_id")
        if not worker_id and port:
            worker_id = worker_id_for(host, port, 0)
        elif not worker_id:  # a new ephemeral worker takes the first free index
            taken = set(state.workers) | state.launching
            worker_id = next(w for i in itertools.count() if (w := worker_id_for(host, 0, i)) not in taken)
        if not state.reserve_worker_id(worker_id):
            return _error_response(f"worker {worker_id} is already connected", status=409)

    # Upsert into DB
    try:
        worker = await state.db_writer.upsert_worker(
            worker_id=worker_id,
            host=host,
            remote_dir=remote_dir,
            ssh_key=ssh_key,
            venv=venv,
            port=port,
            devices=json.dumps(devices) if devices else None,
            status="offline",
        )
    except Exception:
        state.launching.discard(worker_id)
        raise

    # Connect in the background; a launch failure marks the worker dead with the reason.
    asyncio.create_task(connect_single_worker(
        db, state,
        host=host,
        remote_dir=remote_dir,
        worker_id=worker_id,
        scratch_dir=scratch_dir,
        ssh_key=ssh_key,
        venv=venv,
        port=port,
        devices=devices,
        manager_port=state.manager_port,
    ))

    return _json_response(worker, status=201)


@routes.delete("/api/workers/{worker_id}")
async def handle_delete_worker(request: web.Request) -> web.Response:
    """Remove a worker dynamically.

    Marks the worker as dead in the DB, sends ``MsgShutdown`` if connected,
    and re-queues any jobs assigned to that worker.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    worker_id = request.match_info["worker_id"]

    worker = await get_worker(db, worker_id)
    if worker is None:
        return _not_found("worker")

    # Cancels its runs (requeued without spending retries), tells the worker
    # to exit, and marks it dead.
    await declare_worker_dead(db, state, worker_id, "", shutdown=True)

    return _json_response({"worker_id": worker_id, "status": "dead"})


@routes.post("/api/workers/{worker_id}/update")
async def handle_update_worker(request: web.Request) -> web.Response:
    """Relaunch a worker so it runs this manager's mlsweep code.

    Meant for a worker that runs older mlsweep code than the manager; the
    relaunch reinstalls the host's venv when its code differs.  A live
    worker is shut down first and its runs are requeued without spending
    retries.  The update runs in the background and the worker's status shows
    ``updating`` until it reconnects or fails with a reason.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    worker_id = request.match_info["worker_id"]
    if await get_worker(db, worker_id) is None:
        return _not_found("worker")
    async with state.lock:
        wc = state.workers.get(worker_id)
        if worker_id in state.launching:
            return _error_response(f"worker {worker_id} is already launching", status=409)
        live = wc is not None and wc.status != "dead"
    if live:
        await declare_worker_dead(db, state, worker_id, "", shutdown=True)
    async with state.lock:
        if not state.reserve_worker_id(worker_id):
            return _error_response(f"worker {worker_id} is already launching", status=409)
    # Before the task starts, so the page shows the update right away.
    await state.db_writer.update_worker_status(worker_id, "updating")
    asyncio.create_task(update_worker(db, state, worker_id))
    return _json_response({"worker_id": worker_id, "status": "updating"}, status=202)


@routes.patch("/api/workers/{worker_id}/concurrency")
async def handle_patch_worker_concurrency(request: web.Request) -> web.Response:
    """Set max_jobs_per_gpu for a live worker.

    Accepts JSON: {"max_jobs_per_gpu": int}.  0 means unlimited.
    Takes effect immediately for the next scheduling pass.
    """
    state: ManagerState = request.config_dict[STATE_KEY]
    worker_id = request.match_info["worker_id"]

    body = await _json_body(request)

    value = body.get("max_jobs_per_gpu")
    if value is None or not isinstance(value, int) or value < 0:
        return _error_response("'max_jobs_per_gpu' (int >= 0) is required")

    wc = state.workers.get(worker_id)
    if wc is None:
        return _not_found("worker")

    async with state.lock:
        wc.max_jobs_per_gpu = value

    state.request_schedule()
    return _json_response({"worker_id": worker_id, "max_jobs_per_gpu": value})


@routes.patch("/api/workers/{worker_id}/devices")
async def handle_patch_worker_devices(request: web.Request) -> web.Response:
    """Add or remove GPU device indices from a live worker.

    Accepts JSON: {"add": [int, ...], "remove": [int, ...], "drain": [int, ...]}.
    Removing a GPU evicts any jobs currently using it (they are re-queued).
    Draining one takes it out of the pool too, but its runs keep going until
    they finish; nothing new is placed on it.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    worker_id = request.match_info["worker_id"]

    body = await _json_body(request)

    add_gpus: list[int] = body.get("add") or []
    remove_gpus: list[int] = body.get("remove") or []
    drain_gpus: list[int] = body.get("drain") or []

    def int_list(v: Any) -> bool:
        return isinstance(v, list) and all(isinstance(g, int) and not isinstance(g, bool) and g >= 0 for g in v)

    if not (int_list(add_gpus) and int_list(remove_gpus) and int_list(drain_gpus)):
        return _error_response("'add', 'remove' and 'drain' must be lists of GPU indices")

    wc = state.workers.get(worker_id)
    if wc is None:
        return _not_found("worker")

    async with state.lock:
        # Runs on a GPU being removed are requeued (no retry spent).
        remove_set = set(remove_gpus)
        evict = [r for r in state.runs_on(worker_id) if remove_set & set(r.nodes[worker_id])]
        wc.gpus = sorted((set(wc.gpus) - remove_set - set(drain_gpus)) | set(add_gpus))
        # A pin to a GPU that just left the pool is dropped by the next pass.
        await requeue_runs_locked(db, state, [r.key for r in evict], lost=False)
        await state.db_writer.update_worker_devices(worker_id, json.dumps(wc.gpus))
    to_evict = [r.run_id for r in evict]
    state.request_schedule()
    draining = sorted({r.run_id for r in state.runs_on(worker_id)
                       if set(drain_gpus) & set(r.nodes[worker_id])})
    return _json_response({"worker_id": worker_id, "gpus": wc.gpus, "evicted": to_evict,
                           "draining": draining})


# ── Scheduler ──────────────────────────────────────────────────────────────────


@routes.get("/api/scheduler")
async def handle_get_scheduler(request: web.Request) -> web.Response:
    """Return the scheduler settings: ``{"placement": "pack" | "spread"}``."""
    state: ManagerState = request.config_dict[STATE_KEY]
    return _json_response({"placement": state.placement})


@routes.patch("/api/scheduler")
async def handle_patch_scheduler(request: web.Request) -> web.Response:
    """Change the GPU placement policy.

    Accepts JSON: {"placement": "pack" | "spread"}.  Persisted across manager
    restarts.  Applies to jobs dispatched from the next scheduling pass on;
    running jobs stay where they are.
    """
    state: ManagerState = request.config_dict[STATE_KEY]
    body = await _json_body(request)
    placement = body.get("placement")
    if placement not in PLACEMENTS:
        return _error_response(f"'placement' must be one of {list(PLACEMENTS)}")
    await state.db_writer.set_setting("placement", placement)
    async with state.lock:
        state.placement = placement
    state.request_schedule()
    return _json_response({"placement": placement})


# ── Artifacts ──────────────────────────────────────────────────────────────────


@routes.get("/api/artifacts/{artifact_id}/meta")
async def handle_get_artifact_meta(request: web.Request) -> web.Response:
    """Get artifact metadata by ID."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    artifact_id = request.match_info["artifact_id"]
    artifact = await get_artifact(db, artifact_id)
    if artifact is None:
        return _not_found("artifact")
    return _json_response(artifact)


@routes.get("/api/artifacts/{artifact_id}", allow_head=False)
async def handle_download_artifact(request: web.Request) -> web.StreamResponse:
    """Download artifact tarball bytes.

    Returns the raw ``.tar.gz`` file stored on disk.  Returns 404 if the
    artifact file does not exist (the artifact may be registered in the DB
    but its data not yet uploaded).
    """
    artifact_id = request.match_info["artifact_id"]

    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    artifacts_dir = mlsweep_dir / "artifacts"
    tarball = artifacts_dir / f"{artifact_id}.tar.gz"

    if not tarball.is_file():
        return _not_found("artifact")

    return web.FileResponse(tarball)


@routes.head("/api/artifacts/{artifact_id}")
async def handle_head_artifact(request: web.Request) -> web.StreamResponse:
    """Check artifact existence (HEAD).

    Returns 200 if the artifact is registered in the DB *and* its tarball
    file exists on disk; 404 otherwise.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    artifact_id = request.match_info["artifact_id"]
    artifact = await get_artifact(db, artifact_id)
    if artifact is None:
        return _not_found("artifact")

    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    artifacts_dir = mlsweep_dir / "artifacts"
    tarball = artifacts_dir / f"{artifact_id}.tar.gz"

    if not tarball.is_file():
        return _not_found("artifact")

    return web.Response(status=200)


@routes.post("/api/artifacts")
async def handle_register_artifact(request: web.Request) -> web.Response:
    """Register or update an artifact."""
    state: ManagerState = request.config_dict[STATE_KEY]
    body = await _json_body(request)

    artifact_id = body.get("artifact_id") or body.get("id")
    if not artifact_id:
        return _error_response("'artifact_id' is required")
    try:
        validate_artifact_id(artifact_id)
    except ValueError as exc:
        return _error_response(str(exc))
    size_bytes = body.get("size_bytes")
    if size_bytes is not None and (not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0):
        return _error_response("'size_bytes' must be a non-negative integer")

    artifact = await state.db_writer.register_artifact(
        artifact_id=artifact_id,
        size_bytes=size_bytes,
        setup_command=body.get("setup_command"),
    )

    return _json_response(artifact, status=201)


@routes.put("/api/artifacts/{artifact_id}/ref")
async def handle_increment_artifact_ref(request: web.Request) -> web.Response:
    """Increment or decrement an artifact's reference count."""
    state: ManagerState = request.config_dict[STATE_KEY]
    artifact_id = request.match_info["artifact_id"]
    body = await _json_body(request)

    delta = body.get("delta", 1)
    if not isinstance(delta, int) or isinstance(delta, bool):
        return _error_response("'delta' must be an integer")

    artifact = await state.db_writer.increment_artifact_ref(artifact_id, delta=delta)
    if artifact is None:
        return _not_found("artifact")
    return _json_response(artifact)


@routes.put("/api/artifacts/{artifact_id}/data")
async def handle_upload_artifact_data(request: web.Request) -> web.Response:
    """Upload artifact binary data (tar.gz).

    Accepts raw binary body.  Saves to
    ``<mlsweep_dir>/artifacts/<artifact_id>.tar.gz``.
    """
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    artifact_id = request.match_info["artifact_id"]

    # Verify artifact exists in DB
    from mlsweep._manager_db import get_artifact
    artifact = await get_artifact(db, artifact_id)
    if artifact is None:
        return _not_found("artifact")

    # Determine storage path
    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    artifacts_dir = mlsweep_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Stream to a temporary file, then rename for an atomic write.
    dest = artifacts_dir / f"{artifact_id}.tar.gz"
    fd, tmp_name = tempfile.mkstemp(dir=artifacts_dir, suffix=".tmp")
    tmp = Path(tmp_name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.content.iter_chunked(1 << 20):
                size += len(chunk)
                if size > _MAX_ARTIFACT_BYTES:
                    return _error_response(
                        f"artifact is larger than {_MAX_ARTIFACT_BYTES // (1024 * 1024)} MB", status=413)
                f.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("Artifact %s stored (%d bytes)", artifact_id, size)

    return _json_response(
        {"artifact_id": artifact_id, "size_bytes": size},
        status=200,
    )


# ===============================================================================
# WebSocket event stream
# ===============================================================================


@routes.get("/ws/experiments/{experiment_id}")
async def handle_ws_experiment(request: web.Request) -> web.StreamResponse:
    """WebSocket event stream for an experiment.

    Clients receive real-time events: job status changes, log messages,
    metrics, and experiment status updates.

    Query params:
      - ``?since=<unix_timestamp>``: replay completed/failed jobs whose
        ``finish_time >= since`` as synthetic ``job_done`` events, and
        started jobs whose ``start_time >= since`` as ``job_started`` events,
        before the live stream begins.
    """
    experiment_id = request.match_info["experiment_id"]
    state: ManagerState = request.config_dict[STATE_KEY]
    db: aiosqlite.Connection = request.config_dict[DB_KEY]

    ws = web.WebSocketResponse(max_msg_size=_MAX_WS_MESSAGE_BYTES)
    await ws.prepare(request)

    # ── ?since= replay ────────────────────────────────────────────────────
    since_str = request.query.get("since")
    if since_str is not None:
        try:
            since_epoch = float(since_str)
        except (ValueError, TypeError):
            await ws.send_json({"error": "invalid 'since' parameter — must be a Unix timestamp"})
            await ws.close()
            return ws

        # Replay job_done events for done/failed jobs finished since `since`
        done_jobs = await list_jobs_since(
            db, experiment_id,
            statuses=['done', 'failed'],
            since_col='finish_time',
            since_ts=since_epoch,
        )
        for job in done_jobs:
            await ws.send_json({
                "type": "job_done",
                "experiment_id": experiment_id,
                "run_id": job.run_id,
                "status": job.status,
                "success": job.status == "done",
                "elapsed": job.elapsed,
                "exit_code": job.exit_code,
                "worker_id": job.worker_id,
            }, dumps=_json_dumps)

        # Replay job_started events for jobs started since `since`
        started_jobs = await list_jobs_since(
            db, experiment_id,
            statuses=['dispatched', 'running', 'done', 'failed', 'cancelled'],
            since_col='start_time',
            since_ts=since_epoch,
        )
        for job in started_jobs:
            await ws.send_json({
                "type": "job_started",
                "experiment_id": experiment_id,
                "run_id": job.run_id,
                "worker_id": job.worker_id,
            }, dumps=_json_dumps)

    # ── Live subscriber loop ──────────────────────────────────────────────

    # Create a bounded queue so we can detect slow/disconnected clients
    # in broadcast() via QueueFull.
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
    state.add_subscriber(experiment_id, queue)

    logger.debug("WebSocket subscriber joined experiment %s", experiment_id)

    # Background task: forward broadcast events from the queue to the
    # WebSocket client.
    async def _forward_events() -> None:
        try:
            while True:
                event = await queue.get()
                try:
                    await ws.send_json(event, dumps=_json_dumps)
                except Exception:
                    # Connection closed or broken
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("WebSocket forward task exited", exc_info=True)

    forward_task = asyncio.create_task(_forward_events())

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                # Clients can send JSON commands (e.g., ping, filter)
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"error": "invalid JSON"})
                    continue

                cmd = data.get("type", "")
                if cmd == "ping":
                    await ws.send_json({"type": "pong"})
                # Future: subscribe/unsubscribe, log-level filters, etc.

            elif msg.type == WSMsgType.ERROR:
                logger.warning("WebSocket error for experiment %s: %s",
                               experiment_id, ws.exception())
    finally:
        forward_task.cancel()
        try:
            await forward_task
        except asyncio.CancelledError:
            pass
        state.remove_subscriber(experiment_id, queue)
        logger.debug("WebSocket subscriber left experiment %s", experiment_id)

    return ws


# ===============================================================================
# Health check
# ===============================================================================


@routes.get("/api/health")
async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    db: aiosqlite.Connection = request.config_dict[DB_KEY]
    state: ManagerState = request.config_dict[STATE_KEY]
    mlsweep_dir: Path = request.config_dict[DIR_KEY]
    return _json_response({
        "status": "ok",
        "version": _VERSION,
        "workers_connected": sum(wc.status == "connected" for wc in state.workers.values()),
        "jobs_pending": await count_pending_jobs(db),
        "jobs_in_flight": len(state.runs),
        "public_url": state.public_url or None,
        # Lets a client on this machine find the manager's folder (see mlsweep fetch).
        "mlsweep_dir": str(mlsweep_dir),
        "mlsweep_dir_id": mlsweep_dir_id(mlsweep_dir, create=True),
    })


# ===============================================================================
# Static files
# ===============================================================================


@routes.get("/")
async def handle_index(request: web.Request) -> web.Response:
    """Redirect to the web UI index page."""
    raise web.HTTPFound("/static/experiments.html" + ("?" + request.query_string if request.query_string else ""))


def _setup_static_routes(app: web.Application, webui_dir: Path) -> None:
    """Add static file serving routes for the web UI.

    Searches for the web UI in order:
    1. ``<mlsweep_dir>/webui/`` (runtime data directory)
     2. ``<mlsweep_package>/webui/`` (development / installed package)
    """
    if webui_dir.exists():
        app.router.add_static("/static/", path=str(webui_dir))
        return

    # Fallback: look for webui/ directory relative to the mlsweep package
    import mlsweep
    pkg_dir = Path(mlsweep.__file__).resolve().parent
    pkg_web_dir = pkg_dir / "webui"
    if pkg_web_dir.exists():
        logger.info("Serving web UI from package directory: %s", pkg_web_dir)
        app.router.add_static("/static/", path=str(pkg_web_dir))
        return

    logger.info("Web UI directory not found — static routes skipped")


def _broadcast_experiment_event(
    request: web.Request,
    experiment_id: str,
    event_type: str,
    **kwargs: Any,
) -> None:
    """Broadcast an event to all WebSocket subscribers of *experiment_id*."""
    state: ManagerState = request.config_dict[STATE_KEY]
    event = {"type": event_type, "experiment_id": experiment_id, **kwargs}
    state.broadcast(experiment_id, event)


# ===============================================================================
# Application factory
# ===============================================================================


def create_app(
    db: aiosqlite.Connection,
    state: ManagerState,
    token: str,
    *,
    mlsweep_dir: str | Path = "~/.mlsweep",
    wandb: WandbSource | None = None,
) -> web.Application:
    """Create and return an aiohttp ``Application``.

    Parameters
    ----------
    db:
        SQLite database connection.
    state:
        In-memory manager state (pending list, in-flight tracking, subscribers).
    token:
        Authentication token; clients must provide it as ``?token=`` or
        ``Authorization: Bearer``.
    mlsweep_dir:
        Root directory of mlsweep data.  Static web UI is served from
        ``<mlsweep_dir>/webui/`` if it exists.
    wandb:
        Where W&B exports get wandb; by default the installed package.
    """
    app = web.Application(
        middlewares=[audit_middleware, error_middleware, auth_middleware, id_middleware, campaign_middleware],
        client_max_size=_MAX_JSON_BODY_BYTES,
    )

    # Store shared objects in app config so handlers can access them
    app[DB_KEY] = db
    app[STATE_KEY] = state
    app[TOKEN_KEY] = token
    app[DIR_KEY] = Path(mlsweep_dir).expanduser().resolve()
    app[EXPORTS_KEY] = {}
    app[EXPORT_TASKS_KEY] = set()
    app[WANDB_KEY] = wandb or WandbSource()
    app[THROTTLE_KEY] = _AuthThrottle()

    # Register REST routes
    from mlsweep._manager_code import code_routes
    app.add_routes(routes)
    app.add_routes(code_routes)

    # Static file serving for web UI
    webui_dir = Path(mlsweep_dir).expanduser().resolve() / "webui"
    _setup_static_routes(app, webui_dir)

    app.on_response_prepare.append(_add_security_headers)

    return app


# ===============================================================================
# Convenience: run the app
# ===============================================================================


def run_app(
    app: web.Application,
    *,
    host: str = "0.0.0.0",
    port: int = 7891,
    **kwargs: Any,
) -> None:
    """Run the aiohttp application (blocking convenience wrapper)."""
    web.run_app(app, host=host, port=port, **kwargs)


# ===============================================================================
# Test helpers (for verification only)
# ===============================================================================

__all__ = [
    "create_app",
    "setup_audit_log",
    "run_app",
    "routes",
]
