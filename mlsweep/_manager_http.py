"""HTTP REST API and WebSocket event stream for mlsweep manager.

Provides:
  - REST endpoints for campaigns, experiments, jobs, workers, artifacts
  - WebSocket event stream at ``/ws/experiments/{id}``
  - Static file serving for a web dashboard

All endpoints accept authentication via ``?token=...`` query parameter or
``Authorization: Bearer <token>`` header.  The token is the manager token
stored in ``manager.token``.

Any endpoint also accepts ``?campaign=NAME``.  Listings return only that
campaign's experiments and jobs, a new experiment is created in it, and a
request naming an experiment from another campaign is refused with 404 (see
``campaign_middleware``).  Without the parameter nothing is restricted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
import itertools
import json
import logging
import os
import re
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

import aiosqlite
from aiohttp import WSMsgType, web

from mlsweep._manager_db import (
    ACTIVE_JOB_STATUSES,
    FINISHED_JOB_STATUSES,
    WorkerRecord,
    count_pending_jobs,
    experiment_summary,
    get_artifact,
    get_experiment,
    get_job,
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
from mlsweep._manager_state import InFlightRun, ManagerState
from mlsweep._manager_workers import (
    _check_experiments_complete_locked,
    _detach_locked,
    cancel_runs_locked,
    connect_single_worker,
    declare_worker_dead,
    requeue_runs_locked,
    worker_id_for,
)
from mlsweep._shared import DEFAULT_CAMPAIGN, _resolve_safe_subpath, validate_campaign

logger = logging.getLogger(__name__)

_VERSION = importlib.metadata.version("mlsweep")

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
        await fill(tmp_path)
    except Exception:
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
    if not _EXPERIMENT_ID_RE.match(experiment_id):
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
    if request.query.get("token") == token:
        return True
    # Authorization header
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == token:
        return True
    return False


@web.middleware
async def auth_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Middleware that enforces token authentication on all routes.

    Skips static file routes (prefix ``/static/``) so the web UI can load
    without a token in every asset request.
    """
    # Allow static files without auth
    if request.path.startswith("/static/"):
        return await handler(request)

    # Allow OPTIONS (CORS preflight) without auth
    if request.method == "OPTIONS":
        return await handler(request)

    token: str = request.config_dict["mlsweep_token"]
    if not _check_auth(request, token):
        return _error_response("unauthorized — provide ?token= or Authorization: Bearer", status=401)

    return await handler(request)


async def _request_json(request: web.Request) -> Any:
    """The request's JSON body, reusing the copy ``campaign_middleware`` parsed."""
    if "json_body" in request:
        return request["json_body"]
    return await request.json()


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
    if campaign is None or request.path.startswith("/static/"):
        return await handler(request)
    try:
        validate_campaign(campaign)
    except ValueError as exc:
        return _error_response(str(exc))
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    for experiment_id in sorted(await _named_experiments(request)):
        exp = await get_experiment(db, experiment_id)
        if exp is not None and exp.campaign != campaign:
            return _wrong_campaign(experiment_id, exp.campaign, campaign)
    return await handler(request)


# ===============================================================================
# Route table
# ===============================================================================

routes = web.RouteTableDef()


# ── Reachability check ────────────────────────────────────────────────────────


@routes.get("/api/reachable")
async def handle_reachable(request: web.Request) -> web.Response:
    """Check whether this manager is reachable via an external host.

    Accepts a bare *host* (hostname or IP, no scheme / path / port)
    and makes an outbound GET to ``http://{host}:{port}/api/health``.
    The manager constructs the full URL internally to prevent URL
    injection and token leakage.
    """
    host = (request.query.get("host", "") or "").strip()
    if not host:
        return _error_response("'host' query parameter is required")

    # Reject anything that looks like a URL rather than a bare host.
    if any(c in host for c in ("://", "/", "?", "#", "@")):
        return _error_response("'host' must be a bare hostname or IP, not a URL")

    # Reject obviously invalid host patterns without reaching DNS.
    if not re.match(r"^[a-zA-Z0-9.\-:\[\]]+$", host):
        return _error_response("'host' contains invalid characters")

    server_port = request.url.port
    target_url = f"http://{host}:{server_port}/api/health"
    token: str = request.config_dict["mlsweep_token"]

    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                target_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=aiohttp.ClientTimeout(total=4),
            ) as resp:
                reachable = resp.status == 200
    except Exception:
        reachable = False

    return _json_response({"reachable": reachable})


# ── Campaigns ──────────────────────────────────────────────────────────────────


@routes.get("/api/campaigns")
async def handle_list_campaigns(request: web.Request) -> web.Response:
    """List campaigns with experiment and job counts (always includes the default)."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    return _json_response(await list_campaigns(db))


@routes.put("/api/experiments/{experiment_id}/campaign")
async def handle_update_experiment_campaign(request: web.Request) -> web.Response:
    """Move an experiment, with all its runs, to another campaign.

    Body: ``{"campaign": NAME}``.  With ``?campaign=`` the experiment must
    currently be in that campaign (enforced by ``campaign_middleware``).
    """
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    status_filter = request.query.get("status")
    experiments = await list_experiments_with_counts(
        db, status=status_filter, campaign=request.query.get("campaign"),  # type: ignore[arg-type]
    )
    return _json_response(experiments)


@routes.post("/api/experiments")
async def handle_create_experiment(request: web.Request) -> web.Response:
    """Create a new experiment.

    Its campaign comes from the body's ``campaign``, else ``?campaign=``, else
    the default campaign.  Re-creating an experiment that already exists in
    another campaign is refused with 409.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    experiment_id = body.get("experiment_id") or body.get("id")
    if not experiment_id:
        return _error_response("'experiment_id' is required")

    # Sanitize experiment_id
    try:
        experiment_id = _sanitize_experiment_id(experiment_id)
    except ValueError as exc:
        return _error_response(str(exc), status=400)

    name = body.get("name") or experiment_id
    controller_id = body.get("controller_id")
    note = body.get("note")
    status = body.get("status", "running")
    expected_jobs = body.get("expected_jobs", 0)
    singular_dims = body.get("singular_dims") or []
    max_concurrent = body.get("max_concurrent", 0)
    skip_rules = body.get("skip_rules") or {}
    metric = body.get("metric")
    goal = body.get("goal")
    if goal is not None and goal not in ("minimize", "maximize"):
        return _error_response("'goal' must be 'minimize' or 'maximize'")
    query_campaign = request.query.get("campaign")
    campaign = body.get("campaign") or query_campaign or DEFAULT_CAMPAIGN
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

    try:
        exp = await state.db_writer.create_experiment(
            experiment_id=experiment_id,
            name=name,
            campaign=campaign,
            controller_id=controller_id,
            note=note,
            status=status,
            expected_jobs=expected_jobs,
            singular_dims=singular_dims,
            max_concurrent=max_concurrent,
            skip_rules=skip_rules,
            metric=metric,
            goal=goal,
        )
    except Exception as exc:
        return _error_response(str(exc), status=500)

    return _json_response(exp, status=201)


@routes.get("/api/experiments/{experiment_id}")
async def handle_get_experiment(request: web.Request) -> web.Response:
    """Get a single experiment by ID."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
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
    DB).  ``aborted`` additionally cancels any of its jobs that are still
    in-flight.  Moving back to ``running`` resumes scheduling.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")
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
        # Aborting stops the sweep for good, so cancel anything still in flight.
        if status == "aborted":
            await cancel_runs_locked(db, state, state.runs_of(experiment_id))
        # Its last job may have finished while it was paused.
        await _check_experiments_complete_locked(db, state, {experiment_id})
    state.request_schedule()

    # Broadcast event
    _broadcast_experiment_event(request, experiment_id, "status_updated", status=status)
    return _json_response(exp)


@routes.put("/api/experiments/{experiment_id}/name")
async def handle_update_experiment_name(request: web.Request) -> web.Response:
    """Update an experiment's display name."""
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")
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
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
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
    default threshold), merged from in-memory run state.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    status_filter = request.query.get("status")
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
        out.append(d)
    return _json_response(out)


@routes.get("/api/experiments/{experiment_id}/download")
async def handle_download_experiment(request: web.Request) -> web.StreamResponse:
    """Stream experiment directory as a ``.tar.gz`` download.

    Returns 404 if the experiment is not found or its directory does not
    exist on disk.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    experiment_id = request.match_info["experiment_id"]

    # Verify experiment exists in DB
    exp = await get_experiment(db, experiment_id)
    if exp is None:
        return _not_found("experiment")

    # Locate experiment directory on disk
    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    experiment_id = request.query.get("experiment_id")
    campaign = request.query.get("campaign")
    status = request.query.get("status", "pending")
    limit_str = request.query.get("limit")

    limit = int(limit_str) if limit_str else None

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
    state: ManagerState = request.config_dict["mlsweep_state"]

    try:
        body = await _request_json(request)
    except Exception:
        return _error_response("invalid JSON body")

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
    except Exception as exc:
        return _error_response(str(exc), status=500)

    state.request_schedule()

    return _json_response(job, status=201)


@routes.post("/api/jobs/bulk")
async def handle_insert_jobs_bulk(request: web.Request) -> web.Response:
    """Insert multiple jobs in a single transaction."""
    state: ManagerState = request.config_dict["mlsweep_state"]

    try:
        body = await _request_json(request)
    except Exception:
        return _error_response("invalid JSON body")

    jobs_data = body if isinstance(body, list) else body.get("jobs", [])

    if not jobs_data:
        return _error_response("provide a JSON array of job objects")

    try:
        records = await state.db_writer.insert_jobs_bulk(jobs_data)
    except Exception as exc:
        return _error_response(str(exc), status=500)

    state.request_schedule()

    return _json_response(records, status=201)


@routes.get("/api/jobs/pending")
async def handle_list_pending_jobs(request: web.Request) -> web.Response:
    """List pending jobs, optionally filtered by experiment and campaign."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    experiment_id = request.query.get("experiment_id")
    limit_str = request.query.get("limit")
    limit = int(limit_str) if limit_str else None
    jobs = await list_pending_jobs(
        db, experiment_id=experiment_id, limit=limit, campaign=request.query.get("campaign"),
    )
    return _json_response(jobs)


@routes.get("/api/jobs/{run_id}")
async def handle_get_job(request: web.Request) -> web.Response:
    """Get a single job by run_id and experiment_id."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    run_id = request.match_info["run_id"]
    experiment_id = request.query.get("experiment_id", "")
    job = await get_job(db, run_id, experiment_id)
    if job is None:
        return _not_found("job")
    return _json_response(job)


@routes.put("/api/jobs/{run_id}/status")
async def handle_update_job_status(request: web.Request) -> web.Response:
    """Update a job's status (and optionally other fields)."""
    state: ManagerState = request.config_dict["mlsweep_state"]
    run_id = request.match_info["run_id"]
    try:
        body = await _request_json(request)
    except Exception:
        return _error_response("invalid JSON body")

    status = body.get("status")
    if not status:
        return _error_response("'status' is required")
    experiment_id = body.get("experiment_id", "")
    if not experiment_id:
        return _error_response("'experiment_id' is required")

    if status in ACTIVE_JOB_STATUSES:
        return _error_response("only the scheduler dispatches jobs", status=400)

    # Only result columns may be set alongside the status.
    kwargs = {k: v for k, v in body.items() if k in ("exit_code", "elapsed")}

    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
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
    state: ManagerState = request.config_dict["mlsweep_state"]
    run_id = request.match_info["run_id"]
    try:
        body = await _request_json(request)
    except Exception:
        return _error_response("invalid JSON body")

    priority = body.get("priority")
    if priority is None:
        return _error_response("'priority' is required")
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
    state: ManagerState = request.config_dict["mlsweep_state"]
    run_id = request.match_info["run_id"]
    try:
        body = await _request_json(request)
    except Exception:
        return _error_response("invalid JSON body")
    experiment_id = body.get("experiment_id", "")
    if not experiment_id:
        return _error_response("'experiment_id' is required")
    label = body.get("label")
    if label is not None:
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
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


@routes.post("/api/jobs/{run_id}/retry")
async def handle_retry_job(request: web.Request) -> web.Response:
    """Retry a failed job (increment retry count, reset to pending).

    Only jobs in a terminal state (``failed``, ``cancelled``, ``done``)
    can be retried.  Running or pending jobs return 409 Conflict.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    run_id = request.match_info["run_id"]
    experiment_id = request.query.get("experiment_id", "")

    job = await state.db_writer.retry_job(run_id, experiment_id)
    if job is None:
        current = await get_job(db, run_id, experiment_id)
        if current is None:
            return _not_found("job")
        if current.status not in FINISHED_JOB_STATUSES:
            return _error_response(
                f"job is {current.status}; only finished jobs can be retried", status=409,
            )
        return _error_response("max_retries reached", status=400)

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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]

    job = await get_job(db, run_id, experiment_id)
    if job is None or job.experiment_id != experiment_id:
        return _not_found("job")

    async with state.lock:
        await cancel_runs_locked(db, state, [(experiment_id, run_id)])

    updated = await get_job(db, run_id, experiment_id)
    return _json_response(updated if updated is not None else job)


@routes.patch("/api/experiments/{experiment_id}/jobs/{run_id}")
async def handle_patch_job(request: web.Request) -> web.Response:
    """Update a job's priority (reorder).

    Accepts JSON: {priority: int}.  Updates the DB and re-sorts the in-memory
    pending list if the job is pending.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    priority = body.get("priority")
    if priority is None or not isinstance(priority, int):
        return _error_response("'priority' (int) is required")

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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
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
    db = request.config_dict["mlsweep_db"]
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

    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]
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
    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]
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
    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]
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
    db = request.config_dict["mlsweep_db"]

    jobs = await list_jobs_by_experiment(db, experiment_id)
    if not jobs:
        return _error_response("no jobs", status=404)

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
    db = request.config_dict["mlsweep_db"]

    jobs = await list_jobs_by_experiment(db, experiment_id)
    if not jobs:
        return _error_response("no jobs", status=404)

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

    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]

    try:
        artifacts_root = mlsweep_dir / "experiments" / experiment_id / run_id / "artifacts"
        file_path = Path(_resolve_safe_subpath(artifacts_root, artifact_path))
    except ValueError:
        return _error_response("path traversal denied", status=403)
    except (OSError, TypeError):
        return _error_response("invalid path", status=400)

    if not file_path.is_file():
        return _not_found("artifact file")

    return web.FileResponse(file_path)


# ── Workers ────────────────────────────────────────────────────────────────────


def _enrich_worker(wr: WorkerRecord, state: ManagerState) -> dict[str, Any]:
    """Merge a DB WorkerRecord with live WorkerConn data.

    Returns a dict suitable for JSON serialisation, containing all DB
    fields plus ``gpus`` (list[int]) and ``gpu_occupancy`` (dict[int,int])
    from the live connection when available.
    """
    d: dict[str, Any] = dataclasses.asdict(wr)

    wc = state.workers.get(wr.worker_id)
    if wc is not None:
        d["gpus"] = wc.gpus
        d["gpu_occupancy"] = state.occupancy(wc)
        d["gpu_stats"] = wc.gpu_stats
        d["max_jobs_per_gpu"] = wc.max_jobs_per_gpu
    else:
        devices_str = d["devices"]
        if isinstance(devices_str, str) and devices_str:
            try:
                d["gpus"] = json.loads(devices_str)
            except (json.JSONDecodeError, TypeError):
                d["gpus"] = []
        else:
            d["gpus"] = []
        d["gpu_occupancy"] = {}
        d["gpu_stats"] = {}
        d["max_jobs_per_gpu"] = 1
    return d


@routes.get("/api/workers")
async def handle_list_workers(request: web.Request) -> web.Response:
    """List all workers, optionally filtered by status.

    Returns DB records enriched with live GPU occupancy from connected
    workers.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    status_filter = request.query.get("status")
    workers = await list_workers(db, status=status_filter)  # type: ignore[arg-type]
    enriched = [_enrich_worker(wr, state) for wr in workers]
    return _json_response(enriched)


@routes.get("/api/workers/{worker_id}")
async def handle_get_worker(request: web.Request) -> web.Response:
    """Get a single worker by ID, enriched with live GPU occupancy."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    worker_id = request.match_info["worker_id"]
    worker = await get_worker(db, worker_id)
    if worker is None:
        return _not_found("worker")
    return _json_response(_enrich_worker(worker, state))


@routes.post("/api/workers")
async def handle_add_worker(request: web.Request) -> web.Response:
    """Add a new worker dynamically.

    Accepts JSON body: {host (required), remote_dir?, ssh_key?, venv?,
    port?, devices?}.  Generates a worker_id from the host, upserts into the DB,
    and spawns a background task to connect to the worker.

    ``remote_dir`` is optional: the training code is shipped as an artifact and
    the worker falls back to its own working directory when it is empty.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    host = body.get("host")
    remote_dir = body.get("remote_dir") or ""
    if not host:
        return _error_response("'host' is required")

    ssh_key = body.get("ssh_key")
    venv = body.get("venv")
    port = body.get("port", 0)
    devices = body.get("devices")

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
    except Exception as exc:
        state.launching.discard(worker_id)
        return _error_response(str(exc), status=500)

    # Connect in the background; a launch failure marks the worker dead with the reason.
    asyncio.create_task(connect_single_worker(
        db, state,
        host=host,
        remote_dir=remote_dir,
        worker_id=worker_id,
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    worker_id = request.match_info["worker_id"]

    worker = await get_worker(db, worker_id)
    if worker is None:
        return _not_found("worker")

    # Cancels its runs (requeued without spending retries), tells the worker
    # to exit, and marks it dead.
    await declare_worker_dead(db, state, worker_id, "", shutdown=True)

    return _json_response({"worker_id": worker_id, "status": "dead"})


@routes.patch("/api/workers/{worker_id}/concurrency")
async def handle_patch_worker_concurrency(request: web.Request) -> web.Response:
    """Set max_jobs_per_gpu for a live worker.

    Accepts JSON: {"max_jobs_per_gpu": int}.  0 means unlimited.
    Takes effect immediately for the next scheduling pass.
    """
    state: ManagerState = request.config_dict["mlsweep_state"]
    worker_id = request.match_info["worker_id"]

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

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

    Accepts JSON: {"add": [int, ...], "remove": [int, ...]}.
    Removing a GPU evicts any jobs currently using it (they are re-queued).
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    worker_id = request.match_info["worker_id"]

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    add_gpus: list[int] = body.get("add") or []
    remove_gpus: list[int] = body.get("remove") or []

    if not isinstance(add_gpus, list) or not isinstance(remove_gpus, list):
        return _error_response("'add' and 'remove' must be lists of ints")

    wc = state.workers.get(worker_id)
    if wc is None:
        return _not_found("worker")

    async with state.lock:
        # Runs on a GPU being removed are requeued (no retry spent).
        remove_set = set(remove_gpus)
        evict = [r for r in state.runs_on(worker_id) if remove_set & set(r.nodes[worker_id])]
        wc.gpus = sorted((set(wc.gpus) - remove_set) | set(add_gpus))
        await requeue_runs_locked(db, state, [r.key for r in evict], lost=False)
        await state.db_writer.update_worker_devices(worker_id, json.dumps(wc.gpus))
    to_evict = [r.run_id for r in evict]
    state.request_schedule()
    return _json_response({"worker_id": worker_id, "gpus": wc.gpus, "evicted": to_evict})


# ── Artifacts ──────────────────────────────────────────────────────────────────


@routes.get("/api/artifacts/{artifact_id}/meta")
async def handle_get_artifact_meta(request: web.Request) -> web.Response:
    """Get artifact metadata by ID."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
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

    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    artifact_id = request.match_info["artifact_id"]
    artifact = await get_artifact(db, artifact_id)
    if artifact is None:
        return _not_found("artifact")

    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]
    artifacts_dir = mlsweep_dir / "artifacts"
    tarball = artifacts_dir / f"{artifact_id}.tar.gz"

    if not tarball.is_file():
        return _not_found("artifact")

    return web.Response(status=200)


@routes.post("/api/artifacts")
async def handle_register_artifact(request: web.Request) -> web.Response:
    """Register or update an artifact."""
    state: ManagerState = request.config_dict["mlsweep_state"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    artifact_id = body.get("artifact_id") or body.get("id")
    if not artifact_id:
        return _error_response("'artifact_id' is required")

    try:
        artifact = await state.db_writer.register_artifact(
            artifact_id=artifact_id,
            size_bytes=body.get("size_bytes"),
            setup_command=body.get("setup_command"),
        )
    except Exception as exc:
        return _error_response(str(exc), status=500)

    return _json_response(artifact, status=201)


@routes.put("/api/artifacts/{artifact_id}/ref")
async def handle_increment_artifact_ref(request: web.Request) -> web.Response:
    """Increment or decrement an artifact's reference count."""
    state: ManagerState = request.config_dict["mlsweep_state"]
    artifact_id = request.match_info["artifact_id"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    delta = body.get("delta", 1)

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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    artifact_id = request.match_info["artifact_id"]

    # Verify artifact exists in DB
    from mlsweep._manager_db import get_artifact
    artifact = await get_artifact(db, artifact_id)
    if artifact is None:
        return _not_found("artifact")

    # Read raw body
    body = await request.read()

    # Determine storage path
    mlsweep_dir: Path = request.config_dict["mlsweep_dir"]
    artifacts_dir = mlsweep_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Write tarball
    dest = artifacts_dir / f"{artifact_id}.tar.gz"
    # Use a temporary file + rename for atomic write
    tmp = dest.with_suffix(".tar.gz.tmp")
    try:
        tmp.write_bytes(body)
        tmp.rename(dest)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return _error_response(f"Failed to write artifact: {exc}", status=500)

    logger.info("Artifact %s stored (%d bytes)", artifact_id, len(body))

    return _json_response(
        {"artifact_id": artifact_id, "size_bytes": len(body)},
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
    state: ManagerState = request.config_dict["mlsweep_state"]
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]

    ws = web.WebSocketResponse(max_msg_size=0)
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
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    return _json_response({
        "status": "ok",
        "version": _VERSION,
        "workers_connected": sum(wc.status == "connected" for wc in state.workers.values()),
        "jobs_pending": await count_pending_jobs(db),
        "jobs_in_flight": len(state.runs),
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
        app.router.add_static("/static/", path=str(webui_dir), show_index=True)
        return

    # Fallback: look for webui/ directory relative to the mlsweep package
    import mlsweep
    pkg_dir = Path(mlsweep.__file__).resolve().parent
    pkg_web_dir = pkg_dir / "webui"
    if pkg_web_dir.exists():
        logger.info("Serving web UI from package directory: %s", pkg_web_dir)
        app.router.add_static("/static/", path=str(pkg_web_dir), show_index=True)
        return

    logger.info("Web UI directory not found — static routes skipped")


def _broadcast_experiment_event(
    request: web.Request,
    experiment_id: str,
    event_type: str,
    **kwargs: Any,
) -> None:
    """Broadcast an event to all WebSocket subscribers of *experiment_id*."""
    state: ManagerState = request.config_dict["mlsweep_state"]
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
    """
    app = web.Application(
        middlewares=[auth_middleware, campaign_middleware], client_max_size=512 * 1024 * 1024,
    )

    # Store shared objects in app config so handlers can access them
    app["mlsweep_db"] = db
    app["mlsweep_state"] = state
    app["mlsweep_token"] = token
    app["mlsweep_dir"] = Path(mlsweep_dir).expanduser().resolve()

    # Register REST routes
    app.add_routes(routes)

    # Static file serving for web UI
    webui_dir = Path(mlsweep_dir).expanduser().resolve() / "webui"
    _setup_static_routes(app, webui_dir)

    # CORS support — allow any origin for development convenience.
    # In production, restrict this to known origins.

    @web.middleware
    async def cors_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        if request.method == "OPTIONS":
            return web.Response(
                status=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                },
            )
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    # Add CORS middleware after auth so OPTIONS skips auth
    app.middlewares.insert(0, cors_middleware)

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
    "run_app",
    "routes",
]
