"""HTTP REST API and WebSocket event stream for mlsweep manager.

Provides:
  - REST endpoints for experiments, jobs, workers, artifacts
  - WebSocket event stream at ``/ws/experiments/{id}``
  - Static file serving for a web dashboard

All endpoints accept authentication via ``?token=...`` query parameter or
``Authorization: Bearer <token>`` header.  The token is the manager token
stored in ``manager.token``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

import aiosqlite
from aiohttp import WSMsgType, web

from mlsweep._manager_db import (
    WorkerRecord,
    experiment_summary,
    get_artifact,
    get_experiment,
    get_job,
    get_logs_for_run,
    get_metrics_for_run,
    get_worker,
    list_experiments_with_counts,
    list_jobs_by_experiment,
    list_jobs_by_status,
    list_jobs_since,
    list_pending_jobs,
    list_workers,
)
from mlsweep._manager_state import ManagerState
from mlsweep._shared import MsgCancel, MsgShutdown, _resolve_safe_subpath, encode

logger = logging.getLogger(__name__)

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


# ── Experiments ────────────────────────────────────────────────────────────────


@routes.get("/api/experiments")
async def handle_list_experiments(request: web.Request) -> web.Response:
    """List all experiments with job counts, optionally filtered by status."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    status_filter = request.query.get("status")
    experiments = await list_experiments_with_counts(db, status=status_filter)  # type: ignore[arg-type]
    return _json_response(experiments)


@routes.post("/api/experiments")
async def handle_create_experiment(request: web.Request) -> web.Response:
    """Create a new experiment."""
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

    try:
        exp = await state.db_writer.create_experiment(
            experiment_id=experiment_id,
            name=name,
            controller_id=controller_id,
            note=note,
            status=status,
            expected_jobs=expected_jobs,
            singular_dims=singular_dims,
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


@routes.put("/api/experiments/{experiment_id}/status")
async def handle_update_experiment_status(request: web.Request) -> web.Response:
    """Update an experiment's status."""
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")
    status = body.get("status")
    if not status:
        return _error_response("'status' is required")
    exp = await state.db_writer.update_experiment_status(experiment_id, status)
    if exp is None:
        return _not_found("experiment")
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


@routes.delete("/api/experiments/{experiment_id}")
async def handle_delete_experiment(request: web.Request) -> web.Response:
    """Delete an experiment and all its jobs."""
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    existed = await state.db_writer.delete_experiment(experiment_id)
    if not existed:
        return _not_found("experiment")
    return _json_response({"deleted": experiment_id})


@routes.get("/api/experiments/{experiment_id}/summary")
async def handle_experiment_summary(request: web.Request) -> web.Response:
    """Get a summary of an experiment (metadata + job counts)."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    experiment_id = request.match_info["experiment_id"]
    summary = await experiment_summary(db, experiment_id)
    if summary["name"] is None:
        return _not_found("experiment")
    return _json_response(summary)


@routes.get("/api/experiments/{experiment_id}/jobs")
async def handle_list_experiment_jobs(request: web.Request) -> web.Response:
    """List jobs for an experiment, optionally filtered by status."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    experiment_id = request.match_info["experiment_id"]
    status_filter = request.query.get("status")
    jobs = await list_jobs_by_experiment(db, experiment_id, status=status_filter)  # type: ignore[arg-type]
    return _json_response(jobs)


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
    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
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
      - status: filter by status (default: 'pending')
      - limit: max number of results
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    experiment_id = request.query.get("experiment_id")
    status = request.query.get("status", "pending")
    limit_str = request.query.get("limit")

    limit = int(limit_str) if limit_str else None

    if experiment_id:
        jobs = await list_jobs_by_experiment(db, experiment_id, status=status)  # type: ignore[arg-type]
        if limit is not None:
            jobs = jobs[:limit]
    elif status == "pending":
        jobs = await list_pending_jobs(db, limit=limit)
    else:
        jobs = await list_jobs_by_status(db, status, limit=limit)  # type: ignore[arg-type]

    return _json_response(jobs)


@routes.post("/api/jobs")
async def handle_insert_job(request: web.Request) -> web.Response:
    """Insert a single job."""
    state: ManagerState = request.config_dict["mlsweep_state"]

    try:
        body = await request.json()
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
            jobs_per_gpu=body.get("jobs_per_gpu", 1),
        )
    except Exception as exc:
        return _error_response(str(exc), status=500)

    # Add to pending list if status is pending
    if job.status == "pending":
        async with state.scheduler_lock:
            state.insert_pending(job)
        # Trigger scheduling
        _trigger_scheduling(request)

    return _json_response(job, status=201)


@routes.post("/api/jobs/bulk")
async def handle_insert_jobs_bulk(request: web.Request) -> web.Response:
    """Insert multiple jobs in a single transaction."""
    state: ManagerState = request.config_dict["mlsweep_state"]

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    jobs_data = body if isinstance(body, list) else body.get("jobs", [])

    if not jobs_data:
        return _error_response("provide a JSON array of job objects")

    try:
        records = await state.db_writer.insert_jobs_bulk(jobs_data)
    except Exception as exc:
        return _error_response(str(exc), status=500)

    # Add pending jobs to state
    async with state.scheduler_lock:
        for job in records:
            if job.status == "pending":
                state.insert_pending(job)

    # Trigger scheduling
    if records:
        _trigger_scheduling(request)

    return _json_response(records, status=201)


@routes.get("/api/jobs/pending")
async def handle_list_pending_jobs(request: web.Request) -> web.Response:
    """List pending jobs, optionally filtered by experiment."""
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    experiment_id = request.query.get("experiment_id")
    limit_str = request.query.get("limit")
    limit = int(limit_str) if limit_str else None
    jobs = await list_pending_jobs(db, experiment_id=experiment_id, limit=limit)
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
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    status = body.get("status")
    if not status:
        return _error_response("'status' is required")
    experiment_id = body.get("experiment_id", "")
    if not experiment_id:
        return _error_response("'experiment_id' is required")

    # Build kwargs from body, excluding 'status' and 'experiment_id'
    kwargs = {k: v for k, v in body.items() if k not in ("status", "experiment_id")}

    job = await state.db_writer.update_job_status(run_id, experiment_id, status, **kwargs)
    if job is None:
        return _not_found("job")

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
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    priority = body.get("priority")
    if priority is None:
        return _error_response("'priority' is required")
    experiment_id = body.get("experiment_id", "")
    if not experiment_id:
        return _error_response("'experiment_id' is required")

    # Update in DB
    job = await state.db_writer.update_job_priority(run_id, experiment_id, priority)
    if job is None:
        return _not_found("job")

    # Update in-memory sorted list
    state.update_priority(run_id, priority)

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
        body = await request.json()
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
    """Cancel a pending job."""
    state: ManagerState = request.config_dict["mlsweep_state"]
    run_id = request.match_info["run_id"]
    experiment_id = request.query.get("experiment_id", "")

    job = await state.db_writer.cancel_job(run_id, experiment_id)
    if job is None:
        return _not_found("job")

    # Remove from pending list
    state.remove_pending(run_id)

    # Broadcast event
    _broadcast_experiment_event(
        request, job.experiment_id, "job_cancelled", run_id=run_id,
    )

    return _json_response(job)


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

    # Fetch current job to check status
    current = await get_job(db, run_id, experiment_id)
    if current is None:
        return _not_found("job")

    if current.status not in ("failed", "cancelled", "done"):
        return _error_response(
            f"job is {current.status}; only terminal jobs can be retried",
            status=409,
        )

    job = await state.db_writer.increment_retry(run_id, experiment_id)
    if job is None:
        return _error_response(
            "job not found or max_retries reached", status=400,
        )

    # Add back to pending list
    async with state.scheduler_lock:
        state.insert_pending(job)

    # Broadcast event
    _broadcast_experiment_event(
        request, job.experiment_id, "job_retried", run_id=run_id,
    )

    # Trigger scheduling
    _trigger_scheduling(request)

    return _json_response(job)


@routes.delete("/api/experiments/{experiment_id}/jobs/{run_id}")
async def handle_delete_job(request: web.Request) -> web.Response:
    """Cancel a job by experiment_id and run_id.

    - If pending: remove from state.pending, mark 'cancelled', broadcast job_done.
    - If dispatched/running: send MsgCancel to the assigned worker, mark
      'cancelled', broadcast job_done.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]

    job = await get_job(db, run_id, experiment_id)
    if job is None:
        return _not_found("job")

    # Verify the job belongs to the given experiment
    if job.experiment_id != experiment_id:
        return _not_found("job")

    status = job.status

    if status == "pending":
        # Remove from pending list
        state.remove_pending(run_id)

        # Mark cancelled in DB
        job = await state.db_writer.cancel_job(run_id, experiment_id)
        if job is None:
            return _error_response("failed to cancel job", status=500)

        # Broadcast job_done
        _broadcast_experiment_event(
            request, experiment_id, "job_done",
            run_id=run_id, status="cancelled", success=False,
        )

    elif status in ("dispatched", "running"):
        # Send MsgCancel to the assigned worker
        wc = None
        if job.worker_id:
            wc = state.workers.get(job.worker_id)
            if wc is not None:
                cancel_msg = encode(MsgCancel(run_id=run_id))
                try:
                    wc.send_queue.put_nowait(cancel_msg)
                except asyncio.QueueFull:
                    pass

        # Mark cancelled in DB
        job = await state.db_writer.update_job_status(run_id, experiment_id, "cancelled")
        if job is None:
            return _error_response("failed to cancel job", status=500)

        # Remove from in-flight
        if_job = state.remove_in_flight(run_id)
        if if_job is not None and wc is not None:
            wc.in_flight.pop(run_id, None)

        # Broadcast job_done
        _broadcast_experiment_event(
            request, experiment_id, "job_done",
            run_id=run_id, status="cancelled", success=False,
        )

    else:
        # Already terminal — mark cancelled if not already
        if status != "cancelled":
            job = await state.db_writer.update_job_status(run_id, experiment_id, "cancelled")
        _broadcast_experiment_event(
            request, experiment_id, "job_done",
            run_id=run_id, status="cancelled", success=False,
        )

    # Trigger scheduling in case we freed up resources
    _trigger_scheduling(request)

    return _json_response(job)


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

    # Update priority in DB (works for any status)
    job = await state.db_writer.update_job_priority(run_id, experiment_id, priority)
    if job is None:
        return _error_response("failed to update priority", status=500)

    # If job is in pending list, update its priority and re-sort
    if job.status == "pending":
        state.update_priority(run_id, priority)
        # Trigger scheduler to re-evaluate
        _trigger_scheduling(request)

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


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/log")
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

    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
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
    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
    artifacts_dir = mlsweep_dir / "experiments" / experiment_id / run_id / "artifacts"
    if not artifacts_dir.is_dir():
        return _error_response("no artifacts", status=404)
    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="mlsweep_")
    os.close(fd)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _zip_directory, tmp_path, artifacts_dir)
    except Exception:
        os.unlink(tmp_path)
        raise
    dl_name = f"{run_id[:12]}-artifacts.zip"
    _schedule_cleanup(tmp_path, delay=300)
    return web.FileResponse(
        tmp_path,
        headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
    )


@routes.get("/api/experiments/{experiment_id}/artifacts.zip")
async def handle_zip_experiment_artifacts(request: web.Request) -> web.StreamResponse:
    """Serve a zip of all artifact files for every run in an experiment."""
    experiment_id = request.match_info["experiment_id"]
    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
    exp_dir = mlsweep_dir / "experiments" / experiment_id
    if not exp_dir.is_dir():
        return _error_response("no experiment artifacts", status=404)
    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="mlsweep_")
    os.close(fd)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _zip_experiment_artifacts, tmp_path, exp_dir)
    except Exception:
        os.unlink(tmp_path)
        raise
    dl_name = f"{experiment_id[:16]}-artifacts.zip"
    _schedule_cleanup(tmp_path, delay=300)
    return web.FileResponse(
        tmp_path,
        headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
    )


@routes.get("/api/experiments/{experiment_id}/metrics.zip")
async def handle_zip_experiment_metrics(request: web.Request) -> web.StreamResponse:
    """Serve a zip of all metrics files (JSONL) for every run in an experiment."""
    experiment_id = request.match_info["experiment_id"]
    db = request.config_dict["mlsweep_db"]

    jobs = await list_jobs_by_experiment(db, experiment_id)
    if not jobs:
        return _error_response("no jobs", status=404)

    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="mlsweep_")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for job in jobs:
                rows = await get_metrics_for_run(db, job.run_id, experiment_id)
                if rows:
                    jsonl = "\n".join(json.dumps(row) for row in rows)
                    zf.writestr(f"{job.run_id}.jsonl", jsonl)
    except Exception:
        os.unlink(tmp_path)
        raise

    dl_name = f"{experiment_id[:16]}-metrics.zip"
    _schedule_cleanup(tmp_path, delay=300)
    return web.FileResponse(
        tmp_path,
        headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
    )


@routes.get("/api/experiments/{experiment_id}/logs.zip")
async def handle_zip_experiment_logs(request: web.Request) -> web.StreamResponse:
    """Serve a zip of all log files for every run in an experiment."""
    experiment_id = request.match_info["experiment_id"]
    db = request.config_dict["mlsweep_db"]

    jobs = await list_jobs_by_experiment(db, experiment_id)
    if not jobs:
        return _error_response("no jobs", status=404)

    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="mlsweep_")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for job in jobs:
                text = await get_logs_for_run(db, job.run_id, experiment_id)
                if text:
                    zf.writestr(f"{job.run_id}.log", text)
    except Exception:
        os.unlink(tmp_path)
        raise

    dl_name = f"{experiment_id[:16]}-logs.zip"
    _schedule_cleanup(tmp_path, delay=300)
    return web.FileResponse(
        tmp_path,
        headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
    )


@routes.get("/api/experiments/{experiment_id}/jobs/{run_id}/artifacts/{path:.*}")
async def handle_get_job_artifact(request: web.Request) -> web.StreamResponse:
    """Serve a file from a job's artifacts/ directory."""
    experiment_id = request.match_info["experiment_id"]
    run_id = request.match_info["run_id"]
    artifact_path = request.match_info["path"]

    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
    file_path = mlsweep_dir / "experiments" / experiment_id / run_id / "artifacts" / artifact_path

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
        d["gpu_occupancy"] = wc.gpu_occupancy
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

    Accepts JSON body: {host (required), remote_dir (required), ssh_key?, venv?,
    port?, devices?}.  Generates a worker_id from the host, upserts into the DB,
    and spawns a background task to connect to the worker.
    """
    db: aiosqlite.Connection = request.config_dict["mlsweep_db"]
    state: ManagerState = request.config_dict["mlsweep_state"]

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body")

    host = body.get("host")
    remote_dir = body.get("remote_dir")
    if not host or not remote_dir:
        return _error_response("'host' and 'remote_dir' are required")

    ssh_key = body.get("ssh_key")
    venv = body.get("venv")
    port = body.get("port", 0)
    devices = body.get("devices")

    # Use explicit worker_id (reconnect) or derive from host/port (new worker).
    worker_id = body.get("worker_id") or f"{host}:{port or 'dynamic'}"

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
        return _error_response(str(exc), status=500)

    # Spawn background connection task
    mlsweep_dir: str = request.config_dict["mlsweep_dir"]
    asyncio.create_task(
        _connect_worker(
            db, state, worker_id, host, remote_dir,
            ssh_key=ssh_key,
            venv=venv,
            port=port,
            devices=devices,
            mlsweep_dir=mlsweep_dir,
        )
    )

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

    # Mark worker dead in DB
    worker = await state.db_writer.update_worker_status(worker_id, "dead")
    if worker is None:
        return _not_found("worker")

    # If connected, send shutdown and clean up
    wc = state.workers.get(worker_id)
    if wc is not None:
        # Send MsgShutdown
        try:
            wc.send_queue.put_nowait(encode(MsgShutdown()))
        except asyncio.QueueFull:
            pass

        # Mark worker as dead in memory; the worker read task will detect
        # the status change and clean up gracefully.
        async with state.scheduler_lock:
            wc.status = "dead"

    # Re-queue any pending jobs from this worker: reset dispatched/running
    # jobs assigned to this worker back to pending
    re_queued_rows = await state.db_writer.reset_worker_jobs(worker_id)
    re_queued: list[tuple[str, str]] = list(re_queued_rows)
    for run_id, experiment_id in re_queued:
        state.broadcast(
            experiment_id,
            {
                "type": "job_done",
                "run_id": run_id,
                "status": "pending",
                "success": False,
                "worker_id": worker_id,
                "orphaned": True,
            },
        )

    # Re-fetch the re-queued jobs and insert into pending list
    for run_id, experiment_id in re_queued:
        job = await get_job(db, run_id, experiment_id)
        if job is not None:
            async with state.scheduler_lock:
                state.insert_pending(job)

    # Trigger scheduling for re-queued jobs
    if state.dispatch_callback is not None and re_queued:
        asyncio.get_running_loop().create_task(state.dispatch_callback())

    return _json_response({"worker_id": worker_id, "status": "dead"})


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

    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
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

    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
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
    mlsweep_dir = Path(request.config_dict["mlsweep_dir"]).expanduser().resolve()
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
    state: ManagerState = request.config_dict["mlsweep_state"]
    return _json_response({
        "status": "ok",
        "workers_connected": len(state.workers),
        "jobs_pending": len(state.pending),
        "jobs_in_flight": len(state.in_flight),
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


# ===============================================================================
# Scheduling trigger
# ===============================================================================


def _trigger_scheduling(request: web.Request) -> None:
    """Schedule the ``schedule_pending`` callback in the event loop.

    This is a best-effort fire-and-forget call.  If the callback is not
    set (e.g., during early startup), it is silently ignored.
    """
    state: ManagerState = request.config_dict["mlsweep_state"]
    if state.dispatch_callback is not None:
        asyncio.get_running_loop().create_task(state.dispatch_callback())


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
# Dynamic worker connection helper
# ===============================================================================


async def _connect_worker(
    db: aiosqlite.Connection,
    state: ManagerState,
    worker_id: str,
    host: str,
    remote_dir: str,
    *,
    ssh_key: str | None = None,
    venv: str | None = None,
    port: int = 0,
    devices: list[int] | None = None,
    mlsweep_dir: str = "~/.mlsweep",
) -> None:
    """Launch and connect to a single worker, registering it in the manager state.

    This is the background task spawned by ``POST /api/workers``.  Delegates
    to ``connect_single_worker`` in ``_manager_workers``.
    """
    from mlsweep._manager_workers import connect_single_worker

    wc = await connect_single_worker(
        db, state,
        host=host,
        remote_dir=remote_dir,
        worker_id=worker_id,
        scratch_dir="/tmp/mlsweep",
        ssh_key=ssh_key,
        venv=venv,
        port=port,
        devices=devices,
    )
    if wc is None:
        await state.db_writer.update_worker_status(worker_id, "dead")
        return

    # Update DB status
    await state.db_writer.update_worker_status(worker_id, "connected")

    logger.info("Dynamic worker %s connected on %s:%d", worker_id, host, wc.port)


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
    app = web.Application(middlewares=[auth_middleware], client_max_size=512 * 1024 * 1024)

    # Store shared objects in app config so handlers can access them
    app["mlsweep_db"] = db
    app["mlsweep_state"] = state
    app["mlsweep_token"] = token
    app["mlsweep_dir"] = str(mlsweep_dir)

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
