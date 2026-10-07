"""Campaigns in the DB layer and the manager HTTP API.

A campaign groups experiments.  These tests cover name validation, the DB
queries that filter by campaign, and the ``?campaign=`` contract of the REST
API: listings are filtered, new experiments land in the campaign, and every
route that names an experiment refuses one from another campaign without
changing anything.  CLI and web UI tests live in test_campaigns_cli.py and
test_campaigns_webui.py.
"""

import asyncio
import json
import time
import urllib.request

import aiosqlite
import pytest

from campaign_helpers import (
    call as _call,
    ok as _ok,
    q as _q,
    seed as _seed,
    shared_manager,
    snapshot as _snapshot,
    uid as _uid,
)

from mlsweep._manager_db import (
    DbWriter,
    create_experiment,
    delete_experiment,
    experiment_summary,
    get_experiment,
    init_db,
    insert_job,
    list_campaigns,
    list_experiments_with_counts,
    list_jobs_by_status,
    list_pending_jobs,
    list_schedulable_jobs,
    update_experiment_campaign,
    update_experiment_status,
    update_job_status,
)
from mlsweep._shared import DEFAULT_CAMPAIGN, validate_campaign
from mlsweep.run_sweep import _WebSocket

_TOKEN = "test-token"


# ── Name validation ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["default", "a", "A-b_c", "x" * 128, "0", "-", "_"])
def test_validate_campaign_accepts(name):
    assert validate_campaign(name) == name


@pytest.mark.parametrize("name", [
    "", "x" * 129, "a b", "a/b", "a.b", "*", "ünï", "a\nb", "a?b", "a&b", "a=b", None, 3, ["a"],
])
def test_validate_campaign_rejects(name):
    with pytest.raises(ValueError):
        validate_campaign(name)


def test_default_campaign_is_valid():
    assert DEFAULT_CAMPAIGN == "default"
    validate_campaign(DEFAULT_CAMPAIGN)


# ── DB layer ────────────────────────────────────────────────────────────────────


def _db_test(fn):
    """Run ``fn(db)`` against a fresh in-memory database."""
    async def run():
        db = await aiosqlite.connect(":memory:")
        try:
            await init_db(db)
            await fn(db)
        finally:
            await db.close()
    asyncio.run(run())


async def _job(db, eid, rid, status="pending"):
    await insert_job(db, run_id=rid, experiment_id=eid, command=["echo"])
    if status != "pending":
        await update_job_status(db, rid, eid, status)


def test_db_migration_files_old_experiments_under_default():
    """A database from before campaigns gains the column; old rows are in the default."""
    async def body(db):
        await db.execute("DROP INDEX idx_experiments_campaign")
        await db.execute("ALTER TABLE experiments DROP COLUMN campaign")
        await db.execute(
            "INSERT INTO experiments (experiment_id, name, submit_time) VALUES ('old', 'old', 0)")
        await db.commit()
        await init_db(db)
        assert (await get_experiment(db, "old")).campaign == "default"
        assert [c["campaign"] for c in await list_campaigns(db)] == ["default"]
        await create_experiment(db, experiment_id="new", name="new", campaign="alpha")
        await init_db(db)  # idempotent
        assert (await get_experiment(db, "new")).campaign == "alpha"
    _db_test(body)


def test_db_experiment_defaults_to_default_campaign():
    async def body(db):
        exp = await create_experiment(db, experiment_id="e", name="e")
        assert exp.campaign == "default"
        assert (await get_experiment(db, "e")).campaign == "default"
    _db_test(body)


def test_db_experiment_stores_campaign():
    async def body(db):
        exp = await create_experiment(db, experiment_id="e", name="e", campaign="alpha")
        assert exp.campaign == "alpha"
        assert (await get_experiment(db, "e")).campaign == "alpha"
    _db_test(body)


def test_db_recreate_keeps_campaign():
    async def body(db):
        await create_experiment(db, experiment_id="e", name="first", campaign="alpha")
        again = await create_experiment(db, experiment_id="e", name="second", campaign="beta")
        assert again.name == "second"
        assert again.campaign == "alpha"
    _db_test(body)


def test_db_move_experiment():
    async def body(db):
        await create_experiment(db, experiment_id="e", name="e", campaign="alpha")
        moved = await update_experiment_campaign(db, "e", "beta")
        assert moved is not None and moved.campaign == "beta"
        assert (await get_experiment(db, "e")).campaign == "beta"
        assert await update_experiment_campaign(db, "missing", "beta") is None
    _db_test(body)


def test_db_move_carries_jobs():
    async def body(db):
        await create_experiment(db, experiment_id="e", name="e", campaign="alpha")
        await _job(db, "e", "r1")
        assert [j.run_id for j in await list_pending_jobs(db, campaign="alpha")] == ["r1"]
        await update_experiment_campaign(db, "e", "beta")
        assert await list_pending_jobs(db, campaign="alpha") == []
        assert [j.run_id for j in await list_pending_jobs(db, campaign="beta")] == ["r1"]
    _db_test(body)


def test_db_list_experiments_filters_by_campaign():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await create_experiment(db, experiment_id="a2", name="a2", campaign="alpha")
        await create_experiment(db, experiment_id="b1", name="b1", campaign="beta")
        await create_experiment(db, experiment_id="d1", name="d1")
        ids = lambda rows: sorted(r["experiment_id"] for r in rows)  # noqa: E731
        assert ids(await list_experiments_with_counts(db)) == ["a1", "a2", "b1", "d1"]
        assert ids(await list_experiments_with_counts(db, campaign="alpha")) == ["a1", "a2"]
        assert ids(await list_experiments_with_counts(db, campaign="beta")) == ["b1"]
        assert ids(await list_experiments_with_counts(db, campaign="default")) == ["d1"]
        assert await list_experiments_with_counts(db, campaign="nothing") == []
        rows = await list_experiments_with_counts(db, campaign="alpha")
        assert all(r["campaign"] == "alpha" for r in rows)
    _db_test(body)


def test_db_list_experiments_status_and_campaign_combine():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await create_experiment(db, experiment_id="a2", name="a2", campaign="alpha", status="paused")
        await create_experiment(db, experiment_id="b1", name="b1", campaign="beta", status="paused")
        rows = await list_experiments_with_counts(db, status="paused", campaign="alpha")
        assert [r["experiment_id"] for r in rows] == ["a2"]
        rows = await list_experiments_with_counts(db, status="paused")
        assert sorted(r["experiment_id"] for r in rows) == ["a2", "b1"]
    _db_test(body)


def test_db_list_campaigns_counts():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await create_experiment(db, experiment_id="a2", name="a2", campaign="alpha")
        await create_experiment(db, experiment_id="b1", name="b1", campaign="beta")
        await _job(db, "a1", "r1")
        await _job(db, "a1", "r2", "done")
        await _job(db, "a2", "r1", "failed")
        await _job(db, "b1", "r1", "done")
        camps = await list_campaigns(db)
        assert [c["campaign"] for c in camps] == ["alpha", "beta", "default"]
        alpha, beta, default = camps
        assert alpha["experiments"] == 2
        assert alpha["job_counts"] == {"total": 3, "done": 1, "failed": 1, "running": 0, "pending": 1,
                                      "xfailed": 0, "cancelled": 0, "dispatched": 0}
        assert beta["experiments"] == 1
        assert beta["job_counts"]["done"] == 1 and beta["job_counts"]["total"] == 1
        assert default == {
            "campaign": "default", "experiments": 0, "last_submit": None,
            "job_counts": {"total": 0, "done": 0, "failed": 0, "running": 0, "pending": 0,
                           "xfailed": 0, "cancelled": 0, "dispatched": 0},
        }
        assert alpha["last_submit"] is not None
    _db_test(body)


def test_db_list_campaigns_empty_db_lists_default_only():
    async def body(db):
        camps = await list_campaigns(db)
        assert [c["campaign"] for c in camps] == ["default"]
    _db_test(body)


def test_db_list_campaigns_default_not_duplicated():
    async def body(db):
        await create_experiment(db, experiment_id="d1", name="d1")
        await create_experiment(db, experiment_id="z1", name="z1", campaign="zeta")
        camps = await list_campaigns(db)
        assert [c["campaign"] for c in camps] == ["default", "zeta"]
        assert camps[0]["experiments"] == 1
    _db_test(body)


def test_db_campaign_disappears_with_its_last_experiment():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await delete_experiment(db, "a1")
        assert [c["campaign"] for c in await list_campaigns(db)] == ["default"]
    _db_test(body)


def test_db_campaign_disappears_when_its_last_experiment_moves():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await update_experiment_campaign(db, "a1", "beta")
        assert [c["campaign"] for c in await list_campaigns(db)] == ["beta", "default"]
    _db_test(body)


def test_db_list_pending_jobs_filters():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await create_experiment(db, experiment_id="b1", name="b1", campaign="beta")
        for rid in ("r1", "r2", "r3"):
            await _job(db, "a1", rid)
        await _job(db, "b1", "r1")
        await _job(db, "b1", "r2", "done")
        keys = lambda jobs: sorted((j.experiment_id, j.run_id) for j in jobs)  # noqa: E731
        assert len(await list_pending_jobs(db)) == 4
        assert keys(await list_pending_jobs(db, campaign="alpha")) == [("a1", "r1"), ("a1", "r2"), ("a1", "r3")]
        assert keys(await list_pending_jobs(db, campaign="beta")) == [("b1", "r1")]
        assert len(await list_pending_jobs(db, campaign="alpha", limit=2)) == 2
        assert await list_pending_jobs(db, experiment_id="a1", campaign="beta") == []
        assert len(await list_pending_jobs(db, experiment_id="a1", campaign="alpha")) == 3
        assert await list_pending_jobs(db, campaign="nothing") == []
    _db_test(body)


def test_db_list_jobs_by_status_filters():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await create_experiment(db, experiment_id="b1", name="b1", campaign="beta")
        await _job(db, "a1", "r1", "done")
        await _job(db, "a1", "r2", "done")
        await _job(db, "b1", "r1", "done")
        await _job(db, "b1", "r2", "failed")
        assert len(await list_jobs_by_status(db, "done")) == 3
        assert sorted(j.run_id for j in await list_jobs_by_status(db, "done", campaign="alpha")) == ["r1", "r2"]
        assert len(await list_jobs_by_status(db, "done", campaign="alpha", limit=1)) == 1
        assert [j.experiment_id for j in await list_jobs_by_status(db, "failed", campaign="beta")] == ["b1"]
        assert await list_jobs_by_status(db, "failed", campaign="alpha") == []
    _db_test(body)


def test_db_summary_includes_campaign():
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        assert (await experiment_summary(db, "a1"))["campaign"] == "alpha"
        assert (await experiment_summary(db, "missing"))["campaign"] is None
    _db_test(body)


def test_db_scheduler_ignores_campaigns():
    """Campaigns organise experiments; they never gate dispatch."""
    async def body(db):
        await create_experiment(db, experiment_id="a1", name="a1", campaign="alpha")
        await create_experiment(db, experiment_id="b1", name="b1", campaign="beta")
        await create_experiment(db, experiment_id="p1", name="p1", campaign="alpha")
        await update_experiment_status(db, "p1", "paused")
        await _job(db, "a1", "r1")
        await _job(db, "b1", "r1")
        await _job(db, "p1", "r1")
        assert sorted(j.experiment_id for j in await list_schedulable_jobs(db)) == ["a1", "b1"]
    _db_test(body)


def test_db_writer_campaign_round_trip():
    async def body(db):
        writer = DbWriter(db)
        task = asyncio.create_task(writer.run())
        try:
            exp = await writer.create_experiment(experiment_id="e", name="e", campaign="alpha")
            assert exp.campaign == "alpha"
            moved = await writer.update_experiment_campaign("e", "beta")
            assert moved.campaign == "beta"
            assert await writer.update_experiment_campaign("missing", "beta") is None
        finally:
            task.cancel()
    _db_test(body)


# ── HTTP API ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "campaigns_http")


# Every route that names an experiment: (method, path, body, status when allowed).
# Paths and bodies take the experiment id.
_ROUTES = {
    "get_experiment": ("GET", "/api/experiments/{e}", None, 200),
    "summary": ("GET", "/api/experiments/{e}/summary", None, 200),
    "list_jobs": ("GET", "/api/experiments/{e}/jobs", None, 200),
    "download": ("GET", "/api/experiments/{e}/download", None, 200),
    "artifacts_zip": ("GET", "/api/experiments/{e}/artifacts.zip", None, 200),
    "metrics_zip": ("GET", "/api/experiments/{e}/metrics.zip", None, 200),
    "logs_zip": ("GET", "/api/experiments/{e}/logs.zip", None, 200),
    "run_metrics": ("GET", "/api/experiments/{e}/jobs/r1/metrics", None, 404),  # none logged
    "run_logs": ("GET", "/api/experiments/{e}/jobs/r1/logs", None, 404),  # none logged
    "run_artifacts": ("GET", "/api/experiments/{e}/jobs/r1/artifacts", None, 200),
    "run_artifacts_zip": ("GET", "/api/experiments/{e}/jobs/r1/artifacts.zip", None, 200),
    "run_artifact_file": ("GET", "/api/experiments/{e}/jobs/r1/artifacts/f.txt", None, 200),
    "set_status": ("PUT", "/api/experiments/{e}/status", {"status": "paused"}, 200),
    "set_name": ("PUT", "/api/experiments/{e}/name", {"name": "renamed"}, 200),
    "set_max_concurrent": ("PUT", "/api/experiments/{e}/max_concurrent", {"max_concurrent": 3}, 200),
    "move": ("PUT", "/api/experiments/{e}/campaign", {"campaign": "elsewhere"}, 200),
    "delete_experiment": ("DELETE", "/api/experiments/{e}", None, 200),
    "delete_job": ("DELETE", "/api/experiments/{e}/jobs/r1", None, 200),
    "patch_job": ("PATCH", "/api/experiments/{e}/jobs/r1", {"priority": 5}, 200),
    "get_job": ("GET", "/api/jobs/r1?experiment_id={e}", None, 200),
    "jobs_query": ("GET", "/api/jobs?experiment_id={e}&status=pending", None, 200),
    "jobs_pending": ("GET", "/api/jobs/pending?experiment_id={e}", None, 200),
    "job_status": ("PUT", "/api/jobs/r1/status", {"experiment_id": "{e}", "status": "cancelled"}, 200),
    "job_priority": ("PUT", "/api/jobs/r1/priority", {"experiment_id": "{e}", "priority": 5}, 200),
    "job_label": ("PUT", "/api/jobs/r1/label", {"experiment_id": "{e}", "label": "x"}, 200),
    "cancel": ("POST", "/api/jobs/r1/cancel?experiment_id={e}", None, 200),
    "retry": ("POST", "/api/jobs/r2/retry?experiment_id={e}", None, 200),
    "insert_job": ("POST", "/api/jobs", {"experiment_id": "{e}", "run_id": "new", "command": ["echo"]}, 201),
    "bulk_list": ("POST", "/api/jobs/bulk",
                  [{"experiment_id": "{e}", "run_id": "new", "command": ["echo"]}], 201),
    "bulk_object": ("POST", "/api/jobs/bulk",
                    {"jobs": [{"experiment_id": "{e}", "run_id": "new", "command": ["echo"]}]}, 201),
}


def _fill(template, eid):
    return json.loads(json.dumps(template).replace("{e}", eid))


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_route_refuses_experiment_from_other_campaign(mgr, route):
    method, path, body, _ = _ROUTES[route]
    eid = _seed(mgr, "alpha")
    before = _snapshot(mgr, eid)
    status, data = _call(mgr, method, _q(path.format(e=eid), "beta"), _fill(body, eid))
    assert status == 404, (status, data)
    assert data["campaign"] == "alpha"
    assert "'alpha'" in data["error"] and "'beta'" in data["error"] and eid in data["error"]
    assert _snapshot(mgr, eid) == before


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_route_allows_experiment_in_campaign(mgr, route):
    method, path, body, ok = _ROUTES[route]
    eid = _seed(mgr, "alpha")
    status, data = _call(mgr, method, _q(path.format(e=eid), "alpha"), _fill(body, eid))
    assert status == ok, (status, data)
    assert not (isinstance(data, dict) and "campaign" in data and "error" in data)


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_route_without_campaign_is_unrestricted(mgr, route):
    method, path, body, ok = _ROUTES[route]
    eid = _seed(mgr, "alpha")
    status, data = _call(mgr, method, path.format(e=eid), _fill(body, eid))
    assert status == ok, (status, data)


def test_bulk_with_one_foreign_experiment_inserts_nothing(mgr):
    mine, theirs = _seed(mgr, "alpha"), _seed(mgr, "beta")
    jobs = [{"experiment_id": mine, "run_id": "n1", "command": ["echo"]},
            {"experiment_id": theirs, "run_id": "n1", "command": ["echo"]}]
    status, data = _call(mgr, "POST", _q("/api/jobs/bulk", "alpha"), jobs)
    assert status == 404 and data["campaign"] == "beta"
    for eid in (mine, theirs):
        _, rows = _call(mgr, "GET", f"/api/experiments/{eid}/jobs")
        assert sorted(j["run_id"] for j in rows) == ["r1", "r2"]


def test_unknown_experiment_with_campaign_gets_plain_404(mgr):
    status, data = _call(mgr, "GET", _q("/api/experiments/no_such_exp", "alpha"))
    assert status == 404
    assert "campaign" not in data


@pytest.mark.parametrize("path", [
    "/api/experiments", "/api/experiments/whatever", "/api/jobs", "/api/campaigns", "/api/workers",
])
@pytest.mark.parametrize("bad", ["a%20b", "a%2Fb", "%2A", "x" * 129])
def test_invalid_campaign_param_is_400(mgr, path, bad):
    status, data = _call(mgr, "GET", f"{path}?campaign={bad}")
    assert status == 400, (status, data)
    assert "campaign" in data["error"]


def test_empty_campaign_param_is_400(mgr):
    status, _ = _call(mgr, "GET", "/api/experiments?campaign=")
    assert status == 400


@pytest.mark.parametrize("path", ["/api/workers", "/api/health", "/api/campaigns"])
def test_campaign_param_ignored_where_no_experiment(mgr, path):
    status, _ = _call(mgr, "GET", _q(path, "anything"))
    assert status == 200


def test_static_files_ignore_campaign_param(mgr):
    req = urllib.request.Request(f"{mgr.url}/static/experiments.html?campaign=not%20valid")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        assert b"campaign-select" in resp.read()


def test_webui_ships_campaign_script(mgr):
    with urllib.request.urlopen(f"{mgr.url}/static/campaign.js", timeout=10) as resp:
        assert b"MLCampaign" in resp.read()


# ── Creating experiments ───────────────────────────────────────────────────────


def _create(server, body, campaign=None):
    path = "/api/experiments" if campaign is None else _q("/api/experiments", campaign)
    return _call(server, "POST", path, body)


def test_create_defaults_to_default_campaign(mgr):
    eid = _uid("e")
    status, data = _create(mgr, {"experiment_id": eid})
    assert status == 201 and data["campaign"] == "default"


def test_create_takes_campaign_from_body(mgr):
    eid = _uid("e")
    status, data = _create(mgr, {"experiment_id": eid, "campaign": "alpha"})
    assert status == 201 and data["campaign"] == "alpha"
    assert _call(mgr, "GET", f"/api/experiments/{eid}")[1]["campaign"] == "alpha"


def test_create_takes_campaign_from_query(mgr):
    eid = _uid("e")
    status, data = _create(mgr, {"experiment_id": eid}, campaign="beta")
    assert status == 201 and data["campaign"] == "beta"


def test_create_body_and_query_may_agree(mgr):
    eid = _uid("e")
    status, data = _create(mgr, {"experiment_id": eid, "campaign": "beta"}, campaign="beta")
    assert status == 201 and data["campaign"] == "beta"


def test_create_body_and_query_must_agree(mgr):
    eid = _uid("e")
    status, data = _create(mgr, {"experiment_id": eid, "campaign": "alpha"}, campaign="beta")
    assert status == 400 and "does not match" in data["error"]
    assert _call(mgr, "GET", f"/api/experiments/{eid}")[0] == 404


@pytest.mark.parametrize("bad", ["a b", "a/b", "*", "x" * 129, 7, ["a"]])
def test_create_rejects_invalid_campaign(mgr, bad):
    eid = _uid("e")
    status, data = _create(mgr, {"experiment_id": eid, "campaign": bad})
    assert status == 400 and "campaign" in data["error"]
    assert _call(mgr, "GET", f"/api/experiments/{eid}")[0] == 404


def test_create_empty_campaign_means_default(mgr):
    eid = _uid("e")
    status, data = _create(mgr, {"experiment_id": eid, "campaign": ""})
    assert status == 201 and data["campaign"] == "default"


def test_recreate_in_same_campaign_updates(mgr):
    eid = _uid("e")
    _create(mgr, {"experiment_id": eid, "name": "one", "campaign": "alpha"})
    status, data = _create(mgr, {"experiment_id": eid, "name": "two", "campaign": "alpha"})
    assert status == 201 and data["name"] == "two" and data["campaign"] == "alpha"


@pytest.mark.parametrize("body_campaign,query_campaign", [("beta", None), (None, "beta"), (None, None)])
def test_recreate_in_other_campaign_conflicts(mgr, body_campaign, query_campaign):
    eid = _uid("e")
    _create(mgr, {"experiment_id": eid, "name": "one", "campaign": "alpha"})
    body = {"experiment_id": eid, "name": "two"}
    if body_campaign:
        body["campaign"] = body_campaign
    status, data = _create(mgr, body, campaign=query_campaign)
    assert status == 409, (status, data)
    assert data["campaign"] == "alpha"
    exp = _call(mgr, "GET", f"/api/experiments/{eid}")[1]
    assert exp["campaign"] == "alpha" and exp["name"] == "one"


# ── Listing ─────────────────────────────────────────────────────────────────────


def test_list_experiments_by_campaign(mgr):
    cx, cy = _uid("cx"), _uid("cy")
    x1, x2, y1 = _seed(mgr, cx), _seed(mgr, cx), _seed(mgr, cy)
    ids = lambda rows: sorted(r["experiment_id"] for r in rows)  # noqa: E731
    assert ids(_ok(mgr, "GET", _q("/api/experiments", cx))) == sorted([x1, x2])
    assert ids(_ok(mgr, "GET", _q("/api/experiments", cy))) == [y1]
    assert {x1, x2, y1} <= set(ids(_ok(mgr, "GET", "/api/experiments")))
    assert _ok(mgr, "GET", _q("/api/experiments", _uid("empty"))) == []
    rows = _ok(mgr, "GET", _q("/api/experiments", cx))
    assert all(r["campaign"] == cx for r in rows)
    assert rows[0]["job_counts"]["total"] == 2


def test_list_experiments_status_and_campaign(mgr):
    cx = _uid("cx")
    x1, x2 = _seed(mgr, cx), _seed(mgr, cx)
    _ok(mgr, "PUT", f"/api/experiments/{x2}/status", {"status": "paused"})
    rows = _ok(mgr, "GET", _q("/api/experiments?status=paused", cx))
    assert [r["experiment_id"] for r in rows] == [x2]
    rows = _ok(mgr, "GET", _q("/api/experiments?status=running", cx))
    assert [r["experiment_id"] for r in rows] == [x1]


def test_list_campaigns_endpoint(mgr):
    cx = _uid("cx")
    _seed(mgr, cx)
    _seed(mgr, cx)
    camps = _ok(mgr, "GET", "/api/campaigns")
    names = [c["campaign"] for c in camps]
    assert names == sorted(names)
    assert "default" in names
    mine = next(c for c in camps if c["campaign"] == cx)
    assert mine["experiments"] == 2
    assert mine["job_counts"]["total"] == 4
    assert mine["job_counts"]["done"] == 2
    assert mine["job_counts"]["pending"] == 2


def test_list_jobs_by_campaign(mgr):
    cx, cy = _uid("cx"), _uid("cy")
    x1, y1 = _seed(mgr, cx), _seed(mgr, cy)
    pending = _ok(mgr, "GET", _q("/api/jobs?status=pending", cx))
    assert [(j["experiment_id"], j["run_id"]) for j in pending] == [(x1, "r1")]
    done = _ok(mgr, "GET", _q("/api/jobs?status=done", cy))
    assert [(j["experiment_id"], j["run_id"]) for j in done] == [(y1, "r2")]
    assert len(_ok(mgr, "GET", _q("/api/jobs?status=done&limit=1", cx))) == 1
    pending = _ok(mgr, "GET", _q("/api/jobs/pending", cy))
    assert [(j["experiment_id"], j["run_id"]) for j in pending] == [(y1, "r1")]
    everything = _ok(mgr, "GET", "/api/jobs/pending")
    assert {(x1, "r1"), (y1, "r1")} <= {(j["experiment_id"], j["run_id"]) for j in everything}


def test_summary_and_get_report_campaign(mgr):
    eid = _seed(mgr, "alpha")
    assert _ok(mgr, "GET", f"/api/experiments/{eid}/summary")["campaign"] == "alpha"
    assert _ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == "alpha"


# ── Moving experiments ──────────────────────────────────────────────────────────


def test_move_experiment(mgr):
    src, dst = _uid("src"), _uid("dst")
    eid = _seed(mgr, src)
    moved = _ok(mgr, "PUT", _q(f"/api/experiments/{eid}/campaign", src), {"campaign": dst})
    assert moved["campaign"] == dst
    assert _ok(mgr, "GET", _q("/api/experiments", src)) == []
    assert [r["experiment_id"] for r in _ok(mgr, "GET", _q("/api/experiments", dst))] == [eid]
    pending = _ok(mgr, "GET", _q("/api/jobs/pending", dst))
    assert [(j["experiment_id"], j["run_id"]) for j in pending] == [(eid, "r1")]
    assert src not in [c["campaign"] for c in _ok(mgr, "GET", "/api/campaigns")]
    status, data = _call(mgr, "GET", _q(f"/api/experiments/{eid}", src))
    assert status == 404 and data["campaign"] == dst


def test_move_to_same_campaign_is_a_no_op(mgr):
    eid = _seed(mgr, "alpha")
    assert _ok(mgr, "PUT", f"/api/experiments/{eid}/campaign", {"campaign": "alpha"})["campaign"] == "alpha"


@pytest.mark.parametrize("body,raw", [
    ({}, None), ({"campaign": ""}, None), ({"campaign": None}, None), ({"campaign": "a b"}, None),
    ({"campaign": "*"}, None), ({"campaign": 5}, None), (None, b"{nope"), ([], None),
])
def test_move_rejects_bad_body(mgr, body, raw):
    eid = _seed(mgr, "alpha")
    status, data = _call(mgr, "PUT", f"/api/experiments/{eid}/campaign", body, raw=raw)
    assert status == 400, (status, data)
    assert _ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == "alpha"


def test_move_unknown_experiment(mgr):
    status, _ = _call(mgr, "PUT", "/api/experiments/no_such_exp/campaign", {"campaign": "x"})
    assert status == 404


def test_delete_removes_empty_campaign(mgr):
    cx = _uid("cx")
    eid = _seed(mgr, cx)
    assert cx in [c["campaign"] for c in _ok(mgr, "GET", "/api/campaigns")]
    _ok(mgr, "DELETE", f"/api/experiments/{eid}")
    assert cx not in [c["campaign"] for c in _ok(mgr, "GET", "/api/campaigns")]


# ── WebSocket stream ────────────────────────────────────────────────────────────


def _ws(server, eid, campaign=None):
    url = server.url.replace("http://", "ws://") + f"/ws/experiments/{eid}?token={_TOKEN}"
    if campaign:
        url += f"&campaign={campaign}"
    return _WebSocket(url, _TOKEN, timeout=5.0)


def test_ws_refuses_other_campaign(mgr):
    eid = _seed(mgr, "alpha")
    ws = _ws(mgr, eid, "beta")
    with pytest.raises(ConnectionError) as exc:
        ws.connect()
    assert "404" in str(exc.value)


@pytest.mark.parametrize("campaign", ["alpha", None])
def test_ws_streams_campaign_moves(mgr, campaign):
    eid = _seed(mgr, "alpha")
    ws = _ws(mgr, eid, campaign)
    ws.connect()
    seen = None
    try:
        _ok(mgr, "PUT", f"/api/experiments/{eid}/campaign", {"campaign": "omega"})
        deadline = time.time() + 10
        for event in ws.iter_events(max_idle=10):
            if event.get("type") == "campaign_updated":
                seen = event
                break
            if time.time() > deadline:
                break
    finally:
        ws.close()
    assert seen == {"type": "campaign_updated", "experiment_id": eid, "campaign": "omega"}
