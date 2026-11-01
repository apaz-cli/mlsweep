"""TensorBoard and W&B export of stored metrics (``mlsweep/_export.py``).

The HTTP tests run the aiohttp app in-process against an in-memory database,
so they can seed metrics directly and swap in a fake ``wandb`` module.
"""

import asyncio
import io
import json
import sys
import threading
import types
import zipfile

import aiosqlite
import pytest
from aiohttp.test_utils import TestClient, TestServer

import mlsweep._manager_http as http
from mlsweep._export import _crc32c, experiment_labels, tfevents_for_run
from mlsweep._manager_db import (
    create_experiment,
    init_db,
    insert_job,
    insert_metric,
    update_job_status,
)
from mlsweep._manager_state import ManagerState

TOKEN = "t"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_crc32c_check_value():
    assert _crc32c(b"123456789") == 0xE3069283


def _load_events(data, tmp_path):
    loader = pytest.importorskip("tensorboard.backend.event_processing.event_file_loader")
    path = tmp_path / "events.out.tfevents.0.test"
    path.write_bytes(data)
    return list(loader.EventFileLoader(str(path)).Load())


def _scalars(events):
    """{tag: [(step, value)]} from loaded events (the loader moves values into tensors)."""
    out = {}
    for ev in events:
        for v in ev.summary.value:
            if v.metadata.plugin_data.plugin_name != "hparams":
                out.setdefault(v.tag, []).append((ev.step, pytest.approx(v.tensor.float_val[0])))
    return out


def _hparams(events):
    plugin_data_pb2 = pytest.importorskip("tensorboard.plugins.hparams.plugin_data_pb2")
    out = {}
    for ev in events:
        for v in ev.summary.value:
            if v.metadata.plugin_data.plugin_name == "hparams":
                out[v.tag] = plugin_data_pb2.HParamsPluginData.FromString(
                    v.metadata.plugin_data.content)
    return out


def test_tfevents_round_trip(tmp_path):
    rows = [{"step": 1, "loss": 0.5, "note": "text is skipped"},
            {"step": 2, "loss": 0.25, "acc": 0.9}]
    data = tfevents_for_run(
        rows, hparams={"lr": 0.1, "opt": "adam", "bn": True, "layers": [1, 2]},
        status="done", start_time=1000.0, end_time=1100.0,
    )
    events = _load_events(data, tmp_path)
    assert events[0].file_version == "brain.Event:2"
    assert _scalars(events) == {"loss": [(1, 0.5), (2, 0.25)], "acc": [(2, 0.9)]}

    hp = _hparams(events)
    start = hp["_hparams_/session_start_info"].session_start_info
    assert start.hparams["lr"].number_value == 0.1
    assert start.hparams["opt"].string_value == "adam"
    assert start.hparams["bn"].bool_value is True
    assert start.hparams["layers"].string_value == "[1, 2]"
    assert start.start_time_secs == 1000.0
    end = hp["_hparams_/session_end_info"].session_end_info
    assert end.status == 1  # STATUS_SUCCESS
    assert end.end_time_secs == 1100.0


@pytest.mark.parametrize("status,code", [("failed", 2), ("cancelled", 2), ("running", None)])
def test_tfevents_session_end_status(tmp_path, status, code):
    data = tfevents_for_run([{"step": 1, "loss": 1.0}], hparams={}, status=status,
                            start_time=0.0, end_time=None)
    end = _hparams(_load_events(data, tmp_path)).get("_hparams_/session_end_info")
    assert (end.session_end_info.status if end else None) == code


# ── HTTP ───────────────────────────────────────────────────────────────────────


async def _seed(db):
    await create_experiment(db, experiment_id="e1", name="my sweep")
    for rid, status, combo in [("r1", "done", {"lr": 0.1}), ("r2", "failed", {"lr": 0.2}),
                               ("r3", "done", {"lr": 0.3})]:
        job = await insert_job(db, run_id=rid, experiment_id="e1", command=["x"], combo=combo)
        await update_job_status(db, rid, "e1", status)
        if rid != "r3":  # r3 logged nothing
            for step in (1, 2):
                await insert_metric(db, job.job_key, job.attempt, step, {"loss": step * combo["lr"]})
    # Campaign beta: an experiment named like e1, and one with no jobs.
    await create_experiment(db, experiment_id="e2", name="my sweep", campaign="beta")
    await create_experiment(db, experiment_id="e3", name="empty", campaign="beta")
    job = await insert_job(db, run_id="r1", experiment_id="e2", command=["x"], combo={"lr": 0.5})
    await update_job_status(db, "r1", "e2", "done")
    await insert_metric(db, job.job_key, job.attempt, 1, {"loss": 0.5})


async def _seed_no_metrics(db):
    """An experiment whose only run logged nothing."""
    await _seed(db)
    await create_experiment(db, experiment_id="e4", name="no metrics")
    job = await insert_job(db, run_id="r4", experiment_id="e4", command=["x"])
    await update_job_status(db, "r4", "e4", "done")


async def _seed_collisions(db):
    """Names that sanitize to the same zip path component."""
    for eid, name, campaign, rid in [
        ("c1", "sweep/a", "team/x", "r/1"),
        ("c2", "sweep_a", "team_x", "r_1"),
        ("c3", "sweep", "team/x", "r/1"),
    ]:
        await create_experiment(db, experiment_id=eid, name=name, campaign=campaign)
        job = await insert_job(db, run_id=rid, experiment_id=eid, command=["x"])
        await update_job_status(db, rid, eid, "done")
        await insert_metric(db, job.job_key, job.attempt, 1, {"loss": 1.0})
    # Two run ids in one experiment that sanitize the same way.
    await create_experiment(db, experiment_id="c4", name="runs")
    for rid in ("run/1", "run_1"):
        job = await insert_job(db, run_id=rid, experiment_id="c4", command=["x"])
        await update_job_status(db, rid, "c4", "done")
        await insert_metric(db, job.job_key, job.attempt, 1, {"loss": 2.0})


def _with_client(fn, seed=_seed):
    async def run():
        db = await aiosqlite.connect(":memory:")
        try:
            await init_db(db)
            await seed(db)
            app = http.create_app(db, ManagerState(token=TOKEN), TOKEN, mlsweep_dir="/nonexistent")
            async with TestClient(TestServer(app)) as client:
                await fn(client)
        finally:
            await db.close()
    asyncio.run(run())


def test_tensorboard_zip(tmp_path):
    async def body(client):
        r = await client.get("/api/experiments/e1/tensorboard.zip", headers=AUTH)
        assert r.status == 200
        zf = zipfile.ZipFile(io.BytesIO(await r.read()))
        names = sorted(zf.namelist())
        assert [n.rsplit("/", 1)[0] for n in names] == ["my sweep/r1", "my sweep/r2"]
        assert all(n.rsplit("/", 1)[1].startswith("events.out.tfevents.") for n in names)
        events = _load_events(zf.read(names[1]), tmp_path)
        assert _scalars(events) == {"loss": [(1, 0.2), (2, 0.4)]}
        assert _hparams(events)["_hparams_/session_start_info"] \
            .session_start_info.hparams["lr"].number_value == 0.2

        r = await client.get("/api/experiments/e1/tensorboard.zip?only_done=1", headers=AUTH)
        names = zipfile.ZipFile(io.BytesIO(await r.read())).namelist()
        assert [n.split("/")[1] for n in names] == ["r1"]

        r = await client.get("/api/experiments/nope/tensorboard.zip", headers=AUTH)
        assert r.status == 404
    _with_client(body)


def test_experiment_labels_disambiguate_repeated_names():
    assert experiment_labels([("a", "x"), ("b", "x"), ("c", "y"), ("d", "d")]) == {
        "a": "x_a", "b": "x_b", "c": "y", "d": "d"}


async def _zip_names(client, path):
    r = await client.get(path, headers=AUTH)
    assert r.status == 200, await r.text()
    return sorted(n.rsplit("/", 1)[0] for n in zipfile.ZipFile(io.BytesIO(await r.read())).namelist())


def test_tensorboard_zip_names_do_not_collide():
    async def body(client):
        r = await client.get("/api/export/tensorboard.zip", headers=AUTH)
        assert r.status == 200
        names = zipfile.ZipFile(io.BytesIO(await r.read())).namelist()
        assert len(names) == len(set(names))
        folders = [n.rsplit("/", 1)[0] for n in names]
        # Campaigns, experiments and runs all sanitize to duplicate names here.
        assert len(names) == 5
        assert len({f.split("/")[0] for f in folders}) == 3
        assert len({tuple(f.split("/")[:2]) for f in folders}) == 4
        assert len(set(folders)) == 5
    _with_client(body, seed=_seed_collisions)


def test_tensorboard_zip_without_metrics_is_404():
    async def body(client):
        r = await client.get("/api/experiments/e4/tensorboard.zip", headers=AUTH)
        assert r.status == 404
        assert (await r.json())["error"] == "no metrics"
    _with_client(body, seed=_seed_no_metrics)


def test_tensorboard_zip_scopes():
    async def body(client):
        assert await _zip_names(client, "/api/export/tensorboard.zip?campaign=beta") == ["my sweep/r1"]
        assert await _zip_names(client, "/api/export/tensorboard.zip?experiment_id=e1") == [
            "my sweep/r1", "my sweep/r2"]
        # Everything: a campaign level on top, and the repeated name made unique.
        assert await _zip_names(client, "/api/export/tensorboard.zip") == [
            "beta/my sweep_e2/r1", "default/my sweep_e1/r1", "default/my sweep_e1/r2"]
        assert await _zip_names(client, "/api/export/tensorboard.zip?only_done=1") == [
            "beta/my sweep_e2/r1", "default/my sweep_e1/r1"]
        r = await client.get("/api/export/tensorboard.zip", headers=AUTH)
        assert 'filename="mlsweep-tensorboard.zip"' in r.headers["Content-Disposition"]
        r = await client.get("/api/export/tensorboard.zip?campaign=nobody", headers=AUTH)
        assert r.status == 404
        r = await client.get("/api/export/tensorboard.zip?campaign=beta&experiment_id=e1", headers=AUTH)
        assert r.status == 404 and (await r.json())["campaign"] == "default"
    _with_client(body)


class _FakeRun:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.logged = []
        self.summary = {}
        self.exit_code = None
        self.url = f"https://wandb.example/{kwargs['name']}"

    def log(self, data, step):
        self.logged.append((step, data))

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class _Runs(list):
    gate: threading.Event
    fail_names: set[str]


@pytest.fixture
def fake_wandb(monkeypatch):
    runs = _Runs()
    mod = types.ModuleType("wandb")
    mod.Settings = lambda **kw: kw
    runs.gate = threading.Event()  # cleared by a test to hold uploads open
    runs.gate.set()
    runs.fail_names = set()  # run names whose init should raise

    def init(**kwargs):
        runs.gate.wait(10)
        if kwargs["name"] in runs.fail_names:
            raise RuntimeError("simulated upload failure")
        runs.append(_FakeRun(kwargs))
        return runs[-1]

    mod.init = init
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.setattr(http, "wandb_info", lambda: {"available": True, "version": "fake", "env_key": False})
    return runs


async def _wait_export(client, export_id):
    for _ in range(200):
        r = await client.get(f"/api/exports/{export_id}", headers=AUTH)
        data = await r.json()
        if data["status"] != "running":
            return data
        await asyncio.sleep(0.02)
    raise AssertionError("export did not finish")


def test_wandb_export_uploads_each_run(fake_wandb):
    async def body(client):
        r = await client.post("/api/experiments/e1/export/wandb", headers=AUTH, json={
            "project": "proj", "entity": "team", "api_key": "secret"})
        assert r.status == 202
        started = await r.json()
        assert "secret" not in json.dumps(started)
        done = await _wait_export(client, started["export_id"])
        assert done["status"] == "done"
        assert [(x["run_id"], x["status"]) for x in done["runs"]] == [
            ("r1", "done"), ("r2", "done"), ("r3", "skipped")]
        assert done["runs"][0]["url"] == "https://wandb.example/r1"

        r1, r2 = fake_wandb
        assert r1.kwargs["project"] == "proj" and r1.kwargs["entity"] == "team"
        assert r1.kwargs["group"] == "my sweep"
        assert r1.kwargs["tags"] == ["default"]
        assert r2.kwargs["config"] == {"lr": 0.2}
        settings = r1.kwargs["settings"]
        assert settings["api_key"] == "secret"
        assert settings["console"] == "off" and settings["x_disable_stats"] is True
        assert r2.logged == [(1, {"loss": 0.2}), (2, {"loss": 0.4})]
        assert r2.summary["mlsweep/status"] == "failed"
        assert r1.exit_code == 0

        r = await client.get("/api/exports?experiment_id=e1", headers=AUTH)
        assert [x["export_id"] for x in await r.json()] == [started["export_id"]]
    _with_client(body)


def test_wandb_export_cancel(fake_wandb):
    async def body(client):
        fake_wandb.gate.clear()
        try:
            r = await client.post("/api/export/wandb", headers=AUTH, json={"project": "p"})
            assert r.status == 202
            export_id = (await r.json())["export_id"]
            for _ in range(200):
                data = await (await client.get(f"/api/exports/{export_id}", headers=AUTH)).json()
                if any(x["status"] == "uploading" for x in data["runs"]):
                    break
                await asyncio.sleep(0.02)
            assert any(x["status"] == "uploading" for x in data["runs"])
            r = await client.post(f"/api/exports/{export_id}/cancel", headers=AUTH)
            assert r.status == 200
            done = await _wait_export(client, export_id)
        finally:
            fake_wandb.gate.set()
        assert done["status"] == "cancelled"
        assert done["done"] == done["total"]
        assert all(x["status"] != "pending" for x in done["runs"])
        assert done["runs"][0]["status"] == "failed"
        assert done["runs"][0]["error"] == "cancelled"
        assert [x["error"] for x in done["runs"][1:]] == ["cancelled"] * 3
        # The blocked init returns after the gate opens, but logs nothing.
        for _ in range(200):
            if fake_wandb and fake_wandb[0].exit_code is not None:
                break
            await asyncio.sleep(0.01)
        assert fake_wandb[0].logged == []
        assert fake_wandb[0].exit_code == 1
    _with_client(body)


def test_wandb_export_run_failure(fake_wandb):
    fake_wandb.fail_names.add("r2")

    async def body(client):
        r = await client.post("/api/experiments/e1/export/wandb", headers=AUTH, json={"project": "p"})
        done = await _wait_export(client, (await r.json())["export_id"])
        by_run = {x["run_id"]: x for x in done["runs"]}
        assert done["status"] == "failed"
        assert done["error"] == "some runs failed to upload"
        assert by_run["r1"]["status"] == "done"
        assert by_run["r2"]["status"] == "failed"
        assert "simulated upload failure" in by_run["r2"]["error"]
        assert by_run["r3"]["status"] == "skipped"
        assert [run.kwargs["name"] for run in fake_wandb] == ["r1"]
    _with_client(body)


@pytest.mark.parametrize("env,expected", [(None, "online"), ("offline", "offline"),
                                          ("disabled", "disabled")])
def test_wandb_export_honors_wandb_mode(fake_wandb, monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("WANDB_MODE", raising=False)
    else:
        monkeypatch.setenv("WANDB_MODE", env)

    async def body(client):
        r = await client.post("/api/experiments/e1/export/wandb", headers=AUTH,
                              json={"project": "p", "only_done": True})
        done = await _wait_export(client, (await r.json())["export_id"])
        assert done["status"] == "done"
        assert [run.kwargs["mode"] for run in fake_wandb] == [expected]
    _with_client(body)


def test_wandb_export_only_done(fake_wandb):
    async def body(client):
        r = await client.post("/api/experiments/e1/export/wandb", headers=AUTH,
                              json={"project": "p", "only_done": True})
        done = await _wait_export(client, (await r.json())["export_id"])
        assert [x["run_id"] for x in done["runs"]] == ["r1", "r3"]
        assert [run.kwargs["name"] for run in fake_wandb] == ["r1"]
    _with_client(body)


def test_wandb_export_campaign_and_all(fake_wandb):
    async def body(client):
        r = await client.post("/api/export/wandb?campaign=beta", headers=AUTH, json={"project": "p"})
        done = await _wait_export(client, (await r.json())["export_id"])
        assert (done["scope"], done["scope_label"], done["experiments"]) == ("campaign", "campaign beta", 2)
        assert [(x["experiment_id"], x["run_id"], x["experiment"]) for x in done["runs"]] == [
            ("e2", "r1", "my sweep")]
        assert fake_wandb[-1].kwargs["tags"] == ["beta"]

        r = await client.post("/api/export/wandb", headers=AUTH, json={"project": "p", "only_done": True})
        done = await _wait_export(client, (await r.json())["export_id"])
        assert done["scope_label"] == "all campaigns"
        assert sorted((x["experiment"], x["run_id"]) for x in done["runs"]) == [
            ("my sweep_e1", "r1"), ("my sweep_e1", "r3"), ("my sweep_e2", "r1")]

        r = await client.get("/api/exports?campaign=beta", headers=AUTH)
        assert [x["scope"] for x in await r.json()] == ["campaign"]
        r = await client.get("/api/exports?experiment_id=e1", headers=AUTH)
        assert [x["scope"] for x in await r.json()] == ["all"]
    _with_client(body)


def test_wandb_export_refuses_overlap(fake_wandb):
    async def body(client):
        fake_wandb.gate.clear()
        try:
            r = await client.post("/api/export/wandb", headers=AUTH, json={"project": "p"})
            assert r.status == 202
            r = await client.post("/api/experiments/e2/export/wandb", headers=AUTH, json={"project": "p"})
            assert r.status == 409
            assert "all campaigns" in (await r.json())["error"]
        finally:
            fake_wandb.gate.set()
        for x in await (await client.get("/api/exports", headers=AUTH)).json():
            await _wait_export(client, x["export_id"])
    _with_client(body)


@pytest.mark.parametrize("path,payload,status", [
    ("/api/experiments/e1/export/wandb", {}, 400),
    ("/api/experiments/e1/export/wandb", {"project": "  "}, 400),
    ("/api/experiments/nope/export/wandb", {"project": "p"}, 404),
])
def test_wandb_export_rejects(fake_wandb, path, payload, status):
    async def body(client):
        r = await client.post(path, headers=AUTH, json=payload)
        assert r.status == status
    _with_client(body)


def test_wandb_export_without_wandb(monkeypatch):
    monkeypatch.setattr(http, "wandb_info", lambda: {"available": False, "version": None, "env_key": False})

    async def body(client):
        r = await client.post("/api/experiments/e1/export/wandb", headers=AUTH, json={"project": "p"})
        assert r.status == 501
        assert "pip install" in (await r.json())["error"]
    _with_client(body)
