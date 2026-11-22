"""Moving runs between experiments, purging runs, garbage collection, and the
logger's refusal to overwrite logged steps."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.error
from pathlib import Path

import pytest

from conftest import _api_get, _api_post, _api_request
from mlsweep.logger import MLSweepLogger

TOKEN = "test-token"


def _post(url: str, path: str, data: object) -> tuple[int, dict]:  # type: ignore[type-arg]
    try:
        return 200, _api_post(url, TOKEN, path, data)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _job(url: str, exp: str, run: str, status: str = "done") -> None:
    _api_post(url, TOKEN, "/api/jobs", {"run_id": run, "experiment_id": exp, "command": ["true"],
                                         "combo": {"lr": 0.1}})
    if status != "pending":
        _api_request(url, TOKEN, "PUT", f"/api/jobs/{run}/status",
                     {"experiment_id": exp, "status": status, "exit_code": 0})


def _run_dir(server, exp: str, run: str) -> Path:  # type: ignore[no-untyped-def]
    d = Path(server.mlsweep_dir) / "experiments" / exp / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "training.log").write_text(f"log of {run}\n")
    return d


def _db(server) -> sqlite3.Connection:  # type: ignore[no-untyped-def]
    return sqlite3.connect(str(Path(server.mlsweep_dir).parent / "manager.db"))


def test_move_runs(manager_server) -> None:  # type: ignore[no-untyped-def]
    server, url = manager_server
    for e in ("src", "dst"):
        _api_post(url, TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": e})
    _job(url, "src", "r1")
    _job(url, "src", "r2", "failed")
    _job(url, "src", "busy", "pending")
    _job(url, "dst", "taken")
    _job(url, "src", "taken")
    _run_dir(server, "src", "r1")

    status, r = _post(url, "/api/experiments/src/jobs/move", {"run_ids": ["r1", "r2"], "to": "dst"})
    assert status == 200 and sorted(r["moved"]) == ["r1", "r2"]
    dst_runs = {j["run_id"] for j in _api_get(url, TOKEN, "/api/experiments/dst/jobs")}
    assert {"r1", "r2", "taken"} <= dst_runs
    assert (Path(server.mlsweep_dir) / "experiments/dst/r1/training.log").read_text() == "log of r1\n"
    assert not (Path(server.mlsweep_dir) / "experiments/src/r1").exists()

    status, r = _post(url, "/api/experiments/src/jobs/move", {"run_ids": ["busy"], "to": "dst"})
    assert status == 409 and "pending" in r["error"]
    status, r = _post(url, "/api/experiments/src/jobs/move", {"run_ids": ["taken"], "to": "dst"})
    assert status == 409 and "already has a run" in r["error"]
    status, r = _post(url, "/api/experiments/src/jobs/move", {"run_ids": ["nope"], "to": "dst"})
    assert status == 409
    status, _ = _post(url, "/api/experiments/src/jobs/move", {"run_ids": ["r1"], "to": "missing"})
    assert status == 404
    status, _ = _post(url, "/api/experiments/src/jobs/move", {"run_ids": [], "to": "dst"})
    assert status == 400


def test_move_respects_campaign(manager_server) -> None:  # type: ignore[no-untyped-def]
    _, url = manager_server
    _api_post(url, TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": "a", "campaign": "one"})
    _api_post(url, TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": "b", "campaign": "two"})
    _job(url, "a", "r")
    status, _ = _post(url, "/api/experiments/a/jobs/move?campaign=one", {"run_ids": ["r"], "to": "b"})
    assert status == 404


def test_purge_runs(manager_server) -> None:  # type: ignore[no-untyped-def]
    server, url = manager_server
    _api_post(url, TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": "p"})
    _job(url, "p", "bad", "failed")
    _job(url, "p", "good")
    _job(url, "p", "queued", "pending")
    d = _run_dir(server, "p", "bad")
    status, r = _post(url, "/api/experiments/p/jobs/purge", {"run_ids": ["bad"]})
    assert status == 200 and r["purged"] == ["bad"] and r["bytes_freed"] > 0
    assert not d.exists()
    assert {j["run_id"] for j in _api_get(url, TOKEN, "/api/experiments/p/jobs")} == {"good", "queued"}
    status, r = _post(url, "/api/experiments/p/jobs/purge", {"run_ids": ["queued"]})
    assert status == 409


def test_gc(manager_server) -> None:  # type: ignore[no-untyped-def]
    server, url = manager_server
    root = Path(server.mlsweep_dir)
    _api_post(url, TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": "kept"})
    _job(url, "kept", "r")
    kept = _run_dir(server, "kept", "r")
    stray_run = _run_dir(server, "kept", "deleted_run")
    gone = _run_dir(server, "gone_exp", "r")
    fresh = _run_dir(server, "fresh_exp", "r")  # too new to collect
    old = time.time() - 7200
    for p in (stray_run, gone.parent, gone):
        os.utime(p, (old, old))
    (root / "artifacts").mkdir(exist_ok=True)
    stray_art = root / "artifacts" / "sha256:abc.tar.gz"
    stray_art.write_bytes(b"x" * 100)
    os.utime(stray_art, (old, old))

    plan = _api_post(url, TOKEN, "/api/gc", {})
    assert plan["dry_run"] is True
    paths = {d["path"] for d in plan["directories"]} | {f["path"] for f in plan["artifact_files"]}
    assert paths == {"experiments/kept/deleted_run", "experiments/gone_exp", "artifacts/sha256:abc.tar.gz"}
    assert stray_run.exists() and gone.exists()  # dry run deletes nothing

    done = _api_post(url, TOKEN, "/api/gc", {"dry_run": False, "vacuum": True})
    assert done["removed_paths"] == 3 and done["vacuumed"]
    assert not stray_run.exists() and not gone.exists() and not stray_art.exists()
    assert kept.exists() and fresh.exists()
    assert _api_post(url, TOKEN, "/api/gc", {})["directories"] == []


def test_gc_removes_orphan_rows(manager_server) -> None:  # type: ignore[no-untyped-def]
    server, url = manager_server
    _api_post(url, TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": "x"})
    _job(url, "x", "r")
    con = _db(server)
    con.execute("INSERT INTO metrics (job_key, attempt, step, data) VALUES (99999, 0, 1, '{}')")
    con.commit()
    con.close()
    plan = _api_post(url, TOKEN, "/api/gc", {})
    assert plan["database"]["metric_rows"] == 1
    _api_post(url, TOKEN, "/api/gc", {"dry_run": False})
    assert _api_post(url, TOKEN, "/api/gc", {})["database"]["metric_rows"] == 0


# ── logger ─────────────────────────────────────────────────────────────────────


def test_logger_refuses_overwrites() -> None:
    log = MLSweepLogger()
    log.log({"loss": 1.0}, step=0)
    log.log({"val_loss": 2.0}, step=0)  # another key at the same step is fine
    log.log({"loss": 0.9})              # auto-increments to 1
    assert log.step == 1
    with pytest.raises(ValueError, match="already logged at step 1"):
        log.log({"loss": 0.8}, step=1)
    with pytest.raises(ValueError, match="backwards"):
        log.log({"x": 1}, step=0)
    with pytest.raises(TypeError):
        log.log({"x": 1}, step=2.5)
    log.log({"loss": 0.7}, step=5.0)    # whole floats and numpy ints are steps
    np = pytest.importorskip("numpy")
    log.log({"loss": 0.6}, step=np.int64(6))
    assert log.step == 6 and isinstance(log.step, int)
