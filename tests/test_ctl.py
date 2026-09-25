"""Tests for the mlsweep lifecycle commands (ls/logs/cancel/retry/resume/stop/
pause/unpause) and the result-ranking logic (best/fetch leaderboard).

Pure functions are unit-tested; the HTTP-facing commands are exercised against a
real manager subprocess via the ``manager_server`` fixture.
"""

import json
import urllib.request

import pytest

from conftest import _api_get, _api_post

from mlsweep import ctl
from mlsweep import run_sweep
from mlsweep.cli import _status_cmd

_TOKEN = "test-token"


def _api_put(url, token, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{url}{path}", data=body, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _mk_exp(url, eid):
    return _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": eid, "name": eid})


def _mk_job(url, eid, rid, combo=None):
    body = {"run_id": rid, "experiment_id": eid, "command": ["echo"]}
    if combo is not None:
        body["combo"] = combo
    return _api_post(url, _TOKEN, "/api/jobs", body)


def _set_status(url, eid, rid, status, exit_code=0):
    return _api_put(url, _TOKEN, f"/api/jobs/{rid}/status",
                    {"experiment_id": eid, "status": status, "exit_code": exit_code})


def _base(url, server):
    return ["--manager", url, "--token", server.token]


# ── Pure helpers ────────────────────────────────────────────────────────────────


def test_select_jobs():
    jobs = [
        {"run_id": "a", "status": "failed"},
        {"run_id": "b", "status": "done"},
        {"run_id": "c", "status": "pending"},
    ]
    assert [j["run_id"] for j in ctl._select_jobs(jobs, [], ["failed"])] == ["a"]
    assert [j["run_id"] for j in ctl._select_jobs(jobs, ["b"], [])] == ["b"]
    assert {j["run_id"] for j in ctl._select_jobs(jobs, ["b"], ["failed"])} == {"a", "b"}
    assert ctl._select_jobs(jobs, [], []) == []


def test_combo_str():
    assert run_sweep._combo_str({"lr": 0.001, "z": 4}) == "lr=0.001  z=4"
    assert run_sweep._combo_str(None) == ""
    assert run_sweep._combo_str("lr=1") == ""


def _job(run_id, status, combo, elapsed=1.0):
    return {"run_id": run_id, "status": status, "combo": json.dumps(combo),
            "elapsed": elapsed, "exit_code": 0}


def test_build_leaderboard_minimize(monkeypatch):
    jobs = [
        _job("a", "done", {"lr": 0.01}),
        _job("b", "done", {"lr": 0.001}),
        _job("c", "failed", {"lr": 0.1}),
        _job("d", "done", {"lr": 0.005}),
    ]
    metrics = {
        "a": [{"loss": 3.5}, {"loss": 3.0}],
        "b": [{"loss": 1.5}, {"loss": 1.0}],
        "d": [{"loss": 2.0}],
    }
    monkeypatch.setattr(run_sweep, "manager_get_job_metrics",
                        lambda m, t, e, rid: metrics.get(rid))
    rows = run_sweep.build_leaderboard("http://x", "t", "exp", "loss", "minimize", jobs=jobs)
    assert [r["run_id"] for r in rows] == ["b", "d", "a", "c"]
    assert rows[0]["value"] == 1.0
    assert rows[0]["final"] == 1.0
    assert rows[0]["combo"] == {"lr": 0.001}
    assert rows[3]["status"] == "failed"
    assert rows[3]["value"] is None


def test_build_leaderboard_maximize(monkeypatch):
    jobs = [
        _job("a", "done", {"lr": 0.01}),
        _job("b", "done", {"lr": 0.001}),
    ]
    metrics = {
        "a": [{"acc": 0.5}, {"acc": 0.9}],
        "b": [{"acc": 0.7}],
    }
    monkeypatch.setattr(run_sweep, "manager_get_job_metrics",
                        lambda m, t, e, rid: metrics.get(rid))
    rows = run_sweep.build_leaderboard("http://x", "t", "exp", "acc", "maximize", jobs=jobs)
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert rows[0]["value"] == 0.9
    assert rows[0]["final"] == 0.9


def test_build_leaderboard_handles_missing_and_non_numeric(monkeypatch):
    jobs = [
        _job("a", "done", {}),          # no metrics returned
        _job("b", "done", {}),          # metrics with no numeric target
        _job("c", "pending", {}),
        _job("d", "done", {}),          # has a value
    ]
    monkeypatch.setattr(run_sweep, "manager_get_job_metrics",
                        lambda m, t, e, rid: {"a": None, "b": [{"loss": "nan"}],
                                              "c": None, "d": [{"loss": 0.5}]}.get(rid))
    rows = run_sweep.build_leaderboard("http://x", "t", "exp", "loss", "minimize", jobs=jobs)
    ids = [r["run_id"] for r in rows]
    assert ids[0] == "d"                # only valued run first
    assert set(ids[1:]) == {"a", "b", "c"}
    assert all(r["value"] is None for r in rows[1:])


def test_build_leaderboard_combo_is_string(monkeypatch):
    jobs = [_job("a", "done", {"z": 8})]
    monkeypatch.setattr(run_sweep, "manager_get_job_metrics",
                        lambda m, t, e, rid: [{"loss": 1.0}])
    rows = run_sweep.build_leaderboard("http://x", "t", "exp", "loss", "minimize", jobs=jobs)
    assert rows[0]["combo"] == {"z": 8}


def test_build_leaderboard_skips_nonfinite(monkeypatch):
    jobs = [
        _job("nan_run", "done", {}),
        _job("inf_run", "done", {}),
        _job("good", "done", {}),
    ]
    monkeypatch.setattr(run_sweep, "manager_get_job_metrics",
                        lambda m, t, e, rid: {
                            "nan_run": [{"loss": float("nan")}],
                            "inf_run": [{"loss": float("inf")}],
                            "good": [{"loss": 1.0}],
                        }.get(rid))
    rows = run_sweep.build_leaderboard("http://x", "t", "exp", "loss", "minimize", jobs=jobs)
    assert rows[0]["run_id"] == "good"
    assert rows[0]["value"] == 1.0
    assert {r["run_id"] for r in rows[1:]} == {"nan_run", "inf_run"}
    assert all(r["value"] is None for r in rows[1:])


def test_wait_until_settled(monkeypatch):
    calls = {"n": 0}

    def fake_http(method, url, token, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return (200, {"job_counts": {"running": 2, "pending": 1}})
        return (200, {"job_counts": {"done": 5, "failed": 1}})

    monkeypatch.setattr(run_sweep, "_http_request", fake_http)
    run_sweep._wait_until_settled("http://x", "t", "exp", interval=0)
    assert calls["n"] == 3


def test_print_leaderboard(capsys):
    rows = [
        {"run_id": "best", "status": "done", "combo": {"lr": 0.001},
         "value": 1.0, "final": 1.0, "elapsed": 1.0, "exit_code": 0},
        {"run_id": "worst", "status": "done", "combo": None,
         "value": 2.0, "final": 2.0, "elapsed": 1.0, "exit_code": 0},
    ]
    run_sweep.print_leaderboard(rows, "loss", "minimize", top=10)
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out
    assert "best" in out
    assert "lr=0.001" in out


# ── Integration (real manager, no worker) ──────────────────────────────────────


def test_ls_lists_experiments_and_runs(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e_one")
    _mk_exp(url, "e_two")
    _mk_job(url, "e_one", "r1", combo={"lr": 0.001})
    _mk_job(url, "e_one", "r2")

    ctl.ls_cmd(_base(url, server))
    out = capsys.readouterr().out
    assert "e_one" in out and "e_two" in out

    ctl.ls_cmd(_base(url, server) + ["e_one"])
    out = capsys.readouterr().out
    assert "r1" in out and "r2" in out

    ctl.ls_cmd(_base(url, server) + ["e_one", "--json"])
    jobs = json.loads(capsys.readouterr().out)
    assert {j["run_id"] for j in jobs} == {"r1", "r2"}


def test_cancel_dry_run_and_real(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _mk_job(url, "e", "r2")

    ctl.cancel_cmd(_base(url, server) + ["e", "r1", "r2", "--dry-run"])
    out = capsys.readouterr().out
    assert "would cancel" in out

    ctl.cancel_cmd(_base(url, server) + ["e", "r1", "r2"])
    capsys.readouterr()
    jobs = _api_get(url, _TOKEN, "/api/experiments/e/jobs")
    assert all(j["status"] == "cancelled" for j in jobs)


def test_cancel_all_requires_yes(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")

    with pytest.raises(SystemExit):
        ctl.cancel_cmd(_base(url, server) + ["e", "--all"])
    assert "without --yes" in capsys.readouterr().out

    ctl.cancel_cmd(_base(url, server) + ["e", "--all", "--yes"])
    capsys.readouterr()
    jobs = _api_get(url, _TOKEN, "/api/experiments/e/jobs")
    assert jobs[0]["status"] == "cancelled"


def test_retry_failed_jobs(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _mk_job(url, "e", "r2")
    _set_status(url, "e", "r1", "failed", exit_code=1)
    _set_status(url, "e", "r2", "failed", exit_code=1)

    ctl.retry_cmd(_base(url, server) + ["e", "--failed"])
    capsys.readouterr()
    jobs = _api_get(url, _TOKEN, "/api/experiments/e/jobs")
    assert all(j["status"] == "pending" and j["retry_count"] == 1 for j in jobs)


def test_stop_requires_yes_then_aborts(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")

    with pytest.raises(SystemExit):
        ctl.stop_cmd(_base(url, server) + ["e"])
    capsys.readouterr()

    ctl.stop_cmd(_base(url, server) + ["e", "--yes"])
    capsys.readouterr()
    exp = _api_get(url, _TOKEN, "/api/experiments/e")
    assert exp["status"] == "aborted"


def test_pause_and_unpause(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")

    ctl.pause_cmd(_base(url, server) + ["e"])
    capsys.readouterr()
    assert _api_get(url, _TOKEN, "/api/experiments/e")["status"] == "paused"

    ctl.unpause_cmd(_base(url, server) + ["e"])
    capsys.readouterr()
    assert _api_get(url, _TOKEN, "/api/experiments/e")["status"] == "running"


def test_resume_requeues_failed(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r_ok")
    _mk_job(url, "e", "r_fail1")
    _mk_job(url, "e", "r_fail2")
    _set_status(url, "e", "r_ok", "done")
    _set_status(url, "e", "r_fail1", "failed", exit_code=1)
    _set_status(url, "e", "r_fail2", "cancelled")

    ctl.resume_cmd(_base(url, server) + ["e"])
    capsys.readouterr()
    jobs = {j["run_id"]: j for j in _api_get(url, _TOKEN, "/api/experiments/e/jobs")}
    assert jobs["r_fail1"]["status"] == "pending"
    assert jobs["r_fail2"]["status"] == "pending"
    assert jobs["r_ok"]["status"] == "done"


def test_best_and_fetch_json(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1", combo={"lr": 0.001})
    _set_status(url, "e", "r1", "done")

    ctl.best_cmd(_base(url, server) + ["--experiment", "e", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert isinstance(rows, list)
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["status"] == "done"
    assert "value" in rows[0]

    run_sweep._fetch_cmd(_base(url, server) + ["--experiment", "e", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["experiment_id"] == "e"
    assert out["metric"] == "loss"
    assert out["goal"] == "minimize"
    assert out["runs"][0]["run_id"] == "r1"


def test_best_human_output(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "done")

    ctl.best_cmd(_base(url, server) + ["--experiment", "e"])
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out
    assert "no completed runs with a metric value" in out  # no worker → no metrics


def test_best_wait_calls_settle(manager_server, capsys, monkeypatch):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "done")

    calls = []
    monkeypatch.setattr(ctl, "_wait_until_settled", lambda m, t, e, i: calls.append((e, i)))
    ctl.best_cmd(_base(url, server) + ["--experiment", "e", "--wait", "--wait-interval", "3", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert calls == [("e", 3)]
    assert rows[0]["run_id"] == "r1"


def test_logs_no_log_exits_nonzero(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    with pytest.raises(SystemExit):
        ctl.logs_cmd(_base(url, server) + ["r1", "--experiment", "e"])
    assert "No log found" in capsys.readouterr().out


def test_status_json(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _status_cmd(_base(url, server) + ["--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["manager_up"] is True
    assert data["token_present"] is True
    assert "local_version" in data
    assert "manager_version" in data
    assert set(data["disk"]) == {"total", "used", "free"}
    assert "e" in data["recent_experiments"]
    # health details are surfaced for machine consumers
    assert "workers_connected" in data
    assert "jobs_pending" in data
    assert "jobs_in_flight" in data


def test_health_includes_version(manager_server):
    _, url = manager_server
    from importlib.metadata import version
    health = _api_get(url, _TOKEN, "/api/health")
    assert health["version"] == version("mlsweep")


def test_apply_returns_failure_count():
    jobs = [{"run_id": "a"}, {"run_id": "b"}]

    def fn(m, t, r, e):
        return {"ok": True} if r == "a" else None

    assert ctl._apply(jobs, fn, "verb", "m", "t", "e", dry_run=False) == 1
    assert ctl._apply(jobs, fn, "verb", "m", "t", "e", dry_run=True) == 0


def test_retry_nonterminal_exits_nonzero(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")  # still pending → retry is rejected (409)
    with pytest.raises(SystemExit) as exc:
        ctl.retry_cmd(_base(url, server) + ["e", "r1"])
    assert exc.value.code == 1
    assert "FAIL" in capsys.readouterr().out
