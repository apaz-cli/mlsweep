"""Tests for the mlsweep lifecycle commands (ls/logs/cancel/retry/resume/stop/
pause/unpause) and the result-ranking logic (best/fetch leaderboard).

Pure functions are unit-tested; the HTTP-facing commands are exercised against a
real manager subprocess via the ``manager_server`` fixture.
"""

import json
import threading
import time

import pytest

from conftest import _api_get, _api_post, _api_request

from mlsweep import ctl
from mlsweep import run_sweep
from mlsweep.cli import _status_cmd

_TOKEN = "test-token"


def _api_put(url, token, path, data=None):
    return _api_request(url, token, "PUT", path, data)


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


def test_rank_leaderboard_minimize():
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
    rows = run_sweep.rank_leaderboard(jobs, metrics, "loss", "minimize")
    assert [r["run_id"] for r in rows] == ["b", "d", "a", "c"]
    assert rows[0]["value"] == 1.0
    assert rows[0]["final"] == 1.0
    assert rows[0]["combo"] == {"lr": 0.001}
    assert rows[3]["status"] == "failed"
    assert rows[3]["value"] is None


def test_rank_leaderboard_maximize():
    jobs = [
        _job("a", "done", {"lr": 0.01}),
        _job("b", "done", {"lr": 0.001}),
    ]
    metrics = {
        "a": [{"acc": 0.5}, {"acc": 0.9}],
        "b": [{"acc": 0.7}],
    }
    rows = run_sweep.rank_leaderboard(jobs, metrics, "acc", "maximize")
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert rows[0]["value"] == 0.9
    assert rows[0]["final"] == 0.9


def test_rank_leaderboard_handles_missing_and_non_numeric():
    jobs = [
        _job("a", "done", {}),          # no metrics returned
        _job("b", "done", {}),          # metrics with no numeric target
        _job("c", "pending", {}),
        _job("d", "done", {}),          # has a value
    ]
    metrics = {"a": None, "b": [{"loss": "nan"}], "d": [{"loss": 0.5}]}
    rows = run_sweep.rank_leaderboard(jobs, metrics, "loss", "minimize")
    ids = [r["run_id"] for r in rows]
    assert ids[0] == "d"                # only valued run first
    assert set(ids[1:]) == {"a", "b", "c"}
    assert all(r["value"] is None for r in rows[1:])


def test_rank_leaderboard_combo_is_string():
    jobs = [_job("a", "done", {"z": 8})]
    rows = run_sweep.rank_leaderboard(jobs, {"a": [{"loss": 1.0}]}, "loss", "minimize")
    assert rows[0]["combo"] == {"z": 8}


def test_rank_leaderboard_skips_nonfinite():
    jobs = [
        _job("nan_run", "done", {}),
        _job("inf_run", "done", {}),
        _job("good", "done", {}),
    ]
    metrics = {
        "nan_run": [{"loss": float("nan")}],
        "inf_run": [{"loss": float("inf")}],
        "good": [{"loss": 1.0}],
    }
    rows = run_sweep.rank_leaderboard(jobs, metrics, "loss", "minimize")
    assert rows[0]["run_id"] == "good"
    assert rows[0]["value"] == 1.0
    assert {r["run_id"] for r in rows[1:]} == {"nan_run", "inf_run"}
    assert all(r["value"] is None for r in rows[1:])


def _finish_later(url, eid, rid, delay):
    """Mark a job done from another thread after *delay* seconds."""
    t = threading.Timer(delay, _set_status, (url, eid, rid, "done"))
    t.start()
    return t


def test_wait_until_settled(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    timer = _finish_later(url, "e", "r1", 1.5)
    start = time.monotonic()
    assert run_sweep._wait_until_settled(url, server.token, "e", interval=1) is False
    assert time.monotonic() - start >= 1.4
    timer.join()
    assert _api_get(url, _TOKEN, "/api/jobs/r1?experiment_id=e")["status"] == "done"


def test_wait_until_settled_reports_failure(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "failed", exit_code=1)
    assert run_sweep._wait_until_settled(url, server.token, "e", interval=1) is True


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


def test_best_wait_waits_for_active_jobs(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    timer = _finish_later(url, "e", "r1", 1.5)
    ctl.best_cmd(_base(url, server) + ["--experiment", "e", "--wait", "--wait-interval", "1", "--json"])
    timer.join()
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["status"] == "done"


# ── wait ────────────────────────────────────────────────────────────────────────


def test_wait_done_exits_zero(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    timer = _finish_later(url, "e", "r1", 1.0)
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--until", "done", "--interval", "0.25"])
    timer.join()
    assert exc.value.code == ctl.WAIT_EXIT_SETTLED


def test_wait_done_with_failure_exits_one(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "failed", exit_code=1)
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--until", "done", "--interval", "0.25"])
    assert exc.value.code == ctl.WAIT_EXIT_FAILURE


def test_wait_any_failure_returns_immediately(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _mk_job(url, "e", "r2")  # still pending
    _set_status(url, "e", "r1", "failed", exit_code=1)
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--until", "any-failure", "--interval", "0.25"])
    assert exc.value.code == ctl.WAIT_EXIT_FAILURE


def test_wait_timeout_exits_two(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")  # never finishes without a worker
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--timeout", "0.2", "--interval", "0.05"])
    assert exc.value.code == ctl.WAIT_EXIT_TIMEOUT


def test_wait_stalled_exits_three(monkeypatch, manager_server):
    server, url = manager_server
    jobs = [{"run_id": "r1", "status": "running", "stall_seconds": 120.0, "stalled": False}]
    monkeypatch.setattr(ctl, "manager_list_experiment_jobs", lambda *a, **k: jobs)
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--until", "stalled", "--stalled-after", "60"])
    assert exc.value.code == ctl.WAIT_EXIT_STALLED


def test_wait_stalled_ignores_fresh_runs(monkeypatch, manager_server):
    server, url = manager_server
    jobs = [{"run_id": "r1", "status": "running", "stall_seconds": 5.0, "stalled": False}]
    monkeypatch.setattr(ctl, "manager_list_experiment_jobs", lambda *a, **k: jobs)
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--until", "stalled",
                                           "--stalled-after", "900", "--timeout", "0.1",
                                           "--interval", "0.05"])
    assert exc.value.code == ctl.WAIT_EXIT_TIMEOUT


def test_resolve_ranking_prefers_experiment_then_defaults(manager_server):
    server, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments",
              {"experiment_id": "e_rank", "metric": "val_acc", "goal": "maximize"})
    assert run_sweep.resolve_ranking(url, server.token, "e_rank") == ("val_acc", "maximize")
    assert run_sweep.resolve_ranking(url, server.token, "e_rank", "loss") == ("loss", "maximize")
    assert run_sweep.resolve_ranking(url, server.token, "e_rank", None, "minimize") == ("val_acc", "minimize")

    _mk_exp(url, "e_plain")
    assert run_sweep.resolve_ranking(url, server.token, "e_plain") == ("loss", "minimize")


def test_fetch_json_uses_experiment_metric_goal(manager_server, capsys):
    server, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments",
              {"experiment_id": "e_rank", "metric": "val_acc", "goal": "maximize"})
    _mk_job(url, "e_rank", "r1")
    _set_status(url, "e_rank", "r1", "done")

    run_sweep._fetch_cmd(_base(url, server) + ["--experiment", "e_rank", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == "val_acc"
    assert out["goal"] == "maximize"

    # Explicit flags still win over the experiment's stored metric/goal.
    run_sweep._fetch_cmd(_base(url, server) + ["--experiment", "e_rank",
                                               "--metric", "loss", "--goal", "minimize", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == "loss"
    assert out["goal"] == "minimize"


def test_best_wait_exits_nonzero_on_failure(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "failed", exit_code=1)
    with pytest.raises(SystemExit) as exc:
        ctl.best_cmd(_base(url, server) + ["--experiment", "e", "--wait",
                                           "--wait-interval", "1", "--json"])
    assert exc.value.code == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["run_id"] == "r1"


def test_fetch_wait_exits_nonzero_on_failure(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "failed", exit_code=1)
    with pytest.raises(SystemExit) as exc:
        run_sweep._fetch_cmd(_base(url, server) + ["--experiment", "e", "--wait",
                                                   "--wait-interval", "1", "--json"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["runs"][0]["status"] == "failed"


class _FakeWS:
    """Stand-in for run_sweep._WebSocket that replays a fixed event list."""

    def __init__(self, url, token, timeout=10.0):
        self.url = url
        self._events = [
            {"type": "job_started", "run_id": "r1", "worker_id": "w"},
            {"type": "job_done", "run_id": "r1", "success": True, "elapsed": 1.0},
            {"type": "experiment_done", "experiment_id": "e"},
        ]

    def connect(self):
        pass

    def iter_events(self):
        yield from self._events

    def close(self):
        pass


def test_watch_events_machine_readable(monkeypatch, capsys):
    monkeypatch.setattr(run_sweep, "_WebSocket", _FakeWS)
    run_sweep._watch_cmd(["e", "--events", "--manager", "http://x", "--token", "t"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert [e["type"] for e in events] == ["job_started", "job_done", "experiment_done"]


def test_watch_json_alias_and_failure_exit(monkeypatch, capsys):
    class _FailWS(_FakeWS):
        def __init__(self, url, token, timeout=10.0):
            super().__init__(url, token, timeout)
            self._events = [
                {"type": "job_done", "run_id": "r1", "success": False, "elapsed": 1.0},
                {"type": "experiment_done", "experiment_id": "e"},
            ]

    monkeypatch.setattr(run_sweep, "_WebSocket", _FailWS)
    with pytest.raises(SystemExit) as exc:
        run_sweep._watch_cmd(["e", "--json", "--manager", "http://x", "--token", "t"])
    assert exc.value.code == 1
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert events[-1]["type"] == "experiment_done"


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
