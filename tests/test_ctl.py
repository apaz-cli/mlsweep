"""Tests for the mlsweep lifecycle commands (ls/logs/cancel/retry/rename/resume/
stop/pause/unpause) and the result-ranking logic (best/fetch leaderboard).

Pure functions are unit-tested; the HTTP-facing commands are exercised against a
real manager subprocess via the ``manager_server`` fixture.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time

import pytest

from conftest import _api_get, _api_post, _api_request, _api_status
from recording_server import RecordingServer

from mlsweep import ctl
from mlsweep import run_sweep
from mlsweep._colors import set_color
from mlsweep.cli import _status_cmd

_TOKEN = "test-token"


def _api_put(url, token, path, data=None):
    return _api_request(url, token, "PUT", path, data)


def _mk_exp(url, eid):
    return _api_post(url, _TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": eid, "name": eid})


def _mk_job(url, eid, rid, combo=None):
    body = {"run_id": rid, "experiment_id": eid, "command": ["echo"]}
    if combo is not None:
        body["combo"] = combo
    return _api_post(url, _TOKEN, "/api/jobs", body)


def _set_status(url, eid, rid, status, exit_code=0):
    return _api_put(url, _TOKEN, f"/api/jobs/{rid}/status",
                    {"experiment_id": eid, "status": status, "exit_code": exit_code})


def _base(url, server):
    return ["--manager", url, "--token", server.token, "--campaign", "test"]


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
        {"run_id": "best", "label": None, "status": "done", "combo": {"lr": 0.001},
         "value": 1.0, "final": 1.0, "elapsed": 1.0, "exit_code": 0},
        {"run_id": "worst", "label": None, "status": "done", "combo": None,
         "value": 2.0, "final": 2.0, "elapsed": 1.0, "exit_code": 0},
    ]
    run_sweep.print_leaderboard(rows, "loss", "minimize", top=10)
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out
    assert "best" in out
    assert "lr=0.001" in out

    rows[0]["label"] = "baseline"
    run_sweep.print_leaderboard(rows, "loss", "minimize", top=10)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "best" in ln and "lr=" in ln)
    assert "(baseline)" in line


def test_rank_leaderboard_carries_label():
    jobs = [{**_job("a", "done", {}), "label": "baseline"}, _job("b", "done", {})]
    rows = run_sweep.rank_leaderboard(jobs, {"a": [{"loss": 1.0}], "b": [{"loss": 2.0}]})
    assert [(r["run_id"], r["label"]) for r in rows] == [("a", "baseline"), ("b", None)]


def _ranked(run_id, combo, value):
    return {"run_id": run_id, "label": None, "status": "done", "combo": combo,
            "value": value, "final": value, "elapsed": 1.0, "exit_code": 0}


def test_grouped_leaderboard_picks_best_per_group(capsys):
    # rows are already best-first (minimize): n1 → c (0.05), then a (0.10)
    rows = [
        _ranked("c", {"est": "n1", "lr": 0.003}, 0.05),
        _ranked("a", {"est": "n1", "lr": 0.001}, 0.10),
        _ranked("b", {"est": "n2", "lr": 0.003}, 0.20),
    ]
    ctl._print_grouped_leaderboard(rows, "est", "loss", "minimize", top=10)
    out = capsys.readouterr().out
    assert "best per est" in out
    assert "est=n1" in out and "est=n2" in out
    assert out.index("0.050000") < out.index("0.200000")


def test_metric_table_grid(capsys):
    rows = [
        _ranked("c", {"est": "n1", "lr": 0.003}, 0.05),
        _ranked("a", {"est": "n1", "lr": 0.001}, 0.10),
        _ranked("b", {"est": "n2", "lr": 0.003}, 0.20),
    ]
    ctl._print_metric_table(rows, "est,lr", "loss", "minimize")
    out = capsys.readouterr().out
    assert "loss by est × lr" in out
    assert "0.05" in out and "0.1" in out and "0.2" in out


def _exp(eid, status="completed", **counts):
    c = {k: counts.get(k, 0) for k in ("done", "failed", "xfailed", "cancelled", "running", "pending", "dispatched")}
    c["total"] = sum(c.values())
    return {"experiment_id": eid, "status": status, "campaign": "test",
            "submit_time": "2026-01-01T00:00:00+00:00", "job_counts": c}


def _section_of(out):
    """Map each bullet's ID to the ``## `` heading it sits under."""
    heading, under = "", {}
    for line in out.splitlines():
        if line.startswith("## "):
            heading = line
        elif line.startswith("- "):
            under[line[2:].split(":")[0]] = heading
    return under


def test_ls_experiments_grouped_by_status_most_urgent_first(capsys):
    ctl._ls_experiments([
        _exp("e_done", done=3),
        _exp("e_allfail", failed=2),
        _exp("e_some", done=1, failed=1),
        _exp("e_run", "running", done=1, running=1, pending=2),
        _exp("e_paused", "paused", pending=4),
    ], "test", show_all=False)
    out = capsys.readouterr().out
    assert out.startswith("# Campaign test: 5 experiments\n")
    headings = [ln for ln in out.splitlines() if ln.startswith("## ")]
    # The manager's "completed" reads "done", the same word runs use.
    assert headings == ["## Running (1)", "## Paused (1)", "## Failed (1)", "## Done (2)"]
    under = _section_of(out)
    # Every run failed: listed as failed.  Some runs failed: still done.
    assert under["e_allfail"] == "## Failed (1)"
    assert under["e_some"] == "## Done (2)"
    assert "- e_run: 4 runs, 1 done, 1 running, 2 pending." in out
    assert "- e_some: 2 runs, 1 done, 1 failed." in out
    assert "0 " not in out  # zero counts are left out


def test_ls_running_experiment_with_nothing_active_says_so(capsys):
    ctl._ls_experiments([_exp("e", "running", done=1, cancelled=2)], "test", show_all=False)
    assert "- e: 3 runs, 1 done, 2 cancelled. Submitted" in (out := capsys.readouterr().out)
    assert "No runs are active." in out


def test_ls_truncates_only_settled_sections_unless_all(capsys):
    n = ctl._LS_LIMIT + 5
    exps = [_exp(f"c{i}", done=1) for i in range(n)] + [_exp(f"f{i}", failed=1) for i in range(n)]
    ctl._ls_experiments(exps, "test", show_all=False)
    out = capsys.readouterr().out
    assert f"## Done ({n}, newest {ctl._LS_LIMIT} shown)" in out
    assert "- 5 more done experiments not shown. Pass --all to list them." in out
    assert f"## Failed ({n})" in out and f"- f{n - 1}:" in out
    assert f"- c{n - 1}:" not in out
    ctl._ls_experiments(exps, "test", show_all=True)
    out = capsys.readouterr().out
    assert f"## Done ({n})" in out and f"- c{n - 1}:" in out and "not shown" not in out


def test_ls_runs_layout(capsys):
    jobs = [
        {"run_id": "r1", "status": "done", "combo": '{"lr": 0.01, "seed": 1}', "elapsed": 125.0,
         "attempt": 2},
        {"run_id": "r2", "status": "failed", "combo": '{"lr": 0.001, "seed": 1}', "elapsed": 13.0,
         "exit_code": 1, "attempt": 3, "worker_id": "w0", "label": "warmup"},
        {"run_id": "r3", "status": "pending", "combo": '{"lr": 0.1, "seed": 1}'},
    ]
    ctl._ls_runs("e", jobs, {"status": "running"}, "alpha", show_all=False)
    out = capsys.readouterr().out
    assert out.startswith("# Experiment e: running, 3 runs, 1 done, 1 failed, 1 pending\n")
    assert "Dims: lr (0.001, 0.01, 0.1), seed (1)\n" in out
    headings = [ln for ln in out.splitlines() if ln.startswith("## ")]
    assert headings == ["## Pending (1)", "## Failed (1)", "## Done (1)"]
    # Dims that never vary are left off the bullets.
    assert "- r1: lr=0.01. 2m05s.\n" in out
    assert "- r3: lr=0.1.\n" in out
    assert "- r2 (warmup): lr=0.001. Exit 1 after 13s, attempt 3 on w0.\n" in out
    assert "  Logs: mlsweep logs e r2 --campaign alpha --tail 50" in out


def test_ls_runs_heading_says_done_for_completed(capsys):
    jobs = [{"run_id": "r1", "status": "done", "combo": "{}", "elapsed": 5.0}]
    ctl._ls_runs("e", jobs, {"status": "completed"}, "test", show_all=False)
    assert capsys.readouterr().out.startswith("# Experiment e: done, 1 run, 1 done\n")


def test_ls_dims_summarizes_long_value_lists():
    assert ctl._dim_values([3, 1, 2]) == "1, 2, 3"
    assert ctl._dim_values([i / 1000 for i in range(1, 51)]) == "50 values from 0.001 to 0.05"
    assert ctl._dim_values([f"v{i}" for i in range(20)]) == "20 values"


def test_ls_color_only_with_flag(capsys):
    ctl._ls_experiments([_exp("e", failed=1)], "test", show_all=False)
    assert "\033[" not in capsys.readouterr().out
    set_color(True)
    try:
        ctl._ls_experiments([_exp("e", failed=1)], "test", show_all=False)
    finally:
        set_color(False)
    out = capsys.readouterr().out
    assert "\033[1m\033[31m## Failed (1)\033[0m" in out
    assert "\033[94me\033[0m" in out  # the experiment ID
    assert "\033[31m1 failed\033[0m" in out  # each count in its status color
    # Stripped of color, the text matches the plain output exactly.
    assert "- e: 1 run, 1 failed." in re.sub(r"\033\[[0-9;]*m", "", out)


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
    assert all(j["status"] == "pending" and j["retry_count"] == 0 for j in jobs)


def test_retry_is_not_capped_by_max_retries(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    for _ in range(4):  # past the default max_retries of 2
        _set_status(url, "e", "r1", "failed", exit_code=1)
        ctl.retry_cmd(_base(url, server) + ["e", "r1", "--dry-run"])
        assert "would retry r1" in capsys.readouterr().out
        ctl.retry_cmd(_base(url, server) + ["e", "r1"])
        capsys.readouterr()
        job = _api_get(url, _TOKEN, "/api/experiments/e/jobs")[0]
        assert (job["status"], job["retry_count"]) == ("pending", 0)


@pytest.mark.parametrize("held", ["paused", "aborted"])
def test_retry_on_held_experiment_says_to_unpause(manager_server, capsys, held):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "failed", exit_code=1)
    _api_put(url, _TOKEN, "/api/experiments/e/status", {"status": held})
    capsys.readouterr()

    ctl.retry_cmd(_base(url, server) + ["e", "--failed"])
    assert "mlsweep unpause e" in capsys.readouterr().out


def test_retry_on_running_experiment_has_no_unpause_note(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "failed", exit_code=1)
    capsys.readouterr()

    ctl.retry_cmd(_base(url, server) + ["e", "--failed"])
    assert "unpause" not in capsys.readouterr().out


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

    with pytest.raises(SystemExit) as exc:
        ctl.best_cmd(_base(url, server) + ["--experiment", "e"])
    assert exc.value.code == 1  # finished, but nothing logged the metric
    captured = capsys.readouterr()
    assert "LEADERBOARD" in captured.out
    assert "no completed runs with a metric value" in captured.out  # no worker → no metrics
    assert "no finished run logged 'loss'" in captured.err


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


def _no_experiment():
    raise AssertionError("the experiment should not be needed")


def test_wait_stalled_exits_three():
    jobs = [{"run_id": "r1", "status": "running", "stall_seconds": 120.0, "stalled": False}]
    verdict = ctl.wait_outcome("e", jobs, "stalled", 60, _no_experiment)
    assert verdict.code == ctl.WAIT_EXIT_STALLED and "r1" in verdict.message
    assert verdict.outcome == "stalled" and [j["run_id"] for j in verdict.runs] == ["r1"]


def test_wait_stalled_ignores_fresh_runs():
    jobs = [{"run_id": "r1", "status": "running", "stall_seconds": 5.0, "stalled": False}]
    assert ctl.wait_outcome("e", jobs, "stalled", 900, _no_experiment) is None
    # The manager's own flag counts when it sends no stall_seconds.
    jobs = [{"run_id": "r1", "status": "running", "stalled": True}]
    assert ctl.wait_outcome("e", jobs, "stalled", 900, _no_experiment).code == ctl.WAIT_EXIT_STALLED


def _verdict(outcome, code, runs=()):
    return ctl.WaitVerdict(code, outcome, outcome, [dict(r) for r in runs])


def test_combine_waits_for_every_experiment_to_settle():
    done = _verdict("settled", ctl.WAIT_EXIT_SETTLED)
    assert ctl.combine_wait_verdicts({"a": done, "b": None}, "done") is None
    both = ctl.combine_wait_verdicts({"a": done, "b": _verdict("cancelled", ctl.WAIT_EXIT_CANCELLED)}, "done")
    assert both.code == ctl.WAIT_EXIT_CANCELLED
    failed = _verdict("failure", ctl.WAIT_EXIT_FAILURE, [{"run_id": "r"}])
    worst = ctl.combine_wait_verdicts({"a": done, "b": failed}, "done")
    assert worst.code == ctl.WAIT_EXIT_FAILURE and worst.runs == [{"run_id": "r", "experiment_id": "b"}]


def test_combine_settled_with_failures_does_not_end_any_done():
    # Under any-done, an experiment that settled with failures keeps waiting for the rest.
    failed = _verdict("failure", ctl.WAIT_EXIT_FAILURE, [{"run_id": "r"}])
    assert ctl.combine_wait_verdicts({"a": failed, "b": None}, "any-done") is None
    # Under any-failure it is the answer.
    assert ctl.combine_wait_verdicts({"a": failed, "b": None}, "any-failure").code == ctl.WAIT_EXIT_FAILURE


def test_combine_reports_finished_runs_from_every_experiment():
    a = _verdict("run_finished", 0, [{"run_id": "r1", "status": "done"}])
    b = _verdict("run_finished", 0, [{"run_id": "r2", "status": "failed"}])
    v = ctl.combine_wait_verdicts({"a": a, "b": b, "c": None}, "any-done")
    assert v.outcome == "run_finished" and v.code == ctl.WAIT_EXIT_SETTLED
    assert [(j["experiment_id"], j["run_id"]) for j in v.runs] == [("a", "r1"), ("b", "r2")]
    assert "a/r1 (done)" in v.message and "b/r2 (failed)" in v.message
    # A pause needs a person, so it wins over a finished run.
    paused = _verdict("paused", ctl.WAIT_EXIT_FAILURE)
    assert ctl.combine_wait_verdicts({"a": a, "b": paused}, "any-done").outcome == "paused"


def test_campaign_candidates_are_active_or_recently_finished():
    def exp(eid, active=0, last=None, **kw):
        return {"experiment_id": eid, "status": "running" if active else "completed",
                "job_counts": {"running": active}, "last_finish_time": last, **kw}
    listing = [exp("live", active=2), exp("old", last="2026-01-01T00:00:00+00:00"),
               exp("fresh", last="2026-06-01T00:00:00+00:00"), exp("never"),
               exp("bayes", optimize='{"budget": 4}', status="running", job_counts={})]
    since = ctl._parse_since("2026-03-01T00:00:00+00:00")
    assert ctl.campaign_wait_candidates(listing, since, set(), True) == ["live", "fresh", "bayes"]
    # One already watched stays, so its last run's finish is not missed.
    assert "old" in ctl.campaign_wait_candidates(listing, since, {"old"}, False)
    # An older manager reports no finish times: check everything once.
    bare = [{k: v for k, v in e.items() if k != "last_finish_time"} for e in listing]
    assert len(ctl.campaign_wait_candidates(bare, since, set(), True)) == 5
    assert ctl.campaign_wait_candidates(bare, since, set(), False) == ["live", "bayes"]


def test_any_failure_with_fail_since_ignores_old_failures():
    jobs = [{"run_id": "r1", "status": "failed", "finish_time": "2026-01-01T00:00:00+00:00"},
            {"run_id": "r2", "status": "running"}]
    since = ctl._parse_since("2026-03-01T00:00:00+00:00")
    assert ctl.wait_outcome("e", jobs, "any-failure", 900, _no_experiment, fail_since=since) is None
    assert ctl.wait_outcome("e", jobs, "any-failure", 900, _no_experiment).code == ctl.WAIT_EXIT_FAILURE


def _json_tail(out):
    """The JSON value at the end of *out*; in-process, the text verdict is printed before it."""
    return json.loads(out[out.index("\n{") + 1:] if not out.startswith("{") else out)


def _sleeper(url, eid, rid, seconds):
    """A real run of *seconds* on the fixture's worker (no GPU needed)."""
    _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": rid, "experiment_id": eid, "gpus_per_run": 0,
        "command": [sys.executable, "-c", f"import time; time.sleep({seconds})"]})


def test_wait_any_done_over_two_experiments(manager_with_worker, capsys):
    server, url = manager_with_worker
    for eid, seconds in (("a", 120), ("b", 1)):
        _mk_exp(url, eid)
        _sleeper(url, eid, "r1", seconds)
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["a", "b", "--until", "any-done", "--json",
                                           "--interval", "0.2", "--timeout", "60"])
    out = _json_tail(capsys.readouterr().out)
    assert exc.value.code == ctl.WAIT_EXIT_SETTLED, out
    assert out["outcome"] == "run_finished" and out["experiment_ids"] == ["a", "b"]
    assert [(r["experiment_id"], r["run_id"], r["status"]) for r in out["runs"]] == [("b", "r1", "done")]
    # The cursor picks up where this call left off: nothing new yet, so it times out.
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["a", "b", "--until", "any-done", "--since", str(out["cursor"]),
                                           "--timeout", "0.5", "--interval", "0.1"])
    assert exc.value.code == ctl.WAIT_EXIT_TIMEOUT


def test_wait_with_no_ids_follows_the_campaign(manager_with_worker, capsys):
    server, url = manager_with_worker
    _mk_exp(url, "old")
    _api_post(url, _TOKEN, "/api/jobs", {"run_id": "r1", "experiment_id": "old", "gpus_per_run": 0,
                                         "command": [sys.executable, "-c", "raise SystemExit(1)"]})
    deadline = time.monotonic() + 60
    while _api_get(url, _TOKEN, "/api/jobs/r1?experiment_id=old")["status"] != "failed":
        assert time.monotonic() < deadline, "the old run never failed"
        time.sleep(0.1)
    _mk_exp(url, "live")
    _sleeper(url, "live", "r1", 120)

    def submit_later():
        _mk_exp(url, "later")
        _sleeper(url, "later", "r1", 0)

    timer = threading.Timer(1.0, submit_later)
    timer.start()
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["--until", "any-done", "--json", "--interval", "0.2", "--timeout", "60"])
    timer.join()
    out = _json_tail(capsys.readouterr().out)
    assert exc.value.code == ctl.WAIT_EXIT_SETTLED, out
    assert out["campaign"] == "test" and "old" not in out["experiment_ids"]
    assert [(r["experiment_id"], r["run_id"]) for r in out["runs"]] == [("later", "r1")]


_COUNTS_DOWN = """
import time
from mlsweep.logger import MLSweepLogger
with MLSweepLogger(total_steps=100) as logger:
    for step in range(1, 6):
        logger.log({"loss": 1.0 / step}, step=step)
        time.sleep(0.2)
    time.sleep(120)
"""


def test_a_running_runs_total_steps_reaches_the_jobs_listing(manager_with_worker):
    server, url = manager_with_worker
    _mk_exp(url, "eta")
    _api_post(url, _TOKEN, "/api/jobs", {"run_id": "r1", "experiment_id": "eta", "gpus_per_run": 0,
                                         "command": [sys.executable, "-c", _COUNTS_DOWN]})
    deadline = time.monotonic() + 60
    while True:
        [job] = _api_get(url, _TOKEN, "/api/experiments/eta/jobs")
        if job.get("last_step") == 5:
            break
        assert time.monotonic() < deadline, job
        time.sleep(0.1)
    assert job["total_steps"] == 100
    assert 0 < job["eta_seconds"] < 95 * 0.2 * 3  # 95 steps left at about 0.2 s each


_QUIET = """
import sys, time
from mlsweep.logger import MLSweepLogger
ticks = iter(range(0, 10**6, 11))  # every heartbeat() call is 11 s after the last
with MLSweepLogger(clock=lambda: float(next(ticks))) as logger:
    logger.log({"loss": 1.0}, step=1)
    while True:
        time.sleep(0.2)
        if sys.argv[1] == "beat":
            logger.heartbeat()
"""


def test_a_heartbeat_keeps_a_quiet_run_from_looking_stalled(manager_with_worker):
    server, url = manager_with_worker
    _mk_exp(url, "hb")
    for rid in ("beat", "quiet"):
        _api_post(url, _TOKEN, "/api/jobs", {"run_id": rid, "experiment_id": "hb", "gpus_per_run": 0,
                                             "command": [sys.executable, "-c", _QUIET, rid]})
    deadline = time.monotonic() + 60
    while True:
        jobs = {j["run_id"]: j for j in _api_get(url, _TOKEN, "/api/experiments/hb/jobs")}
        if all(j.get("last_step") == 1 for j in jobs.values()) and jobs["quiet"]["stall_seconds"] > 3:
            break
        assert time.monotonic() < deadline, jobs
        time.sleep(0.2)
    assert jobs["beat"]["stall_seconds"] < 1.5, jobs["beat"]


def test_wait_with_no_ids_and_nothing_active_settles(manager_server):
    server, url = manager_server
    _mk_exp(url, "old")
    _mk_job(url, "old", "r1")
    _set_status(url, "old", "r1", "done")
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["--interval", "0.05"])
    assert exc.value.code == ctl.WAIT_EXIT_SETTLED


def test_wait_started_needs_a_logged_step_from_every_run():
    running = {"status": "running", "last_step": 3}
    quiet = {"status": "running"}  # started, but no metric yet
    queued = {"status": "pending"}
    done = {"status": "done", "finish_time": "2026-01-01T00:00:00+00:00"}
    ok = ctl.wait_outcome("e", [{**running, "run_id": "a"}, {**done, "run_id": "b"}], "started", 900,
                          _no_experiment)
    assert ok.code == ctl.WAIT_EXIT_SETTLED and ok.outcome == "started"
    for other in (quiet, queued):
        jobs = [{**running, "run_id": "a"}, {**other, "run_id": "b"}]
        assert ctl.wait_outcome("e", jobs, "started", 900, lambda: {"status": "running"}) is None
    failed = [{**running, "run_id": "a"}, {"status": "failed", "run_id": "b"}]
    assert ctl.wait_outcome("e", failed, "started", 900, _no_experiment).code == ctl.WAIT_EXIT_FAILURE


def test_combine_started_over_several_experiments():
    started = _verdict("started", ctl.WAIT_EXIT_SETTLED)
    assert ctl.combine_wait_verdicts({"a": started, "b": None}, "started") is None
    both = ctl.combine_wait_verdicts({"a": started, "b": _verdict("settled", 0)}, "started")
    assert both.outcome == "started" and both.code == ctl.WAIT_EXIT_SETTLED
    failed = _verdict("failure", ctl.WAIT_EXIT_FAILURE, [{"run_id": "r"}])
    assert ctl.combine_wait_verdicts({"a": failed, "b": None}, "started").code == ctl.WAIT_EXIT_FAILURE


_STEPS_THEN_SLEEPS = """
import sys, time
from mlsweep.logger import MLSweepLogger
with MLSweepLogger() as logger:
    if sys.argv[1] == "fail":
        raise SystemExit(1)
    time.sleep(float(sys.argv[1]))
    logger.log({"loss": 1.0}, step=1)
    time.sleep(120)
"""


def test_wait_until_started_end_to_end(manager_with_worker):
    server, url = manager_with_worker
    _mk_exp(url, "st")
    for rid, arg in (("quick", "0"), ("slow", "1.5")):
        _api_post(url, _TOKEN, "/api/jobs", {"run_id": rid, "experiment_id": "st", "gpus_per_run": 0,
                                             "command": [sys.executable, "-c", _STEPS_THEN_SLEEPS, arg]})
    start = time.monotonic()
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["st", "--until", "started", "--interval", "0.2", "--timeout", "60"])
    assert exc.value.code == ctl.WAIT_EXIT_SETTLED
    assert time.monotonic() - start >= 1.0  # it waited for the slow one's first step
    assert {j["run_id"]: j.get("last_step") for j in _api_get(url, _TOKEN, "/api/experiments/st/jobs")} == {
        "quick": 1, "slow": 1}
    _mk_exp(url, "bad")
    _api_post(url, _TOKEN, "/api/jobs", {"run_id": "r", "experiment_id": "bad", "gpus_per_run": 0,
                                         "command": [sys.executable, "-c", _STEPS_THEN_SLEEPS, "fail"]})
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["bad", "--until", "started", "--interval", "0.2", "--timeout", "60"])
    assert exc.value.code == ctl.WAIT_EXIT_FAILURE


def test_finished_since_takes_a_duration_or_a_time():
    now = ctl._parse_since("2026-11-26T12:00:00+00:00")
    assert ctl.parse_since_or_ago("2h", now=now) == now - 7200
    assert ctl.parse_since_or_ago("2026-11-26T11:00:00+00:00", now=now) == now - 3600
    assert ctl.parse_since_or_ago(str(now - 60), now=now) == now - 60
    with pytest.raises(ValueError):
        ctl.parse_since_or_ago("yesterday", now=now)


def test_finished_since_keeps_runs_that_finished_after_it():
    jobs = [{"run_id": "old", "status": "done", "finish_time": "2026-11-26T09:00:00+00:00"},
            {"run_id": "new", "status": "failed", "finish_time": "2026-11-26T11:30:00+00:00"},
            {"run_id": "newer", "status": "done", "finish_time": "2026-11-26T11:45:00+00:00"},
            {"run_id": "busy", "status": "running", "finish_time": None}]
    since = ctl._parse_since("2026-11-26T10:00:00+00:00")
    assert [j["run_id"] for j in ctl.finished_since(jobs, since)] == ["new", "newer"]


def test_ls_finished_since_across_the_campaign(manager_with_worker, capsys):
    server, url = manager_with_worker
    for eid, code in (("x", 0), ("y", 1)):
        _mk_exp(url, eid)
        _api_post(url, _TOKEN, "/api/jobs", {"run_id": "r", "experiment_id": eid, "gpus_per_run": 0,
                                             "command": [sys.executable, "-c", f"raise SystemExit({code})"]})
    deadline = time.monotonic() + 60
    while {j["status"] for e in ("x", "y") for j in _api_get(url, _TOKEN, f"/api/experiments/{e}/jobs")} \
            != {"done", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    capsys.readouterr()
    ctl.ls_cmd(_base(url, server) + ["--finished-since", "1h", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert sorted((r["experiment_id"], r["status"]) for r in rows) == [("x", "done"), ("y", "failed")]
    ctl.ls_cmd(_base(url, server) + ["--finished-since", "1h"])
    out = capsys.readouterr().out
    assert out.index("## Failed (1)") < out.index("## Done (1)") and "y/r" in out and "Finished" in out
    ctl.ls_cmd(_base(url, server) + ["x", "--finished-since", "1h", "--json"])
    assert [r["experiment_id"] for r in json.loads(capsys.readouterr().out)] == ["x"]


def test_resolve_ranking_prefers_experiment_then_defaults(manager_server):
    server, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments",
              {"campaign": "test", "experiment_id": "e_rank", "metric": "val_acc", "goal": "maximize"})
    assert run_sweep.resolve_ranking(url, server.token, "e_rank") == ("val_acc", "maximize")
    assert run_sweep.resolve_ranking(url, server.token, "e_rank", "loss") == ("loss", "maximize")
    assert run_sweep.resolve_ranking(url, server.token, "e_rank", None, "minimize") == ("val_acc", "minimize")

    _mk_exp(url, "e_plain")
    assert run_sweep.resolve_ranking(url, server.token, "e_plain") == ("loss", "minimize")


def test_fetch_json_uses_experiment_metric_goal(manager_server, capsys):
    server, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments",
              {"campaign": "test", "experiment_id": "e_rank", "metric": "val_acc", "goal": "maximize"})
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
    """An event stream for ``_watch_cmd`` that replays a fixed event list."""

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


def test_watch_events_machine_readable(capsys):
    run_sweep._watch_cmd(["e", "--events", "--manager", "http://x", "--token", "t", "--campaign", "test"], connect=_FakeWS)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert [e["type"] for e in events] == ["job_started", "job_done", "experiment_done"]


def test_watch_json_alias_and_failure_exit(capsys):
    class _FailWS(_FakeWS):
        def __init__(self, url, token, timeout=10.0):
            super().__init__(url, token, timeout)
            self._events = [
                {"type": "job_done", "run_id": "r1", "success": False, "elapsed": 1.0},
                {"type": "experiment_done", "experiment_id": "e"},
            ]

    with pytest.raises(SystemExit) as exc:
        run_sweep._watch_cmd(["e", "--json", "--manager", "http://x", "--token", "t", "--campaign", "test"], connect=_FailWS)
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


def test_apply_reports_each_run():
    jobs = [{"run_id": "a"}, {"run_id": "b"}]

    campaigns = []

    def fn(m, t, r, e, campaign=None):
        campaigns.append(campaign)
        return {"ok": True} if r == "a" else None

    results = ctl._apply(jobs, fn, "verb", "m", "t", "e", dry_run=False)
    assert results == [{"run_id": "a", "ok": True}, {"run_id": "b", "ok": False}]
    assert ctl._failed(results) == 1
    dry = ctl._apply(jobs, fn, "verb", "m", "t", "e", dry_run=True)
    assert dry == [{"run_id": "a", "ok": None}, {"run_id": "b", "ok": None}] and ctl._failed(dry) == 0
    assert ctl._failed(ctl._apply(jobs, fn, "verb", "m", "t", "e", dry_run=False, campaign="c")) == 1
    assert campaigns == [None, None, "c", "c"]


def test_retry_nonterminal_exits_nonzero(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")  # still pending → retry is rejected (409)
    with pytest.raises(SystemExit) as exc:
        ctl.retry_cmd(_base(url, server) + ["e", "r1"])
    assert exc.value.code == 1
    assert "FAIL" in capsys.readouterr().out


# ── rename ─────────────────────────────────────────────────────────────────────


def test_rename_sets_and_clears_name(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1", combo={"lr": 0.001})
    _mk_job(url, "e", "r2")

    ctl.rename_cmd(_base(url, server) + ["e", "r1", "warmup ablation"])
    assert "renamed r1 to 'warmup ablation'" in capsys.readouterr().out
    labels = {j["run_id"]: j["label"] for j in _api_get(url, _TOKEN, "/api/experiments/e/jobs")}
    assert labels == {"r1": "warmup ablation", "r2": None}

    # ls shows the name beside the run ID; best --json carries it.
    ctl.ls_cmd(_base(url, server) + ["e"])
    assert "- r1 (warmup ablation)" in capsys.readouterr().out
    _set_status(url, "e", "r1", "done")
    ctl.best_cmd(_base(url, server) + ["--experiment", "e", "--json"])
    rows = {r["run_id"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["r1"]["label"] == "warmup ablation"

    ctl.rename_cmd(_base(url, server) + ["e", "r1", "--clear"])
    assert "cleared the name of r1" in capsys.readouterr().out
    assert _api_get(url, _TOKEN, "/api/jobs/r1?experiment_id=e")["label"] is None


def test_rename_sets_the_description_and_name_together_and_clears_it(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    ctl.rename_cmd(_base(url, server) + ["e", "--experiment-name", "warmup", "--description", " Does warmup help? ",
                                         "--json"])
    captured = capsys.readouterr()
    said = captured.out + captured.err
    assert "renamed experiment e to 'warmup'" in said and "set the description of e" in said
    assert json.loads(captured.out[captured.out.index("{"):]) == {
        "experiment_id": "e", "name": "warmup", "description": "Does warmup help?"}
    exp = _api_get(url, _TOKEN, "/api/experiments/e")
    assert (exp["name"], exp["note"]) == ("warmup", "Does warmup help?")

    ctl.ls_cmd(_base(url, server) + ["e"])
    assert "Description: Does warmup help?" in capsys.readouterr().out

    ctl.rename_cmd(_base(url, server) + ["e", "--clear-description"])
    assert "cleared the description of e" in capsys.readouterr().out
    assert _api_get(url, _TOKEN, "/api/experiments/e")["note"] is None


@pytest.mark.parametrize("args,err", [
    (["--description", "x", "--clear-description"], "not both"),
    (["--description", "  "], "must not be empty"),
    (["r1", "--description", "x"], "take no RUN"),
])
def test_rename_description_argument_errors(manager_server, capsys, args, err):
    server, url = manager_server
    with pytest.raises(SystemExit) as exc:
        ctl.rename_cmd(_base(url, server) + ["e", *args])
    assert exc.value.code == 2
    assert err in capsys.readouterr().err


def test_description_route_validates_and_clears(manager_server):
    _, url = manager_server
    _mk_exp(url, "e")
    assert _api_put(url, _TOKEN, "/api/experiments/e/description", {"description": "why"})["note"] == "why"
    assert _api_put(url, _TOKEN, "/api/experiments/e/description", {"description": ""})["note"] is None
    for bad in ({}, {"description": 3}):
        assert _api_status(url, _TOKEN, "PUT", "/api/experiments/e/description", bad)[0] == 400
    assert _api_status(url, _TOKEN, "PUT", "/api/experiments/missing/description", {"description": "x"})[0] == 404


@pytest.mark.parametrize("args", [["r1"], ["r1", "  "], ["r1", "x", "--clear"]])
def test_rename_needs_exactly_one_of_name_or_clear(manager_server, capsys, args):
    server, url = manager_server
    with pytest.raises(SystemExit) as exc:
        ctl.rename_cmd(_base(url, server) + ["e", *args])
    assert exc.value.code == 2
    assert "NAME or --clear" in capsys.readouterr().err


def test_rename_unknown_run_exits_nonzero(manager_server, capsys):
    server, url = manager_server
    _mk_exp(url, "e")
    with pytest.raises(SystemExit) as exc:
        ctl.rename_cmd(_base(url, server) + ["e", "missing", "x"])
    assert exc.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_rename_refuses_run_in_another_campaign(manager_server, capsys):
    server, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": "e", "campaign": "other"})
    _mk_job(url, "e", "r1")
    with pytest.raises(SystemExit) as exc:
        ctl.rename_cmd(_base(url, server) + ["e", "r1", "x", "--campaign", "test"])
    assert exc.value.code == 1
    assert _api_get(url, _TOKEN, "/api/jobs/r1?experiment_id=e")["label"] is None


def test_wait_unknown_experiment_is_not_success(manager_server):
    server, url = manager_server
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["does_not_exist", "--timeout", "1", "--interval", "0.05"])
    assert exc.value.code == ctl.WAIT_EXIT_UNKNOWN


def test_wait_unreachable_manager_is_not_a_run_failure():
    # Nothing listens on the discard port.
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(["e", "--manager", "http://127.0.0.1:9", "--token", "t", "--campaign", "test"])
    assert exc.value.code == ctl.WAIT_EXIT_UNKNOWN
    assert ctl.wait_outcome("e", None, "done", 900, _no_experiment).code == ctl.WAIT_EXIT_UNKNOWN


def test_wait_cancelled_runs_are_not_a_clean_finish(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")
    _set_status(url, "e", "r1", "cancelled")
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--interval", "0.05"])
    assert exc.value.code == ctl.WAIT_EXIT_CANCELLED


def test_wait_until_settled_exits_on_unknown_experiment(manager_server):
    server, url = manager_server
    with pytest.raises(SystemExit) as exc:
        run_sweep._wait_until_settled(url, server.token, "does_not_exist", interval=1)
    assert exc.value.code == 1


def test_leaderboard_names_the_runs_it_leaves_out(capsys):
    rows = [
        {"run_id": "a", "label": None, "status": "done", "combo": {}, "value": 1.0, "final": 2.0},
        {"run_id": "b", "label": None, "status": "failed", "combo": {}, "value": None, "final": None},
        {"run_id": "c", "label": None, "status": "done", "combo": {}, "value": None, "final": None},
    ]
    run_sweep.print_leaderboard(rows, "m")
    out = capsys.readouterr().out
    assert "1 failed" in out and "1 without this metric" in out and "best step" in out


def test_run_reads_manager_from_env_with_no_flag(tmp_path):
    code = ("import argparse\nfrom mlsweep import run_sweep\np = argparse.ArgumentParser()\n"
            "run_sweep._add_connection_args(p)\nprint(p.parse_args([]).manager)")
    env = {**os.environ, "MLSWEEP_MANAGER": "http://elsewhere:1"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "http://elsewhere:1"


def test_wait_does_not_call_an_unfinished_bayesian_budget_done():
    jobs = [{"run_id": "r1", "status": "done"}]
    running_bayes = {"status": "running", "optimize": '{"budget": 4}'}
    assert ctl.wait_outcome("e", jobs, "done", 900, lambda: running_bayes) is None
    finished = {"status": "completed", "optimize": '{"budget": 4}'}
    assert ctl.wait_outcome("e", jobs, "done", 900, lambda: finished).code == ctl.WAIT_EXIT_SETTLED


def test_wait_after_stop_does_not_wait_for_pending_runs_forever(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")  # pending: no worker
    ctl.stop_cmd(_base(url, server) + ["e", "--yes"])
    assert _api_get(url, _TOKEN, "/api/experiments/e/jobs")[0]["status"] == "cancelled"
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_base(url, server) + ["e", "--timeout", "5", "--interval", "0.05"])
    assert exc.value.code == ctl.WAIT_EXIT_CANCELLED


def test_fetch_wait_settles_after_stop(manager_server):
    server, url = manager_server
    _mk_exp(url, "e")
    _mk_job(url, "e", "r1")  # pending: no worker
    ctl.stop_cmd(_base(url, server) + ["e", "--yes"])
    assert run_sweep._wait_until_settled(url, server.token, "e", interval=1) is False


def test_workers_update_and_export_cancel_report_unknown_ids(manager_server, capsys):
    server, url = manager_server
    with pytest.raises(SystemExit):
        ctl.workers_cmd(["update", "nope"] + _base(url, server))
    with pytest.raises(SystemExit):
        ctl.export_cmd(["cancel", "nope"] + _base(url, server))
    out = capsys.readouterr().out
    assert "update worker" in out and "cancel export" in out


@pytest.mark.parametrize("argv, path, status", [
    (["workers", "update", "w0"], "/api/workers/w0/update", 202),
    (["export", "cancel", "x1"], "/api/exports/x1/cancel", 200),
])
def test_workers_update_and_export_cancel_post(capsys, argv, path, status):
    with RecordingServer(status, {"worker_id": "w0", "status": "updating"}) as server:
        cmd = ctl.workers_cmd if argv[0] == "workers" else ctl.export_cmd
        cmd(argv[1:] + ["--manager", server.url, "--token", "t"])
    assert server.calls == [("POST", path)]
    assert ("updating" if argv[0] == "workers" else "cancelling") in capsys.readouterr().out



def test_wait_until_any_done_reports_runs_finished_since_the_cursor():
    jobs = [
        {"run_id": "old", "status": "done", "finish_time": "2026-01-01T00:00:10+00:00"},
        {"run_id": "new", "status": "failed", "finish_time": "2026-01-01T00:00:30+00:00"},
        {"run_id": "busy", "status": "running", "finish_time": None},
    ]
    t20 = ctl._parse_since("2026-01-01T00:00:20+00:00")
    verdict = ctl.wait_outcome("e", jobs, "any-done", 900, _no_experiment, since=t20)
    assert verdict.code == ctl.WAIT_EXIT_SETTLED and verdict.outcome == "run_finished"
    assert [j["run_id"] for j in verdict.runs] == ["new"]
    out = verdict.to_json("e", jobs, t20)
    assert out["cursor"] == ctl._parse_since("2026-01-01T00:00:30+00:00")
    assert out["counts"] == {"done": 1, "failed": 1, "running": 1}
    # Nothing finished after the cursor: keep waiting.
    assert ctl.wait_outcome("e", jobs, "any-done", 900, _no_experiment, since=out["cursor"]) is None


def test_ls_shows_each_running_runs_step_and_average_step_time():
    job = {"run_id": "r1", "status": "running", "start_time": None, "last_step": 1200,
           "avg_step_seconds": 2.5, "combo": "{}"}
    [line] = ctl._run_bullet(job, set(), set(), "e", "test")
    assert "step 1200 at 2.5 s/step on average" in line
    fast = {**job, "avg_step_seconds": 0.05}
    assert "at 20 steps/s on average" in ctl._run_bullet(fast, set(), set(), "e", "test")[0]
    # Before the step has advanced there is no average yet.
    assert ctl._run_bullet({**job, "avg_step_seconds": None}, set(), set(), "e", "test")[0].endswith(
        "step 1200.")


def test_ls_shows_the_gpu_and_the_time_left():
    job = {"run_id": "r1", "status": "running", "start_time": None, "last_step": 500, "total_steps": 1500,
           "avg_step_seconds": 2.0, "eta_seconds": 2000.0, "combo": "{}", "worker_id": "w:7890",
           "dispatched_gpu_ids": "[3]"}
    [line] = ctl._run_bullet(job, set(), set(), "e", "test")
    assert "step 500 of 1500 at 2 s/step on average, about 33m20s left on w:7890 GPU 3." in line
    two = {**job, "dispatched_gpu_ids": "[0, 1]", "total_steps": None, "eta_seconds": None}
    [line] = ctl._run_bullet(two, set(), set(), "e", "test")
    assert line.endswith("step 500 at 2 s/step on average on w:7890 GPUs 0,1.")


def test_in_flight_runs_estimate_the_time_left():
    from mlsweep._manager_state import InFlightRun

    run = InFlightRun(experiment_id="e", run_id="r", job_key=1, attempt=1, primary="w")
    run.note_step(10, None, 100.0)
    run.note_step(20, None, 120.0)
    assert run.eta_seconds(120.0) is None  # the script never said where it ends
    run.note_step(30, None, 140.0, total_steps=100)
    assert run.eta_seconds(140.0) == 140.0  # 70 steps at 2 s
    assert run.eta_seconds(150.0) == 130.0  # still falling between steps
    assert run.eta_seconds(1e6) == 0.0


def test_status_says_when_each_gpu_frees_up():
    from mlsweep.cli import gpu_free_lines

    runs = {"1": [{"run_id": "a", "eta_seconds": 600.0}, {"run_id": "b", "eta_seconds": 7200.0}],
            "0": [{"run_id": "c", "eta_seconds": 60.0}, {"run_id": "d", "eta_seconds": None}],
            "2": [{"run_id": "e"}], "3": []}
    assert gpu_free_lines(runs) == [
        "GPU 0: 2 runs, next ends in about 1m00s (1 without an estimate)",
        "GPU 1: 2 runs, next ends in about 10m00s, free in about 2h00m",
        "GPU 2: 1 run, no estimate (the script sets total_steps on its MLSweepLogger)",
    ]


def test_in_flight_runs_average_their_step_time():
    from mlsweep._manager_state import InFlightRun

    run = InFlightRun(experiment_id="e", run_id="r", job_key=1, attempt=1, primary="w")
    assert run.avg_step_seconds is None
    run.note_step(10, None, 100.0)
    run.note_step(10, None, 101.0)  # same step: no change
    assert run.avg_step_seconds is None
    run.note_step(20, None, 120.0)
    run.note_step(30, None, 130.0)
    assert run.avg_step_seconds == 1.5  # 30 s over 20 steps
