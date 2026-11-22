"""The manager runs Bayesian sweeps by itself: it chooses each run, with no client attached."""

from __future__ import annotations

import json
import sys
import urllib.error

import pytest

pytest.importorskip("optuna")
pytestmark = pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")

from cluster_harness import TOKEN, Cluster, stays, wait_until  # noqa: E402
from mlsweep import ctl  # noqa: E402
from mlsweep._bayes import PastTrial, suggest  # noqa: E402
from mlsweep._sweep import options_from_wire, options_to_wire, validate_optimize_spec, validate_options  # noqa: E402

# Logs a metric equal to --x, after --sleep seconds.  --bs above 1 fails like an OOM.
OBJECTIVE = """
import argparse, sys, time
from mlsweep.logger import MLSweepLogger
p = argparse.ArgumentParser()
p.add_argument("--x", type=float, default=0.5)
p.add_argument("--bs", type=int, default=1)
p.add_argument("--sleep", type=float, default=0.0)
p.add_argument("--fail", action="store_true")
a = p.parse_args()
time.sleep(a.sleep)
if a.fail or a.bs > 1:
    sys.exit(1)
with MLSweepLogger() as logger:
    logger.log({"m": a.x}, step=1)
"""

X = {".x": {"distribution": "uniform", "min": 0.0, "max": 1.0, "flags": "--x"}}


def _spec(*args: str, budget: int = 4, options: dict | None = None, **fields) -> dict:
    return {
        "metric": "m", "goal": "minimize", "budget": budget, "n_initial": 2,
        "sweep_name": "s", "options": options_to_wire(options or X),
        "job": {"command": [sys.executable, "-c", OBJECTIVE, *args], "gpus_per_run": 0},
        **fields,
    }


@pytest.fixture
def cluster(tmp_path):
    c = Cluster(tmp_path, [{"devices": [0], "jobs": 0}])
    yield c
    c.close()


def _start(c: Cluster, spec: dict, **fields) -> str:
    return c.experiment("bo", optimize=spec, **fields)


def _view(c: Cluster, eid: str) -> dict:
    return c.get(f"/api/experiments/{eid}/optimize")


def _wait_status(c: Cluster, eid: str, status: str, timeout: float = 60) -> dict:
    wait_until(lambda: _view(c, eid)["status"] == status, timeout, f"{eid} to be {status}")
    return _view(c, eid)


# ── Running to the budget ─────────────────────────────────────────────────────


def test_a_sweep_runs_to_its_budget_with_nobody_attached(cluster):
    eid = _start(cluster, _spec(budget=5))
    # The first trials exist as soon as the sweep does, up to n_initial at once.
    assert len(cluster.jobs(eid)) == 2

    view = _wait_status(cluster, eid, "completed")
    jobs = cluster.jobs(eid).values()
    assert view["told"] == 5 and view["running"] == view["failed"] == 0
    assert [j["status"] for j in jobs] == ["done"] * 5
    assert sorted(j["run_id"] for j in jobs) == [f"s_bayes_{i:04d}" for i in range(1, 6)]
    assert view["best"]["value"] == min(float(c["x"]) for c in (_combo(j) for j in jobs))
    assert view["best"]["combo"]["x"] == view["best"]["value"]
    assert cluster.get(f"/api/experiments/{eid}")["metric"] == "m"


def _combo(job: dict) -> dict:
    return json.loads(job["combo"])


def test_a_run_that_exits_right_after_logging_keeps_its_metric(cluster):
    """Its logger's connection may still be waiting to be accepted when the process is gone."""
    eid = cluster.experiment("short")
    rids = [f"r{i}" for i in range(24)]
    cluster.post("/api/jobs/bulk", [
        {"run_id": r, "experiment_id": eid, "gpus_per_run": 0,
         "command": [sys.executable, "-c", OBJECTIVE, "--x", "0.5"]} for r in rids
    ])
    cluster.wait_statuses(eid, {"done": 24})
    assert not [r for r in rids
                if cluster.status_code("GET", f"/api/experiments/{eid}/jobs/{r}/metrics") != 200]


def test_max_concurrent_one_runs_trials_one_at_a_time(cluster):
    eid = _start(cluster, _spec("--sleep", "0.2", budget=3), max_concurrent=1)
    assert len(cluster.jobs(eid)) == 1
    _wait_status(cluster, eid, "completed")
    jobs = sorted(cluster.jobs(eid).values(), key=lambda j: j["start_time"])
    assert len(jobs) == 3
    for earlier, later in zip(jobs, jobs[1:]):
        assert earlier["finish_time"] <= later["dispatch_time"]


def test_singular_probes_count_as_one_trial(cluster):
    options = {**X, ".bs": {"values": [2, 1], "flags": "--bs", "singular": True}}
    eid = _start(cluster, _spec(budget=2, options=options))
    view = _wait_status(cluster, eid, "completed")
    assert view["told"] == 2 and view["failed"] == 0
    statuses = sorted(j["status"] for j in cluster.jobs(eid).values())
    assert statuses == ["done", "done", "xfailed", "xfailed"]


def test_a_small_discrete_space_completes_when_used_up(cluster):
    options = {".x": {"values": [0.25, 0.75], "flags": "--x"}}
    eid = _start(cluster, _spec(budget=10, options=options))
    view = _wait_status(cluster, eid, "completed", timeout=90)
    assert view["told"] == 2 and "used up" in view["note"]
    assert len(cluster.jobs(eid)) == 2


def test_the_sweep_survives_a_manager_crash(cluster):
    eid = _start(cluster, _spec("--sleep", "1", budget=5))
    wait_until(lambda: _view(cluster, eid)["told"] >= 2, 60, "two results")
    cluster.kill_manager()
    cluster.start_manager()
    view = _wait_status(cluster, eid, "completed")
    jobs = cluster.jobs(eid)
    assert view["told"] == 5 and len(jobs) == 5
    assert {j["status"] for j in jobs.values()} == {"done"}


# ── When runs fail ────────────────────────────────────────────────────────────


def test_a_broken_script_pauses_the_sweep_instead_of_running_forever(cluster):
    eid = _start(cluster, _spec("--fail", budget=50, max_failures=3))
    view = _wait_status(cluster, eid, "paused")
    assert view["failed"] >= 3 and "in a row" in view["note"]
    n = len(cluster.jobs(eid))
    stays(lambda: len(cluster.jobs(eid)) == n, 1.0, "no new trials while paused")

    # Unpausing forgives those failures and gives it another try.
    cluster.set_exp_status(eid, "running")
    wait_until(lambda: len(cluster.jobs(eid)) > n, 30, "new trials after unpause")
    _wait_status(cluster, eid, "paused")


def test_giving_up_cancels_queued_runs_so_wait_returns(tmp_path, capsys):
    """One GPU and four trials: three are still queued when the first failure pauses it."""
    c = Cluster(tmp_path, [{"devices": [0], "jobs": 1}])
    try:
        spec = _spec("--fail", budget=50, n_initial=4, max_failures=1)
        spec["job"]["gpus_per_run"] = 1
        eid = _start(c, spec)
        with pytest.raises(SystemExit) as exc:
            ctl.wait_cmd(_cli(c, eid) + ["--timeout", "60", "--interval", "0.1"])
        assert exc.value.code == ctl.WAIT_EXIT_FAILURE, capsys.readouterr().out
        assert _view(c, eid)["status"] == "paused"
        assert not {j["status"] for j in c.jobs(eid).values()} & {"pending", "dispatched", "running"}
    finally:
        c.close()


def test_a_metric_that_is_never_logged_counts_as_failure(cluster):
    eid = _start(cluster, _spec(budget=50, max_failures=2, metric="typo"))
    view = _wait_status(cluster, eid, "paused")
    assert view["told"] == 0 and "typo" in view["note"]
    assert {j["status"] for j in cluster.jobs(eid).values() if j["status"] != "pending"} <= {"done", "running"}


# ── Control ───────────────────────────────────────────────────────────────────


def test_a_cancelled_trial_is_replaced_and_does_not_use_budget(cluster):
    eid = _start(cluster, _spec("--sleep", "60", budget=3))
    first = sorted(cluster.jobs(eid))[0]
    cluster.cancel(eid, first)
    wait_until(lambda: len(cluster.jobs(eid)) == 3, 20, "a replacement trial")
    view = _view(cluster, eid)
    assert view["abandoned"] == 1 and view["running"] == 2 and view["told"] == 0
    cluster.set_exp_status(eid, "aborted")


def test_stop_ends_it_and_resume_continues(cluster, capsys):
    eid = _start(cluster, _spec("--sleep", "60", budget=3))
    cluster.set_exp_status(eid, "aborted")
    assert {j["status"] for j in cluster.jobs(eid).values()} == {"cancelled"}
    stays(lambda: len(cluster.jobs(eid)) == 2, 1.0, "no new trials after stop")

    ctl.resume_cmd(_cli(cluster, eid))
    wait_until(lambda: len(cluster.jobs(eid)) == 4, 20, "new trials after resume")
    assert _view(cluster, eid)["status"] == "running"
    cluster.set_exp_status(eid, "aborted")


def test_pause_holds_new_runs_and_unpause_finishes(cluster):
    eid = _start(cluster, _spec("--sleep", "0.3", budget=4))
    cluster.set_exp_status(eid, "paused")
    wait_until(lambda: _view(cluster, eid)["status"] == "paused", 5, "paused")
    wait_until(lambda: all(j["status"] in ("done", "pending") for j in cluster.jobs(eid).values()),
               20, "started runs to end")
    n = len(cluster.jobs(eid))
    stays(lambda: len(cluster.jobs(eid)) == n, 1.0, "no new trials while paused")
    cluster.set_exp_status(eid, "running")
    assert _wait_status(cluster, eid, "completed")["told"] == 4


def test_finish_stops_new_trials_and_a_bigger_budget_continues(cluster, capsys):
    eid = _start(cluster, _spec("--sleep", "0.5", budget=50))
    ctl.optimize_cmd(_cli(cluster, eid) + ["--finish"])
    view = _wait_status(cluster, eid, "completed")
    assert view["told"] == 2 and view["finishing"]
    assert len(cluster.jobs(eid)) == 2

    ctl.optimize_cmd(_cli(cluster, eid) + ["--budget", "3"])
    view = _wait_status(cluster, eid, "completed")
    assert view["told"] == 3 and view["budget"] == 3 and not view["finishing"]

    capsys.readouterr()
    ctl.optimize_cmd(_cli(cluster, eid) + ["--budget", "+1"])
    assert "of 4 trials have results" in capsys.readouterr().out
    assert _wait_status(cluster, eid, "completed")["told"] == 4


def test_lowering_the_budget_leaves_existing_runs_alone(cluster):
    eid = _start(cluster, _spec("--sleep", "0.5", budget=50))
    assert cluster.patch(f"/api/experiments/{eid}/optimize", {"budget": 1})["budget"] == 1
    assert _wait_status(cluster, eid, "completed")["told"] == 2


def test_raising_the_cap_starts_more_trials(cluster, capsys):
    eid = _start(cluster, _spec("--sleep", "60", budget=5), max_concurrent=1)
    assert len(cluster.jobs(eid)) == 1
    ctl.optimize_cmd(_cli(cluster, eid) + ["--max-concurrent", "3"])
    wait_until(lambda: len(cluster.jobs(eid)) == 3, 20, "two more trials")
    cluster.set_exp_status(eid, "aborted")


def test_retrying_a_run_replaces_its_result(cluster):
    eid = _start(cluster, _spec(budget=2))
    _wait_status(cluster, eid, "completed")
    rid = sorted(cluster.jobs(eid))[0]
    cluster.post(f"/api/jobs/{rid}/retry?experiment_id={eid}")
    view = _wait_status(cluster, eid, "completed")
    jobs = cluster.jobs(eid)
    assert view["told"] == 2 and len(jobs) == 2 and jobs[rid]["attempt"] == 2


def test_purging_a_result_gets_it_replaced(cluster):
    eid = _start(cluster, _spec(budget=2))
    _wait_status(cluster, eid, "completed")
    rid = sorted(cluster.jobs(eid))[0]
    cluster.post(f"/api/experiments/{eid}/jobs/purge", {"run_ids": [rid]})
    assert rid not in cluster.jobs(eid)
    assert cluster.patch(f"/api/experiments/{eid}/optimize", {"budget": 2})["told"] == 1
    assert _wait_status(cluster, eid, "completed")["told"] == 2


def test_a_deleted_sweep_does_not_disturb_the_next(cluster):
    gone = _start(cluster, _spec("--sleep", "60"))
    cluster.delete(f"/api/experiments/{gone}")
    other = cluster.experiment("next", optimize=_spec(budget=2))
    assert _wait_status(cluster, other, "completed")["told"] == 2


# ── What the API and CLI accept ───────────────────────────────────────────────


@pytest.mark.parametrize("change", [
    {"budget": 0},
    {"goal": "best"},
    {"metric": ""},
    {"options": {"x": {}}},
    {"sweep_name": None},
    {"job": {"command": []}},
    {"job": {"command": ["a"], "gpus_per_run": -1}},
    {"options": options_to_wire({".x": {"distribution": "uniform", "min": 0, "max": 1,
                                        "flags": "--x", "singular": True}})},
])
def test_a_bad_spec_is_refused(cluster, change):
    code = cluster.status_code("POST", "/api/experiments", {"campaign": "test", "experiment_id": "bad", "optimize": {**_spec(), **change}})
    assert code == 400
    assert cluster.status_code("GET", "/api/experiments/bad") == 404


def test_a_spec_cannot_be_added_to_an_existing_experiment_or_carry_jobs(cluster):
    eid = cluster.experiment("plain")
    body = {"experiment_id": eid, "campaign": "test", "optimize": _spec()}
    assert cluster.status_code("POST", "/api/experiments", body) == 409
    bulk = {"campaign": "test", "optimize": _spec(), "jobs": [cluster.job_body("newone", "r", 1)]}
    assert cluster.status_code("POST", "/api/experiments/newone/jobs/bulk", bulk) == 400


def test_optimize_endpoint_is_only_for_bayesian_sweeps(cluster):
    eid = cluster.experiment("plain")
    assert cluster.status_code("GET", f"/api/experiments/{eid}/optimize") == 404
    assert cluster.status_code("PATCH", f"/api/experiments/{eid}/optimize", {"budget": 2}) == 404
    with pytest.raises(urllib.error.HTTPError):
        cluster.get("/api/experiments/nope/optimize")


def test_patch_rejects_bad_values(cluster):
    eid = _start(cluster, _spec("--sleep", "60"))
    path = f"/api/experiments/{eid}/optimize"
    for body in ({"budget": 0}, {"budget": True}, {"finish": "yes"}, {"max_failures": 0}):
        assert cluster.status_code("PATCH", path, body) == 400, body
    cluster.set_exp_status(eid, "aborted")


def test_wait_exits_only_when_the_manager_completes_it(cluster, capsys):
    """A failed or cancelled run in a sweep with budget left doesn't end `wait`."""
    options = {**X, ".bs": {"values": [2, 1], "flags": "--bs", "singular": True}}
    eid = _start(cluster, _spec(budget=3, options=options), max_concurrent=1)
    with pytest.raises(SystemExit) as exc:
        ctl.wait_cmd(_cli(cluster, eid) + ["--timeout", "60", "--interval", "0.1"])
    assert exc.value.code == ctl.WAIT_EXIT_SETTLED, capsys.readouterr().out
    assert _view(cluster, eid)["told"] == 3


def _cli(c: Cluster, eid: str) -> list[str]:
    return [eid, "--manager", c.url, "--token", TOKEN, "--campaign", "test"]


# ── Choosing trials ───────────────────────────────────────────────────────────


def _checked(raw: dict) -> dict:
    return validate_optimize_spec(raw)


def test_suggest_numbers_runs_and_builds_their_flags():
    spec = _checked(_spec(extra_flags=["--extra"]))
    runs = suggest(spec, [], 3, next_number=7)
    assert [r["name"] for r in runs] == ["s_bayes_0007", "s_bayes_0008", "s_bayes_0009"]
    for r in runs:
        assert r["overrides"] == ["--extra", "--x", str(r["combo"]["x"])]


def test_suggest_expands_singular_probes_most_aggressive_first():
    options = {**X, ".bs": {"values": [64, 32], "flags": "--bs", "singular": True}}
    runs = suggest(_checked(_spec(options=options)), [], 2)
    assert [r["combo"]["bs"] for r in runs] == [64, 32, 64, 32]
    assert runs[0]["combo"]["x"] == runs[1]["combo"]["x"] != runs[2]["combo"]["x"]


def test_suggest_rebuilds_from_every_kind_of_dim():
    """Combos from earlier calls replay through the study, subdim branches and fixed dims included."""
    options = {
        ".fixed": {"flags": ["--fixed"]},
        ".n": {"distribution": "int_uniform", "min": 1, "max": 9, "flags": "--n"},
        ".lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-1, "flags": "--lr"},
        ".opt": {".adam": {"flags": ["--adam"]},
                 ".sgd": {"flags": ["--sgd"], ".m": {"values": [0.1, 0.9], "flags": "--m"}}},
    }
    spec = _checked(_spec(options=options, budget=30))
    history: list[PastTrial] = []
    for i in range(12):
        run = suggest(spec, history, 1, i + 1)[0]
        assert run["combo"]["fixed"] is None and "--fixed" in run["overrides"]
        outcome = ["complete", "failed", "abandoned", "running"][i % 4]
        history.append(PastTrial(run["combo"], outcome, 0.5 if outcome == "complete" else None))


def test_suggest_does_not_repeat_a_tried_combo_unless_it_was_abandoned():
    options = {".x": {"values": ["a", "b", "c"], "flags": "--x"}}
    spec = _checked(_spec(options=options))
    history = [PastTrial({"x": "a"}, "complete", 1.0), PastTrial({"x": "b"}, "failed"),
               PastTrial({"x": "c"}, "abandoned")]
    assert [r["combo"]["x"] for r in suggest(spec, history, 2)] == ["c"]
    history[2] = PastTrial({"x": "c"}, "running")
    assert suggest(spec, history, 1) == []


def test_suggest_spreads_trials_asked_together():
    """Without constant_liar, TPE would answer the same way to every ask in a batch."""
    spec = _checked(_spec(n_initial=2))
    history = [PastTrial({"x": i / 10}, "complete", i / 10) for i in range(10)]
    xs = [r["combo"]["x"] for r in suggest(spec, history, 4, 11)]
    assert len(set(xs)) == 4


def test_suggest_learns_from_results():
    """After enough results, trials concentrate where the metric is best."""
    spec = _checked(_spec(n_initial=4, budget=100))
    history = [PastTrial({"x": i / 40}, "complete", abs(i / 40 - 0.1)) for i in range(40)]
    xs = [r["combo"]["x"] for r in suggest(spec, history, 8, 41)]
    assert sum(x < 0.4 for x in xs) >= 6, xs


def test_options_survive_the_wire_with_flags_keyed_by_int():
    options = {".bs": {"flags": {64: ["--bs", "64"], True: ["--on"]}},
               ".o": {".a": {"flags": "--a", ".n": {"values": [1], "flags": "--n"}}}}
    decoded = options_from_wire(options_to_wire(options))
    assert decoded == options
    validate_options(decoded, method="bayes")
    assert decoded[".bs"]["_values"] == [64, True]


def test_options_to_wire_drops_synthesized_fields():
    options = {".x": {"values": [1, 2], "flags": "--x"}}
    before = options_to_wire(options)
    validate_options(options, method="bayes")
    assert options_to_wire(options) == before


@pytest.mark.parametrize("bad", [
    {".x": {"values": [(1, 2)], "flags": {(1, 2): ["--x"]}}},
    {".x": {"values": [float("nan")], "flags": "--x"}},
    {".x": {"values": [print], "flags": "--x"}},
])
def test_options_that_json_cannot_carry_are_refused(bad):
    with pytest.raises(ValueError, match="plain JSON"):
        options_to_wire(bad)
