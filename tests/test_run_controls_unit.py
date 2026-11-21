"""MAX_RUNTIME, PRUNE and METRIC_CHECK: parsing, the pruning rule, and pruned Bayesian trials."""

import pytest

from mlsweep._manager_controls import PruneTracker, metric_value, missing_metric_reason
from mlsweep._sweep import (
    build_controls,
    format_duration,
    parse_duration,
    prune_rungs,
    validate_controls,
)


@pytest.mark.parametrize("raw,seconds", [
    (90, 90.0), (1.5, 1.5), ("45s", 45.0), ("90m", 5400.0), ("2h", 7200.0), ("1h30m", 5400.0),
    ("1d 2h", 93600.0), ("0.5h", 1800.0),
])
def test_parse_duration(raw, seconds):
    assert parse_duration(raw) == seconds


@pytest.mark.parametrize("raw", ["soon", "2x", "", "1h 5", 0, -3, True, None, "0s"])
def test_parse_duration_rejects(raw):
    with pytest.raises(ValueError):
        parse_duration(raw)


def test_format_duration():
    assert [format_duration(s) for s in (45, 5400, 7200, 93600, 0)] == ["45s", "1h30m", "2h", "1d2h", "0s"]


def test_controls_defaults_and_inheritance():
    assert build_controls() is None
    # METRIC_CHECK is on by default whenever a metric is declared.
    assert build_controls(metric="m") == {"metric_check": {"metric": "m", "runs": 2}}
    assert build_controls(metric="m", metric_check=False) is None
    assert build_controls(metric="m", metric_check=5)["metric_check"]["runs"] == 5
    c = build_controls(max_runtime="2h", prune={"min_steps": 100}, metric="val", goal="maximize")
    assert c["max_runtime"] == 7200.0
    assert c["prune"] == {"metric": "val", "goal": "maximize", "strategy": "median", "min_steps": 100,
                          "every": 100, "n_warmup": 5}
    asha = build_controls(prune={"metric": "l", "goal": "minimize", "strategy": "asha", "min_steps": 10})
    assert asha["prune"]["reduction_factor"] == 3 and "every" not in asha["prune"]


@pytest.mark.parametrize("prune,match", [
    ({"min_steps": 10}, "needs a metric"),
    ({"metric": "l", "min_steps": 10}, "goal"),
    ({"metric": "l", "goal": "minimize"}, "min_steps"),
    ({"metric": "l", "goal": "minimize", "min_steps": 10, "strategy": "hyperband"}, "strategy"),
    ({"metric": "l", "goal": "minimize", "min_steps": 10, "reduction_factor": 3}, "for strategy 'asha'"),
    ({"metric": "l", "goal": "minimize", "min_steps": 10, "strategy": "asha", "every": 5}, "for strategy 'median'"),
    ({"metric": "l", "goal": "minimize", "min_steps": 10, "n_warmup": 0}, "n_warmup"),
    ({"metric": "l", "goal": "minimize", "min_steps": 10, "rungs": 3}, "unknown key"),
])
def test_bad_prune_specs(prune, match):
    with pytest.raises(ValueError, match=match):
        build_controls(prune=prune)


def test_validate_controls_round_trips_and_rejects_junk():
    c = build_controls(max_runtime=60, prune={"min_steps": 4, "strategy": "asha"}, metric="m", goal="minimize")
    assert validate_controls(c) == c
    assert validate_controls(None) is None
    with pytest.raises(ValueError):
        validate_controls({"max_runtime": 60, "note": "x"})
    with pytest.raises(ValueError):
        validate_controls({"metric_check": {"metric": "m", "runs": 0}})


def test_rungs():
    rungs = prune_rungs({"strategy": "asha", "min_steps": 10, "reduction_factor": 3})
    assert [next(rungs) for _ in range(4)] == [10, 30, 90, 270]
    rungs = prune_rungs({"strategy": "median", "min_steps": 10, "every": 5})
    assert [next(rungs) for _ in range(3)] == [10, 15, 20]


def _tracker(strategy="median", **kw):
    spec = {"metric": "loss", "goal": "minimize", "strategy": strategy, "min_steps": 10, "n_warmup": 2, **kw}
    spec.setdefault("every", 10) if strategy == "median" else spec.setdefault("reduction_factor", 2)
    return PruneTracker(spec, loaded=True)


def test_median_prunes_a_run_worse_than_the_median_once_warmed_up():
    t = _tracker()
    assert t.observe("a", 1, 10, 1.0) is None  # nobody to compare with
    assert t.observe("b", 1, 10, 2.0) is None  # one other run: still warming up
    assert t.observe("c", 1, 10, 1.4) is None  # median of 1.0 and 2.0 is 1.5
    why = t.observe("d", 1, 12, 1.6)           # median of 1.0, 2.0, 1.4 is 1.4
    assert why and "pruned at step 10" in why and "median 1.4" in why


def test_a_rung_value_is_the_best_so_far_and_steps_between_rungs_do_not_count():
    t = _tracker()
    for rid, v in (("a", 1.0), ("b", 1.0)):
        t.observe(rid, 1, 10, v)
    assert t.observe("c", 1, 5, 0.5) is None  # before the first rung
    assert t.observe("c", 1, 10, 9.0) is None  # best so far is still 0.5
    assert t.values[10]["c"] == 0.5


def test_maximize_flips_the_comparison():
    t = PruneTracker({"metric": "acc", "goal": "maximize", "strategy": "median", "min_steps": 1, "every": 1,
                      "n_warmup": 2}, loaded=True)
    t.observe("a", 1, 1, 0.9)
    t.observe("b", 1, 1, 0.8)
    assert t.observe("c", 1, 1, 0.95) is None
    assert t.observe("d", 1, 1, 0.1) is not None


def test_asha_keeps_the_top_fraction():
    t = _tracker("asha")  # reduction_factor 2: keep the best half
    for rid, v in (("a", 1.0), ("b", 2.0)):
        t.observe(rid, 1, 10, v)
    # 3 runs at the rung keep the best 3 // 2 = 1, which is a's 1.0.
    assert "outside the best 1 of 3" in t.observe("c", 1, 10, 1.5)
    assert t.observe("d", 1, 10, 0.5) is None  # now the best of 4 (keep 2)
    # Rungs grow by the factor: the next one is at step 20.
    assert t.observe("d", 1, 19, 0.4) is None and 20 not in t.values


def test_a_pruned_run_is_not_judged_again_and_a_new_attempt_starts_over():
    t = _tracker()
    for rid, v in (("a", 1.0), ("b", 1.0)):
        t.observe(rid, 1, 10, v)
    assert t.observe("c", 1, 10, 5.0) is not None
    assert t.observe("c", 1, 20, 5.0) is None  # already stopped
    assert t.observe("c", 2, 10, 0.1) is None  # a retry is a fresh run
    assert t.values[10]["c"] == 0.1


def test_judge_false_records_without_stopping():
    t = _tracker()
    for rid, v in (("a", 1.0), ("b", 1.0)):
        t.observe(rid, 1, 10, v)
    assert t.observe("c", -1, 10, 5.0, judge=False) is None
    assert t.values[10]["c"] == 5.0


def test_metric_value_and_missing_metric_reason():
    assert metric_value({"m": 1}, "m") == 1.0
    assert metric_value({"m": float("nan")}, "m") is None
    assert metric_value({"m": True}, "m") is None
    assert metric_value({}, "m") is None
    why = missing_metric_reason("val_loss", {"step", "val/loss", "train_loss"})
    assert why.startswith("missing metric 'val_loss'") and "did you mean 'val/loss'" in why
    assert "no metrics at all" in missing_metric_reason("m", {"step"})


def test_bayes_learns_from_pruned_trials():
    pytest.importorskip("optuna")
    from mlsweep._bayes import PastTrial, suggest
    from mlsweep._sweep import options_to_wire, validate_optimize_spec

    opts = {".x": {"distribution": "uniform", "min": 0.0, "max": 1.0, "flags": "--x"}}
    spec = validate_optimize_spec({"metric": "m", "goal": "minimize", "budget": 10, "n_initial": 2,
                                   "sweep_name": "s", "options": options_to_wire(opts),
                                   "job": {"command": ["x"]}})
    history = [PastTrial({"x": 0.1}, "complete", 0.1), PastTrial({"x": 0.9}, "pruned", 0.9, 40),
               PastTrial({"x": 0.5}, "pruned", None, 0)]
    out = suggest(spec, history, 2, 4)
    assert [v["name"] for v in out] == ["s_bayes_0004", "s_bayes_0005"]
