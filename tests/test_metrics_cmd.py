"""Tests for ``mlsweep metrics``: a read-only, on-demand view of what runs logged."""

import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import _api_post, _wait_for_job
from mlsweep.ctl import metrics_cmd, parse_aggs, pivot_metrics, select_metrics, summarize_run

REPO_ROOT = Path(__file__).parent.parent

ROWS = [
    {"step": 1, "loss": 3.0, "val@1": 4.0, "val@4": 3.5},
    {"step": 2, "loss": 2.5},
    {"step": 3, "loss": 2.0, "val@1": 3.8, "val@4": 3.1, "val@16": 3.6},
]


def test_select_keeps_matching_keys_and_drops_empty_steps():
    out = select_metrics(ROWS, re.compile(r"^val@"))
    assert [r["step"] for r in out] == [1, 3]
    assert set(out[1]) == {"step", "val@1", "val@4", "val@16"}
    assert select_metrics(ROWS, None) == ROWS


def test_pivot_uses_latest_matching_step_at_or_before_requested():
    pat = re.compile(r"^val@(\d+)$")
    assert pivot_metrics(ROWS, pat, None) == (3, {"1": 3.8, "4": 3.1, "16": 3.6})
    assert pivot_metrics(ROWS, pat, 2) == (1, {"1": 4.0, "4": 3.5})   # step 2 has no val@ keys
    assert pivot_metrics(ROWS, pat, 0) == (None, {})


# A run at step 1250 that evaluates every 500 steps and logs a divergence flag.
RUN = [
    {"step": 1, "train_loss": 9.0, "diverged": 0, "tokens": 100},
    {"step": 500, "train_loss": 5.0, "val_loss": 5.2, "diverged": 1, "tokens": 50_000},
    {"step": 1000, "train_loss": 4.0, "val_loss": 4.1, "diverged": 0, "tokens": 100_000},
    {"step": 1250, "train_loss": 3.9, "diverged": 0, "tokens": 125_000},
]


def test_last_says_when_a_value_is_older_than_the_run():
    step, values, where = summarize_run(RUN, re.compile("loss"))
    assert step == 1250
    assert values == {"train_loss": 3.9, "val_loss": 4.1}
    assert where == {"train_loss": 1250, "val_loss": 1000}


def test_agg_per_key():
    aggs = parse_aggs(["final", "max:diverged", "min:^val"])
    _, values, where = summarize_run(RUN, re.compile("loss|diverged"), aggs=aggs)
    assert values == {"max(diverged)": 1, "train_loss": 3.9, "min(val_loss)": 4.1}
    assert where["max(diverged)"] == 500 and where["min(val_loss)"] == 1000
    _, means, _ = summarize_run(RUN, re.compile("^train"), aggs=parse_aggs(["mean"]))
    assert means == {"mean(train_loss)": pytest.approx((9.0 + 5.0 + 4.0 + 3.9) / 4)}
    with pytest.raises(ValueError):
        parse_aggs(["median"])


def test_values_at_chosen_steps_use_the_latest_at_or_before():
    _, values, where = summarize_run(RUN, re.compile("^val"), steps=[500, 1200, 10])
    assert values == {"val_loss@500": 5.2, "val_loss@1200": 4.1, "val_loss@10": None}
    assert where["val_loss@1200"] == 1000


def test_values_at_a_matched_x():
    # Equal tokens, not equal steps: the first step where tokens reach each value.
    _, values, _ = summarize_run(RUN, re.compile("^val"), x="tokens", at=[60_000, 1e9])
    assert values == {"step@tokens=60000": 1000, "val_loss@tokens=60000": 4.1,
                      "step@tokens=1e+09": None, "val_loss@tokens=1e+09": None}


_SCRIPT = """
from mlsweep.logger import MLSweepLogger
with MLSweepLogger() as lg:
    for step in (1, 2):
        lg.log({"loss": 3.0 - step, "val@1": 4.0 - step, "val@4": OFFSET - step}, step=step)
"""


@pytest.mark.parametrize("shape", ["--csv", "--json", "--last"])
def test_tail_outside_table_view_points_at_last(shape, capsys):
    with pytest.raises(SystemExit) as exc:
        metrics_cmd(["--experiment", "e", "--tail", "1", shape])
    assert exc.value.code == 2
    assert "--last" in capsys.readouterr().err


def test_metrics_command_end_to_end(manager_with_worker):
    server, url = manager_with_worker
    token = server.token
    _api_post(url, token, "/api/experiments", {"campaign": "test", "experiment_id": "metrics_cmd"})
    for run_id, offset in (("a", 3.5), ("b", 3.0)):
        _api_post(url, token, "/api/jobs", {
            "run_id": run_id, "experiment_id": "metrics_cmd", "gpus_per_run": 0,
            "command": [sys.executable, "-c", _SCRIPT.replace("OFFSET", str(offset))],
        })
    for run_id in ("a", "b"):
        job = _wait_for_job(url, token, run_id, "metrics_cmd", timeout=60)
        assert job and job["status"] == "done", job

    env = {**os.environ, "MLSWEEP_TOKEN": token}
    base = [sys.executable, "-c", "from mlsweep.cli import main; main()", "metrics",
            "--manager", url, "--experiment", "metrics_cmd", "--campaign", "test"]

    def run(*extra):
        p = subprocess.run(base + list(extra), capture_output=True, text=True, env=env, cwd=REPO_ROOT)
        assert p.returncode == 0, p.stderr
        return p.stdout

    table = run("a", "--keys", "^loss$")
    assert "== a" in table and table.strip().splitlines()[-1].split() == ["2", "1"]

    pivot = run("--keys", r"^val@(\d+)$", "--pivot").strip().splitlines()
    assert pivot[0].split() == ["x", "a", "@2", "b", "@2"]
    assert pivot[-1].split() == ["4", "1.5", "1"]            # val@4 at step 2: 3.5-2, 3.0-2

    rows = list(csv.DictReader(io.StringIO(run("b", "--keys", "^loss$", "--csv"))))
    assert [(r["run"], r["step"], r["key"], float(r["value"])) for r in rows] == \
        [("b", "1", "loss", 2.0), ("b", "2", "loss", 1.0)]

    # --json has one shape in every view: runs, each with its dims and its data.
    data = json.loads(run("a", "--json"))
    assert data["experiment_id"] == "metrics_cmd" and [r["run_id"] for r in data["runs"]] == ["a"]
    [a] = data["runs"]
    assert set(a) == {"run_id", "experiment_id", "dims", "rows"} and a["rows"][0]["step"] == 1 and "val@4" in a["rows"][0]
    last = json.loads(run("--keys", "^loss$", "--last", "--json"))
    assert {r["run_id"]: r["last"] for r in last["runs"]} == {"a": {"loss": 1.0}, "b": {"loss": 1.0}}
    pivot_json = json.loads(run("--keys", r"^val@(\d+)$", "--pivot", "--json"))
    by_run = {r["run_id"]: r["pivot"] for r in pivot_json["runs"]}
    assert by_run["a"]["step"] == 2 and by_run["a"]["values"]["4"] == 1.5

    # One row per run: values at chosen steps, aggregates, and two experiments at once.
    at = run("--keys", "^loss$", "--step", "1,2").strip().splitlines()
    assert at[0].split() == ["run", "step", "loss@1", "loss@2"] and at[1].split() == ["a", "2", "2", "1"]
    agg = json.loads(run("--keys", "^loss$", "--agg", "max", "--json"))
    assert {r["run_id"]: (r["values"], r["steps"]) for r in agg["runs"]} == {
        "a": ({"max(loss)": 2.0}, {"max(loss)": 1}), "b": ({"max(loss)": 2.0}, {"max(loss)": 1})}
    _api_post(url, token, "/api/experiments", {"campaign": "test", "experiment_id": "metrics_cmd2"})
    _api_post(url, token, "/api/jobs", {"run_id": "a", "experiment_id": "metrics_cmd2", "gpus_per_run": 0,
                                        "command": [sys.executable, "-c", _SCRIPT.replace("OFFSET", "9")]})
    assert _wait_for_job(url, token, "a", "metrics_cmd2", timeout=60)["status"] == "done"
    both = run("--experiment", "metrics_cmd,metrics_cmd2", "a", "--keys", "^val@4$", "--last")
    assert [line.split()[0] for line in both.strip().splitlines()[1:]] == ["metrics_cmd/a", "metrics_cmd2/a"]
    both_json = json.loads(run("--experiment", "metrics_cmd,metrics_cmd2", "--keys", "^loss$", "--last", "--json"))
    assert both_json["experiment_ids"] == ["metrics_cmd", "metrics_cmd2"]
    assert {(r["experiment_id"], r["run_id"]) for r in both_json["runs"]} == {
        ("metrics_cmd", "a"), ("metrics_cmd", "b"), ("metrics_cmd2", "a")}

    bad = subprocess.run(base + ["--keys", "^loss$", "--pivot"], capture_output=True, text=True,
                         env=env, cwd=REPO_ROOT)
    assert bad.returncode == 2                                  # --pivot needs a capture group
