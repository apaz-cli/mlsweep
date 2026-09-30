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
from mlsweep.ctl import pivot_metrics, select_metrics

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


_SCRIPT = """
from mlsweep.logger import MLSweepLogger
with MLSweepLogger() as lg:
    for step in (1, 2):
        lg.log({"loss": 3.0 - step, "val@1": 4.0 - step, "val@4": OFFSET - step}, step=step)
"""


def test_metrics_command_end_to_end(manager_with_worker):
    server, url = manager_with_worker
    token = server.token
    _api_post(url, token, "/api/experiments", {"experiment_id": "metrics_cmd"})
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
            "--manager", url, "--experiment", "metrics_cmd"]

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

    data = json.loads(run("a", "--json"))
    assert data["a"][0]["step"] == 1 and "val@4" in data["a"][0]

    bad = subprocess.run(base + ["--keys", "^loss$", "--pivot"], capture_output=True, text=True,
                         env=env, cwd=REPO_ROOT)
    assert bad.returncode == 2                                  # --pivot needs a capture group
