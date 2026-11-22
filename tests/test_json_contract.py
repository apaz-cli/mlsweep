"""Every ``--json`` command keeps one contract (see ``mlsweep/_jsonout.py``).

stdout carries one JSON value: the result, or ``{"error", "exit_code"}`` with
that same nonzero exit code.  Everything else goes to stderr.  These run each
command as a subprocess against a manager address nothing listens on, which is
the case where stray text used to end up in front of the JSON.
"""

import json
import os
import subprocess
import sys

import pytest

from conftest import _api_post, _start_manager, _teardown_manager

DOWN = ["--manager", "http://127.0.0.1:9", "--token", "t"]

# Each command, with arguments that get it as far as talking to the manager.
COMMANDS = [
    ["ls"], ["ls", "e"], ["logs", "e", "r"], ["metrics", "--experiment", "e"],
    ["cancel", "e", "--active"], ["retry", "e", "r"], ["resume", "e"], ["reschedule", "e", "r"],
    ["rename", "e", "r", "n"], ["stop", "e", "--yes"], ["pause", "e"], ["unpause", "e"],
    ["optimize", "e"], ["controls", "e"], ["best", "--experiment", "e"], ["wait", "e", "--timeout", "1"],
    ["fetch", "--experiment", "e"], ["campaign", "ls"], ["campaign", "move", "e", "x"],
    ["move", "e", "r", "--to", "e2"], ["purge", "e", "--failed", "--yes"], ["gc"], ["delete", "e", "--yes"],
    ["code", "e"], ["export", "tensorboard"], ["export", "wandb", "--project", "p"], ["export", "cancel", "x"],
    ["workers", "ls"], ["workers", "rm", "w"], ["scheduler"],
]


def _run(*argv, env=None):
    return subprocess.run([sys.executable, "-m", "mlsweep", *argv], capture_output=True, text=True,
                          timeout=120, env={**os.environ, **(env or {})})


def _one_value(stdout: str):
    """The single JSON value on stdout; fails on anything else there."""
    return json.loads(stdout)


# Commands that fail without a manager: a missing file, an unknown name.
LOCAL = [["context", "show", "nope"], ["notes", "status", "/no/such/notes"],
         ["notes", "show", "HEAD", "x.md", "--dir", "/no/such/notes"], ["run", "/no/such/sweep.py", "--validate"]]


@pytest.mark.parametrize("argv", LOCAL, ids=lambda a: " ".join(a))
def test_a_local_failure_is_one_error_object(argv):
    r = _run(*argv, "--json")
    assert r.returncode != 0, r.stdout
    out = _one_value(r.stdout)
    assert out["exit_code"] == r.returncode and out["error"], out


@pytest.mark.parametrize("argv", COMMANDS, ids=lambda a: " ".join(a))
def test_a_failure_is_one_error_object(argv):
    r = _run(*argv, "--json", *DOWN)
    assert r.returncode != 0, r.stdout
    out = _one_value(r.stdout)
    assert set(out) >= {"error", "exit_code"} and out["exit_code"] == r.returncode, out
    assert out["error"]


def test_bad_arguments_are_an_error_object_too():
    r = _run("ls", "--json", "--no-such-flag")
    assert r.returncode == 2
    out = _one_value(r.stdout)
    assert out["exit_code"] == 2 and "--no-such-flag" in out["error"]
    assert "usage" in r.stderr


def test_lint_errors_carry_the_findings(tmp_path):
    sweep = tmp_path / "s.py"
    sweep.write_text('COMMAND = "x"\nOPTIONS = {}\nOPTIMZE = {}\n')
    r = _run("run", str(sweep), "--validate", "--json")
    assert r.returncode == 1
    out = _one_value(r.stdout)
    assert out["exit_code"] == 1 and any("OPTIMZE" in f["message"] for f in out["findings"])


def test_lint_reports_findings_as_its_result_even_when_it_fails():
    """Like wait, lint exits 1 on a bad file but its result already says why."""
    r = _run("lint", "/no/such/sweep.py", "--json")
    assert r.returncode == 1
    [finding] = _one_value(r.stdout)["/no/such/sweep.py"]
    assert finding["severity"] == "error" and "cannot load" in finding["message"]


def test_a_command_without_json_says_so():
    r = _run("gen_makefile", "--json")
    assert r.returncode == 2 and "error" in _one_value(r.stdout)


@pytest.fixture(scope="module")
def manager(tmp_path_factory):
    proc, server, url = _start_manager(tmp_path_factory.mktemp("json_contract"), with_worker=False)
    yield server, url
    _teardown_manager(proc)


def test_results_are_one_value_with_human_text_on_stderr(manager, tmp_path):
    server, url = manager
    conn = ["--manager", url, "--token", server.token]
    _api_post(url, server.token, "/api/experiments", {"experiment_id": "jc"})
    _api_post(url, server.token, "/api/jobs", {"experiment_id": "jc", "run_id": "r1", "command": ["true"]})

    r = _run("pause", "jc", "--json", *conn)
    assert r.returncode == 0 and _one_value(r.stdout) == {"experiment_id": "jc", "status": "paused"}
    assert "paused jc" in r.stderr

    r = _run("cancel", "jc", "--active", "--json", *conn)
    out = _one_value(r.stdout)
    assert out["action"] == "cancel" and out["runs"] == [{"run_id": "r1", "ok": True}]

    r = _run("wait", "jc", "--json", "--interval", "0.1", *conn)
    out = _one_value(r.stdout)
    assert r.returncode == out["exit_code"] == 5 and out["outcome"] == "cancelled"
    assert [x["run_id"] for x in out["runs"]] == ["r1"]

    sweep = tmp_path / "s.py"
    sweep.write_text('COMMAND = ["true"]\nOPTIONS = {".a": {"values": [1, 2], "flags": "--a"}}\n')
    r = _run("run", str(sweep), "--validate", "--json", *conn)
    out = _one_value(r.stdout)
    assert r.returncode == 0 and out["combinations"] == 2 and out["findings"] == []
