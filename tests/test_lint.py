"""``mlsweep lint`` and the checks ``mlsweep run`` makes before submitting."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mlsweep._lint import find_sweep_files, lint_experiments, lint_file


def _sweep(tmp_path, body: str, name: str = "s.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def _messages(path, severity=None):
    return [f.message for f in lint_file(path) if severity is None or f.severity == severity]


GOOD = """
DESCRIPTION = "Is 1e-3 too high?"
COMMAND = "python train.py"
METRIC = "val_loss"
GOAL = "minimize"
BASE_LR = 3e-4
OPTIONS = {".lr": {"values": [BASE_LR, 1e-3], "flags": "--lr"}}
"""


def test_a_clean_sweep_has_no_findings(tmp_path):
    assert lint_file(_sweep(tmp_path, GOOD)) == []


def test_a_misspelled_setting_is_an_error_with_a_suggestion(tmp_path):
    path = _sweep(tmp_path, GOOD + 'OPTIMZE = {"method": "bayes"}\nGPUS_PER_RUNN = 2\n')
    errors = _messages(path, "error")
    assert any("OPTIMZE" in m and "OPTIMIZE" in m for m in errors)
    assert any("GPUS_PER_RUNN" in m and "GPUS_PER_RUN" in m for m in errors)


def test_own_constants_are_left_alone_and_an_underscore_silences_a_near_miss(tmp_path):
    assert lint_file(_sweep(tmp_path, GOOD + "_METRICS = ['a', 'b']\nLAYERS = 12\n")) == []
    assert _messages(_sweep(tmp_path, GOOD + "METRICS = ['a']\n", "t.py"), "error")


def test_unknown_optimize_and_prune_keys_are_errors(tmp_path):
    path = _sweep(tmp_path, GOOD + 'PRUNE = {"min_step": 10}\n')
    assert any("'min_step'" in m and "min_steps" in m for m in _messages(path, "error"))
    path = _sweep(tmp_path, GOOD + 'OPTIMIZE = {"method": "bayes", "metric": "m", "goal": "minimize", '
                                   '"budget": 4, "bugdet": 5}\n', "t.py")
    assert any("'bugdet'" in m for m in _messages(path, "error"))


def test_a_budget_without_bayes_is_an_error(tmp_path):
    path = _sweep(tmp_path, GOOD + 'OPTIMIZE = {"metric": "val_loss", "budget": 20}\n')
    assert any("method='bayes'" in m for m in _messages(path, "error"))


def test_two_dims_setting_one_flag_is_a_warning(tmp_path):
    path = _sweep(tmp_path, """
        COMMAND = "python train.py"
        EXTRA_FLAGS = ["--seed", "1"]
        OPTIONS = {
            ".lr": {"values": [1, 2], "flags": "--lr"},
            ".warm": {"values": [3], "flags": "--lr"},
            ".seed": {"values": [7], "flags": "--seed"},
        }
    """)
    warnings = _messages(path, "warning")
    assert any(m.startswith("--lr is passed more than once") for m in warnings)
    assert any(m.startswith("--seed is passed more than once") for m in warnings)
    # A negative number is a value, not a flag.
    path = _sweep(tmp_path, 'DESCRIPTION = "d"\nCOMMAND = "x"\nOPTIONS = {".a": {"values": [-1, -2], "flags": "--a"}}\n',
                  "n.py")
    assert lint_file(path) == []


def test_flags_for_values_no_run_uses_and_repeated_values(tmp_path):
    path = _sweep(tmp_path, """
        COMMAND = "python train.py"
        OPTIONS = {
            ".opt": {"values": ["adam"], "flags": {"adam": ["--opt", "adam"], "sgd": ["--opt", "sgd"]}},
            ".bs": {"values": [8, 8], "flags": "--bs"},
        }
    """)
    assert any("['sgd']" in m for m in _messages(path, "warning"))
    assert any("values repeat" in m for m in _messages(path, "error"))


def test_load_errors_and_bad_controls_are_reported(tmp_path):
    assert _messages(_sweep(tmp_path, "COMMAND = 'x'\n"), "error")  # no OPTIONS
    path = _sweep(tmp_path, GOOD + 'MAX_RUNTIME = "soon"\n', "t.py")
    assert any("MAX_RUNTIME" in m for m in _messages(path, "error"))
    assert any("cannot load" in m for m in _messages(_sweep(tmp_path, "1/0\n", "z.py"), "error"))


def test_metric_without_goal_and_goal_without_metric_warn(tmp_path):
    path = _sweep(tmp_path, 'COMMAND = "x"\nMETRIC = "m"\nOPTIONS = {}\n')
    assert any("no GOAL" in m for m in _messages(path, "warning"))


def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "mlsweep", *args], capture_output=True, text=True, cwd=cwd)


def test_lint_cli_exit_codes(tmp_path):
    good = _sweep(tmp_path, GOOD, "good.py")
    warn = _sweep(tmp_path, 'COMMAND = "x"\nMETRIC = "m"\nOPTIONS = {}\n', "warn.py")
    bad = _sweep(tmp_path, GOOD + "OPTIMZE = {}\n", "bad.py")
    assert _cli("lint", str(good)).returncode == 0
    assert _cli("lint", str(warn)).returncode == 0
    assert _cli("lint", "--strict", str(warn)).returncode == 1
    r = _cli("lint", str(good), str(bad))
    assert r.returncode == 1 and "OPTIMZE" in r.stdout


@pytest.mark.parametrize("extra", [["--validate"], ["--dry-run", "--campaign", "test"]])
def test_run_refuses_a_sweep_with_lint_errors(tmp_path, extra):
    bad = _sweep(tmp_path, GOOD + "OPTIMZE = {}\n")
    r = _cli("run", str(bad), *extra, cwd=tmp_path)
    assert r.returncode == 1
    assert "OPTIMZE" in r.stdout and "nothing was submitted" in r.stdout


def test_run_validate_prints_warnings_and_goes_ahead(tmp_path):
    path = _sweep(tmp_path, 'COMMAND = "x"\nMETRIC = "m"\nOPTIONS = {}\n')
    r = _cli("run", str(path), "--validate", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no GOAL" in r.stdout and "Total combinations" in r.stdout


# ── descriptions, and linting everything ──────────────────────────────────────


def test_a_missing_description_warns_and_a_wrong_type_is_an_error(tmp_path):
    no_desc = GOOD.replace('DESCRIPTION = "Is 1e-3 too high?"\n', "")
    assert any("no DESCRIPTION" in m for m in _messages(_sweep(tmp_path, no_desc), "warning"))
    blank = _sweep(tmp_path, no_desc + 'DESCRIPTION = "  "\n', "b.py")
    assert any("no DESCRIPTION" in m for m in _messages(blank, "warning"))
    wrong = _sweep(tmp_path, no_desc + "DESCRIPTION = 3\n", "w.py")
    assert any("DESCRIPTION must be a str" in m for m in _messages(wrong, "error"))
    assert _messages(_sweep(tmp_path, no_desc + "DESCRIPTON = 'x'\n", "t.py"), "error")  # misspelled


def test_run_validate_leaves_out_the_description_warning(tmp_path):
    path = _sweep(tmp_path, GOOD.replace('DESCRIPTION = "Is 1e-3 too high?"\n', ""))
    r = _cli("run", str(path), "--validate", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DESCRIPTION" not in r.stdout


def test_find_sweep_files_goes_by_the_shebang(tmp_path):
    shebang = "#!/usr/bin/env mlsweep_run\n"
    _sweep(tmp_path, shebang + GOOD, "a.py")
    (tmp_path / "sub").mkdir()
    _sweep(tmp_path / "sub", shebang + GOOD, "b.py")
    _sweep(tmp_path, GOOD, "no_shebang.py")
    _sweep(tmp_path, "#!/usr/bin/env python\n", "other.py")
    for skipped in (".hidden", "node_modules", "venv"):
        (tmp_path / skipped).mkdir()
        _sweep(tmp_path / skipped, shebang + GOOD, "c.py")
    (tmp_path / "myenv").mkdir()
    (tmp_path / "myenv" / "pyvenv.cfg").write_text("")
    _sweep(tmp_path / "myenv", shebang + GOOD, "d.py")
    found = sorted(Path(p).resolve() for p in find_sweep_files(tmp_path))
    assert found == [(tmp_path / "a.py").resolve(), (tmp_path / "sub" / "b.py").resolve()]


def test_lint_experiments_flags_a_missing_description():
    report = lint_experiments([
        {"experiment_id": "e1", "campaign": "c", "note": "why"},
        {"experiment_id": "e2", "campaign": "c", "note": None},
        {"experiment_id": "e3", "campaign": "d", "note": "  "},
    ])
    assert report["c/e1"] == []
    assert ["e2" in f.message and "--description" in f.message for f in report["c/e2"]] == [True]
    assert len(report["d/e3"]) == 1


def test_lint_with_no_files_checks_the_project_and_every_campaign(tmp_path, manager_server):
    import json

    from conftest import _api_post

    server, url = manager_server
    _api_post(url, server.token, "/api/experiments", {"campaign": "c1", "experiment_id": "has", "note": "why"})
    _api_post(url, server.token, "/api/experiments", {"campaign": "c2", "experiment_id": "lacks"})
    (tmp_path / ".git").mkdir()  # the project root is the git root, or else the current directory
    _sweep(tmp_path, "#!/usr/bin/env mlsweep_run\n" + GOOD, "good.py")
    conn = ["--manager", url, "--token", server.token]

    r = _cli("lint", *conn, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "good.py" in r.stdout
    assert "experiment c2/lacks: no description" in r.stdout and "has" not in r.stdout.replace("lacks", "")
    assert _cli("lint", "--strict", *conn, cwd=tmp_path).returncode == 1

    out = json.loads(_cli("lint", "--json", *conn, cwd=tmp_path).stdout)
    assert out["good.py"] == [] and out["experiment:c1/has"] == [] and out["experiment:c2/lacks"]
    bad_manager = ["--manager", "http://127.0.0.1:9", "--token", "x"]
    out = json.loads(_cli("lint", "--json", *bad_manager, cwd=tmp_path).stdout)
    assert out["good.py"] == [] and "experiments not checked" in out["manager"][0]["message"]

    # Files named on the command line are checked alone; the manager is not asked.
    r = _cli("lint", "--manager", "http://127.0.0.1:9", "--token", "x", "good.py", cwd=tmp_path)
    assert r.returncode == 0 and "experiment" not in r.stdout

    # A manager that can't be reached is a warning; the files are still checked.
    r = _cli("lint", "--manager", "http://127.0.0.1:9", "--token", "x", cwd=tmp_path)
    assert r.returncode == 0 and "experiments not checked" in r.stdout and "good.py" in r.stdout


def test_run_stores_the_description_and_warns_without_one(tmp_path, manager_server):
    import os

    from conftest import _api_get

    server, url = manager_server
    project = tmp_path / "proj"
    project.mkdir()
    no_desc = GOOD.replace('DESCRIPTION = "Is 1e-3 too high?"\n', "")
    _sweep(project, no_desc, "plain.py")
    _sweep(project, GOOD, "described.py")
    env = {**os.environ, "MLSWEEP_DIR": str(tmp_path / "state")}

    def submit(*args):
        r = subprocess.run([sys.executable, "-m", "mlsweep", "run", *args, "--manager", url,
                            "--token", server.token, "--campaign", "test"],
                           capture_output=True, text=True, cwd=project, env=env, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    def note(exp):
        return _api_get(url, server.token, f"/api/experiments/{exp}")["note"]

    out = submit("plain.py", "--experiment", "e1")
    assert "e1 has no description" in out and "mlsweep rename e1 --description" in out
    assert note("e1") is None

    out = submit("described.py", "--experiment", "e2")
    assert "no description" not in out and note("e2") == "Is 1e-3 too high?"

    submit("described.py", "--experiment", "e3", "--description", "flag wins")
    assert note("e3") == "flag wins"
    submit("plain.py", "--experiment", "e4", "--note", "older flag")
    assert note("e4") == "older flag"

    # --append keeps the description the experiment already has, and doesn't warn.
    out = submit("plain.py", "--experiment", "e2", "--append")
    assert "no description" not in out and note("e2") == "Is 1e-3 too high?"
