"""``mlsweep lint`` and the checks ``mlsweep run`` makes before submitting."""

import subprocess
import sys
import textwrap

import pytest

from mlsweep._lint import lint_file


def _sweep(tmp_path, body: str, name: str = "s.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def _messages(path, severity=None):
    return [f.message for f in lint_file(path) if severity is None or f.severity == severity]


GOOD = """
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
    path = _sweep(tmp_path, 'COMMAND = "x"\nOPTIONS = {".a": {"values": [-1, -2], "flags": "--a"}}\n', "n.py")
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
