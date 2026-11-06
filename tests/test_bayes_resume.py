"""A Bayesian sweep picks up where it left off when its submitter comes back."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from mlsweep._bayes import BayesianOptimizer  # noqa: E402
from mlsweep._sweep import load_sweep_file, validate_options  # noqa: E402

SWEEP = """
COMMAND = ["python", "train.py"]
OPTIONS = {
    ".lr": {"distribution": "log_uniform", "min": 1e-3, "max": 1e-1, "flags": "--lr"},
    ".arch": {"values": ["a", "b"], "flags": "--arch"},
}
OPTIMIZE = {"method": "bayes", "metric": "m", "goal": "minimize", "budget": 5, "n_initial": 2}
"""


def _optimizer(tmp_path: Path) -> BayesianOptimizer:
    path = tmp_path / "s.py"
    path.write_text(SWEEP)
    info = load_sweep_file(str(path))
    validate_options(info["options"], method="bayes")
    return BayesianOptimizer("s", info["options"], info["optimize"])


def test_restore_counts_results_and_continues_the_numbering(tmp_path: Path) -> None:
    first = _optimizer(tmp_path)
    old = first.suggest(n=3)
    for v in old[:2]:
        first.tell(v["combo"], 0.5)

    resumed = _optimizer(tmp_path)
    n_active = resumed.restore([(v["combo"], 0.5) for v in old[:2]], [old[2]["combo"]],
                               [v["name"] for v in old])
    assert resumed._told == 2 and n_active == 1
    assert len(resumed._study.trials) == 3

    new = resumed.suggest(n=1)
    assert new[0]["name"] not in {v["name"] for v in old}

    # A run that was active when we resumed is still counted when it finishes.
    resumed.tell(old[2]["combo"], 0.1)
    assert resumed._told == 3
    assert resumed._study.best_value == pytest.approx(0.1)
