"""Snapshot tests for generate_variations.

On first run, snapshots are written to tests/snapshots/ and the test fails
with a prompt to review them. Once reviewed, subsequent runs compare against
the saved snapshots.

Delete a snapshot file to regenerate it.
"""

import copy
import json
from pathlib import Path

import pytest

from mlsweep._sweep import generate_variations, validate_options


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _snapshot(name: str, opts: dict, sweep_name: str = "s", extra_flags: tuple = (), exclude=None) -> None:
    """Validate opts, generate variations, compare or create snapshot.

    Snapshot format: {"input": {...}, "variations": [...]}
    The input section records the original opts before validate_options mutates them.
    Comparison is against variations only; the input section is for human review.
    """
    opts_input = copy.deepcopy(opts)  # capture before validate_options adds _values/_flags/etc.
    validate_options(opts)
    variations = generate_variations(sweep_name, opts, exclude_fn=exclude, extra_flags=extra_flags)
    actual = [
        {"name": v["name"], "overrides": v["overrides"], "combo": v["combo"]}
        for v in variations
    ]
    input_section = {
        "sweep_name": sweep_name,
        "options": opts_input,
        "extra_flags": list(extra_flags),
        **({"exclude": "(see test)"} if exclude is not None else {}),
    }
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    if not path.exists():
        content = json.dumps(input_section, indent=2) + "\n\n###############\n\n" + json.dumps(actual, indent=2)
        path.write_text(content)
        pytest.fail(f"Snapshot created — review {path} and re-run.")
    top, _, bottom = path.read_text().partition("\n\n###############\n\n")
    expected = json.loads(bottom)
    assert actual == expected, f"Variations differ from snapshot {path}"


# ── Scenarios ──────────────────────────────────────────────────────────────────

def test_simple_cartesian():
    """Two value dims — basic cartesian product, names and flags."""
    opts = {
        ".lr": {"values": [1e-4, 1e-3], "flags": "--lr", "name": "lr"},
        ".bs": {"values": [32, 64], "flags": "--bs", "name": "bs"},
    }
    _snapshot("simple_cartesian", opts)


def test_subdim_with_nested_child():
    """Subdim with a nested child dim — dotted run names, conditional flags."""
    opts = {
        ".opt": {
            "name": "opt",
            ".adam": {"flags": ["--optimizer", "adam"]},
            ".muon": {
                "flags": ["--optimizer", "muon"],
                ".lr_scale": {"values": [0.1, 1.0], "flags": "--lr-scale", "name": "lrs"},
            },
        },
        ".bs": {"values": [32, 64], "flags": "--bs", "name": "bs"},
    }
    _snapshot("subdim_nested", opts)


def test_singular_ordering():
    """Singular dim varies slowest; non-singular dims fill the inner loop."""
    opts = {
        ".bs": {"values": [64, 32, 16], "flags": "--bs", "name": "bs", "singular": True},
        ".lr": {"values": [1e-4, 1e-3], "flags": "--lr", "name": "lr"},
    }
    _snapshot("singular_ordering", opts)


def test_multiple_singular_diagonal():
    """Two singular dims advance diagonally (sum-of-indices order)."""
    opts = {
        ".bs": {"values": [64, 32], "flags": "--bs", "name": "bs", "singular": True},
        ".ac": {
            "flags": {
                "none": ["--ac", "none"],
                "full": ["--ac", "full"],
            },
            "name": "ac",
            "singular": True,
        },
        ".lr": {"values": [1e-3, 1e-4], "flags": "--lr", "name": "lr"},
    }
    _snapshot("multiple_singular_diagonal", opts)


def test_fixed_dim():
    """Fixed dim always appends its flags and does not appear in the run name."""
    opts = {
        ".prec": {"flags": ["--dtype", "bfloat16"], "name": None},
        ".lr": {"values": [1e-4, 1e-3], "flags": "--lr", "name": "lr"},
    }
    _snapshot("fixed_dim", opts)


def test_none_name_suppresses_segment():
    """name=None omits the dim from the run name entirely."""
    opts = {
        ".seed": {"values": [1, 2], "flags": "--seed", "name": None},
        ".lr": {"values": [1e-4, 1e-3], "flags": "--lr", "name": "lr"},
    }
    _snapshot("none_name", opts)


def test_extra_flags_prepended():
    """EXTRA_FLAGS appear before per-dim flags in overrides."""
    opts = {
        ".lr": {"values": [1e-4, 1e-3], "flags": "--lr", "name": "lr"},
    }
    _snapshot("extra_flags", opts, extra_flags=("--steps", "1000"))


def test_exclude_filters_combos():
    """EXCLUDE removes matching combos before dispatch."""
    opts = {
        ".lr": {"values": [1e-4, 1e-3], "flags": "--lr", "name": "lr"},
        ".bs": {"values": [32, 64], "flags": "--bs", "name": "bs"},
    }
    _snapshot("exclude", opts, exclude=lambda c: c["lr"] == 1e-3 and c["bs"] == 64)


def test_bool_value_abbreviation():
    """True/False values in run names are abbreviated as T/F."""
    opts = {
        ".amp": {"values": [True, False], "flags": "--amp", "name": "amp"},
    }
    _snapshot("bool_abbreviation", opts)


def test_monotonic_decreasing_trial_order():
    """Decreasing monotonic dim is tried smallest-first."""
    opts = {
        ".bs": {"values": [64, 32, 16, 8], "flags": "--bs", "name": "bs", "monotonic": "decreasing"},
    }
    _snapshot("monotonic_decreasing", opts)


def test_dict_flags_with_explicit_values():
    """Dict flags with explicit values list respects the values order."""
    opts = {
        ".ac": {
            "values": ["none", "op", "full"],
            "flags": {
                "none": ["--ac.mode", "none"],
                "op":   ["--ac.mode", "selective", "--ac.option", "op"],
                "full": ["--ac.mode", "full"],
            },
            "name": "ac",
        },
    }
    _snapshot("dict_flags", opts)
