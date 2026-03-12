import pytest
from mlsweep._sweep import validate_options


# ── Valid configurations ────────────────────────────────────────────────────────

def test_value_dim_str_flags():
    opts = {".lr": {"values": [1e-4, 1e-3], "flags": "--lr", "name": "lr"}}
    validate_options(opts)
    assert opts[".lr"]["_values"] == [1e-4, 1e-3]
    assert opts[".lr"]["_flags"] == {1e-4: ["--lr", "0.0001"], 1e-3: ["--lr", "0.001"]}
    assert opts[".lr"]["_sub_opts_map"] == {}


def test_value_dim_dict_flags_infers_values():
    opts = {
        ".ac": {
            "flags": {
                "none": ["--ac", "none"],
                "full": ["--ac", "full"],
            }
        }
    }
    validate_options(opts)
    assert opts[".ac"]["_values"] == ["none", "full"]
    assert opts[".ac"]["_flags"] == {"none": ["--ac", "none"], "full": ["--ac", "full"]}


def test_value_dim_none_flags_rejected():
    # flags=None is not supported; use name=None to suppress the dim from the run name instead.
    with pytest.raises(ValueError, match="flags=None is not supported"):
        validate_options({".seed": {"values": [1, 2, 3], "flags": None, "name": "seed"}})


def test_fixed_dim():
    opts = {".prec": {"flags": ["--dtype", "bfloat16"]}}
    validate_options(opts)
    assert opts[".prec"]["_values"] == [None]
    assert opts[".prec"]["_flags"] == {None: ["--dtype", "bfloat16"]}
    assert opts[".prec"]["_sub_opts_map"] == {}


def test_fixed_dim_str_flag():
    opts = {".flag": {"flags": "--some-flag"}}
    validate_options(opts)
    assert opts[".flag"]["_flags"] == {None: ["--some-flag"]}


def test_subdim_basic():
    opts = {
        ".opt": {
            "name": "opt",
            ".adam": {"flags": ["--optimizer", "adam"]},
            ".muon": {"flags": ["--optimizer", "muon"]},
        }
    }
    validate_options(opts)
    assert opts[".opt"]["_values"] == ["adam", "muon"]
    assert opts[".opt"]["_flags"] == {
        "adam": ["--optimizer", "adam"],
        "muon": ["--optimizer", "muon"],
    }
    assert opts[".opt"]["_sub_opts_map"] == {}


def test_subdim_with_children():
    opts = {
        ".opt": {
            ".adam": {"flags": ["--optimizer", "adam"]},
            ".muon": {
                "flags": ["--optimizer", "muon"],
                ".lr_scale": {"values": [0.1, 1.0], "flags": "--lr-scale", "name": "lrs"},
            },
        }
    }
    validate_options(opts)
    assert opts[".opt"]["_values"] == ["adam", "muon"]
    assert "muon" in opts[".opt"]["_sub_opts_map"]
    child = opts[".opt"]["_sub_opts_map"]["muon"]
    assert ".lr_scale" in child
    assert child[".lr_scale"]["_values"] == [0.1, 1.0]


def test_continuous_dim_grid_mode():
    opts = {".lr": {"distribution": "log_uniform", "min": 1e-4, "max": 1e-1, "samples": 4, "flags": "--lr"}}
    validate_options(opts, method="grid")
    assert len(opts[".lr"]["_values"]) == 4
    for v in opts[".lr"]["_values"]:
        assert 1e-4 <= v <= 1e-1


def test_continuous_dim_bayes_mode():
    opts = {".lr": {"distribution": "log_uniform", "min": 1e-4, "max": 1e-1, "flags": "--lr"}}
    validate_options(opts, method="bayes")
    assert opts[".lr"]["_type"] == "continuous"
    assert opts[".lr"]["_values"] == []
    assert opts[".lr"]["_flags"] == {}


def test_continuous_int_uniform():
    opts = {".layers": {"distribution": "int_uniform", "min": 2, "max": 8, "samples": 3, "flags": "--layers"}}
    validate_options(opts, method="grid")
    assert len(opts[".layers"]["_values"]) == 3
    for v in opts[".layers"]["_values"]:
        assert isinstance(v, int)
        assert 2 <= v <= 8


def test_monotonic_decreasing_reverses():
    opts = {".bs": {"values": [64, 32, 16, 8], "flags": "--bs", "monotonic": "decreasing"}}
    validate_options(opts)
    assert opts[".bs"]["_values"] == [8, 16, 32, 64]


def test_monotonic_increasing_preserves():
    opts = {".bs": {"values": [8, 16, 32, 64], "flags": "--bs", "monotonic": "increasing"}}
    validate_options(opts)
    assert opts[".bs"]["_values"] == [8, 16, 32, 64]


def test_singular_flag_preserved():
    opts = {".bs": {"values": [64, 32], "flags": "--bs", "singular": True}}
    validate_options(opts)
    assert opts[".bs"].get("singular") is True


# ── Error cases ─────────────────────────────────────────────────────────────────

def test_key_without_dot():
    with pytest.raises(ValueError, match="must start with '.'"):
        validate_options({"lr": {"values": [1e-3], "flags": "--lr"}})


def test_unknown_metadata_key():
    with pytest.raises(ValueError, match="Unknown metadata key"):
        validate_options({".lr": {"values": [1e-3], "flags": "--lr", "typo": True}})


def test_both_values_and_subdims():
    with pytest.raises(ValueError, match="both 'values' and subdimensions"):
        validate_options({
            ".opt": {
                "values": ["adam"],
                ".adam": {"flags": "--adam"},
            }
        })


def test_distribution_with_subdims():
    with pytest.raises(ValueError, match="cannot have both 'distribution' and subdimensions"):
        validate_options({
            ".lr": {
                "distribution": "log_uniform",
                "min": 1e-4,
                "max": 1e-1,
                ".sub": {"flags": "--x"},
            }
        })


def test_distribution_with_values():
    with pytest.raises(ValueError, match="cannot have both 'distribution' and 'values'"):
        validate_options({
            ".lr": {
                "distribution": "log_uniform",
                "min": 1e-4,
                "max": 1e-1,
                "values": [1e-3],
            }
        })


def test_distribution_missing_min():
    with pytest.raises(ValueError, match="requires 'min' and 'max'"):
        validate_options({".lr": {"distribution": "log_uniform", "max": 1e-1}})


def test_distribution_missing_max():
    with pytest.raises(ValueError, match="requires 'min' and 'max'"):
        validate_options({".lr": {"distribution": "log_uniform", "min": 1e-4}})


def test_distribution_min_ge_max():
    with pytest.raises(ValueError, match="min must be < max"):
        validate_options({".lr": {"distribution": "log_uniform", "min": 1e-1, "max": 1e-4}})


def test_invalid_distribution_name():
    with pytest.raises(ValueError, match="distribution must be"):
        validate_options({".lr": {"distribution": "gaussian", "min": 0.0, "max": 1.0}})


def test_samples_in_bayes_mode():
    with pytest.raises(ValueError, match="'samples' is not used in bayes mode"):
        validate_options(
            {".lr": {"distribution": "log_uniform", "min": 1e-4, "max": 1e-1, "samples": 5, "flags": "--lr"}},
            method="bayes",
        )


def test_samples_missing_in_grid_mode():
    with pytest.raises(ValueError, match="'samples'.*required in grid mode"):
        validate_options(
            {".lr": {"distribution": "log_uniform", "min": 1e-4, "max": 1e-1, "flags": "--lr"}},
            method="grid",
        )


def test_invalid_monotonic_value():
    with pytest.raises(ValueError, match="monotonic must be"):
        validate_options({".bs": {"values": [8, 16], "flags": "--bs", "monotonic": "up"}})


def test_dict_flags_value_not_list():
    with pytest.raises(ValueError, match="must be a list"):
        validate_options({".ac": {"flags": {"none": "--ac none"}}})


def test_subdim_key_collision_with_ancestor():
    with pytest.raises(ValueError, match="collides with ancestor or sibling"):
        validate_options({
            ".opt": {
                ".adam": {"flags": "--adam"},
                ".muon": {
                    "flags": "--muon",
                    ".opt": {"values": [1], "flags": "--x"},
                },
            }
        })


def test_missing_flags_for_value():
    with pytest.raises(ValueError, match="missing flags for value"):
        validate_options({
            ".lr": {
                "values": [1e-3, 1e-4],
                "flags": {1e-3: ["--lr", "0.001"]},  # missing 1e-4
            }
        })
