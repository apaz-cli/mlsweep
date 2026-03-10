import pytest
from mlsweep._sweep import validate_options, should_skip, generate_variations


def _opts(monotonic, values, flags="--batch-size"):
    return {".bs": {"values": values, "flags": flags, "monotonic": monotonic}}


def _combo(bs):
    return {"bs": bs}


def test_increasing_values_order_preserved():
    opts = _opts("increasing", [8, 16, 32, 64])
    validate_options(opts)
    assert opts[".bs"]["_values"] == [8, 16, 32, 64]


def test_decreasing_values_reversed():
    opts = _opts("decreasing", [64, 32, 16, 8])
    validate_options(opts)
    assert opts[".bs"]["_values"] == [8, 16, 32, 64]


def test_decreasing_equivalent_to_increasing_reversed():
    opts_inc = _opts("increasing", [8, 16, 32, 64])
    opts_dec = _opts("decreasing", [64, 32, 16, 8])
    validate_options(opts_inc)
    validate_options(opts_dec)
    assert opts_inc[".bs"]["_values"] == opts_dec[".bs"]["_values"]


# Skip rule: if a value at index fi fails, skip all candidates at index ci >= fi.
# This includes the failed value itself (fi == ci).

def test_increasing_skip_after_fail():
    opts = _opts("increasing", [8, 16, 32, 64])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    # _values = [8, 16, 32, 64]; failed at 16 (index 1)
    assert not should_skip(_combo(8),  [_combo(16)], [], stripped)  # index 0 < 1, not skipped
    assert     should_skip(_combo(16), [_combo(16)], [], stripped)  # index 1 >= 1, skipped
    assert     should_skip(_combo(32), [_combo(16)], [], stripped)  # index 2 >= 1, skipped
    assert     should_skip(_combo(64), [_combo(16)], [], stripped)  # index 3 >= 1, skipped


def test_decreasing_skip_after_fail():
    opts = _opts("decreasing", [64, 32, 16, 8])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    # _values = [8, 16, 32, 64] (reversed); failed at 32 (index 2)
    assert not should_skip(_combo(8),  [_combo(32)], [], stripped)  # index 0 < 2, not skipped
    assert not should_skip(_combo(16), [_combo(32)], [], stripped)  # index 1 < 2, not skipped
    assert     should_skip(_combo(32), [_combo(32)], [], stripped)  # index 2 >= 2, skipped
    assert     should_skip(_combo(64), [_combo(32)], [], stripped)  # index 3 >= 2, skipped


def test_increasing_no_skip_without_fail():
    opts = _opts("increasing", [8, 16, 32, 64])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    for v in [8, 16, 32, 64]:
        assert not should_skip(_combo(v), [], [], stripped)


def test_decreasing_no_skip_without_fail():
    opts = _opts("decreasing", [64, 32, 16, 8])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    for v in [64, 32, 16, 8]:
        assert not should_skip(_combo(v), [], [], stripped)


def test_increasing_does_not_skip_before_failure():
    # Values before the failing index are not skipped
    opts = _opts("increasing", [8, 16, 32, 64])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    assert not should_skip(_combo(8),  [_combo(16)], [], stripped)
    assert not should_skip(_combo(8),  [_combo(32)], [], stripped)


def test_decreasing_does_not_skip_before_failure():
    # _values = [8, 16, 32, 64]; values with lower index than failure are not skipped
    opts = _opts("decreasing", [64, 32, 16, 8])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    assert not should_skip(_combo(8),  [_combo(32)], [], stripped)
    assert not should_skip(_combo(16), [_combo(32)], [], stripped)


def test_decreasing_trial_order_in_variations():
    opts = {".bs": {"values": [64, 32, 16, 8], "flags": "--bs", "monotonic": "decreasing"}}
    validate_options(opts)
    vars_ = generate_variations("s", opts)
    names = [v["name"] for v in vars_]
    assert names == ["s_bs8", "s_bs16", "s_bs32", "s_bs64"]


def test_increasing_trial_order_in_variations():
    opts = {".bs": {"values": [8, 16, 32, 64], "flags": "--bs", "monotonic": "increasing"}}
    validate_options(opts)
    vars_ = generate_variations("s", opts)
    names = [v["name"] for v in vars_]
    assert names == ["s_bs8", "s_bs16", "s_bs32", "s_bs64"]


def test_skip_only_fires_when_other_dims_match():
    opts = {
        ".bs":  {"values": [8, 16, 32], "flags": "--bs",  "monotonic": "increasing"},
        ".lr":  {"values": [1e-3, 1e-4], "flags": "--lr"},
    }
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    failed = [{"bs": 16, "lr": 1e-3}]
    assert     should_skip({"bs": 32, "lr": 1e-3}, failed, [], stripped)
    assert not should_skip({"bs": 32, "lr": 1e-4}, failed, [], stripped)


def test_decreasing_skip_only_fires_when_other_dims_match():
    opts = {
        ".bs":  {"values": [64, 32, 16], "flags": "--bs",  "monotonic": "decreasing"},
        ".lr":  {"values": [1e-3, 1e-4], "flags": "--lr"},
    }
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    # _values for bs = [16, 32, 64]; failed at 32 (index 1) → skip 32 and 64 (indices >= 1)
    failed = [{"bs": 32, "lr": 1e-3}]
    assert     should_skip({"bs": 64, "lr": 1e-3}, failed, [], stripped)
    assert not should_skip({"bs": 64, "lr": 1e-4}, failed, [], stripped)
