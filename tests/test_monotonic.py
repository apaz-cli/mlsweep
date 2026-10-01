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
    opts = _opts("increasing", [8, 16, 32, 64])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    assert not should_skip(_combo(8),  [_combo(16)], [], stripped)
    assert not should_skip(_combo(8),  [_combo(32)], [], stripped)


def test_decreasing_does_not_skip_before_failure():
    opts = _opts("decreasing", [64, 32, 16, 8])
    validate_options(opts)
    stripped = {k[1:]: v for k, v in opts.items()}
    assert not should_skip(_combo(8),  [_combo(32)], [], stripped)
    assert not should_skip(_combo(16), [_combo(32)], [], stripped)


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


def _should_skip_reference(combo, failed, succeeded, options):
    """The original pairwise definition of the skip rules, which SkipIndex must match."""
    for fc in failed:
        for key, opt in options.items():
            if not opt.get("monotonic"):
                continue
            if not all(fc.get(k) == combo.get(k) for k in options if k != key):
                continue
            vals = opt["_values"]
            try:
                fi, ci = vals.index(fc[key]), vals.index(combo[key])
            except (ValueError, TypeError, KeyError):
                continue
            if fi <= ci:
                return True
    for sc in succeeded:
        for key, opt in options.items():
            if not opt.get("singular"):
                continue
            if not all(sc.get(k) == combo.get(k)
                       for k in options if k != key and not options[k].get("singular")):
                continue
            if sc.get(key) != combo.get(key):
                return True
    return False


def test_skip_index_matches_the_pairwise_definition():
    import random
    from mlsweep._sweep import SkipIndex
    rng = random.Random(7)
    values = [0, 1, 2, 3, "a", [1, 2], {"x": 1}, None, 1.5]
    for _ in range(300):
        dims = rng.sample(["a", "b", "c", "d"], rng.randint(1, 4))
        options = {}
        for d in dims:
            kind = rng.choice(["monotonic", "singular", "plain"])
            options[d] = {"monotonic": kind == "monotonic", "singular": kind == "singular",
                          "_values": rng.sample(values, rng.randint(1, 4))}

        def combo():
            c = {d: rng.choice(options[d]["_values"] + [rng.choice(values)]) for d in dims}
            if rng.random() < 0.1:
                c.pop(rng.choice(dims))  # a combo may lack a dim
            return c
        failed = [combo() for _ in range(rng.randint(0, 6))]
        succeeded = [combo() for _ in range(rng.randint(0, 6))]
        index = SkipIndex(failed, succeeded, options)
        for _ in range(20):
            c = combo()
            assert index.skips(c) == _should_skip_reference(c, failed, succeeded, options), \
                (c, failed, succeeded, options)
