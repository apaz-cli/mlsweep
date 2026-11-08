"""Choosing Bayesian trials with optuna TPE.

``suggest`` keeps nothing between calls.  It rebuilds a study from the trials so
far, which the manager reads from its jobs table.  So a restart, a retried run or
a purged run can't leave an optimizer out of step with the database.  Rebuilding
is cheap: 500 past trials take about 15 ms to add and 7 ms to ask from.
"""

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import optuna
import optuna.exceptions
from optuna.samplers import TPESampler
from optuna.trial import FixedTrial, TrialState, create_trial

from mlsweep._sweep import lex_key, options_from_wire, singular_dims, validate_options

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# How many samples to draw looking for one new combo, before calling a discrete
# search space used up.  A continuous dim never repeats, so this is only ever
# reached when few combos are left.
_MAX_DRAWS_PER_TRIAL = 50

TrialOutcome = Literal["complete", "failed", "running", "abandoned"]

_OPTUNA_STATE: dict[TrialOutcome, TrialState] = {
    "complete": TrialState.COMPLETE,
    "failed": TrialState.FAIL,
    "abandoned": TrialState.FAIL,
    "running": TrialState.RUNNING,
}


@dataclass
class PastTrial:
    """A trial the study should know about.  *value* is set only if complete."""

    combo: dict[str, Any]
    outcome: TrialOutcome
    value: float | None = None


def _build_lex_combo(
    trial: Any,
    options: dict[str, Any],
    combo: dict[str, Any] | None = None,
    flags: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Recursively build a lex combo dict and flag list using an optuna trial.

    options: dot-prefixed dimension keys (non-singular dims only at the top level;
             subdim children are recursed via _sub_opts_map).
    Returns (combo_dict, flags_list) where combo keys are stripped (no leading dot).
    """
    if combo is None:
        combo = {}
    if flags is None:
        flags = []

    for key, opt in options.items():
        dim = key[1:]  # strip leading dot

        if opt.get("singular"):
            continue  # singular dims are handled separately by _expand_singular_probes

        sub_opts_map = opt["_sub_opts_map"]
        values = opt["_values"]

        if values == [None]:
            # Fixed dim — always append its flags, no optuna call, record None in combo
            flags.extend(opt["_flags"].get(None, []))
            combo[dim] = None

        elif sub_opts_map:
            # Subdim — branches are the mutually-exclusive values; suggest one
            branch = trial.suggest_categorical(dim, values)
            combo[dim] = branch
            flags.extend(opt["_flags"].get(branch, []))
            # Recurse into the selected branch's child dims
            children = sub_opts_map.get(branch, {})
            if children:
                _build_lex_combo(trial, children, combo, flags)

        elif opt.get("_type") == "continuous":
            # Continuous dim — use suggest_float or suggest_int
            dist = opt["distribution"]
            lo, hi = float(opt["min"]), float(opt["max"])
            flags_spec = opt.get("flags")
            if dist == "log_uniform":
                val: Any = trial.suggest_float(dim, lo, hi, log=True)
            elif dist == "uniform":
                val = trial.suggest_float(dim, lo, hi, log=False)
            else:  # int_uniform
                val = trial.suggest_int(dim, int(lo), int(hi))
            combo[dim] = val
            if flags_spec is not None:
                flags.extend([flags_spec, str(val)])

        else:
            # Discrete value dim — suggest one of the enumerated values
            val = trial.suggest_categorical(dim, values)
            combo[dim] = val
            flags.extend(opt["_flags"].get(val, []))

    return combo, flags


def _build_effective_options(combo: dict[str, Any], all_options: dict[str, Any]) -> dict[str, Any]:
    """Build an effective_options dict for per-combo option lookup.

    Keys are stripped (no leading dot). Adds selected subdim children so that
    lookups work correctly across subdim branches.
    """
    effective: dict[str, Any] = {k[1:]: v for k, v in all_options.items()}
    # Add the selected subdim branch's children
    for k, v in all_options.items():
        sub_map = v["_sub_opts_map"]
        dim = k[1:]
        if sub_map and dim in combo:
            for ck, cv in sub_map.get(combo[dim], {}).items():
                effective[ck[1:]] = cv
    return effective


def _expand_singular_probes(
    sweep_name: str,
    lex_combo: dict[str, Any],
    lex_flags: list[str],
    singular_options: dict[str, Any],
    all_options: dict[str, Any],
    extra_flags: list[str],
    run_counter: int,
) -> tuple[list[dict[str, Any]], int]:
    """Expand a lex combo into probe variations, one per diagonal-order singular combination.

    If there are no singular dims, returns a single variation.
    The diagonal ordering mirrors _expand_tree so that the most-aggressive singular
    values are tried first (sum of indices ascending).
    """
    if not singular_options:
        full_combo = dict(lex_combo)
        full_overrides = list(extra_flags) + list(lex_flags)
        name = f"{sweep_name}_bayes_{run_counter:04d}"
        effective = _build_effective_options(full_combo, all_options)
        return ([{
            "name": name,
            "overrides": full_overrides,
            "combo": full_combo,
            "effective_options": effective,
        }], run_counter + 1)

    # Build diagonal-order index tuples over singular dim values
    sing_keys = list(singular_options.keys())
    raw = list(itertools.product(*(range(len(singular_options[k]["_values"])) for k in sing_keys)))
    raw.sort(key=lambda idx: (sum(idx), idx))

    variations = []
    counter = run_counter
    for idx in raw:
        singular_combo: dict[str, Any] = {}
        singular_flags: list[str] = []
        for i, ki in enumerate(idx):
            k = sing_keys[i]
            dim = k[1:]
            val = singular_options[k]["_values"][ki]
            singular_combo[dim] = val
            singular_flags.extend(singular_options[k]["_flags"].get(val, []))

        full_combo = {**lex_combo, **singular_combo}
        full_overrides = list(extra_flags) + list(lex_flags) + singular_flags
        name = f"{sweep_name}_bayes_{counter:04d}"
        effective = _build_effective_options(full_combo, all_options)
        variations.append({
            "name": name,
            "overrides": full_overrides,
            "combo": full_combo,
            "effective_options": effective,
        })
        counter += 1

    return variations, counter


def suggest(
    spec: dict[str, Any],
    history: list[PastTrial],
    n: int,
    next_number: int = 1,
) -> list[dict[str, Any]]:
    """Choose up to *n* new trials given *history*, and return their runs.

    *spec* is the sweep's ``optimize`` spec.  A trial becomes one run, or one per
    singular probe, numbered from *next_number*.  Combos already tried are not
    tried again, except abandoned ones.  So this returns fewer than *n* trials
    when a discrete search space is nearly used up.
    """
    options = options_from_wire(spec["options"])
    validate_options(options, method="bayes")
    singular = singular_dims(options)
    lex_options = {k: v for k, v in options.items() if not v.get("singular")}
    singular_options = {k: v for k, v in options.items() if v.get("singular")}

    # constant_liar keeps trials asked at the same time from landing on one spot.
    study = optuna.create_study(
        direction="minimize" if spec["goal"] == "minimize" else "maximize",
        sampler=TPESampler(multivariate=True, group=True, constant_liar=True,
                           n_startup_trials=min(spec["n_initial"], spec["budget"])),
    )
    seen: set[tuple[tuple[str, str], ...]] = set()
    for past in history:
        fixed = FixedTrial({k: v for k, v in past.combo.items() if k not in singular and v is not None})
        try:
            _build_lex_combo(fixed, lex_options)
        except ValueError:
            continue  # a run that isn't one of these trials, added to the experiment by hand
        study.add_trial(create_trial(
            params=fixed.params, distributions=fixed.distributions,
            state=_OPTUNA_STATE[past.outcome], value=past.value,
        ))
        if past.outcome != "abandoned":
            seen.add(lex_key(past.combo, singular))

    variations: list[dict[str, Any]] = []
    counter = next_number
    for _ in range(n):
        for _ in range(_MAX_DRAWS_PER_TRIAL):
            trial = study.ask()
            lex_combo, lex_flags = _build_lex_combo(trial, lex_options)
            key = lex_key(lex_combo, singular)
            if key not in seen:
                break
            study.tell(trial, state=TrialState.FAIL)  # a repeat: never run, so nothing learned
        else:
            return variations
        seen.add(key)
        probes, counter = _expand_singular_probes(
            spec["sweep_name"], lex_combo, lex_flags, singular_options, options,
            spec["extra_flags"], counter,
        )
        variations.extend(probes)
    return variations
