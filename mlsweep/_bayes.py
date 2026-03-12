"""Bayesian optimization backend (optuna TPE) for mlsweep."""

import itertools
from typing import Any


def _import_optuna() -> Any:
    """Lazy import of optuna. Raises ImportError with install hint if missing."""
    try:
        import warnings as _warnings
        import optuna
        from optuna.samplers import TPESampler  # noqa: F401
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        return optuna
    except ImportError:
        raise ImportError(
            "Bayesian optimization requires optuna. "
            "Install with: pip install 'mlsweep[bayes]'"
        )


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

        sub_opts_map = opt.get("_sub_opts_map", {})
        values = opt.get("_values", [])

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
    """Build an effective_options dict for use with _treatment_key and should_skip.

    Keys are stripped (no leading dot). Adds selected subdim children so that
    _treatment_key and should_skip work correctly across subdim branches.
    """
    effective: dict[str, Any] = {k[1:]: v for k, v in all_options.items()}
    # Add the selected subdim branch's children
    for k, v in all_options.items():
        sub_map = v.get("_sub_opts_map", {})
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


class BayesianOptimizer:
    """TPE-based Bayesian optimizer for mlsweep sweeps.

    Manages an optuna study, mapping lex combo suggestions to variations
    (including singular probe expansion) and recording results back to optuna.
    """

    def __init__(
        self,
        sweep_name: str,
        options: dict[str, Any],
        optimize_cfg: dict[str, Any],
        extra_flags: list[str] | None = None,
    ) -> None:
        optuna = _import_optuna()
        from optuna.samplers import TPESampler
        self._optuna = optuna
        self._sweep_name = sweep_name
        self._budget: int = optimize_cfg["budget"]
        self._goal: str = optimize_cfg["goal"]
        self._options = options
        self._extra_flags: list[str] = list(extra_flags or [])
        self._lex_options = {k: v for k, v in options.items() if not v.get("singular")}
        self._singular_options = {k: v for k, v in options.items() if v.get("singular")}
        direction = "minimize" if self._goal == "minimize" else "maximize"
        self._study: Any = optuna.create_study(
            direction=direction,
            sampler=TPESampler(multivariate=True, group=True),
        )
        self._lex_key_to_trial: dict[tuple[Any, ...], Any] = {}
        self._told: int = 0
        self._run_counter: int = 1  # 1-based so names start at _bayes_0001

    def suggest(self, n: int = 1) -> list[dict[str, Any]]:
        """Ask optuna for n lex combos and expand each into singular probe variations."""
        all_vars: list[dict[str, Any]] = []
        for _ in range(n):
            if self.exhausted:
                break
            trial = self._study.ask()
            lex_combo, lex_flags = _build_lex_combo(trial, self._lex_options)
            lex_key = tuple(lex_combo[k] for k in sorted(lex_combo))
            self._lex_key_to_trial[lex_key] = trial
            probe_vars, self._run_counter = _expand_singular_probes(
                self._sweep_name, lex_combo, lex_flags,
                self._singular_options, self._options,
                self._extra_flags, self._run_counter,
            )
            all_vars.extend(probe_vars)
        return all_vars

    def tell(self, combo: dict[str, Any], metric_value: float | None) -> None:
        """Report a result back to optuna. metric_value=None counts as a failure."""
        singular_names = {k[1:] for k in self._singular_options}
        lex_key = tuple(combo[k] for k in sorted(combo) if k not in singular_names)
        trial = self._lex_key_to_trial.pop(lex_key, None)
        if trial is None:
            return
        if metric_value is None:
            self._study.tell(trial, state=self._optuna.trial.TrialState.FAIL)
        else:
            self._study.tell(trial, metric_value)
            self._told += 1

    @property
    def exhausted(self) -> bool:
        return self._told >= self._budget
