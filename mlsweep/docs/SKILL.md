---
name: mlsweep
description: Operate mlsweep hyperparameter sweeps from submission through results (manager → run → watch/fetch → best/cancel/retry). Use when the user mentions mlsweep, sweeps, or running experiments on a GPU cluster.
---

# mlsweep skill

## When to use

The user wants to submit a sweep, check its status, fetch results, find the best
run, or stop/retry jobs.

## Quick reference

```sh
mlsweep status                                  # manager / token / GPU / result-path diagnostics
mlsweep run sweeps/my_sweep.py --validate       # list all combos, no submission
mlsweep run sweeps/my_sweep.py --stream         # submit + live status
mlsweep watch <experiment_id>                   # live terminal status
mlsweep fetch --experiment <id> --wait          # block until done, then leaderboard (exits 1 on failure)
mlsweep best  --experiment <id>                 # top runs by metric (--group-by DIM, --table D1,D2, --json)
mlsweep wait  <id> --until done                 # exit 0 clean, 1 failure, 2 timeout, 3 stalled
                                                # (also --until any-failure, --until stalled --stalled-after 900)
mlsweep watch <id> --events                     # one JSON event per line (alias: --json)
mlsweep ls                                      # experiments grouped by status (`mlsweep ls <id>` lists runs)
mlsweep ls --all                                # also every done experiment or run (default shows 20)
mlsweep logs <run_id> --experiment <id>         # tail a run's training.log
mlsweep metrics --experiment <id> [runs] --keys REGEX [--last|--pivot] [--with-dims] [--json|--csv]  # logged metrics, on demand
mlsweep cancel <id> --failed                    # cancel jobs (also --running / --all --yes)
mlsweep retry  <id> --failed                    # re-queue failed jobs
mlsweep stop   <id> --yes                       # abort a sweep
mlsweep rename <id> <run_id> "name"             # display name next to the run ID (--clear removes it)
mlsweep campaign                                # list campaigns (* = current)
mlsweep campaign move <id> <name>               # move an experiment to another campaign
mlsweep ls --all-campaigns                      # experiments from every campaign
```

## Facts agents need

- Manager: start once with `mlsweep manager`. Dashboard at http://localhost:7891.
- Token: auto-read from `~/.mlsweep/manager.token` (or `MLSWEEP_TOKEN`, or `--token`).
- Results: `~/.mlsweep/experiments/<experiment_id>/<run>/{metrics.jsonl, training.log, artifacts/}`.
- Campaigns group experiments. Every command works in the campaign from `--campaign NAME`, else `$MLSWEEP_CAMPAIGN`, else `default`. `--all-campaigns` covers all of them. A command given an experiment from another campaign exits 1 and names that campaign, so rerun it with `--campaign <that one>`. `mlsweep run` refuses `--all-campaigns`. Each Bash call is a fresh shell, so to avoid repeating `--campaign` set it inline per command: `MLSWEEP_CAMPAIGN="Campaign Name" mlsweep ...` (an `export` in one shell does not persist to the next).
- `watch`/`fetch`/`best`/`status`/`ls`/`logs` default `--manager` to `http://localhost:7891` (or `$MLSWEEP_MANAGER`). `run` requires `--manager`.
- Sweep files are `COMMAND` + `OPTIONS` dicts. CLI flags use dashes. Dimensions multiply, so never enumerate combos by hand. A control arm that ignores a dimension goes in a subdim.
- `run` ships the project directory as it is on disk, uncommitted files included (`.git`, `.venv` and caches are skipped). Do not commit just to submit. Run `--validate` first and check the combo count.
- Never start a CUDA process yourself (smoke tests included) on GPUs a worker owns. Running jobs then fail with `CUDA-capable device(s) is/are busy`. Submit a 1-run sweep instead.
- `logger.log()` without `step=` advances the step on every call. Log train and val metrics in one dict, or pass `step=`.
- A run that exits with an error is not retried automatically. Only runs lost with their worker are, up to `max_retries` (default 2). `retry` re-queues any finished run with a fresh budget, so read its log first.
- `METRIC = "val_loss"` / `GOAL = "minimize"` at the top of a sweep file record the ranking metric. `fetch`/`best` default `--metric`/`--goal` to the experiment's stored values (OPTIMIZE wins if both are present), then `loss`/`minimize`.
- Bayesian sweeps declare `OPTIMIZE = {"metric": ..., "goal": ...}`. The leaderboard ranks by that metric.
- `mlsweep wait EXP --until done|any-failure|stalled` exits `0` clean, `1` failure, `2` timeout, `3` stalled (`--timeout`, `--interval`, `--stalled-after`). `fetch --wait`/`best --wait` exit `1` if the experiment settled with failures. `wait` is safe to re-issue. For a new sweep, `--until any-failure --timeout 900` first catches early crashes.
- Colored terminal output is off by default; pass `--color` to `mlsweep` (or `mlsweep run`/`manager`/`worker`) to enable it. `--json`/`--csv` and raw `logs`/`metrics` stay plain.

Full reference: `mlsweep docs` or `mlsweep --help <topic>` (readme, sweep_configuration, mlsweep, examples).
