---
name: mlsweep
description: Run, monitor, and control mlsweep hyperparameter sweeps (manager → run → watch/fetch → best/cancel/retry). Use when the user mentions mlsweep, sweeps, or running experiments on a GPU cluster.
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
mlsweep fetch --experiment <id> --wait          # block until done, then leaderboard
mlsweep best  --experiment <id>                 # top runs by metric (--json for machines)
mlsweep ls                                      # list experiments (`mlsweep ls <id>` lists runs)
mlsweep logs <run_id> --experiment <id>         # tail a run's training.log
mlsweep cancel <id> --failed                    # cancel jobs (also --running / --all --yes)
mlsweep retry  <id> --failed                    # re-queue failed jobs
mlsweep stop   <id> --yes                       # abort a sweep
```

## Facts agents need

- Manager: start once with `mlsweep manager`. Dashboard at http://localhost:7891.
- Token: auto-read from `~/.mlsweep/manager.token` (or `MLSWEEP_TOKEN`, or `--token`).
- Results: `~/.mlsweep/experiments/<experiment_id>/<run>/{metrics.jsonl, training.log, artifacts/}`.
- `watch`/`fetch`/`best`/`status`/`ls`/`logs` default `--manager` to `http://localhost:7891` (or `$MLSWEEP_MANAGER`). `run` requires `--manager`.
- Sweep files are `COMMAND` + `OPTIONS` dicts. CLI flags use dashes.
- Bayesian sweeps declare `OPTIMIZE = {"metric": ..., "goal": ...}`. The leaderboard ranks by that metric.

Full reference: `mlsweep docs` or `mlsweep --help <topic>` (readme, sweep_configuration, mlsweep, examples).
