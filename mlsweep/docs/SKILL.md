---
name: mlsweep
description: Use when running machine learning experiments. Hyperparameter sweeps from submission through results (manager → run → wait/fetch → best/cancel/retry), and read or write shared research notes. Use when the user mentions mlsweep, sweeps, running experiments on a GPU cluster, the notes or knowledgebase, or recording what an experiment found.
---

# mlsweep skill

## Pitfalls

- Never start a CUDA process yourself (smoke tests included) on GPUs a worker owns. Running jobs then OOM or fail with `CUDA-capable device(s) is/are busy`. Submit a 1-run sweep instead.
- `retry` and `resume` rerun the code submitted with the experiment. After fixing a bug, submit a new sweep with `mlsweep run`. Use `retry` for failures outside the code, such as a lost worker or a full disk.
- `run` ships the project directory as it is on disk, uncommitted files included (`.git`, `.venv`, `outputs` and caches are skipped; symlinks are not followed). Do not commit just to submit. Leave large local folders (old runs, checkpoints) out with `UPLOAD_EXCLUDE = ["/runs"]` in the sweep file or `--exclude`; the run prints how much of the upload was new. `mlsweep code EXP [PATH]` shows the code an experiment ran (`--lines 40-60` or `-n` numbers it and pages it within a budget, and a note's `mlsweep:code/...` link works as the argument), `mlsweep code search PATTERN [EXP]` finds lines in it or across the campaign, and `mlsweep code diff A B` shows what changed between two.
- `logger.log()` without `step=` advances the step on every call. Log train and val metrics in one dict, or pass `step=`. Logging a step lower than the last one, or the same key twice at one step, raises `ValueError` (it would overwrite data).
- A run that exits with an error is not retried automatically. Only runs lost with their worker are, up to `max_retries` (default 2). `retry` re-queues any finished run with a fresh budget, so read its log first.

## What needs permission

- Your own experiments, meaning the ones you submitted: `cancel`, `stop`, `pause` and `retry` are yours to use. Abort a sweep you can see is broken rather than letting it use the cluster.
- Anyone else's experiments: leave them alone unless the user asks. If one you did not start looks broken, ask.
- `purge`, `delete` and `gc --yes` destroy data, and `export wandb` publishes it. Run them only when the user asks.
- `workers add`, `workers rm`, `workers forget`, `workers set`, `workers update`, `scheduler --placement` and `scheduler --pause` change the cluster for everyone. Ask first.
- If the manager is unreachable, check the URL and token, then ask. Start a manager only if you were told to. Its worker takes every visible GPU, and a manager started from your session can die with that session.
- For anything else, use your judgment and ask the user when in doubt.

## Workflow

Do what the user asked: keep going through steps 2 to 6 if they said to, otherwise record what you have and finish. If budget, cluster conflicts or anything else is unclear, ask them.

1. Orient. Run `mlsweep status` and `mlsweep ls`. Run `mlsweep notes clone ~/notes`, which is safe to repeat, then `git -C ~/notes pull`. Read `AGENTS.md` and `STATUS.md` there if it exists, and search the notes for what is already known, dead ends included.
2. Write one sweep file per question, so each becomes its own experiment, then run `mlsweep run sweeps/my_sweep.py --validate` and check the combo count. `mlsweep run sweeps/my_sweep.py --compare <id>` instead compares the file's runs with an existing experiment's, by command line. It refuses a file with lint errors, such as a misspelled setting (`mlsweep lint` runs the checks alone); read the warnings too.
3. Submit with `mlsweep run sweeps/my_sweep.py`. It prints the experiment ID. Put the ID in `STATUS.md` and push, so that another session can pick up from there.
4. Monitor. Run `mlsweep wait <id> --until any-failure --timeout 300` to catch early crashes, then `mlsweep wait <id> --until stalled --timeout 300`. Repeat while it exits 2. Keep `--timeout` below any command limit you have, or run `wait` in the background. `wait` takes several IDs, or none for every experiment in the campaign with active runs, so one wait covers everything you have queued. Don't write your own watcher that parses `ls`.
5. Handle failures. Read `mlsweep logs <id> <run_id>`. When a run restarted, moved or vanished and you don't know why, `mlsweep history <id> <run_id>` says what happened and who did it. If the code was wrong, fix it, stop the old sweep if it is still running, and go back to step 2. If the code was fine, use `mlsweep retry <id> --failed`. If the cause is unclear, for example a worker error, work it out with the user.
6. Record. Compare with `mlsweep best --experiment <id>` and `mlsweep metrics`, then write what you learned in the notes, with failed arms and dead ends. Update `STATUS.md`, commit and push.

### What `wait` exits with

| Exit | Meaning | Next |
|---|---|---|
| 0 | every run succeeded | compare and record |
| 1 | a run failed, or the manager paused the sweep (it prints `PAUSED` and why) | read its log; for a pause, usually METRIC names a key the script doesn't log |
| 2 | timed out, runs still active | wait again; if runs stay pending, check `mlsweep status` and `mlsweep ls <id>` |
| 3 | a running run logged nothing for `--stalled-after` seconds (default 900) | read its log; if the script is just quiet for long stretches, raise this or have it call `logger.heartbeat()` there |
| 4 | no answer: manager unreachable, wrong ID, or no runs | check the ID and `mlsweep status`; the runs may be fine |
| 5 | no failures, but some runs were cancelled or the sweep was stopped | the results are partial, so say so in the notes |

`--until stalled` also exits when the sweep settles, so it covers `--until done`. Neither notices runs stuck in pending, so look at `ls` if a timeout repeats.

To hear about each run as it finishes, use `mlsweep wait <id> --until any-done --since <cursor> --json` in a loop, passing the previous result's `cursor` each time. It exits 0 with `outcome: run_finished` and the finished runs, or with the sweep's final outcome once it settles. Leave out `<id>` to hear about every run in the campaign, including experiments submitted later. Each run in the result then names its `experiment_id`.

Every `--json` command prints exactly one JSON value on stdout, and everything else on stderr. A failure is `{"error": ..., "exit_code": N}` with that exit code.

## Sweep files

Dimension keys start with a dot. Dimensions multiply, so never enumerate combos by hand. A control arm that ignores a dimension goes in a subdim.

```python
COMMAND = ["python", "train.py"]
OPTIONS = {
    ".lr": {"values": [1e-3, 3e-4], "flags": "--lr"},
    ".seed": {"values": [0, 1, 2], "flags": "--seed"},  # 6 runs
}
METRIC = "val_loss"  # ranks runs in fetch/best
GOAL = "minimize"
```

`fetch`/`best` default `--metric`/`--goal` to the experiment's stored values, then `loss`/`minimize`. `best` exits 1 if no finished run logged the metric.

- When `METRIC` is set, the manager pauses the sweep if its first 2 finished runs never logged it (`METRIC_CHECK`). `mlsweep ls <id>` shows each run's reason with the keys it did log.
- `MAX_RUNTIME = "2h"` stops and fails runs that run longer.
- `PRUNE = {"min_steps": 500}` stops runs worse than the median of the others at step 500, 1000, and so on (`"strategy": "asha"` keeps the best third instead). Pruned runs have status `pruned`; they are results, not failures.

The training script logs with:

```python
from mlsweep.logger import MLSweepLogger

with MLSweepLogger(total_steps=num_steps) as logger:
    logger.log({"val_loss": 0.42}, step=step)
```

`total_steps` is optional. With it, `mlsweep ls <id>` shows each running run's time left and `mlsweep status` when each GPU frees up, so pass it when the script knows its step count. In a long stretch that logs nothing (an evaluation pass, a phase that doesn't log), call `logger.heartbeat()` so the run doesn't look stalled.

### Bayesian sweeps

These declare `OPTIMIZE = {"method": "bayes", "metric": ..., "goal": ..., "budget": N}`, which wins over `METRIC`/`GOAL`.

- The manager runs the sweep, so submit it like any other and use `wait`. No process has to stay attached. It chooses new trials from the results so far as each one comes in.
- `mlsweep optimize <id>` shows the budget, the failures and the best run. `--budget N`, `--budget +N`, `--finish` and `--max-concurrent K` adjust it. Fewer at once means each trial learns from more results.
- After `max_failures` failed trials in a row the manager pauses the sweep, and `wait` returns. `optimize` shows why. If the metric name or the code was wrong, stop it and submit again. Otherwise use `mlsweep unpause <id>`.
- `mlsweep cancel` on one of its runs gets the trial replaced. To end the sweep, use `mlsweep stop <id> --yes`, or `mlsweep optimize <id> --finish` to let running runs end.

## Quick reference

```sh
# submit
mlsweep run sweeps/my_sweep.py --validate       # list all combos, no submission
mlsweep run sweeps/my_sweep.py                  # submit (--note TEXT, --max-concurrent N, --experiment NAME)
mlsweep run sweeps/my_sweep.py --stream         # submit + live status, stays attached

# monitor
mlsweep status                                  # manager, token and workers (its GPU list is this machine's)
mlsweep ls                                      # experiments grouped by status (`mlsweep ls <id>` lists runs: step, time left, worker and GPU)
mlsweep ls --all                                # also every done experiment or run (default shows 20)
mlsweep ls --finished-since 2h                  # runs that finished lately, across the campaign, failures first
mlsweep wait <id> --until stalled --timeout 300 # block; see the exit table above
mlsweep wait <id> --until started --timeout 600 # smoke test: every run has logged a step (exits 1 if one fails)
mlsweep wait --until any-done --since <cursor> --json  # every run that finishes in the campaign
mlsweep watch <id>                              # live terminal status
mlsweep watch <id> --events                     # one JSON event per line (alias: --json)
mlsweep logs <id> <run_id>                      # tail a run's training.log

# results
mlsweep best --experiment <id>                  # top runs by metric (--group-by DIM, --table D1,D2, --json)
mlsweep metrics --experiment <id> [runs] --keys REGEX [--last|--pivot] [--with-dims] [--json|--csv]  # logged metrics, on demand
mlsweep metrics --experiment <id> --keys 'val_loss|diverged' --agg final --agg max:diverged  # one row per run
mlsweep metrics --active --keys '^val_loss$' --last   # every experiment with runs still going (or --experiment A,B)
mlsweep metrics --experiment <id> --keys '^val_loss$' --step 500,1000,1500   # values at chosen steps
mlsweep metrics --experiment <id> --keys '^val_loss$' --x tokens --at 1e8,2e8 # compare runs at equal tokens
mlsweep fetch --experiment <id> --wait          # block until done, leaderboard, download (exits 1 on failure)
mlsweep export tensorboard --experiment <id>    # zip of TensorBoard logs
mlsweep sql "SELECT ..."                        # read-only SQL, for what no command answers (`mlsweep sql` lists tables)

# control your own sweeps
mlsweep stop <id> --yes                         # abort: cancel every unfinished run
mlsweep cancel <id> --running                   # cancel some runs (also --pending, run IDs, or --all --yes)
mlsweep retry <id> --failed                     # re-queue failed runs, same code
mlsweep resume <id>                             # re-queue failed and cancelled runs, same code
mlsweep pause <id>                              # stop dispatching new jobs
mlsweep unpause <id>                            # undo pause (`resume` does not)
mlsweep rename <id> <run_id> "name"             # display name next to the run ID (--clear removes it)

# only when the user asks
mlsweep purge <id> --failed --yes               # delete finished runs with their files
mlsweep delete <id> --yes                       # delete an experiment and its runs
mlsweep gc                                      # dry run (--yes deletes unreferenced files)
mlsweep export wandb --project P                # publish to W&B

# organize
mlsweep move <id> <run_id> --to <id2>           # move runs; a running one moves when it finishes (--create makes <id2>)
mlsweep history <id> [run_id]                   # what happened to its runs and who did it (dispatches, requeues, cancels, deletes)
mlsweep campaign                                # list campaigns (* = current)
mlsweep campaign move <id> <name>               # move an experiment to another campaign
mlsweep ls --all-campaigns                      # experiments from every campaign

# cluster (reading is fine; ask before changing anything)
mlsweep workers                                 # list workers
mlsweep workers add user@host --tunnel          # connect a machine whose firewall lets only SSH in
mlsweep workers set <worker_id> --jobs N        # tune a worker
mlsweep scheduler                               # show or set GPU placement (--placement pack|spread)
mlsweep scheduler --pause [<worker_id>]         # start no new jobs (everywhere by default); --unpause
```

## Facts agents need

- Manager and dashboard: every command uses `--manager`, else `$MLSWEEP_MANAGER`, else http://localhost:7891.
- Token: auto-read from `~/.mlsweep/manager.token` (or `MLSWEEP_TOKEN`, or `--token`). On another machine you need one of the last two.
- `MLSWEEP_SANDBOX=1` means you are in a dashboard terminal on the manager. The manager, the token and `~/notes` are already set up, and there are no GPUs, so run everything that needs one as a sweep.
- Results live on the manager's host, at `~/.mlsweep/experiments/<experiment_id>/<run>/{metrics.jsonl, training.log, artifacts/}`. From another machine, use `metrics`, `logs` and `fetch`.
- `fetch --wait`/`best --wait` exit `1` if the experiment settled with failures.
- `--experiment NAME` refuses a name already in use. Pass `--append` to add the new runs to that experiment.
- CLI flags use dashes. Colored output is off unless you pass `--color`. `--json`/`--csv` and raw `logs`/`metrics` stay plain.

## Campaigns

- Campaigns group experiments. Every command works in the campaign from `--campaign NAME`, else the context's campaign, else `$MLSWEEP_CAMPAIGN`. There is no default campaign, so pass one. `--all-campaigns` covers all of them, except that `mlsweep run` refuses it. A command given an experiment from another campaign exits 1 and names that campaign, so rerun it with `--campaign <that one>`.
- If each of your commands runs in a fresh shell, an `export` does not persist to the next call. Instead, save the manager and campaign once as a context with `mlsweep context add NAME --manager URL --campaign "Campaign Name" --use`. Every later command uses it with no flags or variables. It is stored in `~/.config/mlsweep/config.toml`, so `--use` changes the default for every session of this user; if other agents share the account, pass `--context NAME` on each command rather than `--use`. A set `$MLSWEEP_MANAGER` (as in a dashboard terminal) skips the current context, so there set the campaign inline: `MLSWEEP_CAMPAIGN="Campaign Name" mlsweep ...`.

## Notes

- The notes are a git repo hosted by the manager. Work in a clone, including on the manager's machine. `mlsweep notes clone ~/notes` is safe to repeat. If several agents share a machine, give each its own folder and `--name`.
- Use normal git. Pull before you start, then commit and push as you go, always before you stop, so a later session finds what you found. If a push is refused, pull, resolve any conflicts so both sides' findings survive, and push again. Never force-push.
- Notes commands run outside the clone need its path, or `MLSWEEP_NOTES=~/notes` inline. Without one they read the manager's own folder, and they refuse to change it.
- `mlsweep notes search WORDS` searches like the Notes tab. `mlsweep notes links PATH` shows backlinks, `mlsweep notes mentions EXP` finds notes about an experiment, and `mlsweep notes history` shows who changed what. `mlsweep notes status` shows unpushed commits. Rename or move notes with `mlsweep notes mv SRC DST`, because plain `mv` breaks links to them.
- Write experiment IDs in full (`rzo_e1_zo_20261001_0309_c37e`) so the dashboard links them.
- The first column of `best` is the run's best step, and `final=` is its last. `mlsweep metrics --experiment <id> --last` gives only the last, so use `--agg min:val_loss` (it shows the step) or `--step N` to check a best-step value. `--last` marks a value logged before the run's latest step with `@step`, so a run at step 1250 with `val_loss 4.10 @1000` was last evaluated at 1000. Reach for Python only for what these don't give. Lint checks the number behind an evidence link, not whether it is the right claim to make, so still read the curve.
- `best` ranks only runs that finished and logged the metric, and its header counts the rest. A failed arm is a result, so report it.
- Notes are for people: plain words, findings first, and say when a term is one you made up. `AGENTS.md` in the notes has the rules.
- Keep `STATUS.md` (one per project folder) short and readable by someone with none of your context: what is running, with experiment IDs, and what comes next.
- `mlsweep notes lint` checks links, experiment and run IDs, chart/table specs, and the numbers behind evidence links, and reports leftover conflict markers and empty files. It always exits 0, so read the report. A first line saying something was "not checked" means those numbers were not verified, and it gives the manager's reason. It needs the manager for the IDs (`--manager`, or `--no-manager` to skip them). `--fix` applies safe link-path fixes, and `--review-prompt` prints a brief for an independent reviewing agent.

### Recording results

Put the evidence in the note, not in a screenshot. These blocks are fetched from
the manager when the note is opened, so they stay current and count as citations.

````
```mlsweep-chart
experiments: <experiment_id>
metric: val_loss
split_by: lr          # color by a hyperparameter; band: minmax shows seed spread
caption: Validation loss by learning rate. {#fig-loss}
```
````

````
```mlsweep-table
experiments: <experiment_id>
metrics: [val_loss, acc]
sort: val_loss
order: asc
```
````

- Chart keys: `experiments` or `runs` (`exp/run`), `metric`, then optional `split_by`, `band`, `smooth`, `log_y`, `x_max`, `where` (`lr=0.001`), `points`, `limit`, `caption`. Table keys add `metrics`, `columns`, `agg` (`final`/`min`/`max`/`mean`), `sort`, `order`.
- Tie every number in prose to its source: `[0.412](mlsweep:<experiment>/<run>/val_loss@20000)`. The tab marks it ✓ when it matches (⚠ when it does not) and links to the run. `@final`, `@min`, `@max`, `@mean` and `@50%` also work. Lint reports unknown experiments/runs and numbers that disagree with the logged metric; it does not judge the claim.
- Footnotes: `text[^1]` and `[^1]: the note`, continuation lines indented.
- Captions: a `Figure: text` line directly under an image or table. Add `{#fig-name}` to fix the anchor and link to it as `[Figure 1](#fig-name)`. Headings take `{#id}` too.
- Full reference: `mlsweep docs notes`.

## Traces

- A trace is a coding-agent session the manager keeps, linked to the experiments it submitted. `mlsweep capture this` uploads the session you are in and prints its reference. Capture keeps prompts, replies, thinking and a one-line summary of each tool call with its status. It never keeps tool output or file contents.
- `mlsweep capture ls --experiment <id>` lists the sessions behind an experiment. `mlsweep capture search WORDS` finds events across traces and gives each one's reference.
- `mlsweep capture show REF` prints one line per event and stops at about 2000 tokens (`--budget`), then gives the command for the next page. `--kind mlsweep` keeps only the mlsweep commands the session ran, and `user`, `assistant`, `thinking` and `tools` work the same way. `--event N`, or `#EN` on the reference, prints one event in full.
- REF is a trace id, a unique prefix of one, or a note reference like `mlsweep:trace/<id>?kind=mlsweep#E412`. Cite an event in a note with the same reference.
- `mlsweep capture export REF` writes the trace as ATIF, which other agent tools read.

## Reading more

`mlsweep docs <topic>` prints each of these. Search them rather than reading them whole.

- `sweep_configuration`: subdims, `EXCLUDE`, Bayesian sweeps, multi-GPU options.
- `mlsweep`: the full guide, including instrumentation, environment variables and troubleshooting.
- `examples`: DDP, multi-node runs, TorchTitan and Prime-RL.
- `notes`: the Notes tab, its markdown, cloning and lint.
- `traces`: capture, the Traces tab, and reading traces from the terminal.
- `readme`: overview and common CLI flags.
