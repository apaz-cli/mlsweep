# mlsweep

`mlsweep` is slightly opinionated but very general solution for managing tons of machine learning runs. It takes flexible combinations of hyperparameters and schedules them across your hardware.

The project contains a persistent manager that owns GPU scheduling and a web dashboard, and workers that execute training jobs. You aren't forced to use our web UI. You can export mlsweep to wandb or tensorboard, and use whatever you use for viewing. The mlsweep logger is also extensible, should you wish. Logs end up on the manager machine.

The main feature of mlsweep is not the logger, or the cluster managment features, but the sweep configuration file. The good stuff. The thing that I've been missing all my machine learning life, and the reason I wrote this library.

`mlsweep` does pretty much everything that wandb does. If you're missing anything, let me know on [Discord](https://discord.gg/w2K2JWJGUb) or [Twitter](https://twitter.com/apaz_cli).

But first, let's install it, and add the logging.

## Setup

Install mlsweep on the machine that will run the manager:

```sh
pip install 'mlsweep[all]'
```

That's it. Remote workers get mlsweep bootstrapped automatically over SSH, no install needed.

You don't need to run the manager, client, or worker on separate machines. By default they all run on the same machine. The single `mlsweep` command gives you all of them (`mlsweep manager` / `mlsweep run` / `mlsweep worker`), and the individual `mlsweep_manager` / `mlsweep_run` / `mlsweep_worker` binaries still work.

## Add logging to your training script

```python
from mlsweep.logger import MLSweepLogger

# If you don't want to use it as a context manager, remember to call .close().
with MLSweepLogger() as logger:
    for step in range(1, num_steps + 1):
        loss = train_step()
        logger.log({"loss": loss}, step=step)

        # Write checkpoints to MLSWEEP_RUN_DIR, they get rsynced back automatically.
        # Call logger.sync() to trigger an immediate rsync mid-run (fire-and-forget).
        if step % 1000 == 0:
            save_checkpoint(os.environ["MLSWEEP_RUN_DIR"], step)
            logger.sync()
```

`MLSweepLogger` is only active when your script is launched by a worker. It checks for `MLSWEEP_WORKER_SOCKET`, which the worker sets. Run your script directly and it's a no-op. The worker always has mlsweep available (bootstrapped or from your venv), so the logger just works.

If your script doesn't use the logger at all that's fine too, mlsweep still dispatches the job and captures stdout/stderr to `training.log`. You just won't get metrics plots.

Metrics land in `<mlsweep-dir>/experiments/<experiment>/<run>/metrics.jsonl` on the manager (default `~/.mlsweep/experiments/...`). Anything written to `MLSWEEP_RUN_DIR` is rsynced to that run's `artifacts/` at the end of every run, and immediately on `logger.sync()`.

## Write a sweep configuration file

Add the following shebang, and use `chmod +x` so that your sweep file can be directly executable.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIONS = {
    ".lr": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".batch_size": {
        "values": [32, 64, 128],
        "flags": "--training.batch_size",
        "name": "bs",
    },
}
```

Running this produces 9 runs named `my_sweep_lr1e-4_bs32`, `my_sweep_lr1e-4_bs64`, etc.

Each run receives its flags appended to `COMMAND`: `python train.py --optimizer.lr 0.0001 --training.batch_size 32`.

See [sweep_configuration.md](docs/sweep_configuration.md) for the full format: subdimensions, monotonic/singular skipping, `EXCLUDE`, `NODES_PER_RUN` and `GPUS_PER_RUN` for training with `torchrun` (see `SET_DIST_ENV`), and more. For end-to-end examples with real frameworks (Prime-RL, TorchTitan), see [examples.md](docs/examples.md).

## Bayesian optimization

If your sweep is specifically for hyperparameter optimization, you can add an `OPTIMIZE` dict to save compute. It uses TPE (via [optuna](https://github.com/optuna/optuna)) to intelligently sample the space and find good configs faster than trying all combinations.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIMIZE = {
    "method": "bayes",
    "metric": "val_loss",
    "goal": "minimize",
    "budget": 40,
}

OPTIONS = {
    # Discrete dim
    ".optimizer": {
        "name": "opt",
        ".adam": {"flags": ["--optimizer", "adam"]},
        ".muon": {"flags": ["--optimizer", "muon"]},
    },
    # Continuous dims
    ".lr": {
        "distribution": "log_uniform",
        "min": 1e-5,
        "max": 1e-1,
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".wd": {
        "distribution": "log_uniform",
        "min": 0.0,
        "max": 0.2,
        "flags": "--optimizer.weight_decay",
        "name": "wd",
    },
}
```

The manager runs the sweep, choosing each trial from the results so far. So `mlsweep run` can exit, and the sweep survives a manager restart. Watch and adjust it with `mlsweep optimize EXP_ID` (`--budget`, `--finish`, `--max-concurrent`), and control it with `pause`, `stop` and `cancel` like any other sweep.

See [sweep_configuration.md](docs/sweep_configuration.md) for continuous ranges, singular dims, all `OPTIMIZE` fields, and the controls.

## Run

mlsweep uses a manager daemon that owns GPU scheduling and persists state. Start it once, then submit sweeps against it.

### 1. Start the manager

```sh
mlsweep manager                                        # local GPUs, dashboard at http://localhost:7891
mlsweep manager --workers workers.toml                 # remote workers
mlsweep manager --port 7891 --host my.server.com       # custom port and externally-reachable hostname
mlsweep manager --mlsweep-dir /data/mlsweep            # custom state dir (DB, token, experiment outputs)
```

Launching the manager also creates a worker process on localhost, unless --workers is passed.

The manager prints dashboard URLs on startup, including a token for authentication:

```
Dashboard: http://localhost:7891/?token=abc123...
```

The token is also saved to `~/.mlsweep/manager.token` so local workers find it automatically.

### 2. Submit a sweep

```sh
mlsweep run sweeps/my_sweep.py --manager http://localhost:7891 --stream   # submit + live status
mlsweep run sweeps/my_sweep.py --validate                                # print all combos, no submission
mlsweep lint sweeps/my_sweep.py                                          # check for misspelled settings and flag clashes
mlsweep lint                                                             # every sweep file here and every experiment
mlsweep run sweeps/my_sweep.py --dry-run                                 # print commands, no submission
```

Every client command resolves the token the same way, in order: `--token`, then `MLSWEEP_TOKEN`, then `~/.mlsweep/manager.token` (saved there by the manager on startup).

### 3. Monitor, rank, and fetch results

```sh
mlsweep status                                       # manager / token / GPUs / result paths
mlsweep watch EXP_ID                                 # live terminal status
mlsweep ls                                           # experiments grouped by status (`mlsweep ls EXP_ID` lists runs, --all lifts the cap of 20 settled)
mlsweep ls --finished-since 2h                       # runs that finished lately, across the campaign
mlsweep logs RUN_ID --experiment EXP_ID              # tail a run's training.log
mlsweep metrics --experiment EXP_ID --keys REGEX     # runs' logged metrics (--last, --agg, --step, --x/--at, --pivot, --json, --csv)
mlsweep best --experiment EXP_ID                     # top runs by metric (--group-by DIM, --table D1,D2)
mlsweep fetch --experiment EXP_ID --wait             # block until done, then leaderboard + download
```

On the manager's machine, `fetch` makes no copy. It prints the manager's own
results folder instead. Pass `--output-dir` to get a copy anyway.

Every command defaults `--manager` to `http://localhost:7891` (override with
`--manager` or `MLSWEEP_MANAGER`). Add `--json` to `status`, `ls`, `best`, or `fetch` for
machine-readable output.

Control a running sweep with `mlsweep cancel EXP --running`, `mlsweep retry EXP --failed`,
`mlsweep resume EXP`, `mlsweep pause EXP`, `mlsweep unpause EXP`, or `mlsweep stop EXP --yes`.
`cancel`, `retry`, and `resume` exit non-zero if any targeted run fails. `retry` and `resume`
rerun the code submitted with the experiment, so after fixing a bug submit a new sweep.

Generated run IDs get long. `mlsweep rename EXP RUN_ID "warmup ablation"` gives a run a
display name, and `--clear` removes it. The name appears next to the run ID in `ls`, `best`,
and the dashboard. The run ID itself never changes, and every other command keeps taking it.

For a project-local interface, run `mlsweep gen_makefile` in your repo to add `make sweep-run` / `sweep-watch` / `sweep-fetch` / `sweep-status` targets.

### 4. Group experiments into campaigns

A campaign is a named group of experiments. Pick one with `--campaign NAME` or
`MLSWEEP_CAMPAIGN`, and every command works inside it. `run` submits under it,
`ls` and `status` list only its experiments, and commands that take an
experiment ID refuse one filed in another campaign. A context can also set it.
There is no default campaign, so a command that needs one exits 1 without it.
`--all-campaigns` covers all of them.

```sh
export MLSWEEP_CAMPAIGN=tokenizer_ablation
mlsweep run sweeps/vocab.py --manager http://localhost:7891 --stream
mlsweep ls                                   # tokenizer_ablation only
mlsweep ls --all-campaigns                   # everything, tagged with its campaign
mlsweep campaign                             # campaigns with experiment and job counts
mlsweep campaign move EXP_ID other_campaign  # re-file an experiment
```

The dashboard has a campaign selector on every page that lists experiments. See
[Campaigns](docs/sweep_configuration.md#campaigns) for the details.

### Remote workers

The manager installs mlsweep on remote machines automatically over SSH, with no manual setup needed. It builds wheels from the local source at startup, SCPs them to the remote, and installs them into `/tmp/mlsweep_venv/`.

Your code is uploaded with each sweep. Its dependencies are installed on the remote from `pyproject.toml` / `requirements.txt` and cached per machine, unless the run directory or `remote_dir` already has a `.venv` (see Run environment in `mlsweep --help sweep_configuration`).

#### 1. Create a workers.toml

```toml
[[workers]]
host = "user@host1"
remote_dir = "/absolute/path/to/project"
ssh_key = "~/.ssh/id_ed25519"
devices = [0, 1, 2, 3]
jobs = 2
```

|    Field     | Required | Notes |
|--------------|----------|-------|
| `host`       | yes      | SSH target |
| `remote_dir` | yes      | Project root on the remote |
| `ssh_key`    | no       | Path to identity file (`-i`) |
| `pass`       | no       | SSH password (needs `sshpass`); or set `MLSWEEP_SSH_PASS` env var |
| `venv`       | no       | Existing venv to prefer over the auto-bootstrapped one. Accepts a project root, venv root, `bin/` dir, activate script, or python binary. |
| `devices`    | no       | Specific GPU IDs to use (worker CLI: `-g`). Default: all visible. |
| `jobs`       | no       | Max concurrent jobs per GPU on this worker (worker CLI: `-j`). Default 1; set to 0 for unlimited. |
| `port`       | no       | Worker TCP port (default: 7890; `0` = ephemeral). |
| `tunnel`     | no       | `true` to reach the worker through SSH instead of dialing its port, for a machine whose firewall lets only SSH in. Default `false`. |
| `scratch_dir` | no      | Where the worker keeps each run's working files while it runs (default: the manager's `--scratch-dir`, `/tmp/mlsweep`). A second worker given a directory that a live worker already uses takes a `.worker-<port>` subdirectory of it. |

#### 2. Start the manager with the workers file

```sh
mlsweep_manager --workers workers.toml
```

## Dashboard

The manager serves a web dashboard at the URL printed at startup (default `http://localhost:7891`). It shows live metrics, per-run logs, file browser, the code each run used, export, and system status. Open it in a browser while your sweep runs. The Rename buttons on the Experiments and Logs pages set a run's display name, the same one `mlsweep rename` sets.

## Useful CLI flags

All flags below assume `--manager http://localhost:7891`:

| Flag | Effect |
|------|--------|
| `--dry-run` | Print commands without running |
| `--validate` | Check config, list all combos, exit |
| `--stream` | Live status in terminal |
| `--experiment NAME` | Custom experiment name |
| `--campaign NAME` | Campaign to submit under (else the context's, else `$MLSWEEP_CAMPAIGN`; required) |
| `--priority N` | Higher values run sooner (default: 0) |
| `--wandb-project P` | Stream metrics to W&B |
| `--tensorboard-dir D` | Write TensorBoard logs |
| `--color` | Enable ANSI color in human-readable output (default: off) |

Colored output is off by default so piping and logging stay clean. Pass
`--color` to any entry point (`mlsweep`, `mlsweep run`, `mlsweep manager`,
`mlsweep worker`) to turn it on. Machine-readable output (`--json`, `--csv`) and
raw `logs`/`metrics` text are never colorized.

Subcommands (`mlsweep <subcommand>`, or `mlsweep --help` for the full grouped list):

```sh
# run
mlsweep manager / run / worker / lint / gen_makefile

# monitor
mlsweep watch EXP_ID                 mlsweep fetch --experiment EXP_ID
mlsweep best  --experiment EXP_ID    mlsweep ls [EXP_ID]
mlsweep logs RUN_ID --experiment EXP_ID    mlsweep status

# control
mlsweep cancel EXP --running         mlsweep retry EXP --failed
mlsweep resume EXP                   mlsweep pause EXP
mlsweep unpause EXP                  mlsweep stop EXP --yes
mlsweep rename EXP RUN_ID NAME       # display name for a run; --clear removes it

# campaigns (every command also takes --campaign NAME / --all-campaigns)
mlsweep campaign [ls]                mlsweep campaign move EXP NAME

# runs and storage
mlsweep move EXP RUN... --to EXP2    mlsweep purge EXP --failed --yes
mlsweep delete EXP --yes             mlsweep gc [--yes] [--vacuum] [--downloads]
mlsweep export tensorboard|wandb     mlsweep rename EXP --experiment-name NAME
mlsweep rename EXP --description TEXT   # or --clear-description

# cluster (the System page)
mlsweep workers [ls|add|rm|set]      mlsweep scheduler [--placement pack|spread]
mlsweep scheduler --pause [W|all]    mlsweep scheduler --unpause [W|all]
mlsweep reschedule EXP RUN [--to W]  mlsweep history EXP [RUN]

# notes (a shared research notes folder in a git repo the manager hosts; mlsweep docs notes)
mlsweep notes clone DIR              mlsweep notes status
mlsweep notes lint                   mlsweep notes links|mentions|search|history|mv

# capture (coding-agent traces, on the dashboard's Traces tab; mlsweep docs traces)
mlsweep capture this                 mlsweep capture status
mlsweep capture on|off               mlsweep capture ls
mlsweep capture import --days 30

# docs
mlsweep docs [topic]                 mlsweep --help <topic>   # readme, sweep_configuration, mlsweep, examples, notes, traces, skill
```

### Shell completion

```sh
# bash (~/.bashrc)
eval "$(mlsweep completion bash)"

# zsh (~/.zshrc, after `autoload -U compinit && compinit`)
eval "$(mlsweep completion zsh)"

# fish
mlsweep completion fish > ~/.config/fish/completions/mlsweep.fish
```

Completion covers subcommands, each subcommand's flags, sweep files, context
names, and experiment IDs and campaigns from the manager in force. A manager
that is down only means no IDs are offered.

### Using with W&B

mlsweep can log all runs to Weights & Biases with no changes to your training script.

```sh
pip install 'mlsweep[wandb]'
export WANDB_API_KEY=your_key_here
mlsweep run sweeps/my_sweep.py --manager http://localhost:7891 --wandb-project my-project
mlsweep run sweeps/my_sweep.py --manager http://localhost:7891 --wandb-project my-project --wandb-entity my-team
```

### Using with TensorBoard

```sh
pip install 'mlsweep[tensorboard]'
mlsweep run sweeps/my_sweep.py --manager http://localhost:7891 --tensorboard-dir ./tb_logs
tensorboard --logdir ./tb_logs
```

### Exporting finished experiments

The flags above stream metrics while `mlsweep run` is attached. To export afterwards, open the dashboard's Export page. Choose what to export with the toggle at the top. You can pick one experiment, one campaign, or everything on the manager. Each choice can be limited to runs that succeeded.

- **TensorBoard:** downloads a zip with one event file per run, in a folder per experiment. An export of everything adds a folder per campaign above those. It needs no extra packages. Each run's sweep parameters appear in the HParams tab, so a sweep shows up as one table. Unzip it and run `tensorboard --logdir` on the folder.
- **W&B:** the manager uploads each run itself. A run's sweep parameters become its config, its experiment becomes its group, and its campaign becomes a tag. This needs `pip install 'mlsweep[wandb]'` where the manager runs. The API key comes from the form, else from the manager's `WANDB_API_KEY` or its saved `wandb login`. Set `WANDB_MODE=offline` in the manager's environment to write runs locally instead of uploading (sync them later with `wandb sync`). The form's key is sent in the request body, so keep the dashboard on localhost or behind HTTPS when you enter one.

Over HTTP, `GET /api/export/tensorboard.zip` and `POST /api/export/wandb` take the scope as a query parameter. Use `?experiment_id=EXP` for one experiment, `?campaign=NAME` for one campaign, and neither for everything. Add `only_done=1` for succeeded runs only. The W&B body is `{"project": ..., "entity": ..., "api_key": ...}`, with only `project` required. Track an upload's progress at `GET /api/exports/EXPORT_ID`.

## Notes

The dashboard's **Notes** tab is an editor for a shared folder of research notes,
papers and plots: an Obsidian-style file tree, markdown with `[[links]]`, search,
graph, history and diffs. Experiment IDs in a note link to their results, and
`mlsweep-chart` / `mlsweep-table` blocks plot live metrics instead of pasting
screenshots. A number written as `[0.42](mlsweep:EXP/RUN/loss@20000)` is checked
against the logged metric when the note is opened.

The folder is a git repo that the manager hosts. Agents on any machine run
`mlsweep notes clone DIR`, then pull, commit, merge and push with normal git.
`mlsweep notes lint` checks links, experiment and run IDs, chart/table specs, the
numbers behind evidence links, and leftover conflict markers.

The manager serves `~/.mlsweep/notes` (creating it) by default, so the tab and the
remote work out of the box. Math renders with `$…$` and `$$…$$`, and footnotes
with `[^1]`.

See [notes.md](docs/notes.md) (`mlsweep docs notes`).

## Traces

Session capture records the coding-agent conversation next to the runs it
produced. An agent runs `mlsweep capture this` inside its session, and that
session uploads to the manager. The dashboard's Traces tab shows it as a
ledger of your prompts, the agent's replies, its thinking, and tool calls one
line each. Tool output and file contents are not captured. Sessions that never
ask are never read. With `mlsweep capture on`, a session that submits a sweep is
uploaded too.

An experiment records the trace it was submitted from. The session id comes
out of the environment the agent runs its commands with, so the link is exact
per agent context, with no shell hooks and no guessing. A note can cite a
trace as `[that idea](mlsweep:trace/9f6c0e22)`, and `mlsweep capture import`
brings in the transcripts you already have.

See [traces.md](docs/traces.md) (`mlsweep docs traces`).

## Dashboard and command line

Everything the dashboard does, an agent can do from the command line, and the
other way round (watching live is the exception; `mlsweep watch` and
`mlsweep wait` cover it in a terminal):

| Dashboard | Command |
|---|---|
| Experiments: rename, edit description, move campaign, delete | `mlsweep rename EXP --experiment-name N --description TEXT`, `mlsweep campaign move`, `mlsweep delete` |
| Experiments: pause, unpause, stop, retry failed, cancel active, delete failed | `mlsweep pause`, `unpause`, `stop`, `retry --failed`, `cancel --running`, `purge --failed` |
| Runs: rename, cancel, retry, move, delete | `mlsweep rename`, `cancel`, `retry`, `move`, `purge` |
| Results, Logs, Artifacts | `mlsweep metrics`, `mlsweep best`, `mlsweep logs`, `mlsweep fetch` |
| Code: a run's files, where they came from, diffs between versions | `mlsweep code EXP [PATH]`, `mlsweep code diff A B`, `mlsweep code checkout EXP DIR` |
| Export (TensorBoard, W&B) | `mlsweep export tensorboard`, `mlsweep export wandb` |
| System: workers, devices, jobs per GPU, placement, pausing | `mlsweep workers ...`, `mlsweep scheduler --placement`, `mlsweep scheduler --pause` |
| System: storage cleanup | `mlsweep gc` |
| Notes | `mlsweep notes ...` (see [notes.md](docs/notes.md)) |
| Traces | `mlsweep capture this\|on\|off\|status\|ls\|import` (see [traces.md](docs/traces.md)) |

## Running an agent

An agent working from your copy of the code can edit it, submit sweeps, watch them and record
results, on the manager's machine or any other. Three things need setting up first.

Install the skill, and run this again after upgrading mlsweep, since it writes a copy:

```sh
mkdir -p ~/.claude/skills/mlsweep && mlsweep docs skill > ~/.claude/skills/mlsweep/SKILL.md
```

Other agents can read it with `mlsweep docs skill`.

Keep a manager running somewhere the agent's session can't end it, such as tmux or a service.
The agent should use this manager and not start its own, because a manager's worker takes
every visible GPU.

On another machine, tell the agent how to connect:

```sh
export MLSWEEP_MANAGER=http://MANAGER:7891
export MLSWEEP_TOKEN=...                     # from ~/.mlsweep/manager.token on the manager
export MLSWEEP_CAMPAIGN=my_campaign          # optional: keeps this work grouped
```

Then give it the question and its limits. The skill covers the rest:

```
Use the mlsweep skill. Find out which warmup schedule gives the lowest val_loss on the
small model. Use at most 40 runs, 2 at a time (--max-concurrent 2). Stop when
three seeds agree on a winner, or when the budget runs out. Record findings and dead
ends in the notes as you go.
```

Limits are yours to set, and the agent asks if something is unclear. By default the skill lets
an agent stop and cancel its own experiments, and asks it to leave others' alone, to get
permission before changing workers or placement, and to delete or publish only on request.

## Troubleshooting

If the error messages are bad or the docs are confusing, hit me up on [Discord](https://discord.gg/w2K2JWJGUb) or [Twitter](https://twitter.com/apaz_cli).
