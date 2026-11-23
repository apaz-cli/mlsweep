"""Top-level ``mlsweep`` command — one entry point for the whole workflow.

See ``_build_help`` for the subcommand list. This module is intentionally light at import time; heavy subcommands are
imported lazily so ``mlsweep --help`` stays fast.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from mlsweep._colors import (
    _BOLD, _BRIGHT_BLUE, _BRIGHT_GREEN, _CYAN, _DIM, _GREEN, _RED,
    _RESET, _YELLOW, set_color, strip_color_flag,
)
from mlsweep._jsonout import emit
from mlsweep._shared import _mlsweep_dir

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_width(s: str) -> int:
    """Visible width of *s* with ANSI escape sequences removed."""
    return len(_ANSI_RE.sub("", s))


def _pad(s: str, width: int) -> str:
    """Right-pad *s* to visible width *width*."""
    return s + " " * max(0, width - _visible_width(s))


def _align(rows: list[tuple[str, str]], indent: str = "  ") -> str:
    """Two-column table: left column padded to the widest, two spaces, right column."""
    width = max(_visible_width(left) for left, _ in rows)
    return "\n".join((f"{indent}{_pad(left, width)}  {right}").rstrip() for left, right in rows)


def _align3(rows: list[tuple[str, str, str]], indent: str = "  ") -> str:
    """Three-column table: command / argument / description, all aligned."""
    cw = max(_visible_width(c) for c, _, _ in rows)
    aw = max(_visible_width(a) for _, a, _ in rows)
    return "\n".join((f"{indent}{_pad(c, cw)}  {_pad(a, aw)}  {d}").rstrip() for c, a, d in rows)


_LABEL_WIDTH = 10
_CONT = " " * (2 + _LABEL_WIDTH + 2)


def _label(label: str) -> str:
    """Left-align a `key:` label to the shared label column."""
    return f"{_BOLD}{_pad(label, _LABEL_WIDTH)}{_RESET}  "


# ── banner ─────────────────────────────────────────────────────────────────────
#
# The banner is shown only for a bare `mlsweep` invocation.  Colors resolve at
# call time via the global toggle, so `_build_banner` must not be cached.

_BANNER_ART = r"""           ___
          /\_ \
  ___ ___ \//\ \     ____  __  __  __     __     __   _____
/' __` __`\ \ \ \   /',__\/\ \/\ \/\ \  /'__`\ /'__`\/\ '__`\
/\ \/\ \/\ \ \_\ \_/\__, `\ \ \_/ \_/ \/\  __//\  __/\ \ \L\ \
\ \_\ \_\ \_\/\____\/\____/\ \___x___/'\ \____\ \____\\ \ ,__/
 \/_/\/_/\/_/\/____/\/___/  \/__//__/   \/____/\/____/ \ \ \/"""
_BANNER_TAIL = r"""                                                        \ \_\
                                                         \/_/"""
_BANNER_TAGLINE = "      an experiment sweep engine"


def _build_banner() -> str:
    """The figlet banner, colored only when the global color toggle is on.

    The tagline sits on the first descender row, to the left of the final
    "p"'s tail, so the art stays connected while the tagline is inside it.
    """
    tail_lines = _BANNER_TAIL.splitlines()
    first = tail_lines[0]
    indent = len(first) - len(first.lstrip())
    tagline_row = (
        f"{_BRIGHT_BLUE}{_BANNER_TAGLINE}"
        f"{_BRIGHT_GREEN}{' ' * max(0, indent - len(_BANNER_TAGLINE))}{first.lstrip()}"
    )
    rest = "\n".join(tail_lines[1:])
    return (f"{_BRIGHT_GREEN}{_BANNER_ART}\n"
            f"{tagline_row}\n"
            f"{_BRIGHT_GREEN}{rest}{_RESET}")


def _build_help(include_banner: bool = False) -> str:
    workflow = [
        (f"1. {_GREEN}mlsweep manager{_RESET}", "start the persistent manager (once)"),
        (f"2. {_GREEN}mlsweep run my_sweep.py{_RESET}", "submit a sweep (--validate first)"),
        (f"3. {_GREEN}mlsweep wait <experiment_id>{_RESET}", "block until it settles (or `watch` for live status)"),
        (f"4. {_GREEN}mlsweep best --experiment <id>{_RESET}", "rank the runs (`fetch` also downloads them)"),
        (f"5. {_GREEN}mlsweep notes clone DIR{_RESET}", "record what you learned in the shared notes"),
        (f"   {_GREEN}mlsweep status{_RESET}", "diagnose manager / token / GPUs / result paths"),
    ]
    sections = [
        ("Run", [
            ("manager [args...]", "start the manager daemon (mlsweep_manager)"),
            ("run SWEEP.py [flags]", "submit a sweep (mlsweep_run)"),
            ("lint SWEEP.py...", "check sweep files for misspelled settings and flag clashes"),
            ("worker [args...]", "start a worker daemon (mlsweep_worker)"),
            ("gen_makefile", "write a standard Makefile into the current directory"),
        ]),
        ("Monitor", [
            ("watch EXP_ID", "live status for an experiment"),
            ("fetch", "download + summarize an experiment's results (leaderboard)"),
            ("best", "top runs by metric"),
            ("wait EXP", "block until done/failed/stalled (exit codes in --help)"),
            ("status | doctor", "diagnose the environment"),
            ("ls [EXP]", "list experiments / runs"),
            ("logs RUN", "tail a run's training.log"),
            ("metrics", "print runs' logged metrics as tables (--keys, --json, --csv)"),
            ("export tensorboard|wandb", "export finished runs (one experiment, a campaign, or all)"),
            ("code EXP [PATH]", "the code an experiment ran (diff A B, checkout EXP DIR)"),
        ]),
        ("Control", [
            ("cancel EXP", "cancel runs (--running/--pending/--all)"),
            ("retry EXP", "re-queue failed runs"),
            ("reschedule EXP RUN [--to W]", "restart a run elsewhere"),
            ("rename EXP RUN NAME", "give a run a display name (--clear; --experiment-name NAME)"),
            ("move EXP RUN... --to EXP2", "move finished runs to another experiment"),
            ("purge EXP --failed --yes", "delete finished runs with their metrics, logs and files"),
            ("gc [--yes]", "clean up files and records nothing refers to"),
            ("delete EXP --yes", "delete an experiment and its runs"),
            ("resume EXP", "re-queue failed and cancelled runs (Bayesian: choose new trials)"),
            ("optimize EXP", "show or adjust a Bayesian sweep (--budget, --finish, --max-concurrent)"),
            ("controls EXP", "show or change run controls (--max-runtime, --prune, --metric-check)"),
            ("stop EXP", "abort a sweep"),
            ("pause EXP", "stop dispatching new jobs"),
            ("unpause EXP", "undo pause"),
        ]),
        ("Cluster", [
            ("workers [ls|add|rm|set]", "list, connect, remove and tune workers"),
            ("scheduler [--placement P]", "show or set GPU placement (pack | spread)"),
        ]),
        ("Campaigns", [
            ("campaign [ls]", "list campaigns with experiment and job counts"),
            ("campaign move EXP NAME", "move an experiment to another campaign"),
        ]),
        ("Notes", [
            ("notes clone DIR", "clone the shared research notes from the manager, then use git"),
            ("notes lint [DIR]", "check links, experiment IDs and conflict markers"),
        ]),
        ("Capture", [
            ("capture [this|on|off]", "upload this agent session (or sweep-submitting ones) to the Traces tab"),
            ("capture [status|ls|import]", "capture state; traces on the manager; backfill old transcripts"),
            ("capture show REF", "a trace as a compact ledger (--kind, --event N, --budget)"),
            ("capture search WORDS", "events across traces, each with its note reference"),
            ("capture export REF", "a trace as ATIF (or --format jsonl)"),
        ]),
        ("Database", [
            ("sql [\"SELECT …\"]", "read-only SQL over the manager's database; no query lists the tables"),
        ]),
        ("Client", [
            ("context [ls|add|use|rm|show]", "named manager/token/campaign settings (--context NAME)"),
            ("completion bash|zsh|fish", "print a shell completion script"),
        ]),
        ("Docs", [
            ("docs [topic]", "read the bundled runbook (README by default)"),
            ("version", "print the version"),
        ]),
    ]
    lines = []
    if include_banner:
        lines += [_build_banner(), ""]
    lines += [
        f"{_BOLD}Workflow:{_RESET}",
        _align(workflow),
        "",
    ]
    for header, rows in sections:
        lines.append(f"{_BOLD}{_CYAN}{header}:{_RESET}")
        lines.append(_align(rows))
        lines.append("")
    lines += [
        "Every command defaults --manager to $MLSWEEP_MANAGER, then http://localhost:7891,",
        "unless a context says otherwise (--context NAME, $MLSWEEP_CONTEXT, `mlsweep context use`).",
        "Each command works in the campaign named by --campaign, else the context's campaign, else",
        "$MLSWEEP_CAMPAIGN. One is required. --all-campaigns covers every campaign (not allowed",
        "when submitting a new sweep).",
        f"Results live on the manager at ~/.mlsweep/experiments/<experiment_id>/<run>/",
        f"{_DIM}Global: --color enables ANSI color (off by default).{_RESET}",
        f"Docs: {_CYAN}mlsweep --help <topic>{_RESET} (readme, sweep_configuration, mlsweep, examples, notes, traces, skill)",
        f"      or {_CYAN}mlsweep docs{_RESET}. {_CYAN}mlsweep gen_makefile{_RESET} adds {_CYAN}make sweep-*{_RESET} targets.",
    ]
    return "\n".join(lines) + "\n"


_DOC_ALIASES = {
    "readme": "README.md",
    "index": "README.md",
    "sweep_configuration": "sweep_configuration.md",
    "sweep_config": "sweep_configuration.md",
    "config": "sweep_configuration.md",
    "configuration": "sweep_configuration.md",
    "sweeps": "sweep_configuration.md",
    "mlsweep": "mlsweep.md",
    "skill": "SKILL.md",
    "examples": "examples.md",
    "notes": "notes.md",
    "traces": "traces.md",
    "capture": "traces.md",
    "sessions": "traces.md",
}

def _build_makefile_template() -> str:
    help_rows = [
        ("make sweep-manager", "", "start the manager daemon"),
        ("make sweep-status", "", "check manager health, GPUs, token"),
        ("make sweep-validate", "S=<path>", "validate a sweep file"),
        ("make sweep-dry-run", "S=<path>", "print commands without running"),
        ("make sweep-run", "S=<path>", "submit + stream live status"),
        ("make sweep-watch", "E=<id>", "watch an experiment"),
        ("make sweep-fetch", "E=<id>", "fetch results + leaderboard"),
        ("make sweep-best", "E=<id>", "top runs by metric"),
        ("make sweep-ls", "E=<id>", "list experiments / runs"),
        ("make sweep-logs", "R=<id> E=<id>", "tail a run's training.log"),
        ("make sweep-cancel", "E=<id>", "cancel running runs"),
        ("make sweep-retry", "E=<id>", "re-queue failed runs"),
        ("make sweep-resume", "E=<id>", "re-queue failed and cancelled runs"),
        ("make sweep-stop", "E=<id>", "abort a sweep (destructive)"),
        ("make sweep-docs", "", "read the full runbook"),
    ]
    help_block = "\n".join(f'\t@echo "{line}"' for line in _align3(help_rows).split("\n"))
    return (
        "# Generated by `mlsweep gen_makefile`. Regenerate with `mlsweep gen_makefile --force`.\n"
        "# This is the standard mlsweep project interface. See `mlsweep docs` for the full runbook.\n"
        "\n"
        "MLSWEEP_MANAGER ?= http://localhost:7891\n"
        "S ?=\n"
        "E ?=\n"
        "R ?=\n"
        "\n"
        ".PHONY: help sweep-manager sweep-status sweep-docs sweep-validate sweep-dry-run sweep-run sweep-watch sweep-fetch sweep-best sweep-ls sweep-logs sweep-cancel sweep-retry sweep-resume sweep-stop sweep-gen\n"
        "\n"
        "help:\n"
        '\t@echo "mlsweep targets:"\n'
        f"{help_block}\n"
        "\n"
        "sweep-manager:\n"
        "\tmlsweep manager\n"
        "\n"
        "sweep-status:\n"
        "\tmlsweep status\n"
        "\n"
        "sweep-docs:\n"
        "\tmlsweep docs\n"
        "\n"
        "sweep-validate:\n"
        "\tmlsweep run $(S) --manager $(MLSWEEP_MANAGER) --validate\n"
        "\n"
        "sweep-dry-run:\n"
        "\tmlsweep run $(S) --manager $(MLSWEEP_MANAGER) --dry-run\n"
        "\n"
        "sweep-run:\n"
        "\tmlsweep run $(S) --manager $(MLSWEEP_MANAGER) --stream\n"
        "\n"
        "sweep-watch:\n"
        "\tmlsweep watch $(E) --manager $(MLSWEEP_MANAGER)\n"
        "\n"
        "sweep-fetch:\n"
        "\tmlsweep fetch --manager $(MLSWEEP_MANAGER) --experiment $(E)\n"
        "\n"
        "sweep-best:\n"
        "\tmlsweep best --manager $(MLSWEEP_MANAGER) --experiment $(E)\n"
        "\n"
        "sweep-ls:\n"
        "\tmlsweep ls $(E) --manager $(MLSWEEP_MANAGER)\n"
        "\n"
        "sweep-logs:\n"
        "\tmlsweep logs $(R) --experiment $(E) --manager $(MLSWEEP_MANAGER)\n"
        "\n"
        "sweep-cancel:\n"
        "\tmlsweep cancel $(E) --running --manager $(MLSWEEP_MANAGER)\n"
        "\n"
        "sweep-retry:\n"
        "\tmlsweep retry $(E) --failed --manager $(MLSWEEP_MANAGER)\n"
        "\n"
        "sweep-resume:\n"
        "\tmlsweep resume $(E) --manager $(MLSWEEP_MANAGER)\n"
        "\n"
        "sweep-stop:\n"
        "\tmlsweep stop $(E) --yes --manager $(MLSWEEP_MANAGER)\n"
        "\n"
        "sweep-gen:\n"
        "\tmlsweep gen_makefile --force\n"
    )


# ── status / doctor ────────────────────────────────────────────────────────────


def _status_cmd(argv: list[str]) -> None:
    import shutil
    from importlib.metadata import version as _pkg_version
    from mlsweep.run_sweep import (
        _add_manager_args,
        _http_request,
        _manager_url,
        _selected_campaign,
        _resolve_token,
        manager_list_experiments,
        manager_list_workers,
    )
    from mlsweep.worker import _query_gpu_stats

    parser = argparse.ArgumentParser(
        prog="mlsweep status",
        description="Check the mlsweep environment: manager, token, GPUs, results.",
    )
    _add_manager_args(parser)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    manager = args.manager.rstrip("/")
    token = _resolve_token(args.token)
    # A diagnostic, so it runs without a campaign and then counts every experiment.
    campaign = None if args.all_campaigns else _selected_campaign(args)
    mlsweep_dir = _mlsweep_dir()
    token_file = mlsweep_dir / "manager.token"
    local_version = _pkg_version("mlsweep")

    status, resp = _http_request("GET", _manager_url(manager, "/api/health"), token, timeout=10)
    health = resp if isinstance(resp, dict) else {}
    manager_up = status not in (0, 401)
    manager_version = health.get("version")

    gpus = _query_gpu_stats()
    disk = shutil.disk_usage(mlsweep_dir)

    exps = (manager_list_experiments(manager, token, campaign=campaign, quiet=True) or []) if manager_up else []
    experiments = [e["experiment_id"] for e in exps if isinstance(e, dict) and e.get("experiment_id")]
    workers = (manager_list_workers(manager, token) or []) if manager_up else []

    if args.json:
        emit({
            "manager": manager,
            "campaign": campaign,
            "results_dir": str(mlsweep_dir / "experiments"),
            "dashboard": f"{manager}/?token=<token>",
            "mlsweep_dir": str(mlsweep_dir),
            "local_version": local_version,
            "manager_up": manager_up,
            "manager_status": status,
            "manager_version": manager_version,
            "workers_connected": health.get("workers_connected"),
            "jobs_pending": health.get("jobs_pending"),
            "jobs_in_flight": health.get("jobs_in_flight"),
            "token_present": bool(token),
            "gpus": gpus,
            "workers": workers,
            "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
            "recent_experiments": experiments[-5:],
            "n_experiments": len(experiments),
        })
        return

    print(f"{_BOLD}{_CYAN}mlsweep status{_RESET}")
    print(f"  {_label('manager:')}{manager}")
    print(f"  {_label('campaign:')}"
          + (campaign or ("all (--all-campaigns)" if args.all_campaigns
                          else "none given (pass --campaign or set $MLSWEEP_CAMPAIGN)")))
    print(f"  {_label('results:')}{mlsweep_dir / 'experiments'} (on the manager's machine; "
          "read them with metrics, logs, fetch)")
    print(f"  {_label('dashboard:')}{manager}/?token=<token>")
    print()

    if status == 0:
        print(f"  {_label('manager:')}{_RED}DOWN{_RESET}, no response from {manager}")
        print(f"{_CONT}wrong URL? set --manager or MLSWEEP_MANAGER")
        print(f"{_CONT}to host one on this machine: {_GREEN}mlsweep manager{_RESET} (its worker takes all visible GPUs)")
    elif status == 401:
        print(f"  {_label('manager:')}{_YELLOW}UP{_RESET}, but rejected the token (401)")
        if not token:
            print(f"{_CONT}no token found; expected at {token_file}")
        else:
            print(f"{_CONT}token present but wrong; check {token_file}")
    else:
        print(f"  {_label('manager:')}{_GREEN}UP{_RESET}  workers={health.get('workers_connected', '?')}"
              f"  jobs_pending={health.get('jobs_pending', '?')}  jobs_in_flight={health.get('jobs_in_flight', '?')}")

    if token:
        print(f"  {_label('token:')}present ({len(token)} chars)")
    else:
        print(f"  {_label('token:')}{_RED}missing{_RESET}, set MLSWEEP_TOKEN or write {token_file}")

    ver = local_version
    if manager_version and manager_version != local_version:
        ver = f"{_YELLOW}{local_version} (manager {manager_version}){_RESET}"
    print(f"  {_label('version:')}{ver}")

    print(f"  {_label('disk:')}{disk.free / 1e9:.1f}G free of {disk.total / 1e9:.1f}G on {mlsweep_dir}")

    if gpus:
        print(f"  {_label('gpus here:')}{len(gpus)} visible on this machine (cluster: see workers)")
        for g in gpus:
            print(f"{_CONT}gpu{g['gpu']}  util={g['util_pct']}%  mem={g['mem_used_mb']}/{g['mem_total_mb']} MiB")
    else:
        print(f"  {_label('gpus here:')}{_YELLOW}none visible{_RESET} on this machine (cluster: see workers)")

    if manager_up and workers:
        print(f"  {_label('workers:')}{len(workers)}")
        for w in workers:
            host = w.get("host") or w.get("worker_id", "?")
            wstatus = w.get("status", "?")
            wgpus = w.get("gpus") or []
            bad = w.get("unhealthy_gpus") or []
            pending = w.get("pending_gpus") or []
            line = f"{_CONT}{host}  {wstatus}  gpus={wgpus}  jobs/GPU={w.get('max_jobs_per_gpu', '?')}"
            if bad:
                line += f"  {_YELLOW}unhealthy={bad}{_RESET}"
            if pending:
                line += f"  probing={pending}"
            print(line)
            reasons = w.get("unhealthy_reasons") or {}
            for why in sorted(set(reasons.values())):
                gs = sorted(int(g) for g, r in reasons.items() if r == why)
                print(f"{_CONT}  {_YELLOW}GPU{'s' if len(gs) > 1 else ''} {gs}: {why}{_RESET}")

    if manager_up and experiments:
        print(f"  {_label('recent:')}{len(experiments)} experiments; latest:")
        for eid in experiments[-5:]:
            print(f"{_CONT}{eid}")
    elif manager_up:
        print(f"  {_label('recent:')}none")

    print()
    print(f"{_BOLD}{_CYAN}Quick start:{_RESET}")
    print("  mlsweep manager")
    print(f"  mlsweep run sweeps/<file>.py --manager {manager} --stream")
    print(f"  mlsweep watch <experiment_id> --manager {manager}")
    print(f"  mlsweep fetch --manager {manager} --experiment <id>")
    print("  (or: mlsweep gen_makefile, then `make sweep-run S=sweeps/<file>.py`)")


# ── docs ───────────────────────────────────────────────────────────────────────


def _docs_cmd(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mlsweep docs",
        description="Read the bundled mlsweep runbook.",
    )
    parser.add_argument("topic", nargs="?", default="README", help="Doc to show")
    parser.add_argument("--list", action="store_true", help="List available docs")
    args = parser.parse_args(argv)

    from importlib.resources import files

    docs_dir = files("mlsweep") / "docs"

    def print_available() -> None:
        print("Available docs (mlsweep docs <name>):")
        for name in sorted(p.name for p in docs_dir.iterdir() if p.name.endswith(".md")):
            print(f"  {name}")

    if args.list:
        print_available()
        return

    key = args.topic.lower().lstrip("./")
    filename = _DOC_ALIASES.get(key, key if key.endswith(".md") else key + ".md")
    target = docs_dir / filename
    if not target.is_file():
        print(f"{_RED}Unknown doc: {args.topic}{_RESET}")
        print_available()
        sys.exit(1)

    text = target.read_text()
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


# ── gen_makefile ───────────────────────────────────────────────────────────────


def _gen_makefile_cmd(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mlsweep gen_makefile",
        description="Write a standard mlsweep Makefile into the current directory.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing Makefile")
    parser.add_argument("--dry-run", action="store_true", help="Print the Makefile instead of writing")
    args = parser.parse_args(argv)

    target = Path.cwd() / "Makefile"
    if args.dry_run:
        sys.stdout.write(_build_makefile_template())
        return

    if target.exists() and not args.force:
        print(f"{_RED}{target} already exists.{_RESET} Use --force to overwrite.")
        sys.exit(1)

    target.write_text(_build_makefile_template())
    print(f"{_GREEN}Wrote {target}{_RESET}")
    print("Now you can use:")
    print(_align([
        ("make sweep-run", "S=sweeps/<file>.py"),
        ("make sweep-watch", "E=<experiment_id>"),
        ("make sweep-fetch", "E=<experiment_id>"),
        ("make sweep-status", ""),
    ]))


# ── lint ───────────────────────────────────────────────────────────────────────


def _lint_cmd(argv: list[str]) -> None:
    from mlsweep._lint import lint_file

    parser = argparse.ArgumentParser(
        prog="mlsweep lint",
        description="Check sweep files for mistakes that would otherwise pass silently: misspelled "
                    "settings, unknown OPTIMIZE/PRUNE keys, flags passed twice, colliding run names. "
                    "`mlsweep run` runs the same checks and refuses a file with errors.",
    )
    parser.add_argument("sweep_files", nargs="+", metavar="SWEEP.py")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on warnings too")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    report = {path: lint_file(path) for path in args.sweep_files}
    if args.json:
        emit({p: [{"severity": f.severity, "message": f.message} for f in fs]
                          for p, fs in report.items()})
    else:
        for path, findings in report.items():
            if not findings:
                print(f"{_GREEN}ok{_RESET}     {path}")
            for f in findings:
                color = _RED if f.severity == "error" else _YELLOW
                print(f"{color}{f.severity}{_RESET}  {path}: {f.message}")
    bad = {"error", "warning"} if args.strict else {"error"}
    sys.exit(1 if any(f.severity in bad for fs in report.values() for f in fs) else 0)


# ── context ────────────────────────────────────────────────────────────────────


def _context_cmd(argv: list[str]) -> None:
    from mlsweep import _context as cx

    parser = argparse.ArgumentParser(
        prog="mlsweep context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Named contexts: a manager URL, where its token is, and a campaign, so switching between "
            "managers is one flag (--context NAME) or one command (context use NAME).\n\n"
            f"They are kept in {cx.config_path()} ($MLSWEEP_CONFIG to move it). A token is never "
            "written there: --token-file or --token-env says where to read it.\n\n"
            "Precedence: flags, then --context / $MLSWEEP_CONTEXT, then $MLSWEEP_MANAGER / $MLSWEEP_TOKEN / "
            "$MLSWEEP_CAMPAIGN, then the current context (skipped when $MLSWEEP_MANAGER is set)."
        ),
    )
    as_json = argparse.ArgumentParser(add_help=False)
    as_json.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Emit JSON")
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("list", aliases=["ls"], parents=[as_json], help="List contexts (the default action)")
    show = sub.add_parser("show", parents=[as_json],
                          help="Show one context (default: the one in force) and what it resolves to")
    show.add_argument("name", nargs="?")
    add = sub.add_parser("add", help="Add or replace a context")
    add.add_argument("name")
    add.add_argument("--manager", required=True, help="Manager URL, e.g. http://lab-box:7891")
    tok = add.add_mutually_exclusive_group()
    tok.add_argument("--token-file", help="File holding the manager token")
    tok.add_argument("--token-env", metavar="VAR", help="Environment variable holding the manager token")
    add.add_argument("--campaign", help="Campaign to work in")
    add.add_argument("--use", action="store_true", help="Make it the current context")
    use = sub.add_parser("use", help="Make a context current (--none: no current context)")
    use.add_argument("name", nargs="?")
    use.add_argument("--none", action="store_true")
    rm = sub.add_parser("rm", aliases=["remove"], help="Remove a context")
    rm.add_argument("name")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    try:
        config = cx.load_config()
        found = cx.contexts(config)
        action = args.action or "list"
        if action in ("list", "ls"):
            current = config.get("current")
            if args.json:
                emit({"config": str(cx.config_path()), "current": current,
                                  "contexts": [c.as_dict() for c in found.values()]})
                return
            if not found:
                print(f"No contexts in {cx.config_path()}. Add one: mlsweep context add NAME --manager URL")
                return
            for name, c in sorted(found.items()):
                mark = f"{_GREEN}*{_RESET}" if name == current else " "
                extra = f"  campaign={c.campaign}" if c.campaign else ""
                print(f"{mark} {_BOLD}{name}{_RESET}  {c.manager or '(default manager)'}{extra}")
            return
        if action == "show":
            ctx = cx.get_context(args.name, config) if args.name else cx.active_context()[0]
            if ctx is None:
                print("No context in force (none current, or $MLSWEEP_MANAGER is set).")
                return
            view = {**ctx.as_dict(), "token": "present" if ctx.token() else "missing"}
            if args.json:
                emit(view)
            else:
                print("\n".join(f"{k}: {v}" for k, v in view.items()))
            return
        if action == "add":
            cx.validate_context_name(args.name)
            if args.campaign:
                from mlsweep._shared import validate_campaign
                validate_campaign(args.campaign)
            entry = {k: v for k, v in (("manager", args.manager.rstrip("/")), ("token_file", args.token_file),
                                       ("token_env", args.token_env), ("campaign", args.campaign)) if v}
            if args.token_file:
                entry["token_file"] = str(Path(args.token_file).expanduser().resolve())
            config.setdefault("contexts", {})[args.name] = entry
            if args.use:
                config["current"] = args.name
            path = cx.save_config(config)
            print(f"{_GREEN}Saved{_RESET} context {args.name} in {path}" + (" (current)" if args.use else ""))
            if not (args.token_file or args.token_env):
                print(f"{_DIM}No token source given: it uses $MLSWEEP_TOKEN or ~/.mlsweep/manager.token.{_RESET}")
            return
        if action == "use":
            if args.none:
                config.pop("current", None)
            elif args.name:
                cx.get_context(args.name, config)
                config["current"] = args.name
            else:
                parser.error("context use needs a NAME (or --none)")
            cx.save_config(config)
            print(f"Current context: {config.get('current') or 'none'}")
            return
        if action in ("rm", "remove"):
            cx.get_context(args.name, config)
            del config["contexts"][args.name]
            if config.get("current") == args.name:
                config.pop("current")
            cx.save_config(config)
            print(f"Removed context {args.name}")
            return
    except (cx.ContextError, ValueError) as e:
        print(f"{_RED}Error: {e}{_RESET}", file=sys.stderr)
        sys.exit(1)


# ── dispatch ───────────────────────────────────────────────────────────────────


# Subcommands implemented in mlsweep.ctl as ``<name>_cmd(argv)``.
_CTL_CMDS = frozenset({
    "ls", "logs", "metrics", "cancel", "retry", "rename", "resume", "optimize", "stop", "pause", "unpause", "best",
    "wait", "campaign", "move", "purge", "gc", "delete", "export", "workers", "scheduler", "reschedule",
    "code", "controls",
})


def _forward(prog: str, fn: Callable[[], None], argv: list[str]) -> None:
    sys.argv = [prog, *argv]
    fn()


def main(argv: list[str] | None = None) -> None:
    from mlsweep._jsonout import contract

    if argv is None:
        argv = sys.argv[1:]

    # The unified CLI dispatches to subcommands that parse their own argv, so
    # consume the global --color flag here and forward the rest untouched.
    argv = strip_color_flag(list(argv))
    with contract(argv):
        _dispatch(argv)


def _dispatch(argv: list[str]) -> None:

    if not argv:
        print(_build_help(include_banner=True))
        return

    if argv[0] in ("-h", "--help", "help"):
        if len(argv) > 1:
            _docs_cmd(argv[1:])
        else:
            print(_build_help())
        return

    cmd, rest = argv[0], argv[1:]

    if cmd in ("-v", "--version", "version"):
        from importlib.metadata import version
        print(version("mlsweep"))
        return

    if cmd == "manager":
        from mlsweep.manager import main as _m
        _forward("mlsweep manager", _m, rest)
        return

    if cmd == "worker":
        from mlsweep.worker import main as _w
        _forward("mlsweep worker", _w, rest)
        return

    if cmd == "run":
        from mlsweep.run_sweep import main as _r
        _forward("mlsweep run", _r, rest)
        return

    if cmd == "watch":
        from mlsweep.run_sweep import _watch_cmd
        _watch_cmd(rest, prog="mlsweep watch")
        return

    if cmd == "fetch":
        from mlsweep.run_sweep import _fetch_cmd
        _fetch_cmd(rest, prog="mlsweep fetch")
        return

    if cmd in ("status", "doctor"):
        _status_cmd(rest)
        return

    if cmd == "docs":
        _docs_cmd(rest)
        return

    if cmd == "lint":
        _lint_cmd(rest)
        return

    if cmd == "context":
        _context_cmd(rest)
        return

    if cmd == "completion":
        from mlsweep._completion import completion_cmd
        completion_cmd(rest)
        return

    if cmd == "__complete":
        from mlsweep._completion import complete_cmd
        complete_cmd(rest)
        return

    if cmd == "gen_makefile":
        _gen_makefile_cmd(rest)
        return

    if cmd == "notes":
        from mlsweep.notes_cli import main as _notes_main
        _notes_main(rest)
        return

    if cmd == "capture":
        from mlsweep._capture import _cmd as _capture_main
        _capture_main(rest)
        return

    if cmd == "sql":
        from mlsweep._sql_cli import main as _sql_main
        _sql_main(rest)
        return

    if cmd in _CTL_CMDS:
        import signal

        import mlsweep.ctl
        # Exit quietly when piped into `head` and the like.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        getattr(mlsweep.ctl, f"{cmd}_cmd")(rest)
        return

    print(f"{_RED}Unknown subcommand: {cmd}{_RESET}\n")
    print(_build_help())
    sys.exit(1)
