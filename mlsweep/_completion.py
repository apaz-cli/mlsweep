"""Shell completion for ``mlsweep``.

``mlsweep completion bash|zsh|fish`` prints a small script that asks
``mlsweep __complete -- WORD...`` for candidates, so the logic lives here once
for every shell.  The words are what follows ``mlsweep`` on the command line;
the last one is the word being completed (empty for a new word).

Experiment IDs and campaign names come from the manager, with a short timeout,
so completion never hangs on a manager that is down.  Flags come from each
subcommand's own ``--help``, so they stay correct as flags change.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
from pathlib import Path

COMMANDS = (
    "manager", "run", "worker", "lint", "gen_makefile", "watch", "fetch", "best", "wait", "status", "doctor",
    "ls", "logs", "metrics", "export", "code", "cancel", "retry", "reschedule", "history", "rename", "move", "purge", "gc",
    "delete", "resume", "optimize", "controls", "stop", "pause", "unpause", "workers", "scheduler", "campaign", "notes",
    "capture", "sql", "context", "completion", "docs", "version", "help",
)

# Subcommands whose first positional argument is an experiment ID.
_EXPERIMENT_FIRST = frozenset({
    "watch", "wait", "ls", "cancel", "retry", "reschedule", "history", "rename", "move", "purge", "delete", "resume",
    "optimize", "controls", "stop", "pause", "unpause", "code", "logs",
})

_SUBCOMMANDS = {
    "context": ("list", "show", "add", "use", "rm"),
    "completion": ("bash", "zsh", "fish"),
    "workers": ("ls", "add", "rm", "forget", "set", "update"),
    "campaign": ("ls", "move"),
    "export": ("tensorboard", "wandb"),
    "notes": ("clone", "lint"),
    "capture": ("this", "on", "off", "status", "ls", "show", "search", "export", "import"),
}

_DOC_TOPICS = ("readme", "sweep_configuration", "mlsweep", "examples", "notes", "traces", "skill")

_FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")

# Flags whose value is not something to complete from the manager.
_TAKES_VALUE = frozenset({"--manager", "--token", "--note", "--description", "--experiment-name", "--priority", "--timeout", "--interval"})


def _flags(cmd: str) -> list[str]:
    """The flags *cmd* accepts, read from its ``--help``."""
    from mlsweep.cli import main

    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            main([cmd, "--help"])
        except SystemExit:
            pass
        except Exception:
            return []
    return sorted(set(_FLAG_RE.findall(out.getvalue())) | {"--help"})


def _manager_get(path: str) -> object:
    """GET *path* from the manager the current settings point at, or None."""
    import argparse
    import json
    from urllib.request import Request, urlopen

    from mlsweep.run_sweep import _add_manager_args, _resolve_token

    parser = argparse.ArgumentParser(add_help=False)
    _add_manager_args(parser)
    args, _ = parser.parse_known_args(_connection_words)
    token = _resolve_token(args.token)
    req = Request(args.manager.rstrip("/") + path, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urlopen(req, timeout=2) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


_connection_words: list[str] = []  # --manager/--token/--context/--campaign already typed


def _experiments() -> list[str]:
    from mlsweep.run_sweep import _add_manager_args, _selected_campaign
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    _add_manager_args(parser)
    args, _ = parser.parse_known_args(_connection_words)
    campaign = None if getattr(args, "all_campaigns", False) else _selected_campaign(args)
    data = _manager_get("/api/experiments" + (f"?campaign={campaign}" if campaign else ""))
    if not isinstance(data, list):
        return []
    return [str(e["experiment_id"]) for e in data if isinstance(e, dict) and e.get("experiment_id")]


def _campaigns() -> list[str]:
    data = _manager_get("/api/campaigns")
    if not isinstance(data, list):
        return []
    return [str(c.get("campaign") or c.get("name")) for c in data if isinstance(c, dict)
            and (c.get("campaign") or c.get("name"))]


def _contexts() -> list[str]:
    from mlsweep._context import ContextError, contexts, load_config

    try:
        return sorted(contexts(load_config()))
    except ContextError:
        return []


def _paths(cur: str, suffix: str) -> list[str]:
    """Files ending in *suffix*, and directories, that start with *cur*."""
    head, _, partial = cur.rpartition("/")
    base = Path(head or ".") if head or not cur.startswith("/") else Path("/")
    prefix = f"{head}/" if head else ("/" if cur.startswith("/") and not head else "")
    try:
        entries = list(base.expanduser().iterdir())
    except OSError:
        return []
    out = []
    for p in entries:
        if not p.name.startswith(partial) or (p.name.startswith(".") and not partial.startswith(".")):
            continue
        if p.is_dir():
            out.append(f"{prefix}{p.name}/")
        elif p.name.endswith(suffix):
            out.append(f"{prefix}{p.name}")
    return sorted(out)


def _connection_flags(words: list[str]) -> list[str]:
    """The connection flags (and values) among *words*, for talking to the right manager."""
    keep = []
    for i, w in enumerate(words):
        if w in ("--manager", "--token", "--context", "--campaign") and i + 1 < len(words):
            keep += [w, words[i + 1]]
        elif w.split("=", 1)[0] in ("--manager", "--token", "--context", "--campaign") and "=" in w:
            keep.append(w)
        elif w == "--all-campaigns":
            keep.append(w)
    return keep


def complete(words: list[str]) -> list[str]:
    """Candidates for the last of *words* (the words after ``mlsweep``)."""
    global _connection_words
    if not words:
        words = [""]
    cur = words[-1]
    if len(words) == 1:
        return [c for c in COMMANDS if c.startswith(cur)]
    _connection_words = _connection_flags(words[:-1])
    cmd, prev = words[0], words[-2]

    def match(cands: list[str] | tuple[str, ...]) -> list[str]:
        return [c for c in cands if c.startswith(cur)]

    if prev == "--context":
        return match(_contexts())
    if prev == "--campaign":
        return match(_campaigns())
    if prev in ("--experiment", "-e", "--to-experiment"):
        return match(_experiments())
    if prev in _TAKES_VALUE:
        return []
    if cur.startswith("-"):
        return match(_flags(cmd))

    positional = [w for i, w in enumerate(words[1:-1], start=1)
                  if not w.startswith("-") and words[i - 1] not in _TAKES_VALUE | {"--context", "--campaign",
                                                                                   "--experiment", "-e"}]
    if cmd in _SUBCOMMANDS and not positional:
        return match(_SUBCOMMANDS[cmd])
    if cmd == "context" and positional and positional[0] in ("show", "use", "rm") and len(positional) == 1:
        return match(_contexts())
    if cmd in ("run", "lint"):
        return _paths(cur, ".py")
    if cmd in ("docs", "help") and not positional:
        return match(_DOC_TOPICS)
    if cmd in _EXPERIMENT_FIRST and not positional:
        return match(_experiments())
    return []


def complete_cmd(argv: list[str]) -> None:
    """``mlsweep __complete -- WORD...``: print one candidate per line."""
    if argv and argv[0] == "--":
        argv = argv[1:]
    os.environ.setdefault("MLSWEEP_COMPLETING", "1")
    try:
        for c in complete(list(argv)):
            print(c)
    except SystemExit:
        pass


_BASH = r'''# mlsweep bash completion.  Load it with:  eval "$(mlsweep completion bash)"
_mlsweep_complete() {
    local IFS=$'\n'
    COMPREPLY=($(mlsweep __complete -- "${COMP_WORDS[@]:1:COMP_CWORD}" 2>/dev/null))
    [[ ${#COMPREPLY[@]} -eq 1 && ${COMPREPLY[0]} == */ ]] && compopt -o nospace 2>/dev/null
}
complete -o default -F _mlsweep_complete mlsweep
'''

_ZSH = r'''#compdef mlsweep
# mlsweep zsh completion.  Load it with:  eval "$(mlsweep completion zsh)"
_mlsweep() {
    local -a cands
    cands=("${(@f)$(mlsweep __complete -- "${(@)words[2,CURRENT]}" 2>/dev/null)}")
    cands=(${cands:#})
    if (( ${#cands} )); then
        compadd -S '' -- ${(M)cands:#*/}
        compadd -- ${cands:#*/}
    else
        _files
    fi
}
compdef _mlsweep mlsweep
'''

_FISH = r'''# mlsweep fish completion.  Load it with:  mlsweep completion fish | source
complete -c mlsweep -f -a '(mlsweep __complete -- (commandline -opc)[2..-1] (commandline -ct) 2>/dev/null)'
'''

SCRIPTS = {"bash": _BASH, "zsh": _ZSH, "fish": _FISH}


def completion_cmd(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mlsweep completion",
        description="Print a shell completion script. Completes subcommands, flags, sweep files, context "
                    "names, and experiment IDs and campaigns from the manager.",
        epilog='bash: eval "$(mlsweep completion bash)" in ~/.bashrc; zsh: eval "$(mlsweep completion zsh)" '
               "in ~/.zshrc (after compinit); fish: mlsweep completion fish > "
               "~/.config/fish/completions/mlsweep.fish",
    )
    parser.add_argument("shell", choices=sorted(SCRIPTS))
    args = parser.parse_args(argv)
    sys.stdout.write(SCRIPTS[args.shell])
