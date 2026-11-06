"""``mlsweep notes``: a shared research notes folder in a git repo the manager hosts.

    mlsweep notes clone DIR [--manager URL]
    mlsweep notes lint DIR [--fix] [--review-prompt]
    mlsweep notes status DIR

See ``mlsweep docs notes`` for the full guide.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote
from typing import Any

from mlsweep import _notes, _notes_git, _notes_lint
from mlsweep._colors import _BOLD, _CYAN, _DIM, _GREEN, _RED, _RESET, _YELLOW
from mlsweep._notes_rules import CONFIG_FILE
from mlsweep._shared import _mlsweep_dir
from mlsweep.ctl import _dur
from mlsweep.run_sweep import _add_connection_args, _http_request, _manager_url, _resolve_token

_USAGE = f"""\
{_BOLD}mlsweep notes{_RESET} — a shared notes folder, in a git repo the manager hosts

{_BOLD}{_CYAN}Use:{_RESET}
  clone DIR                      clone the notes from the manager (--manager URL), then use git as usual
  status [DIR]                   remote, unpushed commits, recent changes
  lint [DIR]                     check links, experiment IDs and conflict markers (--fix, --json)
  lint [DIR] --review-prompt     print a brief for an independent reviewing agent

{_BOLD}{_CYAN}Read and reorganize{_RESET} (add --dir DIR, --json):
  links PATH                     outgoing links, backlinks, experiments a note cites
  mentions EXPERIMENT            notes that mention an experiment
  search WORDS...                search like the Notes tab does
  mv SRC DST                     move or rename, rewriting links to it
  history [PATH]                 who changed what (one file, following renames)
  show REV PATH | diff REV [PATH] | restore PATH REV

DIR defaults to $MLSWEEP_NOTES, else the clone you are in, else ~/.mlsweep/notes
(<mlsweep-dir>/notes), which the manager serves and creates. Guide: `mlsweep docs notes`.
"""


def default_notes_dir() -> Path:
    """$MLSWEEP_NOTES, else <mlsweep-dir>/notes (``~/.mlsweep/notes``)."""
    env = os.environ.get("MLSWEEP_NOTES")
    if env:
        return Path(env).expanduser().resolve()
    return (_mlsweep_dir() / "notes").resolve()


def _default_dir(arg: str | None, *, writes: bool = False) -> Path:
    """The folder to work on.  Falling back to the manager's own folder is never silent."""
    if arg:
        return Path(arg).expanduser().resolve()
    if os.environ.get("MLSWEEP_NOTES"):
        return default_notes_dir()
    try:
        top = _notes_git.git_text(Path.cwd(), "rev-parse", "--show-toplevel", check=False)
    except _notes_git.GitError:
        top = ""
    if top and (Path(top) / CONFIG_FILE).exists():
        return Path(top).resolve()
    root = default_notes_dir()
    if writes:
        print(f"{_RED}Not in a notes clone, so this would change the manager's folder {root}.{_RESET} "
              "Pass a clone's path or set MLSWEEP_NOTES; pass that folder to mean it.", file=sys.stderr)
        sys.exit(2)
    if not root.is_dir():
        print(f"{_RED}No notes here: not in a clone, and {root} does not exist.{_RESET} "
              "Pass a clone's path, set MLSWEEP_NOTES, or run `mlsweep notes clone DIR`.", file=sys.stderr)
        sys.exit(2)
    print(f"{_DIM}Not in a notes clone: using the manager's folder {root}.{_RESET}", file=sys.stderr)
    return root


def _require_repo(path: Path) -> None:
    if not _notes_git.is_repo(path):
        print(f"{_RED}{path} is not a notes clone.{_RESET} "
              "Create one with `mlsweep notes clone DIR`.", file=sys.stderr)
        sys.exit(2)


# ── clone ──────────────────────────────────────────────────────────────────────


def _clone(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="mlsweep notes clone",
        description="Clone the notes from the manager. Plain `git pull` and `git push` then work, "
                    "using a copy of the token kept privately in the clone's .git folder.")
    ap.add_argument("dir")
    _add_connection_args(ap)
    ap.add_argument("--name", help="commit author name (default: your git config, else user@host)")
    ap.add_argument("--email")
    args = ap.parse_args(argv)
    token = _resolve_token(args.token)
    if not token:
        print(f"{_RED}No token: pass --token, or set MLSWEEP_TOKEN.{_RESET}", file=sys.stderr)
        sys.exit(2)
    url = _manager_url(args.manager, "/notes.git")
    hint = f"Outside the clone, pass its path to notes commands or set MLSWEEP_NOTES={args.dir}."
    try:
        if _notes_git.update_clone(args.dir, url, token=token, name=args.name, email=args.email):
            print(f"{args.dir} is already a clone of {url}. Run `git pull` there.")
            print(hint)
            return
        _notes_git.clone_repo(url, args.dir, token=token, name=args.name, email=args.email)
    except _notes_git.GitError as exc:
        print(f"{_RED}{exc}{_RESET}", file=sys.stderr)
        sys.exit(1)
    print(f"{_GREEN}cloned {url} into {args.dir}{_RESET}")
    print("Use git as usual: pull, commit, push. Read AGENTS.md first.")
    print(hint)


# ── status ─────────────────────────────────────────────────────────────────────


def _ago(ts: float) -> str:
    return f"{_dur(time.time() - ts)} ago"


def _status(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlsweep notes status")
    ap.add_argument("dir", nargs="?")
    ap.add_argument("-n", type=int, default=10, help="recent commits to show")
    args = ap.parse_args(argv)
    root = _default_dir(args.dir)
    _require_repo(root)
    info = _notes_git.repo_info(root)
    print(f"{_BOLD}{root}{_RESET}")
    print(f"  remote   {info['remote'] or '(none: local only)'}")
    print(f"  branch   {info['branch']} @ {info['head'][:10] or '(no commits)'}")
    dirty = _notes_git.status_entries(root)
    if info.get("ahead") is not None:
        print(f"  unpushed {info['ahead']} commit(s)")
    print(f"  changed  {len(dirty)} file(s) not yet committed")
    cfg = _notes_git.read_config(root)
    print(f"  protected {', '.join(cfg['protected'])}  (max file {cfg['max_file_mb']} MB)")
    entries = _notes_git.log_entries(root, limit=args.n, with_files=True)
    if entries:
        print(f"\n{_BOLD}Recent changes{_RESET}")
        for e in entries:
            files = e.get("files", [])
            shown = ", ".join(f["path"] for f in files[:3]) + (f" +{len(files) - 3}" if len(files) > 3 else "")
            print(f"  {_DIM}{_ago(e['time']):>11}{_RESET}  {_CYAN}{e['author']:<20}{_RESET} {shown}")


# ── lint ───────────────────────────────────────────────────────────────────────


def _fetch_known(manager: str, token: str) -> set[str]:
    """Experiment IDs on the manager; OSError if it won't say."""
    path = "/api/notes/experiment-ids"
    status, data = _http_request("GET", _manager_url(manager, path), token, timeout=60, quiet=True)
    if status != 200 or not isinstance(data, dict):
        raise OSError(f"HTTP {status}" if status else "no response")
    return set(data.get("known", []))


def _fetch_run_ids(manager: str, token: str, experiment: str) -> set[str]:
    """Run IDs of *experiment*; OSError if the manager won't say."""
    path = f"/api/experiments/{quote(experiment, safe='')}/jobs"
    status, data = _http_request("GET", _manager_url(manager, path), token, timeout=10, quiet=True)
    if status != 200 or not isinstance(data, list):
        raise OSError(f"HTTP {status}" if status else "no response")
    return {str(j["run_id"]) for j in data if isinstance(j, dict) and j.get("run_id")}


def _lint(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="mlsweep notes lint",
        description="Check the notes: links, experiment IDs (via the manager) and merge "
                    "leftovers. Numbers are not checked; compare them with `mlsweep metrics`. "
                    "Reports only; --fix applies the safe link-path fixes.")
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--fix", action="store_true", help="apply safe fixes (link paths) and report the rest")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--review-prompt", action="store_true",
                    help="print a brief for an independent reviewing agent, with this report")
    _add_connection_args(ap)
    ap.add_argument("--no-manager", action="store_true", help="skip the experiment ID check")
    args = ap.parse_args(argv)
    root = _default_dir(args.dir, writes=args.fix)
    if not root.is_dir():
        print(f"{_RED}{root} is not a directory{_RESET}", file=sys.stderr)
        sys.exit(2)

    known: set[str] | None = None
    note = ""
    if not args.no_manager:
        token = _resolve_token(args.token)
        try:
            known = _fetch_known(args.manager, token)
        except OSError as exc:
            note = f"experiment IDs not checked: manager {args.manager} unreachable ({exc})"

    result = _notes_lint.lint(root, known_experiments=known)
    if args.fix:
        changed = _notes_lint.apply_fixes(root, result["findings"])
        if changed and not args.json:
            print(f"{_GREEN}fixed links in {len(changed)} file(s):{_RESET} " + ", ".join(changed))
        result = _notes_lint.lint(root, known_experiments=known)
    if args.json:
        print(json.dumps(result, indent=1))
        return
    if args.review_prompt:
        print(_notes_lint.review_prompt(root, result, time.strftime("%Y-%m-%d")))
        return
    if note:
        print(f"{_YELLOW}{note}{_RESET}")
    print(_notes_lint.format_report(result))
    if any(f["tier"] == "fix" for f in result["findings"]):
        print(f"\n{_DIM}Run with --fix to apply the safe fixes.{_RESET}")


# ── reading and reorganizing (what the Notes tab can do) ───────────────────────


def _known_experiments(manager: str, token: str) -> set[str]:
    """Experiment IDs on the manager, or an empty set if it can't be reached."""
    try:
        return _fetch_known(manager, token)
    except OSError:
        return set()


def _add_dir_args(ap: argparse.ArgumentParser, manager: bool = False) -> None:
    ap.add_argument("--dir", help="notes folder (default: $MLSWEEP_NOTES, else the clone you are in, "
                                  "else ~/.mlsweep/notes)")
    ap.add_argument("--json", action="store_true")
    if manager:
        _add_connection_args(ap)


def _rel_in(root: Path, path: str) -> str:
    """A path given on the command line, as a path inside the notes folder."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        try:
            return str(p.resolve().relative_to(root))
        except ValueError:
            pass
    return path.strip("/")


def _mv(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="mlsweep notes mv",
        description="Rename or move a note, file or folder, rewriting [[links]] and relative "
                    "links that point at it (plain `mv` would break them).")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--no-links", action="store_true", help="do not rewrite links")
    _add_dir_args(ap)
    args = ap.parse_args(argv)
    root = _default_dir(args.dir, writes=True)
    src, dst = _rel_in(root, args.src), _rel_in(root, args.dst)
    if (root / dst).is_dir() and not (root / src).is_dir() or args.dst.endswith("/"):
        dst = f"{dst.rstrip('/')}/{Path(src).name}"
    try:
        r = _notes.move_path(root, src, dst, update_links=not args.no_links)
    except (FileNotFoundError, FileExistsError, _notes.PathError) as exc:
        print(f"{_RED}{type(exc).__name__}: {exc}{_RESET}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(r, indent=1))
        return
    print(f"{_GREEN}moved {r['from']} -> {r['to']}{_RESET}")
    for p in r["links_updated_in"]:
        print(f"  updated links in {p}")


def _links(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="mlsweep notes links",
        description="A note's outgoing links, backlinks (who links here, with the line), and "
                    "the experiments it cites: the Notes tab's side panel.")
    ap.add_argument("path")
    _add_dir_args(ap, manager=True)
    args = ap.parse_args(argv)
    root = _default_dir(args.dir)
    rel = _rel_in(root, args.path)
    if not (root / rel).exists() and (root / (rel + ".md")).exists():
        rel += ".md"
    known = _known_experiments(args.manager, _resolve_token(args.token))
    idx = _notes.build_index(root, known)
    data = idx.as_dict()
    out: dict[str, Any] = {
        "path": rel,
        "outgoing": [{"target": l["target"], "resolved": l["resolved"], "line": l["line"]}
                     for l in data["links"].get(rel, [])],
        "backlinks": _notes.backlinks(idx, rel),
        "experiments": sorted(idx.mentions.get(rel, {})),
    }
    if args.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{_BOLD}{rel}{_RESET}")
    print(f"{_CYAN}Outgoing{_RESET} ({len(out['outgoing'])})")
    for l in out["outgoing"]:
        print(f"  {l['line']:>4}  {l['resolved'] or _RED + l['target'] + ' (missing)' + _RESET}")
    print(f"{_CYAN}Backlinks{_RESET} ({len(out['backlinks'])})")
    for b in out["backlinks"]:
        print(f"  {b['src']}:{b['line']}  {_DIM}{b['context'][:100]}{_RESET}")
    print(f"{_CYAN}Experiments cited{_RESET} ({len(out['experiments'])})" +
          ("" if known else f"  {_DIM}(manager unreachable: only IDs it knows are recognized){_RESET}"))
    for e in out["experiments"]:
        print(f"  {e}")


def _mentions(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="mlsweep notes mentions",
        description="Notes that mention an experiment ID or one of its run IDs, with the lines "
                    "(the Experiments page's citation counts).")
    ap.add_argument("experiment")
    _add_dir_args(ap, manager=True)
    args = ap.parse_args(argv)
    root = _default_dir(args.dir)
    known = {args.experiment}
    note = ""
    try:
        known |= _fetch_run_ids(args.manager, _resolve_token(args.token), args.experiment)
    except OSError as exc:
        note = f"run IDs not checked: manager {args.manager} unreachable ({exc})"
    idx = _notes.build_index(root, known)
    hits = {p: sorted({ln for lines in m.values() for ln in lines})
            for p, m in idx.mentions.items()}
    if note:
        print(f"{_YELLOW}{note}{_RESET}", file=sys.stderr)
    if args.json:
        print(json.dumps(hits, indent=1))
        return
    if not hits:
        print(f"No note mentions {args.experiment} or any of its runs.")
        return
    for p, lines in sorted(hits.items()):
        text = (root / p).read_text(errors="replace").split("\n")
        for ln in lines:
            print(f"{p}:{ln}  {_DIM}{text[ln - 1].strip()[:110]}{_RESET}")


def _search(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlsweep notes search",
                                 description="Search notes: every word must match (\"quoted phrases\" stay whole).")
    ap.add_argument("query", nargs="+")
    _add_dir_args(ap)
    args = ap.parse_args(argv)
    root = _default_dir(args.dir)
    files, _ = _notes.walk(root)
    res = _notes.search(root, files, " ".join(args.query))
    if args.json:
        print(json.dumps(res, indent=1))
        return
    for r in res:
        print(f"{_BOLD}{r['path']}{_RESET}")
        for m in r["matches"][:5]:
            print(f"  {m['line']:>5}  {m['text'].strip()[:120]}")


def _history(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="mlsweep notes history",
        description="Who changed what: every machine's commits, or one file's (following renames).")
    ap.add_argument("path", nargs="?")
    ap.add_argument("-n", type=int, default=20)
    _add_dir_args(ap)
    args = ap.parse_args(argv)
    root = _default_dir(args.dir)
    _require_repo(root)
    rel = _rel_in(root, args.path) if args.path else None
    entries = _notes_git.log_entries(root, rel, limit=args.n, with_files=rel is None)
    if args.json:
        print(json.dumps(entries, indent=1))
        return
    for e in entries:
        what = e.get("path", "") if rel else ", ".join(f["path"] for f in e.get("files", [])[:4])
        print(f"{_CYAN}{e['short']}{_RESET} {_DIM}{_ago(e['time']):>8}{_RESET}  {e['author']:<22} {what}")


def _show(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlsweep notes show",
                                 description="Print a file as it was at a commit (see `history`).")
    ap.add_argument("rev")
    ap.add_argument("path")
    ap.add_argument("--dir")
    args = ap.parse_args(argv)
    root = _default_dir(args.dir)
    data = _notes_git.show_file(root, args.rev, _rel_in(root, args.path))
    if data is None:
        print(f"{_RED}{args.path} is not in {args.rev}{_RESET}", file=sys.stderr)
        sys.exit(1)
    sys.stdout.buffer.write(data)


def _diff(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlsweep notes diff", description="What a commit changed.")
    ap.add_argument("rev")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dir")
    args = ap.parse_args(argv)
    root = _default_dir(args.dir)
    sys.stdout.write(_notes_git.commit_diff(root, args.rev, _rel_in(root, args.path) if args.path else None))


def _restore(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="mlsweep notes restore",
        description="Put a file back the way it was at a commit (the current version stays in "
                    "history; commit the result).")
    ap.add_argument("path")
    ap.add_argument("rev")
    ap.add_argument("--from-path", help="the file's name at REV, if it was renamed since")
    ap.add_argument("--dir")
    args = ap.parse_args(argv)
    root = _default_dir(args.dir, writes=True)
    rel = _rel_in(root, args.path)
    data = _notes_git.show_file(root, args.rev, args.from_path or rel)
    if data is None:
        print(f"{_RED}{args.from_path or rel} is not in {args.rev} (renamed? pass --from-path){_RESET}",
              file=sys.stderr)
        sys.exit(1)
    _notes.atomic_write(_notes.resolve(root, rel), data)
    print(f"{_GREEN}restored {rel} from {args.rev}{_RESET}")


# ── dispatch ───────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return
    cmd, rest = argv[0], argv[1:]
    handlers = {
        "clone": _clone, "status": _status, "lint": _lint,
        "mv": _mv, "links": _links, "mentions": _mentions, "search": _search,
        "history": _history, "show": _show, "diff": _diff, "restore": _restore,
    }
    fn = handlers.get(cmd)
    if fn is None:
        print(f"{_RED}Unknown notes command: {cmd}{_RESET}\n")
        print(_USAGE)
        sys.exit(1)
    fn(rest)
