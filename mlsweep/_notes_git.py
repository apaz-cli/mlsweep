"""Git plumbing for mlsweep notes: a shared folder of files in a git repo.

The manager's notes folder is an ordinary repo and also the remote.  It serves
itself at ``/notes.git`` (see ``_notes_http``), and a push updates its files in
place.  Everywhere else the notes are a normal clone, and agents use normal git:
pull, commit, merge, push.  ``commit_all`` records what the dashboard wrote.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

from mlsweep._notes_rules import CONFIG_FILE, DEFAULT_CONFIG, parse_config

GIT_TIMEOUT = 300
TOKEN_FILE = "mlsweep-token"

# Reads the token from this clone's .git, so it stays out of .git/config and argv.
_CREDENTIAL_HELPER = ("!f() { test \"$1\" = get && printf 'username=mlsweep\\npassword=%s\\n' "
                      "\"$(cat \"$(git rev-parse --git-dir)/" + TOKEN_FILE + "\")\"; }; f")

# What makes the manager's repo a safe remote.
_SERVED_CONFIG = {
    "receive.denyCurrentBranch": "updateInstead",  # a push updates the folder the dashboard shows
    "receive.denyNonFastForwards": "true",         # history stays recoverable
    "receive.denyDeletes": "true",
    "receive.fsckObjects": "true",
    "core.symlinks": "false",                      # a pushed symlink cannot point out of the folder
    "http.receivepack": "true",
}

# updateInstead rewrites the folder before it finds out the ref update failed, so a
# push based on an old commit would leave the folder with that push's files while the
# branch stays put, and the next commit would revert everyone else's work.  This hook
# runs first, and pushes hold the tree lock, so the branch cannot move after it passes.
_PRE_RECEIVE_HOOK = """\
#!/bin/sh
# Installed by the mlsweep manager, which rewrites it at startup.
while read -r old new ref; do
    current=$(git rev-parse -q --verify "$ref") || current=$(printf '%s' "$old" | tr 0-9a-f 0)
    if [ "$current" != "$old" ]; then
        echo "$ref moved while you were pushing. Run git pull, then push again." >&2
        exit 1
    fi
done
"""

GITIGNORE = """\
.DS_Store
*.swp
*.swo
*~
.obsidian/workspace*
.obsidian/cache
__pycache__/
.ipynb_checkpoints/
"""

README = """\
# Research notes

A shared folder for notes, papers, plots and anything else, in a git repo hosted
by the mlsweep manager. Get a clone with `mlsweep notes clone DIR`, then use git as
usual. The dashboard's Notes tab browses and edits the same repo.

- `sources/` holds originals (papers, blog posts, data). Add files here, but do
  not edit or delete them.
- Everything else is free-form: any folders, any files.
- `[[Note name]]` links to another note, `![[plot.png]]` embeds a file, and
  experiment IDs written anywhere link to their results in the dashboard.

See AGENTS.md for the rules agents follow.
"""

AGENTS_MD = """\
# Rules for agents working here

This is a shared repo of research notes, papers, plots and data. Most work here is
on the notes themselves: reading, summarizing, drafting, reorganizing, fixing links
and formatting. When a request is open-ended, assume that is what is wanted. If you
are asked for something else (a sweep, code, a question), do that instead.

## Always

This is a git repo. Run `git pull` before you start. Commit and `git push` as you go,
so another agent can pick up from what you pushed, and always before you stop. If the
push is refused because someone else got there first, run `git pull`, resolve any
conflicts so both sides' work survives, and push again. Never force-push.

1. Read before you write. `grep -ri` the folder for what is already known.
2. Prefer a new file to rewriting someone else's. To correct a note, add a dated
   line or a new note that links to it (`[[old note]]`); do not delete what it said.
3. `sources/` is read-only for you, except to add new files.
4. Each project folder keeps its own `STATUS.md` (a repo with one project can keep it
   at the top). Keep it short and readable by someone with none of your context: what
   is running, with experiment IDs, and what comes next. Read it when you start, and
   update it before you stop.

Markdown is GitHub-flavored plus `[[links]]`, `![[embeds]]`, `$math$`, `> [!note]`
callouts, `[^footnotes]`, numbered figure captions (`Figure: ...`), live data blocks
fetched from the manager (`mlsweep-chart`, `mlsweep-table`), and links to the code an
experiment ran (`[text](mlsweep:code/EXP/path#L40-60)`, and `mlsweep-code` and
`mlsweep-diff` blocks). Pandoc `:::` divs do not render. Rename notes with `mlsweep notes mv`, which keeps links
working. Full reference: `mlsweep docs notes`.

## Write for people

Notes are read by people, often quickly and without your context. Write so that a
person can follow them on the first read. Clarity for a reader matters more than
looking thorough.

1. Use plain words and short sentences. Say what was found first, then why it
   matters, then the details. If a paragraph needs a second read, rewrite it.
2. Be concrete. Give numbers with their conditions instead of adjectives like
   "significant" or "robust".
3. Mark the terms you made up. A name you coined for a method or an effect means
   nothing to anyone else. The first time a note uses one, say that it is your own
   name and explain it in ordinary words, for example: "what I call *hub chaining*
   (my name for it, not a standard term) estimates ...". Standard terms can be used
   as they are, with a link or citation when they are not common knowledge. When you
   use a standard term for your own particular construction, say what yours is.
4. Long documents drift into jargon. Before you push a long note, reread it from the
   top as a newcomer would, and cut jargon, stacked qualifiers, and lists that would
   read better as sentences. Several short notes are easier to read than one long one.

## When a note reports experiment results

These apply to any note that gives numbers from a sweep.

1. Every number must come from that experiment's logged metrics, and the note must
   name the experiment and run. Write it as an evidence link,
   `[0.42](mlsweep:EXPERIMENT/RUN/metric@STEP)`, so a reader can check it with one
   click and lint reports a number that disagrees with the log. Plot with a
   `mlsweep-chart` block instead of pasting a screenshot.
2. State the conditions a result holds under (steps, seeds, model size).
   One seed is one seed.
3. Record dead ends too, with the reason, so nobody retries them.
"""

SOURCES_README = """\
# sources

Originals: papers, blog posts, datasets, transcripts. Add files freely, but do not
change or delete them once they are pushed.
"""


class GitError(RuntimeError):
    """A git command failed."""


# ── git helpers ────────────────────────────────────────────────────────────────


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "GIT_TERMINAL_PROMPT": "0",      # never block on a credential prompt
        "LC_ALL": "C",                   # stable, parseable messages
    })
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    env.pop("GIT_INDEX_FILE", None)
    return env


def git(repo: str | Path, *args: str, check: bool = True, input: bytes | None = None,
        timeout: float = GIT_TIMEOUT) -> subprocess.CompletedProcess[bytes]:
    """Run ``git -C repo ARGS`` and return the completed process (bytes output)."""
    try:
        p = subprocess.run(["git", "-C", str(repo), "-c", "core.quotepath=false", *args], input=input,
                           capture_output=True, env=_git_env(), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args[:2])} timed out after {timeout:.0f}s") from exc
    except FileNotFoundError as exc:
        raise GitError("git is not installed") from exc
    if check and p.returncode != 0:
        msg = (p.stderr or p.stdout).decode(errors="replace").strip()
        raise GitError(f"git {' '.join(args[:3])}: {msg or f'exit {p.returncode}'}")
    return p


def git_text(repo: str | Path, *args: str, check: bool = True) -> str:
    """Run git and return stdout decoded and stripped."""
    return git(repo, *args, check=check).stdout.decode(errors="replace").strip()


def is_repo(path: str | Path) -> bool:
    """True if *path* is the top of a git work tree."""
    p = Path(path)
    if not p.is_dir():
        return False
    try:
        r = git(p, "rev-parse", "--show-toplevel", check=False)
    except GitError:
        return False
    if r.returncode != 0:
        return False
    return Path(r.stdout.decode().strip()).resolve() == p.resolve()


def git_dir(repo: str | Path) -> Path:
    """The repo's .git directory."""
    d = git_text(repo, "rev-parse", "--absolute-git-dir")
    return Path(d)


def current_branch(repo: str | Path) -> str:
    """The checked-out branch name (``main`` if HEAD is unborn)."""
    r = git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    name = r.stdout.decode().strip()
    return name or "main"


def has_commits(repo: str | Path) -> bool:
    return bool(head(repo))


def remote_url(repo: str | Path, remote: str = "origin") -> str | None:
    r = git(repo, "remote", "get-url", remote, check=False)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def head(repo: str | Path) -> str:
    r = git(repo, "rev-parse", "-q", "--verify", "HEAD", check=False)
    return r.stdout.decode().strip()


def read_config(repo: str | Path) -> dict[str, Any]:
    """The repo's ``.mlsweep-notes.json`` merged over the defaults."""
    p = Path(repo) / CONFIG_FILE
    return parse_config(p.read_text(errors="replace") if p.is_file() else None)


def unpushed(repo: str | Path, branch: str) -> int | None:
    """Commits not on the remote yet, or None if there is nothing to compare with."""
    r = git(repo, "rev-list", "--count", f"origin/{branch}..HEAD", check=False)
    return int(r.stdout.decode().strip() or 0) if r.returncode == 0 else None


def repo_info(repo: str | Path) -> dict[str, Any]:
    """Branch, remote and head; plus ``ahead`` (see ``unpushed``) if it has a remote."""
    info: dict[str, Any] = {"branch": current_branch(repo), "remote": remote_url(repo), "head": head(repo)}
    if info["remote"]:
        info["ahead"] = unpushed(repo, info["branch"])
    return info


def _identity(repo: str | Path) -> tuple[str, str]:
    """Commit name and email: the repo's git config, else user@host."""
    name = git_text(repo, "config", "--get", "user.name", check=False)
    email = git_text(repo, "config", "--get", "user.email", check=False)
    fallback = f"{os.environ.get('USER', 'user')}@{socket.gethostname()}"
    return name or fallback, email or fallback


# ── setup ──────────────────────────────────────────────────────────────────────


def write_skeleton(root: Path) -> list[str]:
    """Create the starter files that are missing.  Returns the paths written."""
    files = {
        ".gitignore": GITIGNORE,
        CONFIG_FILE: json.dumps(DEFAULT_CONFIG, indent=2) + "\n",
        "README.md": README,
        "AGENTS.md": AGENTS_MD,
        "sources/README.md": SOURCES_README,
    }
    written = []
    for rel, text in files.items():
        p = root / rel
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        written.append(rel)
    return written


def init_repo(root: str | Path, *, branch: str = "main", name: str | None = None,
              email: str | None = None) -> None:
    """Make *root* a repo with the starter files, committing anything already there."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not is_repo(root):
        git(root, "init", "-q", "-b", branch)
    _set_identity(root, name, email)
    write_skeleton(root)
    commit_all(root, message="Initialize notes")


def configure_served(repo: str | Path) -> None:
    """Set what the manager's repo needs in order to act as the remote."""
    for key, value in _SERVED_CONFIG.items():
        git(repo, "config", key, value)
    hook = git_dir(repo) / "hooks" / "pre-receive"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text(_PRE_RECEIVE_HOOK)
    hook.chmod(0o755)


def _set_identity(repo: Path, name: str | None, email: str | None) -> None:
    if name:
        git(repo, "config", "user.name", name)
    if email:
        git(repo, "config", "user.email", email)


def clone_repo(url: str, dest: str | Path, *, token: str, name: str | None = None,
               email: str | None = None) -> None:
    """Clone the manager's notes into *dest*, so plain ``git pull`` and ``git push`` work.

    The token goes in a private file that only this clone's credential helper reads.
    """
    dest = Path(dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise GitError(f"{dest} exists and is not empty")
    existed = dest.exists()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        git(dest, "init", "-q")
        fd = os.open(git_dir(dest) / TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token)
        # The empty value drops helpers from the user's global config, which might answer first.
        git(dest, "config", "--add", "credential.helper", "")
        git(dest, "config", "--add", "credential.helper", _CREDENTIAL_HELPER)
        git(dest, "config", "pull.rebase", "false")
        git(dest, "remote", "add", "origin", url)
        git(dest, "fetch", "-q", "origin")
        refs = git_text(dest, "ls-remote", "--symref", "origin", "HEAD")
        branch = refs.split("refs/heads/", 1)[1].split()[0] if "refs/heads/" in refs else "main"
        git(dest, "checkout", "-q", "-b", branch, "--track", f"origin/{branch}")
        have_name, have_email = _identity(dest)
        _set_identity(dest, name or have_name, email or have_email)
    except GitError:
        shutil.rmtree(dest if not existed else git_dir(dest), ignore_errors=True)
        raise


def update_clone(dest: str | Path, url: str, *, token: str, name: str | None = None,
                 email: str | None = None) -> bool:
    """If *dest* is already a clone of *url*, refresh its token and identity and return True."""
    dest = Path(dest).expanduser().resolve()
    if not is_repo(dest) or remote_url(dest) != url:
        return False
    fd = os.open(git_dir(dest) / TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    if name or email:
        have_name, have_email = _identity(dest)
        _set_identity(dest, name or have_name, email or have_email)
    return True


# ── commits ────────────────────────────────────────────────────────────────────


def status_entries(repo: str | Path) -> list[tuple[str, str]]:
    """``git status --porcelain=v1 -z`` as (XY, path) entries."""
    parts = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.split(b"\0")
    entries: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        item = parts[i]
        i += 1
        if not item:
            continue
        xy = item[:2].decode()
        entries.append((xy, item[3:].decode(errors="surrogateescape")))
        if xy[0] in "RC":
            i += 1
    return entries


def commit_all(repo: str | Path, message: str | None = None) -> int:
    """Commit everything that changed in the working tree.  Returns the files committed."""
    if not status_entries(repo):
        return 0
    git(repo, "add", "-A")
    changed = [c for c in git_text(repo, "diff", "--cached", "--name-only", "-z").split("\0") if c]
    if not changed:
        return 0
    name, email = _identity(repo)
    if message is None:
        shown = ", ".join(changed[:4]) + (f" and {len(changed) - 4} more" if len(changed) > 4 else "")
        message = f"{name}: {shown}"
    git(repo, "-c", f"user.name={name}", "-c", f"user.email={email}",
        "commit", "-q", "--no-verify", "-m", message)
    return len(changed)


# ── history ────────────────────────────────────────────────────────────────────

_LOG_FMT = "%H%x1f%h%x1f%an%x1f%ae%x1f%at%x1f%s"


def log_entries(repo: str | Path, path: str | None = None, *, limit: int = 50,
                skip: int = 0, with_files: bool = False) -> list[dict[str, Any]]:
    """Recent commits, newest first, optionally only those touching *path*."""
    if not has_commits(repo):
        return []
    args = ["log", f"--max-count={limit}", f"--skip={skip}", f"--format=%x1e{_LOG_FMT}"]
    if with_files or path:
        args += ["--name-status", "-M"]
    if path:
        args += ["--follow", "--", path]
    out = git_text(repo, *args, check=False)
    entries: list[dict[str, Any]] = []
    for block in out.split("\x1e"):
        block = block.strip("\n")
        if not block:
            continue
        lines = block.split("\n")
        fields = lines[0].split("\x1f")
        if len(fields) < 6:
            continue
        e: dict[str, Any] = {
            "hash": fields[0], "short": fields[1], "author": fields[2], "email": fields[3],
            "time": int(fields[4]), "subject": fields[5],
        }
        files = []
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split("\t")
            status = parts[0][:1]
            files.append({"status": status, "path": parts[-1],
                          **({"from": parts[1]} if status in "RC" and len(parts) > 2 else {})})
        if with_files:
            e["files"] = files
        if path:
            # The file's name as of this commit (it may have been renamed since).
            e["path"] = files[-1]["path"] if files else path
            e["status"] = files[-1]["status"] if files else "M"
        entries.append(e)
    return entries


def show_file(repo: str | Path, rev: str, path: str) -> bytes | None:
    """File contents at *rev*, or None if it did not exist there."""
    r = git(repo, "show", f"{rev}:{path}", check=False)
    return r.stdout if r.returncode == 0 else None


def commit_diff(repo: str | Path, rev: str, path: str | None = None) -> str:
    """Unified diff a commit introduced, optionally limited to *path*."""
    args = ["show", "--format=", "--no-color", "-M", "--find-renames", rev]
    if path:
        args += ["--", path]
    return git(repo, *args, check=False).stdout.decode(errors="replace")


def last_commit(repo: str | Path, path: str) -> dict[str, Any] | None:
    entries = log_entries(repo, path, limit=1)
    return entries[0] if entries else None
