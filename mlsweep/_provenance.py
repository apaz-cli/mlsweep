"""What the client records beside a code snapshot at submit time.

Everything here is optional metadata for people reading the experiment later.
The snapshot itself (see ``mlsweep._snapshot``) is what runs; this only says
where it came from: the sweep file, the command line, the host, and the
version control state when the project has any.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from mlsweep._notes_git import _git_env
from mlsweep._snapshot import Manifest

# Caps on what is recorded, so a huge uncommitted change can't bloat the database.
MAX_DIFF_BYTES = 1 << 20
MAX_SOURCE_BYTES = 256 * 1024
MAX_LISTED_PATHS = 1000


def _git(root: Path, *args: str, timeout: float = 20) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, env=_git_env(),
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def _strip_credentials(url: str) -> str:
    """A remote URL without any user:password in it."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme and parts.netloc and "@" in parts.netloc:
        host = parts.netloc.rsplit("@", 1)[1]
        user = parts.netloc.rsplit("@", 1)[0].split(":", 1)[0]
        # ssh://git@host keeps its (non-secret) user name; https://user:token@ loses both.
        netloc = f"{user}@{host}" if parts.scheme in ("ssh", "git+ssh") else host
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url


def git_provenance(root: Path, manifest: Manifest) -> dict[str, Any] | None:
    """Commit, branch, remote, dirty state and diff of the git repo at *root*, or None."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top.strip()).resolve() != root.resolve():
        return None
    commit = (_git(root, "rev-parse", "HEAD") or "").strip() or None
    info: dict[str, Any] = {"type": "git", "commit": commit}
    branch = (_git(root, "symbolic-ref", "--short", "-q", "HEAD") or "").strip()
    info["branch"] = branch or None
    if commit:
        head = (_git(root, "log", "-1", "--format=%s%x00%an%x00%cI") or "").strip().split("\0")
        subject, author, committed_at = (head + ["", "", ""])[:3]
        info["subject"] = subject[:500] or None
        info["author"] = author[:200] or None
        info["committed_at"] = committed_at or None
    remote = (_git(root, "config", "--get", "remote.origin.url") or "").strip()
    info["remote"] = _strip_credentials(remote) if remote else None

    status = (_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no") or "").split("\0")
    changed: list[str] = []
    i = 0
    while i < len(status):
        entry = status[i]
        if len(entry) > 3:
            changed.append(entry[3:])
            # A rename or copy is followed by its source path, with no status letters.
            if entry[0] in "RC":
                i += 1
        i += 1
    info["dirty"] = bool(changed)
    info["changed_files"] = changed[:MAX_LISTED_PATHS]
    diff = ""
    if changed and commit:
        diff = _git(root, "diff", "HEAD", "--no-color", "--no-ext-diff", timeout=60) or ""
    raw = diff.encode("utf-8")
    info["diff"] = raw[:MAX_DIFF_BYTES].decode("utf-8", errors="ignore") if raw else None
    info["diff_truncated"] = len(raw) > MAX_DIFF_BYTES

    tracked_out = _git(root, "ls-files", "-z")
    if tracked_out is not None:
        tracked = set(tracked_out.split("\0"))
        untracked = [f for f in manifest.files if f.path not in tracked]
        info["untracked"] = [f.path for f in untracked[:MAX_LISTED_PATHS]]
        info["untracked_count"] = len(untracked)
        info["untracked_bytes"] = sum(f.size for f in untracked)
    return info


def submit_provenance(
    *,
    root: Path,
    manifest: Manifest,
    snapshot_id: str,
    sweep_file: str,
) -> dict[str, Any]:
    """The metadata recorded with one submission of *snapshot_id*."""
    sweep_path = Path(sweep_file).resolve()
    try:
        rel = sweep_path.relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = None
    meta: dict[str, Any] = {
        "snapshot": snapshot_id,
        "sweep_file": rel or str(sweep_path),
        "sweep_file_in_snapshot": bool(rel) and rel in manifest.by_path(),
        "argv": sys.argv[1:],
        "host": socket.gethostname(),
        "root": str(root),
    }
    if not meta["sweep_file_in_snapshot"]:
        try:
            data = sweep_path.read_bytes()[:MAX_SOURCE_BYTES]
            meta["sweep_source"] = data.decode("utf-8", errors="replace")
        except OSError:
            pass
    meta["vcs"] = git_provenance(root, manifest) if (root / ".git").exists() else None
    return meta


def untracked_summary(vcs: dict[str, Any] | None, manifest: Manifest, top: int = 4) -> list[tuple[str, int]]:
    """The biggest top-level folders (or files) shipped that git does not track."""
    if not vcs or "untracked" not in vcs:
        return []
    names = set(vcs.get("untracked") or [])
    sizes: dict[str, int] = {}
    for f in manifest.files:
        if f.path in names:
            head = f.path.split("/", 1)[0] + ("/" if "/" in f.path else "")
            sizes[head] = sizes.get(head, 0) + f.size
    return sorted(sizes.items(), key=lambda kv: -kv[1])[:top]
