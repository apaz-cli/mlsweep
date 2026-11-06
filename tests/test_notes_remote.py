"""The manager as the notes' git remote: real git, over real HTTP, against a real manager."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_notes_api import _req, _start_manager, _teardown_manager

from mlsweep import _notes_git


def _env(**extra: str) -> dict[str, str]:
    # Only what git needs: pytest prints these on failure, so no secrets from the environment.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "PYTHONPATH") if k in os.environ}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", **extra)
    return env


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=_env(), check=check)


def _commit(repo: Path, path: str, text: str, message: str = "change") -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


@pytest.fixture
def remote(tmp_path: Path):  # type: ignore[no-untyped-def]
    notes = tmp_path / "notes"
    _notes_git.init_repo(notes, name="dash", email="d@x")
    proc, server, url = _start_manager(tmp_path, extra_args=("--notes", str(notes)))
    server.notes = notes
    server.clones = tmp_path / "clones"
    yield server
    _teardown_manager(proc)


def _clone(server, name: str) -> Path:  # type: ignore[no-untyped-def]
    dest = server.clones / name
    done = subprocess.run(
        [sys.executable, "-c", "from mlsweep.cli import main; main()", "notes", "clone", str(dest),
         "--manager", server.url, "--name", name, "--email", f"{name}@x"],
        capture_output=True, text=True, env=_env(MLSWEEP_TOKEN=server.token), timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    return dest


def test_clone_keeps_the_token_private(remote) -> None:  # type: ignore[no-untyped-def]
    a = _clone(remote, "a")
    assert (a / "AGENTS.md").is_file()
    assert remote.token not in (a / ".git" / "config").read_text()
    token_file = a / ".git" / "mlsweep-token"
    assert token_file.read_text() == remote.token and stat.S_IMODE(token_file.stat().st_mode) == 0o600


def test_push_reaches_the_dashboard(remote) -> None:  # type: ignore[no-untyped-def]
    a = _clone(remote, "a")
    _commit(a, "found.md", "a finding\n")
    _git(a, "push")
    assert (remote.notes / "found.md").read_text() == "a finding\n"
    _, f, _ = _req(remote.url, "GET", "/api/notes/file?path=found.md")
    assert f["content"] == "a finding\n" and f["last_commit"]["author"] == "a"
    assert _notes_git.status_entries(remote.notes) == []


def test_dashboard_save_reaches_a_pull(remote) -> None:  # type: ignore[no-untyped-def]
    a = _clone(remote, "a")
    _req(remote.url, "PUT", "/api/notes/file", {"path": "dash.md", "content": "from the tab\n", "base_hash": None})
    _req(remote.url, "POST", "/api/notes/commit")
    _git(a, "pull", "-q")
    assert (a / "dash.md").read_text() == "from the tab\n"


def test_push_commits_pending_saves_first(remote) -> None:  # type: ignore[no-untyped-def]
    a = _clone(remote, "a")
    _req(remote.url, "PUT", "/api/notes/file", {"path": "pending.md", "content": "unsaved\n", "base_hash": None})
    (remote.notes / "direct.md").write_text("edited in the folder\n")
    _commit(a, "found.md", "a finding\n")
    # The pending saves land first, so this push is behind until it merges them.
    refused = _git(a, "push", check=False)
    assert refused.returncode != 0 and "git pull" in refused.stderr
    # A refused push must leave the folder alone: nothing lost, nothing half-applied.
    assert _notes_git.status_entries(remote.notes) == []
    assert (remote.notes / "pending.md").is_file() and not (remote.notes / "found.md").exists()
    _git(a, "pull", "-q", "--no-edit")
    _git(a, "push")
    for name, text in (("pending.md", "unsaved\n"), ("direct.md", "edited in the folder\n"), ("found.md", "a finding\n")):
        assert (remote.notes / name).read_text() == text


def test_racing_pushes_never_corrupt_the_folder(remote) -> None:  # type: ignore[no-untyped-def]
    clones = [_clone(remote, f"c{i}") for i in range(4)]

    def agent(repo: Path, who: int) -> None:
        for n in range(6):
            _commit(repo, f"{who}-{n}.md", f"{who} {n}\n")
            _req(remote.url, "PUT", "/api/notes/file",
                 {"path": f"tab-{who}-{n}.md", "content": "from the tab\n", "base_hash": None})
            for _ in range(60):
                if _git(repo, "push", check=False).returncode == 0:
                    break
                _git(repo, "pull", "-q", "--no-edit")
            else:
                raise AssertionError("push never landed")

    with ThreadPoolExecutor(len(clones)) as pool:
        for future in [pool.submit(agent, repo, i) for i, repo in enumerate(clones)]:
            future.result()
    _req(remote.url, "POST", "/api/notes/commit")
    assert _notes_git.status_entries(remote.notes) == []
    tracked = set(_git(remote.notes, "ls-files").stdout.split())
    for who in range(len(clones)):
        for n in range(6):
            assert f"{who}-{n}.md" in tracked and (remote.notes / f"{who}-{n}.md").is_file()
            assert f"tab-{who}-{n}.md" in tracked and (remote.notes / f"tab-{who}-{n}.md").is_file()


def test_agents_merge_their_own_conflicts(remote) -> None:  # type: ignore[no-untyped-def]
    a, b = _clone(remote, "a"), _clone(remote, "b")
    _commit(a, "n.md", "from a\n")
    _commit(b, "n.md", "from b\n")
    _git(a, "push")
    refused = _git(b, "push", check=False)
    assert refused.returncode != 0 and "fetch first" in refused.stderr
    assert _git(b, "pull", "--no-edit", check=False).returncode != 0
    assert "<<<<<<<" in (b / "n.md").read_text()
    _commit(b, "n.md", "from a\nfrom b\n", "merge")
    _git(b, "push")
    assert (remote.notes / "n.md").read_text() == "from a\nfrom b\n"


def test_history_cannot_be_rewritten(remote) -> None:  # type: ignore[no-untyped-def]
    a = _clone(remote, "a")
    before = _notes_git.head(remote.notes)
    _commit(a, "x.md", "x\n")
    _git(a, "commit", "--amend", "-qm", "rewritten")
    _git(a, "push")
    _git(a, "commit", "--amend", "-qm", "rewritten again")
    assert _git(a, "push", "--force", check=False).returncode != 0
    assert _git(a, "push", "origin", "--delete", "main", check=False).returncode != 0
    assert _notes_git.head(remote.notes) != before
    assert _git(remote.notes, "log", "-1", "--format=%s").stdout.strip() == "rewritten"


def test_a_pushed_symlink_arrives_as_a_plain_file(remote) -> None:  # type: ignore[no-untyped-def]
    a = _clone(remote, "a")
    (a / "link").symlink_to("/etc/passwd")
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "link")
    _git(a, "push")
    served = remote.notes / "link"
    assert not served.is_symlink() and served.read_text() == "/etc/passwd"
    assert _notes_git.status_entries(remote.notes) == []


def test_large_push_and_compressed_fetch(remote) -> None:  # type: ignore[no-untyped-def]
    a, b = _clone(remote, "a"), _clone(remote, "b")
    (a / "big.bin").write_bytes(os.urandom(8 * 1024 * 1024))
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "big")
    _git(a, "push")
    assert (remote.notes / "big.bin").stat().st_size == 8 * 1024 * 1024
    # Enough local commits that git gzips the list of what it already has.
    for i in range(120):
        _git(b, "commit", "-q", "--allow-empty", "-m", f"local {i}")
    _git(b, "fetch", "-q", "origin")
    assert (_git(b, "rev-parse", "origin/main").stdout == _git(a, "rev-parse", "HEAD").stdout)


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


REFS = "/notes.git/info/refs?service=git-upload-pack"


def test_needs_the_token(remote) -> None:  # type: ignore[no-untyped-def]
    status, headers, _ = _get(remote.url + REFS)
    assert status == 401 and "Basic" in headers["WWW-Authenticate"]
    dest = remote.clones / "anon"
    dest.parent.mkdir(exist_ok=True)
    assert subprocess.run(["git", "clone", "-q", f"{remote.url}/notes.git", str(dest)],
                          capture_output=True, env=_env()).returncode != 0


def test_asking_without_credentials_does_not_lock_anyone_out(remote) -> None:  # type: ignore[no-untyped-def]
    for _ in range(40):  # more than the failed-login limit
        assert _get(remote.url + REFS)[0] == 401
    assert _get(remote.url + "/api/notes/status", {"Authorization": f"Bearer {remote.token}"})[0] == 200


@pytest.mark.parametrize("path", ["config", "HEAD", ".git/config", "objects/info/packs", "info/refs",
                                  "info/refs?service=git-upload-archive", "../manager.token"])
def test_serves_nothing_but_fetch_and_push(remote, path: str) -> None:  # type: ignore[no-untyped-def]
    status, _, body = _get(f"{remote.url}/notes.git/{path}", {"Authorization": f"Bearer {remote.token}"})
    assert status == 404 and remote.token.encode() not in body and b"repositoryformatversion" not in body


def test_status_names_the_remote_without_sync_state(remote) -> None:  # type: ignore[no-untyped-def]
    _, st, _ = _req(remote.url, "GET", "/api/notes/status")
    assert st["remote"] == f"{remote.url}/notes.git"
    assert not {"role", "syncing", "last_sync", "ahead"} & set(json.loads(json.dumps(st)))


def test_cloning_again_is_safe_and_refreshes_the_token(remote) -> None:  # type: ignore[no-untyped-def]
    dest = _clone(remote, "again")
    (dest / ".git" / "mlsweep-token").write_text("stale")
    (dest / "kept.md").write_text("mine\n")
    env = {**os.environ, "MLSWEEP_TOKEN": remote.token}
    done = subprocess.run(
        [sys.executable, "-c", "from mlsweep.cli import main; main()", "notes", "clone", str(dest),
         "--manager", remote.url], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and "already a clone" in done.stdout
    assert (dest / ".git" / "mlsweep-token").read_text() == remote.token
    assert (dest / "kept.md").exists()


def test_clone_refuses_a_folder_that_is_something_else(remote, tmp_path) -> None:  # type: ignore[no-untyped-def]
    other = tmp_path / "other"
    other.mkdir()
    (other / "file").write_text("x")
    env = {**os.environ, "MLSWEEP_TOKEN": remote.token}
    done = subprocess.run(
        [sys.executable, "-c", "from mlsweep.cli import main; main()", "notes", "clone", str(other),
         "--manager", remote.url], capture_output=True, text=True, env=env)
    assert done.returncode == 1 and "not empty" in done.stderr
