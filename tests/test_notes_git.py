"""Notes git plumbing (``_notes_git``), on local repos.  ``test_notes_remote`` covers the remote."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mlsweep import _notes_git as g


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_init_writes_skeleton_and_commits(tmp_path: Path) -> None:
    g.init_repo(tmp_path / "n", name="me", email="m@x")
    for f in (".gitignore", ".mlsweep-notes.json", "README.md", "AGENTS.md", "sources/README.md"):
        assert (tmp_path / "n" / f).is_file(), f
    assert not g.status_entries(tmp_path / "n") and g.has_commits(tmp_path / "n")


def test_init_commits_existing_files(tmp_path: Path) -> None:
    d = tmp_path / "existing"
    d.mkdir()
    (d / "old.md").write_text("old notes\n")
    g.init_repo(d)
    assert "old.md" in _git(d, "ls-files")


def test_init_keeps_files_that_exist(tmp_path: Path) -> None:
    d = tmp_path / "n"
    d.mkdir()
    (d / "AGENTS.md").write_text("mine\n")
    g.init_repo(d)
    assert (d / "AGENTS.md").read_text() == "mine\n"


def test_commit_all_counts_and_names_files(tmp_path: Path) -> None:
    g.init_repo(tmp_path / "n", name="me", email="m@x")
    repo = tmp_path / "n"
    assert g.commit_all(repo) == 0
    (repo / "a b.md").write_text("1\n")
    (repo / "c.md").write_text("2\n")
    assert g.commit_all(repo) == 2
    assert _git(repo, "log", "-1", "--format=%s") == "me: a b.md, c.md"
    assert not g.status_entries(repo)


def test_configure_served_keeps_history_safe(tmp_path: Path) -> None:
    g.init_repo(tmp_path / "n")
    g.configure_served(tmp_path / "n")
    for key, want in (("receive.denyCurrentBranch", "updateInstead"), ("receive.denyNonFastForwards", "true"),
                      ("receive.denyDeletes", "true"), ("core.symlinks", "false")):
        assert _git(tmp_path / "n", "config", "--get", key) == want


def test_log_entries_and_history(tmp_path: Path) -> None:
    g.init_repo(tmp_path / "n")
    repo = tmp_path / "n"
    (repo / "a.md").write_text("1\n")
    g.commit_all(repo, message="first")
    (repo / "a.md").write_text("2\n")
    g.commit_all(repo, message="second")
    entries = g.log_entries(repo, "a.md")
    assert [e["subject"] for e in entries] == ["second", "first"]
    feed = g.log_entries(repo, limit=1, with_files=True)
    assert feed[0]["files"] == [{"status": "M", "path": "a.md"}]
    assert g.show_file(repo, entries[1]["hash"], "a.md") == b"1\n"
    assert "+2" in g.commit_diff(repo, entries[0]["hash"], "a.md")


def test_starter_agents_md_lints_clean(tmp_path: Path) -> None:
    from mlsweep import _notes_lint
    (tmp_path / "AGENTS.md").write_text(g.AGENTS_MD)
    found = _notes_lint.lint(tmp_path, known_experiments=None)["findings"]
    assert [f for f in found if f.get("kind") == "broken-link"] == []
