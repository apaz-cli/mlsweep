"""Link parsing, resolution and the lint tiers (``_notes``, ``_notes_lint``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlsweep import _notes, _notes_lint

EXP = "es_archs_20261013_0528_88e3"
FO = "rzo_e1_fo_20261001_0309_ff11"


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def test_clean_rel_rejects_escapes() -> None:
    for bad in ["../x", "a/../../x", "/etc/passwd", ".git/config", "a/.GIT/x", "a//b", "a\\b", "a\x00b", ""]:
        with pytest.raises(_notes.PathError):
            _notes.clean_rel(bad)
    assert _notes.clean_rel("notes/a b.md") == "notes/a b.md"


def test_resolve_refuses_symlinks_out(tmp_path: Path) -> None:
    (tmp_path / "out").symlink_to("/etc")
    with pytest.raises(_notes.PathError):
        _notes.resolve(tmp_path, "out/passwd")


def test_links_and_resolution() -> None:
    text = ("[[Note]] [[dir/x|alias]] ![[plot.png]] [rel](../b/c.md#h) `[[code]]`\n"
            "```\n[[fenced]]\n```\n")
    links = _notes.extract_links(text)
    assert [l.target for l in links] == ["Note", "dir/x", "plot.png", "../b/c.md"]
    r = _notes.Resolver(["a/Note.md", "dir/x.md", "img/plot.png", "b/c.md", "z/Note.md"])
    for l in links:
        r.resolve("a/src.md", l)
    assert [l.resolved for l in links] == ["a/Note.md", "dir/x.md", "img/plot.png", "b/c.md"]
    assert links[0].ambiguous == ["a/Note.md", "z/Note.md"]  # same folder wins


def test_rewrite_links_on_rename() -> None:
    r = _notes.Resolver(["a/Old.md", "a/src.md"])
    text = "see [[Old]], [[Old#Sec|s]], [x](Old.md) and `[[Old]]`"
    new, n = _notes.rewrite_links(text, "a/src.md", {"a/Old.md": "a/New name.md"}, r)
    assert n == 3
    assert new == "see [[New name]], [[New name#Sec|s]], [x](New%20name.md) and `[[Old]]`"


def test_mentions_and_search(tmp_path: Path) -> None:
    _write(tmp_path, {"n.md": f"Ran {EXP}.\nAnother line about LR warmup\n", "other.txt": "lr warmup only"})
    assert _notes.find_mentions((tmp_path / "n.md").read_text(), {EXP}) == {EXP: [1]}
    files, _ = _notes.walk(tmp_path)
    res = _notes.search(tmp_path, files, "lr warmup")
    assert {r["path"] for r in res} == {"n.md", "other.txt"}
    assert res[0]["matches"]


def test_numbers_are_left_to_agents(tmp_path: Path) -> None:
    _write(tmp_path, {"r.md": f"# Result\n\n{EXP} reaches 2.396, and Python 3.13 ran it.\n"})
    res = _notes_lint.lint(tmp_path, known_experiments={EXP})
    assert res["findings"] == [] and res["stats"]["experiments_checked"] is True


def test_unknown_experiment_and_links(tmp_path: Path) -> None:
    _write(tmp_path, {
        "a/n.md": "[[Real]] [[Missing note]] [x](gone/b.md) ![[nope.png]] [moved](old/place/t.md)\n"
                  "cites fake_sweep_20260101_0000_abcd\n",
        "a/Real.md": "x\n",
        "deep/t.md": "target\n",
        "sources/blog.md": "[[broken in a source]] fake2_20260101_0000_abcd\n",
    })
    res = _notes_lint.lint(tmp_path, known_experiments=set())
    kinds = sorted((f["kind"], f["path"]) for f in res["findings"])
    assert ("unknown-experiment", "a/n.md") in kinds
    assert all(p != "sources/blog.md" for _, p in kinds)  # originals are not linted
    fixes = [f for f in res["findings"] if f["tier"] == "fix"]
    assert [f["detail"]["new"] for f in fixes] == ["deep/t.md"]
    broken = {f["detail"]["target"] for f in res["findings"] if f["kind"] == "broken-link"}
    assert broken == {"Missing note", "gone/b.md", "nope.png"}
    changed = _notes_lint.apply_fixes(tmp_path, res["findings"])
    assert changed == ["a/n.md"]
    assert "[moved](../deep/t.md)" in (tmp_path / "a/n.md").read_text()


def test_merge_leftovers(tmp_path: Path) -> None:
    _write(tmp_path, {
        "m.md": "<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> b\n",
        "empty.md": "",
    })
    kinds = {f["kind"] for f in _notes_lint.lint(tmp_path)["findings"]}
    assert kinds == {"conflict-marker", "empty-file"}


def test_review_prompt_includes_report(tmp_path: Path) -> None:
    _write(tmp_path, {"n.md": "[[Nothing]]\n"})
    res = _notes_lint.lint(tmp_path)
    text = _notes_lint.review_prompt(tmp_path, res, "2026-01-01")
    assert "reviews/2026-01-01-review.md" in text and "Nothing" in text


def test_cli_mv_links_mentions(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys
    _write(tmp_path, {"a/Old.md": "x\n", "b/src.md": f"see [[Old]] and {EXP}\n"})
    run = lambda *a: subprocess.run([sys.executable, "-m", "mlsweep", "notes", *a, "--dir", str(tmp_path)],  # noqa: E731
                                    capture_output=True, text=True, env={**__import__("os").environ, "MLSWEEP_MANAGER": "http://127.0.0.1:9"})
    r = run("mv", "a/Old.md", "a/New.md")
    assert r.returncode == 0, r.stderr
    assert "[[New]]" in (tmp_path / "b/src.md").read_text()
    out = json.loads(run("links", "a/New.md", "--json").stdout)
    assert out["backlinks"][0]["src"] == "b/src.md"
    assert json.loads(run("mentions", EXP, "--json").stdout) == {"b/src.md": [1]}


def test_index_cache_rereads_only_changed_files(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.md").write_text("see [[c]]\n")
    (tmp_path / "b.md").write_text(f"{EXP}\n")
    reads: list[str] = []
    real = _notes.read_text

    def counting(path: Path, limit: int = _notes.MAX_TEXT_BYTES) -> str | None:
        reads.append(path.name)
        return real(path, limit)
    monkeypatch.setattr(_notes, "read_text", counting)
    cache = _notes.ParseCache()
    first = _notes.build_index(tmp_path, {EXP}, cache)
    assert sorted(reads) == ["a.md", "b.md"]

    # A new file is read, and an unchanged note's link now finds it.  The Index
    # already handed out keeps what it said.
    reads.clear()
    (tmp_path / "c.md").write_text("new\n")
    second = _notes.build_index(tmp_path, {EXP}, cache)
    assert reads == ["c.md"]
    assert second.links["a.md"][0].resolved == "c.md"
    assert first.links["a.md"][0].resolved is None
    assert second.mentions["b.md"] == {EXP: [1]}

    # Forgetting an ID drops its mentions without reading anything.
    reads.clear()
    assert _notes.build_index(tmp_path, set(), cache).mentions == {}
    assert reads == []

    # A new ID is found in notes that have not changed, and in text files.
    (tmp_path / "data.txt").write_text(f"x\n{EXP}\n")
    third = _notes.build_index(tmp_path, {EXP}, cache)
    assert third.mentions == {"b.md": {EXP: [1]}, "data.txt": {EXP: [2]}}
    assert reads == ["data.txt"]


def test_move_folder_updates_links_to_every_moved_note(tmp_path: Path) -> None:
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "one.md").write_text("[[two]]\n")
    (tmp_path / "old" / "two.md").write_text("two\n")
    (tmp_path / "top.md").write_text("[[old/one]] [[two|2]] [x](old/two.md#h)\n")
    res = _notes.move_path(tmp_path, "old", "new")
    assert res["moved"] == 2
    assert (tmp_path / "top.md").read_text() == "[[new/one]] [[two|2]] [x](new/two.md#h)\n"


@pytest.mark.parametrize("before,after", [
    ("[Old.md](Old.md)", "[Old.md](New.md)"),                    # the label is left alone
    ("[x](Old.md?v=2#sec)", "[x](New.md?v=2#sec)"),              # query and fragment stay
    ('[x](Old.md "Old.md")', '[x](New.md "Old.md")'),            # so is a title
    ("![Old.md](Old.md)", "![Old.md](New.md)"),
    ("[x](//Old.md)", "[x](//Old.md)"),                          # another host, not this file
    ("[x](https://a.example/Old.md)", "[x](https://a.example/Old.md)"),
])
def test_rewrite_changes_only_the_href(before: str, after: str) -> None:
    new, n = _notes.rewrite_links(before, "s.md", {"Old.md": "New.md"}, _notes.Resolver(["Old.md", "s.md"]))
    assert new == after and n == (0 if before == after else 1)


def test_lint_fix_changes_only_the_href(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "Real.md").write_text("x\n")
    (tmp_path / "s.md").write_text('[Real.md](Real.md) and [q](Real.md?x=1#h) and [t](Real.md "Real.md")\n')
    res = _notes_lint.lint(tmp_path)
    assert _notes_lint.apply_fixes(tmp_path, res["findings"]) == ["s.md"]
    assert (tmp_path / "s.md").read_text() == \
        '[Real.md](dir/Real.md) and [q](dir/Real.md?x=1#h) and [t](dir/Real.md "Real.md")\n'
    assert not [f for f in _notes_lint.lint(tmp_path)["findings"] if f["kind"] == "link-path"]


def test_mentions_stay_exact_when_an_id_ends_in_punctuation(tmp_path: Path) -> None:
    """"exp." and "exp" compete for one token, so a cached index must not differ."""
    (tmp_path / "a.md").write_text("exp_1. and\nexp_1\n")
    cache = _notes.ParseCache()
    for known in ({"exp_1."}, {"exp_1.", "exp_1"}, {"exp_1"}):
        assert _notes.build_index(tmp_path, known, cache).mentions == _notes.build_index(tmp_path, known).mentions


def _outside_a_clone(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "MLSWEEP_NOTES"}
    env["MLSWEEP_DIR"] = str(tmp_path / "state")
    return subprocess.run([sys.executable, "-m", "mlsweep", "notes", *args], cwd=tmp_path,
                          capture_output=True, text=True, env=env)


def test_writes_refuse_to_fall_back_to_the_managers_folder(tmp_path: Path) -> None:
    (tmp_path / "state" / "notes").mkdir(parents=True)
    (tmp_path / "state" / "notes" / "a.md").write_text("a\n")
    done = _outside_a_clone(tmp_path, "mv", "a.md", "b.md")
    assert done.returncode == 2 and "manager's folder" in done.stderr
    assert (tmp_path / "state" / "notes" / "a.md").exists()
    assert _outside_a_clone(tmp_path, "lint", "--fix", "--no-manager").returncode == 2


def test_reads_say_when_they_fall_back_and_fail_with_no_notes(tmp_path: Path) -> None:
    assert _outside_a_clone(tmp_path, "search", "a").returncode == 2
    (tmp_path / "state" / "notes").mkdir(parents=True)
    (tmp_path / "state" / "notes" / "a.md").write_text("alpha\n")
    done = _outside_a_clone(tmp_path, "search", "alpha")
    assert done.returncode == 0 and "a.md" in done.stdout and "manager's folder" in done.stderr


def test_chart_blocks_and_evidence_links(tmp_path: Path) -> None:
    _write(tmp_path, {"n.md": (
        "# Results\n\n"
        "```mlsweep-chart\n"
        "experiments: [real, ghost]\n"
        "metric: val_loss\n"
        "runs: [real/run_a, real/run_missing]\n"
        "```\n\n"
        "The loss was [0.42](mlsweep:real/run_a/val_loss@20000).\n"
        "Bad: [x](mlsweep:real/run_a) and [y](mlsweep:real/run_b/loss).\n"
        "Code is ignored: `[z](mlsweep:real/run_a/loss@1)`.\n"
        "\n```mlsweep-table\nnot a key value line\n```\n"
    )})
    res = _notes_lint.lint(tmp_path, known_experiments={"real"},
                           known_runs={"run_a": ["real"]})
    by_kind = {}
    for f in res["findings"]:
        by_kind.setdefault(f["kind"], []).append(f)
    assert res["stats"]["data_blocks"] == 2
    assert [f["detail"]["id"] for f in by_kind["unknown-experiment"]] == ["ghost"]
    assert {f["detail"]["id"] for f in by_kind["unknown-run"]} == {"real/run_missing"}
    assert len(by_kind["evidence-ref"]) == 2
    assert len(by_kind["block-spec"]) == 1
    assert any("key: value" in f["message"] for f in by_kind["block-spec"])

    # Without a manager, specs are still parsed and only structural problems show.
    res = _notes_lint.lint(tmp_path)
    kinds = {f["kind"] for f in res["findings"]}
    assert "unknown-experiment" not in kinds and "unknown-run" not in kinds
    assert "block-spec" in kinds


def test_caption_lines_without_a_figure_are_not_findings(tmp_path: Path) -> None:
    _write(tmp_path, {"n.md": "Figure: this is just a sentence.\n"})
    assert _notes_lint.lint(tmp_path)["findings"] == []


def test_stale_evidence_numbers(tmp_path: Path) -> None:
    _write(tmp_path, {"n.md": (
        "Matches [1.0](mlsweep:real/run_a/loss@2) and "
        "does not [9.0](mlsweep:real/run_a/loss@2).\n")})
    evidence = {"real/run_a/loss@2": {"ok": True, "ref": "real/run_a/loss@2",
                                      "value": 1.001, "step": 2}}
    res = _notes_lint.lint(tmp_path, known_experiments={"real"},
                           known_runs={"run_a": ["real"]}, evidence=evidence)
    stale = [f for f in res["findings"] if f["kind"] == "stale-evidence"]
    assert len(stale) == 1
    assert "9" in stale[0]["message"] and stale[0]["detail"]["value"] == 1.001
    assert res["stats"]["evidence_checked"] is True

    # Without resolved values the number check is skipped, not guessed.
    res = _notes_lint.lint(tmp_path, known_experiments={"real"},
                           known_runs={"run_a": ["real"]})
    assert not [f for f in res["findings"] if f["kind"] == "stale-evidence"]
    assert res["stats"]["evidence_checked"] is False


def test_evidence_refs_skips_code_and_sources(tmp_path: Path) -> None:
    _write(tmp_path, {
        "n.md": ("[a](mlsweep:e/r/loss@1) `[b](mlsweep:e/r/loss@2)`\n"
                 "```\n[c](mlsweep:e/r/loss@3)\n```\n"),
        "sources/paper.md": "[d](mlsweep:e/r/loss@4)\n",
    })
    index = _notes.build_index(tmp_path, set())
    assert _notes_lint.evidence_refs(index, tmp_path) == ["e/r/loss@1"]
