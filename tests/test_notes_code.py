"""Code in notes: ``mlsweep:code/...`` links, ``mlsweep-code`` / ``mlsweep-diff`` blocks,
their lint checks, and the code page they open.

Runs against a real manager holding two snapshots of a small project.  The
browser tests are skipped when no Chromium-family browser is installed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from browser import Browser, find_chromium
from campaign_helpers import TOKEN, call, shared_manager, static_url, uid
from mlsweep import _notes_lint
from mlsweep._notes_code import CodeRefError, parse_code_ref
from mlsweep._snapshot import scan_tree
from mlsweep.run_sweep import upload_snapshot

CHROME = find_chromium()

TRAIN_V1 = "import sys\n\ndef lr(step):\n    return 1e-3\n\nprint(lr(0))\n"
TRAIN_V2 = "import sys\n\ndef lr(step):\n    return 1e-3 * min(1, step / 100)\n\nprint(lr(0))\n"


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):  # type: ignore[no-untyped-def]
    yield from shared_manager(tmp_path_factory, "notes_code")


@pytest.fixture(scope="module")
def exps(mgr, tmp_path_factory) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Two experiments, the second with a warmup added to train.py and a new file."""
    root = tmp_path_factory.mktemp("proj")
    (root / "train.py").write_text(TRAIN_V1)
    (root / "data").mkdir()
    (root / "data" / "blob.bin").write_bytes(b"\0\1\2" * 100)
    (root / "big.py").write_text("".join(f"x{i} = {i}\n" for i in range(450)))
    out: dict[str, Any] = {}
    for name, edit in (("a", None), ("b", TRAIN_V2)):
        if edit:
            (root / "train.py").write_text(edit)
            (root / "warmup.py").write_text("STEPS = 100\n")
        uploaded = upload_snapshot(mgr.url, TOKEN, scan_tree(root))
        assert uploaded is not None
        sid = uploaded[0]
        exp = uid(f"code_{name}")
        status, _ = call(mgr, "POST", f"/api/experiments/{exp}/jobs/bulk", {
            "campaign": "test",
            "code": {"snapshot": sid, "sweep_file": "train.py", "sweep_file_in_snapshot": True},
            "jobs": [{"run_id": "r1", "command": ["true"], "artifact_id": sid}],
        })
        assert status == 201
        out[name], out[name + "_snap"] = exp, sid
    legacy = uid("code_legacy")
    call(mgr, "POST", "/api/artifacts", {"artifact_id": "sha256:" + "e" * 64})
    call(mgr, "POST", f"/api/experiments/{legacy}/jobs/bulk",
         {"campaign": "test", "jobs": [{"run_id": "r1", "command": ["true"], "artifact_id": "sha256:" + "e" * 64}]})
    out["legacy"] = legacy
    # An experiment appended to with newer code: its runs used both snapshots.
    multi = uid("code_multi")
    for i, key in enumerate(("a_snap", "b_snap")):
        status, _ = call(mgr, "POST", f"/api/experiments/{multi}/jobs/bulk", {
            "campaign": "test",
            "code": {"snapshot": out[key]},
            "jobs": [{"run_id": f"r{i}", "command": ["true"], "artifact_id": out[key]}],
        })
        assert status == 201
    out["multi"] = multi
    # An experiment with git provenance, including uncommitted changes.
    dirty = uid("code_dirty")
    diff = ("diff --git a/train.py b/train.py\nindex 1..2 100644\n--- a/train.py\n+++ b/train.py\n"
            "@@ -3,2 +3,2 @@\n def lr(step):\n-    return 1e-3\n+    return 1e-3 * min(1, step / 100)\n")
    status, _ = call(mgr, "POST", f"/api/experiments/{dirty}/jobs/bulk", {
        "campaign": "test",
        "code": {"snapshot": out["b_snap"], "sweep_file": "train.py", "sweep_file_in_snapshot": True,
                 "host": "box", "argv": ["sweeps/x.py"],
                 "vcs": {"type": "git", "commit": "a" * 40, "branch": "main", "subject": "Warmup",
                         "dirty": True, "changed_files": ["train.py"], "diff": diff}},
        "jobs": [{"run_id": "r1", "command": ["true"], "artifact_id": out["b_snap"]}],
    })
    assert status == 201
    out["dirty"] = dirty
    return out


def _resolve(mgr, refs: list[str]) -> dict[str, dict[str, Any]]:  # type: ignore[no-untyped-def]
    status, body = call(mgr, "POST", "/api/notes/evidence", {"refs": refs})
    assert status == 200
    return body["results"]  # type: ignore[no-any-return]


# ── references ────────────────────────────────────────────────────────────────


def test_parse_code_ref() -> None:
    r = parse_code_ref("mlsweep:code/exp_1/yoco/train.py#L40-60")
    assert (r.target, r.path, r.start, r.end) == ("exp_1", "yoco/train.py", 40, 60)
    assert parse_code_ref("code/exp_1/a.py#L7").end == 7
    assert parse_code_ref("code/snap:0123abcd/a.py").target == "snap:0123abcd"
    assert parse_code_ref("code/exp_1").path == ""
    for bad in ("code/../x", "code/exp/../../etc", "code/exp_1#L4", "code/exp/a.py#L9-3", "code/bad id/x",
                "code/exp/.git/config", "exp/run/metric"):
        with pytest.raises(CodeRefError):
            parse_code_ref(bad)


def test_resolve_files_lines_and_errors(mgr, exps) -> None:  # type: ignore[no-untyped-def]
    a, b = exps["a"], exps["b"]
    refs = {
        "file": f"code/{a}/train.py#L3-4",
        "tree": f"code/{a}",
        "dir": f"code/{a}/data",
        "snap": f"code/{exps['b_snap'][:17]}/warmup.py",
        "missing": f"code/{a}/warmup.py",
        "past_end": f"code/{a}/train.py#L5-99",
        "binary_lines": f"code/{a}/data/blob.bin#L1",
        "no_exp": "code/nope_123/train.py",
        "legacy": f"code/{exps['legacy']}/train.py",
    }
    res = _resolve(mgr, list(refs.values()))
    got = {k: res[v] for k, v in refs.items()}
    assert got["file"]["ok"] and got["file"]["kind"] == "file" and got["file"]["lines"] == 6
    assert got["file"]["experiment"] == a and got["file"]["snapshot"] == exps["a_snap"]
    assert got["tree"]["ok"] and got["tree"]["kind"] == "tree" and got["tree"]["files"] == 3
    assert got["dir"]["ok"] and got["dir"]["kind"] == "dir"
    assert got["snap"]["ok"] and got["snap"]["snapshot"] == exps["b_snap"]
    assert not got["missing"]["ok"] and "no file `warmup.py`" in got["missing"]["error"]
    assert not got["past_end"]["ok"] and "has 6 lines" in got["past_end"]["error"]
    assert not got["binary_lines"]["ok"] and "binary" in got["binary_lines"]["error"]
    assert not got["no_exp"]["ok"] and "no experiment" in got["no_exp"]["error"]
    assert not got["legacy"]["ok"] and "before snapshots" in got["legacy"]["error"]
    # Evidence links still resolve through the same endpoint.
    assert "ok" in _resolve(mgr, [f"{b}/r1/loss@final"])[f"{b}/r1/loss@final"]
    status, body = call(mgr, "GET", f"/api/code/resolve?ref=code/{a}/train.py%23L3")
    assert status == 200 and body["ok"] and body["start"] == 3


# ── blocks ────────────────────────────────────────────────────────────────────


def test_code_block_shows_the_lines(mgr, exps) -> None:  # type: ignore[no-untyped-def]
    spec = f"experiment: {exps['b']}\npath: train.py\nlines: 3-4\n"
    status, body = call(mgr, "POST", "/api/notes/block", {"kind": "code", "spec": spec})
    assert status == 200, body
    assert body["lines"] == ["def lr(step):", "    return 1e-3 * min(1, step / 100)"]
    assert (body["start"], body["end"], body["total_lines"]) == (3, 4, 6)
    assert body["ref"] == f"code/{exps['b']}/train.py#L3-4"
    status, body = call(mgr, "POST", "/api/notes/block",
                        {"kind": "code", "spec": f"experiment: {exps['b']}\npath: nope.py"})
    assert status == 400 and "no file" in body["error"]


def test_diff_block_shows_what_changed(mgr, exps) -> None:  # type: ignore[no-untyped-def]
    spec = f"from: {exps['a']}\nto: {exps['b']}\n"
    status, body = call(mgr, "POST", "/api/notes/block", {"kind": "diff", "spec": spec})
    assert status == 200, body
    assert body["counts"] == {"changed": 1, "added": 1, "removed": 0} and not body["same"]
    files = {f["path"]: f for f in body["files"]}
    assert files["warmup.py"]["status"] == "A"
    train = files["train.py"]
    assert train["status"] == "M" and train["added"] == 1 and train["removed"] == 1
    assert "+    return 1e-3 * min(1, step / 100)" in train["diff"]
    status, body = call(mgr, "POST", "/api/notes/block",
                        {"kind": "diff", "spec": f"from: {exps['a']}\nto: {exps['a']}\n"})
    assert status == 200 and body["same"] and body["files"] == []


# ── lint ──────────────────────────────────────────────────────────────────────


def test_lint_reports_code_links_that_do_not_resolve(mgr, exps, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    a, b = exps["a"], exps["b"]
    (tmp_path / "n.md").write_text(
        f"Good: [lr](mlsweep:code/{a}/train.py#L3-4). Bad: [w](mlsweep:code/{a}/warmup.py).\n\n"
        f"```mlsweep-code\nexperiment: {b}\npath: train.py\nlines: 50-60\n```\n\n"
        f"```mlsweep-diff\nfrom: {a}\nto: nope_999\n```\n\n"
        f"```mlsweep-code\nexperiment: {b}\n```\n")
    index = _notes_lint._notes.build_index(tmp_path, {a, b})
    refs = _notes_lint.evidence_refs(index, tmp_path)
    resolved = _resolve(mgr, refs)
    result = _notes_lint.lint(tmp_path, known_experiments={a, b}, evidence=resolved, index=index)
    msgs = [f["message"] for f in result["findings"] if f["kind"] in ("code-ref", "block-spec")]
    assert any("warmup.py" in m for m in msgs)
    assert any("has 6 lines" in m for m in msgs)
    assert any("nope_999" in m for m in msgs)
    assert any("`path` is required" in m for m in msgs)
    assert not any("train.py#L3-4" in m for m in msgs)
    assert not any(f["kind"] == "evidence-ref" for f in result["findings"])


def test_lint_catches_the_common_mistakes(mgr, exps, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    a, b, multi = exps["a"], exps["b"], exps["multi"]
    (tmp_path / "n.md").write_text(
        f"Forgot the prefix: [lr](mlsweep:{a}/train.py#L3).\n"
        f"Appended code: [lr](mlsweep:code/{multi}/train.py#L3).\n"
        f"Pinned: [lr](mlsweep:code/{exps['a_snap'][:17]}/train.py#L3).\n"
        f"Space in a path: [x](mlsweep:code/{a}/no%20such.py).\n\n"
        f"```mlsweep-code\nexperiment: {a}\npath: big.py\n```\n\n"
        f"```mlsweep-code\nexperiment: {a}\npath: big.py\nlines: 1-40\n```\n\n"
        f"```mlsweep-diff\nfrom: {a}\nto: {b}\npath: nowhere/\n```\n\n"
        f"```mlsweep-diff\nfrom: {a}\nto: {b}\npath: warmup.py\n```\n")
    index = _notes_lint._notes.build_index(tmp_path, {a, b, multi})
    resolved = _resolve(mgr, _notes_lint.evidence_refs(index, tmp_path))
    result = _notes_lint.lint(tmp_path, known_experiments={a, b, multi}, evidence=resolved, index=index)
    by_line: dict[int, list[str]] = {}
    for f in result["findings"]:
        by_line.setdefault(f["line"], []).append(f["message"])
    assert any("starts with `mlsweep:code/`" in m for m in by_line[1])
    assert any("used 2 code versions" in m for m in by_line[2])
    assert 3 not in by_line                      # a snapshot ID pins the version
    assert any("no file `no such.py`" in m for m in by_line[4])
    assert any("shows 450 lines" in m for m in by_line[6])
    assert 11 not in by_line                     # lines: 1-40 fits
    assert any("in neither experiment's code" in m for m in by_line[17])
    assert 23 not in by_line                     # warmup.py exists on one side, so the diff is not empty


# ── browser ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def page(mgr):  # type: ignore[no-untyped-def]
    if CHROME is None:
        pytest.skip("no Chromium-family browser installed")
    b = Browser(CHROME)
    yield b
    b.close()


def test_code_page_shows_the_file_with_the_lines_highlighted(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    # A note's link: ?ref=code/EXP/path#Llines.
    page.goto(static_url(mgr, "code.html", ref=f"code/{exps['b']}/train.py%23L3-4"))
    page.wait_for("document.querySelectorAll('table.src tr').length === 6")
    assert page.eval("[...document.querySelectorAll('tr.hl td.n')].map(t => t.textContent).join(',')") == "3,4"
    assert "min(1, step / 100)" in page.eval("document.querySelectorAll('table.src td.c')[3].textContent")
    assert page.eval("document.getElementById('sel').textContent") == "lines 3–4"
    assert page.eval("document.querySelectorAll('#file-tree .tnode.file').length") == 4
    assert page.eval("document.getElementById('exp-select').value") == exps["b"]
    assert "train.py" in page.eval("document.getElementById('hdr-meta').textContent")
    # Clicking a line number selects it and updates the address.
    page.eval("document.querySelector('td.n[data-n=\"6\"]').click()")
    q = page.eval("Object.fromEntries(new URLSearchParams(location.search))")
    assert (q["experiment"], q["path"], q["lines"]) == (exps["b"], "train.py", "6")


def test_code_page_follows_the_run(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    # An experiment whose runs used two versions shows the chosen run's.
    page.goto(static_url(mgr, "code.html", experiment=exps["multi"], run="r0", path="train.py"))
    page.wait_for("document.querySelectorAll('table.src tr').length === 6")
    assert "return 1e-3" in page.eval("document.querySelectorAll('table.src td.c')[3].textContent")
    assert "v1 of 2" in page.eval("document.getElementById('hdr-meta').textContent")
    assert page.eval("document.querySelector('.job-item.selected').dataset.runId") == "r0"
    # Clicking the other run switches to its code.
    page.eval("document.querySelector('.job-item[data-run-id=\"r1\"]').click()")
    page.wait_for("(document.querySelectorAll('table.src td.c')[3] || {}).textContent?.includes('min(1, step / 100)')")
    assert "v2 of 2" in page.eval("document.getElementById('hdr-meta').textContent")


def test_code_page_explains_a_legacy_experiment(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, "code.html", ref=f"code/{exps['legacy']}"))
    page.wait_for("document.getElementById('no-sel').textContent.includes('before snapshots')")


def test_notes_render_code_links_and_blocks(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, "notes.html"))
    page.wait_for("typeof renderMarkdown === 'function' && typeof annotateCode === 'function'")
    a, b = exps["a"], exps["b"]
    text = (f"See [the schedule](mlsweep:code/{b}/train.py#L3-4) and [gone](mlsweep:code/{a}/warmup.py).\n\n"
            f"```mlsweep-code\nexperiment: {b}\npath: train.py\nlines: 3-4\n```\n\n"
            f"```mlsweep-diff\nfrom: {a}\nto: {b}\n```\n")
    ident = page.eval(
        "(() => { const d = document.createElement('div'); d.className = 'doc'; document.body.append(d); "
        "renderMarkdown(%s, 'n.md', d); return d.id = 'out-' + Math.random().toString(36).slice(2); })()"
        % json.dumps(text))
    root = f"document.getElementById({json.dumps(ident)})"
    page.wait_for(f"{root}.querySelectorAll('a.code-ok, a.code-bad').length === 2")
    good = page.eval(f"(() => {{ const x = {root}.querySelector('a.code-ok'); return [x.textContent, x.getAttribute('href'), x.title]; }})()")
    assert good[0] == "the schedule" and "code.html" in good[1] and "lines 3–4" in good[2]
    assert "no file" in page.eval(f"{root}.querySelector('a.code-bad').title")
    page.wait_for(f"{root}.querySelector('.nb-code .nb-src') !== null && {root}.querySelector('.nb-diff') !== null")
    assert page.eval(f"{root}.querySelector('.nb-code .nb-gutter').textContent") == "3\n4"
    assert "min(1, step / 100)" in page.eval(f"{root}.querySelector('.nb-code .nb-src').textContent")
    assert page.eval(f"{root}.querySelectorAll('.nb-diff-file').length") == 2
    assert page.eval(f"{root}.querySelectorAll('.nb-diff-table tr.add').length") >= 1
    assert "code.html" in page.eval(f"{root}.querySelector('.nb-code').closest('.nb-block').querySelector('.nb-open').href")


@pytest.mark.parametrize("page_name, button", [("logs", "code-btn"), ("files", "run-code-btn")])
def test_logs_and_artifacts_link_to_the_runs_code(page, mgr, exps, page_name, button) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, f"{page_name}.html", experiment=exps["b"], run="r1"))
    page.wait_for(f"document.getElementById('{button}').style.display === ''")
    href = page.eval(f"document.getElementById('{button}').getAttribute('href')")
    assert href.startswith("code.html?") and f"experiment={exps['b']}" in href and "run=r1" in href



# ── the code API the Code tab uses ────────────────────────────────────────────


def test_experiment_code_is_light_and_the_full_submission_is_on_demand(mgr, exps) -> None:  # type: ignore[no-untyped-def]
    status, info = call(mgr, "GET", f"/api/experiments/{exps['dirty']}/code")
    assert status == 200
    vcs = info["submissions"][0]["meta"]["vcs"]
    assert "diff" not in vcs and vcs["diff_bytes"] > 0
    status, full = call(mgr, "GET", f"/api/code/submissions/{info['submissions'][0]['submission_id']}")
    assert status == 200 and "+    return 1e-3 * min(1, step / 100)" in full["meta"]["vcs"]["diff"]
    assert call(mgr, "GET", "/api/code/submissions/999999")[0] == 404


def test_code_stays_visible_after_its_runs_are_purged(mgr, exps) -> None:  # type: ignore[no-untyped-def]
    exp = uid("code_purged")
    call(mgr, "POST", f"/api/experiments/{exp}/jobs/bulk", {
        "campaign": "test",
        "code": {"snapshot": exps["a_snap"]},
        "jobs": [{"run_id": "r1", "command": ["true"], "artifact_id": exps["a_snap"]}]})
    call(mgr, "PUT", "/api/jobs/r1/status", {"experiment_id": exp, "status": "done", "exit_code": 0})
    status, _ = call(mgr, "POST", f"/api/experiments/{exp}/jobs/purge", {"run_ids": ["r1"]})
    assert status == 200
    status, info = call(mgr, "GET", f"/api/experiments/{exp}/code")
    assert info["versions"] == [{"artifact_id": exps["a_snap"], "runs": 0, "kind": "snapshot", "files": 3,
                                 "size_bytes": info["versions"][0]["size_bytes"]}]
    assert _resolve(mgr, [f"code/{exp}/train.py"])[f"code/{exp}/train.py"]["ok"]


def test_diff_usage_zip_and_caching(mgr, exps) -> None:  # type: ignore[no-untyped-def]
    import io
    import zipfile
    status, d = call(mgr, "GET", f"/api/code/diff?from={exps['a']}&to={exps['b_snap']}")
    assert status == 200 and {"path": "warmup.py", "status": "A"} in d["changes"]
    status, usage = call(mgr, "GET", f"/api/snapshots/{exps['a_snap']}/usage")
    assert status == 200 and exps["a"] in {e["experiment_id"] for e in usage["experiments"]}
    status, body = call(mgr, "GET", f"/api/snapshots/{exps['b_snap']}/archive.zip")
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        assert zf.read("train.py").decode() == TRAIN_V2 and "data/blob.bin" in zf.namelist()
    req = urllib.request.Request(f"{mgr.url}/api/snapshots/{exps['b_snap']}/files/train.py",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req) as r:
        etag, cache = r.headers["ETag"], r.headers["Cache-Control"]
    assert "immutable" in cache
    req.add_header("If-None-Match", etag)
    try:
        urllib.request.urlopen(req)
        raise AssertionError("expected 304")
    except urllib.error.HTTPError as e:
        assert e.code == 304


# ── the Code tab ──────────────────────────────────────────────────────────────


def test_code_page_opens_a_directory(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, "code.html", ref=f"code/{exps['a']}/data"))
    page.wait_for("document.querySelector('table.dirlist') !== null")
    assert page.eval("[...document.querySelectorAll('table.dirlist td.name')].map(t => t.textContent)") == ["blob.bin"]
    assert page.eval("document.querySelector('#file-tree .tnode.active').dataset.path") == "data"
    # The breadcrumb's first part goes back to the top.
    page.eval("document.querySelector('#crumbs a').click()")
    page.wait_for("[...document.querySelectorAll('table.dirlist td.name')].some(t => t.textContent === 'data/')")


def test_code_page_versions_and_compare(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, "code.html", experiment=exps["multi"], path="train.py"))
    page.wait_for("document.querySelectorAll('table.src tr').length === 6")
    assert page.eval("document.getElementById('version-select').options.length") == 2
    # Compare the latest (v2) with v1, picked from a list rather than typed.
    page.eval("document.getElementById('compare-btn').click()")
    page.wait_for("!!document.querySelector('.ml-pick')")
    groups = page.eval("[...document.querySelectorAll('.ml-pick-group')].map(g => g.textContent)")
    assert groups[0] == "This experiment's other versions" and any(g.startswith("Experiments") for g in groups)
    page.eval("[...document.querySelectorAll('.ml-pick-item')].find(x => x.textContent.startsWith('v1')).click()")
    page.wait_for("document.getElementById('compare-bar').classList.contains('on')")
    page.wait_for("document.querySelector('table.src.diff') !== null")
    marks = page.eval("Object.fromEntries([...document.querySelectorAll('#file-tree .tnode.file')]"
                      ".map(r => [r.dataset.path, r.querySelector('.mark').textContent]))")
    assert marks == {"train.py": "M", "warmup.py": "A"}
    assert page.eval("document.querySelectorAll('table.src.diff tr.add').length") == 1
    assert "compare=" in page.eval("location.search")
    page.eval("document.getElementById('compare-stop').click()")
    page.wait_for("!document.getElementById('compare-bar').classList.contains('on')")
    # Back returns to the comparison.
    page.eval("history.back()")
    page.wait_for("document.getElementById('compare-bar').classList.contains('on')")


def test_code_page_details_show_the_uncommitted_diff(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, "code.html", experiment=exps["dirty"]))
    page.wait_for("document.getElementById('hdr-meta').textContent.includes('uncommitted')")
    assert "aaaaaaaa main" in page.eval("document.getElementById('hdr-meta').textContent")
    page.eval("document.querySelector('#hdr-meta a.warn').click()")
    page.wait_for("document.querySelector('#details table.src.diff tr.add') !== null")
    text = page.eval("document.getElementById('details').textContent")
    assert "Warmup" in text and "mlsweep run sweeps/x.py" in text and "box" in text


def test_results_run_bar_links_to_code(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    src = (Path(__file__).parent.parent / "mlsweep" / "webui" / "page-results.js").read_text()
    assert 'runLink("code", r.hash)' in src



def _ctx_labels(page):  # type: ignore[no-untyped-def]
    page.wait_for("!!document.querySelector('.ml-ctx')")
    return page.eval("[...document.querySelectorAll('.ml-ctx-item')].map(b => b.querySelector('.l').textContent)")


def _right_click(page, selector):  # type: ignore[no-untyped-def]
    page.eval(f"""(() => {{ const e = document.querySelector({json.dumps(selector)}); const r = e.getBoundingClientRect();
        e.dispatchEvent(new MouseEvent('contextmenu', {{bubbles: true, cancelable: true, clientX: r.x + 5, clientY: r.y + 5}})); }})()""")


def test_code_page_right_click_menus(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, "code.html", experiment=exps["multi"], path="train.py"))
    page.wait_for("document.querySelectorAll('table.src tr').length === 6")
    # A file in the tree.
    _right_click(page, '#file-tree .tnode[data-path="train.py"]')
    labels = _ctx_labels(page)
    assert "Compare with the previous version" in labels and "Copy note link" in labels and "Download file" in labels
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent.startsWith('Compare with the previous')).click()")
    page.wait_for("document.querySelector('table.src.diff') !== null")
    page.eval("document.getElementById('compare-stop').click()")
    page.wait_for("document.querySelectorAll('table.src:not(.diff) tr').length === 6")
    # A line number: selects the line, then offers links to it.
    _right_click(page, 'td.n[data-n="4"]')
    labels = _ctx_labels(page)
    assert labels[0] == "Copy note link to line 4" and "Copy as a code block for a note" in labels
    assert page.eval("[...document.querySelectorAll('tr.hl td.n')].map(t => t.textContent).join(',')") == "4"
    page.eval("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}))")
    # A run in the sidebar that used other code.
    _right_click(page, '.job-item[data-run-id="r0"]')
    labels = _ctx_labels(page)
    assert labels[:2] == ["Show this run's code", "Compare with the code shown now"] and "Open logs" in labels
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent.startsWith('Compare with the code')).click()")
    page.wait_for("document.getElementById('compare-bar').classList.contains('on')")
    assert "v1 of this experiment" in page.eval("document.getElementById('compare-text').textContent")


def test_notes_complete_experiment_ids(page, mgr, exps) -> None:  # type: ignore[no-untyped-def]
    page.goto(static_url(mgr, "notes.html"))
    page.wait_for("typeof expContext === 'function' && S.experiments.size > 0")
    assert page.eval("expContext('see [lr](mlsweep:code/code_b', false)") == "code_b"
    assert page.eval("expContext('the value [0.4](mlsweep:', false)") == ""
    assert page.eval("expContext('experiment: code_', true)") == "code_"
    assert page.eval("expContext('experiments: a, code_', true)") == "code_"
    assert page.eval("expContext('experiment: code_', false)") is None   # only inside a data block
    assert page.eval("expContext('just prose', true)") is None
    ids = page.eval(f"expMatches({json.dumps(exps['b'][:12])}).map(x => x.id)")
    assert exps["b"] in ids
