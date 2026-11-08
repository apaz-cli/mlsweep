"""The Notes HTTP API on a real manager started with ``--notes``."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import _start_manager, _teardown_manager
from mlsweep import _notes_git

TOKEN = "test-token"


def _req(url: str, method: str, path: str, data: object = None, *, raw: bool = False,
         token: str | None = TOKEN, headers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    body = json.dumps(data).encode() if data is not None else None
    h = {"Authorization": f"Bearer {token}"} if token else {}
    if body is not None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    r = urllib.request.Request(url + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            content = resp.read()
            return resp.status, (content if raw else json.loads(content)), dict(resp.headers)
    except urllib.error.HTTPError as e:
        content = e.read()
        try:
            parsed = json.loads(content)
        except ValueError:
            parsed = content
        return e.code, parsed, dict(e.headers)


@pytest.fixture
def notes_manager(tmp_path: Path):  # type: ignore[no-untyped-def]
    notes = tmp_path / "notes"
    _notes_git.init_repo(notes, name="dash", email="d@x")
    (notes / "zo").mkdir()
    (notes / "zo" / "Results.md").write_text("# Results\n\nSee [[Plan]].\n")
    (notes / "Plan.md").write_text("# Plan\n\n## Steps\n\nRun it.\n")
    (notes / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (notes / "page.html").write_text("<script>alert(1)</script>")
    _notes_git.commit_all(notes)
    proc, server, url = _start_manager(tmp_path, extra_args=("--notes", str(notes)))
    server.notes = notes
    yield server
    _teardown_manager(proc)


def test_notes_folder_created_by_default(manager_server):  # type: ignore[no-untyped-def]
    server, url = manager_server
    status, st, _ = _req(url, "GET", "/api/notes/status")
    assert status == 200 and st["enabled"] and st["root"] == str(Path(server.mlsweep_dir).resolve() / "notes")
    assert (Path(server.mlsweep_dir) / "notes" / "AGENTS.md").is_file()


def test_disabled_with_no_notes(tmp_path: Path):  # type: ignore[no-untyped-def]
    proc, _, url = _start_manager(tmp_path, extra_args=("--no-notes",))
    try:
        _check_disabled(url)
    finally:
        _teardown_manager(proc)


def _check_disabled(url: str) -> None:
    status, body, _ = _req(url, "GET", "/api/notes/status")
    assert status == 200 and body == {"enabled": False}
    status, body, _ = _req(url, "GET", "/api/notes/tree")
    assert status == 404 and body["enabled"] is False
    status, body, _ = _req(url, "GET", "/api/notes/mentions")
    assert body == {"enabled": False, "mentions": {}, "runs": {}}


def test_requires_token(notes_manager) -> None:  # type: ignore[no-untyped-def]
    for path in ("/api/notes/tree", "/api/notes/file?path=Plan.md", "/api/notes/raw?path=fig.png"):
        assert _req(notes_manager.url, "GET", path, token=None)[0] == 401
        assert _req(notes_manager.url, "GET", path, token="wrong")[0] == 401


def test_status_tree_file(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    _, st, _ = _req(url, "GET", "/api/notes/status")
    assert st["enabled"] and st["branch"] == "main" and st["protected"] == ["sources/"]
    assert st["remote"] == f"{url}/notes.git"
    _, tree, _ = _req(url, "GET", "/api/notes/tree")
    paths = {f["path"] for f in tree["files"]}
    assert {"Plan.md", "zo/Results.md", "fig.png", "sources/README.md"} <= paths
    assert not any(p.startswith(".git/") for p in paths)
    _, f, _ = _req(url, "GET", "/api/notes/file?path=Plan.md")
    assert f["content"].startswith("# Plan") and f["hash"] == hashlib.sha256(f["content"].encode()).hexdigest()
    assert f["last_commit"]["author"] == "dash"
    _, src, _ = _req(url, "GET", "/api/notes/file?path=sources/README.md")
    assert src["protected"] is True


def test_save_is_compare_and_swap(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    _, f, _ = _req(url, "GET", "/api/notes/file?path=Plan.md")
    status, saved, _ = _req(url, "PUT", "/api/notes/file", {"path": "Plan.md", "content": "v2\n", "base_hash": f["hash"]})
    assert status == 200 and (notes_manager.notes / "Plan.md").read_text() == "v2\n"
    # Someone else (a push) changes the file; a save based on the old version is refused.
    (notes_manager.notes / "Plan.md").write_text("from an agent\n")
    status, body, _ = _req(url, "PUT", "/api/notes/file", {"path": "Plan.md", "content": "v3\n", "base_hash": saved["hash"]})
    assert status == 409 and body["current_content"] == "from an agent\n"
    assert (notes_manager.notes / "Plan.md").read_text() == "from an agent\n"
    # Creating over an existing file is refused too.
    status, body, _ = _req(url, "PUT", "/api/notes/file", {"path": "Plan.md", "content": "x", "base_hash": None})
    assert status == 409 and body["exists"]
    status, body, _ = _req(url, "PUT", "/api/notes/file", {"path": "new/deep/note.md", "content": "hi\n", "base_hash": None})
    assert status == 200 and (notes_manager.notes / "new/deep/note.md").read_text() == "hi\n"


@pytest.mark.parametrize("path", ["../escape.md", "/etc/passwd", ".git/config", "a/../../x", ".GIT/hooks/x", "a\\b"])
def test_paths_are_confined(notes_manager, path: str) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    q = urllib.request.quote(path)
    assert _req(url, "GET", f"/api/notes/file?path={q}")[0] == 400
    assert _req(url, "GET", f"/api/notes/raw?path={q}")[0] == 400
    assert _req(url, "PUT", "/api/notes/file", {"path": path, "content": "x", "base_hash": None})[0] == 400


def test_symlinks_out_are_refused(notes_manager, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    os.symlink(secret, notes_manager.notes / "link.md")
    os.symlink(tmp_path, notes_manager.notes / "linkdir")
    url = notes_manager.url
    assert _req(url, "GET", "/api/notes/file?path=link.md")[0] == 400
    assert _req(url, "GET", "/api/notes/raw?path=linkdir/secret.txt")[0] == 400
    assert _req(url, "PUT", "/api/notes/file", {"path": "linkdir/x.md", "content": "x", "base_hash": None})[0] == 400
    _, tree, _ = _req(url, "GET", "/api/notes/tree")
    assert "link.md" not in {f["path"] for f in tree["files"]}


def test_raw_never_serves_active_content(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    status, body, h = _req(url, "GET", "/api/notes/raw?path=page.html", raw=True)
    assert status == 200 and h["Content-Type"].startswith("text/plain")
    assert "sandbox" in h["Content-Security-Policy"] and h["X-Content-Type-Options"] == "nosniff"
    status, body, h = _req(url, "GET", "/api/notes/raw?path=fig.png", raw=True)
    assert h["Content-Type"] == "image/png" and body.startswith(b"\x89PNG")
    (notes_manager.notes / "x.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><script>1</script></svg>")
    _, _, h = _req(url, "GET", "/api/notes/raw?path=x.svg", raw=True)
    assert "sandbox" in h["Content-Security-Policy"]
    _, _, h = _req(url, "GET", "/api/notes/raw?path=page.html&download=1", raw=True)
    assert "attachment" in h["Content-Disposition"] and h["Content-Type"] == "application/octet-stream"


def test_move_updates_links(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    status, r, _ = _req(url, "POST", "/api/notes/move", {"from": "Plan.md", "to": "plans/Depth plan.md"})
    assert status == 200 and r["links_updated_in"] == ["zo/Results.md"]
    assert "[[Depth plan]]" in (notes_manager.notes / "zo/Results.md").read_text()
    assert _req(url, "POST", "/api/notes/move", {"from": "zo/Results.md", "to": "plans/Depth plan.md"})[0] == 409
    status, _, _ = _req(url, "POST", "/api/notes/move", {"from": "zo", "to": "zo/inner"})
    assert status == 400


def test_delete_and_upload(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    boundary = "XyZ"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"../../plot.png\"\r\n"
            f"Content-Type: image/png\r\n\r\nPNGDATA\r\n--{boundary}--\r\n").encode()
    r = urllib.request.Request(url + "/api/notes/upload?dir=figs", data=body, method="POST",
                               headers={"Authorization": f"Bearer {TOKEN}",
                                        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    saved = json.loads(urllib.request.urlopen(r).read())["saved"]
    assert saved == ["figs/plot.png"] and (notes_manager.notes / "figs/plot.png").read_bytes() == b"PNGDATA"
    assert _req(url, "DELETE", "/api/notes/file?path=figs")[0] == 409  # not empty
    assert _req(url, "DELETE", "/api/notes/file?path=figs&recursive=1")[0] == 200
    assert not (notes_manager.notes / "figs").exists()


def test_search_index_history(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    _, s, _ = _req(url, "GET", "/api/notes/search?q=" + urllib.request.quote('"run it." steps'))
    assert [r["path"] for r in s["results"]] == ["Plan.md"]
    _, idx, _ = _req(url, "GET", "/api/notes/index")
    assert idx["links"]["zo/Results.md"][0]["resolved"] == "Plan.md"
    _, h, _ = _req(url, "GET", "/api/notes/history?path=Plan.md")
    rev = h["commits"][0]["hash"]
    _, show, _ = _req(url, "GET", f"/api/notes/show?path=Plan.md&rev={rev}")
    assert show["content"].startswith("# Plan")
    assert _req(url, "GET", "/api/notes/show?path=Plan.md&rev=HEAD;rm")[0] == 400
    _, d, _ = _req(url, "GET", f"/api/notes/diff?rev={rev}")
    assert "Plan.md" in d["diff"]


def test_saves_are_committed(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    _req(url, "PUT", "/api/notes/file", {"path": "saved.md", "content": "hello\n", "base_hash": None})
    status, r, _ = _req(url, "POST", "/api/notes/commit")
    assert status == 200 and r["error"] is None and r["committed"] == 1
    tree = subprocess.run(["git", "-C", str(notes_manager.notes), "ls-tree", "-r", "--name-only", "main"],
                          capture_output=True, text=True).stdout
    assert "saved.md" in tree


def test_lint_endpoints(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    (notes_manager.notes / "bad.md").write_text("[[Nope]] and fake_20260101_0000_abcd\n")
    _, res, _ = _req(url, "GET", "/api/notes/lint")
    kinds = {f["kind"] for f in res["findings"]}
    assert {"broken-link", "unknown-experiment"} <= kinds and res["stats"]["experiments_checked"] is True
    _, fix, _ = _req(url, "POST", "/api/notes/lint/fix")
    assert fix["changed"] == []
    _, prompt, _ = _req(url, "GET", "/api/notes/review-prompt")
    assert "reviews/" in prompt["prompt"]
    _, ids, _ = _req(url, "GET", "/api/notes/experiment-ids")
    assert isinstance(ids["known"], list)


@pytest.mark.parametrize("method,path", [("POST", "/api/notes/mkdir"), ("POST", "/api/notes/move"),
                                         ("PUT", "/api/notes/file")])
@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2]"])
def test_bad_json_is_a_400(notes_manager, method: str, path: str, raw: bytes) -> None:  # type: ignore[no-untyped-def]
    req = urllib.request.Request(notes_manager.url + path, data=raw, method=method,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=10)
    assert err.value.code == 400
    assert "error" in json.loads(err.value.read())


@pytest.mark.parametrize("path", ["/api/notes/search?q=x&limit=abc", "/api/notes/history?limit=abc",
                                  "/api/notes/history?skip=abc"])
def test_bad_integer_is_a_400(notes_manager, path: str) -> None:  # type: ignore[no-untyped-def]
    status, body, _ = _req(notes_manager.url, "GET", path)
    assert status == 400 and "must be an integer" in body["error"]


def test_tree_reports_protected_dirs_including_globs(notes_manager) -> None:  # type: ignore[no-untyped-def]
    url = notes_manager.url
    (notes_manager.notes / "drafts" / "old").mkdir(parents=True)
    (notes_manager.notes / "drafts" / "old" / "x.md").write_text("x\n")
    (notes_manager.notes / ".mlsweep-notes.json").write_text(json.dumps({"protected": ["sources/", "drafts/*"]}))
    tree = _req(url, "GET", "/api/notes/tree")[1]
    assert "sources" in tree["protected_dirs"] and "drafts/old" in tree["protected_dirs"]
    assert "zo" not in tree["protected_dirs"]


def test_mentions_count_runs_and_roll_them_up(notes_manager) -> None:  # type: ignore[no-untyped-def]
    """A note citing a run credits its experiment; the run has its own count."""
    url, notes = notes_manager.url, notes_manager.notes
    for eid in ("exp_runs", "exp_other"):
        assert _req(url, "POST", "/api/experiments", {"experiment_id": eid, "name": eid})[0] == 201
    # run1 is only exp_runs'; seed0 exists in both, so it needs the note to name
    # the experiment as well.
    for eid, run in (("exp_runs", "run1"), ("exp_runs", "seed0"), ("exp_other", "seed0")):
        assert _req(url, "POST", "/api/jobs",
                    {"experiment_id": eid, "run_id": run, "command": ["echo"]})[0] == 201
    (notes / "runs.md").write_text(
        "exp_runs did well on run1\n"
        "the seed0 run belongs to it too\n")
    for _ in range(40):
        _, body, _ = _req(url, "GET", "/api/notes/mentions")
        if body["mentions"].get("exp_runs"):
            break
        time.sleep(0.1)
    assert body["mentions"]["exp_runs"] == ["runs.md"]
    assert body["runs"]["exp_runs"] == {"run1": ["runs.md"], "seed0": ["runs.md"]}
    # The note names exp_runs, not exp_other, so the shared run ID credits only it.
    assert "exp_other" not in body["mentions"]
    assert "exp_other" not in body["runs"]


def _seed_metrics(db_path: Path, experiment: str, rows: dict[str, list[tuple[int, dict]]]) -> None:
    """Write metric rows straight into the manager's DB (WAL allows a second writer)."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        for run_id, points in rows.items():
            found = con.execute(
                "SELECT job_key, attempt FROM jobs WHERE experiment_id = ? AND run_id = ?",
                (experiment, run_id)).fetchone()
            assert found, (experiment, run_id)
            job_key, attempt = found
            for step, data in points:
                con.execute(
                    "INSERT OR REPLACE INTO metrics (job_key, attempt, step, data) VALUES (?, ?, ?, ?)",
                    (job_key, attempt, step, json.dumps(data)))
        con.commit()
    finally:
        con.close()


def test_chart_and_table_blocks(notes_manager, tmp_path) -> None:
    url = notes_manager.url
    exp = "e_notes_blocks"
    _req(url, "POST", "/api/experiments", {"experiment_id": exp})
    for run, lr in (("r0", 0.001), ("r1", 0.001), ("r2", 0.01)):
        status, _, _ = _req(url, "POST", "/api/jobs", {
            "run_id": run, "experiment_id": exp, "command": ["echo"], "combo": {"lr": lr}})
        assert status == 201
    _seed_metrics(tmp_path / "manager.db", exp, {
        "r0": [(1, {"loss": 1.0}), (2, {"loss": 0.8})],
        "r1": [(1, {"loss": 1.2}), (2, {"loss": 1.0})],
        "r2": [(1, {"loss": 2.0}), (2, {"loss": 1.5})],
    })

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "chart",
        "spec": f"experiments: [{exp}]\nmetric: loss\nsplit_by: lr\nband: minmax",
    })
    assert status == 200, data
    groups = {s["name"]: s for s in data["series"]}
    assert groups["lr=0.001"]["y"] == [1.1, 0.9]
    assert groups["lr=0.001"]["lo"] == [1.0, 0.8]
    assert groups["lr=0.001"]["hi"] == [1.2, 1.0]

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "table",
        "spec": f"experiments: {exp}\nmetrics: [loss]\nsort: loss\norder: desc",
    })
    assert status == 200, data
    assert [c["key"] for c in data["columns"]] == ["run", "lr", "loss"]
    assert [r["run"] for r in data["rows"]] == ["r2", "r1", "r0"]
    assert data["rows"][-1]["metrics"]["loss"]["final"] == 0.8

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "chart", "spec": f"experiments: [{exp}]\nmetric: loss\nruns: [{exp}/r0]"})
    assert status == 200 and [s["name"] for s in data["series"]] == ["r0"]

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "chart", "spec": "runs: [" + exp + "/r0]\nmetric: loss"})
    assert status == 200 and [s["name"] for s in data["series"]] == ["r0"]

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "chart", "spec": "runs: [r0]\nmetric: loss"})
    assert status == 200 and [(s["experiment"], s["run"]) for s in data["series"]] == [(exp, "r0")]

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "chart", "spec": f"experiments: [{exp}]\nmetric: loss\nwhere: lr=0.01"})
    assert status == 200 and [s["name"] for s in data["series"]] == ["r2"]

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "chart", "spec": f"experiments: [no_such_exp]\nmetric: loss"})
    assert status == 200 and data["series"] == []
    assert any("unknown experiment" in w for w in data["warnings"])

    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "chart", "spec": "metric: loss"})
    assert status == 400 and "experiments" in data["error"]
    status, data, _ = _req(url, "POST", "/api/notes/block", {
        "kind": "nope", "spec": "experiments: x"})
    assert status == 400


def test_evidence_links(notes_manager, tmp_path) -> None:
    url = notes_manager.url
    exp = "e_evidence"
    _req(url, "POST", "/api/experiments", {"experiment_id": exp})
    _req(url, "POST", "/api/jobs", {"run_id": "r0", "experiment_id": exp, "command": ["echo"]})
    _seed_metrics(tmp_path / "manager.db", exp, {
        "r0": [(100, {"loss": 2.0}), (200, {"loss": 1.25})],
    })

    status, data, _ = _req(url, "POST", "/api/notes/evidence", {"refs": [
        f"{exp}/r0/loss@200", f"{exp}/r0/loss", f"{exp}/nope/loss@1", "broken",
    ]})
    assert status == 200, data
    res = data["results"]
    assert res[f"{exp}/r0/loss@200"]["value"] == 1.25
    assert res[f"{exp}/r0/loss"]["value"] == 1.25
    assert res[f"{exp}/nope/loss@1"]["ok"] is False
    assert res["broken"]["ok"] is False

    status, data, _ = _req(url, "POST", "/api/notes/evidence", {"refs": "not a list"})
    assert status == 400


def test_data_endpoints_need_a_token(notes_manager) -> None:
    for path in ("/api/notes/block", "/api/notes/evidence", "/api/notes/run-ids"):
        if path.endswith("run-ids"):
            assert _req(notes_manager.url, "GET", path, token=None)[0] == 401
        else:
            assert _req(notes_manager.url, "POST", path, {"kind": "chart", "spec": "x"},
                        token=None)[0] == 401
    status, data, _ = _req(notes_manager.url, "GET", "/api/notes/run-ids")
    assert status == 200 and data["runs"] == {}


def test_run_ids_endpoint(notes_manager) -> None:
    url = notes_manager.url
    _req(url, "POST", "/api/experiments", {"experiment_id": "e_run_ids"})
    _req(url, "POST", "/api/jobs", {"run_id": "r0", "experiment_id": "e_run_ids",
                                    "command": ["echo"]})
    status, data, _ = _req(url, "GET", "/api/notes/run-ids")
    assert status == 200 and data["runs"]["r0"] == ["e_run_ids"]


def test_lint_flags_stale_evidence_numbers(notes_manager, tmp_path) -> None:
    url = notes_manager.url
    exp = "e_lint_stale"
    _req(url, "POST", "/api/experiments", {"experiment_id": exp})
    _req(url, "POST", "/api/jobs", {"run_id": "r0", "experiment_id": exp, "command": ["echo"]})
    _seed_metrics(tmp_path / "manager.db", exp, {"r0": [(2, {"loss": 2.0})]})
    note = (f"Matches [2.0](mlsweep:{exp}/r0/loss@2) and "
            f"is stale [3.0](mlsweep:{exp}/r0/loss@2).\n")
    status, _, _ = _req(url, "PUT", "/api/notes/file",
                        {"path": "stale.md", "content": note, "base_hash": None})
    assert status == 200
    status, data, _ = _req(url, "GET", "/api/notes/lint")
    assert status == 200
    stale = [f for f in data["findings"] if f["kind"] == "stale-evidence"]
    assert len(stale) == 1
    assert stale[0]["path"] == "stale.md" and stale[0]["line"] == 1
    assert data["stats"]["evidence_checked"] is True
    assert data["kinds"]["stale-evidence"]
