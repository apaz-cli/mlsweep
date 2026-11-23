"""Right-click menus across the dashboard, driven through headless Chromium.

Every list a person works through has a right-click menu: experiment and run
rows, workers, GPU tiles and run chips, the Logs and Artifacts sidebars, traces
and their events, and the notes panels and tables.  Each offers what the row's
own buttons and clicks do.  A right-click on a link inside a row keeps the
browser's menu (open in new tab, copy link).

Skipped when no Chromium-family browser is installed (set MLSWEEP_TEST_CHROME
to point at one).
"""

import json
import sqlite3

import pytest

from browser import Browser, find_chromium
from campaign_helpers import TOKEN, call, ok, shared_manager, static_url, uid
from mlsweep._snapshot import scan_tree
from mlsweep.run_sweep import upload_snapshot

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "context_menus")


@pytest.fixture
def page():
    b = Browser(CHROME)
    yield b
    b.close()


def _experiment(mgr, runs, *, status="done"):
    eid = uid("ctx")
    ok(mgr, "POST", "/api/experiments", {"campaign": "test", "experiment_id": eid, "name": eid})
    for rid in runs:
        ok(mgr, "POST", "/api/jobs", {"experiment_id": eid, "run_id": rid, "command": ["echo"]})
        ok(mgr, "PUT", f"/api/jobs/{rid}/status", {"experiment_id": eid, "status": status, "exit_code": 0})
    return eid


def _right_click(page, selector):
    """Dispatch a contextmenu on *selector*; return whether the page took it."""
    return page.eval(f"""(() => {{ const e = document.querySelector({json.dumps(selector)});
        const r = e.getBoundingClientRect();
        const ev = new MouseEvent('contextmenu', {{bubbles: true, cancelable: true,
            clientX: r.x + Math.min(8, r.width / 2), clientY: r.y + Math.min(8, r.height / 2)}});
        e.dispatchEvent(ev); return ev.defaultPrevented; }})()""")


def _row_menu(page):
    page.wait_for("!!document.querySelector('.row-menu')")
    return page.eval("[...document.querySelectorAll('.row-menu-item .mi-label')].map(x => x.textContent)")


def _ctx_menu(page):
    page.wait_for("!!document.querySelector('.ml-ctx')")
    return page.eval("[...document.querySelectorAll('.ml-ctx-item .l')].map(x => x.textContent)")


_notes_menu = _ctx_menu  # the notes page uses the same menu as every other page


def _escape(page):
    page.eval("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}))")


def test_experiment_and_run_rows(mgr, page):
    eid = _experiment(mgr, ["run_a"])
    page.goto(static_url(mgr, "experiments.html", campaign="*"))
    page.wait_for(f"!!document.getElementById('exp-row-{eid}')")

    # The experiment row: its View menu, then Copy ID and its Manage menu.
    assert _right_click(page, f"#exp-row-{eid} .count-cell")
    labels = _row_menu(page)
    assert labels[:4] == ["Results", "Logs", "Artifacts", "Code"], labels
    assert labels[-4:] == ["Copy experiment ID", "Rename…", "Move to campaign…", "Delete…"], labels
    # It opens at the pointer, not under the row's buttons.
    left = page.eval("parseFloat(document.querySelector('.row-menu').style.left)")
    cell = page.eval(f"document.querySelector('#exp-row-{eid} .count-cell').getBoundingClientRect().x")
    assert abs(left - (cell + 8)) < 2, (left, cell)
    # The button menus are unchanged.
    _escape(page)
    page.eval(f"document.querySelector('#exp-row-{eid} .btn-manage').click()")
    page.wait_for("!!document.querySelector('.row-menu')")
    assert _row_menu(page) == ["Rename…", "Move to campaign…", "Delete…"]
    _escape(page)

    # A run row, and the details row under it, give the run's menu.
    page.eval(f"toggleExpand({json.dumps(eid)})")
    page.wait_for(f"!!document.querySelector('.run-row[data-exp-id=\"{eid}\"][data-run-id=\"run_a\"]')")
    assert _right_click(page, f'.run-row[data-run-id="run_a"][data-exp-id="{eid}"] .combo-cell')
    labels = _row_menu(page)
    assert labels[:4] == ["Results", "Artifacts", "Logs", "Code"], labels
    assert "Copy run ID" in labels and labels[-1] == "Delete…", labels
    hrefs = page.eval("[...document.querySelectorAll('.row-menu a')].map(a => a.getAttribute('href'))")
    assert any(f"logs.html?experiment={eid}" in h and "run=run_a" in h for h in hrefs), hrefs
    # A Manage verb runs from the right-click menu like from the button.
    page.eval("[...document.querySelectorAll('.row-menu-item')].find(b => b.textContent.includes('Rename')).click()")
    page.wait_for("!!document.querySelector('.ml-dialog input')")
    page.eval("document.querySelector('.ml-dialog input').value = 'renamed'; "
              "document.querySelector('.ml-dialog input').dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}))")
    page.wait_for(f"(jobsCache[{json.dumps(eid)}] || []).some(j => j.label === 'renamed')")

    # A right-click on a link in a row keeps the browser's menu.
    link = page.eval(f"!!document.querySelector('#exp-row-{eid} a[href]')")
    if link:
        assert not _right_click(page, f"#exp-row-{eid} a[href]")


def test_code_entry_counts_the_snapshot_files(mgr, page, tmp_path):
    """View's Code entry says how many files the code has: the experiment's
    latest code for the experiment, each run's own for a run."""
    eid = uid("ctx_code")
    for i, n in enumerate((2, 3)):
        (tmp_path / f"f{n}.py").write_text("x = 1\n")  # 2 files, then 3
        if i == 0:
            (tmp_path / "train.py").write_text("print(1)\n")
        sid = upload_snapshot(mgr.url, TOKEN, scan_tree(tmp_path))[0]
        status, _ = call(mgr, "POST", f"/api/experiments/{eid}/jobs/bulk", {
            "campaign": "test", "code": {"snapshot": sid},
            "jobs": [{"run_id": f"run{n}", "command": ["true"], "artifact_id": sid}]})
        assert status == 201
    exp = next(e for e in ok(mgr, "GET", "/api/experiments") if e["experiment_id"] == eid)
    assert exp["code_files"] == 3
    jobs = {j["run_id"]: j["code_files"] for j in ok(mgr, "GET", f"/api/experiments/{eid}/jobs")}
    assert jobs == {"run2": 2, "run3": 3}

    def code_count():
        page.wait_for("!!document.querySelector('.row-menu')")
        return page.eval("""[...document.querySelectorAll('.row-menu-item')]
            .find(x => x.querySelector('.mi-label').textContent === 'Code').querySelector('.mi-count')?.textContent""")

    page.goto(static_url(mgr, "experiments.html", campaign="*"))
    page.wait_for(f"!!document.getElementById('exp-row-{eid}')")
    page.eval(f"document.querySelector('#exp-row-{eid} .btn-menu').click()")
    assert code_count() == "3"
    _escape(page)
    page.eval(f"toggleExpand({json.dumps(eid)})")
    page.wait_for(f"!!document.querySelector('.run-row[data-exp-id=\"{eid}\"][data-run-id=\"run2\"]')")
    page.eval(f"document.querySelector('.run-row[data-exp-id=\"{eid}\"][data-run-id=\"run2\"] .btn-menu').click()")
    assert code_count() == "2"
    _escape(page)
    _right_click(page, f'.run-row[data-exp-id="{eid}"][data-run-id="run3"] .combo-cell')
    assert code_count() == "3"


def test_system_page_rows(mgr, page):
    # No worker connects in this test, so nothing really runs: draw an
    # in-flight run and a worker the way /api/jobs and /api/workers report them.
    eid = _experiment(mgr, ["live_r"])
    page.goto(static_url(mgr, "system.html"))
    page.wait_for("typeof renderInFlight === 'function' && !!document.querySelector('#inflight-tbody tr')")
    job = {"experiment_id": eid, "run_id": "live_r", "status": "running", "worker_id": "gpu1:7000",
           "dispatched_gpu_ids": "[0]", "start_time": "2026-11-23T00:00:00Z"}
    page.eval(f"renderInFlight([{json.dumps(job)}])")
    assert _right_click(page, '#inflight-tbody tr[data-run-id="live_r"] td')
    labels = _row_menu(page)
    assert labels[:3] == ["Results", "Artifacts", "Logs"] and labels[-3:] == ["Copy run ID", "Rename…", "Cancel"], labels
    _escape(page)

    worker = {"worker_id": "gpu1:7000", "host": "gpu1", "status": "connected", "gpus": [0, 1],
              "gpu_runs": {"0": [{"experiment_id": eid, "run_id": "live_r"}]}, "max_jobs_per_gpu": 0}
    page.eval(f"renderInFlight([{json.dumps(job)}]); renderWorkers([{json.dumps(worker)}])")
    assert _right_click(page, '.run-chip[data-run="live_r"] .rc-meta')
    labels = _row_menu(page)
    assert labels[0] == "Move to gpu1 GPU 1" and "Requeue anywhere" in labels and labels[-1] == "Cancel run", labels
    assert page.eval("!!document.querySelector('.run-chip.menu-open')")
    _escape(page)
    page.wait_for("!document.querySelector('.run-chip.menu-open')")

    page.eval(f"renderInFlight([{json.dumps(job)}]); renderWorkers([{json.dumps(worker)}])")
    assert _right_click(page, '.gpu-tile[data-gpu="1"] .gt-bars')
    assert _row_menu(page) == ["Drain GPU 1", "Remove GPU 1 now", "Select"]
    _escape(page)

    page.eval(f"renderInFlight([{json.dumps(job)}]); renderWorkers([{json.dumps(worker)}])")
    assert _right_click(page, ".worker-card .worker-detail")
    assert _row_menu(page) == ["Drain all GPUs", "Disconnect", "Copy worker ID"]
    _escape(page)


def test_logs_run_list(mgr, page):
    eid = _experiment(mgr, ["r1", "r2"])
    page.goto(static_url(mgr, "logs.html", experiment=eid))
    page.wait_for("document.querySelectorAll('.job-item').length === 2")
    assert _right_click(page, '.job-item[data-run-id="r2"]')
    labels = _ctx_menu(page)
    assert labels == ["Show logs", "Download log", "Rename…", "Open results", "Open artifacts", "Copy run ID"], labels
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent === 'Show logs').click()")
    page.wait_for("document.querySelector('.job-item.selected')?.dataset.runId === 'r2'")


def test_artifacts_sidebar_files_and_cards(mgr, page):
    eid = _experiment(mgr, ["r1", "r2"])
    for rid in ("r1", "r2"):
        art = mgr.mlsweep_dir / "experiments" / eid / rid / "artifacts"
        art.mkdir(parents=True)
        (art / "out.txt").write_text(f"from {rid}")
    page.goto(static_url(mgr, "files.html", experiment=eid))
    page.wait_for("document.querySelectorAll('.run-item').length === 2")
    assert _right_click(page, '.run-item[data-run-id="r1"] .run-info')
    # Every run starts in the comparison; the menu takes one out and puts it back.
    labels = _ctx_menu(page)
    assert labels[:3] == ["Browse files", "Leave out of the comparison", "Download all files (.zip)"], labels
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent.startsWith('Leave out')).click()")
    page.wait_for("!checkedRuns.has('r1') && !document.querySelector('.run-item[data-run-id=\"r1\"] input').checked")
    _right_click(page, '.run-item[data-run-id="r1"] .run-info')
    assert "Add to the comparison" in _ctx_menu(page)
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent.startsWith('Add to')).click()")
    page.wait_for("checkedRuns.has('r1')")
    _right_click(page, '.run-item[data-run-id="r1"] .run-info')
    _ctx_menu(page)
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent === 'Browse files').click()")
    page.wait_for("!!document.querySelector('#file-tree .file-row[data-path=\"out.txt\"]')")

    assert _right_click(page, '#file-tree .file-row[data-path="out.txt"]')
    assert _ctx_menu(page) == ["Preview", "Download", "Open in a new tab", "Compare across runs", "Copy path"]
    # Compare across runs shows the file for each checked run.
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent === 'Compare across runs').click()")
    page.wait_for("document.querySelectorAll('#compare-grid .run-card').length === 2")
    assert page.eval("document.getElementById('file-path-sel').value") == "out.txt"
    assert _right_click(page, '#compare-grid .run-card[data-run-id="r2"] .card-header')
    labels = _ctx_menu(page)
    assert labels[:2] == ["Download this file", "Open this file in a new tab"], labels
    assert "Leave out of the comparison" in labels


def test_trace_list_and_events(mgr, page):
    keep, drop = uid("tr"), uid("tr")
    for sid in (keep, drop):
        ok(mgr, "POST", "/api/capture/sessions", {"session": {
            "session_id": sid, "agent": "claude_code", "title": sid, "state": "active"},
            "events": [{"seq": 1, "ts": 1.0, "kind": "user", "text": "first prompt"},
                       {"seq": 2, "ts": 2.0, "kind": "assistant", "text": "an answer"}]})
    page.goto(static_url(mgr, "traces.html", session=keep))
    page.wait_for(f"!!document.querySelector('.session-item[data-session=\"{drop}\"]')")
    page.eval(f"openSession({json.dumps(keep)})")
    page.wait_for("document.querySelectorAll('#ledger tr.event').length === 2")

    # An event: inspect it or copy a way back to it.
    assert _right_click(page, '#ledger tr.event[data-seq="2"] .content-col')
    assert _ctx_menu(page) == ["Inspect", "Copy text", "Copy note link", "Copy note reference"]
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent.startsWith('Inspect')).click()")
    page.wait_for("document.getElementById('inspector').classList.contains('open')")

    # A trace that is not open: the ⋯ menu's entries for it, and deleting it
    # leaves the open trace alone.
    assert _right_click(page, f'.session-item[data-session="{drop}"]')
    labels = _ctx_menu(page)
    assert labels == ["Open", "Copy note reference", "Copy trace id", "Download transcript", "Download as ATIF",
                      "Delete trace"], labels
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(b => b.textContent === 'Delete trace').click()")
    page.wait_for("!!document.querySelector('.ml-dialog')")
    page.eval("[...document.querySelectorAll('.ml-dialog button')].find(b => b.textContent === 'Confirm').click()")
    page.wait_for(f"!document.querySelector('.session-item[data-session=\"{drop}\"]')")
    assert page.eval("current && current.session_id") == keep
    assert page.eval("document.querySelectorAll('#ledger tr.event').length") == 2


def _insert_metrics(server, experiment, run, points):
    con = sqlite3.connect(server.mlsweep_dir.parent / "manager.db")
    try:
        job_key, attempt = con.execute(
            "SELECT job_key, attempt FROM jobs WHERE experiment_id = ? AND run_id = ?",
            (experiment, run)).fetchone()
        for step, data in points:
            con.execute("INSERT OR REPLACE INTO metrics (job_key, attempt, step, data) VALUES (?, ?, ?, ?)",
                        (job_key, attempt, step, json.dumps(data)))
        con.commit()
    finally:
        con.close()


def test_notes_panels_and_tables(mgr, page):
    eid = _experiment(mgr, ["r0"])
    _insert_metrics(mgr, eid, "r0", [(1, {"loss": 1.0}), (2, {"loss": 0.25})])
    word = uid("zebra")
    ok(mgr, "PUT", "/api/notes/file", {"path": "ctx/target.md", "content": f"# Target\n\n{word}\n", "base_hash": None})
    ok(mgr, "PUT", "/api/notes/file", {"path": "ctx/source.md", "base_hash": None, "content":
        f"# Source\n\nSee [[target]].\n\n```mlsweep-table\nexperiments: {eid}\nmetrics: [loss]\n```\n"})

    # A search hit.
    page.goto(static_url(mgr, "notes.html"))
    page.wait_for("typeof switchPane === 'function' && S.fileMap.has('ctx/target.md')")
    page.eval(f"searchFor({json.dumps(word)})")
    page.wait_for("!!document.querySelector('#search-results .sr, .sr')")
    assert _right_click(page, ".sr .sr-path")
    labels = _notes_menu(page)
    assert labels == ["Open in new tab", "Open to the side", "Reveal in file tree", "Copy path", "Copy wikilink"], labels

    # A backlink in the right panel.
    page.eval("mlCloseContextMenu(); openFile('ctx/target.md')")
    page.wait_for("[...document.querySelectorAll('.ritem')].some(r => r.title === 'ctx/source.md')")
    page.eval("""(() => { const t = [...document.querySelectorAll('.ritem')].find(r => r.title === 'ctx/source.md');
        t.id = 'backlink-under-test'; })()""")
    assert _right_click(page, "#backlink-under-test")
    assert _notes_menu(page)[0] == "Open in new tab"
    page.eval("[...document.querySelectorAll('.ml-ctx-item')].find(r => r.querySelector('.l').textContent === 'Open in new tab').click()")
    page.wait_for("S.tabs.filter(t => t.path === 'ctx/source.md').length === 1")

    # A run in a table block: its pages, and on a metric cell an evidence link.
    page.wait_for("!!document.querySelector('.nb-table td[data-metric=\"loss\"]')", timeout=20)
    assert _right_click(page, '.nb-table td[data-metric="loss"]')
    labels = _notes_menu(page)
    assert labels[:4] == ["Open results", "Open logs", "Open artifacts", "Open code"], labels
    assert labels[-2:] == ["Copy value", "Copy evidence link"], labels
    # The run's own link keeps the browser's menu.
    page.eval("mlCloseContextMenu()")
    assert not _right_click(page, ".nb-table a.nb-run")
