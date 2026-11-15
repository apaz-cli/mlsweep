"""Layout of the experiments page, driven through headless Chromium.

Runs are rows of the experiments table, so they share its columns.  Action
cells must stay real table cells.  As flex boxes they stopped stretching to
their row, so any row taller than its buttons drew a stepped bottom border
under Actions.  A long run ID wraps rather than widening the Name column.

Skipped when no Chromium-family browser is installed (set MLSWEEP_TEST_CHROME
to point at one).
"""

import json

import pytest

from browser import Browser, find_chromium
from campaign_helpers import ok, shared_manager, static_url, uid
from conftest import _start_manager, _teardown_manager
from mlsweep import _notes_git

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "experiments_webui")


def _experiment(mgr, run_ids, status, name=None):
    eid = uid("layout")
    ok(mgr, "POST", "/api/experiments", {"experiment_id": eid, "name": name or eid})
    for rid in run_ids:
        ok(mgr, "POST", "/api/jobs", {"experiment_id": eid, "run_id": rid, "command": ["echo"]})
        ok(mgr, "PUT", f"/api/jobs/{rid}/status", {"experiment_id": eid, "status": "done", "exit_code": 0})
    ok(mgr, "PUT", f"/api/experiments/{eid}/status", {"status": status})
    return eid


@pytest.mark.parametrize("width", [1300, 900])
def test_runs_share_the_experiment_columns(mgr, width):
    eids = [
        # A name that wraps makes its row taller than the buttons in Actions.
        _experiment(mgr, ["short"], "completed",
                    name="tokenizer ablation with a name long enough to wrap onto a second line " * 2),
        _experiment(mgr, ["a_much_longer_run_id_lr0.0003_bs512_seed1", "r2"], "running"),
    ]
    b = Browser(CHROME)
    try:
        b.call("Emulation.setDeviceMetricsOverride", width=width, height=900,
               deviceScaleFactor=1, mobile=False)
        b.goto(static_url(mgr, "experiments.html"))
        width_js = "Math.round(document.getElementById('exp-table').getBoundingClientRect().width)"
        b.wait_for(f"!!document.querySelector('.exp-row[data-exp-id={json.dumps(eids[-1])}]')")
        for eid in eids:
            b.eval(f"toggleExpand({json.dumps(eid)})")
            b.wait_for(f"runRowsOf({json.dumps(eid)}).some(r => r.classList.contains('run-row'))")

        # Every cell of a row ends on the same line, so its bottom border is unbroken.
        stepped = b.eval("""[...document.querySelectorAll('.exp-row, .run-row')]
            .filter(r => new Set([...r.cells].map(c => Math.round(c.getBoundingClientRect().bottom))).size > 1)
            .map(r => r.textContent.trim().slice(0, 40))""")
        assert stepped == []
        assert b.eval("[...document.querySelectorAll('td.actions-cell')]"
                      ".every(c => getComputedStyle(c).display === 'table-cell')")

        # A run's Status and Actions sit under its experiment's, in one table.
        assert b.eval("document.querySelectorAll('#exp-table table').length") == 0
        lefts = b.eval("""[...document.querySelectorAll('.exp-row, .run-row')].map(r => {
            const xs = [...r.cells].map(c => Math.round(c.getBoundingClientRect().left));
            return [xs[1], xs[xs.length - 1]];
        })""")
        assert b.eval("document.querySelectorAll('.run-row').length") == 3
        assert len({tuple(x) for x in lefts}) == 1, lefts

        # Run rows never widen the table, and no run ID spills out of its cell.
        assert b.eval("[...document.querySelectorAll('.run-row td:first-child')]"
                      ".every(c => c.scrollWidth <= c.clientWidth)")
        with_runs = b.eval(width_js)
        b.eval("document.querySelectorAll('#exp-tbody tr[data-runs-of]').forEach(r => r.remove())")
        assert b.eval(width_js) == with_runs
    finally:
        b.close()


@pytest.fixture(scope="module")
def notes_mgr(tmp_path_factory):
    """A manager whose notes cite one experiment and one of its runs."""
    root = tmp_path_factory.mktemp("experiments_notes")
    notes = root / "notes"
    _notes_git.init_repo(notes, name="dash", email="d@x")
    proc, server, _ = _start_manager(root, extra_args=("--notes", str(notes)))
    yield server, notes
    _teardown_manager(proc)


def _menu_items(b):
    """[label, count] for the open menu; count is None where the entry has none."""
    return b.eval("""[...document.querySelectorAll('.row-menu-item')].map(x => {
        const count = x.querySelector('.mi-count');
        return [x.querySelector('.mi-label').textContent, count ? count.textContent : null];
    })""")


def test_menus_and_citation_counts(notes_mgr):
    mgr, notes = notes_mgr
    # Run IDs are words the notes index can recognise: at least three characters.
    eid = _experiment(mgr, ["run_a", "run_b"], "completed", name="cited_one")
    other = _experiment(mgr, ["run_a"], "completed", name="uncited_one")
    # run_b is unique to eid; run_a is shared with `other`, so the note naming
    # eid is what attributes it.
    (notes / "a.md").write_text(f"{eid} improved run_b\nrun_a too\n")
    b = Browser(CHROME)
    try:
        b.call("Emulation.setDeviceMetricsOverride", width=1300, height=900,
               deviceScaleFactor=1, mobile=False)
        b.goto(static_url(mgr, "experiments.html", campaign="*"))
        b.wait_for(f"!!document.getElementById('exp-row-{eid}')")

        # Counts live in the menus, not in the row.
        assert b.eval("document.querySelectorAll('.doc-count').length") == 0

        # View holds the pages with what each leads to, the experiment (and
        # campaign) carried along: two finished runs, two logs, no artifacts.
        b.eval(f"document.querySelector('#exp-row-{eid} .btn-menu').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Results", "2"], ["Logs", "2"], ["Artifacts", "0"],
                                  ["Notes", "1"], ["Export", "2"]]
        hrefs = b.eval("[...document.querySelectorAll('.row-menu a')].map(a => a.getAttribute('href'))")
        assert any("results.html?experiment=" + eid in h for h in hrefs), hrefs
        assert any("logs.html?experiment=" + eid in h for h in hrefs), hrefs
        assert any("files.html?experiment=" + eid in h for h in hrefs), hrefs
        assert any("notes.html?mentions=" + eid in h for h in hrefs), hrefs
        assert any("export.html?experiment=" + eid in h for h in hrefs), hrefs

        # Manage holds the verbs, Delete behind a divider.
        b.eval(f"document.querySelector('#exp-row-{eid} .btn-manage').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Rename\u2026", None], ["Move to campaign\u2026", None],
                                  ["Delete\u2026", None]]
        assert b.eval("document.querySelectorAll('.row-menu hr').length") == 1

        # An experiment nothing cites still offers Notes (0): visible, inert.
        b.eval(f"document.querySelector('#exp-row-{other} .btn-menu').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Results", "1"], ["Logs", "1"], ["Artifacts", "0"],
                                  ["Notes", "0"], ["Export", "1"]]
        assert b.eval("""(() => { const it = [...document.querySelectorAll('.row-menu-item')]
            .find(x => x.querySelector('.mi-label').textContent === 'Notes');
            return it.tagName === 'SPAN' && it.classList.contains('disabled')
                && it.title.includes('No note mentions'); })()""")

        # Runs get the same two menus, and a count for their own citations.
        b.eval(f"toggleExpand({json.dumps(eid)})")
        b.wait_for(f"(jobsCache[{json.dumps(eid)}] || []).length === 2")
        b.wait_for(f"runRowsOf({json.dumps(eid)}).some(r => r.classList.contains('run-row'))")
        b.eval(f"document.querySelector('#exp-tbody tr[data-runs-of={eid}] .btn-menu').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Results", None], ["Artifacts", None], ["Logs", None],
                                  ["Notes", "1"]]
        run_hrefs = b.eval("[...document.querySelectorAll('.row-menu a')].map(a => a.getAttribute('href'))")
        # Results goes to the run's experiment; the other pages carry the run.
        results_hrefs = [h for h in run_hrefs if h.startswith("results.html")]
        assert len(results_hrefs) == 1 and "run=" not in results_hrefs[0], run_hrefs
        assert f"experiment={eid}" in results_hrefs[0], run_hrefs
        assert all("run=" in h for h in run_hrefs
                   if not h.startswith(("results.html", "notes.html"))), run_hrefs
        assert any("notes.html?mentions=run_a" in h for h in run_hrefs), run_hrefs
        b.eval(f"document.querySelector('#exp-tbody tr[data-runs-of={eid}] .btn-manage').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Rename\u2026", None], ["Move to experiment\u2026", None],
                                  ["Delete\u2026", None]]
    finally:
        b.close()
