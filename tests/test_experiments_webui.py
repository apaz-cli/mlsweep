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


def _experiment(mgr, run_ids, status, name=None, capture_session=None, note=None):
    eid = uid("layout")
    body = {"campaign": "test", "experiment_id": eid, "name": name or eid}
    if capture_session:
        body["capture_session"] = capture_session  # the agent session that submitted it
    if note:
        body["note"] = note
    ok(mgr, "POST", "/api/experiments", body)
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


def test_note_is_a_hover_description(mgr):
    """The note is no longer text in the row; hovering the name shows it large."""
    note = ("A description long enough to need several lines in the tooltip, so it stays "
            "readable instead of squeezing the experiment name and the count columns.")
    eid = _experiment(mgr, ["run_a"], "completed", name="noted_experiment", note=note)
    b = Browser(CHROME)
    try:
        b.call("Emulation.setDeviceMetricsOverride", width=1300, height=900,
               deviceScaleFactor=1, mobile=False)
        b.goto(static_url(mgr, "experiments.html"))
        b.wait_for(f"!!document.getElementById('exp-row-{eid}')")

        # The row carries the name but not the description text.
        row_text = b.eval(f"document.getElementById('exp-row-{eid}').textContent")
        assert "noted_experiment" in row_text and note not in row_text
        assert b.eval(f"!document.getElementById('exp-row-{eid}').querySelector('.exp-note')")
        assert b.eval(f"document.querySelector('#exp-row-{eid} .exp-name').dataset.note") == note

        # Hovering the name opens a large, wrapping tooltip with the whole note.
        b.eval(f"""(() => {{ const el = document.querySelector('#exp-row-{eid} .exp-name');
            el.dispatchEvent(new PointerEvent('pointerover', {{bubbles: true}})); }})()""")
        b.wait_for("document.querySelector('.exp-note-tip').hidden === false")
        tip = b.eval("""(() => { const t = document.querySelector('.exp-note-tip');
            const s = getComputedStyle(t), r = t.getBoundingClientRect();
            return [t.textContent, s.fontSize, s.whiteSpace, r.left, r.right, innerWidth]; })()""")
        assert tip[0] == note
        assert float(tip[1].removesuffix("px")) >= 14, tip
        assert tip[2] == "normal", tip
        assert tip[3] >= 0 and tip[4] <= tip[5], tip

        # Leaving the name closes it.
        b.eval(f"""(() => {{ const el = document.querySelector('#exp-row-{eid} .exp-name');
            el.dispatchEvent(new PointerEvent('pointerout', {{bubbles: true, relatedTarget: document.body}})); }})()""")
        b.wait_for("document.querySelector('.exp-note-tip').hidden === true")
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
    # One agent session submitted eid.
    trace = uid("trace")
    ok(mgr, "POST", "/api/capture/sessions", {"session": {
        "session_id": trace, "agent": "claude_code", "title": "t", "state": "active"},
        "events": [{"seq": 1, "ts": 1.0, "kind": "user", "text": "go"}]})
    eid = _experiment(mgr, ["run_a", "run_b"], "completed", name="cited_one", capture_session=trace)
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
        b.wait_for("MLViews.state.traceCounts !== null")

        # The Refs column holds the notes badge over the traces badge: the icon
        # and count of the sessions that submitted the experiment, linking to them.
        b.wait_for(f"!!document.querySelector('#exp-row-{eid} .refs-cell .trace-ref')")
        chip = b.eval(f"""(() => {{ const a = document.querySelector('#exp-row-{eid} .refs-cell a.trace-ref');
            return [a.textContent, a.getAttribute('href'), !!a.querySelector('svg')]; }})()""")
        assert chip[0] == "1" and chip[2], chip
        assert f"traces.html?experiment={eid}" in chip[1], chip
        # Both badges always show; a zero is red and links nowhere.
        zero = b.eval(f"""(() => {{ const t = document.querySelector('#exp-row-{other} .refs-cell .trace-ref');
            return [t.tagName, t.textContent, t.classList.contains('zero')]; }})()""")
        assert zero == ["SPAN", "0", True], zero
        headers = b.eval("[...document.querySelectorAll('#exp-table th')].map(th => th.textContent)")
        assert "Refs" in headers and "Notes" not in headers and "Traces" not in headers, headers
        b.eval(f"document.querySelector('#exp-row-{eid} .btn-menu').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Results", "2"], ["Logs", "2"], ["Artifacts", "0"],
                                  ["Code", None], ["Notes", "1"], ["Traces", "1"], ["Export", "2"]]
        hrefs = b.eval("[...document.querySelectorAll('.row-menu a')].map(a => a.getAttribute('href'))")
        assert any("results.html?experiment=" + eid in h for h in hrefs), hrefs
        assert any("logs.html?experiment=" + eid in h for h in hrefs), hrefs
        assert any("files.html?experiment=" + eid in h for h in hrefs), hrefs
        assert any("notes.html?mentions=" + eid in h for h in hrefs), hrefs
        assert any("export.html?experiment=" + eid in h for h in hrefs), hrefs

        # Manage holds the verbs, Delete behind a divider.
        b.eval(f"document.querySelector('#exp-row-{eid} .btn-manage').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Rename\u2026", None], ["Edit description\u2026", None],
                                  ["Move to campaign\u2026", None], ["Delete\u2026", None]]
        assert b.eval("document.querySelectorAll('.row-menu hr').length") == 1

        # An experiment nothing cites still offers Notes (0): visible, inert.
        b.eval(f"document.querySelector('#exp-row-{other} .btn-menu').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Results", "1"], ["Logs", "1"], ["Artifacts", "0"],
                                  ["Code", None], ["Notes", "0"], ["Traces", "0"], ["Export", "1"]]
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
                                  ["Code", None], ["Notes", "1"], ["Traces", None]]
        run_hrefs = b.eval("[...document.querySelectorAll('.row-menu a')].map(a => a.getAttribute('href'))")
        # Results goes to the run's experiment; the other pages carry the run.
        results_hrefs = [h for h in run_hrefs if h.startswith("results.html")]
        assert len(results_hrefs) == 1 and "run=" not in results_hrefs[0], run_hrefs
        assert f"experiment={eid}" in results_hrefs[0], run_hrefs
        assert all("run=" in h for h in run_hrefs
                   if not h.startswith(("results.html", "notes.html", "traces.html"))), run_hrefs
        assert any("notes.html?mentions=run_a" in h for h in run_hrefs), run_hrefs
        b.eval(f"document.querySelector('#exp-tbody tr[data-runs-of={eid}] .btn-manage').click()")
        b.wait_for("!!document.querySelector('.row-menu')")
        assert _menu_items(b) == [["Rename\u2026", None], ["Move to experiment\u2026", None],
                                  ["Delete\u2026", None]]
    finally:
        b.close()


def test_note_trace_links_are_checked(notes_mgr):
    """A note's mlsweep:trace/ links open the trace; a short id works, a missing one shows ⚠."""
    mgr, notes = notes_mgr
    trace = "7e57ab1e-" + uid("t").replace("_", "-")[-12:]
    ok(mgr, "POST", "/api/capture/sessions", {"session": {
        "session_id": trace, "agent": "pi", "title": "Warmup idea", "state": "active"},
        "events": [{"seq": 1, "ts": 1.0, "kind": "user", "text": "try warmup"}]})
    (notes / "traces.md").write_text(
        f"[full](mlsweep:trace/{trace}) [short](mlsweep:trace/{trace[:8]}) "
        "[gone](mlsweep:trace/00000000-dead-beef-0000-000000000000) "
        f"[event](mlsweep:trace/{trace[:8]}?kind=user#E1) [no event](mlsweep:trace/{trace}#E9)\n")
    b = Browser(CHROME)
    try:
        b.goto(static_url(mgr, "notes.html", path="traces.md"))
        b.wait_for("[...document.querySelectorAll('a.trace-ev')].length === 5 && "
                   "[...document.querySelectorAll('a.trace-ev')].every(a => "
                   "a.classList.contains('trace-ok') || a.classList.contains('trace-bad'))")
        links = b.eval("""[...document.querySelectorAll('a.trace-ev')].map(a =>
            [a.textContent, a.classList.contains('trace-ok'), a.getAttribute('href'), a.title])""")
        full, short, gone, event, no_event = links
        assert full[1] and f"session={trace}" in full[2] and "Warmup idea" in full[3], full
        # The short id opens the full one.
        assert short[1] and short[2] == full[2], short
        assert not gone[1] and gone[0].endswith("⚠") and "no such trace" in gone[3], gone
        # A link to one event opens the trace there, filtered as it says, and
        # names the event; an event the trace lacks shows ⚠.
        assert event[1] and f"session={trace}" in event[2], event
        assert "kind=user" in event[2] and "event=1" in event[2], event
        assert "user event 1" in event[3] and "try warmup" in event[3], event
        assert not no_event[1] and "no event 9" in no_event[3], no_event
        # Following it lands on that event with the inspector open.
        b.goto(static_url(mgr, "traces.html", session=trace, kind="user", event="1"))
        b.wait_for("document.getElementById('inspector').classList.contains('open')")
        assert b.eval("document.querySelector('#ledger tbody tr.selected').dataset.seq") == "1"
        assert b.eval("document.querySelector('#kind-chips button.active').dataset.kind") == "user"
        link = b.eval("eventNoteLink(eventsBySeq.get(1))")
        assert link == f"[“try warmup”](mlsweep:trace/{trace}#E1)", link
    finally:
        b.close()
