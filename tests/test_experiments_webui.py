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
