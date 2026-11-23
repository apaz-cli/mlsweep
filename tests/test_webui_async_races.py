"""Race regressions for dashboard pages that switch what they show.

Each test delays the response for the view being left, starts a newer view, and
checks that the older answer cannot land on top of it.  A user clicking two
experiments quickly (or a slow first request next to a fast second one) used to
see the first one's runs, metrics or files under the second one's name.

Skipped when no Chromium-family browser is installed (set MLSWEEP_TEST_CHROME
to point at one).
"""

import json

import pytest

from browser import Browser, find_chromium
from campaign_helpers import TOKEN, ok, shared_manager, static_url, uid
from mlsweep._snapshot import scan_tree
from mlsweep.run_sweep import upload_snapshot

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "webui_async_races")


def _experiment(mgr, prefix, run_id, artifact=None):
    eid = uid(prefix)
    ok(mgr, "POST", "/api/experiments", {
        "campaign": "test", "experiment_id": eid, "name": eid,
    })
    ok(mgr, "POST", "/api/jobs", {
        "experiment_id": eid, "run_id": run_id, "command": ["true"],
    })
    ok(mgr, "PUT", f"/api/jobs/{run_id}/status", {
        "experiment_id": eid, "status": "done", "exit_code": 0,
    })
    if artifact:
        path = mgr.mlsweep_dir / "experiments" / eid / run_id / "artifacts"
        path.mkdir(parents=True)
        (path / artifact).write_text(f"from {run_id}\n")
    return eid


def _delay_jobs(page, experiment_id, ms=800):
    """Make the jobs list for *experiment_id* answer late, as a slow network would."""
    path = f"/api/experiments/{experiment_id}/jobs"
    page.eval(f"""(() => {{
        const original = apiFetch;
        window.__raceOriginalFetch = original;
        apiFetch = (url, opts) => url === {json.dumps(path)}
            ? new Promise(resolve => setTimeout(() => resolve(original(url, opts)), {ms}))
            : original(url, opts);
    }})()""")


def test_results_drops_a_late_answer_for_the_previous_experiment(mgr):
    old = _experiment(mgr, "slow_results", "old_run")
    new = _experiment(mgr, "fast_results", "new_run")
    page = Browser(CHROME)
    try:
        page.goto(static_url(mgr, "results.html", experiment=old))
        page.wait_for(f"LOADED_EXP === {json.dumps(old)} && DATA && DATA.experiment === {json.dumps(old)}")
        _delay_jobs(page, old)
        page.eval(f"window.__raceStart = Date.now(); loadExperiment({json.dumps(old)}); "
                  f"loadExperiment({json.dumps(new)});")
        page.wait_for(f"Date.now() - window.__raceStart > 1500 && LOADED_EXP === {json.dumps(new)}")
        assert page.eval("DATA && DATA.experiment") == new
        assert page.eval("document.getElementById('exp-name').textContent") == new
    finally:
        page.close()


def test_artifacts_drops_a_late_answer_for_the_previous_experiment(mgr):
    old = _experiment(mgr, "slow_files", "old_file_run", artifact="old.txt")
    new = _experiment(mgr, "fast_files", "new_file_run", artifact="new.txt")
    page = Browser(CHROME)
    try:
        page.goto(static_url(mgr, "files.html", experiment=old))
        page.wait_for(f"loadedExp === {json.dumps(old)} && jobs.some(j => j.run_id === 'old_file_run')")
        _delay_jobs(page, old)
        page.eval(f"window.__raceStart = Date.now(); loadExperiment({json.dumps(old)}); "
                  f"loadExperiment({json.dumps(new)});")
        page.wait_for(f"Date.now() - window.__raceStart > 1500 && loadedExp === {json.dumps(new)}")
        assert page.eval("jobs.map(j => j.run_id)") == ["new_file_run"]
        assert page.eval("Object.keys(fileCache)") == ["new_file_run"]
    finally:
        page.close()


def _coded_experiment(mgr, tmp_path, prefix, run_id):
    """An experiment with a code snapshot and one run that used it."""
    root = tmp_path / prefix
    root.mkdir()
    (root / "train.py").write_text(f"print({prefix!r})\n")
    sid = upload_snapshot(mgr.url, TOKEN, scan_tree(root))[0]
    eid = uid(prefix)
    ok(mgr, "POST", f"/api/experiments/{eid}/jobs/bulk", {
        "campaign": "test", "code": {"snapshot": sid},
        "jobs": [{"run_id": run_id, "command": ["true"], "artifact_id": sid}],
    })
    return eid, sid


def test_code_drops_a_late_answer_for_the_previous_experiment(mgr, tmp_path):
    old, old_sid = _coded_experiment(mgr, tmp_path, "slow_code", "old_code_run")
    new, new_sid = _coded_experiment(mgr, tmp_path, "fast_code", "new_code_run")
    page = Browser(CHROME)
    try:
        page.goto(static_url(mgr, "code.html", experiment=new))
        page.wait_for(f"S.expId === {json.dumps(new)} && S.snapshot === {json.dumps(new_sid)}")
        old_jobs = f"/api/experiments/{old}/jobs"
        page.eval(f"""(() => {{
            const original = apiFetch;
            apiFetch = (url, opts) => url === {json.dumps(old_jobs)}
                ? new Promise(resolve => setTimeout(() => resolve(original(url, opts)), 800))
                : original(url, opts);
        }})()""")
        page.eval(f"window.__raceStart = Date.now(); selectExperiment({json.dumps(old)}); "
                  f"selectExperiment({json.dumps(new)});")
        page.wait_for(f"Date.now() - window.__raceStart > 1500 && S.expId === {json.dumps(new)}")
        assert page.eval("S.expId") == new
        assert page.eval("S.snapshot") == new_sid
        assert page.eval("S.jobs.map(j => j.run_id)") == ["new_code_run"]
        assert page.eval("document.getElementById('hdr-title').textContent").find(new) >= 0
    finally:
        page.close()



def test_logs_drops_a_late_answer_for_the_previous_experiment(mgr):
    old = _experiment(mgr, "slow_logs", "old_log_run")
    new = _experiment(mgr, "fast_logs", "new_log_run")
    page = Browser(CHROME)
    try:
        page.goto(static_url(mgr, "logs.html", experiment=old))
        page.wait_for("document.querySelector('#job-list').textContent.includes('old_log_run')")
        page.wait_for(f"[...document.getElementById('exp-select').options]"
                      f".filter(o => o.value === {json.dumps(old)} || o.value === {json.dumps(new)}).length === 2")
        _delay_jobs(page, old)
        page.eval(f"""(() => {{
            window.__raceStart = Date.now();
            const sel = document.getElementById('exp-select');
            sel.value = {json.dumps(old)};
            sel.dispatchEvent(new Event('change'));
            sel.value = {json.dumps(new)};
            sel.dispatchEvent(new Event('change'));
        }})()""")
        page.wait_for("Date.now() - window.__raceStart > 1500")
        shown = page.eval("document.querySelector('#job-list').textContent")
        assert "new_log_run" in shown and "old_log_run" not in shown, shown
        assert page.eval("document.getElementById('exp-select').value") == new
        assert page.eval("document.getElementById('exp-dl-all').getAttribute('href')").find(new) >= 0
    finally:
        page.close()


def test_results_merges_overlapping_metric_history_without_duplicates(mgr):
    """A full-history load or a reconnect replay must not append steps the live
    stream already delivered; the old code pushed both copies into the chart."""
    eid = _experiment(mgr, "dedupe_metrics", "run1")
    page = Browser(CHROME)
    try:
        page.goto(static_url(mgr, "results.html", experiment=eid))
        page.wait_for("DATA && DATA.experiment === " + json.dumps(eid))
        page.eval("""
            METRIC_CACHE = {};
            mergeJsonlIntoCache([{step: 1, loss: 1}, {step: 2, loss: 2}], "run1");
            mergeJsonlIntoCache([{step: 2, loss: 2}, {step: 3, loss: 3}], "run1");
            _pendingMetrics = {run1: {loss: {steps: [3, 4], values: [3, 4]}}};
            _flushPendingMetrics();
        """)
        assert page.eval("METRIC_CACHE.loss.run1.steps") == [1, 2, 3, 4]
        assert page.eval("METRIC_CACHE.loss.run1.values") == [1, 2, 3, 4]
    finally:
        page.close()
