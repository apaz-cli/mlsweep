"""Renaming runs in the web dashboard, driven through headless Chromium.

The experiments and logs pages each have a Rename button.  A rename made
anywhere, whether on another page or with ``mlsweep rename``, reaches open logs
and results pages over the experiment's WebSocket.  The experiments page polls.
It redraws right after a rename and whenever its tab regains focus.  The system
page never cuts IDs short.

Skipped when no Chromium-family browser is installed (set MLSWEEP_TEST_CHROME
to point at one).
"""

import json
import time

import pytest

from browser import Browser, find_chromium
from campaign_helpers import ok, seed, shared_manager, static_url

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "rename_webui")


@pytest.fixture(scope="module")
def page(mgr):
    b = Browser(CHROME)
    b.url = lambda name, **params: static_url(mgr, name, **params)
    yield b
    b.close()


def _label(mgr, eid, rid):
    return ok(mgr, "GET", f"/api/jobs/{rid}?experiment_id={eid}")["label"]


def _rename(mgr, eid, rid, label):
    ok(mgr, "PUT", f"/api/jobs/{rid}/label", {"experiment_id": eid, "label": label})


def _answer_prompt(b, answer):
    """Make the next rename dialog return *answer* (None = Cancel)."""
    b.eval(f"window.mlPrompt = async () => {json.dumps(answer)}")


def _open_log(b, eid, rid):
    b.goto(b.url("logs.html", experiment=eid, run=rid))
    b.wait_for("document.getElementById('log-header').style.display === ''")
    time.sleep(0.5)  # let the WebSocket connect before anything is renamed


def _runs_text(eid):
    """JS for the text of an experiment's run rows on the experiments page."""
    return f"runRowsOf({json.dumps(eid)}).map(r => r.textContent).join(' ')"


def _log_header(b):
    return b.eval("document.getElementById('log-run-id').textContent")


def _log_list_entry(b, rid):
    return b.eval(f"document.querySelector('.job-item[data-run-id={json.dumps(rid)}] .job-run-id')"
                  ".textContent")


def test_logs_page_renames_selected_run(mgr, page):
    eid = seed(mgr, "default", prefix="logs")
    _open_log(page, eid, "r1")

    _answer_prompt(page, "warmup ablation")
    page.eval("document.getElementById('rename-btn').click()")
    page.wait_for(f"document.getElementById('log-run-id').textContent.startsWith('warmup ablation')")
    assert _label(mgr, eid, "r1") == "warmup ablation"
    assert _log_list_entry(page, "r1").startswith("warmup ablation")
    assert _label(mgr, eid, "r2") is None

    # An empty answer clears the name; Cancel changes nothing.
    _answer_prompt(page, "")
    page.eval("document.getElementById('rename-btn').click()")
    page.wait_for("document.getElementById('log-run-id').textContent === 'r1'")
    assert _label(mgr, eid, "r1") is None
    _answer_prompt(page, None)
    page.eval("document.getElementById('rename-btn').click()")
    assert _label(mgr, eid, "r1") is None


def test_logs_page_follows_renames_made_elsewhere(mgr, page):
    eid = seed(mgr, "default", prefix="live")
    _open_log(page, eid, "r1")

    _rename(mgr, eid, "r1", "from the cli")
    page.wait_for("document.getElementById('log-run-id').textContent.startsWith('from the cli')",
                  timeout=5)
    _rename(mgr, eid, "r2", "other run")
    page.wait_for(f"document.querySelector('.job-item[data-run-id=\"r2\"] .job-run-id')"
                  f".textContent.startsWith('other run')", timeout=5)
    assert _log_header(page).startswith("from the cli")


def test_results_page_follows_renames(mgr, page):
    eid = seed(mgr, "default", prefix="results")
    page.goto(page.url("results.html", experiment=eid))
    page.wait_for("DATA && DATA.runs.length === 2 && _ws && _ws.readyState === 1")

    _rename(mgr, eid, "r1", "baseline")
    page.wait_for("DATA.runs.find(r => r.hash === 'r1').name === 'baseline'", timeout=5)
    _rename(mgr, eid, "r1", None)
    page.wait_for("DATA.runs.find(r => r.hash === 'r1').name === 'r1'", timeout=5)


def test_experiments_page_renames_and_escapes(mgr, page):
    eid = seed(mgr, "default", prefix="exps")
    page.goto(page.url("experiments.html"))
    page.wait_for(f"!!document.querySelector('.exp-row[data-exp-id={json.dumps(eid)}]')")
    page.eval(f"toggleExpand({json.dumps(eid)})")
    page.wait_for(f"(jobsCache[{json.dumps(eid)}] || []).length === 2")

    # A name is text, never markup.
    _answer_prompt(page, "<b>bold</b>")
    page.eval(f"renameJob({json.dumps(eid)}, 'r1')")
    page.wait_for(f"{_runs_text(eid)}.includes('<b>bold</b>')")
    assert _label(mgr, eid, "r1") == "<b>bold</b>"
    assert page.eval(f"!runRowsOf({json.dumps(eid)}).some(r => r.querySelector('b'))")


def test_experiments_page_redraws_from_manager_after_rename(mgr, page):
    eid = seed(mgr, "default", prefix="redraw")
    page.goto(page.url("experiments.html"))
    page.wait_for(f"!!document.querySelector('.exp-row[data-exp-id={json.dumps(eid)}]')")
    page.eval(f"toggleExpand({json.dumps(eid)})")
    page.wait_for(f"(jobsCache[{json.dumps(eid)}] || []).length === 2")
    jobs_text = _runs_text(eid)

    # r2 is renamed behind the page's back.  Renaming r1 in the page redraws
    # from the manager, so r2's new name shows well before the 10 s poll.
    _rename(mgr, eid, "r2", "renamed elsewhere")
    _answer_prompt(page, "renamed here")
    page.eval(f"renameJob({json.dumps(eid)}, 'r1')")
    page.wait_for(f"{jobs_text}.includes('renamed here') && {jobs_text}.includes('renamed elsewhere')",
                  timeout=3)

    # Coming back to the tab also redraws.
    _rename(mgr, eid, "r1", "renamed while away")
    page.eval("document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for(f"{jobs_text}.includes('renamed while away')", timeout=3)


def test_system_page_shows_whole_run_and_experiment_ids(mgr, page):
    # Only the scheduler marks jobs running, so render a running job directly.
    eid = "system_page_experiment_with_a_long_id_20261001_1144_6ba4"
    rid = "sweep_optmuon.lrs0.001_bs512_warmup2000_seed3"
    job = {"experiment_id": eid, "run_id": rid, "label": "<i>muon</i>", "status": "running",
           "worker_id": "localhost:1", "dispatched_gpu_ids": "[0]", "elapsed": 5}
    page.goto(page.url("system.html"))
    page.eval("document.body.style.width = '600px'")  # narrow, so long IDs must wrap
    # Let the page's first poll land, so it can't overwrite the row (the next is 5 s away).
    page.wait_for("!document.getElementById('inflight-tbody').textContent.includes('Loading')")
    page.eval(f"renderInFlight([{json.dumps(job)}])")

    row = "document.querySelector('#inflight-tbody tr')"
    run_cell, exp_cell = page.eval(f"[...{row}.cells].slice(0, 2).map(c => c.textContent)")
    assert run_cell == f"<i>muon</i> {rid}"
    assert exp_cell == eid
    # Long IDs wrap inside the table rather than spilling out of it.
    assert page.eval(f"[...{row}.cells].every(c => c.scrollWidth <= c.clientWidth)")
