"""Experiment descriptions in the web dashboard, driven through headless Chromium.

An experiment without one gets a "no description" button on the Experiments
page. That button, the row menu's "Edit description…" item and a click on the
description on Results all open the same multi-line prompt, which saves through
``PUT /api/experiments/EXP/description``.

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
    yield from shared_manager(tmp_path_factory, "description_webui")


@pytest.fixture(scope="module")
def page(mgr):
    b = Browser(CHROME)
    b.url = lambda name, **params: static_url(mgr, name, **params)
    yield b
    b.close()


def _note(mgr, eid):
    return ok(mgr, "GET", f"/api/experiments/{eid}")["note"]


def _wait_note(mgr, eid, want, timeout=5.0):
    """The dialog closes before its save lands, so wait for the manager to have *want*."""
    deadline = time.monotonic() + timeout
    while _note(mgr, eid) != want and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _note(mgr, eid) == want


def _answer_dialog(b, text):
    """Type *text* into the open description dialog and save with Ctrl+Enter."""
    b.wait_for("!!document.querySelector('.ml-dialog textarea')")
    b.eval(f"""(() => {{
      const t = document.querySelector('.ml-dialog textarea');
      t.value = {json.dumps(text)};
      t.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', bubbles: true }}));  // a new line, not a save
      if (!document.querySelector('.ml-dialog textarea')) throw new Error('Enter closed the dialog');
      t.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', ctrlKey: true, bubbles: true }}));
    }})()""")
    b.wait_for("!document.querySelector('.ml-dialog')")


def _row(eid):
    return f"document.querySelector('.exp-row[data-exp-id={json.dumps(eid)}]')"


def test_experiments_page_adds_and_edits_a_description(mgr, page):
    eid = seed(mgr, "test", "desc")
    page.goto(page.url("experiments.html", campaign="test"))
    page.wait_for(f"!!{_row(eid)}?.querySelector('.exp-no-desc')")

    page.eval(f"{_row(eid)}.querySelector('.exp-no-desc').click()")
    _answer_dialog(page, "first try\nsecond line")
    _wait_note(mgr, eid, "first try\nsecond line")
    page.wait_for(f"!{_row(eid)}.querySelector('.exp-no-desc')")
    assert page.eval(f"{_row(eid)}.querySelector('.exp-name').dataset.note") == "first try\nsecond line"

    # The row menu edits it; an empty answer removes it and the button comes back.
    page.eval(f"void editExpDescription({json.dumps(eid)})")  # eval would wait for the dialog
    page.wait_for("!!document.querySelector('.ml-dialog textarea')")
    assert page.eval("document.querySelector('.ml-dialog textarea').value") == "first try\nsecond line"
    _answer_dialog(page, "  ")
    _wait_note(mgr, eid, None)
    page.wait_for(f"!!{_row(eid)}.querySelector('.exp-no-desc')")
    labels = page.eval("expManageItems('x').map(i => i.label)")
    assert "Edit description…" in labels


def test_results_page_shows_and_edits_the_description(mgr, page):
    eid = seed(mgr, "test", "desc")
    ok(mgr, "PUT", f"/api/experiments/{eid}/description", {"description": "from the cli"})
    page.goto(page.url("results.html", campaign="test", experiment=eid))
    page.wait_for("DATA && DATA.experiment === " + json.dumps(eid))
    page.wait_for("document.getElementById('sweep-note').textContent === 'from the cli'")

    page.eval("document.getElementById('sweep-note').click()")
    _answer_dialog(page, "edited on results")
    _wait_note(mgr, eid, "edited on results")
    page.wait_for("document.getElementById('sweep-note').textContent === 'edited on results'")
