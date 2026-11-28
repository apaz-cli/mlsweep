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
    # The textarea must sit inside the dialog: it used to spill out of the
    # 340px-wide box and hang over the page behind it.
    box = b.eval("""(() => {
      const d = document.querySelector('.ml-dialog');
      const dr = d.getBoundingClientRect();
      const tr = d.querySelector('textarea').getBoundingClientRect();
      return [dr.left, dr.right, dr.top, dr.bottom, tr.left, tr.right, tr.top, tr.bottom, innerWidth];
    })()""")
    dleft, dright, dtop, dbottom, tleft, tright, ttop, tbottom, vw = box
    assert dleft <= tleft and tright <= dright, f"the textarea spills out of its dialog: {box}"
    assert dtop <= ttop and tbottom <= dbottom, f"the textarea spills out of its dialog: {box}"
    assert dleft >= 0 and dright <= vw, f"the dialog is off screen: {box}"
    b.eval(f"""(() => {{
      const t = document.querySelector('.ml-dialog textarea');
      t.value = {json.dumps(text)};
      t.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', bubbles: true }}));  // a new line, not a save
      if (!document.querySelector('.ml-dialog textarea')) throw new Error('Enter closed the dialog');
      t.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', ctrlKey: true, bubbles: true }}));
    }})()""")
    b.wait_for("!document.querySelector('.ml-dialog')")


def _hover_name(b, over=True):
    """Hover the experiment's name on Results (or leave it) without a real mouse."""
    kind = "'pointerover'" if over else "'pointerout'"
    related = "null" if over else "document.body"
    b.eval(f"""(() => {{ const el = document.getElementById('exp-name');
        el.dispatchEvent(new PointerEvent({kind}, {{ bubbles: true, relatedTarget: {related} }})); }})()""")


def _wait_tip(b, text):
    b.wait_for("(t => !!t && !t.hidden && t.textContent === %s)(document.querySelector('.ml-hover-tip'))"
               % json.dumps(text))


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

    # The description is a tooltip on the name, not a line in the sidebar.
    assert page.eval("!document.getElementById('sweep-note')")
    _hover_name(page)
    _wait_tip(page, "from the cli")
    page.wait_for("document.getElementById('exp-name').classList.contains('has-note')")
    _hover_name(page, over=False)
    page.wait_for("document.querySelector('.ml-hover-tip').hidden")

    # Tabbing to the name shows it too, for people who never use a mouse.
    page.call("Emulation.setFocusEmulationEnabled", enabled=True)
    page.eval("document.getElementById('exp-name').focus()")
    _wait_tip(page, "from the cli")
    page.eval("document.getElementById('exp-name').blur()")
    page.wait_for("document.querySelector('.ml-hover-tip').hidden")

    # A click on the name edits it, and the tooltip follows the new text.
    page.eval("document.getElementById('exp-name').click()")
    _answer_dialog(page, "edited on results")
    _wait_note(mgr, eid, "edited on results")
    page.wait_for("DATA.note === 'edited on results'")
    _hover_name(page)
    _wait_tip(page, "edited on results")

    # With no description the tooltip says how to add one.
    ok(mgr, "PUT", f"/api/experiments/{eid}/description", {"description": None})
    page.eval(f"void refreshExperiments().then(() => loadExperiment({json.dumps(eid)}))")
    page.wait_for("DATA && DATA.experiment === " + json.dumps(eid) + " && DATA.note === null")
    _hover_name(page)
    _wait_tip(page, "No description — click the name to add one.")
    page.wait_for("document.querySelector('.ml-hover-tip').classList.contains('placeholder')")
