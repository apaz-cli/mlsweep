"""Notes autosave races: flushing must wait for a save that is already out."""

import json
import time

import pytest
from browser import Browser, find_chromium
from campaign_helpers import call, ok, shared_manager, static_url, uid

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "notes_save_race")


def test_flush_saves_waits_for_an_in_flight_save(mgr):
    path = f"save-race-{uid('note')}.md"
    ok(mgr, "PUT", "/api/notes/file", {"path": path, "content": "old\n", "base_hash": None})
    page = Browser(CHROME)
    try:
        page.goto(static_url(mgr, "notes.html", path=path))
        page.wait_for(f"S.tabs.some(t => t.path === {json.dumps(path)} && t.doc && !t.saving)")
        # Make the tab dirty first; CodeMirror emits its change event after the
        # operation, so wait for the dirty flag before starting the save.
        page.eval(f"""(() => {{
            const t = S.tabs.find(x => x.path === {json.dumps(path)});
            t.doc.setValue("new text\\n");
            return true;
        }})()""")
        page.wait_for(f"S.tabs.find(t => t.path === {json.dumps(path)}).dirty === true")
        page.eval(f"""(() => {{
            window.__origApiJSON = apiJSON;
            window.__saveStarted = false;
            window.__saveRelease = null;
            window.__flushDone = false;
            apiJSON = (p, method, data) => {{
                if (method === "PUT" && data && data.path === {json.dumps(path)}) {{
                    window.__saveStarted = true;
                    return new Promise(resolve => {{
                        window.__saveRelease = () => resolve(window.__origApiJSON(p, method, data));
                    }});
                }}
                return window.__origApiJSON(p, method, data);
            }};
            const t = S.tabs.find(x => x.path === {json.dumps(path)});
            saveTab(t);
            flushSaves().then(() => {{ window.__flushDone = true; }});
            return true;
        }})()""")
        page.wait_for("window.__saveStarted === true")
        time.sleep(0.2)
        assert page.eval("window.__flushDone") is False
        page.eval("window.__saveRelease(); true")
        page.wait_for("window.__flushDone === true")
        status, body = call(mgr, "GET", f"/api/notes/file?path={path}")
        assert status == 200
        assert body["content"] == "new text\n"
    finally:
        page.close()
