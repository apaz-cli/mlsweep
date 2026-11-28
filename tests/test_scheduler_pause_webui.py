"""The System page's scheduling pause: the header button and a worker's own pause.

Skipped when no Chromium-family browser is installed (set MLSWEEP_TEST_CHROME
to point at one).
"""

import json

import pytest

from browser import Browser, find_chromium
from campaign_helpers import ok, shared_manager, static_url

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "scheduler_pause_webui")


@pytest.fixture
def page():
    b = Browser(CHROME)
    yield b
    b.close()


def test_the_pause_button_pauses_and_unpauses_everywhere(mgr, page):
    page.goto(static_url(mgr, "system.html"))
    page.wait_for("document.getElementById('btn-pause').textContent === 'Pause scheduling'")
    assert page.eval("document.getElementById('pause-banner').classList.contains('hidden')")

    page.eval("document.getElementById('btn-pause').click()")
    page.wait_for("!document.getElementById('pause-banner').classList.contains('hidden')")
    assert page.eval("document.getElementById('btn-pause').textContent") == "Unpause scheduling"
    assert ok(mgr, "GET", "/api/scheduler")["paused"] is True

    # Another tab sees the pause on its next refresh.
    page.goto(static_url(mgr, "system.html"))
    page.wait_for("document.getElementById('btn-pause').textContent === 'Unpause scheduling'")
    page.eval("document.getElementById('btn-pause').click()")
    page.wait_for("document.getElementById('pause-banner').classList.contains('hidden')")
    assert ok(mgr, "GET", "/api/scheduler")["paused"] is False


def test_a_paused_worker_shows_it_and_offers_to_unpause(mgr, page):
    page.goto(static_url(mgr, "system.html"))
    page.wait_for("typeof renderWorkers === 'function'")
    worker = {"worker_id": "gpu1:7000", "host": "gpu1", "status": "connected", "gpus": [0],
              "gpu_runs": {}, "max_jobs_per_gpu": 1, "paused": True}
    page.eval(f"refresh = async () => {{}}; renderWorkers([{json.dumps(worker)}])")
    assert page.eval("[...document.querySelectorAll('.worker-header .badge')].map(b => b.textContent)") \
        == ["connected", "paused"]
    assert page.eval("workerItems('gpu1:7000')[0].label") == "Unpause scheduling here"
    page.eval(f"renderWorkers([{json.dumps({**worker, 'paused': False})}])")
    assert page.eval("document.querySelectorAll('.worker-header .badge').length") == 1
    assert page.eval("workerItems('gpu1:7000')[0].label") == "Pause scheduling here"
