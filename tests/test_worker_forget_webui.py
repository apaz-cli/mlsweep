"""The System page's Disconnect and the Connect dialog's Forget.

Skipped when no Chromium-family browser is installed (set MLSWEEP_TEST_CHROME
to point at one).
"""

import time

import pytest

from browser import Browser, find_chromium
from campaign_helpers import ok, shared_manager, static_url

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "worker_forget_webui")


@pytest.fixture
def page():
    b = Browser(CHROME)
    yield b
    b.close()


def test_a_disconnected_worker_leaves_the_grid_and_the_connect_dialog_forgets_it(mgr, page):
    """Disconnect takes a worker off the System page; the Connect dialog still
    offers it, and its Forget button drops it from the list."""
    wid = "forget-me"
    ok(mgr, "POST", "/api/workers", {"worker_id": wid, "host": "user@nosuchhost.invalid"})
    deadline = time.monotonic() + 60
    while ok(mgr, "GET", f"/api/workers/{wid}")["status"] != "dead":
        assert time.monotonic() < deadline, "the launch never failed"
        time.sleep(0.2)

    page.goto(static_url(mgr, "system.html"))
    page.wait_for(f"!!document.querySelector('.worker-card[data-worker=\"{wid}\"]')")
    ok(mgr, "DELETE", f"/api/workers/{wid}")
    page.eval("refresh()")
    page.wait_for(f"!document.querySelector('.worker-card[data-worker=\"{wid}\"]')")

    page.eval("document.getElementById('btn-connect-worker').click()")
    forget = f".btn-forget[data-worker-id=\"{wid}\"]"
    page.wait_for(f"!!document.querySelector('{forget}')")
    page.eval(f"document.querySelector('{forget}').click()")
    page.wait_for(f"!document.querySelector('{forget}')")
    assert wid not in {w["worker_id"] for w in ok(mgr, "GET", "/api/workers")}
