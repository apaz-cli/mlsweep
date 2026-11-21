"""Every dashboard page gets its token, nav, theme and API helpers from chrome.js."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from browser import Browser, find_chromium
from campaign_helpers import shared_manager, static_url


WEBUI = Path(__file__).parent.parent / "mlsweep" / "webui"
PAGES = ["experiments", "results", "logs", "files", "code", "notes", "export", "system"]
CHROME = find_chromium()


@pytest.mark.parametrize("page_name", PAGES)
def test_pages_do_not_redefine_the_shared_parts(page_name: str) -> None:
    text = (WEBUI / f"{page_name}.html").read_text()
    if page_name == "notes":
        text += (WEBUI / "notes.js").read_text()
    for pattern in (r"function apiFetch\b", r"function pageHref\b", r"function setConnStatus\b",
                    r"function authURL\b", r"const TOKEN\b", r"\.nav-link\s*\{", r'id="nav-\w+" href'):
        assert not re.search(pattern, text), pattern
    assert '<script src="chrome.js"></script>' in text or page_name == "notes"
    assert text.index("chrome.js") < text.index("campaign.js")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):  # type: ignore[no-untyped-def]
    yield from shared_manager(tmp_path_factory, "webui_chrome")


@pytest.fixture(scope="module")
def browser():  # type: ignore[no-untyped-def]
    if CHROME is None:
        pytest.skip("no Chromium")
    b = Browser(CHROME)
    yield b
    b.close()


@pytest.mark.parametrize("page_name", PAGES)
def test_nav_is_the_same_on_every_page(browser, mgr, page_name: str) -> None:  # type: ignore[no-untyped-def]
    browser.goto(static_url(mgr, f"{page_name}.html"))
    labels = browser.eval("[...document.querySelectorAll('nav a.nav-link')].map(a => a.textContent)")
    assert labels == ["Experiments", "Results", "Logs", "Artifacts", "Code", "Notes", "Export", "System"]
    active = browser.eval("[...document.querySelectorAll('a.nav-link.active')].map(a => a.textContent)")
    assert active == [dict(zip(PAGES, labels))[page_name]]
    # Campaign links are added to these, but the token never is.
    hrefs = browser.eval("[...document.querySelectorAll('a.nav-link[id]')].map(a => a.getAttribute('href'))")
    assert len(hrefs) == 7 and not any("token" in h for h in hrefs)
    assert browser.eval("!!document.getElementById('theme-toggle')")
    assert browser.eval("!!document.getElementById('nav-sync')") == (page_name == "notes")
    assert browser.eval("!!document.getElementById('conn-dot')") == (page_name != "notes")


def test_token_leaves_the_address_bar_but_keeps_the_query(browser, mgr) -> None:  # type: ignore[no-untyped-def]
    browser.goto(static_url(mgr, "files.html", experiment="abc"))
    assert "token" not in browser.eval("location.search")
    assert "experiment=abc" in browser.eval("location.search")
    assert browser.eval("localStorage.getItem('mlsweep_token')") == browser.eval("TOKEN") != ""
    # The next page finds it in localStorage.
    browser.goto(f"{mgr.url}/static/system.html")
    assert browser.eval("apiFetch('/api/workers').then(r => r.status)") == 200


def test_theme_toggle_persists_and_announces(browser, mgr) -> None:  # type: ignore[no-untyped-def]
    browser.goto(static_url(mgr, "system.html"))
    browser.eval("localStorage.removeItem('theme')")
    browser.goto(f"{mgr.url}/static/system.html")
    assert browser.eval("document.documentElement.getAttribute('data-theme')") == "dark"
    browser.eval("window._told = 0; document.addEventListener('mlthemechange', () => window._told++);"
                 "document.getElementById('theme-toggle').click()")
    assert browser.eval("[document.documentElement.getAttribute('data-theme'), localStorage.getItem('theme'),"
                        " document.getElementById('theme-toggle').textContent, window._told]") == [None, "light", "☾", 1]
    browser.goto(f"{mgr.url}/static/logs.html")
    assert browser.eval("document.documentElement.getAttribute('data-theme')") is None
    browser.eval("localStorage.removeItem('theme')")


def test_conn_status(browser, mgr) -> None:  # type: ignore[no-untyped-def]
    browser.goto(static_url(mgr, "export.html"))
    browser.eval("setConnStatus('reconnecting')")
    assert browser.eval("[document.getElementById('conn-dot').className, document.getElementById('conn-label').textContent]") \
        == ["reconnecting", "Reconnecting…"]


def test_an_outdated_worker_offers_the_update(browser, mgr) -> None:  # type: ignore[no-untyped-def]
    browser.goto(static_url(mgr, "system.html"))
    browser.eval("""window._posted = [];
      window.apiFetch = (path, opts) => { if (opts && opts.method === 'POST') window._posted.push(path);
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 })); };
      window.refresh = () => {};
      renderWorkers([
        { worker_id: 'gpu1:7890', host: 'gpu1', port: 7890, status: 'dead', outdated: true,
          last_error: 'worker speaks protocol 1, this manager 2' },
        { worker_id: 'gpu2:7890', host: 'gpu2', port: 7890, status: 'dead', outdated: false },
        { worker_id: 'gpu3:7890', host: 'gpu3', port: 7890, status: 'updating', outdated: false },
        { worker_id: 'gpu4:7890', host: 'gpu4', port: 7890, status: 'connected', outdated: true },
        { worker_id: 'gpu5:7890', host: 'gpu5', port: 7890, status: 'connected', outdated: false }]);""")
    cards = browser.eval("[...document.querySelectorAll('.worker-card')].map(c => "
                         "[!!c.querySelector('.btn-update'), !!c.querySelector('.btn-disconnect'), "
                         "!!c.querySelector('.worker-warning')])")
    assert cards == [[True, False, False], [False, False, False], [False, False, False],
                     [True, True, True], [False, True, False]]
    browser.eval("document.querySelector('.btn-update').click()")
    browser.wait_for("window._posted.length === 1")
    assert browser.eval("window._posted") == ["/api/workers/gpu1%3A7890/update"]


@pytest.mark.parametrize("page_name,select_id", [("files", "exp-sel"), ("logs", "exp-select")])
def test_a_long_experiment_name_keeps_the_sidebar_buttons_inside(browser, mgr, page_name, select_id) -> None:  # type: ignore[no-untyped-def]
    """The select must shrink below its longest option, or the refresh button is clipped."""
    browser.goto(static_url(mgr, f"{page_name}.html"))
    rights = browser.eval(f"""(() => {{
      const s = document.getElementById({select_id!r});
      const o = document.createElement('option'); o.textContent = 'x'.repeat(120); s.append(o); s.value = o.value;
      document.getElementById('exp-dl-all').style.display = '';
      const edge = document.getElementById('sidebar').getBoundingClientRect().right;
      return [...s.parentElement.children].map(e => e.getBoundingClientRect().right <= edge);
    }})()""")
    assert rights == [True, True, True]
