"""Campaigns in the web dashboard, driven through headless Chromium.

The experiments page has a campaign selector at the top; the results, logs and
artifacts pages have one above their experiment selector.  Each filters the
experiments to the selected campaign, remembers the choice, and carries it in
the URL and in links to the other pages.  Skipped when no Chromium-family
browser is installed (set MLSWEEP_TEST_CHROME to point at one).
"""

import json

import pytest

from browser import Browser, find_chromium
from campaign_helpers import TOKEN, ok, seed, shared_manager

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")

# Experiments seeded once per module, as campaign -> experiment names, newest last.
_LAYOUT = {"default": ["d1"], "alpha": ["a1", "a2"], "beta": ["b1"]}

# Each page's (campaign select, experiment select) ids.
def _js_list(values):
    """*values* sorted, as the JS string literal ``JSON.stringify`` would produce."""
    return json.dumps(json.dumps(sorted(values), separators=(",", ":")))


_SIDEBARS = {
    "results.html": ("campaign-sel", "experiment-sel"),
    "logs.html": ("campaign-select", "exp-select"),
    "files.html": ("campaign-sel", "exp-sel"),
}


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "campaigns_webui")


@pytest.fixture(scope="module")
def seeded(mgr):
    """``{name: experiment_id}`` for the experiments in ``_LAYOUT``."""
    ids = {}
    for campaign, names in _LAYOUT.items():
        for name in names:
            ids[name] = seed(mgr, campaign, prefix=name)
    return ids


@pytest.fixture(scope="module")
def _browser():
    b = Browser(CHROME)
    yield b
    b.close()


@pytest.fixture
def page(mgr, seeded, _browser):
    """The browser, with remembered campaign cleared, plus a URL builder."""
    _browser.goto(f"{mgr.url}/static/experiments.html")
    _browser.eval("localStorage.clear()")

    def url(name, **params):
        query = "&".join(f"{k}={v}" for k, v in {"token": TOKEN, **params}.items())
        return f"{mgr.url}/static/{name}?{query}"

    _browser.url = url
    return _browser


def _options(b, select_id):
    return b.eval(f"[...document.getElementById('{select_id}').options].map(o => o.value)")


def _selected(b, select_id):
    return b.eval(f"document.getElementById('{select_id}').value")


def _choose(b, select_id, value):
    b.eval(f"""(() => {{ const s = document.getElementById('{select_id}');
        s.value = {json.dumps(value)}; s.dispatchEvent(new Event('change')); }})()""")


def _rows(b):
    """Experiment ids shown in the experiments table, once it has rendered."""
    b.wait_for("document.getElementById('exp-table').style.display !== '' "
               "|| document.querySelectorAll('.exp-row').length > 0")
    return sorted(b.eval("[...document.querySelectorAll('.exp-row')].map(r => r.dataset.expId)"))


def _wait_rows(b, expected):
    b.wait_for(f"JSON.stringify([...document.querySelectorAll('.exp-row')]"
               f".map(r => r.dataset.expId).sort()) === {_js_list(expected)}")


def _every(mgr):
    """Every experiment id on the manager, the expected view of "All campaigns"."""
    return sorted(e["experiment_id"] for e in ok(mgr, "GET", "/api/experiments"))


def _campaign_options(mgr):
    """The campaign select's expected values, All first and then every campaign by name."""
    return ["*"] + [c["campaign"] for c in ok(mgr, "GET", "/api/campaigns")]


def _ids(seeded, *names):
    return sorted(seeded[n] for n in names)


def _campaign_param(b):
    return b.eval("new URLSearchParams(location.search).get('campaign')")


# ── Experiments page ───────────────────────────────────────────────────────────


def test_experiments_page_starts_in_default_campaign(page, mgr, seeded):
    page.goto(page.url("experiments.html"))
    _wait_rows(page, _ids(seeded, "d1"))
    assert _selected(page, "campaign-select") == "default"
    assert _options(page, "campaign-select") == _campaign_options(mgr)
    assert page.eval("document.querySelector('#campaign-select option').textContent") == "All campaigns"
    assert _campaign_param(page) == "default"


def test_experiments_selector_is_at_the_top(page, seeded):
    page.goto(page.url("experiments.html"))
    page.wait_for("document.getElementById('campaign-select').options.length > 1")
    # The selector sits in the page header, before the table.
    assert page.eval("""document.getElementById('campaign-select')
        .compareDocumentPosition(document.getElementById('exp-table'))
        & Node.DOCUMENT_POSITION_FOLLOWING""")
    assert page.eval("!!document.querySelector('#exp-header #campaign-select')")


def test_experiments_page_filters_on_change(page, seeded):
    page.goto(page.url("experiments.html"))
    _wait_rows(page, _ids(seeded, "d1"))
    _choose(page, "campaign-select", "alpha")
    _wait_rows(page, _ids(seeded, "a1", "a2"))
    assert _campaign_param(page) == "alpha"
    assert page.eval("localStorage.getItem('mlsweep_campaign')") == "alpha"
    _choose(page, "campaign-select", "beta")
    _wait_rows(page, _ids(seeded, "b1"))


def test_experiments_all_campaigns_shows_everything_tagged(page, mgr, seeded):
    page.goto(page.url("experiments.html", campaign="*"))
    _wait_rows(page, _every(mgr))
    tags = page.eval("""Object.fromEntries([...document.querySelectorAll('.exp-row')].map(r =>
        [r.dataset.expId, r.querySelector('.campaign-tag')?.textContent]))""")
    assert tags[seeded["a1"]] == "alpha"
    assert tags[seeded["b1"]] == "beta"
    assert tags[seeded["d1"]] == "default"


def test_experiments_single_campaign_has_no_tags(page, seeded):
    page.goto(page.url("experiments.html", campaign="alpha"))
    _wait_rows(page, _ids(seeded, "a1", "a2"))
    assert page.eval("document.querySelectorAll('.campaign-tag').length") == 0


def test_selection_is_remembered_across_visits(page, seeded):
    page.goto(page.url("experiments.html", campaign="beta"))
    _wait_rows(page, _ids(seeded, "b1"))
    page.goto(page.url("experiments.html"))
    _wait_rows(page, _ids(seeded, "b1"))
    assert _selected(page, "campaign-select") == "beta"


def test_url_beats_remembered_selection(page, seeded):
    page.goto(page.url("experiments.html", campaign="beta"))
    _wait_rows(page, _ids(seeded, "b1"))
    page.goto(page.url("experiments.html", campaign="alpha"))
    _wait_rows(page, _ids(seeded, "a1", "a2"))


def test_links_carry_the_campaign(page, seeded):
    page.goto(page.url("experiments.html", campaign="alpha"))
    _wait_rows(page, _ids(seeded, "a1", "a2"))
    hrefs = page.eval("""Object.fromEntries([...document.querySelectorAll('a.nav-link')]
        .filter(a => a.id).map(a => [a.id, a.getAttribute('href')]))""")
    for nav in ("nav-results", "nav-logs", "nav-files", "nav-system"):
        assert "campaign=alpha" in hrefs[nav], hrefs
    _choose(page, "campaign-select", "beta")
    page.wait_for("document.getElementById('nav-logs').getAttribute('href').includes('campaign=beta')")
    # Row links are built at render time and rewritten when clicked.
    href = page.eval("""(() => { const a = document.querySelector('.exp-row a.btn-link');
        a.addEventListener('click', e => e.preventDefault(), { once: true });
        a.click(); return a.getAttribute('href'); })()""")
    assert "campaign=beta" in href and "experiment=" in href


def test_empty_campaign_explains_how_to_submit(page, seeded):
    page.goto(page.url("experiments.html", campaign="nobody_here"))
    page.wait_for("document.getElementById('empty-msg').style.display === ''")
    page.wait_for("document.getElementById('empty-msg').textContent.includes('nobody_here')")
    text = page.eval("document.getElementById('empty-msg').textContent")
    assert "No experiments in campaign nobody_here" in text
    assert "--campaign nobody_here" in text
    assert "nobody_here" in _options(page, "campaign-select")


def test_default_campaign_hint_omits_flag(page, mgr, seeded):
    page.goto(page.url("experiments.html", campaign="default"))
    _wait_rows(page, _ids(seeded, "d1"))
    assert "--campaign" not in page.eval("cmdBlock('http://x')")


def _prompt(b, value):
    """Answer the open mlPrompt dialog with *value*."""
    b.wait_for("!!document.querySelector('.ml-dialog input')")
    b.eval(f"""(() => {{ document.querySelector('.ml-dialog input').value = {json.dumps(value)};
        document.querySelector('.ml-dialog .ml-btn-ok').click(); }})()""")


def _click_move(b, eid):
    b.eval(f"""[...document.getElementById('exp-row-{eid}').querySelectorAll('button')]
        .find(x => x.textContent === 'Move').click()""")


def test_move_button_moves_experiment(page, mgr):
    eid = seed(mgr, "gamma", prefix="mv")
    page.goto(page.url("experiments.html", campaign="gamma"))
    _wait_rows(page, [eid])
    _click_move(page, eid)
    _prompt(page, "delta")
    _wait_rows(page, [])
    assert ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == "delta"
    page.wait_for("[...document.getElementById('campaign-select').options].some(o => o.value === 'delta')")
    _choose(page, "campaign-select", "delta")
    _wait_rows(page, [eid])


def test_move_button_rejects_invalid_name(page, mgr):
    eid = seed(mgr, "gamma2", prefix="mv")
    page.goto(page.url("experiments.html", campaign="gamma2"))
    _wait_rows(page, [eid])
    _click_move(page, eid)
    _prompt(page, "not valid!")
    page.wait_for("document.querySelector('.ml-dialog-message')?.textContent.includes('campaign must be')")
    page.eval("document.querySelector('.ml-dialog .ml-btn-ok').click()")
    assert ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == "gamma2"


def test_move_button_cancel_changes_nothing(page, mgr):
    eid = seed(mgr, "gamma3", prefix="mv")
    page.goto(page.url("experiments.html", campaign="gamma3"))
    _wait_rows(page, [eid])
    _click_move(page, eid)
    page.wait_for("!!document.querySelector('.ml-dialog .ml-btn-cancel')")
    page.eval("document.querySelector('.ml-dialog .ml-btn-cancel').click()")
    assert ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == "gamma3"
    assert _rows(page) == [eid]


def test_index_redirect_keeps_campaign(page, mgr, seeded):
    page.call("Page.navigate", url=f"{mgr.url}/static/index.html?token={TOKEN}&campaign=beta")
    page.wait_for("location.pathname.endsWith('experiments.html') && document.readyState === 'complete'")
    _wait_rows(page, _ids(seeded, "b1"))


# ── Results, logs and artifacts pages ──────────────────────────────────────────


def _exp_options(b, sel):
    b.wait_for(f"document.getElementById('{sel}').options.length > 0 "
               f"&& document.getElementById('{sel}').options[0].textContent !== 'Loading…'")
    return sorted(v for v in _options(b, sel) if v)


def _wait_exp_options(b, sel, expected):
    b.wait_for(f"JSON.stringify([...document.getElementById('{sel}').options].map(o => o.value)"
               f".filter(Boolean).sort()) === {_js_list(expected)}")


def _shown(b, name):
    """Which experiment the page has loaded, read from its own state."""
    return b.eval({
        "results.html": "LOADED_EXP",
        "logs.html": "document.getElementById('exp-dl-all').getAttribute('href')",
        "files.html": "loadedExp",
    }[name])


def _wait_shown(b, name, eid):
    if name == "logs.html":
        b.wait_for(f"(document.getElementById('exp-dl-all').getAttribute('href') || '')"
                   f".includes('/experiments/{eid}/') "
                   f"&& document.getElementById('exp-dl-all').style.display !== 'none'")
    else:
        b.wait_for(f"{'LOADED_EXP' if name == 'results.html' else 'loadedExp'} === {json.dumps(eid)}")


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_campaign_select_sits_above_experiment_select(page, name):
    camp, exp = _SIDEBARS[name]
    page.goto(page.url(name))
    page.wait_for(f"document.getElementById('{camp}').options.length > 1")
    assert page.eval(f"""document.getElementById('{camp}').compareDocumentPosition(
        document.getElementById('{exp}')) & Node.DOCUMENT_POSITION_FOLLOWING""")
    assert page.eval(f"""(() => {{
        const c = document.getElementById('{camp}').getBoundingClientRect();
        const e = document.getElementById('{exp}').getBoundingClientRect();
        return c.bottom <= e.top; }})()""")


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_lists_only_campaign_experiments(page, mgr, seeded, name):
    camp, exp = _SIDEBARS[name]
    page.goto(page.url(name, campaign="alpha"))
    _wait_exp_options(page, exp, _ids(seeded, "a1", "a2"))
    assert _selected(page, camp) == "alpha"
    assert _options(page, camp) == _campaign_options(mgr)
    _wait_shown(page, name, seeded["a2"])  # newest first


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_defaults_to_default_campaign(page, seeded, name):
    camp, exp = _SIDEBARS[name]
    page.goto(page.url(name))
    _wait_exp_options(page, exp, _ids(seeded, "d1"))
    assert _selected(page, camp) == "default"
    _wait_shown(page, name, seeded["d1"])


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_campaign_change_reloads(page, seeded, name):
    camp, exp = _SIDEBARS[name]
    page.goto(page.url(name, campaign="alpha"))
    _wait_shown(page, name, seeded["a2"])
    _choose(page, camp, "beta")
    _wait_exp_options(page, exp, _ids(seeded, "b1"))
    _wait_shown(page, name, seeded["b1"])
    assert _campaign_param(page) == "beta"
    assert page.eval("localStorage.getItem('mlsweep_campaign')") == "beta"


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_all_campaigns_keeps_current_experiment(page, mgr, seeded, name):
    camp, exp = _SIDEBARS[name]
    page.goto(page.url(name, campaign="beta"))
    _wait_shown(page, name, seeded["b1"])
    _choose(page, camp, "*")
    _wait_exp_options(page, exp, _every(mgr))
    assert _selected(page, exp) == seeded["b1"]
    _wait_shown(page, name, seeded["b1"])


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_deep_link_follows_experiment_campaign(page, seeded, name):
    camp, exp = _SIDEBARS[name]
    page.goto(page.url(name, campaign="alpha"))  # remembered: alpha
    _wait_shown(page, name, seeded["a2"])
    page.goto(page.url(name, experiment=seeded["b1"]))
    _wait_shown(page, name, seeded["b1"])
    assert _selected(page, camp) == "beta"
    assert _exp_options(page, exp) == _ids(seeded, "b1")


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_deep_link_beats_campaign_param(page, seeded, name):
    camp, _ = _SIDEBARS[name]
    page.goto(page.url(name, campaign="default", experiment=seeded["a1"]))
    _wait_shown(page, name, seeded["a1"])
    assert _selected(page, camp) == "alpha"


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_empty_campaign(page, seeded, name):
    camp, exp = _SIDEBARS[name]
    page.goto(page.url(name, campaign="nobody_here"))
    assert page.wait_for(f"document.getElementById('{camp}').value") == "nobody_here"
    page.wait_for(f"document.getElementById('{exp}').textContent.includes('No experiments')")
    page.wait_for("document.body.innerText.includes('No experiments in this campaign')")
    if name == "results.html":
        assert page.eval("LOADED_EXP") is None
    elif name == "files.html":
        assert page.eval("loadedExp") is None
    else:
        assert page.eval("document.getElementById('exp-dl-all').style.display") == "none"
    # Coming back loads an experiment again.
    _choose(page, camp, "alpha")
    _wait_shown(page, name, seeded["a2"])


@pytest.mark.parametrize("name", sorted(_SIDEBARS))
def test_sidebar_nav_links_carry_campaign(page, seeded, name):
    camp, _ = _SIDEBARS[name]
    page.goto(page.url(name, campaign="alpha"))
    _wait_shown(page, name, seeded["a2"])
    page.wait_for("[...document.querySelectorAll('a.nav-link')].filter(a => a.id)"
                  ".every(a => a.getAttribute('href').includes('campaign=alpha'))")
    _choose(page, camp, "beta")
    page.wait_for("[...document.querySelectorAll('a.nav-link')].filter(a => a.id)"
                  ".every(a => a.getAttribute('href').includes('campaign=beta'))")


def test_logs_page_lists_jobs_of_campaign_experiment(page, seeded):
    page.goto(page.url("logs.html", campaign="beta"))
    page.wait_for("document.querySelectorAll('.job-item').length === 2")
    assert sorted(page.eval("[...document.querySelectorAll('.job-item')].map(e => e.dataset.runId)")) == ["r1", "r2"]


def test_files_page_lists_runs_of_campaign_experiment(page, seeded):
    page.goto(page.url("files.html", campaign="beta"))
    _wait_shown(page, "files.html", seeded["b1"])
    page.wait_for("document.getElementById('run-list').textContent.includes('r1')")


def test_results_page_names_loaded_experiment(page, seeded):
    page.goto(page.url("results.html", campaign="beta"))
    page.wait_for(f"document.getElementById('exp-name').textContent === {json.dumps(seeded['b1'])}")


def test_system_page_carries_campaign(page):
    page.goto(page.url("system.html", campaign="alpha"))
    page.wait_for("document.getElementById('nav-logs').getAttribute('href').includes('campaign=alpha')")
    assert _campaign_param(page) == "alpha"


def test_campaign_follows_from_page_to_page(page, seeded):
    page.goto(page.url("experiments.html", campaign="beta"))
    _wait_rows(page, _ids(seeded, "b1"))
    page.eval("document.getElementById('nav-logs').click()")
    page.wait_for("location.pathname.endsWith('logs.html')")
    _wait_shown(page, "logs.html", seeded["b1"])
    assert _selected(page, "campaign-select") == "beta"
