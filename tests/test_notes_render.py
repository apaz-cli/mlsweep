"""The Notes tab's markdown additions, rendered in headless Chromium.

Covers footnotes, numbered figure captions, evidence links, and the
``mlsweep-chart`` / ``mlsweep-table`` blocks.  The block and evidence responses
are stubbed, so this checks the rendering contract, not the manager's metrics
queries (those are covered in ``test_notes_blocks.py`` and ``test_notes_api.py``).

Skipped when no Chromium-family browser is installed.
"""

import json
import time

import pytest

from browser import Browser, find_chromium
from campaign_helpers import shared_manager, static_url

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "notes_render")


@pytest.fixture(scope="module")
def page(mgr):
    b = Browser(CHROME)
    b.url = lambda name, **params: static_url(mgr, name, **params)
    b.goto(static_url(mgr, "notes.html"))
    b.wait_for("typeof renderMarkdown === 'function' && typeof Resolver !== 'undefined'")
    yield b
    b.close()


def render(page, text, files=("fig.png",)):  # type: ignore[no-untyped-def]
    files_js = json.dumps([{"path": p} for p in files])
    return page.eval(
        "(() => { S.files = %s; "
        "S.fileMap = new Map(); Resolver.rebuild(); "
        "const d = document.createElement('div'); document.body.append(d); "
        "renderMarkdown(%s, 'n.md', d); return d.id = 'out-' + Math.random().toString(36).slice(2); })()"
        % (files_js, json.dumps(text)))


def out(page, ident, expr):  # type: ignore[no-untyped-def]
    return page.eval(f"(() => {{ const r = document.getElementById({json.dumps(ident)}); return {expr}; }})()")


def test_footnotes_render_with_backlinks(page) -> None:
    ident = render(page, "One seed is one seed.[^a] And again.[^a] Second.[^b]\n\n"
                         "[^a]: Repeat with three seeds.\n"
                         "[^b]: Report the spread.\n")
    page.wait_for(f"document.getElementById({json.dumps(ident)}).querySelectorAll('.fn-list li').length === 2")
    assert out(page, ident, "[...r.querySelectorAll('sup.fn-ref')].map(s => s.textContent)") == ["1", "1", "2"]
    assert out(page, ident, "r.querySelectorAll('.fn-list li')[0].textContent.trim()").startswith(\
        "Repeat with three seeds.") and "↩" in out(page, ident, "r.querySelectorAll('.fn-list li')[0].textContent")
    assert out(page, ident, "r.querySelectorAll('.fn-list li')[0].id") == "fn-1"


def test_footnote_definition_can_be_in_another_block(page) -> None:
    # Live mode renders one block at a time but numbers over the whole note.
    ident = page.eval(
        "(() => { const d = document.createElement('div'); document.body.append(d); "
        "d.id = 'live-fn'; "
        "const whole = 'A.[^1]\\n\\n[^1]: in another block\\n'; "
        "const fn = collectFootnotes(whole); "
        "renderMarkdown('A.[^1]', 'n.md', d, "
        "  { frontmatter: false, fnDefs: fn.defs, fnNumbers: footnoteNumbers(fn.body, fn.defs) }); "
        "return d.id; })()")
    page.wait_for(f"document.getElementById({json.dumps(ident)}).querySelectorAll('.fn-ref').length === 1")
    assert out(page, ident, "r.querySelector('.fn-ref').textContent") == "1"
    assert out(page, ident, "r.querySelectorAll('.fn-list li').length") == 0


def test_caption_under_an_image_is_numbered(page) -> None:
    ident = render(page, "![[fig.png]]\nFigure: A plot worth citing {#fig-a}\n\n"
                         "![[fig.png]]\nTable: A second figure\n\n"
                         "See [Figure 1](#fig-a).\n")
    page.wait_for(f"document.getElementById({json.dumps(ident)}).querySelectorAll('figure.note-fig').length === 2")
    assert out(page, ident, "r.querySelector('figure.note-fig').id") == "fig-a"
    assert out(page, ident, "[...r.querySelectorAll('figcaption .fig-kind')].map(s => s.textContent)") == \
        ["Figure 1.", "Table 1."]
    assert out(page, ident, "[...r.querySelectorAll('figure.note-fig')][1].id") == "tbl-1"


def test_evidence_links_are_checked(page) -> None:
    results = {
        "refs": {
            "e/r0/loss@200": {"ok": True, "ref": "e/r0/loss@200", "experiment": "e", "run": "r0",
                              "metric": "loss", "at": "200", "status": "done", "value": 1.25,
                              "step": 200, "exact": True},
        },
    }
    ident = page.eval(
        "(() => { window.__origFetch = window.fetch; "
        "window.fetch = (u, o) => String(u).includes('/api/notes/evidence') "
        "  ? Promise.resolve(new Response(%s, {status: 200, headers: {'Content-Type': 'application/json'}})) "
        "  : window.__origFetch(u, o); "
        "const d = document.createElement('div'); document.body.append(d); d.id = 'ev-out'; "
        "renderMarkdown('[1.25](mlsweep:e/r0/loss@200) and [9.99](mlsweep:e/r0/loss@200)', 'n.md', d); "
        "return d.id; })()" % json.dumps(json.dumps({"results": results["refs"]})))
    page.wait_for(f"document.querySelectorAll('#{ident} a.metric-ev .ev-mark').length === 2")
    marks = out(page, ident, "[...r.querySelectorAll('a.metric-ev')].map(a => [a.className, a.textContent, a.dataset.evValue])")
    assert marks[0][-1] == "1.25"
    assert "ev-ok" in marks[0][0] and marks[0][1].endswith("✓")
    assert "ev-bad" in marks[1][0] and marks[1][1].endswith("⚠")
    assert out(page, ident, "r.querySelector('a.metric-ev').getAttribute('href')").startswith("results.html?")
    page.eval("window.fetch = window.__origFetch")


def test_chart_and_table_blocks_render(page) -> None:
    chart = {
        "kind": "chart", "metric": "loss", "x": "step",
        "title": "loss by lr · exp",
        "spec": {"metric": "loss", "caption": "Seed spread", "id": "fig-seeds", "log_y": False},
        "series": [
            {"name": "lr=0.001", "full_name": "e/full-run", "experiment": "e", "run": "",
             "status": "", "params": {}, "runs": 2,
             "x": [1, 2], "y": [1.1, 0.9], "lo": [1.0, 0.8], "hi": [1.2, 1.0], "final": 0.9},
        ],
        "warnings": [],
    }
    table = {
        "kind": "table",
        "title": "loss · exp",
        "spec": {"caption": "", "agg": "final"},
        "warnings": ["1 run logged no loss"],
        "columns": [{"key": "run", "kind": "run"}, {"key": "lr", "kind": "param"},
                    {"key": "loss", "kind": "metric"}],
        "rows": [{"run": "r0", "experiment": "e", "status": "done", "params": {"lr": 0.001},
                  "metrics": {"loss": {"final": 0.9, "step": 2, "min": 0.9, "max": 1.1,
                                       "mean": 1.0, "value": 0.9}}}],
    }
    md = ("```mlsweep-chart\nexperiments: [e]\nmetric: loss\n"
          "caption: Seed spread {#fig-seeds}\n```\n\n"
          "```mlsweep-table\nexperiments: [e]\nmetrics: [loss]\n```")
    js = (
        "(() => { window.__origFetch = window.fetch; "
        "window.fetch = (u, o) => { "
        "  if (!String(u).includes('/api/notes/block')) return window.__origFetch(u, o); "
        "  const body = JSON.parse(o.body); "
        "  const payload = body.kind === 'chart' ? " + json.dumps(chart) + " : " + json.dumps(table) + "; "
        "  return Promise.resolve(new Response(JSON.stringify(payload), "
        "    {status: 200, headers: {'Content-Type': 'application/json'}})); }; "
        "const d = document.createElement('div'); document.body.append(d); d.id = 'blocks-out'; "
        "renderMarkdown(" + json.dumps(md) + ", 'n.md', d); return d.id; })()"
    )
    ident = page.eval(js)
    page.wait_for(f"document.querySelector('#{ident} .js-plotly-plot') !== null", timeout=20)
    page.wait_for(f"document.querySelectorAll('#{ident} .nb-table tbody tr').length === 1")
    assert out(page, ident, "[...r.querySelectorAll('figure.note-fig figcaption .fig-kind')].map(s => s.textContent)") == \
        ["Figure 1."]
    assert out(page, ident, "r.querySelector('figure.note-fig').id") == "fig-seeds"
    assert out(page, ident, "r.querySelector('.nb-table td.nb-num').textContent") == "0.9"
    # The card must be taller than the plot: a fixed-height card clipped the x-axis.
    geom = out(page, ident, "(() => { const c = r.querySelector('.nb-block.nb-chart'); "
                            "const p = c.querySelector('.nb-plot'); "
                            "return [c.clientHeight, p.clientHeight, "
                            "c.querySelectorAll('.xtick text').length]; })()")
    assert geom[0] > geom[1] and geom[2] > 1
    assert out(page, ident, "[...r.querySelectorAll('.nb-title')].map(t => t.textContent)") == \
        ["loss by lr · exp", "loss · exp"]
    assert "lr=0.001" in out(page, ident, "r.querySelector('.nb-legend').textContent")
    assert "e/full-run" in out(page, ident, "r.querySelector('.nb-legend-item').title")
    assert out(page, ident, "r.querySelectorAll('.nb-legend-band').length") == 1
    assert out(page, ident, "[...r.querySelectorAll('.nb-head .nb-refresh')].map(b => b.textContent)") == \
        ["↻", "⤓", "↻", "⧉"]
    md = out(page, ident, "nbTableMarkdown(r.querySelector('table.nb-table'))")
    assert md.splitlines()[0].startswith("| run") and "| r0" in md
    assert out(page, ident, "r.querySelector('.nb-warn').textContent").startswith("1 run logged")
    page.eval("window.fetch = window.__origFetch")


def test_bad_spec_shows_the_error(page) -> None:
    ident = page.eval(
        "(() => { window.__origFetch = window.fetch; "
        "window.fetch = (u, o) => String(u).includes('/api/notes/block') "
        "  ? Promise.resolve(new Response(JSON.stringify({error: 'chart needs `metric`'}), "
        "    {status: 400, headers: {'Content-Type': 'application/json'}})) "
        "  : window.__origFetch(u, o); "
        "const d = document.createElement('div'); document.body.append(d); d.id = 'bad-out'; "
        "renderMarkdown('```mlsweep-chart\\nexperiments: [e]\\n```', 'n.md', d); return d.id; })()")
    page.wait_for(f"document.querySelector('#{ident} .nb-error') !== null")
    assert "chart needs" in out(page, ident, "r.querySelector('.nb-error').textContent")
    page.eval("window.fetch = window.__origFetch")


def _insert_metrics(server, experiment: str, rows: dict[str, list[tuple[int, dict]]]) -> None:
    import sqlite3
    con = sqlite3.connect(server.mlsweep_dir.parent / "manager.db")
    try:
        for run, points in rows.items():
            job_key, attempt = con.execute(
                "SELECT job_key, attempt FROM jobs WHERE experiment_id = ? AND run_id = ?",
                (experiment, run)).fetchone()
            for step, data in points:
                con.execute(
                    "INSERT OR REPLACE INTO metrics (job_key, attempt, step, data) VALUES (?, ?, ?, ?)",
                    (job_key, attempt, step, json.dumps(data)))
        con.commit()
    finally:
        con.close()


def test_real_metrics_end_to_end(mgr, page) -> None:
    from campaign_helpers import ok, uid
    exp = uid("render")
    ok(mgr, "POST", "/api/experiments", {"campaign": "test", "experiment_id": exp})
    for run, lr in (("r0", 0.001), ("r1", 0.01)):
        ok(mgr, "POST", "/api/jobs", {"run_id": run, "experiment_id": exp,
                                      "command": ["echo"], "combo": {"lr": lr}})
    _insert_metrics(mgr, exp, {
        "r0": [(1, {"loss": 1.0}), (2, {"loss": 0.5})],
        "r1": [(1, {"loss": 2.0}), (2, {"loss": 1.5})],
    })

    md = ("```mlsweep-chart\nexperiments: " + exp + "\nmetric: loss\n```\n\n"
          "At step 2 the loss was [0.5](mlsweep:" + exp + "/r0/loss@2).")
    ident = page.eval(
        "(() => { const d = document.createElement('div'); document.body.append(d); "
        "d.id = 'real-out'; renderMarkdown(" + json.dumps(md) + ", 'n.md', d); return d.id; })()")
    page.wait_for(f"document.querySelector('#{ident} .js-plotly-plot') !== null", timeout=20)
    names = out(page, ident, "r.querySelector('.js-plotly-plot').data.map(t => t.name)")
    assert sorted(n for n in names if n) == ["r0", "r1"]
    title = out(page, ident, "r.querySelector('.nb-title').textContent")
    assert "loss" in title and exp in title
    page.wait_for(f"document.querySelector('#{ident} a.metric-ev.ev-ok') !== null")
    assert out(page, ident, "r.querySelector('a.metric-ev').dataset.evValue") == "0.5"


def test_chart_block_renders_in_live_mode(mgr, page) -> None:
    from campaign_helpers import ok, uid
    exp = uid("live")
    ok(mgr, "POST", "/api/experiments", {"campaign": "test", "experiment_id": exp})
    ok(mgr, "POST", "/api/jobs", {"run_id": "r0", "experiment_id": exp,
                                  "command": ["echo"], "combo": {"lr": 0.001}})
    _insert_metrics(mgr, exp, {"r0": [(1, {"loss": 1.0}), (2, {"loss": 0.4})]})
    note = ("# Chart\n\n```mlsweep-chart\nexperiments: " + exp + "\nmetric: loss\n"
            "caption: Loss curve {#fig-live}\n```\n\nThe end was [0.4](mlsweep:" + exp + "/r0/loss@2).\n")
    ok(mgr, "PUT", "/api/notes/file", {"path": "chart-live.md", "content": note, "base_hash": None})
    page.goto(page.url("notes.html", path="chart-live.md"))
    page.wait_for("document.querySelector('.lv-doc .js-plotly-plot') !== null", timeout=20)
    assert page.eval("getComputedStyle(document.querySelector('.group.focused .mode-seg')).display") != "none"
    assert page.eval("document.querySelector('.lv-doc figure.note-fig').id") == "fig-live"
    page.wait_for("document.querySelector('.lv-doc a.metric-ev.ev-ok') !== null")


def test_renders_into_a_detached_container(page) -> None:
    """Live builds a block detached from the document, then appends it."""
    ident = page.eval(
        "(() => { S.files = [{path: 'fig.png'}]; S.fileMap = new Map(); Resolver.rebuild(); "
        "const d = document.createElement('div'); "
        "renderMarkdown('A[^1]\\n\\n![[fig.png]]\\nFigure: cap\\n\\n[^1]: note', 'n.md', d); "
        "const id = 'det-' + Math.random().toString(36).slice(2); d.id = id; "
        "document.body.append(d); return id; })()")
    page.wait_for(f"document.getElementById({json.dumps(ident)}).querySelectorAll('figure.note-fig').length === 1")
    assert out(page, ident, "r.querySelectorAll('.fn-list li').length") == 1
    assert out(page, ident, "r.querySelector('figcaption .fig-kind').textContent") == "Figure 1."


def test_table_run_link_column_renders_as_a_link(page) -> None:
    ident = page.eval(
        "(() => { const d = document.createElement('div'); document.body.append(d); "
        "d.id = 'run-link-table'; d.append(nbTableBox({"
        "  columns: [{key: 'arch', kind: 'param'}, {key: 'ppl', kind: 'metric'}],"
        "  rows: [{run: 'exp_ngram', label: 'ngram', experiment: 'exp', status: 'done',"
        "          params: {arch: 'ngram'}, metrics: {ppl: {value: 9.9, final: 9.9, step: 2}}}],"
        "  run_link_column: 'arch', spec: {agg: 'final'}})); return d.id; })()")
    page.wait_for(f"document.querySelector('#{ident} table.nb-table') !== null")
    assert out(page, ident, "r.querySelector('tbody td a.nb-run').textContent") == "ngram"
    assert out(page, ident, "r.querySelector('tbody td a.nb-run').getAttribute('href')").startswith("results.html?")
    assert "exp/exp_ngram" in out(page, ident, "r.querySelector('tbody td a.nb-run').title")
    assert out(page, ident, "[...r.querySelectorAll('thead th')].map(th => th.firstChild.textContent)") == \
        ["arch", "ppl"]


def test_shell_selection_starts_at_the_line(mgr, page) -> None:
    """The non-markdown editor used to inset each line by 16px, so the selection
    background started 16px right of the line box.  The line starts at the box."""
    notes = mgr.mlsweep_dir / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "selection-test.sh").write_text("#!/bin/sh\necho hello\n")
    page.goto(page.url("notes.html", path="selection-test.sh"))
    page.wait_for("!!document.querySelector('.editor-wrap .CodeMirror')", timeout=20)
    page.eval("const cm = document.querySelector('.CodeMirror').CodeMirror;"
              " cm.focus(); cm.execCommand('selectAll')")
    geom = page.eval(
        "(() => { const line = document.querySelector('.CodeMirror-line');"
        " const l = line.getBoundingClientRect();"
        " const sels = [...document.querySelectorAll('.CodeMirror-selected')]"
        "   .map(s => s.getBoundingClientRect());"
        " return {line: l.left, sels: sels.map(s => s.left),"
        "         pad: getComputedStyle(line).paddingLeft,"
        "         n: document.querySelector('.CodeMirror').CodeMirror.getSelection().length}; })()")
    assert geom["n"] > 0
    assert geom["pad"] == "0px"
    assert geom["sels"] and all(abs(geom["line"] - left) <= 1 for left in geom["sels"])
    assert page.eval("getComputedStyle(document.querySelector('.group.focused .mode-seg')).display") == "none"
    # Even a stale stored mode cannot put a text file into Live or Split.
    page.eval("activeTab().mode = 'split'; renderActive()")
    assert page.eval("document.querySelectorAll('.split-divider').length") == 0
    assert page.eval("document.querySelectorAll('.editor-wrap .CodeMirror').length") == 1


def test_source_heading_hash_matches_the_heading(mgr, page) -> None:
    notes = mgr.mlsweep_dir / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "heading-style.md").write_text("# Heading\n\nText\n")
    page.goto(page.url("notes.html", path="heading-style.md"))
    page.wait_for("!!document.querySelector('.lv-doc')", timeout=20)
    page.eval("setMode('source')")
    page.wait_for("!!document.querySelector('.editor-wrap .CodeMirror-line')", timeout=20)
    styles = page.eval(
        "(() => { const line = [...document.querySelectorAll('.CodeMirror-line')]"
        ".find(l => l.textContent.startsWith('#'));"
        " const hash = line.querySelector('.cm-formatting-header');"
        " const text = line.querySelector('.cm-header:not(.cm-formatting)');"
        " const cs = e => getComputedStyle(e);"
        " return {weight: [cs(hash).fontWeight, cs(text).fontWeight],"
        "         color: [cs(hash).color, cs(text).color]}; })()")
    assert styles["weight"] == ["700", "700"]
    assert styles["color"][0] == styles["color"][1]

    code = page.eval(
        "(() => ({font: getComputedStyle(document.querySelector('.CodeMirror')).fontFamily,"
        "        numbers: document.querySelectorAll('.CodeMirror-linenumber').length}))()")
    assert "mono" in code["font"].lower()
    assert code["numbers"] > 0

    page.eval("setMode('split')")
    page.wait_for("!!document.querySelector('.split-divider')", timeout=20)
    split = page.eval(
        "(() => ({font: getComputedStyle(document.querySelector('.CodeMirror')).fontFamily,"
        "        numbers: document.querySelectorAll('.CodeMirror-linenumber').length,"
        "        preview: !!document.querySelector('.doc')}))()")
    assert "mono" in split["font"].lower()
    assert split["numbers"] > 0 and split["preview"]


def test_sidebar_rows_line_up_with_the_center_panes(mgr, page) -> None:
    notes = mgr.mlsweep_dir / "notes"
    (notes / "align-test.md").write_text("# Align\n\nText\n")
    page.goto(page.url("notes.html", path="align-test.md"))
    page.wait_for("!!document.querySelector('.group.focused .doc-toolbar:not(.hidden)')", timeout=20)
    geom = page.eval(
        "(() => { const r = s => document.querySelector(s).getBoundingClientRect();"
        " const g = '.group.focused ';"
        " return [r('#left-head').bottom, r(g + '.tabstrip').bottom,"
        "         r('#left-tabs').bottom, r(g + '.doc-toolbar').bottom,"
        "         r('#left-body').top, r(g + '.view').top]; })()")
    assert abs(geom[0] - geom[1]) <= 1   # header row / tab strip
    assert abs(geom[2] - geom[3]) <= 1   # pane tabs / doc toolbar
    assert abs(geom[4] - geom[5]) <= 1   # content starts at the same y


def test_markdown_source_highlights_notes_syntax_and_code(mgr, page) -> None:
    notes = mgr.mlsweep_dir / "notes"
    (notes / "highlight-test.md").write_text(
        "---\ntitle: Test\n---\n# Heading\n\n"
        "> [!warning] Careful\n> Body\n\n"
        "Figure: A caption\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "`inline code` and ![[fig.png]] [[Note]] #tag [0.4](mlsweep:e/r/loss@1) ^block\n\n"
        "[^1]: note\n\n"
        "```rust\nfn main() { let x = 1; }\n```\n\n"
        "```go\nfunc main() { x := 1 }\n```\n")
    page.goto(page.url("notes.html", path="highlight-test.md"))
    page.wait_for("!!document.querySelector('.lv-doc')", timeout=20)
    assert page.eval("document.querySelector('.editor-wrap') === null")
    page.eval("setMode('source')")
    page.wait_for("!!document.querySelector('.editor-wrap.md-source .CodeMirror-line')", timeout=20)
    time.sleep(0.4)
    counts = page.eval(
        "(() => { const out = {}; for (const c of ["
        "'cm-frontmatter-key','cm-frontmatter-value','cm-frontmatter-mark',"
        "'cm-callout-warning','cm-caption','cm-footnote-def','cm-wikilink',"
        "'cm-wikilink-embed','cm-tag','cm-blockid','cm-evidence','cm-table-pipe',"
        "'cm-table-header-text','cm-keyword'])"
        "  out[c] = document.querySelectorAll('.' + c).length; return out; })()")
    for name, count in counts.items():
        assert count > 0, name
    assert page.eval("!!CodeMirror.modes.rust && !!CodeMirror.modes.go")
    inline = page.eval(
        "(() => { const e = document.querySelector('.cm-formatting-code + .cm-comment');"
        " if (!e) return null; const s = getComputedStyle(e);"
        " return [s.backgroundColor, s.borderTopWidth, s.paddingLeft, s.color]; })()")
    assert inline is not None
    assert inline[0] in ("rgba(0, 0, 0, 0)", "transparent")
    assert inline[1] == "0px" and inline[2] == "0px"

    embed = page.eval(
        "(() => { const bracket = document.querySelector('.cm-formatting.cm-wikilink-embed');"
        " const body = document.querySelector('.cm-wikilink-embed:not(.cm-formatting)');"
        " if (!bracket || !body) return null;"
        " return [getComputedStyle(bracket).color, getComputedStyle(body).color]; })()")
    assert embed is not None and embed[0] == embed[1]
    assert page.eval(
        "getComputedStyle(document.querySelector('.cm-header-1')).fontSize"
        " === getComputedStyle(document.querySelector('.CodeMirror-line')).fontSize")

    page.eval("setMode('live')")
    page.wait_for("!!document.querySelector('.lv-doc pre code')", timeout=20)
    rendered = page.eval(
        "({langs: [...document.querySelectorAll('.lv-doc pre code')].map(c => c.dataset.lang),"
        "  kw: document.querySelectorAll('.lv-doc pre code .cm-keyword,.lv-doc pre code .cm-builtin').length})")
    assert "rust" in rendered["langs"] and "go" in rendered["langs"]
    assert rendered["kw"] > 0
