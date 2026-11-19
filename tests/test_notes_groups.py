"""Editor groups in the Notes tab: splitting, dragging tabs, resizing, and the
saved layout, in headless Chromium.

Drags are dispatched as DragEvents with a real DataTransfer at real
coordinates, so they go through the same handlers a mouse drag does.

Skipped when no Chromium-family browser is installed.
"""

import json

import pytest

from browser import Browser, find_chromium
from campaign_helpers import shared_manager, static_url

CHROME = find_chromium()
pytestmark = pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")

FILES = {"ga.md": "# A\n\nalpha\n", "gb.md": "# B\n\nbeta\n", "gc.md": "# C\n\ngamma\n", "gd.txt": "plain\n",
         "ge.md": "# E\n\nepsilon\n", "gf.txt": "another plain\n"}

# drag(srcSelector, targetSelector, fx, fy): drag the source onto the target at a
# fraction of its box (0..1 in each direction), as the browser would.
DRAG_JS = """
window.__drag = (src, dst, fx, fy) => {
  const s = typeof src === "string" ? document.querySelector(src) : src;
  const d = typeof dst === "string" ? document.querySelector(dst) : dst;
  const r = d.getBoundingClientRect();
  const x = r.left + r.width * fx, y = r.top + r.height * fy;
  const dt = new DataTransfer();
  const ev = (type, target) => target.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, composed: true, clientX: x, clientY: y, dataTransfer: dt }));
  ev("dragstart", s);
  const at = document.elementFromPoint(x, y) || d;
  ev("dragenter", at); ev("dragover", at); ev("drop", at);
  ev("dragend", s);
};
"""


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "notes_groups")


@pytest.fixture(scope="module")
def browser(mgr):
    notes = mgr.mlsweep_dir / "notes"
    for name, text in FILES.items():
        (notes / name).write_text(text)
    b = Browser(CHROME)
    yield b
    b.close()


@pytest.fixture
def page(mgr, browser):
    """The Notes tab with ga, gb and gc open in one group, and nothing saved before."""
    browser.goto(static_url(mgr, "notes.html"))
    browser.eval("localStorage.clear()")
    browser.goto(static_url(mgr, "notes.html"))
    browser.wait_for("S.ready && S.fileMap.has('gc.md') && Groups.size === 1", timeout=20)
    browser.eval(DRAG_JS)
    for name in ("ga.md", "gb.md", "gc.md"):
        browser.eval(f"openFile({json.dumps(name)}, {{ newTab: true }})")
        browser.wait_for(f"activeTab() && activeTab().path === {json.dumps(name)} && !!activeTab().meta")
    return browser


def strips(page):
    """Each group's tab titles, in layout order."""
    return page.eval("groupOrder().map(id => [...Groups.get(id).strip.querySelectorAll('.dtab .tt')].map(e => e.textContent))")


def tab_sel(title):
    return f"[...document.querySelectorAll('.dtab')].find(d => d.querySelector('.tt').textContent === {json.dumps(title)})"


def test_file_tree_always_shows_the_extension(page) -> None:
    name = lambda path: f"document.querySelector('#tree .tnode.file[data-path={json.dumps(path)}] .tname').textContent"
    assert page.eval(name("ga.md")) == "ga.md"
    assert page.eval(name("gd.txt")) == "gd.txt"


def double_click(page, path) -> None:  # type: ignore[no-untyped-def]
    page.eval(f"""(() => {{
      const row = document.querySelector('#tree .tnode.file[data-path={json.dumps(path)}]');
      for (const detail of [1, 2]) row.dispatchEvent(new MouseEvent('click', {{ bubbles: true, detail }}));
      row.dispatchEvent(new MouseEvent('dblclick', {{ bubbles: true, detail: 2 }}));
    }})()""")


def test_double_click_opens_a_new_tab_for_the_same_or_another_type(page) -> None:
    page.eval("activate(S.tabs.find(t => t.path === 'ga.md').id)")
    double_click(page, "gf.txt")
    page.wait_for("activeTab().path === 'gf.txt'")
    assert strips(page) == [["ga", "gf.txt", "gb", "gc"]]
    page.eval("activate(S.tabs.find(t => t.path === 'ga.md').id)")
    double_click(page, "ge.md")
    page.wait_for("activeTab().path === 'ge.md'")
    assert strips(page) == [["ga", "ge", "gf.txt", "gb", "gc"]]


def test_tree_click_selects_and_double_click_opens_a_new_tab(page) -> None:
    page.eval("activate(S.tabs.find(t => t.path === 'ga.md').id)")
    row = "document.querySelector('#tree .tnode.file[data-path=\"gd.txt\"]')"
    page.eval(f"{row}.click()")
    page.wait_for(f"{row}.classList.contains('selected')")
    assert page.eval("activeTab().path") == "ga.md"
    assert strips(page) == [["ga", "gb", "gc"]]
    page.eval(f"{row}.dispatchEvent(new MouseEvent('dblclick', {{ bubbles: true }}))")
    page.wait_for("activeTab().path === 'gd.txt'")
    assert strips(page) == [["ga", "gd.txt", "gb", "gc"]]


def test_dragging_a_tab_to_the_right_edge_splits(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    assert strips(page) == [["ga", "gb"], ["gc"]]
    assert page.eval("S.layout.kind === 'split' && S.layout.dir === 'row'")
    # Side by side, the new group on the right, focused, with its own toolbar.
    boxes = page.eval("groupOrder().map(id => { const r = Groups.get(id).root.getBoundingClientRect(); return [r.left, r.top, r.width]; })")
    assert boxes[0][0] < boxes[1][0] and boxes[0][1] == boxes[1][1]
    assert abs(boxes[0][2] - boxes[1][2]) < 4
    assert page.eval("activeTab().path") == "gc.md"
    assert page.eval("document.querySelectorAll('.group .doc-toolbar:not(.hidden)').length") == 2


def test_dragging_to_the_bottom_edge_splits_down(page) -> None:
    page.eval(f"__drag({tab_sel('ga')}, '.group .view', 0.5, 0.95)")
    assert strips(page) == [["gb", "gc"], ["ga"]]
    assert page.eval("S.layout.dir") == "col"


def test_reordering_within_a_strip(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, {tab_sel('ga')}, 0.1, 0.5)")
    assert strips(page) == [["gc", "ga", "gb"]]
    page.eval(f"__drag({tab_sel('gc')}, {tab_sel('gb')}, 0.9, 0.5)")
    assert strips(page) == [["ga", "gb", "gc"]]


def test_moving_the_last_tab_out_closes_its_group(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    assert len(strips(page)) == 2
    # Back onto the first group's strip: the right group is now empty, so it goes.
    page.eval(f"__drag({tab_sel('gc')}, {tab_sel('ga')}, 0.1, 0.5)")
    assert strips(page) == [["gc", "ga", "gb"]]
    assert page.eval("S.layout.kind") == "group"


def test_dropping_on_the_middle_of_another_group_moves_the_tab(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    page.eval(f"__drag({tab_sel('ga')}, Groups.get(groupOrder()[1]).view, 0.5, 0.5)")
    assert strips(page) == [["gb"], ["gc", "ga"]]


def test_a_lone_tab_dropped_on_its_own_edge_stays(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    right = "Groups.get(groupOrder()[1]).view"
    page.eval(f"__drag({tab_sel('gc')}, {right}, 0.95, 0.5)")
    assert strips(page) == [["ga", "gb"], ["gc"]]


def test_nested_splits_and_collapse(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    right = "Groups.get(groupOrder()[1]).view"
    page.eval(f"__drag({tab_sel('gb')}, {right}, 0.5, 0.95)")
    assert strips(page) == [["ga"], ["gc"], ["gb"]]
    assert page.eval("S.layout.dir === 'row' && S.layout.children[1].kind === 'split' && S.layout.children[1].dir === 'col'")
    # Closing the bottom-right group's tab collapses the column back into the row.
    page.eval("closeTab(S.tabs.find(t => t.path === 'gb.md').id)")
    page.wait_for("Groups.size === 2")
    assert page.eval("S.layout.dir === 'row' && S.layout.children.every(c => c.kind === 'group')")


def test_both_groups_edit_in_source_at_once(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    page.eval("setMode('source')")
    page.eval("activate(S.tabs.find(t => t.path === 'gb.md').id); setMode('source')")
    page.wait_for("document.querySelectorAll('.editor-wrap .CodeMirror').length === 2")
    texts = page.eval("[...document.querySelectorAll('.editor-wrap .CodeMirror')].map(c => c.CodeMirror.getValue())")
    assert sorted(texts) == sorted([FILES["gb.md"], FILES["gc.md"]])
    # Moving a tab whose doc is in one editor into the other group's editor must not fail.
    page.eval(f"__drag({tab_sel('gc')}, {tab_sel('gb')}, 0.9, 0.5)")
    assert strips(page) == [["ga", "gb", "gc"]]
    assert page.eval("activeTab().path") == "gc.md"
    page.wait_for("document.querySelector('.editor-wrap .CodeMirror') && document.querySelector('.editor-wrap .CodeMirror').CodeMirror.getValue() === %s"
                  % json.dumps(FILES["gc.md"]))


def test_an_edit_in_the_unfocused_group_saves(mgr, page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    page.eval("setMode('source')")
    page.eval("activate(S.tabs.find(t => t.path === 'gb.md').id)")
    assert page.eval("activeTab().path") == "gb.md"
    page.eval("S.tabs.find(t => t.path === 'gc.md').doc.replaceRange('more\\n', { line: 3, ch: 0 })")
    page.wait_for("!S.tabs.find(t => t.path === 'gc.md').dirty", timeout=10)
    assert (mgr.mlsweep_dir / "notes" / "gc.md").read_text() == FILES["gc.md"] + "more\n"
    (mgr.mlsweep_dir / "notes" / "gc.md").write_text(FILES["gc.md"])


def test_layout_and_sizes_survive_a_reload(mgr, page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    # Drag the sash 200px to the right.
    page.eval("""(() => {
      const s = document.querySelector('.sash'), r = s.getBoundingClientRect();
      const o = { bubbles: true, clientX: r.left, clientY: r.top + 50 };
      s.dispatchEvent(new MouseEvent('mousedown', o));
      window.dispatchEvent(new MouseEvent('mousemove', { ...o, clientX: r.left + 200 }));
      window.dispatchEvent(new MouseEvent('mouseup', { ...o, clientX: r.left + 200 }));
    })()""")
    widths = page.eval("groupOrder().map(id => Groups.get(id).root.getBoundingClientRect().width)")
    assert widths[0] - widths[1] > 350
    page.goto(static_url(mgr, "notes.html"))
    page.wait_for("Groups.size === 2 && S.tabs.length === 3 && S.tabs.every(t => t.meta)", timeout=20)
    assert strips(page) == [["ga", "gb"], ["gc"]]
    again = page.eval("groupOrder().map(id => Groups.get(id).root.getBoundingClientRect().width)")
    assert abs(again[0] - widths[0]) < 3
    assert page.eval("activeTab().path") == "gc.md"


def test_old_saved_tabs_become_one_group(mgr, page) -> None:
    page.eval("localStorage.clear(); localStorage.setItem('notes.tabs', JSON.stringify([{type:'file',path:'ga.md',mode:'live'},{type:'file',path:'gd.txt',mode:'source'}]));"
              " localStorage.setItem('notes.activeTab', JSON.stringify({type:'file',path:'gd.txt'}))")
    page.goto(static_url(mgr, "notes.html"))
    page.wait_for("S.tabs.length === 2 && S.tabs.every(t => t.meta)", timeout=20)
    assert strips(page) == [["ga", "gd.txt"]]
    assert page.eval("activeTab().path") == "gd.txt"


def test_split_keys_and_closing_an_empty_group(page) -> None:
    page.eval("document.body.dispatchEvent(new KeyboardEvent('keydown', { key: '\\\\', bubbles: true }))")
    assert strips(page) == [["ga", "gb"], ["gc"]]
    # The only tab: splitting leaves an empty group to open something in.
    page.eval("document.body.dispatchEvent(new KeyboardEvent('keydown', { key: '|', bubbles: true }))")
    assert strips(page) == [["ga", "gb"], ["gc"], []]
    assert page.eval("activeTab()") is None
    assert page.eval("!!focusedGroup().strip.querySelector('.group-close')")
    page.eval("openFile('gd.txt')")
    page.wait_for("activeTab() && activeTab().path === 'gd.txt'")
    assert strips(page) == [["ga", "gb"], ["gc"], ["gd.txt"]]
    page.eval("closeTab(activeTab().id)")
    page.wait_for("Groups.size === 2")
    assert strips(page) == [["ga", "gb"], ["gc"]]


def test_tree_file_dropped_on_a_strip_opens_there(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    page.eval("revealInTree('gd.txt')")
    row = "document.querySelector(\"#tree .tnode[data-path='gd.txt']\")"
    left_tab = tab_sel("ga")
    page.eval(f"__drag({row}, {left_tab}, 0.9, 0.5)")
    page.wait_for("activeTab() && activeTab().path === 'gd.txt'")
    assert strips(page) == [["ga", "gd.txt", "gb"], ["gc"]]


def test_opening_a_file_open_in_another_group_focuses_it(page) -> None:
    page.eval(f"__drag({tab_sel('gc')}, '.group .view', 0.95, 0.5)")
    page.eval("activate(S.tabs.find(t => t.path === 'ga.md').id)")
    page.eval("openFile('gc.md')")
    page.wait_for("activeTab().path === 'gc.md'")
    assert strips(page) == [["ga", "gb"], ["gc"]]
    assert page.eval("focusedGroup() === Groups.get(groupOrder()[1])")


def test_a_file_opened_before_the_layout_is_restored_is_kept(page) -> None:
    # As if gc had been clicked while the page was still starting: the saved layout lacks it.
    page.eval("localStorage.setItem('notes.layout', JSON.stringify({ tree: { kind: 'group', tabs: [{ type: 'file', path: 'ga.md', mode: 'live' }], active: 0 }, focus: 0 }))")
    page.eval("activate(S.tabs.find(t => t.path === 'gc.md').id)")
    page.eval("restoreLayout()")
    page.wait_for("S.tabs.length === 3 && S.tabs.every(t => t.meta)")
    assert strips(page) == [["ga", "gb", "gc"]]
    assert page.eval("activeTab().path") == "gc.md"
