// mlsweep Notes: editor groups.  The centre is a tree of splits whose leaves
// are groups.  Each group has its own tab strip, toolbar, banner, view and
// editor.  A tab lives in exactly one group (t.group); S.tabs still lists
// every tab, so saving and sync code need not know about groups.
"use strict";

// node = { kind: "group", id } | { kind: "split", dir: "row" | "col", children: [node], sizes: [number] }
const Groups = new Map();
let groupSeq = 0;
S.layout = null;
S.focus = null;     // the focused group's id
// The focused group's active tab, read by everything that acts on "the open file".
Object.defineProperty(S, "active", { configurable: true, get() { const g = focusedGroup(); return g ? g.active : null; } });

const focusedGroup = () => Groups.get(S.focus) || null;
const groupOf = t => (t && Groups.get(t.group)) || null;
const tabById = id => S.tabs.find(t => t.id === id) || null;
const isShown = t => { const g = groupOf(t); return !!g && g.active === t.id; };
const editorOf = t => { const g = groupOf(t); return g ? g.editor : null; };
function eachEditor(fn) { for (const g of Groups.values()) if (g.editor.cm) fn(g.editor); }
function groupOrder() {
  const out = [];
  const walk = n => n.kind === "group" ? out.push(n.id) : n.children.forEach(walk);
  if (S.layout) walk(S.layout);
  return out;
}

function makeGroup() {
  const root = $("group-tpl").content.firstElementChild.cloneNode(true);
  const q = c => root.querySelector(c);
  const g = {
    id: ++groupSeq, tabs: [], active: null, root,
    strip: q(".tabstrip"), toolbar: q(".doc-toolbar"), crumbs: q(".crumbs"), saveState: q(".save-state"),
    modeSeg: q(".mode-seg"), banner: q(".banner"), view: q(".view"), zone: q(".drop-zone"),
  };
  root.dataset.group = g.id;
  g.editor = makeEditor(g);
  root.addEventListener("mousedown", () => focusGroup(g.id), true);
  root.addEventListener("focusin", () => focusGroup(g.id));
  q(".btn-back").onclick = () => navigate(-1);
  q(".btn-fwd").onclick = () => navigate(1);
  g.modeSeg.querySelectorAll("button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
  q(".btn-more").onclick = e => moreMenu(e);
  q(".btn-right").onclick = () => toggleRight();
  wireGroupDrop(g);
  Groups.set(g.id, g);
  return g;
}

function addTab(t, g, index = g.tabs.length) {
  if (!S.tabs.includes(t)) S.tabs.push(t);
  t.group = g.id;
  g.tabs.splice(Math.max(0, Math.min(index, g.tabs.length)), 0, t.id);
}

function markFocus() {
  for (const g of Groups.values()) g.root.classList.toggle("focused", g.id === S.focus);
  $("groups").classList.toggle("multi", Groups.size > 1);
}
// Clicking into a group makes its tab the open file, without redrawing it.
function focusGroup(id) {
  if (S.focus === id || !Groups.has(id)) return;
  S.focus = id;
  markFocus();
  const t = activeTab();
  refreshRight(); saveTabs();
  if (t && t.path) revealInTree(t.path);
  else document.querySelectorAll("#tree .tnode.active").forEach(n => n.classList.remove("active"));
}

// ── The split tree ─────────────────────────────────────────────────────────────
function findNode(node, match, parent = null, index = -1) {
  if (match(node)) return { node, parent, index };
  if (node.kind !== "split") return null;
  for (let i = 0; i < node.children.length; i++) {
    const r = findNode(node.children[i], match, node, i);
    if (r) return r;
  }
  return null;
}
const findLeaf = id => findNode(S.layout, n => n.kind === "group" && n.id === id);
const firstLeaf = n => n.kind === "group" ? n.id : firstLeaf(n.children[0]);

// Put group *ng* beside group *gid*.  *side* is left, right, up or down.
function splitBeside(gid, side, ng) {
  const dir = side === "left" || side === "right" ? "row" : "col";
  const before = side === "left" || side === "up";
  const leaf = { kind: "group", id: ng.id };
  const at = findLeaf(gid);
  if (at.parent && at.parent.dir === dir) {
    const p = at.parent, half = p.sizes[at.index] / 2, i = at.index + (before ? 0 : 1);
    p.sizes[at.index] = half;
    p.children.splice(i, 0, leaf); p.sizes.splice(i, 0, half);
  } else {
    const split = { kind: "split", dir, children: before ? [leaf, at.node] : [at.node, leaf], sizes: [1, 1] };
    if (at.parent) at.parent.children[at.index] = split; else S.layout = split;
  }
}

// Take a group out of the tree.  The last group always stays.
function removeGroup(gid) {
  const at = findLeaf(gid);
  if (!at || !at.parent) return false;
  const p = at.parent;
  const freed = p.sizes[at.index];
  p.children.splice(at.index, 1); p.sizes.splice(at.index, 1);
  const nb = Math.max(0, at.index - 1);
  p.sizes[nb] += freed;
  const next = firstLeaf(p.children[nb]);
  if (p.children.length === 1) {
    const only = p.children[0];
    const up = findNode(S.layout, n => n === p);
    if (!up.parent) S.layout = only;
    else if (only.kind === "split" && only.dir === up.parent.dir) {
      // A row inside a row is one row: lift its children up, keeping proportions.
      const share = up.parent.sizes[up.index], total = only.sizes.reduce((a, b) => a + b, 0);
      up.parent.children.splice(up.index, 1, ...only.children);
      up.parent.sizes.splice(up.index, 1, ...only.sizes.map(s => s / total * share));
    } else up.parent.children[up.index] = only;
  }
  const g = Groups.get(gid);
  g.editor.release();
  g.root.remove();
  Groups.delete(gid);
  if (S.focus === gid) S.focus = next;
  return true;
}

function renderLayout() {
  const build = node => {
    if (node.kind === "group") return Groups.get(node.id).root;
    const box = el("div", "split " + node.dir);
    node.children.forEach((c, i) => {
      if (i) box.append(sash(node, i - 1));
      const e = build(c);
      e.style.flex = `${node.sizes[i]} 1 0`;
      box.append(e);
    });
    return box;
  };
  const top = build(S.layout);
  top.style.flex = "1 1 0";
  $("groups").replaceChildren(top);
  markFocus();
  requestAnimationFrame(() => eachEditor(ed => ed.cm.refresh()));
}

function sash(node, i) {
  const s = el("div", "sash");
  const horiz = node.dir === "row";
  const sides = () => [s.previousElementSibling, s.nextElementSibling];
  s.addEventListener("mousedown", e => {
    e.preventDefault(); s.classList.add("dragging");
    const [a, b] = sides();
    const px = x => horiz ? x.getBoundingClientRect().width : x.getBoundingClientRect().height;
    const pa = px(a), both = pa + px(b), total = node.sizes[i] + node.sizes[i + 1];
    const start = horiz ? e.clientX : e.clientY, MIN = Math.min(140, both / 2);
    let raf = 0;
    const move = ev => {
      const na = Math.max(MIN, Math.min(both - MIN, pa + (horiz ? ev.clientX : ev.clientY) - start));
      node.sizes[i] = total * na / both; node.sizes[i + 1] = total - node.sizes[i];
      a.style.flexGrow = node.sizes[i]; b.style.flexGrow = node.sizes[i + 1];
      if (!raf) raf = requestAnimationFrame(() => { raf = 0; eachEditor(ed => ed.cm.refresh()); });
    };
    const up = () => {
      s.classList.remove("dragging");
      window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up);
      saveTabs();
    };
    window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
  });
  s.addEventListener("dblclick", () => {
    const [a, b] = sides(), half = (node.sizes[i] + node.sizes[i + 1]) / 2;
    node.sizes[i] = node.sizes[i + 1] = half;
    a.style.flexGrow = b.style.flexGrow = half;
    eachEditor(ed => ed.cm.refresh());
    saveTabs();
  });
  s.title = "Drag to resize; double-click to make both sides equal";
  return s;
}

// ── Moving tabs between groups ─────────────────────────────────────────────────
function moveTab(t, to, index = to.tabs.length) {
  const from = groupOf(t);
  if (isShown(t)) stashScroll(t);
  if (from) {
    const i = from.tabs.indexOf(t.id);
    from.tabs.splice(i, 1);
    if (from === to && i < index) index--;
    if (from.active === t.id) from.active = from.tabs.length ? from.tabs[Math.min(i, from.tabs.length - 1)] : null;
  }
  addTab(t, to, index);
  to.active = t.id;
  S.focus = to.id;
  if (from && from !== to) {
    if (!from.tabs.length && removeGroup(from.id)) renderLayout();
    else renderGroup(from);
  }
  renderTabs(); renderGroup(to); markFocus(); saveTabs();
}

// A new group beside *g*, focused, with nothing in it yet.
function newGroupBeside(g, side) {
  const ng = makeGroup();
  splitBeside(g.id, side, ng);
  renderLayout();
  return ng;
}

// Split right/down from a tab.  A tab is in one group at a time, so it moves;
// the only tab in a group leaves an empty group beside it to open something in.
function splitTab(t, side) {
  const g = groupOf(t);
  if (!g) return;
  const ng = newGroupBeside(g, side);
  if (g.tabs.length > 1) moveTab(t, ng, 0);
  else { S.focus = ng.id; markFocus(); renderTabs(); renderGroup(ng); saveTabs(); refreshRight(); }
}

function closeGroup(g) {
  if (Groups.size < 2) return;
  for (const id of [...g.tabs]) { const t = tabById(id); if (t && t.dirty) saveTab(t); }
  S.tabs = S.tabs.filter(t => t.group !== g.id);
  if (removeGroup(g.id)) { renderLayout(); renderTabs(); renderGroup(focusedGroup()); saveTabs(); }
}

// Open *path* in group *g* (at *index* in its strip), or in a new group on *side* of it.
async function openIn(path, g, { index = null, side = null } = {}) {
  const target = side ? newGroupBeside(g, side) : g;
  S.focus = target.id; markFocus();
  const open = S.tabs.find(x => x.type === "file" && x.path === path);
  if (open) {
    if (groupOf(open) !== target || index != null) moveTab(open, target, index != null ? index : target.tabs.length);
    else activate(open.id);
    return;
  }
  await openFile(path, { newTab: true, index });
}

// The group to the right of the focused one, made if there isn't one.
function sideGroup() {
  const g = focusedGroup();
  const at = findLeaf(g.id);
  if (at.parent && at.parent.dir === "row" && at.index < at.parent.children.length - 1)
    return Groups.get(firstLeaf(at.parent.children[at.index + 1]));
  return newGroupBeside(g, "right");
}
async function openToSide(path) {
  const g = focusedGroup();
  const open = S.tabs.find(x => x.type === "file" && x.path === path);
  if (open && groupOf(open) !== g) { activate(open.id); return; }
  await openIn(path, sideGroup());
}

// ── Drag and drop ──────────────────────────────────────────────────────────────
// A tab dragged onto a strip goes to that position.  Onto a group's body, its
// edges split the group and its middle moves the tab there.  A file dragged
// from the tree does the same, except over a text editor, where dropping it
// inserts a link (see Editor).
let TabDrag = null;      // { id } while a tab is being dragged
let stripsStale = false; // a redraw waits until the drag ends, so the dragged tab stays put

function startTabDrag(e, t, d) {
  TabDrag = { id: t.id };
  e.dataTransfer.effectAllowed = "move";
  e.dataTransfer.setData("application/x-mlsweep-tab", String(t.id));
  d.classList.add("dragging");
}
function endTabDrag() {
  if (!TabDrag && !document.querySelector(".dtab.dragging, .drop-before, .drop-after, .drop-zone.show")) return;
  TabDrag = null;
  document.querySelectorAll(".dtab.dragging").forEach(n => n.classList.remove("dragging"));
  clearDropMarks();
  if (stripsStale) { stripsStale = false; renderTabs(); }
}
document.addEventListener("dragend", endTabDrag);
function clearDropMarks() {
  document.querySelectorAll(".drop-before, .drop-after").forEach(n => n.classList.remove("drop-before", "drop-after"));
  for (const g of Groups.values()) g.zone.classList.remove("show");
}

function stripIndex(g, x) {
  const tabs = [...g.strip.querySelectorAll(".dtab")];
  const i = tabs.findIndex(d => { const r = d.getBoundingClientRect(); return x < r.left + r.width / 2; });
  return i < 0 ? tabs.length : i;
}
function markStrip(g, i) {
  clearDropMarks();
  const tabs = [...g.strip.querySelectorAll(".dtab")];
  if (i < tabs.length) tabs[i].classList.add("drop-before");
  else if (tabs.length) tabs[tabs.length - 1].classList.add("drop-after");
}

const ZONE_EDGE = 0.3;
function bodyRect(g) {
  const r = g.root.getBoundingClientRect(), top = g.strip.getBoundingClientRect().bottom;
  return { left: r.left, top, width: r.width, height: r.bottom - top, offset: top - r.top };
}
function zoneAt(g, e) {
  const r = bodyRect(g);
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const [side, d] = Object.entries({ left: x, right: 1 - x, up: y, down: 1 - y }).sort((a, b) => a[1] - b[1])[0];
  return d < ZONE_EDGE ? side : "center";
}
function showZone(g, zone) {
  clearDropMarks();
  const r = bodyRect(g), z = g.zone.style;
  const half = { left: [0, 0, .5, 1], right: [.5, 0, .5, 1], up: [0, 0, 1, .5], down: [0, .5, 1, .5], center: [0, 0, 1, 1] }[zone];
  z.left = half[0] * 100 + "%"; z.width = half[2] * 100 + "%";
  z.top = r.offset + half[1] * r.height + "px"; z.height = half[3] * r.height + "px";
  g.zone.classList.add("show");
}
// A tree file over a text editor keeps the editor's own drop: insert a link.
const overEditor = e => !!e.target.closest(".CodeMirror, textarea");

function wireGroupDrop(g) {
  const s = g.strip;
  s.addEventListener("dragover", e => {
    if (!TabDrag && !dragPath) return;
    e.preventDefault(); e.stopPropagation();
    e.dataTransfer.dropEffect = "move";
    markStrip(g, stripIndex(g, e.clientX));
  });
  s.addEventListener("dragleave", e => { if (!s.contains(e.relatedTarget)) clearDropMarks(); });
  s.addEventListener("drop", e => {
    if (!TabDrag && !dragPath) return;
    e.preventDefault(); e.stopPropagation();
    const i = stripIndex(g, e.clientX);
    clearDropMarks();
    if (TabDrag) { const t = tabById(TabDrag.id); endTabDrag(); if (t) moveTab(t, g, i); }
    else { const p = dragPath; dragPath = null; openIn(p, g, { index: i }); }
  });

  // Capture, so CodeMirror and the upload handler never see a tab being dropped.
  const body = e => !e.target.closest(".tabstrip") && (TabDrag || (dragPath && !overEditor(e)));
  g.root.addEventListener("dragover", e => {
    if (!body(e)) { if (!e.target.closest(".tabstrip")) g.zone.classList.remove("show"); return; }
    e.preventDefault(); e.stopPropagation();
    e.dataTransfer.dropEffect = "move";
    showZone(g, zoneAt(g, e));
  }, true);
  g.root.addEventListener("dragleave", e => { if (!g.root.contains(e.relatedTarget)) g.zone.classList.remove("show"); });
  g.root.addEventListener("drop", e => {
    if (!body(e)) return;
    e.preventDefault(); e.stopPropagation();
    const zone = zoneAt(g, e);
    clearDropMarks();
    if (TabDrag) { const t = tabById(TabDrag.id); endTabDrag(); if (t) dropTab(t, g, zone); }
    else { const p = dragPath; dragPath = null; openIn(p, g, { side: zone === "center" ? null : zone }); }
  }, true);
}

function dropTab(t, g, zone) {
  const from = groupOf(t);
  if (zone === "center") { if (from !== g) moveTab(t, g); else activate(t.id); return; }
  if (from === g && g.tabs.length === 1) return;   // splitting a group off its only tab changes nothing
  moveTab(t, newGroupBeside(g, zone), 0);
}

// ── Saving and restoring the layout ────────────────────────────────────────────
const PERSISTED = new Set(["file", "graph"]);
function serializeLayout() {
  const ser = n => {
    if (n.kind === "split") return { kind: "split", dir: n.dir, sizes: n.sizes, children: n.children.map(ser) };
    const g = Groups.get(n.id);
    const tabs = g.tabs.map(tabById).filter(t => t && PERSISTED.has(t.type));
    return { kind: "group", tabs: tabs.map(t => ({ type: t.type, path: t.path, mode: t.mode })), active: tabs.findIndex(t => t.id === g.active) };
  };
  return { tree: ser(S.layout), focus: groupOrder().indexOf(S.focus) };
}

// One empty group, so there is somewhere to draw before the layout is restored.
function initLayout() {
  const g = makeGroup();
  S.layout = { kind: "group", id: g.id };
  S.focus = g.id;
  renderLayout();
}

async function restoreLayout() {
  let saved = store.get("layout", null);
  if (!saved || !saved.tree) {
    // Before groups: one strip of tabs.
    const tabs = store.get("tabs", []), act = store.get("activeTab", null);
    saved = { tree: { kind: "group", tabs, active: act ? tabs.findIndex(s => s.type === act.type && s.path === act.path) : 0 }, focus: 0 };
  }
  for (const g of Groups.values()) g.root.remove();
  Groups.clear();
  S.tabs = [];
  const wantActive = new Map();
  const build = n => {
    if (n && n.kind === "split" && Array.isArray(n.children) && n.children.length) {
      const kids = n.children.map((c, i) => ({ node: build(c), size: Number((n.sizes || [])[i]) > 0 ? Number(n.sizes[i]) : 1 }));
      if (kids.length === 1) return kids[0].node;
      return { kind: "split", dir: n.dir === "col" ? "col" : "row", children: kids.map(k => k.node), sizes: kids.map(k => k.size) };
    }
    const g = makeGroup();
    let graph = !!S.tabs.find(t => t.type === "graph");
    for (const st of (n && n.tabs) || []) {
      if (st.type === "graph") { if (!graph) { addTab(newTab({ type: "graph", title: "Graph" }), g); graph = true; } }
      else if (st.path && S.fileMap.has(st.path) && !S.tabs.some(t => t.path === st.path && t.type === "file")) {
        const mode = { read: "live", edit: "source" }[st.mode] || st.mode || "live";
        addTab(newTab({ path: st.path, title: titleFor(st.path), kind: kindOf(st.path), mode }), g);
      }
    }
    wantActive.set(g.id, g.tabs[(n && n.active) | 0] || g.tabs[0] || null);
    return { kind: "group", id: g.id };
  };
  S.layout = build(saved.tree);
  await Promise.all(S.tabs.filter(t => t.type === "file").map(t => loadTabContent(t).catch(() => { t.missing = true; })));
  for (const t of S.tabs.filter(t => t.missing)) { const g = groupOf(t); g.tabs = g.tabs.filter(id => id !== t.id); }
  S.tabs = S.tabs.filter(t => !t.missing);
  for (const g of Groups.values()) {
    const want = wantActive.get(g.id);
    g.active = g.tabs.includes(want) ? want : (g.tabs[0] || null);
  }
  const order = groupOrder();
  S.focus = order[saved.focus] || order[0];
  for (const id of order) if (!Groups.get(id).tabs.length) removeGroup(id);
  renderLayout();
  renderAll();
}

function renderAll() {
  renderTabs();
  for (const g of Groups.values()) renderGroup(g, { keepEditor: true });
}
