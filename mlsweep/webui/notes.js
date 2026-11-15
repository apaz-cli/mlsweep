// mlsweep Notes: an Obsidian-style editor over the manager's notes clone.
//
// Everything the server sends that came from a file is untrusted: rendered
// markdown goes through DOMPurify, names are set with textContent, and raw
// files are fetched from /api/notes/raw, which never serves active content.
"use strict";

// ── API ─────────────────────────────────────────────────────────────
const _qp = new URLSearchParams(location.search);

async function api(path, opts = {}) {
  const r = await apiFetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  if (!r.ok) { const e = new Error((body && body.error) || `HTTP ${r.status}`); e.status = r.status; e.body = body; throw e; }
  return body;
}
function apiJSON(path, method, data) {
  return api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}
const q = encodeURIComponent;
const TREE_ORDER = new Intl.Collator(undefined, { sensitivity: "base", numeric: true });
function rawURL(path, extra = "") { return authURL(`/api/notes/raw?path=${q(path)}${extra}`); }


document.addEventListener("mlthemechange", () => { if (typeof Graph !== "undefined") Graph.redraw(); });

// ── Small utilities ────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const esc = escHtml;  // from dialog.js
const basename = p => p.split("/").pop();
const dirname = p => { const i = p.lastIndexOf("/"); return i < 0 ? "" : p.slice(0, i); };
const joinPath = (d, n) => d ? d + "/" + n : n;
const extOf = p => { const n = basename(p); const i = n.lastIndexOf("."); return i <= 0 ? "" : n.slice(i + 1).toLowerCase(); };
const stemOf = p => { const n = basename(p); const i = n.lastIndexOf("."); return i <= 0 ? n : n.slice(0, i); };
const isMd = p => /\.(md|markdown)$/i.test(p);
// The display name for an experiment; the full ID stays in titles and links.
const shortExp = id => (S.experiments.get(id) || {}).name || id;
const HEADING_RE = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const TASK_RE = /^(\s*(?:[-*+]|\d+[.)])\s+\[)( |x|X)(\])/;
const LV_FENCE = /^\s{0,3}(`{3,}|~{3,})/;

// Fence state after `line`: an open ``` / ~~~ marker, "$$" inside display math, or null.
function lvNextFence(fence, line) {
  if (fence === "$$") return line.includes("$$") ? null : fence;
  if (fence) return line.trim().startsWith(fence) && line.trim().replace(/[`~]/g, "") === "" ? null : fence;
  const m = LV_FENCE.exec(line);
  if (m) return m[1];
  if (/^\s*\$\$/.test(line) && (line.match(/\$\$/g) || []).length === 1) return "$$";
  return null;
}
// Calls fn(line, i) for each of `lines` outside code and display math.
function forEachProseLine(lines, fn) {
  let fence = null;
  for (let i = 0; i < lines.length; i++) {
    const was = fence;
    fence = lvNextFence(fence, lines[i]);
    if (!was && !fence && fn(lines[i], i) === false) return;
  }
}
// Where the box of the n-th task in `lines` is: { line, ch }, or null.
function findTask(lines, want) {
  let n = 0, found = null;
  forEachProseLine(lines, (text, i) => {
    const m = TASK_RE.exec(text);
    if (m && n++ === want) { found = { line: i, ch: m[1].length }; return false; }
  });
  return found;
}
// What to write inside [[ ]] to link to p from the note src: the bare name if that finds it.
const wikiName = (p, src) => isMd(p) ? (Resolver.wiki(src, stemOf(p)) === p ? stemOf(p) : p.replace(/\.md$/i, "")) : basename(p);
// A link to a note, or an embed of any other file.
const embedText = p => isMd(p) ? `[[${stemOf(p)}]]` : `![[${basename(p)}]]`;
const attachDir = t => t && t.path ? dirname(t.path) : "";
const treeRow = path => [...document.querySelectorAll("#tree .tnode.file")].find(n => n.dataset.path === path);
function downloadFile(path) { const a = el("a"); a.href = rawURL(path, "&download=1"); a.download = basename(path); a.click(); }
// Front matter lines as a key/value block.
function propsEl(lines) {
  const props = el("div", "props");
  for (const line of lines) {
    const m = /^([^:#]+):\s*(.*)$/.exec(line);
    if (m) props.append(el("span", "k", m[1].trim()), el("span", "v", m[2].trim()));
  }
  return props;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem("notes." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("notes." + k, JSON.stringify(v)); } catch {} },
};
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function ago(ts) {
  const d = Date.now() / 1000 - ts;
  if (d < 45) return "just now";
  if (d < 90) return "1m ago";
  if (d < 3600) return Math.round(d / 60) + "m ago";
  if (d < 86400) return Math.round(d / 3600) + "h ago";
  if (d < 7 * 86400) return Math.round(d / 86400) + "d ago";
  return new Date(ts * 1000).toLocaleDateString();
}
function dayLabel(ts) {
  const d = new Date(ts * 1000), now = new Date();
  const k = x => x.toDateString();
  if (k(d) === k(now)) return "Today";
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (k(d) === k(y)) return "Yesterday";
  return d.toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
}
const AVATAR_COLORS = ["#4c9be8", "#e07b00", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#1a56c4"];
function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function avatar(name) {
  const a = el("span", "avatar", (name || "?").replace(/[^A-Za-z0-9]/g, "").slice(0, 1) || "?");
  a.style.background = AVATAR_COLORS[hashStr(name || "") % AVATAR_COLORS.length];
  a.title = name;
  return a;
}
function toast(msg, kind = "") {
  const t = el("div", "toast " + kind, msg);
  $("toasts").appendChild(t);
  setTimeout(() => { t.style.transition = "opacity .3s"; t.style.opacity = "0"; setTimeout(() => t.remove(), 320); }, kind === "err" ? 6000 : 3200);
}

const ICONS = {
  md: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="M5.5 8h5M5.5 10.5h5M5.5 5.5h2.5"/></svg>',
  image: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><rect x="1.5" y="2.5" width="13" height="11" rx="1.5"/><circle cx="5.5" cy="6" r="1.3"/><path d="m2 12 4-4 3 3 2-2 3.5 3.5"/></svg>',
  pdf: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="M5.3 11.5V8.2h1.1a1 1 0 0 1 0 2H5.3M8.6 11.5V8.2h.8c.9 0 1.3.7 1.3 1.65s-.4 1.65-1.3 1.65z"/></svg>',
  text: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="m6.5 7.5-1.5 1.5 1.5 1.5M9.5 7.5l1.5 1.5-1.5 1.5"/></svg>',
  binary: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="M9.5 1.5v3h3"/></svg>',
  dir: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M1.5 4.5v8a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-6a1 1 0 0 0-1-1H8L6.5 3.5h-4a1 1 0 0 0-1 1z"/></svg>',
  graph: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="4" cy="4" r="2"/><circle cx="12" cy="5" r="2"/><circle cx="7" cy="12.5" r="2"/><path d="M5.8 4.3 10 4.8M5 5.8l1.4 4.9M10.8 6.6 8.3 11"/></svg>',
  diff: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 2v8M1 6h6M9 12h6"/><circle cx="12" cy="4" r="2"/></svg>',
  term: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="1.5" y="2.5" width="13" height="11" rx="1.5"/><path d="m4.5 6 2 2-2 2M8.5 10.5h3"/></svg>',
};
// The tree's own `kind` (from the server's _notes.kind_of) when the file is
// known; the extension rules below only cover paths not in the tree yet.
function kindOf(path) {
  const known = S.fileMap.get(path);
  if (known) return known.kind;
  const e = extOf(path);
  if (e === "md" || e === "markdown") return "md";
  if (/^(png|jpe?g|gif|webp|avif|bmp|ico|svg)$/.test(e)) return "image";
  if (e === "pdf") return "pdf";
  if (/^(txt|log|json|jsonl|ya?ml|toml|ini|cfg|conf|csv|tsv|py|sh|bash|zsh|js|ts|css|html?|xml|tex|bib|rst|c|h|cc|cpp|hpp|cu|rs|go|java|lua|sql|r|jl|ipynb|diff|patch|env|mk|cmake|gitignore|gitattributes)$/.test(e)) return "text";
  if (/^(Makefile|Dockerfile|LICENSE|README|TODO|\.gitignore|\.gitattributes)$/.test(basename(path))) return "text";
  return "binary";
}
const iconFor = path => ICONS[kindOf(path)] || ICONS.binary;

// ── State ──────────────────────────────────────────────────────────────────────
const S = {
  enabled: null,          // null = unknown, false = notes not configured
  status: null,
  version: "",
  files: [],              // [{path,size,mtime,kind}]
  fileMap: new Map(),
  dirs: [],
  index: null,            // {links, mentions, experiments}
  indexVersion: "",
  experiments: new Map(), // id -> {status, campaign, name}
  tabs: [],               // see newTab(); each is in one group (notes-layout.js)
  openDirs: new Set(store.get("openDirs", [])),
  recent: store.get("recent", []),
  navBack: [], navFwd: [],
  seenMtime: store.get("seenMtime", Date.now() / 1000),
  lint: null,
  ownSaves: new Map(),    // path -> mtime of our own last save (not "fresh")
  protectedDirs: new Set(),
};


// ── Link resolution (mirrors _notes.Resolver; tests/test_notes_parity.py holds them together) ──
const Resolver = {
  byName: new Map(), lower: new Map(),
  rebuild() {
    this.byName = new Map(); this.lower = new Map();
    for (const path of S.files.map(f => f.path).sort()) {
      this.lower.set(path.toLowerCase(), path);
      const add = k => { const a = this.byName.get(k) || []; a.push(path); this.byName.set(k, a); };
      add(basename(path).toLowerCase());
      if (isMd(path)) add(stemOf(path).toLowerCase());
    }
  },
  closest(src, cands) {
    const sd = dirname(src || "");
    const same = cands.filter(c => dirname(c) === sd);
    if (same.length) return same[0];
    const byCode = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
    return [...cands].sort((a, b) => (a.split("/").length - b.split("/").length) || a.length - b.length || byCode(a, b))[0];
  },
  wiki(src, target) {
    target = target.trim().replace(/^\/+/, "");
    if (!target) return null;
    const cands = /\.(md|markdown)$/i.test(target) ? [target] : [target, target + ".md"];
    let name = target;
    if (target.includes("/")) {
      for (const c of cands) { const hit = this.lower.get(c.toLowerCase()); if (hit) return hit; }
      name = basename(target);
    }
    const hits = this.byName.get(name.toLowerCase());
    return hits ? this.closest(src, hits) : null;
  },
  relative(src, href) {
    if (href.startsWith("#") || href.startsWith("//") || /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(href)) return null;
    let p = href.split("#")[0].split("?")[0];
    try { p = decodeURIComponent(p); } catch {}
    if (!p) return null;
    const parts = p.startsWith("/") ? [] : dirname(src || "").split("/").filter(Boolean);
    for (const seg of p.split("/")) {
      if (seg === "" || seg === ".") continue;
      if (seg === "..") { if (!parts.length) return null; parts.pop(); } else parts.push(seg);
    }
    const j = parts.join("/");
    return this.lower.get(j.toLowerCase()) || (extOf(j) ? null : this.lower.get((j + ".md").toLowerCase())) ||
      this.lower.get(p.replace(/^\/+/, "").toLowerCase()) || null;
  },
};

// ── Status & sync ──────────────────────────────────────────────────────────────
let lastCommitReq = 0;
function renderSyncState() {
  const st = S.status;
  const dots = [$("nav-sync-dot"), $("foot-dot")];
  let cls = "off", nav = "Notes off", foot = "Notes are not enabled";
  if (st && st.enabled) {
    if (st.last_error) { cls = "err"; nav = "Git error"; foot = st.last_error; }
    else { cls = ""; nav = "Notes"; foot = `${st.branch} · ${st.head}`; }
  }
  for (const d of dots) d.className = "sync-dot " + cls;
  $("nav-sync-text").textContent = nav;
  $("foot-text").textContent = foot;
  $("foot-text").title = st && st.last_error ? st.last_error : (st && st.enabled ? `${st.root}
clone: mlsweep notes clone DIR
remote: ${st.remote}
head: ${st.head}` : "");
  if (st && st.enabled) {
    $("vault-name").innerHTML = "";
    $("vault-name").append(document.createTextNode(st.name), el("span", "branch", st.branch));
    $("vault-name").title = st.root;
  }
}
async function commitNow() {
  if (Date.now() - lastCommitReq < 1500) return;
  lastCommitReq = Date.now();
  await flushSaves();
  $("btn-sync").disabled = true;
  try {
    const r = await api("/api/notes/commit", { method: "POST" });
    S.status = r.status; renderSyncState();
    if (r.error) toast("Commit failed: " + r.error, "err");
    else toast(r.committed ? `Committed ${r.committed} file(s)` : "Nothing to commit", "ok");
    await poll(true);
  } catch (e) { toast("Commit failed: " + e.message, "err"); }
  finally { $("btn-sync").disabled = false; }
}
$("btn-sync").onclick = commitNow;
$("nav-sync").onclick = commitNow;

let polling = false;
async function poll(force = false) {
  if (polling) return;
  polling = true;
  try {
    const st = await api("/api/notes/status");
    const wasEnabled = S.enabled;
    S.enabled = !!st.enabled;
    S.status = st;
    renderSyncState();
    if (!S.enabled) { if (wasEnabled !== false) showSetup(); return; }
    if (force || st.version !== S.version) {
      const first = !S.version;
      S.version = st.version;
      await loadTree();
      if (!first) await checkOpenTabs();
      S.index = null;
      // Views drawn from the folder (graph, start page) catch up.
      for (const g of Groups.values()) { const t = tabById(g.active); if (!t || !tabType(t).ownRefresh) renderGroup(g); }
      refreshRight();
    }
  } catch (e) {
    if (e.status === 401) { $("foot-text").textContent = "Unauthorized: open the dashboard with ?token=…"; }
    else { $("nav-sync-dot").className = "sync-dot err"; $("nav-sync-text").textContent = "Manager unreachable"; }
  } finally { polling = false; }
}

async function loadExperiments() {
  try {
    const exps = await api("/api/experiments");
    S.experiments = new Map(exps.map(e => [e.experiment_id, { status: e.status, campaign: e.campaign, name: e.name, jobs: e.job_counts }]));
  } catch { /* keep the old list */ }
}

async function ensureIndex() {
  if (S.index && S.indexVersion === S.version) return S.index;
  try {
    S.index = await api("/api/notes/index");
    S.indexVersion = S.version;
    for (const [id, m] of Object.entries(S.index.experiments || {})) if (!S.experiments.has(id)) S.experiments.set(id, m);
  } catch { S.index = S.index || { links: {}, mentions: {} }; }
  return S.index;
}

// ── File tree ──────────────────────────────────────────────────────────────────
// A click that would replace the current tab waits briefly: if a second click
// turns it into a double-click, the tab is kept and the file opens in a new one.
let treeClick = null;
const TREE_CLICK_MS = 300;
function runTreeClick() {
  const c = treeClick;
  if (!c) return;
  treeClick = null;
  openFile(c.path);
}
function laterTreeClick(path) {
  if (treeClick && treeClick.path === path) clearTimeout(treeClick.timer);
  else runTreeClick();
  treeClick = { path, timer: setTimeout(runTreeClick, TREE_CLICK_MS) };
}
function cancelTreeClick() {
  if (!treeClick) return;
  clearTimeout(treeClick.timer);
  treeClick = null;
}
async function loadTree() {
  const t = await api("/api/notes/tree");
  S.files = t.files; S.dirs = t.dirs; S.protectedDirs = new Set(t.protected_dirs); S.version = t.version || S.version;
  S.fileMap = new Map(S.files.map(f => [f.path, f]));
  Resolver.rebuild();
  renderTree();
}

function buildTree() {
  const root = { name: "", path: "", dirs: new Map(), files: [] };
  const ensure = path => {
    let node = root;
    if (!path) return node;
    let acc = "";
    for (const part of path.split("/")) {
      acc = acc ? acc + "/" + part : part;
      if (!node.dirs.has(part)) node.dirs.set(part, { name: part, path: acc, dirs: new Map(), files: [] });
      node = node.dirs.get(part);
    }
    return node;
  };
  for (const d of S.dirs) ensure(d);
  for (const f of S.files) ensure(dirname(f.path)).files.push(f);
  return root;
}

function renderTree() {
  const tree = $("tree");
  const scroll = tree.parentElement.scrollTop;
  tree.innerHTML = "";
  if (!S.files.length && !S.dirs.length) {
    tree.innerHTML = `<div class="tree-empty">This folder is empty.<br>Create a note with the icon above, or drop files here.</div>`;
    return;
  }
  const activePath = activeTab() && activeTab().path;
  const fresh = Date.now() / 1000 - 15 * 60;
  const walk = (node, depth, container) => {
    const dirs = [...node.dirs.values()].sort((a, b) => TREE_ORDER.compare(a.name, b.name));
    const files = node.files.sort((a, b) => TREE_ORDER.compare(basename(a.path), basename(b.path)));
    for (const d of dirs) {
      const open = S.openDirs.has(d.path);
      const row = el("div", "tnode dir" + (open ? " open" : ""));
      row.style.paddingLeft = (8 + depth * 14) + "px";
      row.dataset.path = d.path; row.dataset.dir = "1";
      row.append(el("span", "chev", "▶"));
      const ic = el("span", "ficon"); ic.innerHTML = ICONS.dir; row.append(ic);
      row.append(el("span", "tname", d.name));
      if (S.protectedDirs.has(d.path)) { const l = el("span", "lock", "🔒"); l.title = "Originals: add files here, but do not change them"; row.append(l); }
      const kids = el("div", "tchildren" + (open ? "" : " closed"));
      container.append(row, kids);
      if (open) walk(d, depth + 1, kids);
      else row._lazy = () => walk(d, depth + 1, kids);
    }
    for (const f of files) {
      const row = el("div", "tnode file" + (f.path === activePath ? " active" : ""));
      row.style.paddingLeft = (8 + depth * 14 + 13) + "px";
      row.dataset.path = f.path;
      const ic = el("span", "ficon"); ic.innerHTML = iconFor(f.path); row.append(ic);
      const nm = el("span", "tname");
      nm.textContent = stemOf(f.path) || basename(f.path);
      const e = extOf(f.path); if (e && stemOf(f.path)) nm.append(el("span", "ext", "." + e));
      row.append(nm);
      if (f.mtime > fresh && f.mtime > S.seenMtime && S.ownSaves.get(f.path) !== f.mtime) { const d = el("span", "fresh"); d.title = "Changed " + ago(f.mtime); row.append(d); }
      row.title = `${f.path}\n${fmtSize(f.size)} · modified ${ago(f.mtime)}`;
      row.draggable = true;
      container.append(row);
    }
  };
  walk(buildTree(), 0, tree);
  tree.parentElement.scrollTop = scroll;
}

$("tree").addEventListener("click", e => {
  const row = e.target.closest(".tnode");
  if (!row || e.target.tagName === "INPUT") return;
  const path = row.dataset.path;
  const plain = !row.dataset.dir && !e.altKey && !e.ctrlKey && !e.metaKey;
  // A plain click replaces the current tab, but the first half of a
  // double-click must not consume the tab that double-click should keep.
  const cur = activeTab();
  if (plain && cur && cur.type === "file" && !cur.dirty
      && !S.tabs.some(t => t.type === "file" && t.path === path)) {
    laterTreeClick(path);
    return;
  }
  runTreeClick();
  if (row.dataset.dir) toggleDir(row);
  else if (e.altKey) openToSide(path);
  else openFile(path, { newTab: e.ctrlKey || e.metaKey });
});
$("tree").addEventListener("dblclick", e => {
  const row = e.target.closest(".tnode.file");
  if (!row || e.target.tagName === "INPUT") return;
  cancelTreeClick();
  openFile(row.dataset.path, { newTab: true });
});
$("tree").addEventListener("auxclick", e => {
  const row = e.target.closest(".tnode.file");
  if (row && e.button === 1) { e.preventDefault(); runTreeClick(); openFile(row.dataset.path, { newTab: true }); }
});
function toggleDir(row) {
  const p = row.dataset.path, open = !S.openDirs.has(p);
  open ? S.openDirs.add(p) : S.openDirs.delete(p);
  store.set("openDirs", [...S.openDirs]);
  row.classList.toggle("open", open);
  const kids = row.nextElementSibling;
  kids.classList.toggle("closed", !open);
  if (open && row._lazy) { row._lazy(); row._lazy = null; }
}
function revealInTree(path) {
  const parts = dirname(path).split("/").filter(Boolean);
  let acc = "", changed = false;
  for (const p of parts) { acc = acc ? acc + "/" + p : p; if (!S.openDirs.has(acc)) { S.openDirs.add(acc); changed = true; } }
  if (changed) { store.set("openDirs", [...S.openDirs]); renderTree(); }
  document.querySelectorAll("#tree .tnode.active").forEach(n => n.classList.remove("active"));
  const row = treeRow(path);
  if (row) { row.classList.add("active"); row.scrollIntoView({ block: "nearest" }); }
}
$("btn-collapse").onclick = () => { S.openDirs.clear(); store.set("openDirs", []); renderTree(); };

// Context menu
function showMenu(x, y, items) {
  const m = $("ctx-menu");
  m.innerHTML = "";
  for (const it of items) {
    if (it === "-") { m.append(el("div", "cm-sep")); continue; }
    const row = el("div", "cm-item" + (it.danger ? " danger" : ""));
    row.append(el("span", "", it.label));
    if (it.key) row.append(el("span", "k", it.key));
    row.onclick = () => { hideMenu(); it.run(); };
    m.append(row);
  }
  m.classList.add("show");
  const r = m.getBoundingClientRect();
  m.style.left = Math.min(x, innerWidth - r.width - 8) + "px";
  m.style.top = Math.min(y, innerHeight - r.height - 8) + "px";
}
function hideMenu() { $("ctx-menu").classList.remove("show"); }
document.addEventListener("mousedown", e => { if (!e.target.closest("#ctx-menu")) hideMenu(); });
window.addEventListener("blur", hideMenu);

$("tree").addEventListener("contextmenu", e => {
  e.preventDefault();
  const row = e.target.closest(".tnode");
  const path = row ? row.dataset.path : "";
  const isDir = !row || !!row.dataset.dir;
  const folder = isDir ? path : dirname(path);
  const items = [];
  if (row && !isDir) {
    items.push({ label: "Open", run: () => openFile(path) }, { label: "Open in new tab", key: "Ctrl+Click", run: () => openFile(path, { newTab: true }) },
      { label: "Open to the side", key: "Alt+Click", run: () => openToSide(path) });
    if (kindOf(path) === "md" || kindOf(path) === "text") items.push({ label: "Edit", run: () => openFile(path, { mode: "source" }) });
    items.push("-");
  }
  items.push({ label: "New note", run: () => newNote(folder) }, { label: "New folder", run: () => newFolder(folder) }, { label: "Upload files here…", run: () => pickUpload(folder) });
  if (S.term.enabled && !S.term.error) items.push({ label: "Open terminal here", run: () => newTerminal(null, folder) });
  if (row) {
    items.push("-", { label: "Rename…", key: "F2", run: () => startRename(row) }, { label: "Move to…", run: () => moveDialog(path) });
    if (!isDir) {
      items.push({ label: "History", run: () => { openFile(path); setTimeout(() => openRightSection("history"), 200); } });
      items.push({ label: "Download", run: () => downloadFile(path) });
    }
    items.push({ label: "Copy path", run: () => copyText(path) });
    if (!isDir && isMd(path)) items.push({ label: "Copy wikilink", run: () => copyText(`[[${stemOf(path)}]]`) });
    items.push("-", { label: isDir ? "Delete folder…" : "Delete…", danger: true, run: () => deletePath(path, isDir) });
  }
  showMenu(e.clientX, e.clientY, items);
});

async function copyText(t) {
  try { await navigator.clipboard.writeText(t); toast("Copied"); }
  catch { await mlPrompt("Copy this:", t, { okLabel: "Done" }); }
}

function uniqueName(folder, stem, ext) {
  let name = `${stem}${ext}`, n = 1;
  while (S.fileMap.has(joinPath(folder, name)) || S.dirs.includes(joinPath(folder, name))) name = `${stem} ${n++}${ext}`;
  return joinPath(folder, name);
}
const BAD_NAME = /[\\\x00-\x1f]|^\.\.?$|^\.git$/i;
async function newNote(folder = null, name = null, opts = {}) {
  if (folder == null) folder = attachDir(activeTab());
  if (name == null) {
    name = await mlPrompt(`New note in ${folder || "the top folder"}:`, "", { okLabel: "Create", placeholder: "Note name" });
    if (name == null) return;
  }
  name = name.trim().replace(/\.md$/i, "") || "Untitled";
  if (name.split("/").some(p => !p || BAD_NAME.test(p))) { toast("That name is not allowed", "err"); return; }
  const path = S.fileMap.has(joinPath(folder, name + ".md")) && !opts.openExisting ? uniqueName(folder, name, ".md") : joinPath(folder, name + ".md");
  if (S.fileMap.has(path)) { openFile(path); return; }
  const title = basename(name);
  try {
    await apiJSON("/api/notes/file", "PUT", { path, content: opts.content ?? `# ${title}\n\n`, base_hash: null });
    await loadTree();
    await openFile(path, { mode: "source", focusEnd: true });
  } catch (e) { toast("Could not create note: " + e.message, "err"); }
}
async function newFolder(parent = "") {
  const name = await mlPrompt(`New folder in ${parent || "the top folder"}:`, "", { okLabel: "Create", placeholder: "Folder name" });
  if (!name) return;
  if (name.split("/").some(p => !p || BAD_NAME.test(p))) { toast("That name is not allowed", "err"); return; }
  const path = joinPath(parent, name.trim());
  try {
    await apiJSON("/api/notes/mkdir", "POST", { path });
    S.openDirs.add(path); if (parent) S.openDirs.add(parent); store.set("openDirs", [...S.openDirs]);
    await loadTree();
    toast("Folder created. Empty folders sync once they hold a file.");
  } catch (e) { toast(e.message, "err"); }
}
function startRename(row) {
  const path = row.dataset.path, isDir = !!row.dataset.dir;
  const nameEl = row.querySelector(".tname");
  const input = el("input", "rename");
  const current = basename(path);
  input.value = !isDir && isMd(path) ? stemOf(path) : current;
  nameEl.replaceWith(input);
  input.focus(); input.select();
  let done = false;
  const finish = async ok => {
    if (done) return; done = true;
    let v = input.value.trim();
    if (!ok || !v || v === input.defaultValue) { renderTree(); return; }
    if (!isDir && isMd(path) && !/\.(md|markdown)$/i.test(v)) v += ".md";
    if (v === current) { renderTree(); return; }
    if (v.includes("/") || BAD_NAME.test(v)) { toast("Use Move to change folders", "err"); renderTree(); return; }
    await movePath(path, joinPath(dirname(path), v));
  };
  input.defaultValue = input.value;
  input.addEventListener("keydown", e => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); e.stopPropagation(); });
  input.addEventListener("blur", () => finish(true));
}
async function moveDialog(path) {
  const to = await mlPrompt(`Move ${path} to (folder/name):`, path, { okLabel: "Move" });
  if (!to || to === path) return;
  await movePath(path, to.replace(/^\/+/, ""));
}
async function movePath(from, to) {
  try {
    await flushSaves();
    const r = await apiJSON("/api/notes/move", "POST", { from, to, update_links: true });
    for (const t of S.tabs) {
      if (t.path === from) { t.path = to; t.title = titleFor(to); }
      else if (t.path && t.path.startsWith(from + "/")) { t.path = to + t.path.slice(from.length); t.title = titleFor(t.path); }
    }
    if (S.openDirs.has(from)) { S.openDirs.delete(from); S.openDirs.add(to); }
    await loadTree();
    for (const p of r.links_updated_in || []) { const t = S.tabs.find(t => t.path === p); if (t) await reloadTab(t, { quiet: true }); }
    renderAll(); saveTabs();
    toast(r.links_updated_in && r.links_updated_in.length ? `Moved; updated links in ${r.links_updated_in.length} note(s)` : "Moved", "ok");
  } catch (e) { toast("Move failed: " + e.message, "err"); renderTree(); }
}
async function deletePath(path, isDir) {
  const ok = await mlConfirm(isDir ? `Delete the folder ${path} and everything in it?\n\nIt stays in the git history.` : `Delete ${path}?\n\nIt stays in the git history, so it can be restored.`, { confirmLabel: "Delete" });
  if (!ok) return;
  try {
    await api(`/api/notes/file?path=${q(path)}${isDir ? "&recursive=1" : ""}`, { method: "DELETE" });
    for (const t of [...S.tabs]) if (t.path === path || (isDir && t.path && t.path.startsWith(path + "/"))) closeTab(t.id, true);
    await loadTree();
    toast("Deleted", "ok");
  } catch (e) { toast("Delete failed: " + e.message, "err"); }
}

// Drag files within the tree to move them; drop OS files to upload.
let dragPath = null;
$("tree").addEventListener("dragstart", e => {
  const row = e.target.closest(".tnode"); if (!row) return;
  dragPath = row.dataset.path;
  e.dataTransfer.effectAllowed = "move";
  e.dataTransfer.setData("text/plain", embedText(dragPath));
});
$("tree").addEventListener("dragend", () => { dragPath = null; document.querySelectorAll(".drop-target").forEach(n => n.classList.remove("drop-target")); });
$("tree").addEventListener("dragover", e => {
  const row = e.target.closest(".tnode");
  const folderRow = row ? (row.dataset.dir ? row : null) : null;
  if (dragPath || (e.dataTransfer.types || []).includes("Files")) {
    e.preventDefault(); e.stopPropagation();
    document.querySelectorAll(".drop-target").forEach(n => n.classList.remove("drop-target"));
    if (folderRow) folderRow.classList.add("drop-target");
    $("drop-overlay").classList.remove("show");
  }
});
$("tree").addEventListener("drop", async e => {
  e.preventDefault(); e.stopPropagation();
  $("drop-overlay").classList.remove("show");
  const row = e.target.closest(".tnode");
  const folder = row ? (row.dataset.dir ? row.dataset.path : dirname(row.dataset.path)) : "";
  document.querySelectorAll(".drop-target").forEach(n => n.classList.remove("drop-target"));
  if (dragPath) {
    const from = dragPath; dragPath = null;
    const to = joinPath(folder, basename(from));
    if (to !== from && !to.startsWith(from + "/")) await movePath(from, to);
  } else if (e.dataTransfer.files.length) {
    await uploadFiles(e.dataTransfer.files, folder);
  }
});

// Uploads
function pickUpload(folder) {
  const inp = $("upload-input");
  inp.onchange = async () => { if (inp.files.length) await uploadFiles(inp.files, folder); inp.value = ""; };
  inp.click();
}
$("btn-upload").onclick = () => pickUpload(attachDir(activeTab()));
async function uploadFiles(fileList, folder) {
  const fd = new FormData();
  for (const f of fileList) fd.append("file", f, f.name);
  try {
    const r = await api(`/api/notes/upload?dir=${q(folder)}`, { method: "POST", body: fd });
    if (folder) { S.openDirs.add(folder); store.set("openDirs", [...S.openDirs]); }
    await loadTree();
    toast(`Uploaded ${r.saved.length} file(s)`, "ok");
    return r.saved;
  } catch (e) { toast("Upload failed: " + e.message, "err"); return []; }
}
let dragDepth = 0;
window.addEventListener("dragenter", e => { if ((e.dataTransfer.types || []).includes("Files") && !dragPath) { dragDepth++; $("drop-overlay").classList.add("show"); } });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop-overlay").classList.remove("show"); } });
window.addEventListener("dragover", e => { if ((e.dataTransfer.types || []).includes("Files")) e.preventDefault(); });
window.addEventListener("drop", async e => {
  dragDepth = 0; $("drop-overlay").classList.remove("show");
  if (!e.dataTransfer.files.length || e.defaultPrevented) return;
  e.preventDefault();
  const t = activeTab();
  const folder = attachDir(t);
  const saved = await uploadFiles(e.dataTransfer.files, folder);
  if (saved.length && t && t.type === "file" && t.kind === "md" && (t.mode !== "live" || (t.live && t.live.ta))) insertEmbeds(saved);
});
$("btn-new-note").onclick = () => newNote();
$("btn-new-folder").onclick = () => newFolder(attachDir(activeTab()));
