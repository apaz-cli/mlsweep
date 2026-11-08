// mlsweep Notes: tabs, file views, markdown rendering, the editor.
"use strict";

// ── Tabs ───────────────────────────────────────────────────────────────────────
let tabSeq = 0;
const titleFor = p => isMd(p) ? stemOf(p) : basename(p);
function newTab(props) {
  return { id: ++tabSeq, type: "file", path: null, title: "", kind: null, mode: "live",
           doc: null, hash: null, saved: "", dirty: false, saving: false, meta: null,
           conflict: null, deleted: false, scroll: {}, ...props };
}
const activeTab = () => S.tabs.find(t => t.id === S.active) || null;
function saveTabs() {
  store.set("tabs", S.tabs.filter(t => t.type === "file" || t.type === "graph").map(t => ({ type: t.type, path: t.path, mode: t.mode })));
  const t = activeTab();
  store.set("activeTab", t ? { type: t.type, path: t.path } : null);
  const u = new URL(location.href);
  if (t && t.type === "file" && t.path) u.searchParams.set("path", t.path); else u.searchParams.delete("path");
  history.replaceState(null, "", u.pathname + (u.search || "") + u.hash);
}

function renderTabs() {
  const strip = $("tabstrip");
  strip.innerHTML = "";
  for (const t of S.tabs) {
    const d = el("div", "dtab" + (t.id === S.active ? " active" : "") + (t.dirty ? " dirty" : ""));
    d.dataset.id = t.id;
    const ic = el("span", "ti");
    ic.innerHTML = t.type === "graph" ? ICONS.graph : t.type === "diff" ? ICONS.diff : iconFor(t.path || "");
    d.append(ic, el("span", "tt", t.title || "Untitled"));
    d.title = t.path || t.title;
    const x = el("span", "x", "×"); x.title = "Close";
    x.onclick = e => { e.stopPropagation(); closeTab(t.id); };
    d.append(x);
    d.onclick = () => activate(t.id);
    d.onauxclick = e => { if (e.button === 1) { e.preventDefault(); closeTab(t.id); } };
    d.oncontextmenu = e => {
      e.preventDefault();
      showMenu(e.clientX, e.clientY, [
        { label: "Close", run: () => closeTab(t.id) },
        { label: "Close others", run: () => S.tabs.filter(o => o.id !== t.id).forEach(o => closeTab(o.id)) },
        { label: "Close all", run: () => [...S.tabs].forEach(o => closeTab(o.id)) },
        ...(t.path ? ["-", { label: "Reveal in file tree", run: () => { switchPane("files"); revealInTree(t.path); } }, { label: "Copy path", run: () => copyText(t.path) }] : []),
      ]);
    };
    strip.append(d);
  }
  const plus = el("button", "icon-btn"); plus.id = "tab-new"; plus.title = "New note (N)";
  plus.innerHTML = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M8 3v10M3 8h10"/></svg>';
  plus.onclick = () => newNote();
  strip.append(plus);
  const a = strip.querySelector(".dtab.active");
  if (a) a.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function activate(id, { pushNav = true } = {}) {
  const prev = activeTab();
  if (prev && prev.id !== id) stashScroll(prev);
  if (pushNav && prev && prev.id !== id && prev.path) { S.navBack.push({ type: prev.type, path: prev.path }); S.navFwd = []; }
  S.active = id;
  renderTabs(); renderActive(); saveTabs();
  const t = activeTab();
  if (t && t.path) { revealInTree(t.path); pushRecent(t.path); }
}
async function closeTab(id, force = false) {
  const t = S.tabs.find(x => x.id === id);
  if (!t) return;
  if (t.dirty && !force) {
    await saveTab(t);
    if (t.dirty && !(await mlConfirm(`${t.title} has unsaved changes that could not be saved. Close anyway?`, { confirmLabel: "Discard" }))) return;
  }
  const i = S.tabs.indexOf(t);
  S.tabs.splice(i, 1);
  if (S.active === id) {
    const next = S.tabs[i] || S.tabs[i - 1];
    S.active = next ? next.id : null;
  }
  renderTabs(); renderActive(); saveTabs();
}
function pushRecent(path) {
  S.recent = [path, ...S.recent.filter(p => p !== path)].slice(0, 30);
  store.set("recent", S.recent);
}
function stashScroll(t) {
  if (t.live && t.live.editing >= 0) lvCommit(t.live);
  const v = $("view").querySelector(".vpane.reader");
  if (v) t.scroll.read = v.scrollTop;
  if (Editor.cm && Editor.tab === t) { t.scroll.edit = Editor.cm.getScrollInfo().top; t.cursor = Editor.cm.getCursor(); }
}

// Open a file: reuse its tab, else replace the active file tab (unless newTab).
async function openFile(path, { newTab: inNew = false, mode = null, line = null, heading = null, focusEnd = false, pushNav = true } = {}) {
  if (!path) return;
  let t = S.tabs.find(x => x.type === "file" && x.path === path);
  if (!t) {
    const cur = activeTab();
    const kind = kindOf(path);
    const defMode = kind === "text" ? "source" : "live";
    if (cur && cur.type === "file" && !inNew && !cur.dirty) {
      stashScroll(cur);
      if (pushNav && cur.path) { S.navBack.push({ type: "file", path: cur.path }); S.navFwd = []; }
      Object.assign(cur, newTab({ id: cur.id, path, title: titleFor(path), kind, mode: mode || defMode }));
      t = cur;
      pushNav = false;
    } else {
      t = newTab({ path, title: titleFor(path), kind, mode: mode || defMode });
      const idx = cur ? S.tabs.indexOf(cur) + 1 : S.tabs.length;
      S.tabs.splice(idx, 0, t);
    }
    try { await loadTabContent(t); }
    catch (e) {
      if (e.status === 404) { toast(`${path} no longer exists`, "err"); S.tabs = S.tabs.filter(x => x !== t); if (S.active === t.id) S.active = S.tabs.length ? S.tabs[S.tabs.length - 1].id : null; renderTabs(); renderActive(); return; }
      toast("Could not open: " + e.message, "err");
    }
  } else if (mode) t.mode = mode;
  t.pendingLine = line; t.pendingHeading = heading; t.focusEnd = focusEnd;
  activate(t.id, { pushNav });
}

async function loadTabContent(t) {
  const meta = await api(`/api/notes/file?path=${q(t.path)}`);
  t.meta = meta; t.kind = meta.kind; t.hash = meta.hash; t.deleted = false;
  if (meta.content != null) {
    t.saved = meta.content;
    const mode = t.kind === "md" ? "gfm" : modeForPath(t.path);
    t.doc = CodeMirror.Doc(meta.content, mode);
    t.doc.on("change", () => onDocChange(t));
  } else t.doc = null;
  t.dirty = false;
}
// Put text that came from outside into the tab's doc, not as an edit of the user's.
function applyRemote(t, content) {
  const cur = t.doc.getCursor();
  t.liveResume = lvSuspend(t);
  t.applyingRemote = true;
  t.doc.setValue(content);
  t.applyingRemote = false;
  t.doc.setCursor(cur);
}
async function reloadTab(t, { quiet = false } = {}) {
  try {
    const meta = await api(`/api/notes/file?path=${q(t.path)}`);
    t.meta = meta; t.hash = meta.hash; t.kind = meta.kind;
    if (meta.content != null && t.doc) {
      if (t.doc.getValue() !== meta.content) {
        const sc = Editor.cm && Editor.tab === t ? Editor.cm.getScrollInfo() : null;
        applyRemote(t, meta.content);
        if (sc) Editor.cm.scrollTo(sc.left, sc.top);
        if (!quiet) toast(`${t.title} was updated by sync`);
      }
      t.saved = meta.content; t.dirty = false;
    } else if (meta.content != null && !t.doc) { await loadTabContent(t); }
    t.conflict = null; t.deleted = false;
    if (t.id === S.active) renderActive({ keepEditor: true });
    renderTabs();
  } catch (e) {
    if (e.status === 404) { t.deleted = true; if (t.id === S.active) renderBanner(t); }
  }
}

// After a sync changed files: refresh open tabs whose file changed on disk.
async function checkOpenTabs() {
  for (const t of S.tabs) {
    if (t.type !== "file" || !t.path || t.saving) continue;
    const f = S.fileMap.get(t.path);
    if (!f) { if (!t.deleted) { t.deleted = true; if (t.id === S.active) renderBanner(t); } continue; }
    if (t.meta && f.mtime === t.meta.mtime && f.size === t.meta.size) continue;
    let meta;
    try { meta = await api(`/api/notes/file?path=${q(t.path)}`); } catch { continue; }
    if (meta.hash === t.hash) { t.meta = meta; continue; }
    if (!t.dirty) {
      t.meta = meta; t.hash = meta.hash;
      if (meta.content != null && t.doc) {
        applyRemote(t, meta.content);
        t.saved = meta.content;
      }
      if (t.id === S.active) renderActive({ keepEditor: true });
      toast(`${t.title} was updated by sync`);
    } else {
      t.conflict = { hash: meta.hash, content: meta.content };
      if (t.id === S.active) renderBanner(t);
    }
  }
}

function navigate(dir) {
  const from = dir < 0 ? S.navBack : S.navFwd, to = dir < 0 ? S.navFwd : S.navBack;
  while (from.length) {
    const target = from.pop();
    const cur = activeTab();
    if (cur && cur.path) to.push({ type: cur.type, path: cur.path });
    if (target.type === "file" && S.fileMap.has(target.path)) { openFile(target.path, { pushNav: false }); return; }
  }
}
$("btn-back").onclick = () => navigate(-1);
$("btn-fwd").onclick = () => navigate(1);

// ── Rendering the active tab ───────────────────────────────────────────────────
function renderActive({ keepEditor = false } = {}) {
  const t = activeTab();
  const view = $("view");
  const toolbar = $("doc-toolbar");
  if (!keepEditor || !t || t.type !== "file") Editor.detach();
  view.innerHTML = "";
  renderBanner(t);
  if (!S.enabled) { toolbar.classList.add("hidden"); if (S.enabled === false) showSetup(); return; }
  if (!t) { toolbar.classList.add("hidden"); renderEmpty(); refreshRight(); return; }
  if (t.type === "graph") { toolbar.classList.add("hidden"); Graph.mount(view, t); refreshRight(); return; }
  if (t.type === "diff") { toolbar.classList.add("hidden"); renderDiffView(view, t); refreshRight(); return; }
  toolbar.classList.remove("hidden");
  renderCrumbs(t);
  renderModeSeg(t);
  updateSaveState(t);

  if (!t.meta) { view.append(el("div", "placeholder", "Loading…")); return; }
  const k = t.kind;
  if ((k === "md" || k === "text") && t.doc) {
    if (k === "text" && t.mode !== "source") t.mode = "source";
    if (t.mode === "live") view.append(livePane(t));
    else if (t.mode === "source") view.append(Editor.attach(t));
    else { const ed = Editor.attach(t); view.append(ed, el("div", "split-divider"), readerPane(t)); }
    if (t.mode !== "live") Editor.refresh(t);
  } else if (k === "image") {
    const p = el("div", "vpane media-view");
    const img = el("img"); img.src = rawURL(t.path) + "&v=" + (t.hash || t.meta.mtime); img.alt = t.path;
    img.onclick = () => lightbox(img.src);
    p.append(img); view.append(p);
  } else if (k === "pdf") {
    const f = el("iframe", "pdf-view"); f.src = rawURL(t.path); f.title = t.path;
    const p = el("div", "vpane"); p.style.overflow = "hidden"; p.append(f); view.append(p);
  } else {
    const p = el("div", "vpane");
    const card = el("div", "file-card");
    const big = el("div", "big"); big.innerHTML = ICONS.binary; big.querySelector("svg").style.cssText = "width:42px;height:42px";
    card.append(big, el("div", "name", basename(t.path)), el("div", "meta", `${fmtSize(t.meta.size)}${t.meta.too_large ? " · too large to show" : " · not a text file"}`));
    const a = el("a", "mini-btn primary", "Download"); a.href = rawURL(t.path, "&download=1"); a.download = basename(t.path);
    card.append(a); p.append(card); view.append(p);
  }
  refreshRight();
}

function renderCrumbs(t) {
  const c = $("crumbs");
  c.innerHTML = "";
  const parts = t.path.split("/");
  parts.forEach((p, i) => {
    if (i) c.append(el("span", "sep", "/"));
    const last = i === parts.length - 1;
    const s = el("span", "c" + (last ? " last" : ""), last && isMd(p) ? stemOf(p) : p);
    if (!last) {
      const dir = parts.slice(0, i + 1).join("/");
      s.onclick = () => { switchPane("files"); S.openDirs.add(dir); store.set("openDirs", [...S.openDirs]); renderTree(); };
    } else {
      s.title = "Rename"; s.style.cursor = "text";
      s.ondblclick = () => { const row = treeRow(t.path); if (row) startRename(row); };
    }
    c.append(s);
  });
  if (t.meta && t.meta.protected) { const b = el("span", "lockbadge", "🔒 protected"); b.title = "An original: by convention, files here are added and not changed."; c.append(b); }
}
function renderModeSeg(t) {
  // Live, Source and Split are ways to read a note.  Other files are simply
  // themselves: text/code in the editor, images and PDFs in place.
  const note = t.kind === "md" && !!t.doc;
  document.querySelectorAll("#mode-seg button").forEach(b => {
    b.classList.toggle("on", b.dataset.mode === t.mode);
    b.disabled = false;
  });
  $("mode-seg").style.display = note ? "" : "none";
}
document.querySelectorAll("#mode-seg button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
function setMode(mode) {
  const t = activeTab();
  if (!t || t.type !== "file" || !t.doc) return;
  if (t.kind === "text" && mode !== "source") return;
  stashScroll(t);
  t.mode = mode;
  renderActive(); saveTabs();
  if (mode !== "live") Editor.cm && Editor.cm.focus();
}

function renderBanner(t) {
  const b = $("banner");
  b.className = ""; b.innerHTML = "";
  if (!t || t.type !== "file") return;
  const msg = el("span", "msg");
  const btn = (label, fn, primary) => { const x = el("button", "mini-btn" + (primary ? " primary" : ""), label); x.onclick = fn; return x; };
  if (t.deleted) {
    b.className = "show err";
    msg.textContent = "This file was deleted or moved on another machine.";
    b.append(msg);
    if (t.doc) b.append(btn("Save my copy", async () => { t.hash = null; t.deleted = false; await saveTab(t, { create: true }); renderBanner(t); await loadTree(); }, true));
    b.append(btn("Close", () => closeTab(t.id, true)));
  } else if (t.conflict) {
    b.className = "show warn";
    msg.textContent = "This file changed on another machine while you were editing it.";
    b.append(msg,
      btn("Compare…", () => showMerge(t), true),
      btn("Keep mine", () => keepMine(t)),
      btn("Take theirs", () => takeTheirs(t)));
  } else if (t.saveError) {
    b.className = "show err";
    msg.textContent = "Saving failed: " + t.saveError;
    b.append(msg, btn("Retry", () => saveTab(t), true));
  }
}

function updateSaveState(t) {
  const s = $("save-state");
  if (!t || t.type !== "file") { s.textContent = ""; return; }
  s.className = "";
  if (t.saving) s.textContent = "Saving…";
  else if (t.conflict) { s.textContent = "Conflict"; s.className = "err"; }
  else if (t.saveError) { s.textContent = "Not saved"; s.className = "err"; }
  else if (t.dirty) { s.textContent = "Edited"; s.className = "dirty"; }
  else s.textContent = "";
  if (!t.dirty && !t.saving && t.meta) {
    const lc = t.meta.last_commit;
    s.textContent = lc ? `${lc.author.split(/[ @(]/)[0]} · ${ago(lc.time)}` : (t.meta.mtime ? `edited ${ago(t.meta.mtime)}` : "");
    s.title = lc ? `Last change: ${lc.author}, ${new Date(lc.time * 1000).toLocaleString()}\n${lc.subject}` : "Not committed yet";
  }
}

function renderEmpty() {
  const v = $("view");
  const p = el("div", "vpane");
  const wrap = el("div", "empty-view");
  const st = S.status || {};
  wrap.innerHTML = `<h1>${esc(st.name || "Notes")}</h1>
    <div class="sub">${S.files.length} files · git remote <code>${esc(st.remote || "")}</code>.
    Agents clone it with <code>mlsweep notes clone DIR</code>; their changes arrive when they push.</div>
    <div class="empty-grid">
      <div><h3>Recent</h3><div class="empty-list" id="empty-recent"></div></div>
      <div><h3>Recently changed</h3><div class="empty-list" id="empty-changed"></div></div>
      <div><h3>Start</h3><div class="empty-list">
        <a id="e-new">Create a note</a><a id="e-open">Open a file…</a><a id="e-graph">Open the graph</a><a id="e-lint">Run lint</a></div></div>
      <div><h3>Keys</h3><div class="kbd-list">
        <span><kbd>T</kbd></span><span>Open a file</span>
        <span><kbd>N</kbd></span><span>New note</span>
        <span><kbd>E</kbd></span><span>Live / Source</span>
        <span><kbd>/</kbd></span><span>Search</span>
        <span><kbd>G</kbd></span><span>Graph</span>
        <span><kbd>X</kbd></span><span>Close tab</span>
        <span><kbd>?</kbd></span><span>All keys</span>
      </div></div>
    </div>`;
  p.append(wrap); v.append(p);
  const rec = wrap.querySelector("#empty-recent");
  const recents = S.recent.filter(r => S.fileMap.has(r)).slice(0, 8);
  if (!recents.length) rec.append(el("div", "muted", "Nothing opened yet."));
  for (const r of recents) { const a = el("a", "", titleFor(r)); a.title = r; a.onclick = () => openFile(r); rec.append(a); }
  const ch = wrap.querySelector("#empty-changed");
  const changed = [...S.files].filter(f => f.kind === "md").sort((a, b) => b.mtime - a.mtime).slice(0, 8);
  for (const f of changed) { const a = el("a", "", `${titleFor(f.path)}`); a.title = `${f.path} · ${ago(f.mtime)}`; a.onclick = () => openFile(f.path); ch.append(a); }
  wrap.querySelector("#e-new").onclick = () => newNote("");
  wrap.querySelector("#e-open").onclick = () => Switcher.open();
  wrap.querySelector("#e-graph").onclick = () => openGraph();
  wrap.querySelector("#e-lint").onclick = () => { switchPane("lint"); runLint(); };
}

function showSetup() {
  $("doc-toolbar").classList.add("hidden");
  $("tree").innerHTML = `<div class="tree-empty">Notes are not enabled on this manager.</div>`;
  const v = $("view"); v.innerHTML = "";
  const p = el("div", "vpane");
  p.innerHTML = `<div class="setup">
    <h1>Notes are off on this manager</h1>
    <p>The manager serves <code>&lt;mlsweep-dir&gt;/notes</code> (usually <code>~/.mlsweep/notes</code>) by default,
    creating it as a local git repo. It is off because the manager was started with <code>--no-notes</code>,
    or because <code>git</code> is not installed on it. Restart it without <code>--no-notes</code>, or point it at a folder:</p>
    <pre>mlsweep manager --notes ~/notes</pre>
    <p>The manager also hosts the notes as a git remote. <b>On any machine</b>:</p>
    <pre>mlsweep notes clone ~/notes --manager http://MANAGER:7891</pre>
    <p>Full guide: <code>mlsweep docs notes</code>.</p></div>`;
  v.append(p);
}

function lightbox(src) { $("lightbox-img").src = src; $("lightbox").classList.add("show"); }
$("lightbox").onclick = () => $("lightbox").classList.remove("show");

// ── Markdown rendering ─────────────────────────────────────────────────────────
const MD = new marked.Marked({ gfm: true, breaks: false });
// Math: $$…$$ (display, may span lines), $…$ (inline; no space just inside
// the dollars and no digit right after the closing one, so "$5 and $10" is
// not math), and ```math fences.  Rendered by KaTeX after sanitizing.
const INLINE_MATH = /^\$(?!\s)((?:\\.|[^\\$\n])+?)(?<![\s\\])\$(?![\d$])/;
const mathSpan = (tex, display) =>
  `<${display ? "div" : "span"} class="math${display ? " math-display" : ""}" data-tex="${esc(tex)}">${esc(tex)}</${display ? "div" : "span"}>`;
MD.use({
  extensions: [{
    name: "blockMath", level: "block",
    start(src) { const m = /(^|\n)\$\$/.exec(src); return m ? m.index + m[1].length : undefined; },
    tokenizer(src) {
      const m = /^\$\$([\s\S]+?)\$\$[ \t]*(?:\n|$)/.exec(src);
      if (m) return { type: "blockMath", raw: m[0], text: m[1].trim() };
    },
    renderer(t) { return mathSpan(t.text, true); },
  }, {
    name: "inlineMath", level: "inline",
    start(src) { const i = src.indexOf("$"); return i < 0 ? undefined : i; },
    tokenizer(src) {
      let m = /^\$\$((?:\\.|[^\\$])+?)\$\$/.exec(src);
      if (m) return { type: "inlineMath", raw: m[0], text: m[1].trim(), display: true };
      m = INLINE_MATH.exec(src);
      if (m) return { type: "inlineMath", raw: m[0], text: m[1], display: false };
    },
    renderer(t) { return mathSpan(t.text, t.display); },
  }, {
    name: "wikilink", level: "inline",
    start(src) { const i = src.indexOf("[["); if (i < 0) return undefined; return src[i - 1] === "!" ? i - 1 : i; },
    tokenizer(src) {
      const m = /^(!?)\[\[([^\[\]\n|#^]*)(?:#([^\[\]\n|]*))?(?:\|([^\[\]\n]*))?\]\]/.exec(src);
      if (m) return { type: "wikilink", raw: m[0], embed: !!m[1], target: m[2].trim(), heading: (m[3] || "").trim(), alias: (m[4] || "").trim() };
    },
    renderer(t) {
      const label = t.alias || (t.target ? stemOf(t.target) : "") + (t.heading ? (t.target ? " › " : "") + t.heading : "");
      if (t.embed) return `<span class="wl-embed" data-wl="${esc(t.target)}" data-heading="${esc(t.heading)}" data-alias="${esc(t.alias)}"></span>`;
      return `<a class="wikilink" data-wl="${esc(t.target)}" data-heading="${esc(t.heading)}">${esc(label || t.target)}</a>`;
    },
  }, {
    name: "footnoteRef", level: "inline",
    start(src) { const i = src.indexOf("[^"); return i < 0 ? undefined : i; },
    tokenizer(src) {
      const m = /^\[\^([^\]\s]+)\]/.exec(src);
      if (m) return { type: "footnoteRef", raw: m[0], id: m[1] };
    },
    renderer(t) {
      if (!FN || !FN.defs.has(t.id)) return esc(t.raw);
      let n = FN.numbers.get(t.id);
      if (!n) { n = FN.next++; FN.numbers.set(t.id, n); }
      const base = "fn-" + FN.prefix + n;
      const refId = "fnref-" + FN.prefix + n + "-" + (++FN.refs);
      if (!FN.firstRef.has(t.id)) FN.firstRef.set(t.id, refId);
      if (!FN_FIRST_REF.has(FN.src + "\u0000" + n)) FN_FIRST_REF.set(FN.src + "\u0000" + n, refId);
      return `<sup class="fn-ref"><a href="#${base}" id="${refId}" data-fn="${esc(t.id)}">${n}</a></sup>`;
    },
  }],
  renderer: {
    code(tok) {
      const lang = (tok.lang || "").trim().split(/\s+/)[0];
      if (/^(math|latex|tex|katex)$/i.test(lang)) return mathSpan(tok.text, true);
      return `<pre><code data-lang="${esc(lang)}">${esc(tok.text)}</code></pre>`;
    },
  },
});
const PURIFY = {
  // DOMPurify's default URI allow-list plus mlsweep:, which links a number to
  // the logged metric (the tab turns it into a normal link after sanitizing).
  ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp|mlsweep):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
  ADD_ATTR: ["data-wl", "data-heading", "data-alias", "data-lang", "data-tex", "data-fn", "target"],
  FORBID_TAGS: ["style", "form", "button", "textarea", "select", "iframe", "object", "embed", "link", "meta", "base"],
  FORBID_ATTR: ["style"],
  ALLOW_UNKNOWN_PROTOCOLS: false,
};

function splitFrontmatter(text) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  if (!m) return [null, text];
  return [m[1], text.slice(m[0].length)];
}
function slugify(s) { return s.toLowerCase().trim().replace(/[^\w\- ]+/g, "").replace(/\s+/g, "-") || "section"; }

const CALLOUT_ICONS = { note: "✎", info: "ℹ", tip: "💡", hint: "💡", important: "❗", warning: "⚠", caution: "⚠", attention: "⚠", danger: "⛔", error: "⛔", bug: "🐞", failure: "✗", fail: "✗", success: "✓", check: "✓", done: "✓", question: "?", help: "?", faq: "?", example: "▸", quote: "❝", cite: "❝", abstract: "≡", summary: "≡", tldr: "≡", todo: "☐", idea: "💡" };

// ── Footnotes and figures ─────────────────────────────────────────────────────
// State for the one synchronous MD.parse() call: a marked renderer cannot take
// arguments, so the renderer reads this.  Cleared again as soon as parsing ends.
let FN = null;
const FN_FIRST_REF = new Map();  // note + number -> the first backlink id
let FN_EMBED_SEQ = 0;            // keeps two embeds of one note from sharing ids

// A definition line is `[^id]: text`, plus any indented continuation lines.
// Fenced blocks and display math are left alone, so code can show the syntax.
function collectFootnotes(text) {
  const lines = text.split("\n");
  const out = [];
  const defs = new Map();
  let fence = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const was = fence;
    fence = lvNextFence(fence, line);
    if (was || fence) { out.push(line); continue; }
    const m = /^\[\^([^\]\s]+)\]:[ \t]*(.*)$/.exec(line);
    if (!m) { out.push(line); continue; }
    let body = m[2];
    while (i + 1 < lines.length) {
      const next = lines[i + 1];
      if (/^(?: {2,}|\t)/.test(next)) { body += "\n" + lines[++i].replace(/^(?: {2,4}|\t)/, ""); continue; }
      if (/^\s*$/.test(next) && i + 2 < lines.length && /^(?: {2,}|\t)/.test(lines[i + 2])) {
        body += "\n\n" + lines[i + 2].replace(/^(?: {2,4}|\t)/, "");
        i += 2;
        continue;
      }
      break;
    }
    if (!defs.has(m[1])) defs.set(m[1], body);
  }
  return { body: out.join("\n"), defs };
}

// Footnote numbers follow the order of first reference in the note.
function footnoteNumbers(body, defs) {
  const numbers = new Map();
  let next = 1, fence = null;
  for (const line of body.split("\n")) {
    const was = fence;
    fence = lvNextFence(fence, line);
    if (was || fence) continue;
    const re = /\[\^([^\]\s]+)\]/g;
    let m;
    while ((m = re.exec(line))) {
      if (!defs.has(m[1]) || numbers.has(m[1])) continue;
      numbers.set(m[1], next++);
    }
  }
  for (const id of defs.keys()) if (!numbers.has(id)) numbers.set(id, next++);
  return numbers;
}

function fnNext(numbers) {
  let n = 1;
  for (const v of numbers.values()) if (v >= n) n = v + 1;
  return n;
}

// The figures in a whole note, in order, so Live mode can number across blocks.
// A figure is a chart/table block with a `caption:`, or a caption line directly
// under an image or a table row.
const NB_CAP_RE = /^\s*(Figure|Table|Chart|Fig\.)\s*[:.]?\s+\S/i;
const NB_IMG_LINE_RE = /^\s*(?:!\[\[[^\]]+\]\]|!\[[^\]]*\]\([^)]+\))\s*$/;
function countFigures(text) {
  let fig = 0, table = 0, fence = null, lang = "", hasCaption = false, prev = "";
  for (const line of text.split("\n")) {
    if (fence) {
      if (line.trim().startsWith(fence) && line.trim().replace(/[`~]/g, "") === "") {
        if (/^mlsweep-(chart|table)$/.test(lang) && hasCaption) {
          if (lang === "mlsweep-table") table++; else fig++;
        }
        fence = null; lang = ""; hasCaption = false;
      } else if (/^\s*caption\s*:/i.test(line) || /"caption"\s*:/.test(line)) {
        hasCaption = true;
      }
      prev = line;
      continue;
    }
    const m = LV_FENCE.exec(line);
    if (m) {
      fence = m[1];
      lang = line.trim().slice(m[1].length).trim().split(/\s+/)[0].toLowerCase();
      hasCaption = false;
      prev = line;
      continue;
    }
    if (NB_CAP_RE.test(line) && (NB_IMG_LINE_RE.test(prev) || /\|/.test(prev))) {
      if (/^\s*Table\b/i.test(line)) table++; else fig++;
    }
    prev = line;
  }
  return { fig, table };
}

function appendFootnotes(container, defs, state, srcPath, depth) {
  if (!defs.size) return;
  const items = [...defs.keys()]
    .map(id => ({ id, n: state.numbers.get(id) || 0 }))
    .sort((a, b) => a.n - b.n);
  const sec = el("section", "footnotes");
  sec.append(el("hr"));
  const ol = el("ol", "fn-list");
  for (const { id, n } of items) {
    const li = el("li");
    li.id = "fn-" + state.prefix + n;
    li.dataset.fn = id;
    const body = el("div", "fn-body");
    li.append(body);
    renderMarkdown(defs.get(id), srcPath, body, { depth: depth + 1, footnote: false, frontmatter: false });
    const back = state.firstRef.get(id) || FN_FIRST_REF.get(srcPath + "\u0000" + n);
    if (back) {
      const a = el("a", "fn-back", "↩");
      a.href = "#" + back;
      a.title = "Back to the reference";
      li.append(a);
    }
    ol.append(li);
  }
  sec.append(ol);
  container.append(sec);
}

function renderMarkdown(text, srcPath, container, opts = {}) {
  const { depth = 0, footnote = true, frontmatter = true, footnotePrefix = "" } = opts;
  const figBase = typeof opts.figBase === "number"
    ? { fig: opts.figBase, table: 0 }
    : (opts.figBase || { fig: 0, table: 0 });
  const [fm, body] = frontmatter ? splitFrontmatter(text) : [null, text];
  const local = footnote ? collectFootnotes(body) : { body, defs: new Map() };
  const defs = new Map(opts.fnDefs || []);
  for (const [id, def] of local.defs) defs.set(id, def);
  const numbers = opts.fnNumbers || footnoteNumbers(local.body, defs);
  const state = { defs, numbers, next: fnNext(numbers), firstRef: new Map(), refs: 0,
                  src: srcPath, prefix: footnotePrefix };
  const prev = FN;
  FN = footnote ? state : null;
  let html;
  try {
    html = DOMPurify.sanitize(MD.parse(splitFigureCaptions(local.body)), PURIFY);
  } finally {
    FN = prev;
  }
  container.innerHTML = html;
  if (fm) {
    const props = propsEl(fm.split("\n"));
    if (props.childNodes.length) container.prepend(props);
  }
  postProcess(container, srcPath, depth, { figBase, footnote: footnote ? state : null });
  if (footnote) appendFootnotes(container, local.defs, state, srcPath, depth);
}

// A caption line directly under an image merges into one paragraph in
// CommonMark.  A blank line before it keeps the image and its caption separate
// for decorateFigures, without changing the note on disk.
function splitFigureCaptions(text) {
  const out = [];
  let fence = null;
  for (const line of text.split("\n")) {
    const was = fence;
    fence = lvNextFence(fence, line);
    if (!was && !fence && NB_CAP_RE.test(line) && out.length && NB_IMG_LINE_RE.test(out[out.length - 1])) {
      out.push("");
    }
    out.push(line);
  }
  return out.join("\n");
}

function nbCaptionParts(p) {
  const text = p.textContent.trim();
  const m = NB_CAP_RE.exec(text);
  if (!m) return null;
  let inner = p.innerHTML.replace(/^\s*(?:Figure|Table|Chart|Fig\.)\s*[:.]?\s+/i, "");
  const idm = /\s*\{#([A-Za-z][\w:.\-]*)\}\s*$/.exec(text);
  let id = "";
  if (idm) {
    id = idm[1];
    inner = inner.replace(/\s*\{#[^}]+\}\s*$/, "");
  }
  const kind = /^fig/i.test(m[1]) ? "Figure" : m[1];
  return { kind, html: inner, id };
}

// Turn caption lines into <figcaption>, and chart/table blocks into rendered
// cards, in one document-order pass so `Figure 1, 2, …` follows the note.
function decorateFigures(root, figState) {
  if (typeof nbBlock !== "function") return;  // an old cached page without notes-blocks.js
  const state = {
    fig: (figState && figState.fig) || 0,
    table: (figState && figState.table) || 0,
  };
  const nodes = [...root.querySelectorAll("pre, p")];
  for (const node of nodes) {
    // parentNode, not isConnected: Live builds a block detached, then appends it.
    if (!node.parentNode) continue;
    if (node.tagName === "PRE") {
      const code = node.firstElementChild;
      if (!code || code.tagName !== "CODE") continue;
      const lang = (code.dataset.lang || "").toLowerCase();
      if (lang !== "mlsweep-chart" && lang !== "mlsweep-table") continue;
      nbBlock(node, lang === "mlsweep-table" ? "table" : "chart", code.textContent, state);
      continue;
    }
    const parts = nbCaptionParts(node);
    if (!parts) continue;
    const prev = node.previousElementSibling;
    if (!prev || !prev.matches) continue;
    // An image on its own line is wrapped in a paragraph, so look inside it.
    let target = prev;
    if (prev.tagName === "P" && prev.childElementCount === 1) {
      const only = prev.firstElementChild;
      if (only && only.matches("img, table, .table-wrap, .nb-block")) target = only;
    }
    const bare = target.matches("figure.note-fig.nb-bare") ? target : null;
    const eligible = bare || target.matches("img, table, .table-wrap, .nb-block");
    if (!eligible) continue;
    const counter = /^table$/i.test(parts.kind) ? "table" : "fig";
    const n = ++state[counter];
    const fig = bare || el("figure", "note-fig");
    if (bare) bare.classList.remove("nb-bare");
    else {
      prev.replaceWith(fig);
      fig.append(target);
      if (target !== prev) prev.remove();
    }
    fig.id = parts.id || ((counter === "table" ? "tbl-" : "fig-") + n);
    const cap = el("figcaption");
    cap.innerHTML = `<span class="fig-kind">${esc(parts.kind)} ${n}.</span> ` + parts.html;
    fig.append(cap);
    node.remove();
  }
}

function postProcess(root, srcPath, depth, opts = {}) {
  // Headings get ids for the outline; `## Title {#id}` fixes one for links.
  const used = new Map();
  root.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach(h => {
    let explicit = null;
    const m = /\s*\{#([A-Za-z][\w:.\-]*)\}\s*$/.exec(h.textContent);
    if (m) { explicit = m[1]; h.textContent = h.textContent.slice(0, m.index).trim(); }
    let id = explicit || "h-" + slugify(h.textContent);
    const n = used.get(id) || 0;
    used.set(id, n + 1);
    if (n) id += "-" + n;
    h.id = id;
  });
  // Callouts: > [!type] Title
  root.querySelectorAll("blockquote").forEach(bq => {
    const p = bq.firstElementChild;
    if (!p || p.tagName !== "P") return;
    const m = /^\[!([\w-]+)\]([+-]?)[ \t]*([^\n]*)/.exec(p.textContent);
    if (!m) return;
    const type = m[1].toLowerCase();
    const div = el("div", `callout c-${type}`);
    const title = el("div", "callout-title");
    title.append(el("span", "ci", CALLOUT_ICONS[type] || "✎"), el("span", "", m[3] || type[0].toUpperCase() + type.slice(1)));
    div.append(title);
    // Remove the marker line from the first paragraph; keep the rest of it.
    const firstText = p.innerHTML.replace(/^\s*\[![\w-]+\][+-]?[^\n<]*(<br>|\n)?/, "");
    if (firstText.trim()) { const np = el("p"); np.innerHTML = firstText; div.append(np); }
    [...bq.children].slice(1).forEach(c => div.append(c));
    bq.replaceWith(div);
  });
  // Tables scroll horizontally.
  root.querySelectorAll("table").forEach(tb => { const w = el("div", "table-wrap"); tb.replaceWith(w); w.append(tb); });
  // Task list checkboxes are the only form control a note can contain.
  root.querySelectorAll("input:not([type=checkbox])").forEach(n => n.remove());
  // Task list items.
  let taskIdx = 0;
  root.querySelectorAll("li > input[type=checkbox]").forEach(cb => {
    const li = cb.parentElement;
    li.classList.add("task"); if (cb.checked) li.classList.add("done");
    cb.removeAttribute("disabled"); cb.dataset.task = String(taskIdx++);
    if (depth > 0) cb.disabled = true;
  });
  // Code highlighting.
  root.querySelectorAll("pre > code[data-lang]").forEach(code => {
    const lang = code.dataset.lang;
    const mode = modeForLang(lang);
    if (lang) code.parentElement.append(el("span", "lang", lang));
    if (mode && CodeMirror.runMode) { const txt = code.textContent; code.textContent = ""; CodeMirror.runMode(txt, mode, code); }
  });
  // Wikilinks.
  root.querySelectorAll("a.wikilink").forEach(a => {
    const target = a.dataset.wl, heading = a.dataset.heading;
    const hit = target ? Resolver.wiki(srcPath, target) : srcPath;
    if (!hit) { a.classList.add("missing"); a.title = `No note "${target}" yet — click to create it`; }
    else a.title = hit + (heading ? " › " + heading : "");
    a.href = hit ? pageHref("notes.html", { path: hit }) : "#";
    a.dataset.path = hit || "";
  });
  // Embeds.
  root.querySelectorAll("span.wl-embed").forEach(sp => {
    const target = sp.dataset.wl, hit = Resolver.wiki(srcPath, target);
    if (!hit) { const a = el("a", "wikilink missing", sp.dataset.alias || target); a.dataset.wl = target; a.href = "#"; a.title = "Missing file"; sp.replaceWith(a); return; }
    const k = kindOf(hit);
    if (k === "image") {
      const img = el("img"); img.src = rawURL(hit); img.alt = sp.dataset.alias || basename(hit); img.loading = "lazy";
      const w = /^(\d+)(?:x(\d+))?$/.exec(sp.dataset.alias || ""); if (w) { img.style.width = w[1] + "px"; if (w[2]) img.style.height = w[2] + "px"; img.alt = basename(hit); }
      sp.replaceWith(img);
    } else if (k === "pdf") {
      const f = el("iframe", "embed-pdf"); f.src = rawURL(hit); f.title = hit; sp.replaceWith(f);
    } else if (k === "md" && depth < 2) {
      const box = el("div", "embed-note");
      const tt = el("div", "embed-title", "↳ " + titleFor(hit)); tt.onclick = () => openFile(hit); box.append(tt);
      const inner = el("div", "embed-body"); box.append(inner); sp.replaceWith(box);
      api(`/api/notes/file?path=${q(hit)}&history=0`).then(m => {
        let text = m.content || "";
        if (sp.dataset.heading) text = sectionOf(text, sp.dataset.heading) ?? text;
        renderMarkdown(text, hit, inner, { depth: depth + 1, footnotePrefix: "e" + hashStr(hit) + (++FN_EMBED_SEQ) + "-" });
      }).catch(() => { inner.textContent = "(could not load)"; });
    } else {
      const a = el("a", "embed-file"); a.innerHTML = iconFor(hit); a.append(" " + basename(hit)); a.dataset.path = hit; a.href = "#"; sp.replaceWith(a);
    }
  });
  // Relative links and images.
  root.querySelectorAll("a[href]").forEach(a => {
    if (a.classList.contains("wikilink") || a.dataset.path) return;
    const href = a.getAttribute("href");
    if (/^mlsweep:/i.test(href) && typeof annotateEvidence === "function") {
      a.dataset.evRef = href.slice("mlsweep:".length);
      a.classList.add("metric-ev");
      a.removeAttribute("href");
      a.title = "Checking against the logged metric…";
      return;
    }
    if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("//")) {
      if (/^(https?|mailto):/i.test(href)) { a.target = "_blank"; a.rel = "noopener noreferrer"; a.classList.add("ext"); }
      else { a.removeAttribute("href"); }
      return;
    }
    if (href.startsWith("#")) { a.dataset.anchor = href.slice(1); return; }
    const hit = Resolver.relative(srcPath, href);
    a.dataset.path = hit || "";
    if (!hit) { a.classList.add("wikilink", "missing"); a.title = "Missing: " + href; }
    const hash = href.includes("#") ? href.split("#")[1] : "";
    if (hash) a.dataset.heading = decodeURIComponent(hash).replace(/-/g, " ");
  });
  root.querySelectorAll("img[src]").forEach(img => {
    const src = img.getAttribute("src");
    if (/^(https?:|data:image\/)/i.test(src) || src.startsWith("/api/notes/raw")) return;
    if (/^[a-z][a-z0-9+.-]*:/i.test(src)) { img.removeAttribute("src"); return; }
    const hit = Resolver.relative(srcPath, src) || Resolver.wiki(srcPath, basename(src.split("?")[0]));
    if (hit) img.src = rawURL(hit); else { img.alt = "Missing image: " + src; img.removeAttribute("src"); }
  });
  // Charts, tables and figure captions, in document order so numbering follows
  // the note.  After embeds, because a caption pairs with the resolved image.
  decorateFigures(root, opts.figBase);
  linkExperiments(root);
  if (typeof annotateEvidence === "function") annotateEvidence(root);
  renderMath(root);
}

// KaTeX writes its own markup into each placeholder; it escapes the TeX and,
// with trust off, refuses \href, \url, \includegraphics and friends.
function renderMath(root) {
  if (typeof katex === "undefined") return;
  root.querySelectorAll(".math[data-tex]").forEach(node => {
    try {
      katex.render(node.dataset.tex, node, {
        displayMode: node.classList.contains("math-display"), throwOnError: false,
        trust: false, strict: "ignore", output: "htmlAndMathml", maxExpand: 1000,
      });
    } catch (e) { node.classList.add("math-error"); node.title = String(e.message || e); }
  });
}

function sectionOf(text, heading) {
  const lines = text.split("\n");
  const want = heading.toLowerCase().trim();
  let start = -1, level = 0;
  for (let i = 0; i < lines.length; i++) {
    const m = HEADING_RE.exec(lines[i]);
    if (!m) continue;
    const title = m[2].replace(/\s*\{#[^}]+\}\s*$/, "");
    if (start < 0 && title.toLowerCase() === want) { start = i; level = m[1].length; continue; }
    if (start >= 0 && m[1].length <= level) return lines.slice(start, i).join("\n");
  }
  return start >= 0 ? lines.slice(start).join("\n") : null;
}

// Known experiment IDs anywhere in text become chips linking to Results.
const EXP_TOKEN = /[A-Za-z0-9][A-Za-z0-9_.\-]{2,127}/g;
function expChip(id) {
  const m = S.experiments.get(id) || {};
  const a = el("a", "exp-chip s-" + (m.status || "unknown"));
  a.href = pageHref("results.html", { experiment: id });
  a.title = `${id} · ${m.status || "?"}${m.campaign ? " · campaign " + m.campaign : ""}${m.jobs ? ` · ${m.jobs.done || 0}/${m.jobs.total || 0} runs done` : ""}\nOpen in Results (Shift+click: Logs)`;
  a.dataset.exp = id;
  a.append(el("span", "d"), document.createTextNode(m.name || id));
  return a;
}
function linkExperiments(root) {
  if (!S.experiments.size) return;
  root.querySelectorAll("code").forEach(c => {
    if (c.closest("pre") || c.closest("a")) return;
    const id = c.textContent.trim();
    if (S.experiments.has(id)) c.replaceWith(expChip(id));
  });
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: n => n.parentElement.closest("a, pre, code, .props") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const text = node.nodeValue;
    EXP_TOKEN.lastIndex = 0;
    let m, last = 0, frag = null;
    while ((m = EXP_TOKEN.exec(text))) {
      let tok = m[0];
      while (tok && !S.experiments.has(tok) && /[.\-]$/.test(tok)) tok = tok.slice(0, -1);
      if (!S.experiments.has(tok)) continue;
      frag = frag || document.createDocumentFragment();
      frag.append(text.slice(last, m.index), expChip(tok));
      last = m.index + tok.length;
      EXP_TOKEN.lastIndex = last;
    }
    if (frag) { frag.append(text.slice(last)); node.replaceWith(frag); }
  }
}

// The rendered side of split mode.
function readerPane(t) {
  const pane = el("div", "vpane reader");
  const doc = el("article", "doc");
  pane.append(doc);
  const text = t.doc ? t.doc.getValue() : (t.meta.content || "");
  if (t.kind === "md") renderMarkdown(text, t.path, doc);
  else { const pre = el("pre"); const code = el("code"); code.textContent = text; pre.append(code); doc.append(pre); }
  doc.addEventListener("click", e => onDocClick(e, t));
  doc.addEventListener("change", e => onTaskToggle(e, t));
  requestAnimationFrame(() => {
    if (t.pendingHeading) { scrollToHeading(pane, t.pendingHeading); t.pendingHeading = null; }
  });
  t.rerender = debounce(() => {
    const st = pane.scrollTop;
    renderMarkdown(t.doc.getValue(), t.path, doc);
    pane.scrollTop = st;
    refreshOutline(t);
  }, 220);
  return pane;
}
function scrollToHeading(pane, heading) {
  const want = heading.toLowerCase();
  const heads = [...pane.querySelectorAll("h1,h2,h3,h4,h5,h6")];
  const h = pane.querySelector("#" + CSS.escape(heading)) ||
    heads.find(x => x.textContent.toLowerCase().trim() === want) ||
    heads.find(x => x.textContent.toLowerCase().trim() === want.replace(/-/g, " ")) ||
    pane.querySelector("#h-" + CSS.escape(slugify(heading)));
  if (h) { h.scrollIntoView({ block: "start" }); h.animate([{ background: "var(--accent-soft)" }, { background: "transparent" }], { duration: 1400 }); }
}

function onDocClick(e, t) {
  const img = e.target.closest("img");
  if (img && !e.target.closest("a")) { lightbox(img.src); return; }
  const chip = e.target.closest("a.exp-chip");
  if (chip) {
    if (e.shiftKey) { e.preventDefault(); location.href = pageHref("logs.html", { experiment: chip.dataset.exp }); }
    return;
  }
  const a = e.target.closest("a");
  if (!a) return;
  if (a.dataset.anchor != null) { e.preventDefault(); scrollToHeading(e.currentTarget.parentElement, decodeURIComponent(a.dataset.anchor)); return; }
  if (a.classList.contains("ext")) return;
  if (a.dataset.path !== undefined || a.classList.contains("wikilink")) {
    e.preventDefault();
    const path = a.dataset.path;
    const newTab = e.ctrlKey || e.metaKey;
    if (path) openFile(path, { newTab, heading: a.dataset.heading || null });
    else if (a.dataset.wl) {
      const target = a.dataset.wl;
      const folder = target.includes("/") ? dirname(target) : dirname(t.path);
      newNote(folder, basename(target));
    }
  }
}
async function onTaskToggle(e, t) {
  const cb = e.target;
  if (!cb.matches("input[type=checkbox][data-task]") || !t.doc) return;
  const at = findTask(t.doc.getValue().split("\n"), Number(cb.dataset.task));
  if (!at) return;
  t.doc.replaceRange(cb.checked ? "x" : " ", at, { line: at.line, ch: at.ch + 1 });
  cb.parentElement.classList.toggle("done", cb.checked);
}

// ── Editor ─────────────────────────────────────────────────────────────────────
function modeForLang(lang) {
  const l = (lang || "").toLowerCase();
  const map = { py: "python", python: "python", sh: "shell", bash: "shell", shell: "shell", zsh: "shell", console: "shell",
    js: "javascript", javascript: "javascript", ts: "text/typescript", typescript: "text/typescript", json: "application/json",
    yaml: "yaml", yml: "yaml", toml: "toml", diff: "diff", patch: "diff", c: "text/x-csrc", cpp: "text/x-c++src", "c++": "text/x-c++src",
    cuda: "text/x-c++src", cu: "text/x-c++src", java: "text/x-java", md: "markdown", markdown: "markdown",
    rs: "text/x-rustsrc", rust: "text/x-rustsrc", go: "text/x-go", golang: "text/x-go" };
  return map[l] || null;
}
function modeForPath(p) {
  const e = extOf(p);
  const map = { py: "python", sh: "shell", bash: "shell", zsh: "shell", js: "javascript", ts: "text/typescript", json: "application/json",
    jsonl: "application/json", yaml: "yaml", yml: "yaml", toml: "toml", diff: "diff", patch: "diff", c: "text/x-csrc", h: "text/x-csrc",
    cc: "text/x-c++src", cpp: "text/x-c++src", hpp: "text/x-c++src", cu: "text/x-c++src", java: "text/x-java", html: "xml", xml: "xml", svg: "xml",
    rs: "text/x-rustsrc", go: "text/x-go" };
  return map[e] || null;
}

const Editor = {
  cm: null, host: null, tab: null,
  ensure() {
    if (this.cm) return;
    this.host = el("div", "vpane"); this.host.id = "editor-wrap";
    this.cm = CodeMirror(this.host, {
      lineWrapping: true, lineNumbers: false, styleActiveLine: true, autoCloseBrackets: "()[]{}``", theme: "mlnotes",
      indentUnit: 2, tabSize: 2, indentWithTabs: false, viewportMargin: 50, placeholder: "Start writing…",
      extraKeys: {
        "Enter": "newlineAndIndentContinueMarkdownList",
        "Tab": cm => cm.somethingSelected() ? cm.indentSelection("add") : cm.execCommand("insertSoftTab"),
        "Shift-Tab": cm => cm.indentSelection("subtract"),
        "Ctrl-S": () => saveActiveNow(), "Cmd-S": () => saveActiveNow(),
        "Ctrl-B": cm => wrapSel(cm, "**"), "Cmd-B": cm => wrapSel(cm, "**"),
        "Ctrl-I": cm => wrapSel(cm, "*"), "Cmd-I": cm => wrapSel(cm, "*"),
        "Ctrl-Space": cm => linkHint(cm, true),
        "Esc": cm => { if (cm.state.completionActive) return CodeMirror.Pass; const t = activeTab(); if (t && t.kind === "md") setMode("live"); else cm.getInputField().blur(); },
      },
    });
    this.overlay = {
      // CodeMirror overlays cannot be stateful.  Frontmatter and fence ranges
      // are precomputed by decorateBlocks and read through Editor.mdInfo.
      token(stream) {
        const info = Editor.mdInfo;
        const lineNo = stream.lineOracle
          ? (stream.lineOracle.lineNo != null ? stream.lineOracle.lineNo : stream.lineOracle.line)
          : null;
        if (info && lineNo != null) {
          if (info.frontmatter && lineNo >= info.frontmatter.start && lineNo <= info.frontmatter.end) {
            if (lineNo === info.frontmatter.start || lineNo === info.frontmatter.end) {
              stream.skipToEnd(); return "frontmatter-mark";
            }
            const colon = stream.string.indexOf(":");
            if (colon > 0 && stream.pos <= colon) {
              stream.match(/^[^:\n]+:/);
              return "frontmatter-key";
            }
            stream.skipToEnd(); return "frontmatter-value";
          }
          if (info.fenced.has(lineNo)) { stream.skipToEnd(); return null; }
        }
        if (stream.sol()) {
          const callout = /^>\s*\[!([\w-]+)\]/.exec(stream.string);
          if (callout) {
            stream.match(callout[0]);
            return "callout callout-" + callout[1].toLowerCase();
          }
          if (stream.match(/^(Figure|Table|Chart):/)) return "caption";
        }
        if (stream.match(/^\[\^[^\]\s]+\]:/)) return "footnote-def";
        if (stream.match(/^\[\^[^\]\s]+\]/)) return "footnote-ref";
        if (stream.match(/^!\[\[[^\]\n]*\]\]/)) return "wikilink wikilink-embed";
        if (stream.match(/^\[\[[^\]\n]*\]\]/)) return "wikilink";
        if (stream.match(/^mlsweep:[^\s)<>]+/)) return "evidence";
        if (stream.match(/^\^[A-Za-z0-9-]+(?=\s*$)/)) return "blockid";
        if (stream.match(/^#[A-Za-z][\w/-]*/)) return "tag";
        if (stream.match(/^\|/)) return "table-pipe";
        if (stream.match(/^\$\$[^$]+\$\$/) || stream.match(INLINE_MATH)) return "math";
        if (stream.match(/^[A-Za-z0-9][A-Za-z0-9_\-]{2,127}/)) {
          return S.experiments.has(stream.current()) ? "expid" : null;
        }
        stream.next();
        while (!stream.eol() && !/[A-Za-z0-9!\[$#>^m|]/.test(stream.peek())) stream.next();
        return null;
      },
    };
    this.overlayAdded = false;
    this.cm.on("changes", debounce(() => this.decorateBlocks(this.tab), 200));
    // Typing "[[" opens link completion (autoCloseBrackets inserts brackets
    // itself, so this listens for changes rather than inputRead).
    this.cm.on("change", (cm, ch) => {
      if (ch.origin === "setValue" || !ch.text || !ch.text.join("").includes("[")) return;
      if (cm.state.completionActive) return;
      const cur = cm.getCursor();
      if (/\[\[$/.test(cm.getLine(cur.line).slice(0, cur.ch))) setTimeout(() => linkHint(cm), 0);
    });
    this.cm.on("paste", (cm, e) => onPaste(cm, e));
    this.cm.on("drop", (cm, e) => {
      if (dragPath) { e.preventDefault(); const pos = cm.coordsChar({ left: e.clientX, top: e.clientY }); cm.replaceRange(embedText(dragPath), pos); dragPath = null; return; }
      if (e.dataTransfer.files.length) {
        e.preventDefault(); e.stopPropagation();
        const pos = cm.coordsChar({ left: e.clientX, top: e.clientY }); cm.setCursor(pos);
        const t = Editor.tab; uploadFiles(e.dataTransfer.files, attachDir(t)).then(insertEmbeds);
      }
    });
    this.cm.on("scroll", debounce(() => syncSplitScroll(), 30));
  },
  attach(t) {
    this.ensure();
    if (this.tab !== t || this.cm.getDoc() !== t.doc) {
      this.cm.swapDoc(t.doc);
      this.tab = t;
      const mode = t.kind === "md" ? { name: "gfm", highlightFormatting: true, fencedCodeBlockHighlighting: true } : modeForPath(t.path);
      this.cm.setOption("mode", mode);
      this.cm.setOption("lineNumbers", true);
    }
    this.host.classList.toggle("md-source", t.kind === "md");
    this.setOverlay(t.kind === "md");
    if (t.kind !== "md") { this.mdInfo = null; this.mdSig = null; }
    this.decorateBlocks(t);
    return this.host;
  },
  setOverlay(on) {
    if (!this.cm || !this.overlay) return;
    if (this.overlayAdded) {
      this.cm.removeOverlay(this.overlay);
      this.overlayAdded = false;
    }
    if (on) {
      this.cm.addOverlay(this.overlay);
      this.overlayAdded = true;
    }
  },
  decorateBlocks(t) {
    if (!this.cm || !t || t.kind !== "md") { this.mdInfo = null; return; }
    const cm = this.cm;
    const info = { frontmatter: null, fenced: new Set() };
    this.mdInfo = info;
    cm.operation(() => {
      cm.eachLine(line => cm.removeLineClass(line, "text", "cm-table-header-text"));
      const n = cm.lineCount();
      const lines = [];
      for (let i = 0; i < n; i++) lines.push(cm.getLine(i));
      if (lines[0] === "---") {
        const close = lines.indexOf("---", 1);
        if (close > 0) info.frontmatter = { start: 0, end: close };
      }
      let fence = null;
      const first = info.frontmatter ? info.frontmatter.end + 1 : 0;
      for (let i = first; i < n; i++) {
        const line = lines[i];
        if (fence) {
          info.fenced.add(i);
          const trim = line.trim();
          if (trim.startsWith(fence) && trim.replace(/[`~]/g, "") === "") fence = null;
          continue;
        }
        const open = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
        if (open) {
          fence = open[1];
          info.fenced.add(i);
          continue;
        }
        const next = lines[i + 1] || "";
        const delim = /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(next) && next.includes("-");
        if (line.includes("|") && delim) {
          cm.addLineClass(i, "text", "cm-table-header-text");
        }
      }
    });
    const sig = (info.frontmatter ? `${info.frontmatter.start}:${info.frontmatter.end}` : "-") +
      "|" + [...info.fenced].join(",");
    if (sig !== this.mdSig) {
      this.mdSig = sig;
      this.cm.refresh();
    }
  },
  refresh(t) {
    requestAnimationFrame(() => {
      if (!this.cm) return;
      this.cm.refresh();
      if (t.pendingLine) { const ln = t.pendingLine - 1; this.cm.setCursor({ line: ln, ch: 0 }); this.cm.scrollIntoView({ line: ln, ch: 0 }, 120); t.pendingLine = null; this.cm.focus(); }
      else if (t.focusEnd) { this.cm.setCursor(this.cm.lineCount(), 0); this.cm.focus(); t.focusEnd = false; }
      else if (t.scroll.edit != null) this.cm.scrollTo(null, t.scroll.edit);
    });
  },
  detach() { if (this.host && this.host.parentElement) this.host.remove(); },
};
function wrapSel(cm, m) {
  const s = cm.getSelection();
  if (s) cm.replaceSelection(m + s + m);
  else { const c = cm.getCursor(); cm.replaceRange(m + m, c); cm.setCursor({ line: c.line, ch: c.ch + m.length }); }
}
function linkHint(cm, force = false) {
  cm.showHint({
    completeSingle: false,
    hint(cm) {
      const cur = cm.getCursor(), line = cm.getLine(cur.line).slice(0, cur.ch);
      const m = /\[\[([^\]\[|#]*)$/.exec(line);
      if (!m && !force) return null;
      const typed = m ? m[1] : "";
      const from = { line: cur.line, ch: cur.ch - typed.length };
      const items = rankPaths(typed, S.files.map(f => f.path), 40).map(p => ({
        text: wikiName(p, Editor.tab && Editor.tab.path) + (m ? "]]" : ""),
        displayText: titleFor(p),
        render(elt, _d, cur) { elt.append(document.createTextNode(cur.displayText)); if (dirname(p)) elt.append(el("span", "hp", dirname(p))); },
      }));
      // Don't double the closing brackets that autoCloseBrackets added.
      const after = cm.getRange(cur, { line: cur.line, ch: cur.ch + 2 });
      const to = after === "]]" ? { line: cur.line, ch: cur.ch + 2 } : cur;
      return { list: items, from, to };
    },
  });
}
function insertEmbeds(paths) {
  if (!paths.length) return;
  const txt = paths.map(embedText).join("\n");
  const t = activeTab(), ta = t && t.live && t.live.ta;
  if (ta && ta.isConnected) { ta.focus(); document.execCommand("insertText", false, txt); return; }
  if (!Editor.cm) return;
  Editor.cm.replaceSelection(txt);
  Editor.cm.focus();
}
async function onPaste(cm, e) {
  const items = [...(e.clipboardData && e.clipboardData.items || [])].filter(i => i.kind === "file");
  if (!items.length) return;
  e.preventDefault();
  const files = [];
  for (const it of items) {
    const f = it.getAsFile(); if (!f) continue;
    const ext = (f.type.split("/")[1] || "bin").replace("jpeg", "jpg").replace(/[^a-z0-9]/g, "");
    const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
    files.push(new File([f], f.name && f.name !== "image.png" ? f.name : `Pasted image ${stamp}.${ext}`, { type: f.type }));
  }
  const t = activeTab();
  const folder = joinPath(attachDir(t), "attachments");
  insertEmbeds(await uploadFiles(files, folder));
}
function syncSplitScroll() {
  const t = activeTab();
  if (!t || t.mode !== "split" || !Editor.cm) return;
  const reader = $("view").querySelector(".vpane.reader");
  if (!reader) return;
  const info = Editor.cm.getScrollInfo();
  const ratio = info.top / Math.max(1, info.height - info.clientHeight);
  reader.scrollTop = ratio * (reader.scrollHeight - reader.clientHeight);
}

// Saving: debounced autosave, compare-and-swap against the version we loaded.
const autosave = new Map();
function onDocChange(t) {
  if (t.applyingRemote) return;
  const dirty = t.doc.getValue() !== t.saved;
  if (dirty !== t.dirty) { t.dirty = dirty; renderTabs(); }
  if (t.id === S.active) { updateSaveState(t); if (t.rerender) t.rerender(); }
  clearTimeout(autosave.get(t.id));
  if (dirty) autosave.set(t.id, setTimeout(() => saveTab(t), 1200));
}
async function saveTab(t, { create = false } = {}) {
  if (t.live && t.live.editing >= 0 && t.live.t === t) lvSyncOpen(t.live);
  if (!t.doc || t.saving || t.conflict) return;
  if (!t.dirty && !create) return;
  clearTimeout(autosave.get(t.id));
  t.saving = true; if (t.id === S.active) updateSaveState(t);
  const content = t.doc.getValue();
  try {
    const r = await apiJSON("/api/notes/file", "PUT", { path: t.path, content, base_hash: create ? null : t.hash });
    t.hash = r.hash; t.saved = content; t.saveError = null;
    t.meta = { ...(t.meta || {}), size: r.size, mtime: r.mtime, hash: r.hash };
    const f = S.fileMap.get(t.path); if (f) { f.mtime = r.mtime; f.size = r.size; }
    S.ownSaves.set(t.path, r.mtime);
    t.dirty = t.doc.getValue() !== t.saved;
    if (t.dirty) autosave.set(t.id, setTimeout(() => saveTab(t), 800));
  } catch (e) {
    if (e.status === 409 && e.body && e.body.current_hash) {
      t.conflict = { hash: e.body.current_hash, content: e.body.current_content };
    } else if (e.status === 409 && e.body && e.body.deleted) {
      t.deleted = true;
    } else t.saveError = e.message;
  } finally {
    t.saving = false;
    renderTabs();
    if (t.id === S.active) { updateSaveState(t); renderBanner(t); }
  }
}
async function saveActiveNow() { const t = activeTab(); if (t && t.type === "file") { await saveTab(t); if (!t.dirty && !t.conflict) toast("Saved", "ok"); } }
async function flushSaves() { for (const t of S.tabs) if (t.dirty) await saveTab(t); }
window.addEventListener("beforeunload", e => { if (S.tabs.some(t => t.dirty)) { flushSaves(); e.preventDefault(); e.returnValue = ""; } });

// ── Line diff (for conflicts) ──────────────────────────────────────────────────
function lineDiff(a, b) {
  const A = a.split("\n"), B = b.split("\n");
  const n = A.length, m = B.length;
  if (n * m > 4e6) return null;
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push([" ", A[i]]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) out.push(["-", A[i++]]);
    else out.push(["+", B[j++]]);
  }
  while (i < n) out.push(["-", A[i++]]);
  while (j < m) out.push(["+", B[j++]]);
  return out;
}
function diffLinesEl(ops) {
  const body = el("div", "diff-body");
  for (const [op, text] of ops) {
    const d = el("div", "dl" + (op === "+" ? " add" : op === "-" ? " del" : ""));
    d.append(el("span", "n", op === " " ? "" : op), el("span", "t", text || " "));
    body.append(d);
  }
  return body;
}
async function keepMine(t) { t.hash = t.conflict.hash; t.conflict = null; t.dirty = true; await saveTab(t); renderBanner(t); }
async function takeTheirs(t) { t.conflict = null; await reloadTab(t, { quiet: true }); }
async function showMerge(t) {
  const theirs = t.conflict.content ?? "";
  const mine = t.doc.getValue();
  const ops = lineDiff(theirs, mine);
  const back = el("div", "ml-dialog-backdrop");
  const dlg = el("div", "ml-dialog merge-dialog");
  dlg.append(el("div", "ml-dialog-message", `${t.path} changed on another machine. Lines in green are only in your version, red only in theirs.`));
  const cols = el("div");
  if (ops) cols.append(diffLinesEl(ops)); else cols.append(el("div", "", "Too large to compare here."));
  cols.firstChild.style.cssText = "max-height:60vh;overflow:auto;border:1px solid var(--border);border-radius:4px;background:var(--doc-bg)";
  const acts = el("div", "ml-dialog-actions");
  const b = (label, cls, fn) => { const x = el("button", "ml-btn " + cls, label); x.onclick = () => { back.remove(); fn && fn(); }; return x; };
  acts.append(
    b("Cancel", "ml-btn-cancel"),
    b("Take theirs", "ml-btn-cancel", () => takeTheirs(t)),
    b("Keep both (merge lines)", "ml-btn-cancel", async () => {
      const merged = ops ? ops.map(o => o[1]).join("\n") : mine + "\n\n" + theirs;
      t.hash = t.conflict.hash; t.conflict = null; t.doc.setValue(merged); t.dirty = true; await saveTab(t); renderBanner(t);
      toast("Merged both versions; review the duplicated lines", "ok");
    }),
    b("Keep mine", "ml-btn-ok", () => keepMine(t)),
  );
  dlg.append(cols, acts);
  back.append(dlg);
  back.addEventListener("keydown", e => { if (e.key === "Escape") back.remove(); });
  document.body.append(back);
  acts.lastChild.focus();
}
