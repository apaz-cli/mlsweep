// mlsweep Notes: side panels, search, changes, lint, graph, switcher, startup.
"use strict";

// ── Left panes ─────────────────────────────────────────────────────────────────
function switchPane(name, { reveal = true } = {}) {
  // Asking for a pane shows the panel, except when restoring the last one at startup.
  if (reveal && document.body.classList.contains("left-collapsed")) toggleLeft();
  document.querySelectorAll(".ltab").forEach(b => b.classList.toggle("active", b.dataset.pane === name));
  document.querySelectorAll(".lpane").forEach(p => p.classList.toggle("active", p.id === "pane-" + name));
  store.set("pane", name);
  if (name === "changes") loadFeed(true);
  if (name === "search") setTimeout(() => $("search-input").focus(), 0);
  if (name === "lint" && !S.lint) runLint();
}
document.querySelectorAll(".ltab").forEach(b => b.onclick = () => switchPane(b.dataset.pane));

// ── Fuzzy ranking (switcher, link hints) ───────────────────────────────────────
function fuzzyScore(query, text) {
  if (!query) return 1;
  const ql = query.toLowerCase(), tl = text.toLowerCase();
  const idx = tl.indexOf(ql);
  if (idx >= 0) return 1000 - idx - text.length * 0.1 + (idx === 0 || /[\/ _-]/.test(tl[idx - 1]) ? 200 : 0);
  let ti = 0, score = 0, streak = 0;
  for (const c of ql) {
    const f = tl.indexOf(c, ti);
    if (f < 0) return -1;
    streak = f === ti ? streak + 1 : 0;
    score += 10 + streak * 5 - (f - ti) * 0.5 + (f === 0 || /[\/ _-]/.test(tl[f - 1]) ? 8 : 0);
    ti = f + 1;
  }
  return score - text.length * 0.1;
}
function rankPaths(query, paths, limit = 50) {
  const scored = [];
  for (const p of paths) {
    const s = Math.max(fuzzyScore(query, titleFor(p)) + 50, fuzzyScore(query, p));
    if (s > 0 || !query) scored.push([s + (isMd(p) ? 5 : 0), p]);
  }
  scored.sort((a, b) => b[0] - a[0] || a[1].localeCompare(b[1]));
  return scored.slice(0, limit).map(x => x[1]);
}
function highlightMatch(text, query) {
  const frag = document.createDocumentFragment();
  if (!query) { frag.append(text); return frag; }
  const tl = text.toLowerCase(), ql = query.toLowerCase();
  const idx = tl.indexOf(ql);
  if (idx >= 0) { frag.append(text.slice(0, idx), el("b", "", text.slice(idx, idx + ql.length)), text.slice(idx + ql.length)); return frag; }
  let ti = 0;
  for (const c of ql) {
    const f = tl.indexOf(c, ti);
    if (f < 0) break;
    frag.append(text.slice(ti, f), el("b", "", text[f]));
    ti = f + 1;
  }
  frag.append(text.slice(ti));
  return frag;
}

// ── Quick switcher ─────────────────────────────────────────────────────────────
const Switcher = {
  items: [], sel: 0,
  open() {
    $("switcher-back").classList.add("show");
    const inp = $("switcher-input"); inp.value = ""; inp.focus();
    this.update();
  },
  close() { $("switcher-back").classList.remove("show"); },
  update() {
    const qv = $("switcher-input").value.trim();
    const recent = S.recent.filter(p => S.fileMap.has(p));
    const paths = qv ? rankPaths(qv, S.files.map(f => f.path), 60) : [...recent, ...S.files.map(f => f.path).filter(p => !recent.includes(p))].slice(0, 60);
    this.items = paths; this.sel = 0;
    const list = $("switcher-list"); list.innerHTML = "";
    if (!paths.length) list.append(el("div", "sw-item", qv ? `No match. Shift+Enter creates "${qv}".` : "No files yet."));
    paths.forEach((p, i) => {
      const row = el("div", "sw-item" + (i === 0 ? " sel" : ""));
      const nm = el("span", "nm"); nm.append(highlightMatch(titleFor(p), qv));
      row.append(nm, el("span", "pth", dirname(p) ? dirname(p) + "/" : ""));
      row.onmousedown = e => { e.preventDefault(); this.close(); openFile(p, { newTab: e.ctrlKey || e.metaKey }); };
      list.append(row);
    });
  },
  move(d) {
    if (!this.items.length) return;
    this.sel = (this.sel + d + this.items.length) % this.items.length;
    [...$("switcher-list").children].forEach((c, i) => c.classList.toggle("sel", i === this.sel));
    $("switcher-list").children[this.sel].scrollIntoView({ block: "nearest" });
  },
};
$("switcher-input").addEventListener("input", () => Switcher.update());
$("switcher-input").addEventListener("keydown", e => {
  if (e.key === "ArrowDown") { e.preventDefault(); Switcher.move(1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); Switcher.move(-1); }
  else if (e.key === "Escape") Switcher.close();
  else if (e.key === "Enter") {
    e.preventDefault();
    const v = $("switcher-input").value.trim();
    Switcher.close();
    if (e.shiftKey || !Switcher.items.length) { if (v) newNote(dirname(v), basename(v), { openExisting: true }); }
    else openFile(Switcher.items[Switcher.sel], { newTab: e.ctrlKey || e.metaKey });
  }
});
$("switcher-back").addEventListener("mousedown", e => { if (e.target.id === "switcher-back") Switcher.close(); });

// ── Search ─────────────────────────────────────────────────────────────────────
let searchSeq = 0;
const doSearch = debounce(async () => {
  const qv = $("search-input").value.trim();
  const out = $("search-results");
  const seq = ++searchSeq;
  if (!qv) { out.innerHTML = ""; $("search-note").textContent = ""; return; }
  $("search-note").textContent = "Searching…";
  try {
    const r = await api(`/api/notes/search?q=${q(qv)}`);
    if (seq !== searchSeq) return;
    out.innerHTML = "";
    $("search-note").textContent = r.results.length ? `${r.results.length} file${r.results.length === 1 ? "" : "s"}` : "No matches";
    const terms = (qv.match(/"[^"]+"|\S+/g) || []).map(t => t.replace(/"/g, "").toLowerCase()).filter(Boolean);
    for (const res of r.results) {
      const box = el("div", "sr");
      const head = el("div", "sr-path");
      const ic = el("span", "ficon"); ic.innerHTML = iconFor(res.path); ic.querySelector("svg").style.cssText = "width:12px;height:12px";
      head.append(ic, el("span", "", titleFor(res.path)));
      if (dirname(res.path)) head.append(el("span", "dir", dirname(res.path)));
      box.append(head);
      // An open tab keeps its view (Live, Source or Split); a new one opens Live.
      const isOpen = () => S.tabs.some(t => t.type === "file" && t.path === res.path);
      box.onclick = () => openFile(res.path, { mode: !isOpen() && res.kind === "md" ? "live" : null });
      box.oncontextmenu = e => fileMenu(e, res.path);
      for (const m of res.matches.slice(0, 5)) {
        const line = el("div", "sr-line");
        line.append(el("span", "ln", m.line));
        line.append(markTerms(m.text.trim(), terms));
        line.onclick = e => {
          e.stopPropagation();
          openFile(res.path, { mode: isOpen() ? null : res.kind === "md" ? "live" : "source", line: m.line, hits: terms });
        };
        box.append(line);
      }
      out.append(box);
    }
  } catch (e) { $("search-note").textContent = "Search failed: " + e.message; }
}, 220);
function markTerms(text, terms) {
  const frag = document.createDocumentFragment();
  if (!terms.length) { frag.append(text); return frag; }
  const re = new RegExp(terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "gi");
  let last = 0, m;
  while ((m = re.exec(text))) { frag.append(text.slice(last, m.index), el("mark", "", m[0])); last = m.index + m[0].length; if (!m[0].length) re.lastIndex++; }
  frag.append(text.slice(last));
  return frag;
}
$("search-input").addEventListener("input", doSearch);
function searchFor(text) { switchPane("search"); $("search-input").value = text; doSearch(); }

// ── Changes feed ───────────────────────────────────────────────────────────────
let feedSkip = 0, feedVersion = "", feedSeq = 0;
async function loadFeed(reset = false) {
  const feed = $("feed");
  if (reset) { if (feedVersion === S.version && feed.childElementCount) return; feedSkip = 0; feed.innerHTML = ""; feedVersion = S.version; }
  const seq = ++feedSeq;
  const skip = feedSkip;
  try {
    const r = await api(`/api/notes/history?limit=40&skip=${skip}`);
    // A reset or a newer page started while this one was in flight; appending
    // it now would duplicate or misorder the feed.
    if (seq !== feedSeq) return;
    const old = feed.querySelector(".load-more"); if (old) old.remove();
    if (!r.commits.length && !feedSkip) { feed.innerHTML = `<div class="tree-empty">No history yet.</div>`; return; }
    let lastDay = feed.dataset.lastDay || "";
    for (const c of r.commits) {
      const day = dayLabel(c.time);
      if (day !== lastDay) { feed.append(el("div", "feed-day", day)); lastDay = day; }
      const box = el("div", "commit");
      const head = el("div", "commit-head");
      const t = el("span", "commit-time", new Date(c.time * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
      t.title = new Date(c.time * 1000).toLocaleString() + "\n" + c.hash;
      head.append(avatar(c.author), el("span", "commit-author", c.author), t);
      head.style.cursor = "pointer"; head.title = c.subject + "\nClick for the full diff";
      head.onclick = () => openDiff(c, null);
      head.oncontextmenu = e => commitMenu(e, c);
      box.append(head);
      const files = el("div", "commit-files");
      for (const f of (c.files || []).slice(0, 12)) {
        const row = el("div", "cf");
        row.append(el("span", "st st-" + f.status, f.status), el("span", "p", f.path));
        const db = el("span", "diffbtn", "diff"); db.title = "Show this file's change";
        db.onclick = e => { e.stopPropagation(); openDiff(c, f.path); };
        row.append(db);
        row.title = f.path;
        row.onclick = () => f.status === "D" ? openDiff(c, f.path) : openFile(f.path);
        row.oncontextmenu = e => fileMenu(e, f.path, [
          { label: "Show this change", run: () => openDiff(c, f.path) },
          { label: "View this version", run: () => showVersion(f.path, c.hash, c) },
          ...commitItems(c)]);
        files.append(row);
      }
      if ((c.files || []).length > 12) files.append(el("div", "cf", `… ${c.files.length - 12} more`));
      box.append(files);
      feed.append(box);
    }
    feed.dataset.lastDay = lastDay;
    feedSkip = skip + r.commits.length;
    if (r.commits.length === 40) {
      const more = el("button", "mini-btn load-more", "Load older changes");
      more.onclick = () => loadFeed(false);
      feed.append(more);
    }
  } catch (e) {
    if (seq !== feedSeq) return;
    feed.innerHTML = "";
    feed.append(el("div", "tree-empty", "Could not load history: " + e.message));
  }
}

// A commit's own entries, for the menus on a change.
function commitItems(c) {
  return [
    { label: "Show the whole commit", run: () => openDiff(c, null) },
    { label: "Copy commit hash", run: () => copyText(c.hash) },
  ];
}
function commitMenu(e, c) {
  e.preventDefault();
  mlMenuAt(e.clientX, e.clientY, [...commitItems(c), { label: "Copy message", run: () => copyText(c.subject || "") }]);
}

// ── Diff view ──────────────────────────────────────────────────────────────────
// *path* is the file's name in that commit; *current* its name now (they differ after a rename).
function openDiff(commit, path, current = null) {
  let t = S.tabs.find(x => x.type === "diff" && x.rev === commit.hash && x.path === path);
  if (!t) {
    t = newTab({ type: "diff", rev: commit.hash, path, current: current || path, commit, title: `${path ? titleFor(path) : "Commit"} @ ${commit.short || commit.hash.slice(0, 7)}` });
    const g = focusedGroup(), cur = activeTab();
    addTab(t, g, cur ? g.tabs.indexOf(cur.id) + 1 : g.tabs.length);
  }
  activate(t.id);
}
async function renderDiffView(view, t) {
  const pane = el("div", "vpane");
  const wrap = el("div", "diff-view");
  pane.append(wrap); view.append(pane);
  const c = t.commit || {};
  const head = el("div", "diff-head");
  head.append(avatar(c.author || "?"));
  const txt = el("div");
  txt.append(el("div", "subj", c.subject || t.rev), el("div", "meta", `${c.author || ""} · ${c.time ? new Date(c.time * 1000).toLocaleString() : ""} · ${t.rev.slice(0, 10)}`));
  head.append(txt);
  if (t.path) {
    const spacer = el("span"); spacer.style.flex = "1"; head.append(spacer);
    const viewBtn = el("button", "mini-btn", "View this version");
    viewBtn.onclick = () => showVersion(t.path, t.rev, c);
    const restore = el("button", "mini-btn primary", "Restore this version");
    restore.title = `Write this version to ${t.current || t.path}`;
    restore.onclick = () => restoreVersion(t.path, t.rev, c, t.current || t.path);
    head.append(viewBtn, restore);
  }
  wrap.append(head);
  try {
    const r = await api(`/api/notes/diff?rev=${q(t.rev)}${t.path ? "&path=" + q(t.path) : ""}`);
    renderUnifiedDiff(wrap, r.diff);
  } catch (e) { wrap.append(el("div", "pane-note", "Could not load the diff: " + e.message)); }
}
function renderUnifiedDiff(wrap, diff) {
  if (!diff.trim()) { wrap.append(el("div", "pane-note", "No textual changes.")); return; }
  let box = null, body = null, oldN = 0, newN = 0;
  for (const line of diff.replace(/\n$/, "").split("\n")) {
    if (line.startsWith("diff --git")) {
      box = el("div", "diff-file"); const fh = el("div", "diff-file-head");
      const m = / b\/(.*)$/.exec(line); const p = m ? m[1] : line;
      const pe = el("span", "p", p); pe.onclick = () => S.fileMap.has(p) && openFile(p);
      fh.append(pe); box.append(fh); body = el("div", "diff-body"); box.append(body); wrap.append(box);
      continue;
    }
    if (!body) continue;
    if (/^(index |--- |\+\+\+ |new file|deleted file|similarity|rename |old mode|new mode)/.test(line)) {
      if (/^(new file|deleted file|rename )/.test(line)) box.querySelector(".diff-file-head").append(el("span", "pane-note", line));
      continue;
    }
    if (line.startsWith("Binary files")) { body.append(el("div", "dl hunk", line)); continue; }
    const d = el("div", "dl");
    if (line.startsWith("@@")) {
      const m = /@@ -(\d+)(?:,\d+)? \+(\d+)/.exec(line); if (m) { oldN = +m[1]; newN = +m[2]; }
      d.className = "dl hunk"; d.append(el("span", "n", ""), el("span", "t", line));
    } else if (line.startsWith("+")) { d.className = "dl add"; d.append(el("span", "n", newN++), el("span", "t", line.slice(1) || " ")); }
    else if (line.startsWith("-")) { d.className = "dl del"; d.append(el("span", "n", oldN++), el("span", "t", line.slice(1) || " ")); }
    else if (line.startsWith("\\")) continue;
    else { d.append(el("span", "n", newN++), el("span", "t", line.slice(1) || " ")); oldN++; }
    body.append(d);
  }
}
async function showVersion(path, rev, c) {
  try {
    const r = await api(`/api/notes/show?path=${q(path)}&rev=${q(rev)}`);
    if (r.binary) { toast("Binary file; use Restore to bring it back", "err"); return; }
    const t = newTab({ type: "version", path, rev, commit: c, content: r.content, title: `${titleFor(path)} @ ${rev.slice(0, 7)}` });
    const g = focusedGroup(), cur = activeTab();
    addTab(t, g, cur ? g.tabs.indexOf(cur.id) + 1 : g.tabs.length);
    activate(t.id);
  } catch (e) { toast(e.message, "err"); }
}
// A read-only old version of a file.
function renderVersionView(view, t) {
  const c = t.commit, path = t.path;
  const pane = el("div", "vpane reader"); const doc = el("article", "doc");
  const note = el("div", "callout c-quote"); note.append(el("div", "callout-title", `Version of ${path} from ${c && c.time ? new Date(c.time * 1000).toLocaleString() : t.rev.slice(0, 10)} — read only`));
  doc.append(note);
  const inner = el("div"); doc.append(inner);
  if (isMd(path)) renderMarkdown(t.content, path, inner, { depth: 1, footnotePrefix: "p" + hashStr(path) + "-" }); else { const pre = el("pre"); pre.append(el("code", "", t.content)); inner.append(pre); }
  pane.append(doc); view.append(pane);
}
async function restoreVersion(srcPath, rev, c, path = srcPath) {
  if (!(await mlConfirm(`Restore ${path} to the version from ${c && c.time ? new Date(c.time * 1000).toLocaleString() : rev.slice(0, 10)}?\n\nThe current version stays in history.`, { confirmLabel: "Restore", danger: false }))) return;
  try {
    // An autosave that lands after this PUT would undo the restore.
    await flushSaves();
    const r = await api(`/api/notes/show?path=${q(srcPath)}&rev=${q(rev)}`);
    if (r.binary) { toast("Restoring binary files is not supported here", "err"); return; }
    let base = null;
    if (S.fileMap.has(path)) { const cur = await api(`/api/notes/file?path=${q(path)}`); base = cur.hash; }
    await apiJSON("/api/notes/file", "PUT", { path, content: r.content, base_hash: base });
    await loadTree();
    const open = S.tabs.find(t => t.type === "file" && t.path === path);
    if (open) await reloadTab(open, { quiet: true });
    openFile(path);
    toast("Restored", "ok");
  } catch (e) { toast("Restore failed: " + e.message, "err"); }
}

// ── Right panel ────────────────────────────────────────────────────────────────
const closedSecs = new Set(store.get("closedSecs", []));
function rsec(id, title, count) {
  const sec = el("div", "rsec" + (closedSecs.has(id) ? " closed" : ""));
  sec.dataset.sec = id;
  const head = el("div", "rsec-head");
  head.append(document.createTextNode(title));
  if (count != null) head.append(el("span", "n", String(count)));
  head.append(el("span", "chev", "▼"));
  head.onclick = () => { sec.classList.toggle("closed"); sec.classList.contains("closed") ? closedSecs.add(id) : closedSecs.delete(id); store.set("closedSecs", [...closedSecs]); };
  const body = el("div", "rsec-body");
  sec.append(head, body);
  return [sec, body];
}
async function openRightSection(id) {
  if (document.body.classList.contains("right-collapsed")) {
    document.body.classList.remove("right-collapsed"); store.set("rightCollapsed", false);
    await refreshRight();
  }
  const sec = document.querySelector(`.rsec[data-sec="${id}"]`);
  if (sec) { sec.classList.remove("closed"); closedSecs.delete(id); sec.scrollIntoView({ block: "start" }); }
}
let rightSeq = 0;
async function refreshRight() {
  const body = $("right-body");
  const t = activeTab();
  const seq = ++rightSeq;
  if (!S.enabled) { body.innerHTML = ""; return; }
  if (document.body.classList.contains("right-collapsed")) return;   // btn-right refreshes it when opened
  if (!t || t.type !== "file") {
    body.innerHTML = "";
    const idx = await ensureIndex();
    if (seq !== rightSeq) return;
    const [sec, b] = rsec("vault", "Folder");
    const mdCount = S.files.filter(f => f.kind === "md").length;
    const exps = Object.keys(idx.mentions).length;
    const info = el("div", "file-info");
    info.append(el("span", "k", "Notes"), el("span", "", mdCount), el("span", "k", "Files"), el("span", "", S.files.length),
      el("span", "k", "Experiments cited"), el("span", "", exps));
    if (S.status && S.status.remote) info.append(el("span", "k", "Remote"), el("span", "", S.status.remote));
    b.append(info); body.append(sec);
    const cited = Object.entries(idx.mentions).sort((a, b) => b[1].length - a[1].length).slice(0, 25);
    const [s2, b2] = rsec("cited", "Most cited experiments", cited.length);
    if (!cited.length) b2.append(el("div", "rempty", "No note mentions an experiment yet."));
    for (const [id, paths] of cited) b2.append(expRow(id, `${paths.length} note${paths.length === 1 ? "" : "s"}`, () => searchFor(id)));
    body.append(s2);
    return;
  }
  const idx = t.kind === "md" ? await ensureIndex() : null;
  if (seq !== rightSeq) return;
  body.innerHTML = "";
  const path = t.path;

  if (t.kind === "md") {
    // Experiments this note mentions.
    const exps = Object.entries(idx.mentions).filter(([, ps]) => ps.includes(path)).map(([id]) => id);
    const [es, eb] = rsec("exps", "Experiments", exps.length);
    if (!exps.length) eb.append(el("div", "rempty", "No experiment IDs in this note."));
    for (const id of exps) eb.append(expRow(id));
    body.append(es);

    const [os, ob] = rsec("outline", "Outline");
    ob.id = "outline-body";
    body.append(os);
    refreshOutline(t);

    // Backlinks.
    const back = [];
    for (const [src, links] of Object.entries(idx.links)) {
      if (src === path) continue;
      for (const l of links) if (l.resolved === path) back.push({ src, ...l });
    }
    const bySrc = new Map();
    for (const b of back) { if (!bySrc.has(b.src)) bySrc.set(b.src, []); bySrc.get(b.src).push(b); }
    const [bs, bb] = rsec("backlinks", "Backlinks", bySrc.size);
    if (!bySrc.size) bb.append(el("div", "rempty", "Nothing links here yet."));
    for (const [src, ls] of bySrc) {
      const it = el("div", "ritem");
      it.append(el("div", "t", titleFor(src)));
      for (const l of ls.slice(0, 2)) it.append(el("div", "ctx", l.context || ""));
      it.title = src;
      it.onclick = () => openFile(src, { line: ls[0].line });
      it.oncontextmenu = e => fileMenu(e, src);
      bb.append(it);
    }
    body.append(bs);

    // Outgoing links.
    const out = idx.links[path] || [];
    const seen = new Set();
    const [ls, lb] = rsec("outgoing", "Outgoing links", out.length);
    if (!out.length) lb.append(el("div", "rempty", "No links."));
    for (const l of out) {
      const key = l.resolved || "?" + l.target;
      if (seen.has(key)) continue; seen.add(key);
      const it = el("div", "ritem");
      const tt = el("div", "t" + (l.resolved ? "" : " missing"), l.resolved ? titleFor(l.resolved) : l.target);
      it.append(tt); it.title = l.resolved || `Missing: ${l.target}`;
      it.onclick = () => l.resolved ? openFile(l.resolved) : newNote(dirname(path), basename(l.target));
      it.oncontextmenu = e => {
        if (l.resolved) return fileMenu(e, l.resolved);
        e.preventDefault();
        mlMenuAt(e.clientX, e.clientY, [
          { label: "Create this note", run: () => newNote(dirname(path), basename(l.target)) },
          { label: "Copy link target", run: () => copyText(l.target) },
        ]);
      };
      lb.append(it);
    }
    body.append(ls);
  } else {
    const [fs, fb] = rsec("info", "File");
    const info = el("div", "file-info");
    const m = t.meta || {};
    info.append(el("span", "k", "Size"), el("span", "", fmtSize(m.size)), el("span", "k", "Modified"), el("span", "", m.mtime ? ago(m.mtime) : ""), el("span", "k", "Path"), el("span", "", path));
    fb.append(info);
    const dl = el("a", "mini-btn", "Download"); dl.href = rawURL(path, "&download=1"); dl.download = basename(path); dl.style.margin = "4px 12px"; dl.style.display = "inline-block";
    fb.append(dl);
    body.append(fs);
    const idx2 = await ensureIndex();
    if (seq !== rightSeq) return;
    const users = Object.entries(idx2.links).filter(([, ls]) => ls.some(l => l.resolved === path)).map(([s]) => s);
    const [us, ub] = rsec("usedby", "Linked from", users.length);
    if (!users.length) ub.append(el("div", "rempty", "No note links to this file."));
    for (const s of users) {
      const it = el("div", "ritem"); it.append(el("div", "t", titleFor(s)));
      it.onclick = () => openFile(s); it.oncontextmenu = e => fileMenu(e, s); ub.append(it);
    }
    body.append(us);
  }

  // History.
  const [hs, hb] = rsec("history", "History");
  body.append(hs);
  try {
    const r = await api(`/api/notes/history?path=${q(path)}&limit=25`);
    if (seq !== rightSeq) return;
    hs.querySelector(".rsec-head").insertBefore(el("span", "n", r.commits.length + (r.commits.length === 25 ? "+" : "")), hs.querySelector(".chev"));
    if (!r.commits.length) hb.append(el("div", "rempty", "Not committed yet; the next sync records it."));
    for (const c of r.commits) {
      const it = el("div", "hist-item");
      const b = el("div", "body");
      b.append(el("div", "who", c.author), el("div", "when", `${ago(c.time)} · ${c.short}`));
      it.append(avatar(c.author), b);
      it.title = `${new Date(c.time * 1000).toLocaleString()}\n${c.subject}`;
      if (c.path && c.path !== path) it.title += `\n(then named ${c.path})`;
      it.onclick = () => openDiff(c, c.path || path, path);
      it.oncontextmenu = e => {
        e.preventDefault();
        const then = c.path || path;
        mlMenuAt(e.clientX, e.clientY, [
          { label: "Show this change", run: () => openDiff(c, then, path) },
          { label: "View this version", run: () => showVersion(then, c.hash, c) },
          { label: "Restore this version…", run: () => restoreVersion(then, c.hash, c, path) },
          "-", ...commitItems(c),
        ]);
      };
      hb.append(it);
    }
  } catch { hb.append(el("div", "rempty", "History unavailable.")); }
}
function expRow(id, note, onclick) {
  const m = S.experiments.get(id) || {};
  const a = el("a", "exp-row");
  a.href = pageHref("results.html", { experiment: id });
  a.append(el("span", "badge badge-" + (m.status || "unknown"), m.status || "?"),
           el("span", "id", m.name || id));
  if (note) a.append(el("span", "commit-time", note));
  a.title = `${id}${m.campaign ? " · campaign " + m.campaign : ""}${m.jobs ? `\n${m.jobs.done || 0}/${m.jobs.total || 0} runs done, ${m.jobs.failed || 0} failed` : ""}`;
  if (onclick) a.onclick = e => { e.preventDefault(); onclick(); };
  return a;
}
function refreshOutline(t) {
  const ob = $("outline-body");
  if (!ob || !t || !t.doc) return;
  ob.innerHTML = "";
  const lines = t.doc.getValue().split("\n");
  let n = 0;
  forEachProseLine(lines, (line, i) => {
    const m = HEADING_RE.exec(line);
    if (!m) return;
    n++;
    const title = m[2].replace(/\s*\{#[^}]+\}\s*$/, "");
    const it = el("div", "outline-item", title);
    it.style.paddingLeft = (12 + (m[1].length - 1) * 12) + "px";
    it.onclick = () => {
      const g = groupOf(t);
      if (!g) return;
      const reader = g.view.querySelector(".vpane.reader"), cm = g.editor.tab === t && g.editor.cm;
      if (t.mode !== "source" && reader) scrollToHeading(reader, title);
      if (t.mode !== "live" && cm) { cm.setCursor({ line: i, ch: 0 }); cm.scrollIntoView({ line: i, ch: 0 }, 80); cm.focus(); }
    };
    ob.append(it);
  });
  if (!n) ob.append(el("div", "rempty", "No headings."));
}
function toggleLeft() {
  // With no tab open the tree is all there is, so it stays.
  if (!S.tabs.length && !document.body.classList.contains("left-collapsed")) return;
  const c = document.body.classList.toggle("left-collapsed");
  store.set("leftCollapsed", c);
  eachEditor(ed => ed.cm.refresh());
}
function toggleRight() {
  const c = document.body.classList.toggle("right-collapsed");
  store.set("rightCollapsed", c);
  if (!c) refreshRight();
  eachEditor(ed => ed.cm.refresh());
}
function moreMenu(e) {
  const t = activeTab(); if (!t || !t.path) return;
  const r = e.currentTarget.getBoundingClientRect();
  mlMenuAt(r.right - 200, r.bottom + 4, [
    { label: "Rename…", run: () => { const row = treeRow(t.path); if (row) startRename(row); else moveDialog(t.path); } },
    { label: "Move to…", run: () => moveDialog(t.path) },
    { label: "Reveal in file tree", run: () => { switchPane("files"); revealInTree(t.path); } },
    { label: "Show history", run: () => openRightSection("history") },
    { label: "Local graph", run: () => openGraph(t.path) },
    { label: "Find mentions of this note", run: () => searchFor(stemOf(t.path)) },
    "-",
    { label: "Split right", hint: "\\", run: () => splitTab(t, "right") },
    { label: "Split down", hint: "|", run: () => splitTab(t, "down") },
    "-",
    { label: "Copy path", run: () => copyText(t.path) },
    { label: "Copy wikilink", run: () => copyText(`[[${stemOf(t.path)}]]`) },
    { label: "Download", run: () => downloadFile(t.path) },
    { label: (document.body.classList.contains("wide-doc") ? "Readable" : "Full") + " line width", run: () => { const w = document.body.classList.toggle("wide-doc"); store.set("wide", w); eachEditor(ed => ed.cm.refresh()); } },
    "-",
    { label: "Delete…", danger: true, run: () => deletePath(t.path, false) },
  ]);
}

// ── Lint ───────────────────────────────────────────────────────────────────────
let linting = false;
async function runLint() {
  if (linting) return;
  linting = true;
  $("lint-run").disabled = true; $("lint-run").textContent = "Running…";
  $("lint-summary").textContent = "Checking links, experiment and run IDs, chart specs, evidence numbers, code links and merge leftovers…";
  try {
    S.lint = await api("/api/notes/lint");
    renderLint();
  } catch (e) { $("lint-summary").textContent = "Lint failed: " + e.message; }
  finally { linting = false; $("lint-run").disabled = false; $("lint-run").textContent = "Run lint"; }
}
function renderLint() {
  const L = S.lint; if (!L) return;
  const body = $("lint-body"); body.innerHTML = "";
  const fixes = L.findings.filter(f => f.tier === "fix");
  const reports = L.findings.filter(f => f.tier === "report");
  const st = L.stats;
  $("lint-summary").textContent = `${st.notes} notes, ${st.links} links, ${st.experiments_cited} experiments cited` +
    (st.data_blocks ? `, ${st.data_blocks} data block${st.data_blocks === 1 ? "" : "s"}` : "") +
    (st.evidence_refs ? `, ${st.evidence_refs} evidence link${st.evidence_refs === 1 ? "" : "s"}` : "") + ". " +
    (L.findings.length ? `${fixes.length} safe fix${fixes.length === 1 ? "" : "es"}, ${reports.length} report${reports.length === 1 ? "" : "s"}.` : "") +
    (st.experiments_checked ? (st.runs_checked ? "" : " Run IDs not checked.") : " Experiment and run IDs not checked.") +
    (st.evidence_checked ? "" : " Evidence numbers not checked.");
  const count = $("lint-count");
  const real = reports.filter(f => !(f.detail && f.detail.missing_note)).length + fixes.length;
  count.textContent = real; count.className = "count" + (real ? (reports.some(f => f.kind === "conflict-marker") ? "" : " warn") : " zero");
  $("lint-fix").disabled = !fixes.length;
  $("lint-fix").textContent = fixes.length ? `Apply ${fixes.length} safe fix${fixes.length === 1 ? "" : "es"}` : "Apply safe fixes";
  if (!L.findings.length) body.append(el("div", "ok-msg", "✓ No findings."));
  const group = (title, items) => {
    if (!items.length) return;
    body.append(el("div", "tier-title", title));
    const byKind = new Map();
    for (const f of items) { if (!byKind.has(f.kind)) byKind.set(f.kind, []); byKind.get(f.kind).push(f); }
    for (const [kind, fs] of byKind) {
      const g = el("div", "lint-group");
      const h = el("div", "lint-group-head");
      h.append(el("span", "", L.kinds[kind] || kind), el("span", "n", fs.length), el("span", "tier" + (fs[0].tier === "fix" ? " fix" : ""), fs[0].tier === "fix" ? "safe fix" : "report"));
      const list = el("div");
      h.onclick = () => list.style.display = list.style.display === "none" ? "" : "none";
      g.append(h, list);
      for (const f of fs.slice(0, 200)) {
        const it = el("div", "lint-item");
        it.append(el("div", "loc", f.line ? `${f.path}:${f.line}` : f.path), el("div", "msg", f.message));
        if (f.detail && f.detail.context) it.append(el("div", "ctx", f.detail.context));
        it.onclick = () => S.fileMap.has(f.path) && openFile(f.path, { mode: isMd(f.path) && f.line ? "split" : null, line: f.line || null });
        it.oncontextmenu = e => fileMenu(e, f.path, [
          { label: "Copy finding", run: () => copyText(`${f.line ? `${f.path}:${f.line}` : f.path}: ${f.message}`) }]);
        list.append(it);
      }
      body.append(g);
    }
  };
  group("Tier 1 · safe fixes", fixes);
  group("Tier 2 · reports (judge each)", reports);
  // Tier 3: reviews written by an independent agent.
  const reviews = S.files.filter(f => f.path.startsWith("reviews/")).sort((a, b) => b.mtime - a.mtime);
  body.append(el("div", "tier-title", "Tier 3 · reviews"));
  const hint = el("div", "lint-item"); hint.style.cursor = "default";
  hint.append(el("div", "msg", "Judgment is left to an independent agent. Copy the review prompt, give it to a fresh agent with access to this folder and the manager; it writes its findings to reviews/."));
  body.append(hint);
  for (const r of reviews.slice(0, 20)) {
    const it = el("div", "review-file");
    const ic = el("span", "ficon"); ic.innerHTML = ICONS.md; ic.querySelector("svg").style.cssText = "width:12px;height:12px";
    it.append(ic, el("span", "", titleFor(r.path)), el("span", "commit-time", ago(r.mtime)));
    it.onclick = () => openFile(r.path);
    it.oncontextmenu = e => fileMenu(e, r.path);
    body.append(it);
  }
}
$("lint-run").onclick = runLint;
$("lint-fix").onclick = async () => {
  const n = S.lint ? S.lint.findings.filter(f => f.tier === "fix").length : 0;
  if (!(await mlConfirm(`Apply ${n} safe fix${n === 1 ? "" : "es"}? Each one points a link whose path is wrong at the only file with that name.`, { confirmLabel: "Apply", danger: false }))) return;
  await flushSaves();
  try {
    const r = await api("/api/notes/lint/fix", { method: "POST" });
    toast(r.changed.length ? `Fixed links in ${r.changed.length} file(s)` : "Nothing to fix", "ok");
    await loadTree();
    for (const p of r.changed) { const t = S.tabs.find(t => t.type === "file" && t.path === p); if (t && !t.dirty) await reloadTab(t, { quiet: true }); }
    await runLint();
  } catch (e) { toast("Fix failed: " + e.message, "err"); }
};
$("lint-review").onclick = async () => {
  try {
    const r = await api("/api/notes/review-prompt");
    await copyText(r.prompt);
  } catch (e) { toast(e.message, "err"); }
};

// ── Graph ──────────────────────────────────────────────────────────────────────
function openGraph(focus = null) {
  let t = S.tabs.find(x => x.type === "graph");
  if (!t) { t = newTab({ type: "graph", title: "Graph" }); const g = focusedGroup(), cur = activeTab(); addTab(t, g, cur ? g.tabs.indexOf(cur.id) + 1 : g.tabs.length); }
  t.focus = focus || null;
  t.title = focus ? `Graph: ${titleFor(focus)}` : "Graph";
  activate(t.id);
}
$("btn-graph").onclick = () => openGraph();

const FOLDER_COLORS = ["#4c9be8", "#e07b00", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f"];
const Graph = {
  canvas: null, ctx: null, nodes: [], edges: [], byId: new Map(), raf: 0, alpha: 0, tab: null,
  view: { x: 0, y: 0, k: 1 }, hover: null, drag: null, opts: store.get("graphOpts", { exps: true, missing: false, files: false, depth: 1, labels: 1.0 }),
  async mount(container, tab) {
    this.tab = tab;
    const wrap = el("div"); wrap.id = "graph-wrap";
    this.canvas = el("canvas"); this.canvas.id = "graph-canvas";
    wrap.append(this.canvas);
    const ctl = el("div", "graph-controls");
    const chk = (key, label) => { const l = el("label"); const c = el("input"); c.type = "checkbox"; c.checked = !!this.opts[key]; c.onchange = () => { this.opts[key] = c.checked; this.save(); this.build(); }; l.append(c, label); return l; };
    const filter = el("input", "text-input"); filter.placeholder = "Filter by path…"; filter.value = this.filter || "";
    filter.oninput = debounce(() => { this.filter = filter.value.trim().toLowerCase(); this.build(); }, 200);
    ctl.append(filter, chk("exps", "Experiments"), chk("files", "Images & other files"), chk("missing", "Notes not written yet"));
    if (tab.focus) {
      const row = el("div", "row"); row.append(el("span", "", "Depth"));
      const r = el("input"); r.type = "range"; r.min = 1; r.max = 4; r.value = this.opts.depth;
      const v = el("span", "", this.opts.depth);
      r.oninput = () => { this.opts.depth = +r.value; v.textContent = r.value; this.save(); this.build(); };
      row.append(r, v); ctl.append(row);
      const all = el("button", "mini-btn", "Show whole graph"); all.onclick = () => openGraph(null); ctl.append(all);
    }
    const lrow = el("div", "row"); lrow.append(el("span", "", "Labels"));
    const lr = el("input"); lr.type = "range"; lr.min = 0; lr.max = 2; lr.step = 0.1; lr.value = this.opts.labels;
    lr.oninput = () => { this.opts.labels = +lr.value; this.save(); this.redraw(); };
    lrow.append(lr); ctl.append(lrow);
    wrap.append(ctl);
    this.legend = el("div", "graph-legend"); wrap.append(this.legend);
    container.append(wrap);
    this.ctx = this.canvas.getContext("2d");
    this.bind();
    new ResizeObserver(() => this.resize()).observe(wrap);
    this.resize();
    await ensureIndex();
    this.build(true);
  },
  save() { store.set("graphOpts", this.opts); },
  resize() {
    if (!this.canvas || !this.canvas.isConnected) return;
    const r = this.canvas.parentElement.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    this.canvas.width = r.width * dpr; this.canvas.height = r.height * dpr;
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    this.redraw();
  },
  build(fresh = false) {
    const idx = S.index || { links: {}, mentions: {} };
    const old = new Map(this.nodes.map(n => [n.id, n]));
    const nodes = new Map(), edges = [];
    const folderOf = p => p.includes("/") ? p.split("/")[0] : "";
    const folders = [...new Set(S.files.filter(f => f.kind === "md").map(f => folderOf(f.path)))].sort();
    this.folderColor = new Map(folders.map((f, i) => [f, FOLDER_COLORS[i % FOLDER_COLORS.length]]));
    const add = (id, kind, label) => {
      if (!nodes.has(id)) { const o = old.get(id); nodes.set(id, { id, kind, label, x: o ? o.x : (Math.random() - .5) * 400, y: o ? o.y : (Math.random() - .5) * 400, vx: 0, vy: 0, deg: 0 }); }
      return nodes.get(id);
    };
    for (const f of S.files) if (f.kind === "md") add(f.path, "note", stemOf(f.path));
    for (const [src, links] of Object.entries(idx.links)) {
      for (const l of links) {
        let tgt = l.resolved;
        if (tgt && !isMd(tgt)) { if (!this.opts.files) continue; add(tgt, "file", basename(tgt)); }
        if (!tgt) { if (!this.opts.missing || !l.target) continue; tgt = "?" + l.target.toLowerCase(); add(tgt, "missing", l.target); }
        if (tgt !== src) edges.push([src, tgt]);
      }
    }
    if (this.opts.exps) {
      for (const [exp, srcs] of Object.entries(idx.mentions)) {
        if (!S.experiments.has(exp)) continue;
        add("exp:" + exp, "exp", shortExp(exp));
        for (const s of srcs) if (nodes.has(s)) edges.push([s, "exp:" + exp]);
      }
    }
    // Local graph: keep nodes within depth of the focus.
    let keep = null;
    if (this.tab && this.tab.focus && nodes.has(this.tab.focus)) {
      keep = new Set([this.tab.focus]);
      const adj = new Map();
      for (const [a, b] of edges) { (adj.get(a) || adj.set(a, []).get(a)).push(b); (adj.get(b) || adj.set(b, []).get(b)).push(a); }
      let frontier = [this.tab.focus];
      for (let d = 0; d < this.opts.depth; d++) {
        const next = [];
        for (const n of frontier) for (const m of adj.get(n) || []) if (!keep.has(m)) { keep.add(m); next.push(m); }
        frontier = next;
      }
    }
    if (this.filter) {
      const f = this.filter;
      const match = new Set([...nodes.keys()].filter(id => id.toLowerCase().includes(f)));
      keep = keep ? new Set([...keep].filter(k => match.has(k))) : match;
    }
    this.nodes = [...nodes.values()].filter(n => !keep || keep.has(n.id));
    this.byId = new Map(this.nodes.map(n => [n.id, n]));
    const seen = new Set();
    this.edges = edges.filter(([a, b]) => { const k = a < b ? a + "\0" + b : b + "\0" + a; if (seen.has(k) || !this.byId.has(a) || !this.byId.has(b)) return false; seen.add(k); return true; })
      .map(([a, b]) => [this.byId.get(a), this.byId.get(b)]);
    for (const n of this.nodes) n.deg = 0;
    for (const [a, b] of this.edges) { a.deg++; b.deg++; }
    this.legend.innerHTML = "";
    for (const [f, c] of this.folderColor) { const s = el("span"); const i = el("i"); i.style.background = c; s.append(i, f || "(top)"); this.legend.append(s); }
    if (this.opts.exps) { const s = el("span"); const i = el("i"); i.style.cssText = "background:transparent;border:2px solid #f0a030;border-radius:2px"; s.append(i, "experiment"); this.legend.append(s); }
    this.legend.append(el("span", "", `${this.nodes.length} nodes · ${this.edges.length} links`));
    if (fresh) this.view = { x: 0, y: 0, k: 1 };
    this.alpha = 1;
    this.start();
  },
  start() { if (!this.raf) this.raf = requestAnimationFrame(() => this.tick()); },
  tick() {
    this.raf = 0;
    if (!this.canvas || !this.canvas.isConnected) return;
    const N = this.nodes, a = this.alpha;
    if (a > 0.005) {
      const rep = 1800, spring = 0.035, len = 70, grav = 0.012;
      for (let i = 0; i < N.length; i++) {
        const p = N[i];
        for (let j = i + 1; j < N.length; j++) {
          const o = N[j];
          let dx = p.x - o.x, dy = p.y - o.y, d2 = dx * dx + dy * dy + 0.01;
          if (d2 > 250000) continue;
          const f = rep / d2 * a;
          const d = Math.sqrt(d2);
          dx /= d; dy /= d;
          p.vx += dx * f; p.vy += dy * f; o.vx -= dx * f; o.vy -= dy * f;
        }
      }
      for (const [s, t] of this.edges) {
        const dx = t.x - s.x, dy = t.y - s.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - len) * spring * a;
        s.vx += dx / d * f; s.vy += dy / d * f; t.vx -= dx / d * f; t.vy -= dy / d * f;
      }
      for (const p of N) {
        if (p === (this.drag && this.drag.node)) { p.vx = p.vy = 0; continue; }
        p.vx -= p.x * grav * a; p.vy -= p.y * grav * a;
        p.vx *= 0.6; p.vy *= 0.6;
        p.x += Math.max(-30, Math.min(30, p.vx)); p.y += Math.max(-30, Math.min(30, p.vy));
      }
      this.alpha *= 0.985;
      this.raf = requestAnimationFrame(() => this.tick());
    }
    this.redraw();
  },
  radius(n) { return n.kind === "exp" ? 4.5 + Math.sqrt(n.deg) * 1.2 : 3.2 + Math.sqrt(n.deg) * 1.6; },
  color(n) {
    if (n.kind === "exp") return "#f0a030";
    if (n.kind === "missing") return getComputedStyle(document.body).getPropertyValue("--text-muted");
    if (n.kind === "file") return "#888";
    return this.folderColor.get(n.id.includes("/") ? n.id.split("/")[0] : "") || "#4c9be8";
  },
  redraw() {
    if (!this.ctx || !this.canvas.isConnected) return;
    const c = this.ctx, v = this.view, css = getComputedStyle(document.body);
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);
    c.translate(this.w / 2 + v.x, this.h / 2 + v.y); c.scale(v.k, v.k);
    const hov = this.hover, active = this.tab && this.tab.focus;
    const nb = new Set();
    if (hov) { nb.add(hov); for (const [a, b] of this.edges) { if (a === hov) nb.add(b); if (b === hov) nb.add(a); } }
    c.lineWidth = 1 / v.k;
    for (const [a, b] of this.edges) {
      const on = hov && (a === hov || b === hov);
      c.strokeStyle = on ? css.getPropertyValue("--radio-checked-border") : css.getPropertyValue("--border-input");
      c.globalAlpha = hov && !on ? 0.25 : 0.9;
      c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    }
    const textCol = css.getPropertyValue("--text-label");
    for (const n of this.nodes) {
      const r = this.radius(n);
      c.globalAlpha = hov && !nb.has(n) ? 0.25 : 1;
      c.fillStyle = this.color(n);
      c.beginPath();
      if (n.kind === "exp") c.rect(n.x - r, n.y - r, r * 2, r * 2); else c.arc(n.x, n.y, r, 0, Math.PI * 2);
      if (n.kind === "missing") { c.strokeStyle = c.fillStyle; c.lineWidth = 1.2 / v.k; c.stroke(); } else c.fill();
      if (n.id === active) { c.strokeStyle = css.getPropertyValue("--text-strong"); c.lineWidth = 2 / v.k; c.stroke(); }
      const showLabel = n === hov || nb.has(n) || v.k * this.opts.labels > 1.15 || (this.opts.labels > 0 && n.deg >= 4 && v.k * this.opts.labels > 0.6);
      if (showLabel && this.opts.labels > 0) {
        c.fillStyle = textCol; c.font = `${(n === hov ? 12.5 : 11) / v.k}px system-ui, sans-serif`; c.textAlign = "center";
        c.fillText(n.label.length > 40 ? n.label.slice(0, 38) + "…" : n.label, n.x, n.y + r + 12 / v.k);
      }
    }
    c.globalAlpha = 1;
  },
  toWorld(ev) {
    const r = this.canvas.getBoundingClientRect(), v = this.view;
    return [(ev.clientX - r.left - this.w / 2 - v.x) / v.k, (ev.clientY - r.top - this.h / 2 - v.y) / v.k];
  },
  pick(ev) {
    const [x, y] = this.toWorld(ev);
    let best = null, bd = Infinity;
    for (const n of this.nodes) { const d = (n.x - x) ** 2 + (n.y - y) ** 2, r = this.radius(n) + 4 / this.view.k; if (d < r * r && d < bd) { bd = d; best = n; } }
    return best;
  },
  bind() {
    const cv = this.canvas;
    cv.addEventListener("mousemove", ev => {
      if (this.drag) {
        if (this.drag.node) { const [x, y] = this.toWorld(ev); this.drag.node.x = x; this.drag.node.y = y; this.alpha = Math.max(this.alpha, 0.3); this.start(); }
        else { this.view.x = this.drag.vx + ev.clientX - this.drag.x; this.view.y = this.drag.vy + ev.clientY - this.drag.y; this.redraw(); }
        this.drag.moved = true;
        return;
      }
      const h = this.pick(ev);
      if (h !== this.hover) { this.hover = h; cv.style.cursor = h ? "pointer" : "grab"; cv.title = h ? (h.kind === "exp" ? h.label + " (experiment)" : h.id) : ""; this.redraw(); }
    });
    cv.addEventListener("mousedown", ev => {
      const n = this.pick(ev);
      this.drag = { node: n, x: ev.clientX, y: ev.clientY, vx: this.view.x, vy: this.view.y, moved: false };
      cv.classList.add("dragging");
    });
    window.addEventListener("mouseup", ev => {
      if (!this.drag) return;
      const d = this.drag; this.drag = null; cv.classList.remove("dragging");
      if (!d.moved && d.node) {
        const n = d.node;
        if (n.kind === "note" || n.kind === "file") openFile(n.id, { newTab: ev.ctrlKey || ev.metaKey });
        else if (n.kind === "exp") location.href = pageHref("results.html", { experiment: n.label });
        else if (n.kind === "missing") newNote("", n.label);
      }
    });
    cv.addEventListener("wheel", ev => {
      ev.preventDefault();
      const r = cv.getBoundingClientRect();
      const mx = ev.clientX - r.left - this.w / 2, my = ev.clientY - r.top - this.h / 2;
      const k0 = this.view.k, k = Math.max(0.15, Math.min(6, k0 * Math.exp(-ev.deltaY * 0.0015)));
      this.view.x = mx - (mx - this.view.x) * k / k0; this.view.y = my - (my - this.view.y) * k / k0; this.view.k = k;
      this.redraw();
    }, { passive: false });
    cv.addEventListener("dblclick", () => { this.view = { x: 0, y: 0, k: 1 }; this.redraw(); });
  },
};

// ── Resizers ───────────────────────────────────────────────────────────────────
// The shared drag handle (MLChrome.resizable), setting the width variable the
// panel's CSS reads, so its cap on a narrow window still applies.
function setupResizer(id, side) {
  const prop = side === "left" ? "--left-w" : "--right-w";
  MLChrome.resizable($(side), {
    name: side === "left" ? "notes_leftW" : "notes_rightW", side, min: 180, max: 560, handle: $(id),
    apply: w => w == null ? document.body.style.removeProperty(prop) : document.body.style.setProperty(prop, w + "px"),
    onResize: () => eachEditor(ed => ed.cm.refresh()),
  });
}
setupResizer("resize-left", "left");
setupResizer("resize-right", "right");
// On a narrow window (half a monitor) the right panel folds away so the document
// keeps the room, and comes back as it was when the window widens.  Its button
// still opens it meanwhile.
const narrowWindow = matchMedia("(max-width: 1100px)");
function fitRightPanel() {
  const c = narrowWindow.matches || store.get("rightCollapsed", false);
  document.body.classList.toggle("right-collapsed", c);
  return c;
}
fitRightPanel();
narrowWindow.addEventListener("change", () => {
  if (!fitRightPanel()) refreshRight();
  eachEditor(ed => ed.cm.refresh());
});
if (store.get("leftCollapsed", false)) document.body.classList.add("left-collapsed");
if (store.get("wide", false)) document.body.classList.add("wide-doc");

// ── Keyboard ───────────────────────────────────────────────────────────────────
// No modifiers except Ctrl+S, so nothing shadows the browser (Ctrl+W, E, G, K,
// O, P, Alt+arrows) or a window manager (Alt+anything).  Single keys, GitHub
// style, work when you are not typing; Esc gets you out of a text field.
const KEYS = [
  ["Anywhere", [
    ["Ctrl+S", "save now (saving is automatic)"],
    ["Esc", "stop typing, so the keys below work"],
  ]],
  ["When not typing", [
    ["T or O", "open a file"],
    ["X", "close tab"],
    ["{ / }", "previous / next tab"],
    ["\\", "split right: move the tab to a group beside this one"],
    ["|", "split down"],
    ["/", "search"],
    ["L / R", "hide or show the left / right panel"],
    ["E", "notes: Live ↔ Source"],
    ["S", "notes: split view"],
    ["G", "graph (Shift+G: this note's graph)"],
    ["[ / ]", "back / forward"],
    ["Ctrl+Z / Ctrl+Shift+Z", "undo / redo in Live"],
    ["F2", "rename"],
    ["?", "this list"],
  ]],
  ["Editing a block (Live)", [
    ["Esc", "done (or click elsewhere)"],
    ["↑ / ↓", "into the block above / below"],
    ["Ctrl+Enter", "new block below"],
    ["[[", "link completion"],
    ["mlsweep:", "experiment ID completion"],
    ["Ctrl+B / Ctrl+I", "bold / italic"],
  ]],
  ["Source", [
    ["Esc", "back to Live"],
    ["Ctrl+Space", "link or experiment ID completion"],
    ["Ctrl+B / Ctrl+I", "bold / italic"],
  ]],
];
function showKeys() {
  const back = el("div", "ml-dialog-backdrop");
  const dlg = el("div", "ml-dialog keys-dialog");
  for (const [title, rows] of KEYS) {
    dlg.append(el("h3", "", title));
    const g = el("div", "kbd-list");
    for (const [k, what] of rows) {
      const kk = el("span");
      const parts = k === "/" ? ["/"] : k.split(" ");
      parts.forEach(part => {
        if (k !== "/" && (part === "/" || part === "or")) kk.append(" " + part + " ");
        else kk.append(el("kbd", "", part));
      });
      g.append(kk, el("span", "", what));
    }
    dlg.append(g);
  }
  dlg.append(el("div", "pane-note", "Full guide, including the markdown reference: mlsweep docs notes"));
  const close = () => back.remove();
  back.onclick = e => { if (e.target === back) close(); };
  back.tabIndex = -1;
  back.addEventListener("keydown", e => { if (e.key === "Escape" || e.key === "?") { e.preventDefault(); close(); } });
  back.append(dlg); document.body.append(back); back.focus();
}
function cycleTab(d) {
  const g = focusedGroup();
  if (!g || !g.tabs.length) return;
  const n = g.tabs.length, i = Math.max(0, g.tabs.indexOf(g.active));
  activate(g.tabs[(i + d + n) % n]);
}
document.addEventListener("keydown", e => {
  const mod = e.ctrlKey || e.metaKey;
  const typing = e.target.matches("input, textarea, select, [contenteditable]") || e.target.closest(".CodeMirror");
  const t = activeTab();
  if (mod && !e.altKey && e.key.toLowerCase() === "s") { e.preventDefault(); saveActiveNow(); return; }
  if (e.key === "Escape") {
    mlCloseContextMenu(); $("lightbox").classList.remove("show");
    // Leave a plain text field, so the single-key shortcuts work again.
    if (e.target.matches("input, select") && !e.target.closest("#switcher, .ml-dialog, .tnode")) e.target.blur();
  }
  if (typing || e.altKey) return;
  if (mod) {
    const k = e.key.toLowerCase();
    if (k === "z" && t && t.mode === "live" && t.doc && !t.live?.ta) {
      e.preventDefault();
      e.shiftKey ? t.doc.redo() : t.doc.undo();
      renderActive();
    } else if (k === "y" && t && t.mode === "live" && t.doc && !t.live?.ta) {
      e.preventDefault(); t.doc.redo(); renderActive();
    }
    return;
  }
  const key = e.key;
  const act = {
    "t": () => Switcher.open(), "o": () => Switcher.open(),
    "x": () => { if (t) closeTab(t.id); else closeGroup(focusedGroup()); },
    "\\": () => { if (t) splitTab(t, "right"); },
    "|": () => { if (t) splitTab(t, "down"); },
    "{": () => cycleTab(-1), "}": () => cycleTab(1),
    "/": () => { switchPane("search"); $("search-input").select(); },
    "e": () => { if (t && t.type === "file") setMode(t.mode === "live" ? "source" : "live"); },
    "s": () => { if (t && t.type === "file") setMode(t.mode === "split" ? "live" : "split"); },
    "g": () => openGraph(), "G": () => openGraph(t && t.path && isMd(t.path) ? t.path : null),
    "[": () => navigate(-1), "]": () => navigate(1),
    "l": () => toggleLeft(), "r": () => toggleRight(),
    "?": () => showKeys(),
    "F2": () => {
      if (!t || !t.path) return;
      const row = treeRow(t.path);
      if (row) startRename(row);
    },
  }[key];
  if (act) { e.preventDefault(); act(); }
});

// ── Startup ────────────────────────────────────────────────────────────────────
async function init() {
  initLayout();
  await loadExperiments();
  await Promise.all([poll(true), loadSessions()]);
  if (!S.enabled) return;
  // Restore the groups and their tabs, then honour ?path= / ?q= / ?mentions=.
  await restoreLayout();
  const want = _qp.get("path");
  if (want && S.fileMap.has(want)) await openFile(want, { pushNav: false });
  if (_qp.get("q") || _qp.get("mentions")) searchFor(_qp.get("q") || _qp.get("mentions"));
  else switchPane(store.get("pane", "files") === "lint" ? "files" : store.get("pane", "files"), { reveal: false });
  if (activeTab() && activeTab().path) revealInTree(activeTab().path);
  S.ready = true;   // the saved layout is back; tests wait for this
  // Mark everything seen so "fresh" dots only show changes from now on.
  setTimeout(() => { S.seenMtime = Date.now() / 1000 - 15 * 60; store.set("seenMtime", Date.now() / 1000); }, 4000);
  setInterval(() => { if (!document.hidden) poll(); }, 4000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  setInterval(loadExperiments, 60000);
  setInterval(renderSyncState, 15000);
}
init();
