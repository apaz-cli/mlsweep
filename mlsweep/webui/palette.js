// Command palette — search, resolve, preview, and act.
// Uses MLSearch (search-core.js) for the actual index.
(function () {
  const PAGES = [
    ["experiments", "Experiments", "▤"], ["results", "Results", "▥"], ["logs", "Logs", "≡"],
    ["files", "Artifacts", "▣"], ["code", "Code", "‹›"], ["notes", "Notes", "✎"],
    ["traces", "Traces", "∿"], ["export", "Export", "⇩"],
    ["system", "System", "⚙"],
  ];

  // Earlier versions kept a search history and per-result usage; forget them.
  try { localStorage.removeItem("mlsweep_search_history"); localStorage.removeItem("mlsweep_search_usage"); } catch {}

  let overlay, input, listEl, footEl, previewEl, scopeSel;
  let query = "";
  let scope = MLPrefs.get("search_scope") || "all";
  let items = [], selected = 0, actionMode = false, actionIndex = 0;
  let searchSeq = 0, debounceTimer = 0, lastFocus = null;
  const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const mark = (text, q) => {
    if (!q) return esc(text);
    const i = text.toLowerCase().indexOf(q.toLowerCase());
    if (i < 0) return esc(text);
    return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + q.length)) + "</mark>" + esc(text.slice(i + q.length));
  };
  const recent = () => MLPrefs.get("palette_recent", []).filter(x => x && x.href);
  const views = () => MLPrefs.get("saved_views", []).filter(x => x && x.href && x.title);
  const searches = () => window.MLSearch ? MLSearch.savedSearches() : [];

  function remember(item) {
    if (!item || !item.href || item.run) return;
    const list = recent().filter(x => x.href !== item.href);
    list.unshift({ title: item.title, sub: item.sub, href: item.href, icon: item.icon, kind: item.kind });
    MLPrefs.set("palette_recent", list.slice(0, 8));
  }

  function pageItems() {
    return PAGES.map(([id, title, icon]) => ({ kind: "page", id: `page:${id}`, title, sub: "Dashboard page", icon, href: `${id}.html`, actions: [{ label: "Open", icon: "↗", run: () => location.href = `${id}.html` }] }));
  }
  function recentItems() {
    return recent().map(x => ({ ...x, kind: "recent", id: `recent:${x.href}`, actions: [{ label: "Open", icon: "↗", run: () => location.href = x.href }] }));
  }
  function savedSearchItems() {
    return searches().map(s => ({
      kind: "saved", id: `search:${s.query}`, title: s.title || s.query, sub: s.query, icon: "★",
      actions: [
        { label: "Run search", icon: "→", keepOpen: true, run: () => { setScope(s.scope || "all"); setQuery(s.query); } },
        { label: "Remove saved search", icon: "×", keepOpen: true, run: () => { MLSearch.removeSearch(s.query); render(); } },
      ],
    }));
  }
  function savedViewItems() {
    return views().map(v => ({ kind: "saved", id: `view:${v.href}`, title: v.title, sub: "Saved view", icon: "★", href: v.href, actions: [{ label: "Open view", icon: "↗", run: () => location.href = v.href }] }));
  }
  async function saveCurrentView() {
    const fallback = (document.title || "Saved view").replace(/^mlsweep — /, "");
    const name = await mlPrompt("Name this view:", fallback, { okLabel: "Save view" });
    if (!name) return;
    const href = location.pathname + location.search + location.hash;
    const list = views().filter(v => v.href !== href);
    list.unshift({ title: name.trim(), href });
    MLPrefs.set("saved_views", list.slice(0, 30));
    render();
  }

  function actionItems() {
    const out = [
      { kind: "action", id: "action:theme", title: "Toggle dark mode", sub: "Switch the dashboard theme", icon: "◐",
        actions: [{ label: "Toggle theme", icon: "◐", run: () => document.getElementById("theme-toggle")?.click() }] },
      { kind: "action", id: "action:saveview", title: "Save current view…", sub: "Remember this filter, metric and tab", icon: "★",
        actions: [{ label: "Save view", icon: "★", keepOpen: true, run: saveCurrentView }] },
    ];
    if (query) out.push({ kind: "action", id: "action:savesearch", title: `Save search: ${query}`, sub: "Pin this query", icon: "★",
      actions: [{ label: "Save search", icon: "★", keepOpen: true, run: async () => { const name = await mlPrompt("Name this search:", query, { okLabel: "Save search" }); if (name) { MLSearch.saveSearch(query, name, scope); render(); } } }] });
    return out;
  }

  function allDefaultItems() {
    const groups = [];
    if (recent().length) groups.push({ group: "Recent" }, ...recentItems());
    if (searches().length) groups.push({ group: "Saved searches" }, ...savedSearchItems());
    if (views().length) groups.push({ group: "Saved views" }, ...savedViewItems());
    groups.push({ group: "Pages" }, ...pageItems());
    groups.push({ group: "Actions" }, ...actionItems());
    return groups;
  }

  function groupResults(results) {
    const groups = new Map();
    const order = ["answer", "action", "run", "experiment", "artifact", "note", "doc", "page", "saved"];
    const labels = { answer: "Answers", action: "Actions", run: "Runs", experiment: "Experiments", artifact: "Artifacts", note: "Notes", doc: "Docs", page: "Pages", saved: "Saved" };
    for (const it of results) {
      const k = it.kind === "saved" || it.kind === "recent" ? "saved" : it.kind;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(it);
    }
    const out = [];
    for (const k of order) {
      if (!groups.has(k)) continue;
      out.push({ group: labels[k] || k }, ...groups.get(k));
    }
    return out;
  }

  function setScope(v) {
    scope = v || "all";
    scopeSel.value = scope;
    MLPrefs.set("search_scope", scope);
  }

  function setQuery(q) {
    query = q;
    input.value = q;
    selected = 0; actionMode = false; actionIndex = 0;
    runSearch();
  }

  function runSearch() {
    clearTimeout(debounceTimer);
    if (!query.trim()) {
      items = allDefaultItems();
      render();
      return;
    }
    const seq = ++searchSeq;
    items = [{ group: "Search" }, { kind: "loading", id: "loading", title: "Searching…", sub: "experiments, runs, notes, docs", icon: "…", actions: [] }];
    render();
    debounceTimer = setTimeout(async () => {
      if (!window.MLSearch) return;
      try {
        const res = await MLSearch.search(query, { scope, limit: 20 });
        if (seq !== searchSeq) return;
        items = groupResults(res.items);
        if (!res.items.length) items.push({ group: "Search" }, { kind: "error", id: "empty", title: "No matches", sub: "Try a run ID, a note, docs:OPTIMIZE, or status:failed", icon: "·", actions: [] });
        items.push({ group: "Actions" },
          { kind: "action", id: "action:savesearch", title: `Save search: ${query}`, sub: "Pin this query", icon: "★", actions: [{ label: "Save search", icon: "★", keepOpen: true, run: saveSearchPrompt }] });
        selected = 0; actionMode = false;
        render(res);
      } catch (e) {
        if (seq !== searchSeq) return;
        items = [{ group: "Search" }, { kind: "error", id: "error", title: `Search failed: ${e.message}`, sub: "", icon: "!", actions: [] }];
        render();
      }
    }, 160);
  }

  async function saveSearchPrompt() {
    const name = await mlPrompt("Name this search:", query, { okLabel: "Save search" });
    if (!name) return;
    MLSearch.saveSearch(query, name);
    render();
  }

  function selectable() { return items.filter(it => !it.group && it.kind !== "loading" && it.kind !== "error"); }
  function selectedItem() { return selectable()[selected]; }

  function itemActions(it) {
    return (it && it.actions && it.actions.length) ? it.actions : [{ label: "Open", icon: "↗", run: () => it?.href && (location.href = it.href) }];
  }

  function render(meta = {}) {
    const sel = selectedItem();
    if (actionMode) actionIndex = Math.min(actionIndex, itemActions(sel).length - 1);
    const selList = items.filter(it => !it.group && it.kind !== "loading" && it.kind !== "error");
    selected = Math.max(0, Math.min(selected, selList.length - 1));
    listEl.innerHTML = items.map(it => {
      if (it.group) return `<div class="pal-group">${esc(it.group)}</div>`;
      const idx = selList.indexOf(it);
      return `<div class="pal-item ${idx === selected ? "sel" : ""}" data-i="${idx}">
        <span class="pal-icon">${esc(it.icon || "·")}</span>
        <span class="pal-text">
          <div class="pal-title">${mark(it.title || "", query.trim())}</div>
          <div class="pal-sub">${mark(it.sub || "", query.trim())}</div>
        </span>
        ${it.status ? `<span class="pal-dot ${esc(it.status)}" title="${esc(it.status)}"></span>` : ""}
        <span class="pal-kind">${esc(it.kind)}</span>
      </div>`;
    }).join("") || `<div class="pal-empty">No matches. Try an experiment, a run ID, a note, or <code>status:failed</code>.</div>`;

    // Preview / actions pane
    if (sel) {
      const acts = itemActions(sel);
      const actionHtml = `<div class="pal-actions">${acts.map((a, i) => `<button class="pal-action ${actionMode && i === actionIndex ? "sel" : ""}" data-action="${i}"><span>${esc(a.icon || "›")}</span>${esc(a.label)}</button>`).join("")}</div>`;
      previewEl.innerHTML = `<div class="pal-preview-kind">${esc(sel.kind)}</div>${sel.preview || `<div class="sp-title">${esc(sel.title || "")}</div><div class="sp-body">${esc(sel.sub || "")}</div>`}${actionHtml}`;
    } else {
      previewEl.innerHTML = `<div class="pal-empty">Type to search.</div>`;
    }

    const total = selectable().length;
    footEl.innerHTML = `<span><kbd>↑</kbd><kbd>↓</kbd> select</span>
      <span><kbd>↵</kbd> ${actionMode ? "run action" : "open"}</span>
      <span><kbd>tab</kbd> actions</span>
      <span><kbd>esc</kbd> ${actionMode ? "back" : "close"}</span>
      <span style="margin-left:auto">${meta.took != null ? `${total} results · ${meta.took}ms` : `${total} items`}</span>`;
  }

  function move(delta) {
    const sel = selectable();
    if (!sel.length) return;
    selected = (selected + delta + sel.length) % sel.length;
    actionMode = false; actionIndex = 0;
    render();
    listEl.querySelector(`.pal-item[data-i="${selected}"]`)?.scrollIntoView({ block: "nearest" });
  }

  function execute(action) {
    const it = selectedItem();
    if (!it) return;
    const act = action || itemActions(it)[actionMode ? actionIndex : 0];
    if (!act) return;
    if (act.run) {
      if (!act.keepOpen) close();
      act.run();
    } else if (act.href) {
      remember(it); close(); location.href = act.href;
    }
  }

  function open() {
    lastFocus = document.activeElement;
    overlay.classList.add("show");
    query = ""; input.value = ""; selected = 0; actionMode = false;
    input.focus();
    runSearch();
  }
  function close() {
    overlay.classList.remove("show");
    input.blur();
    if (lastFocus?.focus) lastFocus.focus();
  }

  function init() {
    overlay = document.createElement("div");
    overlay.id = "palette-back";
    overlay.innerHTML = `<div class="palette" role="dialog" aria-modal="true" aria-label="Search">
      <div class="palette-input-wrap">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="7" cy="7" r="4.75"/><path d="m10.5 10.5 3 3"/></svg>
        <input id="palette-input" type="text" placeholder="Search runs, experiments, notes, docs… or paste an ID/URL" autocomplete="off" spellcheck="false">
        <select id="palette-scope" title="Search scope"><option value="all">Everything</option><option value="experiment">This experiment</option><option value="campaign">This campaign</option></select>
      </div>
      <div class="palette-body"><div class="palette-list" id="palette-list"></div><div class="pal-preview" id="pal-preview"></div></div>
      <div class="pal-footer" id="palette-foot"></div>
    </div>`;
    document.body.append(overlay);
    input = overlay.querySelector("#palette-input");
    listEl = overlay.querySelector("#palette-list");
    previewEl = overlay.querySelector("#pal-preview");
    footEl = overlay.querySelector("#palette-foot");
    scopeSel = overlay.querySelector("#palette-scope");
    scopeSel.value = scope;
    if (!MLSearch.context().experiment) {
      const opt = scopeSel.querySelector('option[value="experiment"]');
      if (opt) { opt.disabled = true; opt.textContent = "This experiment (open a sweep)"; }
    }

    input.addEventListener("input", () => { query = input.value; selected = 0; actionMode = false; runSearch(); });
    scopeSel.addEventListener("change", () => {
      scope = scopeSel.value;
      MLPrefs.set("search_scope", scope);
      runSearch();
    });
    listEl.addEventListener("click", e => {
      const row = e.target.closest(".pal-item");
      if (!row) return;
      selected = +row.dataset.i; actionMode = false; render();
      execute();
    });
    previewEl.addEventListener("click", e => {
      const btn = e.target.closest("[data-action]");
      if (!btn) return;
      actionMode = true; actionIndex = +btn.dataset.action; render(); execute();
    });
    overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
    input.addEventListener("keydown", e => {
      if (e.key === "Escape") {
        e.preventDefault();
        if (actionMode) { actionMode = false; render(); } else close();
      } else if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Tab") {
        e.preventDefault();
        const acts = itemActions(selectedItem());
        if (!acts.length) return;
        actionMode = true;
        actionIndex = (actionIndex + (e.shiftKey ? -1 : 1) + acts.length) % acts.length;
        render();
      } else if (e.key === "Enter") { e.preventDefault(); execute(); }
    });

    document.addEventListener("keydown", e => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        overlay.classList.contains("show") ? close() : open();
      }
    });
    document.getElementById("palette-open")?.addEventListener("click", open);
    runSearch();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
