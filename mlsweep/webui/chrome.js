// What every dashboard page shares: the login token, API calls, the nav bar,
// the theme, and the connection indicator.  Load this before campaign.js, then
// call MLChrome.nav() right after the page's <nav data-page="...">.  Anything
// inside that <nav> stays, in place of the connection indicator.

// The token comes from ?token= once, then from localStorage.  It is taken out
// of the address bar, so it never ends up in a copied link or a Referer header.
const TOKEN = (() => {
  const url = new URL(location.href);
  const fromUrl = url.searchParams.get("token");
  const token = fromUrl || localStorage.getItem("mlsweep_token") || "";
  if (token) localStorage.setItem("mlsweep_token", token);
  if (fromUrl) {
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + url.search + url.hash);
  }
  return token;
})();

// Remembered settings, one scheme for every page: localStorage key
// "mlsweep_<name>" (names like "logs_wrap", "notes_tabs"), the value as JSON.
// A value written before it was JSON (a bare "1" or campaign name) still reads
// back.  Storage can be off (a private window); then nothing is remembered.
const MLPrefs = (() => {
  const key = name => "mlsweep_" + name;
  function get(name, fallback = null) {
    let raw = null;
    try { raw = localStorage.getItem(key(name)); } catch { return fallback; }
    if (raw == null) return fallback;
    try { return JSON.parse(raw); } catch { return raw; }
  }
  function set(name, value) {
    try { localStorage.setItem(key(name), JSON.stringify(value)); } catch { /* storage off */ }
  }
  function remove(name) {
    try { localStorage.removeItem(key(name)); } catch { /* storage off */ }
  }
  // Keys from before the scheme, moved once: theme, notes.<name>, and
  // sidebarCollapsed:<page>.  The login token keeps its own key.
  try {
    const moves = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k === "theme") moves.push([k, "theme", v => v]);
      else if (k.startsWith("notes.")) moves.push([k, "notes_" + k.slice(6), v => JSON.parse(v)]);
      else if (k.startsWith("sidebarCollapsed:")) moves.push([k, `sidebar_collapsed_${k.slice(17)}`, v => v === "1"]);
    }
    for (const [old, name, convert] of moves) {
      const v = localStorage.getItem(old);
      localStorage.removeItem(old);
      if (localStorage.getItem(key(name)) != null) continue;
      try { set(name, convert(v)); } catch { /* unreadable: dropped */ }
    }
  } catch { /* storage off */ }
  return { get, set, remove };
})();

// A new *tag* element with class *cls* and text *text*, both optional.
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

async function apiFetch(path, opts = {}) {
  // X-Mlsweep-Client names the dashboard in a run's history (`mlsweep history`).
  const r = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${TOKEN}`, "X-Mlsweep-Client": "dashboard" } });
  if (r.status === 401) showLoginProblem("The manager rejected this page's token. Open the link it printed at startup, or run `mlsweep status` to find it.");
  else if (r.status === 429) showLoginProblem("Too many failed logins from this address. Try again in a minute.");
  return r;
}

// Every page calls the API first thing, so a bad token is explained in one place
// and pages need only handle their own failures.
function showLoginProblem(message) {
  if (document.getElementById("login-problem")) return;
  const box = document.createElement("div");
  box.id = "login-problem";
  box.setAttribute("role", "alert");
  box.textContent = message;
  box.style.cssText = "position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:1000;max-width:90vw;"
    + "padding:10px 14px;border:1px solid var(--bad);border-radius:6px;background:var(--surface);color:var(--text);font-size:13px";
  document.body.append(box);
}

// *path* for the places a request can't carry a header: images, downloads.
function authURL(path) {
  return TOKEN ? path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(TOKEN) : path;
}

// Copy *text* and confirm with a small toast.  (The Notes page has its own.)
function copyText(text, label = "Copied") {
  if (!navigator.clipboard?.writeText) return;
  navigator.clipboard.writeText(text).then(() => {
    document.getElementById("copy-toast")?.remove();
    const t = Object.assign(document.createElement("div"), { id: "copy-toast", textContent: label });
    document.body.append(t);
    setTimeout(() => t.remove(), 1200);
  }).catch(() => {});
}

// The WebSocket at *path* (a page can't set headers on one, so the token rides along).
function wsURL(path) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}?token=${encodeURIComponent(TOKEN)}`;
}

// What a trace is called: its title, else its working directory's last part, else its id.
function traceName(s) {
  return s.title || (s.cwd ? s.cwd.split("/").filter(Boolean).pop() : s.session_id);
}

// A link to another dashboard page, with query *extra*.
function pageHref(page, extra = {}) {
  const query = new URLSearchParams(extra).toString();
  return query ? `${page}?${query}` : page;
}

// "connected", "reconnecting" or "disconnected".
function setConnStatus(state) {
  document.getElementById("conn-dot").className = state === "connected" ? "" : state;
  document.getElementById("conn-label").textContent = state === "disconnected" ? "Disconnected"
    : state === "reconnecting" ? "Reconnecting…" : "";
}

// The pages' CSP bars inline handlers, so markup built as a string names its
// handler instead: <button ${act("rename", id)}>.  bindActions({rename}) on the
// page then runs rename(id) on a click, with the element as `this`.  The nearest
// data-act wins, and an empty one (data-act="") keeps a click from reaching rows
// and cells around it.
function act(name, ...args) {
  return `data-act="${escHtml(name)}" data-args="${escHtml(JSON.stringify(args))}"`;
}

function bindActions(actions, eventName = "click") {
  document.addEventListener(eventName, e => {
    const el = e.target.closest ? e.target.closest("[data-act]") : null;
    if (!el || !el.dataset.act || !Object.hasOwn(actions, el.dataset.act)) return;
    actions[el.dataset.act].apply(el, JSON.parse(el.dataset.args || "[]"));
  });
}

// Small monochrome glyphs for the nav tabs and the row menus' View entries,
// drawn with the text colour.  Keyed by page, except files.html is "artifacts".
const VIEW_ICONS = {
  experiments: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5.5 1.5h5M6.5 1.5v4.3l-3.8 6.6a1.4 1.4 0 0 0 1.2 2.1h8.2a1.4 1.4 0 0 0 1.2-2.1L9.5 5.8V1.5"/><path d="M4.6 9.5h6.8"/></svg>',
  results: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13.5V2.5M2 13.5h12"/><path d="m4.5 10 3-3.5 2.5 2 3.5-5"/></svg>',
  logs: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><path d="M3 4h10M3 8h10M3 12h6"/></svg>',
  artifacts: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M8 1.5 14 4.5v7L8 14.5 2 11.5v-7z"/><path d="M2 4.5 8 7.5l6-3M8 7.5v7"/></svg>',
  notes: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="m2.5 13.5.8-3.2 7.4-7.4 2.4 2.4-7.4 7.4z"/><path d="m9.5 3.7 2.4 2.4"/></svg>',
  traces: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12.5v-9"/><path d="M6 13.5V6.5"/><path d="M10 12.5V3"/><path d="M14 13.5V8.5"/></svg>',
  code: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="m5.5 4.5-3.5 3.5 3.5 3.5M10.5 4.5 14 8l-3.5 3.5"/></svg>',
  export: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v7.5"/><path d="m4.8 6.5 3.2 3.2 3.2-3.2"/><path d="M3 13.5h10"/></svg>',
  system: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2.5" width="12" height="4.5" rx="1"/><rect x="2" y="9" width="12" height="4.5" rx="1"/><path d="M4.6 4.75h.01M4.6 11.25h.01M7 4.75h.01M7 11.25h.01" stroke-width="2"/></svg>',
};

// File-type glyphs for the file trees (notes, code), keyed by fileKindOf().
const FILE_ICONS = {
  md: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="M5.5 8h5M5.5 10.5h5M5.5 5.5h2.5"/></svg>',
  image: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><rect x="1.5" y="2.5" width="13" height="11" rx="1.5"/><circle cx="5.5" cy="6" r="1.3"/><path d="m2 12 4-4 3 3 2-2 3.5 3.5"/></svg>',
  pdf: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="M5.3 11.5V8.2h1.1a1 1 0 0 1 0 2H5.3M8.6 11.5V8.2h.8c.9 0 1.3.7 1.3 1.65s-.4 1.65-1.3 1.65z"/></svg>',
  text: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="m6.5 7.5-1.5 1.5 1.5 1.5M9.5 7.5l1.5 1.5-1.5 1.5"/></svg>',
  binary: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M3.5 1.5h6l3 3v10h-9z"/><path d="M9.5 1.5v3h3"/></svg>',
  dir: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M1.5 4.5v8a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-6a1 1 0 0 0-1-1H8L6.5 3.5h-4a1 1 0 0 0-1 1z"/></svg>',
};

// "md", "image", "pdf", "text" or "binary", from *path*'s extension.
function fileKindOf(path) {
  const base = path.slice(path.lastIndexOf("/") + 1);
  const dot = base.lastIndexOf(".");
  const e = dot > 0 ? base.slice(dot + 1).toLowerCase() : base.startsWith(".") ? base.slice(1).toLowerCase() : "";
  if (e === "md" || e === "markdown") return "md";
  if (/^(png|jpe?g|gif|webp|avif|bmp|ico|svg)$/.test(e)) return "image";
  if (e === "pdf") return "pdf";
  if (/^(txt|log|json|jsonl|ya?ml|toml|ini|cfg|conf|csv|tsv|py|sh|bash|zsh|js|ts|css|html?|xml|tex|bib|rst|c|h|cc|cpp|hpp|cu|rs|go|java|lua|sql|r|jl|ipynb|diff|patch|env|mk|cmake|gitignore|gitattributes)$/.test(e)) return "text";
  if (/^(Makefile|Dockerfile|LICENSE|README|TODO|\.gitignore|\.gitattributes)$/.test(base)) return "text";
  return "binary";
}

// A panel with its left part marked: the glyph for showing and hiding a side panel.
const PANEL_ICON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="1.5" y="2.5" width="13" height="11" rx="1.5"/><path d="M5.5 2.5v11"/></svg>';

const MLChrome = (() => {
  // [page, label, palette glyph]: the nav bar and the command palette both list these.
  const PAGES = [
    ["experiments", "Experiments", "▤"], ["results", "Results", "▥"], ["logs", "Logs", "≡"],
    ["files", "Artifacts", "▣"], ["code", "Code", "‹›"], ["notes", "Notes", "✎"],
    ["traces", "Traces", "∿"], ["export", "Export", "⇩"], ["system", "System", "⚙"],
  ];
  const root = document.documentElement;

  function showTheme(dark) {
    if (dark) root.setAttribute("data-theme", "dark"); else root.removeAttribute("data-theme");
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.textContent = dark ? "☀" : "☾";
      btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
    }
  }
  const prefersDark = () => MLPrefs.get("theme") !== "light";
  showTheme(prefersDark());   // now, in <head>, so a light page never flashes dark

  function make(tag, props) { return Object.assign(document.createElement(tag), props); }

  function connIndicator() {
    const box = make("span", { style: "display:flex;align-items:center;gap:5px" });
    box.append(make("span", { id: "conn-dot" }), make("span", { id: "conn-label" }));
    return box;
  }

  function nav() {
    const bar = document.querySelector("nav[data-page]");
    const extras = [...bar.children];
    const logo = make("img", { src: "/static/favicon.ico", width: 28, height: 28, alt: "",
                               style: "border-radius:3px;flex-shrink:0;margin-left:-8px;margin-right:4px" });
    const links = PAGES.map(([page, label]) => {
      const a = page === bar.dataset.page
        ? make("a", { className: "nav-link active", href: "#" })
        : make("a", { className: "nav-link", id: `nav-${page}`, href: pageHref(`${page}.html`) });
      a.innerHTML = VIEW_ICONS[page === "files" ? "artifacts" : page];
      a.append(make("span", { className: "nav-label", textContent: label }));
      a.title = label;   // the label hides on a narrow window, leaving the icon
      return a;
    });
    const btn = make("button", { id: "theme-toggle" });
    const searchBtn = make("button", { id: "palette-open", className: "nav-search-btn", type: "button",
                                       title: "Search runs, experiments, notes and docs" });
    searchBtn.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="7" cy="7" r="4.75"/><path d="m10.5 10.5 3 3"/></svg><span class="nav-label">Search</span> <kbd>${/Mac|iPhone|iPad/.test(navigator.platform || "") ? "⌘K" : "Ctrl K"}</kbd>`;
    bar.replaceChildren(logo, ...links, make("span", { className: "nav-spacer" }),
                        searchBtn, ...(extras.length ? extras : [connIndicator()]), btn);
    showTheme(prefersDark());
    btn.addEventListener("click", () => {
      const dark = root.getAttribute("data-theme") !== "dark";
      MLPrefs.set("theme", dark ? "dark" : "light");
      showTheme(dark);
      document.dispatchEvent(new Event("mlthemechange"));
    });
  }

  const pageName = () => document.querySelector("nav[data-page]")?.dataset.page || "page";

  // Lets *sidebar* fold to a thin rail and back, with a button in its corner
  // that stays on the rail.  Each page remembers its own choice.
  function collapsibleSidebar(sidebar) {
    const name = `sidebar_collapsed_${pageName()}`;
    const btn = make("button", { type: "button", className: "ghost-btn side-toggle" });
    btn.innerHTML = PANEL_ICON;
    const show = (collapsed, save) => {
      sidebar.classList.toggle("collapsed", collapsed);
      btn.title = collapsed ? "Show the sidebar" : "Hide the sidebar";
      btn.setAttribute("aria-expanded", String(!collapsed));
      if (save) MLPrefs.set(name, collapsed);
      window.dispatchEvent(new Event("resize"));   // charts and editors refit the new width
    };
    btn.addEventListener("click", () => show(!sidebar.classList.contains("collapsed"), true));
    sidebar.classList.add("has-side-toggle");
    sidebar.prepend(btn);
    show(!!MLPrefs.get(name, false), false);
  }

  // Lets the user drag a handle on *panel*'s inner edge to resize it; the
  // panel sits at the left of its row, or at the right with side "right".
  // Double-click the handle for the default width again.  The width is kept
  // as setting *name*.  *apply(px | null)* sets it on the page; by default the
  // panel's flex-basis, capped at 45% of the window so a narrow one keeps room.
  // *handle* is an existing element to use; otherwise one is added.
  function resizable(panel, { name, side = "left", min = 160, max = 640, handle = null, apply = null, onResize = null }) {
    const set = apply || (w => {
      if (w == null) panel.style.removeProperty("flex-basis");
      else panel.style.flexBasis = `min(${w}px, 45vw)`;
    });
    const h = handle || make("div", { className: "resizer" });
    if (!handle) { if (side === "left") panel.after(h); else panel.before(h); }
    h.title = "Drag to resize; double-click for the default width";
    const saved = MLPrefs.get(name);
    if (typeof saved === "number") set(saved);
    const refit = () => { window.dispatchEvent(new Event("resize")); if (onResize) onResize(); };
    h.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      e.preventDefault();
      h.setPointerCapture(e.pointerId);
      h.classList.add("dragging");
      document.body.classList.add("resizing");
      const startX = e.clientX, start = panel.getBoundingClientRect().width;
      let width = start;
      const move = ev => {
        const dx = side === "left" ? ev.clientX - startX : startX - ev.clientX;
        width = Math.round(Math.max(min, Math.min(max, start + dx)));
        set(width);
        if (onResize) onResize();
      };
      const up = () => {
        h.classList.remove("dragging");
        document.body.classList.remove("resizing");
        h.removeEventListener("pointermove", move);
        h.removeEventListener("pointerup", up);
        h.removeEventListener("pointercancel", up);
        if (width !== start) { MLPrefs.set(name, width); refit(); }
      };
      h.addEventListener("pointermove", move);
      h.addEventListener("pointerup", up);
      h.addEventListener("pointercancel", up);
    });
    h.addEventListener("dblclick", () => { MLPrefs.remove(name); set(null); refit(); });
  }

  return { PAGES: PAGES.map(([page]) => page), pages: PAGES, nav, collapsibleSidebar, resizable };
})();

// The Ctrl/Cmd+K palette (palette.js, searching through search-core.js) on every page.
window.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("palette-back")) return;
  const core = document.createElement("script");
  core.src = "search-core.js";
  core.onload = () => document.body.append(Object.assign(document.createElement("script"), { src: "palette.js" }));
  document.body.append(core);
});
