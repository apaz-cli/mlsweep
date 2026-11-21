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

async function apiFetch(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${TOKEN}` } });
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
  code: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="m5.5 4.5-3.5 3.5 3.5 3.5M10.5 4.5 14 8l-3.5 3.5"/></svg>',
  export: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v7.5"/><path d="m4.8 6.5 3.2 3.2 3.2-3.2"/><path d="M3 13.5h10"/></svg>',
  system: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2.5" width="12" height="4.5" rx="1"/><rect x="2" y="9" width="12" height="4.5" rx="1"/><path d="M4.6 4.75h.01M4.6 11.25h.01M7 4.75h.01M7 11.25h.01" stroke-width="2"/></svg>',
};

const MLChrome = (() => {
  const PAGES = [
    ["experiments", "Experiments"], ["results", "Results"], ["logs", "Logs"], ["files", "Artifacts"],
    ["code", "Code"], ["notes", "Notes"], ["export", "Export"], ["system", "System"],
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
  const prefersDark = () => localStorage.getItem("theme") !== "light";
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
      a.append(label);
      return a;
    });
    const btn = make("button", { id: "theme-toggle" });
    const searchBtn = make("button", { id: "palette-open", className: "nav-search-btn", type: "button",
                                       title: "Search runs, experiments, notes and docs" });
    searchBtn.innerHTML = `Search <kbd>${/Mac|iPhone|iPad/.test(navigator.platform || "") ? "⌘K" : "Ctrl K"}</kbd>`;
    bar.replaceChildren(logo, ...links, make("span", { className: "nav-spacer" }),
                        searchBtn, ...(extras.length ? extras : [connIndicator()]), btn);
    showTheme(prefersDark());
    btn.addEventListener("click", () => {
      const dark = root.getAttribute("data-theme") !== "dark";
      localStorage.setItem("theme", dark ? "dark" : "light");
      showTheme(dark);
      document.dispatchEvent(new Event("mlthemechange"));
    });
  }

  return { PAGES: PAGES.map(([page]) => page), nav };
})();

// The Ctrl/Cmd+K palette (palette.js, searching through search-core.js) on every page.
window.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("palette-back")) return;
  const core = document.createElement("script");
  core.src = "search-core.js";
  core.onload = () => document.body.append(Object.assign(document.createElement("script"), { src: "palette.js" }));
  document.body.append(core);
});
