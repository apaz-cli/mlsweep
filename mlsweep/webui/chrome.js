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

function apiFetch(path, opts = {}) {
  return fetch(path, { ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${TOKEN}` } });
}

// *path* for the places a request can't carry a header: images, downloads.
function authURL(path) {
  return TOKEN ? path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(TOKEN) : path;
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

const MLChrome = (() => {
  const PAGES = [
    ["experiments", "Experiments"], ["results", "Results"], ["logs", "Logs"], ["files", "Artifacts"],
    ["notes", "Notes"], ["export", "Export"], ["system", "System"],
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
    const links = PAGES.map(([page, label]) => page === bar.dataset.page
      ? make("a", { className: "nav-link active", href: "#", textContent: label })
      : make("a", { className: "nav-link", id: `nav-${page}`, href: pageHref(`${page}.html`), textContent: label }));
    const btn = make("button", { id: "theme-toggle" });
    bar.replaceChildren(logo, ...links, make("span", { className: "nav-spacer" }),
                        ...(extras.length ? extras : [connIndicator()]), btn);
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
