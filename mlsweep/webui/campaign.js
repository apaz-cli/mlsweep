// Campaign selection shared by the dashboard pages.
//
// The selected campaign comes from ?campaign= in the URL, else the last one
// picked (localStorage), else "default".  "*" selects every campaign.  The
// choice is written back to the URL and to localStorage, and links to other
// dashboard pages carry it along.
const MLCampaign = (function () {
  const ALL = "*";
  const DEFAULT = "default";
  const KEY = "mlsweep_campaign";
  const PAGES = /(^|\/)(experiments|results|logs|files|system|index)\.html$/;

  function load() {
    try { return localStorage.getItem(KEY); } catch { return null; }
  }
  function save(c) {
    try { localStorage.setItem(KEY, c); } catch {}
  }

  let current = new URLSearchParams(location.search).get("campaign") || load() || DEFAULT;
  save(current);
  const listeners = [];

  function campaignOf(exp) { return (exp && exp.campaign) || DEFAULT; }

  // A same-site dashboard URL with this campaign selected.
  function href(url) {
    const u = new URL(url, location.href);
    if (u.origin !== location.origin || !PAGES.test(u.pathname)) return url;
    u.searchParams.set("campaign", current);
    return u.pathname.split("/").pop() + u.search + u.hash;
  }

  function linkNav() {
    document.querySelectorAll("a.nav-link").forEach(a => {
      const raw = a.getAttribute("href");
      if (raw && raw !== "#") a.setAttribute("href", href(raw));
    });
  }

  // Show the selection in the address bar and in links to the other pages.
  function publish() {
    const u = new URL(location.href);
    u.searchParams.set("campaign", current);
    history.replaceState(null, "", u.pathname + u.search + u.hash);
    linkNav();
  }

  function set(c) {
    current = c || DEFAULT;
    save(current);
    publish();
    listeners.forEach(fn => fn(current));
  }

  // Campaign names to offer. These are the campaigns of *exps*, plus the default and the current one.
  function names(exps) {
    const s = new Set(exps.map(campaignOf));
    s.add(DEFAULT);
    if (current !== ALL) s.add(current);
    return [...s].sort();
  }

  function filter(exps) {
    return current === ALL ? exps : exps.filter(e => campaignOf(e) === current);
  }

  // When a deep-linked experiment is in another campaign, select that campaign.
  // Only the first call per page acts, so callers can run it on every fetch.
  let followed = false;
  function follow(exps, expId) {
    if (followed) return;
    followed = true;
    const e = expId && exps.find(x => x.experiment_id === expId);
    if (e && current !== ALL && campaignOf(e) !== current) set(campaignOf(e));
  }

  // Rebuilt only when the names or the selection change, so polling does not
  // close an open dropdown.
  function fill(select, exps) {
    const opts = [[ALL, "All campaigns"], ...names(exps).map(n => [n, n])];
    const key = current + "\n" + opts.map(o => o[0]).join("\n");
    if (select.dataset.campaignKey === key) return;
    select.dataset.campaignKey = key;
    select.innerHTML = "";
    for (const [value, label] of opts) {
      const o = document.createElement("option");
      o.value = value;
      o.textContent = label;
      o.selected = value === current;
      select.appendChild(o);
    }
  }

  // Fill an experiment *select* with *exps*, selecting the first of
  // *preferred* that is listed, else the first (newest) one.  Returns the
  // selected experiment id, or null when *exps* is empty.
  function fillExperiments(select, exps, ...preferred) {
    const ids = new Set(exps.map(e => e.experiment_id));
    const chosen = preferred.find(id => id && ids.has(id)) || (exps[0]?.experiment_id ?? null);
    select.innerHTML = "";
    for (const e of exps) {
      const o = document.createElement("option");
      o.value = e.experiment_id;
      o.textContent = e.name || e.experiment_id;
      o.selected = e.experiment_id === chosen;
      select.appendChild(o);
    }
    if (!exps.length) select.innerHTML = `<option value="" disabled selected>No experiments</option>`;
    return chosen;
  }

  // Wire *select* to the selection; *onChange* runs after each user change.
  function bind(select, onChange) {
    select.addEventListener("change", () => set(select.value));
    if (onChange) listeners.push(onChange);
  }

  // Links built after load (and clicked before linkNav ran) still get the campaign.
  function rewrite(e) {
    const a = e.target.closest && e.target.closest("a[href]");
    if (a && a.getAttribute("href") !== "#") a.setAttribute("href", href(a.getAttribute("href")));
  }
  document.addEventListener("click", rewrite, true);
  document.addEventListener("auxclick", rewrite, true);
  // After the page script has run (it may strip ?token= from the URL).
  document.addEventListener("DOMContentLoaded", publish);

  return {
    ALL, DEFAULT,
    get: () => current,
    set, campaignOf, names, filter, follow, fill, fillExperiments, bind, href, linkNav,
  };
})();
