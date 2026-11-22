// Row menus, shared by the Experiments and System pages.
// Each row has one menu of places to go ("View") and one of things to change
// ("Manage"), so a row costs two controls however many pages and verbs exist.
// Needs chrome.js (act, escHtml, pageHref, apiFetch, VIEW_ICONS) loaded first.
// notes are optional: with no notes folder there is no Notes entry at all.
let notesEnabled = false;
let noteMentions = {};      // experiment id -> [note paths] (its own or its runs')
let noteRunMentions = {};   // experiment id -> {run id: [note paths]}
// capture is optional too: a manager started with --no-capture has no traces.
let captureEnabled = false;
let traceCounts = {};       // experiment id -> linked trace count

let rowMenu = null;

// POST a run verb (cancel, retry) and report a failure; the caller refreshes.
async function postRunAction(expId, runId, action) {
  const r = await apiFetch(`/api/jobs/${encodeURIComponent(runId)}/${action}?experiment_id=${encodeURIComponent(expId)}`, { method: "POST" });
  if (!r.ok) { const body = await r.json().catch(() => ({})); await mlAlert(`${action} failed: ${body.error || r.status}`); }
}

function closeRowMenu() { if (rowMenu) { rowMenu.remove(); rowMenu = null; } }

function rowMenuItem(it) {
  const cls = `row-menu-item${it.danger ? " danger" : ""}${it.disabled ? " disabled" : ""}`;
  const title = it.title ? ` title="${escHtml(it.title)}"` : "";
  const icon = it.icon ? `<span class="mi-icon">${VIEW_ICONS[it.icon] || ""}</span>` : "";
  const count = it.count != null ? `<span class="mi-count">${it.count}</span>` : "";
  const body = `${icon}<span class="mi-label">${escHtml(it.label)}</span>${count}`;
  if (it.disabled) return `<span class="${cls}"${title}>${body}</span>`;
  if (it.href) return `<a class="${cls}" href="${it.href}"${title}>${body}</a>`;
  return `<button class="${cls}"${title} ${it.act}>${body}</button>`;
}

function openRowMenu(anchor, key, items) {
  closeRowMenu();
  const menu = document.createElement("div");
  menu.className = "row-menu";
  menu.dataset.for = key;
  menu.setAttribute("role", "menu");
  menu.innerHTML = items.map(it =>
    (it.divider ? "<hr>" : "") + rowMenuItem(it)).join("");
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  const w = menu.offsetWidth, h = menu.offsetHeight;
  menu.style.left = `${Math.max(8, Math.min(r.right - w, window.innerWidth - w - 8))}px`;
  menu.style.top = `${r.bottom + 4 + h > window.innerHeight - 8 ? Math.max(8, r.top - h - 4) : r.bottom + 4}px`;
  rowMenu = menu;
}

// A second click on the same button closes the menu it opened.
function toggleMenu(anchor, key, items) {
  if (rowMenu && rowMenu.dataset.for === key) return closeRowMenu();
  openRowMenu(anchor, key, items);
}

function noteCount(expId, runId) {
  if (runId) return ((noteRunMentions[expId] || {})[runId] || []).length;
  return (noteMentions[expId] || []).length;
}

function notesItem(expId, runId) {
  const what = runId || expId;
  const n = noteCount(expId, runId);
  if (!notesEnabled) return null;
  const item = { label: "Notes", icon: "notes", count: n };
  if (!n) return { ...item, disabled: true, title: `No note mentions ${what} yet` };
  return {
    ...item,
    href: pageHref("notes.html", { mentions: what }),
    title: `${n} note${n === 1 ? "" : "s"} mention ${what}`,
  };
}

// Where a run's View menu leads.
function tracesItem(expId, runId) {
  if (!captureEnabled) return null;
  const what = runId || expId;
  const item = { label: "Traces", icon: "traces" };
  if (runId) return { ...item, href: pageHref("traces.html", { experiment: expId }),
                      title: "The coding-agent sessions around this run" };
  const n = traceCounts[expId] ?? 0;
  if (!traceCounts.__loaded) return { ...item, href: pageHref("traces.html", { experiment: expId }),
                                     title: "The coding-agent sessions around this" };
  if (!n) return { ...item, disabled: true, count: n, title: `No trace submitted ${what} yet` };
  return {
    ...item, count: n,
    href: pageHref("traces.html", { experiment: expId }),
    title: `${n} trace${n === 1 ? "" : "s"} submitted ${what}`,
  };
}

function toggleRunView(expId, runId) {
  toggleMenu(this, `runview:${expId}/${runId}`, [
    { label: "Results", icon: "results", href: pageHref("results.html", { experiment: expId }) },
    { label: "Artifacts", icon: "artifacts", href: pageHref("files.html", { experiment: expId, run: runId }) },
    { label: "Logs", icon: "logs", href: pageHref("logs.html", { experiment: expId, run: runId }) },
    { label: "Code", icon: "code", href: pageHref("code.html", { experiment: expId, run: runId }) },
    notesItem(expId, runId),
    tracesItem(expId, runId),
  ].filter(Boolean));
}

// A click elsewhere, Esc, or scrolling the list closes the menu.
document.addEventListener("click", e => {
  if (!rowMenu) return;
  if (e.target.closest(".row-menu-item")) return closeRowMenu();
  if (e.target.closest(".row-menu") || e.target.closest(".btn-manage") || e.target.closest(".btn-menu")) return;
  closeRowMenu();
});
document.addEventListener("keydown", e => { if (e.key === "Escape") closeRowMenu(); });
window.addEventListener("scroll", closeRowMenu, true);
window.addEventListener("resize", closeRowMenu);

// What cites each experiment and each of its runs (empty with no --notes).
// onChange runs whenever the answer differs from the last one.
function watchNoteMentions(onChange) {
  async function poll() {
    try {
      const r = await apiFetch("/api/notes/mentions");
      if (!r.ok) return;
      const d = await r.json();
      const next = d.mentions || {};
      const nextRuns = d.runs || {};
      const nextEnabled = d.enabled !== false;
      const changed = nextEnabled !== notesEnabled
        || JSON.stringify(next) !== JSON.stringify(noteMentions)
        || JSON.stringify(nextRuns) !== JSON.stringify(noteRunMentions);
      noteMentions = next;
      noteRunMentions = nextRuns;
      notesEnabled = nextEnabled;
      if (changed) onChange();
    } catch { /* notes are optional */ }
  }
  poll();
  setInterval(() => { if (!document.hidden) poll(); }, 30000);
}

// Which experiment how many traces link to, so the View menu can show the count
// (and go disabled without one).  The count endpoint answers 404 on a manager
// with capture off; then there is no Traces entry either.
function watchTraceCounts(onChange) {
  let loaded = false;
  async function poll() {
    try {
      const r = await apiFetch("/api/capture/sessions?limit=200&state=all");
      if (r.status === 404) { captureEnabled = false; return; }
      if (!r.ok) return;
      const sessions = await r.json();
      captureEnabled = true;
      const next = {};
      for (const s of sessions) for (const e of s.experiments || []) next[e] = (next[e] || 0) + 1;
      next.__loaded = true;
      const changed = !loaded || JSON.stringify(next) !== JSON.stringify(traceCounts);
      loaded = true;
      traceCounts = next;
      if (changed) onChange();
    } catch { /* capture is optional */ }
  }
  poll();
  setInterval(() => { if (!document.hidden) poll(); }, 30000);
}
