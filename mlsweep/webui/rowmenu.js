// Row menus, shared by the Experiments and System pages.
// Each row has one menu of places to go ("View") and one of things to change
// ("Manage"), so a row costs two controls however many pages and verbs exist.
// Needs chrome.js (act, escHtml, pageHref, apiFetch) loaded first.
// notes are optional: with no notes folder there is no Notes entry at all.
let notesEnabled = false;
let noteMentions = {};      // experiment id -> [note paths] (its own or its runs')
let noteRunMentions = {};   // experiment id -> {run id: [note paths]}

let rowMenu = null;

// Small monochrome glyphs for the View menu, drawn with the text colour.
const VIEW_ICONS = {
  results: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13.5V2.5M2 13.5h12"/><path d="m4.5 10 3-3.5 2.5 2 3.5-5"/></svg>',
  logs: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><path d="M3 4h10M3 8h10M3 12h6"/></svg>',
  artifacts: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M8 1.5 14 4.5v7L8 14.5 2 11.5v-7z"/><path d="M2 4.5 8 7.5l6-3M8 7.5v7"/></svg>',
  notes: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="m2.5 13.5.8-3.2 7.4-7.4 2.4 2.4-7.4 7.4z"/><path d="m9.5 3.7 2.4 2.4"/></svg>',
  export: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v7.5"/><path d="m4.8 6.5 3.2 3.2 3.2-3.2"/><path d="M3 13.5h10"/></svg>',
};

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
function toggleRunView(expId, runId) {
  toggleMenu(this, `runview:${expId}/${runId}`, [
    { label: "Results", icon: "results", href: pageHref("results.html", { experiment: expId }) },
    { label: "Artifacts", icon: "artifacts", href: pageHref("files.html", { experiment: expId, run: runId }) },
    { label: "Logs", icon: "logs", href: pageHref("logs.html", { experiment: expId, run: runId }) },
    notesItem(expId, runId),
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
