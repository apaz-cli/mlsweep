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
let traceCounts = null;     // experiment id -> linked trace count (null until loaded)

let rowMenu = null;

// POST a run verb (cancel, retry) and report a failure; the caller refreshes.
async function postRunAction(expId, runId, action) {
  const r = await apiFetch(`/api/jobs/${encodeURIComponent(runId)}/${action}?experiment_id=${encodeURIComponent(expId)}`, { method: "POST" });
  if (!r.ok) { const body = await r.json().catch(() => ({})); await mlAlert(`${action} failed: ${body.error || r.status}`); }
}

// A menu's "Copy … ID" entry (an act() handler, so it takes plain arguments).
function copyId(id, toast) { copyText(id, toast); }

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

// anchor is the button the menu drops from, or a {x, y} point (a right-click).
function openRowMenu(anchor, key, items) {
  closeRowMenu();
  const menu = document.createElement("div");
  menu.className = "row-menu";
  menu.dataset.for = key;
  menu.setAttribute("role", "menu");
  menu.innerHTML = items.map(it =>
    (it.divider ? "<hr>" : "") + rowMenuItem(it)).join("");
  document.body.appendChild(menu);
  const w = menu.offsetWidth, h = menu.offsetHeight;
  if (anchor.getBoundingClientRect) {
    const r = anchor.getBoundingClientRect();
    menu.style.left = `${Math.max(8, Math.min(r.right - w, window.innerWidth - w - 8))}px`;
    menu.style.top = `${r.bottom + 4 + h > window.innerHeight - 8 ? Math.max(8, r.top - h - 4) : r.bottom + 4}px`;
  } else {
    menu.style.left = `${Math.max(8, Math.min(anchor.x, window.innerWidth - w - 8))}px`;
    menu.style.top = `${Math.max(8, Math.min(anchor.y, window.innerHeight - h - 8))}px`;
  }
  rowMenu = menu;
}

// A row's View and Manage menus as one list, Manage below a divider.
function joinMenus(view, manage) {
  if (!view.length || !manage.length) return [...view, ...manage];
  return [...view, { ...manage[0], divider: true }, ...manage.slice(1)];
}

// Right-clicking a row opens menu(row) at the pointer.  menu returns
// [key, items], or null to leave the browser's menu alone.
function bindRowContextMenu(container, selector, menu) {
  container.addEventListener("contextmenu", e => {
    const row = e.target.closest(selector);
    if (!row || !container.contains(row) || mlNativeMenuWanted(e, row)) return;
    const got = menu(row);
    if (!got) return;
    e.preventDefault();
    openRowMenu({ x: e.clientX, y: e.clientY }, got[0], got[1]);
  });
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
  const item = { label: "Traces", icon: "traces", href: pageHref("traces.html", { experiment: expId }) };
  // A run has no traces of its own: it shows its experiment's.
  if (runId || !traceCounts) return { ...item, title: `The coding-agent sessions around ${runId || expId}` };
  const n = traceCounts[expId] ?? 0;
  if (!n) return { ...item, href: null, disabled: true, count: n, title: `No trace submitted ${expId} yet` };
  return { ...item, count: n, title: `${n} trace${n === 1 ? "" : "s"} submitted ${expId}` };
}

// codeFiles: how many files the run's code snapshot has (null: no snapshot).
function toggleRunView(expId, runId, codeFiles = null) {
  toggleMenu(this, `runview:${expId}/${runId}`, runViewItems(expId, runId, codeFiles));
}

function codeItem(href, codeFiles) {
  return { label: "Code", icon: "code", href, count: codeFiles ?? null,
           title: codeFiles != null ? `${codeFiles} file${codeFiles === 1 ? "" : "s"} of code` : null };
}

function runViewItems(expId, runId, codeFiles = null) {
  return [
    { label: "Results", icon: "results", href: pageHref("results.html", { experiment: expId }) },
    { label: "Artifacts", icon: "artifacts", href: pageHref("files.html", { experiment: expId, run: runId }) },
    { label: "Logs", icon: "logs", href: pageHref("logs.html", { experiment: expId, run: runId }) },
    codeItem(pageHref("code.html", { experiment: expId, run: runId }), codeFiles),
    notesItem(expId, runId),
    tracesItem(expId, runId),
  ].filter(Boolean);
}

// A click elsewhere, Esc, or scrolling the list closes the menu.
document.addEventListener("click", e => {
  if (!rowMenu) return;
  if (e.target.closest(".row-menu-item")) return closeRowMenu();
  if (e.target.closest(".row-menu") || e.target.closest(".btn-manage") || e.target.closest(".btn-menu")) return;
  closeRowMenu();
});
// A right-click anywhere else closes it too (a row's own handler then opens its menu).
document.addEventListener("contextmenu", e => { if (!e.target.closest(".row-menu")) closeRowMenu(); }, true);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeRowMenu(); });
window.addEventListener("scroll", closeRowMenu, true);
window.addEventListener("resize", closeRowMenu);

// What cites each experiment and each of its runs (empty with no --notes).
// onChange runs whenever the answer differs from the last one.
function watchNoteMentions(onChange) {
  let seq = 0;
  async function poll() {
    const mine = ++seq;
    try {
      const r = await apiFetch("/api/notes/mentions");
      if (mine !== seq || !r.ok) return;
      const d = await r.json();
      if (mine !== seq) return;
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
// (and go disabled without one).  The counts endpoint answers 404 on a manager
// with capture off; then there is no Traces entry either.
function watchTraceCounts(onChange) {
  let seq = 0;
  async function poll() {
    const mine = ++seq;
    try {
      const r = await apiFetch("/api/capture/counts");
      if (mine !== seq) return;
      if (r.status === 404) {
        if (captureEnabled) { captureEnabled = false; onChange(); }
        return;
      }
      if (!r.ok) return;
      const next = (await r.json()).experiments || {};
      if (mine !== seq) return;
      const changed = !captureEnabled || JSON.stringify(next) !== JSON.stringify(traceCounts);
      captureEnabled = true;
      traceCounts = next;
      if (changed) onChange();
    } catch { /* capture is optional */ }
  }
  poll();
  setInterval(() => { if (!document.hidden) poll(); }, 30000);
}
