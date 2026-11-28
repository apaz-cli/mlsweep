// Row menus, shared by the Experiments and System pages.
// Each row has one menu of places to go ("View") and one of things to change
// ("Manage"), so a row costs two controls however many pages and verbs exist.
// Needs chrome.js (act, escHtml, pageHref, apiFetch, VIEW_ICONS) and viewmenu.js
// (what the View menus list) loaded first.
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

// codeFiles: how many files the run's code snapshot has (null: no snapshot).
function toggleRunView(expId, runId, codeFiles = null) {
  toggleMenu(this, `runview:${expId}/${runId}`, MLViews.runItems(expId, runId, { codeFiles }));
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
