// Custom styled confirm/alert dialogs that match the mlsweep UI theme.
(function () {
  const css = `
.ml-dialog-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center; z-index: 1000;
  animation: ml-fade-in 0.1s ease;
}
@keyframes ml-fade-in { from { opacity: 0; } to { opacity: 1; } }
.ml-dialog {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px 24px; width: 340px; max-width: 95vw;
  display: flex; flex-direction: column; gap: 16px;
  animation: ml-slide-in 0.1s ease;
}
@keyframes ml-slide-in { from { transform: translateY(-6px); opacity: 0; } to { transform: none; opacity: 1; } }
.ml-dialog-message {
  font-size: 13px; color: var(--text); line-height: 1.5; white-space: pre-wrap;
  font-family: system-ui, -apple-system, sans-serif;
}
.ml-dialog-actions { display: flex; justify-content: flex-end; gap: 8px; }
.ml-btn {
  font-size: 12px; padding: 5px 14px; border-radius: 4px; cursor: pointer;
  font-family: system-ui, -apple-system, sans-serif;
}
.ml-btn-cancel {
  border: 1px solid var(--border); background: var(--btn-bg); color: var(--btn-text);
}
.ml-btn-cancel:hover { background: var(--btn-hover); }
.ml-btn-ok {
  border: 1px solid var(--radio-checked-border); background: var(--radio-checked-bg);
  color: var(--radio-checked-color); font-weight: 600;
}
.ml-btn-ok:hover { opacity: 0.85; }
.ml-btn-danger {
  border: 1px solid color-mix(in srgb, var(--bad) 33%, transparent); background: var(--btn-bg); color: var(--bad); font-weight: 600;
}
.ml-btn-danger:hover { background: color-mix(in srgb, var(--bad) 9%, transparent); }

/* mlPick: a searchable list instead of typing an ID */
.ml-dialog.ml-pick { width: 520px; gap: 10px; padding: 16px 18px 14px; }
.ml-pick-search { width: 100%; box-sizing: border-box; padding: 7px 10px; font-size: 13px; border: 1px solid var(--border-input); border-radius: 5px; background: var(--btn-bg); color: var(--text); outline: none; }
.ml-pick-search:focus { border-color: var(--accent, var(--radio-checked-border)); }
.ml-pick-list { max-height: min(52vh, 440px); overflow-y: auto; margin: 0 -6px; padding: 0 6px; }
.ml-pick-group { font-size: 10px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: .06em; padding: 10px 8px 4px; }
.ml-pick-item { display: flex; align-items: baseline; gap: 10px; padding: 6px 8px; border-radius: 5px; cursor: pointer; font-family: system-ui, -apple-system, sans-serif; }
.ml-pick-item .l { font-size: 12.5px; color: var(--text-strong); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; flex: 1; }
.ml-pick-item .s { font-size: 11px; color: var(--text-muted); white-space: nowrap; flex-shrink: 0; }
.ml-pick-item .mono { font-family: ui-monospace, monospace; font-size: 11.5px; }
.ml-pick-item.on { background: var(--radio-checked-bg); }
.ml-pick-item.on .l { color: var(--radio-checked-color); }
.ml-pick-item.disabled { opacity: .45; cursor: default; }
.ml-pick-empty { padding: 18px 8px; text-align: center; font-size: 12px; color: var(--text-muted); }
.ml-pick-hint { font-size: 10.5px; color: var(--text-muted); }

/* mlContextMenu: what a right-click offers */
.ml-ctx { position: fixed; z-index: 1100; min-width: 210px; max-width: 340px; background: var(--surface); border: 1px solid var(--border-input); border-radius: 7px; box-shadow: 0 8px 28px rgba(0,0,0,.28); padding: 4px; font-family: system-ui, -apple-system, sans-serif; animation: ml-fade-in 0.08s ease; }
.ml-ctx-item { display: flex; align-items: center; gap: 10px; width: 100%; text-align: left; background: none; border: 0; border-radius: 4px; padding: 6px 10px; font-size: 12.5px; color: var(--text); cursor: pointer; }
.ml-ctx-item:hover, .ml-ctx-item.on { background: var(--btn-hover); }
.ml-ctx-item .l { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ml-ctx-item .h { font-size: 10.5px; color: var(--text-muted); white-space: nowrap; }
.ml-ctx-item.danger { color: var(--bad); }
.ml-ctx-item[disabled] { opacity: .45; cursor: default; background: none; }
.ml-ctx-sep { height: 1px; background: var(--border); margin: 4px 6px; }
.ml-ctx-title { font-size: 10px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: .06em; padding: 6px 10px 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
`;

  const styleEl = document.createElement("style");
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  function showDialog({ message, buttons, extraContent = null, onExtraKeydown = null }) {
    return new Promise(resolve => {
      const backdrop = document.createElement("div");
      backdrop.className = "ml-dialog-backdrop";

      const dlg = document.createElement("div");
      dlg.className = "ml-dialog";
      dlg.setAttribute("role", "dialog");
      dlg.setAttribute("aria-modal", "true");

      const msg = document.createElement("div");
      msg.className = "ml-dialog-message";
      msg.textContent = message;

      const actions = document.createElement("div");
      actions.className = "ml-dialog-actions";

      function close(value) {
        backdrop.remove();
        resolve(value);
      }

      backdrop.addEventListener("keydown", e => {
        if (e.key === "Escape") close(null);
      });

      if (extraContent && onExtraKeydown) {
        extraContent.addEventListener("keydown", e => onExtraKeydown(e, close));
      }

      for (const { label, value, getValue, cls } of buttons) {
        const btn = document.createElement("button");
        btn.className = `ml-btn ${cls}`;
        btn.textContent = label;
        btn.onclick = () => close(getValue ? getValue() : value);
        actions.appendChild(btn);
      }

      if (extraContent) dlg.append(msg, extraContent, actions);
      else dlg.append(msg, actions);
      backdrop.appendChild(dlg);
      document.body.appendChild(backdrop);
      // Focus once the dialog is in the document; focusing a detached
      // element does nothing, and keystrokes would go to the page behind.
      if (extraContent) {
        extraContent.focus();
        extraContent.select?.();
      } else {
        actions.querySelector("button:last-child").focus();
      }
    });
  }

  // mlConfirm(message, opts?) → Promise<boolean>
  // opts: { confirmLabel, cancelLabel, danger }
  window.mlConfirm = function (message, opts = {}) {
    const {
      confirmLabel = "Confirm",
      cancelLabel = "Cancel",
      danger = true,
    } = opts;
    return showDialog({
      message,
      buttons: [
        { label: cancelLabel, value: false, cls: "ml-btn-cancel" },
        { label: confirmLabel, value: true,  cls: danger ? "ml-btn-danger" : "ml-btn-ok" },
      ],
    });
  };

  // mlAlert(message, opts?) → Promise<void>
  // opts: { okLabel }
  window.mlAlert = function (message, opts = {}) {
    const { okLabel = "OK" } = opts;
    return showDialog({
      message,
      buttons: [{ label: okLabel, value: null, cls: "ml-btn-ok" }],
    });
  };

  // mlPrompt(message, defaultValue?, opts?) → Promise<string|null>
  // opts: { okLabel, cancelLabel, placeholder }
  window.mlPrompt = function (message, defaultValue = "", opts = {}) {
    const { okLabel = "OK", cancelLabel = "Cancel", placeholder = "" } = opts;
    const input = document.createElement("input");
    input.type = "text";
    input.value = defaultValue;
    input.placeholder = placeholder;
    input.style.cssText = "width:100%;padding:6px 8px;font-size:13px;border:1px solid var(--border-input);border-radius:4px;background:var(--btn-bg);color:var(--text);outline:none;box-sizing:border-box;";
    return showDialog({
      message,
      extraContent: input,
      onExtraKeydown: (e, close) => { if (e.key === "Enter") close(input.value.trim()); },
      buttons: [
        { label: cancelLabel, value: null, cls: "ml-btn-cancel" },
        { label: okLabel, getValue: () => input.value.trim(), cls: "ml-btn-ok" },
      ],
    });
  };

  // mlPick(message, items, opts?) → Promise<value|null>
  // A searchable list to choose from, so nobody has to type an ID.
  // items: [{ value, label, sub?, group?, mono?, disabled? }], shown in order, under
  // their group headings.  opts: { placeholder, okLabel, create: { label, prompt } }:
  // *create* adds an item that asks for a new name instead (e.g. a new campaign).
  window.mlPick = function (message, items, opts = {}) {
    const { placeholder = "Search…", create = null } = opts;
    return new Promise(resolve => {
      const backdrop = document.createElement("div");
      backdrop.className = "ml-dialog-backdrop";
      const dlg = document.createElement("div");
      dlg.className = "ml-dialog ml-pick";
      dlg.setAttribute("role", "dialog");
      dlg.setAttribute("aria-modal", "true");
      const msg = document.createElement("div");
      msg.className = "ml-dialog-message";
      msg.textContent = message;
      const search = document.createElement("input");
      search.className = "ml-pick-search";
      search.placeholder = placeholder;
      search.autocomplete = "off";
      search.spellcheck = false;
      const list = document.createElement("div");
      list.className = "ml-pick-list";
      list.setAttribute("role", "listbox");
      const hint = document.createElement("div");
      hint.className = "ml-pick-hint";
      hint.textContent = "↑↓ to move · Enter to choose · Esc to cancel";
      const actions = document.createElement("div");
      actions.className = "ml-dialog-actions";
      actions.style.alignItems = "center";
      const cancel = document.createElement("button");
      cancel.className = "ml-btn ml-btn-cancel";
      cancel.textContent = "Cancel";
      const spacer = document.createElement("span");
      spacer.style.flex = "1";
      actions.append(hint, spacer, cancel);
      dlg.append(msg, search, list, actions);
      backdrop.append(dlg);

      let shown = [], active = 0;
      const CREATE = {};
      async function choose(item) {
        if (!item || item.disabled) return;
        if (item.value === CREATE) {
          backdrop.remove();
          const name = await mlPrompt(create.prompt || create.label, search.value.trim(), { okLabel: "OK" });
          resolve(name || null);
          return;
        }
        backdrop.remove();
        resolve(item.value);
      }
      function render() {
        const q = search.value.trim().toLowerCase();
        shown = items.filter(it => !q || [it.label, it.sub, String(it.value)].some(t => t && String(t).toLowerCase().includes(q)));
        if (create) shown.push({ value: CREATE, label: create.label, group: "" });
        active = Math.max(0, Math.min(active, shown.length - 1));
        while (shown[active] && shown[active].disabled && active < shown.length - 1) active++;
        list.replaceChildren();
        if (!shown.length) {
          const e = document.createElement("div");
          e.className = "ml-pick-empty";
          e.textContent = "Nothing matches.";
          list.append(e);
        }
        let group = null;
        shown.forEach((it, i) => {
          if ((it.group || "") !== group) {
            group = it.group || "";
            if (group) {
              const g = document.createElement("div");
              g.className = "ml-pick-group";
              g.textContent = group;
              list.append(g);
            }
          }
          const row = document.createElement("div");
          row.className = "ml-pick-item" + (i === active ? " on" : "") + (it.disabled ? " disabled" : "");
          row.setAttribute("role", "option");
          const l = document.createElement("span");
          l.className = "l" + (it.mono ? " mono" : "");
          l.textContent = it.label;
          row.append(l);
          if (it.sub) {
            const s2 = document.createElement("span");
            s2.className = "s";
            s2.textContent = it.sub;
            row.append(s2);
          }
          row.title = it.title || String(it.value === CREATE ? "" : it.value);
          row.onmouseenter = () => { if (!it.disabled) { active = i; paint(); } };
          row.onclick = () => choose(it);
          list.append(row);
        });
      }
      function paint() {
        [...list.querySelectorAll(".ml-pick-item")].forEach((r, i) => r.classList.toggle("on", i === active));
        const on = list.querySelectorAll(".ml-pick-item")[active];
        if (on) on.scrollIntoView({ block: "nearest" });
      }
      function step(d) {
        if (!shown.length) return;
        let i = active;
        do { i = (i + d + shown.length) % shown.length; } while (shown[i].disabled && i !== active);
        active = i;
        paint();
      }
      search.addEventListener("input", () => { active = 0; render(); });
      backdrop.addEventListener("keydown", e => {
        if (e.key === "Escape") { backdrop.remove(); resolve(null); }
        else if (e.key === "ArrowDown") { e.preventDefault(); step(1); }
        else if (e.key === "ArrowUp") { e.preventDefault(); step(-1); }
        else if (e.key === "Enter") { e.preventDefault(); choose(shown[active]); }
      });
      backdrop.addEventListener("mousedown", e => { if (e.target === backdrop) { backdrop.remove(); resolve(null); } });
      cancel.onclick = () => { backdrop.remove(); resolve(null); };
      render();
      document.body.append(backdrop);
      search.focus();
    });
  };

  // mlContextMenu(event, items) shows a right-click menu at the pointer;
  // mlMenuAt(x, y, items) shows the same menu at a point (a button's corner).
  // items: [{ label, hint?, run, danger?, disabled? } | "-" | { title } | null].
  let ctx = null;
  function closeCtx() { if (ctx) { ctx.remove(); ctx = null; } }
  window.mlCloseContextMenu = closeCtx;
  window.mlContextMenu = function (event, items) {
    event.preventDefault();
    event.stopPropagation();
    window.mlMenuAt(event.clientX, event.clientY, items);
  };
  window.mlMenuAt = function (x, y, items) {
    closeCtx();
    const menu = document.createElement("div");
    menu.className = "ml-ctx";
    menu.setAttribute("role", "menu");
    for (const it of items) {
      if (!it) continue;
      if (it === "-") {
        // No separator at the top or twice in a row (items around it were left out).
        if (menu.lastChild && menu.lastChild.className !== "ml-ctx-sep") {
          const sep = document.createElement("div"); sep.className = "ml-ctx-sep"; menu.append(sep);
        }
        continue;
      }
      if (it.title) { const t = document.createElement("div"); t.className = "ml-ctx-title"; t.textContent = it.title; menu.append(t); continue; }
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ml-ctx-item" + (it.danger ? " danger" : "");
      b.setAttribute("role", "menuitem");
      if (it.disabled) b.disabled = true;
      const l = document.createElement("span");
      l.className = "l";
      l.textContent = it.label;
      b.append(l);
      if (it.hint) { const h = document.createElement("span"); h.className = "h"; h.textContent = it.hint; b.append(h); }
      b.onclick = () => { closeCtx(); it.run(); };
      menu.append(b);
    }
    while (menu.lastChild && menu.lastChild.className === "ml-ctx-sep") menu.lastChild.remove();
    document.body.append(menu);
    const w = menu.offsetWidth, h = menu.offsetHeight;
    menu.style.left = Math.max(4, Math.min(x, innerWidth - w - 4)) + "px";
    menu.style.top = Math.max(4, Math.min(y, innerHeight - h - 4)) + "px";
    ctx = menu;
  };
  // A right-click on a link, a field or selected text inside a row keeps the
  // browser's own menu (open in new tab, copy, paste); the row's menu is for
  // the rest of the row.  Pass the row so a row that is itself a link counts.
  window.mlNativeMenuWanted = function (event, row) {
    const t = event.target;
    if (!t.closest) return false;
    const inner = t.closest("a[href], input, textarea, select");
    if ((inner && inner !== row) || t.isContentEditable) return true;
    const sel = getSelection();
    return !!(sel && !sel.isCollapsed && String(sel).trim() && sel.containsNode(t, true));
  };
  // A run's other pages, as right-click menu items: every page but *skip*.
  // *code*: whether the run has a code snapshot to open.
  window.mlRunPageItems = function (expId, runId, { skip = "", code = true } = {}) {
    const q = { experiment: expId, run: runId };
    return [
      ["results", "Open results", { experiment: expId }],
      ["logs", "Open logs", q],
      ["files", "Open artifacts", q],
      code ? ["code", "Open code", q] : null,
    ].filter(p => p && p[0] !== skip)
      .map(([page, label, query]) => ({ label, run: () => { location.href = pageHref(`${page}.html`, query); } }));
  };
  document.addEventListener("mousedown", e => { if (ctx && !ctx.contains(e.target)) closeCtx(); }, true);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeCtx(); });
  window.addEventListener("blur", closeCtx);
  window.addEventListener("resize", closeCtx);
  document.addEventListener("scroll", closeCtx, true);
})();

// Experiments as mlPick items: the current campaign's first, newest first in each group.
function mlExperimentItems(exps, { exclude = [], campaign = null } = {}) {
  const skip = new Set(exclude);
  const when = e => Date.parse(e.submit_time) || 0;
  const day = e => { const d = new Date(e.submit_time); return isNaN(d) ? "" : d.toLocaleDateString([], { month: "short", day: "numeric" }); };
  const camp = e => e.campaign;
  const sorted = exps.filter(e => !skip.has(e.experiment_id)).sort((a, b) => when(b) - when(a));
  const item = (e, group) => ({
    value: e.experiment_id, label: e.name && e.name !== e.experiment_id ? e.name : e.experiment_id,
    sub: [campaign ? null : camp(e), day(e), e.status].filter(Boolean).join(" · "),
    title: e.experiment_id, group,
  });
  if (!campaign) return sorted.map(e => item(e, ""));
  return [...sorted.filter(e => camp(e) === campaign).map(e => item(e, `Campaign ${campaign}`)),
          ...sorted.filter(e => camp(e) !== campaign).map(e => ({ ...item(e, "Other campaigns"), sub: [camp(e), day(e), e.status].filter(Boolean).join(" · ") }))];
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// Human-readable byte count ("" for null).
function fmtSize(b) {
  if (b == null) return "";
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  if (b < 1073741824) return (b / 1048576).toFixed(1) + " MB";
  return (b / 1073741824).toFixed(2) + " GB";
}

// A status badge.  common.css has the colors.
function badge(cls, label = cls) {
  return `<span class="badge badge-${escHtml(cls)}">${escHtml(label)}</span>`;
}

// "5m ago" from an ISO time.
function relTime(iso) {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60e3) return "just now";
  if (diff < 3600e3) return Math.floor(diff / 60e3) + "m ago";
  if (diff < 86400e3) return Math.floor(diff / 3600e3) + "h ago";
  return Math.floor(diff / 86400e3) + "d ago";
}

// "1h2m" from seconds; *empty* for null.
function fmtElapsed(s, empty = "—") {
  if (s == null) return empty;
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  if (h > 0) return `${h}h${m}m`;
  if (m > 0) return `${m}m${sec}s`;
  return `${sec}s`;
}

// A run's display name: its label, or its run ID if it has none.
function runDisplayName(j) {
  return j.label || j.run_id;
}

// A run's display name as HTML: its label with the run ID muted beside it.
function runNameHtml(j) {
  if (!j.label) return escHtml(j.run_id);
  return `${escHtml(j.label)} <span style="color:var(--text-muted);font-size:9px">${escHtml(j.run_id)}</span>`;
}

// Ask for a run's new display name and save it.  Resolves to the new label
// (null = cleared), or undefined if cancelled, unchanged, or the save failed.
async function mlRenameRun(expId, runId, current) {
  const input = await mlPrompt(`Rename run ${runId}:`, current || "", { okLabel: "Rename", placeholder: runId });
  if (input === null) return undefined;
  const label = input.trim() || null;
  if (label === (current || null)) return undefined;
  const r = await apiFetch(`/api/jobs/${encodeURIComponent(runId)}/label`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experiment_id: expId, label }),
  });
  if (r.ok) return label;
  await mlAlert(`Rename failed: ${r.status}`);
  return undefined;
}
