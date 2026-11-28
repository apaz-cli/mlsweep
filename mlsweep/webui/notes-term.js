// mlsweep Notes: terminal tabs.  Each is a shell on the manager's machine, in
// its sandbox (mlsweep/_sessions.py), shown with xterm.js.  The manager owns
// the shell, so closing the browser leaves it running; the tab reattaches and
// catches up from the last byte it saw.  Closing the tab ends the shell.
"use strict";

S.term = { enabled: false, error: null, config_error: null, commands: [], sessions: [] };

let sessionsSeq = 0;
async function loadSessions() {
  const seq = ++sessionsSeq;
  try {
    const info = await api("/api/sessions");
    if (seq === sessionsSeq) Object.assign(S.term, info);
  } catch { if (seq === sessionsSeq) S.term.enabled = false; }
  return S.term;
}

function loadXterm() {
  if (typeof Terminal !== "undefined") return Promise.resolve();
  return nbLoadScript("vendor-xterm.js", "the terminal library", "vendor-xterm.css");
}

function termTheme() {
  return {
    background: nbCssVar("--doc-bg") || "#1f1f1f", foreground: nbCssVar("--text") || "#ddd",
    cursor: nbCssVar("--text-strong") || "#fff", cursorAccent: nbCssVar("--doc-bg") || "#1f1f1f",
    selectionBackground: (nbCssVar("--radio-checked-border") || "#4c9be8") + "55",
  };
}
document.addEventListener("mlthemechange", () => {
  for (const t of S.tabs) if (t.type === "term" && t.term && t.term.x) t.term.x.options.theme = termTheme();
});

// Open a new terminal (optionally starting one of terminals.json's *command*s) as a tab in the focused group.
async function newTerminal(command = null, folder = "") {
  if (!S.term.enabled) { toast("Terminals are off. The manager was started with --no-terminals.", "err"); return; }
  if (S.term.error) { toast(S.term.error, "err"); return; }
  let info;
  const body = {};
  if (command) body.command = command;
  if (folder) body.folder = folder;
  try { info = await apiJSON("/api/sessions", "POST", body); }
  catch (e) { toast("Could not open a terminal: " + e.message, "err"); return; }
  openTerminalTab(info.id, folder ? `${info.title}: ${basename(folder)}` : info.title);
}
function openTerminalTab(sid, title, g = focusedGroup()) {
  const open = S.tabs.find(t => t.type === "term" && t.sid === sid);
  if (open) { activate(open.id); return; }
  const t = newTab({ type: "term", sid, title: title || "Terminal" });
  const cur = tabById(g.active);
  addTab(t, g, cur ? g.tabs.indexOf(cur.id) + 1 : g.tabs.length);
  activate(t.id);
}

async function termMenu(e) {
  e.preventDefault();
  const at = [e.clientX, e.clientY];
  await loadSessions();   // terminals.json is read afresh, so the menu shows its commands as they are now
  const items = [{ label: "New terminal", run: () => newTerminal() }];
  for (const c of S.term.commands) items.push({ label: `New terminal: ${c}`, run: () => newTerminal(c) });
  if (S.term.config_error) items.push("-", { label: "terminals.json has a problem…", run: () => toast(S.term.config_error, "err") });
  const loose = S.term.sessions.filter(s => !s.exited && !S.tabs.some(t => t.type === "term" && t.sid === s.id));
  if (loose.length) {
    items.push("-");
    for (const s of loose) items.push({ label: `Attach: ${s.title} (${ago(s.created)})`, run: () => openTerminalTab(s.id, s.title) });
  }
  mlMenuAt(at[0], at[1], items);
}

// ── The tab's view ─────────────────────────────────────────────────────────────
// The xterm and its socket belong to the tab, so moving the tab to another group,
// or redrawing its group, keeps the screen and the connection.
async function renderTermView(view, t) {
  const pane = el("div", "vpane term-pane");
  view.append(pane);
  if (!t.term) {
    t.term = { host: el("div", "term-host"), x: null, fit: null, ws: null, offset: 0, exited: false, closing: false, retries: 0 };
  }
  pane.append(t.term.host);
  if (!t.term.x) {
    try { await loadXterm(); }
    catch (e) { pane.replaceChildren(el("div", "placeholder", e.message)); return; }
    if (!t.term.host.isConnected || t.term.x) return;   // the tab closed or another render made it
    startTerm(t);
  }
  requestAnimationFrame(() => {
    fitTerm(t);
    if (isShown(t) && groupOf(t).id === S.focus) t.term.x.focus();
  });
}

function startTerm(t) {
  const T = t.term;
  const x = new Terminal({
    cursorBlink: true, allowProposedApi: true, scrollback: 10000, theme: termTheme(),
    // Answer "how big are you in pixels / cells" (CSI 14t, 16t, 18t), which image previewers ask.
    windowOptions: { getWinSizePixels: true, getCellSizePixels: true, getWinSizeChars: true },
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", "mlsweep Nerd Symbols", monospace', fontSize: 13,
  });
  T.x = x;
  T.fit = new FitAddon.FitAddon();
  x.loadAddon(T.fit);
  x.loadAddon(new WebLinksAddon.WebLinksAddon((e, uri) => window.open(uri, "_blank", "noopener")));
  x.loadAddon(new Unicode11Addon.Unicode11Addon());
  x.unicode.activeVersion = "11";
  x.open(T.host);
  const enc = new TextEncoder();
  const send = data => { if (T.ws && T.ws.readyState === WebSocket.OPEN) T.ws.send(data); };
  x.onData(d => {
    if (T.exited) { closeTab(t.id); return; }   // "press any key to close"
    send(enc.encode(d));
  });
  x.onBinary(d => send(Uint8Array.from(d, c => c.charCodeAt(0))));
  x.onResize(() => send(termSize(t)));
  x.onTitleChange(title => {
    const next = title || t.baseTitle || "Terminal";
    if (next !== t.title) { t.title = next; renderTabs(); }
  });
  t.baseTitle = t.title;
  x.attachCustomKeyEventHandler(e => {
    if (e.type !== "keydown") return true;
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key.toLowerCase();
    // Copy a selection rather than interrupting; paste goes through the browser's paste event.
    if (mod && k === "c" && x.hasSelection()) { copyText(x.getSelection()); x.clearSelection(); return false; }
    if (mod && k === "v") return false;
    // Ctrl+S would freeze output (XOFF); here it means save, as everywhere on the page.
    if (mod && k === "s") return false;
    return true;
  });
  T.ro = new ResizeObserver(debounce(() => fitTerm(t), 40));
  T.ro.observe(T.host);
  connectTerm(t);
}

// The resize message: cells, and the pixels they cover, which the pty reports to programs.
function termSize(t) {
  const x = t.term.x;
  const css = x._core && x._core._renderService && x._core._renderService.dimensions && x._core._renderService.dimensions.css;
  const cell = css && css.cell;
  const width = cell && cell.width ? Math.round(cell.width * x.cols) : 0;
  const height = cell && cell.height ? Math.round(cell.height * x.rows) : 0;
  return JSON.stringify({ type: "resize", cols: x.cols, rows: x.rows, width, height });
}

function fitTerm(t) {
  const T = t.term;
  if (!T || !T.x || !T.host.isConnected || !T.host.clientWidth) return;
  try { T.fit.fit(); } catch { /* not laid out yet */ }
}

function connectTerm(t) {
  const T = t.term;
  const ws = new WebSocket(wsURL(`/ws/sessions/${encodeURIComponent(t.sid)}`) + `&since=${T.offset}`);
  ws.binaryType = "arraybuffer";
  T.ws = ws;
  ws.onopen = () => { T.retries = 0; ws.send(termSize(t)); };
  ws.onmessage = ev => {
    if (typeof ev.data !== "string") { T.offset += ev.data.byteLength; T.x.write(new Uint8Array(ev.data)); return; }
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m.type === "reset") T.x.reset();
    else if (m.type === "hello") T.offset = m.offset;
    else if (m.type === "exit") termEnded(t, `exited with code ${m.code}`);
  };
  ws.onclose = async () => {
    if (T.ws !== ws || T.closing || T.exited) return;
    // The socket dropped.  If the shell is still there, pick up where we left off.
    await new Promise(r => setTimeout(r, Math.min(5000, 500 * 2 ** T.retries++)));
    if (T.closing || T.exited || !S.tabs.includes(t)) return;
    await loadSessions();
    if (S.term.sessions.some(s => s.id === t.sid && !s.exited)) connectTerm(t);
    else termEnded(t, "ended (the manager may have restarted)");
  };
}

function termEnded(t, why) {
  const T = t.term;
  if (T.exited) return;
  T.exited = true;
  T.x.write(`\r\n\x1b[31m[terminal ${why}; press any key to close the tab]\x1b[0m\r\n`);
  t.title = (t.baseTitle || "Terminal") + " (ended)";
  renderTabs();
}

// When a terminal tab closes, so does its shell.
function disposeTerm(t) {
  const T = t.term;
  if (T) {
    T.closing = true;
    if (T.ws) T.ws.close();
    if (T.ro) T.ro.disconnect();
    if (T.x) T.x.dispose();
  }
  if (!T || !T.exited) api(`/api/sessions/${encodeURIComponent(t.sid)}`, { method: "DELETE" }).catch(() => {});
}
