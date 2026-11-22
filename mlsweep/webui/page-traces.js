// The Traces page: coding-agent sessions the capture watcher uploaded,
// readable as a ledger (narrow event-tag lane, then the content) with an
// inspector pane for the selected event.  See `mlsweep docs traces`.

const params = new URLSearchParams(location.search);

MLChrome.collapsibleSidebar(document.getElementById("sidebar"));
MLChrome.resizable(document.getElementById("sidebar"), { name: "sidebar_width_traces", min: 200, max: 520 });

// ── DOM refs ──────────────────────────────────────────────────────────────────
const searchEl = document.getElementById("search");
const sessionList = document.getElementById("session-list");
const agentChips = document.getElementById("agent-chips");
const traceTitle = document.getElementById("trace-title");
const traceAgent = document.getElementById("trace-agent");
const traceMeta = document.getElementById("trace-meta");
const traceLinks = document.getElementById("trace-links");
const inFilter = document.getElementById("in-filter");
const dlBtn = document.getElementById("dl-btn");
const refreshBtn = document.getElementById("refresh-btn");
const deleteBtn = document.getElementById("delete-btn");
const ledgerWrap = document.getElementById("ledger-wrap");
const ledger = document.getElementById("ledger");
const noSel = document.getElementById("no-sel");
const inspector = document.getElementById("inspector");
const inspectorKind = document.getElementById("inspector-kind");
const inspectorWhere = document.getElementById("inspector-where");
const inspectorText = document.getElementById("inspector-text");
const inspectorOverview = document.getElementById("inspector-overview");
const inspectorClose = document.getElementById("inspector-close");

// ── State ─────────────────────────────────────────────────────────────────────
let sessions = [];            // what /api/capture/sessions last answered
let searching = false;        // the search box shows matches, not the list
let agentFilter = "";
let currentId = null;
let events = [];              // the current session's events
let pollTimer = null;
let selectedSeq = null;

const AGENT_LABELS = { pi: "pi", claude_code: "claude code" };

// ── Session list ──────────────────────────────────────────────────────────────

function timeAgo(ts) {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "now";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

function sessionTitle(s) {
  return s.title || (s.cwd ? s.cwd.split("/").filter(Boolean).pop() : s.session_id);
}

function sessionRow(s) {
  const div = document.createElement("div");
  div.className = "session-item" + (s.session_id === currentId ? " selected" : "");
  div.dataset.session = s.session_id;
  const top = document.createElement("div");
  top.className = "session-item-top";
  const dot = document.createElement("span");
  dot.className = `state-dot ${s.state || ""}`;
  dot.title = s.state || "";
  const title = document.createElement("span");
  title.className = "session-title";
  title.textContent = sessionTitle(s);
  title.title = s.session_id;
  const agent = document.createElement("span");
  agent.className = "session-agent";
  agent.textContent = AGENT_LABELS[s.agent] || s.agent || "?";
  top.append(dot, title, agent);
  const bottom = document.createElement("div");
  bottom.className = "session-item-bottom";
  const cwd = document.createElement("span");
  cwd.className = "session-cwd";
  cwd.textContent = s.cwd || "";
  const when = document.createElement("span");
  when.textContent = timeAgo(s.last_seen_at);
  when.title = new Date((s.last_seen_at || 0) * 1000).toLocaleString();
  bottom.append(cwd, when);
  if (s.event_count) {
    const n = document.createElement("span");
    n.textContent = `${s.event_count}`;
    n.title = `${s.event_count} events`;
    bottom.append(n);
  }
  if ((s.experiments || []).length) {
    const links = document.createElement("span");
    links.className = "session-links";
    links.textContent = `${s.experiments.length} exp`;
    links.title = s.experiments.join(", ");
    bottom.append(links);
  }
  div.append(top, bottom);
  if (s.snippet) {
    const snip = document.createElement("div");
    snip.className = "snippet";
    snip.textContent = s.snippet;
    div.append(snip);
  }
  div.addEventListener("click", () => openSession(s.session_id));
  return div;
}

function renderSessions() {
  sessionList.replaceChildren();
  if (!sessions.length) {
    const msg = document.createElement("div");
    msg.className = "list-msg";
    msg.textContent = searching && searchEl.value ? "No traces match." : "No traces yet.";
    sessionList.append(msg);
    return;
  }
  for (const s of sessions) sessionList.append(sessionRow(s));
}

async function loadSessions() {
  try {
    const q = searchEl.value.trim();
    let path;
    if (q) {
      searching = true;
      path = `/api/capture/search?q=${encodeURIComponent(q)}&limit=50`;
      const r = await apiFetch(path);
      sessions = r.ok ? await r.json() : [];
    } else {
      searching = false;
      path = `/api/capture/sessions?limit=100&state=all` + (agentFilter ? `&agent=${agentFilter}` : "");
      const r = await apiFetch(path);
      sessions = r.ok ? await r.json() : [];
    }
    if (searching) sessions = sessions.filter(s => !agentFilter || s.agent === agentFilter);
    renderSessions();
  } catch { /* offline: keep what we had */ }
}

searchEl.addEventListener("input", () => { clearTimeout(loadSessions._t); loadSessions._t = setTimeout(loadSessions, 250); });
agentChips.addEventListener("click", e => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  agentFilter = chip.dataset.agent;
  agentChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c === chip));
  loadSessions();
});

// ── The ledger ────────────────────────────────────────────────────────────────

function shortText(e) {
  if (e.kind === "tool_result") {
    const status = e.ok === false ? "error" : "ok";
    return `<span class="event-${e.ok === false ? "err" : "ok"}">${escHtml(status)}</span>`
      + (e.duration_ms != null ? `<span class="event-arrow">·</span>${e.duration_ms} ms` : "");
  }
  return escHtml(e.text || "");
}

function kindLabel(e) {
  if (e.kind === "tool") return e.name || "tool";
  if (e.kind === "tool_result") return "result";
  return e.kind;
}

function kindTagClass(e) {
  if (e.kind === "tool" || e.kind === "tool_result") return `kind-${e.kind}`;
  return `kind-${e.kind}`;
}

// A user event starts a turn; turn numbers count from the first one.
let turnCount = 0;

function ledgerRow(e, index) {
  const tr = document.createElement("tr");
  tr.className = `event kind-${e.kind}` + (e.kind === "user" ? " turn-start" : "");
  tr.dataset.seq = e.seq;
  if (e.seq === selectedSeq) tr.classList.add("selected");
  const tag = document.createElement("td");
  tag.className = "tag-col";
  if (e.kind === "user") {
    const label = document.createElement("span");
    label.className = "turn-label";
    label.textContent = `#${++turnCount}`;
    tag.append(label);
  }
  const pill = document.createElement("span");
  pill.className = `kind-tag ${kindTagClass(e)}`;
  pill.textContent = kindLabel(e);
  tag.append(pill);
  const content = document.createElement("td");
  const text = document.createElement("span");
  text.className = "event-text";
  text.innerHTML = shortText(e);
  content.append(text);
  if (e.kind === "tool" && e.text) {
    const arrow = document.createElement("span");
    arrow.className = "event-arrow";
    arrow.textContent = "→";
    arrow.title = "result below";
    content.append(arrow);
  }
  tr.append(tag, content);
  tr.addEventListener("click", () => inspect(e, tr));
  return tr;
}

function renderLedger() {
  turnCount = 0;
  const tbody = ledger.querySelector("tbody");
  tbody.replaceChildren();
  const filter = inFilter.value.trim().toLowerCase();
  let shown = 0;
  for (const e of events) {
    if (filter && !(`${e.text || ""} ${e.name || ""}`.toLowerCase().includes(filter))) continue;
    tbody.append(ledgerRow(e, shown++));
  }
  ledger.style.display = events.length ? "" : "none";
  noSel.style.display = events.length ? "none" : "";
  if (!events.length && !filter) {
    noSel.querySelector("div").textContent = "This trace is empty.";
  } else {
    noSel.innerHTML = `<div>Pick a trace from the left, or turn capture on:</div>` +
      `<div style="font-family:var(--font-mono);font-size:11px;color:var(--text-dim)">mlsweep capture on</div>`;
  }
  ledgerWrap.scrollTop = ledgerWrap.scrollHeight;
}

inFilter.addEventListener("input", () => renderLedger());

// ── Inspector ─────────────────────────────────────────────────────────────────

function inspect(e, tr) {
  selectedSeq = e.seq;
  ledger.querySelectorAll("tr.selected").forEach(x => x.classList.remove("selected"));
  if (tr) tr.classList.add("selected");
  inspector.classList.add("open");
  inspectorKind.className = `kind-tag ${kindTagClass(e)}`;
  inspectorKind.textContent = kindLabel(e);
  inspectorWhere.textContent = `event ${e.seq} · ${new Date((e.ts || 0) * 1000).toLocaleString()}`;
  inspectorText.className = "event-full" + (e.kind === "tool" ? " tool" : "");
  inspectorText.textContent = e.text || (e.kind === "tool_result" ? (e.ok === false ? "error" : "ok") : "");
  const rows = [];
  const put = (k, v) => { if (v != null && v !== "") rows.push([k, String(v)]); };
  put("model", e.model);
  if (e.tokens) put("tokens", `${e.tokens.input ?? "?"} in · ${e.tokens.output ?? "?"} out`);
  put("tool", e.name);
  put("tool id", e.tool_id);
  put("result", e.kind === "tool_result" ? (e.ok === false ? "error" : "ok") : null);
  put("duration", e.duration_ms != null ? `${e.duration_ms} ms` : null);
  inspectorOverview.replaceChildren(
    ...rows.map(([k, v]) => {
      const d = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      dd.title = v;
      d.append(dt, dd);
      return d;
    }));
}

inspectorClose.addEventListener("click", () => {
  inspector.classList.remove("open");
  selectedSeq = null;
  ledger.querySelectorAll("tr.selected").forEach(x => x.classList.remove("selected"));
});

// ── Session header + events ──────────────────────────────────────────────────

function renderHeader(s) {
  traceTitle.textContent = sessionTitle(s) || "Trace";
  traceTitle.title = s.session_id;
  traceAgent.style.display = "";
  traceAgent.textContent = AGENT_LABELS[s.agent] || s.agent || "";
  traceMeta.textContent = [
    s.model, s.cwd,
    `${s.event_count ?? "?"} events`,
    new Date((s.started_at || 0) * 1000).toLocaleString(),
  ].filter(Boolean).join("  ·  ");
  traceLinks.replaceChildren();
  for (const eid of s.experiments || []) {
    const a = document.createElement("a");
    a.className = "link-chip";
    a.href = pageHref("experiments.html", { id: eid });
    a.textContent = eid;
    traceLinks.append(a);
  }
  dlBtn.style.display = "";
  dlBtn.href = authURL(`/api/capture/sessions/${encodeURIComponent(s.session_id)}/transcript`);
  deleteBtn.style.display = "";
  inFilter.style.display = "";
}

async function openSession(id) {
  currentId = id;
  selectedSeq = null;
  inspector.classList.remove("open");
  document.querySelectorAll(".session-item").forEach(x =>
    x.classList.toggle("selected", x.dataset.session === id));
  const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(id)}`);
  if (!r.ok) return;
  const s = await r.json();
  renderHeader(s);
  await loadEvents(s, 0);
  renderSessions();
  // A live session grows: poll while it stays active.
  clearInterval(pollTimer);
  if (s.state === "active") {
    pollTimer = setInterval(async () => {
      if (s.state !== "active") return clearInterval(pollTimer);
      await loadEvents(s, events.length ? events[events.length - 1].seq + 1 : 0, { append: true });
    }, 5000);
  }
}

async function loadEvents(s, fromSeq, { append = false } = {}) {
  // Pages of 1000 until the session is caught up (bounded, sessions are small).
  // Collect into a local list: the page that ends the loop is empty, and must
  // not replace the pages that came before it.
  const loaded = [];
  let next = fromSeq;
  for (let page = 0; page < 20; page++) {
    const r = await apiFetch(
      `/api/capture/sessions/${encodeURIComponent(s.session_id)}/events?from_seq=${next}&limit=1000`);
    if (!r.ok) return;
    const body = await r.json();
    if (!body.events.length) break;
    loaded.push(...body.events);
    next = body.next_from_seq;
  }
  if (append) events.push(...loaded);
  else events = loaded;
  renderLedger();
}

refreshBtn.addEventListener("click", () => { loadSessions(); if (currentId) openSession(currentId); });

deleteBtn.addEventListener("click", async () => {
  if (!currentId) return;
  const yes = await window.mlConfirm(`Delete trace ${currentId}? This cannot be undone.`);
  if (!yes) return;
  const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(currentId)}`, { method: "DELETE" });
  if (r.ok) {
    currentId = null;
    ledger.style.display = "none";
    noSel.style.display = "";
    traceTitle.textContent = "No trace selected";
    traceMeta.textContent = "";
    traceAgent.style.display = "none";
    traceLinks.replaceChildren();
    dlBtn.style.display = "none";
    deleteBtn.style.display = "none";
    loadSessions();
  }
});

// ── Boot ──────────────────────────────────────────────────────────────────────
const wanted = params.get("session");
loadSessions().then(() => { if (wanted) openSession(wanted); });
setInterval(loadSessions, 15000);
