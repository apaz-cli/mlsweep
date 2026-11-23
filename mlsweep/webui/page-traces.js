// The Traces page: coding-agent sessions the capture watcher uploaded,
// readable as a ledger (time, event tag, then the content) with an inspector
// pane for the selected event.  See `mlsweep docs traces`.

const params = new URLSearchParams(location.search);

MLChrome.collapsibleSidebar(document.getElementById("sidebar"));
MLChrome.resizable(document.getElementById("sidebar"), { name: "sidebar_width_traces", min: 200, max: 520 });

// ── DOM refs ──────────────────────────────────────────────────────────────────
const searchEl = document.getElementById("search");
const sessionList = document.getElementById("session-list");
const moreBtn = document.getElementById("more-btn");
const listInfo = document.getElementById("list-info");
const agentChips = document.getElementById("agent-chips");
const stateChips = document.getElementById("state-chips");
const traceTitle = document.getElementById("trace-title");
const traceAgent = document.getElementById("trace-agent");
const traceState = document.getElementById("trace-state");
const traceMeta = document.getElementById("trace-meta");
const traceLinks = document.getElementById("trace-links");
const linkBtn = document.getElementById("link-btn");
const traceNotes = document.getElementById("trace-notes");
const inFilter = document.getElementById("in-filter");
const dlBtn = document.getElementById("dl-btn");
const menuBtn = document.getElementById("menu-btn");
const refreshBtn = document.getElementById("refresh-btn");
const turnNav = document.getElementById("turn-nav");
const turnPrev = document.getElementById("turn-prev");
const turnNext = document.getElementById("turn-next");
const turnPos = document.getElementById("turn-pos");
const ledgerWrap = document.getElementById("ledger-wrap");
const ledger = document.getElementById("ledger");
const noSel = document.getElementById("no-sel");
const inspector = document.getElementById("inspector");
const inspectorKind = document.getElementById("inspector-kind");
const inspectorWhere = document.getElementById("inspector-where");
const inspectorText = document.getElementById("inspector-text");
const inspectorOverview = document.getElementById("inspector-overview");
const inspectorCopy = document.getElementById("inspector-copy");
const inspectorClose = document.getElementById("inspector-close");

// ── State ─────────────────────────────────────────────────────────────────────
let sessions = [];            // what the list/search endpoint last answered
let searching = false;        // the search box shows matches, not the list
let captureOff = false;       // the manager answered 404: started with --no-capture
let agentFilter = params.get("agent") || "";
let stateFilter = params.get("state") || "all";
let experimentFilter = params.get("experiment") || "";
let listLimit = 100;

let current = null;           // the open session's header, as the manager last told it
let events = [];
let eventsBySeq = new Map();  // seq -> event
let rowsBySeq = new Map();    // seq -> <tr> (only rows the current filter shows)
let toolRowByCallId = new Map();
let resultByToolId = new Map();
let turnOfSeq = new Map();    // seq -> turn number
let turnStarts = [];          // seq of each user prompt, in order
let turnCount = 0;            // turns seen so far (also the counter while appending)
let selectedSeq = null;
let pollTimer = null;
let pinned = false;           // the user is reading the tail: follow new events
let notesCache = new Map();   // session id -> how many notes cite it
let notesEnabled = true;      // the manager answered 404: no notes served

const AGENT_LABELS = { pi: "pi", claude_code: "claude code" };
const MAX_LIST = 1000;
const POLL_MS = 5000;

// Search snippets mark matches with these (see SEARCH_MARK_* in _manager_db.py).
const MARK_START = "\u0001";
const MARK_END = "\u0002";

// ── Small utilities ──────────────────────────────────────────────────────────

function timeAgo(ts) {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "now";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

function fmtClock(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

function fmtDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function sessionTitle(s) {
  return s.title || (s.cwd ? s.cwd.split("/").filter(Boolean).pop() : s.session_id);
}

function agentLabel(s) {
  return AGENT_LABELS[s.agent] || s.agent || "";
}

// The page's state, in the address bar: a reload or a copied link keeps the
// trace, the selected event, and the filters.
function syncURL() {
  const q = new URLSearchParams();
  if (current) q.set("session", current.session_id);
  if (selectedSeq != null) q.set("event", String(selectedSeq));
  if (searchEl.value.trim()) q.set("q", searchEl.value.trim());
  if (agentFilter) q.set("agent", agentFilter);
  if (stateFilter && stateFilter !== "all") q.set("state", stateFilter);
  if (experimentFilter) q.set("experiment", experimentFilter);
  const qs = q.toString();
  history.replaceState(null, "", location.pathname + (qs ? `?${qs}` : ""));
}

function updateChips(container, attr, value) {
  container.querySelectorAll(".chip").forEach(chip => {
    const on = (chip.dataset[attr] || "") === value;
    chip.classList.toggle("active", on);
    chip.setAttribute("aria-pressed", String(on));
  });
}

function snippetNodes(text) {
  const frag = document.createDocumentFragment();
  let rest = String(text || "");
  for (let guard = 0; rest && guard < 100; guard++) {
    const start = rest.indexOf(MARK_START);
    const end = start < 0 ? -1 : rest.indexOf(MARK_END, start + 1);
    if (start < 0 || end < 0) { frag.append(document.createTextNode(rest)); break; }
    if (start > 0) frag.append(document.createTextNode(rest.slice(0, start)));
    const mark = document.createElement("mark");
    mark.textContent = rest.slice(start + 1, end);
    frag.append(mark);
    rest = rest.slice(end + 1);
  }
  return frag;
}

// ── Sidebar: sessions ────────────────────────────────────────────────────────

function sessionRow(s) {
  const div = document.createElement("div");
  div.className = "session-item" + (current && s.session_id === current.session_id ? " selected" : "");
  div.dataset.session = s.session_id;
  div.tabIndex = 0;
  div.setAttribute("role", "option");
  div.setAttribute("aria-selected", String(Boolean(current && s.session_id === current.session_id)));

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
  agent.textContent = agentLabel(s) || "?";
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
    snip.append(snippetNodes(s.snippet));
    div.append(snip);
  }
  div.addEventListener("click", () => openSession(s.session_id));
  div.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      openSession(s.session_id);
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      e.stopPropagation();
      const items = [...sessionList.querySelectorAll(".session-item")];
      const at = items.indexOf(div);
      const next = items[Math.max(0, Math.min(items.length - 1, at + (e.key === "ArrowDown" ? 1 : -1)))];
      if (next) next.focus();
    }
  });
  return div;
}

function renderSessions() {
  sessionList.replaceChildren();
  if (!sessions.length) {
    const msg = document.createElement("div");
    msg.className = "list-msg";
    msg.textContent = captureOff ? "Capture is off on this manager."
      : searching || activeFilters().length ? "No traces match."
      : "No traces yet.";
    sessionList.append(msg);
    return;
  }
  for (const s of sessions) sessionList.append(sessionRow(s));
}

function activeFilters() {
  const out = [];
  if (agentFilter) out.push(agentLabel({ agent: agentFilter }));
  if (stateFilter !== "all") out.push(stateFilter);
  if (experimentFilter) out.push(`exp ${experimentFilter}`);
  if (searchEl.value.trim()) out.push(`“${searchEl.value.trim()}”`);
  return out;
}

function renderListInfo() {
  listInfo.replaceChildren();
  if (captureOff) {
    listInfo.textContent = "Capture is off on this manager.";
    moreBtn.hidden = true;
    return;
  }
  const n = sessions.length;
  const count = searching ? `${n} match${n === 1 ? "" : "es"}` : `${n} trace${n === 1 ? "" : "s"}`;
  const filters = activeFilters();
  listInfo.append(document.createTextNode([count, ...filters].join(" · ")));
  if (filters.length) {
    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "clear";
    clear.title = "Clear search and filters";
    clear.addEventListener("click", clearFilters);
    listInfo.append(" ", clear);
  }
  moreBtn.hidden = searching || captureOff || n < listLimit || listLimit >= MAX_LIST;
}

function matchesFilters(s) {
  if (agentFilter && s.agent !== agentFilter) return false;
  if (stateFilter !== "all" && s.state !== stateFilter) return false;
  if (experimentFilter && !(s.experiments || []).includes(experimentFilter)) return false;
  return true;
}

async function loadSessions() {
  const q = searchEl.value.trim();
  const atTop = sessionList.scrollTop;
  try {
    let r;
    if (q) {
      searching = true;
      r = await apiFetch(`/api/capture/search?q=${encodeURIComponent(q)}&limit=50`);
    } else {
      searching = false;
      const query = new URLSearchParams({ limit: String(listLimit), state: stateFilter });
      if (agentFilter) query.set("agent", agentFilter);
      if (experimentFilter) query.set("experiment", experimentFilter);
      r = await apiFetch(`/api/capture/sessions?${query}`);
    }
    if (r.status === 404) {
      captureOff = true;
      sessions = [];
    } else {
      if (!r.ok) return;
      captureOff = false;
      sessions = await r.json();
      if (searching) sessions = sessions.filter(matchesFilters);
    }
    renderSessions();
    renderListInfo();
    sessionList.scrollTop = atTop;
    if (!current) showNoSelection();
  } catch { /* offline: keep what we had */ }
}

function clearFilters() {
  searchEl.value = "";
  agentFilter = "";
  stateFilter = "all";
  experimentFilter = "";
  listLimit = 100;
  updateChips(agentChips, "agent", "");
  updateChips(stateChips, "state", "all");
  loadSessions();
  syncURL();
}

// The pane on the right when nothing is open.
function showNoSelection(message, hint) {
  noSel.replaceChildren();
  const line = document.createElement("div");
  line.textContent = message
    || (captureOff ? "Capture is off on this manager." : "Pick a trace from the left.");
  noSel.append(line);
  const text = hint === undefined
    ? (captureOff ? "Start the manager without --no-capture to record traces." : "mlsweep capture this")
    : hint;
  if (text) {
    const sub = document.createElement("div");
    sub.style.cssText = "font-family:var(--font-mono);font-size:11px;color:var(--text-dim)";
    sub.textContent = text;
    noSel.append(sub);
  }
  noSel.style.display = "";
  ledger.style.display = "none";
}

// ── The ledger ───────────────────────────────────────────────────────────────

function indexEvents() {
  eventsBySeq = new Map();
  resultByToolId = new Map();
  turnOfSeq = new Map();
  turnStarts = [];
  turnCount = 0;
  for (const e of events) indexOne(e);
}

function indexOne(e) {
  if (e.kind === "user") {
    turnCount += 1;
    turnStarts.push(e.seq);
  }
  turnOfSeq.set(e.seq, turnCount);
  eventsBySeq.set(e.seq, e);
  if (e.kind === "tool_result" && e.tool_id) resultByToolId.set(e.tool_id, e);
}

function matchesEventFilter(e, needle) {
  return `${e.text || ""} ${e.name || ""} ${e.kind || ""}`.toLowerCase().includes(needle);
}

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

function makeRow(e) {
  const turn = turnOfSeq.get(e.seq) || 0;
  const tr = document.createElement("tr");
  tr.className = `event kind-${e.kind}` + (e.kind === "user" ? " turn-start" : "");
  tr.dataset.seq = e.seq;
  tr.dataset.turn = String(turn);
  if (e.seq === selectedSeq) tr.classList.add("selected");

  const time = document.createElement("td");
  time.className = "time-col";
  time.textContent = fmtClock(e.ts);
  time.title = e.ts ? new Date(e.ts * 1000).toLocaleString() : "";

  const tag = document.createElement("td");
  tag.className = "tag-col";
  if (e.kind === "user") {
    const label = document.createElement("span");
    label.className = "turn-label";
    label.textContent = `#${turn}`;
    tag.append(label);
  }
  const pill = document.createElement("span");
  pill.className = `kind-tag kind-${e.kind}`;
  pill.textContent = kindLabel(e);
  tag.append(pill);

  const content = document.createElement("td");
  content.className = "content-col";
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

  tr.append(time, tag, content);
  tr.addEventListener("click", () => selectEvent(e.seq, { open: true }));
  return tr;
}

// A tool call shows its result inline (status and how long), as well as in its
// own `result` row: the pairing is what people look for.
function decorateToolRows() {
  for (const [toolId, tr] of toolRowByCallId) {
    const result = resultByToolId.get(toolId);
    if (!result) continue;
    const content = tr.querySelector(".content-col");
    if (!content) continue;
    let status = content.querySelector(".event-status");
    if (!status) {
      status = document.createElement("span");
      status.className = "event-status";
      status.title = "Show the result";
      status.addEventListener("click", ev => { ev.stopPropagation(); jumpToSeq(result.seq); });
      content.append(status);
    }
    const ok = result.ok !== false;
    status.textContent = `${ok ? "✓" : "✗"}${result.duration_ms != null ? ` ${result.duration_ms} ms` : ""}`;
    status.classList.toggle("ok", ok);
    status.classList.toggle("err", !ok);
  }
}

function buildLedger({ keepScroll = false } = {}) {
  const anchor = keepScroll ? firstVisibleSeq() : null;
  const tbody = ledger.querySelector("tbody");
  tbody.replaceChildren();
  rowsBySeq.clear();
  toolRowByCallId.clear();
  const needle = inFilter.value.trim().toLowerCase();
  for (const e of events) {
    if (needle && !matchesEventFilter(e, needle)) continue;
    const tr = makeRow(e);
    tbody.append(tr);
    rowsBySeq.set(e.seq, tr);
    if (e.kind === "tool" && e.tool_id) toolRowByCallId.set(e.tool_id, tr);
  }
  decorateToolRows();
  if (!events.length) {
    showNoSelection("This trace is empty.");
  } else {
    noSel.style.display = "none";
    ledger.style.display = "";
    if (anchor != null && rowsBySeq.has(anchor)) scrollRowToTop(rowsBySeq.get(anchor));
    else if (!keepScroll) { ledgerWrap.scrollTop = 0; pinned = false; }
  }
  updateTurnNav();
}

function appendEvents(fresh) {
  if (!fresh.length) return;
  const wasPinned = pinned || atBottom();
  const needle = inFilter.value.trim().toLowerCase();
  const tbody = ledger.querySelector("tbody");
  for (const e of fresh) {
    events.push(e);
    indexOne(e);
    if (needle && !matchesEventFilter(e, needle)) continue;
    const tr = makeRow(e);
    tbody.append(tr);
    rowsBySeq.set(e.seq, tr);
    if (e.kind === "tool" && e.tool_id) toolRowByCallId.set(e.tool_id, tr);
  }
  decorateToolRows();
  if (current) {
    current.event_count = Math.max(current.event_count || 0, events.length);
    renderHeader(current, { notes: false });
  }
  if (wasPinned) { ledgerWrap.scrollTop = ledgerWrap.scrollHeight; pinned = true; }
  updateTurnNav();
}

function atBottom() {
  return ledgerWrap.scrollTop + ledgerWrap.clientHeight >= ledgerWrap.scrollHeight - 12;
}

function firstVisibleSeq() {
  const top = ledgerWrap.getBoundingClientRect().top;
  for (const e of events) {
    const tr = rowsBySeq.get(e.seq);
    if (tr && tr.getBoundingClientRect().bottom > top) return e.seq;
  }
  return null;
}

function scrollRowToTop(tr) {
  const offset = tr.getBoundingClientRect().top - ledgerWrap.getBoundingClientRect().top;
  ledgerWrap.scrollTop = Math.max(0, ledgerWrap.scrollTop + offset);
}

function scrollRowIntoView(tr) {
  const wrap = ledgerWrap.getBoundingClientRect();
  const box = tr.getBoundingClientRect();
  if (box.top >= wrap.top + 4 && box.bottom <= wrap.bottom - 4) return;
  ledgerWrap.scrollTop += box.top - wrap.top - Math.max(0, wrap.height / 2 - box.height / 2);
}

// ── Turns ────────────────────────────────────────────────────────────────────

function currentTurn() {
  if (selectedSeq != null && turnOfSeq.has(selectedSeq)) {
    const tr = rowsBySeq.get(selectedSeq);
    if (tr) {
      const box = tr.getBoundingClientRect();
      const wrap = ledgerWrap.getBoundingClientRect();
      if (box.bottom > wrap.top && box.top < wrap.bottom) return turnOfSeq.get(selectedSeq);
    }
  }
  if (!turnStarts.length) return 0;
  const wrapTop = ledgerWrap.getBoundingClientRect().top + 8;
  let seen = 0;
  for (const seq of turnStarts) {
    const tr = rowsBySeq.get(seq);
    if (!tr) continue;                       // filtered out, not proof it is past
    if (tr.getBoundingClientRect().top <= wrapTop) seen = turnOfSeq.get(seq);
    else break;
  }
  return seen || 1;
}

function updateTurnNav() {
  const total = turnStarts.length;
  const show = Boolean(current) && total > 0;
  turnNav.style.display = show ? "" : "none";
  if (!show) return;
  const turn = Math.min(total, Math.max(1, currentTurn()));
  turnPos.textContent = `${turn} / ${total}`;
  turnPrev.disabled = turn <= 1;
  turnNext.disabled = turn >= total;
}

function gotoPrompt(delta) {
  if (!current || !turnStarts.length) return;
  const target = Math.min(turnStarts.length, Math.max(1, currentTurn() + delta));
  jumpToSeq(turnStarts[target - 1]);
}

// ── Selection and inspector ──────────────────────────────────────────────────

function selectEvent(seq, { open = false } = {}) {
  selectedSeq = seq;
  for (const [rowSeq, tr] of rowsBySeq) tr.classList.toggle("selected", rowSeq === seq);
  const event = eventsBySeq.get(seq);
  if (event && (open || inspector.classList.contains("open"))) inspect(event);
  syncURL();
  updateTurnNav();
}

function jumpToSeq(seq) {
  if (!rowsBySeq.has(seq) && inFilter.value.trim()) {
    inFilter.value = "";
    buildLedger({ keepScroll: true });
  }
  const tr = rowsBySeq.get(seq);
  if (!tr) return;
  scrollRowIntoView(tr);
  selectEvent(seq, { open: true });
  pinned = atBottom();
}

function inspect(e) {
  inspector.classList.add("open");
  inspectorKind.className = `kind-tag kind-${e.kind}`;
  inspectorKind.textContent = kindLabel(e);
  inspectorWhere.textContent = `event ${e.seq} · ${e.ts ? new Date(e.ts * 1000).toLocaleString() : "no timestamp"}`;
  inspectorText.className = "event-full" + (e.kind === "tool" ? " tool" : "");
  inspectorText.textContent = e.kind === "tool_result"
    ? (e.ok === false ? "error" : "ok") + (e.duration_ms != null ? ` · ${e.duration_ms} ms` : "")
    : (e.text || "");
  const rows = [];
  const put = (key, value) => { if (value != null && value !== "") rows.push([key, String(value)]); };
  put("seq", e.seq);
  put("model", e.model);
  if (e.tokens) put("tokens", `${e.tokens.input ?? "?"} in · ${e.tokens.output ?? "?"} out`);
  put("tool", e.name);
  put("tool id", e.tool_id);
  if (e.kind === "tool_result") {
    put("result", e.ok === false ? "error" : "ok");
    put("duration", e.duration_ms != null ? `${e.duration_ms} ms` : null);
  }
  const result = e.kind === "tool" && e.tool_id ? resultByToolId.get(e.tool_id) : null;
  if (e.kind === "tool") {
    rows.push(["result", result
      ? (result.ok === false ? "error" : "ok") + (result.duration_ms != null ? ` · ${result.duration_ms} ms` : "")
      : "not recorded", result ? () => jumpToSeq(result.seq) : null]);
  }
  inspectorOverview.replaceChildren(...rows.map(([key, value, run]) => {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    if (run) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "overview-link";
      button.textContent = value;
      button.title = "Show this event";
      button.addEventListener("click", run);
      dd.append(button);
    } else {
      dd.textContent = value;
      dd.title = value;
    }
    wrap.append(dt, dd);
    return wrap;
  }));
}

function closeInspector() {
  inspector.classList.remove("open");
  selectedSeq = null;
  for (const tr of rowsBySeq.values()) tr.classList.remove("selected");
  syncURL();
  updateTurnNav();
}

// ── Header ───────────────────────────────────────────────────────────────────

function renderLinks(s) {
  traceLinks.replaceChildren();
  for (const eid of s.experiments || []) {
    const wrap = document.createElement("span");
    wrap.className = "chip-wrap";
    const link = document.createElement("a");
    link.className = "link-chip";
    link.href = pageHref("experiments.html", { id: eid });
    link.textContent = eid;
    link.title = eid;
    const unlink = document.createElement("button");
    unlink.type = "button";
    unlink.className = "unlink";
    unlink.textContent = "×";
    unlink.title = `Unlink ${eid} from this trace`;
    unlink.setAttribute("aria-label", `Unlink ${eid}`);
    unlink.addEventListener("click", ev => { ev.preventDefault(); ev.stopPropagation(); unlinkExperiment(eid); });
    wrap.append(link, unlink);
    traceLinks.append(wrap);
  }
}

function renderHeader(s, { notes = true } = {}) {
  const started = s.started_at ? new Date(s.started_at * 1000).toLocaleString() : "";
  const duration = s.started_at && s.last_seen_at > s.started_at
    ? ` · ${fmtDuration(s.last_seen_at - s.started_at)}` : "";
  traceTitle.textContent = sessionTitle(s) || "Trace";
  traceTitle.title = s.session_id;
  document.title = `mlsweep — ${sessionTitle(s) || "Trace"}`;
  traceAgent.style.display = "";
  traceAgent.textContent = agentLabel(s);
  traceState.style.display = "";
  traceState.className = `trace-state ${s.state || ""}`.trim();
  traceState.textContent = s.state || "";
  traceMeta.textContent = [
    s.model, s.cwd, `${events.length || s.event_count || 0} events${duration}`, started,
  ].filter(Boolean).join("  ·  ");
  renderLinks(s);
  traceLinks.style.display = (s.experiments || []).length ? "" : "none";
  linkBtn.style.display = "";
  dlBtn.style.display = "";
  dlBtn.href = authURL(`/api/capture/sessions/${encodeURIComponent(s.session_id)}/transcript`);
  menuBtn.style.display = "";
  inFilter.style.display = "";
  if (notes) loadNotes(s);
}

async function loadNotes(s) {
  const id = s.session_id;
  traceNotes.style.display = "none";
  traceNotes.removeAttribute("href");
  if (notesCache.has(id)) { showNotes(notesCache.get(id), id); return; }
  if (!notesEnabled) return;
  try {
    const r = await apiFetch(`/api/notes/search?q=${encodeURIComponent(id)}&limit=20`);
    if (r.status === 404) { notesEnabled = false; return; }
    const count = r.ok ? ((await r.json()).results || []).length : 0;
    notesCache.set(id, count);
    if (current && current.session_id === id) showNotes(count, id);
  } catch {
    notesCache.set(id, 0);
  }
}

function showNotes(count, id) {
  if (!id || !count) return;
  traceNotes.href = pageHref("notes.html", { q: id });
  traceNotes.textContent = `cited in ${count} note${count === 1 ? "" : "s"}`;
  traceNotes.style.display = "";
}

async function linkExperiment() {
  if (!current) return;
  try {
    if (!linkExperiment.exps) {
      const r = await apiFetch("/api/experiments");
      if (!r.ok) return;
      linkExperiment.exps = await r.json();
    }
    const items = mlExperimentItems(linkExperiment.exps, { exclude: current.experiments || [] });
    const pick = await mlPick("Link this trace to an experiment:", items,
      { placeholder: "Search experiments" });
    if (!pick) return;
    const r = await apiFetch("/api/capture/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: current.session_id, experiment_id: pick, source: "manual" }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      await mlAlert(body.error || `Link failed (${r.status})`);
      return;
    }
    await refreshCurrent();
  } catch { /* linking is a convenience */ }
}

async function unlinkExperiment(id) {
  if (!current) return;
  const r = await apiFetch("/api/capture/link", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: current.session_id, experiment_id: id }),
  });
  if (r.ok) await refreshCurrent();
}

function openMenu() {
  if (!current) return;
  const sid = current.session_id;
  const rect = menuBtn.getBoundingClientRect();
  mlContextMenu({
    clientX: rect.left, clientY: rect.bottom + 2,
    preventDefault() {}, stopPropagation() {},
  }, [
    { label: "Copy note reference", hint: "mlsweep:trace/…",
      run: () => copyText(`mlsweep:trace/${sid}`, "Copied trace reference") },
    { label: "Copy trace id", run: () => copyText(sid, "Copied trace id") },
    { label: "Download transcript",
      run: () => window.open(authURL(`/api/capture/sessions/${encodeURIComponent(sid)}/transcript`), "_blank") },
    "-",
    { label: "Delete trace", danger: true, run: deleteCurrent },
  ]);
}

// ── Opening, loading, polling ────────────────────────────────────────────────

function markSelectedSession() {
  sessionList.querySelectorAll(".session-item").forEach(item => {
    const on = Boolean(current && item.dataset.session === current.session_id);
    item.classList.toggle("selected", on);
    item.setAttribute("aria-selected", String(on));
  });
}

async function openSession(id, { eventSeq = null } = {}) {
  const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(id)}`);
  if (r.status === 404) {
    await loadSessions();
    showNoSelection("That trace is gone.", "Pick another from the left.");
    return;
  }
  if (!r.ok) return;
  const s = await r.json();
  current = s;
  clearInterval(pollTimer);
  selectedSeq = null;
  inspector.classList.remove("open");
  inFilter.value = "";
  // Nothing from the previous trace survives the switch, even if the load fails.
  events = [];
  eventsBySeq.clear();
  rowsBySeq.clear();
  toolRowByCallId.clear();
  resultByToolId.clear();
  turnOfSeq.clear();
  turnStarts = [];
  turnCount = 0;
  ledger.querySelector("tbody").replaceChildren();
  showNoSelection("Loading trace…", "");
  markSelectedSession();
  await loadEvents(s, 0);
  renderHeader(s);
  updateTurnNav();
  markSelectedSession();
  syncURL();
  if (eventSeq != null) jumpToSeq(eventSeq);
  startPoll();
}

async function refreshCurrent() {
  if (!current) return;
  const keep = selectedSeq;
  await openSession(current.session_id);
  if (keep != null) jumpToSeq(keep);
}

async function loadEvents(s, fromSeq) {
  // Pages of 1000 until the session is caught up.  The page that ends the loop
  // is empty, and must not replace the pages that came before it.
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
  if (fromSeq === 0) {
    events = loaded;
    indexEvents();
    buildLedger();
  } else {
    appendEvents(loaded);
  }
}

function startPoll() {
  clearInterval(pollTimer);
  if (!current || current.state !== "active") return;
  pollTimer = setInterval(async () => {
    if (!current || document.hidden) return;
    const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(current.session_id)}`);
    if (r.status === 404) { clearInterval(pollTimer); return; }
    if (!r.ok) return;
    const s = await r.json();
    const grew = (s.event_count || 0) > events.length;
    const stateChanged = s.state !== current.state;
    current = s;
    renderHeader(s, { notes: false });
    if (grew) {
      await loadEvents(s, events.length ? events[events.length - 1].seq + 1 : 0);
    }
    if (stateChanged) updateTurnNav();
    if (s.state !== "active") clearInterval(pollTimer);
  }, POLL_MS);
}

async function deleteCurrent() {
  if (!current) return;
  const id = current.session_id;
  const yes = await mlConfirm(`Delete trace ${id}? Its events and experiment links go with it.`);
  if (!yes) return;
  const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) {
    await mlAlert(`Delete failed (${r.status})`);
    return;
  }
  clearInterval(pollTimer);
  current = null;
  events = [];
  eventsBySeq.clear();
  rowsBySeq.clear();
  toolRowByCallId.clear();
  resultByToolId.clear();
  turnOfSeq.clear();
  turnStarts = [];
  turnCount = 0;
  selectedSeq = null;
  notesCache.delete(id);
  inspector.classList.remove("open");
  inFilter.value = "";
  inFilter.style.display = "none";
  ledger.querySelector("tbody").replaceChildren();
  traceTitle.textContent = "No trace selected";
  traceTitle.title = "";
  document.title = "mlsweep — Traces";
  traceMeta.textContent = "";
  traceAgent.style.display = "none";
  traceState.style.display = "none";
  traceLinks.replaceChildren();
  traceLinks.style.display = "none";
  linkBtn.style.display = "none";
  traceNotes.style.display = "none";
  dlBtn.style.display = "none";
  menuBtn.style.display = "none";
  turnNav.style.display = "none";
  showNoSelection();
  syncURL();
  await loadSessions();
}

// ── Wiring ───────────────────────────────────────────────────────────────────

searchEl.addEventListener("input", () => {
  clearTimeout(searchEl._t);
  searchEl._t = setTimeout(() => { listLimit = 100; loadSessions(); syncURL(); }, 250);
});

agentChips.addEventListener("click", e => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  agentFilter = chip.dataset.agent || "";
  updateChips(agentChips, "agent", agentFilter);
  listLimit = 100;
  loadSessions();
  syncURL();
});

stateChips.addEventListener("click", e => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  stateFilter = chip.dataset.state || "all";
  updateChips(stateChips, "state", stateFilter);
  listLimit = 100;
  loadSessions();
  syncURL();
});

moreBtn.addEventListener("click", () => {
  listLimit = Math.min(MAX_LIST, listLimit + 100);
  loadSessions();
});

inFilter.addEventListener("input", () => {
  clearTimeout(inFilter._t);
  inFilter._t = setTimeout(() => buildLedger({ keepScroll: true }), 150);
});

turnPrev.addEventListener("click", () => gotoPrompt(-1));
turnNext.addEventListener("click", () => gotoPrompt(1));
linkBtn.addEventListener("click", linkExperiment);
menuBtn.addEventListener("click", openMenu);
refreshBtn.addEventListener("click", () => { loadSessions(); if (current) refreshCurrent(); });

inspectorCopy.addEventListener("click", () => copyText(inspectorText.textContent, "Copied event"));
inspectorClose.addEventListener("click", closeInspector);

let scrollQueued = false;
ledgerWrap.addEventListener("scroll", () => {
  pinned = atBottom();
  if (scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(() => { scrollQueued = false; updateTurnNav(); });
}, { passive: true });

// Keyboard: / searches, j/k (or arrows) move, Enter opens the inspector,
// Esc closes it, g/G jump to the ends, [ and ] step between prompts.
document.addEventListener("keydown", e => {
  const target = e.target;
  const typing = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA"
    || target.tagName === "SELECT" || target.isContentEditable);
  if (e.key === "Escape") {
    if (target === searchEl && searchEl.value) {
      searchEl.value = "";
      loadSessions();
      syncURL();
      return;
    }
    if (inspector.classList.contains("open")) { closeInspector(); return; }
    if (typing) target.blur();
    return;
  }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "/") {
    e.preventDefault();
    searchEl.focus();
    searchEl.select();
  } else if (e.key === "j" || e.key === "ArrowDown") {
    e.preventDefault();
    moveSelection(1);
  } else if (e.key === "k" || e.key === "ArrowUp") {
    e.preventDefault();
    moveSelection(-1);
  } else if (e.key === "Enter") {
    if (selectedSeq != null) {
      const event = eventsBySeq.get(selectedSeq);
      if (event) inspect(event);
    }
  } else if (e.key === "g") {
    e.preventDefault();
    moveSelectionTo(0);
  } else if (e.key === "G") {
    e.preventDefault();
    moveSelectionTo(-1);
  } else if (e.key === "[") {
    e.preventDefault();
    gotoPrompt(-1);
  } else if (e.key === "]") {
    e.preventDefault();
    gotoPrompt(1);
  }
});

function moveSelection(delta) {
  const rows = [...ledger.querySelectorAll("tbody tr")];
  if (!rows.length) return;
  const at = rows.findIndex(r => Number(r.dataset.seq) === selectedSeq);
  const index = at < 0 ? (delta > 0 ? 0 : rows.length - 1)
    : Math.min(rows.length - 1, Math.max(0, at + delta));
  const tr = rows[index];
  scrollRowIntoView(tr);
  selectEvent(Number(tr.dataset.seq), { open: inspector.classList.contains("open") });
  pinned = atBottom();
}

function moveSelectionTo(index) {
  const rows = [...ledger.querySelectorAll("tbody tr")];
  if (!rows.length) return;
  const tr = index === 0 ? rows[0] : rows[rows.length - 1];
  scrollRowIntoView(tr);
  selectEvent(Number(tr.dataset.seq));
  pinned = atBottom();
}

// ── Boot ─────────────────────────────────────────────────────────────────────
searchEl.value = params.get("q") || "";
if (!["all", "active", "closed", "imported"].includes(stateFilter)) stateFilter = "all";
if (!AGENT_LABELS[agentFilter]) agentFilter = "";
updateChips(agentChips, "agent", agentFilter);
updateChips(stateChips, "state", stateFilter);

loadSessions().then(() => {
  const wanted = params.get("session");
  const eventSeq = Number(params.get("event"));
  if (wanted) openSession(wanted, { eventSeq: Number.isFinite(eventSeq) && eventSeq > 0 ? eventSeq : null });
  else showNoSelection();
});

setInterval(() => { if (!document.hidden) loadSessions(); }, 15000);
