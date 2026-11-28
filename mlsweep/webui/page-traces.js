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
const traceNotes = document.getElementById("trace-notes");
const inFilter = document.getElementById("in-filter");
const dlBtn = document.getElementById("dl-btn");
const menuBtn = document.getElementById("menu-btn");
const refreshBtn = document.getElementById("refresh-btn");
const turnNav = document.getElementById("turn-nav");
const turnPrev = document.getElementById("turn-prev");
const turnNext = document.getElementById("turn-next");
const turnPos = document.getElementById("turn-pos");
const cmdNav = document.getElementById("cmd-nav");
const cmdPrev = document.getElementById("cmd-prev");
const cmdNext = document.getElementById("cmd-next");
const cmdPos = document.getElementById("cmd-pos");
const ledgerBar = document.getElementById("ledger-bar");
const kindChips = document.getElementById("kind-chips");
const ledgerWrap = document.getElementById("ledger-wrap");
const ledger = document.getElementById("ledger");
const noSel = document.getElementById("no-sel");
const inspector = document.getElementById("inspector");
const inspectorKind = document.getElementById("inspector-kind");
const inspectorWhere = document.getElementById("inspector-where");
const inspectorText = document.getElementById("inspector-text");
const inspectorOverview = document.getElementById("inspector-overview");
const inspectorCopy = document.getElementById("inspector-copy");
const inspectorLink = document.getElementById("inspector-link");
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
let resultByToolId = new Map();
let toolSeqById = new Map();  // tool_id -> seq of the call (its result folds into that row)
let turnOfSeq = new Map();    // seq -> turn number
let turnStarts = [];          // seq of each user prompt, in order
let cmdSeqs = [];             // seq of each tool call that runs mlsweep, in order
let cmdSet = new Set();       // the same seqs, to test one
let kindFilter = params.get("kind") || "";  // "" | user | assistant | thinking | tools | mlsweep
let linksExpanded = false;    // the header lists every linked experiment, not the first few
let selectedSeq = null;
let pollTimer = null;
let openToken = 0;            // bumped by each openSession: a late answer for another trace is dropped
let desiredSession = null;    // the trace the reader asked for, even while its load is in flight
let listSeq = 0;              // bumped by each loadSessions: an older list answer is dropped
let pinned = false;           // the user is reading the tail: follow new events
let notesCache = new Map();   // session id -> how many notes cite it
let notesEnabled = true;      // the manager answered 404: no notes served

const AGENT_LABELS = { pi: "pi", claude_code: "claude code" };
const MAX_LIST = 1000;
const KINDS = ["user", "assistant", "thinking", "tools", "mlsweep"];
const LINKS_SHOWN = 3;        // experiment chips in the header before "+N more"

// A tool call that runs the mlsweep CLI: the word in command position (start
// of the command, or after ; && || | ( $( or a newline), past env assignments
// and wrappers like env/timeout/uv run, or `python -m mlsweep…`.  A path that
// merely ends in /mlsweep (cd ~/git/mlsweep) is not one.
const MLSWEEP_RE = /(?:^|[;&|(`\n]\s*|\$\(\s*)(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:(?:sudo|env|nohup|uv\s+run|timeout\s+\S+)\s+)*(?:[^\s=]*\/)?(?:mlsweep(?=\s|$)|python[0-9.]*\s+-m\s+mlsweep\b)/;
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

function agentLabel(s) {
  return AGENT_LABELS[s.agent] || s.agent || "";
}

// The page's state, in the address bar: a reload or a copied link keeps the
// trace, the selected event, and the filters.
function syncURL() {
  const q = new URLSearchParams();
  if (current) q.set("session", current.session_id);
  if (selectedSeq != null) q.set("event", String(selectedSeq));
  if (current && kindFilter) q.set("kind", kindFilter);
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
  title.textContent = traceName(s);
  title.title = s.session_id;
  top.append(dot, title);

  const bottom = document.createElement("div");
  bottom.className = "session-item-bottom";
  const cwd = document.createElement("span");
  cwd.className = "session-cwd";
  cwd.textContent = s.cwd || "";
  const agent = document.createElement("span");
  agent.className = "session-agent";
  agent.textContent = agentLabel(s) || "?";
  const when = document.createElement("span");
  when.textContent = timeAgo(s.last_seen_at);
  when.title = new Date((s.last_seen_at || 0) * 1000).toLocaleString();
  bottom.append(cwd, agent, when);
  if (s.event_count) {
    const n = document.createElement("span");
    n.textContent = `${s.event_count} ev`;
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
  const seq = ++listSeq;
  const q = searchEl.value.trim();
  const isSearch = !!q;
  const atTop = sessionList.scrollTop;
  try {
    let r;
    if (isSearch) {
      r = await apiFetch(`/api/capture/search?q=${encodeURIComponent(q)}&limit=50`);
    } else {
      const query = new URLSearchParams({ limit: String(listLimit), state: stateFilter });
      if (agentFilter) query.set("agent", agentFilter);
      if (experimentFilter) query.set("experiment", experimentFilter);
      r = await apiFetch(`/api/capture/sessions?${query}`);
    }
    // Typing, a filter or a poll started another list request; this answer is stale.
    if (seq !== listSeq) return;
    let next;
    if (r.status === 404) {
      captureOff = true;
      next = [];
    } else {
      if (!r.ok) return;
      captureOff = false;
      next = await r.json();
      if (seq !== listSeq) return;
      if (isSearch) next = next.filter(matchesFilters);
    }
    searching = isSearch;
    sessions = next;
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
  toolSeqById = new Map();
  turnOfSeq = new Map();
  turnStarts = [];
  cmdSeqs = [];
  cmdSet = new Set();
  for (const e of events) indexOne(e);
}

function indexOne(e) {
  if (e.kind === "user") turnStarts.push(e.seq);
  turnOfSeq.set(e.seq, turnStarts.length);
  eventsBySeq.set(e.seq, e);
  if (e.kind === "tool_result" && e.tool_id) resultByToolId.set(e.tool_id, e);
  if (e.kind === "tool" && e.tool_id) toolSeqById.set(e.tool_id, e.seq);
  if (e.kind === "tool" && MLSWEEP_RE.test(e.text || "")) {
    cmdSeqs.push(e.seq);
    cmdSet.add(e.seq);
  }
}

// Whether a (indexed) event is a tool call that runs mlsweep.
function isMlsweep(e) {
  return cmdSet.has(e.seq);
}

// Forget the open trace's events and rows (the header is the caller's).
function resetTrace() {
  events = [];
  indexEvents();
  rowsBySeq.clear();
  selectedSeq = null;
  inspector.classList.remove("open");
  inFilter.value = "";
  ledger.querySelector("tbody").replaceChildren();
}

// The mlsweep invocations in a command, each from the word `mlsweep` to the
// end of its pipeline stage: what an mlsweep row shows instead of the whole
// script around it.
function mlsweepCalls(text) {
  const out = [];
  const re = new RegExp(MLSWEEP_RE.source, "g");
  for (let m; (m = re.exec(text)) && out.length < 4;) {
    const start = m.index + m[0].length - "mlsweep".length;
    const rest = text.slice(start);
    const stop = rest.search(/\s*(?:;|&&|\|\||\||\n|\)|`)/);
    // Redirections (2>&1, >/dev/null) are noise next to the subcommand.
    const call = (stop < 0 ? rest : rest.slice(0, stop)).replace(/\s+\d*(?:>>?|<)&?\s*\S+/g, "").trim();
    out.push(call.length > 160 ? call.slice(0, 159) + "…" : call);
    if (re.lastIndex === m.index) re.lastIndex += 1;
  }
  return out;
}

function matchesKind(e) {
  if (!kindFilter) return true;
  if (kindFilter === "mlsweep") return isMlsweep(e);
  if (kindFilter === "tools") return e.kind === "tool" || e.kind === "tool_result";
  return e.kind === kindFilter;
}

function shownByFilters(e, needle) {
  return !hiddenRow(e) && matchesKind(e) && (!needle || matchesEventFilter(e, needle));
}

// The kind buttons, each with how many rows it would show.
function updateKindCounts() {
  const counts = { "": 0 };
  for (const k of KINDS) counts[k] = 0;
  for (const e of events) {
    if (hiddenRow(e)) continue;
    counts[""] += 1;
    if (isMlsweep(e)) counts.mlsweep += 1;
    const k = e.kind === "tool" || e.kind === "tool_result" ? "tools" : e.kind;
    if (k in counts && k !== "mlsweep") counts[k] += 1;
  }
  kindChips.querySelectorAll("button").forEach(b => {
    const k = b.dataset.kind || "";
    const on = k === kindFilter;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", String(on));
    let n = b.querySelector(".n");
    if (!n) { n = document.createElement("span"); n.className = "n"; b.append(n); }
    n.textContent = String(counts[k] ?? 0);
    b.disabled = !on && !counts[k];
  });
}

// Rows the ledger leaves out: thinking with no text (Claude Code records
// redacted thinking as an empty block).
function hiddenRow(e) {
  return foldedInto(e) != null || (e.kind === "thinking" && !(e.text || "").trim());
}

// A result whose call is in the trace shows on the call's row, not its own.
function foldedInto(e) {
  if (e.kind !== "tool_result" || !e.tool_id) return null;
  return toolSeqById.get(e.tool_id) ?? null;
}

function resultStatus(result) {
  const ok = result.ok !== false;
  return { ok, text: (ok ? "ok" : "error") + (result.duration_ms != null ? ` · ${fmtMs(result.duration_ms)}` : "") };
}

function fmtMs(ms) {
  return ms < 1000 ? `${ms} ms` : ms < 60000 ? `${(ms / 1000).toFixed(1)} s` : fmtDuration(ms / 1000);
}

function matchesEventFilter(e, needle) {
  return `${e.text || ""} ${e.name || ""} ${e.kind || ""}`.toLowerCase().includes(needle);
}

function shortText(e) {
  if (e.kind === "tool_result") {
    const status = e.ok === false ? "error" : "ok";
    return `<span class="event-${e.ok === false ? "err" : "ok"}">${escHtml(status)}</span>`
      + (e.duration_ms != null ? `<span class="event-arrow">·</span>${escHtml(fmtMs(e.duration_ms))}` : "");
  }
  if (isMlsweep(e)) {
    const calls = mlsweepCalls(e.text || "");
    if (calls.length) return calls.map(c => `<span class="cmd">${escHtml(c)}</span>`).join('<span class="event-arrow">·</span>');
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
  tr.className = `event ev-${e.kind}` + (e.kind === "user" ? " turn-start" : "");
  tr.dataset.seq = e.seq;
  tr.dataset.turn = String(turn);
  if (e.seq === selectedSeq) tr.classList.add("selected");

  const time = document.createElement("td");
  time.className = "time-col";
  time.textContent = fmtClock(e.ts);
  time.title = e.ts ? new Date(e.ts * 1000).toLocaleString() : "";

  const tag = document.createElement("td");
  tag.className = "tag-col";
  const pill = document.createElement("span");
  const cmd = isMlsweep(e);
  if (cmd) tr.classList.add("is-mlsweep");
  pill.className = `kind-tag kind-${cmd ? "mlsweep" : e.kind}`;
  pill.textContent = cmd ? "mlsweep" : kindLabel(e);
  pill.title = cmd ? `${kindLabel(e)}: runs mlsweep` : kindLabel(e);
  tag.append(pill);
  if (e.kind === "user") {
    const label = document.createElement("span");
    label.className = "turn-label";
    label.textContent = `#${turn}`;
    tag.append(label);
  }

  const content = document.createElement("td");
  content.className = "content-col";
  const line = document.createElement("div");
  line.className = "content-line";
  const text = document.createElement("span");
  text.className = "event-text";
  text.innerHTML = shortText(e);
  if (e.kind === "tool") text.title = e.text || "";
  line.append(text);
  content.append(line);

  tr.append(time, tag, content);
  tr.addEventListener("click", () => selectEvent(e.seq, { open: true }));
  return tr;
}

// A tool call shows its result inline (status and how long); the result gets
// no row of its own unless its call is missing from the trace.
function decorateToolRow(toolId) {
  const result = resultByToolId.get(toolId);
  const line = rowsBySeq.get(toolSeqById.get(toolId))?.querySelector(".content-line");
  if (!result || !line) return;
  let status = line.querySelector(".event-status");
  if (!status) {
    status = document.createElement("span");
    status.className = "event-status";
    line.append(status);
  }
  const ok = result.ok !== false;
  const mark = document.createElement("span");
  mark.className = "mark";
  mark.textContent = ok ? "✓" : "✗";
  status.replaceChildren(mark, result.duration_ms != null ? fmtMs(result.duration_ms) : (ok ? "ok" : "error"));
  status.title = ok ? "Succeeded" : "Failed";
  status.classList.toggle("ok", ok);
  status.classList.toggle("err", !ok);
}

function addRow(e, tbody) {
  const tr = makeRow(e);
  tbody.append(tr);
  rowsBySeq.set(e.seq, tr);
}

function buildLedger({ keepScroll = false } = {}) {
  const anchor = keepScroll ? firstVisibleSeq() : null;
  const tbody = ledger.querySelector("tbody");
  tbody.replaceChildren();
  rowsBySeq.clear();
  const needle = inFilter.value.trim().toLowerCase();
  for (const e of events) {
    if (shownByFilters(e, needle)) addRow(e, tbody);
  }
  for (const toolId of resultByToolId.keys()) decorateToolRow(toolId);
  updateKindCounts();
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
  const last = events.length ? events[events.length - 1].seq : 0;
  for (const e of fresh) {
    if (e.seq <= last || eventsBySeq.has(e.seq)) continue;  // already shown
    events.push(e);
    indexOne(e);
    if (shownByFilters(e, needle)) addRow(e, tbody);
    if (e.tool_id) decorateToolRow(e.tool_id);  // a result lands on its call's row
  }
  updateKindCounts();
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

// Rows are in document order, so the first one in view is a binary search.
function firstVisibleSeq() {
  const rows = ledger.tBodies[0].rows;
  const top = ledgerWrap.getBoundingClientRect().top;
  let lo = 0;
  let hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].getBoundingClientRect().bottom > top) hi = mid;
    else lo = mid + 1;
  }
  return lo < rows.length ? Number(rows[lo].dataset.seq) : null;
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

// Where the reader is: the selected row while it is on screen, else the first
// row in view.
function readingSeq() {
  if (selectedSeq != null && rowsBySeq.has(selectedSeq)) {
    const box = rowsBySeq.get(selectedSeq).getBoundingClientRect();
    const wrap = ledgerWrap.getBoundingClientRect();
    if (box.bottom > wrap.top && box.top < wrap.bottom) return selectedSeq;
  }
  return firstVisibleSeq();
}

function currentTurn(seq = readingSeq()) {
  if (!turnStarts.length) return 0;
  return seq == null ? 1 : Math.max(1, turnOfSeq.get(seq) || 0);
}

function updateTurnNav() {
  const at = current ? readingSeq() : null;
  const total = turnStarts.length;
  const show = Boolean(current) && total > 0;
  turnNav.style.display = show ? "" : "none";
  if (show) {
    const turn = Math.min(total, Math.max(1, currentTurn(at)));
    turnPos.textContent = `${turn} / ${total}`;
    turnPrev.disabled = turn <= 1;
    turnNext.disabled = turn >= total;
  }
  updateCmdNav(at);
}

// The mlsweep stepper: which command the reader is at or past, of how many.
function updateCmdNav(at) {
  const total = cmdSeqs.length;
  const show = Boolean(current) && total > 0;
  cmdNav.style.display = show ? "" : "none";
  if (!show) return;
  const passed = at == null ? 0 : cmdSeqs.filter(seq => seq <= at).length;
  const on = at != null && cmdSet.has(at);
  cmdPos.textContent = `${on ? passed : passed ? `${passed}+` : "–"} / ${total}`;
  cmdPrev.disabled = at == null || !cmdSeqs.some(seq => seq < at);
  cmdNext.disabled = at != null && !cmdSeqs.some(seq => seq > at);
}

function gotoCommand(delta) {
  if (!current || !cmdSeqs.length) return;
  const at = readingSeq();
  const target = at == null ? cmdSeqs[0]
    : delta > 0 ? cmdSeqs.find(seq => seq > at)
    : [...cmdSeqs].reverse().find(seq => seq < at);
  if (target != null) jumpToSeq(target);
}

// The first row the filters show in turn *turn*: its prompt, or (when the
// kind filter hides prompts) the turn's first shown event.
function firstRowOfTurn(turn) {
  const start = turnStarts[turn - 1];
  if (rowsBySeq.has(start)) return start;
  for (const e of events) {
    if (e.seq > start && turnOfSeq.get(e.seq) === turn && rowsBySeq.has(e.seq)) return e.seq;
    if (turnOfSeq.get(e.seq) > turn) break;
  }
  return null;
}

function gotoPrompt(delta) {
  if (!current || !turnStarts.length) return;
  for (let t = currentTurn() + delta; t >= 1 && t <= turnStarts.length; t += delta) {
    const seq = firstRowOfTurn(t);
    if (seq != null) { jumpToSeq(seq); return; }
  }
}

// ── Selection and inspector ──────────────────────────────────────────────────

function selectEvent(seq, { open = false } = {}) {
  rowsBySeq.get(selectedSeq)?.classList.remove("selected");
  selectedSeq = seq;
  rowsBySeq.get(seq)?.classList.add("selected");
  const event = eventsBySeq.get(seq);
  if (event && (open || inspector.classList.contains("open"))) inspect(event);
  syncURL();
  updateTurnNav();
}

function jumpToSeq(seq) {
  const folded = eventsBySeq.has(seq) ? foldedInto(eventsBySeq.get(seq)) : null;
  if (folded != null) seq = folded;
  if (!rowsBySeq.has(seq) && (inFilter.value.trim() || kindFilter)) {
    inFilter.value = "";
    kindFilter = "";
    buildLedger({ keepScroll: true });
    syncURL();
  }
  // An event the ledger never shows (empty thinking) opens the next row that is shown.
  if (!rowsBySeq.has(seq)) {
    const next = events.find(e => e.seq > seq && rowsBySeq.has(e.seq));
    if (next) seq = next.seq;
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
  const status = e.kind === "tool_result" ? resultStatus(e) : null;
  inspectorText.className = "event-full"
    + (e.kind === "tool" ? " tool" : e.kind === "thinking" ? " thinking" : "")
    + (status ? ` result ${status.ok ? "ok" : "err"}` : "");
  inspectorText.textContent = status ? status.text : (e.text || "");
  const rows = [];
  const put = (key, value) => { if (value != null && value !== "") rows.push([key, String(value)]); };
  put("seq", e.seq);
  put("model", e.model);
  if (e.tokens) put("tokens", `${e.tokens.input ?? "?"} in · ${e.tokens.output ?? "?"} out`);
  put("tool", e.name);
  put("tool id", e.tool_id);
  if (e.kind === "tool_result") {
    put("result", e.ok === false ? "error" : "ok");
    put("duration", e.duration_ms != null ? fmtMs(e.duration_ms) : null);
  }
  const result = e.kind === "tool" && e.tool_id ? resultByToolId.get(e.tool_id) : null;
  if (e.kind === "tool") {
    rows.push(["result", result ? resultStatus(result).text : "not recorded", null]);
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
  rowsBySeq.get(selectedSeq)?.classList.remove("selected");
  selectedSeq = null;
  syncURL();
  updateTurnNav();
}

// ── Header ───────────────────────────────────────────────────────────────────

function renderLinks(s) {
  traceLinks.replaceChildren();
  const all = s.experiments || [];
  const shown = linksExpanded ? all : all.slice(0, LINKS_SHOWN);
  traceLinks.classList.toggle("expanded", linksExpanded && all.length > LINKS_SHOWN);
  for (const eid of shown) {
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
  if (all.length > LINKS_SHOWN) {
    const more = document.createElement("button");
    more.type = "button";
    more.className = "more-links";
    more.textContent = linksExpanded ? "show fewer" : `+${all.length - LINKS_SHOWN} more`;
    more.title = linksExpanded ? "" : `${all.length} experiments in all`;
    more.addEventListener("click", () => { linksExpanded = !linksExpanded; renderLinks(s); });
    traceLinks.append(more);
  }
}

function renderHeader(s, { notes = true } = {}) {
  const started = s.started_at ? new Date(s.started_at * 1000).toLocaleString() : "";
  const duration = s.started_at && s.last_seen_at > s.started_at
    ? ` · ${fmtDuration(s.last_seen_at - s.started_at)}` : "";
  traceTitle.textContent = traceName(s) || "Trace";
  traceTitle.title = s.session_id;
  document.title = `mlsweep — ${traceName(s) || "Trace"}`;
  traceAgent.style.display = "";
  traceAgent.textContent = agentLabel(s);
  traceState.style.display = "";
  traceState.className = `trace-state ${s.state || ""}`.trim();
  traceState.textContent = s.state || "";
  const parts = [
    [s.model, "mono"], [s.cwd, "mono"],
    [`${events.length || s.event_count || 0} events${duration}`], [started],
  ].filter(([text]) => text);
  // A continued session's transcript repeats the earlier one's events; they stay there.
  if (s.continues) parts.push([`continues ${s.continues.slice(0, 8)}`, "", s.continues]);
  for (const later of s.continued_by || []) parts.push([`continued in ${later.slice(0, 8)}`, "", later]);
  traceMeta.replaceChildren(...parts.flatMap(([text, cls, session], i) => {
    const span = document.createElement(session ? "a" : "span");
    if (cls) span.className = cls;
    span.textContent = text;
    if (session) {
      span.href = pageHref("traces.html", { session });
      span.title = session;
    }
    if (!i) return [span];
    const sep = document.createElement("span");
    sep.className = "sep";
    sep.textContent = "·";
    return [sep, span];
  }));
  renderLinks(s);
  traceLinks.style.display = (s.experiments || []).length ? "" : "none";
  dlBtn.style.display = "";
  dlBtn.href = authURL(`/api/capture/sessions/${encodeURIComponent(s.session_id)}/transcript`);
  menuBtn.style.display = "";
  inFilter.style.display = "";
  ledgerBar.style.display = "";
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
    { label: "Copy note reference", hint: kindFilter ? `mlsweep:trace/…?kind=${kindFilter}` : "mlsweep:trace/…",
      run: () => copyText(traceRef(kindFilter, null), "Copied trace reference") },
    { label: "Copy trace id", run: () => copyText(sid, "Copied trace id") },
    { label: "Download transcript",
      run: () => window.open(authURL(`/api/capture/sessions/${encodeURIComponent(sid)}/transcript`), "_blank") },
    { label: "Download as ATIF", hint: "for other agent tools",
      run: () => window.open(authURL(`/api/capture/sessions/${encodeURIComponent(sid)}/atif?download=1`), "_blank") },
    "-",
    { label: "Delete trace", danger: true, run: deleteCurrent },
  ]);
}

// ── Note links ───────────────────────────────────────────────────────────────

// `mlsweep:trace/<id>`, filtered to *kind* and opened at *seq* when given.
function traceRef(kind, seq, sid = current.session_id) {
  return `mlsweep:trace/${sid}` + (kind ? `?kind=${kind}` : "") + (seq != null ? `#E${seq}` : "");
}

// Right-click a trace in the list: what the ⋯ menu offers for the open one.
sessionList.addEventListener("contextmenu", ev => {
  const item = ev.target.closest(".session-item");
  if (!item || mlNativeMenuWanted(ev, item)) return;
  const sid = item.dataset.session;
  const isOpen = current && current.session_id === sid;
  const s = sessions.find(x => x.session_id === sid);
  mlContextMenu(ev, [
    { title: s ? traceName(s) : sid },
    isOpen ? null : { label: "Open", run: () => openSession(sid) },
    { label: "Copy note reference", hint: "mlsweep:trace/…", run: () => copyText(traceRef("", null, sid), "Copied trace reference") },
    { label: "Copy trace id", run: () => copyText(sid, "Copied trace id") },
    { label: "Download transcript",
      run: () => window.open(authURL(`/api/capture/sessions/${encodeURIComponent(sid)}/transcript`), "_blank") },
    { label: "Download as ATIF", hint: "for other agent tools",
      run: () => window.open(authURL(`/api/capture/sessions/${encodeURIComponent(sid)}/atif?download=1`), "_blank") },
    "-",
    { label: "Delete trace", danger: true, run: () => deleteTrace(sid) },
  ]);
});

// Right-click an event: inspect it or copy a way back to it.
ledger.addEventListener("contextmenu", ev => {
  const tr = ev.target.closest("tr.event");
  if (!tr || !current || mlNativeMenuWanted(ev, tr)) return;
  const e = eventsBySeq.get(Number(tr.dataset.seq));
  if (!e) return;
  selectEvent(e.seq);
  mlContextMenu(ev, [
    { label: "Inspect", hint: "Enter", run: () => selectEvent(e.seq, { open: true }) },
    { label: "Copy text", run: () => copyText(e.text || "", "Copied event") },
    { label: "Copy note link", hint: `#E${e.seq}`, run: () => copyText(eventNoteLink(e), "Copied note link") },
    { label: "Copy note reference", run: () => copyText(traceRef("", e.seq), "Copied trace reference") },
  ]);
});

// A markdown link to one event, its text a short description a note can keep.
function eventNoteLink(e) {
  const clip = (text, n) => {
    const one = String(text || "").replace(/\s+/g, " ").trim();
    return one.length > n ? one.slice(0, n - 1) + "…" : one;
  };
  let label;
  if (isMlsweep(e)) {
    const calls = mlsweepCalls(e.text || "");
    label = clip(calls.find(c => /^mlsweep\s+run\b/.test(c)) || calls[0] || e.text, 60);
  }
  else if (e.kind === "user") label = `“${clip(e.text, 50)}”`;
  else if (e.kind === "tool") label = `${e.name || "tool"} call ${e.seq}`;
  else label = `${kindLabel(e)} ${e.seq}`;
  label = label.replace(/[[\]]/g, "") || `event ${e.seq}`;
  return `[${label}](${traceRef("", e.seq)})`;
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
  desiredSession = id;
  const token = ++openToken;
  const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(id)}`);
  if (token !== openToken) return;  // another trace was opened meanwhile
  if (r.status === 404) {
    if (desiredSession === id) desiredSession = null;
    await loadSessions();
    if (token === openToken) showNoSelection("That trace is gone.", "Pick another from the left.");
    return;
  }
  if (!r.ok) return;
  const s = await r.json();
  if (token !== openToken) return;  // the body arrived after another trace was opened
  if (!current || current.session_id !== s.session_id) linksExpanded = false;
  current = s;
  clearInterval(pollTimer);
  // Nothing from the previous trace survives the switch, even if the load fails.
  resetTrace();
  showNoSelection("Loading trace…", "");
  markSelectedSession();
  await loadEvents(s, 0);
  if (token !== openToken) return;
  renderHeader(s);
  updateTurnNav();
  markSelectedSession();
  syncURL();
  if (eventSeq != null) jumpToSeq(eventSeq);
  startPoll();
}

async function refreshCurrent() {
  // If the reader picked another trace whose load is still in flight, refresh
  // that one rather than the trace still on screen.
  const id = desiredSession || (current && current.session_id);
  if (!id) return;
  const keep = selectedSeq;
  await openSession(id);
  if (keep != null && current && current.session_id === id) jumpToSeq(keep);
}

const EVENTS_PAGE = 1000;

async function loadEvents(s, fromSeq) {
  // Pages until a short one says the session is caught up.  What arrived
  // before a failed page is still shown.
  const token = openToken;
  const loaded = [];
  let next = fromSeq;
  for (;;) {
    const r = await apiFetch(
      `/api/capture/sessions/${encodeURIComponent(s.session_id)}/events?from_seq=${next}&limit=${EVENTS_PAGE}`);
    if (token !== openToken) return;  // the reader moved on to another trace
    if (!r.ok) break;
    const body = await r.json();
    if (token !== openToken) return;
    loaded.push(...body.events);
    next = body.next_from_seq;
    if (body.events.length < EVENTS_PAGE) break;
  }
  if (token !== openToken) return;
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
  let busy = false;
  const timer = pollTimer = setInterval(async () => {
    if (!current || document.hidden || busy) return;
    const token = openToken;
    const sid = current.session_id;
    busy = true;
    try {
      const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(sid)}`);
      if (token !== openToken) return;  // another trace is open now
      if (r.status === 404) { clearInterval(timer); return; }
      if (!r.ok) return;
      const s = await r.json();
      if (token !== openToken || !current || current.session_id !== sid) return;
      const grew = (s.event_count || 0) > events.length;
      const stateChanged = s.state !== current.state;
      current = s;
      renderHeader(s, { notes: false });
      if (grew) {
        await loadEvents(s, events.length ? events[events.length - 1].seq + 1 : 0);
      }
      if (stateChanged) updateTurnNav();
      if (s.state !== "active") clearInterval(timer);
    } catch { /* offline: try again next tick */ } finally {
      busy = false;
    }
  }, POLL_MS);
}

async function deleteCurrent() {
  if (current) await deleteTrace(current.session_id);
}

// Delete a trace; the open one also clears the view.
async function deleteTrace(id) {
  const yes = await mlConfirm(`Delete trace ${id}? Its events and experiment links go with it.`);
  if (!yes) return;
  const r = await apiFetch(`/api/capture/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) {
    await mlAlert(`Delete failed (${r.status})`);
    return;
  }
  notesCache.delete(id);
  if (!current || current.session_id !== id) { await loadSessions(); return; }
  clearInterval(pollTimer);
  openToken += 1;  // a load still in flight is for the trace just deleted
  desiredSession = null;
  current = null;
  resetTrace();
  inFilter.style.display = "none";
  ledgerBar.style.display = "none";
  ledger.querySelector("tbody").replaceChildren();
  traceTitle.textContent = "No trace selected";
  traceTitle.title = "";
  document.title = "mlsweep — Traces";
  traceMeta.textContent = "";
  traceAgent.style.display = "none";
  traceState.style.display = "none";
  traceLinks.replaceChildren();
  traceLinks.style.display = "none";
  traceNotes.style.display = "none";
  dlBtn.style.display = "none";
  menuBtn.style.display = "none";
  updateTurnNav();
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
cmdPrev.addEventListener("click", () => gotoCommand(-1));
cmdNext.addEventListener("click", () => gotoCommand(1));
kindChips.addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || b.disabled) return;
  kindFilter = b.dataset.kind || "";
  buildLedger({ keepScroll: true });
  syncURL();
});
turnNext.addEventListener("click", () => gotoPrompt(1));
menuBtn.addEventListener("click", openMenu);
refreshBtn.addEventListener("click", () => { loadSessions(); if (current) refreshCurrent(); });

inspectorCopy.addEventListener("click", () => copyText(inspectorText.textContent, "Copied event"));
inspectorLink.addEventListener("click", () => {
  const e = eventsBySeq.get(selectedSeq);
  if (current && e) copyText(eventNoteLink(e), "Copied note link");
});
inspectorClose.addEventListener("click", closeInspector);

let scrollQueued = false;
ledgerWrap.addEventListener("scroll", () => {
  pinned = atBottom();
  if (scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(() => { scrollQueued = false; updateTurnNav(); });
}, { passive: true });

// Keyboard: / searches, j/k (or arrows) move, Enter opens the inspector,
// Esc closes it, g/G jump to the ends, [ and ] step between prompts, m/M to
// the next/previous mlsweep command.
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
  } else if (e.key === "m" || e.key === "M") {
    e.preventDefault();
    gotoCommand(e.key === "m" ? 1 : -1);
  } else if (e.key === "[") {
    e.preventDefault();
    gotoPrompt(-1);
  } else if (e.key === "]") {
    e.preventDefault();
    gotoPrompt(1);
  }
});

function moveSelection(delta) {
  const rows = ledger.tBodies[0].rows;
  if (!rows.length) return;
  const at = rowsBySeq.get(selectedSeq);
  const tr = !at ? rows[delta > 0 ? 0 : rows.length - 1]
    : (delta > 0 ? at.nextElementSibling : at.previousElementSibling) || at;
  scrollRowIntoView(tr);
  selectEvent(Number(tr.dataset.seq), { open: inspector.classList.contains("open") });
  pinned = atBottom();
}

function moveSelectionTo(index) {
  const rows = ledger.tBodies[0].rows;
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
if (!KINDS.includes(kindFilter)) kindFilter = "";
updateChips(agentChips, "agent", agentFilter);
updateChips(stateChips, "state", stateFilter);

loadSessions().then(() => {
  const wanted = params.get("session");
  const eventSeq = Number(params.get("event"));
  if (wanted) openSession(wanted, { eventSeq: Number.isFinite(eventSeq) && eventSeq > 0 ? eventSeq : null });
  else showNoSelection();
});

setInterval(() => { if (!document.hidden) loadSessions(); }, 15000);
