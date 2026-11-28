// Where a run or an experiment leads: its View menu, shared by every page.
// The Experiments tab's View ▾ and a right-click on a run or an experiment
// anywhere list the same pages in the same order, with the same counts.
// Needs chrome.js (pageHref, apiFetch, copyText) and dialog.js (mlContextMenu).
const MLViews = (function () {
  // notes are optional: with no notes folder there is no Notes entry at all.
  // capture is optional too: a manager started with --no-capture has no traces.
  const S = {
    notesEnabled: false,
    noteMentions: {},      // experiment id -> [note paths] (its own or its runs')
    noteRunMentions: {},   // experiment id -> {run id: [note paths]}
    captureEnabled: false,
    traceCounts: null,     // experiment id -> linked trace count (null until loaded)
    exps: new Map(),       // experiment id -> its /api/experiments record
  };
  const listeners = [];
  const changed = () => listeners.forEach(fn => fn());

  // ── the data the counts come from ─────────────────────────────────────────

  let notesSeq = 0;
  async function pollNotes() {
    const mine = ++notesSeq;
    try {
      const r = await apiFetch("/api/notes/mentions");
      if (mine !== notesSeq || !r.ok) return;
      const d = await r.json();
      if (mine !== notesSeq) return;
      const next = d.mentions || {}, nextRuns = d.runs || {}, nextEnabled = d.enabled !== false;
      const differs = nextEnabled !== S.notesEnabled
        || JSON.stringify(next) !== JSON.stringify(S.noteMentions)
        || JSON.stringify(nextRuns) !== JSON.stringify(S.noteRunMentions);
      Object.assign(S, { noteMentions: next, noteRunMentions: nextRuns, notesEnabled: nextEnabled });
      if (differs) changed();
    } catch { /* notes are optional */ }
  }

  // The counts endpoint answers 404 on a manager with capture off; then there
  // is no Traces entry either.
  let tracesSeq = 0;
  async function pollTraces() {
    const mine = ++tracesSeq;
    try {
      const r = await apiFetch("/api/capture/counts");
      if (mine !== tracesSeq) return;
      if (r.status === 404) {
        if (S.captureEnabled) { S.captureEnabled = false; changed(); }
        return;
      }
      if (!r.ok) return;
      const next = (await r.json()).experiments || {};
      if (mine !== tracesSeq) return;
      const differs = !S.captureEnabled || JSON.stringify(next) !== JSON.stringify(S.traceCounts);
      Object.assign(S, { captureEnabled: true, traceCounts: next });
      if (differs) changed();
    } catch { /* capture is optional */ }
  }

  // Experiment records, for their run counts and code size.  A page that lists
  // experiments hands its own over (setExperiments); the rest fetch them.
  let expsAt = 0, expsSeq = 0;
  function setExperiments(list) {
    S.exps = new Map((list || []).map(e => [e.experiment_id, e]));
    expsAt = Date.now();
  }
  async function pollExperiments() {
    const mine = ++expsSeq;
    try {
      const r = await apiFetch("/api/experiments");
      if (mine !== expsSeq || !r.ok) return;
      const list = await r.json();
      if (mine === expsSeq) setExperiments(list);
    } catch { /* the menu just goes without counts */ }
  }

  // Everything loads once, then the mentions and traces every 30s.  A menu
  // opened after the experiments went stale refreshes them for the next one.
  const STALE_MS = 30000;
  function refreshIfStale() {
    if (Date.now() - expsAt > STALE_MS) { expsAt = Date.now(); pollExperiments(); }
  }
  function start() {
    pollNotes(); pollTraces(); refreshIfStale();
    setInterval(() => { if (!document.hidden) { pollNotes(); pollTraces(); } }, STALE_MS);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();

  // ── the items ─────────────────────────────────────────────────────────────

  // Counted the way each page lists them: results and export are finished
  // runs, logs are every run, artifacts are the runs with artifact files.
  function expCounts(expId) {
    const exp = S.exps.get(expId) || {};
    const c = exp.job_counts || {};
    return {
      total: c.total || exp.expected_jobs || 0,
      finished: (c.done || 0) + (c.pruned || 0) + (c.failed || 0) + (c.xfailed || 0),
      withArtifacts: c.with_artifacts || 0,
    };
  }

  function noteCount(expId, runId) {
    if (runId) return ((S.noteRunMentions[expId] || {})[runId] || []).length;
    return (S.noteMentions[expId] || []).length;
  }

  function notesItem(expId, runId) {
    if (!S.notesEnabled) return null;
    const what = runId || expId;
    const n = noteCount(expId, runId);
    const item = { page: "notes", label: "Notes", icon: "notes", count: n };
    if (!n) return { ...item, disabled: true, title: `No note mentions ${what} yet` };
    return { ...item, href: pageHref("notes.html", { mentions: what }),
             title: `${n} note${n === 1 ? "" : "s"} mention ${what}` };
  }

  function tracesItem(expId, runId) {
    if (!S.captureEnabled) return null;
    const item = { page: "traces", label: "Traces", icon: "traces", href: pageHref("traces.html", { experiment: expId }) };
    // A run has no traces of its own: it shows its experiment's.
    if (runId || !S.traceCounts) return { ...item, title: `The coding-agent sessions around ${runId || expId}` };
    const n = S.traceCounts[expId] ?? 0;
    if (!n) return { ...item, href: null, disabled: true, count: n, title: `No trace submitted ${expId} yet` };
    return { ...item, count: n, title: `${n} trace${n === 1 ? "" : "s"} submitted ${expId}` };
  }

  // codeFiles: how many files the code snapshot has (null: no snapshot, or not known).
  function codeItem(href, codeFiles) {
    return { page: "code", label: "Code", icon: "code", href, count: codeFiles ?? null,
             title: codeFiles != null ? `${codeFiles} file${codeFiles === 1 ? "" : "s"} of code` : null };
  }

  const leaveOut = skip => it => it && it.page !== skip;

  // A run's View menu.  skip: the page this one is (it leaves itself out).
  function runItems(expId, runId, { codeFiles = null, skip = "" } = {}) {
    refreshIfStale();
    const q = { experiment: expId, run: runId };
    return [
      { page: "results", label: "Results", icon: "results", href: pageHref("results.html", { experiment: expId }) },
      { page: "files", label: "Artifacts", icon: "artifacts", href: pageHref("files.html", q) },
      { page: "logs", label: "Logs", icon: "logs", href: pageHref("logs.html", q) },
      codeItem(pageHref("code.html", q), codeFiles),
      notesItem(expId, runId),
      tracesItem(expId, runId),
    ].filter(leaveOut(skip));
  }

  // An experiment's View menu.
  function expItems(expId, { skip = "" } = {}) {
    refreshIfStale();
    const n = expCounts(expId);
    const exp = S.exps.get(expId) || {};
    const q = { experiment: expId };
    return [
      { page: "results", label: "Results", icon: "results", count: n.finished, href: pageHref("results.html", q) },
      { page: "logs", label: "Logs", icon: "logs", count: n.total, href: pageHref("logs.html", q) },
      { page: "files", label: "Artifacts", icon: "artifacts", count: n.withArtifacts, href: pageHref("files.html", q) },
      { ...codeItem(pageHref("code.html", q), exp.code_files),
        title: "The code this experiment ran, with its file tree and where it came from"
          + (exp.code_files != null ? ` (${exp.code_files} file${exp.code_files === 1 ? "" : "s"})` : "") },
      notesItem(expId),
      tracesItem(expId),
      { page: "export", label: "Export", icon: "export", count: n.finished, href: pageHref("export.html", q) },
    ].filter(leaveOut(skip));
  }

  return {
    state: S,
    onChange(fn) { listeners.push(fn); },
    setExperiments, noteCount, runItems, expItems,
  };
})();

// The same View items for the right-click menus pages build themselves.
function mlRunPageItems(expId, runId, { skip = "", codeFiles = null } = {}) {
  return MLViews.runItems(expId, runId, { skip, codeFiles });
}
function mlExpPageItems(expId, { skip = "" } = {}) {
  return MLViews.expItems(expId, { skip });
}

// The whole right-click menu for a run (*runId* set) or an experiment: a title,
// its View items, then copying its IDs.  *link*: the element is a link, so the
// menu also offers what the browser's menu would.
function mlEntityMenuItems(expId, runId, { title = "", skip = "", codeFiles = null, link = null } = {}) {
  return [
    { title: title || runId || expId },
    link ? { label: "Open link in a new tab", run: () => window.open(link, "_blank", "noopener") } : null,
    link ? "-" : null,
    ...(runId ? mlRunPageItems(expId, runId, { skip, codeFiles }) : mlExpPageItems(expId, { skip })),
    "-",
    runId ? { label: "Copy run ID", run: () => copyText(runId, "Copied the run ID") } : null,
    { label: "Copy experiment ID", run: () => copyText(expId, "Copied the experiment ID") },
  ];
}

// Anything that names a run or an experiment gets its menu by saying which:
//   data-ml-exp="EXP" [data-ml-run="RUN"] [data-ml-title] [data-ml-skip="logs"] [data-ml-code="12"]
// A page's own menu (one that already took the right-click) wins.
document.addEventListener("contextmenu", e => {
  if (e.defaultPrevented || !e.target.closest) return;
  const el = e.target.closest("[data-ml-exp]");
  if (!el || !el.dataset.mlExp || mlNativeMenuWanted(e, el)) return;
  const d = el.dataset;
  const link = el.matches("a[href]") && el.getAttribute("href") !== "#" ? el.href : null;
  mlContextMenu(e, mlEntityMenuItems(d.mlExp, d.mlRun || "", {
    title: d.mlTitle || "", skip: d.mlSkip || "", link,
    codeFiles: d.mlCode != null && d.mlCode !== "" ? Number(d.mlCode) : null,
  }));
});

// Mark *el* as naming an experiment, or one of its runs.
function mlTagEntity(el, expId, runId = "", { title = "", skip = "", codeFiles = null } = {}) {
  if (!el) return el;
  const d = el.dataset;
  if (!expId) { delete d.mlExp; delete d.mlRun; return el; }
  d.mlExp = expId;
  for (const [k, v] of [["mlRun", runId], ["mlTitle", title], ["mlSkip", skip],
                        ["mlCode", codeFiles != null ? String(codeFiles) : ""]]) {
    if (v) d[k] = v; else delete d[k];
  }
  return el;
}

// The same as HTML attributes, for markup built as a string.
function mlEntityAttrs(expId, runId = "", { skip = "", title = "", codeFiles = null } = {}) {
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  return `data-ml-exp="${esc(expId)}"` + (runId ? ` data-ml-run="${esc(runId)}"` : "")
    + (skip ? ` data-ml-skip="${esc(skip)}"` : "") + (title ? ` data-ml-title="${esc(title)}"` : "")
    + (codeFiles != null ? ` data-ml-code="${esc(codeFiles)}"` : "");
}
