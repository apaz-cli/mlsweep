// The Code tab: the code a run used, where it came from, and what changed.
//
// Pick a campaign, an experiment and a run, as on Logs and Artifacts.  With no
// run picked it shows the code last submitted to the experiment; the version
// menu picks any other.  "Compare" diffs it against another version or another
// experiment's code, marking the tree and showing each changed file's diff.
// "Details" shows the provenance mlsweep run recorded (command, host, commit,
// uncommitted changes).  Click a line number to select it, shift-click for a
// range; "Copy" gives a note link, a pinned link or a code block for them.
//
// URL: ?experiment=E&run=R&path=P&lines=40-60&compare=X (or snapshot=ID), or
// ?ref=code/TARGET[/PATH][#Lstart-end] as a note's mlsweep:code/... link writes it.
"use strict";

const MAX_TEXT_BYTES = 4 << 20;

const $ = id => document.getElementById(id);
const short = sid => (sid || "").slice(0, 17);

const S = {
  exps: [],
  expId: null, jobs: [], info: null,
  runId: null,
  snapshot: null, manifest: null,
  snapOnly: false, usage: null,       // opened by snapshot ID: which experiments used it
  path: "", start: null, end: null,
  compare: "",                        // what the code is compared against: a snapshot or experiment ID
  diff: null,                         // /api/code/diff for it
  filter: "", changedOnly: true,
  details: false,
};
const manifests = new Map();          // snapshot -> Promise<manifest> (one fetch per snapshot)
const fileCache = new Map();          // snapshot + path -> text (content never changes)
// Bumped by every navigation (experiment, version, file, popstate).  Async work
// carries the token it began with and drops its result if the view moved on.
let navSeq = 0;
const openDirs = new Set();
const TREE_ORDER = new Intl.Collator(undefined, { sensitivity: "base", numeric: true });   // as the notes tree sorts

MLChrome.collapsibleSidebar(document.getElementById("sidebar"));
MLChrome.resizable(document.getElementById("sidebar"), { name: "sidebar_width_code", min: 180, max: 520 });
MLChrome.resizable(document.getElementById("tree-pane"), { name: "code_tree_width", min: 150, max: 560 });

// ── the URL ──────────────────────────────────────────────────────────────────

function parseRef(raw) {
  let text = String(raw || "").trim().replace(/^mlsweep:/i, "").replace(/^code\//, "");
  if (!text) return null;
  let frag = "";
  const hash = text.indexOf("#");
  if (hash >= 0) { frag = text.slice(hash + 1); text = text.slice(0, hash); }
  const slash = text.indexOf("/");
  return {
    target: slash < 0 ? text : text.slice(0, slash),
    path: slash < 0 ? "" : decodeURIComponent(text.slice(slash + 1)).replace(/^\/+|\/+$/g, ""),
    lines: frag.replace(/L/gi, ""),
  };
}

function parseLines(text) {
  const m = /^(\d+)(?:-(\d+))?$/.exec(String(text || "").trim());
  if (!m) return [null, null];
  const a = Number(m[1]), b = Number(m[2] || m[1]);
  return [Math.min(a, b), Math.max(a, b)];
}

// What the URL asks for.
function urlState() {
  const p = new URLSearchParams(location.search);
  // A missing path (null) means "the default file"; "" is the top folder.
  const out = { exp: p.get("experiment"), run: p.get("run"), snap: p.get("snapshot"), path: p.get("path"),
                lines: p.get("lines") || "", compare: p.get("compare") || "" };
  const ref = parseRef(p.get("ref"));
  if (ref) {
    if (ref.target.startsWith("snap:")) { out.snap = ref.target; out.exp = null; } else out.exp = ref.target;
    out.path = ref.path; out.lines = ref.lines;
  }
  return out;
}

function stateUrl() {
  const u = new URL(location.href);
  for (const k of ["ref", "experiment", "run", "snapshot", "path", "lines", "compare"]) u.searchParams.delete(k);
  if (S.snapOnly) u.searchParams.set("snapshot", S.snapshot);
  else if (S.expId) u.searchParams.set("experiment", S.expId);
  if (S.runId) u.searchParams.set("run", S.runId);
  else if (!S.snapOnly && S.snapshot && S.snapshot !== latestSnapshot()) u.searchParams.set("snapshot", short(S.snapshot));
  if (S.path) u.searchParams.set("path", S.path);
  if (S.start != null) u.searchParams.set("lines", S.end !== S.start ? `${S.start}-${S.end}` : String(S.start));
  if (S.compare) u.searchParams.set("compare", S.compare);
  return u.pathname + u.search;
}

// A new history entry for a new place (file, run, version); line selection only replaces it.
function syncUrl(push) {
  const url = stateUrl();
  if (url === location.pathname + location.search) return;
  if (push) history.pushState(null, "", url); else history.replaceState(null, "", url);
  document.title = `mlsweep — Code${S.path ? " · " + S.path.split("/").pop() : ""}`;
}

// ── data ─────────────────────────────────────────────────────────────────────

async function getJSON(path) {
  const r = await apiFetch(path);
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  if (!r.ok) throw new Error((body && body.error) || `HTTP ${r.status}`);
  return body;
}

function manifestOf(sid) {
  let pending = manifests.get(sid);
  if (!pending) {
    pending = getJSON(`/api/snapshots/${encodeURIComponent(sid)}`).then(body => body.manifest);
    // A failed fetch is retryable; a concurrent caller shares this one request.
    pending.catch(() => manifests.delete(sid));
    manifests.set(sid, pending);
  }
  return pending;
}

const versions = () => (S.info && S.info.versions) || [];
const submissions = () => (S.snapOnly ? (S.usage && S.usage.submissions) : (S.info && S.info.submissions)) || [];

// The snapshot an experiment ID means: the latest submitted, else the last its runs used.
function latestSnapshot() {
  const ids = versions().map(v => v.artifact_id);
  const subs = ((S.info && S.info.submissions) || []).map(s => s.snapshot).filter(s => ids.includes(s));
  return subs.length ? subs[subs.length - 1] : ids[ids.length - 1] || null;
}
function versionNumber(sid) { return versions().findIndex(v => v.artifact_id === sid) + 1; }
function submissionOf(sid) {
  const subs = submissions().filter(s => s.snapshot === sid);
  return subs[subs.length - 1] || null;
}
function fileEntry(path) { return S.manifest.files.find(f => f[0] === path); }
function isDir(path) {
  if (!path) return true;
  const pre = path + "/";
  return S.manifest.files.some(f => f[0].startsWith(pre)) || (S.manifest.dirs || []).includes(path);
}
function changeOf(path) {
  return S.diff ? (S.diff.changes.find(c => c.path === path) || null) : null;
}

// ── sidebar ──────────────────────────────────────────────────────────────────

function renderJobs() {
  const list = $("job-list");
  const n = versions().length;
  $("runs-note").textContent = S.snapOnly ? "" : n > 1 ? `· ${n} code versions` : n === 1 ? "· one code version" : "";
  if (S.snapOnly) {
    const used = (S.usage && S.usage.experiments) || [];
    list.innerHTML = `<div class="job-list-msg">Opened by snapshot ID.${used.length
      ? "<br>Used by " + used.map(e => `<b>${escHtml(e.name || e.experiment_id)}</b>`).join(", ") + "." : ""}
      <br>Choose an experiment to browse its runs.</div>`;
    return;
  }
  if (!S.jobs.length) { list.innerHTML = `<div class="job-list-msg">No runs</div>`; return; }
  const sorted = [...S.jobs].sort((a, b) => new Date(b.submit_time) - new Date(a.submit_time));
  list.innerHTML = sorted.map(j => {
    let combo = {};
    try { combo = typeof j.combo === "string" ? JSON.parse(j.combo || "{}") : (j.combo || {}); } catch { combo = {}; }
    const comboStr = Object.entries(combo).map(([k, v]) => `${k}=${v}`).join(" ");
    const v = n > 1 ? versionNumber(j.artifact_id) : 0;
    const cls = j.run_id === S.runId ? " selected" : (!S.runId && j.artifact_id === S.snapshot ? " same" : "");
    return `<div class="job-item${cls}" data-run-id="${escHtml(j.run_id)}" title="Click to see the code this run used. Right-click for more.">
      <div class="job-item-top">
        <span class="job-run-id" title="${escHtml(j.run_id)}">${runNameHtml(j)}</span>
        ${v ? `<span class="ver-tag" title="${escHtml(short(j.artifact_id))}">v${v}</span>` : ""}
        <span class="badge badge-${escHtml(j.status)}">${escHtml(j.status)}</span>
      </div>
      ${comboStr ? `<div class="job-combo" title="${escHtml(comboStr)}">${escHtml(comboStr)}</div>` : ""}
    </div>`;
  }).join("");
  list.querySelectorAll(".job-item").forEach(item => item.addEventListener("click", () => {
    const job = S.jobs.find(j => j.run_id === item.dataset.runId);
    if (!job) return;
    // A second click on the selected run goes back to the experiment's latest code.
    if (S.runId === job.run_id) showCode(latestSnapshot(), null, { push: true });
    else showCode(job.artifact_id, job, { push: true });
  }));
  list.querySelectorAll(".job-item").forEach(item => item.addEventListener("contextmenu", e => {
    const job = S.jobs.find(j => j.run_id === item.dataset.runId);
    if (job) runMenu(e, job);
  }));
}

// ── header ───────────────────────────────────────────────────────────────────

function metaItem(label, value) {
  const s = el("span");
  if (label) s.append(el("span", "k", label));
  if (typeof value === "string") s.append(el("span", "v", value)); else s.append(value);
  return s;
}

function renderHeader() {
  const job = S.runId ? S.jobs.find(j => j.run_id === S.runId) : null;
  const title = $("hdr-title");
  title.replaceChildren();
  $("hdr-kind").textContent = job ? "Run" : S.snapOnly ? "Snapshot" : "Experiment";
  if (job) title.innerHTML = `<span class="mono">${runNameHtml(job)}</span>`;
  else if (S.snapOnly) title.append(el("span", "mono", short(S.snapshot)));
  else {
    const exp = S.exps.find(e => e.experiment_id === S.expId);
    title.append(document.createTextNode((exp && exp.name) || S.expId));
    title.title = S.expId;
  }
  $("hdr-badge").innerHTML = job ? `<span class="badge badge-${escHtml(job.status)}">${escHtml(job.status)}</span>` : "";

  // Version menu: every version of the experiment's code.
  const vs = $("version-select");
  vs.replaceChildren();
  const latest = latestSnapshot();
  for (const [i, v] of versions().entries()) {
    const o = el("option", null, `v${i + 1}${v.artifact_id === latest ? " (latest)" : ""} · ${short(v.artifact_id).slice(5)}`
      + (v.runs ? ` · ${v.runs} run${v.runs === 1 ? "" : "s"}` : " · no runs"));
    o.value = v.artifact_id;
    o.disabled = v.kind !== "snapshot";
    o.selected = v.artifact_id === S.snapshot;
    vs.append(o);
  }
  vs.style.display = versions().length > 1 ? "" : "none";

  $("compare-btn").classList.toggle("on", !!S.compare);

  // One line of where it came from.
  const meta = $("hdr-meta");
  meta.replaceChildren();
  const files = S.manifest.files;
  const total = files.reduce((n, f) => n + f[2], 0);
  const snapEl = el("span", "v", short(S.snapshot));
  snapEl.title = S.snapshot;
  meta.append(metaItem("code", snapEl), metaItem("", `${files.length} files, ${fmtSize(total)}`));
  if (versions().length > 1 && !S.snapOnly) {
    const n = versionNumber(S.snapshot);
    const v = metaItem("version", `v${n} of ${versions().length}`);
    if (S.snapshot !== latest) { v.querySelector(".v").classList.add("warn"); v.title = "Older than the experiment's latest code"; }
    meta.append(v);
  }
  const sub = submissionOf(S.snapshot);
  const m = (sub && sub.meta) || {};
  const vcs = m.vcs;
  if (vcs && vcs.commit) {
    const g = metaItem("git", `${vcs.commit.slice(0, 8)} ${vcs.branch || "detached"}`);
    g.title = vcs.subject || "";
    meta.append(g);
    if (vcs.dirty) {
      const a = el("a", "warn", `+${(vcs.changed_files || []).length} uncommitted`);
      a.title = "Show the uncommitted changes";
      a.onclick = () => openDetails(true);
      meta.append(metaItem("", a));
    }
  }
  if (m.sweep_file) {
    if (m.sweep_file_in_snapshot) {
      const a = el("a", null, m.sweep_file);
      a.onclick = () => openPath(m.sweep_file, { push: true });
      a.title = "Open the sweep file";
      meta.append(metaItem("sweep", a));
    } else meta.append(metaItem("sweep", m.sweep_file));
  }
  if (sub) meta.append(metaItem("submitted", `${fmtWhen(sub.submit_time)}${m.host ? " · " + m.host : ""}`));
  else if (!S.snapOnly) meta.append(metaItem("", el("span", null, "submitted before mlsweep recorded where code came from")));
  if (S.snapOnly && S.usage) {
    for (const e of S.usage.experiments.slice(0, 3)) {
      const a = el("a", null, e.name || e.experiment_id);
      a.href = pageHref("code.html", { experiment: e.experiment_id });
      meta.append(metaItem("used by", a));
    }
  }

  $("zip-link").href = authURL(`/api/snapshots/${encodeURIComponent(S.snapshot)}/archive.zip?name=${
    encodeURIComponent((S.runId || S.expId || short(S.snapshot)).replace(/[^\w.-]+/g, "_"))}`);
  $("details-btn").classList.toggle("on", S.details);
  renderCompareBar();
  if (S.details) renderDetails();
}

function fmtWhen(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return iso || "";
  return d.toLocaleString([], { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// ── provenance panel ─────────────────────────────────────────────────────────

let fullSub = null;   // the full submission (with its diff), fetched on demand

function openDetails(scrollToDiff) {
  S.details = true;
  $("details-btn").classList.add("on");
  renderDetails(scrollToDiff);
}

async function renderDetails(scrollToDiff) {
  const box = $("details");
  box.classList.toggle("on", S.details);
  if (!S.details) return;
  const token = navSeq, snapshot = S.snapshot;
  const sub = submissionOf(snapshot);
  box.replaceChildren();
  const grid = el("div", "prov-grid");
  const row = (k, v) => { grid.append(el("div", "k", k)); const d = el("div", "v"); d.append(v); grid.append(d); };
  const txt = s => document.createTextNode(s);
  row("Snapshot", txt(S.snapshot));
  if (!sub) {
    row("Provenance", el("span", "muted", "None recorded. This code was submitted before mlsweep recorded where code came from."));
    box.append(grid);
    return;
  }
  const m = sub.meta || {};
  row("Submitted", txt(`${fmtWhen(sub.submit_time)}${m.host ? " from " + m.host : ""}`));
  if (m.root) row("Project", txt(m.root));
  if (m.sweep_file) row("Sweep file", txt(m.sweep_file + (m.sweep_file_in_snapshot ? "" : "  (outside the snapshot; its source was kept)")));
  const vcs = m.vcs;
  if (vcs && vcs.commit) {
    row("Commit", txt(`${vcs.commit}${vcs.branch ? "  on " + vcs.branch : ""}`));
    if (vcs.subject) row("Message", txt(vcs.subject + (vcs.author ? `  (${vcs.author})` : "")));
    if (vcs.remote) row("Remote", txt(vcs.remote));
    if (vcs.dirty) {
      const ul = el("ul", "prov-list");
      for (const f of vcs.changed_files || []) ul.append(el("li", null, f));
      row("Uncommitted", ul);
    } else row("Uncommitted", el("span", "muted", "none; the tree matched the commit"));
    if (vcs.untracked_count) {
      row("Not in git", txt(`${vcs.untracked_count} file(s), ${fmtSize(vcs.untracked_bytes)}: `
        + (vcs.untracked || []).slice(0, 8).join(", ") + (vcs.untracked_count > 8 ? ", …" : "")));
    }
  } else row("Git", el("span", "muted", "not a git repository"));
  box.append(grid);
  if (m.argv && m.argv.length) {
    const sec = el("div", "prov-section");
    sec.append(el("h4", null, "Command"), el("div", "prov-cmd", "mlsweep run " + m.argv.join(" ")));
    box.append(sec);
  }
  if ((vcs && vcs.diff_bytes) || m.sweep_source_bytes) {
    let fullSubForThis = fullSub;
    if (!fullSubForThis || fullSubForThis.submission_id !== sub.submission_id) {
      try { fullSubForThis = await getJSON(`/api/code/submissions/${sub.submission_id}`); }
      catch { fullSubForThis = null; }
    }
    if (token !== navSeq || S.snapshot !== snapshot || !S.details) return;
    fullSub = fullSubForThis;
    const full = (fullSubForThis && fullSubForThis.meta) || {};
    if (full.vcs && full.vcs.diff) {
      const sec = el("div", "prov-section");
      sec.append(el("h4", null, "Uncommitted changes" + (full.vcs.diff_truncated ? " (cut at 1 MB)" : "")));
      sec.append(diffTable(parseGitDiff(full.vcs.diff)));
      box.append(sec);
      if (scrollToDiff) sec.scrollIntoView({ block: "start" });
    }
    if (full.sweep_source) {
      const sec = el("div", "prov-section");
      sec.append(el("h4", null, `Sweep file ${m.sweep_file}`), el("div", "prov-cmd", full.sweep_source));
      box.append(sec);
    }
  }
}

// ── comparing ────────────────────────────────────────────────────────────────

async function loadCompare(token = navSeq) {
  const compare = S.compare, snapshot = S.snapshot;
  S.diff = null;
  if (!compare) return;
  try {
    const diff = await getJSON(`/api/code/diff?from=${encodeURIComponent(compare)}&to=${encodeURIComponent(snapshot)}&context=4`);
    if (token !== navSeq || S.compare !== compare || S.snapshot !== snapshot) return;
    S.diff = diff;
  } catch (e) {
    if (token !== navSeq || S.compare !== compare || S.snapshot !== snapshot) return;
    await mlAlert(`Can't compare with ${compare}: ${e.message}`);
    if (token === navSeq) S.compare = "";
  }
}

function renderCompareBar() {
  const bar = $("compare-bar");
  bar.classList.toggle("on", !!S.diff);
  if (!S.diff) return;
  const d = S.diff, c = d.counts;
  const text = $("compare-text");
  text.replaceChildren(document.createTextNode("Compared with "), el("b", null, compareLabel()),
                       document.createTextNode(" · "));
  if (d.same) text.append(document.createTextNode("the same code"));
  else {
    text.append(el("span", "n mod", `${c.changed} changed`), document.createTextNode(", "),
                el("span", "n add", `${c.added} added`), document.createTextNode(", "),
                el("span", "n del", `${c.removed} removed`));
  }
  $("changed-only").checked = S.changedOnly;
}

// What the code is compared with, in words: "v1", a run, or an experiment's name.
function compareLabel() {
  const d = S.diff;
  if (!d) return "";
  const v = versionNumber(d.from.snapshot);
  if (!d.from.experiment && v) return `v${v} of this experiment`;
  if (d.from.experiment) {
    const e = S.exps.find(x => x.experiment_id === d.from.experiment);
    return (e && e.name) || d.from.experiment;
  }
  const runs = S.jobs.filter(j => j.artifact_id === d.from.snapshot);
  return runs.length ? `the code of ${runs[0].label || runs[0].run_id}` : short(d.from.snapshot);
}

// Choose what to compare with: another version, a run that used other code, or another experiment.
async function pickCompare() {
  const items = [];
  const cur = versionNumber(S.snapshot);
  for (const [i, v] of versions().entries()) {
    if (v.artifact_id === S.snapshot || v.kind !== "snapshot") continue;
    items.push({ value: v.artifact_id, group: "This experiment's other versions", mono: true,
                 label: `v${i + 1}${i + 1 === cur - 1 ? "  (previous)" : ""}${v.artifact_id === latestSnapshot() ? "  (latest)" : ""}`,
                 sub: `${short(v.artifact_id)} · ${v.runs} run${v.runs === 1 ? "" : "s"}` });
  }
  const seen = new Set([S.snapshot]);
  for (const j of S.jobs) {
    if (!j.artifact_id || !j.artifact_id.startsWith("snap:") || seen.has(j.artifact_id)) continue;
    seen.add(j.artifact_id);
    items.push({ value: j.artifact_id, group: "Runs that used other code", mono: true,
                 label: j.label || j.run_id, sub: `v${versionNumber(j.artifact_id)} · ${j.status}` });
  }
  const here = S.expId ? MLCampaign.campaignOf(S.exps.find(e => e.experiment_id === S.expId)) : null;
  items.push(...mlExperimentItems(S.exps, { exclude: S.expId ? [S.expId] : [], campaign: here })
    .map(it => ({ ...it, group: it.group ? `Experiments · ${it.group.replace(/^Campaign /, "")}` : "Experiments" })));
  const target = await mlPick("Compare this code with:", items, { placeholder: "Search versions, runs and experiments" });
  if (target) await compareWith(target);
}

async function compareWith(target, path) {
  S.compare = target;
  S.changedOnly = true;
  await showCode(S.snapshot, S.runId ? S.jobs.find(j => j.run_id === S.runId) : null,
                 { push: true, ...(path !== undefined ? { path, lines: "" } : {}) });
}

// ── right-click menus ────────────────────────────────────────────────────────

function previousVersion() {
  const n = versionNumber(S.snapshot);
  return n > 1 ? versions()[n - 2].artifact_id : null;
}

function treeMenu(e, path, dir) {
  const name = path.split("/").pop() || "code";
  const prev = previousVersion();
  const inSnapshot = dir || !!fileEntry(path);
  mlContextMenu(e, [
    { title: dir ? name + "/" : name },
    { label: dir ? "Open folder" : "Open", run: () => openPath(path, { push: true }) },
    !dir && inSnapshot ? { label: "Open in a new tab", run: () => window.open(pathHref(path), "_blank") } : null,
    "-",
    !dir && prev && !S.compare ? { label: "Compare with the previous version", hint: `v${versionNumber(prev)}`, run: () => compareWith(prev, path) } : null,
    !dir && S.compare ? { label: "Show the whole file, not the diff", run: () => { S.compare = ""; showCode(S.snapshot, S.runId ? S.jobs.find(j => j.run_id === S.runId) : null, { push: true, path, lines: "" }); } } : null,
    "-",
    inSnapshot ? { label: "Copy note link", run: () => copyRef("link", path) } : null,
    inSnapshot ? { label: "Copy link to this version", run: () => copyRef("pinned", path) } : null,
    { label: "Copy path", run: () => copyText(path, "Copied the path") },
    "-",
    !dir && inSnapshot ? { label: "Download file", run: () => window.open(authURL(fileUrl(path)), "_blank") } : null,
    dir ? { label: "Download all the code", hint: "zip", run: () => { location.href = $("zip-link").href; } } : null,
  ]);
}

function linesMenu(e, n) {
  if (S.start == null || n < S.start || n > S.end) { S.start = S.end = n; paintSelection(false); syncUrl(false); }
  const what = S.end !== S.start ? `lines ${S.start}–${S.end}` : `line ${S.start}`;
  mlContextMenu(e, [
    { title: `${S.path.split("/").pop()} · ${what}` },
    { label: `Copy note link to ${what}`, run: () => copyRef("link", S.path, true) },
    { label: "Copy link to this version", run: () => copyRef("pinned", S.path, true) },
    { label: "Copy as a code block for a note", run: () => copyRef("block", S.path, true) },
    "-",
    { label: `Copy the text of ${what}`, run: () => {
      const rows = [...$("view").querySelectorAll("table.src tr")].slice(S.start - 1, S.end);
      copyText(rows.map(r => r.querySelector("td.c").textContent).join("\n"), "Copied");
    } },
    { label: "Clear the selection", run: () => { S.start = S.end = null; paintSelection(false); syncUrl(false); } },
  ]);
}

function runMenu(e, job) {
  const differs = job.artifact_id && job.artifact_id !== S.snapshot && job.artifact_id.startsWith("snap:");
  mlContextMenu(e, [
    { title: job.label || job.run_id },
    { label: "Show this run's code", run: () => showCode(job.artifact_id, job, { push: true }) },
    differs ? { label: "Compare with the code shown now", hint: `v${versionNumber(job.artifact_id)}`, run: () => compareWith(job.artifact_id) } : null,
    "-",
    { label: "Open logs", run: () => { location.href = pageHref("logs.html", { experiment: job.experiment_id, run: job.run_id }); } },
    { label: "Open artifacts", run: () => { location.href = pageHref("files.html", { experiment: job.experiment_id, run: job.run_id }); } },
    { label: "Open results", run: () => { location.href = pageHref("results.html", { experiment: job.experiment_id }); } },
    "-",
    { label: "Copy run ID", run: () => copyText(job.run_id, "Copied the run ID") },
  ]);
}

// ── file tree ────────────────────────────────────────────────────────────────

function buildTree(files, dirs) {
  const root = { dirs: new Map(), files: [] };
  const dirOf = p => {
    let node = root;
    for (const part of p.split("/")) {
      if (!node.dirs.has(part)) node.dirs.set(part, { dirs: new Map(), files: [] });
      node = node.dirs.get(part);
    }
    return node;
  };
  for (const d of dirs || []) dirOf(d);
  for (const f of files) {
    const i = f.path.lastIndexOf("/");
    (i < 0 ? root : dirOf(f.path.slice(0, i))).files.push({ ...f, name: f.path.slice(i + 1) });
  }
  return root;
}

function treeFiles() {
  // The snapshot's files, plus those deleted since what it's compared with.
  let files = S.manifest.files.map(f => ({ path: f[0], size: f[2], status: (changeOf(f[0]) || {}).status || "" }));
  if (S.diff) {
    for (const c of S.diff.changes) if (c.status === "D") files.push({ path: c.path, size: null, status: "D" });
    if (S.changedOnly) files = files.filter(f => f.status);
  }
  const q = S.filter.trim().toLowerCase();
  if (q) files = files.filter(f => f.path.toLowerCase().includes(q));
  return files;
}

function renderTree() {
  const tree = $("file-tree");
  const scroll = tree.scrollTop;
  tree.replaceChildren();
  const files = treeFiles();
  const title = $("tree-title");
  title.replaceChildren(S.diff && S.changedOnly ? "Changed files" : "Files", el("span", "count", String(files.length)));
  const expandAll = !!S.filter.trim() || (S.diff && S.changedOnly);
  if (!files.length) {
    tree.append(el("div", "tree-msg", S.filter ? "No file matches." : S.diff ? "Nothing changed." : "No files."));
    return;
  }
  const changedDirs = new Set();
  for (const f of files) if (f.status) {
    const ps = f.path.split("/");
    for (let i = 1; i < ps.length; i++) changedDirs.add(ps.slice(0, i).join("/"));
  }
  // The rows are the notes tree's: a chevron on folders, a glyph per file type,
  // the extension dimmed.
  const draw = (node, prefix, depth, host) => {
    for (const name of [...node.dirs.keys()].sort(TREE_ORDER.compare)) {
      const p = prefix ? `${prefix}/${name}` : name;
      const open = expandAll || openDirs.has(p);
      const row = el("div", "tnode dir" + (open ? " open" : "") + (p === S.path ? " active" : ""));
      row.style.paddingLeft = (8 + depth * 14) + "px";
      row.dataset.path = p;
      const chev = el("span", "chev", "▶");
      const icon = el("span", "ficon"); icon.innerHTML = FILE_ICONS.dir;
      row.append(chev, icon, el("span", "tname", name));
      if (changedDirs.has(p)) { const dot = el("span", "dot"); dot.title = "Something in here changed"; row.append(dot); }
      const kids = el("div", "tchildren" + (open ? "" : " closed"));
      const setOpen = want => {
        row.classList.toggle("open", want); kids.classList.toggle("closed", !want);
        if (want) openDirs.add(p); else openDirs.delete(p);
      };
      chev.onclick = e => { e.stopPropagation(); setOpen(!row.classList.contains("open")); };
      // A click shows the folder's listing and opens it; a click on the folder
      // already shown folds it.
      row.onclick = () => {
        if (p === S.path && row.classList.contains("open")) { setOpen(false); return; }
        setOpen(true); openPath(p, { push: true });
      };
      row.oncontextmenu = e => treeMenu(e, p, true);
      host.append(row, kids);
      draw(node.dirs.get(name), p, depth + 1, kids);
    }
    for (const f of node.files.sort((a, b) => TREE_ORDER.compare(a.name, b.name))) {
      const row = el("div", "tnode file" + (f.status ? " ch-" + f.status : "") + (f.path === S.path ? " active" : ""));
      row.style.paddingLeft = (8 + depth * 14 + 17) + "px";
      row.dataset.path = f.path;
      row.title = f.size == null ? f.path : `${f.path}\n${fmtSize(f.size)}`;
      const icon = el("span", "ficon"); icon.innerHTML = FILE_ICONS[fileKindOf(f.name)] || FILE_ICONS.binary;
      const name = el("span", "tname");
      const dot = f.name.lastIndexOf(".");
      if (dot > 0) name.append(f.name.slice(0, dot), el("span", "ext", f.name.slice(dot)));
      else name.textContent = f.name;
      row.append(icon, name);
      if (f.status) row.append(el("span", "mark", f.status === "mode" ? "M" : f.status));
      row.onclick = () => openPath(f.path, { push: true });
      row.oncontextmenu = e => treeMenu(e, f.path, false);
      host.append(row);
    }
  };
  draw(buildTree(files, S.diff && S.changedOnly ? [] : S.manifest.dirs), "", 0, tree);
  tree.scrollTop = scroll;
}

// Opens the folders down to *path*, marks its row and scrolls it into view.
function revealInTree(path) {
  const parts = path.split("/");
  let added = false;
  for (let i = 1; i <= parts.length; i++) {
    const p = parts.slice(0, i).join("/");
    if (i < parts.length || isDir(path)) { if (!openDirs.has(p)) { openDirs.add(p); added = true; } }
  }
  if (added) renderTree();
  const rows = document.querySelectorAll("#file-tree .tnode");
  rows.forEach(r => r.classList.toggle("active", r.dataset.path === path));
  [...rows].find(r => r.dataset.path === path)?.scrollIntoView({ block: "nearest" });
}

// "Wrap" folds long lines in the file and diff views instead of scrolling sideways.
$("wrap-lines").checked = !!MLPrefs.get("code_wrap", false);
$("view").classList.toggle("wrap", $("wrap-lines").checked);
$("wrap-lines").addEventListener("change", e => {
  MLPrefs.set("code_wrap", e.target.checked);
  $("view").classList.toggle("wrap", e.target.checked);
});

$("tree-collapse").onclick = () => { openDirs.clear(); renderTree(); };

// ── highlighting ─────────────────────────────────────────────────────────────

function modeForPath(p) {
  const e = (p.split(".").pop() || "").toLowerCase();
  const map = { py: "python", sh: "shell", bash: "shell", zsh: "shell", js: "javascript", ts: "text/typescript",
    json: "application/json", jsonl: "application/json", yaml: "yaml", yml: "yaml", toml: "toml", diff: "diff",
    patch: "diff", c: "text/x-csrc", h: "text/x-csrc", cc: "text/x-c++src", cpp: "text/x-c++src", hpp: "text/x-c++src",
    cu: "text/x-c++src", java: "text/x-java", html: "xml", xml: "xml", md: "markdown", rs: "text/x-rustsrc", go: "text/x-go" };
  return map[e] || null;
}

// *text*'s lines as DOM fragments, highlighted when CodeMirror knows the language.
function highlightLines(text, path) {
  const lines = [document.createDocumentFragment()];
  const push = (txt, style) => {
    txt.split("\n").forEach((part, i) => {
      if (i > 0) lines.push(document.createDocumentFragment());
      if (part) lines[lines.length - 1].append(style ? el("span", style.split(" ").map(s => "cm-" + s).join(" "), part)
                                                    : document.createTextNode(part));
    });
  };
  const mode = modeForPath(path);
  if (mode && typeof CodeMirror !== "undefined" && CodeMirror.runMode) {
    try { CodeMirror.runMode(text, mode, push); }
    catch { lines.length = 1; lines[0] = document.createDocumentFragment(); push(text, null); }
  } else push(text, null);
  if (text.endsWith("\n") && lines.length > 1) lines.pop();
  return lines;
}

// ── diffs ────────────────────────────────────────────────────────────────────

// A `git diff` as [{path, lines}] (the unified-diff lines after each file's header).
function parseGitDiff(text) {
  const files = [];
  let cur = null;
  for (const line of text.split("\n")) {
    const m = /^diff --git a\/(.*) b\/(.*)$/.exec(line);
    if (m) { cur = { path: m[2], lines: [] }; files.push(cur); continue; }
    if (!cur || /^(index |--- |\+\+\+ |new file mode|deleted file mode|old mode|new mode|similarity|rename |Binary files)/.test(line)) {
      if (cur && line.startsWith("Binary files")) cur.lines.push("@@ binary file @@");
      continue;
    }
    cur.lines.push(line);
  }
  return files;
}

// Rows of old number, new number, sign and line, for each file's unified diff.
function diffTable(files) {
  const table = el("table", "src diff");
  const body = el("tbody");
  for (const f of files) {
    if (files.length > 1 || f.path !== S.path) {
      const tr = el("tr", "file");
      const td = el("td", null, f.path);
      td.colSpan = 4;
      tr.append(td);
      body.append(tr);
    }
    let oldN = 0, newN = 0;
    const lines = f.lines.filter((l, i) => !(i === f.lines.length - 1 && l === ""));
    const texts = highlightLines(lines.map(l => (l.startsWith("@@") ? "" : l.slice(1))).join("\n"), f.path);
    lines.forEach((line, i) => {
      const tr = el("tr");
      const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/.exec(line);
      if (hunk || line.startsWith("@@")) {
        if (hunk) { oldN = Number(hunk[1]); newN = Number(hunk[2]); }
        tr.className = "hunk";
        tr.append(el("td", "n", "⋯"), el("td", "n", "⋯"), el("td", "sign"),
                  el("td", null, hunk ? `from line ${newN}${hunk[3].trim() ? " · " + hunk[3].trim() : ""}` : line.replace(/@/g, "").trim()));
        body.append(tr);
        return;
      }
      if (line.startsWith("\\")) return;   // "\ No newline at end of file"
      const kind = line[0] === "+" ? "add" : line[0] === "-" ? "del" : "ctx";
      tr.className = kind;
      const c = el("td", "c");
      c.append(texts[i] || document.createTextNode(""));
      tr.append(el("td", "n", kind === "add" ? "" : String(oldN)), el("td", "n", kind === "del" ? "" : String(newN)),
                el("td", "sign", kind === "add" ? "+" : kind === "del" ? "−" : ""), c);
      if (kind !== "add") oldN++;
      if (kind !== "del") newN++;
      body.append(tr);
    });
  }
  table.append(body);
  return table;
}

// ── the viewer ───────────────────────────────────────────────────────────────

function paintSelection(scroll) {
  const rows = $("view").querySelectorAll("table.src:not(.diff) tr");
  rows.forEach(tr => tr.classList.remove("hl"));
  if (S.start != null) {
    for (let n = S.start; n <= S.end; n++) if (rows[n - 1]) rows[n - 1].classList.add("hl");
    const first = rows[S.start - 1];
    if (scroll && first) {
      // A third of the way down, so the lines above give some context.
      requestAnimationFrame(() => { $("view").scrollTop = Math.max(0, first.offsetTop - $("view").clientHeight / 3); });
    }
  }
  const sel = $("sel");
  sel.classList.toggle("tip", S.start == null);
  sel.textContent = S.start == null ? "click a line number to select it · right-click for links"
    : S.end !== S.start ? `lines ${S.start}–${S.end}` : `line ${S.start}`;
}

function renderCrumbs() {
  const crumbs = $("crumbs");
  crumbs.replaceChildren();
  const root = el("a", null, S.snapOnly ? short(S.snapshot) : (S.expId || "code"));
  root.onclick = () => openPath("", { push: true });
  crumbs.append(root);
  if (!S.path) return;
  const parts = S.path.split("/");
  parts.forEach((part, i) => {
    crumbs.append(el("span", "sep", "/"));
    if (i === parts.length - 1) { crumbs.append(el("b", null, part)); return; }
    const a = el("a", null, part);
    const p = parts.slice(0, i + 1).join("/");
    a.onclick = () => openPath(p, { push: true });
    crumbs.append(a);
  });
}

function renderDir(path) {
  const pre = path ? path + "/" : "";
  const dirs = new Map(), files = [];
  for (const f of S.manifest.files) {
    if (!f[0].startsWith(pre)) continue;
    const rest = f[0].slice(pre.length);
    const slash = rest.indexOf("/");
    if (slash < 0) files.push(f);
    else {
      const d = rest.slice(0, slash);
      dirs.set(d, (dirs.get(d) || 0) + f[2]);
    }
  }
  for (const d of S.manifest.dirs || []) {
    if (d.startsWith(pre) && !d.slice(pre.length).includes("/")) dirs.set(d.slice(pre.length), dirs.get(d.slice(pre.length)) || 0);
  }
  const table = el("table", "dirlist");
  const body = el("tbody");
  for (const [d, size] of [...dirs].sort((a, b) => a[0].localeCompare(b[0]))) {
    const tr = el("tr", "d");
    tr.append(el("td", "name", d + "/"), el("td", "size", fmtSize(size)));
    tr.onclick = () => { openDirs.add(pre + d); openPath(pre + d, { push: true }); };
    body.append(tr);
  }
  for (const f of files.sort((a, b) => a[0].localeCompare(b[0]))) {
    const tr = el("tr");
    tr.append(el("td", "name", f[0].slice(pre.length)), el("td", "size", fmtSize(f[2])));
    tr.onclick = () => openPath(f[0], { push: true });
    body.append(tr);
  }
  table.append(body);
  $("file-meta").textContent = `${dirs.size} folder${dirs.size === 1 ? "" : "s"}, ${files.length} file${files.length === 1 ? "" : "s"}`;
  $("view").replaceChildren(table);
}

async function fileText(path, entry, snapshot = S.snapshot) {
  const key = snapshot + "\n" + path;
  if (fileCache.has(key)) return fileCache.get(key);
  const url = `/api/snapshots/${encodeURIComponent(snapshot)}/files/${path.split("/").map(encodeURIComponent).join("/")}`;
  const r = await apiFetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  const text = bytes.subarray(0, 8192).includes(0) ? null : new TextDecoder().decode(bytes);
  if (entry[2] < (512 << 10)) fileCache.set(key, text);
  return text;
}

async function openPath(path, { keepLines = false, push = false, token = null } = {}) {
  // A click in the tree starts a new navigation; showCode passes its own token
  // so its follow-up calls are not mistaken for a newer view.
  const seq = token ?? ++navSeq;
  const snapshot = S.snapshot;
  S.path = path || "";
  if (!keepLines) S.start = S.end = null;
  if (S.path) revealInTree(S.path);
  else document.querySelectorAll("#file-tree .tnode.active").forEach(r => r.classList.remove("active"));
  renderCrumbs();
  syncUrl(push);
  const view = $("view"), meta = $("file-meta"), raw = $("raw-link");
  const entry = fileEntry(S.path);
  const change = changeOf(S.path);
  raw.style.display = entry ? "" : "none";
  meta.textContent = "";
  $("sel").textContent = "";

  if (!entry && isDir(S.path) && !(change && change.status === "D")) { renderDir(S.path); return; }
  if (change) {
    const fd = S.diff.files.find(f => f.path === S.path);
    if (fd && fd.diff && fd.diff.length) {
      meta.textContent = `+${fd.added || 0} −${fd.removed || 0}` + (fd.truncated ? " · cut short" : "");
      if (entry) raw.href = authURL(`/api/snapshots/${encodeURIComponent(S.snapshot)}/files/${S.path.split("/").map(encodeURIComponent).join("/")}`);
      view.replaceChildren(diffTable([{ path: S.path, lines: fd.diff }]));
      return;
    }
    if (change.status === "D") { view.replaceChildren(el("div", "msg", "Deleted in this version.")); return; }
  }
  if (!entry) {
    view.replaceChildren(el("div", "msg err", `No file ${S.path} in this code.`));
    return;
  }
  const url = `/api/snapshots/${encodeURIComponent(S.snapshot)}/files/${S.path.split("/").map(encodeURIComponent).join("/")}`;
  raw.href = authURL(url);
  // As on the artifacts page: models and arrays get the manager's summary,
  // images are shown.  (An SVG is source here, so it stays text.)
  if (MLModelPreview.isModel(S.path)) {
    meta.textContent = fmtSize(entry[2]);
    const box = el("div", "file-preview");
    view.replaceChildren(box);
    MLModelPreview.render(url, box, false, { info: url.replace("/files/", "/file-info/") });
    return;
  }
  if (fileKindOf(S.path) === "image" && !/\.svg$/i.test(S.path)) {
    meta.textContent = fmtSize(entry[2]);
    const img = el("img");
    img.src = authURL(url); img.alt = S.path;
    const box = el("div", "file-preview");
    box.append(img);
    view.replaceChildren(box);
    return;
  }
  if (entry[2] > MAX_TEXT_BYTES) {
    meta.textContent = fmtSize(entry[2]);
    view.replaceChildren(el("div", "msg", `${fmtSize(entry[2])}, too large to show here. The download button next to the path has it.`));
    return;
  }
  view.replaceChildren(el("div", "msg", "Loading…"));
  let text;
  try { text = await fileText(S.path, entry, snapshot); }
  catch (e) {
    if (seq !== navSeq || S.path !== path || S.snapshot !== snapshot) return;
    view.replaceChildren(el("div", "msg err", `Could not load ${S.path}: ${e.message}`));
    return;
  }
  if (seq !== navSeq || S.path !== path || S.snapshot !== snapshot) return;   // another file was opened meanwhile
  if (text == null) {
    meta.textContent = fmtSize(entry[2]);
    view.replaceChildren(el("div", "msg", `Binary file, ${fmtSize(entry[2])}. The download button next to the path has it.`));
    return;
  }
  const lines = highlightLines(text, S.path);
  meta.textContent = `${lines.length} lines · ${fmtSize(entry[2])}`;
  const table = el("table", "src");
  const body = el("tbody");
  lines.forEach((frag, i) => {
    const tr = el("tr");
    const n = el("td", "n", String(i + 1));
    n.dataset.n = String(i + 1);
    const c = el("td", "c");
    c.append(frag);
    tr.append(n, c);
    body.append(tr);
  });
  table.append(body);
  table.addEventListener("contextmenu", e => {
    const td = e.target.closest("td.n");
    if (td) linesMenu(e, Number(td.dataset.n));
  });
  table.addEventListener("click", e => {
    const td = e.target.closest("td.n");
    if (!td) return;
    const n = Number(td.dataset.n);
    if (e.shiftKey && S.start != null) { S.end = Math.max(n, S.end); S.start = Math.min(n, S.start); }
    else if (S.start === n && S.end === n) { S.start = S.end = null; }
    else { S.start = S.end = n; }
    paintSelection(false);
    syncUrl(false);
  });
  view.replaceChildren(table);
  paintSelection(true);
}

// ── showing some code ────────────────────────────────────────────────────────

function showMessage(text, err) {
  $("code-header").style.display = "none";
  $("code-inner").style.display = "none";
  const box = $("no-sel");
  box.style.display = "";
  box.textContent = text;
  box.style.color = err ? "var(--bad)" : "";
}

// The file to show when none is asked for: the sweep file, else a README.
function defaultPath() {
  const have = new Set(S.manifest.files.map(f => f[0]));
  const m = (submissionOf(S.snapshot) || {}).meta || {};
  for (const p of [m.sweep_file_in_snapshot ? m.sweep_file : null, "README.md", "readme.md", "README"]) {
    if (p && have.has(p)) return p;
  }
  return "";
}

async function showCode(sid, job, { push = false, path = null, lines = "", compare = null } = {}) {
  const seq = ++navSeq;
  S.runId = job ? job.run_id : null;
  renderJobs();
  if (!sid) {
    showMessage(job ? "This run has no code recorded: it ran from the worker's own directory."
                    : "No code is recorded for this experiment.");
    syncUrl(push);
    return;
  }
  if (!sid.startsWith("snap:")) {
    showMessage("This ran from a code archive made before snapshots, which can't be shown.");
    syncUrl(push);
    return;
  }
  let manifest;
  try {
    manifest = await manifestOf(sid);
  } catch (e) {
    if (seq !== navSeq) return;
    showMessage(`Could not load this code: ${e.message}`, true);
    return;
  }
  if (seq !== navSeq) return;
  S.manifest = manifest;
  S.snapshot = sid;
  if (compare !== null) S.compare = compare;
  if (S.compare === sid) S.compare = "";
  await loadCompare(seq);
  if (seq !== navSeq) return;
  if (path !== null) { S.path = path; [S.start, S.end] = parseLines(lines); }
  const known = !S.path || fileEntry(S.path) || isDir(S.path) || changeOf(S.path);
  if (!known || (path === null && !S.path)) { S.path = known && S.path ? S.path : defaultPath(); S.start = S.end = null; }
  renderJobs();
  renderHeader();
  $("no-sel").style.display = "none";
  $("code-header").style.display = "";
  $("code-inner").style.display = "";
  renderTree();
  await openPath(S.path, { keepLines: true, push, token: seq });
}

async function loadExperimentData(expId, token = navSeq) {
  const [jobs, info] = await Promise.all([
    getJSON(`/api/experiments/${encodeURIComponent(expId)}/jobs`),
    getJSON(`/api/experiments/${encodeURIComponent(expId)}/code`),
  ]);
  if (token !== navSeq || S.expId !== expId) return false;
  S.jobs = jobs; S.info = info;
  return true;
}

async function selectExperiment(expId, want = {}) {
  const token = ++navSeq;
  S.snapOnly = false; S.usage = null;
  S.expId = expId;
  S.jobs = []; S.info = null; S.runId = null;
  $("job-list").innerHTML = `<div class="job-list-msg">Loading…</div>`;
  try { if (!await loadExperimentData(expId, token)) return; }
  catch (e) {
    if (token !== navSeq) return;
    showMessage(`Could not load ${expId}: ${e.message}`, true);
    return;
  }
  const job = want.run && S.jobs.find(j => j.run_id === want.run);
  let sid = job ? job.artifact_id : latestSnapshot();
  if (!job && want.snap) {
    const v = versions().find(x => x.artifact_id.startsWith(want.snap));
    if (v) sid = v.artifact_id;
  }
  await showCode(sid, job || null, { push: want.push, path: want.path ?? null, lines: want.lines || "",
                                     compare: want.compare ?? "" });
}

async function showSnapshotOnly(target, want = {}) {
  const token = ++navSeq;
  let res;
  try { res = await getJSON(`/api/code/resolve?ref=${encodeURIComponent("code/" + target)}`); }
  catch (e) { res = { ok: false, error: e.message }; }
  if (token !== navSeq) return;
  if (!res.ok) { showMessage(res.error || "not found", true); return; }
  S.snapOnly = true;
  S.expId = null; S.jobs = []; S.info = null;
  try { S.usage = await getJSON(`/api/snapshots/${encodeURIComponent(res.snapshot)}/usage`); } catch { S.usage = null; }
  if (token !== navSeq) return;
  expSelect.value = "";
  await showCode(res.snapshot, null, { path: want.path ?? null, lines: want.lines || "", compare: want.compare ?? "" });
}

// ── experiments ──────────────────────────────────────────────────────────────

const campaignSelect = $("campaign-select");
const expSelect = $("exp-select");

async function showCampaign(want) {
  MLCampaign.fill(campaignSelect, S.exps);
  const chosen = MLCampaign.fillExperiments(expSelect, MLCampaign.filter(S.exps), want && want.exp, S.expId);
  if (!chosen) { S.expId = null; showMessage("No experiments in this campaign."); $("job-list").innerHTML = ""; return; }
  if (chosen !== S.expId || S.snapOnly || want) await selectExperiment(chosen, want || {});
}

let expsSeq = 0;
async function loadExperiments() {
  const seq = ++expsSeq;
  try {
    const exps = await getJSON("/api/experiments");
    if (seq !== expsSeq) return false;
    S.exps = exps;
    setConnStatus("connected");
    return true;
  } catch {
    if (seq !== expsSeq) return false;
    setConnStatus("disconnected");
    return false;
  }
}

// Show what the URL asks for (on load, and on Back/Forward).
async function applyUrl() {
  const want = urlState();
  if (want.snap && !want.exp) {
    MLCampaign.fill(campaignSelect, S.exps);
    MLCampaign.fillExperiments(expSelect, MLCampaign.filter(S.exps));
    await showSnapshotOnly(want.snap, want);
    return;
  }
  if (want.exp) MLCampaign.follow(S.exps, want.exp);
  await showCampaign(want.exp ? want : null);
}

MLCampaign.bind(campaignSelect, () => showCampaign(null));
expSelect.addEventListener("change", () => { S.path = ""; S.compare = ""; selectExperiment(expSelect.value, { push: true }); });
$("refresh-btn").addEventListener("click", async () => {
  await loadExperiments();
  if (S.expId && !S.snapOnly) {
    const expId = S.expId, token = ++navSeq;
    try { if (!await loadExperimentData(expId, token)) return; } catch { return; }
    const job = S.runId && S.jobs.find(j => j.run_id === S.runId);
    const latest = latestSnapshot();
    await showCode(job ? job.artifact_id : (versionNumber(S.snapshot) ? S.snapshot : latest), job || null);
  }
});
$("version-select").addEventListener("change", e => showCode(e.target.value, null, { push: true }));
$("compare-btn").addEventListener("click", () => pickCompare());
$("compare-change").addEventListener("click", () => pickCompare());
$("compare-stop").addEventListener("click", () => {
  S.compare = "";
  showCode(S.snapshot, S.runId ? S.jobs.find(j => j.run_id === S.runId) : null, { push: true });
});
$("changed-only").addEventListener("change", e => { S.changedOnly = e.target.checked; renderTree(); });
$("details-btn").addEventListener("click", () => {
  S.details = !S.details;
  $("details-btn").classList.toggle("on", S.details);
  renderDetails();
});
$("tree-filter").addEventListener("input", e => { S.filter = e.target.value; renderTree(); });

// The Copy menu: a note link (following or pinned), a code block, or the path.
const copyPop = $("copy-pop");
$("copy-btn").addEventListener("click", e => { e.stopPropagation(); copyPop.classList.toggle("on"); });
document.addEventListener("click", e => { if (!e.target.closest(".menu-wrap")) copyPop.classList.remove("on"); });
copyPop.addEventListener("click", e => {
  const b = e.target.closest("button[data-copy]");
  if (!b) return;
  copyPop.classList.remove("on");
  if (b.dataset.copy === "path") { copyText(S.path || "/", "Copied the path"); return; }
  copyRef(b.dataset.copy, S.path, true);
});

function fileUrl(path) {
  return `/api/snapshots/${encodeURIComponent(S.snapshot)}/files/${path.split("/").map(encodeURIComponent).join("/")}`;
}
function pathHref(path) {
  const u = new URL(location.href);
  u.searchParams.set("path", path);
  u.searchParams.delete("lines");
  return u.pathname + u.search;
}

// Copy a reference for a note: "link" follows the experiment's latest code,
// "pinned" names this version, "block" is a mlsweep-code block.
function copyRef(kind, path, withLines) {
  const pinned = kind === "pinned" || S.snapOnly || !S.expId || S.snapshot !== latestSnapshot();
  const target = pinned ? short(S.snapshot) : S.expId;
  const lines = !withLines || S.start == null ? "" : S.end !== S.start ? `${S.start}-${S.end}` : String(S.start);
  if (kind === "block") {
    if (!fileEntry(path)) { mlAlert("Open a file first."); return; }
    copyText("```mlsweep-code\n" + `${pinned ? "snapshot" : "experiment"}: ${target}\npath: ${path}\n`
             + (lines ? `lines: ${lines}\n` : "") + "```", "Copied a code block");
    return;
  }
  const encoded = (path || "").split("/").map(encodeURIComponent).join("/");
  const name = (path ? path.split("/").pop() : (S.expId || target)) + (lines ? ` L${lines.replace("-", "–")}` : "");
  copyText(`[${name}](mlsweep:code/${target}${path ? "/" + encoded : ""}${lines ? "#L" + lines : ""})`,
           pinned ? "Copied a link to this version" : "Copied a note link");
}

window.addEventListener("popstate", () => applyUrl());

(async () => {
  if (await loadExperiments()) await applyUrl();
})();
