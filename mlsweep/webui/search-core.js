// Search for the command palette (palette.js), built on the existing APIs.
// Log and artifact search look at a bounded set of runs, since a full index
// would need the manager's help.
(() => {
  if (window.MLSearch) return;

  const DOCS = [
    { id: "sweep", title: "Sweep configuration", keys: "sweep config options command shebang mlsweep_run", snippet:
      "#!/usr/bin/env mlsweep_run\n\nCOMMAND = [\"python\", \"train.py\"]\n\nOPTIONS = {\n    \".lr\": {\"values\": [1e-4, 3e-4], \"flags\": \"--lr\"},\n}",
      body: "A sweep file is an executable Python file. COMMAND is the training command; OPTIONS describes each dimension and its flags." },
    { id: "optimize", title: "Bayesian optimization (OPTIMIZE)", keys: "optimize bayes optuna metric goal budget", snippet:
      "OPTIMIZE = {\n    \"method\": \"bayes\",\n    \"metric\": \"val_loss\",\n    \"goal\": \"minimize\",\n    \"budget\": 40,\n}",
      body: "Use optuna/TPE to sample the space instead of trying every combination. budget limits the number of runs." },
    { id: "subdims", title: "Subdimensions", keys: "subdimension nested options groups", snippet:
      "OPTIONS = {\n  \".optimizer\": {\n    \".adam\": {\"flags\": [\"--opt\", \"adam\"]},\n    \".muon\": {\"flags\": [\"--opt\", \"muon\"]},\n  },\n}",
      body: "A key beginning with a dot can contain child dimensions; choosing a child selects its flags." },
    { id: "exclude", title: "EXCLUDE and skip rules", keys: "exclude skip monotonic singular filter", snippet:
      "def EXCLUDE(combo: dict) -> bool:\n    return combo[\"weight_decay\"] == 0.0 and combo[\"wd_compensation\"] is True",
      body: "Return True to drop a combination before it is scheduled. Use subdims when a dimension only applies within certain values of another." },
    { id: "dist", title: "Distributed runs", keys: "torchrun nodes NODES_PER_RUN GPUS_PER_RUN SET_DIST_ENV", snippet:
      "NODES_PER_RUN = 2\nGPUS_PER_RUN = 8\nSET_DIST_ENV = True",
      body: "Request multiple nodes/GPUs per run. The worker sets the usual distributed environment variables." },
    { id: "return", title: "Returning artifact files", keys: "return files artifacts return_files", snippet:
      "RETURN_FILES = [\"checkpoints/best.pt\", \"samples/*.png\"]",
      body: "Files written to MLSWEEP_RUN_DIR are synced back. RETURN_FILES can restrict what is kept." },
    { id: "logger", title: "Logging metrics", keys: "logger MLSweepLogger log metrics", snippet:
      "from mlsweep.logger import MLSweepLogger\nwith MLSweepLogger() as logger:\n    logger.log({\"loss\": loss}, step=step)",
      body: "The logger is a no-op unless the script was launched by a worker. It writes metrics.jsonl and exposes MLSWEEP_RUN_DIR." },
    { id: "manager", title: "Manager and workers", keys: "manager worker ssh connect GPU scheduler", snippet:
      "mlsweep manager --port 7891\nmlsweep workers add user@host --remote-dir DIR",
      body: "The manager owns scheduling and the dashboard. Workers run jobs; remote ones are started over SSH." },
    { id: "campaigns", title: "Campaigns", keys: "campaign group experiments tags", snippet:
      "mlsweep_run --campaign my-campaign sweep.py",
      body: "Campaigns group experiments and are the main filter in the dashboard. The selector remembers your last choice." },
    { id: "export", title: "Exporting to W&B / TensorBoard", keys: "export wandb tensorboard", snippet:
      "mlsweep export wandb --project my-project\nmlsweep export tensorboard",
      body: "Export a run, experiment, campaign, or everything to W&B or TensorBoard. Raw metrics/logs/artifacts are also downloadable." },
    { id: "values", title: "Value types and flags", keys: "values flags name list range int float bool", snippet:
      "\".batch_size\": {\"values\": [32, 64, 128], \"flags\": \"--batch-size\", \"name\": \"bs\"}",
      body: "values is the list to sweep; flags is appended verbatim. name shortens generated run names." },
    { id: "cli", title: "CLI commands", keys: "cli status pause resume stop purge cancel retry", snippet:
      "mlsweep status\nmlsweep pause EXPERIMENT\nmlsweep resume EXPERIMENT\nmlsweep stop EXPERIMENT",
      body: "CLI verbs mirror the dashboard: status, pause, resume, stop, cancel, retry, purge, gc, export, notes." },
    { id: "notes", title: "Notes and git", keys: "notes git clone sync markdown comments", snippet:
      "mlsweep notes clone DIR\ncd DIR && git pull",
      body: "The Notes tab is a git repository hosted by the manager. Edit files locally or in the dashboard; changes sync through git." },
    { id: "artifacts", title: "Artifacts and checkpoints", keys: "artifacts files checkpoint rsync MLSWEEP_RUN_DIR", snippet:
      "save_checkpoint(os.environ[\"MLSWEEP_RUN_DIR\"], step)\nlogger.sync()",
      body: "Artifacts are rsynced at the end of a run and immediately on logger.sync(). The Artifacts tab previews text, JSON, CSV, and images." },
    { id: "placement", title: "Scheduler placement", keys: "placement pack spread gpu", snippet:
      "mlsweep scheduler --placement pack   # or spread (also on the System page)",
      body: "Pack fills busy GPUs first, keeping whole machines free. Spread puts each job on the least busy GPU." },
  ];

  const ICONS = { run: "▶", experiment: "◈", note: "✎", artifact: "▣", doc: "▤", answer: "◎", action: "⚡", page: "▦" };

  let exps = null, expsAt = 0, expMap = new Map();
  let jobs = null, jobsAt = 0;
  let workers = null, workersAt = 0;
  let loadingJobs = null;

  const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const qs = (page, params) => `${page}.html?` + new URLSearchParams(params).toString();
  const context = () => {
    const u = new URL(location.href);
    return {
      page: u.pathname.split("/").pop(),
      campaign: u.searchParams.get("campaign") || "default",
      experiment: u.searchParams.get("experiment") || "",
      run: u.searchParams.get("run") || "",
    };
  };

  async function getExperiments(force = false) {
    if (!force && exps && Date.now() - expsAt < 60000) return exps;
    try {
      const r = await apiFetch("/api/experiments");
      if (r.ok) { exps = await r.json(); expsAt = Date.now(); expMap = new Map(exps.map(e => [e.experiment_id, e])); }
    } catch {}
    return exps || (exps = []);
  }

  async function getAllJobs(force = false) {
    if (!force && jobs && Date.now() - jobsAt < 60000) return jobs;
    if (loadingJobs) return loadingJobs;
    loadingJobs = (async () => {
      await getExperiments();
      const statuses = ["running", "dispatched", "pending", "failed", "done", "cancelled"];
      const lists = await Promise.all(statuses.map(async status => {
        try {
          const r = await apiFetch(`/api/jobs?status=${status}`);
          return r.ok ? await r.json() : [];
        } catch { return []; }
      }));
      const all = lists.flat();
      const seen = new Set();
      jobs = all.filter(j => {
        const key = `${j.experiment_id}/${j.run_id}`;
        if (seen.has(key)) return false;
        seen.add(key);
        const e = expMap.get(j.experiment_id);
        j.campaign = j.campaign || e?.campaign || "default";
        j.experiment_name = e?.name || j.experiment_id;
        return true;
      });
      jobsAt = Date.now();
      return jobs;
    })();
    try { return await loadingJobs; } finally { loadingJobs = null; }
  }

  async function getWorkers(force = false) {
    if (!force && workers && Date.now() - workersAt < 30000) return workers;
    try {
      const r = await apiFetch("/api/workers");
      if (r.ok) { workers = await r.json(); workersAt = Date.now(); }
    } catch {}
    return workers || (workers = []);
  }

  function parseSince(v) {
    if (!v) return null;
    const now = Date.now();
    const m = v.match(/^(\d+)(m|h|d|w)$/i);
    if (m) return now - (+m[1] * { m: 60e3, h: 3600e3, d: 86400e3, w: 7 * 86400e3 }[m[2].toLowerCase()]);
    if (/^today$/i.test(v)) { const d = new Date(); d.setHours(0, 0, 0, 0); return d.getTime(); }
    if (/^yesterday$/i.test(v)) { const d = new Date(); d.setHours(0, 0, 0, 0); return d.getTime() - 86400e3; }
    const t = Date.parse(v); return isNaN(t) ? null : t;
  }

  function parseQuery(raw) {
    let text = String(raw || "").trim();
    const filters = { status: "", campaign: "", worker: "", type: "", has: "", file: "", log: "", note: "", docs: "", since: null };
    for (const m of text.matchAll(/\b(status|campaign|worker|type|has|file|log|note|docs):("([^"]*)"|\S+)/gi)) {
      const key = m[1].toLowerCase();
      const val = (m[3] ?? m[2]).replace(/^["']|["']$/g, "");
      filters[key] = val;
      text = text.replace(m[0], " ");
    }
    for (const m of text.matchAll(/\bsince:(\S+)/gi)) {
      filters.since = parseSince(m[1].trim());
      text = text.replace(m[0], " ");
    }
    const notePref = text.match(/(?:^|\s)@(\S+)/);
    if (notePref) { filters.note = notePref[1]; text = text.replace(notePref[0], " "); }
    const bang = text.match(/(?:^|\s)!(\w+)/);
    if (bang) { filters.status = bang[1]; text = text.replace(bang[0], " "); }
    const clauses = [];
    for (const m of text.matchAll(/\b([A-Za-z_][\w./-]*)\s*(<=|>=|!=|=|<|>)\s*("[^"]*"|\S+)/g)) {
      clauses.push({ key: m[1], op: m[2], value: m[3].replace(/^["']|["']$/g, "") });
      text = text.replace(m[0], " ");
    }
    const answer = {};
    if (/\bbest\b/i.test(text)) {
      const m = text.match(/\bbest\s+(?:run\s+)?(?:for\s+)?([\w./-]+)/i);
      answer.best = m ? m[1] : "";
      text = text.replace(/\bbest\b.*$/i, " ");
    }
    if (/\bwhat changed|\bchanges\b|\bsince yesterday\b/i.test(text)) { answer.changes = true; text = text.replace(/(what changed|changes|since yesterday)/ig, " "); }
    if (/\bwhat failed|\bfailed\b/i.test(text) && !filters.status) { answer.failed = true; text = text.replace(/what failed|failed/ig, " "); }
    if (/\brunning\b/i.test(text) && !filters.status) { answer.running = true; text = text.replace(/running/ig, " "); }
    if (/\bidle\b|\bworkers?\b/i.test(text)) { answer.workers = true; text = text.replace(/idle|workers?/ig, " "); }
    if (/\bstalled\b/i.test(text)) { answer.stalled = true; text = text.replace(/stalled/ig, " "); }
    const terms = text.split(/\s+/).map(t => t.trim().toLowerCase()).filter(Boolean);
    return { raw: String(raw || ""), text: terms.join(" "), terms, filters, clauses, answer };
  }

  function campaignOf(j) { return j.campaign || expMap.get(j.experiment_id)?.campaign || "default"; }
  function jobText(j) {
    return `${j.run_id} ${j.label || ""} ${j.experiment_id} ${j.experiment_name || ""} ${j.worker_id || ""} ${j.status} ${j.command || ""} ${campaignOf(j)}`.toLowerCase();
  }
  function expText(e) {
    return `${e.name || ""} ${e.experiment_id} ${e.campaign || ""} ${e.note || ""} ${e.status || ""}`.toLowerCase();
  }
  function matchSince(j, since) {
    if (!since) return true;
    const t = j.finish_time || j.start_time || j.submit_time || 0;
    return new Date(t).getTime() >= since;
  }
  function matchClauses(obj, clauses) {
    for (const c of clauses) {
      let actual;
      if (c.key === "status") actual = obj.status;
      else if (c.key === "campaign") actual = obj.campaign;
      else if (c.key === "worker") actual = obj.worker_id;
      else if (c.key === "name") actual = obj.name || obj.run_id;
      else if (c.key === "run" || c.key === "run_id") actual = obj.run_id;
      else if (c.key === "experiment") actual = obj.experiment_id;
      else actual = obj[c.key];
      if (actual === undefined || actual === null) return false;
      const an = Number(actual), en = Number(c.value);
      const numeric = isFinite(an) && isFinite(en) && String(actual).trim() !== "";
      const a = numeric ? an : String(actual).toLowerCase();
      const b = numeric ? en : c.value.toLowerCase();
      const ok = c.op === "=" ? a === b : c.op === "!=" ? a !== b : c.op === "<" ? a < b : c.op === ">" ? a > b : c.op === "<=" ? a <= b : a >= b;
      if (!ok) return false;
    }
    return true;
  }
  function matchJob(j, p, scope) {
    const ctx = context();
    if (scope === "experiment" && ctx.experiment && j.experiment_id !== ctx.experiment) return false;
    if (scope === "campaign" && ctx.campaign && campaignOf(j) !== ctx.campaign) return false;
    if (p.filters.status && !String(j.status).toLowerCase().includes(p.filters.status.toLowerCase())) return false;
    if (p.filters.campaign && !campaignOf(j).toLowerCase().includes(p.filters.campaign.toLowerCase())) return false;
    if (p.filters.worker && !String(j.worker_id || "").toLowerCase().includes(p.filters.worker.toLowerCase())) return false;
    if (p.filters.type && !["run", "runs"].includes(p.filters.type.toLowerCase())) return false;
    if (!matchSince(j, p.filters.since)) return false;
    let combo = {};
    try { combo = JSON.parse(j.combo || "{}"); } catch {}
    for (const c of p.clauses) {
      if (!(c.key in combo)) continue; // metric clauses are handled by Results answers
      const av = String(combo[c.key]), bv = c.value;
      const an = Number(av), bn = Number(bv);
      const numeric = isFinite(an) && isFinite(bn) && av.trim() !== "";
      const a = numeric ? an : av.toLowerCase(), b = numeric ? bn : bv.toLowerCase();
      const ok = c.op === "=" ? a === b : c.op === "!=" ? a !== b : c.op === "<" ? a < b : c.op === ">" ? a > b : c.op === "<=" ? a <= b : a >= b;
      if (!ok) return false;
    }
    const hay = jobText(j);
    return p.terms.every(t => hay.includes(t));
  }
  function matchExp(e, p, scope) {
    const ctx = context();
    if (scope === "experiment" && ctx.experiment && e.experiment_id !== ctx.experiment) return false;
    if (scope === "campaign" && ctx.campaign && (e.campaign || "default") !== ctx.campaign) return false;
    if (p.filters.status && !String(e.status || "").toLowerCase().includes(p.filters.status.toLowerCase())) return false;
    if (p.filters.campaign && !String(e.campaign || "").toLowerCase().includes(p.filters.campaign.toLowerCase())) return false;
    if (p.filters.type && !["experiment", "experiments"].includes(p.filters.type.toLowerCase())) return false;
    if (p.filters.has === "artifacts" && !(e.job_counts?.with_artifacts > 0)) return false;
    if (p.filters.since && new Date(e.submit_time || 0).getTime() < p.filters.since) return false;
    if (!matchClauses({ ...e, campaign: e.campaign || "default" }, p.clauses.filter(c => ["status", "campaign", "name", "experiment"].includes(c.key)))) return false;
    const hay = expText(e);
    return p.terms.every(t => hay.includes(t));
  }

  function runHref(j) { return qs("results", { campaign: campaignOf(j), experiment: j.experiment_id, run: j.run_id }); }
  function logsHref(j) { return qs("logs", { campaign: campaignOf(j), experiment: j.experiment_id, run: j.run_id }); }
  function filesHref(j, file) { return qs("files", { campaign: campaignOf(j), experiment: j.experiment_id, run: j.run_id, ...(file ? { file } : {}) }); }
  function expResultsHref(e) { return qs("results", { campaign: e.campaign || "default", experiment: e.experiment_id }); }
  function expFilesHref(e) { return qs("files", { campaign: e.campaign || "default", experiment: e.experiment_id }); }
  function codeHref(j) { return qs("code", { campaign: campaignOf(j), experiment: j.experiment_id, run: j.run_id }); }
  function expCodeHref(e) { return qs("code", { campaign: e.campaign || "default", experiment: e.experiment_id }); }
  function expLogsHref(e) { return qs("logs", { campaign: e.campaign || "default", experiment: e.experiment_id }); }
  function copy(text, label) { if (window.copyText) copyText(text, label); }
  function go(href) { location.href = href; }

  async function cancelRun(j) {
    if (!await mlConfirm(`Cancel run ${j.label || j.run_id}?`, { confirmLabel: "Cancel run", cancelLabel: "Keep running" })) return;
    const r = await apiFetch(`/api/jobs/${encodeURIComponent(j.run_id)}/cancel?experiment_id=${encodeURIComponent(j.experiment_id)}`, { method: "POST" });
    if (r.ok) location.reload(); else mlAlert("Cancel failed: " + r.status);
  }
  async function retryRun(j) {
    if (!await mlConfirm(`Retry run ${j.label || j.run_id}?`, { confirmLabel: "Retry" })) return;
    const r = await apiFetch(`/api/jobs/${encodeURIComponent(j.run_id)}/retry?experiment_id=${encodeURIComponent(j.experiment_id)}`, { method: "POST" });
    if (r.ok) location.reload(); else mlAlert("Retry failed: " + r.status);
  }
  async function setExpStatus(e, status) {
    const label = status === "paused" ? "Pause" : status === "running" ? "Resume" : "Stop";
    if (!await mlConfirm(`${label} experiment ${e.name || e.experiment_id}?`, { confirmLabel: label })) return;
    const r = await apiFetch(`/api/experiments/${encodeURIComponent(e.experiment_id)}/status`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
    });
    if (r.ok) location.reload(); else mlAlert(`${label} failed: ` + r.status);
  }

  function runItem(j) {
    const e = expMap.get(j.experiment_id) || {};
    const combo = (() => { try { return Object.entries(JSON.parse(j.combo || "{}")).map(([k, v]) => `${k}=${v}`).join("  "); } catch { return ""; } })();
    const cmdText = (() => { try { return JSON.parse(j.command || "[]").join(" "); } catch { return j.command || ""; } })();
    return {
      kind: "run", id: `run:${j.run_id}`, title: j.label || j.run_id, sub: `${j.status} · ${j.experiment_name || j.experiment_id} · ${j.worker_id?.split(":")[0] || "unassigned"}${combo ? " · " + combo : ""}`,
      status: j.status, icon: ICONS.run,
      href: runHref(j),
      preview: `<div class="sp-title">${esc(j.label || j.run_id)}</div>
        <div class="sp-row"><span class="sp-k">Status</span>${esc(j.status || "unknown")}</div>
        <div class="sp-row"><span class="sp-k">Experiment</span>${esc(j.experiment_name || j.experiment_id)}</div>
        <div class="sp-row"><span class="sp-k">Worker</span>${esc(j.worker_id || "—")}</div>
        <div class="sp-row"><span class="sp-k">Combo</span>${esc(combo || "—")}</div>
        <div class="sp-cmd">${esc(cmdText.slice(0, 300))}</div>`,
      actions: [
        { label: "Highlight in Results", icon: "▥", run: () => go(runHref(j)) },
        { label: "Open logs", icon: "≡", run: () => go(logsHref(j)) },
        { label: "Open artifacts", icon: "▣", run: () => go(filesHref(j)) },
        { label: "Open code", icon: "‹›", run: () => go(codeHref(j)) },
        { label: "Copy run ID", icon: "⧉", run: () => copy(j.run_id, "Run ID copied") },
        { label: "Copy command", icon: "⌘", run: () => copy((() => { try { return JSON.parse(j.command || "[]").join(" "); } catch { return j.command || ""; } })(), "Command copied") },
        ...(["running", "dispatched", "pending"].includes(j.status) ? [{ label: "Cancel run", icon: "■", run: () => cancelRun(j) }] : []),
        ...(["failed", "cancelled", "xfailed"].includes(j.status) ? [{ label: "Retry run", icon: "↻", run: () => retryRun(j) }] : []),
      ],
    };
  }

  function expItem(e) {
    const c = e.job_counts || {};
    return {
      kind: "experiment", id: `exp:${e.experiment_id}`, title: e.name || e.experiment_id,
      sub: `${e.status || "unknown"} · ${e.campaign || "default"} · ${c.done || 0}/${c.total || e.expected_jobs || 0} finished`,
      status: e.status, icon: ICONS.experiment,
      href: expResultsHref(e),
      preview: `<div class="sp-title">${esc(e.name || e.experiment_id)}</div>
        <div class="sp-row"><span class="sp-k">Status</span>${esc(e.status || "unknown")}</div>
        <div class="sp-row"><span class="sp-k">Campaign</span>${esc(e.campaign || "default")}</div>
        <div class="sp-row"><span class="sp-k">Runs</span>${c.done || 0} done · ${c.failed || 0} failed · ${(c.running || 0) + (c.dispatched || 0)} active</div>
        <div class="sp-cmd">${esc(e.note || "No experiment note.")}</div>`,
      actions: [
        { label: "Open results", icon: "▥", run: () => go(expResultsHref(e)) },
        { label: "Open logs", icon: "≡", run: () => go(expLogsHref(e)) },
        { label: "Open artifacts", icon: "▣", run: () => go(expFilesHref(e)) },
        { label: "Open code", icon: "‹›", run: () => go(expCodeHref(e)) },
        { label: "Copy experiment ID", icon: "⧉", run: () => copy(e.experiment_id, "Experiment ID copied") },
        ...(e.status === "running" ? [{ label: "Pause experiment", icon: "❚❚", run: () => setExpStatus(e, "paused") }] : []),
        ...(e.status === "paused" ? [{ label: "Resume experiment", icon: "▶", run: () => setExpStatus(e, "running") }] : []),
        ...(["running", "paused"].includes(e.status) ? [{ label: "Stop experiment", icon: "■", run: () => setExpStatus(e, "aborted") }] : []),
      ],
    };
  }

  function noteItem(res) {
    const match = res.matches?.[0];
    return {
      kind: "note", id: `note:${res.path}`, title: res.path, sub: match ? `line ${match.line}: ${match.text.trim()}` : "note",
      icon: ICONS.note, href: `notes.html?path=${encodeURIComponent(res.path)}`,
      preview: `<div class="sp-title">${esc(res.path)}</div>${(res.matches || []).slice(0, 5).map(m => `<div class="sp-note"><span>${m.line}</span>${esc(m.text.trim())}</div>`).join("")}`,
      actions: [
        { label: "Open note", icon: "↗", run: () => go(`notes.html?path=${encodeURIComponent(res.path)}`) },
        { label: "Open at match", icon: "→", run: () => go(`notes.html?path=${encodeURIComponent(res.path)}`) },
        { label: "Copy path", icon: "⧉", run: () => copy(res.path, "Path copied") },
      ],
    };
  }

  function docItem(d) {
    return {
      kind: "doc", id: `doc:${d.id}`, title: d.title, sub: d.keys, icon: ICONS.doc,
      preview: `<div class="sp-title">${esc(d.title)}</div><div class="sp-body">${esc(d.body)}</div><pre class="sp-code">${esc(d.snippet)}</pre>`,
      actions: [
        { label: "Copy snippet", icon: "⧉", run: () => copy(d.snippet, "Snippet copied") },
        { label: "Copy with explanation", icon: "⌘", run: () => copy(`${d.body}\n\n${d.snippet}`, "Doc copied") },
      ],
    };
  }

  function artifactItem(j, path) {
    return {
      kind: "artifact", id: `artifact:${j.run_id}:${path}`, title: path,
      sub: `artifact · ${j.label || j.run_id}`, icon: ICONS.artifact,
      href: filesHref(j, path),
      preview: `<div class="sp-title">${esc(path)}</div><div class="sp-row"><span class="sp-k">Run</span>${esc(j.label || j.run_id)}</div><div class="sp-row"><span class="sp-k">Experiment</span>${esc(j.experiment_name || j.experiment_id)}</div>`,
      actions: [
        { label: "Open artifacts", icon: "▣", run: () => go(filesHref(j, path)) },
        { label: "Highlight in Results", icon: "▥", run: () => go(runHref(j)) },
        { label: "Copy path", icon: "⧉", run: () => copy(path, "Path copied") },
      ],
    };
  }

  function answerItem(title, sub, preview, actions = []) {
    return { kind: "answer", id: `answer:${title}`, title, sub, icon: ICONS.answer, preview, actions };
  }

  async function searchNotes(q) {
    if (!q || q.length < 3) return [];
    try {
      const r = await apiFetch(`/api/notes/search?q=${encodeURIComponent(q)}&limit=20`);
      if (!r.ok) return [];
      return (await r.json()).results || [];
    } catch { return []; }
  }

  async function searchArtifacts(p) {
    const term = (p.filters.file || "").toLowerCase();
    if (!term) return [];
    const ctx = context();
    if (!ctx.experiment) return [];
    const js = (await getAllJobs()).filter(j => j.experiment_id === ctx.experiment).slice(0, 24);
    const found = [];
    await Promise.all(js.map(async j => {
      try {
        const r = await apiFetch(`/api/experiments/${encodeURIComponent(j.experiment_id)}/jobs/${encodeURIComponent(j.run_id)}/artifacts`);
        if (!r.ok) return;
        for (const f of await r.json()) if (f.path.toLowerCase().includes(term)) found.push(artifactItem(j, f.path));
      } catch {}
    }));
    return found.slice(0, 20);
  }

  async function searchLogs(p) {
    const term = (p.filters.log || "").toLowerCase();
    if (!term) return [];
    const candidates = (await getAllJobs()).filter(j => ["failed", "running", "dispatched"].includes(j.status)).slice(0, 14);
    const results = [];
    await Promise.all(candidates.map(async j => {
      try {
        const r = await apiFetch(`/api/experiments/${encodeURIComponent(j.experiment_id)}/jobs/${encodeURIComponent(j.run_id)}/logs`);
        if (!r.ok) return;
        const text = await r.text();
        const line = text.split("\n").find(l => l.toLowerCase().includes(term));
        if (!line) return;
        results.push({ ...runItem(j), sub: `log match · ${line.trim().slice(0, 100)}`, preview: `<div class="sp-title">${esc(j.label || j.run_id)}</div><div class="sp-note">${esc(line.trim())}</div>` });
      } catch {}
    }));
    return results.slice(0, 10);
  }

  async function answers(p) {
    const out = [];
    const ctx = context();
    const allJobs = p.answer.best || p.answer.failed || p.answer.running || p.answer.stalled ? await getAllJobs() : [];
    if (p.answer.best) {
      const metric = p.answer.best;
      if (typeof DATA !== "undefined" && DATA && DATA.runs) {
        const vals = DATA.runs.map(r => {
          const s = (typeof METRIC_CACHE !== "undefined" && METRIC_CACHE[metric]?.[r.hash]) || {};
          const v = (s.values || []).filter(x => isFinite(x));
          return { r, v: v.length ? v[v.length - 1] : null };
        }).filter(x => x.v !== null);
        if (vals.length) {
          const minimize = typeof optMinimize !== "undefined" ? optMinimize : true;
          const best = vals.sort((a, b) => minimize ? a.v - b.v : b.v - a.v)[0];
          const shown = Number(best.v.toPrecision(6));
          out.push(answerItem(`Best ${metric}: ${shown}`, `${best.r.name} · current experiment`, `<div class="sp-title">Best ${esc(metric)}</div><div class="sp-row"><span class="sp-k">Run</span>${esc(best.r.name)}</div><div class="sp-row"><span class="sp-k">Value</span>${esc(String(shown))}</div>`, [
            { label: "Highlight in Results", icon: "▥", run: () => go(qs("results", { campaign: ctx.campaign, experiment: DATA.experiment, run: best.r.hash })) },
          ]));
        }
      }
    }
    if (p.answer.running) {
      const running = allJobs.filter(j => j.status === "running" || j.status === "dispatched");
      out.push(answerItem(`${running.length} running now`, running.slice(0, 3).map(j => j.label || j.run_id).join(" · "), `<div class="sp-title">${running.length} running runs</div>${running.slice(0, 6).map(j => `<div class="sp-row"><span class="sp-k">${esc(j.status)}</span>${esc(j.label || j.run_id)}</div>`).join("")}`, [
        { label: "Open System", icon: "⚙", run: () => go("system.html") },
      ]));
    }
    if (p.answer.failed) {
      const failed = allJobs.filter(j => ["failed", "xfailed"].includes(j.status));
      out.push(answerItem(`${failed.length} failed runs`, "Across all experiments", `<div class="sp-title">${failed.length} failed runs</div>${failed.slice(0, 6).map(j => `<div class="sp-row"><span class="sp-k">fail</span>${esc(j.label || j.run_id)}</div>`).join("")}`, [
        { label: "Open Experiments", icon: "▤", run: () => go("experiments.html") },
      ]));
    }
    if (p.answer.stalled) {
      const stalled = allJobs.filter(j => j.stalled);
      out.push(answerItem(`${stalled.length} stalled runs`, stalled.length ? "No recent progress" : "Nothing looks stuck", `<div class="sp-title">${stalled.length} stalled runs</div>${stalled.slice(0, 6).map(j => `<div class="sp-row"><span class="sp-k">stall</span>${esc(j.label || j.run_id)}</div>`).join("")}`, []));
    }
    if (p.answer.workers) {
      const ws = await getWorkers();
      const connected = ws.filter(w => w.status === "connected");
      out.push(answerItem(`${connected.length} connected workers`, `${connected.reduce((n, w) => n + (w.gpus || []).length, 0)} GPUs total`, `<div class="sp-title">Workers</div>${ws.map(w => `<div class="sp-row"><span class="sp-k">${esc(w.status)}</span>${esc(w.host)} · ${(w.gpus || []).length} GPUs</div>`).join("")}`, [{ label: "Open System", icon: "⚙", run: () => go("system.html") }]));
    }
    if (p.answer.changes) {
      const since = Date.now() - 86400e3;
      const recent = (await getExperiments()).filter(e => new Date(e.submit_time || 0).getTime() > since);
      const failed = (await getAllJobs()).filter(j => ["failed", "xfailed"].includes(j.status) && matchSince(j, since));
      out.push(answerItem(`Changes in the last 24h`, `${recent.length} experiments · ${failed.length} failures`, `<div class="sp-title">Last 24 hours</div><div class="sp-row"><span class="sp-k">new</span>${recent.length} experiments</div><div class="sp-row"><span class="sp-k">failed</span>${failed.length} runs</div>`, [{ label: "Open Experiments", icon: "▤", run: () => go("experiments.html") }]));
    }
    return out;
  }

  async function resolvePasted(raw) {
    const text = String(raw || "").trim();
    if (!/^https?:\/\//i.test(text)) return [];
    let u;
    try { u = new URL(text); } catch { return []; }
    if (!u.pathname.includes("/static/") && !u.pathname.match(/\.html$/)) return [];
    u.searchParams.delete("token");
    const page = u.pathname.split("/").pop().replace(/\.html$/, "");
    const exp = u.searchParams.get("experiment");
    const run = u.searchParams.get("run");
    if (exp) {
      await getExperiments();
      if (run) {
        const all = await getAllJobs();
        const j = all.find(x => x.experiment_id === exp && x.run_id === run);
        if (j) return [runItem(j)];
      }
      const e = expMap.get(exp);
      if (e) return [expItem(e)];
    }
    const path = u.searchParams.get("path");
    if (path) return [noteItem({ path, matches: [] })];
    return [{
      kind: "page", id: "pasted-link", title: "Open pasted link", sub: `${u.host}${u.pathname}`,
      icon: ICONS.page, href: u.pathname + u.search + u.hash,
      actions: [{ label: "Open link", icon: "↗", run: () => location.href = u.pathname + u.search + u.hash }],
    }];
  }

  async function search(raw, opts = {}) {
    const scope = opts.scope || "all";
    const p = parseQuery(raw);
    const t0 = performance.now();
    const pasted = await resolvePasted(raw);
    if (pasted.length) return { items: pasted, parsed: p, took: Math.round(performance.now() - t0), searchedJobs: 0 };
    const experiments = await getExperiments();
    const allJobs = (p.terms.length || p.filters.status || p.filters.campaign || p.filters.worker || p.filters.type || p.answer.best || p.answer.failed || p.answer.running)
      ? await getAllJobs() : [];
    const items = [];
    const ans = await answers(p);
    items.push(...ans);

    if (p.filters.file) {
      if (context().experiment) items.push(...await searchArtifacts(p));
      else items.push(answerItem("Artifact search", "Open a page for one experiment to search its artifacts", `<div class="sp-title">Artifact search</div><div class="sp-body">file: searches the artifacts of the experiment the current page shows. Open Results, Logs or Artifacts for an experiment first.</div>`, [
        { label: "Open Artifacts", icon: "▣", run: () => location.href = "files.html" },
      ]));
    }
    if (p.filters.log) items.push(...await searchLogs(p));

    const onlyNotes = !!p.filters.note;
    const onlyDocs = !!p.filters.docs;
    const onlyFiles = !!p.filters.file;
    const onlyLogs = !!p.filters.log;
    const onlyAnswer = !p.text && !onlyNotes && !onlyDocs && !onlyFiles && !onlyLogs && Object.keys(p.answer).length > 0;
    if (!onlyNotes && !onlyDocs && !onlyFiles && !onlyLogs && !onlyAnswer) {
      const jobsToShow = allJobs.filter(j => matchJob(j, p, scope)).sort((a, b) => (b.status === "running") - (a.status === "running") || new Date(b.submit_time || 0) - new Date(a.submit_time || 0));
      for (const j of jobsToShow.slice(0, opts.limit || 20)) items.push(runItem(j));

      const expsToShow = experiments.filter(e => matchExp(e, p, scope)).sort((a, b) => new Date(b.submit_time || 0) - new Date(a.submit_time || 0));
      for (const e of expsToShow.slice(0, opts.limit || 12)) items.push(expItem(e));
    }

    if (!onlyDocs && (!p.filters.type || ["note", "notes"].includes(p.filters.type.toLowerCase()))) {
      const notes = await searchNotes(p.filters.note || p.text);
      for (const n of notes.slice(0, opts.limit || 10)) items.push(noteItem(n));
    }

    if (!onlyNotes && (!p.filters.type || ["doc", "docs", "help"].includes(p.filters.type.toLowerCase()))) {
      const q = (p.filters.docs || p.text).toLowerCase();
      const docs = q ? DOCS.filter(d => `${d.title} ${d.keys}`.toLowerCase().includes(q) || q.split(/\s+/).every(t => `${d.title} ${d.keys} ${d.body}`.toLowerCase().includes(t))) : [];
      for (const d of docs.slice(0, 10)) items.push(docItem(d));
    }

    return {
      items,
      parsed: p,
      took: Math.round(performance.now() - t0),
      searchedJobs: allJobs.length,
    };
  }

  function savedSearches() { return MLPrefs.get("saved_searches", []); }
  function saveSearch(query, title, scope) {
    const all = savedSearches().filter(s => s.query !== query);
    all.unshift({ query, title: title || query, scope: scope || "all", at: Date.now() });
    MLPrefs.set("saved_searches", all.slice(0, 30));
  }
  function removeSearch(query) { MLPrefs.set("saved_searches", savedSearches().filter(s => s.query !== query)); }

  window.MLSearch = {
    search, parseQuery, searchNotes, savedSearches, saveSearch, removeSearch,
    getExperiments, getAllJobs, getWorkers, docs: DOCS, context,
    href: { run: runHref, logs: logsHref, files: filesHref, results: expResultsHref },
    icons: ICONS,
  };
})();
