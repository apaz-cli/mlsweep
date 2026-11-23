const params = new URLSearchParams(location.search);

const esc = s => escHtml(s ?? "");
const recall = name => { const v = MLPrefs.get(name); return v == null ? "" : String(v); };
function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }




// ── State ─────────────────────────────────────────────────────────────────────
const campaignSelect = document.getElementById("campaign-select");
const expSelect      = document.getElementById("exp-select");
const scopeSummary   = document.getElementById("scope-summary");
const tbDownload     = document.getElementById("tb-download");
const tbCmd          = document.getElementById("tb-cmd");
const wbProject      = document.getElementById("wb-project");
const wbEntity       = document.getElementById("wb-entity");
const wbKey          = document.getElementById("wb-key");
const wbSubmit       = document.getElementById("wb-submit");
const wbError        = document.getElementById("wandb-error");
const exportList     = document.getElementById("export-list");

const SCOPES = ["experiment", "campaign", "all"];
let scope = params.get("experiment") ? "experiment"
          : SCOPES.includes(recall("export_scope")) ? recall("export_scope") : "experiment";
let allExps = [];
let currentExpId = null;
let onlyDone = false;
let wandbAvailable = false;
let pollTimer = null;
let firstLoad = true;

wbProject.value = recall("wandb_project");
wbEntity.value  = recall("wandb_entity");

// ── Scope ─────────────────────────────────────────────────────────────────────
// The experiments the current scope covers.
function scopeExps() {
  if (scope === "experiment") return allExps.filter(e => e.experiment_id === currentExpId);
  if (scope === "campaign") return allExps.filter(e => MLCampaign.campaignOf(e) === MLCampaign.get());
  return allExps;
}

// Query parameters that select the scope on the export endpoints.
function scopeQuery() {
  if (scope === "experiment") return { experiment_id: currentExpId };
  if (scope === "campaign") return { campaign: MLCampaign.get() };
  return {};
}

function scopeReady() {
  return scope !== "experiment" || !!currentExpId;
}

function zipName() {
  if (scope === "experiment") return `${(currentExpId || "experiment").slice(0, 16)}-tensorboard.zip`;
  if (scope === "campaign") return `${MLCampaign.get()}-tensorboard.zip`;
  return "mlsweep-tensorboard.zip";
}

// The campaign select lists real campaigns only in campaign scope; "All
// campaigns" there would mean the Everything scope.
function fillCampaigns() {
  if (scope === "campaign" && MLCampaign.get() === MLCampaign.ALL) {
    const newest = MLCampaign.names(allExps)[0];
    if (newest) MLCampaign.set(newest);
  }
  MLCampaign.fill(campaignSelect, allExps, scope !== "campaign");
}

function showScope(prefer) {
  MLPrefs.set("export_scope", scope);
  document.querySelectorAll("#scope-toggle button").forEach(b => b.classList.toggle("active", b.dataset.scope === scope));
  document.getElementById("campaign-field").classList.toggle("hidden", scope === "all");
  document.getElementById("experiment-field").classList.toggle("hidden", scope !== "experiment");
  fillCampaigns();
  if (scope === "experiment") {
    currentExpId = MLCampaign.fillExperiments(expSelect, MLCampaign.filter(allExps), prefer, currentExpId);
  }
  render();
  pollExports();
}

function render() {
  const exps = scopeExps();
  const total = exps.reduce((n, e) => n + (onlyDone ? e.job_counts.done : e.job_counts.total), 0);
  const runs = plural(total, onlyDone ? "succeeded run" : "run");
  const nCampaigns = new Set(exps.map(MLCampaign.campaignOf)).size;
  let where;
  if (scope === "experiment") where = exps.length ? `experiment <b>${esc(exps[0].name || exps[0].experiment_id)}</b>` : null;
  else if (scope === "campaign") where = `${plural(exps.length, "experiment")} in campaign <b>${esc(MLCampaign.get())}</b>`;
  else where = `${plural(exps.length, "experiment")} across ${plural(nCampaigns, "campaign")}`;
  scopeSummary.innerHTML = where ? `Exporting <b>${esc(runs)}</b> from ${where}.` : "No experiment selected.";

  const ready = scopeReady() && total > 0;
  const q = { ...scopeQuery(), ...(onlyDone ? { only_done: "1" } : {}) };
  tbDownload.classList.toggle("disabled", !ready);
  tbDownload.href = ready ? authURL(pageHref("/api/export/tensorboard.zip", q)) : "#";
  tbCmd.textContent = `unzip ${zipName()} -d tb_logs\ntensorboard --logdir tb_logs`;

  wbSubmit.disabled = !ready || !wandbAvailable || !wbProject.value.trim();
  wbSubmit.textContent = total ? `Upload ${runs} to W&B` : "Upload to W&B";

  const single = scope === "experiment" && currentExpId;
  const enc = single ? encodeURIComponent(currentExpId) : "";
  for (const [elId, path] of [["dl-metrics", "metrics.zip"], ["dl-logs", "logs.zip"],
                              ["dl-artifacts", "artifacts.zip"], ["dl-all", "download"]]) {
    document.getElementById(elId).href = single ? authURL(`/api/experiments/${enc}/${path}`) : "#";
  }
  document.getElementById("raw-list").classList.toggle("hidden", !single);
  document.getElementById("raw-note").classList.toggle("hidden", scope === "experiment");
}

async function loadExperiments() {
  try {
    const r = await apiFetch("/api/experiments");
    if (!r.ok) { setConnStatus("disconnected"); return; }
    allExps = await r.json();
    setConnStatus("connected");
    const urlExp = firstLoad ? params.get("experiment") : null;
    firstLoad = false;
    MLCampaign.follow(allExps, urlExp);
    showScope(urlExp);
  } catch {
    setConnStatus("disconnected");
  }
}

MLCampaign.bind(campaignSelect, () => showScope(null));
expSelect.addEventListener("change", () => { currentExpId = expSelect.value; render(); });
document.querySelectorAll("#scope-toggle button").forEach(b => {
  b.addEventListener("click", () => { scope = b.dataset.scope; showScope(null); });
});
document.querySelectorAll("#runs-toggle button").forEach(b => {
  b.addEventListener("click", () => {
    onlyDone = b.dataset.onlyDone === "1";
    document.querySelectorAll("#runs-toggle button").forEach(x => x.classList.toggle("active", x === b));
    render();
  });
});
// Job counts change as sweeps run.  Refresh them without rebuilding the
// selects, which would close one the user has open.
setInterval(async () => {
  try {
    const r = await apiFetch("/api/experiments");
    if (r.ok) { allExps = await r.json(); render(); }
  } catch {}
}, 15000);

// ── W&B ───────────────────────────────────────────────────────────────────────
async function loadWandbInfo() {
  try {
    const r = await apiFetch("/api/export/wandb");
    if (!r.ok) return;
    const info = await r.json();
    wandbAvailable = info.available;
    document.getElementById("wandb-missing").classList.toggle("hidden", info.available);
    document.getElementById("wandb-version").textContent =
      info.available ? `wandb ${info.version || ""}`.trim() : "not installed";
    if (info.env_key) {
      document.getElementById("wb-key-hint").innerHTML =
        "Leave blank to use the <code>WANDB_API_KEY</code> the manager was started with. " +
        "The key is not stored, but it is sent in the request; use HTTPS if the dashboard is remote.";
    }
    for (const el of [wbProject, wbEntity, wbKey]) el.disabled = !info.available;
    render();
  } catch {}
}

wbProject.addEventListener("input", render);

wbSubmit.addEventListener("click", async () => {
  if (!scopeReady()) return;
  const project = wbProject.value.trim();
  const entity  = wbEntity.value.trim();
  MLPrefs.set("wandb_project", project);
  MLPrefs.set("wandb_entity", entity);
  wbError.classList.add("hidden");
  wbSubmit.disabled = true;
  try {
    const r = await apiFetch(pageHref("/api/export/wandb", scopeQuery()), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project, entity: entity || null, api_key: wbKey.value.trim() || null, only_done: onlyDone }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    wbKey.value = "";
    pollExports();
  } catch (e) {
    wbError.textContent = e.message;
    wbError.classList.remove("hidden");
  } finally {
    render();
  }
});

let lastExports = [];

function renderExports(exports) {
  lastExports = exports;
  document.getElementById("export-list-title").classList.toggle("hidden", !exports.length);
  exportList.innerHTML = exports.map(e => {
    const dest = (e.entity ? e.entity + "/" : "") + e.project;
    const pct = e.total ? Math.round(100 * e.done / e.total) : 0;
    const many = e.scope !== "experiment";
    const runs = e.runs.map((r, ri) => [r, ri]).filter(([r]) => r.status !== "pending").map(([r, ri]) => {
      const detail = r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener">open</a>`
                   : r.error ? `<span class="err">${esc(r.error)}</span>`
                   : `<span>${esc(r.status)}</span>`;
      const name = many ? `${r.experiment} / ${r.run_id}` : r.run_id;
      return `<div class="export-run" data-run="${ri}"><span class="rid" title="${esc(name)}">${esc(name)}</span>${detail}</div>`;
    }).join("");
    return `<div class="export-item" data-export="${esc(e.export_id)}">
      <div class="export-top">
        <span class="badge badge-${esc(e.status)}">${esc(e.status)}</span>
        <span class="export-dest" title="${esc(e.scope_label)} → ${esc(dest)}">${esc(e.scope_label)} → <code>${esc(dest)}</code></span>
        <span class="card-note">${e.done}/${e.total}</span>
        ${e.status === "running" ? `<button class="btn-sm" data-cancel="${esc(e.export_id)}">Cancel</button>` : ""}
      </div>
      <div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div>
      ${e.error ? `<div class="error">${esc(e.error)}</div>` : ""}
      ${runs ? `<details${e.status !== "done" ? " open" : ""}><summary>Runs</summary><div class="export-runs">${runs}</div></details>` : ""}
    </div>`;
  }).join("");
}

exportList.addEventListener("click", async ev => {
  const id = ev.target.dataset && ev.target.dataset.cancel;
  if (!id) return;
  ev.target.disabled = true;
  await apiFetch(`/api/exports/${encodeURIComponent(id)}/cancel`, { method: "POST" }).catch(() => {});
  pollExports();
});

async function cancelExport(id) {
  await apiFetch(`/api/exports/${encodeURIComponent(id)}/cancel`, { method: "POST" }).catch(() => {});
  pollExports();
}

// Right-click an upload: cancel it or copy where it went.  On one of its
// runs: open that run's upload, logs, or copy its ID or error.
exportList.addEventListener("contextmenu", ev => {
  const item = ev.target.closest(".export-item");
  const e = item && lastExports.find(x => x.export_id === item.dataset.export);
  if (!e || mlNativeMenuWanted(ev, item)) return;
  const dest = (e.entity ? e.entity + "/" : "") + e.project;
  const runEl = ev.target.closest(".export-run");
  const r = runEl ? e.runs[+runEl.dataset.run] : null;
  const exp = r ? r.experiment_id : null;
  mlContextMenu(ev, [
    r ? { title: r.run_id } : null,
    r?.url ? { label: "Open in W&B", run: () => window.open(r.url, "_blank", "noopener") } : null,
    r && exp ? { label: "Open logs", run: () => { location.href = pageHref("logs.html", { experiment: exp, run: r.run_id }); } } : null,
    r ? { label: "Copy run ID", run: () => copyText(r.run_id, "Copied the run ID") } : null,
    r?.url ? { label: "Copy W&B link", run: () => copyText(r.url, "Copied the link") } : null,
    r?.error ? { label: "Copy error", run: () => copyText(r.error, "Copied the error") } : null,
    r ? "-" : null,
    { title: e.scope_label },
    e.status === "running" ? { label: "Cancel upload", danger: true, run: () => cancelExport(e.export_id) } : null,
    { label: "Copy destination", hint: dest, run: () => copyText(dest, "Copied the destination") },
    e.error ? { label: "Copy error", run: () => copyText(e.error, "Copied the error") } : null,
  ]);
});

// Every recent upload is listed, whatever the scope.  Poll every 2s while
// one is running; stop once all have finished.
async function pollExports() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  try {
    const r = await apiFetch("/api/exports");
    if (!r.ok) return;
    const exports = await r.json();
    // Keep a <details> the user toggled open or shut across re-renders.
    const open = [...exportList.querySelectorAll("details")].map(d => d.open);
    renderExports(exports);
    exportList.querySelectorAll("details").forEach((d, i) => { if (i < open.length) d.open = open[i]; });
    if (exports.some(e => e.status === "running")) pollTimer = setTimeout(pollExports, 2000);
  } catch {}
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadWandbInfo();
loadExperiments();
