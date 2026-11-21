// ── Helpers ───────────────────────────────────────────────────────────────────
// ── Connect worker modal ──────────────────────────────────────────────────────
(function() {
  const modal     = document.getElementById("connect-modal");
  const errEl     = document.getElementById("connect-error");
  const submit    = document.getElementById("connect-submit");
  const knownList = document.getElementById("known-workers-list");
  const divider   = document.getElementById("connect-divider");

  // Workers fetched on open; keyed by worker_id for reconnect lookups.
  let _knownWorkers = [];

  function clearForm() {
    ["fw-host","fw-remote-dir","fw-venv","fw-ssh-key","fw-devices"].forEach(id => {
      document.getElementById(id).value = "";
    });
    errEl.textContent = "";
    errEl.classList.add("hidden");
  }

  async function openModal() {
    clearForm();
    knownList.innerHTML = "";
    knownList.classList.add("hidden");
    divider.classList.add("hidden");

    modal.classList.remove("hidden");
    document.getElementById("fw-host").focus();

    // Fetch all known workers; show inactive ones as quick-reconnect options.
    try {
      const res = await apiFetch("/api/workers");
      if (res.ok) {
        const workers = await res.json();
        _knownWorkers = workers;
        const inactive = workers.filter(w => w.status !== "connected");
        if (inactive.length) {
          knownList.innerHTML = inactive.map(w => `
            <div class="known-worker-row">
              <div class="known-worker-info">
                <span class="known-worker-host">${escHtml(w.host)}</span>
                <span class="known-worker-dir">${escHtml(w.remote_dir)}</span>
              </div>
              <button class="btn-reconnect" data-worker-id="${escHtml(w.worker_id)}">Reconnect</button>
            </div>`).join("");
          knownList.classList.remove("hidden");
          divider.classList.remove("hidden");
        }
      }
    } catch (_) {}
  }

  function closeModal() { modal.classList.add("hidden"); }

  // Reconnect a previously registered worker with one click.
  knownList.addEventListener("click", async e => {
    const btn = e.target.closest(".btn-reconnect");
    if (!btn) return;
    const workerId = btn.dataset.workerId;
    const w = _knownWorkers.find(x => x.worker_id === workerId);
    if (!w) return;

    btn.disabled = true;
    btn.textContent = "Connecting…";
    errEl.classList.add("hidden");

    const devices = w.gpus && w.gpus.length ? w.gpus : null;

    try {
      const res = await apiFetch("/api/workers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          worker_id: w.worker_id,
          host: w.host, remote_dir: w.remote_dir,
          venv: w.venv || null, ssh_key: w.ssh_key || null,
          devices,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        errEl.textContent = data.error || `Error ${res.status}`;
        errEl.classList.remove("hidden");
        btn.disabled = false;
        btn.textContent = "Reconnect";
        return;
      }
      closeModal();
      refresh();
    } catch (e) {
      errEl.textContent = "Network error: " + e.message;
      errEl.classList.remove("hidden");
      btn.disabled = false;
      btn.textContent = "Reconnect";
    }
  });

  document.getElementById("btn-connect-worker").addEventListener("click", openModal);
  document.getElementById("connect-cancel").addEventListener("click", closeModal);
  modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });

  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) closeModal();
  });

  submit.addEventListener("click", async () => {
    const host       = document.getElementById("fw-host").value.trim();
    const remote_dir = document.getElementById("fw-remote-dir").value.trim();
    const venv       = document.getElementById("fw-venv").value.trim() || null;
    const ssh_key    = document.getElementById("fw-ssh-key").value.trim() || null;
    const devStr     = document.getElementById("fw-devices").value.trim();
    const devices    = devStr ? devStr.split(",").map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n)) : null;

    if (!host) {
      errEl.textContent = "Host is required.";
      errEl.classList.remove("hidden");
      return;
    }

    submit.disabled = true;
    submit.textContent = "Connecting…";
    errEl.classList.add("hidden");

    try {
      const res = await apiFetch("/api/workers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host, remote_dir, venv, ssh_key, devices }),
      });
      const data = await res.json();
      if (!res.ok) {
        errEl.textContent = data.error || `Error ${res.status}`;
        errEl.classList.remove("hidden");
        return;
      }
      closeModal();
      refresh();
    } catch (e) {
      errEl.textContent = "Network error: " + e.message;
      errEl.classList.remove("hidden");
    } finally {
      submit.disabled = false;
      submit.textContent = "Connect";
    }
  });
})();

// ── Disconnect worker ─────────────────────────────────────────────────────────
async function disconnectWorker(workerId) {
  if (!await mlConfirm(`Disconnect worker "${workerId}"?\nIn-flight jobs will be re-queued.`, { confirmLabel: "Disconnect" })) return;
  try {
    const res = await apiFetch(`/api/workers/${encodeURIComponent(workerId)}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      await mlAlert("Failed to disconnect: " + (data.error || res.status));
      return;
    }
    refresh();
  } catch (e) {
    await mlAlert("Network error: " + e.message);
  }
}

// ── Update an outdated worker ────────────────────────────────────────────────
async function updateWorker(workerId, isLive) {
  if (isLive && !await mlConfirm(`Update worker "${workerId}"?\nIt restarts, and its in-flight jobs will be re-queued.`, { confirmLabel: "Update" })) return;
  this.disabled = true;
  this.textContent = "Updating…";
  try {
    const res = await apiFetch(`/api/workers/${encodeURIComponent(workerId)}/update`, { method: "POST" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      await mlAlert("Failed to update: " + (data.error || res.status));
    }
  } catch (e) {
    await mlAlert("Network error: " + e.message);
  }
  refresh();
}

// ── Worker live controls ───────────────────────────────────────────────────────
async function setWorkerConcurrency(workerId, value) {
  try {
    const res = await apiFetch(`/api/workers/${encodeURIComponent(workerId)}/concurrency`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_jobs_per_gpu: value }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      await mlAlert("Failed: " + (data.error || res.status));
    }
  } catch (e) {
    await mlAlert("Network error: " + e.message);
  }
}

// ── Scheduler placement ────────────────────────────────────────────────────────
function renderPlacement(placement) {
  for (const btn of document.querySelectorAll("#placement-toggle button")) {
    btn.classList.toggle("active", btn.dataset.placement === placement);
  }
}

async function setPlacement(placement) {
  try {
    const res = await apiFetch("/api/scheduler", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ placement }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      await mlAlert("Failed: " + (data.error || res.status));
      return;
    }
    renderPlacement((await res.json()).placement);
  } catch (e) {
    await mlAlert("Network error: " + e.message);
  }
}

for (const btn of document.querySelectorAll("#placement-toggle button")) {
  btn.addEventListener("click", () => setPlacement(btn.dataset.placement));
}

// ── Workers: GPU pools ────────────────────────────────────────────────────────
// Each card shows every GPU the worker reports.  A GPU is in the pool (the
// scheduler may place runs there) or out of it; an out-of-pool GPU whose runs
// are still going is "draining".  Runs are chips on their GPU: drag one onto
// another GPU, on any worker, to restart it there, or open its menu.  Click
// GPUs to select them, then use the bar under them to drain, remove or return
// several at once.

let lastWorkers = [];
const selectedGpus = new Map();   // worker_id → Set of GPU indices
let dragging = null;              // {expId, runId} while a run chip is dragged
let pendingRender = false;

function apiJSON(method, path, body) {
  return apiFetch(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      return data;
    });
}

function gpuState(w, g) {
  const inPool = (w.gpus || []).includes(g);
  const runs = (w.gpu_runs || {})[g] || [];
  if ((w.unhealthy_gpus || []).includes(g)) return "unhealthy";
  if (inPool) return "pool";
  return runs.length ? "draining" : "out";
}

const STATE_LABEL = { pool: "in pool", draining: "draining", out: "out of pool", unhealthy: "unhealthy" };

function allGpusOf(w) {
  const ids = new Set([...(w.gpus || []), ...(w.unhealthy_gpus || [])]);
  for (const k of Object.keys(w.gpu_stats || {})) ids.add(+k);
  for (const k of Object.keys(w.gpu_runs || {})) ids.add(+k);
  return [...ids].sort((a, b) => a - b);
}

function jobOf(expId, runId) {
  return inFlight.find(j => j.experiment_id === expId && j.run_id === runId);
}

function elapsedSince(iso) {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  return fmtElapsed(s);
}

function runChip(w, r) {
  const j = jobOf(r.experiment_id, r.run_id);
  const name = j ? runNameHtml(j) : escHtml(r.run_id);
  const live = w.status === "connected";
  return `<div class="run-chip${j?.status === "dispatched" ? " starting" : ""}" data-act="" ${live ? 'draggable="true"' : ""}
      data-exp="${escHtml(r.experiment_id)}" data-run="${escHtml(r.run_id)}"
      title="${escHtml(r.run_id)} · ${escHtml(r.experiment_id)}${live ? "\nDrag onto another GPU to restart it there" : ""}">
    <span class="rc-name">${name}</span>
    <span class="rc-meta">${escHtml(r.experiment_id)}${j?.start_time ? " · " + elapsedSince(j.start_time) : j?.status === "dispatched" ? " · starting" : ""}</span>
    <button class="rc-menu btn-menu" aria-haspopup="true" title="Move, requeue, logs" ${act("toggleChipMenu", w.worker_id, r.experiment_id, r.run_id)}>⋯</button>
  </div>`;
}

// "NVIDIA GeForce RTX 5090" → "RTX 5090"; the full name is in the tooltip.
function shortGpuName(name) {
  return name.replace(/^(NVIDIA|AMD)\s+/i, "").replace(/^(GeForce|Radeon)\s+/i, "").trim();
}

function gpuTile(w, g) {
  const st = gpuState(w, g);
  const runs = (w.gpu_runs || {})[g] || [];
  const stat = (w.gpu_stats || {})[g] || (w.gpu_stats || {})[String(g)];
  const util = stat?.util_pct ?? null;
  const memUsed = stat?.mem_used_mb ?? -1, memTotal = stat?.mem_total_mb ?? -1;
  const memPct = memTotal > 0 ? Math.round(100 * memUsed / memTotal) : 0;
  const cap = w.max_jobs_per_gpu || 0;
  const pins = (w.pins || []).filter(p => p.gpus.includes(g));
  const live = w.status === "connected";
  const sel = selectedGpus.get(w.worker_id)?.has(g);
  const slots = st === "pool"
    ? (cap ? `<span class="gt-slots${runs.length > cap ? " over" : runs.length === cap ? " full" : ""}" title="${runs.length} of ${cap} jobs on this GPU${runs.length > cap ? " (over the limit: a run was moved here by hand)" : ""}">${runs.length}/${cap}</span>`
           : `<span class="gt-slots${runs.length ? " full" : ""}" title="No per-GPU limit">${runs.length}/∞</span>`)
    : "";
  const toggle = live && st !== "unhealthy"
    ? `<label class="gt-switch" title="${st === "pool" ? "Take out of the pool; its runs keep going" : "Put back in the pool"}" data-act="">
         <input type="checkbox" ${st === "pool" ? "checked" : ""} ${act("toggleGpuPool", w.worker_id, g, st === "pool")}><span></span></label>`
    : "";
  return `<div class="gpu-tile st-${st}${sel ? " selected" : ""}" data-worker="${escHtml(w.worker_id)}" data-gpu="${g}"
      ${live && st !== "unhealthy" ? act("toggleGpuSelect", w.worker_id, g) : ""}>
    <div class="gt-head">
      <span class="gt-id">GPU ${g}</span>
      ${stat?.name ? `<span class="gt-name" title="${escHtml(stat.name)}">${escHtml(shortGpuName(stat.name))}</span>` : ""}
      <span class="gt-state">${STATE_LABEL[st]}</span>
      ${slots}
      ${toggle}
    </div>
    <div class="gt-bars">
      <span class="gt-label">%util</span>
      <span class="gt-pct">${util !== null ? util + "%" : "—"}</span>
      <span class="gt-amt"></span>
      <div class="gt-bar"><i style="width:${util ?? 0}%"></i></div>
      <span class="gt-label">%mem</span>
      <span class="gt-pct">${memTotal > 0 ? memPct + "%" : "—"}</span>
      <span class="gt-amt">${memTotal > 0 ? `${(memUsed / 1024).toFixed(1)}/${Math.round(memTotal / 1024)}G` : ""}</span>
      <div class="gt-bar mem"><i style="width:${memPct}%"></i></div>
    </div>
    <div class="gt-runs">
      ${runs.map(r => runChip(w, r)).join("")}
      ${pins.map(p => `<div class="run-chip pinned" title="Starts here next">
          <span class="rc-name">${escHtml(p.run_id)}</span><span class="rc-meta">next here</span></div>`).join("")}
      ${!runs.length && !pins.length ? `<div class="gt-empty">${st === "pool" ? "idle" : st === "out" ? "not used" : ""}</div>` : ""}
    </div>
  </div>`;
}

function selectionBar(w) {
  const sel = [...(selectedGpus.get(w.worker_id) || [])].filter(g => allGpusOf(w).includes(g)).sort((a, b) => a - b);
  if (!sel.length) return "";
  const states = sel.map(g => gpuState(w, g));
  const runsOn = sel.reduce((n, g) => n + ((w.gpu_runs || {})[g] || []).length, 0);
  const anyIn = states.includes("pool"), anyOut = states.some(s => s === "out" || s === "draining");
  return `<div class="sel-bar">
    <span>GPU${sel.length > 1 ? "s" : ""} ${sel.join(", ")}</span>
    ${anyIn ? `<button class="btn-link" title="Out of the pool; runs on them finish first" ${act("applyGpus", w.worker_id, "drain", sel)}>Drain</button>` : ""}
    ${anyIn || runsOn ? `<button class="btn-link danger" title="Out of the pool now; their runs restart elsewhere" ${act("applyGpus", w.worker_id, "remove", sel)}>Remove now${runsOn ? ` (requeue ${runsOn})` : ""}</button>` : ""}
    ${anyOut ? `<button class="btn-link" ${act("applyGpus", w.worker_id, "add", sel)}>Return to pool</button>` : ""}
    <button class="btn-link" ${act("clearGpuSelect", w.worker_id)}>Clear</button>
  </div>`;
}

function poolSummary(w) {
  const all = allGpusOf(w);
  const n = { pool: 0, draining: 0, out: 0, unhealthy: 0 };
  for (const g of all) n[gpuState(w, g)]++;
  const runs = new Set(Object.values(w.gpu_runs || {}).flat().map(r => r.experiment_id + "/" + r.run_id)).size;
  const parts = [`${n.pool} of ${all.length} GPU${all.length === 1 ? "" : "s"} in pool`];
  if (n.draining) parts.push(`${n.draining} draining`);
  if (n.unhealthy) parts.push(`${n.unhealthy} unhealthy`);
  parts.push(`${runs} run${runs === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

function renderWorkers(workers) {
  lastWorkers = workers;
  const grid = document.getElementById("workers-grid");
  // Don't redraw under a drag, an open menu or an input being edited.
  const active = document.activeElement;
  if (dragging || (rowMenu && grid.contains(document.querySelector(".run-chip.menu-open")))
      || (active && active.tagName === "INPUT" && active.type !== "checkbox" && grid.contains(active))) {
    pendingRender = true;
    return;
  }
  pendingRender = false;
  if (!workers.length) {
    grid.innerHTML = `<div class="no-data">No workers connected.</div>`;
    return;
  }

  grid.innerHTML = workers.map(w => {
    const status   = w.status || "offline";
    const isLive   = status === "connected";
    const canDisconnect = status !== "dead" && status !== "updating";
    const canUpdate     = w.outdated && (status === "dead" || isLive);
    const maxJobs  = w.max_jobs_per_gpu || 0;
    const gpus = allGpusOf(w);

    return `<div class="worker-card${isLive ? "" : " offline"}">
      <div class="worker-header">
        <span class="worker-host">${escHtml(w.host)}${w.port ? `:${escHtml(w.port)}` : ""}</span>
        ${badge(status)}
        <span class="worker-summary">${poolSummary(w)}</span>
        ${isLive ? `<label class="jobs-per-gpu" title="Most runs the scheduler puts on one GPU (0 = no limit)">jobs/GPU
          <input class="worker-control-input worker-control-narrow" type="number" min="0" value="${escHtml(maxJobs)}"
            ${act("workerConcurrencyChanged", w.worker_id)}></label>` : ""}
        <button class="btn-link btn-manage" aria-haspopup="true" title="Pool, update, disconnect" ${act("toggleWorkerMenu", w.worker_id)}>⋯</button>
      </div>
      ${w.last_error ? `<div class="worker-error">${escHtml(w.last_error)}</div>` : ""}
      ${w.outdated && isLive ? `<div class="worker-warning">Runs older mlsweep code than the manager.</div>` : ""}
      ${gpus.length ? `<div class="gpu-grid">${gpus.map(g => gpuTile(w, g)).join("")}</div>` : `<div class="no-data">No GPUs reported.</div>`}
      ${isLive ? selectionBar(w) : ""}
      <div class="worker-actions">
        <div class="worker-detail">
          ${w.remote_dir  ? `<div><span>dir</span> ${escHtml(w.remote_dir)}</div>` : ""}
          ${w.venv        ? `<div><span>venv</span> ${escHtml(w.venv)}</div>` : ""}
          ${w.ssh_key     ? `<div><span>key</span> ${escHtml(w.ssh_key)}</div>` : ""}
          ${w.scratch_dir ? `<div><span>scratch</span> ${escHtml(w.scratch_dir)}</div>` : ""}
        </div>
        <span class="worker-actions-buttons">
          <span class="worker-last-seen">${w.last_seen ? relTime(w.last_seen) : ""}</span>
          ${canUpdate ? `<button class="btn-update" title="Install this manager's mlsweep on ${escHtml(w.host)} and restart the worker" ${act("updateWorker", w.worker_id, isLive)}>Update worker</button>` : ""}
          ${canDisconnect ? `<button class="btn-disconnect" ${act("disconnectWorker", w.worker_id)}>Disconnect</button>` : ""}
        </span>
      </div>
    </div>`;
  }).join("");
}

// ── GPU pool actions ──────────────────────────────────────────────────────────

async function patchDevices(workerId, body) {
  try { await apiJSON("PATCH", `/api/workers/${encodeURIComponent(workerId)}/devices`, body); }
  catch (e) { await mlAlert("Failed: " + e.message); }
  await refresh();
}

function toggleGpuSelect(workerId, g) {
  let set = selectedGpus.get(workerId);
  if (!set) selectedGpus.set(workerId, set = new Set());
  if (set.has(g)) set.delete(g); else set.add(g);
  renderWorkers(lastWorkers);
}

function clearGpuSelect(workerId) {
  selectedGpus.delete(workerId);
  renderWorkers(lastWorkers);
}

// The switch on a tile: out of the pool lets runs finish (nothing is killed).
function toggleGpuPool(workerId, g, inPool) {
  patchDevices(workerId, inPool ? { drain: [g] } : { add: [g] });
}

async function applyGpus(workerId, how, gpus) {
  const w = lastWorkers.find(x => x.worker_id === workerId);
  if (how === "remove") {
    const n = gpus.reduce((k, g) => k + ((w?.gpu_runs || {})[g] || []).length, 0);
    if (n && !await mlConfirm(`Requeue the ${n} run${n === 1 ? "" : "s"} on GPU ${gpus.join(", ")}?\n`
        + "They stop now and restart on other GPUs, from scratch unless the script resumes from a checkpoint.",
        { confirmLabel: "Remove and requeue" })) return;
  }
  selectedGpus.delete(workerId);
  await patchDevices(workerId, { [how]: gpus });
}

function toggleWorkerMenu(workerId) {
  const w = lastWorkers.find(x => x.worker_id === workerId);
  const all = w ? allGpusOf(w) : [];
  const live = w?.status === "connected";
  const items = [];
  if (live) {
    const inPool = all.filter(g => gpuState(w, g) === "pool");
    const outPool = all.filter(g => ["out", "draining"].includes(gpuState(w, g)));
    if (inPool.length) items.push({ label: "Drain all GPUs", title: "Nothing new starts here; runs finish", act: act("applyGpus", workerId, "drain", inPool) });
    if (outPool.length) items.push({ label: "Return all GPUs to the pool", act: act("applyGpus", workerId, "add", outPool) });
  }
  toggleMenu(this, `worker:${workerId}`, items.length ? items : [{ label: "No pool changes for an offline worker", disabled: true }]);
}

// ── Moving runs ───────────────────────────────────────────────────────────────

async function reschedule(expId, runId, workerId, gpus) {
  const body = workerId ? { worker_id: workerId, ...(gpus ? { gpus } : {}) } : {};
  try {
    await apiJSON("POST", `/api/jobs/${encodeURIComponent(runId)}/reschedule?experiment_id=${encodeURIComponent(expId)}`, body);
  } catch (e) { await mlAlert("Could not move the run: " + e.message); }
  await refresh();
}

async function moveRunTo(expId, runId, workerId, g) {
  const j = jobOf(expId, runId);
  const w = lastWorkers.find(x => x.worker_id === workerId);
  const there = ((w?.gpu_runs || {})[g] || []).length;
  const cap = w?.max_jobs_per_gpu || 0;
  const share = cap && there >= cap ? `\nGPU ${g} already has ${there} run${there === 1 ? "" : "s"}; this one will share it.` : "";
  if (!await mlConfirm(`Restart ${j?.label || runId} on ${workerId} GPU ${g}?\n`
      + `It stops now and starts again there, from scratch unless the script resumes from a checkpoint.${share}`,
      { confirmLabel: "Move run" })) return;
  await reschedule(expId, runId, workerId, [g]);
}

async function requeueRun(expId, runId) {
  const j = jobOf(expId, runId);
  if (!await mlConfirm(`Requeue ${j?.label || runId}?\nIt stops now and the scheduler starts it again wherever there is room.`,
      { confirmLabel: "Requeue" })) return;
  await reschedule(expId, runId, null, null);
}

// A worker's short name: its host, or its full ID when two workers share a host.
function workerName(w) {
  return lastWorkers.filter(x => x.host === w.host).length > 1 ? w.worker_id : w.host;
}

function toggleChipMenu(workerId, expId, runId) {
  const targets = [];
  for (const w of lastWorkers) {
    if (w.status !== "connected") continue;
    for (const g of w.gpus || []) {
      const on = ((w.gpu_runs || {})[g] || []);
      if (w.worker_id === workerId && on.some(r => r.run_id === runId && r.experiment_id === expId)) continue;
      const cap = w.max_jobs_per_gpu || 0;
      targets.push({
        label: `Move to ${workerName(w)} GPU ${g}`, count: cap ? `${on.length}/${cap}` : `${on.length}`,
        act: act("moveRunTo", expId, runId, w.worker_id, g),
      });
    }
  }
  const chip = this.closest(".run-chip");
  chip?.classList.add("menu-open");
  toggleMenu(this, `chip:${expId}/${runId}`, [
    ...targets,
    { label: "Requeue anywhere", act: act("requeueRun", expId, runId), divider: targets.length > 0 },
    { label: "Logs", icon: "logs", href: pageHref("logs.html", { experiment: expId, run: runId }), divider: true },
    { label: "Results", icon: "results", href: pageHref("results.html", { experiment: expId, run: runId }) },
    { label: "Cancel run", act: act("cancelInFlight", expId, runId), danger: true, divider: true },
  ]);
}

// Drag a run chip onto a GPU tile in the pool.
const grid = document.getElementById("workers-grid");
grid.addEventListener("dragstart", e => {
  const chip = e.target.closest?.(".run-chip[draggable]");
  if (!chip) return;
  dragging = { expId: chip.dataset.exp, runId: chip.dataset.run, from: chip.closest(".gpu-tile") };
  e.dataTransfer.effectAllowed = "move";
  e.dataTransfer.setData("text/plain", chip.dataset.run);
  chip.classList.add("dragging");
  document.body.classList.add("dragging-run");
});
grid.addEventListener("dragend", () => {
  dragging = null;
  document.body.classList.remove("dragging-run");
  grid.querySelectorAll(".drop-ok, .dragging").forEach(el => el.classList.remove("drop-ok", "dragging"));
  if (pendingRender) renderWorkers(lastWorkers);
});
function dropTarget(e) {
  const tile = e.target.closest?.(".gpu-tile.st-pool");
  return tile && dragging && tile !== dragging.from ? tile : null;
}
grid.addEventListener("dragover", e => {
  const tile = dropTarget(e);
  grid.querySelectorAll(".drop-ok").forEach(el => { if (el !== tile) el.classList.remove("drop-ok"); });
  if (!tile) return;
  e.preventDefault();
  tile.classList.add("drop-ok");
});
grid.addEventListener("drop", e => {
  const tile = dropTarget(e);
  if (!tile) return;
  e.preventDefault();
  const { expId, runId } = dragging;
  moveRunTo(expId, runId, tile.dataset.worker, +tile.dataset.gpu);
});
document.addEventListener("click", () => {
  // The chip menu closed (rowmenu.js closes on any click): let refreshes redraw.
  setTimeout(() => {
    if (!rowMenu) {
      grid.querySelectorAll(".run-chip.menu-open").forEach(c => c.classList.remove("menu-open"));
      if (pendingRender) renderWorkers(lastWorkers);
    }
  }, 0);
});

// ── Render in-flight jobs ──────────────────────────────────────────────────────
let inFlight = [];

function renderInFlight(jobs) {
  inFlight = jobs;
  const tbody = document.getElementById("inflight-tbody");
  if (!jobs.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="no-data">No jobs currently running.</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map(j => `<tr>
    <td class="mono id-cell">${runNameHtml(j)}</td>
    <td class="mono id-cell">${escHtml(j.experiment_id || "—")}</td>
    <td>${badge(j.status)}</td>
    <td class="mono">${j.worker_id ? escHtml(j.worker_id.split(":")[0]) : "—"}</td>
    <td class="mono">${j.dispatched_gpu_ids ? escHtml(JSON.parse(j.dispatched_gpu_ids || "[]").join(",")) : "—"}</td>
    <td>${j.elapsed != null ? fmtElapsed(j.elapsed) : elapsedSince(j.start_time) || "—"}</td>
    <td class="actions-cell"><div class="actions">
      <button class="btn-link btn-menu" aria-haspopup="true" title="Results, Artifacts, Logs, Notes" ${act("toggleRunView", j.experiment_id, j.run_id)}>View ▾</button>
      <button class="btn-link btn-manage" aria-haspopup="true" title="Rename or cancel" ${act("toggleInFlightMenu", j.experiment_id, j.run_id)}>Manage</button>
    </div></td>
  </tr>`).join("");
}

// Every in-flight run is dispatched or running, so Manage offers what applies
// to an active run.  Moving and deleting wait until it finishes.
function toggleInFlightMenu(expId, runId) {
  toggleMenu(this, `run:${expId}/${runId}`, [
    { label: "Rename…", act: act("renameInFlight", expId, runId) },
    { label: "Cancel", act: act("cancelInFlight", expId, runId), danger: true },
  ]);
}

async function renameInFlight(expId, runId) {
  const job = inFlight.find(j => j.experiment_id === expId && j.run_id === runId);
  if (await mlRenameRun(expId, runId, job?.label) !== undefined) await refresh();
}

async function cancelInFlight(expId, runId) {
  if (!await mlConfirm(`Cancel run ${runId}?`, { confirmLabel: "Cancel run", cancelLabel: "Keep running" })) return;
  await postRunAction(expId, runId, "cancel");
  await refresh();
}

watchNoteMentions(() => {});

// ── Poll ──────────────────────────────────────────────────────────────────────
async function refresh() {
  try {
    const [wRes, jRes, dRes, sRes] = await Promise.all([
      apiFetch("/api/workers"),
      apiFetch("/api/jobs?status=running"),
      apiFetch("/api/jobs?status=dispatched"),
      apiFetch("/api/scheduler"),
    ]);
    if (!wRes.ok || !jRes.ok || !dRes.ok || !sRes.ok) {
      setConnStatus("disconnected");
      return;
    }
    const workers = await wRes.json();
    const jobs    = [...(await jRes.json()), ...(await dRes.json())];
    setConnStatus("connected");
    renderPlacement((await sRes.json()).placement);
    renderInFlight(jobs);   // first: the worker cards name runs from it
    renderWorkers(workers);
  } catch (e) {
    setConnStatus("disconnected");
  }
}

refresh();
setInterval(refresh, 5000);

// ── Storage cleanup ───────────────────────────────────────────────────────────
function renderGc(plan, done) {
  const box = document.getElementById("gc-box");
  const db = plan.database;
  const objects = plan.objects || { count: 0, bytes: 0 };
  const items = [...plan.directories, ...plan.artifact_files];
  if (objects.count) items.push({ path: `store/objects (${objects.count} stored file${objects.count === 1 ? "" : "s"})`,
                                  reason: "no code snapshot lists them", bytes: objects.bytes });
  const rows = db.metric_rows + db.log_rows + db.job_node_rows + plan.artifact_rows;
  let html = `<div class="gc-sum"><b>${done ? "Deleted" : "Can free"} ${fmtSize(plan.bytes)}</b> in ${items.length} path${items.length === 1 ? "" : "s"}` +
    `, plus ${rows} database record${rows === 1 ? "" : "s"} (${db.metric_rows} metric, ${db.log_rows} log, ${db.job_node_rows} run-node, ` +
    `${plan.artifact_rows} code).${done && plan.vacuumed ? " Database compacted." : ""}</div>`;
  if (items.length) {
    html += "<table>" + items.slice(0, 300).map(it =>
      `<tr><td>${escHtml(it.path)}</td><td class="why">${escHtml(it.reason)}</td><td class="num">${fmtSize(it.bytes)}</td></tr>`).join("") + "</table>";
    if (items.length > 300) html += `<div class="gc-sum">… and ${items.length - 300} more</div>`;
  } else if (!rows) html += `<div class="no-data">Nothing to clean up.</div>`;
  box.innerHTML = html;
  document.getElementById("gc-run").disabled = done || (!items.length && !rows);
}
async function runGc(dryRun) {
  const vacuum = document.getElementById("gc-vacuum").checked;
  if (!dryRun && !(await mlConfirm("Delete everything listed? This cannot be undone.", { confirmLabel: "Delete" }))) return;
  const btn = document.getElementById(dryRun ? "gc-scan" : "gc-run");
  btn.disabled = true; const label = btn.textContent; btn.textContent = dryRun ? "Scanning…" : "Deleting…";
  try {
    const r = await apiFetch("/api/gc", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ dry_run: dryRun, vacuum: !dryRun && vacuum }) });
    const plan = await r.json();
    if (!r.ok) throw new Error(plan.error || r.status);
    renderGc(plan, !dryRun);
  } catch (e) { await mlAlert("Cleanup failed: " + e.message); }
  finally { btn.textContent = label; if (dryRun) btn.disabled = false; }
}
document.getElementById("gc-scan").onclick = () => runGc(true);
document.getElementById("gc-run").onclick = () => runGc(false);

bindActions({
  disconnectWorker, updateWorker, toggleRunView, toggleInFlightMenu, renameInFlight, cancelInFlight,
  toggleGpuSelect, clearGpuSelect, applyGpus, toggleWorkerMenu, toggleChipMenu, moveRunTo, requeueRun,
});
bindActions({
  toggleGpuPool(workerId, g, inPool) { toggleGpuPool(workerId, g, inPool); },
  workerConcurrencyChanged(workerId) { setWorkerConcurrency(workerId, +this.value).then(refresh); },
}, "change");
