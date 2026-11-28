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
    document.getElementById("fw-tunnel").checked = false;
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
        renderKnown();
      }
    } catch (_) {}
  }

  // Previously connected workers that are not connected now, each with
  // Reconnect and Forget.
  function renderKnown() {
    const inactive = _knownWorkers.filter(w => w.status !== "connected");
    knownList.innerHTML = inactive.map(w => `
      <div class="known-worker-row">
        <div class="known-worker-info">
          <span class="known-worker-host">${escHtml(w.host)}</span>
          <span class="known-worker-dir">${escHtml(w.remote_dir)}</span>
        </div>
        <button class="btn-forget" data-worker-id="${escHtml(w.worker_id)}"
          title="Remove from this list; its runs, if any, are re-queued">Forget</button>
        <button class="btn-reconnect" data-worker-id="${escHtml(w.worker_id)}">Reconnect</button>
      </div>`).join("");
    knownList.classList.toggle("hidden", !inactive.length);
    divider.classList.toggle("hidden", !inactive.length);
  }

  function closeModal() { modal.classList.add("hidden"); }

  knownList.addEventListener("click", async e => {
    const forget = e.target.closest(".btn-forget");
    if (!forget) return;
    const workerId = forget.dataset.workerId;
    forget.disabled = true;
    errEl.classList.add("hidden");
    try {
      const res = await apiFetch(`/api/workers/${encodeURIComponent(workerId)}?forget=1`, { method: "DELETE" });
      if (!res.ok && res.status !== 404) {
        const data = await res.json().catch(() => ({}));
        errEl.textContent = "Failed to forget: " + (data.error || res.status);
        errEl.classList.remove("hidden");
        forget.disabled = false;
        return;
      }
      _knownWorkers = _knownWorkers.filter(w => w.worker_id !== workerId);
      renderKnown();
      refresh();
    } catch (err) {
      errEl.textContent = "Network error: " + err.message;
      errEl.classList.remove("hidden");
      forget.disabled = false;
    }
  });

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
          devices, port: w.port || 0, tunnel: !!w.tunnel,
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
    const tunnel     = document.getElementById("fw-tunnel").checked;

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
        body: JSON.stringify({ host, remote_dir, venv, ssh_key, devices, tunnel }),
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
// Works on a worker in any state, one that never started included, and takes
// it out of the pool.  It leaves the page and a manager restart does not
// relaunch it; the Connect dialog offers it again, or forgets it.
async function disconnectWorker(workerId) {
  const w = lastWorkers.find(x => x.worker_id === workerId);
  const runs = Object.values(w?.gpu_runs || {}).reduce((n, rs) => n + rs.length, 0);
  const msg = `Disconnect worker "${workerId}" and remove it from the pool?`
    + (runs ? `\nIts ${runs} in-flight run${runs === 1 ? "" : "s"} will be re-queued.` : "")
    + "\nThe Connect dialog can reconnect or forget it later.";
  if (!await mlConfirm(msg, { confirmLabel: "Disconnect" })) return;
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

// ── Scheduling pause ───────────────────────────────────────────────────────────
// Paused, the scheduler starts nothing new (on one worker, or anywhere), so
// runs can be placed by hand; running jobs carry on.  Unpausing everywhere
// also lifts each worker's own pause.
let clusterPaused = false;

function renderPause(scheduler) {
  clusterPaused = !!scheduler.paused;
  const btn = document.getElementById("btn-pause");
  btn.textContent = clusterPaused ? "Unpause scheduling" : "Pause scheduling";
  btn.classList.toggle("active", clusterPaused);
  btn.title = clusterPaused
    ? "Start scheduling again everywhere, including on workers paused one by one"
    : "Start no new jobs on any worker, so you can place runs by hand. Running jobs carry on.";
  document.getElementById("pause-banner").classList.toggle("hidden", !clusterPaused);
}

async function setPaused(workerId, paused) {
  try {
    renderPause(await apiJSON("POST", `/api/scheduler/${paused ? "pause" : "unpause"}`,
                              workerId ? { worker_id: workerId } : {}));
  } catch (e) {
    await mlAlert("Failed: " + e.message);
  }
  await refresh();
}

document.getElementById("btn-pause").addEventListener("click", () => setPaused(null, !clusterPaused));

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
  if ((w.pending_gpus || []).includes(g)) return "probing";
  if (inPool) return "pool";
  return runs.length ? "draining" : "out";
}

const STATE_LABEL = { pool: "in pool", draining: "draining", out: "out of pool", unhealthy: "unhealthy", probing: "probing" };
// Tile states the user cannot select or toggle: the worker's health probe holds them.
const HELD = new Set(["unhealthy", "probing"]);

function allGpusOf(w) {
  const ids = new Set([...(w.gpus || []), ...(w.unhealthy_gpus || []), ...(w.pending_gpus || [])]);
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
  const toggle = live && !HELD.has(st)
    ? `<label class="gt-switch" title="${st === "pool" ? "Take out of the pool; its runs keep going" : "Put back in the pool"}" data-act="">
         <input type="checkbox" ${st === "pool" ? "checked" : ""} ${act("toggleGpuPool", w.worker_id, g, st === "pool")}><span></span></label>`
    : "";
  const held = st === "unhealthy"
    ? (w.unhealthy_reasons || {})[g] || "Failed the worker's CUDA probe"
    : st === "probing" ? "The worker is still checking this GPU can create a CUDA context" : "";
  return `<div class="gpu-tile st-${st}${sel ? " selected" : ""}" data-worker="${escHtml(w.worker_id)}" data-gpu="${g}"
      ${held ? `title="${escHtml(held)}"` : ""}
      ${live && !HELD.has(st) ? act("toggleGpuSelect", w.worker_id, g) : ""}>
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
  const n = { pool: 0, draining: 0, out: 0, unhealthy: 0, probing: 0 };
  for (const g of all) n[gpuState(w, g)]++;
  const runs = new Set(Object.values(w.gpu_runs || {}).flat().map(r => r.experiment_id + "/" + r.run_id)).size;
  const parts = [`${n.pool} of ${all.length} GPU${all.length === 1 ? "" : "s"} in pool`];
  if (n.draining) parts.push(`${n.draining} draining`);
  if (n.unhealthy) parts.push(`${n.unhealthy} unhealthy`);
  if (n.probing) parts.push(`${n.probing} probing`);
  parts.push(`${runs} run${runs === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

function renderWorkers(workers) {
  // A disconnected worker is out of the pool; the Connect dialog lists it.
  lastWorkers = workers = workers.filter(w => w.status !== "disconnected");
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
    const canUpdate     = w.outdated && (status === "dead" || isLive);
    const maxJobs  = w.max_jobs_per_gpu || 0;
    const gpus = allGpusOf(w);

    return `<div class="worker-card${isLive ? "" : " offline"}" data-worker="${escHtml(w.worker_id)}">
      <div class="worker-header">
        <span class="worker-host">${escHtml(w.host)}${w.port ? `:${escHtml(w.port)}` : ""}</span>
        ${badge(status)}
        ${w.paused ? `<span title="The scheduler starts no new jobs here">${badge("paused")}</span>` : ""}
        <span class="worker-summary">${poolSummary(w)}</span>
        ${isLive ? `<label class="jobs-per-gpu" title="Most runs the scheduler puts on one GPU (0 = no limit)">jobs/GPU
          <input class="worker-control-input worker-control-narrow" type="number" min="0" value="${escHtml(maxJobs)}"
            ${act("workerConcurrencyChanged", w.worker_id)}></label>` : ""}
        <button class="btn-link btn-manage" aria-haspopup="true" title="Pause, pool, update, disconnect" ${act("toggleWorkerMenu", w.worker_id)}>⋯</button>
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
          ${w.tunnel      ? `<div><span>via</span> SSH tunnel</div>` : ""}
          ${w.scratch_dir ? `<div><span>scratch</span> ${escHtml(w.scratch_dir)}</div>` : ""}
        </div>
        <span class="worker-actions-buttons">
          <span class="worker-last-seen">${w.last_seen ? relTime(w.last_seen) : ""}</span>
          ${canUpdate ? `<button class="btn-update" title="Install this manager's mlsweep on ${escHtml(w.host)} and restart the worker" ${act("updateWorker", w.worker_id, isLive)}>Update worker</button>` : ""}
          <button class="btn-disconnect" title="Remove this worker from the pool" ${act("disconnectWorker", w.worker_id)}>Disconnect</button>
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
  toggleMenu(this, `worker:${workerId}`, workerItems(workerId));
}

// The worker's pool changes; withLifecycle adds Update and Disconnect, which
// the card shows as buttons of their own.
function workerItems(workerId, withLifecycle = false) {
  const w = lastWorkers.find(x => x.worker_id === workerId);
  const all = w ? allGpusOf(w) : [];
  const live = w?.status === "connected";
  const items = [];
  if (w) items.push(w.paused
    ? { label: "Unpause scheduling here", act: act("setPaused", workerId, false) }
    : { label: "Pause scheduling here", title: "Start no new jobs here; running jobs carry on", act: act("setPaused", workerId, true) });
  if (live) {
    const inPool = all.filter(g => gpuState(w, g) === "pool");
    const outPool = all.filter(g => ["out", "draining"].includes(gpuState(w, g)));
    if (inPool.length) items.push({ label: "Drain all GPUs", title: "Nothing new starts here; runs finish", act: act("applyGpus", workerId, "drain", inPool) });
    if (outPool.length) items.push({ label: "Return all GPUs to the pool", act: act("applyGpus", workerId, "add", outPool) });
  }
  if (items.length > 1) items[1].divider = true;
  if (withLifecycle && w) {
    const status = w.status || "offline";
    const more = [];
    if (w.outdated && (status === "dead" || live)) more.push({ label: "Update worker", act: act("updateWorker", workerId, live) });
    more.push({ label: "Disconnect", act: act("disconnectWorker", workerId), danger: true });
    more.push({ label: "Copy worker ID", act: act("copyId", workerId, "Copied the worker ID") });
    return joinMenus(items, more);
  }
  return items;
}


// One GPU's own pool changes, for a right-click on its tile.
function gpuItems(w, g) {
  const st = gpuState(w, g);
  if (w.status !== "connected") return [];
  if (HELD.has(st)) return [{ label: `GPU ${g} is ${STATE_LABEL[st].toLowerCase()}`, disabled: true,
                              title: st === "unhealthy" ? (w.unhealthy_reasons || {})[g] || "" : "" }];
  const runs = ((w.gpu_runs || {})[g] || []).length;
  const sel = selectedGpus.get(w.worker_id)?.has(g);
  const items = [];
  if (st === "pool") {
    items.push({ label: `Drain GPU ${g}`, title: "Nothing new starts here; its runs finish", act: act("applyGpus", w.worker_id, "drain", [g]) });
    items.push({ label: `Remove GPU ${g} now${runs ? ` (requeue ${runs})` : ""}`, danger: true,
                 title: "Out of the pool now; its runs restart elsewhere", act: act("applyGpus", w.worker_id, "remove", [g]) });
  } else {
    items.push({ label: `Return GPU ${g} to the pool`, act: act("applyGpus", w.worker_id, "add", [g]) });
  }
  items.push({ label: sel ? "Deselect" : "Select", title: "Select several GPUs to change them together",
               act: act("toggleGpuSelect", w.worker_id, g), divider: true });
  return items;
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
  const chip = this.closest(".run-chip");
  chip?.classList.add("menu-open");
  toggleMenu(this, `chip:${expId}/${runId}`, chipItems(workerId, expId, runId));
}

function chipItems(workerId, expId, runId) {
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
  return [
    ...targets,
    { label: "Requeue anywhere", act: act("requeueRun", expId, runId), divider: targets.length > 0 },
    { label: "Logs", icon: "logs", href: pageHref("logs.html", { experiment: expId, run: runId }), divider: true },
    { label: "Results", icon: "results", href: pageHref("results.html", { experiment: expId, run: runId }) },
    { label: "Cancel run", act: act("cancelInFlight", expId, runId), danger: true, divider: true },
  ];
}

// Right-click a run chip, a GPU tile or the rest of a worker card.
bindRowContextMenu(document.getElementById("workers-grid"), ".run-chip[data-run], .gpu-tile, .worker-card", el => {
  const card = el.closest(".worker-card");
  const w = lastWorkers.find(x => x.worker_id === card?.dataset.worker);
  if (!w) return null;
  if (el.classList.contains("run-chip")) {
    el.classList.add("menu-open");
    const { exp, run } = el.dataset;
    return [`ctxchip:${exp}/${run}`, chipItems(w.worker_id, exp, run)];
  }
  if (el.classList.contains("gpu-tile")) {
    const items = gpuItems(w, +el.dataset.gpu);
    if (items.length) return [`ctxgpu:${w.worker_id}/${el.dataset.gpu}`, items];
  }
  return [`ctxworker:${w.worker_id}`, workerItems(w.worker_id, true)];
});

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
// The chip menu closed (rowmenu.js closes on any click, Esc or right-click
// elsewhere): let refreshes redraw.
function releaseChips() {
  setTimeout(() => {
    if (!rowMenu || !rowMenu.dataset.for.includes("chip:")) {
      grid.querySelectorAll(".run-chip.menu-open").forEach(c => c.classList.remove("menu-open"));
      if (pendingRender) renderWorkers(lastWorkers);
    }
  }, 0);
}
document.addEventListener("click", releaseChips);
document.addEventListener("keydown", e => { if (e.key === "Escape") releaseChips(); });
document.addEventListener("contextmenu", releaseChips);

// ── Render in-flight jobs ──────────────────────────────────────────────────────
let inFlight = [];

function renderInFlight(jobs) {
  inFlight = jobs;
  const tbody = document.getElementById("inflight-tbody");
  if (!jobs.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="no-data">No jobs currently running.</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map(j => `<tr data-exp-id="${escHtml(j.experiment_id)}" data-run-id="${escHtml(j.run_id)}">
    <td class="mono id-cell">${runNameHtml(j)}</td>
    <td class="mono id-cell">${escHtml(j.experiment_id || "—")}</td>
    <td>${badge(j.status)}</td>
    <td class="mono">${j.worker_id ? escHtml(j.worker_id.split(":")[0]) : "—"}</td>
    <td class="mono">${j.dispatched_gpu_ids ? escHtml(JSON.parse(j.dispatched_gpu_ids || "[]").join(",")) : "—"}</td>
    <td>${j.elapsed != null ? fmtElapsed(j.elapsed) : elapsedSince(j.start_time) || "—"}</td>
    <td class="actions-cell"><div class="actions">
      <button class="btn-link btn-menu" aria-haspopup="true" title="Results, Artifacts, Logs, Notes" ${act("toggleRunView", j.experiment_id, j.run_id, j.code_files ?? null)}>View ▾</button>
      <button class="btn-link btn-manage" aria-haspopup="true" title="Rename or cancel" ${act("toggleInFlightMenu", j.experiment_id, j.run_id)}>Manage</button>
    </div></td>
  </tr>`).join("");
}

// Every in-flight run is dispatched or running, so Manage offers what applies
// to an active run.  Moving and deleting wait until it finishes.
function toggleInFlightMenu(expId, runId) {
  toggleMenu(this, `run:${expId}/${runId}`, inFlightItems(expId, runId));
}

function inFlightItems(expId, runId) {
  return [
    { label: "Rename…", act: act("renameInFlight", expId, runId) },
    { label: "Cancel", act: act("cancelInFlight", expId, runId), danger: true },
  ];
}

bindRowContextMenu(document.getElementById("inflight-tbody"), "tr[data-run-id]", row => {
  const { expId, runId } = row.dataset;
  return [`ctx:${expId}/${runId}`, joinMenus(runViewItems(expId, runId, jobOf(expId, runId)?.code_files),
    [{ label: "Copy run ID", act: act("copyId", runId, "Copied the run ID") }, ...inFlightItems(expId, runId)])];
});

async function renameInFlight(expId, runId) {
  const job = inFlight.find(j => j.experiment_id === expId && j.run_id === runId);
  if (await mlRenameRun(expId, runId, job?.label) !== undefined) await refresh();
}

async function cancelInFlight(expId, runId) {
  if (!await mlConfirm(`Cancel run ${runId}?`, { confirmLabel: "Cancel run", cancelLabel: "Keep running" })) return;
  await postRunAction(expId, runId, "cancel");
  await refresh();
}

// ── Render the job queue ───────────────────────────────────────────────────────
// Pending runs in the order the scheduler takes them (priority, then submit
// time).  A paused experiment's runs keep their place but are held.
const QUEUE_LIMIT = 200;
let queued = [];

function renderQueue(jobs, total, pausedExps) {
  queued = jobs;
  const tbody = document.getElementById("queue-tbody");
  document.getElementById("queue-count").textContent = total ? `(${total})` : "";
  const more = document.getElementById("queue-more");
  more.classList.toggle("hidden", total <= jobs.length);
  more.textContent = total > jobs.length ? `Showing the first ${jobs.length} of ${total}.` : "";
  if (!jobs.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="no-data">The queue is empty.</td></tr>`;
    return;
  }
  let pos = 0;
  tbody.innerHTML = jobs.map(j => {
    const held = pausedExps.has(j.experiment_id);
    if (!held) pos++;
    const gpus = j.gpus_per_run === 0 ? "CPU" : j.nodes_per_run > 1 ? `${j.gpus_per_run} × ${j.nodes_per_run} nodes` : String(j.gpus_per_run);
    const status = held ? badge("paused", "held")
      : j.retry_count ? badge("pending", `retry ${j.retry_count}`) : badge("pending");
    return `<tr class="${held ? "held" : ""}" data-exp-id="${escHtml(j.experiment_id)}" data-run-id="${escHtml(j.run_id)}"${held ? ' title="Its experiment is paused; resume it to schedule this run"' : ""}>
    <td class="mono">${held ? "—" : pos}</td>
    <td class="mono id-cell">${runNameHtml(j)}</td>
    <td class="mono id-cell">${escHtml(j.experiment_id || "—")}</td>
    <td>${status}</td>
    <td class="mono">${escHtml(gpus)}</td>
    <td class="mono">${j.priority}</td>
    <td>${elapsedSince(j.submit_time) || "—"}</td>
    <td class="actions-cell"><div class="actions">
      <button class="btn-link btn-menu" aria-haspopup="true" title="Results, Artifacts, Logs, Notes" ${act("toggleRunView", j.experiment_id, j.run_id, j.code_files ?? null)}>View ▾</button>
      <button class="btn-link btn-manage" aria-haspopup="true" title="Rename or cancel" ${act("toggleQueuedMenu", j.experiment_id, j.run_id)}>Manage</button>
    </div></td>
  </tr>`;
  }).join("");
}

function queuedItems(expId, runId) {
  return [
    { label: "Rename…", act: act("renameQueued", expId, runId) },
    { label: "Cancel", act: act("cancelQueued", expId, runId), danger: true },
  ];
}

function toggleQueuedMenu(expId, runId) {
  toggleMenu(this, `queued:${expId}/${runId}`, queuedItems(expId, runId));
}

bindRowContextMenu(document.getElementById("queue-tbody"), "tr[data-run-id]", row => {
  const { expId, runId } = row.dataset;
  const job = queued.find(j => j.experiment_id === expId && j.run_id === runId);
  return [`qctx:${expId}/${runId}`, joinMenus(runViewItems(expId, runId, job?.code_files),
    [{ label: "Copy run ID", act: act("copyId", runId, "Copied the run ID") }, ...queuedItems(expId, runId)])];
});

async function renameQueued(expId, runId) {
  const job = queued.find(j => j.experiment_id === expId && j.run_id === runId);
  if (await mlRenameRun(expId, runId, job?.label) !== undefined) await refresh();
}

async function cancelQueued(expId, runId) {
  if (!await mlConfirm(`Cancel queued run ${runId}?`, { confirmLabel: "Cancel run", cancelLabel: "Keep it" })) return;
  await postRunAction(expId, runId, "cancel");
  await refresh();
}

watchNoteMentions(() => {});
watchTraceCounts(() => {});

// ── Poll ──────────────────────────────────────────────────────────────────────
let refreshSeq = 0;
async function refresh() {
  const seq = ++refreshSeq;
  try {
    const responses = await Promise.all([
      apiFetch("/api/workers"),
      apiFetch("/api/jobs?status=running"),
      apiFetch("/api/jobs?status=dispatched"),
      apiFetch("/api/scheduler"),
      apiFetch(`/api/jobs?status=pending&limit=${QUEUE_LIMIT}`),
      apiFetch("/api/experiments?status=paused"),
      apiFetch("/api/health"),
    ]);
    if (seq !== refreshSeq) return;   // a newer refresh owns the page now
    if (!responses.every(r => r.ok)) {
      setConnStatus("disconnected");
      return;
    }
    const [workers, running, dispatched, scheduler, pending, paused, health] =
      await Promise.all(responses.map(r => r.json()));
    if (seq !== refreshSeq) return;

    // The two job queries are separate snapshots.  A run that changed status
    // between them can appear in both (or neither); keep one row per run and
    // prefer the later state.
    const byRun = new Map();
    for (const j of [...running, ...dispatched]) {
      const key = `${j.experiment_id}\u0000${j.run_id}`;
      const old = byRun.get(key);
      if (!old || (old.status === "dispatched" && j.status === "running")) byRun.set(key, j);
    }

    setConnStatus("connected");
    renderPlacement(scheduler.placement);
    renderPause(scheduler);
    renderInFlight([...byRun.values()]);   // first: the worker cards name runs from it
    renderWorkers(workers);
    renderQueue(pending, Math.max(health.jobs_pending, pending.length), new Set(paused.map(e => e.experiment_id)));
  } catch (e) {
    if (seq === refreshSeq) setConnStatus("disconnected");
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
let gcBusy = false;
async function runGc(dryRun) {
  // A scan that finishes after a delete would redraw the deleted plan and
  // re-enable the delete button; only one cleanup request at a time.
  if (gcBusy) return;
  const vacuum = document.getElementById("gc-vacuum").checked;
  if (!dryRun && !(await mlConfirm("Delete everything listed? This cannot be undone.", { confirmLabel: "Delete" }))) return;
  gcBusy = true;
  const btn = document.getElementById(dryRun ? "gc-scan" : "gc-run");
  btn.disabled = true; const label = btn.textContent; btn.textContent = dryRun ? "Scanning…" : "Deleting…";
  try {
    const r = await apiFetch("/api/gc", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ dry_run: dryRun, vacuum: !dryRun && vacuum }) });
    const plan = await r.json();
    if (!r.ok) throw new Error(plan.error || r.status);
    renderGc(plan, !dryRun);
  } catch (e) { await mlAlert("Cleanup failed: " + e.message); }
  finally { gcBusy = false; btn.textContent = label; if (dryRun) btn.disabled = false; }
}
document.getElementById("gc-scan").onclick = () => runGc(true);
document.getElementById("gc-run").onclick = () => runGc(false);

bindActions({
  disconnectWorker, updateWorker, toggleRunView, toggleInFlightMenu, renameInFlight, cancelInFlight,
  toggleQueuedMenu, renameQueued, cancelQueued,
  toggleGpuSelect, clearGpuSelect, applyGpus, toggleWorkerMenu, toggleChipMenu, moveRunTo, requeueRun, copyId,
  setPaused,
});
bindActions({
  toggleGpuPool(workerId, g, inPool) { toggleGpuPool(workerId, g, inPool); },
  workerConcurrencyChanged(workerId) { setWorkerConcurrency(workerId, +this.value).then(refresh); },
}, "change");
