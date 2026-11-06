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

async function setWorkerGpus(el, workerId, currentGpus) {
  const newIds = el.value.split(",").map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
  const oldSet = new Set(currentGpus);
  const newSet = new Set(newIds);
  const add    = newIds.filter(n => !oldSet.has(n));
  const remove = currentGpus.filter(n => !newSet.has(n));
  if (!add.length && !remove.length) return;
  if (remove.length) {
    const gpuList = remove.join(", ");
    if (!await mlConfirm(`Remove GPU${remove.length > 1 ? "s" : ""} ${gpuList}?\nAny jobs using them will be re-queued.`, { confirmLabel: "Remove" })) {
      el.value = currentGpus.join(",");
      return;
    }
  }
  try {
    const res = await apiFetch(`/api/workers/${encodeURIComponent(workerId)}/devices`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ add, remove }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      await mlAlert("Failed: " + (data.error || res.status));
      el.value = currentGpus.join(",");
      return;
    }
    refresh();
  } catch (e) {
    await mlAlert("Network error: " + e.message);
    el.value = currentGpus.join(",");
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

// ── GPU stat list ──────────────────────────────────────────────────────────────
function renderGpuList(gpus, occ, gpuStats) {
  if (!gpus.length) return "";
  const header = `<div class="gpu-stat-row">
    <span class="gpu-stat-id"></span>
    <div class="gpu-bar-wrap" style="background:none"></div>
    <span class="gpu-stat-pct gpu-stat-col-label">util</span>
    <span class="gpu-stat-mem gpu-stat-col-label">memory</span>
    <span class="gpu-stat-jobs gpu-stat-col-label">jobs</span>
  </div>`;
  return `<div class="gpu-stat-list">${header}${gpus.map(gid => {
    const used  = occ[gid] ?? occ[String(gid)] ?? 0;
    const stat  = gpuStats && (gpuStats[gid] ?? gpuStats[String(gid)]);
    const util  = stat?.util_pct ?? null;
    const memUsed  = stat?.mem_used_mb  ?? -1;
    const memTotal = stat?.mem_total_mb ?? -1;
    const memStr = memTotal > 0
      ? `${(memUsed / 1024).toFixed(1)}/${Math.round(memTotal / 1024)}G`
      : "";
    return `<div class="gpu-stat-row">
      <span class="gpu-stat-id">GPU ${escHtml(gid)}</span>
      <div class="gpu-bar-wrap"><div class="gpu-bar-fill" style="width:${util ?? 0}%"></div></div>
      <span class="gpu-stat-pct">${util !== null ? util + "%" : "—"}</span>
      <span class="gpu-stat-mem">${memStr}</span>
      <span class="gpu-stat-jobs">${used}</span>
    </div>`;
  }).join("")}</div>`;
}

// ── Render workers ─────────────────────────────────────────────────────────────
function renderWorkers(workers) {
  const grid = document.getElementById("workers-grid");
  // Don't clobber an input the user is actively editing
  const active = document.activeElement;
  if (active && active.tagName === "INPUT" && grid.contains(active)) return;
  if (!workers.length) {
    grid.innerHTML = `<div class="no-data">No workers connected.</div>`;
    return;
  }

  grid.innerHTML = workers.map(w => {
    const gpus     = w.gpus || [];
    const occ      = w.gpu_occupancy || {};
    const gpuStats = w.gpu_stats || {};
    const status   = w.status || "offline";
    const isLive   = status === "connected";
    const canDisconnect = status !== "dead";
    const maxJobs  = w.max_jobs_per_gpu || 0;

    return `<div class="worker-card">
      <div class="worker-header">
        <span class="worker-host">${escHtml(w.host)}${w.port ? `:${escHtml(w.port)}` : ""}</span>
        ${badge(status)}
      </div>
      ${w.last_error ? `<div class="worker-error">${escHtml(w.last_error)}</div>` : ""}
      ${renderGpuList(gpus, occ, gpuStats)}
      ${isLive ? `<div class="worker-controls">
        <div class="worker-control-row">
          <span class="worker-control-label">GPUs</span>
          <input class="worker-control-input" type="text" value="${escHtml(gpus.join(","))}"
            placeholder="e.g. 0,1,2,3"
            ${act("workerGpusChanged", w.worker_id, gpus)}>
        </div>
        <div class="worker-control-row">
          <span class="worker-control-label">Jobs/GPU</span>
          <input class="worker-control-input worker-control-narrow" type="number" min="0" value="${escHtml(maxJobs)}"
            ${act("workerConcurrencyChanged", w.worker_id)}>
          <span class="worker-control-hint">0 = unlimited</span>
        </div>
      </div>` : ""}
      <div class="worker-detail">
        ${w.remote_dir  ? `<div><span>dir</span> ${escHtml(w.remote_dir)}</div>` : ""}
        ${w.venv        ? `<div><span>venv</span> ${escHtml(w.venv)}</div>` : ""}
        ${w.ssh_key     ? `<div><span>key</span> ${escHtml(w.ssh_key)}</div>` : ""}
        ${w.scratch_dir ? `<div><span>scratch</span> ${escHtml(w.scratch_dir)}</div>` : ""}
      </div>
      <div class="worker-actions">
        <span class="worker-last-seen">${w.last_seen ? relTime(w.last_seen) : ""}</span>
        ${canDisconnect ? `<button class="btn-disconnect" ${act("disconnectWorker", w.worker_id)}>Disconnect</button>` : ""}
      </div>
    </div>`;
  }).join("");
}

// ── Render in-flight jobs ──────────────────────────────────────────────────────
function renderInFlight(jobs) {
  const tbody = document.getElementById("inflight-tbody");
  if (!jobs.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="no-data">No jobs currently running.</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map(j => `<tr>
    <td class="mono id-cell">${runNameHtml(j)}</td>
    <td class="mono id-cell">${escHtml(j.experiment_id || "—")}</td>
    <td>${badge(j.status)}</td>
    <td class="mono">${j.worker_id ? escHtml(j.worker_id.split(":")[0]) : "—"}</td>
    <td class="mono">${j.dispatched_gpu_ids ? escHtml(JSON.parse(j.dispatched_gpu_ids || "[]").join(",")) : "—"}</td>
    <td>${fmtElapsed(j.elapsed)}</td>
  </tr>`).join("");
}

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
    renderWorkers(workers);
    renderInFlight(jobs);
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
  const items = [...plan.directories, ...plan.artifact_files];
  const rows = db.metric_rows + db.log_rows + db.job_node_rows + plan.artifact_rows;
  let html = `<div class="gc-sum"><b>${done ? "Deleted" : "Can free"} ${fmtSize(plan.bytes)}</b> in ${items.length} path${items.length === 1 ? "" : "s"}` +
    `, plus ${rows} database record${rows === 1 ? "" : "s"} (${db.metric_rows} metric, ${db.log_rows} log, ${db.job_node_rows} run-node, ` +
    `${plan.artifact_rows} archive).${done && plan.vacuumed ? " Database compacted." : ""}</div>`;
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

bindActions({ disconnectWorker });
bindActions({
  workerGpusChanged(workerId, gpus) { setWorkerGpus(this, workerId, gpus); },
  workerConcurrencyChanged(workerId) { setWorkerConcurrency(workerId, +this.value); },
}, "change");
