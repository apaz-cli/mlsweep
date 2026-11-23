// ── Auth & URL params ──────────────────────────────────────────────────────────
const _p       = new URLSearchParams(location.search);
const INIT_EXP  = _p.get("experiment") || "";
const INIT_RUN  = _p.get("run")  || "";
const INIT_FILE = _p.get("file") || "";
let _initDone = false;





MLChrome.collapsibleSidebar(document.getElementById("sidebar"));
MLChrome.resizable(document.getElementById("sidebar"), { name: "sidebar_width_files", min: 180, max: 520 });
MLChrome.resizable(document.getElementById("file-tree"), { name: "artifacts_tree_width", min: 140, max: 520 });

// ── State ──────────────────────────────────────────────────────────────────────
let jobs        = [];           // JobRecord[] for current experiment
let fileCache   = {};           // run_id → [{path, size, modified}]
let checkedRuns = new Set();    // run_ids selected in sidebar
let browseRun   = null;         // run_id shown in Browse tab
let browseFile  = null;         // file path highlighted in Browse tree
let compareFile = null;         // file path shown in Compare grid

// ── Helpers ────────────────────────────────────────────────────────────────────
function fmtModified(iso) {
  if (!iso) return "";
  const d = new Date(iso), now = Date.now(), diff = now - d.getTime();
  if (diff < 60e3) return "just now";
  if (diff < 3600e3) return Math.floor(diff / 60e3) + "m ago";
  if (diff < 86400e3) return Math.floor(diff / 3600e3) + "h ago";
  if (diff < 7 * 86400e3) return Math.floor(diff / 86400e3) + "d ago";
  return d.toLocaleDateString();
}
function comboStr(job) {
  try { const c = typeof job.combo==="string" ? JSON.parse(job.combo) : (job.combo||{}); return Object.entries(c).map(([k,v])=>`${k}=${v}`).join("  "); }
  catch { return ""; }
}
function fileIcon(p) {
  const e = (p.split(".").pop()||"").toLowerCase();
  if (/^(png|jpg|jpeg|gif|svg|webp|bmp)$/.test(e)) return "🖼";
  if (/^(pth|pt|ckpt|safetensors|bin|npz|npy|gguf|onnx|pkl|pickle)$/.test(e)) return "🧠";
  if (/^(json|yaml|yml|toml|cfg|ini|conf)$/.test(e)) return "⚙";
  if (/^(csv|tsv)$/.test(e)) return "📊";
  if (/^(py|sh|bash)$/.test(e)) return "📝";
  if (/^(txt|log|md)$/.test(e)) return "📄";
  if (/^(zip|tar|gz|bz2)$/.test(e)) return "📦";
  return "📎";
}
const isImage = p => /\.(png|jpg|jpeg|gif|svg|webp|bmp)$/i.test(p);
const isText  = p => /\.(txt|log|json|yaml|yml|toml|cfg|ini|conf|csv|tsv|md|py|sh|bash|xml)$/i.test(p);

function artifactURL(expId, runId, filePath) {
  return `/api/experiments/${encodeURIComponent(expId)}/jobs/${encodeURIComponent(runId)}/artifacts/${filePath.split("/").map(encodeURIComponent).join("/")}`;
}
function runZipURL(expId, runId) {
  return authURL(`/api/experiments/${encodeURIComponent(expId)}/jobs/${encodeURIComponent(runId)}/artifacts.zip`);
}
function expZipURL(expId) {
  return authURL(`/api/experiments/${encodeURIComponent(expId)}/artifacts.zip`);
}

// ── Tab management ─────────────────────────────────────────────────────────────
let activeTab = "browse";
document.querySelectorAll(".tab").forEach(btn =>
  btn.addEventListener("click", () => switchTab(btn.dataset.tab))
);
function switchTab(name) {
  activeTab = name;
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.getElementById("compare-pane").classList.toggle("hidden", name !== "compare");
  document.getElementById("browse-pane").classList.toggle("hidden", name !== "browse");
}

// ── Sidebar: run list ──────────────────────────────────────────────────────────
function renderRunList() {
  const list = document.getElementById("run-list");
  list.innerHTML = "";
  if (!jobs.length) {
    list.innerHTML = `<div style="color:var(--text-muted);font-size:12px;padding:4px">No runs.</div>`;
    return;
  }
  for (const j of jobs) {
    const item = document.createElement("div");
    item.className = "run-item" + (j.run_id === browseRun ? " browse-hl" : "");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checkedRuns.has(j.run_id);
    cb.addEventListener("change", e => {
      e.stopPropagation();
      cb.checked ? checkedRuns.add(j.run_id) : checkedRuns.delete(j.run_id);
      if (activeTab === "compare") renderCompareGrid();
    });

    const info = document.createElement("div");
    info.className = "run-info";
    const combo = comboStr(j);
    const elapsed = fmtElapsed(j.elapsed, "");
    info.innerHTML = `<div class="run-id-text">${runNameHtml(j)}</div>` +
      (combo ? `<div class="run-combo-text">${escHtml(combo)}</div>` : "") +
      (elapsed ? `<div class="run-elapsed-text">${escHtml(elapsed)}</div>` : "");

    const bdg = document.createElement("span");
    bdg.className = `badge badge-${j.status}`;
    bdg.textContent = j.status;

    item.append(cb, info, bdg);
    item.dataset.runId = j.run_id;
    item.addEventListener("click", e => {
      if (e.target === cb) return;
      browseRunFiles(j.run_id);
    });
    list.appendChild(item);
  }
}

function browseRunFiles(runId) {
  browseRun = runId; browseFile = null;
  document.querySelectorAll(".run-item").forEach(el => el.classList.toggle("browse-hl", el.dataset.runId === runId));
  switchTab("browse");
  loadBrowse(runId);
}

function downloadURL(url, name) {
  Object.assign(document.createElement("a"), { href: authURL(url), download: name || "" }).click();
}

function setChecked(runId, on) {
  on ? checkedRuns.add(runId) : checkedRuns.delete(runId);
  renderRunList();
  if (activeTab === "compare") renderCompareGrid();
}

function compareAcrossRuns(path) {
  compareFile = path;
  rebuildFileSel();
  document.getElementById("file-path-sel").value = path;
  switchTab("compare");
  renderCompareGrid();
}

// A run's menu, from the sidebar or a compare card.  path: the file the card shows.
function runMenu(e, j, path) {
  const eid = expId(), q = { experiment: eid, run: j.run_id };
  const go = page => () => { location.href = pageHref(page, q); };
  const checked = checkedRuns.has(j.run_id);
  mlContextMenu(e, [
    { title: j.label || j.run_id },
    path ? { label: "Download this file", run: () => downloadURL(artifactURL(eid, j.run_id, path), path.split("/").pop()) } : null,
    path ? { label: "Open this file in a new tab", run: () => window.open(authURL(artifactURL(eid, j.run_id, path)), "_blank", "noopener") } : null,
    path ? "-" : null,
    { label: "Browse files", run: () => browseRunFiles(j.run_id) },
    { label: checked ? "Leave out of the comparison" : "Add to the comparison", run: () => setChecked(j.run_id, !checked) },
    { label: "Download all files (.zip)", run: () => downloadURL(runZipURL(eid, j.run_id), `${j.run_id}.zip`) },
    "-",
    { label: "Open logs", run: go("logs.html") },
    j.artifact_id ? { label: "Open code", run: go("code.html") } : null,
    { label: "Copy run ID", run: () => copyText(j.run_id, "Copied the run ID") },
  ]);
}

document.getElementById("run-list").addEventListener("contextmenu", e => {
  const item = e.target.closest(".run-item");
  const j = item && jobs.find(x => x.run_id === item.dataset.runId);
  if (j && !mlNativeMenuWanted(e, item)) runMenu(e, j, null);
});
document.getElementById("compare-grid").addEventListener("contextmenu", e => {
  const card = e.target.closest(".run-card");
  const j = card && jobs.find(x => x.run_id === card.dataset.runId);
  if (!j || mlNativeMenuWanted(e, card)) return;
  const found = (fileCache[j.run_id] || []).some(f => f.path === compareFile);
  runMenu(e, j, found ? compareFile : null);
});
document.getElementById("file-tree").addEventListener("contextmenu", e => {
  const row = e.target.closest(".file-row");
  if (!row || !browseRun || mlNativeMenuWanted(e, row)) return;
  const runId = browseRun, path = row.dataset.path, eid = expId();
  const f = (fileCache[runId] || []).find(x => x.path === path);
  mlContextMenu(e, [
    { title: path },
    f && browseFile !== path ? { label: "Preview", run: () => row.click() } : null,
    { label: "Download", run: () => downloadURL(artifactURL(eid, runId, path), path.split("/").pop()) },
    { label: "Open in a new tab", run: () => window.open(authURL(artifactURL(eid, runId, path)), "_blank", "noopener") },
    { label: "Compare across runs", run: () => compareAcrossRuns(path) },
    "-",
    { label: "Copy path", run: () => copyText(path, "Copied the path") },
  ]);
});

document.getElementById("sel-all").onclick = () => {
  jobs.forEach(j => checkedRuns.add(j.run_id)); renderRunList();
  if (activeTab === "compare") renderCompareGrid();
};
document.getElementById("sel-none").onclick = () => {
  checkedRuns.clear(); renderRunList();
  if (activeTab === "compare") renderCompareGrid();
};
document.getElementById("sel-done").onclick = () => {
  checkedRuns.clear();
  jobs.filter(j => j.status === "done").forEach(j => checkedRuns.add(j.run_id));
  renderRunList(); if (activeTab === "compare") renderCompareGrid();
};
document.getElementById("sel-failed").onclick = () => {
  const FAILED_STATUSES = new Set(["failed", "xfailed", "cancelled"]);
  checkedRuns.clear();
  jobs.filter(j => FAILED_STATUSES.has(j.status)).forEach(j => checkedRuns.add(j.run_id));
  renderRunList(); if (activeTab === "compare") renderCompareGrid();
};

// ── File list loading ──────────────────────────────────────────────────────────
function expId() { return document.getElementById("exp-sel").value; }

async function fetchFileList(runId) {
  if (fileCache[runId] !== undefined) return fileCache[runId];
  try {
    const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId())}/jobs/${encodeURIComponent(runId)}/artifacts`);
    fileCache[runId] = r.ok ? await r.json() : [];
  } catch { fileCache[runId] = []; }
  return fileCache[runId];
}

async function fetchFileLists(runIds) {
  await Promise.all(runIds.map(id => fetchFileList(id)));
}

// ── File path selector ─────────────────────────────────────────────────────────
function rebuildFileSel() {
  const all = new Set();
  for (const fs of Object.values(fileCache)) for (const f of fs) all.add(f.path);

  const sel = document.getElementById("file-path-sel");
  const prev = sel.value;
  sel.innerHTML = `<option value="">— select file path —</option>`;
  for (const p of [...all].sort()) {
    const opt = document.createElement("option");
    opt.value = p; opt.textContent = p;
    sel.appendChild(opt);
  }
  const target = prev || compareFile;
  if (target && all.has(target)) sel.value = target;
}

document.getElementById("file-path-sel").addEventListener("change", e => {
  compareFile = e.target.value || null;
  renderCompareGrid();
});
document.getElementById("only-with-file").addEventListener("change", () => renderCompareGrid());

// ── Compare grid ───────────────────────────────────────────────────────────────
async function renderCompareGrid() {
  const grid = document.getElementById("compare-grid");
  const info = document.getElementById("compare-info");

  if (!compareFile) {
    grid.innerHTML = `<div class="placeholder">Choose a file path above to compare across runs.</div>`;
    info.textContent = ""; return;
  }

  const selected = jobs.filter(j => checkedRuns.has(j.run_id));
  if (!selected.length) {
    grid.innerHTML = `<div class="placeholder">No runs selected — check some runs in the sidebar.</div>`;
    info.textContent = ""; return;
  }

  await fetchFileLists(selected.map(j => j.run_id));

  const hasFile  = j => (fileCache[j.run_id]||[]).some(f => f.path === compareFile);
  const onlyWith = document.getElementById("only-with-file").checked;
  const toShow   = onlyWith ? selected.filter(hasFile) : selected;
  const withCnt  = selected.filter(hasFile).length;

  info.textContent = `${withCnt} / ${selected.length} runs have this file`;

  if (!toShow.length) {
    grid.innerHTML = `<div class="placeholder">No selected runs have this file.</div>`;
    return;
  }

  grid.innerHTML = "";
  for (const j of toShow) {
    const found = hasFile(j);
    const combo = comboStr(j);
    const card  = document.createElement("div");
    card.className = "run-card";
    card.dataset.runId = j.run_id;

    const dlHref = found ? escHtml(authURL(artifactURL(expId(), j.run_id, compareFile))) : "";
    const fname  = compareFile.split("/").pop();

    card.innerHTML = `
      <div class="card-header">
        <span class="card-run-id" title="${escHtml(j.run_id)}">${runNameHtml(j)}</span>
        <span class="badge badge-${escHtml(j.status)}">${escHtml(j.status)}</span>
      </div>
      ${combo ? `<div class="card-combo" title="${escHtml(combo)}">${escHtml(combo)}</div>` : ""}
      <div class="card-body"></div>
      <div class="card-footer">
        ${found ? `<a class="btn-sm" href="${dlHref}" download="${escHtml(fname)}">Download</a>` : ""}
      </div>`;

    const body = card.querySelector(".card-body");
    if (found) renderPreviewInto(expId(), j.run_id, compareFile, body, true);
    else body.innerHTML = `<span class="no-file">—</span>`;

    grid.appendChild(card);
  }
}

// ── Preview rendering ──────────────────────────────────────────────────────────
function renderPreviewInto(eid, runId, filePath, el, compact) {
  const url = artifactURL(eid, runId, filePath);
  el.innerHTML = "";

  if (isImage(filePath)) {
    const img = document.createElement("img");
    img.src = authURL(url);
    img.alt = filePath;
    img.addEventListener("click", () => openLightbox(authURL(url)));
    el.appendChild(img);
  } else if (isText(filePath)) {
    el.innerHTML = `<pre style="opacity:.4">Loading…</pre>`;
    apiFetch(url).then(r => r.ok ? r.text() : null).then(text => {
      if (text === null) { el.innerHTML = `<span class="no-file">Failed to load</span>`; return; }
      const lines = text.split("\n");
      const maxL = compact ? 40 : Infinity;
      const pre = document.createElement("pre");
      pre.textContent = lines.slice(0, maxL).join("\n") + (lines.length > maxL ? `\n… (${lines.length} lines)` : "");
      el.innerHTML = ""; el.appendChild(pre);
    }).catch(() => { el.innerHTML = `<span class="no-file">Error loading file</span>`; });
  } else if (MLModelPreview.isModel(filePath)) {
    MLModelPreview.render(url, el, compact);
  } else {
    el.innerHTML = `<span class="no-file" style="font-size:11px">Binary — use Download</span>`;
  }
}

// ── Lightbox ───────────────────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.remove("hidden");
}
document.getElementById("lightbox").addEventListener("click", () =>
  document.getElementById("lightbox").classList.add("hidden")
);
document.addEventListener("keydown", e => {
  if (e.key === "Escape") document.getElementById("lightbox").classList.add("hidden");
});

// ── Browse tab ─────────────────────────────────────────────────────────────────
async function loadBrowse(runId) {
  const tree = document.getElementById("file-tree");
  const pane = document.getElementById("preview-pane");
  const toolbar = document.getElementById("browse-toolbar");
  toolbar.style.display = "flex";
  document.getElementById("run-dl-zip").href = runZipURL(expId(), runId);
  const _browseJob = jobs.find(j => j.run_id === runId);
  document.getElementById("browse-run-label").innerHTML = _browseJob ? runNameHtml(_browseJob) : escHtml(runId);
  const codeBtn = document.getElementById("run-code-btn");
  codeBtn.href = pageHref("code.html", { experiment: expId(), run: runId });
  codeBtn.style.display = _browseJob && _browseJob.artifact_id ? "" : "none";
  tree.innerHTML = `<div style="padding:8px 12px;font-size:11px;color:var(--text-muted)">Loading…</div>`;
  pane.innerHTML = `<div class="placeholder">Loading files…</div>`;

  const files = await fetchFileList(runId);
  rebuildFileSel();

  tree.innerHTML = "";
  if (!files.length) {
    tree.innerHTML = `<div style="padding:8px 12px;font-size:11px;color:var(--text-muted)">No artifact files.</div>`;
    pane.innerHTML = `<div class="placeholder">This run has no artifact files.</div>`;
    return;
  }

  pane.innerHTML = `<div class="placeholder">Select a file on the left to preview it.</div>`;

  for (const f of files) {
    const row = document.createElement("div");
    row.className = "file-row" + (f.path === browseFile ? " selected" : "");
    row.dataset.path = f.path;
    row.innerHTML = `<span class="file-icon">${fileIcon(f.path)}</span><span class="file-name" title="${escHtml(f.path)}">${escHtml(f.path)}</span><span class="file-size">${fmtSize(f.size)}</span><span class="file-modified">${fmtModified(f.modified)}</span>`;
    row.addEventListener("click", () => {
      browseFile = f.path;
      tree.querySelectorAll(".file-row").forEach(r => r.classList.remove("selected"));
      row.classList.add("selected");
      showBrowsePreview(runId, f);
    });
    tree.appendChild(row);
  }

  // Auto-open browseFile if set (e.g. from INIT_FILE)
  if (browseFile) {
    const f = files.find(f => f.path === browseFile);
    if (f) {
      tree.querySelector(`[data-path="${CSS.escape(browseFile)}"]`)?.classList.add("selected");
      showBrowsePreview(runId, f);
    }
  }
}

function showBrowsePreview(runId, file) {
  const pane = document.getElementById("preview-pane");
  const eid  = expId();
  const url  = artifactURL(eid, runId, file.path);
  const fname = file.path.split("/").pop();

  pane.innerHTML = `
    <div class="preview-header">
      <span class="preview-filename">${escHtml(file.path)}</span>
      <span class="preview-meta">${fmtSize(file.size)}${file.modified ? ` · ${fmtModified(file.modified)}` : ""}</span>
      <a class="btn-sm" href="${escHtml(authURL(url))}" download="${escHtml(fname)}">Download</a>
    </div>
    <div id="preview-content"></div>`;

  renderPreviewInto(eid, runId, file.path, document.getElementById("preview-content"), false);
}

// ── Experiment loading ─────────────────────────────────────────────────────────
let loadedExp = null;  // experiment currently shown

async function loadExperiment(eid) {
  if (!eid) return;
  loadedExp = eid;
  jobs = []; fileCache = {}; checkedRuns.clear();
  browseRun = null; browseFile = null; compareFile = null;

  document.getElementById("file-path-sel").innerHTML = `<option value="">— select file path —</option>`;
  document.getElementById("compare-info").textContent = "";
  document.getElementById("compare-grid").innerHTML = `<div class="placeholder">Loading runs…</div>`;
  document.getElementById("browse-toolbar").style.display = "none";
  document.getElementById("file-tree").innerHTML = "";
  document.getElementById("preview-pane").innerHTML = `<div class="placeholder">Click a run in the sidebar, then select a file.</div>`;
  document.getElementById("run-list").innerHTML = `<div style="color:var(--text-muted);font-size:12px;padding:4px">Loading…</div>`;

  // Keep Results nav link in sync
  document.getElementById("nav-results").href = pageHref("results.html", { experiment: eid });
  MLCampaign.linkNav();

  const expDl = document.getElementById("exp-dl-all");
  expDl.href = expZipURL(eid);
  expDl.style.display = "flex";

  try {
    const r = await apiFetch(`/api/experiments/${encodeURIComponent(eid)}/jobs`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    jobs = await r.json();
    setConnStatus("connected");
  } catch (e) {
    setConnStatus("disconnected");
    document.getElementById("run-list").innerHTML = `<div style="color:var(--bad);font-size:12px;padding:4px">Error loading runs</div>`;
    return;
  }

  // Default selection: done runs
  jobs.filter(j => j.status === "done").forEach(j => checkedRuns.add(j.run_id));
  renderRunList();

  // Apply init params once
  if (!_initDone) {
    _initDone = true;
    if (INIT_RUN && jobs.find(j => j.run_id === INIT_RUN)) {
      browseRun = INIT_RUN;
      switchTab("browse");
      loadBrowse(INIT_RUN);
    } else if (INIT_FILE) {
      compareFile = INIT_FILE;
    }
  }

  // Eagerly fetch file lists for terminal runs, then populate selector + grid
  const doneIds = jobs.filter(j => ["done","pruned","failed","xfailed","cancelled"].includes(j.status)).map(j => j.run_id);
  fetchFileLists(doneIds).then(() => {
    rebuildFileSel();
    if (activeTab === "compare") {
      if (compareFile) {
        document.getElementById("file-path-sel").value = compareFile;
        renderCompareGrid();
      } else {
        document.getElementById("compare-grid").innerHTML = `<div class="placeholder">Choose a file path above to compare across runs.</div>`;
      }
    }
  });
}

// ── Campaign and experiment selectors ──────────────────────────────────────────
const campaignSel = document.getElementById("campaign-sel");
let allExps = [];      // every experiment, for the campaign list

function clearExperiment() {
  loadedExp = null;
  jobs = []; fileCache = {}; checkedRuns.clear();
  browseRun = null; browseFile = null; compareFile = null;
  document.getElementById("exp-dl-all").style.display = "none";
  document.getElementById("file-path-sel").innerHTML = `<option value="">— select file path —</option>`;
  document.getElementById("compare-info").textContent = "";
  document.getElementById("compare-grid").innerHTML = `<div class="placeholder">No experiments in this campaign.</div>`;
  document.getElementById("browse-toolbar").style.display = "none";
  document.getElementById("file-tree").innerHTML = "";
  document.getElementById("preview-pane").innerHTML = `<div class="placeholder">No experiments in this campaign.</div>`;
  document.getElementById("run-list").innerHTML = `<div style="color:var(--text-muted);font-size:12px;padding:4px">No experiments in this campaign</div>`;
}

// Fill the experiment select from the selected campaign, keeping the shown
// experiment when it is still listed, and load the selection if it changed.
function showCampaign() {
  MLCampaign.fill(campaignSel, allExps);
  const id = MLCampaign.fillExperiments(document.getElementById("exp-sel"),
                                        MLCampaign.filter(allExps), loadedExp || INIT_EXP);
  if (!id) clearExperiment();
  else if (id !== loadedExp) loadExperiment(id);
}

async function fetchExperiments() {
  try {
    const r = await apiFetch("/api/experiments");
    if (!r.ok) { setConnStatus("disconnected"); return; }
    allExps = await r.json();
    setConnStatus("connected");
    MLCampaign.follow(allExps, INIT_EXP);
    showCampaign();
  } catch { setConnStatus("disconnected"); }
}

MLCampaign.bind(campaignSel, showCampaign);

document.getElementById("exp-sel").addEventListener("change", e => loadExperiment(e.target.value));
document.getElementById("refresh-btn").addEventListener("click", fetchExperiments);

document.getElementById("run-dl-zip").addEventListener("click", async e => {
  if (browseRun && fileCache[browseRun]?.length === 0) {
    e.preventDefault();
    if (await mlConfirm("This run has no artifact files. Download anyway?", { confirmLabel: "Download", danger: false }))
      window.location.href = e.currentTarget.href;
  }
});

document.getElementById("exp-dl-all").addEventListener("click", async e => {
  const cached = Object.values(fileCache);
  if (cached.length > 0 && !cached.some(fs => fs.length > 0)) {
    e.preventDefault();
    if (await mlConfirm("No runs in this experiment have artifact files. Download anyway?", { confirmLabel: "Download", danger: false }))
      window.location.href = e.currentTarget.href;
  }
});

fetchExperiments();
