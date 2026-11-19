const _urlParams = new URLSearchParams(location.search);
document.addEventListener("mlthemechange", () => { if (DATA) buildActiveChart(); });
const INIT_EXP = _urlParams.get("experiment") || "";
const INIT_RUN = _urlParams.get("run") || "";

// ── Constants & state (from visualize.py verbatim) ────────────────────────────

const PALETTE_LIGHT = [
  "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed",
  "#a16207", "#db2777", "#475569", "#65a30d", "#0891b2",
  "#60a5fa", "#fb923c", "#4ade80", "#f87171", "#a78bfa",
];
const PALETTE_DARK = [
  "#60a5fa", "#fb923c", "#4ade80", "#f87171", "#c084fc",
  "#facc15", "#f472b6", "#94a3b8", "#a3e635", "#22d3ee",
  "#93c5fd", "#fdba74", "#86efac", "#fca5a5", "#d8b4fe",
];
const currentPalette = () =>
  document.documentElement.getAttribute("data-theme") === "dark" ? PALETTE_DARK : PALETTE_LIGHT;

let DATA         = null;  // experiment metadata (dims, runs, metricNames, subDims)
let METRIC_CACHE = {};    // metric_name → {run_hash: {steps, values}}, loaded on demand
let colorBy      = null;  // dim name to color by (curves tab)
let metric       = "loss";
let smoothAlpha  = 0;     // EMA alpha; 0 = disabled
let filters      = {};    // dim → Set of visible values
let METRIC_NAMES = [];    // populated by buildMetricSelect

let activeTab    = "curves";
const dirtyTabs  = new Set(["curves", "sensitivity", "optimization"]);
let optMinimize  = true;   // true = lower metric is better
let selectedRun  = INIT_RUN || null;  // run highlighted in the curves chart
let hiddenRuns   = new Set();         // runs hidden with the highlight bar's Hide
let _chartRunHashes = [];             // line-trace order in chart-curves
let JOBS         = new Map();         // run_id → job record, for status and the step timer

// ── Live state (SSE replaced with WS) ────────────────────────────────────────
let LIVE_RUNS      = new Set();  // run hashes currently status=running
let _ws            = null;       // WebSocket for live updates
// run_hash → line-trace index in chart-curves (populated by buildCurvesChart)
let _traceIndexMap = {};
// last EMA accumulator: {metric: {run_hash: s_prev}} for incremental smoothing
let LAST_EMA_S     = {};
let _pendingMetrics = {};   // runId → metricName → {steps, values} accumulated between RAF flushes
let _rafPending     = false;

// ── Tab management (from visualize.py verbatim) ───────────────────────────────

function renderTab(name) {
  if (!DATA) return;
  if (name !== "optimization" && !METRIC_CACHE[metric]) return;
  dirtyTabs.delete(name);
  if (name === "curves") buildCurvesChart();
  else if (name === "sensitivity") buildSensitivityChart();
  else if (name === "optimization") buildOptimizationChart();
}

function setTab(name) {
  activeTab = name;
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".curves-only").forEach(el =>
    el.style.display = (name === "curves") ? "" : "none");
  document.getElementById("opt-minimize-label").style.display =
    name === "optimization" ? "flex" : "none";
  renderRunBar();
  document.querySelectorAll(".chart-pane").forEach(el => el.style.display = "none");
  const pane = document.getElementById(`chart-${name}`);
  pane.style.display = "";
  if (dirtyTabs.has(name)) renderTab(name);
  else if (name === "optimization") {
    try { Plotly.relayout("opt-bestsofar", {}); } catch {}
    document.querySelectorAll(".opt-scatter-panel").forEach(el => { try { Plotly.relayout(el, {}); } catch {} });
  } else {
    Plotly.relayout(pane, {});
  }
}

// Mark all tabs dirty and re-render the active one. Called when data changes.
function buildActiveChart() {
  if (typeof Plotly === "undefined") return;  // Plotly not yet loaded
  dirtyTabs.add("curves");
  dirtyTabs.add("sensitivity");
  dirtyTabs.add("optimization");
  renderTab(activeTab);
}



// ── Shared helpers (from visualize.py verbatim) ───────────────────────────────

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Exponential moving average; treats null/non-finite values as gaps.
function ema(vals, alpha) {
  if (alpha === 0) return vals;
  const out = [];
  let s = null;
  for (const v of vals) {
    if (v === null || !isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : alpha * s + (1 - alpha) * v;
    out.push(s);
  }
  return out;
}

// Map an array of dim values to palette colors.
function colorMap(dimValues) {
  const m = {};
  const palette = currentPalette();
  dimValues.forEach((v, i) => { m[v] = palette[i % palette.length]; });
  return m;
}

// Interpolate blue (#1565C0) → red (#C62828).
function lerpColor(t) {
  const r = Math.round(0x15 + (0xC6 - 0x15) * t);
  const g = Math.round(0x65 + (0x28 - 0x65) * t);
  const b = Math.round(0xC0 + (0x28 - 0xC0) * t);
  return `rgb(${r},${g},${b})`;
}

// Return a run-ID → color map based on each run's final value of metricName.
function metricColorsForRuns(runs, metricName) {
  const cache = METRIC_CACHE[metricName] || {};
  const finals = runs.map(r => {
    const vals = (cache[r.hash] || {}).values || [];
    for (let i = vals.length - 1; i >= 0; i--)
      if (vals[i] !== null && isFinite(vals[i])) return vals[i];
    return null;
  });
  const finite = finals.filter(v => v !== null);
  if (!finite.length) return new Map(runs.map(r => [r.hash, "#888"]));
  const lo = Math.min(...finite), hi = Math.max(...finite);
  return new Map(runs.map((r, i) => {
    const v = finals[i];
    const t = v === null ? 0.5 : (hi === lo ? 0.5 : (v - lo) / (hi - lo));
    return [r.hash, lerpColor(t)];
  }));
}

function orderedComboEntries(combo, skipDim) {
  return Object.keys(DATA.dims)
    .filter(k => k !== skipDim && k in combo)
    .map(k => [k, combo[k]]);
}

function comboLabel(combo, skipDim) {
  return orderedComboEntries(combo, skipDim)
    .map(([k, v]) => `${k}=${v}`)
    .join("  ");
}

function hoverText(combo) {
  return orderedComboEntries(combo, null)
    .map(([k, v]) => `<b>${escHtml(k)}</b>: ${escHtml(v)}`)
    .join("<br>");
}

function visibleRuns() {
  return DATA.runs.filter(r => !hiddenRuns.has(r.hash) &&
    Object.entries(filters).every(([dim, vals]) => !(dim in r.combo) || vals.has(r.combo[dim]))
  );
}

function yRangeOf(runs) {
  const cache = METRIC_CACHE[metric] || {};
  let lo = Infinity, hi = -Infinity;
  for (const r of runs)
    for (const v of ((cache[r.hash] || {}).values || []))
      if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) return undefined;
  const pad = Math.max((hi - lo) * 0.05, 1e-6);
  return [lo - pad, hi + pad];
}

// ── Shared helpers for analysis tabs (from visualize.py verbatim) ─────────────

// Return the final logged value for a run from the currently-loaded metric.
function runScalar(runHash) {
  const cache = METRIC_CACHE[metric] || {};
  const vals = (cache[runHash]?.values || []).filter(v => v !== null && isFinite(v));
  return vals.length ? vals[vals.length - 1] : null;
}

// For each dim, compute per-value means and the overall effect size
// (max_mean - min_mean). Returns array sorted descending by effect size.
function computeDimEffects(runs) {
  return Object.entries(DATA.dims).map(([dim, values]) => {
    const valueMeans = values.map(val => {
      const matching = runs.filter(r => r.combo[dim] === val);
      const scalars = matching.map(r => runScalar(r.hash)).filter(v => v !== null);
      const mean = scalars.length ? scalars.reduce((a, b) => a + b, 0) / scalars.length : null;
      return { val, mean, scalars, count: scalars.length };
    }).filter(vm => vm.mean !== null);
    const means = valueMeans.map(vm => vm.mean);
    const effectSize = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    return { dim, effectSize, valueMeans };
  }).sort((a, b) => b.effectSize - a.effectSize);
}

// ── Tab: Curves (from visualize.py verbatim) ──────────────────────────────────

function buildCurvesChart() {
  let runs = visibleRuns();

  // When coloring by a subdim, hide runs that don't have that dim.
  const subDimInfo = DATA.subDims || {};
  if (!colorBy.startsWith("_m:") && subDimInfo[colorBy])
    runs = runs.filter(r => colorBy in r.combo);

  const isMetricColor = colorBy.startsWith("_m:");
  let getColor, getGroup, showLegendFor, legendGroupTitle;

  if (isMetricColor) {
    const colorMap_ = metricColorsForRuns(DATA.runs, colorBy.slice(3));
    getColor        = r     => colorMap_.get(r.hash) || "#888";
    getGroup        = r     => r.hash;
    showLegendFor   = ()    => false;
    legendGroupTitle = ()   => undefined;
  } else {
    const cmap      = colorMap(DATA.dims[colorBy] || []);
    const firstSeen = new Set();
    getColor  = r => cmap[r.combo[colorBy]] || "#888";
    getGroup  = r => String(r.combo[colorBy]);
    showLegendFor   = group => { const f = !firstSeen.has(group); if (f) firstSeen.add(group); return f; };
    legendGroupTitle = (group, isFirst) => isFirst ? { text: colorBy, font: { size: 11 } } : undefined;
  }

  const plotBg   = cssVar("--plot-bg");
  const plotGrid = cssVar("--plot-grid");
  const plotText = cssVar("--plot-text");

  const mcache = METRIC_CACHE[metric] || {};
  const lineTraces = runs.map(r => {
    const color   = getColor(r);
    const group   = getGroup(r);
    const isFirst = showLegendFor(group);
    const mdata   = mcache[r.hash] || {};
    return {
      x: mdata.steps  || [],
      y: ema(mdata.values || [], smoothAlpha),
      mode: "lines",
      line: { color, width: 1.5 },
      name: group,
      legendgroup: group,
      legendgrouptitle: legendGroupTitle(group, isFirst),
      showlegend: isFirst,
      hovertemplate: `<b>${escHtml(r.name)}</b><br>` + hoverText(r.combo) + "<br><b>step</b>: %{x}<br><b>" + metric + "</b>: %{y:.4f}<extra></extra>",
      customdata: [comboLabel(r.combo, colorBy)],
    };
  });

  // One dot per run at the final data point.
  const dotTraces = runs.map(r => {
    const color = getColor(r);
    const group = getGroup(r);
    const mdata = mcache[r.hash] || {};
    const steps  = mdata.steps  || [];
    const vals   = ema(mdata.values || [], smoothAlpha);
    const lastX  = steps.length  ? [steps[steps.length - 1]]  : [];
    const lastY  = vals.length   ? [vals[vals.length - 1]]    : [];
    return {
      x: lastX, y: lastY,
      mode: "markers",
      marker: { color, size: 6, line: { color: plotBg, width: 1.5 } },
      legendgroup: group,
      showlegend: false,
      hoverinfo: "skip",
    };
  });

  const traces = [...lineTraces, ...dotTraces];

  Plotly.react("chart-curves", traces, {
    margin: { t: 20, r: 20, b: 50, l: 60 },
    xaxis: { title: "step", gridcolor: plotGrid, color: plotText },
    yaxis: { title: metric, gridcolor: plotGrid, color: plotText, autorange: true },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { groupclick: "toggleitem", font: { size: 11, color: plotText } },
    hovermode: "closest",
  }, { responsive: true, scrollZoom: true, displaylogo: false });

  // Capture trace-index map for incremental WS updates.
  _traceIndexMap = {};
  runs.forEach((r, i) => { _traceIndexMap[r.hash] = i; });
  _chartRunHashes = runs.map(r => r.hash);
  if (selectedRun && !_chartRunHashes.includes(selectedRun)) selectedRun = null;
  bindCurveEvents();
  applyRunSelection();

  // Resync EMA state from full cache (after a full rebuild, accumulator is reset).
  if (!LAST_EMA_S[metric]) LAST_EMA_S[metric] = {};
  for (const r of runs) {
    if (smoothAlpha === 0) { LAST_EMA_S[metric][r.hash] = null; continue; }
    const vals = (mcache[r.hash] || {}).values || [];
    let s = null;
    for (const v of vals) {
      if (v === null || !isFinite(v)) continue;
      s = s === null ? v : smoothAlpha * s + (1 - smoothAlpha) * v;
    }
    LAST_EMA_S[metric][r.hash] = s;
  }

  const liveCount = runs.filter(r => LIVE_RUNS.has(r.hash)).length;
  document.getElementById("status").innerHTML =
    `${runs.length} / ${DATA.runs.length} runs visible` +
    (liveCount ? `  •  ${liveCount} live` : "") + hiddenNote();
}

// ── Highlight and click-to-filter (curves tab) ───────────────────────────────
// Click a curve to highlight its run; click it again, click empty chart space,
// or press Escape to clear.  The bar above the chart names the highlighted run;
// its settings filter to that value and Hide drops the run.  A legend entry
// toggles its value in the sidebar filter, and a double-click shows only it.

function setSelectedRun(hash) {
  selectedRun = hash;
  applyRunSelection();
}

function applyRunSelection() {
  const n = _chartRunHashes.length;
  const gd = document.getElementById("chart-curves");
  if (n && gd.data && gd.data.length === 2 * n) {
    const on = i => !selectedRun || _chartRunHashes[i] === selectedRun;
    const idx = [...Array(2 * n).keys()];
    Plotly.restyle(gd, {
      opacity: idx.map(i => on(i % n) ? 1 : 0.12),
      "line.width": idx.map(i => selectedRun && on(i % n) ? 2.6 : 1.5),
    }, idx);
  }
  renderRunBar();
}

function bindCurveEvents() {
  const gd = document.getElementById("chart-curves");
  if (gd._mlBound || !gd.on) return;
  gd._mlBound = true;
  let lastCurveClick = 0, downAt = null;
  gd.on("plotly_click", e => {
    const pt = e.points && e.points[0];
    if (!pt) return;
    lastCurveClick = Date.now();
    const hash = _chartRunHashes[pt.curveNumber % _chartRunHashes.length];
    setSelectedRun(hash === selectedRun ? null : hash);
  });
  // A click that hit no curve (and was not the end of a zoom drag) clears.
  gd.addEventListener("mousedown", e => { downAt = [e.clientX, e.clientY]; });
  gd.addEventListener("click", e => {
    const moved = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4;
    if (moved || e.target.closest(".modebar, .legend")) return;
    setTimeout(() => { if (Date.now() - lastCurveClick > 300 && selectedRun) setSelectedRun(null); }, 0);
  });
  gd.on("plotly_legendclick", e => {
    const val = e.data[e.curveNumber]?.legendgroup;
    if (val === undefined || colorBy.startsWith("_m:")) return false;
    const set = filters[colorBy];
    if (set.has(val)) set.delete(val); else set.add(val);
    filtersChanged();
    return false;
  });
  gd.on("plotly_legenddoubleclick", e => {
    const val = e.data[e.curveNumber]?.legendgroup;
    if (val === undefined || colorBy.startsWith("_m:")) return false;
    showOnly(colorBy, val);
    return false;
  });
}

// Show only runs whose *dim* is *val*; again on an already-isolated value shows all.
function showOnly(dim, val) {
  const set = filters[dim];
  const isolated = set.size === 1 && set.has(val);
  filters[dim] = isolated ? new Set(DATA.dims[dim]) : new Set([val]);
  filtersChanged();
}

function filtersChanged() {
  buildFilters();
  buildActiveChart();
}

function hideRun(hash) {
  hiddenRuns.add(hash);
  if (selectedRun === hash) selectedRun = null;
  buildActiveChart();
}

function showHiddenRuns() {
  hiddenRuns = new Set();
  buildActiveChart();
}

function runLink(page, hash) {
  return MLCampaign.href(pageHref(`${page}.html`, { experiment: DATA.experiment, run: hash }));
}

function renderRunBar() {
  const bar = document.getElementById("run-bar");
  const r = selectedRun && activeTab === "curves" && DATA ? DATA.runs.find(x => x.hash === selectedRun) : null;
  bar.style.display = r ? "" : "none";
  if (!r) { MLStepTimer.tick(); return; }
  const color = document.getElementById("chart-curves").data?.[_chartRunHashes.indexOf(r.hash)]?.line?.color || "#888";
  const job = JOBS.get(r.hash);
  const pills = orderedComboEntries(r.combo, null).map(([k, v]) =>
    `<button class="rb-pill" title="Show only runs with ${escHtml(k)}=${escHtml(v)} (again to show all)" ${act("showOnly", k, v)}>`
    + `<span class="rb-k">${escHtml(k)}</span>=${escHtml(v)}</button>`).join("");
  document.getElementById("run-bar-main").innerHTML =
    `<span class="color-swatch" style="background:${escHtml(color)}"></span>`
    + `<span class="rb-name" title="${escHtml(r.hash)}">${escHtml(r.name)}</span>`
    + (job ? badge(job.status) : "")
    + `<span class="rb-pills">${pills}</span>`;
  document.getElementById("run-bar-links").innerHTML =
    `<a class="btn-link" href="${escHtml(runLink("logs", r.hash))}">Logs</a>`
    + `<a class="btn-link" href="${escHtml(runLink("files", r.hash))}">Artifacts</a>`
    + `<button class="btn-link" title="Hide this run from the charts" ${act("hideRun", r.hash)}>Hide</button>`
    + `<button class="btn-link" title="Clear the highlight (Esc)" ${act("setSelectedRun", null)}>✕</button>`;
  MLStepTimer.tick();
}

function hiddenNote() {
  return hiddenRuns.size
    ? `  ·  <button class="link-btn" title="Show the hidden runs again" ${act("showHiddenRuns")}>${hiddenRuns.size} hidden, show</button>`
    : "";
}

document.addEventListener("keydown", e => {
  if (e.key === "Escape" && selectedRun && !e.target.closest("input, textarea, select")) setSelectedRun(null);
});

// ── Tab: Sensitivity (from visualize.py verbatim) ────────────────────────────

function buildSensitivityChart() {
  const runs    = visibleRuns();
  const plotBg  = cssVar("--plot-bg");
  const plotText = cssVar("--plot-text");
  const plotGrid = cssVar("--plot-grid");

  const effects = computeDimEffects(runs);
  if (!effects.length) {
    Plotly.react("chart-sensitivity", [], { paper_bgcolor: plotBg, plot_bgcolor: plotBg }, { responsive: true, displaylogo: false });
    document.getElementById("status").textContent = "No data";
    return;
  }

  const traces = [];

  // Dumbbell connector lines: from min_mean to max_mean for each dim.
  effects.forEach(e => {
    if (e.valueMeans.length < 2) return;
    const xVals = e.valueMeans.map(vm => vm.mean);
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [xMin, xMax],
      y: [e.dim, e.dim],
      line: { color: cssVar("--text-dim"), width: 2 },
      showlegend: false,
      hoverinfo: "skip",
    });
  });

  // One scatter trace per unique value label for consistent cross-dim coloring.
  const allLabels = [...new Set(
    effects.flatMap(e => e.valueMeans.map(vm => String(vm.val)))
  )];
  const palette = currentPalette();
  const labelColorMap = Object.fromEntries(allLabels.map((l, i) => [l, palette[i % palette.length]]));

  allLabels.forEach(label => {
    const xs = [], ys = [], customdata = [];
    effects.forEach(e => {
      const vm = e.valueMeans.find(v => String(v.val) === label);
      if (!vm) return;
      xs.push(vm.mean);
      ys.push(e.dim);
      customdata.push({
        dim: e.dim, val: label, count: vm.count,
        effectSize: e.effectSize.toFixed(4),
      });
    });
    if (!xs.length) return;
    traces.push({
      type: "scatter",
      mode: "markers",
      x: xs,
      y: ys,
      name: label,
      marker: {
        size: 12,
        color: labelColorMap[label],
        line: { color: plotBg, width: 2 },
      },
      customdata,
      hovertemplate:
        "<b>%{customdata.dim} = %{customdata.val}</b><br>" +
        `Mean ${metric}: %{x:.4f}<br>` +
        "n=%{customdata.count}<br>" +
        "Effect size: %{customdata.effectSize}<extra></extra>",
    });
  });

  // Effect size text annotations on the right.
  const allMeans = effects.flatMap(e => e.valueMeans.map(vm => vm.mean));
  const xMax = allMeans.length ? Math.max(...allMeans) : 1;
  const xMin = allMeans.length ? Math.min(...allMeans) : 0;
  const xPad = Math.max((xMax - xMin) * 0.22, 1e-6);

  const annotations = effects.map(e => ({
    x: xMax + xPad * 0.05,
    y: e.dim,
    xanchor: "left",
    yanchor: "middle",
    text: `Δ${e.effectSize.toFixed(4)}`,
    showarrow: false,
    font: { size: 10, color: plotText },
  }));

  Plotly.react("chart-sensitivity", traces, {
    margin: { t: 50, r: 20, b: 60, l: 110 },
    xaxis: {
      title: `Mean ${metric} (final value)`,
      gridcolor: plotGrid, color: plotText,
      range: [xMin - xPad * 0.1, xMax + xPad],
    },
    yaxis: {
      autorange: "reversed",   // top = highest effect size
      color: plotText,
      tickfont: { size: 11 },
    },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { font: { size: 10, color: plotText } },
    annotations,
    title: {
      text: "Dimension sensitivity — ranked by effect size (top = most impact)",
      font: { color: plotText, size: 13 }, x: 0.5,
    },
    hovermode: "closest",
  }, { responsive: true, displaylogo: false });

  document.getElementById("status").textContent =
    `${runs.length} runs · ${effects.length} dimensions`;
}

// ── Tab: Optimization ────────────────────────────────────────────────────────

function buildOptimizationChart() {
  if (!DATA) return;
  dirtyTabs.delete("optimization");

  const plotBg   = cssVar("--plot-bg");
  const plotGrid = cssVar("--plot-grid");
  const plotText = cssVar("--plot-text");

  const allRuns = visibleRuns();

  // Pair each visible run with its trial index (position in submission order)
  // and its final scalar value for the current metric.
  const trialData = allRuns.map((r, i) => ({ r, trialIdx: i, v: runScalar(r.hash) }));
  const withData  = trialData.filter(d => d.v !== null);

  if (!withData.length) {
    Plotly.react("opt-bestsofar", [], {
      margin: { t: 20, r: 20, b: 40, l: 50 },
      paper_bgcolor: plotBg, plot_bgcolor: plotBg,
      annotations: [{ text: "No metric data yet", showarrow: false,
        font: { color: plotText, size: 13 }, xref: "paper", yref: "paper", x: 0.5, y: 0.5 }],
    }, { responsive: true, displaylogo: false });
    document.getElementById("opt-scatter-grid").innerHTML = "";
    document.getElementById("status").textContent = "No data";
    return;
  }

  const n         = allRuns.length;
  const trialColor = d => lerpColor(n > 1 ? d.trialIdx / (n - 1) : 0.5);
  const comboText  = d => Object.entries(d.r.combo).map(([k, v]) => `${k}=${v}`).join("  ");

  // ── Best-so-far curve ───────────────────────────────────────────────────────
  let best = null;
  const bestLine = withData.map(d => {
    if (best === null || (optMinimize ? d.v < best : d.v > best)) best = d.v;
    return best;
  });
  const trialNums = withData.map(d => d.trialIdx + 1);

  Plotly.react("opt-bestsofar", [
    {
      x: trialNums, y: withData.map(d => d.v),
      mode: "markers", name: "trial",
      marker: { color: withData.map(trialColor), size: 7, line: { color: plotBg, width: 1 } },
      text: withData.map(comboText),
      showlegend: false,
      hovertemplate: "trial %{x}<br>" + metric + ": %{y:.4f}<br>%{text}<extra></extra>",
    },
    {
      x: trialNums, y: bestLine,
      mode: "lines", name: optMinimize ? "best (min)" : "best (max)",
      line: { color: cssVar("--accent"), width: 2, shape: "hv" },
      hovertemplate: "after trial %{x}<br>best: %{y:.4f}<extra></extra>",
    },
  ], {
    height: 240,
    margin: { t: 16, r: 20, b: 48, l: 60 },
    xaxis: { title: "trial", gridcolor: plotGrid, color: plotText },
    yaxis: { title: metric, gridcolor: plotGrid, color: plotText },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { font: { size: 11, color: plotText }, x: 1, xanchor: "right", y: 1 },
    hovermode: "closest",
  }, { responsive: true, displaylogo: false });

  // ── Scatter grid ────────────────────────────────────────────────────────────
  const grid = document.getElementById("opt-scatter-grid");
  for (const el of grid.querySelectorAll(".opt-scatter-panel")) Plotly.purge(el);
  grid.innerHTML = "";

  // Color dots blue→red by trial order so you can see where the optimizer searched over time.

  // Sort dimensions by effect size so the most informative panels come first.
  const effects = computeDimEffects(allRuns);

  for (const e of effects) {
    const pts = withData.filter(d => e.dim in d.r.combo);
    if (!pts.length) continue;

    const xRaw   = pts.map(d => d.r.combo[e.dim]);
    const xNum   = xRaw.map(Number);
    const numeric = xNum.every(v => isFinite(v));

    const panel = document.createElement("div");
    panel.className = "opt-scatter-panel";
    panel.dataset.dim = e.dim;
    grid.appendChild(panel);

    Plotly.react(panel, [{
      x: numeric ? xNum : xRaw,
      y: pts.map(d => d.v),
      mode: "markers",
      marker: { color: pts.map(trialColor), size: 8, line: { color: plotBg, width: 1 } },
      text: pts.map(d => `trial ${d.trialIdx + 1}`),
      hovertemplate: e.dim + ": %{x}<br>" + metric + ": %{y:.4f}<br>%{text}<extra></extra>",
      showlegend: false,
    }], {
      height: 220,
      margin: { t: 38, r: 10, b: 52, l: 60 },
      title: { text: `${e.dim} vs ${metric}   Δ${e.effectSize.toFixed(3)}`, font: { size: 11, color: plotText }, x: 0.5 },
      xaxis: { title: { text: e.dim, font: { size: 10, color: plotText } }, gridcolor: plotGrid, color: plotText, type: numeric ? "linear" : "category" },
      yaxis: { title: { text: metric, font: { size: 10, color: plotText } }, gridcolor: plotGrid, color: plotText },
      paper_bgcolor: plotBg, plot_bgcolor: plotBg,
      hovermode: "closest",
    }, { responsive: true, displaylogo: false });
  }

  document.getElementById("status").textContent =
    `${withData.length} / ${allRuns.length} runs with ${metric}  ·  ${effects.length} dimensions`;
}

// ── Sidebar builders (from visualize.py verbatim) ─────────────────────────────

function buildFilters() {
  const subDimInfo = DATA.subDims || {};
  const subDimSet  = new Set(Object.keys(subDimInfo));

  // childrenOf["parentDim:parentValue"] = [childDim, ...]
  const childrenOf = {};
  for (const [dim, info] of Object.entries(subDimInfo)) {
    const key = `${info.parentDim}:${info.parentValue}`;
    (childrenOf[key] = childrenOf[key] || []).push(dim);
  }

  const container = document.getElementById("filters");
  container.innerHTML = "";

  function makeFilterGroup(dim, values, extraClass) {
    const isColorDim = !colorBy.startsWith("_m:") && dim === colorBy;
    const cmap_      = isColorDim ? colorMap(values) : {};

    const group = document.createElement("div");
    group.className = "filter-group" + (extraClass ? " " + extraClass : "");

    const header = document.createElement("div");
    header.className = "filter-header";
    header.innerHTML = `<span class="axis-label">${escHtml(dim)}</span>`;
    const btns = document.createElement("div");
    btns.className = "filter-btns";
    ["all", "none"].forEach(action => {
      const b = document.createElement("button");
      b.textContent = action;
      b.onclick = () => {
        filters[dim] = action === "all" ? new Set(values) : new Set();
        group.querySelectorAll(`input[data-dim="${CSS.escape(dim)}"]`)
             .forEach(cb => { cb.checked = action === "all"; });
        buildActiveChart();
      };
      btns.appendChild(b);
    });
    header.appendChild(btns);
    group.appendChild(header);

    const list = document.createElement("div");
    list.className = "checkbox-list";
    for (const val of values) {
      const row = document.createElement("label");
      row.className = "check-row";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.dataset.dim = dim;
      cb.checked = filters[dim]?.has(val) ?? true;
      cb.addEventListener("change", () => {
        if (cb.checked) filters[dim].add(val); else filters[dim].delete(val);
        buildActiveChart();
      });
      row.appendChild(cb);

      if (isColorDim) {
        const swatch = document.createElement("div");
        swatch.className = "color-swatch";
        swatch.style.background = cmap_[val] || "#ccc";
        row.appendChild(swatch);
      }

      const lbl = document.createElement("span");
      lbl.className = "check-label";
      lbl.textContent = val;
      row.appendChild(lbl);
      const only = document.createElement("button");
      only.className = "only-btn";
      only.textContent = "only";
      only.title = `Show only ${dim}=${val} (again to show all)`;
      only.addEventListener("click", e => { e.preventDefault(); showOnly(dim, val); });
      row.appendChild(only);
      list.appendChild(row);

      // Nest any child dims under this value, hidden when unchecked.
      for (const childDim of (childrenOf[`${dim}:${val}`] || [])) {
        const childGroup = makeFilterGroup(childDim, DATA.dims[childDim] || [], "sub-filter-group");
        if (!cb.checked) childGroup.classList.add("hidden");
        cb.addEventListener("change", () => childGroup.classList.toggle("hidden", !cb.checked));
        list.appendChild(childGroup);
      }
    }

    group.appendChild(list);
    return group;
  }

  for (const [dim, values] of Object.entries(DATA.dims)) {
    if (subDimSet.has(dim)) continue;  // rendered inline under parent
    container.appendChild(makeFilterGroup(dim, values, null));
  }
}

function buildColorBySelect() {
  const container  = document.getElementById("color-by");
  const subDimInfo = DATA.subDims || {};
  container.innerHTML = "";

  for (const dim of Object.keys(DATA.dims)) {
    const id    = "cbr-" + dim;
    const input = document.createElement("input");
    input.type = "radio"; input.name = "color-by"; input.id = id; input.value = dim;

    const lbl = document.createElement("label");
    lbl.htmlFor = id;
    const info = subDimInfo[dim];
    lbl.textContent = info ? `${dim} (${info.parentValue})` : dim;
    if (dim === colorBy) lbl.classList.add("checked");

    input.addEventListener("change", () => {
      container.querySelectorAll("label").forEach(l => l.classList.remove("checked"));
      lbl.classList.add("checked");
      colorBy = dim;
      buildFilters();
      buildActiveChart();
    });

    container.appendChild(input);
    container.appendChild(lbl);
  }
}

function buildMetricSelect() {
  METRIC_NAMES = DATA.metricNames;

  const sel = document.getElementById("metric-sel");
  sel.innerHTML = "";
  for (const m of METRIC_NAMES) {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    if (m === metric) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.onchange = () => { metric = sel.value; buildActiveChart(); };
}

// ── Init (from visualize.py verbatim, with manager-format data) ───────────────

function init(data) {
  DATA    = data;
  colorBy = Object.keys(DATA.dims)[0] || "_none_";

  const noteEl = document.getElementById("sweep-note");
  if (DATA.note) {
    noteEl.textContent = DATA.note;
    noteEl.style.display = "";
  } else {
    noteEl.style.display = "none";
  }

  filters = Object.fromEntries(
    Object.entries(DATA.dims).map(([dim, vals]) => [dim, new Set(vals)])
  );

  // Wire up one-time event listeners.
  if (!init._wired) {
    init._wired = true;

    // Smoothing slider (curves-only).
    document.getElementById("smoothing").addEventListener("input", e => {
      smoothAlpha = parseFloat(e.target.value);
      document.getElementById("smooth-val").textContent = smoothAlpha.toFixed(2);
      if (activeTab === "curves") buildCurvesChart();
    });

    // Tab buttons.
    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => setTab(btn.dataset.tab));
    });

    // Optimization: minimize/maximize toggle.
    document.getElementById("opt-minimize").addEventListener("change", e => {
      optMinimize = e.target.checked;
      if (activeTab === "optimization") buildOptimizationChart();
    });
  }

  metric = DATA.metricNames.includes(metric) ? metric
    : (DATA.metricNames.find(m => m.includes("loss")) || DATA.metricNames[0]);

  buildMetricSelect();
  buildColorBySelect();
  buildFilters();

  // Re-apply tab visibility in case experiment was switched.
  setTab(activeTab);
}

// ── Incremental EMA (from visualize.py verbatim) ──────────────────────────────

function _applyEmaIncremental(newValues, lastS) {
  if (smoothAlpha === 0) return [newValues, null];
  const out = [];
  let s = lastS;
  for (const v of newValues) {
    if (v === null || !isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : smoothAlpha * s + (1 - smoothAlpha) * v;
    out.push(s);
  }
  return [out, s];
}

// ── Experiment selector list update (from visualize.py verbatim) ──────────────

let ALL_EXPS   = [];    // every experiment, for the campaign list
let LOADED_EXP = null;  // experiment whose data is shown

// Refill the campaign and experiment selects; keeps the selected experiment
// when it is still listed, else selects *prefer* or the first one.
function _updateExperimentList(experiments, prefer) {
  ALL_EXPS = experiments;
  MLCampaign.fill(document.getElementById("campaign-sel"), ALL_EXPS);
  const exps = MLCampaign.filter(ALL_EXPS);
  const sel = document.getElementById("experiment-sel");
  MLCampaign.fillExperiments(sel, exps, prefer || sel.value);
}

// The selected campaign has no experiments, so show nothing.
function clearExperiment() {
  LOADED_EXP = null;
  if (_ws) { _ws.onclose = null; _ws.close(); _ws = null; }
  clearTimeout(_jobRefreshTimer);
  DATA = null; METRIC_CACHE = {}; LIVE_RUNS = new Set();
  if (typeof Plotly !== "undefined") {
    Plotly.purge("chart-curves");
    Plotly.purge("chart-sensitivity");
    Plotly.purge("opt-bestsofar");
    document.querySelectorAll(".opt-scatter-panel").forEach(el => Plotly.purge(el));
  }
  document.getElementById("opt-scatter-grid").innerHTML = "";
  ["metric-sel", "color-by", "filters"].forEach(id => { document.getElementById(id).innerHTML = ""; });
  document.getElementById("exp-dl-metrics").style.display = "none";
  const el = document.getElementById("exp-name");
  el.classList.remove("loading");
  el.textContent = "No experiments in this campaign";
  document.getElementById("status").textContent = "";
}

// Load the selected experiment if it is not the one shown.
function showSelected() {
  const target = document.getElementById("experiment-sel").value;
  if (!target) clearExperiment();
  else if (target !== LOADED_EXP) loadExperiment(target);
}

// ── JSONL parsing (new — not in visualize.py) ─────────────────────────────────

function parseJsonl(text) {
  const rows = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    try { rows.push(JSON.parse(t)); } catch {}
  }
  return rows;
}

// Parse JSONL rows for one run into METRIC_CACHE entries.
// Each row: {step: N, metric1: v1, metric2: v2, ...}
function mergeJsonlIntoCache(rows, runId) {
  for (const row of rows) {
    const step = row.step;
    if (step == null) continue;
    for (const [k, v] of Object.entries(row)) {
      if (k === "step") continue;
      if (!METRIC_CACHE[k]) METRIC_CACHE[k] = {};
      if (!METRIC_CACHE[k][runId]) METRIC_CACHE[k][runId] = { steps: [], values: [] };
      const val = (typeof v === "number" && isFinite(v)) ? v : null;
      METRIC_CACHE[k][runId].steps.push(step);
      METRIC_CACHE[k][runId].values.push(val);
    }
  }
}

// ── Metric batch buffer (replaces per-step Plotly calls with one RAF flush) ────

function _scheduleFlush() {
  if (_rafPending) return;
  _rafPending = true;
  requestAnimationFrame(_flushPendingMetrics);
}

function _flushPendingMetrics() {
  _rafPending = false;
  const pending = _pendingMetrics;
  _pendingMetrics = {};
  if (!DATA) return;

  let gotCurrentMetric   = false;
  let metricNamesChanged = false;
  const incrementalUpdates = [];

  for (const [runId, perMetric] of Object.entries(pending)) {
    for (const [mname, {steps, values}] of Object.entries(perMetric)) {
      if (!METRIC_CACHE[mname]) METRIC_CACHE[mname] = {};
      if (!METRIC_CACHE[mname][runId]) METRIC_CACHE[mname][runId] = { steps: [], values: [] };
      METRIC_CACHE[mname][runId].steps.push(...steps);
      METRIC_CACHE[mname][runId].values.push(...values);

      if (!DATA.metricNames.includes(mname)) {
        DATA.metricNames.push(mname);
        metricNamesChanged = true;
      }

      if (mname === metric) {
        gotCurrentMetric = true;
        if (!LAST_EMA_S[metric]) LAST_EMA_S[metric] = {};
        const [newSmoothed, newS] = _applyEmaIncremental(values, LAST_EMA_S[metric][runId] ?? null);
        LAST_EMA_S[metric][runId] = newS;
        const lineIdx = _traceIndexMap[runId];
        if (lineIdx !== undefined) {
          const n      = Object.keys(_traceIndexMap).length;
          const dotIdx = n + lineIdx;
          const lastStep = steps[steps.length - 1];
          const lastY    = newSmoothed[newSmoothed.length - 1];
          incrementalUpdates.push({ lineIdx, dotIdx, newSteps: steps, newSmoothed, lastStep, lastY });
        }
      }
    }
  }

  if (metricNamesChanged) {
    DATA.metricNames.sort();
    if (!metric || !DATA.metricNames.includes(metric)) {
      metric = DATA.metricNames.find(m => m.includes("loss")) || DATA.metricNames[0];
      buildMetricSelect();
      buildActiveChart();
      return;
    }
    buildMetricSelect();
  }
  if (!gotCurrentMetric) return;

  if (activeTab === "curves" && !dirtyTabs.has("curves") && smoothAlpha === 0) {
    const lineIdxs = [], lineXs = [], lineYs = [];
    const dotIdxs  = [], dotXs  = [], dotYs  = [];
    for (const u of incrementalUpdates) {
      lineIdxs.push(u.lineIdx); lineXs.push(u.newSteps);  lineYs.push(u.newSmoothed);
      dotIdxs.push(u.dotIdx);   dotXs.push([u.lastStep]); dotYs.push([u.lastY]);
    }
    if (lineIdxs.length) {
      Plotly.extendTraces("chart-curves", { x: lineXs, y: lineYs }, lineIdxs);
      Plotly.restyle("chart-curves",      { x: dotXs,  y: dotYs  }, dotIdxs);
    }
    const vis = Object.keys(_traceIndexMap).length;
    document.getElementById("status").innerHTML =
      `${vis} / ${DATA.runs.length} runs visible` +
      (LIVE_RUNS.size ? `  •  ${LIVE_RUNS.size} live` : "") + hiddenNote();
  } else {
    buildActiveChart();
  }
  dirtyTabs.add("sensitivity");
}

// ── WebSocket live updates (replaces connectSSE from visualize.py) ────────────

function connectWS(expId) {
  if (_ws) { _ws.close(); _ws = null; }
  LIVE_RUNS       = new Set();
  LAST_EMA_S      = {};
  _pendingMetrics = {};
  _rafPending     = false;

  const url = wsURL(`/ws/experiments/${encodeURIComponent(expId)}`);
  _ws = new WebSocket(url);

  let pingInterval = null;
  _ws.onopen = () => {
    setConnStatus("connected");
    pingInterval = setInterval(() => {
      if (_ws && _ws.readyState === WebSocket.OPEN)
        _ws.send(JSON.stringify({ type: "ping" }));
    }, 30000);
  };

  _ws.onclose = () => {
    clearInterval(pingInterval);
    const sel = document.getElementById("experiment-sel");
    if (sel && sel.value === expId) {
      setConnStatus("reconnecting");
      // Auto-reconnect if this experiment is still selected.
      setTimeout(() => {
        if (document.getElementById("experiment-sel")?.value === expId) connectWS(expId);
      }, 3000);
    } else {
      setConnStatus("disconnected");
    }
  };

  _ws.onerror = () => {};

  _ws.onmessage = e => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { return; }
    if (!DATA) return;

    if (evt.type === "metric") {
      MLStepTimer.onMetric(evt);
      const runId = evt.run_id;
      const step  = evt.step;
      if (!_pendingMetrics[runId]) _pendingMetrics[runId] = {};
      for (const [mname, val] of Object.entries(evt.data || {})) {
        const fval = (typeof val === "number" && isFinite(val)) ? val : null;
        if (!_pendingMetrics[runId][mname]) _pendingMetrics[runId][mname] = { steps: [], values: [] };
        _pendingMetrics[runId][mname].steps.push(step);
        _pendingMetrics[runId][mname].values.push(fval);
      }
      _scheduleFlush();
    }

    else if (evt.type === "job_started" || evt.type === "job_dispatched") {
      LIVE_RUNS.add(evt.run_id);
      setJobStatus(evt.run_id, evt.type === "job_started" ? "running" : "dispatched");
      // If this run isn't in DATA.runs yet (started after page load), schedule a
      // refresh to add it to the chart and give _traceIndexMap an entry for it.
      if (DATA && !DATA.runs.some(r => r.hash === evt.run_id))
        _scheduleJobRefresh(expId);
    }

    else if (evt.type === "job_done") {
      LIVE_RUNS.delete(evt.run_id);
      setJobStatus(evt.run_id, evt.status || (evt.success ? "done" : "failed"));
      // Retroactively reclassified singular-probe failures arrive in xfailed[].
      if (evt.xfailed && evt.xfailed.length) {
        for (const id of evt.xfailed) LIVE_RUNS.delete(id);
      }
      // New run may have appeared; refresh job list after a short delay.
      _scheduleJobRefresh(expId);
    }

    else if (evt.type === "job_updated" && "label" in evt) {
      // A rename: the run's legend and hover name follow its display name.
      const run = DATA.runs.find(r => r.hash === evt.run_id);
      if (run) {
        run.name = runDisplayName(evt);
        dirtyTabs.add("curves");  // only the curves legend and hover show names
        if (activeTab === "curves" && typeof Plotly !== "undefined") renderTab("curves");
      }
    }

    else if (evt.type === "experiment_done" || evt.type === "status_updated") {
      const el = document.getElementById("exp-name");
      if (el) el.textContent = expId + (evt.status ? ` (${evt.status})` : " (done)");
    }
  };
}

function rememberJobs(jobs) {
  for (const j of jobs) { JOBS.set(j.run_id, j); MLStepTimer.seed(j); }
  if (selectedRun) renderRunBar();
}

function setJobStatus(runId, status) {
  const j = JOBS.get(runId);
  if (j) j.status = status;
  if (runId === selectedRun) renderRunBar();
}

// combo is stored as a JSON string in the DB; parse it.
function parseCombo(raw) {
  if (!raw) return {};
  if (typeof raw === "string") { try { return JSON.parse(raw); } catch { return {}; } }
  return raw;
}

// Debounced job list refresh to pick up new runs.
let _jobRefreshTimer = null;
function _scheduleJobRefresh(expId) {
  clearTimeout(_jobRefreshTimer);
  _jobRefreshTimer = setTimeout(() => _refreshJobs(expId), 3000);
}

async function _refreshJobs(expId) {
  try {
    const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/jobs`);
    if (!r.ok) return;
    const jobs = await r.json();
    rememberJobs(jobs);
    const existing = new Set(DATA.runs.map(r => r.hash));
    let added = false;
    for (const j of jobs) {
      if (!existing.has(j.run_id)) {
        const combo = {};
        for (const [k, v] of Object.entries(parseCombo(j.combo))) combo[k] = String(v);
        DATA.runs.push({ name: runDisplayName(j), hash: j.run_id, combo });
        for (const [k, v] of Object.entries(combo)) {
          if (!DATA.dims[k]) { DATA.dims[k] = []; filters[k] = new Set(); }
          if (!DATA.dims[k].includes(v)) { DATA.dims[k].push(v); DATA.dims[k].sort(); filters[k].add(v); }
        }
        added = true;
      }
    }
    if (added) { buildColorBySelect(); buildFilters(); }
    // Fetch metrics for any new runs.
    await _loadMetrics(expId, jobs.filter(j => !existing.has(j.run_id)));
    // Sync any newly discovered metric names and auto-select if unset.
    let metricNamesChanged = false;
    for (const mname of Object.keys(METRIC_CACHE)) {
      if (!DATA.metricNames.includes(mname)) {
        DATA.metricNames.push(mname);
        metricNamesChanged = true;
      }
    }
    if (metricNamesChanged) {
      DATA.metricNames.sort();
      if (!metric || !DATA.metricNames.includes(metric)) {
        metric = DATA.metricNames.find(m => m.includes("loss")) || DATA.metricNames[0];
      }
      buildMetricSelect();
    }
    if (added || metricNamesChanged) buildActiveChart();
  } catch {}
}

// ── Metric loading (new — replaces file-based approach from visualize.py) ─────

async function _loadMetrics(expId, jobs) {
  await Promise.all(jobs.map(async j => {
    try {
      const r = await apiFetch(
        `/api/experiments/${encodeURIComponent(expId)}/jobs/${encodeURIComponent(j.run_id)}/metrics`
      );
      if (!r.ok) return;
      const text = await r.text();
      mergeJsonlIntoCache(parseJsonl(text), j.run_id);
    } catch {}
  }));
}

// ── Load experiment (replaces loadExperiment from visualize.py) ───────────────

async function loadExperiment(expId) {
  LOADED_EXP = expId;
  // Keep Artifacts nav link in sync with the viewed experiment
  document.getElementById("nav-files").href = pageHref("files.html", { experiment: expId });
  MLCampaign.linkNav();

  const dlBtn = document.getElementById("exp-dl-metrics");
  dlBtn.href = authURL(`/api/experiments/${encodeURIComponent(expId)}/metrics.zip`);
  dlBtn.style.display = "";

  const expNameEl = document.getElementById("exp-name");
  expNameEl.classList.add("loading");
  expNameEl.textContent = "Fetching data…";
  document.getElementById("status").textContent = "";

  if (_ws) { _ws.close(); _ws = null; }
  clearTimeout(_jobRefreshTimer);
  DATA = null; METRIC_CACHE = {}; _traceIndexMap = {}; LAST_EMA_S = {}; LIVE_RUNS = new Set();
  // The ?run= highlight applies to the experiment named in the address only.
  selectedRun = expId === INIT_EXP && !loadExperiment._done ? (INIT_RUN || null) : null;
  loadExperiment._done = true;
  hiddenRuns = new Set(); JOBS = new Map(); MLStepTimer.clear();
  _pendingMetrics = {}; _rafPending = false;
  dirtyTabs.add("curves"); dirtyTabs.add("sensitivity"); dirtyTabs.add("optimization");
  if (typeof Plotly !== "undefined") {
    Plotly.purge("chart-curves");
    Plotly.purge("chart-sensitivity");
    Plotly.purge("opt-bestsofar");
    document.querySelectorAll(".opt-scatter-panel").forEach(el => Plotly.purge(el));
  }
  document.getElementById("opt-scatter-grid").innerHTML = "";

  try {
    const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/jobs`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const jobs = await r.json();
    rememberJobs(jobs);

    // Build manifest-like DATA structure.
    const dimValues = {};
    for (const j of jobs) {
      for (const [k, v] of Object.entries(parseCombo(j.combo))) {
        (dimValues[k] = dimValues[k] || new Set()).add(String(v));
      }
    }
    const dims = {};
    for (const [k, vs] of Object.entries(dimValues)) dims[k] = [...vs].sort();

    LIVE_RUNS = new Set(
      jobs.filter(j => j.status === "running" || j.status === "dispatched").map(j => j.run_id)
    );

    const manifest = {
      experiment: expId,
      dims,
      runs: jobs.map(j => ({
        name: runDisplayName(j),
        hash: j.run_id,
        combo: Object.fromEntries(Object.entries(parseCombo(j.combo)).map(([k, v]) => [k, String(v)])),
      })),
      metricNames: [],
      subDims: {},
      note: null,
    };

    // Load all metrics from JSONL files, then discover metric names.
    await _loadMetrics(expId, jobs);

    manifest.metricNames = Object.keys(METRIC_CACHE).sort();

    init(manifest);

    expNameEl.classList.remove("loading");
    expNameEl.textContent = expId;

    // Connect WS for live updates.
    connectWS(expId);

    // Initial chart.
    buildActiveChart();

  } catch (err) {
    expNameEl.classList.remove("loading");
    expNameEl.textContent = "Error";
    document.getElementById("status").textContent = "Failed to load: " + err;
    setConnStatus("disconnected");
  }
}

// ── Refresh experiment list (from visualize.py, adapted to manager API) ───────

async function refreshExperiments() {
  try {
    const r = await apiFetch("/api/experiments");
    const experiments = await r.json();
    _updateExperimentList(experiments);
    showSelected();
  } catch (err) {
    document.getElementById("status").textContent = "Error refreshing: " + err;
  }
}

bindActions({ showOnly, hideRun, setSelectedRun, showHiddenRuns });

// The step timer in the highlight bar shows while the highlighted run is running.
MLStepTimer.mount(document.getElementById("run-bar-timer"), () =>
  selectedRun && activeTab === "curves" && JOBS.get(selectedRun)?.status === "running" ? selectedRun : null);

// ── Bootstrap (replaces the bottom fetch block from visualize.py) ─────────────

apiFetch("/api/experiments")
  .then(r => r.json())
  .then(experiments => {
    MLCampaign.follow(experiments, INIT_EXP);
    _updateExperimentList(experiments, INIT_EXP);

    const sel = document.getElementById("experiment-sel");
    sel.onchange = () => loadExperiment(sel.value);
    document.getElementById("refresh-experiments").onclick = refreshExperiments;
    MLCampaign.bind(document.getElementById("campaign-sel"), () => {
      _updateExperimentList(ALL_EXPS);
      showSelected();
    });

    if (sel.value) loadExperiment(sel.value);
    else {
      const el = document.getElementById("exp-name");
      el.classList.remove("loading");
      el.textContent = experiments.length ? "No experiments in this campaign" : "No experiments";
    }
  })
  .catch(err => {
    const el = document.getElementById("exp-name");
    el.classList.remove("loading");
    el.textContent = "Error";
    document.getElementById("status").textContent = "Error fetching experiments: " + err;
    setConnStatus("disconnected");
  });
