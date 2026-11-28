// mlsweep Notes: data-backed markdown blocks and evidence links.
//
// ```mlsweep-chart and ```mlsweep-table blocks are sent to the manager as
// written; it evaluates them against the metrics database and returns plotted
// or tabulated data.  The manager parses the spec, so lint and the tab agree.
//
// [0.42](mlsweep:EXP/RUN/metric@20000) ties a number in prose to the metric it
// came from: this file asks the manager for that value, marks the number ✓
// when it matches (⚠ when it does not) and links it to the run's Results.
//
// Code: [text](mlsweep:code/EXP/path#L40-60) links to a file as an experiment
// ran it, on the code page.  ```mlsweep-code shows those lines in the note and
// ```mlsweep-diff what changed between two experiments' code.
"use strict";

// Fence language -> block kind the manager evaluates.
const NB_BLOCK_KINDS = {
  "mlsweep-chart": "chart", "mlsweep-table": "table", "mlsweep-code": "code", "mlsweep-diff": "diff",
};

// The code page for a code reference (code/EXP[/path][#Lstart-end]).
function codeHref(ref) { return pageHref("code.html", { ref: String(ref).replace(/^mlsweep:/i, "") }); }

const NB_CACHE = new Map();          // kind + spec -> { at, data }
const NB_CACHE_MS = 20_000;
const NB_VIEWS = new Set();          // { div, data, kind } redrawn on a theme change
const EV_CACHE = new Map();          // ref -> { at, data }
const EV_CACHE_MS = 30_000;
const NB_PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
];

function nbCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Numeric-ish formatting shared by hover text and table cells.
function nbNum(v) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (n !== 0 && (Math.abs(n) < 1e-3 || Math.abs(n) >= 1e6)) return n.toExponential(3);
  return String(Number(n.toPrecision(5)));
}

// Plotly writes labels into the SVG, but keep stray markup out of its hover HTML.
function nbLabel(s) { return String(s == null ? "" : s).replace(/[<>]/g, ""); }

// The one field a block needs before its data arrives: its caption (and id).
function nbSpecField(raw, key) {
  const text = String(raw || "").trim();
  if (text.startsWith("{")) {
    try {
      const o = JSON.parse(text);
      return o && o[key];
    } catch { return undefined; }
  }
  for (const line of text.split("\n")) {
    const m = /^([A-Za-z0-9_-]+)\s*:\s*(.*)$/.exec(line.trim());
    if (!m) continue;
    if (m[1].toLowerCase().replace(/-/g, "_") !== key) continue;
    let v = m[2].trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    return v;
  }
  return undefined;
}

function nbFigureMeta(raw) {
  const caption = nbSpecField(raw, "caption");
  if (!caption) return null;
  const m = /\s*\{#([A-Za-z][\w:.\-]*)\}\s*$/.exec(caption);
  return m
    ? { text: caption.slice(0, m.index).trim(), id: m[1] }
    : { text: String(caption).trim(), id: "" };
}

// Before the data arrives, say what the spec asks for; the server sends the
// final title (which also covers runs picked by `experiments`).
function nbInitialTitle(raw, kind) {
  if (kind === "code") return nbSpecField(raw, "title") || nbSpecField(raw, "path") || "Code";
  if (kind === "diff") {
    const a = nbSpecField(raw, "from"), b = nbSpecField(raw, "to");
    return nbSpecField(raw, "title") || (a && b ? `${a} → ${b}` : "Diff");
  }
  return nbSpecField(raw, "title") || nbSpecField(raw, "metric")
    || (kind === "table" ? "Table" : "Chart");
}

// Load a script (and its stylesheet, if any) once; a failed load can be retried.
const _scriptLoads = new Map();
function nbLoadScript(src, what, css) {
  if (!_scriptLoads.has(src)) {
    _scriptLoads.set(src, new Promise((resolve, reject) => {
      if (css) {
        const l = document.createElement("link");
        l.rel = "stylesheet"; l.href = css;
        document.head.append(l);
      }
      const s = document.createElement("script");
      s.src = src;
      s.onload = () => resolve();
      s.onerror = () => { _scriptLoads.delete(src); reject(new Error(`could not load ${what}`)); };
      document.head.append(s);
    }));
  }
  return _scriptLoads.get(src);
}

function nbPlotly() {
  if (typeof Plotly !== "undefined") return Promise.resolve();
  return nbLoadScript("plotly-basic-2.35.2.min.js", "the chart library");
}

async function nbFetchBlock(kind, spec) {
  const key = kind + "\n" + spec;
  const hit = NB_CACHE.get(key);
  if (hit && Date.now() - hit.at < NB_CACHE_MS) return hit.data;
  const r = await apiFetch("/api/notes/block", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, spec }),
  });
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  if (!r.ok) throw new Error((body && body.error) || `HTTP ${r.status}`);
  NB_CACHE.set(key, { at: Date.now(), data: body });
  return body;
}

function nbError(box, message) {
  box.innerHTML = "";
  const d = el("div", "nb-error");
  d.append(el("span", "nb-error-icon", "!"), document.createTextNode(message));
  box.append(d);
}

function nbWarnings(box, warnings) {
  if (!warnings || !warnings.length) return;
  const d = el("div", "nb-warn");
  d.textContent = warnings.join(" · ");
  box.append(d);
}

// ── charts ───────────────────────────────────────────────────────────────────

function nbChartBox(data, hiddenSeries) {
  const wrap = el("div", "nb-chart-wrap");
  const hidden = hiddenSeries || new Set();
  if (!data.series.length) {
    wrap.append(el("div", "nb-empty", "No data to plot."));
    return wrap;
  }
  const traces = [];
  const groups = [];                    // series index -> trace indices
  data.series.forEach((s, i) => {
    const color = NB_PALETTE[i % NB_PALETTE.length];
    const x = s.x || [], y = s.y || [];
    const mine = [];
    if (s.lo && s.hi) {
      mine.push(traces.length);
      traces.push({ x, y: s.lo, mode: "lines", line: { width: 0, color },
                    hoverinfo: "skip", showlegend: false });
      mine.push(traces.length);
      traces.push({ x, y: s.hi, mode: "lines", line: { width: 0, color },
                    fill: "tonexty", fillcolor: color + "3a",
                    hoverinfo: "skip", showlegend: false });
    }
    mine.push(traces.length);
    traces.push({
      x, y, mode: "lines", name: nbLabel(s.name),
      line: { color, width: 1.9 },
      hovertemplate: "<b>%{x}</b><br>%{y:.5g}<extra>" + nbLabel(s.name) + "</extra>",
    });
    groups.push(mine);
  });

  // The legend lives above the fixed-height plot: Plotly's own horizontal
  // legend is clipped by the card, and these names are long.
  const legend = el("div", "nb-legend");
  const plot = el("div", "nb-plot");
  const highlight = (idx, on) => {
    if (typeof Plotly === "undefined" || !plot.isConnected || !plot.data) return;
    const other = [];
    groups.forEach((g, j) => { if (j !== idx) other.push(...g); });
    Plotly.restyle(plot, { opacity: on ? 0.2 : 1 }, other);
    Plotly.restyle(plot, { opacity: 1 }, groups[idx]);
  };
  const toggle = i => {
    if (hidden.has(i)) hidden.delete(i); else hidden.add(i);
    if (typeof Plotly !== "undefined" && plot.data) {
      Plotly.restyle(plot, { visible: hidden.has(i) ? "legendonly" : true }, groups[i]);
    }
    legend.querySelectorAll(".nb-legend-item")[i].classList.toggle("off", hidden.has(i));
  };
  data.series.forEach((s, i) => {
    const item = el("span", "nb-legend-item");
    const sw = el("span", "nb-swatch");
    sw.style.background = NB_PALETTE[i % NB_PALETTE.length];
    const label = el("span", "nb-legend-label", nbLabel(s.name));
    item.append(sw, label);
    item.title = nbLabel(s.name) + (s.runs > 1 ? ` · ${s.runs} runs` : "") +
      (s.full_name ? "\n" + nbLabel(s.full_name) : "") +
      "\nClick to hide or show this line";
    item.classList.toggle("off", hidden.has(i));
    item.onmouseenter = () => highlight(i, true);
    item.onmouseleave = () => highlight(i, false);
    item.onclick = () => toggle(i);
    legend.append(item);
  });
  if (data.series.some(s => s.lo && s.hi)) {
    const item = el("span", "nb-legend-item nb-legend-band");
    const sw = el("span", "nb-swatch nb-band");
    item.append(sw, el("span", "nb-legend-label", "min–max across runs"));
    item.title = "The shaded band spans the runs in each group";
    legend.append(item);
  }
  wrap.append(legend, plot);

  const plotBg = nbCssVar("--plot-bg") || "transparent";
  const grid = nbCssVar("--plot-grid") || "#333";
  const text = nbCssVar("--plot-text") || "#aaa";
  const layout = {
    margin: { t: 10, r: 16, b: 44, l: 58 },
    paper_bgcolor: plotBg,
    plot_bgcolor: plotBg,
    font: { color: text, size: 11 },
    hovermode: "closest",
    showlegend: false,
    xaxis: {
      title: { text: data.x || "step", font: { size: 11 } },
      gridcolor: grid, zeroline: false, color: text,
      type: data.spec && data.spec.log_x ? "log" : "linear",
    },
    yaxis: {
      title: { text: data.metric || "", font: { size: 11 } },
      gridcolor: grid, zeroline: false, color: text,
      type: data.spec && data.spec.log_y ? "log" : "linear",
    },
  };
  plot.append(el("div", "nb-loading", "Loading chart…"));
  nbPlotly().then(() => {
    plot.innerHTML = "";
    Plotly.react(plot, traces, layout, {
      responsive: true, displaylogo: false, displayModeBar: false,
    });
    for (const i of hidden) Plotly.restyle(plot, { visible: "legendonly" }, groups[i]);
    requestAnimationFrame(() => { try { if (plot.isConnected) Plotly.Plots.resize(plot); } catch {} });
    for (const v of NB_VIEWS) if (!v.div.isConnected) NB_VIEWS.delete(v);
    NB_VIEWS.add({ div: wrap, data, hidden, kind: "chart" });
  }).catch(e => nbError(plot, e.message || String(e)));
  return wrap;
}

// ── tables ───────────────────────────────────────────────────────────────────

function nbMetricCell(summary, agg) {
  if (!summary) return el("td", "nb-missing", "—");
  const value = summary.value != null ? summary.value : summary.final;
  const td = el("td", "nb-num", nbNum(value));
  const bits = [];
  if (summary.final != null) bits.push("final " + nbNum(summary.final) + " @ " + summary.step);
  if (summary.min != null) bits.push("min " + nbNum(summary.min));
  if (summary.max != null) bits.push("max " + nbNum(summary.max));
  if (summary.mean != null) bits.push("mean " + nbNum(summary.mean));
  td.title = bits.join(" · ");
  if (agg && summary[agg] != null && agg !== "final") td.classList.add("nb-agg");
  return td;
}

function nbTableBox(data) {
  const wrap = el("div", "nb-table-wrap");
  if (!data.rows.length) {
    wrap.append(el("div", "nb-empty", "No runs to show."));
    return wrap;
  }
  const spec = data.spec || {};
  let sortKey = spec.sort || null;
  let sortOrder = spec.order === "desc" ? "desc" : "asc";
  const table = el("table", "nb-table");
  const thead = el("thead");
  const tbody = el("tbody");
  const hr = el("tr");
  const headers = [];

  const valueOf = (row, key) => {
    const col = data.columns.find(c => c.key === key);
    if (!col) return null;
    if (col.kind === "run") return row.label || row.run;
    if (col.kind === "param") return row.params ? row.params[key] : null;
    const summary = row.metrics ? row.metrics[key] : null;
    return summary ? (summary.value != null ? summary.value : summary.final) : null;
  };
  const compare = (a, b) => {
    const av = valueOf(a, sortKey), bv = valueOf(b, sortKey);
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = (typeof av === "number" && typeof bv === "number")
      ? av - bv
      : String(av).localeCompare(String(bv), undefined, { numeric: true });
    return sortOrder === "desc" ? -cmp : cmp;
  };
  const runLink = (row, text) => {
    const a = el("a", "nb-run", text);
    a.href = pageHref("results.html", { experiment: row.experiment, run: row.run });
    a.title = `${row.experiment}/${row.run} · ${row.status || "?"}` +
      "\nShift+click for Logs";
    a.onclick = e => {
      if (!e.shiftKey) return;
      e.preventDefault();
      location.href = pageHref("logs.html", { experiment: row.experiment, run: row.run });
    };
    return a;
  };
  function drawBody() {
    tbody.innerHTML = "";
    const rows = sortKey ? [...data.rows].sort(compare) : data.rows;
    for (const row of rows) {
      const tr = el("tr");
      for (const c of data.columns) {
        if (c.kind === "run") {
          const td = el("td");
          td.append(runLink(row, row.label || row.run));
          tr.append(td);
        } else if (c.kind === "param") {
          const v = row.params ? row.params[c.key] : undefined;
          const text = v == null ? "—" : String(v);
          const td = el("td", "nb-param");
          if (data.run_link_column === c.key) td.append(runLink(row, text));
          else td.textContent = text;
          tr.append(td);
        } else {
          const td = nbMetricCell(row.metrics ? row.metrics[c.key] : null, spec.agg);
          td.dataset.metric = c.key;
          tr.append(td);
        }
      }
      tr.oncontextmenu = e => nbRunMenu(e, row, spec);
      tbody.append(tr);
    }
    for (const h of headers) {
      const active = h.c.key === sortKey;
      h.th.classList.toggle("nb-sorted", active);
      h.arrow.textContent = active ? (sortOrder === "desc" ? "▼" : "▲") : "";
      if (active) h.th.setAttribute("aria-sort", sortOrder === "desc" ? "descending" : "ascending");
      else h.th.removeAttribute("aria-sort");
    }
  }
  for (const c of data.columns) {
    const th = el("th", c.kind === "metric" ? "nb-num" : "");
    th.append(document.createTextNode(c.key));
    const arrow = el("span", "nb-sort");
    th.append(arrow);
    th.classList.add("nb-sortable");
    th.title = c.kind === "param" ? "hyperparameter · click to sort" : "Click to sort";
    th.onclick = () => {
      if (sortKey === c.key) sortOrder = sortOrder === "asc" ? "desc" : "asc";
      else { sortKey = c.key; sortOrder = "asc"; }
      drawBody();
    };
    headers.push({ th, c, arrow });
    hr.append(th);
  }
  thead.append(hr);
  table.append(thead, tbody);
  drawBody();
  wrap.append(table);
  return wrap;
}

// Right-click a table row: the run's pages, and on a metric cell, its value
// or an evidence link a note can cite.
function nbRunMenu(e, row, spec) {
  if (mlNativeMenuWanted(e, null)) return;
  e.preventDefault();
  e.stopPropagation();
  const items = [
    ...MLViews.runItems(row.experiment, row.run),
    "-",
    { label: "Copy run ID", run: () => copyText(row.run) },
  ];
  const metric = e.target.closest("td[data-metric]")?.dataset.metric;
  const summary = metric && row.metrics ? row.metrics[metric] : null;
  const value = summary ? (summary.value != null ? summary.value : summary.final) : null;
  if (value != null) {
    const at = ["min", "max", "mean"].includes(spec.agg) ? spec.agg : "final";
    items.push({ label: "Copy value", run: () => copyText(String(value)) },
      { label: "Copy evidence link", hint: `@${at}`,
        run: () => copyText(`[${nbNum(value)}](mlsweep:${row.experiment}/${row.run}/${metric}@${at})`) });
  }
  mlMenuAt(e.clientX, e.clientY, items);
}

// ── block wiring ─────────────────────────────────────────────────────────────

// ── code and diff blocks ─────────────────────────────────────────────────────

// Lines with their numbers; highlighted like the note's code blocks when the
// file type is known.  (Class names avoid nb-code / nb-diff, which the block's
// card already carries.)
function nbCodeBox(data) {
  const holder = el("div", "nb-srcbox");
  const table = el("div", "nb-src-lines");
  const gutter = el("pre", "nb-gutter");
  const code = el("pre", "nb-src");
  const inner = el("code");
  code.append(inner);
  gutter.textContent = data.lines.map((_, i) => String(data.start + i)).join("\n");
  const text = data.lines.join("\n");
  const mode = typeof modeForPath === "function" ? modeForPath(data.path) : null;
  if (mode && typeof CodeMirror !== "undefined" && CodeMirror.runMode) CodeMirror.runMode(text, mode, inner);
  else inner.textContent = text;
  table.append(gutter, code);
  holder.append(table);
  return holder;
}

// One file's unified diff as rows of old number, new number and the line.
function nbDiffRows(lines) {
  const table = el("table", "nb-diff-table");
  let oldN = 0, newN = 0;
  for (const line of lines) {
    const tr = el("tr");
    const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/.exec(line);
    if (hunk) {
      oldN = Number(hunk[1]); newN = Number(hunk[2]);
      tr.className = "hunk";
      tr.append(el("td", "ln", "⋯"), el("td", "ln", "⋯"),
                el("td", "dl", `from line ${newN}${hunk[3] && hunk[3].trim() ? " · " + hunk[3].trim() : ""}`));
      table.append(tr);
      continue;
    }
    const sign = line[0];
    const kind = sign === "+" ? "add" : sign === "-" ? "del" : "ctx";
    tr.className = kind;
    tr.append(el("td", "ln", kind === "add" ? "" : String(oldN)),
              el("td", "ln", kind === "del" ? "" : String(newN)),
              el("td", "dl", line.slice(1) || " "));
    if (kind !== "add") oldN++;
    if (kind !== "del") newN++;
    table.append(tr);
  }
  return table;
}

function nbDiffBox(data) {
  const box = el("div", "nb-diffbox");
  const c = data.counts;
  const summary = el("div", "nb-diff-sum");
  summary.textContent = data.same ? "Both ran the same code."
    : `${c.changed} changed, ${c.added} added, ${c.removed} removed` + (data.path ? ` under ${data.path}/` : "")
      + (data.files.length > 1 ? " · click a file to show or hide it" : "");
  summary.title = `${data.from.short} → ${data.to.short}`;
  box.append(summary);
  const many = data.files.length > 1;
  for (const f of data.files) {
    const det = el("details", "nb-diff-file");
    const size = (f.diff || []).length;
    // Whole new or deleted files and long diffs start closed when there are others to see.
    det.open = !many || (f.status === "M" && size <= 60);
    const sum = el("summary");
    const label = { A: "added", D: "deleted", M: "changed", mode: "mode" }[f.status] || f.status;
    sum.append(el("span", "nb-diff-status st-" + f.status.toLowerCase(), label),
               el("span", "nb-diff-path", f.path));
    if (f.added || f.removed) {
      sum.append(el("span", "nb-diff-n add", "+" + (f.added || 0)), el("span", "nb-diff-n del", "−" + (f.removed || 0)));
    }
    det.append(sum);
    if (f.binary) det.append(el("div", "nb-diff-note", "Binary or large file; not shown."));
    else if (f.status === "mode") det.append(el("div", "nb-diff-note", `Mode ${f.old_mode} → ${f.new_mode}`));
    else {
      const body = el("div", "nb-diff-body");
      body.append(nbDiffRows(f.diff));
      if (f.truncated) body.append(el("div", "nb-diff-note", "Cut short; the code page has the whole file."));
      det.append(body);
    }
    box.append(det);
  }
  return box;
}

function nbRenderBody(kind, data) {
  const body = kind === "chart" ? nbChartBox(data) : kind === "table" ? nbTableBox(data)
    : kind === "code" ? nbCodeBox(data) : nbDiffBox(data);
  if (data.warnings && data.warnings.length) {
    const holder = el("div");
    holder.append(body);
    nbWarnings(holder, data.warnings);
    return holder;
  }
  return body;
}

const nbSlug = s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-")
  .replace(/^-+|-+$/g, "").slice(0, 60) || "block";

function nbTableMarkdown(table) {
  if (!table || !table.tBodies.length) return "";
  const clean = v => String(v).replace(/\|/g, "\\|");
  const head = [...table.tHead.rows[0].cells].map(th => clean(th.firstChild ? th.firstChild.textContent : th.textContent));
  const body = [...table.tBodies[0].rows].map(tr => [...tr.cells].map(td => clean(td.textContent.trim())));
  const numeric = [...table.tHead.rows[0].cells].map(th => th.classList.contains("nb-num"));
  const widths = head.map((h, i) => Math.max(h.length, ...body.map(r => r[i].length)));
  const line = cells => "| " + cells.map((c, i) => c.padEnd(widths[i])).join(" | ") + " |";
  const rule = "| " + widths.map((w, i) => {
    const n = Math.max(3, w);
    return numeric[i] ? "-".repeat(n - 1) + ":" : "-".repeat(n);
  }).join(" | ") + " |";
  return [line(head), rule, ...body.map(line)].join("\n");
}

function nbBlock(pre, kind, raw, figState) {
  const figure = el("figure", "note-fig");
  const card = el("div", "nb-block nb-" + kind);
  const head = el("div", "nb-head");
  head.title = "Click the header to edit this block's markdown";
  const title = el("span", "nb-title", nbInitialTitle(raw, kind));
  title.title = title.textContent;
  head.append(title);
  const body = el("div", "nb-body");
  body.append(el("div", "nb-loading", "Loading…"));
  const refresh = el("button", "nb-refresh", "↻");
  refresh.title = "Fetch the latest numbers";
  refresh.onclick = e => { e.stopPropagation(); load(true); };
  head.append(refresh);
  if (kind === "chart") {
    const png = el("button", "nb-refresh", "⤓");
    png.title = "Download this chart as a PNG";
    png.onclick = e => {
      e.stopPropagation();
      const plot = figure.querySelector(".nb-plot");
      if (plot && plot.data && typeof Plotly !== "undefined") {
        Plotly.downloadImage(plot, {
          format: "png", width: 1400, height: 700,
          filename: nbSlug(title.textContent || "chart"),
        });
      }
    };
    head.append(png);
  } else if (kind === "code" || kind === "diff") {
    const open = el("a", "nb-refresh nb-open", "↗");
    open.title = "Open on the code page";
    open.addEventListener("click", e => e.stopPropagation());
    head.append(open);
  } else {
    const copy = el("button", "nb-refresh", "⧉");
    copy.title = "Copy this table as a markdown table";
    copy.onclick = e => {
      e.stopPropagation();
      const table = figure.querySelector("table.nb-table");
      if (table) copyText(nbTableMarkdown(table));
    };
    head.append(copy);
  }
  card.append(head, body);

  const meta = nbFigureMeta(raw);
  if (meta) {
    const counter = kind === "table" ? "table" : "fig";
    const n = ++figState[counter];
    figure.id = meta.id || ((counter === "table" ? "tbl-" : "fig-") + n);
    figure.append(card);
    const cap = el("figcaption");
    const label = counter === "table" ? "Table" : "Figure";
    cap.append(el("span", "fig-kind", `${label} ${n}.`), document.createTextNode(" " + meta.text));
    figure.append(cap);
  } else {
    figure.classList.add("nb-bare");
    figure.append(card);
  }

  let loadSeq = 0;
  async function load(force) {
    const seq = ++loadSeq;
    if (force) NB_CACHE.delete(kind + "\n" + raw);
    body.innerHTML = "";
    body.append(el("div", "nb-loading", "Loading…"));
    try {
      const data = await nbFetchBlock(kind, raw);
      // A refresh clicked while this fetch was out owns the block now.
      if (seq !== loadSeq) return;
      body.innerHTML = "";
      body.append(nbRenderBody(kind, data));
      const spec = data.spec || {};
      title.textContent = data.title || spec.title
        || (kind === "chart" ? spec.metric || "Chart" : "Table");
      title.title = title.textContent;
      const open = head.querySelector(".nb-open");
      if (open) {
        open.href = kind === "code" ? codeHref(data.ref)
          : codeHref(`code/${data.to.experiment || data.to.snapshot}${data.path ? "/" + data.path : ""}`);
      }
    } catch (e) {
      if (seq === loadSeq) nbError(body, e.message || String(e));
    }
  }
  load(false);
  pre.replaceWith(figure);
  return figure;
}

// ── evidence links ───────────────────────────────────────────────────────────

function nbClaimed(text) {
  const cleaned = String(text).replace(/,/g, "").replace(/[^0-9eE+.\-]/g, "");
  const n = Number(cleaned);
  return cleaned !== "" && Number.isFinite(n) ? n : null;
}

function nbEvidenceMark(a, ev) {
  a.querySelectorAll(".ev-mark").forEach(m => m.remove());
  a.classList.remove("ev-ok", "ev-bad", "ev-unknown");
  const claimed = nbClaimed(a.textContent);
  const mark = el("span", "ev-mark");
  a.append(mark);
  if (!ev || !ev.ok) {
    a.classList.add("ev-unknown");
    mark.textContent = "?";
    a.title = (ev && ev.error) || "could not check this number";
    return;
  }
  const got = Number(ev.value);
  const where = `${ev.experiment}/${ev.run} · ${ev.metric} @ ${ev.step}`;
  a.dataset.evValue = String(ev.value);
  a.dataset.evStep = String(ev.step);
  a.href = pageHref("results.html", { experiment: ev.experiment, run: ev.run });
  mlTagEntity(a, ev.experiment, ev.run);
  if (claimed == null || !Number.isFinite(got)) {
    a.classList.add("ev-unknown");
    mark.textContent = "?";
    a.title = `${where} = ${nbNum(ev.value)}`;
    return;
  }
  const tolerance = Math.max(Math.abs(got) * 0.005, 0.005);
  const ok = Math.abs(got - claimed) <= tolerance;
  a.classList.add(ok ? "ev-ok" : "ev-bad");
  mark.textContent = ok ? "✓" : "⚠";
  a.title = ok
    ? `${where} = ${nbNum(got)}${ev.exact ? "" : " (nearest logged step)"} · matches the note`
    : `${where} = ${nbNum(got)}, but the note says ${nbNum(claimed)}. Click for the experiment.`;
}

// Resolve `mlsweep:` link refs the cache lacks (or has stale), in one request.
async function evResolve(refs) {
  const missing = [...new Set(refs)].filter(r => {
    const hit = EV_CACHE.get(r);
    return !hit || Date.now() - hit.at >= EV_CACHE_MS;
  });
  if (!missing.length) return;
  try {
    const r = await apiFetch("/api/notes/evidence", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refs: missing }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error((body && body.error) || `HTTP ${r.status}`);
    for (const [ref, data] of Object.entries(body.results || {})) EV_CACHE.set(ref, { at: Date.now(), data });
  } catch (e) {
    for (const ref of missing) EV_CACHE.set(ref, { at: Date.now(), data: { ok: false, error: e.message } });
  }
}

// Every kind of `mlsweep:` link in *root*, resolved together, then marked.
async function annotateLinks(root) {
  const anchors = root.querySelectorAll("a[data-ev-ref], a[data-code-ref], a[data-trace-ref]");
  if (!anchors.length) return;
  await evResolve([...anchors].map(a => a.dataset.evRef || a.dataset.codeRef || a.dataset.traceRef));
  annotateEvidence(root);
  annotateCode(root);
  annotateTraces(root);
}

async function annotateEvidence(root) {
  const anchors = [...root.querySelectorAll("a[data-ev-ref]")];
  if (!anchors.length) return;
  await evResolve(anchors.map(a => a.dataset.evRef));
  for (const a of anchors) {
    const hit = EV_CACHE.get(a.dataset.evRef);
    nbEvidenceMark(a, hit && hit.data);
  }
}

// ── code links ───────────────────────────────────────────────────────────────

// Says where a code link leads; marks it ⚠ when the file or lines are gone.
async function annotateCode(root) {
  const anchors = [...root.querySelectorAll("a[data-code-ref]")];
  if (!anchors.length) return;
  await evResolve(anchors.map(a => a.dataset.codeRef));
  for (const a of anchors) {
    const hit = EV_CACHE.get(a.dataset.codeRef);
    const res = hit && hit.data;
    a.querySelectorAll(".ev-mark").forEach(m => m.remove());
    a.classList.remove("code-ok", "code-bad");
    if (!res || !res.ok) {
      a.classList.add("code-bad");
      a.append(el("span", "ev-mark", "⚠"));
      a.title = (res && res.error) || "could not check this code link";
      continue;
    }
    a.classList.add("code-ok");
    const lines = res.start ? (res.end && res.end !== res.start ? ` lines ${res.start}–${res.end}` : ` line ${res.start}`) : "";
    const what = res.kind === "tree" ? `the code (${res.files} files)` : res.kind === "dir" ? `${res.path}/` : res.path + lines;
    a.title = `${what} as ${res.experiment || "snapshot"} ran it · ${res.short}`
      + (res.versions > 1 ? ` (latest of ${res.versions} code versions)` : "");
  }
}

// ── trace links ──────────────────────────────────────────────────────────────

// Says which session a trace link opens; marks it ⚠ when the trace is gone or a
// short id matches more than one.
async function annotateTraces(root) {
  const anchors = [...root.querySelectorAll("a[data-trace-ref]")];
  if (!anchors.length) return;
  await evResolve(anchors.map(a => a.dataset.traceRef));
  for (const a of anchors) {
    const hit = EV_CACHE.get(a.dataset.traceRef);
    const res = hit && hit.data;
    a.querySelectorAll(".ev-mark").forEach(m => m.remove());
    a.classList.remove("trace-ok", "trace-bad");
    if (!res || !res.ok) {
      a.classList.add("trace-bad");
      a.append(el("span", "ev-mark", "⚠"));
      a.title = (res && res.error) || "could not check this trace link";
      continue;
    }
    a.classList.add("trace-ok");
    // A short id opens the full one, so the Traces tab's URL names the trace.
    if (res.session_id) a.href = traceHref(a.dataset.traceRef, res.session_id);
    const exps = res.experiments || [];
    const trace = `${res.agent} trace${res.title ? `: ${res.title}` : ""}`;
    const ev = res.event;
    a.title = ev
      ? `${ev.kind === "tool" ? ev.name || "tool" : ev.kind} event ${ev.seq} of ${trace}`
        + (ev.text ? `\n${ev.text}` : "")
      : `${trace} · ${res.event_count ?? 0} events`
        + (exps.length ? ` · ${exps.length} experiment${exps.length === 1 ? "" : "s"}` : "")
        + (res.kind ? ` · showing ${res.kind} only` : "");
  }
}

// Re-plot charts after a theme change, from the data already fetched.
function nbRedraw() {
  for (const view of [...NB_VIEWS]) {
    if (!view.div.isConnected) { NB_VIEWS.delete(view); continue; }
    const replacement = nbChartBox(view.data, view.hidden);
    view.div.replaceWith(replacement);
    NB_VIEWS.delete(view);
  }
}
document.addEventListener("mlthemechange", () => nbRedraw());
