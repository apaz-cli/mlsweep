// mlsweep Notes: data-backed markdown blocks and evidence links.
//
// ```mlsweep-chart and ```mlsweep-table blocks are sent to the manager as
// written; it evaluates them against the metrics database and returns plotted
// or tabulated data.  The manager parses the spec, so lint and the tab agree.
//
// [0.42](mlsweep:EXP/RUN/metric@20000) ties a number in prose to the metric it
// came from: this file asks the manager for that value, marks the number ✓
// when it matches (⚠ when it does not) and links it to the run's Results.
"use strict";

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
let _plotlyPromise = null;

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
  return nbSpecField(raw, "title") || nbSpecField(raw, "metric")
    || (kind === "table" ? "Table" : "Chart");
}

function nbPlotly() {
  if (typeof Plotly !== "undefined") return Promise.resolve();
  if (_plotlyPromise) return _plotlyPromise;
  _plotlyPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "plotly-basic-2.35.2.min.js";
    s.onload = () => resolve();
    s.onerror = () => { _plotlyPromise = null; reject(new Error("could not load the chart library")); };
    document.head.append(s);
  });
  return _plotlyPromise;
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
          tr.append(nbMetricCell(row.metrics ? row.metrics[c.key] : null, spec.agg));
        }
      }
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

// ── block wiring ─────────────────────────────────────────────────────────────

function nbRenderBody(kind, data) {
  const body = kind === "chart" ? nbChartBox(data) : nbTableBox(data);
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

  async function load(force) {
    if (force) NB_CACHE.delete(kind + "\n" + raw);
    body.innerHTML = "";
    body.append(el("div", "nb-loading", "Loading…"));
    try {
      const data = await nbFetchBlock(kind, raw);
      body.innerHTML = "";
      body.append(nbRenderBody(kind, data));
      const spec = data.spec || {};
      title.textContent = data.title || spec.title
        || (kind === "chart" ? spec.metric || "Chart" : "Table");
      title.title = title.textContent;
    } catch (e) {
      nbError(body, e.message || String(e));
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

async function annotateEvidence(root) {
  const anchors = [...root.querySelectorAll("a[data-ev-ref]")];
  if (!anchors.length) return;
  const refs = [...new Set(anchors.map(a => a.dataset.evRef))];
  const missing = refs.filter(r => {
    const hit = EV_CACHE.get(r);
    return !hit || Date.now() - hit.at >= EV_CACHE_MS;
  });
  if (missing.length) {
    try {
      const r = await apiFetch("/api/notes/evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refs: missing }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error((body && body.error) || `HTTP ${r.status}`);
      for (const [ref, data] of Object.entries(body.results || {})) {
        EV_CACHE.set(ref, { at: Date.now(), data });
      }
    } catch (e) {
      for (const ref of missing) EV_CACHE.set(ref, { at: Date.now(), data: { ok: false, error: e.message } });
    }
  }
  for (const a of anchors) {
    const hit = EV_CACHE.get(a.dataset.evRef);
    nbEvidenceMark(a, hit && hit.data);
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
