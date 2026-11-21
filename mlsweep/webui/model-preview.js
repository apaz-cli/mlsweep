// Previews of model and array artifacts (safetensors, PyTorch, pickle, NumPy,
// GGUF, ONNX).  The manager parses the file (/artifact-info/...) and this
// renders the summary.  Needs escHtml, fmtSize and apiFetch from chrome.js
// and dialog.js, and model-preview.css.

const MLModelPreview = (() => {
  const EXT = /\.(safetensors|pt|pth|ckpt|bin|pkl|pickle|npy|npz|gguf|onnx)$/i;
  const FORMAT_NAMES = {
    safetensors: "safetensors", pytorch: "PyTorch checkpoint", "pytorch-legacy": "PyTorch checkpoint (legacy format)",
    torchscript: "TorchScript module", pickle: "Python pickle", npy: "NumPy array", npz: "NumPy archive",
    gguf: "GGUF", onnx: "ONNX model",
  };
  const PICKLED = new Set(["pytorch", "pytorch-legacy", "torchscript", "pickle"]);
  const FLAT_LIMIT = 2000;     // rows shown for a filtered tensor list

  function fmtCount(n) {
    if (n == null) return "?";
    const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];
    for (const [v, u] of units) if (n >= v) return (n / v).toFixed(n >= v * 100 ? 0 : 1).replace(/\.0$/, "") + u;
    return String(n);
  }
  const fmtShape = s => "[" + s.join(", ") + "]";
  const fmtVal = v => typeof v === "number" && !Number.isInteger(v) ? String(+v.toPrecision(5)) : String(v);

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function infoURL(artifactURL) { return artifactURL.replace(/\/artifacts\//, "/artifact-info/"); }

  // ── Summary ────────────────────────────────────────────────────────────────
  function summaryChips(info) {
    const chips = [`<span class="mp-chip mp-format">${escHtml(FORMAT_NAMES[info.format] || info.format)}</span>`];
    chips.push(`<span class="mp-chip" title="${info.total_params.toLocaleString()} elements">${fmtCount(info.total_params)} elements</span>`);
    chips.push(`<span class="mp-chip">${info.tensor_count.toLocaleString()} ${info.format === "npy" ? "array" : "tensors"}</span>`);
    if (info.total_tensor_bytes) chips.push(`<span class="mp-chip">${fmtSize(info.total_tensor_bytes)} of data</span>`);
    const dtypes = Object.entries(info.dtypes || {}).sort((a, b) => b[1] - a[1]);
    for (const [d, n] of dtypes.slice(0, 6)) {
      const pct = info.total_params ? Math.round(100 * n / info.total_params) : 0;
      chips.push(`<span class="mp-chip mp-dtype" title="${n.toLocaleString()} elements">${escHtml(d)}${dtypes.length > 1 ? ` ${pct}%` : ""}</span>`);
    }
    return `<div class="mp-chips">${chips.join("")}</div>`;
  }

  // Scalars worth comparing across runs: top-level pickle entries (epoch,
  // global_step, …) and string/number metadata.
  function headlineScalars(info) {
    const out = [];
    const tree = info.tree;
    if (tree && typeof tree === "object" && !Array.isArray(tree) && !("__tensor__" in tree)) {
      for (const [k, v] of Object.entries(tree.__object__ ? (tree.state || {}) : tree)) {
        if (v === null || ["number", "string", "boolean"].includes(typeof v)) out.push([k, v]);
      }
    }
    for (const [k, v] of Object.entries(info.metadata || {})) {
      if (["number", "string", "boolean"].includes(typeof v)) out.push([k, v]);
    }
    return out;
  }

  function kvTable(rows) {
    const t = el("table", "mp-kv");
    for (const [k, v] of rows) {
      const tr = t.insertRow();
      tr.insertCell().textContent = k;
      const td = tr.insertCell();
      if (v && typeof v === "object" && "array" in v && Array.isArray(v.head))   // a GGUF array
        td.textContent = `${v.array.toLocaleString()} items: ${v.head.map(fmtVal).join(", ")}${v.array > v.head.length ? ", …" : ""}`;
      else td.textContent = typeof v === "object" && v !== null ? JSON.stringify(v) : fmtVal(v);
    }
    return t;
  }

  function section(title, body, open = true) {
    const d = el("details", "mp-section");
    d.open = open;
    d.appendChild(el("summary", null, escHtml(title)));
    d.appendChild(body);
    return d;
  }

  // ── Tensor tree, grouped on "." ─────────────────────────────────────────────
  function buildGroups(tensors) {
    const root = { name: "", children: new Map(), tensors: [], numel: 0, count: 0 };
    for (const t of tensors) {
      const parts = t.name.split(".");
      let node = root;
      node.numel += t.numel; node.count++;
      for (const p of parts.slice(0, -1)) {
        if (!node.children.has(p)) node.children.set(p, { name: p, children: new Map(), tensors: [], numel: 0, count: 0 });
        node = node.children.get(p);
        node.numel += t.numel; node.count++;
      }
      node.tensors.push(t);
    }
    return collapseChains(root);
  }

  // "model" → "layers" → … with one child each reads better as one row.
  function collapseChains(node) {
    for (const [k, c] of node.children) node.children.set(k, collapseChains(c));
    if (node.name && node.children.size === 1 && !node.tensors.length) {
      const [only] = node.children.values();
      return { ...only, name: node.name + "." + only.name };
    }
    return node;
  }

  function tensorRow(t, label) {
    const row = el("div", "mp-trow");
    const vals = t.values ? `<span class="mp-vals" title="${escHtml(t.values.map(fmtVal).join(", "))}">= ${escHtml(t.values.map(fmtVal).join(", "))}</span>` : "";
    row.innerHTML =
      `<span class="mp-tname" title="${escHtml(t.name)}">${escHtml(label)}</span>` +
      `<span class="mp-tdtype">${escHtml(t.dtype)}${t.kind && t.kind !== "tensor" ? ` <span class="mp-kind">${escHtml(t.kind)}</span>` : ""}</span>` +
      `<span class="mp-tshape">${escHtml(fmtShape(t.shape))}</span>` +
      `<span class="mp-tnumel" title="${t.numel.toLocaleString()} elements${t.bytes != null ? ", " + fmtSize(t.bytes) : ""}">${fmtCount(t.numel)}</span>` + vals;
    return row;
  }

  function groupNode(node, depth) {
    const d = el("details", "mp-group");
    const s = el("summary", null,
      `<span class="mp-gname">${escHtml(node.name)}</span>` +
      `<span class="mp-gmeta">${node.count.toLocaleString()} · ${fmtCount(node.numel)}</span>`);
    d.appendChild(s);
    const body = el("div", "mp-gbody");
    d.appendChild(body);
    let filled = false;
    const fill = () => {
      if (filled) return;
      filled = true;
      fillGroup(body, node, depth + 1);
    };
    d._fill = fill;
    d.addEventListener("toggle", () => { if (d.open) fill(); });
    if (depth < 1 && node.count <= 400) { d.open = true; fill(); }
    return d;
  }

  function fillGroup(container, node, depth) {
    for (const c of node.children.values()) container.appendChild(groupNode(c, depth));
    for (const t of node.tensors) container.appendChild(tensorRow(t, t.name.split(".").pop()));
  }

  // A button that opens every <details> under *container*, or closes them all.
  // Groups fill in lazily, so each pass fills what it opens and the loop goes
  // again for the children that just appeared.
  function expandAllButton(container) {
    const btn = el("button", "btn-sm mp-expand", "Expand all");
    btn.type = "button";
    btn.addEventListener("click", () => {
      const expand = btn.textContent === "Expand all";
      if (expand) {
        for (let closed; (closed = container.querySelectorAll("details:not([open])")).length;)
          for (const d of closed) { d._fill?.(); d.open = true; }
      } else {
        for (const d of container.querySelectorAll("details[open]")) d.open = false;
      }
      btn.textContent = expand ? "Collapse all" : "Expand all";
    });
    return btn;
  }

  function tensorSection(info) {
    const wrap = el("div", "mp-tensors");
    const bar = el("div", "mp-filterbar");
    const input = el("input", "mp-filter");
    input.type = "search"; input.placeholder = "Filter tensors by name";
    bar.appendChild(input);
    if (info.tensors_truncated) bar.appendChild(el("span", "mp-note", `showing the first ${info.tensors.length.toLocaleString()} of ${info.tensor_count.toLocaleString()}`));
    const list = el("div", "mp-tlist");
    const expand = expandAllButton(list);
    bar.insertBefore(expand, input.nextSibling);
    wrap.append(bar, list);

    const groups = buildGroups(info.tensors);
    const showTree = () => {
      list.innerHTML = ""; fillGroup(list, groups, 0);
      expand.textContent = "Expand all";
      expand.hidden = !list.querySelector("details");     // a flat list has nothing to expand
    };
    showTree();
    let timer = null;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const q = input.value.trim().toLowerCase();
        if (!q) { showTree(); return; }
        list.innerHTML = "";
        expand.hidden = true;
        const hits = info.tensors.filter(t => t.name.toLowerCase().includes(q));
        for (const t of hits.slice(0, FLAT_LIMIT)) list.appendChild(tensorRow(t, t.name));
        const numel = hits.reduce((a, t) => a + t.numel, 0);
        list.prepend(el("div", "mp-note", `${hits.length.toLocaleString()} match · ${fmtCount(numel)} elements` +
          (hits.length > FLAT_LIMIT ? ` · first ${FLAT_LIMIT} shown` : "")));
      }, 120);
    });
    return wrap;
  }

  // ── Pickle structure: everything that is not a tensor ─────────────────────
  function isTensorRef(v) { return v && typeof v === "object" && !Array.isArray(v) && "__tensor__" in v; }

  function hasNonTensor(v) {
    if (isTensorRef(v)) return false;
    if (v === null || typeof v !== "object") return true;
    const vals = Array.isArray(v) ? v : Object.values(v);
    return vals.length === 0 || vals.some(hasNonTensor);
  }

  function structNode(key, v, info, depth) {
    const label = key == null ? "" : `<span class="mp-skey">${escHtml(String(key))}</span>`;
    if (isTensorRef(v)) {
      const t = info.tensors[v.__tensor__];
      const desc = t ? `${t.dtype}${fmtShape(t.shape)}${t.values ? " = " + t.values.map(fmtVal).join(", ") : ""}` : "tensor";
      return el("div", "mp-sleaf", `${label}<span class="mp-stensor">${escHtml(desc)}</span>`);
    }
    if (v === null || typeof v !== "object") {
      const cls = typeof v === "string" ? "mp-sstr" : "mp-snum";
      const text = typeof v === "string" ? JSON.stringify(v) : String(v);
      return el("div", "mp-sleaf", `${label}<span class="${cls}">${escHtml(text)}</span>`);
    }
    let entries, tag = "";
    if (Array.isArray(v)) { entries = v.map((x, i) => [i, x]); tag = `list(${v.length})`; }
    else if (v.__object__) {
      tag = v.__object__;
      entries = [];
      for (const part of ["args", "kwargs", "state", "items", "list"]) if (part in v) entries.push([part, v[part]]);
      if (entries.length === 1 && entries[0][0] === "state" && entries[0][1] && typeof entries[0][1] === "object" && !Array.isArray(entries[0][1]))
        entries = Object.entries(entries[0][1]);
    } else { entries = Object.entries(v); tag = `dict(${entries.length})`; }

    const refs = entries.filter(([, x]) => isTensorRef(x));
    if (entries.length && refs.length === entries.length) {
      const numel = refs.reduce((a, [, r]) => a + ((info.tensors[r.__tensor__] || {}).numel || 0), 0);
      tag += ` · ${refs.length} tensors, ${fmtCount(numel)} elements`;
    }
    const d = el("details", "mp-snode");
    d.appendChild(el("summary", null, `${label}<span class="mp-stag">${escHtml(tag)}</span>`));
    const body = el("div", "mp-sbody");
    d.appendChild(body);
    let filled = false;
    const fill = () => {
      if (filled) return;
      filled = true;
      for (const [k, x] of entries) body.appendChild(structNode(k, x, info, depth + 1));
    };
    d._fill = fill;
    d.addEventListener("toggle", () => { if (d.open) fill(); });
    // Open the outer levels unless they are just a wall of tensors.
    if (depth < 2 && entries.length <= 60 && refs.length < entries.length) { d.open = true; fill(); }
    return d;
  }

  function valueInfoTable(rows) {
    const t = el("table", "mp-kv");
    for (const r of rows) {
      const tr = t.insertRow();
      tr.insertCell().textContent = r.name;
      tr.insertCell().textContent = `${r.dtype}${fmtShape(r.shape)}`;
    }
    return t;
  }

  // ── Rendering ──────────────────────────────────────────────────────────────
  function renderFull(info, target) {
    target.innerHTML = "";
    const root = el("div", "mp-root");
    root.innerHTML = summaryChips(info);
    const meta = Object.entries(info.metadata || {});
    if (meta.length) root.appendChild(section("Metadata", kvTable(meta), meta.length <= 40));
    if (info.inputs?.length || info.outputs?.length) {
      const io = el("div", "mp-io");
      if (info.inputs?.length) { io.appendChild(el("div", "mp-subtitle", "Inputs")); io.appendChild(valueInfoTable(info.inputs)); }
      if (info.outputs?.length) { io.appendChild(el("div", "mp-subtitle", "Outputs")); io.appendChild(valueInfoTable(info.outputs)); }
      root.appendChild(section("Graph inputs and outputs", io));
    }
    if (info.ops && Object.keys(info.ops).length) {
      const total = Object.values(info.ops).reduce((a, b) => a + b, 0);
      root.appendChild(section(`Operators (${total.toLocaleString()} nodes)`, kvTable(Object.entries(info.ops)), false));
    }
    if (PICKLED.has(info.format) && info.tree !== undefined && hasNonTensor(info.tree)) {
      const wrap = el("div");
      const s = el("div", "mp-struct");
      s.appendChild(structNode(null, info.tree, info, 0));
      if (info.tree_truncated) s.appendChild(el("div", "mp-note", "Large structure, parts elided (…)"));
      const bar = el("div", "mp-filterbar");
      bar.appendChild(expandAllButton(s));
      wrap.append(bar, s);
      root.appendChild(section("Structure", wrap));
    }
    if (info.tensor_count) root.appendChild(section(info.format === "npy" ? "Array" : "Tensors", tensorSection(info)));
    else root.appendChild(el("div", "mp-note", "No tensors found."));
    target.appendChild(root);
  }

  function renderCompact(info, target) {
    target.innerHTML = "";
    const root = el("div", "mp-root mp-compact");
    root.innerHTML = summaryChips(info);
    const scalars = headlineScalars(info).slice(0, 8);
    if (scalars.length) root.appendChild(kvTable(scalars));
    else if (info.tensors.length) {
      const top = [...info.tensors].sort((a, b) => b.numel - a.numel).slice(0, 4);
      const list = el("div", "mp-tlist");
      for (const t of top) list.appendChild(tensorRow(t, t.name));
      root.appendChild(list);
    }
    target.appendChild(root);
  }

  // Render a preview of *artifactURL* into *target*.  Resolves false when the
  // manager could not read the file, after showing why.  *info* is the summary's
  // own URL when it is not the artifact's (a file in a code snapshot).
  async function render(artifactURL, target, compact, { info: summaryURL = null } = {}) {
    target.innerHTML = `<span class="no-file">Reading file…</span>`;
    let info;
    try {
      const r = await apiFetch(summaryURL || infoURL(artifactURL));
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { msg = (await r.json()).error || msg; } catch {}
        target.innerHTML = `<span class="no-file">${r.status === 422 ? "Binary file, no preview: " : "Preview failed: "}${escHtml(msg)}</span>`;
        return false;
      }
      info = await r.json();
    } catch (e) {
      target.innerHTML = `<span class="no-file">Preview failed: ${escHtml(String(e))}</span>`;
      return false;
    }
    (compact ? renderCompact : renderFull)(info, target);
    return true;
  }

  return { isModel: p => EXT.test(p), render, fmtCount };
})();
