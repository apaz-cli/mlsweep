// mlsweep Notes: Live mode, after mded.  The note is shown rendered; click a
// block to edit its markdown in place, press Esc or click elsewhere and it
// renders again.  Blocks you don't touch stay byte-for-byte as they were.
//
// The tab's CodeMirror doc stays the one source of truth: committing a block
// rewrites the doc's text, so saving, undo, sync conflicts and Source mode
// all keep working unchanged.
"use strict";

// ── Lossless block splitting: serialize(parse(t)) === t ───────────────────────
const LV_BLANK = /^\s*$/;
const LV_LIST = /^\s{0,3}([-*+]|\d+[.)])\s/;
const LV_DIV_OPEN = /^:{3,}\s*[^:\s]/;
const LV_DIV_CLOSE = /^:{3,}\s*$/;

function lvDepthDelta(line) {
  if (LV_DIV_OPEN.test(line)) return 1;
  if (LV_DIV_CLOSE.test(line)) return -1;
  return (line.match(/<div\b/gi) || []).length - (line.match(/<\/div\s*>/gi) || []).length;
}
function lvContainerEnd(L, i) {
  if (lvDepthDelta(L[i]) <= 0) return -1;
  let depth = 0, fence = null;
  for (let j = i; j < L.length; j++) {
    const was = fence;
    fence = lvNextFence(fence, L[j]);
    if (was || fence) continue;
    depth += lvDepthDelta(L[j]);
    if (depth <= 0) return j + 1;
  }
  return -1;
}

function lvParse(text, allowFrontmatter = true) {
  const L = text.split("\n");
  const blocks = [];
  let i = 0;
  // YAML frontmatter is one block, shown as a properties table.
  if (allowFrontmatter && L[0] === "---") {
    const end = L.indexOf("---", 1);
    if (end > 0) { blocks.push({ pre: [], lines: L.slice(0, end + 1), fm: true }); i = end + 1; }
  }
  while (i < L.length) {
    const pre = [];
    while (i < L.length && LV_BLANK.test(L[i])) pre.push(L[i++]);
    if (i >= L.length) return { blocks, tail: pre };
    const start = i;
    const end = lvContainerEnd(L, i);
    if (end > 0) {
      i = end;
    } else {
      const isList = LV_LIST.test(L[i]);
      let fence = null;
      while (i < L.length) {
        if (!fence && LV_BLANK.test(L[i])) {
          let j = i;
          while (j < L.length && LV_BLANK.test(L[j])) j++;
          const next = L[j];
          const continues = next !== undefined && isList && (/^(\s{2,}|\t)/.test(next) || LV_LIST.test(next));
          if (!continues) break;
          i = j; continue;
        }
        fence = lvNextFence(fence, L[i++]);
      }
    }
    blocks.push({ pre, lines: L.slice(start, i) });
  }
  return { blocks, tail: [] };
}
function lvSerialize(d) {
  const out = [];
  for (const b of d.blocks) out.push(...b.pre, ...b.lines);
  out.push(...d.tail);
  return out.join("\n");
}
const lvSrc = b => b.lines.join("\n");

// ── Rendering one block ────────────────────────────────────────────────────────
function lvRenderBlock(el, b, path) {
  el.innerHTML = "";
  if (b.fm) {
    const props = propsEl(b.lines.slice(1, -1));
    if (props.children.length) el.append(props); else lvGhost(el, b);
    return;
  }
  el.innerHTML = DOMPurify.sanitize(MD.parse(lvSrc(b)), PURIFY);
  postProcess(el, path, 0);
  const visible = el.textContent.trim() !== "" || el.querySelector("img,hr,iframe,svg,table,canvas,.katex,.embed-note,input");
  if (!visible) lvGhost(el, b);
}
// Blocks that render to nothing (comments, link definitions) show their source, greyed.
function lvGhost(el, b) {
  el.innerHTML = "";
  const g = el.appendChild(document.createElement("div"));
  g.className = "lv-ghost";
  g.textContent = lvSrc(b);
}

// ── The Live pane ──────────────────────────────────────────────────────────────
function livePane(t) {
  const pane = el("div", "vpane reader live");
  const art = el("article", "doc lv-doc");
  pane.append(art);
  const L = { t, pane, art, doc: null, editing: -1, ta: null };
  t.live = L;
  t.rerender = null;
  lvDraw(L);
  art.addEventListener("mousedown", e => lvMouseDown(L, e));
  art.addEventListener("click", e => {
    // The block under the click may already have turned into a textarea.
    if (!e.target.isConnected || e.target.closest("textarea")) return;
    onDocClick(e, t);
  });
  art.addEventListener("change", e => lvTaskToggle(L, e));
  requestAnimationFrame(() => {
    if (t.pendingHeading) { scrollToHeading(pane, t.pendingHeading); t.pendingHeading = null; }
    else if (t.pendingLine) { lvRevealLine(L, t.pendingLine); t.pendingLine = null; }
    else if (t.scroll.read != null) pane.scrollTop = t.scroll.read;
    if (t.liveResume) { lvResume(L, t.liveResume); t.liveResume = null; }
    else if (t.focusEnd) { t.focusEnd = false; lvEdit(L, Math.max(0, L.doc.blocks.length - 1)); }
  });
  return pane;
}

function lvDraw(L) {
  L.doc = lvParse(L.t.doc.getValue());
  L.editing = -1; L.ta = null;
  L.art.innerHTML = "";
  for (const b of L.doc.blocks) L.art.append(lvBlockEl(L, b));
  const add = el("div", "lv-add", "+ click to add a block");
  add.title = "Add a block at the end (Ctrl+Enter inside a block adds one below it)";
  L.art.append(add);
}
function lvBlockEl(L, b) {
  const d = el("div", "lv-block");
  lvRenderBlock(d, b, L.t.path);
  return d;
}
const lvIndex = (L, node) => [...L.art.children].indexOf(node);

// Replace blocks [i, i+del) with items, in the model and on the page.
function lvSplice(L, i, del, ...items) {
  L.doc.blocks.splice(i, del, ...items);
  const gone = [...L.art.children].slice(i, i + del);
  L.art.children[i].before(...items.map(b => lvBlockEl(L, b)));   // the "add" row is always last
  gone.forEach(n => n.remove());
}

// Write the model back into the tab's doc (autosave and dirty state follow).
function lvWriteBack(L) {
  const text = lvSerialize(L.doc);
  if (text === L.t.doc.getValue()) return;
  const last = L.t.doc.lastLine();
  L.t.doc.replaceRange(text, { line: 0, ch: 0 }, { line: last, ch: L.t.doc.getLine(last).length }, "+live");
}

function lvAutosize(ta) { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + 2 + "px"; }

function lvEdit(L, i, caret = "end") {
  if (L.editing >= 0) {
    const was = L.editing, n = lvCommit(L);
    if (i > was) i += n - 1;
  }
  if (i < 0 || i >= L.doc.blocks.length) return;
  L.editing = i;
  const node = L.art.children[i];
  const ta = document.createElement("textarea");
  ta.className = "lv-src";
  ta.spellcheck = true;
  ta.value = lvSrc(L.doc.blocks[i]);
  node.classList.add("editing");
  node.replaceChildren(ta);
  L.ta = ta;
  lvAutosize(ta);
  ta.focus({ preventScroll: true });
  const pos = caret === "start" ? 0 : typeof caret === "number" ? caret : ta.value.length;
  ta.setSelectionRange(pos, pos);
  ta.addEventListener("input", () => { lvAutosize(ta); lvMarkTyping(L); });
  ta.addEventListener("keydown", e => lvKey(L, e));
  // Switching windows also blurs; keep editing in that case.
  ta.addEventListener("blur", () => { if (L.editing === i && L.ta === ta && ta.isConnected && document.hasFocus()) lvCommit(L); });
  ta.addEventListener("paste", e => onPaste(null, e));
  linkHintTextarea(ta);
}

// While a block is open the tab counts as edited, so a sync never replaces
// the text under the cursor.
function lvMarkTyping(L) {
  const t = L.t;
  if (!t.dirty) { t.dirty = true; renderTabs(); updateSaveState(t); }
  clearTimeout(autosave.get(t.id));
  autosave.set(t.id, setTimeout(() => { if (L.editing >= 0 && L.ta) lvCommit(L, { keepOpen: true }); }, 2500));
}

// Fold the textarea back into the document, re-splitting it into blocks.
// Returns the number of blocks it became.
function lvCommit(L, { keepOpen = false } = {}) {
  const i = L.editing;
  if (i < 0) return 1;
  const ta = L.ta;
  const old = L.doc.blocks[i];
  const text = ta ? ta.value : lvSrc(old);
  if (keepOpen) {
    lvSyncOpen(L);
    if (L.t.dirty) saveTab(L.t);
    return 1;
  }
  L.editing = -1; L.ta = null;
  if (!LV_BLANK.test(text) && text === lvSrc(old)) {
    L.art.children[i].replaceWith(lvBlockEl(L, old));
    lvAfterChange(L);
    return 1;
  }
  let repl = [];
  if (!LV_BLANK.test(text)) {
    const sub = lvParse(text, i === 0);
    repl = sub.blocks;
    if (repl.length) repl[0].pre = old.pre.concat(repl[0].pre);
  }
  lvSplice(L, i, 1, ...repl);
  lvWriteBack(L);
  lvAfterChange(L);
  return repl.length;
}

// Write the open block's text into the doc, leaving it open (autosave while typing).
function lvSyncOpen(L) {
  const i = L.editing;
  if (i < 0 || !L.ta) return;
  const old = L.doc.blocks[i], text = L.ta.value;
  if (text !== lvSrc(old) && !LV_BLANK.test(text)) {
    L.doc.blocks[i] = { ...old, lines: text.split("\n"), fm: old.fm && text.startsWith("---") };
    lvWriteBack(L);
  }
  L.t.dirty = L.t.doc.getValue() !== L.t.saved;
}

function lvAfterChange(L) {
  const t = L.t;
  t.dirty = t.doc.getValue() !== t.saved;
  renderTabs(); updateSaveState(t);
  clearTimeout(autosave.get(t.id));
  if (t.dirty) autosave.set(t.id, setTimeout(() => saveTab(t), 600));
  refreshOutline(t);
}

function lvInsert(L, at) {
  if (L.editing >= 0) { const n = lvCommit(L); if (at > L.editing) at += n - 1; }
  const b = { pre: at === 0 && !L.doc.blocks.length ? [] : [""], lines: [""] };
  if (at === 0 && L.doc.blocks.length) { b.pre = L.doc.blocks[0].pre; L.doc.blocks[0].pre = [""]; }
  lvSplice(L, at, 0, b);
  lvEdit(L, at);
}

function lvKey(L, e) {
  const ta = e.target, i = L.editing;
  const before = ta.value.slice(0, ta.selectionStart), after = ta.value.slice(ta.selectionEnd);
  const mod = e.ctrlKey || e.metaKey;
  if (e.key === "Escape") {
    if (ta._hint && ta._hint.isConnected) return;
    e.preventDefault(); lvCommit(L);
  } else if (e.key === "Enter" && mod) {
    e.preventDefault();
    const n = lvCommit(L); lvInsert(L, i + n);
  } else if (e.key === "ArrowUp" && !e.shiftKey && !mod && !before.includes("\n") && i > 0) {
    e.preventDefault(); lvCommit(L); lvEdit(L, i - 1);
  } else if (e.key === "ArrowDown" && !e.shiftKey && !mod && !after.includes("\n") && ta.selectionStart === ta.selectionEnd) {
    e.preventDefault();
    const n = lvCommit(L);
    if (i + n < L.doc.blocks.length) lvEdit(L, i + n, "start");
  } else if (e.key === "Backspace" && ta.value === "") {
    e.preventDefault(); lvCommit(L); if (i > 0) lvEdit(L, i - 1);
  } else if (e.key === "Tab" && !mod) {
    e.preventDefault();
    document.execCommand("insertText", false, e.shiftKey ? "" : "  ");
  } else if (mod && !e.shiftKey && (e.key === "b" || e.key === "i")) {
    e.preventDefault();
    const m = e.key === "b" ? "**" : "*";
    const s = ta.value.slice(ta.selectionStart, ta.selectionEnd);
    document.execCommand("insertText", false, m + s + m);
    if (!s) ta.setSelectionRange(ta.selectionStart - m.length, ta.selectionStart - m.length);
  } else if (mod && e.key.toLowerCase() === "s") {
    e.preventDefault(); e.stopPropagation(); lvCommit(L, { keepOpen: true }); saveActiveNow();
  }
}

function lvMouseDown(L, e) {
  if (e.button !== 0) return;
  if (e.target.closest("textarea")) return;
  // Links, experiment chips and checkboxes act; everything else edits.
  if (e.target.closest("a, input, .embed-note .embed-title")) return;
  const blk = e.target.closest(".lv-block");
  if (blk) {
    e.preventDefault();
    const i = lvIndex(L, blk);
    if (i !== L.editing) lvEdit(L, i);
  } else if (e.target.closest(".lv-add")) {
    e.preventDefault();
    lvInsert(L, L.doc.blocks.length);
  }
}

function lvTaskToggle(L, e) {
  const cb = e.target;
  if (!cb.matches("input[type=checkbox][data-task]")) return;
  const blk = cb.closest(".lv-block");
  const i = lvIndex(L, blk);
  const b = L.doc.blocks[i];
  const lines = b.lines.slice();
  const at = findTask(lines, Number(cb.dataset.task));
  if (at) lines[at.line] = lines[at.line].slice(0, at.ch) + (cb.checked ? "x" : " ") + lines[at.line].slice(at.ch + 1);
  L.doc.blocks[i] = { ...b, lines };
  cb.parentElement.classList.toggle("done", cb.checked);
  lvWriteBack(L);
  lvAfterChange(L);
}

// Scroll to the block containing (1-based) line `line`.
function lvRevealLine(L, line) {
  let n = 0;
  for (let i = 0; i < L.doc.blocks.length; i++) {
    const b = L.doc.blocks[i];
    n += b.pre.length;
    if (line <= n + b.lines.length) {
      const node = L.art.children[i];
      node.scrollIntoView({ block: "center" });
      node.animate([{ background: "var(--accent-soft)" }, { background: "transparent" }], { duration: 1400 });
      return;
    }
    n += b.lines.length;
  }
}

// Before the doc is replaced (sync, undo): remember the open block.
function lvSuspend(t) {
  const L = t.live;
  if (!L || L.editing < 0 || !L.ta || !L.ta.isConnected) return null;
  return { orig: lvSrc(L.doc.blocks[L.editing]), value: L.ta.value,
           start: L.ta.selectionStart, end: L.ta.selectionEnd, index: L.editing };
}
// After a redraw: reopen it where its original text now is.
function lvResume(L, s) {
  let i = L.doc.blocks.findIndex(b => lvSrc(b) === s.orig);
  if (i < 0) i = Math.min(s.index, L.doc.blocks.length - 1);
  if (i < 0) return;
  lvEdit(L, i);
  L.ta.value = s.value;
  lvAutosize(L.ta);
  L.ta.setSelectionRange(s.start, s.end);
}

// ── [[ completion for the block textarea ───────────────────────────────────────
function linkHintTextarea(ta) {
  ta.addEventListener("input", () => {
    const before = ta.value.slice(0, ta.selectionStart);
    const m = /\[\[([^\]\[|#\n]*)$/.exec(before);
    if (!m) { hideTaHint(ta); return; }
    const items = rankPaths(m[1], S.files.map(f => f.path), 12);
    if (!items.length) { hideTaHint(ta); return; }
    showTaHint(ta, items, m[1].length);
  });
  ta.addEventListener("keydown", e => {
    const h = ta._hint;
    if (!h || !h.isConnected) return;
    const rows = [...h.children];
    const on = "CodeMirror-hint-active";
    let sel = rows.findIndex(r => r.classList.contains(on));
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); e.stopImmediatePropagation();
      rows[sel]?.classList.remove(on);
      sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length;
      rows[sel].classList.add(on); rows[sel].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" || e.key === "Tab") {
      e.preventDefault(); e.stopImmediatePropagation(); rows[Math.max(0, sel)].onmousedown(e);
    } else if (e.key === "Escape") {
      e.preventDefault(); e.stopImmediatePropagation(); hideTaHint(ta);
    }
  }, true);
  ta.addEventListener("blur", () => setTimeout(() => hideTaHint(ta), 150));
}
function hideTaHint(ta) { if (ta._hint) { ta._hint.remove(); ta._hint = null; } }
function showTaHint(ta, items, typed) {
  hideTaHint(ta);
  const h = el("ul", "CodeMirror-hints");
  h.style.position = "fixed";
  const r = ta.getBoundingClientRect();
  h.style.left = r.left + 12 + "px";
  h.style.top = Math.min(innerHeight - 220, r.bottom + 2) + "px";
  items.forEach((p, k) => {
    const li = el("li", "CodeMirror-hint" + (k === 0 ? " CodeMirror-hint-active" : ""));
    li.append(document.createTextNode(titleFor(p)));
    if (dirname(p)) li.append(el("span", "hp", dirname(p)));
    li.onmousedown = e => {
      e.preventDefault();
      const name = wikiName(p, null);
      const start = ta.selectionStart - typed;
      const rest = ta.value.slice(ta.selectionStart);
      const close = rest.startsWith("]]") ? "" : "]]";
      ta.setSelectionRange(start, ta.selectionStart);
      document.execCommand("insertText", false, name + close);
      if (!close) ta.setSelectionRange(ta.selectionStart + 2, ta.selectionStart + 2);
      hideTaHint(ta);
    };
    h.append(li);
  });
  document.body.append(h);
  ta._hint = h;
}
