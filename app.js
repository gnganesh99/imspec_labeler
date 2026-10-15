// ImSpec Labeler UI: folder access, keyboard labeling, canvas / uPlot rendering.
// Python (worker.js + py/reader.py) parses files; all folder IO happens here.
"use strict";

const LOG_DIR = "labeled";
const LOG_FILE = "classification_log.csv";
// one row per file and mode: binary / multiclass rows fill label (+ class_id), score rows fill score
// flatten: the flatten type applied to the image when the entry was made (empty for spectra)
// array: the training array saved for this entry (a file in labeled/training_data/arrays/), empty when none was saved
// sample_mean / sample_std: mean and std of the labeled array's non-NaN values, in the units of the file
const LOG_COLUMNS = ["file", "type", "channel", "flatten", "mode", "label", "class_id", "score", "tags", "note", "array",
                     "sample_mean", "sample_std", "timestamp"];
const DEFER_LABEL = "Deferred";   // always available on key D; copies go to labeled/deferred/
const UNSURE_LABEL = "Unsure";    // always available on key U; copies go to labeled/unsure/
const noOutput = (label) => label === DEFER_LABEL || label === UNSURE_LABEL;   // logged, but no class / score for training
const COPY_MODES = ["none", "original", "image"];
const SUPPORTED = /\.(sxm|dat|3ds|jpe?g|png|bmp|gif|webp|tiff?)$/i;
const PREFETCH = 3;
const VIEW_CACHE_SIZE = 12;
const GUTTER = 128;   // stage padding (left + right) that keeps the ‹ › buttons clear of the content
const FLATTENS = ["none", "offset", "line", "plane"];
const DEFAULT_SETTINGS = {
  sxm: { flatten: "offset", cmap: "afmhot" }, "3ds": { flatten: "none", cmap: "viridis" },
  dat: {}, img: { flatten: "none", cmap: "gray" },
};

const APP_VERSION = "2026-10-15 · Open folder in the header, Bad left / Good right, note box in the panel, Unsure (U)";   // shown in the header: tells which build runs
const $ = (id) => document.getElementById(id);
const state = {
  dir: null, config: mergeConfig({}), userConfig: {}, mode: "binary", scoreBuf: "",
  files: [], rows: new Map(), done: new Set(), pos: -1,   // rows: `${mode}|${file}` -> log row; done: files done in this mode
  current: null,    // {name, type, key, summary, view}
  gen: 0, note: "", workerReady: false, settings: loadSettings(),
  copyAs: loadPref("spm-labeler-copy-as", "none"),   // "none" | "original" | "image" (jpeg of the view)
};
const viewCache = new Map();   // `${key}|${settingsKey}` -> {summary, view}
let ioChain = Promise.resolve();
let plot = null;

// --------------------------------------------------------------------------- worker RPC
const worker = new Worker("worker.js?v=" + BUILD_V);   // BUILD_V: see index.html
const pending = new Map();
let nextId = 1;
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.status) return onWorkerStatus(m);
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
};
function call(op, args, prio = 0, transfer = []) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, prio, gen: state.gen, ...args }, transfer);
  });
}
function onWorkerStatus(m) {
  if (m.status === "ready") {
    state.workerReady = true;
    setStatus("");
  } else if (m.status === "error") {
    setStatus("Python failed to load: " + m.error);
  } else {
    setStatus(m.status + (state.dir ? "" : " (first visit downloads ~25 MB, cached afterwards)"));
  }
}

// --------------------------------------------------------------------------- settings
function loadSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("spm-labeler-settings") || "{}"); } catch {}
  const out = {};
  for (const t of Object.keys(DEFAULT_SETTINGS)) {
    out[t] = { channel: null, direction: "forward", flatten: "none", cmap: "viridis", index: null,
               transpose: false, origin: "auto", scale: "ticks", clip: 1,   // View panel: display only, no re-render
               ...DEFAULT_SETTINGS[t], ...(saved[t] || {}) };
  }
  return out;
}
function saveSettings() {
  try { localStorage.setItem("spm-labeler-settings", JSON.stringify(state.settings)); } catch {}
}
function settingsKey(s) { return [s.channel, s.direction, s.flatten, s.index].join("|"); }

// --------------------------------------------------------------------------- folder
// a file's identity for the caches (Python's parsed files, the view cache): the folder counter keeps two
// folders that hold files of the same name, size and date apart
function fileKey(name, file) { return `${state.folderId}|${name}|${file.size}|${file.lastModified}`; }

async function openFolder(handle) {
  if (datasetBuilding) return toast("Wait until Build dataset has finished, then open the other folder.");
  try {
    if (!handle) handle = await window.showDirectoryPicker({ id: "spm-labeler", mode: "readwrite" });
    else if ((await handle.requestPermission({ mode: "readwrite" })) !== "granted") return;
  } catch (e) {
    if (e.name !== "AbortError") toast(e.message);
    return;
  }
  await ioChain;   // entries made in the previous folder are written there before switching
  state.folderId = (state.folderId || 0) + 1;
  $("note").value = "";   // a note typed for the previous folder
  state.dir = handle;
  rememberHandle(handle);
  $("folder").textContent = handle.name;
  const userConfig = await readJson("labeler.json");
  const files = [];
  for await (const [name, h] of handle.entries()) {
    if (h.kind === "file" && SUPPORTED.test(name)) files.push(name);
  }
  // oldest first by modified time (the browser does not expose creation time); name breaks ties
  state.mtimes = new Map();
  for (let i = 0; i < files.length; i += 100) {
    await Promise.all(files.slice(i, i + 100).map(async (n) => {
      state.mtimes.set(n, (await (await handle.getFileHandle(n)).getFile()).lastModified);
    }));
  }
  state.files = files.sort((a, b) => state.mtimes.get(a) - state.mtimes.get(b) || naturalCompare(a, b));
  setLabels(await readLog());
  applyConfig(userConfig);   // mode, classes, copy mode ... from labeler.json
  await loadDatasetConfig();   // build settings for the training dataset
  viewCache.clear();
  $("message").classList.add("hidden");
  $("side").classList.remove("hidden");
  $("hbtns").classList.remove("hidden");
  $("modeBox").classList.remove("hidden");
  $("cfgText").disabled = false;
  if (!state.files.length) return showMessage("No supported files in this folder", "");
  goTo(nextUndone(0));
}

// settings live in labeled/ (nothing else is written to the data folder). A labeler.json left in the data
// folder by an earlier version is still read when there is none in labeled/, and is never changed or deleted.
async function readJson(name) {
  const dirs = [];
  try { dirs.push(await state.dir.getDirectoryHandle(LOG_DIR)); } catch {}
  dirs.push(state.dir);
  for (const dir of dirs) {
    try {
      const f = await (await dir.getFileHandle(name)).getFile();
      return JSON.parse(await f.text());
    } catch (e) {
      if (e.name !== "NotFoundError") { toast(`${name} ignored: ${e.message}`); return {}; }
    }
  }
  return {};
}

function naturalCompare(a, b) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function fileType(name) {
  const ext = name.split(".").pop().toLowerCase();
  return ["sxm", "dat", "3ds"].includes(ext) ? ext : "img";
}

const rowKey = (mode, file) => `${mode}|${file}`;
const rowOf = (file, mode = state.mode) => state.rows.get(rowKey(mode, file));

function setLabels(rows) {
  state.rows = new Map(rows.map((r) => [rowKey(r.mode, r.file), r]));
  refreshDone();
}
// the files that are done in the current mode (deferred ones count)
function refreshDone() {
  state.done = new Set();
  for (const r of state.rows.values()) if (r.mode === state.mode) state.done.add(r.file);
}

function showLabelInfo(name) {
  const r = name ? rowOf(name) : null;
  const el = $("labelInfo");
  el.classList.toggle("hidden", !r);
  if (!r) return;
  const what = r.label ? `Labeled: ${r.label}${r.class_id !== "" ? ` (class_id ${r.class_id})` : ""}` : `Scored: ${r.score}`;
  el.textContent = `${what} · ${r.timestamp}` + (r.note ? ` · ${r.note}` : "") +
    (state.mode === "score" ? " — type a score to overwrite" : " — press a label key to overwrite");
}

function nextUndone(from) {
  for (let i = from; i < state.files.length; i++) if (!state.done.has(state.files[i])) return i;
  return state.files.length;
}

// --------------------------------------------------------------------------- showing files
async function goTo(pos) {
  state.pos = pos;
  state.gen++;
  worker.postMessage({ op: "cancel", gen: state.gen });
  updateProgress();
  renderFileList();
  if (pos >= state.files.length) return showEnd();
  const name = state.files[pos];
  const type = fileType(name);
  state.current = { name, type, key: null, summary: null, view: null };
  state.disp = null;
  state.scoreBuf = "";
  updateScoreBox();
  clearLine();
  hideTip();
  resetMeta("Loading…");
  $("fileName").textContent = name;
  $("fileInfo").textContent = `${pos + 1} of ${state.files.length} · ${type === "img" ? "image" : "." + type} · ${localTimestamp(new Date(state.mtimes.get(name)))}`;
  showLabelInfo(name);
  try {
    const res = await fetchView(name, state.settings[type], 0);
    if (state.current.name !== name) return;
    Object.assign(state.current, res);
    draw();
  } catch (e) {
    if (state.current.name !== name || e.message === "cancelled") return;
    showMessage("Could not read this file", e.message);
    resetMeta("No metadata: this file could not be read.");
  }
  prefetch(pos);
}

async function fetchView(name, settings, prio) {
  const file = await (await state.dir.getFileHandle(name)).getFile();
  const key = fileKey(name, file);
  const ck = key + "|" + settingsKey(settings);
  if (viewCache.has(ck)) return { key, ...viewCache.get(ck) };
  if (!state.workerReady) setStatus("Loading Python…");
  const bytes = await file.arrayBuffer();
  const res = await call("show", { key, name, bytes, settings }, prio, [bytes]);
  cacheView(ck, res);
  return { key, ...res };
}

// parse the file in Python again (its cache holds fewer files than viewCache does)
async function reloadInPython(cur) {
  const file = await (await state.dir.getFileHandle(cur.name)).getFile();
  const bytes = await file.arrayBuffer();
  await call("show", { key: cur.key, name: cur.name, bytes, settings: state.settings[cur.type] }, 0, [bytes]);
}

function cacheView(ck, res) {
  viewCache.delete(ck);
  viewCache.set(ck, res);
  while (viewCache.size > VIEW_CACHE_SIZE) viewCache.delete(viewCache.keys().next().value);
}

function prefetch(pos) {
  let i = pos;
  for (let n = 0; n < PREFETCH; n++) {
    i = nextUndone(i + 1);
    if (i >= state.files.length) break;
    const name = state.files[i];
    fetchView(name, state.settings[fileType(name)], 1).catch(() => {});
  }
}

// re-render the current file after a settings change
let renderSeq = 0;
async function rerender() {
  const cur = state.current;
  if (!cur || !cur.key) return;
  const seq = ++renderSeq;
  const s = state.settings[cur.type];
  saveSettings();
  const ck = cur.key + "|" + settingsKey(s);
  let res = viewCache.get(ck);
  if (!res) {
    try {
      res = await call("render", { key: cur.key, settings: s }, 0);
      if (res.notLoaded) res = await fetchView(cur.name, s, 0);
      else cacheView(ck, res);
    } catch (e) {
      return toast(e.message);
    }
  }
  if (state.current !== cur || seq !== renderSeq) return;   // a newer file or setting won
  Object.assign(cur, res);
  draw();
  prefetch(state.pos);   // neighbours with the new settings
}

// --------------------------------------------------------------------------- drawing
function draw() {
  const { type, summary, view } = state.current;
  const s = state.settings[type];
  $("message").classList.add("hidden");

  const sel = $("channel");
  sel.replaceChildren(...summary.channels.map((c) => new Option(c, c)));
  sel.value = view.channel;
  $("fallback").classList.toggle("hidden", !view.fallback);

  const dirs = summary.directions[view.channel] || [];
  const isImage = view.kind === "image" || view.kind === "cube";
  $("dirRow").classList.toggle("hidden", dirs.length < 2 || view.kind === "spectrum");
  for (const b of $("direction").children) b.classList.toggle("on", b.dataset.dir === view.direction);
  $("flattenRow").classList.toggle("hidden", !isImage);
  $("flatten").value = s.flatten;
  $("cmapRow").classList.toggle("hidden", !isImage);
  $("cmap").value = s.cmap;
  $("sliceRow").classList.toggle("hidden", view.kind !== "cube");
  if (view.kind === "cube") {
    $("slice").max = view.n - 1;
    $("slice").value = view.index;
    $("sliceText").textContent = `${view.index + 1}/${view.n} · ${fmt(view.slice_value)} ${unitOf(view.slice_label)}`;
  }

  const showImg = view.kind !== "spectrum";
  $("canvasWrap").classList.toggle("hidden", !showImg);
  $("scale").classList.toggle("hidden", !showImg);
  $("cbar").classList.toggle("hidden", view.kind === "rgb");
  if (showImg) drawImage(view, s);
  else { state.disp = null; $("axes").replaceChildren(); }
  const spec = view.kind === "spectrum" ? view : view.spectrum;
  $("plot").classList.toggle("hidden", !spec);
  if (spec) drawPlot(spec, view.kind === "cube" ? view.slice_value : null, showImg);
  else if (plot) { plot.destroy(); plot = null; }
  syncPanels(view, s);
}

function lut(name) {
  const hex = COLORMAPS[name] || COLORMAPS.gray;
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) out.set([0, 2, 4].map((o) => parseInt(hex.substr(i * 6 + o, 2), 16)), i * 3);
  return out;
}
const lutCache = {};

function drawImage(view, s) {
  const disp = buildDisplay(view, s);   // transpose / origin applied here; ticks are drawn separately
  state.disp = disp;
  const c = $("img");
  c.width = disp.cols;
  c.height = disp.rows;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(disp.cols, disp.rows);
  const range = view.kind === "rgb" ? null : contrastRange(view, s);
  paintImage(img, disp, range, (lutCache[s.cmap] ||= lut(s.cmap)));
  if (range) drawColorbar(s.cmap, view, range);
  ctx.putImageData(img, 0, 0);

  // fit the stage, keeping the physical aspect ratio; axis ticks need room around the image
  const ticks = s.scale === "ticks";
  const pad = (state.pad = { l: ticks ? 70 : 0, t: ticks ? 10 : 0, r: ticks ? 16 : 0, b: ticks ? 50 : 0 });
  $("imgBox").style.padding = `${pad.t}px ${pad.r}px ${pad.b}px ${pad.l}px`;
  const stage = $("stage");
  const plotH = view.spectrum ? 220 : 0;
  const maxW = stage.clientWidth - GUTTER - (view.kind === "rgb" ? 0 : 112) - pad.l - pad.r;
  const maxH = stage.clientHeight - 64 - plotH - pad.t - pad.b;
  const aspect = view.extent[0] / view.extent[1] || view.w / view.h;
  let w = Math.max(50, maxW), h = w / aspect;
  if (h > maxH) { h = Math.max(50, maxH); w = h * aspect; }
  c.style.width = Math.floor(w) + "px";
  c.style.height = Math.floor(h) + "px";
  $("cbarCanvas").style.height = Math.floor(h * 0.8) + "px";
  $("scale").textContent = view.extent_units === "px"
    ? `${disp.cols} × ${disp.rows} px shown`
    : `${fmt(view.extent[0])} × ${fmt(view.extent[1])} ${view.extent_units} · ${disp.cols} × ${disp.rows} px`;
  drawOverlay();
}

function drawColorbar(cmap, view, range) {
  const c = $("cbarCanvas");
  const ctx = c.getContext("2d");
  const table = (lutCache[cmap] ||= lut(cmap));
  for (let i = 0; i < 256; i++) {
    ctx.fillStyle = `rgb(${table[i * 3]},${table[i * 3 + 1]},${table[i * 3 + 2]})`;
    ctx.fillRect(0, 255 - i, 1, 1);
  }
  const u = view.units ? " " + view.units : "";
  $("cbarMax").textContent = fmt(range.vmax) + u;
  $("cbarMin").textContent = fmt(range.vmin) + u;
}

function drawPlot(spec, marker, below) {
  const el = $("plot");
  if (plot) { plot.destroy(); plot = null; }
  const colors = ["#2f6fde", "#e0812f", "#1f8a4c", "#c43d3d"];
  const css = getComputedStyle(document.documentElement);
  const text = css.getPropertyValue("--muted").trim(), grid = css.getPropertyValue("--border").trim();
  const axis = (label) => ({ label, stroke: text, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } });
  const data = [spec.x, ...spec.series.map((s) => s.y.map((v) => (v === undefined ? null : v)))];
  const width = Math.min($("stage").clientWidth - GUTTER, 1100);
  const height = below ? 200 : Math.max(240, Math.min($("stage").clientHeight - 60, 640));
  plot = new uPlot({
    width, height,
    scales: { x: { time: false } },
    axes: [axis(spec.x_label), axis(spec.y_label)],
    series: [{ label: spec.x_label }, ...spec.series.map((s, i) => ({
      label: s.label, stroke: colors[i % colors.length], width: 1.5, spanGaps: false,
    }))],
    cursor: { drag: { x: true, y: true } },
    hooks: marker === null ? {} : { draw: [(u) => {
      const x = u.valToPos(marker, "x", true);
      const ctx = u.ctx;
      ctx.save();
      ctx.strokeStyle = "#c43d3d";
      ctx.lineWidth = 1.5 * devicePixelRatio;
      ctx.beginPath(); ctx.moveTo(x, u.bbox.top); ctx.lineTo(x, u.bbox.top + u.bbox.height); ctx.stroke();
      ctx.restore();
    }] },
  }, data, el);
}

function fmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const a = Math.abs(v);
  return a !== 0 && (a < 0.01 || a >= 1e5) ? v.toExponential(2) : +v.toPrecision(4) + "";
}
function unitOf(label) { const m = /\(([^)]*)\)$/.exec(label || ""); return m ? m[1] : ""; }

function showMessage(title, detail) {
  for (const id of ["canvasWrap", "scale", "plot"]) $(id).classList.add("hidden");
  const m = $("message");
  m.replaceChildren();
  const h = document.createElement("h2");
  h.textContent = title;
  const p = document.createElement("p");
  p.textContent = detail;
  m.append(h, p);
  m.classList.remove("hidden");
}

function showEnd() {
  state.current = null;
  const left = state.files.filter((f) => !state.done.has(f)).length;
  $("fileName").textContent = "";
  $("fileInfo").textContent = "";
  showLabelInfo(null);
  resetMeta("No file selected.");
  showMessage(left ? "End of folder" : "All files labeled",
              left ? `${left} skipped file(s) still unlabeled.` : `Log: ${LOG_DIR}/${LOG_FILE}`);
  if (left) {
    const b = document.createElement("button");
    b.className = "primary";
    b.textContent = "Go to first unlabeled";
    b.onclick = () => goTo(nextUndone(0));
    $("message").append(b);
  }
}

function updateProgress() {
  const n = state.files.filter((f) => state.done.has(f)).length;
  $("progress").textContent = state.files.length ? `${n} / ${state.files.length} ${state.mode === "score" ? "scored" : "labeled"}` : "";
  for (const id of ["prevBtn", "nextBtn"]) $(id).classList.toggle("hidden", !state.files.length);
  $("prevBtn").disabled = state.pos <= 0;
  $("nextBtn").disabled = state.pos >= state.files.length - 1;
}

// --------------------------------------------------------------------------- file list
// searchable list of all files, oldest first, with their entries in the current mode; click a row to jump there
function setupFileList() {
  const filter = $("fileFilter");
  const options = state.mode === "score"
    ? [["__all", "All files"], ["__unlabeled", "Unscored"], ["__scored", "Scored"], [DEFER_LABEL, DEFER_LABEL], [UNSURE_LABEL, UNSURE_LABEL]]
    : [["__all", "All files"], ["__unlabeled", "Unlabeled"],
       ...[...new Set([...(state.mode === "binary" ? binaryClasses() : multiclassList(state.config).map((c) => c.name)), DEFER_LABEL, UNSURE_LABEL])].map((c) => [c, c])];
  filter.replaceChildren(...options.map(([v, t]) => new Option(t, v)));
  $("fileSearch").value = "";
  renderFileList();
}

function labelClass(label) {
  if (noOutput(label)) return "defer";
  if (state.mode !== "binary") return "";
  const classes = binaryClasses();
  if (classes.length >= 2 && label === classes[0]) return "good";
  if (classes.length >= 2 && label === classes[1]) return "bad";
  return "";
}

// what the list shows for a row of the current mode
function chipText(r) { return !r ? "" : r.label || r.score; }

function renderFileList() {
  if (!$("fileListBox").open) return;
  const q = $("fileSearch").value.trim().toLowerCase(), f = $("fileFilter").value;
  const frag = document.createDocumentFragment();
  let shown = 0, curRow = null;
  state.files.forEach((name, i) => {
    const r = rowOf(name), text = chipText(r);
    const scored = r && !noOutput(r.label) && r.score !== "";
    if (f === "__unlabeled" ? r : f === "__scored" ? !scored : f !== "__all" && !(r && r.label === f)) return;
    if (q && !name.toLowerCase().includes(q) && !text.toLowerCase().includes(q)) return;
    const row = document.createElement("div");
    row.className = "frow" + (i === state.pos ? " cur" : "");
    row.dataset.i = i;
    const others = MODES.filter((m) => m !== state.mode).map((m) => {
      const o = rowOf(name, m);
      return o ? `${MODE_LABELS[m]}: ${chipText(o)}` : null;
    }).filter(Boolean);
    row.title = [localTimestamp(new Date(state.mtimes.get(name))), ...others].join("\n");
    const n = document.createElement("span"), nm = document.createElement("span"), chip = document.createElement("span");
    n.className = "n"; n.textContent = i + 1;
    nm.className = "nm"; nm.textContent = name;
    chip.className = "chip " + labelClass(r ? r.label : ""); chip.textContent = text || "–";
    row.append(n, nm, chip);
    frag.append(row);
    if (i === state.pos) curRow = row;
    shown++;
  });
  $("fileList").replaceChildren(frag);
  $("fileCount").textContent = `${shown} / ${state.files.length}`;
  if (curRow) curRow.scrollIntoView({ block: "nearest" });
}

// --------------------------------------------------------------------------- labeling
// binary / multiclass / score entries all go through record(); the buttons and keys live in modes.js
function label(cls) { record({ label: cls }); }

function record(fields) {
  const cur = state.current;
  if (!cur) return;
  const view = cur.view;
  let channel = "";
  if (view) {
    const dirs = cur.summary.directions[view.channel] || [];
    channel = view.kind === "spectrum" || dirs.length < 2 ? view.channel : `${view.channel} ${view.direction}`;
  }
  const mode = state.mode;
  // none / offset (line offset) / line (line-wise fit) / plane: only images have one (spectra and colour photos do not)
  const flatten = view && (view.kind === "image" || view.kind === "cube") ? state.settings[cur.type].flatten : "";
  const row = {
    file: cur.name, type: cur.type, channel, flatten, mode, label: "", class_id: "", score: "", tags: "",
    note: $("note").value.trim(), array: "", sample_mean: "", sample_std: "", timestamp: localTimestamp(), ...fields,
  };
  // what is on screen now (dataset.js): its mean / std always go into the log; the training array is saved when
  // that is switched on, except for deferred / unsure files, which have no output
  const snap = view ? arraySnapshot(cur, view, state.settings[cur.type]) : null;
  if (snap) {
    snap.save = arraysOn() && !noOutput(row.label);
    if (snap.save) row.array = snap.array;
  }
  $("note").value = "";   // the note went with this entry
  const previous = rowOf(cur.name);   // set when reviewing an already labeled file
  state.rows.set(rowKey(mode, cur.name), row);
  state.done.add(cur.name);
  const copy = mode === "score" ? "none" : state.copyAs;   // scores are never copied
  // snapshot now: the canvas is redrawn for the next file before the IO runs
  const snapshot = copy === "image" && row.label ? snapshotView() : null;
  queueIO(async () => {
    if (snap) {   // first, so that the statistics are part of the row that is written
      try {
        const meta = await saveArray(snap);
        row.sample_mean = numText(meta.sample_mean);
        row.sample_std = numText(meta.sample_std);
      } catch (e) {
        row.array = "";
        toast(`Could not ${snap.save ? "save the training array" : "get the statistics"} for ${cur.name}: ${e.message}`, 8000);
      }
    }
    if (previous) {   // overwrite: drop the old row (of this mode) and its copies, so Undo reverts the latest action
      const rows = (await readLog()).filter((r) => !(r.file === cur.name && r.mode === mode));
      await writeLog([...rows, row]);
      await removeCopies(previous);
      await removeArrayIfUnused(previous.array);
    } else await appendLog(row);
    if (row.label && snapshot) await writeJpeg(snapshot, jpegName(cur.name), row.label);
    else if (row.label && copy === "original") await copyInto(cur.name, row.label);
  }, `Could not save the entry for ${cur.name}`);
  const shown = row.label || row.score, was = previous ? chipText(previous) : "";
  toast(`${cur.name} → ${shown}${previous ? " (was " + was + ")" : ""}`);
  goTo(previous ? Math.min(state.pos + 1, state.files.length) : nextUndone(state.pos + 1));
}

// remove the copy (original or jpeg) made for a logged row, if any
async function removeCopies(row) {
  if (!row.label) return;   // score rows have no copies
  try {
    const logDir = await state.dir.getDirectoryHandle(LOG_DIR);
    const dst = await logDir.getDirectoryHandle(folderName(row.label));
    for (const n of [row.file, jpegName(row.file)]) await dst.removeEntry(n).catch(() => {});
  } catch {}
}

function skip() {
  if (state.pos < state.files.length) goTo(nextUndone(state.pos + 1));
}

function undo() {
  queueIO(async () => {
    const rows = await readLog();
    const at = rows.map((r) => r.mode).lastIndexOf(state.mode);   // the last entry made in this mode
    if (at < 0) return toast("Nothing to undo in this mode");
    const [last] = rows.splice(at, 1);
    await writeLog(rows);
    await removeCopies(last);
    setLabels(rows);
    await removeArrayIfUnused(last.array);
    toast(`Undid ${last.file} → ${chipText(last)}`);
    const i = state.files.indexOf(last.file);
    goTo(i >= 0 ? i : nextUndone(0));
  }, "Undo failed");
}

function queueIO(fn, errorText) {
  // a failed write is most often the log (classification_log.csv) being open in Excel or another program
  ioChain = ioChain.then(fn).catch((e) => toast(
    `${errorText}: ${String(e.message).replace(/\.$/, "")}. This could be due to the CSV being open in another program (such as Excel). ` +
    "Close it and continue.", 12000));
  return ioChain;
}

function localTimestamp(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function folderName(label) { return label === DEFER_LABEL ? "deferred" : label === UNSURE_LABEL ? "unsure" : safeName(label); }
function safeName(s) { return String(s).replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").trim() || "_"; }

async function logHandle(create) {
  const logDir = await state.dir.getDirectoryHandle(LOG_DIR, { create });
  return logDir.getFileHandle(LOG_FILE, { create });
}

async function readLog() {
  try {
    const text = await (await (await logHandle(false)).getFile()).text();
    const [header, ...rows] = parseCsv(text);
    if (!header) return [];
    return rows.filter((r) => r.length > 1 || r[0])
      .map((r) => {
        const row = Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""]));
        for (const c of LOG_COLUMNS) row[c] ??= "";
        row.mode ||= "binary";   // logs from before the mode column existed
        return row;
      });
  } catch (e) {
    if (e.name === "NotFoundError") return [];
    throw e;
  }
}

async function appendLog(row) {
  const fh = await logHandle(true);
  const file = await fh.getFile();
  if (file.size && (await file.slice(0, 500).text()).split(/\r?\n/)[0] !== LOG_COLUMNS.join(",")) {
    await writeLog([...(await readLog()), row]);   // an older log: rewrite it with the current columns
    return;
  }
  const w = await fh.createWritable({ keepExistingData: true });
  await w.seek(file.size);
  let text = file.size ? "" : csvLine(LOG_COLUMNS);
  if (file.size && !(await file.slice(-1).text()).endsWith("\n")) text += "\n";
  await w.write(text + csvLine(LOG_COLUMNS.map((c) => row[c] ?? "")));
  await w.close();
}

async function writeLog(rows) {
  const fh = await logHandle(true);
  const w = await fh.createWritable();
  await w.write([LOG_COLUMNS, ...rows.map((r) => LOG_COLUMNS.map((c) => r[c] ?? ""))].map(csvLine).join(""));
  await w.close();
}

async function copyInto(name, cls) {
  const src = await (await state.dir.getFileHandle(name)).getFile();
  const logDir = await state.dir.getDirectoryHandle(LOG_DIR, { create: true });
  const dst = await logDir.getDirectoryHandle(folderName(cls), { create: true });
  const w = await (await dst.getFileHandle(name, { create: true })).createWritable();
  await w.write(src);
  await w.close();
}

function jpegName(name) { return name.replace(/\.[^.]+$/, "") + ".jpg"; }

// copy of what is on screen: the image canvas, or the spectrum plot
function snapshotView() {
  const v = state.current && state.current.view;
  if (!v) return null;
  const src = v.kind === "spectrum" ? plot && plot.ctx.canvas : $("img");
  if (!src || !src.width) return null;
  const c = document.createElement("canvas");
  c.width = src.width;
  c.height = src.height;
  const ctx = c.getContext("2d");
  ctx.fillStyle = v.kind === "spectrum" ? getComputedStyle(document.body).backgroundColor : "#808080";
  ctx.fillRect(0, 0, c.width, c.height);   // jpeg has no transparency (no-data pixels, plot background)
  ctx.drawImage(src, 0, 0);
  const cur = state.current;
  if (v.kind !== "spectrum" && state.settings[cur.type].scale === "bar") drawScaleBarOnCanvas(ctx, v, c.width, c.height);
  return c;
}

async function writeJpeg(canvas, name, cls) {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  const logDir = await state.dir.getDirectoryHandle(LOG_DIR, { create: true });
  const dst = await logDir.getDirectoryHandle(folderName(cls), { create: true });
  const w = await (await dst.getFileHandle(name, { create: true })).createWritable();
  await w.write(blob);
  await w.close();
}

function loadPref(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}

function setCopyAs(value) {
  state.copyAs = value;
  $("copyAs").value = value;
  try { localStorage.setItem("spm-labeler-copy-as", value); } catch {}
}

function csvLine(values) {
  return values.map((v) => {
    v = String(v);
    return /[",\r\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
  }).join(",") + "\n";
}

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

// --------------------------------------------------------------------------- controls
function changeSetting(patch) {
  if (!state.current) return;
  Object.assign(state.settings[state.current.type], patch);
  rerender();
}

function cycleChannel(step) {
  const cur = state.current;
  if (!cur || !cur.summary) return;
  const chans = cur.summary.channels;
  const i = chans.indexOf(cur.view.channel);
  changeSetting({ channel: chans[(i + step + chans.length) % chans.length] });
}

function cycle(list, value, step = 1) { return list[(list.indexOf(value) + step + list.length) % list.length]; }

function onKey(e) {
  const tag = e.target.tagName;
  if (e.target.id === "note") {   // Enter / Esc leave the box; the note stays there until the next label
    if (e.key === "Enter" || e.key === "Escape") { e.preventDefault(); e.target.blur(); }
    return;
  }
  // while typing in a text field (including the settings editor) the keys belong to the field, not to the shortcuts
  if (tag === "TEXTAREA" || e.target.isContentEditable) return;
  if (tag === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
  if (!state.dir || e.ctrlKey && e.key !== "z" || e.metaKey || e.altKey) return;
  const view = state.current && state.current.view;
  const s = state.current && state.settings[state.current.type];
  let handled = true;
  if (modeKey(e)) { /* binary / multiclass class keys, typing a score */ }
  else if (e.key === "ArrowUp") cycleChannel(-1);
  else if (e.key === "ArrowDown") cycleChannel(1);
  else if (e.key === "Escape") { if (state.line) clearLine(); }
  else if (e.key === " ") e.shiftKey ? goTo(nextUndone(0)) : skip();
  else if (e.key === "[") goTo(Math.max(0, state.pos - 1));
  else if (e.key === "]") { if (state.pos < state.files.length - 1) goTo(state.pos + 1); }
  else if (e.key === "Home") goTo(0);
  else if (e.key === "End") goTo(state.files.length - 1);
  else if (e.key.toLowerCase() === "z") undo();
  else if (e.key.toLowerCase() === "d") label(DEFER_LABEL);
  else if (e.key.toLowerCase() === "u") label(UNSURE_LABEL);
  else if (!s) handled = false;
  else if (e.key.toLowerCase() === "b" && view) changeSetting({ direction: view.direction === "forward" ? "backward" : "forward" });
  else if (e.key.toLowerCase() === "f") changeSetting({ flatten: cycle(FLATTENS, s.flatten, e.shiftKey ? -1 : 1) });
  else if (e.key.toLowerCase() === "c") changeSetting({ cmap: cycle(Object.keys(COLORMAPS), s.cmap, e.shiftKey ? -1 : 1) });
  else if ((e.key === "PageUp" || e.key === "PageDown") && view && view.kind === "cube")
    changeSetting({ index: Math.min(view.n - 1, Math.max(0, (s.index ?? view.index) + (e.key === "PageDown" ? 1 : -1))) });
  else if (e.key.toLowerCase() === "n" && state.config.notes) $("note").focus();
  else handled = false;
  if (handled) {
    e.preventDefault();
    if (tag === "SELECT" || tag === "BUTTON" || tag === "INPUT") e.target.blur();
  }
}

let toastTimer;
function toast(text, ms = 1800) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}

function setStatus(text) { $("status").textContent = text; }

// remember the last folder (IndexedDB is the only place a directory handle can be stored)
function idb() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open("spm-labeler", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("handles");
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function rememberHandle(h) {
  try { (await idb()).transaction("handles", "readwrite").objectStore("handles").put(h, "last"); } catch {}
}
async function lastHandle() {
  try {
    const db = await idb();
    return await new Promise((res) => {
      const r = db.transaction("handles").objectStore("handles").get("last");
      r.onsuccess = () => res(r.result);
      r.onerror = () => res(null);
    });
  } catch { return null; }
}

// --------------------------------------------------------------------------- startup
(async function init() {
  $("cmap").replaceChildren(...Object.keys(COLORMAPS).map((c) => new Option(c, c)));
  if (!window.showDirectoryPicker) {
    $("browserWarn").classList.remove("hidden");
    $("openBtn").disabled = true;
    return;
  }
  $("ver").textContent = "v" + APP_VERSION.split(" · ")[0];
  $("ver").title = APP_VERSION;
  $("openBtn").onclick = () => openFolder();
  $("headerOpenBtn").onclick = (e) => { e.currentTarget.blur(); openFolder(); };   // switch folders without a reload
  const last = await lastHandle();
  if (last) {
    $("reopenBtn").textContent = "Reopen " + last.name;
    $("reopenBtn").classList.remove("hidden");
    $("reopenBtn").onclick = () => openFolder(last);
  }
  document.addEventListener("keydown", onKey);
  initTools();
  initModes();
  initDataset();
  $("channel").onchange = (e) => { changeSetting({ channel: e.target.value }); e.target.blur(); };
  $("flatten").onchange = (e) => { changeSetting({ flatten: e.target.value }); e.target.blur(); };
  $("cmap").onchange = (e) => { changeSetting({ cmap: e.target.value }); e.target.blur(); };
  $("fileListBox").ontoggle = renderFileList;
  $("fileSearch").oninput = renderFileList;
  $("fileSearch").onkeydown = (e) => {
    if (e.key === "Enter") {   // jump to the first match
      const first = $("fileList").firstElementChild;
      if (first) { goTo(+first.dataset.i); e.target.blur(); }
    } else if (e.key === "Escape") e.target.blur();
  };
  $("fileFilter").onchange = (e) => { renderFileList(); e.target.blur(); };
  $("fileList").onclick = (e) => {
    const row = e.target.closest(".frow");
    if (row) goTo(+row.dataset.i);
  };
  const showTheme = () => {
    const t = document.documentElement.dataset.theme || "";
    for (const b of $("theme").children) b.classList.toggle("on", b.dataset.theme === t);
  };
  showTheme();
  for (const b of $("theme").children) b.onclick = () => {
    const t = b.dataset.theme;
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    try { t ? localStorage.setItem("spm-labeler-theme", t) : localStorage.removeItem("spm-labeler-theme"); } catch {}
    showTheme();
    b.blur();
    if (state.current && state.current.view) draw();   // the plot reads theme colours when drawn
  };
  $("copyAs").value = state.copyAs;
  $("copyAs").onchange = (e) => { setCopyAs(e.target.value); e.target.blur(); };
  $("slice").oninput = (e) => changeSetting({ index: +e.target.value });
  $("slice").onchange = (e) => e.target.blur();
  for (const b of $("direction").children) b.onclick = () => { changeSetting({ direction: b.dataset.dir }); b.blur(); };
  $("prevBtn").onclick = (e) => { goTo(Math.max(0, state.pos - 1)); e.target.blur(); };
  $("nextBtn").onclick = (e) => { goTo(state.pos + 1); e.target.blur(); };
  $("skipBtn").onclick = (e) => { skip(); e.target.blur(); };
  $("undoBtn").onclick = (e) => { undo(); e.target.blur(); };
})();
