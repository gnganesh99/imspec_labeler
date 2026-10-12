// Training data, all inside labeled/training_data/. While labeling, every labeled file gets one .npy in
// training_data/arrays/ (full resolution, as viewed, NaN kept) plus a small .json describing it.
// "Build dataset" stacks them into training_data/, per mode and kind of data:
//   <mode>_<kind>_input.npy   (n, ...)  float32      kind: image | rgb | spectrum
//   <mode>_<kind>_output.npy  (n,)      class_id (binary / multiclass) or score
//   <mode>_<kind>_aux.npy     (n, k)    scan size / pixel size or bias range, and each sample's own mean / std
//   <mode>_<kind>_normalize.json        how the arrays were normalized (needed again at inference)
//   <mode>_<kind>_meta.csv, <mode>_<kind>_info.json
// Uses globals from app.js / modes.js (state, call, readLog, toast, MODES, ...) only inside functions.
"use strict";

const TRAIN_DIR = "training_data";   // labeled/training_data: the built files
const ARRAY_DIR = "arrays";          // labeled/training_data/arrays: one array per labeled file
const DATASET_FILE = "dataset.json";
const DEFAULT_DATASET = {
  image: { size: null, method: "interpolate" },        // size null: every image must already have the same size
  spectrum: { points: null, bias_range: "union" },     // points null: as many as the longest spectrum
  normalize: "zscore",   // inputs: each sample by its own mean / std; aux and scores: over the whole dataset. Or "none".
};
// aux columns. log: taken before the z-score, for positive values that span orders of magnitude
const AUX_COLUMNS = {
  image: [["width", true], ["height", true], ["pixel_x", true], ["pixel_y", true], ["sample_mean", false], ["sample_std", true]],
  rgb: [["width", true], ["height", true], ["pixel_x", true], ["pixel_y", true], ["sample_mean", false], ["sample_std", true]],
  spectrum: [["bias_min", false], ["bias_max", false], ["n_points", false], ["sample_mean", false], ["sample_std", true]],
};
let datasetConfig = structuredClone(DEFAULT_DATASET);

function arraysOn() {
  try { return localStorage.getItem("spm-labeler-arrays") !== "off"; } catch { return true; }   // on by default
}

// --------------------------------------------------------------------------- one array per labeled file
// kind: what is on screen ("image" | "cube" | "rgb" | "spectrum")
function arrayNameFor(file, kind, channel, direction, flatten, index) {
  const part = (x) => safeName(String(x)).replace(/\s+/g, "_");
  if (kind === "spectrum" || kind === "rgb") return `${file}__${part(channel)}.npy`;   // all directions / no flatten
  return `${file}__${part(channel)}__${direction}__${flatten}${kind === "cube" ? "__slice" + index : ""}.npy`;
}

// taken at the moment of labeling: the view changes before the file is written
function arraySnapshot(cur, view, s) {
  const index = view.kind === "cube" ? view.index : null;
  return {
    name: cur.name, key: cur.key, type: cur.type,
    kind: view.kind === "cube" ? "image" : view.kind,
    array: arrayNameFor(cur.name, view.kind, view.channel, view.direction, s.flatten, index),
    save: true,   // false: only the sample's mean / std are wanted (for the log), no file
    settings: { channel: view.channel, direction: view.direction, flatten: s.flatten, index },
  };
}

async function exportArray(snap) {
  const ask = () => call("export", { key: snap.key, settings: snap.settings, withArray: snap.save !== false }, 0);
  let res = await ask();
  if (res.notLoaded) {   // Python dropped the file from its cache: parse it again
    await reloadInPython(snap);
    res = await ask();
  }
  if (!res.npy && snap.save !== false) throw new Error("no array returned");
  return res;
}

async function trainDir(create) {
  const dir = await state.dir.getDirectoryHandle(LOG_DIR, { create });
  return dir.getDirectoryHandle(TRAIN_DIR, { create });
}

async function arrayDir(create) {
  return (await trainDir(create)).getDirectoryHandle(ARRAY_DIR, { create });
}

// arrays saved by the first version of this feature lived in labeled/arrays/: move them to the new place
async function migrateOldArrays() {
  try {
    const labeled = await state.dir.getDirectoryHandle(LOG_DIR);
    const old = await labeled.getDirectoryHandle(ARRAY_DIR);
    const dir = await arrayDir(true);
    for await (const [name, h] of old.entries()) {
      if (h.kind !== "file") continue;
      await writeFile(dir, name, await h.getFile());
      await old.removeEntry(name);
    }
    await labeled.removeEntry(ARRAY_DIR).catch(() => {});   // only goes if it is empty now
  } catch {}   // no old folder: nothing to do
}

async function writeFile(dir, name, data) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.write(data);
  await w.close();
}

// returns the description of the sample (with sample_mean / sample_std); writes the files unless snap.save is false
async function saveArray(snap) {
  const res = await exportArray(snap);
  if (snap.save === false) return res.meta;
  const dir = await arrayDir(true);
  await writeFile(dir, snap.array, res.npy);
  const meta = { file: snap.name, array: snap.array, ...res.meta, saved: localTimestamp() };
  await writeFile(dir, snap.array.replace(/\.npy$/, ".json"), JSON.stringify(meta, null, 1) + "\n");
  return meta;
}

// after an undo / overwrite: delete the array unless another log row (any mode) still uses it
async function removeArrayIfUnused(name) {
  if (!name) return;
  for (const r of state.rows.values()) if (r.array === name) return;
  try {
    const dir = await arrayDir(false);
    await dir.removeEntry(name).catch(() => {});
    await dir.removeEntry(name.replace(/\.npy$/, ".json")).catch(() => {});
  } catch {}
}

// --------------------------------------------------------------------------- .npy writing
function npyHeader(descr, shape) {
  const dict = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const pad = (64 - ((10 + dict.length + 1) % 64)) % 64;   // 10 = magic + version + header length
  const text = dict + " ".repeat(pad) + "\n";
  const out = new Uint8Array(10 + text.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, text.length & 255, text.length >> 8]);
  for (let i = 0; i < text.length; i++) out[10 + i] = text.charCodeAt(i);
  return out;
}

function npyFile(descr, shape, typed) {
  return new Blob([npyHeader(descr, shape), typed]);
}

// --------------------------------------------------------------------------- build settings (dataset.json)
function validateDataset(c) {
  if (!c || typeof c !== "object" || Array.isArray(c)) return "The settings must be a JSON object { ... }.";
  const im = c.image, sp = c.spectrum;
  if (im !== undefined) {
    if (!im || typeof im !== "object") return 'image must look like {"size": [256, 256], "method": "interpolate"}.';
    const ok = (v) => Number.isInteger(v) && v >= 1 && v <= 8192;
    if (im.size !== undefined && im.size !== null && !(Array.isArray(im.size) && im.size.length === 2 && im.size.every(ok)))
      return "image.size must be null (all images already the same size) or [height, width] in pixels.";
    if (im.method !== undefined && !["interpolate", "pad_crop"].includes(im.method)) return 'image.method must be "interpolate" or "pad_crop".';
  }
  if (sp !== undefined) {
    if (!sp || typeof sp !== "object") return 'spectrum must look like {"points": 256, "bias_range": "union"}.';
    if (sp.points !== undefined && sp.points !== null && !(Number.isInteger(sp.points) && sp.points >= 2 && sp.points <= 1e6))
      return "spectrum.points must be null (as many as the longest spectrum) or a whole number of at least 2.";
    const b = sp.bias_range;
    const pair = Array.isArray(b) && b.length === 2 && b.every(Number.isFinite) && b[0] < b[1];
    if (b !== undefined && !["union", "overlap", "strict"].includes(b) && !pair)
      return 'spectrum.bias_range must be "union", "overlap", "strict" or [min, max].';
  }
  if (c.normalize !== undefined && !["none", "zscore"].includes(c.normalize)) return 'normalize must be "zscore" or "none".';
  return null;
}

function mergeDataset(user) {
  user = user && typeof user === "object" ? user : {};
  return { image: { ...DEFAULT_DATASET.image, ...(user.image || {}) },
           spectrum: { ...DEFAULT_DATASET.spectrum, ...(user.spectrum || {}) },
           normalize: user.normalize ?? DEFAULT_DATASET.normalize };
}

function checkDatasetText() {
  let parsed = null, error = null;
  try { parsed = JSON.parse($("dsText").value); error = validateDataset(parsed); } catch (e) { error = "Not valid JSON: " + e.message; }
  $("dsMsg").textContent = error || "";
  $("dsMsg").style.color = "var(--bad)";
  $("dsBuild").disabled = !!error || !state.dir;
  return error ? null : mergeDataset(parsed);
}

// called when a folder is opened
async function loadDatasetConfig() {
  await migrateOldArrays();
  const user = await readJson(DATASET_FILE);
  datasetConfig = validateDataset(user) ? structuredClone(DEFAULT_DATASET) : mergeDataset(user);
  $("dsText").value = JSON.stringify(datasetConfig, null, 2);
  $("dsText").disabled = false;
  $("dsStatus").textContent = "";
  checkDatasetText();
}

// --------------------------------------------------------------------------- build
const dsLog = (line, replaceLast) => {
  const el = $("dsStatus");
  const lines = el.textContent ? el.textContent.split("\n") : [];
  if (replaceLast && lines.length) lines.pop();
  lines.push(line);
  el.textContent = lines.join("\n");
};

async function buildDataset() {
  const cfg = checkDatasetText();
  if (!cfg) return;
  $("dsBuild").disabled = true;
  $("dsStatus").textContent = "";
  try {
    datasetConfig = cfg;
    await writeJson(DATASET_FILE, cfg);
    await ioChain;                                   // labels still being written
    let rows = await readLog();
    rows = await backfillArrays(rows);
    const outDir = await trainDir(true);
    const arrays = await arrayDir(true);
    const order = new Map(state.files.map((f, i) => [f, i]));
    let built = 0, logChanged = false;
    for (const mode of MODES) {
      const items = [];
      for (const row of rows) {
        if (row.mode !== mode || row.label === DEFER_LABEL || !row.array) continue;
        try {
          const meta = JSON.parse(await (await (await arrays.getFileHandle(row.array.replace(/\.npy$/, ".json"))).getFile()).text());
          await arrays.getFileHandle(row.array);
          items.push({ row, meta });
        } catch { dsLog(`${mode}: the array of ${row.file} is missing, left out`); }
      }
      if (await fillStats(items, arrays)) logChanged = true;
      items.sort((a, b) => (order.get(a.row.file) ?? 1e9) - (order.get(b.row.file) ?? 1e9));
      for (const kind of ["image", "rgb", "spectrum"]) {
        const group = items.filter((it) => it.meta.kind === kind);
        if (!group.length) continue;
        try {
          dsLog(`${mode} · ${kind}: …`);
          const shape = await buildGroup(mode, kind, group, cfg, outDir, arrays);
          dsLog(`${mode} · ${kind}: ${group.length} samples → ${mode}_${kind}_input.npy (${shape.join(", ")})`, true);
          built++;
        } catch (e) {
          dsLog(`${mode} · ${kind}: not built. ${e.message}`, true);
        }
      }
    }
    if (logChanged) { await writeLog(rows); setLabels(rows); }
    dsLog(built ? `Done ${localTimestamp()}. Files are in ${LOG_DIR}/${TRAIN_DIR}/.` : "Nothing to build yet: label some files first.");
  } catch (e) {
    dsLog("Build failed: " + e.message);
  }
  $("dsBuild").disabled = false;
}

// rows labeled before arrays were saved (or with saving off): make their arrays now from the original files
async function backfillArrays(rows) {
  const todo = rows.filter((r) => !r.array && r.label !== DEFER_LABEL && state.files.includes(r.file));
  if (!todo.length) return rows;
  let done = 0, failed = 0;
  dsLog(`Making arrays for ${todo.length} earlier label(s)…`);
  for (const r of todo) {
    try {
      const file = await (await state.dir.getFileHandle(r.file)).getFile();
      const m = /^(.*) (forward|backward)$/.exec(r.channel);
      const snap = {
        name: r.file, key: `${r.file}|${file.size}|${file.lastModified}`, type: fileType(r.file), array: "",
        settings: { channel: m ? m[1] : r.channel, direction: m ? m[2] : "forward", flatten: r.flatten || "none", index: null },
      };
      await reloadInPython(snap);
      const res = await exportArray(snap);
      const kind = res.meta.slice_index !== undefined ? "cube" : res.kind;
      snap.settings.index = res.meta.slice_index ?? null;
      snap.array = arrayNameFor(r.file, kind, snap.settings.channel, res.meta.direction || snap.settings.direction, snap.settings.flatten, snap.settings.index);
      const dir = await arrayDir(true);
      await writeFile(dir, snap.array, res.npy);
      await writeFile(dir, snap.array.replace(/\.npy$/, ".json"),
                      JSON.stringify({ file: r.file, array: snap.array, ...res.meta, saved: localTimestamp() }, null, 1) + "\n");
      r.array = snap.array;
      done++;
    } catch { failed++; }
    dsLog(`Making arrays for earlier labels: ${done + failed} / ${todo.length}`, true);
  }
  if (done) { await writeLog(rows); setLabels(rows); }
  dsLog(`Arrays made for ${done} earlier label(s)` + (failed ? `, ${failed} could not be made (file unreadable or channel missing)` : ""), true);
  return rows;
}

const numText = (v) => (Number.isFinite(v) ? String(v) : "");

// samples saved before the statistics were recorded get them now; the log is given them as well
async function fillStats(items, arrays) {
  let changed = false;
  for (const it of items) {
    if (!Number.isFinite(it.meta.sample_std)) {
      const bytes = await (await (await arrays.getFileHandle(it.row.array)).getFile()).arrayBuffer();
      const st = await call("stats", { bytes, kind: it.meta.kind }, 0, [bytes]);
      it.meta.sample_mean = st.sample_mean ?? null;
      it.meta.sample_std = st.sample_std ?? null;
      await writeFile(arrays, it.row.array.replace(/\.npy$/, ".json"), JSON.stringify(it.meta, null, 1) + "\n");
    }
    const mean = numText(it.meta.sample_mean), std = numText(it.meta.sample_std);
    if (it.row.sample_mean !== mean || it.row.sample_std !== std) {
      it.row.sample_mean = mean;
      it.row.sample_std = std;
      changed = true;
    }
  }
  return changed;
}

// z-score over the whole dataset (non-NaN values only). A column without spread gives 0 everywhere.
function datasetZscore(values, log) {
  let floor = null, v = values;
  if (log) {   // values <= 0 (e.g. the std of a constant image) are raised to the smallest positive one first
    const positive = values.filter((x) => Number.isFinite(x) && x > 0);
    floor = positive.length ? Math.min(...positive) : 1;
    v = values.map((x) => (Number.isFinite(x) ? Math.log(Math.max(x, floor)) : NaN));
  }
  const fin = v.filter(Number.isFinite);
  const mean = fin.length ? fin.reduce((a, b) => a + b, 0) / fin.length : 0;
  const std = fin.length ? Math.sqrt(fin.reduce((a, b) => a + (b - mean) ** 2, 0) / fin.length) : 0;
  const scale = fin.reduce((a, b) => Math.max(a, Math.abs(b)), 1e-300);
  const constant = !(std > 1e-9 * scale);
  return { z: v.map((x) => (Number.isFinite(x) ? (constant ? 0 : (x - mean) / std) : NaN)),
           stats: { ...(log ? { log: true, floor } : { log: false }), mean, std: constant ? 0 : std, constant } };
}

async function buildGroup(mode, kind, items, cfg, outDir, arrays) {
  const n = items.length;
  let params, sample;
  if (kind === "spectrum") {
    const nRows = Math.max(...items.map((it) => it.meta.shape[0]));
    const range = cfg.spectrum.bias_range;
    if (range === "strict") {
      const lengths = [...new Set(items.map((it) => it.meta.shape[1]))];
      if (lengths.length > 1) throw new Error(`the spectra have different lengths (${lengths.join(", ")} points); use "bias_range": "union".`);
      params = { grid: null, rows: nRows };
      sample = [nRows, lengths[0]];
    } else {
      const mins = items.map((it) => it.meta.bias_min), maxs = items.map((it) => it.meta.bias_max);
      const [lo, hi] = Array.isArray(range) ? range
        : range === "overlap" ? [Math.max(...mins), Math.min(...maxs)] : [Math.min(...mins), Math.max(...maxs)];
      if (!(hi > lo)) throw new Error("the spectra have no bias range in common.");
      const points = cfg.spectrum.points ?? Math.max(...items.map((it) => it.meta.n_points));
      params = { grid: { min: lo, max: hi, points }, rows: nRows };
      sample = [nRows, points];
    }
  } else {
    let size = cfg.image.size;
    if (!size) {
      const sizes = [...new Set(items.map((it) => `${it.meta.shape[0]} x ${it.meta.shape[1]}`))];
      if (sizes.length > 1) throw new Error(`the images have different sizes (${sizes.join(", ")}); set "image": {"size": [height, width]}.`);
      size = items[0].meta.shape.slice(0, 2);
    }
    params = { size, method: cfg.image.method };
    sample = kind === "rgb" ? [...size, 3] : [...size];
  }
  const zs = cfg.normalize === "zscore";
  const prefix = `${mode}_${kind}`;
  const norm = { normalize: cfg.normalize, built: localTimestamp(), samples: n,
                 note: "All statistics use non-NaN values only. NaN stays NaN in the arrays." };

  // inputs: streamed sample by sample, so the memory use stays small however many files there are
  const w = await (await outDir.getFileHandle(`${prefix}_input.npy`, { create: true })).createWritable();
  try {
    await w.write(npyHeader("<f4", [n, ...sample]));
    for (let i = 0; i < n; i++) {
      const bytes = await (await (await arrays.getFileHandle(items[i].row.array)).getFile()).arrayBuffer();
      const own = { ...params, normalize: cfg.normalize, mean: items[i].meta.sample_mean ?? null, std: items[i].meta.sample_std ?? null };
      const res = await call("transform", { bytes, kind, params: own }, 0, [bytes]);
      if (res.shape.join() !== sample.join()) throw new Error(`${items[i].row.file} gave shape (${res.shape.join(", ")}), expected (${sample.join(", ")}).`);
      await w.write(res.data);
      if (i % 10 === 9) dsLog(`${mode} · ${kind}: ${i + 1} / ${n}`, true);
    }
    await w.close();
  } catch (e) {
    await w.abort().catch(() => {});
    throw e;
  }

  // outputs
  const info = { mode, kind, samples: n, built: localTimestamp(), order: "files oldest first (as listed in the app)",
                 input: { file: `${prefix}_input.npy`, dtype: "float32", shape: [n, ...sample], nan: "kept where there is no data" },
                 transform: params, normalize: cfg.normalize };
  norm.input = zs
    ? { scope: "per sample", formula: "(x - sample_mean) / sample_std",
        statistics: "each sample's own mean and std, from its saved full-resolution array" +
          (kind === "spectrum" ? " (signal rows only; the bias row is not normalized)" : "") +
          "; listed per sample in " + `${prefix}_meta.csv`, constant_sample: "std 0 gives 0 everywhere" }
    : { scope: "none", formula: "values as saved (the units of the file)" };
  let out;
  if (mode === "score") {
    const raw = items.map((it) => Number(it.row.score));
    const d = datasetZscore(raw, false);
    out = { descr: "<f8", data: Float64Array.from(zs ? d.z : raw) };
    info.output = { file: `${prefix}_output.npy`, dtype: "float64", meaning: zs ? "score, z-scored over the dataset" : "score" };
    norm.output = zs ? { scope: "dataset", formula: "(score - mean) / std", mean: d.stats.mean, std: d.stats.std, constant: d.stats.constant,
                         back: "score = z * std + mean" }
                     : { scope: "none", mean: d.stats.mean, std: d.stats.std };
  } else {
    norm.output = { scope: "none", formula: "class numbers are never normalized" };
    const ids = items.map((it) => it.row.class_id);
    const numeric = ids.every((v) => v !== "" && Number.isFinite(Number(v)));
    if (numeric && ids.every((v) => Number.isInteger(Number(v)))) {
      out = { descr: "<i4", data: Int32Array.from(ids, Number) };
      info.output = { file: `${prefix}_output.npy`, dtype: "int32", meaning: "class_id" };
    } else if (numeric) {
      out = { descr: "<f8", data: Float64Array.from(ids, Number) };
      info.output = { file: `${prefix}_output.npy`, dtype: "float64", meaning: "class_id" };
    } else {   // ids that are text or missing (older logs): number the label names instead
      const names = [...new Set(items.map((it) => it.row.label))].sort();
      out = { descr: "<i4", data: Int32Array.from(items, (it) => names.indexOf(it.row.label)) };
      info.output = { file: `${prefix}_output.npy`, dtype: "int32", meaning: "index into classes (class_id was not a number for every row)" };
    }
    info.output.classes = {};   // label name -> the number stored in the output array
    items.forEach((it, i) => { info.output.classes[it.row.label] = out.data[i]; });
  }
  await writeFile(outDir, `${prefix}_output.npy`, npyFile(out.descr, [n], out.data));

  // aux: scan size / bias range and the sample's own mean / std, as measured (before any resizing).
  // With "zscore" each column is z-scored over the dataset (after a log where marked); the raw values are in meta.csv.
  const cols = AUX_COLUMNS[kind].map(([name]) => name);
  const aux = new Float64Array(n * cols.length);
  norm.aux = { scope: zs ? "dataset" : "none",
               formula: zs ? "z = (v - mean) / std, with v = ln(max(x, floor)) for log columns, else v = x; constant columns are 0"
                           : "raw values (the statistics below are what a z-score would use)",
               columns: {} };
  AUX_COLUMNS[kind].forEach(([name, log], j) => {
    const raw = items.map((it) => (Number.isFinite(it.meta[name]) ? it.meta[name] : NaN));
    const d = datasetZscore(raw, log);
    (zs ? d.z : raw).forEach((v, i) => { aux[i * cols.length + j] = v; });
    norm.aux.columns[name] = d.stats;
  });
  await writeFile(outDir, `${prefix}_aux.npy`, npyFile("<f8", [n, cols.length], aux));
  await writeFile(outDir, `${prefix}_normalize.json`, JSON.stringify(norm, null, 2) + "\n");
  info.aux = { file: `${prefix}_aux.npy`, dtype: "float64", columns: cols, values: zs ? "z-scores (see normalize.json)" : "raw",
               units: kind === "spectrum" ? items[0].meta.bias_units : items[0].meta.extent_units };
  if (kind === "spectrum") info.input.rows = "row 0 = bias, then the signal (forward, backward if recorded)";

  // meta: one line per sample, same order as the arrays
  const head = ["index", "file", "channel", "direction", "flatten", "array", "label", "class_id", "score", "original_shape",
                "units", "extent_units", "width", "height", "pixel_x", "pixel_y", "slice_index", "slice_value",
                "bias_min", "bias_max", "n_points", "sample_mean", "sample_std", "labeled"];
  const lines = items.map((it, i) => [i, it.row.file, it.meta.channel, it.meta.direction ?? "", it.meta.flatten ?? "", it.row.array,
    it.row.label, it.row.class_id, it.row.score, (it.meta.shape || []).join("x"), it.meta.units ?? "", it.meta.extent_units ?? it.meta.bias_units ?? "",
    it.meta.width ?? "", it.meta.height ?? "", it.meta.pixel_x ?? "", it.meta.pixel_y ?? "", it.meta.slice_index ?? "", it.meta.slice_value ?? "",
    it.meta.bias_min ?? "", it.meta.bias_max ?? "", it.meta.n_points ?? "", numText(it.meta.sample_mean), numText(it.meta.sample_std),
    it.row.timestamp]);
  await writeFile(outDir, `${prefix}_meta.csv`, [head, ...lines].map(csvLine).join(""));
  await writeFile(outDir, `${prefix}_info.json`, JSON.stringify(info, null, 2) + "\n");
  return [n, ...sample];
}

// --------------------------------------------------------------------------- panel
function initDataset() {
  $("dsSave").checked = arraysOn();
  $("dsSave").onchange = (e) => {
    try { localStorage.setItem("spm-labeler-arrays", e.target.checked ? "on" : "off"); } catch {}
    e.target.blur();
  };
  $("dsText").value = JSON.stringify(DEFAULT_DATASET, null, 2);
  $("dsText").oninput = checkDatasetText;
  $("dsText").onkeydown = (e) => { if (e.key === "Escape") e.target.blur(); };
  $("dsBuild").onclick = (e) => { buildDataset(); e.currentTarget.blur(); };
  checkDatasetText();
}
