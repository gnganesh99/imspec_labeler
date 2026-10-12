// Labeling modes (binary / multiclass / score), the labeler.json settings and their editor box.
// Uses globals from app.js (state, record, label, toast, goTo, ...) only inside functions.
"use strict";

const MODES = ["binary", "multiclass", "score"];
const MODE_LABELS = { binary: "Binary", multiclass: "Multiclass", score: "Score" };
const DEFAULT_CONFIG = {
  mode: "binary",
  classes: ["Good", "Bad"],                 // binary mode uses the first two
  multiclass: { start: 1, classes: [] },    // names, or {"name": ..., "id": ...}; up to 10
  score: { min: null, max: null },
  notes: true,
  copy_as: null,                            // none | original | image (null: the app's own setting)
};

function mergeConfig(user) {
  user = user && typeof user === "object" ? user : {};
  const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
  return { ...DEFAULT_CONFIG, ...user,
           multiclass: { ...DEFAULT_CONFIG.multiclass, ...obj(user.multiclass) },
           score: { ...DEFAULT_CONFIG.score, ...obj(user.score) } };
}

// The settings box shows only the options of the current mode:
//   binary:     { "classes": [ {"name": "Good", "id": 1}, {"name": "Bad", "id": 0} ] }
//   multiclass: { "start": 1, "classes": [ ... ] }
//   score:      { "min": null, "max": null }
// (notes, copy_as and the other modes' options stay in labeler.json untouched)
function modeSettingsForEditor(user, mode) {
  const c = mergeConfig(user);
  if (mode === "binary") return { classes: binaryList(c).map(({ name, id }) => ({ name, id })) };
  if (mode === "multiclass") return { start: c.multiclass.start, classes: c.multiclass.classes };
  return { min: c.score.min, max: c.score.max };
}

// merge what was typed in the settings box back into the whole labeler.json
function withModeSettings(user, mode, obj) {
  const next = { ...user, mode };
  if (mode === "binary") next.classes = obj.classes;
  else if (mode === "multiclass") next.multiclass = { start: obj.start ?? 1, classes: obj.classes ?? [] };
  else next.score = { min: obj.min ?? null, max: obj.max ?? null };
  return next;
}

const classNameOf = (x) => (typeof x === "string" ? x : x && typeof x === "object" ? x.name : null);
const classIdOf = (x) => (x && typeof x === "object" && x.id !== undefined && x.id !== null && x.id !== "" ? x.id : null);

// null if fine, else a message for the user
function validateModeSettings(mode, o) {
  if (!o || typeof o !== "object" || Array.isArray(o)) return "The settings must be a JSON object { ... }.";
  const named = (x) => { const n = classNameOf(x); return typeof n === "string" && n.trim() ? n : null; };
  const checkClasses = (list, what) => {
    const seen = new Set();
    for (const x of list) {
      const n = named(x);
      if (!n) return `each ${what} is a name like "Good" or {"name": "Good", "id": 1}.`;
      const id = classIdOf(x);
      if (id !== null && typeof id !== "number" && typeof id !== "string") return `the id of "${n}" must be a number or text.`;
      if (seen.has(n)) return `"${n}" is listed twice.`;
      seen.add(n);
    }
    return null;
  };
  if (mode === "binary") {
    if (!Array.isArray(o.classes) || o.classes.length !== 2) return 'classes must list exactly two classes, e.g. ["Good", "Bad"] or ["Yes", "No"].';
    return checkClasses(o.classes, "class");
  }
  if (mode === "multiclass") {
    if (o.start !== undefined && o.start !== 0 && o.start !== 1) return "start must be 0 or 1.";
    if (!Array.isArray(o.classes)) return 'classes must be a list, e.g. ["one", "two", "three"].';
    if (o.classes.length > 10) return "at most 10 classes (keys 0-9).";
    return checkClasses(o.classes, "class");
  }
  for (const k of ["min", "max"]) {
    if (o[k] !== undefined && o[k] !== null && !Number.isFinite(o[k])) return `${k} must be a number or null.`;
  }
  if (Number.isFinite(o.min) && Number.isFinite(o.max) && o.min > o.max) return "min is larger than max.";
  return null;
}

// --------------------------------------------------------------------------- classes
const BINARY_KEYS = [["→", "1"], ["←", "0"]];   // [arrow, number] per class
// binary: two classes with a name (-> label) and an id (-> class_id); the id defaults to the key number (1, then 0)
function binaryList(cfg) {
  return cfg.classes.slice(0, 2).map((x, i) => ({ name: classNameOf(x), id: classIdOf(x) ?? +BINARY_KEYS[i][1] }));
}
function binaryClasses() { return binaryList(state.config).map((c) => c.name); }
function pickBinary(i) {
  const c = binaryList(state.config)[i];
  if (c) record({ label: c.name, class_id: String(c.id) });
}

// multiclass: with start 1 the keys are 1..9 then 0, with start 0 they are 0..9.
// class_id defaults to the key number; the user can set it per class.
function multiclassList(cfg) {
  const start = cfg.multiclass.start === 0 ? 0 : 1;
  return (cfg.multiclass.classes || []).slice(0, 10).map((c, i) => {
    const key = String((start + i) % 10);
    const hasId = c && typeof c === "object" && c.id !== undefined && c.id !== null && c.id !== "";
    return { name: classNameOf(c), key, id: hasId ? c.id : +key };
  });
}

// --------------------------------------------------------------------------- sidebar
function buildLabelButtons() {
  const wrap = $("labelButtons");
  wrap.replaceChildren();
  const mode = state.mode;
  const add = (text, keys, cls, onclick, title) => {
    const b = document.createElement("button");
    b.innerHTML = `<span></span><span>${keys.map((k) => `<kbd>${k}</kbd>`).join("")}</span>`;
    b.firstChild.textContent = text;
    if (cls) b.className = cls;
    if (title) b.title = title;
    b.onclick = () => { onclick(); b.blur(); };
    wrap.append(b);
  };
  if (mode === "binary") {
    binaryList(state.config).forEach((c, i) => add(c.name, BINARY_KEYS[i], i === 0 ? "good" : "bad", () => pickBinary(i), `class_id ${c.id}`));
  } else if (mode === "multiclass") {
    const list = multiclassList(state.config);
    list.forEach((c) => add(c.name, [c.key], "", () => pickClass(c), `class_id ${c.id}`));
    if (!list.length) {
      const hint = document.createElement("div");
      hint.className = "sub";
      hint.style.gridColumn = "1 / -1";
      hint.innerHTML = 'No classes yet. Add them under <b>Data → Labeling settings</b> (multiclass.classes). ';
      const open = document.createElement("button");
      open.textContent = "Open settings";
      open.onclick = () => { openSettings(); open.blur(); };
      hint.append(open);
      wrap.append(hint);
    }
  }
  add("Defer", ["D"], "", () => label(DEFER_LABEL));

  const keys = [["↑ ↓", "channel"], ["B", "forward / backward"], ["D", "defer (labeled/deferred/)"], ["F", "flatten"], ["C", "colormap"],
                ["PgUp PgDn", "slice (3ds, stacks)"], ["Space", "skip"], ["[ ]", "previous / next file"],
                ["Home End", "first / last file"], ["⇧ Space", "first unlabeled"], ["Z", "undo last entry"]];
  if (mode === "score") keys.unshift(["0-9 . -", "type a score"], ["Enter", "save the score"], ["Backspace", "edit"], ["Esc", "clear what you typed"]);
  if (state.config.notes) keys.push(["N", "note for this file"]);
  $("keys").replaceChildren(...keys.flatMap(([k, d]) => {
    const a = document.createElement("span"); a.innerHTML = k.split(" ").map((x) => `<kbd>${x}</kbd>`).join(" ");
    const b = document.createElement("span"); b.textContent = d;
    return [a, b];
  }));
}

function pickClass(c) { record({ label: c.name, class_id: String(c.id) }); }

// what the mode selector and the surrounding controls show (no file loading here)
function renderMode() {
  $("modeSel").value = state.mode;
  if (state.dir) refreshConfigText();
  $("copyRow").classList.toggle("hidden", state.mode === "score");   // scores are never copied
  buildLabelButtons();
  setupFileList();
  updateScoreBox();
}

function setMode(mode) {
  if (!MODES.includes(mode) || mode === state.mode) return;
  state.mode = state.config.mode = mode;
  state.scoreBuf = "";
  state.userConfig = { ...state.userConfig, mode };
  if (state.dir) {
    writeJson("labeler.json", state.userConfig).catch((e) => toast(`Could not save labeler.json: ${e.message}`));
    refreshConfigText();
  }
  refreshDone();
  renderMode();
  updateProgress();
  if (state.pos >= 0 && state.files.length) goTo(state.pos);   // same file, now shown for this mode
}

// --------------------------------------------------------------------------- score mode
function updateScoreBox() {
  const on = state.mode === "score";
  $("scoreBox").classList.toggle("hidden", !on);
  if (!on) return;
  const el = $("scoreValue");
  el.textContent = state.scoreBuf;
  el.classList.toggle("empty", !state.scoreBuf);
  const { min, max } = state.config.score;
  const r = state.current && rowOf(state.current.name);
  const parts = [];
  if (r && r.label === DEFER_LABEL) parts.push("Deferred.");
  else if (r) parts.push(`Scored ${r.score}.`);
  parts.push(Number.isFinite(min) || Number.isFinite(max)
    ? `Range ${Number.isFinite(min) ? min : "−∞"} to ${Number.isFinite(max) ? max : "∞"}.` : "Any number, negatives and decimals allowed.");
  parts.push("Just start typing; Enter saves.");
  $("scoreHint").textContent = parts.join(" ");
}

function commitScore() {
  const text = state.scoreBuf.trim();
  if (!/^-?(\d+\.?\d*|\.\d+)$/.test(text)) return toast("Type a number first");
  const value = Number(text), { min, max } = state.config.score;
  if ((Number.isFinite(min) && value < min) || (Number.isFinite(max) && value > max)) {
    return toast(`The score must be between ${Number.isFinite(min) ? min : "−∞"} and ${Number.isFinite(max) ? max : "∞"}`);
  }
  state.scoreBuf = "";
  record({ score: String(value) });
}

// mode-specific keys; returns true when the key was used
function modeKey(e) {
  const mode = state.mode, k = e.key;
  if (mode === "binary") {
    if (k === "ArrowRight" || k === "1") { pickBinary(0); return true; }
    if (k === "ArrowLeft" || k === "0") { pickBinary(1); return true; }
    return false;
  }
  if (mode === "multiclass") {
    if (/^[0-9]$/.test(k)) {
      const c = multiclassList(state.config).find((x) => x.key === k);
      if (c) pickClass(c);
      return true;   // numbers without a class do nothing
    }
    return false;
  }
  // score
  if (/^[0-9.\-]$/.test(k)) {
    const next = state.scoreBuf + k;
    if (/^-?\d*\.?\d*$/.test(next)) { state.scoreBuf = next; updateScoreBox(); }
    return true;
  }
  if (k === "Backspace") { state.scoreBuf = state.scoreBuf.slice(0, -1); updateScoreBox(); return true; }
  if (k === "Enter") { commitScore(); return true; }
  if (k === "Escape" && state.scoreBuf) { state.scoreBuf = ""; updateScoreBox(); return true; }
  return false;
}

// --------------------------------------------------------------------------- settings (labeler.json)
async function writeJson(name, obj) {
  const dir = await state.dir.getDirectoryHandle(LOG_DIR, { create: true });   // labeled/
  const fh = await dir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(JSON.stringify(obj, null, 2) + "\n");
  await w.close();
}

const CFG_HINTS = {
  binary: "Two classes, for example Good / Bad, Yes / No or High / Low. The name is written to label and the id to " +
    "class_id (default 1 for the first class, 0 for the second). Keys: 1 or → for the first, 0 or ← for the second.",
  multiclass: 'Up to 10 classes, for example ["one", "two", "three"]. The name is written to label and the id to class_id ' +
    '(default: the number of its key; set your own with {"name": "three", "id": 30}). "start": 1 gives the keys 1-9 then 0, ' +
    '"start": 0 gives 0-9.',
  score: "Scores are typed on the keyboard and written to the score column. Optional limits: min and max (null = no limit).",
};

function refreshConfigText() {
  $("cfgText").value = JSON.stringify(modeSettingsForEditor(state.userConfig, state.mode), null, 2);
  $("cfgMode").textContent = `(${MODE_LABELS[state.mode]} mode)`;
  $("cfgHint").textContent = CFG_HINTS[state.mode];
  checkConfigText();
}

// returns the parsed settings, or null (and shows why)
function checkConfigText() {
  const msg = $("cfgMsg");
  let parsed = null, error = null;
  try { parsed = JSON.parse($("cfgText").value); error = validateModeSettings(state.mode, parsed); } catch (e) { error = "Not valid JSON: " + e.message; }
  msg.textContent = error || "Looks good. Save to apply it to this folder.";
  msg.style.color = error ? "var(--bad)" : "var(--muted)";
  $("cfgSave").disabled = !!error || !state.dir;
  return error ? null : parsed;
}

function applyConfig(user) {
  state.userConfig = user;
  state.config = mergeConfig(user);
  if (!MODES.includes(state.config.mode)) state.config.mode = "binary";
  state.mode = state.config.mode;
  // "copy_as" picks the copy mode; the older "copy_files": false means "none"
  const cfgCopy = user.copy_files === false ? "none" : state.config.copy_as;
  if (COPY_MODES.includes(cfgCopy)) setCopyAs(cfgCopy);
  else if (!COPY_MODES.includes(state.copyAs)) setCopyAs("none");
  refreshDone();
  renderMode();
  refreshConfigText();
}

async function saveConfig() {
  const parsed = checkConfigText();
  if (!parsed) return;
  const next = withModeSettings(state.userConfig, state.mode, parsed);
  try {
    await writeJson("labeler.json", next);
  } catch (e) {
    return toast(`Could not save labeler.json: ${e.message}`);
  }
  applyConfig(next);
  updateProgress();
  if (state.pos >= 0 && state.files.length) goTo(state.pos);
  toast(`Saved the ${MODE_LABELS[state.mode]} settings`);
}

function openSettings() {
  if (!panels.data) $("dataBtn").click();
  $("cfgBox").open = true;
  $("cfgBox").scrollIntoView({ block: "nearest" });
}

function initModes() {
  $("modeSel").onchange = (e) => { setMode(e.target.value); e.target.blur(); };
  $("cfgText").oninput = checkConfigText;
  $("cfgText").onkeydown = (e) => { if (e.key === "Escape") e.target.blur(); };   // Esc leaves the editor; then the shortcuts work again
  refreshConfigText();
  $("cfgSave").onclick = (e) => { saveConfig(); e.currentTarget.blur(); };
  $("cfgReset").onclick = (e) => {
    $("cfgText").value = JSON.stringify(modeSettingsForEditor({}, state.mode), null, 2);
    checkConfigText();
    e.currentTarget.blur();
  };
}
