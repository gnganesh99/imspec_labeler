// Image display model (transpose / origin), axes + scale bar overlay, pixel readout, line profile,
// metadata panel and the View / Tools drawer. Uses globals from app.js ($, state, fmt, call, draw, ...).
"use strict";

function loadJson(key) { try { return JSON.parse(localStorage.getItem(key) || "{}"); } catch { return {}; } }
function saveJson(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} }

const tools = { readout: true, width: 5, ...loadJson("spm-labeler-tools"), profile: false };
const panels = { view: false, tools: false, data: false, meta: true, profile: false, ...loadJson("spm-labeler-panels") };
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// --------------------------------------------------------------------------- display model
function f32(view) {
  if (!view._f32) {
    const b = view.data, aligned = b.byteOffset % 4 === 0;
    const src = aligned ? b : b.slice();
    view._f32 = new Float32Array(src.buffer, src.byteOffset, src.byteLength / 4);
  }
  return view._f32;
}

// The file data is row-major with row 0 = first row. Display = optional transpose, then the origin:
// "lower" puts row 0 at the bottom, "upper" at the top. Axis ticks are not part of this transform.
function buildDisplay(view, s) {
  const transposed = !!s.transpose;
  const origin = s.origin === "auto" ? view.origin : s.origin;
  const rows = transposed ? view.w : view.h, cols = transposed ? view.h : view.w;
  const rgb = view.kind === "rgb", w = view.w;
  const src = rgb ? view.data : f32(view);
  const vals = rgb ? new Uint8ClampedArray(rows * cols * 4) : new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const rr = origin === "lower" ? rows - 1 - i : i;
    for (let j = 0; j < cols; j++) {
      const from = transposed ? j * w + rr : rr * w + j;
      if (rgb) {
        const a = from * 4, b = (i * cols + j) * 4;
        vals[b] = src[a]; vals[b + 1] = src[a + 1]; vals[b + 2] = src[a + 2]; vals[b + 3] = 255;
      } else vals[i * cols + j] = src[from];
    }
  }
  return { rows, cols, vals, rgb };
}

// Colour range of an image. The default (1 % clipped at each end) is the range the Python side computed; the View
// panel's contrast slider changes the clipped percentage (0 = full range), taken from the sorted pixel values.
const DEFAULT_CLIP = 1;
function contrastRange(view, s) {
  const clip = s.clip ?? DEFAULT_CLIP;
  if (clip === DEFAULT_CLIP) return { vmin: view.vmin, vmax: view.vmax };
  if (!view._sorted) {
    const a = f32(view), f = new Float32Array(a.length);
    let n = 0;
    for (let i = 0; i < a.length; i++) if (a[i] === a[i]) f[n++] = a[i];
    view._sorted = f.subarray(0, n).sort();
  }
  const v = view._sorted;
  if (!v.length) return { vmin: view.vmin, vmax: view.vmax };
  const at = (p) => v[clamp(Math.round(p * (v.length - 1)), 0, v.length - 1)];
  let lo = at(clip / 100), hi = at(1 - clip / 100);
  if (!(hi > lo)) { lo = v[0]; hi = v[v.length - 1]; }
  return { vmin: lo, vmax: hi };
}

function paintImage(imageData, disp, range, table) {
  const px = imageData.data, v = disp.vals;
  if (disp.rgb) { px.set(v); return; }
  const view = range;   // { vmin, vmax }
  const span = (view.vmax - view.vmin) || 1;
  for (let i = 0, j = 0; i < v.length; i++, j += 4) {
    const x = v[i];
    if (x !== x) continue;   // NaN: no data, stays transparent
    const t = (x - view.vmin) / span;
    const k = (t <= 0 ? 0 : t >= 1 ? 255 : Math.round(t * 255)) * 3;
    px[j] = table[k]; px[j + 1] = table[k + 1]; px[j + 2] = table[k + 2]; px[j + 3] = 255;
  }
}

// --------------------------------------------------------------------------- axes / scale bar
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

function niceTicks(max, target = 5) {
  if (!(max > 0)) return { ticks: [0], step: 1 };
  const step = niceStep(max / target), ticks = [];
  for (let v = 0; v <= max * 1.0001; v += step) ticks.push(+v.toPrecision(12));
  return { ticks, step };
}

function scaleBarSpec(view) {
  const nice = niceStep(view.extent[0] * 0.2);
  return { frac: nice / view.extent[0], text: `${+nice.toPrecision(3)} ${view.extent_units}` };
}

function svgEl(name, attrs, text) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

function drawOverlay() {
  const svg = $("axes"), cur = state.current, view = cur && cur.view;
  svg.replaceChildren();
  if (!view || !state.disp || view.kind === "spectrum") return;
  const s = state.settings[cur.type], c = $("img"), pad = state.pad;
  const L = pad.l, T = pad.t, W = c.clientWidth, H = c.clientHeight;
  const [ex, ey] = view.extent, eu = view.extent_units;

  if (s.scale === "ticks") {
    const axis = (max, size, horizontal) => {
      const { ticks, step } = niceTicks(max, clamp(Math.floor(size / (horizontal ? 75 : 50)), 2, 8));   // fewer ticks on a small image
      const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
      for (const v of ticks) {
        if (horizontal) {
          const x = L + (W * v) / max;
          svg.append(svgEl("line", { x1: x, x2: x, y1: T + H, y2: T + H + 4, stroke: "currentColor" }),
                     svgEl("text", { x, y: T + H + 19, "text-anchor": "middle", fill: "currentColor" }, v.toFixed(dec)));
        } else {
          const y = T + H * (1 - v / max);
          svg.append(svgEl("line", { x1: L - 4, x2: L, y1: y, y2: y, stroke: "currentColor" }),
                     svgEl("text", { x: L - 8, y: y + 5, "text-anchor": "end", fill: "currentColor" }, v.toFixed(dec)));
        }
      }
    };
    axis(ex, W, true);
    axis(ey, H, false);
    svg.append(svgEl("text", { x: L + W / 2, y: T + H + 40, "text-anchor": "middle", fill: "currentColor" }, `x (${eu})`),
               svgEl("text", { x: 14, y: T + H / 2, "text-anchor": "middle", fill: "currentColor",
                               transform: `rotate(-90 14 ${T + H / 2})` }, `y (${eu})`));
  } else if (s.scale === "bar") {
    const { frac, text } = scaleBarSpec(view);
    const x1 = L + W * 0.96, x0 = x1 - W * frac, y = T + H * 0.93;
    svg.append(svgEl("rect", { x: x0 - 1, y: y - 3, width: x1 - x0 + 2, height: 6, fill: "#000", opacity: 0.7 }),
               svgEl("rect", { x: x0, y: y - 2, width: x1 - x0, height: 4, fill: "#fff" }),
               svgEl("text", { x: (x0 + x1) / 2, y: y - 9, "text-anchor": "middle", fill: "#fff", stroke: "#000",
                               "stroke-width": 3, "paint-order": "stroke", "font-size": 15, "font-weight": 600 }, text));
  }

  const ln = state.line;
  if (ln) {
    const a = [L + ln.fx0 * W, T + ln.fy0 * H], b = [L + ln.fx1 * W, T + ln.fy1 * H];
    svg.append(svgEl("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: "#000", "stroke-width": 4, opacity: 0.5 }),
               svgEl("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: "#ffd400", "stroke-width": 2 }),
               svgEl("circle", { cx: a[0], cy: a[1], r: 4, fill: "#ffd400", stroke: "#000" }),
               svgEl("circle", { cx: b[0], cy: b[1], r: 4, fill: "#ffd400", stroke: "#000" }));
  }
}

// scale bar on a jpeg copy of the image (same look as the overlay)
function drawScaleBarOnCanvas(ctx, view, W, H) {
  const { frac, text } = scaleBarSpec(view);
  const x1 = W * 0.96, x0 = x1 - W * frac, y = H * 0.93, k = Math.max(1, W / 400);
  ctx.fillStyle = "rgba(0,0,0,.7)";
  ctx.fillRect(x0 - k, y - 3 * k, x1 - x0 + 2 * k, 6 * k);
  ctx.fillStyle = "#fff";
  ctx.fillRect(x0, y - 2 * k, x1 - x0, 4 * k);
  ctx.font = `600 ${15 * k}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.lineWidth = 3 * k;
  ctx.strokeStyle = "#000";
  ctx.strokeText(text, (x0 + x1) / 2, y - 9 * k);
  ctx.fillText(text, (x0 + x1) / 2, y - 9 * k);
}

// --------------------------------------------------------------------------- pixel readout
function fracOf(e) {
  const r = $("img").getBoundingClientRect();
  return { fx: clamp((e.clientX - r.left) / r.width, 0, 1), fy: clamp((e.clientY - r.top) / r.height, 0, 1) };
}

function showTip(e) {
  const tip = $("tip"), view = state.current && state.current.view, d = state.disp;
  if (!tools.readout || !view || !d) return tip.classList.add("hidden");
  const { fx, fy } = fracOf(e);
  const j = Math.min(d.cols - 1, Math.floor(fx * d.cols)), i = Math.min(d.rows - 1, Math.floor(fy * d.rows));
  const eu = view.extent_units ? " " + view.extent_units : "";
  let text = `x ${fmt(fx * view.extent[0])}, y ${fmt((1 - fy) * view.extent[1])}${eu}\n`;
  if (d.rgb) {
    const k = (i * d.cols + j) * 4;
    text += `RGB ${d.vals[k]}, ${d.vals[k + 1]}, ${d.vals[k + 2]}`;
  } else {
    const v = d.vals[i * d.cols + j];
    text += v === v ? `${view.title} ${fmt(v)} ${view.units}` : "no data";
  }
  tip.textContent = text;
  tip.classList.remove("hidden");
  const x = Math.min(e.clientX + 16, innerWidth - tip.offsetWidth - 6);
  const y = Math.min(e.clientY + 16, innerHeight - tip.offsetHeight - 6);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
function hideTip() { $("tip").classList.add("hidden"); }

// --------------------------------------------------------------------------- line profile
function bilinear(a, rows, cols, x, y) {
  if (x < -0.5 || y < -0.5 || x > cols - 0.5 || y > rows - 0.5) return NaN;
  x = clamp(x, 0, cols - 1);
  y = clamp(y, 0, rows - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, cols - 1), y1 = Math.min(y0 + 1, rows - 1);
  const tx = x - x0, ty = y - y0;
  let sum = 0, wsum = 0;
  for (const [xx, yy, w] of [[x0, y0, (1 - tx) * (1 - ty)], [x1, y0, tx * (1 - ty)],
                             [x0, y1, (1 - tx) * ty], [x1, y1, tx * ty]]) {
    const v = a[yy * cols + xx];
    if (v === v && w > 0) { sum += w * v; wsum += w; }
  }
  return wsum ? sum / wsum : NaN;
}

// samples along the line (one per display pixel); each sample is the mean over `width` pixels across it
function computeProfile(d, view, line, width) {
  const { rows, cols, vals } = d, [ex, ey] = view.extent;
  const dx = (line.fx1 - line.fx0) * cols, dy = (line.fy1 - line.fy0) * rows;
  const lenPx = Math.hypot(dx, dy);
  if (lenPx < 2) return null;
  const n = Math.round(lenPx) + 1;
  const length = Math.hypot((line.fx1 - line.fx0) * ex, (line.fy1 - line.fy0) * ey);
  const px = -dy / lenPx, py = dx / lenPx, half = (width - 1) / 2;
  const dist = new Array(n), val = new Array(n);
  for (let k = 0; k < n; k++) {
    const t = k / (n - 1);
    const cx = (line.fx0 + t * (line.fx1 - line.fx0)) * cols, cy = (line.fy0 + t * (line.fy1 - line.fy0)) * rows;
    let sum = 0, cnt = 0;
    for (let o = -half; o <= half + 1e-9; o += 1) {
      const v = bilinear(vals, rows, cols, cx + px * o - 0.5, cy + py * o - 0.5);
      if (v === v) { sum += v; cnt++; }
    }
    dist[k] = t * length;
    val[k] = cnt ? sum / cnt : NaN;
  }
  return { dist, val, length };
}

let profPlot = null, profPlotKey = "";

function updateProfile() {
  const cur = state.current, view = cur && cur.view, d = state.disp, line = state.line;
  const usable = view && d && (view.kind === "image" || view.kind === "cube");
  const prof = usable && line ? computeProfile(d, view, line, tools.width) : null;
  state.profile = prof;
  $("profCsv").disabled = !prof;
  $("profClear").disabled = !line;
  if (!prof) {
    if (profPlot) { profPlot.destroy(); profPlot = null; profPlotKey = ""; }
    $("profStats").textContent = "";
    return;
  }
  const eu = view.extent_units, xl = `distance (${eu})`, yl = `${view.title} (${view.units})`;
  const key = xl + "|" + yl;
  const data = [prof.dist, prof.val.map((v) => (v === v ? v : null))];
  if (profPlot && key === profPlotKey) {
    const width = Math.max(240, $("profPlot").clientWidth);
    if (profPlot.width !== width) profPlot.setSize({ width, height: 170 });   // the drawer width can change
    profPlot.setData(data);
  } else {
    if (profPlot) profPlot.destroy();
    const css = getComputedStyle(document.documentElement);
    const text = css.getPropertyValue("--muted").trim(), grid = css.getPropertyValue("--border").trim();
    const axis = (label) => ({ label, stroke: text, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, size: 52 });
    profPlot = new uPlot({
      width: Math.max(240, $("profPlot").clientWidth), height: 170, legend: { show: false },
      scales: { x: { time: false } }, axes: [axis(xl), axis(yl)],
      series: [{}, { stroke: css.getPropertyValue("--accent").trim(), width: 1.5 }],
    }, data, $("profPlot"));
    profPlotKey = key;
  }
  const finite = prof.val.filter((v) => v === v);
  const lo = Math.min(...finite), hi = Math.max(...finite);
  $("profStats").textContent = `length ${fmt(prof.length)} ${eu} · min ${fmt(lo)}, max ${fmt(hi)}, Δ ${fmt(hi - lo)} ${view.units}` +
    ` · width ${tools.width} px`;
}

function clearLine() {
  state.line = null;
  state.profile = null;
  if (state.disp) drawOverlay(); else $("axes").replaceChildren();
  updateProfile();
}

function exportProfile() {
  const cur = state.current, view = cur && cur.view, p = state.profile, ln = state.line;
  if (!p) return;
  const s = state.settings[cur.type], [ex, ey] = view.extent, eu = view.extent_units;
  const lines = [
    `# file: ${cur.name}`,
    `# channel: ${view.channel}${view.kind !== "image" || view.direction ? " " + view.direction : ""}, flatten: ${s.flatten}`,
    `# line: (${fmt(ln.fx0 * ex)}, ${fmt((1 - ln.fy0) * ey)}) -> (${fmt(ln.fx1 * ex)}, ${fmt((1 - ln.fy1) * ey)}) ${eu}`,
    `# averaging width: ${tools.width} px`,
    `distance (${eu}),${view.title} (${view.units})`,
  ];
  for (let i = 0; i < p.dist.length; i++) lines.push(`${+p.dist[i].toPrecision(7)},${p.val[i] === p.val[i] ? +p.val[i].toPrecision(7) : ""}`);
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${cur.name.replace(/\.[^.]+$/, "")}_${view.channel.replace(/[^\w.-]+/g, "_")}_profile.csv`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --------------------------------------------------------------------------- metadata
// forget the previous file's header right away (called when another file is opened)
function resetMeta(note) {
  state.meta = null;
  state.metaTag = "";
  renderMeta(note);
}

async function refreshMeta() {
  const box = $("metaBox"), cur = state.current;
  if (!box.open) return;
  if (!cur || !cur.key || !cur.view) { state.meta = null; state.metaTag = ""; return renderMeta(); }
  const tag = [cur.key, cur.view.channel, cur.view.direction].join("|");
  if (tag === state.metaTag) return;
  const ask = () => call("meta", { key: cur.key, channel: cur.view.channel, direction: cur.view.direction }, 0);
  try {
    let res = await ask();
    if (res.notLoaded) {   // Python dropped this file from its cache: parse it again, then retry
      await reloadInPython(cur);
      res = await ask();
    }
    if (state.current !== cur) return;
    if (!Array.isArray(res.facts)) throw new Error("unexpected reply from the worker; hard-refresh the page (Ctrl+Shift+R)");
    state.meta = res;
    state.metaTag = tag;
  } catch (e) {
    if (state.current !== cur) return;
    state.meta = { facts: [["Error", e.message]], all: [] };
  }
  renderMeta();
}

// note: placeholder text shown instead of fields (loading / unreadable file)
function renderMeta(note) {
  const m = state.meta || { facts: [], all: [] };
  const cell = (t) => { const e = document.createElement("span"); e.textContent = t; return e; };
  if (typeof note === "string") {
    const n = cell(note);
    n.style.gridColumn = "1 / -1";
    $("metaFacts").replaceChildren(n);
  } else $("metaFacts").replaceChildren(...m.facts.flatMap(([k, v]) => [cell(k), cell(v)]));
  const q = $("metaSearch").value.trim().toLowerCase();
  const rows = m.all.filter(([k, v]) => !q || k.toLowerCase().includes(q) || v.toLowerCase().includes(q)).slice(0, 400);
  $("metaAll").replaceChildren(...rows.map(([k, v]) => {
    const row = document.createElement("div");
    row.append(cell(k), cell(v));
    return row;
  }));
}

// --------------------------------------------------------------------------- panels
// called by draw() after every redraw
function syncPanels(view, s) {
  const clip = s.clip ?? DEFAULT_CLIP;
  $("vClip").value = clip;
  $("clipText").textContent = clip === 0 ? "(full range)" : `(${clip}% clipped at each end)`;
  $("clipRow").classList.toggle("hidden", view.kind !== "image" && view.kind !== "cube");
  $("vTranspose").checked = !!s.transpose;
  $("vOrigin").value = s.origin;
  for (const b of $("vScale").children) b.classList.toggle("on", b.dataset.scale === s.scale);
  const image = view.kind === "image" || view.kind === "cube";
  for (const id of ["vTranspose", "vOrigin"]) $(id).disabled = view.kind === "spectrum";
  $("profMode").disabled = !image;
  $("profMode").classList.toggle("on", tools.profile && image);
  $("imgBox").classList.toggle("drawmode", tools.profile && image);
  $("profNote").textContent = image ? "Turn on Draw line, then drag on the image. Values are in the channel's units."
    : view.kind === "rgb" ? "Profiles need a single-channel image: pick Gray, R, G or B."
    : "Profiles are available for images.";
  updateProfile();
  refreshMeta();
}

function changeView(patch) {
  const cur = state.current;
  if (!cur || !cur.view) return;
  Object.assign(state.settings[cur.type], patch);
  saveSettings();
  draw();
}

function initTools() {
  const setPanels = () => {
    $("viewPanel").classList.toggle("hidden", !panels.view);
    $("toolsPanel").classList.toggle("hidden", !panels.tools);
    $("dataPanel").classList.toggle("hidden", !panels.data);
    $("drawer").classList.toggle("hidden", !panels.view && !panels.tools && !panels.data);
    $("viewBtn").classList.toggle("on", panels.view);
    $("toolsBtn").classList.toggle("on", panels.tools);
    $("dataBtn").classList.toggle("on", panels.data);
    saveJson("spm-labeler-panels", panels);
  };
  setPanels();
  $("viewBtn").onclick = (e) => { panels.view = !panels.view; setPanels(); e.currentTarget.blur(); };
  $("toolsBtn").onclick = (e) => { panels.tools = !panels.tools; setPanels(); e.currentTarget.blur(); };
  $("dataBtn").onclick = (e) => { panels.data = !panels.data; setPanels(); e.currentTarget.blur(); };

  let clipTimer = 0;   // redraw at most every ~30 ms while the slider moves
  $("vClip").oninput = (e) => {
    const value = +e.target.value;
    clearTimeout(clipTimer);
    clipTimer = setTimeout(() => changeView({ clip: value }), 30);
  };
  $("vClip").onchange = (e) => e.target.blur();
  $("vClipReset").onclick = (e) => { changeView({ clip: DEFAULT_CLIP }); e.currentTarget.blur(); };
  $("vTranspose").onchange = (e) => { changeView({ transpose: e.target.checked }); e.target.blur(); };
  $("vOrigin").onchange = (e) => { changeView({ origin: e.target.value }); e.target.blur(); };
  for (const b of $("vScale").children) b.onclick = () => { changeView({ scale: b.dataset.scale }); b.blur(); };

  $("profMode").onclick = (e) => {
    tools.profile = !tools.profile;
    $("profMode").classList.toggle("on", tools.profile);
    $("imgBox").classList.toggle("drawmode", tools.profile);
    e.currentTarget.blur();
  };
  $("profClear").onclick = (e) => { clearLine(); e.currentTarget.blur(); };
  $("profCsv").onclick = (e) => { exportProfile(); e.currentTarget.blur(); };
  $("profWidth").value = tools.width;
  $("profWidth").oninput = (e) => {
    const v = parseInt(e.target.value, 10);
    if (!(v >= 1)) return;
    tools.width = clamp(v, 1, 99);
    saveJson("spm-labeler-tools", { readout: tools.readout, width: tools.width });
    updateProfile();
  };
  $("profWidth").onkeydown = (e) => { if (e.key === "Enter" || e.key === "Escape") e.target.blur(); };
  $("tReadout").checked = tools.readout;
  $("tReadout").onchange = (e) => {
    tools.readout = e.target.checked;
    saveJson("spm-labeler-tools", { readout: tools.readout, width: tools.width });
    if (!tools.readout) hideTip();
    e.target.blur();
  };
  // Line profile is a drop down; collapsing it also leaves drawing mode, so no hidden crosshair mode stays on
  $("profBox").open = !!panels.profile;
  $("profBox").ontoggle = () => {
    panels.profile = $("profBox").open;
    saveJson("spm-labeler-panels", panels);
    if (!panels.profile && tools.profile) {
      tools.profile = false;
      $("profMode").classList.remove("on");
      $("imgBox").classList.remove("drawmode");
    }
    if (panels.profile) updateProfile();   // the plot needs the width of the opened section
  };
  $("metaBox").open = panels.meta !== false;
  $("metaBox").ontoggle = () => { panels.meta = $("metaBox").open; saveJson("spm-labeler-panels", panels); refreshMeta(); };
  $("metaSearch").oninput = renderMeta;

  // the narrow-window panels start below the header, whatever its height
  const header = document.querySelector("header");
  const setHeader = () => document.documentElement.style.setProperty("--hdr", header.offsetHeight + "px");
  new ResizeObserver(setHeader).observe(header);
  setHeader();

  // redraw when the stage changes size (drawer opened / closed, window resized)
  let last = "";
  new ResizeObserver(() => {
    const st = $("stage"), size = st.clientWidth + "x" + st.clientHeight;
    if (size === last) return;
    last = size;
    requestAnimationFrame(() => state.current && state.current.view && draw());
  }).observe($("stage"));

  // pointer handling on the image: readout, and dragging out the profile line
  const img = $("img");
  let dragging = false;
  img.addEventListener("pointerdown", (e) => {
    const view = state.current && state.current.view;
    if (!tools.profile || !state.disp || !view || (view.kind !== "image" && view.kind !== "cube")) return;
    const { fx, fy } = fracOf(e);
    state.line = { fx0: fx, fy0: fy, fx1: fx, fy1: fy };
    dragging = true;
    img.setPointerCapture(e.pointerId);
    hideTip();
    e.preventDefault();
  });
  img.addEventListener("pointermove", (e) => {
    if (!dragging) return showTip(e);
    const { fx, fy } = fracOf(e);
    state.line.fx1 = fx;
    state.line.fy1 = fy;
    drawOverlay();
    updateProfile();
  });
  const end = () => {
    if (!dragging) return;
    dragging = false;
    const ln = state.line;
    const tooShort = state.disp && Math.hypot((ln.fx1 - ln.fx0) * state.disp.cols, (ln.fy1 - ln.fy0) * state.disp.rows) < 2;
    if (tooShort) state.line = null;
    drawOverlay();
    updateProfile();
  };
  img.addEventListener("pointerup", end);
  img.addEventListener("pointercancel", end);
  img.addEventListener("pointerleave", hideTip);
}
