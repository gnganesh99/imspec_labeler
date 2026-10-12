// Runs Pyodide + py/reader.py off the main thread.
// Messages in:  {id, op: "show" | "render" | "meta" | "export" | "transform" | "stats" | "cancel", prio, gen, ...args}
// Messages out: {id, ok, result | error} and {status: "..."} while starting up.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const PYODIDE_PACKAGES = ["numpy", "pillow", "pyyaml", "packaging", "toolz", "cloudpickle"];
importScripts(PYODIDE_URL + "pyodide.js");

let py, reader, dataset;
const queue = [];   // pending jobs; prio 0 (current file) runs before prio 1 (prefetch)
let running = false;

const ready = (async () => {
  self.postMessage({ status: "Loading Python…" });
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  self.postMessage({ status: "Loading packages…" });
  await py.loadPackage(PYODIDE_PACKAGES, { messageCallback: () => {} });
  const wheels = await (await fetch("wheels/manifest.json", { cache: "no-cache" })).json();
  await py.loadPackage(wheels.map((w) => new URL("wheels/" + w, self.location).href),
                       { messageCallback: () => {} });
  for (const f of ["compat.py", "reader.py", "dataset.py"]) {
    const src = await (await fetch("py/" + f, { cache: "no-cache" })).text();
    py.FS.writeFile("/home/pyodide/" + f, src);
  }
  self.postMessage({ status: "Importing SciFiReaders…" });
  py.runPython(`
import sys
sys.path.insert(0, "/home/pyodide")
import compat
compat.install()
import reader, dataset, SciFiReaders
`);
  reader = py.pyimport("reader");
  dataset = py.pyimport("dataset");
  self.postMessage({ status: "ready" });
})().catch((e) => self.postMessage({ status: "error", error: String(e) }));

self.onmessage = (ev) => {
  const job = ev.data;
  if (job.op === "cancel") {   // drop prefetches made for an older position
    for (let i = queue.length - 1; i >= 0; i--) {
      if (queue[i].prio > 0 && queue[i].gen < job.gen) {
        self.postMessage({ id: queue[i].id, ok: false, error: "cancelled" });
        queue.splice(i, 1);
      }
    }
    return;
  }
  queue.push(job);
  pump();
};

async function pump() {
  if (running) return;
  running = true;
  await ready;
  while (queue.length) {
    queue.sort((a, b) => a.prio - b.prio);
    const job = queue.shift();
    try {
      const result = run(job);
      // large arrays are handed over, not copied: the view, an exported .npy, a transformed sample
      const big = result && [result.view && result.view.data, result.npy, result.data].filter((b) => b && b.buffer);
      const transfer = big ? big.map((b) => b.buffer) : [];
      self.postMessage({ id: job.id, ok: true, result }, transfer);
    } catch (e) {
      self.postMessage({ id: job.id, ok: false, error: pyError(e) });
    }
    await new Promise((r) => setTimeout(r, 0));   // let newer messages arrive
  }
  running = false;
}

function run(job) {
  if (!reader) throw new Error("Python failed to start");
  if (job.op === "transform") {   // build step: one saved sample -> its block of the stacked input array
    return toJs(dataset.transform(new Uint8Array(job.bytes), job.kind, JSON.stringify(job.params)));
  }
  if (job.op === "stats") {       // mean / std of an array saved before these were recorded
    return toJs(dataset.stats(new Uint8Array(job.bytes), job.kind));
  }
  if (job.op === "show") {
    // bytes are only written to the in-memory FS when the file is not parsed yet
    if (!reader.is_loaded(job.key)) {
      const path = "/tmp/" + job.name.replace(/[\/\\]/g, "_");
      py.FS.writeFile(path, new Uint8Array(job.bytes));
      try { reader.load(path, job.key); } finally { py.FS.unlink(path); }
    }
  } else if (!reader.is_loaded(job.key)) {
    return { notLoaded: true };
  }
  if (job.op === "meta") {
    return toJs(reader.metadata.callKwargs(job.key, { channel: job.channel, direction: job.direction }));
  }
  const s = job.settings || {};
  // JS null becomes pyodide's jsnull, not None, so unset values are left out
  const kwargs = { direction: s.direction || "forward", flatten: s.flatten || "none" };
  if (s.channel != null) kwargs.channel = s.channel;
  if (s.index != null) kwargs.index = s.index;
  if (job.op === "export") {   // the training array (or only its statistics) of what is on screen
    return toJs(reader.export.callKwargs(job.key, { ...kwargs, with_array: job.withArray !== false }));
  }
  const summary = toJs(reader.load("", job.key));
  const view = toJs(reader.render.callKwargs(job.key, kwargs));
  return { summary, view };
}

function toJs(proxy) {
  const out = proxy.toJs({ dict_converter: Object.fromEntries, create_pyproxies: false });
  proxy.destroy();
  return out;
}

function pyError(e) {
  const msg = String(e && e.message || e);
  // keep the last line of a Python traceback ("ValueError: ...")
  const lines = msg.trim().split("\n");
  return lines[lines.length - 1] || msg;
}
