// Runs the Python harness off the main thread: ~150 window simulations would freeze the
// page for the length of the run otherwise.
//
// Pyodide is pinned: 0.29.5 ships Python 3.13 and pandas 2.3.3, the repo's own versions.
// The 314.x line moves to pandas 3.0, which the harness has never been tested on.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.5/full/pyodide.js");

async function boot() {
  postMessage({ status: "Loading Python…" });
  const py = await loadPyodide();
  postMessage({ status: "Loading pandas and numpy…" });
  // tzdata is not optional here. Without it, every tz_localize (twice a round, in
  // calendar.at) retries `import tzdata`, and a failed import is not cached, so each
  // call walks sys.path again. That made scoring ~30x slower than native.
  await py.loadPackage(["numpy", "pandas", "tzdata"]);
  postMessage({ status: "Loading the harness and prices…" });
  const manifest = await (await fetch("manifest.json")).json();
  for (const path of manifest.files) {
    const resp = await fetch(path);
    // A missing file would otherwise be written as an HTML error page and fail later
    // as a confusing SyntaxError or a malformed price table.
    if (!resp.ok) throw new Error(`could not fetch ${path}: HTTP ${resp.status}`);
    const dir = "/app/" + path.split("/").slice(0, -1).join("/");
    py.FS.mkdirTree(dir);
    py.FS.writeFile("/app/" + path, new Uint8Array(await resp.arrayBuffer()));
  }
  // Parse the prices once, here, so the first score doesn't pay for it.
  py.runPython("import sys; sys.path.insert(0, '/app'); import webapp; webapp._MARKET = webapp.load_market()");
  postMessage({ ready: true, meta: manifest.meta });
  return py;
}

const ready = boot().catch((err) => {
  postMessage({ fatal: String(err) });
  throw err;
});

onmessage = async (event) => {
  const py = await ready;
  const { name, text, start, end, strict, sizing } = event.data;
  try {
    // Keep the uploaded file's own name: it becomes the strategy name when the file
    // doesn't state one, and it is the name the rejection message cites.
    const safe = name.replace(/[^A-Za-z0-9._-]/g, "_") || "decisions.json";
    py.FS.mkdirTree("/upload");
    py.FS.writeFile("/upload/" + safe, text);
    const score = py.globals.get("webapp").score;
    const out = score("/upload/" + safe, start, end, strict, sizing);
    postMessage({ result: out });
  } catch (err) {
    postMessage({ error: String(err) });
  }
};
