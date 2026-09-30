// Runs Python off the main thread, for both pages: the private scorer (module `webapp`,
// ~100 window simulations that would freeze the page) and the public board (module
// `boardapp`, the ranking). The page names the module; the worker imports it and calls
// its boot().
//
// Pyodide is pinned: 0.29.5 ships Python 3.13 and pandas 2.3.3, the repo's own versions.
// The 314.x line moves to pandas 3.0, which the harness has never been tested on.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.5/full/pyodide.js");

// The page, not the worker, fetches the harness and prices and posts them here. On a
// private Space the worker's own requests reached HF without the viewer's login and got
// a 401 "Invalid username or password." page, which then failed as a JSON SyntaxError.
let files;
const filesArrived = new Promise((resolve) => { files = resolve; });

async function boot() {
  postMessage({ status: "Loading Python…" });
  const py = await loadPyodide();
  postMessage({ status: "Loading pandas and numpy…" });
  // tzdata is not optional here. Without it, every tz_localize (twice a round, in
  // calendar.at) retries `import tzdata`, and a failed import is not cached, so each
  // call walks sys.path again. That made scoring ~30x slower than native.
  await py.loadPackage(["numpy", "pandas", "tzdata"]);
  postMessage({ status: "Loading the harness and prices…" });
  const { contents, meta, module } = await filesArrived;
  if (!/^[a-z_]+$/.test(module)) throw new Error(`bad module name ${module}`);
  for (const [path, bytes] of Object.entries(contents)) {
    py.FS.mkdirTree("/app/" + path.split("/").slice(0, -1).join("/"));
    py.FS.writeFile("/app/" + path, new Uint8Array(bytes));
  }
  // boot() does any one-time work (the scorer parses its prices once, here).
  py.runPython(`import sys; sys.path.insert(0, '/app'); import ${module} as app; app.boot()`);
  postMessage({ ready: true, meta });
  return py;
}

const ready = boot().catch((err) => {
  postMessage({ fatal: String(err) });
  throw err;
});

onmessage = async (event) => {
  if (event.data.init) return files(event.data.init);
  const py = await ready;
  if (event.data.board) {
    try {
      py.globals.set("_board_texts", py.toPy(event.data.board));
      postMessage({ board: py.runPython("app.board(list(_board_texts))") });
    } catch (err) {
      postMessage({ boardError: String(err) });
    }
    return;
  }
  const { name, text, start, end, strict, sizing } = event.data;
  try {
    // Keep the uploaded file's own name: it becomes the strategy name when the file
    // doesn't state one, and it is the name the rejection message cites.
    const safe = name.replace(/[^A-Za-z0-9._-]/g, "_") || "decisions.json";
    py.FS.mkdirTree("/upload");
    py.FS.writeFile("/upload/" + safe, text);
    const score = py.globals.get("app").score;
    const out = score("/upload/" + safe, start, end, strict, sizing);
    postMessage({ result: out });
  } catch (err) {
    postMessage({ error: String(err) });
  }
};
