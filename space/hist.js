// Small histograms of a metric's per-window values, shared by the board and the scorer.
// The build inlines this file into both pages (see tools/build_holdout_space.py), so
// neither page fetches a script that a private Space's auth might refuse.
//
// The bins come from Python (icaif/leaderboard.py), shared down a column so shapes are
// comparable across rows. The y scale is each histogram's own: a board with cash on it
// would otherwise flatten every other row, since cash puts all its windows in one bin.

const HIST = {
  fmt: (metric, v) => metric === "sharpe_ratio" ? v.toFixed(2) : (v * 100).toFixed(1) + "%",

  // counts: n bins; edges: n+1; median, zero: where to draw the marker lines (or null).
  svg({ counts, edges, metric, median = null, zero = null, muted = false, w = 132, h = 40 }) {
    const n = counts.length, lo = edges[0], hi = edges[n];
    const x = (v) => ((v - lo) / (hi - lo)) * w;
    const top = Math.max(1, ...counts);
    const slot = w / n, bw = Math.max(1, slot - 2);          // 2px surface gap between bars
    const fill = muted ? "var(--hist-muted)" : "var(--hist)";
    let out = `<svg class="hist" viewBox="0 0 ${w} ${h + 1}" width="${w}" height="${h + 1}" role="img"` +
      ` aria-label="${metric} across windows">`;
    if (zero !== null && zero > lo && zero < hi)
      out += `<line x1="${x(zero)}" x2="${x(zero)}" y1="0" y2="${h}" class="hist-zero"/>`;
    for (let i = 0; i < n; i++) {
      const bh = counts[i] ? Math.max(2, (counts[i] / top) * (h - 2)) : 0;
      const bx = i * slot + (slot - bw) / 2, by = h - bh, r = Math.min(2, bw / 2, bh);
      // Rounded at the data end, square at the baseline.
      if (bh) out += `<path fill="${fill}" d="M${bx},${h}V${by + r}Q${bx},${by} ${bx + r},${by}` +
        `H${bx + bw - r}Q${bx + bw},${by} ${bx + bw},${by + r}V${h}Z"/>`;
      // A full-height, invisible hit target: easier to hover than a 2px sliver.
      out += `<rect x="${i * slot}" y="0" width="${slot}" height="${h}" fill="transparent" class="hist-hit"` +
        ` data-tip="${HIST.fmt(metric, edges[i])} to ${HIST.fmt(metric, edges[i + 1])}: ${counts[i]} window${counts[i] === 1 ? "" : "s"}"/>`;
    }
    out += `<line x1="0" x2="${w}" y1="${h + 0.5}" y2="${h + 0.5}" class="hist-base"/>`;
    if (median !== null) out += `<line x1="${x(median)}" x2="${x(median)}" y1="2" y2="${h}" class="hist-median"/>`;
    return out + "</svg>";
  },

  // Edges for a single set of values (the scorer's one file), same bin count as the board.
  edgesFor(values, n = 16) {
    let lo = Math.min(...values), hi = Math.max(...values);
    if (hi === lo) { lo -= 0.5e-3; hi += 0.5e-3; }
    return Array.from({ length: n + 1 }, (_, i) => i === n ? hi : lo + (i * (hi - lo)) / n);
  },

  counts(values, edges) {
    const c = new Array(edges.length - 1).fill(0);
    for (const v of values) { let i = 0; while (i < c.length - 1 && v >= edges[i + 1]) i++; c[i]++; }
    return c;
  },

  median(values) {
    const s = [...values].sort((a, b) => a - b), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  },

  // One tooltip for the whole page, filled with textContent (the tip text is data).
  installTooltips() {
    const tip = document.createElement("div");
    tip.className = "hist-tip"; tip.hidden = true; document.body.appendChild(tip);
    document.addEventListener("pointermove", (e) => {
      const t = e.target.closest && e.target.closest(".hist-hit");
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.dataset.tip; tip.hidden = false;
      const r = tip.getBoundingClientRect();
      tip.style.left = Math.min(window.innerWidth - r.width - 8, e.clientX + 12) + "px";
      tip.style.top = (e.clientY - r.height - 10) + "px";
    });
  },
};
