"""Build the holdout's two static pages, and optionally deploy them to HuggingFace.

    .venv/bin/python tools/build_holdout_space.py            # build output/space/, output/board/
    .venv/bin/python tools/build_holdout_space.py --push     # build, then deploy both
    .venv/bin/python tools/build_holdout_space.py --sync     # copy missing entries dataset -> board

Two pages, because only one of them may be public (icaif/space_hub.py has the table):
- the private scorer (output/space/), which carries the price file and the kit;
- the public board (output/board/), which carries only the ranking code (ranking.py,
  leaderboard.py), the references' results and the page. Its file list is exact and
  checked. A price or kit file in the board build stops the deploy, because the board
  is public and that data is not ours to redistribute.

The Space is an allowlist, not a copy of the repo. It ships the harness, the ledger,
the calendar and the organizers' validator and metric calculator, plus a 2026-only
price file (JSON: the office network blocks .csv downloads). Nothing else goes: no models, no features, no strategy code, no organizer
panel. `icaif/` is mostly strategy code, and one stray import would carry it into the
upload. So the build imports the harness from the built folder alone and refuses to
finish if anything outside the allowlist loaded.

The page runs the harness in the browser through Pyodide (space/worker.js), so the
Space is static: HF hosts Gradio Spaces only on a paid plan. The build scores the
template natively through the page's own entry point (space/webapp.py) and requires
the CLI's numbers, so the only step not checked here is Pyodide itself.

The build also scores the leaderboard's reference strategies natively, since they need
information bars neither page ships. --push deploys through `space_hub.publish`, which
checks each Space's visibility and mirrors its folder, but never touches submitted
entries.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))

import webapp  # noqa: E402
from icaif import baselines, calendar, data, holdout, leaderboard, markets, space_hub  # noqa: E402

REPO_ID = space_hub.REPO_ID
# Static because a Gradio Space needs a paid plan: Team/Enterprise for the org, PRO for
# a personal account. Both refused with HF 402 on 2026-09-30.
ICAIF_MODULES = ["__init__", "calendar", "data", "holdout", "kit", "leaderboard", "ranking",
                 "sim", "windows"]
KIT_FILES = ["kit/__init__.py", "kit/config.py", "kit/contracts.py", "kit/evaluation.py",
             "universe.json"]
SPACE_FILES = ["index.html", "worker.js", "webapp.py", "README.md"]
BOARD_FILES = {"index.html": "board/index.html", "README.md": "board/README.md",
               "boardapp.py": "board/boardapp.py", "worker.js": "space/worker.js",
               "icaif/__init__.py": "icaif/__init__.py", "icaif/ranking.py": "icaif/ranking.py",
               "icaif/leaderboard.py": "icaif/leaderboard.py"}
BOARD_PY = ["boardapp.py", "icaif/__init__.py", "icaif/ranking.py", "icaif/leaderboard.py"]
# What the worker writes into Pyodide's filesystem; the page's own HTML/JS is not.
PY_FILES = ["webapp.py", *(f"icaif/{m}.py" for m in ICAIF_MODULES),
            *(f"starter-kit/{f}" for f in KIT_FILES),
            "data/prices.json", "data/market.json"]
# From just before the holdout, so a Space scoring a start in early January has fills;
# the organizer panel (licensed to participants) is never an input here.
PRICES_FROM = "2025-12-01"


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "icaif").mkdir(parents=True)
    (out / "starter-kit" / "kit").mkdir(parents=True)
    (out / "data").mkdir()
    for m in ICAIF_MODULES:
        shutil.copy2(data.ROOT / "icaif" / f"{m}.py", out / "icaif" / f"{m}.py")
    for f in KIT_FILES:
        shutil.copy2(data.ROOT / "starter-kit" / f, out / "starter-kit" / f)
    for f in SPACE_FILES:
        shutil.copy2(data.ROOT / "space" / f, out / f)
    inline_hist(out / "index.html")

    market = markets.research_market("alpaca")
    since = market.exec_prices.index >= PRICES_FROM
    (out / "data" / "prices.json").write_text(json.dumps({
        "exec_prices": webapp.frame_doc(market.exec_prices[since]),
        "closes": webapp.frame_doc(market.closes[market.closes.index >= PRICES_FROM])}))
    snapshot = sorted((data.ROOT / "data" / "public").glob("alpaca_30m_2*.parquet"))[-1].name
    days = [d for d in market.days if str(d) >= PRICES_FROM]
    meta = {"snapshot": snapshot, "first_day": str(days[0]), "last_day": str(days[-1]),
            "holdout": [str(holdout.HOLDOUT_START), str(holdout.HOLDOUT_END)],
            "degraded_days": [d for d in market.issues.get("degraded_days", [])
                              if d >= PRICES_FROM]}
    (out / "data" / "market.json").write_text(json.dumps(meta, indent=1))
    missing = [f for f in PY_FILES if not (out / f).exists()]
    if missing:
        sys.exit(f"manifest names files the build did not write: {missing}")
    meta["reference_names"] = list(REFERENCES)
    (out / "manifest.json").write_text(json.dumps({"files": PY_FILES, "meta": meta}, indent=1))
    return meta, market, snapshot


def inline_hist(page: Path) -> None:
    """Put space/hist.js into the page itself, at its <!--HIST--> marker.

    Inlined rather than served: on the private Space a separately fetched script
    depends on HF's auth reaching that request too, and one fewer file is one fewer
    thing the board's allowlist must name.
    """
    html = page.read_text()
    if html.count("<!--HIST-->") != 1:
        sys.exit(f"{page} must have exactly one <!--HIST--> marker")
    js = (data.ROOT / "space" / "hist.js").read_text()
    page.write_text(html.replace("<!--HIST-->", f"<script>\n{js}</script>"))


def build_board(out: Path, market, snapshot: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "icaif").mkdir(parents=True)
    for dst, src in BOARD_FILES.items():
        shutil.copy2(data.ROOT / src, out / dst)
    inline_hist(out / "index.html")
    refs = write_references(out, market, snapshot)
    (out / "manifest.json").write_text(json.dumps(
        {"files": BOARD_PY, "module": "boardapp", "references": refs,
         "meta": {"holdout": [str(holdout.HOLDOUT_START), str(holdout.HOLDOUT_END)]}}, indent=1))
    # Exact, not "at least": the board is public, so anything extra is published.
    shipped = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    expected = set(BOARD_FILES) | set(refs) | {"manifest.json"}
    if shipped != expected:
        sys.exit(f"the public board build differs from its allowlist: extra "
                 f"{sorted(shipped - expected)}, missing {sorted(expected - shipped)}")


# The leaderboard's anchors. Each runs as itself, fresh in every window, not replayed
# from a file. inv_vol_hold_75 is the best baseline found so far (README: Baselines vs
# the field); cash and ew_hold are the do-nothing corners.
REFERENCES = {
    "cash": (baselines.Cash, "Never trades."),
    "ew_hold": (baselines.EqualWeightHold, "Buys 1/30 of each stock at the first round, then holds."),
    "inv_vol_hold_75": (baselines.scaled(baselines.InverseVolHold, 0.75),
                        "Inverse-vol weights at 75% gross, bought once and held."),
}


def write_references(out: Path, market, snapshot: str) -> list[str]:
    (out / "references").mkdir()
    start, end = holdout.HOLDOUT_START, holdout.HOLDOUT_END
    paths = []
    for name, (factory, note) in REFERENCES.items():
        wins, _ = holdout.rolling_runs(factory, market, start, end)
        entry = leaderboard.make_entry(
            name, leaderboard.REFERENCE, wins, span=(start, end), sizing=leaderboard.BOARD_SIZING,
            market_snapshot=snapshot, author="baseline", note=note, submitted_at="")
        path = f"references/{name}.json"
        (out / path).write_text(json.dumps(entry))
        paths.append(path)
    return paths


def check_board(out: Path) -> None:
    """The public board must import nothing but ranking and leaderboard, and must rank."""
    probe = f"""
import sys, json
sys.path = [p for p in sys.path if p not in ("", {str(data.ROOT)!r}, {str(data.ROOT / "tools")!r})]
sys.path.insert(0, {str(out)!r})
import boardapp
loaded = sorted(m for m in sys.modules if m.startswith(("icaif", "kit", "webapp")))
assert loaded == ["icaif", "icaif.leaderboard", "icaif.ranking"], loaded
refs = json.load(open({str(out / "manifest.json")!r}))["references"]
b = json.loads(boardapp.board([open({str(out)!r} + "/" + p).read() for p in refs]))
assert "error" not in b and b["entrants"] == len(refs), b
print(loaded, b["entrants"], "references ranked on", b["windows"], "windows")
"""
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=out)
    if r.returncode:
        sys.exit(f"built board does not import or rank cleanly:\n{r.stderr}")
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache)
    print(f"board imports only: {r.stdout.strip()}")


def check(out: Path) -> None:
    """Import the harness from the built folder only, and score a file with it.

    Run in a subprocess with the repo off sys.path: if a copied module imported one
    that was not copied, this is where it fails, not on the Space after upload.
    """
    probe = f"""
import sys
sys.path = [p for p in sys.path if p not in ("", {str(data.ROOT)!r}, {str(data.ROOT / "tools")!r})]
sys.path.insert(0, {str(out)!r})
import json, pandas as pd
import webapp
from icaif import holdout, leaderboard
loaded = sorted(m for m in sys.modules if m.startswith("icaif"))
outside = [m for m, mod in sys.modules.items() if m.startswith(("icaif", "kit"))
           and getattr(mod, "__file__", None) and not mod.__file__.startswith({str(out)!r})]
assert not outside, outside
allowed = {{"icaif"}} | {{"icaif." + m for m in {ICAIF_MODULES!r} if m != "__init__"}}
assert set(loaded) <= allowed, set(loaded) - allowed
print(json.dumps(loaded))
"""
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=out)
    if r.returncode:
        sys.exit(f"built Space does not import cleanly:\n{r.stderr}")
    # The probe's bytecode would otherwise be uploaded with the source.
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache)
    print(f"imports from the build only: {r.stdout.strip()}")


def parity(out: Path) -> None:
    """The page's entry point, on the shipped CSVs, must score a file as the CLI does.

    A file that rebalances only at rounds 1 and 4 exercises holds as well as trades,
    and its weights differ by window, so a window scored with another's decisions shows.
    """
    full = markets.research_market("alpaca")
    webapp._MARKET = webapp.load_market(out)
    trimmed = webapp._MARKET[0]
    if not (trimmed.exec_prices.equals(full.exec_prices.loc[trimmed.exec_prices.index])
            and trimmed.closes.equals(full.closes.loc[trimmed.closes.index])):
        sys.exit("shipped prices are not bit-identical to the full market's")
    spans, _ = holdout.window_spans(full, holdout.HOLDOUT_START, holdout.HOLDOUT_END)
    windows = {}
    for i, (ws, span) in enumerate(spans.items()):
        w = {t: (1 / 40 if (j + i) % 3 else 1 / 50) for j, t in enumerate(full.tickers)}
        cash = 1 - sum(w.values())
        windows[str(ws)] = [{"round_id": holdout.round_id(d, r["round"]), "cash": cash, "weights": w}
                            for d in span for r in calendar.rounds_for(d) if r["round"] in (1, 4)]
    probe = out.parent / "space_parity.json"
    probe.write_text(json.dumps({"strategy": "parity", "windows": windows}))
    dec = holdout.load_decisions(probe, full)
    a_roll, _ = holdout.rolling(dec, full)
    page = json.loads(webapp.score(str(probe), str(holdout.HOLDOUT_START),
                                   str(holdout.HOLDOUT_END), False, "pre_fee"))
    probe.unlink()
    b_roll = page["windows"]
    # Prices are bitwise equal; the last digits differ only because numpy's dot product
    # sums in an order that depends on memory layout. Anything above 1e-12 is data.
    close = lambda x, y: abs(x - y) <= 1e-12 * max(1.0, abs(x))  # noqa: E731
    diff = {}
    if len(b_roll) != len(a_roll):
        diff["windows"] = (len(a_roll), len(b_roll))
    else:
        diff.update({f"window {w['window_start']} {k}": (row[k], w[k])
                     for (_, row), w in zip(a_roll.iterrows(), b_roll)
                     for k in holdout.METRICS if not close(row[k], w[k])})
    if diff:
        sys.exit(f"the page's entry point scores differently from the CLI: {diff}")
    print(f"page entry point matches the CLI: {len(b_roll)} windows (to 1e-12)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=data.ROOT / "output" / "space")
    ap.add_argument("--board-out", type=Path, default=data.ROOT / "output" / "board")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--board-id", default=space_hub.BOARD_ID)
    ap.add_argument("--sync", action="store_true",
                    help="only copy dataset entries missing from the board (needs downloads, "
                         "so not on the office network), then stop")
    args = ap.parse_args()
    if args.sync:
        copied = space_hub.sync(args.board_id)
        print(f"copied {len(copied)} entries to the board: {copied}")
        return

    meta, market, snapshot = build(args.out)
    print(f"built {args.out}: prices {meta['first_day']}..{meta['last_day']} "
          f"from {meta['snapshot']}")
    check(args.out)
    parity(args.out)
    build_board(args.board_out, market, snapshot)
    check_board(args.board_out)
    if args.push:
        space_hub.publish(args.out, args.repo_id, "private")
        space_hub.publish(args.board_out, args.board_id, "public")


if __name__ == "__main__":
    main()
