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
the calendar and the organizers' validator and metric calculator, plus a price file
(JSON: the office network blocks .csv downloads) holding Dec 2025 on and every fixed
suite's windows (icaif/suites.py), nothing between them. Nothing else goes: no models, no features, no strategy code, no organizer
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
from datetime import date
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))

import webapp  # noqa: E402
from icaif import baselines, calendar, data, holdout, leaderboard, markets, space_hub, suites  # noqa: E402

REPO_ID = space_hub.REPO_ID
# Static because a Gradio Space needs a paid plan: Team/Enterprise for the org, PRO for
# a personal account. Both refused with HF 402 on 2026-09-30.
ICAIF_MODULES = ["__init__", "calendar", "data", "holdout", "kit", "leaderboard", "ranking",
                 "sim", "suites", "windows"]
KIT_FILES = ["kit/__init__.py", "kit/config.py", "kit/contracts.py", "kit/evaluation.py",
             "universe.json"]
SPACE_FILES = ["index.html", "worker.js", "webapp.py", "README.md"]
BOARD_FILES = {"index.html": "board/index.html", "README.md": "board/README.md",
               "boardapp.py": "board/boardapp.py", "worker.js": "space/worker.js",
               "icaif/__init__.py": "icaif/__init__.py", "icaif/ranking.py": "icaif/ranking.py",
               "icaif/leaderboard.py": "icaif/leaderboard.py", "icaif/suites.py": "icaif/suites.py"}
BOARD_PY = ["boardapp.py", "icaif/__init__.py", "icaif/ranking.py", "icaif/leaderboard.py",
            "icaif/suites.py"]
# What the worker writes into Pyodide's filesystem; the page's own HTML/JS is not.
PY_FILES = ["webapp.py", *(f"icaif/{m}.py" for m in ICAIF_MODULES),
            *(f"starter-kit/{f}" for f in KIT_FILES),
            "data/prices.json", "data/market.json"]
# From just before the holdout, so a Space scoring a start in early January has fills;
# the organizer panel (licensed to participants) is never an input here. Fixed suites'
# windows before this date are added session by session (`shipped_days`).
PRICES_FROM = "2025-12-01"


def shipped_days(market) -> list:
    """The sessions the price file carries: PRICES_FROM on, plus every fixed suite's windows.

    Only the windows, not the months between them: the file is downloaded on every page
    load, and Alpaca's prices are not ours to hand out beyond what scoring needs.
    """
    days = {d for d in market.days if str(d) >= PRICES_FROM}
    for suite in suites.SUITES.values():
        spans, _ = holdout.suite_spans(market, suite)
        days |= {d for span in spans.values() for d in span}
    return sorted(days)


def _on_days(frame, days: set):
    return frame[[ts.date() in days for ts in frame.index]]


def price_file_problems(shipped, full) -> list[str]:
    """Why the shipped market would score a suite differently from the full one; [] if none.

    Two ways to be wrong while every check on the numbers passes: a price that differs
    (every fill drifts a little), or a session missing from a window. The second is worse
    in a fixed suite, whose windows are islands in the file: the harness refuses a short
    window, but a rolling suite would just score fewer windows, or windows whose 15
    sessions skip a day. So every suite's windows on the shipped market must be the full
    market's, session for session and round for round.
    """
    problems = []
    for name in ("exec_prices", "closes"):
        mine, theirs = getattr(shipped, name), getattr(full, name)
        extra = mine.index.difference(theirs.index)
        if len(extra):
            problems.append(f"{name}: {len(extra)} timestamps the full market lacks, first {extra[0]}")
        elif not mine.equals(theirs.loc[mine.index]):
            problems.append(f"{name}: not bit-identical to the full market's")
    for suite in suites.SUITES.values():
        want, _ = holdout.suite_spans(full, suite)
        try:
            got, _ = holdout.suite_spans(shipped, suite)
        except holdout.DecisionFileError as err:
            problems.append(f"suite {suite.name}: {err}")
            continue
        if got != want:
            off = sorted((set(got) ^ set(want)) | {k for k in got if got[k] != want.get(k)})
            problems.append(f"suite {suite.name}: windows differ from the full market's, first at {off[0]}")
            continue
        days = {d for span in want.values() for d in span}
        for name in ("exec_prices", "closes"):
            a = _on_days(getattr(shipped, name), days).index
            b = _on_days(getattr(full, name), days).index
            if not a.equals(b):
                gone = b.difference(a)
                problems.append(f"suite {suite.name}: {name} lacks {len(gone)} of its rounds"
                                + (f", first {gone[0]}" if len(gone) else ""))
    return problems


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
    days = shipped_days(market)
    keep = set(days)
    (out / "data" / "prices.json").write_text(json.dumps({
        "exec_prices": webapp.frame_doc(_on_days(market.exec_prices, keep)),
        "closes": webapp.frame_doc(_on_days(market.closes, keep))}))
    snapshot = sorted((data.ROOT / "data" / "public").glob("alpaca_30m_2*.parquet"))[-1].name
    meta = {"snapshot": snapshot, "first_day": str(days[0]), "last_day": str(days[-1]),
            "trading_days": len(days), "continuous_from": PRICES_FROM,
            "holdout": [str(holdout.HOLDOUT_START), str(holdout.HOLDOUT_END)],
            # The page's suite selector reads these, never a copy of its own.
            "suites": [s.doc() for s in suites.SUITES.values()],
            "degraded_days": [d for d in market.issues.get("degraded_days", [])
                              if date.fromisoformat(d) in keep]}
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


def build_board(out: Path, refs: dict, snapshot: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "icaif").mkdir(parents=True)
    for dst, src in BOARD_FILES.items():
        shutil.copy2(data.ROOT / src, out / dst)
    inline_hist(out / "index.html")
    refs = write_references(out, refs, snapshot)
    (out / "manifest.json").write_text(json.dumps(
        {"files": BOARD_PY, "module": "boardapp", "references": refs,
         "meta": {"holdout": [str(holdout.HOLDOUT_START), str(holdout.HOLDOUT_END)],
                  "suites": [s.doc() for s in suites.SUITES.values()]}}, indent=1))
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


def score_references(market) -> dict:
    """Every suite's references, {suite: {name: per-window DataFrame}}, on the full market.

    Natively, here: inv_vol_hold_75 reads information bars, which neither page ships.
    """
    out = {}
    for suite in suites.SUITES.values():
        out[suite.name] = {name: holdout.rolling_runs(factory, market, suite)[0]
                           for name, (factory, _) in REFERENCES.items()}
        print(f"references scored on {suite.name}: {len(out[suite.name]['cash'])} windows")
    return out


def write_references(out: Path, scored: dict, snapshot: str) -> list[str]:
    """One file per suite and reference, references/<suite>/<name>.json, each naming its
    suite: without it, a second suite's cash would be a second holdout cash."""
    paths = []
    for suite in suites.SUITES.values():
        (out / "references" / suite.name).mkdir(parents=True)
        for name, (_, note) in REFERENCES.items():
            entry = leaderboard.make_entry(
                name, leaderboard.REFERENCE, scored[suite.name][name], span=tuple(suite.span),
                sizing=leaderboard.BOARD_SIZING, market_snapshot=snapshot, author="baseline",
                note=note, submitted_at="", suite=suite.name)
            path = f"references/{suite.name}/{name}.json"
            (out / path).write_text(json.dumps(entry))
            paths.append(path)
    return paths


def check_board(out: Path) -> None:
    """The public board must import nothing but the ranking code, and must rank every suite."""
    probe = f"""
import sys, json
sys.path = [p for p in sys.path if p not in ("", {str(data.ROOT)!r}, {str(data.ROOT / "tools")!r})]
sys.path.insert(0, {str(out)!r})
import boardapp
loaded = sorted(m for m in sys.modules if m.startswith(("icaif", "kit", "webapp")))
assert loaded == ["icaif", "icaif.leaderboard", "icaif.ranking", "icaif.suites"], loaded
refs = json.load(open({str(out / "manifest.json")!r}))["references"]
b = json.loads(boardapp.board([open({str(out)!r} + "/" + p).read() for p in refs]))
assert "error" not in b, b
ranked = {{"holdout": b, **b["suites"]}}
assert sorted(ranked) == {sorted(suites.SUITES)!r}, sorted(ranked)
assert all(s["entrants"] == {len(REFERENCES)} for s in ranked.values()) and not b["excluded"], b["excluded"]
print(loaded, ", ".join(f"{{n}}: {{s['entrants']}} references on {{s['windows']}} windows"
                        for n, s in ranked.items()))
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


def parity(out: Path, full) -> None:
    """The page's entry point, on the shipped prices, must score every suite as the CLI does.

    A file that rebalances only at rounds 1 and 4 exercises holds as well as trades,
    and its weights differ by window, so a window scored with another's decisions shows.
    """
    webapp._MARKET = webapp.load_market(out)
    problems = price_file_problems(webapp._MARKET[0], full)
    if problems:
        sys.exit("the shipped price file would score differently:\n  " + "\n  ".join(problems))
    for suite in suites.SUITES.values():
        _parity_suite(out, full, suite)


def _parity_suite(out: Path, full, suite) -> None:
    spans, _ = holdout.suite_spans(full, suite)
    windows = {}
    for i, (ws, span) in enumerate(spans.items()):
        w = {t: (1 / 40 if (j + i) % 3 else 1 / 50) for j, t in enumerate(full.tickers)}
        cash = 1 - sum(w.values())
        windows[str(ws)] = [{"round_id": holdout.round_id(d, r["round"]), "cash": cash, "weights": w}
                            for d in span for r in calendar.rounds_for(d) if r["round"] in (1, 4)]
    probe = out.parent / "space_parity.json"
    probe.write_text(json.dumps({"strategy": "parity", "suite": suite.name, "windows": windows}))
    dec = holdout.load_decisions(probe, full, suite)
    a_roll, _ = holdout.rolling(dec, full)
    page = json.loads(webapp.score(str(probe), "", "", False, "pre_fee", suite.name))
    probe.unlink()
    if "rejected" in page:
        sys.exit(f"the page rejected the {suite.name} parity file: {page['rejected']}")
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
        sys.exit(f"the page's entry point scores {suite.name} differently from the CLI: {diff}")
    print(f"page entry point matches the CLI on {suite.name}: {len(b_roll)} windows (to 1e-12)")


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
    parity(args.out, market)
    build_board(args.board_out, score_references(market), snapshot)
    check_board(args.board_out)
    if args.push:
        space_hub.publish(args.out, args.repo_id, "private")
        space_hub.publish(args.board_out, args.board_id, "public")


if __name__ == "__main__":
    main()
