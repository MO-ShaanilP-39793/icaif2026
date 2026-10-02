"""Does booking part of a winner beat holding it? (Roadmap step 5)

    .venv/bin/python tools/trim_report.py             # choose, on 2016-10 .. 2025 (~12 min)
    .venv/bin/python tools/trim_report.py --holdout   # score the choice once, Jan-Jun 2026

The rule (`icaif/trim.py`): at each morning review from day 2, trim `fraction` of a held
name that is still a winner (gain since entry >= a sigma sqrt n), has turned (given back
>= b sigma from its high-water mark), and whose expected give-back over the sessions left
beats the 20 bps round trip plus `rank_hit`. sigma is the name's HAR 3-session forecast.
Once per name, at most three a window. The book underneath is the rule desk's own,
`q_riskparity_entry_regime`: with no trim it is that candidate trade for trade (a test
checks), so a score difference is the trims' doing alone. A test also holds the book
scored here to the desk that would trade it (`Desk(trim=...)`), trade for trade.

**Three ways to expect a give-back**, each a variant with its own chosen settings:

- `trailing`: what a name has given back, it gives back again (the classic profit-take).
- `expanding`: minus the mean forward return of past states that met the same two
  conditions, over every window that ended before this one began (`trim.GiveBack`).
- `rolling3y`: the same over the three years before.

**Discipline**, as tools/har_sizing_report.py: settings are chosen on the 146
non-overlapping windows from 2016-10 to the last that ends in 2025, by the lowest mean
score on the no-clone field (the default field without inv_vol_hold, whose near-clone of
the reference hands any other book ~0.1). The choice goes to reports/trim_choice.json,
which must be committed before --holdout runs; the holdout then scores exactly that
choice on the 109 rolling Jan-Jun 2026 windows and counts every look in
reports/trim_holdout.json.

**Win**, as step 3's: against both references, the mean paired score difference is below
zero on both fields in both splits, and on the selection windows more than 2 SE below
zero on the no-clone field. Only a winner becomes the rule desk's trim, and then the
candidate it must reproduce.

**Grid**, fixed before any score was seen:
- a {0.5, 1}, b {1, 2}, fraction {0.25, 0.5}, for every estimator;
- rank_hit {0, 20 bps} for the two estimated give-backs; for `trailing` only 0, since a
  give-back of b sigma (1% and more) already clears any cost on the grid.

SE: selection windows don't overlap, so SE = sd / sqrt(n). Holdout windows share 14 of 15
days, so their SE divides by the independent count, as the leaderboard's does. Writes
reports/trim_{selection,holdout}.csv (per window) beside the JSON.
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import har_sizing_report as H  # noqa: E402  (the step-3 race: fields, pairing, the win rule)
from icaif import data, markets, trim as TR, windows  # noqa: E402
from icaif.leaderboard import independent_windows  # noqa: E402

A, B, FRACTION = (0.5, 1.0), (1.0, 2.0), (0.25, 0.5)
RANK_HIT = (0.0, 0.002)
VARIANTS = TR.ESTIMATORS
REPORTS = data.ROOT / "reports"
CHOICE = REPORTS / "trim_choice.json"
HOLDOUT = REPORTS / "trim_holdout.json"


def _rel(path: Path) -> str:
    return str(path.relative_to(data.ROOT)) if path.is_relative_to(data.ROOT) else str(path)


def grid() -> dict:
    out = {}
    for est in VARIANTS:
        hits = (0.0,) if est == "trailing" else RANK_HIT
        out[est] = [TR.TrimSettings(a, b, f, est, lam) for a in A for b in B for f in FRACTION
                    for lam in hits]
    return out


class Recorder:
    """Keeps every book a factory makes, so the trims it made can be counted."""

    def __init__(self, factory):
        self.factory, self.books = factory, []

    def __call__(self):
        b = self.factory()
        self.books.append(b)
        return b

    def stats(self) -> dict:
        trims = [t for b in self.books for t in b.trims]
        return {"trims": len(trims), "windows_with_a_trim": sum(bool(b.trims) for b in self.books),
                "windows": len(self.books),
                "mean_give_back_at_trim": float(np.mean([t["give_back"] for t in trims])) if trims else None,
                "mean_expected_give_back": float(np.mean([t["egb"] for t in trims])) if trims else None}


def history(market, har) -> TR.GiveBack:
    """Every non-overlapping window's states, 2016 on; each read only after it ended."""
    return TR.GiveBack.build(market, har, windows.window_starts(market))


def run_grid(settings: list, market, har, gb, starts) -> dict:
    """{label: (unranked runs, trim stats)}, one setting at a time."""
    out = {}
    for s in settings:
        rec = Recorder(TR.trimmed(TR.TrimRule(s, None if s.estimator == "trailing" else gb), har))
        out[s.label()] = (windows.run_field({s.label(): rec}, market, starts), rec.stats())
    return out


def rank(runs: dict, field_runs: dict) -> pd.DataFrame:
    return pd.concat([windows.rank_against_field(run, field).assign(field=f)
                      for run, _ in runs.values() for f, field in field_runs.items()],
                     ignore_index=True)


def select(market, har, t0):
    starts = H.selection_starts(market)
    print(f"selection: {len(starts)} windows, {starts[0]} .. {starts[-1]}")
    gb = history(market, har)
    print(f"give-back history: {len(gb.states)} states ({time.time() - t0:.0f}s)")
    settings = {s.label(): s for ss in grid().values() for s in ss}
    runs = run_grid(list(settings.values()), market, har, gb, starts)
    print(f"{len(settings)} settings run ({time.time() - t0:.0f}s)")
    field_runs = H.fields(market, starts)
    allres = pd.concat([H.race(H.references(), market, starts, field_runs), rank(runs, field_runs)],
                       ignore_index=True)
    summ = H.summarise(allres)
    allres.to_csv(REPORTS / "trim_selection.csv", index=False)
    pick = {est: settings[H.best(summ, [s.label() for s in ss])] for est, ss in grid().items()}
    stats = {k: st for k, (_, st) in runs.items()}

    H.show(summ, title="selection 2016-10..2025:")
    era = allres.assign(era=np.where(pd.to_datetime(allres.window) < pd.Timestamp("2023-01-01"),
                                     "2016-22", "2023-25"))
    chosen = [pick[v].label() for v in VARIANTS]
    for e, g in era.groupby("era"):
        H.show(H.summarise(g), names=list(H.REFS) + chosen, title=f"era {e}:")
    print("\ntrims made by each setting (selection windows):")
    for name in settings:
        st = stats[name]
        print(f"  {name:34s} {st['trims']:4d} trims in {st['windows_with_a_trim']:3d} windows")

    choice = {"made": str(date.today()), "objective": f"lowest mean score, {H.OBJECTIVE} field",
              "selection_windows": {"n": len(starts), "first": str(starts[0]), "last": str(starts[-1])},
              "grid": {"a": A, "b": B, "fraction": FRACTION, "rank_hit": RANK_HIT,
                       "trailing_rank_hit": [0.0], "cost": TR.COST, "horizon": TR.HORIZON,
                       "min_trim": TR.MIN_TRIM, "max_trims": TR.MAX_TRIMS, "min_states": TR.MIN_STATES},
              "win": "against both references: paired diff < 0 on both fields in both splits, "
                     "and > 2 SE below 0 on the no_clone field in the selection windows",
              "variants": {v: {"settings": pick[v].to_json(), "label": pick[v].label(),
                               "selection": H.paired(summ, pick[v].label()),
                               **stats[pick[v].label()]} for v in VARIANTS},
              "references": {r: {"selection": H.paired(summ, r)} for r in H.REFS}}
    CHOICE.write_text(json.dumps(choice, indent=1, default=str) + "\n")
    print("\nchosen (not yet scored on the holdout):")
    for v in VARIANTS:
        c = choice["variants"][v]
        nc = c["selection"]
        print(f"  {c['label']:34s} vs hold {nc['inv_vol_hold_75']['no_clone']['diff']:+.4f} "
              f"(SE {nc['inv_vol_hold_75']['no_clone']['se']:.4f}), vs rule "
              f"{nc['q_riskparity_entry_regime']['no_clone']['diff']:+.4f} "
              f"(SE {nc['q_riskparity_entry_regime']['no_clone']['se']:.4f}) on no_clone; "
              f"{c['trims']} trims in {c['windows_with_a_trim']} windows")
    print(f"\nwrote {_rel(CHOICE)}: commit it before --holdout")


def score_holdout(market, har, again: bool):
    rel = str(_rel(CHOICE))
    if (H._git("ls-files", "--error-unmatch", rel).returncode
            or H._git("diff", "--quiet", "HEAD", "--", rel).returncode):
        raise SystemExit(f"{rel} is not committed as it stands: commit the choice before the holdout sees it")
    looks = json.loads(HOLDOUT.read_text())["looks"] if HOLDOUT.exists() else []
    if looks and not again:
        raise SystemExit(f"the holdout was already scored {len(looks)} time(s); --again counts another look")
    choice = json.loads(CHOICE.read_text())
    commit = H._git("log", "-1", "--format=%H", "--", rel).stdout.strip()

    starts = H.holdout_starts(market)
    n_eff = independent_windows(pd.DataFrame(
        {"window_start": [str(s) for s in starts],
         "window_end": [str(market.days[market.days.index(s) + windows.WINDOW_DAYS - 1]) for s in starts]}))
    print(f"holdout: {len(starts)} rolling windows, {starts[0]} .. {starts[-1]}, "
          f"{n_eff} independent; choice from {commit[:7]}")
    field_runs = H.fields(market, starts)
    gb = history(market, har)
    rec, cands = {}, dict(H.references())
    for v in VARIANTS:
        c = choice["variants"][v]
        s = TR.TrimSettings(**c["settings"])
        rec[c["label"]] = Recorder(TR.trimmed(TR.TrimRule(s, None if v == "trailing" else gb), har))
        cands[c["label"]] = rec[c["label"]]
    res = H.race(cands, market, starts, field_runs)
    res.to_csv(REPORTS / "trim_holdout.csv", index=False)
    summ = H.summarise(res, n_eff)
    H.show(summ, title="holdout Jan-Jun 2026:")

    out = {"variants": {}, "references": {r: {"holdout": H.paired(summ, r)} for r in H.REFS}}
    for v in VARIANTS:
        c = choice["variants"][v]
        h = H.paired(summ, c["label"])
        out["variants"][v] = {"label": c["label"], "holdout": h,
                              **H.verdict(c["selection"], h), **rec[c["label"]].stats()}
    looks.append({"at": pd.Timestamp.now(tz="UTC").isoformat(), "choice_commit": commit,
                  "windows": len(starts), "n_eff": n_eff})
    HOLDOUT.write_text(json.dumps({"looks": looks, **out}, indent=1, default=str) + "\n")
    print("\nverdict (both references, both fields, both splits; 2 SE on no_clone selection):")
    for v in VARIANTS:
        o = out["variants"][v]
        failed = [k for k, ok in o["checks"].items() if not ok]
        print(f"  {o['label']:34s} {'WINS' if o['wins'] else 'does not win'}"
              + ("" if o["wins"] else f": fails {len(failed)}, e.g. {failed[0]}")
              + f"; {o['trims']} trims in {o['windows_with_a_trim']} windows")
    print(f"\nwrote {_rel(HOLDOUT)} (look {len(looks)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="score the committed choice on Jan-Jun 2026")
    ap.add_argument("--again", action="store_true", help="score the holdout again, counted as another look")
    args = ap.parse_args()
    t0 = time.time()
    market = markets.research_market()
    har = H.forecasts(market)
    if args.holdout:
        score_holdout(market, har, args.again)
    else:
        select(market, har, t0)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
