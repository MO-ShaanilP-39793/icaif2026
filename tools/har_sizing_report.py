"""Does the HAR forecast improve the one decision made at entry? (Roadmap step 3)

    .venv/bin/python tools/har_sizing_report.py             # choose, on 2016-10 .. 2025
    .venv/bin/python tools/har_sizing_report.py --holdout   # score the choice once, Jan-Jun 2026

Four questions, each a book bought once at entry and then held, so none pays turnover
its reference doesn't (`icaif/har_sizing.py`):

1. Weights: inverse-vol on each name's HAR forecast, at the hold's 75%.
2. Exposure: the inverse-vol hold at e = 0.75 x typical / forecast of the basket, clipped.
3. Both.
4. Inside the rule desk's book (risk parity at the HMM's exposure): (a) HAR vols on risk
   parity's covariance; (b) HAR exposure instead of, averaged with, or capping the
   HMM's; (c) both.

Each is compared with both references, inv_vol_hold_75 and q_riskparity_entry_regime,
ranked alone against the default field and against that field without inv_vol_hold:
the reference's near-clone, which hands any book that isn't a copy ~0.1 of score
(TODO, 2026-09-29).

**Discipline.** Every setting is chosen on the 2016-2025 windows alone: the 146
non-overlapping 15-day windows from 2016-10 (the first quarter the 15-session basket fit
has 100 training rows for, so every horizon is compared on the same windows) to the last
that ends in 2025. The choice is the lowest mean score on the no-clone field, the one
without the artefact. It is written to reports/har_sizing_choice.json, which must be
committed before --holdout runs; the holdout then scores exactly that choice on the 109
rolling Jan-Jun 2026 windows and counts every look in reports/har_sizing_holdout.json.
A choice tuned on the holdout would not be evidence.

**Win.** A variant wins if, against both references, its mean paired score difference is
below zero on both fields in both splits, and on the selection windows it is more than
2 SE below zero on the no-clone field. The holdout's ~8 independent windows can't carry a
2-SE bar, so there the sign alone counts. Only a winner is wired into the desk.

**Grid**, fixed before any score was seen:
- 1: the weights' horizon, 1, 3 or 15 sessions.
- 2: horizon {1, 3, 15} x typical window {250, 750} sessions x clip {[0.5, 0.95],
  [0.25, 0.95], [0.6, 0.9]}; e_ref 0.75 throughout.
- 3: 1's choice with 2's, no further search.
- 4a: risk parity's vol horizon {1, 3, 15}, at the HMM's exposure.
- 4b: mode {har, mean, min}, with 2's exposure settings, on the sample vols.
- 4c: 4a's choice with 4b's.

SE: selection windows don't overlap, so SE = sd / sqrt(n). Holdout windows share 14 of
15 days, so their SE divides by the independent count instead, as the leaderboard's does.
Writes reports/har_sizing_{selection,holdout}.csv (per window) beside the JSON.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import baselines, data, har_sizing as hs, holdout, markets  # noqa: E402
from icaif import quant_strategies as qs, windows  # noqa: E402
from icaif.agents.signals import VolForecasts  # noqa: E402
from icaif.leaderboard import independent_windows  # noqa: E402

HORIZONS = (1, 3, 15)
TYPICAL_DAYS = (250, 750)
CLIPS = ((0.5, 0.95), (0.25, 0.95), (0.6, 0.9))
RP_MODES = ("har", "mean", "min")
SELECT_START, SELECT_END = date(2016, 10, 1), date(2025, 12, 31)
REFS = ("inv_vol_hold_75", "q_riskparity_entry_regime")
FIELDS = ("default", "no_clone")
OBJECTIVE = "no_clone"
VARIANTS = ("1_weights", "2_exposure", "3_both", "4a_rp_sizing", "4b_rp_exposure", "4c_rp_both")
REPORTS = data.ROOT / "reports"
CHOICE = REPORTS / "har_sizing_choice.json"
HOLDOUT = REPORTS / "har_sizing_holdout.json"


# ----------------------------------------------------------------------------- candidates

def references() -> dict:
    return {"inv_vol_hold_75": baselines.scaled(baselines.InverseVolHold, hs.E_REF),
            "q_riskparity_entry_regime": qs.CANDIDATES["q_riskparity_entry_regime"]}


def exposure(s: dict) -> hs.HarExposure:
    return hs.HarExposure(s["horizon"], s["typical_days"], s["lo"], s["hi"])


def build(variant: str, s: dict, har: VolForecasts):
    """The factory for one variant at settings `s`, as written in the choice file."""
    if variant == "1_weights":
        return hs.inverse_vol_hold(har, s["weights_h"])
    if variant == "2_exposure":
        return hs.inverse_vol_hold(har, None, exposure(s["exposure"]))
    if variant == "3_both":
        return hs.inverse_vol_hold(har, s["weights_h"], exposure(s["exposure"]))
    if variant == "4a_rp_sizing":
        return hs.risk_parity_entry(har, s["vol_h"])
    if variant == "4b_rp_exposure":
        return hs.risk_parity_entry(har, None, s["mode"], exposure(s["exposure"]))
    if variant == "4c_rp_both":
        return hs.risk_parity_entry(har, s["vol_h"], s["mode"], exposure(s["exposure"]))
    raise ValueError(variant)


def label(variant: str, s: dict) -> str:
    parts = [variant]
    for k in ("weights_h", "vol_h"):
        if k in s:
            parts.append(f"h{s[k]}")
    if "mode" in s:
        parts.append(s["mode"])
    if "exposure" in s:
        e = s["exposure"]
        parts.append(f"e:h{e['horizon']}_t{e['typical_days']}_{e['lo']:g}-{e['hi']:g}")
    return " ".join(parts)


class Recorder:
    """Keeps every book a factory makes, so its entry exposure and fallbacks can be read."""

    def __init__(self, factory):
        self.factory, self.books = factory, []

    def __call__(self):
        b = self.factory()
        self.books.append(b)
        return b

    def stats(self) -> dict:
        e = [b.entry["exposure"] for b in self.books if b.entry is not None]
        return {"entry_exposure_mean": float(np.mean(e)) if e else None,
                "entry_exposure_min": float(np.min(e)) if e else None,
                "entry_exposure_max": float(np.max(e)) if e else None,
                "windows_with_fallback": sum(bool(b.fallbacks) for b in self.books),
                "fallbacks": [f for b in self.books for f in b.fallbacks][:5]}


# ----------------------------------------------------------------------------- scoring

def fields(market, starts) -> dict:
    default = windows.run_field(baselines.FIELD, market, starts)
    return {"default": default, "no_clone": default[default.strategy != "inv_vol_hold"]}


def race(cands: dict, market, starts, field_runs: dict) -> pd.DataFrame:
    rows = []
    for name, factory in cands.items():
        run = windows.run_field({name: factory}, market, starts)
        for f, field in field_runs.items():
            rows.append(windows.rank_against_field(run, field).assign(field=f))
    return pd.concat(rows, ignore_index=True)


def summarise(res: pd.DataFrame, n_eff: int | None = None) -> pd.DataFrame:
    """Per field and strategy: mean score and SE, paired difference from each reference."""
    out = []
    for f, g in res.groupby("field"):
        piv = g.pivot(index="window", columns="strategy", values="overall_score")
        k = n_eff or len(piv)
        by = g.groupby("strategy")
        for s in piv.columns:
            row = {"field": f, "strategy": s, "windows": len(piv), "n_eff": k,
                   "score": piv[s].mean(), "se": piv[s].std() / np.sqrt(k),
                   "top3": (by.get_group(s)["position"] <= 3).mean()}
            for ref in REFS:
                d = piv[s] - piv[ref]
                row[f"diff_{ref}"], row[f"se_{ref}"] = d.mean(), d.std() / np.sqrt(k)
            gs = by.get_group(s)
            row.update(ret=gs["cumulative_return"].mean(), sharpe=gs["sharpe_ratio"].mean(),
                       mdd=gs["maximum_drawdown"].mean(), turnover=gs["turnover"].mean())
            out.append(row)
    return pd.DataFrame(out)


def paired(summ: pd.DataFrame, strategy: str) -> dict:
    """{ref: {field: {diff, se, score, se_score, top3}}} for one strategy."""
    s = summ[summ.strategy == strategy].set_index("field")
    return {ref: {f: {"diff": float(s.at[f, f"diff_{ref}"]), "se": float(s.at[f, f"se_{ref}"]),
                      "score": float(s.at[f, "score"]), "se_score": float(s.at[f, "se"]),
                      "top3": float(s.at[f, "top3"])} for f in FIELDS} for ref in REFS}


def best(summ: pd.DataFrame, names: list[str]) -> str:
    s = summ[(summ.field == OBJECTIVE) & summ.strategy.isin(names)]
    return s.sort_values(["score", "strategy"]).iloc[0]["strategy"]


def verdict(select: dict, hold: dict) -> dict:
    checks = {}
    for ref in REFS:
        for f in FIELDS:
            checks[f"selection {f} vs {ref} < 0"] = select[ref][f]["diff"] < 0
            checks[f"holdout {f} vs {ref} < 0"] = hold[ref][f]["diff"] < 0
        nc = select[ref]["no_clone"]
        checks[f"selection no_clone vs {ref} more than 2 SE below 0"] = nc["diff"] + 2 * nc["se"] < 0
    return {"wins": all(checks.values()), "checks": checks}


def show(summ: pd.DataFrame, names=None, title=""):
    cols = ["strategy", "score", "se", "diff_inv_vol_hold_75", "se_inv_vol_hold_75",
            "diff_q_riskparity_entry_regime", "se_q_riskparity_entry_regime", "top3",
            "ret", "mdd", "turnover"]
    for f in FIELDS:
        s = summ[summ.field == f]
        if names is not None:
            s = s[s.strategy.isin(names)]
        print(f"\n=== {title} {f} field, {int(s['windows'].iloc[0])} windows "
              f"(n_eff {int(s['n_eff'].iloc[0])}); lower is better ===")
        print(s.sort_values("score")[cols].round(4).to_string(index=False))


# ----------------------------------------------------------------------------- the two splits

def selection_starts(market) -> list[date]:
    days = market.days
    return [s for s in windows.window_starts(market) if s >= SELECT_START
            and days[days.index(s) + windows.WINDOW_DAYS - 1] <= SELECT_END]


def holdout_starts(market) -> list[date]:
    days = market.days
    return [s for s in windows.window_starts(market, stride=1) if s >= holdout.HOLDOUT_START
            and days[days.index(s) + windows.WINDOW_DAYS - 1] <= holdout.HOLDOUT_END]


def forecasts(market) -> VolForecasts:
    har = VolForecasts.from_bars(market.info_bars, market.tickers, horizons=HORIZONS)
    saved = data.ROOT / "data" / "derived" / "vol_forecasts.parquet"
    if saved.exists():
        # The saved history (vol_report --save-forecasts) and these forecasts must be one
        # model: the same quarterly refits from 2017 on, read from the same bars.
        s = pd.read_parquet(saved)
        for h in (1, 3):
            a = s[f"har_h{h}"].unstack("ticker")
            a.index = pd.DatetimeIndex(a.index)
            b = har.panels[h].frame.reindex(index=a.index, columns=a.columns)
            rel = ((b - a).abs() / a).stack().max()
            print(f"h{h}: max relative difference from {saved.name}: {rel:.2e}")
    return har


def select(market, har, t0):
    starts = selection_starts(market)
    print(f"selection: {len(starts)} windows, {starts[0]} .. {starts[-1]}")
    field_runs = fields(market, starts)
    grid = {"1_weights": [{"weights_h": h} for h in HORIZONS],
            "2_exposure": [{"exposure": {"horizon": h, "typical_days": t, "lo": lo, "hi": hi,
                                         "e_ref": hs.E_REF}}
                           for h in HORIZONS for t in TYPICAL_DAYS for lo, hi in CLIPS],
            "4a_rp_sizing": [{"vol_h": h} for h in HORIZONS]}
    rec, settings = {}, {}

    def stage(pairs):
        cands = {}
        for v, s in pairs:
            name = label(v, s)
            rec[name], settings[name] = Recorder(build(v, s, har)), (v, s)
            cands[name] = rec[name]
        return race(cands, market, starts, field_runs)

    res = [race(references(), market, starts, field_runs)]
    res.append(stage([(v, s) for v, ss in grid.items() for s in ss]))
    summ = summarise(pd.concat(res, ignore_index=True))
    pick = {v: settings[best(summ, [label(v, s) for s in ss])][1] for v, ss in grid.items()}
    print(f"stage 1 done ({time.time() - t0:.0f}s)")

    expo = pick["2_exposure"]["exposure"]
    stage2 = [("3_both", {"weights_h": pick["1_weights"]["weights_h"], "exposure": expo})]
    stage2 += [("4b_rp_exposure", {"mode": m, "exposure": expo}) for m in RP_MODES]
    res.append(stage(stage2))
    summ = summarise(pd.concat(res, ignore_index=True))
    pick["3_both"] = stage2[0][1]
    pick["4b_rp_exposure"] = settings[best(summ, [label(v, s) for v, s in stage2[1:]])][1]
    pick["4c_rp_both"] = {"vol_h": pick["4a_rp_sizing"]["vol_h"],
                          "mode": pick["4b_rp_exposure"]["mode"], "exposure": expo}
    res.append(stage([("4c_rp_both", pick["4c_rp_both"])]))
    allres = pd.concat(res, ignore_index=True)
    summ = summarise(allres)
    allres.to_csv(REPORTS / "har_sizing_selection.csv", index=False)

    show(summ, title="selection 2016-10..2025:")
    era = allres.assign(era=np.where(pd.to_datetime(allres.window) < pd.Timestamp("2023-01-01"),
                                     "2016-22", "2023-25"))
    chosen = [label(v, pick[v]) for v in VARIANTS]
    for e, g in era.groupby("era"):
        show(summarise(g), names=list(REFS) + chosen, title=f"era {e}:")

    choice = {"made": str(date.today()), "objective": f"lowest mean score, {OBJECTIVE} field",
              "selection_windows": {"n": len(starts), "first": str(starts[0]), "last": str(starts[-1])},
              "grid": {"horizons": HORIZONS, "typical_days": TYPICAL_DAYS, "clips": CLIPS,
                       "rp_modes": RP_MODES, "e_ref": hs.E_REF},
              "win": "against both references: paired diff < 0 on both fields in both splits, "
                     "and > 2 SE below 0 on the no_clone field in the selection windows",
              "variants": {v: {"settings": pick[v], "label": label(v, pick[v]),
                               "selection": paired(summ, label(v, pick[v])),
                               **rec[label(v, pick[v])].stats()} for v in VARIANTS},
              "references": {r: {"selection": paired(summ, r)} for r in REFS}}
    CHOICE.write_text(json.dumps(choice, indent=1, default=str) + "\n")
    print("\nchosen (not yet scored on the holdout):")
    for v in VARIANTS:
        c = choice["variants"][v]
        nc = c["selection"]
        print(f"  {c['label']:48s} vs hold {nc['inv_vol_hold_75']['no_clone']['diff']:+.4f} "
              f"(SE {nc['inv_vol_hold_75']['no_clone']['se']:.4f}), vs rule "
              f"{nc['q_riskparity_entry_regime']['no_clone']['diff']:+.4f} "
              f"(SE {nc['q_riskparity_entry_regime']['no_clone']['se']:.4f}) on no_clone; "
              f"mean entry exposure {c['entry_exposure_mean']:.3f}, "
              f"{c['windows_with_fallback']} windows fell back")
    print(f"\nwrote {CHOICE.relative_to(data.ROOT)}: commit it before --holdout")


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=data.ROOT, capture_output=True, text=True)


def score_holdout(market, har, again: bool):
    rel = str(CHOICE.relative_to(data.ROOT))
    if _git("ls-files", "--error-unmatch", rel).returncode or _git("diff", "--quiet", "HEAD", "--", rel).returncode:
        raise SystemExit(f"{rel} is not committed as it stands: commit the choice before the holdout sees it")
    looks = json.loads(HOLDOUT.read_text())["looks"] if HOLDOUT.exists() else []
    if looks and not again:
        raise SystemExit(f"the holdout was already scored {len(looks)} time(s); --again counts another look")
    choice = json.loads(CHOICE.read_text())
    commit = _git("log", "-1", "--format=%H", "--", rel).stdout.strip()

    starts = holdout_starts(market)
    n_eff = independent_windows(pd.DataFrame(
        {"window_start": [str(s) for s in starts],
         "window_end": [str(market.days[market.days.index(s) + windows.WINDOW_DAYS - 1]) for s in starts]}))
    print(f"holdout: {len(starts)} rolling windows, {starts[0]} .. {starts[-1]}, "
          f"{n_eff} independent; choice from {commit[:7]}")
    field_runs = fields(market, starts)
    rec = {}
    cands = dict(references())
    for v in VARIANTS:
        c = choice["variants"][v]
        rec[c["label"]] = Recorder(build(v, c["settings"], har))
        cands[c["label"]] = rec[c["label"]]
    res = race(cands, market, starts, field_runs)
    res.to_csv(REPORTS / "har_sizing_holdout.csv", index=False)
    summ = summarise(res, n_eff)
    show(summ, title="holdout Jan-Jun 2026:")

    out = {"variants": {}, "references": {r: {"holdout": paired(summ, r)} for r in REFS}}
    for v in VARIANTS:
        c = choice["variants"][v]
        h = paired(summ, c["label"])
        out["variants"][v] = {"label": c["label"], "holdout": h,
                              **verdict(c["selection"], h), **rec[c["label"]].stats()}
    looks.append({"at": pd.Timestamp.now(tz="UTC").isoformat(), "choice_commit": commit,
                  "windows": len(starts), "n_eff": n_eff})
    HOLDOUT.write_text(json.dumps({"looks": looks, **out}, indent=1, default=str) + "\n")
    print("\nverdict (both references, both fields, both splits; 2 SE on no_clone selection):")
    for v in VARIANTS:
        o = out["variants"][v]
        failed = [k for k, ok in o["checks"].items() if not ok]
        print(f"  {o['label']:48s} {'WINS' if o['wins'] else 'does not win'}"
              + ("" if o["wins"] else f": fails {len(failed)}, e.g. {failed[0]}"))
    print(f"\nwrote {HOLDOUT.relative_to(data.ROOT)} (look {len(looks)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="score the committed choice on Jan-Jun 2026")
    ap.add_argument("--again", action="store_true", help="score the holdout again, counted as another look")
    args = ap.parse_args()
    t0 = time.time()
    market = markets.research_market()
    har = forecasts(market)
    if args.holdout:
        score_holdout(market, har, args.again)
    else:
        select(market, har, t0)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
