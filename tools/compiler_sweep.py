"""Sweep compiler levers, choosing on 2023-24 windows and confirming on 2025-26.

    .venv/bin/python tools/compiler_sweep.py [--sanity-only | --report-only]

Each candidate is ranked alone against FIELD (`windows.rank_against_field`), so the
sweep's own candidates never crowd each other's ranks. References, scored the same
way: inv_vol_hold at 75% gross (the best baseline) and a cash entrant (it ties the
field's own cash, as any copy of a field member does).

**Why two splits.** ~150 candidates on ~33 windows will produce a winner by luck
alone; the best choose-split score is biased low (better) by the selection. Only
the confirm split, never looked at while choosing, says whether it is real, and a
plateau (neighbouring settings scoring alike) says more than any single point.

**Sanity check first.** tilt=0 with a cadence longer than any window buys the
inverse-vol book once and holds it: the same strategy as inv_vol_hold at 75%, with
daily rather than hourly vol. If the two disagree by more than noise, something in
the compiler's plumbing is wrong and every sweep number inherits it.

Windows start on or after 2023-01-03 (predictions are out of sample only from then)
and end by the last prediction date. Writes reports/compiler_sweep.csv.
"""

import argparse
import itertools
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, compiler, data, markets, windows  # noqa: E402
from icaif.compiler import Levers  # noqa: E402

OOS_START = date(2023, 1, 3)
CONFIRM_START = date(2025, 1, 1)
NEVER = 10 ** 6  # a cadence longer than any window: enter once, hold
LEVER_COLS = ["weighting", "tilt", "top_k", "rebalance_every", "exposure", "band", "gamma"]


def grid() -> dict[str, Levers]:
    out = {}
    for tilt, every, exp, band, gamma in itertools.product(
            (0, 0.25, 0.5, 1, 2), (1, 5, 10, 15), (0.5, 0.75), (0.02, 0.05), (0, 0.5)):
        if tilt == 0 and gamma != 0:
            continue  # gamma only moves the tilt's percentile; at tilt 0 it is a duplicate
        out[f"tilt{tilt}_e{every}_x{exp}_b{band}_g{gamma}"] = Levers(
            weighting="tilt", tilt=tilt, rebalance_every=every, exposure=exp, band=band,
            gamma=gamma)
    for k, every, gamma in itertools.product((10, 20), (5, 15), (0, 0.5)):
        out[f"top{k}_e{every}_x0.75_b0.05_g{gamma}"] = Levers(
            top_k=k, rebalance_every=every, exposure=0.75, band=0.05, gamma=gamma)
    return out


def split_stats(ranked: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Per split: mean score, SE, paired difference from the reference and its SE."""
    out = {}
    r = ranked.set_index("window")
    d_all = r["overall_score"] - ref.set_index("window")["overall_score"]
    for name, mask in (("choose", r.index < CONFIRM_START), ("confirm", r.index >= CONFIRM_START)):
        s, d = r.loc[mask], d_all[mask]
        n = len(s)
        out.update({
            f"{name}_score": s["overall_score"].mean(),
            f"{name}_se": s["overall_score"].std() / n ** 0.5,
            f"{name}_diff_ivh75": d.mean(),
            f"{name}_diff_se": d.std() / n ** 0.5,
            f"{name}_return": s["cumulative_return"].mean(),
            f"{name}_sharpe": s["sharpe_ratio"].mean(),
            f"{name}_mdd": s["maximum_drawdown"].mean(),
            f"{name}_turnover": s["turnover"].mean(),
            f"{name}_windows": n,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print the summary from reports/compiler_sweep.csv without running")
    args = ap.parse_args()
    if args.report_only:
        report(pd.read_csv(data.ROOT / "reports" / "compiler_sweep.csv", index_col=0))
        return

    t0 = time.time()
    market = markets.research_market()
    scores = compiler.load_daily_scores()
    vol = compiler.trailing_daily_vol()
    last_pred = scores.frame.index.max().date()
    days = market.days
    starts = [s for s in windows.window_starts(market) if s >= OOS_START
              and days[days.index(s) + windows.WINDOW_DAYS - 1] <= last_pred]
    field = windows.run_field(baselines.FIELD, market, starts)

    def score(name, factory):
        return windows.rank_against_field(windows.run_field({name: factory}, market, starts), field)

    ref = score("inv_vol_hold_75", baselines.scaled(baselines.InverseVolHold, 0.75))
    cash = score("cash", baselines.Cash)
    hold0 = score("tilt0_hold", compiler.compiled(scores, vol, Levers(
        weighting="tilt", tilt=0.0, rebalance_every=NEVER, exposure=0.75)))
    a, b = hold0.set_index("window"), ref.set_index("window")
    d = a["overall_score"] - b["overall_score"]
    sanity = {
        "windows": len(starts), "first": str(starts[0]), "last": str(starts[-1]),
        "tilt0_hold_score": round(a["overall_score"].mean(), 3),
        "inv_vol_hold_75_score": round(b["overall_score"].mean(), 3),
        "paired_diff": round(d.mean(), 3), "paired_diff_se": round(d.std() / len(d) ** 0.5, 3),
        "return_corr": round(a["cumulative_return"].corr(b["cumulative_return"]), 4),
        "mean_abs_return_gap_bps": round((a["cumulative_return"] - b["cumulative_return"])
                                         .abs().mean() * 1e4, 2),
        "turnover": [round(a["turnover"].mean() * 100, 3), round(b["turnover"].mean() * 100, 3)],
    }
    print(json.dumps({"sanity": sanity}, indent=1))
    if args.sanity_only:
        return

    rows = {"inv_vol_hold_75": {"weighting": "reference", **split_stats(ref, ref)},
            "cash": {"weighting": "reference", **split_stats(cash, ref)}}
    for name, lv in grid().items():
        ranked = score(name, compiler.compiled(scores, vol, lv))
        rows[name] = {**{c: getattr(lv, c) for c in LEVER_COLS}, **split_stats(ranked, ref)}
    table = pd.DataFrame(rows).T
    table.index.name = "candidate"
    out = data.ROOT / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "compiler_sweep.csv")

    report(table, time.time() - t0)


def report(table: pd.DataFrame, seconds: float = float("nan")) -> None:
    table = table.copy()
    num_cols = [c for c in table.columns if c != "weighting"]
    table[num_cols] = table[num_cols].astype(float)
    cols = ["choose_score", "choose_se", "choose_diff_ivh75", "choose_diff_se",
            "confirm_score", "confirm_se", "confirm_diff_ivh75", "confirm_diff_se",
            "confirm_turnover"]
    num = table[cols].copy()
    num["confirm_turnover"] *= 100
    print(f"\n{len(table) - 2} candidates, {seconds:.0f} s; "
          f"choose windows {int(table.iloc[0]['choose_windows'])}, "
          f"confirm {int(table.iloc[0]['confirm_windows'])}")
    print("\nReferences")
    print(num.loc[["inv_vol_hold_75", "cash"]].round(3).to_string())
    cands = num.drop(index=["inv_vol_hold_75", "cash"]).sort_values("choose_score")
    print("\nBest ten on the choose split (turnover in %; diff < 0 beats the hold)")
    print(cands.head(10).round(3).to_string())

    # The plateau is read off the tilt family, where tilt=0 is the hold itself in the
    # compiler's own plumbing (daily vol). Measuring a tilt against tilt=0 at the same
    # settings isolates what the scores add; against inv_vol_hold_75 it also carries
    # the daily-vs-hourly vol difference, which is not the model.
    tilt = table[table.weighting == "tilt"]
    best = tilt.loc[tilt["choose_score"].idxmin()]
    fixed = (tilt[["exposure", "band", "gamma"]] == best[["exposure", "band", "gamma"]]).all(axis=1)
    sl = tilt[fixed | ((tilt.tilt == 0) & (tilt[["exposure", "band"]] == best[["exposure", "band"]])
                       .all(axis=1))]
    print(f"\nPlateau: tilt x rebalance_every at exposure {best.exposure}, band {best.band}, "
          f"gamma {best.gamma} (the best tilt setting on choose)")
    for split in ("choose", "confirm"):
        piv = sl.pivot_table(index="tilt", columns="rebalance_every",
                             values=f"{split}_diff_ivh75")
        print(f"{split}: overall-score difference from inv_vol_hold_75")
        print(piv.round(3).to_string())
        print(f"{split}: difference from tilt=0 at the same cadence (what the scores add)")
        print((piv - piv.loc[0.0]).round(3).to_string())
    print("\nConfirm-split turnover % by tilt x rebalance_every (same slice)")
    print((sl.pivot_table(index="tilt", columns="rebalance_every", values="confirm_turnover")
           * 100).round(2).to_string())
    top = num.drop(index=["inv_vol_hold_75", "cash"])
    print(f"\nconfirm/choose rank correlation across all candidates: "
          f"{top['choose_score'].corr(top['confirm_score'], method='spearman'):.2f}")

if __name__ == "__main__":
    main()
