"""Does the ranking of our candidates depend on who else is playing?

    .venv/bin/python tools/field_sensitivity.py

Our default field (`baselines.FIELD`) is five near-holds out of nine, which makes any
trade after entry cost several turnover ranks. The real field is more likely LLM
agents trading daily or hourly. This ranks the same candidates against both fields
and prints each metric's mean rank on its own, so a candidate that buys return and
Sharpe with turnover shows where it gains and where it pays.

Windows start 2023-01-03, where the model scores are out of sample, so the model-tilt
candidates are judged on the same windows as the rest. Writes
reports/field_sensitivity.csv.
"""

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, compiler, data, markets, quant_strategies as qs, windows  # noqa: E402
from icaif.compiler import Levers  # noqa: E402

OOS_START = date(2023, 1, 3)
RANKS = ["rank_cumulative_return", "rank_sharpe_ratio", "rank_maximum_drawdown", "rank_turnover"]


def main() -> None:
    t0 = time.time()
    market = markets.research_market()
    scores = compiler.load_daily_scores()
    vol = compiler.trailing_daily_vol()
    last = scores.frame.index.max().date()
    days = market.days
    starts = [s for s in windows.window_starts(market) if s >= OOS_START
              and days[days.index(s) + windows.WINDOW_DAYS - 1] <= last]
    tilt = lambda **kw: compiler.compiled(scores, vol, Levers(weighting="tilt", **kw))  # noqa: E731
    cands = {
        "inv_vol_hold_75": baselines.scaled(baselines.InverseVolHold, 0.75),
        "inv_vol_hold_100": baselines.InverseVolHold,
        "rp_entry_regime": qs.CANDIDATES["q_riskparity_entry_regime"],
        "rp_hold_90": qs.CANDIDATES["q_riskparity_fixed90"],
        "rp_hold_100": qs.book("risk_parity", lambda: qs.Fixed(1.0), qs.ENTRY_ONLY),
        "ew_hold_100": baselines.EqualWeightHold,
        "tilt1_weekly_90": tilt(tilt=1.0, rebalance_every=5, exposure=0.9, band=0.02),
        "tilt1_weekly_100": tilt(tilt=1.0, rebalance_every=5, exposure=1.0, band=0.02),
        "tilt2_weekly_100": tilt(tilt=2.0, rebalance_every=5, exposure=1.0, band=0.02),
        "tilt1_daily_100": tilt(tilt=1.0, rebalance_every=1, exposure=1.0, band=0.02),
        "compiled_default": compiler.compiled(scores, vol),
        "ou_tilt": qs.CANDIDATES["q_ou_tilt"],
        "invvol_voltarget": qs.CANDIDATES["q_invvol_voltarget"],
    }
    cand_runs = {n: windows.run_field({n: f}, market, starts) for n, f in cands.items()}
    rows = []
    for fname, field_def in (("default", baselines.FIELD), ("active", baselines.ACTIVE_FIELD)):
        field = windows.run_field(field_def, market, starts)
        for n, run in cand_runs.items():
            r = windows.rank_against_field(run, field)
            rows.append(r.assign(field=fname))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(data.ROOT / "reports" / "field_sensitivity.csv", index=False)

    for fname in ("default", "active"):
        g = res[res.field == fname].groupby("strategy")
        s = pd.DataFrame({"score": g["overall_score"].mean(),
                          "se": g["overall_score"].std() / g["overall_score"].count() ** 0.5,
                          **{c.replace("rank_", "r_"): g[c].mean() for c in RANKS},
                          "ret": g["cumulative_return"].mean(), "sharpe": g["sharpe_ratio"].mean(),
                          "mdd": g["maximum_drawdown"].mean(), "turnover": g["turnover"].mean()})
        n = len(baselines.FIELD if fname == "default" else baselines.ACTIVE_FIELD) + 1
        print(f"\n=== {fname} field ({n} entrants; lower ranks are better) ===")
        print(s.sort_values("score").round(4).to_string())
    a = res.pivot_table(index=["field", "window"], columns="strategy", values="overall_score")
    print("\nrank of each candidate within the candidate list, by field:")
    print(a.groupby(level="field").mean().rank(axis=1).T.sort_values("active").to_string())
    print(f"\n{len(starts)} windows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
