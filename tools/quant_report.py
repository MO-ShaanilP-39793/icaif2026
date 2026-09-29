"""Race the quant models against the inverse-vol hold, window by window.

    .venv/bin/python tools/quant_report.py

Every candidate is ranked alone against the baseline field in each 15-day window
(`windows.rank_against_field`), as the compiler sweep does. None of these models reads
the ML scores, so every window since 2016 is out of sample for them, and the split is
by era instead: 2016-22 to look at, 2023-26 to confirm. Nothing in `quant_strategies`
was tuned on either; its constants are set a priori, and a candidate that only wins
the first split is a candidate that fit 2020.

Also reports H1 2026 on rolling (stride 1) windows, the period the hold was last
judged on. Writes reports/quant_windows.csv and reports/quant_summary.csv.
"""

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, data, markets, quant_strategies as qs, windows  # noqa: E402

CONFIRM_START = date(2023, 1, 1)
REFERENCE = "inv_vol_hold_75"


def race(market, starts, candidates):
    field = windows.run_field(baselines.FIELD, market, starts)
    rows = [windows.rank_against_field(windows.run_field({n: f}, market, starts), field)
            for n, f in candidates.items()]
    return pd.concat(rows, ignore_index=True)


def summarise(res: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    out = []
    ref = res[res.strategy == REFERENCE].set_index("window")["overall_score"]
    for (name, part), g in res.assign(split=split).groupby(["strategy", "split"]):
        g = g.set_index("window")
        d = g["overall_score"] - ref.reindex(g.index)
        out.append({"strategy": name, "split": part, "windows": len(g),
                    "score": g["overall_score"].mean(),
                    "diff_vs_hold": d.mean(), "diff_se": d.std() / len(d) ** 0.5,
                    "beats_hold": (d < 0).mean(), "loses_to_hold": (d > 0).mean(),
                    "top3": (g["position"] <= 3).mean(),
                    "ret": g["cumulative_return"].mean(), "sharpe": g["sharpe_ratio"].mean(),
                    "mdd": g["maximum_drawdown"].mean(), "worst_mdd": g["maximum_drawdown"].max(),
                    "turnover": g["turnover"].mean()})
    return pd.DataFrame(out)


def main() -> None:
    t0 = time.time()
    market = markets.research_market()
    cands = {REFERENCE: baselines.scaled(baselines.InverseVolHold, 0.75),
             "cash": baselines.Cash, **qs.CANDIDATES}

    # Skip the first 60 sessions so the shape and regime fits always have history.
    starts = [s for s in windows.window_starts(market) if s >= market.days[60]]
    res = race(market, starts, cands)
    split = pd.Series(["choose" if w < CONFIRM_START else "confirm" for w in res.window],
                      index=res.index)
    summ = summarise(res, split)

    h1 = [d for d in market.days if date(2026, 1, 1) <= d <= date(2026, 6, 30)]
    roll = [s for s in windows.window_starts(market, stride=1)
            if h1[0] <= s and market.days[market.days.index(s) + 14] <= h1[-1]]
    res_h1 = race(market, roll, cands)
    summ_h1 = summarise(res_h1, pd.Series("h1_2026_rolling", index=res_h1.index))

    allsum = pd.concat([summ, summ_h1], ignore_index=True)
    out = data.ROOT / "reports"
    pd.concat([res.assign(split=split), res_h1.assign(split="h1_2026_rolling")]).to_csv(
        out / "quant_windows.csv", index=False)
    allsum.to_csv(out / "quant_summary.csv", index=False)

    cols = ["strategy", "windows", "score", "diff_vs_hold", "diff_se", "beats_hold",
            "loses_to_hold", "top3", "ret", "sharpe", "mdd", "worst_mdd", "turnover"]
    for part in ("choose", "confirm", "h1_2026_rolling"):
        s = allsum[allsum.split == part].sort_values("score")[cols]
        print(f"\n=== {part} (lower score is better; diff < 0 beats the hold) ===")
        print(s.round(4).to_string(index=False))
    print(f"\n{len(starts)} windows + {len(roll)} rolling, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
