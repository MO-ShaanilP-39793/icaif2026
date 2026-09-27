"""Rank the compiled model strategy against the baseline field, out of sample.

    .venv/bin/python tools/compiler_report.py

Daily d5 predictions are walk-forward out of sample only from 2023, so only windows
starting on or after 2023-01-03, and ending by the last prediction date, are scored.
A window run past the predictions would hold on all-NaN scores and flatter the
compiler with a quiet tail. Fills are Alpaca's :30 opens (markets.research_market).

Two rankings, because they answer different questions:
- `joint_*`: everything (FIELD + candidates + inv_vol_hold at 75%) ranked in one field,
  so cash, the 75% hold and each variant are compared in the same ranking. Variants
  crowd each other here: a near-copy takes a rank from its sibling.
- `solo_*`: each candidate ranked as the only extra entrant against FIELD
  (`windows.rank_against_field`), the way the lever sweep will score it.
  `solo_diff_vs_ivh75` is the paired per-window difference from inv_vol_hold at 75%
  scored the same way, with its SE: the number that says whether the model helps.

Writes reports/compiler_summary.csv and prints the summary.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, compiler, data, markets, ranking, windows  # noqa: E402
from icaif.compiler import Levers  # noqa: E402

OOS_START = date(2023, 1, 3)
REFERENCE = "inv_vol_hold_75"


def candidates(scores, vol) -> dict:
    c = lambda **kw: compiler.compiled(scores, vol, Levers(**kw))  # noqa: E731
    return {
        "compiled_default": c(),
        "compiled_gamma0": c(gamma=0.0),
        "compiled_exposure1": c(exposure=1.0),
        "ew_top10_no_band_no_buffer": c(weighting="equal", band=0.0, buffer=0),
        # Isolates what the band and the buffer buy: the default in every other lever.
        "compiled_no_band_no_buffer": c(band=0.0, buffer=0),
        REFERENCE: baselines.scaled(baselines.InverseVolHold, 0.75),
    }


def joint_rank(results: pd.DataFrame) -> pd.DataFrame:
    metrics = list(ranking.METRICS)
    rows = []
    for w, g in results.groupby("window"):
        m = g.set_index("strategy")
        r = ranking.rank_window(m[metrics])
        rows.append(pd.concat([m[metrics + ["invalid_rounds"]], r], axis=1)
                    .reset_index().assign(window=w))
    return pd.concat(rows, ignore_index=True)


def paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float]:
    """Mean and SE of a's per-window overall score minus b's (negative: a is better)."""
    d = (a.set_index("window")["overall_score"] - b.set_index("window")["overall_score"]).dropna()
    return float(d.mean()), float(d.std() / len(d) ** 0.5)


def main() -> None:
    t0 = time.time()
    market = markets.research_market()
    scores = compiler.load_daily_scores()
    vol = compiler.trailing_daily_vol()
    last_pred = scores.frame.index.max().date()
    starts = [s for s in windows.window_starts(market) if s >= OOS_START]
    days = market.days
    starts = [s for s in starts
              if days[days.index(s) + windows.WINDOW_DAYS - 1] <= last_pred]

    field = windows.run_field(baselines.FIELD, market, starts)
    cands = {name: windows.run_field({name: f}, market, starts)
             for name, f in candidates(scores, vol).items()}

    joint = joint_rank(pd.concat([field, *cands.values()], ignore_index=True))
    summary = windows.summarise(joint).add_prefix("joint_")
    solo = {name: windows.rank_against_field(r, field) for name, r in cands.items()}
    for name, s in solo.items():
        summary.loc[name, "solo_score"] = s["overall_score"].mean()
        summary.loc[name, "solo_se"] = s["overall_score"].std() / len(s) ** 0.5
        summary.loc[name, "solo_diff_vs_ivh75"], summary.loc[name, "solo_diff_se"] = paired(
            s, solo[REFERENCE])
    by_name = {n: g for n, g in joint.groupby("strategy")}
    for name in summary.index:
        summary.loc[name, "joint_diff_vs_cash"], summary.loc[name, "joint_diff_vs_cash_se"] = paired(
            by_name[name], by_name["cash"])

    out = data.ROOT / "reports"
    out.mkdir(exist_ok=True)
    summary.index.name = "strategy"
    summary.to_csv(out / "compiler_summary.csv")

    print(json.dumps({"windows": len(starts), "first": str(starts[0]), "last": str(starts[-1]),
                      "last_prediction": str(last_pred), "seconds": round(time.time() - t0, 1)},
                     indent=1))
    show = pd.DataFrame({
        "score": summary["joint_mean_overall_score"],
        "se": summary["joint_se_overall_score"],
        "ret%": summary["joint_mean_return"] * 100,
        "sharpe": summary["joint_mean_sharpe"],
        "mdd%": summary["joint_mean_mdd"] * 100,
        "turn%": summary["joint_mean_turnover"] * 100,
        "d_cash": summary["joint_diff_vs_cash"],
        "d_cash_se": summary["joint_diff_vs_cash_se"],
        "solo": summary["solo_score"],
        "solo_d_ivh75": summary["solo_diff_vs_ivh75"],
        "solo_d_se": summary["solo_diff_se"],
    })
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
