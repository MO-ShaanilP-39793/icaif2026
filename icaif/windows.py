"""Competition-shaped windows: every strategy, fresh $1M, 15 trading days, ranked.

The official phase is one 15-day window, so the honest unit of evaluation is the
distribution over many such windows, not one multi-year equity curve. A 2.8-year
Sharpe says little about a 15-day rank.
"""

from datetime import date

import pandas as pd

from icaif import ranking, sim

WINDOW_DAYS = 15


def window_starts(market: sim.Market, n_days: int = WINDOW_DAYS, stride: int = WINDOW_DAYS,
                  skip_degraded: bool = True) -> list[date]:
    """Start days of windows that fit in the market, skipping any that touch a degraded day.

    A degraded day carries stand-in prices (a return of zero that never happened, then a
    catch-up), so a window across one looks calmer than the market was.
    """
    degraded = set(market.issues.get("degraded_days", [])) if skip_degraded else set()
    days = market.days
    starts = []
    for i in range(0, len(days) - n_days + 1, stride):
        span = days[i:i + n_days]
        if not degraded & {str(d) for d in span}:
            starts.append(span[0])
    return starts


def run_field(field: dict, market: sim.Market, starts: list[date],
              n_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """One row per (window, strategy): the four metrics, the ranks, invalid-round count."""
    rows = []
    for start in starts:
        window = {}
        for name, factory in field.items():
            res = sim.run(factory(), market, start, n_days)
            window[name] = {**res.metrics(), "invalid_rounds": len(res.invalid_rounds)}
        metrics = pd.DataFrame(window).T
        ranks = ranking.rank_window(metrics[list(ranking.METRICS)])
        for name in field:
            rows.append({"window": start, "strategy": name,
                         **metrics.loc[name].to_dict(), **ranks.loc[name].to_dict()})
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Per strategy, across windows, sorted by the objective (mean overall score)."""
    g = results.groupby("strategy")
    out = pd.DataFrame({
        "mean_overall_score": g["overall_score"].mean(),
        # The measurement floor: two strategies closer than ~2 SE apart are not ordered
        # by ~40 windows, however the means look.
        "se_overall_score": g["overall_score"].std() / g["overall_score"].count() ** 0.5,
        "mean_position": g["position"].mean(),
        "wins": g["position"].apply(lambda s: int((s == 1).sum())),
        "mean_return": g["cumulative_return"].mean(),
        "mean_sharpe": g["sharpe_ratio"].mean(),
        "mean_mdd": g["maximum_drawdown"].mean(),
        "mean_turnover": g["turnover"].mean(),
        "invalid_rounds": g["invalid_rounds"].sum(),
    })
    return out.sort_values("mean_overall_score")
