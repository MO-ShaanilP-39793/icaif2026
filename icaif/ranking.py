"""The competition's ranking, applied to one window's field (`docs/evaluation.md`).

Per metric: return and Sharpe rank high-to-low, drawdown and turnover low-to-high, and
equal values share the average of the ranks they occupy. The Overall Rank Score is the
mean of the four ranks (lower is better). Ties on it break by higher return, higher
Sharpe, lower drawdown, then lower turnover.

The objective we optimise is a strategy's mean Overall Rank Score across windows. It is
deliberately not Sharpe: a high-Sharpe strategy that churns can lose two of the four
ranks to a buy-and-hold, and a Sharpe objective would never notice.
"""

import pandas as pd

METRICS = {  # metric -> ascending? (True = lower is better)
    "cumulative_return": False,
    "sharpe_ratio": False,
    "maximum_drawdown": True,
    "turnover": True,
}


def rank_window(metrics: pd.DataFrame) -> pd.DataFrame:
    """`metrics`: one row per strategy (index), the four metric columns.

    Returns per-metric ranks, `overall_score` and `position` (1 = winner).
    """
    ranks = pd.DataFrame(index=metrics.index)
    for m, ascending in METRICS.items():
        ranks[f"rank_{m}"] = metrics[m].rank(ascending=ascending, method="average")
    ranks["overall_score"] = ranks[[f"rank_{m}" for m in METRICS]].mean(axis=1)
    order = pd.concat([ranks["overall_score"], metrics], axis=1).sort_values(
        ["overall_score", "cumulative_return", "sharpe_ratio",
         "maximum_drawdown", "turnover"],
        ascending=[True, False, False, True, True])
    # Exact four-metric ties share a position (1, 1, 3), as the rules specify.
    keys = order[list(METRICS)].apply(tuple, axis=1)
    position, prev = {}, None
    for i, (name, key) in enumerate(keys.items(), start=1):
        position[name] = position[prev] if prev is not None and key == keys[prev] else i
        prev = name
    ranks["position"] = pd.Series(position)
    return ranks
