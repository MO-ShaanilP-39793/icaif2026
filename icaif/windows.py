"""Competition-shaped windows: every strategy, fresh $1M, 15 trading days, ranked.

The official phase is one 15-day window, so the honest unit of evaluation is the
distribution over many such windows, not one multi-year equity curve. A 2.8-year
Sharpe says little about a 15-day rank.
"""

from datetime import date

import numpy as np
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


def rank_against_field(candidate_results: pd.DataFrame, field_results: pd.DataFrame) -> pd.DataFrame:
    """Rank ONE candidate in each window against a fixed, precomputed field.

    `candidate_results`: one row per window (a `window` column and the four metrics),
    e.g. `run_field({name: factory}, ...)`. `field_results`: `run_field(FIELD, ...)`.
    Returns rows shaped like `run_field`'s, so `summarise` reads them.

    The candidate enters each window as the only extra entrant, exactly as
    `ranking.rank_window` would rank the field plus it, but without re-simulating the
    field or ranking the candidates of a sweep against each other. That matters: in a
    joint ranking, near-copies from the same sweep crowd each other's ranks, and the
    best lever setting would be the one least like its neighbours rather than the one
    that beats the field. A window absent from the field is dropped, never scored
    against an empty field (where every candidate would place first).
    """
    metrics = list(ranking.METRICS)
    cand = candidate_results.set_index("window")
    rows = []
    for w, f in field_results.groupby("window", sort=True):
        if w not in cand.index:
            continue
        c = cand.loc[w]
        fm = f[metrics].to_numpy(dtype=float)
        cm = c[metrics].to_numpy(dtype=float)
        # Average ranks with one extra entrant: 1 + (field strictly better) + half the
        # field tied with it. Each field member moves down by the mirror image.
        cand_rank = np.empty(len(metrics))
        field_rank = np.empty_like(fm)
        for j, (m, ascending) in enumerate(ranking.METRICS.items()):
            better = fm[:, j] < cm[j] if ascending else fm[:, j] > cm[j]
            equal = fm[:, j] == cm[j]
            cand_rank[j] = 1 + better.sum() + 0.5 * equal.sum()
            own = pd.Series(fm[:, j]).rank(ascending=ascending, method="average").to_numpy()
            field_rank[:, j] = own + (~better & ~equal) + 0.5 * equal
        overall = cand_rank.mean()
        field_overall = field_rank.mean(axis=1)
        key = (overall, -cm[0], -cm[1], cm[2], cm[3])
        ahead = sum((fo, -r[0], -r[1], r[2], r[3]) < key for fo, r in zip(field_overall, fm))
        name = c["strategy"] if "strategy" in cand.columns else "candidate"
        rows.append({"window": w, "strategy": name,
                     **{k: c[k] for k in cand.columns
                        if k != "strategy" and not k.startswith("rank_")
                        and k not in ("overall_score", "position")},
                     **{f"rank_{m}": r for m, r in zip(metrics, cand_rank)},
                     "overall_score": overall, "position": 1 + int(ahead)})
    return pd.DataFrame(rows)
