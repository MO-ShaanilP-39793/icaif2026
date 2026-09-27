"""Labels: two ranking targets on a horizon of rounds, not a quarter.

**Composite** (every horizon): alphabt-features' path score, below.
**Upside on fills** (3 and 5 days): alphaBT's Target 2, rescaled. The mean of the top-k
fills on the path, against entry. Target 1 (the same on bar highs) is left out on
purpose: that upside is only capturable with a resting limit order, and here the only
prices we can trade at are round opens.

Each target comes in two forms. `_pct` is the cross-sectional percentile in [0, 1], the
regression target: bounded, so no single outlier dominates, and it keeps the ordering
a binary cut discards. `_label` is 1 for the top 40% (alphaBT's cut), run as an ablation.

For a decision executing at round i and a horizon of h rounds, the path is the fill
price at rounds i+1 .. i+h. We sample only fills, not every bar, because a strategy can
only trade at a round, so a peak between rounds cannot be captured. Components, as in
`alphabt-features/src/targets.py::_compute_raw_metrics`:

- reward_to_risk = max upside / (max drawdown + floor)                       weight 0.4
- terminal       = return from entry to round i+h                            weight 0.3
- path_sharpe    = (mean of the top-k path prices / entry - 1) / step vol    weight 0.3

Each component is ranked across the 30 names at the decision and the weighted ranks are
summed into the composite.

Two constants are rescaled from the quarterly original, and both choices are
deliberate. The quarterly `+0.01` drawdown floor is small beside a quarter's ~10% moves
but larger than a typical 1-day move, so left alone it flattens every short-horizon
reward_to_risk towards upside / 1%. It scales by sqrt(horizon / 63 sessions). The top-5
of ~63 closes becomes the top 8% of the path, at least 1.

Entry is the round's own fill: the organizer-grid guess before Nov 2023 (median error
11 bp, small beside a multi-day move) and the exact public open after
(`markets.label_exec_prices`).
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZONS = {"h7": 7, "h21": 21, "h35": 35}
UPSIDE_HORIZONS = ("h21", "h35")
# The daily model decides once a day, at round 1, so its path is the 09:30 open of
# each following session: 3 and 5 steps, one per session.
DAILY_HORIZONS = {"d3": 3, "d5": 5}
WEIGHTS = {"reward_to_risk": 0.4, "terminal": 0.3, "path_sharpe": 0.3}
TOP_FRAC = 0.40
QUARTER_SESSIONS = 63
ROUNDS_PER_SESSION = 7


def _floor(h: int, per_session: int = ROUNDS_PER_SESSION) -> float:
    return 0.01 * np.sqrt((h / per_session) / QUARTER_SESSIONS)


def components(exec_prices: pd.DataFrame, h: int,
               per_session: int = ROUNDS_PER_SESSION) -> dict[str, pd.DataFrame]:
    """Raw components at every step with a full h-step path ahead (step x ticker).

    A step is a round (7 a session) for the intraday model and a session for the daily
    model; `per_session` rescales the drawdown floor to the same calendar span.
    """
    px = exec_prices.sort_index().to_numpy(dtype=float)
    n = px.shape[0] - h
    if n <= 0:
        raise ValueError(f"need more than {h} rounds of prices")
    entry = px[:n]
    # windows[i] = rounds i .. i+h, shape (n, tickers, h+1)
    windows = sliding_window_view(px, h + 1, axis=0)[:n]
    path = windows[:, :, 1:]
    rel = path / entry[:, :, None]
    upside = rel.max(axis=2) - 1
    drawdown = 1 - rel.min(axis=2)
    step = np.diff(np.log(windows), axis=2)
    k = max(1, int(round(0.08 * h)))
    top_k = np.sort(rel, axis=2)[:, :, -k:].mean(axis=2) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "reward_to_risk": upside / (drawdown + _floor(h, per_session)),
            "terminal": rel[:, :, -1] - 1,
            "path_sharpe": top_k / (step.std(axis=2) + 1e-10),
            "upside": top_k,
        }
    idx = exec_prices.sort_index().index[:n]
    return {name: pd.DataFrame(v, index=idx, columns=exec_prices.columns)
            for name, v in out.items()}


def unit_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile scaled to exactly [0, 1] (worst 0, best 1)."""
    r = panel.rank(axis=1)
    n = panel.notna().sum(axis=1)
    return r.sub(1).div((n - 1).where(n > 1), axis=0)


def top_label(score: pd.DataFrame) -> pd.DataFrame:
    """1 for the top 40% at each decision, exactly round(n x 0.4) names.

    Ties at the cut are broken by column order (tickers sorted). With average ranks, a
    tie straddling the cut labels one name too many, and the base rate drifts with how
    often scores tie.
    """
    n = score.notna().sum(axis=1)
    k = (n * TOP_FRAC).round()
    return (score.rank(axis=1, method="first", ascending=False)
            .le(k, axis=0)).astype(float).where(score.notna())


def build(exec_prices: pd.DataFrame, horizons: dict[str, int] = HORIZONS,
          upside_horizons: tuple[str, ...] = UPSIDE_HORIZONS,
          per_session: int = ROUNDS_PER_SESSION,
          universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Labels for every horizon: index (execution, ticker).

    Columns per horizon: `<h>_pct` and `<h>_label` (composite), `<h>_terminal` (plain
    forward return, a reference), `<h>_end` (the round the path ends at, for purging
    folds); for 3 and 5 days also `<h>_up_pct` and `<h>_up_label` (upside on fills).
    A ticker with any missing price on the path gets no label, rather than a label
    computed on a hole.

    `universe` (step x ticker, bool) restricts who is ranked at each step: the daily
    model's cross-section is that day's universe. Paths still read every price, so a
    name that leaves the universe mid-path keeps its label.
    """
    ordered = exec_prices.sort_index()
    frames = []
    for name, h in horizons.items():
        comps = components(ordered, h, per_session)
        idx = comps["terminal"].index
        valid = ~np.isnan(sliding_window_view(ordered.to_numpy(float), h + 1, axis=0)[:len(idx)]
                          ).any(axis=2)
        valid = pd.DataFrame(valid, index=idx, columns=ordered.columns)
        if universe is not None:
            valid &= universe.reindex(index=idx, columns=ordered.columns).fillna(False).astype(bool)
        score = sum(w * comps[c].where(valid).rank(axis=1, pct=True) for c, w in WEIGHTS.items())
        cols = {f"{name}_pct": unit_rank(score),
                f"{name}_label": top_label(score),
                f"{name}_terminal": comps["terminal"].where(valid)}
        if name in upside_horizons:
            up = comps["upside"].where(valid)
            cols[f"{name}_up_pct"] = unit_rank(up)
            cols[f"{name}_up_label"] = top_label(up)
        f = pd.concat({c: v.stack(future_stack=True) for c, v in cols.items()}, axis=1)
        f.index.names = ["execution", "ticker"]
        end = pd.Series(ordered.index[h:h + len(idx)], index=idx)
        f[f"{name}_end"] = f.index.get_level_values("execution").map(end)
        frames.append(f)
    return pd.concat(frames, axis=1).sort_index()
