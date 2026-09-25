"""Labels: alphabt-features' composite path score, on a horizon of rounds, not a quarter.

For a decision executing at round i and a horizon of h rounds, the path is the fill
price at rounds i+1 .. i+h. We sample only fills, not every bar, because a strategy can
only trade at a round, so a peak between rounds cannot be captured. Components, as in
`alphabt-features/src/targets.py::_compute_raw_metrics`:

- reward_to_risk = max upside / (max drawdown + floor)                       weight 0.4
- terminal       = return from entry to round i+h                            weight 0.3
- path_sharpe    = (mean of the top-k path prices / entry - 1) / step vol    weight 0.3

Each component is ranked across the 30 names at the decision, the weighted ranks are
summed, and the top 30% is label 1.

Two constants are rescaled from the quarterly original, and both choices are
deliberate. The quarterly `+0.01` drawdown floor is small beside a quarter's ~10% moves
but larger than a typical 1-day move, so left alone it flattens every short-horizon
reward_to_risk towards upside / 1%. It scales by sqrt(horizon / 63 sessions). The top-5
of ~63 closes becomes the top 8% of the path, at least 1.

Entry is the round's own fill: the organizer-grid guess before 2026 (median error 11 bp,
small beside a 1-3 day move) and the exact public open after.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZONS = {"h7": 7, "h21": 21}
WEIGHTS = {"reward_to_risk": 0.4, "terminal": 0.3, "path_sharpe": 0.3}
TOP_FRAC = 0.30
QUARTER_SESSIONS = 63
ROUNDS_PER_SESSION = 7


def _floor(h: int) -> float:
    return 0.01 * np.sqrt((h / ROUNDS_PER_SESSION) / QUARTER_SESSIONS)


def components(exec_prices: pd.DataFrame, h: int) -> dict[str, pd.DataFrame]:
    """Raw components at every round with a full h-round path ahead (round x ticker)."""
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
            "reward_to_risk": upside / (drawdown + _floor(h)),
            "terminal": rel[:, :, -1] - 1,
            "path_sharpe": top_k / (step.std(axis=2) + 1e-10),
        }
    idx = exec_prices.sort_index().index[:n]
    return {name: pd.DataFrame(v, index=idx, columns=exec_prices.columns)
            for name, v in out.items()}


def build(exec_prices: pd.DataFrame) -> pd.DataFrame:
    """Labels for every horizon: index (execution, ticker).

    Columns per horizon: `<h>_score` (composite rank score), `<h>_label` (top 30% = 1),
    `<h>_end` (the round the label's path ends at, for purging folds). A ticker with any
    missing price on the path gets no label, rather than a label computed on a hole.
    """
    ordered = exec_prices.sort_index()
    frames = []
    for name, h in HORIZONS.items():
        comps = components(ordered, h)
        idx = comps["terminal"].index
        valid = ~np.isnan(sliding_window_view(ordered.to_numpy(float), h + 1, axis=0)[:len(idx)]
                          ).any(axis=2)
        valid = pd.DataFrame(valid, index=idx, columns=ordered.columns)
        score = sum(w * comps[c].where(valid).rank(axis=1, pct=True) for c, w in WEIGHTS.items())
        # Ties at the cut are broken by column order (tickers sorted), so each decision
        # labels exactly round(30 x 0.3) = 9 names. With average ranks, a tie straddling
        # the cut labels 10, and the base rate drifts with how often scores tie.
        n = score.notna().sum(axis=1)
        k = (n * TOP_FRAC).round()
        label = (score.rank(axis=1, method="first", ascending=False)
                 .le(k, axis=0)).astype(float).where(score.notna())
        end = pd.Series(ordered.index[h:h + len(idx)], index=idx)
        f = pd.concat({f"{name}_score": score.stack(future_stack=True),
                       f"{name}_label": label.stack(future_stack=True),
                       f"{name}_terminal": comps["terminal"].where(valid).stack(future_stack=True)},
                      axis=1)
        f.index.names = ["execution", "ticker"]
        f[f"{name}_end"] = f.index.get_level_values("execution").map(end)
        frames.append(f)
    return pd.concat(frames, axis=1).sort_index()
