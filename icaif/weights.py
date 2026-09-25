"""The last step before a decision leaves our hands: weights the backend will accept.

The backend validates `Decimal(str(w))`, so the float a computation produces is what it
checks. `0.1 + 0.2` is `0.30000000000000004`, over the 0.30 cap, and one such weight makes
the whole decision invalid -- the round holds, silently, at whatever the book was.
Rounding *down* to a fixed grid keeps every weight and the total on the safe side.
"""

import math

CAP = 0.30
GRID = 1e-6


def safe(weights: dict, tickers: list[str]) -> dict:
    """All `tickers` present, each floored to the grid, capped, total <= 1."""
    w = {t: max(0.0, min(CAP, float(weights.get(t, 0.0)))) for t in tickers}
    total = sum(w.values())
    if total > 1.0:
        w = {t: v / total for t, v in w.items()}
    # Floor after scaling: the cap and the total both survive a floor, never a round-up.
    return {t: math.floor(v / GRID) * GRID for t, v in w.items()}


def floor_to_grid(x: float) -> float:
    return math.floor(x / GRID) * GRID
