"""Reference strategies, and the synthetic field our strategies are ranked against.

The competition scores a *rank* across four metrics, so a strategy is only good
relative to its rivals. We cannot see the real field, so we stand in plausible ones:
- the do-nothing corners: cash, buy-and-hold;
- the kit's own example agent, which many teams will start from;
- the kind of agent we expect most entries to be, one that trades every hour.

Each entry is a *factory*: every window starts from a fresh $1M book, and a strategy
that kept state across windows would carry one window's position into the next.
"""

import numpy as np

from icaif import weights as W

ROUNDS_PER_DAY = 7


def _ew(tickers):
    return W.safe({t: 1 / len(tickers) for t in tickers}, tickers)


class Cash:
    """Never trades: zero turnover, zero drawdown, zero return, zero Sharpe."""

    def __call__(self, ctx):
        return None


class EqualWeightHold:
    """Buy 1/30 of everything at the first round, then never trade."""

    def __init__(self):
        self.done = False

    def __call__(self, ctx):
        if self.done:
            return None
        self.done = True
        return _ew(ctx.market.tickers)


class EqualWeightDaily:
    """Rebalance to 1/30 each at every round 1."""

    def __call__(self, ctx):
        return _ew(ctx.market.tickers) if ctx.round == 1 else None


class InverseVolHold:
    """Weights ~ 1 / realised vol over the last ~20 days of bars, bought once and held."""

    def __init__(self, lookback_bars: int = 20 * ROUNDS_PER_DAY):
        self.lookback = lookback_bars
        self.done = False

    def __call__(self, ctx):
        if self.done:
            return None
        closes = ctx.recent_closes(self.lookback + 1)
        vol = np.log(closes).diff().std()
        if vol.isna().any() or len(closes) < self.lookback // 2:
            return None  # not enough history yet; hold cash until there is
        self.done = True
        inv = 1.0 / vol
        return W.safe((inv / inv.sum()).to_dict(), ctx.market.tickers)


class KitMomentum:
    """The starter kit's rule-based example, run every round (`agents/rule_based_agent.py`):
    the top 5 positive 5-interval momentum names at 20% each; no signal means cash."""

    def __call__(self, ctx):
        closes = ctx.recent_closes(6)
        if len(closes) < 6:
            return None
        score = closes.iloc[-1] / closes.iloc[0] - 1
        top = score[score > 0].sort_values(ascending=False, kind="stable").head(5)
        return W.safe({t: 0.20 for t in top.index}, ctx.market.tickers)


class MomentumDaily:
    """Top 5 by prior-day return, 20% each, rebalanced at round 1 only."""

    def __call__(self, ctx):
        if ctx.round != 1:
            return None
        closes = ctx.recent_closes(ROUNDS_PER_DAY + 1)
        if len(closes) < ROUNDS_PER_DAY + 1:
            return None
        score = closes.iloc[-1] / closes.iloc[0] - 1
        top = score.sort_values(ascending=False, kind="stable").head(5)
        return W.safe({t: 0.20 for t in top.index}, ctx.market.tickers)


class RandomChurn:
    """A stand-in for a noisy hourly LLM agent: fully invested, and every round moves 20%
    of its target toward a fresh random portfolio. Seeded so a field is reproducible."""

    def __init__(self, seed: int = 0, step: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.step = step
        self.target = None

    def __call__(self, ctx):
        n = len(ctx.market.tickers)
        fresh = np.minimum(self.rng.dirichlet(np.ones(n)), W.CAP)
        fresh = fresh / fresh.sum()
        self.target = fresh if self.target is None else (
            (1 - self.step) * self.target + self.step * fresh)
        return W.safe(dict(zip(ctx.market.tickers, self.target)), ctx.market.tickers)


class ConcentratedHold:
    """30% in each of the 3 strongest names over the prior ~5 days, bought once: the
    high-variance corner of the field."""

    def __init__(self):
        self.done = False

    def __call__(self, ctx):
        if self.done:
            return None
        closes = ctx.recent_closes(5 * ROUNDS_PER_DAY + 1)
        if len(closes) < 2:
            return None
        self.done = True
        top = (closes.iloc[-1] / closes.iloc[0] - 1).sort_values(ascending=False).head(3)
        return W.safe({t: 0.30 for t in top.index}, ctx.market.tickers)


class Scaled:
    """Any strategy with its weights scaled by `gross`: the exposure dial on its own.

    Scaling leaves Sharpe roughly unchanged and shrinks return, drawdown and turnover
    together, so it trades the return rank against the drawdown and turnover ranks.
    """

    def __init__(self, inner, gross: float):
        self.inner, self.gross = inner, gross

    def __call__(self, ctx):
        w = self.inner(ctx)
        if w is None:
            return None
        return W.safe({t: v * self.gross for t, v in w.items()}, ctx.market.tickers)


def scaled(factory, gross: float):
    return lambda: Scaled(factory(), gross)


FIELD = {
    "cash": Cash,
    "ew_hold": EqualWeightHold,
    "ew_daily": EqualWeightDaily,
    "inv_vol_hold": InverseVolHold,
    "kit_momentum_hourly": KitMomentum,
    "momentum_daily": MomentumDaily,
    "random_churn": RandomChurn,
    "concentrated_hold": ConcentratedHold,
}
