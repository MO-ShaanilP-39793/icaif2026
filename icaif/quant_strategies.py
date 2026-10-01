"""The quant models of `quant.py` as simulator strategies, one decision a day at round 1.

Each is a factory like `baselines`: a fresh book per window, so no state leaks from one
window into the next. They all decide at round 1 and hold through rounds 2-7, because
every model here reads daily data and an hourly re-decision on the same inputs is
turnover bought for nothing.

The book has a *shape* (inverse-vol, minimum variance or risk parity, fixed on the
first day) and an *exposure* set daily by a policy (fixed, vol target, drawdown
control, HMM regime, or the most cautious of several). Exposure changes smaller than
the band do not trade; a change that does trade rescales the drifted book rather than
resetting it to the first day's shape, so the dial costs |Δexposure| of turnover and
nothing more.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from icaif import quant
from icaif import weights as W

HISTORY_DAYS = 750  # ~3 years for the regime and reference-vol fits
SHAPE_DAYS = 60     # covariance / vol window for the book's shape


def daily_closes(ctx, n_days: int) -> pd.DataFrame:
    """The last `n_days` session closes the decision may see, from its bars only.

    Built once per market from the bar panel and cut at the deadline by the timestamp
    of each day's last bar, so a round-1 decision never sees the session it trades in.

    A session still trading has no close yet. In a backtest each day's last bar ends at
    the close, after all of that day's deadlines, so the cut alone suffices. A live
    market holds only the bars fetched so far, and today's latest bar would pass the
    cut as today's "close": every daily return and the regime fit would carry a partial
    day, and the event trigger (a move since the prior close) would measure today's
    price against itself and never fire.
    """
    m = ctx.market
    if not hasattr(m, "_quant_daily"):
        bars = m.recent_closes(pd.Timestamp("2100-01-01", tz="UTC"), len(m.info_bars))
        last = bars.groupby(bars.index.date).tail(1)
        m._quant_daily = last
        m._quant_daily_ends = pd.DatetimeIndex(last.index).asi8
    deadline = pd.Timestamp(ctx.deadline)
    k = np.searchsorted(m._quant_daily_ends, deadline.value, side="right")
    if k and m._quant_daily.index[k - 1].date() >= deadline.tz_convert(m._quant_daily.index.tz).date():
        k -= 1
    return m._quant_daily.iloc[max(0, k - n_days):k]


def _log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    return np.log(closes).diff().iloc[1:]


def _basket(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight basket of the 30. NaN on a day any member is missing, since a mean
    over whoever happens to be present is a different basket."""
    return returns.mean(axis=1).where(returns.notna().all(axis=1))


# ----------------------------------------------------------------------------- shapes

def shape_inverse_vol(returns: pd.DataFrame) -> Optional[pd.Series]:
    vol = returns.tail(20).std()
    if vol.isna().any() or (vol <= 0).any():
        return None
    inv = 1.0 / vol
    return inv / inv.sum()


def shape_min_variance(returns: pd.DataFrame) -> Optional[pd.Series]:
    cov = quant.shrunk_cov(returns)
    if cov is None or len(cov) < len(returns.columns):
        return None
    return quant.min_variance(cov, 1.0, cap=W.CAP)


def shape_risk_parity(returns: pd.DataFrame) -> Optional[pd.Series]:
    cov = quant.shrunk_cov(returns)
    if cov is None or len(cov) < len(returns.columns):
        return None
    return quant.risk_parity(cov, 1.0, cap=W.CAP)


def shape_blend(alpha: float):
    """alpha x risk parity + (1 - alpha) x inverse-vol, both fully invested.

    The two shapes win against different fields (risk parity against the near-hold
    field, inverse-vol against the active one), and the real field is unobservable
    until the end; a blend is the hedge between them. Both parts are capped at 30%
    per name, so their mix is too.
    """
    def shape(returns):
        rp, iv = shape_risk_parity(returns), shape_inverse_vol(returns)
        if rp is None or iv is None:
            return None
        return alpha * rp + (1 - alpha) * iv.reindex(rp.index)
    return shape


SHAPES = {"inverse_vol": shape_inverse_vol, "min_variance": shape_min_variance,
          "risk_parity": shape_risk_parity, "blend25": shape_blend(0.25),
          "blend50": shape_blend(0.50), "blend75": shape_blend(0.75)}


# ----------------------------------------------------------------------------- exposure policies

class Fixed:
    def __init__(self, e: float = 0.75):
        self.e = e

    def __call__(self, basket: pd.Series, nav: list[float]) -> float:
        return self.e


class VolTarget:
    """e = e_ref x (median EWMA vol over 3y) / (today's EWMA vol), clipped.

    The reference is the basket's own typical vol, so the policy holds e_ref in an
    ordinary market and only leans when vol is unusual; a fixed annual target would
    hold a different exposure in every era.
    """

    def __init__(self, e_ref=0.75, e_min=0.25, e_max=0.95, lam=0.94):
        self.e_ref, self.e_min, self.e_max, self.lam = e_ref, e_min, e_max, lam

    def __call__(self, basket, nav):
        r = basket.dropna()
        if len(r) < 60:
            return self.e_ref
        vol = np.sqrt((r ** 2).ewm(alpha=1 - self.lam).mean())
        return quant.vol_target_exposure(float(vol.iloc[-1]), float(vol.median()),
                                         self.e_ref, self.e_min, self.e_max)


class Drawdown:
    """Grossman–Zhou on the book's own NAV: e_max at a peak, zero at the floor."""

    def __init__(self, e_max=0.75, dd_limit=0.04):
        self.e_max, self.dd_limit = e_max, dd_limit

    def __call__(self, basket, nav):
        return quant.drawdown_exposure(np.array(nav), self.e_max, self.dd_limit)


class Regime:
    """Exposure blended by tomorrow's probability of the turbulent HMM state.

    The model is fit once per window on the three years before it, never refit inside
    it, so a window's regime calls use parameters that could have existed at its start.
    """

    def __init__(self, e_calm=0.85, e_turbulent=0.30):
        self.e_calm, self.e_turb = e_calm, e_turbulent
        self.model = None
        self.last_p = np.nan

    def __call__(self, basket, nav):
        r = basket.dropna().to_numpy()
        if self.model is None:
            if len(r) < 250:
                return 0.5 * (self.e_calm + self.e_turb)
            self.model = quant.fit_hmm2(r)
        p = quant.hmm_next_turbulent(r[-250:], self.model)
        self.last_p = p
        return float(self.e_calm * (1 - p) + self.e_turb * p)


class Cautious:
    """The lowest exposure any of its policies asks for."""

    def __init__(self, *policies):
        self.policies = policies

    def __call__(self, basket, nav):
        return min(p(basket, nav) for p in self.policies)


# ----------------------------------------------------------------------------- the strategy

class QuantBook:
    """A shape held from day one, sized each morning by an exposure policy."""

    def __init__(self, shape: str = "inverse_vol", policy: Optional[Callable] = None,
                 band: float = 0.03):
        self.shape_fn = SHAPES[shape]
        self.policy = policy or Fixed()
        self.band = band
        self.shape: Optional[pd.Series] = None
        self.nav: list[float] = []
        self.log: list[dict] = []

    def _book_value(self, ctx, tickers):
        shares = pd.Series(ctx.shares, dtype=float).reindex(tickers).fillna(0.0)
        last = ctx.recent_closes(1)
        px = last.iloc[-1].reindex(tickers) if len(last) else pd.Series(np.nan, index=tickers)
        value = shares * px
        return value, ctx.cash + float(value.sum())

    def __call__(self, ctx):
        if ctx.round != 1:
            return None
        tickers = ctx.market.tickers
        value, nav = self._book_value(ctx, tickers)
        if not np.isfinite(nav) or nav <= 0:
            return None  # cannot value the book; holding beats guessing
        self.nav.append(nav)
        closes = daily_closes(ctx, HISTORY_DAYS + 1)
        rets = _log_returns(closes)
        if self.shape is None:
            shape = self.shape_fn(rets.tail(SHAPE_DAYS))
            if shape is None:
                return None  # not enough history yet: cash until there is
            self.shape = shape.reindex(tickers).fillna(0.0)
        e = float(np.clip(self.policy(_basket(rets), self.nav), 0.0, 1.0))
        gross = float(value.sum()) / nav
        self.log.append({"day": ctx.day, "exposure": e, "gross": gross})
        if gross <= W.GRID:
            if e <= W.GRID:
                return None
            target = self.shape * e
        else:
            if abs(e - gross) <= self.band:
                return None
            current = value / nav
            target = current * (e / gross)
        return W.safe(target.clip(upper=W.CAP).to_dict(), tickers)


class OUTilt:
    """Inverse-vol book tilted toward names stretched below their residual OU mean.

    Weight x clip(1 - k s, 0, 2): s = -2 doubles a name, s = +1 at k = 1 drops it. Names
    with no reversion inside the window (s NaN) keep their inverse-vol weight. Re-tilted
    every `every` sessions, and only when some name would move by more than the band.
    """

    def __init__(self, exposure=0.75, k=0.5, every=5, lookback=60, band=0.01):
        self.exposure, self.k, self.every, self.lookback, self.band = (
            exposure, k, every, lookback, band)
        self.session = -1

    def __call__(self, ctx):
        if ctx.round != 1:
            return None
        self.session += 1
        if self.session % self.every:
            return None
        tickers = ctx.market.tickers
        rets = _log_returns(daily_closes(ctx, self.lookback + 1))
        base = shape_inverse_vol(rets)
        if base is None:
            return None
        s = quant.s_scores(rets)["s"].reindex(tickers).astype(float)
        tilt = (1 - self.k * s.fillna(0.0)).clip(0.0, 2.0)
        w = base.reindex(tickers).fillna(0.0) * tilt
        if w.sum() <= 0:
            return None
        from icaif.compiler import _water_fill
        target = pd.Series(_water_fill(w.to_numpy(), self.exposure, W.CAP), index=tickers)
        shares = pd.Series(ctx.shares, dtype=float).reindex(tickers).fillna(0.0)
        last = ctx.recent_closes(1)
        if len(last) and (shares > 0).any():
            value = shares * last.iloc[-1].reindex(tickers)
            current = value / (ctx.cash + float(value.sum()))
            if float((target - current).abs().max()) <= self.band:
                return None
        return W.safe(target.to_dict(), tickers)


def book(shape="inverse_vol", policy_factory=Fixed, band=0.03):
    """A `windows.run_field` factory: a fresh book and a fresh policy per window."""
    return lambda: QuantBook(shape, policy_factory(), band)


ENTRY_ONLY = 2.0  # a band no exposure change can cross: decide at entry, then hold


def _cautious():
    return Cautious(VolTarget(), Drawdown(), Regime())


CANDIDATES = {
    # Shape, at the hold's own 75%, bought once.
    "q_invvol_fixed75": book("inverse_vol", Fixed),
    "q_minvar_fixed75": book("min_variance", Fixed),
    "q_riskparity_fixed75": book("risk_parity", Fixed),
    # The exposure dial, moved daily past a 3-point band.
    "q_invvol_voltarget": book("inverse_vol", VolTarget),
    "q_invvol_drawdown": book("inverse_vol", Drawdown),
    "q_invvol_regime": book("inverse_vol", Regime),
    "q_invvol_cautious": book("inverse_vol", _cautious),
    "q_minvar_cautious": book("min_variance", _cautious),
    # The same dials read once, at entry, then held.
    "q_riskparity_fixed90": book("risk_parity", lambda: Fixed(0.90), ENTRY_ONLY),
    "q_riskparity_entry_regime": book("risk_parity", Regime, ENTRY_ONLY),
    "q_invvol_entry_regime": book("inverse_vol", Regime, ENTRY_ONLY),
    "q_riskparity_entry_voltarget": book("risk_parity", VolTarget, ENTRY_ONLY),
    # Residual mean reversion.
    "q_ou_tilt": lambda: OUTilt(),
}
