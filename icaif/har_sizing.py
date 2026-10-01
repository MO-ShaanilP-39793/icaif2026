"""Day-1 sizing from the HAR forecast (Roadmap step 3): decided once at entry, then held.

Every rule that trades after entry has lost to the hold (`tools/quant_report.py`: the
EWMA vol target, drawdown control, the HMM dialled daily), because each trade pays 0.1%
and a turnover rank. The entry is the one decision that costs no extra turnover: the
book is bought anyway. So the HAR forecast, which beats trailing vol out of sample in
every year (`tools/vol_report.py`), is tried there and only there:

- **Weights.** Inverse-vol on each name's HAR forecast, in place of the 20 days of
  trailing hourly vol `InverseVolHold` uses; and inside risk parity, HAR vols in place of
  the sample vols on the covariance's diagonal, with the shrunk sample correlations kept.
- **Exposure.** The basket's HAR forecast against its own typical level,
  e = e_ref x typical / forecast in vol, clipped, read once at entry. For the rule
  desk's book it replaces the HMM's exposure, is averaged with it, or caps it.

With HAR switched off each book is its reference trade for trade (a test checks), so a
score difference is the forecast's doing alone.

**A missing forecast falls back to the reference, and says so.** A name without one on
the entry day (a hole in its recent bars) would otherwise drop out of the book or block
the entry. The book takes the reference's trailing shape instead, and a basket without a
forecast takes the reference's exposure. Each is appended to `fallbacks`: a book that
quietly ran on the reference would score as a HAR book that happened to tie.
"""

from typing import Optional

import numpy as np
import pandas as pd

from icaif import baselines, quant, quant_strategies as qs, vol
from icaif import weights as W
from icaif.agents.signals import VolForecasts

# The hold's own gross, chosen from the exposure scan (README "Baselines"). Fixed here,
# so an exposure rule is judged on when it leans, not on a different average level.
E_REF = 0.75
MODES = ("regime", "har", "mean", "min")


def har_vols(har: VolForecasts, ctx, horizon: int) -> pd.Series:
    """Forecast daily vol per name (and `_MKT`) for the decision's own day; NaN where none."""
    return np.sqrt(har.for_day(ctx.day, ctx.deadline)[f"har_h{horizon}"])


def har_covariance(cov: pd.DataFrame, har_vol: pd.Series) -> pd.DataFrame:
    """`cov` with its vols replaced by HAR's: correlation x outer(har_vol, har_vol).

    HAR forecasts each name's variance, not how names move together, so the
    correlations stay the shrunk sample's. Risk parity is scale-free, so only the
    forecasts' relative levels move the book.
    """
    c = cov.to_numpy()
    sd = np.sqrt(np.diag(c))
    v = har_vol.reindex(cov.index).to_numpy(dtype=float)
    return pd.DataFrame(c / np.outer(sd, sd) * np.outer(v, v), index=cov.index, columns=cov.columns)


def _missing(hv: pd.Series) -> list[str]:
    return sorted(hv.index[~(hv > 0)])  # NaN compares False, so it counts as missing


class HarExposure:
    """e = e_ref x typical / forecast (in vol), clipped to [lo, hi]; None without either.

    `typical` is the median of the basket's own forecasts over the last `typical_days`
    sessions, forecasts of a mean against forecasts of a mean. Realised variance would be
    the wrong yardstick: it is right-skewed, so its median sits below any forecast of its
    mean, and every entry would lean low with nothing looking wrong. With fewer than 60%
    of those sessions forecast (`vol.MIN_WINDOW_SHARE`) a median of a few weeks would
    stand in for a typical level, so there is none.
    """

    def __init__(self, horizon: int, typical_days: int, lo: float, hi: float,
                 e_ref: float = E_REF):
        if not 0 < lo <= e_ref <= hi <= 1:
            raise ValueError(f"need 0 < lo <= e_ref <= hi <= 1, got {lo}, {e_ref}, {hi}")
        self.horizon, self.typical_days, self.lo, self.hi, self.e_ref = (
            horizon, typical_days, lo, hi, e_ref)

    def __call__(self, har: VolForecasts, day, deadline) -> Optional[float]:
        hist = har.trailing(day, deadline, self.typical_days, self.horizon)[vol.MARKET]
        if not len(hist) or hist.index[-1].date() != day:
            return None
        now = float(hist.iloc[-1])
        if not now > 0 or hist.notna().sum() < np.ceil(vol.MIN_WINDOW_SHARE * self.typical_days):
            return None
        return quant.vol_target_exposure(np.sqrt(now), float(np.sqrt(hist.median())),
                                         self.e_ref, self.lo, self.hi)

    def settings(self) -> dict:
        return {"horizon": self.horizon, "typical_days": self.typical_days,
                "lo": self.lo, "hi": self.hi, "e_ref": self.e_ref}


class HarInverseVolHold:
    """`InverseVolHold` bought once at gross e, with HAR in its weights, its gross, or both.

    `weights_h` None keeps the hold's own trailing weights; `exposure` None keeps e_ref.
    The entry is gated by `InverseVolHold` itself, so every variant enters on the
    reference's own round and only its sizing differs.
    """

    def __init__(self, har: Optional[VolForecasts], weights_h: Optional[int] = None,
                 exposure: Optional[HarExposure] = None, e_ref: float = E_REF):
        if (weights_h is not None or exposure is not None) and har is None:
            raise ValueError("HAR sizing needs the forecasts")
        self.inner = baselines.InverseVolHold()
        self.har, self.weights_h, self.exposure, self.e_ref = har, weights_h, exposure, e_ref
        self.fallbacks: list[str] = []
        self.entry: Optional[dict] = None

    def __call__(self, ctx):
        if self.inner.done:
            return None
        w = self.inner(ctx)
        if w is None:
            return None
        tickers = ctx.market.tickers
        if self.weights_h is not None:
            hv = har_vols(self.har, ctx, self.weights_h).reindex(tickers)
            if _missing(hv):
                self.fallbacks.append(f"{ctx.day} weights: no h{self.weights_h} forecast "
                                      f"for {', '.join(_missing(hv))}; trailing weights")
            else:
                inv = 1.0 / hv
                w = W.safe((inv / inv.sum()).to_dict(), tickers)
        e = self.e_ref
        if self.exposure is not None:
            got = self.exposure(self.har, ctx.day, ctx.deadline)
            if got is None:
                self.fallbacks.append(f"{ctx.day} exposure: no basket forecast or typical "
                                      f"level; e_ref {self.e_ref}")
            else:
                e = got
        self.entry = {"day": ctx.day, "exposure": e}
        return W.safe({t: v * e for t, v in w.items()}, tickers)


class HarRiskParity(qs.QuantBook):
    """`q_riskparity_entry_regime` with HAR in its sizing, its exposure, or both.

    `vol_h`: HAR vols on the covariance's diagonal (None: the sample's own). `mode`: the
    entry exposure is the HMM's ("regime", the rule's), HAR's ("har"), their mean, or
    the lower of the two ("min"). Bought once, then held by the entry-only band.
    """

    def __init__(self, har: Optional[VolForecasts], vol_h: Optional[int] = None,
                 mode: str = "regime", exposure: Optional[HarExposure] = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if mode != "regime" and exposure is None:
            raise ValueError(f"mode {mode!r} needs a HarExposure")
        if (vol_h is not None or mode != "regime") and har is None:
            raise ValueError("HAR sizing needs the forecasts")
        super().__init__("risk_parity", qs.Regime(), qs.ENTRY_ONLY)
        self.har, self.vol_h, self.mode, self.exposure = har, vol_h, mode, exposure
        self.fallbacks: list[str] = []
        self.entry: Optional[dict] = None

    def _shape(self, ctx, rets):
        if self.vol_h is None:
            return super()._shape(ctx, rets)
        cov = quant.shrunk_cov(rets.tail(qs.SHAPE_DAYS))
        if cov is None or len(cov) < len(rets.columns):
            return None  # the gate `shape_risk_parity` applies: cash until there is history
        hv = har_vols(self.har, ctx, self.vol_h).reindex(cov.index)
        if _missing(hv):
            self.fallbacks.append(f"{ctx.day} shape: no h{self.vol_h} forecast for "
                                  f"{', '.join(_missing(hv))}; sample vols")
            return super()._shape(ctx, rets)
        return quant.risk_parity(har_covariance(cov, hv), 1.0, cap=W.CAP)

    def _exposure(self, ctx, rets):
        # Asked every day, as the rule asks it, so the "regime" book is the rule's own
        # down to its HMM's state, and a HAR mode differs from it only in the number used.
        e_rule = super()._exposure(ctx, rets)
        if self.entry is not None:
            return e_rule  # held by the entry-only band: no later exposure trades
        e_har = None if self.mode == "regime" else self.exposure(self.har, ctx.day, ctx.deadline)
        if self.mode == "regime":
            e = e_rule
        elif e_har is None:
            self.fallbacks.append(f"{ctx.day} exposure: no basket forecast or typical level; "
                                  f"the HMM's {e_rule:.3f}")
            e = e_rule
        else:
            e = {"har": e_har, "mean": 0.5 * (e_rule + e_har), "min": min(e_rule, e_har)}[self.mode]
        self.entry = {"day": ctx.day, "exposure": e, "regime": e_rule, "har": e_har}
        return e


def inverse_vol_hold(har, weights_h=None, exposure=None):
    """A `windows.run_field` factory: a fresh book per window."""
    return lambda: HarInverseVolHold(har, weights_h, exposure)


def risk_parity_entry(har, vol_h=None, mode="regime", exposure=None):
    """A `windows.run_field` factory: a fresh book (and HMM) per window."""
    return lambda: HarRiskParity(har, vol_h, mode, exposure)
