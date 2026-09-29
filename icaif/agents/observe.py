"""What an agent sees: a point-in-time, JSON-able snapshot built from daily closes.

The desk hands this module only closes a decision may see (`quant_strategies.
daily_closes` cuts them at the deadline), so nothing here can look ahead by itself;
the test that rewrites the future and requires the observation unchanged guards the
seam anyway.

**Anonymised mode** exists because the model has read 2016-25 market history. Shown
"NVDA, 2024-05-20", it can recall what happened next, and a replay that scores well
on memory is not evidence the agent can judge. So in replays each window gets its own
random name codes (S01-S30, sorted by code, so even the alphabetical order of real
tickers is gone), dates become "day k of 15", no price level appears, macro levels
become z-scores and changes (`macro.readings`), and headlines, which name companies,
are dropped. 8-K events stay: "director or officer change" names no one. Only windows after the model's training cutoff can
be replayed with real names and still count.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from icaif import quant, quant_strategies as qs

ANNUAL = np.sqrt(252)


class Anonymizer:
    """A per-window bijection between tickers and codes; identity when disabled."""

    def __init__(self, tickers: list[str], seed: Optional[int]):
        self.enabled = seed is not None
        if self.enabled:
            order = np.random.default_rng(seed).permutation(len(tickers))
            codes = [f"S{i + 1:02d}" for i in range(len(tickers))]
            self.to_code = {tickers[j]: codes[i] for i, j in enumerate(order)}
        else:
            self.to_code = {t: t for t in tickers}
        self.to_ticker = {c: t for t, c in self.to_code.items()}

    def code(self, ticker: str) -> str:
        return self.to_code[ticker]

    def ticker(self, code: str) -> str:
        """Raises KeyError on a code the agent invented, which the desk treats as an
        invalid answer: a guessed mapping would sell a name the agent never named."""
        return self.to_ticker[code]


@dataclass
class BookState:
    weights: pd.Series            # current weights, every ticker
    nav: list[float]              # NAV at each round 1 so far, first = 1,000,000
    traded_notional: float = 0.0  # to date, for the turnover the agent has spent
    entered: bool = False


@dataclass
class Readings:
    """The market-level numbers the rule and the agents both read (unrounded)."""

    returns: pd.DataFrame = field(repr=False)
    basket: pd.Series = field(repr=False)
    ewma_vol: float
    vol_ratio: float
    p_turbulent_next: float
    hmm: Optional[quant.HMM2]


def readings(closes: pd.DataFrame, hmm: Optional[quant.HMM2]) -> Readings:
    rets = np.log(closes).diff().iloc[1:]
    basket = qs._basket(rets)
    b = basket.dropna()
    ewma = np.sqrt((b ** 2).ewm(alpha=0.06).mean()) if len(b) else pd.Series(dtype=float)
    vol = float(ewma.iloc[-1]) if len(ewma) else np.nan
    ratio = float(vol / ewma.median()) if len(ewma) > 60 else np.nan
    p = quant.hmm_next_turbulent(b.to_numpy()[-250:], hmm) if hmm is not None and len(b) else np.nan
    return Readings(rets, basket, vol, ratio, p, hmm)


def fit_regime(closes: pd.DataFrame) -> Optional[quant.HMM2]:
    """The HMM fit a window uses throughout: on the history before its first round."""
    b = qs._basket(np.log(closes).diff().iloc[1:]).dropna().to_numpy()
    return quant.fit_hmm2(b) if len(b) >= 250 else None


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def observation(closes: pd.DataFrame, rd: Readings, book: BookState, anon: Anonymizer, *,
                day: int, window_days: int, round_no: int,
                scores: Optional[pd.Series] = None,
                earnings: Optional[dict] = None,
                news: Optional[dict] = None,
                macro: Optional[dict] = None,
                filings: Optional[dict] = None,
                calendar_date: Optional[str] = None) -> dict:
    tickers = list(closes.columns)
    rets = rd.returns
    tail = rets.tail(qs.SHAPE_DAYS)
    shapes = {}
    for name in ("inverse_vol", "risk_parity"):
        s = qs.SHAPES[name](tail)
        shapes[name] = s.reindex(tickers) if s is not None else pd.Series(np.nan, index=tickers)
    cov = quant.shrunk_cov(tail)
    ou = quant.s_scores(tail) if len(tail) >= 20 else pd.DataFrame(index=tickers, columns=["s"])
    pct = scores.rank(pct=True) if scores is not None else None

    nav = np.array(book.nav, dtype=float)
    peak = nav.max() if len(nav) else np.nan
    names = []
    for t in tickers:
        r = rets[t]
        row = {
            "name": anon.code(t),
            "vol_ann_20d": _r(r.tail(20).std() * ANNUAL),
            "ret_1d": _r(r.iloc[-1] if len(r) else np.nan),
            "ret_5d": _r(r.tail(5).sum()),
            "ret_20d": _r(r.tail(20).sum()),
            "weight_now": _r(book.weights.get(t, 0.0)),
            "weight_if_inverse_vol": _r(shapes["inverse_vol"][t]),
            "weight_if_risk_parity": _r(shapes["risk_parity"][t]),
            "ou_s_score": _r(ou["s"].get(t, np.nan) if "s" in ou else np.nan, 2),
        }
        if pct is not None:
            row["model_score_pct"] = _r(pct.get(t, np.nan), 2)
        if earnings is not None:
            row["earnings_in_sessions"] = earnings.get(t)
        if filings is not None:
            row["recent_8k_filings"] = filings.get(t, [])
        if news is not None and not anon.enabled:
            row["headlines"] = news.get(t, [])[:5]
        names.append(row)
    names.sort(key=lambda x: x["name"])

    avg_corr = None
    if cov is not None:
        sd = np.sqrt(np.diag(cov.to_numpy()))
        corr = cov.to_numpy() / np.outer(sd, sd)
        avg_corr = _r(corr[np.triu_indices_from(corr, 1)].mean())
    hmm = rd.hmm
    obs = {
        "clock": {"day": day, "of": window_days, "round": round_no,
                  "sessions_left_after_today": window_days - day},
        "book": {
            "return_to_date": _r(nav[-1] / nav[0] - 1 if len(nav) else 0.0),
            "drawdown_from_peak": _r(1 - nav[-1] / peak if len(nav) else 0.0),
            "gross": _r(float(book.weights.sum())),
            "entered": book.entered,
            "turnover_spent": _r(book.traded_notional / nav[0] if len(nav) else 0.0),
        },
        "market": {
            "basket_ret_1d": _r(rd.basket.iloc[-1] if len(rd.basket) else np.nan),
            "basket_ret_5d": _r(rd.basket.tail(5).sum()),
            "basket_ret_20d": _r(rd.basket.tail(20).sum()),
            "basket_vol_ann_ewma": _r(rd.ewma_vol * ANNUAL),
            "vol_vs_3y_median": _r(rd.vol_ratio, 3),
            "p_turbulent_next_session": _r(rd.p_turbulent_next, 6),
            "regime_persistence_days": (None if hmm is None else
                                        {"calm": _r(hmm.persistence[0], 1),
                                         "turbulent": _r(hmm.persistence[1], 1)}),
            "avg_pairwise_corr_60d": avg_corr,
        },
        "names": names,
    }
    if macro is not None:
        obs["macro"] = macro
    if calendar_date and not anon.enabled:
        obs["clock"]["date"] = calendar_date
    return obs
