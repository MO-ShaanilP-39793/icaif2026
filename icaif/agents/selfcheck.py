"""Code's report on a draft entry book, for the PM to confirm or revise (desk v3).

The free desk's one-shot books were concentrated: on 2026-04-13 it opened with 8 names
at 85% gross, 15% of it in one bank, and lost to the inverse-vol hold in three of four
windows. Nothing in its observation said what that book's risk was, so it never weighed
it. The report states the draft's concentration, expected volatility, market beta, its
recent drawdown, the weight reporting results soon and the entry's cost, beside the same numbers for the
inverse-vol and risk-parity books at the draft's own gross, so a difference is the
draft's choice of names and not of cash.

Everything is read from closes cut at the deadline (`qs.daily_closes`), so the report is
as point in time as the observation it sits beside. It reads no HAR forecast and no
model score: an ablation that drops one of those streams must not get it back here.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from icaif import compiler, quant, quant_strategies as qs

ANNUAL = np.sqrt(252)
LOOKBACK_SESSIONS = 15   # one window's length, looking back
EARNINGS_HORIZON = 10    # the earnings calendar's reach (`earnings.NEXT_KNOWN_SESSIONS`)


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def book_stats(w: pd.Series, tail: pd.DataFrame, cov: Optional[pd.DataFrame],
               recent: pd.DataFrame, earnings: Optional[dict], code: Callable[[str], str]) -> dict:
    """`w`: weights of NAV per ticker; `tail`: the shape window's daily log returns;
    `recent`: the last LOOKBACK_SESSIONS of them; `earnings`: {ticker: sessions to the
    reaction} within the calendar's horizon (None: no calendar)."""
    w = w.reindex(tail.columns).fillna(0.0).astype(float)
    gross = float(w.sum())
    held = w[w > compiler.HELD].sort_values(ascending=False)
    out = {"gross": _r(gross), "cash": _r(1.0 - gross), "names_held": int(len(held)),
           "largest": [{"name": code(t), "weight": _r(v)} for t, v in held.head(3).items()],
           "entry_turnover": _r(gross), "entry_fee_bps_of_nav": _r(gross * 10.0, 2)}
    if gross <= compiler.HELD:
        return out
    share = w / gross
    out["effective_names"] = _r(1.0 / float((share ** 2).sum()), 1)
    if cov is not None:
        c = cov.reindex(index=w.index, columns=w.index).fillna(0.0).to_numpy()
        var = float(w.to_numpy() @ c @ w.to_numpy())
        sd = np.sqrt(np.clip(np.diag(c), 0.0, None))
        out["expected_vol_ann"] = _r(np.sqrt(var) * ANNUAL)
        out["diversification_ratio"] = _r(float(w.to_numpy() @ sd) / np.sqrt(var), 2) if var > 0 else None
    basket = tail.mean(axis=1)
    port = tail.fillna(0.0) @ w
    if basket.var() > 0:
        out["beta_to_basket"] = _r(float(np.cov(port, basket)[0, 1] / basket.var()), 2)
    if earnings is not None:
        out[f"weight_reporting_within_{EARNINGS_HORIZON}_sessions"] = _r(
            float(sum(w[t] for t, n in earnings.items() if t in w.index and n <= EARNINGS_HORIZON)))
    # The drawdown a static book would have taken over the last window's sessions,
    # rebalanced daily to these weights: an approximation, stated as one. Its return is
    # left out on purpose. In the first paid window (2025-02-03) the PM confirmed an
    # 11-name, beta-1.09 book citing "the backtested performance", and the only
    # performance it was shown was this line's trailing return, 5.96% against inverse-
    # vol's 4.21%. A trailing return rewards whatever just rose; shown beside a draft it
    # argues for the draft's own momentum, and a check meant to surface risk becomes one
    # that endorses it.
    simple = np.expm1(recent.fillna(0.0)) @ w
    nav = (1.0 + simple).cumprod()
    out[f"last_{LOOKBACK_SESSIONS}_sessions_if_held"] = {
        "max_drawdown": _r(float((1.0 - nav / nav.cummax().clip(lower=1.0)).max())) if len(nav) else None}
    return out


def report(book: pd.Series, closes: pd.DataFrame, earnings: Optional[dict],
           code: Callable[[str], str]) -> dict:
    """The draft's numbers and, at its own gross, the inverse-vol and risk-parity books'.

    `book`: the weights the draft would buy, per ticker (cash is the rest); `closes`:
    `qs.daily_closes` at the deadline.
    """
    rets = np.log(closes).diff().iloc[1:]
    tail = rets.tail(qs.SHAPE_DAYS)
    recent = rets.tail(LOOKBACK_SESSIONS)
    cov = quant.shrunk_cov(tail)
    gross = float(book.sum())
    out = {"your_book": book_stats(book, tail, cov, recent, earnings, code)}
    for name in ("inverse_vol", "risk_parity"):
        shape = qs.SHAPES[name](tail)
        if shape is not None and gross > compiler.HELD:
            ref = shape.reindex(tail.columns).fillna(0.0) * gross
            out[f"{name}_at_your_gross"] = book_stats(ref, tail, cov, recent, earnings, code)
    out["notes"] = ("Volatility and beta from the last 60 sessions' shrunk covariance; "
                    f"the last-{LOOKBACK_SESSIONS}-session drawdown holds these weights daily, "
                    "an approximation. Fees are 0.1% of what is bought.")
    return out
