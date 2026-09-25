"""The competition's portfolio ledger, round by round, for backtests and paper trading.

One simulator serves every use (backtest, paper trading, a shadow of the live book), so
the rules live here once. The rules are the kit's (`starter-kit/docs/rules.md`,
`docs/evaluation.md`):

- A round's decision executes at that round's execution price. Target value is
  weight x NAV before the trade; fractional shares are allowed.
- The fee is 0.1% of buy-plus-sell notional, taken from cash, so it lands in that
  round's return.
- A missing or invalid decision holds: no trade, no fee, zero notional. The period still
  counts, and so does its market move.
- Period k ends at the next round's execution price, before that round trades. The
  last period of a run ends at the final official close.
- Drawdown sees the initial NAV, every period endpoint and every 16:00 close.

**Sizing against the fee is an assumption.** The rules say how much is charged, not
whether targets are sized before or after the charge. `pre_fee` (the default) sizes on
NAV before the fee, so a fully invested target leaves cash slightly negative, by
0.1% x notional. `post_fee` shrinks the targets so the fee is covered. Validation
receipts show real share counts; check which one the backend does before trusting the
difference.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd

from icaif import calendar, kit

INITIAL_NAV = 1_000_000.0
FEE_RATE = 0.001


@dataclass
class Market:
    """Execution prices on the live grid, plus the bars a decision may look at.

    `exec_prices`: index = execution timestamps (one per round that runs), columns =
    tickers. `closes`: index = each day's official close timestamp. `info_bars`: any
    canonical bar frame; a decision sees only bars whose `end` <= its deadline.
    """

    exec_prices: pd.DataFrame
    closes: pd.DataFrame
    info_bars: pd.DataFrame
    issues: dict = field(default_factory=dict)

    def __post_init__(self):
        self.info_bars = self.info_bars.sort_values("end").reset_index(drop=True)
        # UTC nanoseconds: a tz-aware column becomes an object array under to_numpy(),
        # which numpy cannot binary-search against a timestamp.
        self._ends = pd.DatetimeIndex(self.info_bars["end"]).asi8
        self.tickers = list(self.exec_prices.columns)
        self.days = sorted({ts.date() for ts in self.exec_prices.index})

    def history(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Bars that had ended by `as_of`. The only door a strategy has to prices.

        Filtering by `end`, not `start`, is the whole guard: a bar that started before
        the deadline but ends after it contains prices from after the deadline.
        """
        n = np.searchsorted(self._ends, pd.Timestamp(as_of).value, side="right")
        return self.info_bars.iloc[:n]

    def recent_closes(self, as_of: pd.Timestamp, n: int) -> pd.DataFrame:
        """The last `n` bar closes (rows = bar end, columns = tickers) ended by `as_of`.

        The same cutoff as `history`, on a wide panel built once. Pivoting `history()`
        every round is ~5k pivots per backtest year per strategy.
        """
        if not hasattr(self, "_close_panel"):
            self._close_panel = (self.info_bars.pivot_table(
                index="end", columns="ticker", values="close", aggfunc="last")
                .sort_index().reindex(columns=self.tickers))
            self._panel_ends = pd.DatetimeIndex(self._close_panel.index).asi8
        k = np.searchsorted(self._panel_ends, pd.Timestamp(as_of).value, side="right")
        return self._close_panel.iloc[max(0, k - n):k]


def market_from_public_60m(bars_60m: pd.DataFrame, info_bars: pd.DataFrame) -> Market:
    """Execution prices from public 60m bars, which sit on the live :30 grid.

    A round's price is the open of the 60m bar starting at its execution time; a day's
    official close is the close of its last bar. Where Yahoo has no such bar, the last
    completed close stands in so the ledger stays valued, and the day is listed in
    `issues["degraded_days"]`. Measured 2026-09-25, those are:

    - every half-day: Yahoo has no 12:30-13:00 bar, so round 4 and the 13:00 close
      are missing;
    - Yahoo outages on 2026-01-30 (after 10:30) and 2026-02-02 (the morning);
    - leading days before every ticker has a first bar (TMO starts 2023-11-06), which
      are dropped outright, since there is no earlier price to stand in.

    A stand-in price is a zero return that never happened, followed by a catch-up move.
    Windows touching a degraded day should be skipped, not scored; the competition's
    own window (Oct 12-30) contains no half-day.
    """
    opens = bars_60m.pivot(index="start", columns="ticker", values="open").sort_index()
    last_close = (bars_60m.pivot(index="end", columns="ticker", values="close")
                  .sort_index().ffill())
    days = sorted({ts.date() for ts in opens.index})
    exec_ts = [r["execution"] for d in days for r in calendar.rounds_for(d)]
    close_ts = [calendar.at(d, calendar.session_close(d)) for d in days]
    exec_raw = opens.reindex(exec_ts)
    close_raw = bars_60m.pivot(index="end", columns="ticker", values="close").reindex(close_ts)

    missing = (exec_raw.isna().groupby(exec_raw.index.date).sum().sum(axis=1)
               + close_raw.isna().groupby(close_raw.index.date).sum().sum(axis=1))
    seen = opens.notna().cummax().all(axis=1)
    first_full = seen.idxmax().date() if seen.any() else None
    if first_full is None:
        raise ValueError("no execution time at which every ticker has a price")
    leading = [d for d in days if d < first_full]
    keep_exec = exec_raw.index.date >= first_full
    keep_close = close_raw.index.date >= first_full

    exec_prices = exec_raw[keep_exec].fillna(
        last_close.reindex(exec_raw.index[keep_exec], method="ffill"))
    closes = close_raw[keep_close].fillna(
        last_close.reindex(close_raw.index[keep_close], method="ffill"))
    # A NaN price poisons NAV even with zero shares held (0 x NaN is NaN), so a gap
    # left here would surface as an unscoreable run rather than a missing bar.
    if exec_prices.isna().any().any() or closes.isna().any().any():
        raise ValueError("a ticker has no price at or before some round; cannot value it")
    degraded = sorted(d for d, n in missing.items() if n and d >= first_full)
    return Market(exec_prices, closes, info_bars, issues={
        "dropped_leading_days": [str(d) for d in leading],
        "degraded_days": [str(d) for d in degraded],
        "stand_in_prices": int(missing[missing.index >= first_full].sum()),
    })


@dataclass
class RoundContext:
    """What a strategy is handed at a round: the clock, its own book, and prices."""

    day: date
    round: int
    deadline: pd.Timestamp
    execution: pd.Timestamp
    shares: dict
    cash: float
    market: Market

    def history(self) -> pd.DataFrame:
        return self.market.history(self.deadline)

    def recent_closes(self, n: int) -> pd.DataFrame:
        return self.market.recent_closes(self.deadline, n)


# A strategy returns target weights for all 30 tickers, or None to skip the round.
Strategy = Callable[[RoundContext], Optional[dict]]


@dataclass
class Result:
    periods: list[dict]
    valuation_points: list[float]
    valuation_times: list[pd.Timestamp]
    ledger: pd.DataFrame
    invalid_rounds: list[dict]

    def metrics(self) -> dict:
        return kit.metrics(
            [{k: p[k] for k in ("nav_before", "nav_after_period", "traded_notional")}
             for p in self.periods],
            self.valuation_points, INITIAL_NAV)


def run(strategy: Strategy, market: Market, start: date, n_days: int,
        sizing: str = "pre_fee") -> Result:
    """Run one competition-shaped window: fresh $1M, `n_days` trading days from `start`."""
    if sizing not in ("pre_fee", "post_fee"):
        raise ValueError(f"unknown sizing {sizing!r}")
    days = [d for d in market.days if d >= start][:n_days]
    if len(days) < n_days:
        raise ValueError(f"only {len(days)} trading days from {start}, wanted {n_days}")
    tickers = market.tickers
    shares = np.zeros(len(tickers))
    cash = INITIAL_NAV

    periods, rows, invalid = [], [], []
    points, times = [INITIAL_NAV], [None]
    for d in days:
        for r in calendar.rounds_for(d):
            price = market.exec_prices.loc[r["execution"]].to_numpy(dtype=float)
            nav_before = cash + float(shares @ price)
            if periods:
                periods[-1]["nav_after_period"] = nav_before
                points.append(nav_before)
                times.append(r["execution"])

            ctx = RoundContext(d, r["round"], r["deadline"], r["execution"],
                               dict(zip(tickers, shares)), cash, market)
            weights = strategy(ctx)
            notional = 0.0
            if weights is not None:
                try:
                    kit.validate_weights(weights)
                except kit.SubmissionError as err:
                    invalid.append({"execution": r["execution"], "reason": str(err)})
                    weights = None
            if weights is not None:
                w = np.array([float(weights[t]) for t in tickers])
                target = _target_shares(w, nav_before, price, shares, sizing)
                trade = target - shares
                notional = float(np.abs(trade) @ price)
                cash -= float(trade @ price) + FEE_RATE * notional
                shares = target
            periods.append({"execution": r["execution"], "nav_before": nav_before,
                            "nav_after_period": None, "traded_notional": notional,
                            "held": weights is None})
            rows.append({"execution": r["execution"], "cash": cash,
                         **dict(zip(tickers, shares))})
        close_ts = calendar.at(d, calendar.session_close(d))
        close_px = market.closes.loc[close_ts].to_numpy(dtype=float)
        points.append(cash + float(shares @ close_px))
        times.append(close_ts)

    # The last period ends at the final close, which is already the last point; drawdown
    # dedupes points sharing a timestamp, so it is not appended twice.
    periods[-1]["nav_after_period"] = points[-1]
    return Result(periods, points, times, pd.DataFrame(rows), invalid)


def _target_shares(w, nav_before, price, shares, sizing):
    target = w * nav_before / price
    if sizing == "post_fee":
        # Shrink only when the fee would not fit in the cash the targets leave. Solve for
        # the scale s where s x invested + fee(s) = NAV; the fee depends on the trade, so
        # iterate (three rounds converge well below a cent).
        invested = float(w.sum()) * nav_before
        s = 1.0
        for _ in range(3):
            fee = FEE_RATE * float(np.abs(s * target - shares) @ price)
            if invested + fee <= nav_before or invested == 0:
                break
            s = (nav_before - fee) / invested
        target = s * target
    return target
