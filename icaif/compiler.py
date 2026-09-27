"""The compiler: model scores, volatility and levers in, one round's legal weights out.

An LLM agent will choose the *levers* (how much to hold, how many names, how often to
trade); everything that turns them into a decision is deterministic and lives here.
The agent never writes a weight, so it can never write an illegal one, and a backtest
of a lever setting is the strategy the agent would actually run.

Rules, in the order they apply:

- **Not selectable** is a NaN score, a NaN or non-positive vol, or a ticker on
  `avoid`. A NaN is missing data, not a bad score: a held name with one keeps its
  weight ("frozen") rather than being sold, because dumping a position whenever a
  feed hiccups is turnover and drawdown bought for nothing.
- **Exits** fire in any round: a held name on `avoid`, and (if `stop_sigma` is set) a
  held name whose price has fallen more than `stop_sigma` daily sigmas below the
  day's round-1 fill. An exit always trades, whatever the band.
- **Selection** happens only in `rebalance_rounds`. The selection score is the
  score's percentile among selectable names divided by vol**gamma, the division
  alphaBT makes after the model because upside targets are largely a volatility bet.
  A held name stays while its selection rank is within top_k + buffer; the rest of
  the top_k slots go to the best new names. Frozen names occupy slots. Other rounds
  only hold or exit, so an hourly re-rank never churns the book.
- **Sizing**: equal or inverse-vol across the book, scaled to `exposure` less the
  frozen weight, capped at 0.30 with the excess spread to uncapped names and, when
  none is left, held as cash. Never over 0.30, never levered.
- **Band**: a name staying in the book is not resized by less than `band`. Turnover
  is one of the four ranked metrics, so a 1% trim that buys nothing still costs a
  rank. Entries and exits ignore the band: a sub-band exit left undone is a stray
  position the selection no longer owns and nothing ever cleans up.
- `weights.safe` last, so a float such as 0.1 + 0.2 never reaches the backend's
  Decimal check.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from icaif import calendar, data
from icaif import weights as W

PREDS = data.ROOT / "output" / "preds" / "daily_d5_pct.parquet"
# A weight at or below one grid step is the residue of flooring, not a position.
HELD = W.GRID
WEIGHTINGS = ("equal", "inverse_vol")


@dataclass(frozen=True)
class Levers:
    """The settings an agent chooses. Out-of-range values raise rather than clamp:
    a clamped lever runs a different strategy from the one the agent believes it chose,
    and its explanation of the round would describe a book that never existed."""

    exposure: float = 0.75
    top_k: int = 10
    gamma: float = 0.5
    weighting: str = "inverse_vol"
    buffer: int = 3
    band: float = 0.02
    rebalance_rounds: tuple = (1,)
    stop_sigma: Optional[float] = None
    avoid: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "rebalance_rounds", tuple(sorted(set(self.rebalance_rounds))))
        object.__setattr__(self, "avoid", frozenset(self.avoid))
        problems = []
        if not 0.0 <= self.exposure <= 1.0:
            problems.append(f"exposure {self.exposure} outside [0, 1]")
        if int(self.top_k) != self.top_k or self.top_k < 1:
            problems.append(f"top_k {self.top_k} is not a positive integer")
        if not np.isfinite(self.gamma):
            problems.append(f"gamma {self.gamma} is not finite")
        if self.weighting not in WEIGHTINGS:
            problems.append(f"weighting {self.weighting!r} not in {WEIGHTINGS}")
        if int(self.buffer) != self.buffer or self.buffer < 0:
            problems.append(f"buffer {self.buffer} is not a non-negative integer")
        if not 0.0 <= self.band < 1.0:
            problems.append(f"band {self.band} outside [0, 1)")
        if not set(self.rebalance_rounds) <= set(calendar.ROUNDS):
            problems.append(f"rebalance_rounds {self.rebalance_rounds} not all in 1-7")
        if self.stop_sigma is not None and not self.stop_sigma > 0:
            problems.append(f"stop_sigma {self.stop_sigma} must be positive or None")
        if problems:
            raise ValueError("; ".join(problems))


def _series(x, index) -> pd.Series:
    if x is None:
        return pd.Series(np.nan, index=index)
    return pd.Series(x, dtype=float).reindex(index)


def _water_fill(raw: np.ndarray, budget: float, cap: float) -> np.ndarray:
    """Split `budget` in proportion to `raw`, no name above `cap`, the rest to cash.

    Clipping alone would leave the book under its exposure whenever an inverse-vol
    favourite hits the cap; spreading without the cap check would push a second name
    over it.
    """
    if budget <= 0 or len(raw) == 0:
        return np.zeros(len(raw))
    w = raw / raw.sum() * budget
    capped = np.zeros(len(raw), dtype=bool)
    for _ in range(len(raw)):
        over = w > cap
        if not over.any():
            break
        capped |= over
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~capped
        if not free.any():
            break  # every name is at the cap: the excess is cash
        w[free] += excess * raw[free] / raw[free].sum()
    return np.minimum(w, cap)


def stop_exits(vol: pd.Series, held: pd.Series, stop_sigma: Optional[float],
               entry_prices=None, last_prices=None) -> pd.Series:
    """Held names whose log move since entry is below -stop_sigma x daily vol.

    Every missing input (no entry yet, no price, no vol) means no exit: a NaN
    comparison is False, so a data gap can never sell a position.
    """
    index = vol.index
    if stop_sigma is None or entry_prices is None or last_prices is None:
        return pd.Series(False, index=index)
    entry, last = _series(entry_prices, index), _series(last_prices, index)
    with np.errstate(divide="ignore", invalid="ignore"):
        move = np.log(last / entry)
    return held & (move < -stop_sigma * vol)


def plan(scores: pd.Series, vol: pd.Series, current_weights, levers: Levers, round_no: int,
         entry_prices=None, last_prices=None) -> tuple[pd.Series, pd.Index]:
    """Unrounded target weights, and the names whose weight the round actually changes.

    A name the compiler leaves alone keeps *exactly* its current weight, so "changed"
    is an exact test. The strategy uses it to skip the round entirely (see
    `CompiledStrategy`), which matters because a weight is re-applied at the execution
    price: a submitted "hold" at last close's weights is a small rebalance of every name.
    """
    tickers = scores.index
    s = scores.astype(float)
    v = _series(vol, tickers)
    cur = _series(current_weights, tickers).fillna(0.0).clip(lower=0.0)
    held = cur > HELD
    avoided = pd.Series(tickers.isin(list(levers.avoid)), index=tickers)
    selectable = s.notna() & v.notna() & (v > 0) & ~avoided
    stopped = stop_exits(v, held, levers.stop_sigma, entry_prices, last_prices) & ~avoided
    frozen = held & ~selectable & ~avoided & ~stopped
    exits = (held & avoided) | stopped

    target = cur.copy()
    target[exits] = 0.0

    if round_no in levers.rebalance_rounds:
        eligible = selectable & ~stopped
        pct = s.where(eligible).rank(pct=True)
        sel = (pct / v.pow(levers.gamma)).where(eligible)
        # Ties keep ticker order, so a rerun of the same inputs picks the same book.
        order = sel.dropna().sort_values(ascending=False, kind="stable").index
        rank = pd.Series(np.arange(1, len(order) + 1), index=order).reindex(tickers)

        slots = max(0, levers.top_k - int(frozen.sum()))
        keep = [t for t in order if held[t] and rank[t] <= levers.top_k + levers.buffer][:slots]
        new = [t for t in order if t not in keep][:slots - len(keep)]
        book = keep + new

        budget = max(0.0, levers.exposure - float(cur[frozen].sum()))
        raw = (np.ones(len(book)) if levers.weighting == "equal"
               else 1.0 / v[book].to_numpy(dtype=float))
        sized = pd.Series(_water_fill(raw, budget, W.CAP), index=book, dtype=float)

        target[~frozen] = 0.0
        target[book] = sized
        stay = pd.Series(tickers.isin(keep), index=tickers)
        small = stay & ((target - cur).abs() < levers.band)
        target[small] = cur[small]

    changed = tickers[(target - cur).abs() > HELD]
    return target, changed


def compile_weights(scores: pd.Series, vol: pd.Series, current_weights, levers: Levers,
                    round_no: int, entry_prices=None, last_prices=None) -> dict[str, float]:
    """One round's weights for every ticker in `scores`, legal by the backend's check."""
    target, _ = plan(scores, vol, current_weights, levers, round_no, entry_prices, last_prices)
    return W.safe(target.to_dict(), list(scores.index))


class LookAheadError(AssertionError):
    """A decision asked for a value dated after its own deadline's day."""


class DailyPanel:
    """date x ticker values that apply to every round of their date, and only that date.

    The one door a compiled strategy has to scores and vol. It serves a row only for
    the day the deadline falls on, so a strategy that asked for tomorrow's prediction
    (an off-by-one in a date lookup reads as a very good model) raises instead.
    A date with no row is all-NaN, which the compiler reads as "hold".
    """

    def __init__(self, frame: pd.DataFrame, tickers: list[str]):
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        self.frame = frame.reindex(columns=tickers)
        self.tickers = tickers
        self._rows: dict = {}

    def for_day(self, day: date, deadline: pd.Timestamp) -> pd.Series:
        if pd.Timestamp(deadline).date() != day:
            raise LookAheadError(f"values for {day} requested at a deadline on {deadline}")
        if day not in self._rows:
            key = pd.Timestamp(day)
            self._rows[day] = (self.frame.loc[key].astype(float) if key in self.frame.index
                               else pd.Series(np.nan, index=self.tickers))
        return self._rows[day]


def load_daily_scores(path: Path = PREDS, column: str = "pred") -> DailyPanel:
    """The daily model's predictions for the 30 names; the row for d was made before d's open.

    The file ranks each day's broad universe (~104 names). Only the relative order among
    the 30 matters here, and the compiler re-ranks them among themselves.
    """
    tickers = sorted(data.load_universe())
    preds = pd.read_parquet(path)[column]
    preds = preds[preds.index.get_level_values("ticker").isin(tickers)]
    return DailyPanel(preds.unstack("ticker"), tickers)


def trailing_daily_vol(daily: Optional[pd.DataFrame] = None, window: int = 20,
                       min_periods: int = 15) -> DailyPanel:
    """20-session std of daily log returns through the close *before* each date.

    Computed as of each close, then shifted one session: the row for d must not
    contain d's own close, which is not known at d's 09:10 deadline. Unshifted, the
    selection would divide by a vol that already knows whether d was a big day.
    `min_periods` matches `daily_features`' vol_20d.
    """
    from icaif import daily_features, external

    tickers = sorted(data.load_universe())
    if daily is None:
        daily = external.load("yahoo_daily_universe")
    close = daily_features.panels(daily[daily["ticker"].isin(tickers)])["close"]
    logret = np.log(close).diff()
    vol = logret.rolling(window, min_periods=min_periods).std().shift(1)
    return DailyPanel(vol, tickers)


class CompiledStrategy:
    """A `sim.Strategy`: each round, look up the day's scores and vol and compile them.

    Current weights are valued at the last bar close the deadline can see. Returns None
    (hold: no trade, no fee) when the compiler changes nothing, because re-submitting
    the current weights re-sizes every name to the execution price and pays for it.
    """

    def __init__(self, scores: DailyPanel, vol: DailyPanel, levers: Levers = Levers()):
        self.scores, self.vol, self.levers = scores, vol, levers

    def __call__(self, ctx):
        lv = self.levers
        if (ctx.round not in lv.rebalance_rounds and lv.stop_sigma is None
                and not any(ctx.shares.get(t, 0) > 0 for t in lv.avoid)):
            # The compiler could only hold here (no selection, no stop, no avoided
            # holding), so skip its work: ~6 of every 7 rounds in a backtest.
            return None
        tickers = ctx.market.tickers
        s = self.scores.for_day(ctx.day, ctx.deadline).reindex(tickers)
        v = self.vol.for_day(ctx.day, ctx.deadline).reindex(tickers)
        shares = pd.Series(ctx.shares, dtype=float).reindex(tickers).fillna(0.0)
        last = ctx.recent_closes(1)
        last_px = last.iloc[-1].reindex(tickers) if len(last) else pd.Series(np.nan, index=tickers)
        if (shares != 0).any():
            value = shares * last_px
            nav = ctx.cash + float(value.sum())
            if not np.isfinite(nav) or nav <= 0:
                return None  # cannot value the book; holding beats guessing
            current = value / nav
        else:
            current = pd.Series(0.0, index=tickers)

        entry = None
        if self.levers.stop_sigma is not None and ctx.round not in self.levers.rebalance_rounds:
            entry = _day_open(ctx)
        target, changed = plan(s, v, current, self.levers, ctx.round, entry, last_px)
        if len(changed) == 0:
            return None
        return W.safe(target.to_dict(), tickers)


def _day_open(ctx) -> Optional[pd.Series]:
    """The day's round-1 fill (the 09:30 bar's open), once that bar has ended.

    At round 2's deadline (10:25) the 09:30-10:30 bar is still open, so its open is not
    yet in the bars a decision may see; the stop is blind until round 3 rather than
    reading a price from the future.
    """
    m = ctx.market
    if not hasattr(m, "_compiler_opens"):
        b = m.info_bars
        m._compiler_opens = b.pivot_table(index="start", columns="ticker", values="open",
                                          aggfunc="last").reindex(columns=m.tickers)
        m._compiler_open_ends = b.groupby("start")["end"].max()
    first = calendar.at(ctx.day, calendar.SESSION_OPEN)
    if first not in m._compiler_opens.index or m._compiler_open_ends[first] > ctx.deadline:
        return None
    return m._compiler_opens.loc[first]


def compiled(scores: DailyPanel, vol: DailyPanel, levers: Levers = Levers()):
    """A factory for `windows.run_field`: a fresh strategy per window."""
    return lambda: CompiledStrategy(scores, vol, levers)
