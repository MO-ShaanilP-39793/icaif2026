"""Our own signals in front of the desk's roles, each served for the decision's day only.

Roadmap step 2. Three inputs the repo already makes, which until now the roles either
never saw (HAR) or saw undocumented:

- **HAR volatility** (`icaif.vol`): each name's and the basket's forecast of mean daily
  variance over the next 1 and 3 sessions. Replays read the walk-forward, refit each
  quarter on targets that had ended; live reads `vol.forecast_next`, which a test in
  test_vol pins to the walk-forward's number for the same session.
- **The daily model's score**, as a rank within the 30 (1 = best). Replays read the
  walk-forward predictions (`compiler.load_daily_scores`), live the frozen model's
  row for the day (`live.daily_scores`).
- **The daily model's ranking of its whole training universe** (`UniverseScores`), about
  100 names a day, as context: only the 30 can be traded. The rank among the 30 stays the
  primary signal, because it is the one measured (mean daily rank IC 0.051, 2023 to Sep
  2026); what the agent can do with the other names' ranks is not measured.
- **Sessions to the next earnings reaction** (`desk.EarningsCalendar` in replays,
  `live.CalendarEarnings` live), unchanged here.

Every lookup goes through `compiler.DailyPanel`'s door, which serves a row only for
the day its deadline falls on. The forecast for D+1 already contains D's realised
variance, so an off-by-one that handed a round-1 decision tomorrow's row would read as
a forecaster that sees the day coming; it raises instead.

**Black-Litterman** takes the scores as views on the risk-parity book. The views are
sized from what the model has actually shown, not from what the agent hopes: Grinold's
alpha = IC x vol x z, with the IC the model earned among the 30 in its test years. The
agent chooses only how much weight the views get, from three levels whose books it is
shown before it chooses.
"""

from typing import Optional

import numpy as np
import pandas as pd

from icaif import compiler, quant, vol
from icaif import weights as W

# Annualising a daily variance for display: sqrt(var x 252), as vol_ann_20d.
TRADING_DAYS = 252

# The daily model's rank IC among the 30 names in its walk-forward test years
# (reports/walkforward_daily.csv, target d5_pct): 0.049, 0.091, 0.033, 0.021 for
# 2023-26. The two latest years, the ones nearest the frozen 2026 model, average 0.027;
# the four average 0.048. Views sized on the four-year mean would trust the model
# about twice as much as its recent record supports.
VIEW_IC = 0.03
# The model ranks 5-session forward returns.
VIEW_HORIZON = 5
# Idzorek confidence per level the Strategist may choose (tools/bl_calibration.py, the
# 61 window entry days with scores, 2023-26). "light" moves a median 10% of the book
# from risk parity toward the scores (IQR 9-13%) and holds at least 28 names; "strong"
# moves 27% (24-31%) and holds at least 21. Full confidence moves 53% and keeps 13
# names: the model's IC does not support a book that concentrated.
VIEW_LEVELS = {"none": 0.0, "light": 0.07, "strong": 0.2}


class VolForecasts:
    """HAR forecasts of mean daily variance, rows (session, ticker), `_MKT` the basket."""

    def __init__(self, frame: pd.DataFrame, tickers: list[str]):
        cols = list(tickers) + [vol.MARKET]
        self.tickers = list(tickers)
        horizons = sorted(int(c[len("har_h"):]) for c in frame.columns
                          if c.startswith("har_h") and c[len("har_h"):].isdigit())
        self.panels = {h: compiler.DailyPanel(frame[f"har_h{h}"].unstack("ticker"), cols)
                       for h in horizons}

    @classmethod
    def from_bars(cls, info_bars: pd.DataFrame, tickers: list[str],
                  horizons: tuple[int, ...] = vol.HORIZONS) -> "VolForecasts":
        """The walk-forward over `info_bars`, from the first quarter with enough history.

        A quarter the history cannot fit raises in `walk_forward` rather than forecasting
        NaN; here the replay simply starts its forecasts at the next quarter, and the
        sessions before have none (null in the observation, not a guess). Every horizon
        starts at the first quarter all of them can fit, so a book comparing horizons
        compares them on the same sessions.
        """
        rv = vol.realised_variance(info_bars)
        sessions = pd.to_datetime(pd.Index(rv.index))
        for start in pd.date_range(sessions.min(), sessions.max(), freq="QS"):
            try:
                return cls(vol.walk_forward(rv, first_test=str(start.date()), horizons=horizons),
                           tickers)
            except ValueError as err:
                if "training rows" not in str(err):
                    raise
        raise ValueError("no quarter of these bars has history enough for a HAR fit")

    @classmethod
    def from_live(cls, forecast: pd.DataFrame, tickers: list[str]) -> "VolForecasts":
        """One session's `vol.forecast_next` result, served for that session only."""
        session = forecast.attrs["session"]
        frame = forecast.copy()
        frame.index = pd.MultiIndex.from_arrays(
            [[session] * len(frame), frame.index], names=["session", "ticker"])
        return cls(frame, tickers)

    def for_day(self, day, deadline) -> pd.DataFrame:
        """Ticker (and `_MKT`) x har_h1, har_h3 for `day`; NaN where there is none."""
        return pd.DataFrame({f"har_h{h}": p.for_day(day, deadline) for h, p in self.panels.items()})

    def trailing(self, day, deadline, n: int, horizon: int) -> pd.DataFrame:
        """Session x ticker (and `_MKT`): the last `n` sessions' `horizon` forecasts to `day`.

        Each was made before its own session opened, so all of them are known at `day`'s
        open, and none after it is served.
        """
        return self.panels[horizon].trailing(day, deadline, n)


class UniverseScores:
    """The daily model's score for every name in each day's training universe.

    Rows are (date, ticker) -> score, a varying set of names a day. Served through
    `compiler.DailyPanel`'s door like the 30's scores: a universe row is made from the
    prior close, so tomorrow's would carry today's, and the door raises for it.
    `tradeable` is the competition's 30; every other name is context the agent may read
    and never trade.
    """

    def __init__(self, scores: pd.Series, tradeable: list[str]):
        wide = scores.unstack("ticker")
        self.panel = compiler.DailyPanel(wide, sorted(wide.columns))
        self.tradeable = set(tradeable)

    @classmethod
    def load(cls, path=None, column: str = "pred") -> "UniverseScores":
        """The walk-forward predictions for every universe name (replays, 2023 on)."""
        from icaif import data

        return cls(pd.read_parquet(path or compiler.PREDS)[column], sorted(data.load_universe()))

    def for_day(self, day, deadline) -> pd.DataFrame:
        """Ticker x rank (1 = best), percentile (1.0 = best), tradeable; best first.

        Only the names scored that day: a name outside the day's universe is absent, not
        last, so a percentile counts the universe the model ranked.
        """
        s = self.panel.for_day(day, deadline).dropna()
        out = pd.DataFrame({"rank": s.rank(ascending=False, method="min"),
                            "percentile": s.rank(pct=True),
                            "tradeable": [t in self.tradeable for t in s.index]})
        return out.sort_values(["rank", "percentile"], ascending=[True, False])


def annualised(var: pd.Series) -> pd.Series:
    return np.sqrt(var * TRADING_DAYS)


def score_ranks(scores: pd.Series) -> pd.Series:
    """1 = the highest score among the names that have one; NaN for a name without.

    A rank, not the score: the model's output is a percentile of a broad universe, and
    only its order among the 30 is what the backtests measured.
    """
    return scores.rank(ascending=False, method="min")


def score_alpha(scores: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Each scored name's expected daily return above the prior's: IC x vol x z.

    z is the normal score of the name's rank among the scored names, so a skewed or
    clipped score distribution cannot make one name's view louder than its rank. vol is
    the name's daily vol from the same covariance the prior was built on, so the view
    and the prior disagree in the same units. NaN for a name without a score.
    """
    from scipy.stats import norm

    s = scores.reindex(cov.index).astype(float)
    ok = s.notna()
    out = pd.Series(np.nan, index=cov.index)
    n = int(ok.sum())
    if n < 2:
        return out
    z = norm.ppf((s[ok].rank(method="average") - 0.5) / n)
    z = z / z.std()
    sd = np.sqrt(np.diag(cov.loc[ok[ok].index, ok[ok].index].to_numpy()))
    out[ok] = VIEW_IC * sd * z / np.sqrt(VIEW_HORIZON)
    return out


def views_book(prior: pd.Series, returns: pd.DataFrame, scores: Optional[pd.Series],
               level: str) -> Optional[pd.Series]:
    """The fully invested book `prior` becomes with the scores as views at `level`.

    None when there is nothing to take a view from (no scores, or a covariance that
    does not cover the book), so a level the desk cannot build is never offered and an
    answer choosing it is rejected rather than quietly bought as the prior.
    """
    c = VIEW_LEVELS[level]
    if c == 0.0:
        return prior
    if scores is None or scores.reindex(prior.index).notna().sum() < 2:
        return None
    cov = quant.shrunk_cov(returns)
    if cov is None or len(cov) < len(prior):
        return None
    return quant.black_litterman(prior, cov, score_alpha(scores, cov), c, cap=W.CAP)
