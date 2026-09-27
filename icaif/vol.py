"""Volatility forecasts: log-HAR on realised variance, per ticker and for the market.

The design doc bets that volatility is more forecastable than direction here, and that
sizing, the exposure dial and the exit bands all lean on it. This module is that bet's
first test: a forecast must beat trailing vol out of sample, or the levers built on it
are dressed-up trailing vol.

**Realised variance** of a session = sum of squared intraday bar log returns (the first
bar from its own open) plus the squared overnight gap from the prior close. A session
sum is close to grid-invariant: the organizer and public grids cut the same session into
different bars, but the squared returns add up to about the same total. A ticker-session
with fewer bars than its peers that day (a data hole) gets NaN, not a smaller number:
missing bars would read as a calm day. So does one whose bars do not span the calendar
session (Yahoo has no 12:30-13:00 bar on half-days, for every name at once, so no peer
shows the hole), and one with no prior close to measure the gap from (the first session,
or the day after a missing one: a gap from two sessions back is two sessions' move).
The market basket is NaN on any session one of its members is, since an average over
the names that happen to be present is a different, noisier basket.

**Model.** For a forecast made before session D opens, for horizon H sessions, the
target is the log of mean daily RV over D .. D+H-1. The regressors are the log of RV on
D-1, the mean over the last 5 sessions and the mean over the last 22, pooled across
tickers (Corsi 2009, in logs). The market basket is fit on its own: pooled in with the
30 names it would get 1/31 of the weight, and its persistence is not theirs.

The forecast is exp(fit) times the fit's smearing factor, the mean of exp(residual) over
its training rows (Duan 1983). exp(fit) alone is a median, and runs low. The lognormal
correction exp(s^2 / 2) assumes the residuals are normal; earnings jumps give them a
right tail, and with it stock forecasts ran 20% low out of sample, so a vol target built
on them would run hot by the same margin. Smearing leaves them 7% low (the basket 4%),
and the whole of that is the top 1% of surprises: drop those and the rest run 13% high.
The remaining bias is unforecast jumps, which a smearing factor cannot fix and an
earnings calendar might.

**Gaps in the regressors.** A missing session costs a name two RV sessions (its own and
the next day's unknown gap). With strict windows that blanked the name's forecasts for
22 sessions. So the lag falls back to the latest RV in D-1 .. D-3, and each window mean
takes the sessions it has, if at least 60% of them. Targets stay strict: a target with a
hole in it is not the thing being forecast.

**Half-days stay in the regressors as ordinary sessions**, on evidence. Their RV is low
(3.5 hours), which looks like it would drag the next forecast down wrongly. It does not:
the sessions after a half-day are the quiet holiday weeks, and with half-days kept the
forecast for the session after already runs 14% high (realised / forecast 0.86 over 21
half-days, 2017-26). Treating them as gaps raised it further and lost 1-day QLIKE (0.473
vs 0.427 for the names, 0.510 vs 0.442 for the basket, on the session after); rescaling
to a full session would raise it more. What does run high is the forecast *of* a
half-day (realised / forecast 0.62): nothing here knows the session is short.

**Walk-forward.** Refit at each quarter start on rows whose whole target window ended
before the refit day. A row whose 3-session target straddles the refit would train on
the first days being forecast.
"""

from datetime import date

import numpy as np
import pandas as pd

from icaif import calendar

HORIZONS = (1, 3)
MARKET = "_MKT"
HAR_COLUMNS = ["x_d", "x_w", "x_m"]
# A window with more than 40% holes is a different window: a 22-session mean over 8
# sessions is a 2-week mean labelled a monthly one.
MIN_WINDOW_SHARE = 0.6
# One missing session blanks two RV sessions (its own and the next gap), so the lag
# reaches three back; further, and "yesterday" is last week.
LAG_FALLBACK = 3
# Fewer complete rows than this and a refit would be a few points' noise presented as
# a coefficient; the caller has asked for a test year its data cannot support.
MIN_TRAIN_ROWS = 100


def _session_closes(close: pd.DataFrame) -> pd.DataFrame:
    """Each session's close: the last bar row of the day, NaN where a name lacks it.

    Not `groupby().last()`, which skips NaN: a name missing its 15:30 bar would take its
    14:30 close, and the next gap would quietly carry the last hour's move as overnight.
    """
    last = close.groupby(pd.Index(close.index.date)).tail(1)
    return last.set_axis(pd.Index(last.index.date))


def _spans_session(bars: pd.DataFrame) -> pd.DataFrame:
    """Session x ticker: do this name's bars cover the whole calendar session?"""
    b = bars.drop_duplicates(["ticker", "start"])
    covered = (b["end"] - b["start"]).groupby([b["start"].dt.date, b["ticker"]]).sum()
    covered = covered.unstack("ticker")
    length = pd.Series([calendar.at(d, calendar.session_close(d))
                        - calendar.at(d, calendar.SESSION_OPEN) for d in covered.index],
                       index=covered.index)
    return covered.eq(length, axis=0)


def realised_variance(info_bars: pd.DataFrame) -> pd.DataFrame:
    """Session x ticker RV, plus a `_MKT` column for the equal-weight basket."""
    b = info_bars.sort_values("end")
    close = b.pivot_table(index="end", columns="ticker", values="close", aggfunc="last")
    opens = b.pivot_table(index="start", columns="ticker", values="open", aggfunc="first")
    day = pd.Index(close.index.date)
    lr = np.log(close).groupby(day).diff()
    # First bar of each session: return from its own open, not from yesterday's close
    # (that part is the overnight gap, counted separately).
    first_end = close.groupby(day).head(1).index
    first_open = opens.groupby(pd.Index(opens.index.date)).head(1)
    first_open.index = first_end
    lr.loc[first_end] = np.log(close.loc[first_end] / first_open)

    sess_open = first_open.set_axis(pd.Index(first_end.date))
    gap = np.log(sess_open / _session_closes(close).shift())

    intraday = (lr ** 2).groupby(day).sum(min_count=1)
    rv = intraday + gap.reindex(intraday.index) ** 2

    n_bars = lr.notna().groupby(day).sum()
    full = n_bars.max(axis=1)
    spans = _spans_session(b).reindex(index=rv.index, columns=rv.columns, fill_value=False)
    rv = rv.where(n_bars.ge(full, axis=0) & spans.astype(bool))

    mkt_bar = lr.mean(axis=1)
    mkt_gap = gap.mean(axis=1)
    mkt = (mkt_bar ** 2).groupby(day).sum() + mkt_gap.reindex(rv.index) ** 2
    rv[MARKET] = mkt.where(rv.notna().all(axis=1))
    rv.index = pd.Index(rv.index, name="session")
    return rv


def _mean_available(lag: pd.DataFrame, n: int) -> pd.DataFrame:
    return lag.rolling(n, min_periods=int(np.ceil(MIN_WINDOW_SHARE * n))).mean()


def _har_frame(rv: pd.DataFrame, h: int) -> pd.DataFrame:
    """Rows (session D, ticker): regressors known before D opens, and D's target."""
    lag = rv.shift(1)
    x_d = np.log(lag.ffill(limit=LAG_FALLBACK - 1))
    x_w = np.log(_mean_available(lag, 5))
    x_m = np.log(_mean_available(lag, 22))
    x_20 = np.log(_mean_available(lag, 20))
    fwd = rv[::-1].rolling(h, min_periods=h).mean()[::-1]
    y = np.log(fwd)
    frame = pd.concat({k: v.stack(future_stack=True) for k, v in
                       {"x_d": x_d, "x_w": x_w, "x_m": x_m, "x_20": x_20, "y": y}.items()},
                      axis=1)
    frame.index.names = ["session", "ticker"]
    sessions = list(rv.index)
    pos = {d: i for i, d in enumerate(sessions)}
    end_pos = frame.index.get_level_values("session").map(pos) + h - 1
    frame["target_end"] = [sessions[p] if p < len(sessions) else None for p in end_pos]
    return frame.replace([np.inf, -np.inf], np.nan)


def _fit(train: pd.DataFrame, refit: pd.Timestamp) -> tuple[np.ndarray, float]:
    """Coefficients and the smearing factor, the mean of exp(residual)."""
    t = train.dropna(subset=HAR_COLUMNS + ["y"])
    if len(t) < MIN_TRAIN_ROWS:
        raise ValueError(f"refit {refit.date()}: {len(t)} complete training rows, "
                         f"need {MIN_TRAIN_ROWS}; start the test later")
    X = np.column_stack([np.ones(len(t)), t[HAR_COLUMNS]])
    beta, *_ = np.linalg.lstsq(X, t["y"].to_numpy(), rcond=None)
    resid = t["y"].to_numpy() - X @ beta
    return beta, float(np.exp(resid).mean())


def _predict(frame: pd.DataFrame, pool: np.ndarray, test: np.ndarray,
             refit: pd.Timestamp) -> pd.DataFrame:
    """Log fit and smearing factor for the `test` rows, from the refit at `refit`.

    The one place a forecast is made, so the walk-forward that is scored and the live
    call that trades cannot drift apart.
    """
    tend = pd.to_datetime(frame["target_end"])
    beta, smear = _fit(frame[pool & (tend < refit)], refit)
    rows = frame[test]
    X = np.column_stack([np.ones(len(rows)), rows[HAR_COLUMNS]])
    return pd.DataFrame({"log": X @ beta, "smear": smear}, index=rows.index)


def _pools(frame: pd.DataFrame) -> list[np.ndarray]:
    is_mkt = np.asarray(frame.index.get_level_values("ticker") == MARKET)
    return [m for m in (~is_mkt, is_mkt) if m.any()]


def walk_forward(rv: pd.DataFrame, first_test: str = "2022-01-01") -> pd.DataFrame:
    """Out-of-sample forecasts of mean daily variance, per horizon.

    Returns rows (session, ticker) with `har_h{H}`, the trailing baselines `rw5_h{H}`
    and `rv20_h{H}` (mean RV over the last 5 and 20 sessions) and the realised
    `y_h{H}`, all in mean daily variance. `harlog_h{H}` is the fit in logs before the
    bias correction: the forecast of log RV itself, which a log-space R^2 must use, or
    it charges HAR a bias it does not have. The other baseline,
    `close_to_close_variance`, is joined by the caller.
    """
    out = []
    sessions = pd.to_datetime(pd.Index(rv.index))
    refits = pd.date_range(first_test, sessions.max(), freq="QS")
    for h in HORIZONS:
        frame = _har_frame(rv, h)
        sess = pd.to_datetime(frame.index.get_level_values("session"))
        preds = []
        for pool in _pools(frame):
            for i, start in enumerate(refits):
                stop = refits[i + 1] if i + 1 < len(refits) else sessions.max() + pd.Timedelta(days=1)
                test = pool & np.asarray((sess >= start) & (sess < stop))
                if test.any():
                    preds.append(_predict(frame, pool, test, start))
        p = pd.concat(preds).sort_index()
        f = frame.loc[p.index]
        out.append(pd.DataFrame({f"har_h{h}": np.exp(p["log"]) * p["smear"],
                                 f"harlog_h{h}": p["log"],
                                 f"rw5_h{h}": np.exp(f["x_w"]),
                                 f"rv20_h{h}": np.exp(f["x_20"]),
                                 f"y_h{h}": np.exp(f["y"])}))
    return pd.concat(out, axis=1)


def _session_to_forecast(as_of: pd.Timestamp) -> date:
    """as_of's own date while its session is still to close, else the next weekday."""
    d = as_of.date()
    if d.weekday() < 5 and as_of < calendar.at(d, calendar.session_close(d)):
        return d
    return (pd.Timestamp(d) + pd.offsets.BDay(1)).date()


def forecast_next(info_bars: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Live: the HAR forecast for the session in progress, or the next one to open.

    Returns one row per name plus `_MKT`, with `har_h1` and `har_h3` in mean daily
    variance (take the square root for daily vol), the same numbers `walk_forward`
    scores for that session. `attrs` carries the session and the refit used.

    Only sessions that had closed by `as_of` are read, so every deadline of a day gets
    the forecast made before its open: a half-finished session would otherwise read as a
    hole, or its first hours as a whole day. The refit is the latest quarter start on or
    before the session, trained on targets that ended before it, as in the walk-forward.

    Raises if the last closed session is more than 4 days before the one forecast: a feed
    that stopped would otherwise pass off a week-old lag as yesterday. A single missing
    weekday is not caught here (there is no holiday calendar); it reads as a gap.
    """
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware (America/New_York)")
    day = info_bars["start"].dt.date
    closes = pd.Series({d: calendar.at(d, calendar.session_close(d)) for d in day.unique()})
    closed = info_bars[day.map(closes).le(as_of).to_numpy()]
    rv = realised_variance(closed)
    session = _session_to_forecast(as_of)
    last = rv.index.max()
    if (session - last).days > 4:
        raise ValueError(f"last closed session {last} is stale for a forecast of {session}")
    rv.loc[session] = np.nan
    refit = pd.Timestamp(session).to_period("Q").start_time
    out = {}
    for h in HORIZONS:
        frame = _har_frame(rv, h)
        target = np.asarray(frame.index.get_level_values("session") == session)
        p = pd.concat([_predict(frame, pool, pool & target, refit) for pool in _pools(frame)])
        out[f"har_h{h}"] = (np.exp(p["log"]) * p["smear"]).droplevel("session")
    res = pd.DataFrame(out)
    res.attrs.update(session=session, refit=refit.date())
    return res


def close_to_close_variance(info_bars: pd.DataFrame) -> pd.DataFrame:
    """20-session variance of daily close-to-close log returns, lagged one session."""
    b = info_bars.sort_values("end")
    close = b.pivot_table(index="end", columns="ticker", values="close", aggfunc="last")
    lr = np.log(_session_closes(close)).diff()
    lr[MARKET] = lr.mean(axis=1).where(lr.notna().all(axis=1))
    var = lr.rolling(20, min_periods=20).var().shift(1)
    var.index = pd.Index(var.index, name="session")
    return var


def qlike(realised: pd.Series, forecast: pd.Series) -> pd.Series:
    """Patton (2011) QLIKE: robust to noise in the realised proxy; 0 at a perfect forecast."""
    r = realised / forecast
    return r - np.log(r) - 1


def oos_r2(actual: pd.Series, forecast: pd.Series) -> float:
    """1 - SSE / SST around the sample's own mean. Bias counts against the forecast."""
    return float(1 - ((actual - forecast) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())
