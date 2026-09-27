from datetime import date

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, data, vol

TICKERS = sorted(data.load_universe())


def _starts(d, grid):
    """Bar starts for a session: the organizer grid (09:30, 10:00, 11:00 ...) or the
    live :30 grid (09:30, 10:30 ...) that `markets.intraday_info_bars` serves."""
    close = calendar.at(d, calendar.session_close(d))
    if grid == "live":
        starts = pd.date_range(calendar.at(d, calendar.SESSION_OPEN), close, freq="60min")
    else:
        starts = [calendar.at(d, calendar.SESSION_OPEN)] + list(
            pd.date_range(calendar.at(d, pd.Timestamp("10:00").time()), close, freq="60min"))
    return [s for s in starts if s < close], close


def _bars(days, closes_fn, grid="organizer"):
    """closes_fn(day_index, bar_index, ticker) -> close; each bar opens at the prior close."""
    rows = []
    for di, d in enumerate(days):
        starts, close = _starts(d, grid)
        for t in TICKERS:
            prev = None
            for bi, s in enumerate(starts):
                c = closes_fn(di, bi, t)
                o = c if prev is None else prev
                end = min(starts[bi + 1] if bi + 1 < len(starts) else close, close)
                rows.append({"ticker": t, "start": s, "end": end, "open": o, "high": max(o, c),
                             "low": min(o, c), "close": c, "volume": 1.0})
                prev = c
    return pd.DataFrame(rows)


def _walk(d, bi, t):
    return 100.0 * (1.01 ** bi) * (1.02 ** d)


def test_realised_variance_is_intraday_squared_returns_plus_the_squared_gap():
    """Day 2 opens 1% above day 1's close and moves +2% in one bar. Its RV is
    ln(1.01)^2 + ln(1.02)^2, with the overnight part counted exactly once."""
    days = [date(2026, 9, 21), date(2026, 9, 22)]

    def px(di, bi, t):
        if di == 0:
            return 100.0
        return 101.0 if bi == 0 else 101.0 * 1.02

    for grid in ("organizer", "live"):
        rv = vol.realised_variance(_bars(days, px, grid))
        assert rv.loc[days[1], "AAPL"] == pytest.approx(np.log(1.01) ** 2 + np.log(1.02) ** 2)


def test_a_session_with_no_prior_close_gets_no_variance_rather_than_a_gapless_one():
    """With no prior close the gap is unknown, not zero. Counting it as zero would
    make the first session, and the day after a name's missing session, read calm; a
    gap from two sessions back would count two sessions' move as one night's. Nor may
    a name missing its last bar lend its 14:30 close to the next day's gap."""
    days = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
    bars = _bars(days, _walk, "live")
    on = bars["start"].dt.date == days[1]
    whole_day = (bars["ticker"] == "AAPL") & on
    last_bar = (bars["ticker"] == "MSFT") & on & (bars["start"].dt.hour == 15)
    bars = bars[~(whole_day | last_bar)]
    rv = vol.realised_variance(bars)
    assert rv.loc[days[0]].isna().all()
    assert np.isnan(rv.loc[days[2], "AAPL"])
    assert np.isnan(rv.loc[days[2], "MSFT"])
    assert rv.loc[days[2], "NVDA"] > 0
    assert rv.loc[days[3], ["AAPL", "MSFT"]].gt(0).all()


def test_a_session_missing_bars_gets_no_variance_rather_than_a_calm_one():
    """A name missing its 12:00 bar has one squared return fewer. The basket averaged
    over the 29 names that remain is a different basket, so it goes too."""
    days = [date(2026, 9, 21), date(2026, 9, 22)]
    bars = _bars(days, _walk)
    bars = bars.drop(bars[(bars["ticker"] == "AAPL") & (bars["start"].dt.hour == 12)
                          & (bars["start"].dt.date == days[1])].index)
    rv = vol.realised_variance(bars)
    assert np.isnan(rv.loc[days[1], "AAPL"])
    assert rv.loc[days[1], "MSFT"] > 0
    assert np.isnan(rv.loc[days[1], vol.MARKET])


def test_a_half_day_missing_its_last_half_hour_for_every_name_is_not_a_calm_session():
    """Yahoo has no 12:30-13:00 bar on half-days, for all 30 names at once, so no peer
    shows the hole. A complete half-day (four bars on the live grid, the last 30
    minutes) is a real session; the truncated one would lose its last move silently."""
    days = [date(2025, 11, 26), date(2025, 11, 28)]
    bars = _bars(days, _walk, "live")
    assert (bars["start"].dt.date == days[1]).sum() == 4 * len(TICKERS)
    rv = vol.realised_variance(bars)
    assert rv.loc[days[1], "AAPL"] == pytest.approx(3 * np.log(1.01) ** 2
                                                   + np.log(1.02 / 1.01 ** 6) ** 2)
    truncated = bars.drop(bars[bars["start"] == calendar.at(days[1], pd.Timestamp("12:30").time())].index)
    assert vol.realised_variance(truncated).loc[days[1]].isna().all()


def _synthetic_rv(seed=0):
    sessions = pd.bdate_range("2021-01-04", "2022-09-30").date
    rng = np.random.default_rng(seed)
    return pd.DataFrame(np.exp(rng.normal(-9, 0.5, (len(sessions), 3))),
                        index=pd.Index(sessions, name="session"), columns=["A", "B", vol.MARKET])


def test_a_forecast_does_not_move_when_the_future_is_rewritten():
    """Rewrite every session from D on (RV x10). Forecasts for sessions up to and
    including D must not change: their regressors end at D-1, and the coefficients were
    fit on targets that ended before the quarter began."""
    rv = _synthetic_rv()
    cut = date(2022, 5, 16)
    later = rv.copy()
    later.loc[later.index >= cut] *= 10
    a = vol.walk_forward(rv)
    b = vol.walk_forward(later)
    upto = a.index.get_level_values("session") <= cut
    for h in vol.HORIZONS:
        pd.testing.assert_series_equal(a.loc[upto, f"har_h{h}"], b.loc[upto, f"har_h{h}"])
    assert not a.loc[~upto, "har_h1"].equals(b.loc[~upto, "har_h1"])


def test_the_market_forecast_is_not_fit_on_the_stocks():
    """Pooled with the 30 names, the basket gets 1/31 of the fit and their level and
    persistence instead of its own. Rewriting every stock must leave it untouched."""
    rv = _synthetic_rv()
    louder = rv.copy()
    louder[["A", "B"]] = louder[["A", "B"]] ** 0.5 * 3e-4
    a = vol.walk_forward(rv).xs(vol.MARKET, level="ticker")
    b = vol.walk_forward(louder).xs(vol.MARKET, level="ticker")
    pd.testing.assert_frame_equal(a, b)
    assert not vol.walk_forward(rv)["har_h1"].equals(vol.walk_forward(louder)["har_h1"])


def test_a_test_year_the_history_cannot_fit_raises_rather_than_forecasting_nan():
    """An empty training set fits beta = 0 with a NaN variance: every forecast of the
    quarter NaN, dropped downstream, and the year's verdict drawn from what survived."""
    with pytest.raises(ValueError, match="training rows"):
        vol.walk_forward(_synthetic_rv(), first_test="2021-01-01")


def _random_bars(first="2021-01-04", last="2022-06-30", seed=1):
    """A random walk on the live grid, half-days included: long enough to refit."""
    rng = np.random.default_rng(seed)
    frames, px = [], np.full(len(TICKERS), 100.0)
    for d in pd.bdate_range(first, last).date:
        starts, close = _starts(d, "live")
        ends = starts[1:] + [close]
        px = px * np.exp(rng.normal(0, 0.01, len(TICKERS)))  # overnight gap
        o = px.copy()
        scale = np.exp(rng.normal(0, 0.4))  # a vol regime that moves day to day
        for s, e in zip(starts, ends):
            c = o * np.exp(rng.normal(0, 0.006 * scale, len(TICKERS)))
            frames.append(pd.DataFrame({"ticker": TICKERS, "start": s, "end": e, "open": o,
                                        "high": np.maximum(o, c), "low": np.minimum(o, c),
                                        "close": c, "volume": 1.0}))
            o = c
        px = o
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def random_bars():
    return _random_bars()


def test_one_missing_session_does_not_blank_a_name_for_weeks(random_bars):
    """A day's bars missing for one name cost it two RV sessions (the day and the next
    gap). With strict 22-session windows every forecast of that name went NaN for a
    month and a day, and whatever was sized on it fell back or stopped."""
    hole = date(2022, 2, 15)
    bars = random_bars[~((random_bars["ticker"] == "AAPL") & (random_bars["start"].dt.date == hole))]
    rv = vol.realised_variance(bars)
    assert rv.loc[[hole, date(2022, 2, 16)], "AAPL"].isna().all()
    fc = vol.walk_forward(rv).xs("AAPL", level="ticker")
    after = fc.loc[[d for d in fc.index if d > hole]]
    assert after["har_h1"].notna().all()
    assert after["har_h3"].notna().all()


def test_the_live_forecast_is_the_walk_forward_forecast_and_reads_no_later_bar(random_bars):
    """forecast_next is what trades and walk_forward is what was scored. Any drift
    between them (another refit, a regressor window off by one, a session read before
    it closed) would trade a forecast nobody measured. It gets every bar, including
    the future, and must ignore all it could not have seen at the deadline."""
    session = date(2022, 5, 16)
    wf = vol.walk_forward(vol.realised_variance(random_bars)).xs(session, level="session")
    for deadline in ("09:10", "12:25"):
        as_of = calendar.at(session, pd.Timestamp(deadline).time())
        live = vol.forecast_next(random_bars, as_of)
        assert live.attrs["session"] == session
        assert live.notna().all().all()
        for h in vol.HORIZONS:
            pd.testing.assert_series_equal(live[f"har_h{h}"], wf.loc[live.index, f"har_h{h}"],
                                           check_names=False)
        assert set(live.index) == set(TICKERS) | {vol.MARKET}
    after_close = vol.forecast_next(random_bars, calendar.at(date(2022, 5, 13), pd.Timestamp("16:30").time()))
    assert after_close.attrs["session"] == session
    pd.testing.assert_frame_equal(after_close, live)


def test_a_stale_feed_raises_rather_than_passing_last_week_off_as_yesterday(random_bars):
    old = random_bars[random_bars["start"].dt.date <= date(2022, 5, 6)]
    with pytest.raises(ValueError, match="stale"):
        vol.forecast_next(old, calendar.at(date(2022, 5, 16), pd.Timestamp("09:10").time()))
