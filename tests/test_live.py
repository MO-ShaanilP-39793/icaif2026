"""The live dry run, offline: every network door replaced by a small synthetic world."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, compiler, data, external, kit, live, sim
from icaif import quant_strategies as qs

OURS = list(data.load_universe())
OTHERS = [f"X{i:02d}" for i in range(12)]
# The frozen 2026 daily model's features, in its order (predictor.features()).
MODEL_FEATURES = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d", "mom_12_1",
    "vol_5d", "vol_20d", "vol_60d", "parkinson_20d", "gap_last", "overnight_share_20d",
    "volume_1d_ratio", "dist_high_20d", "dist_low_20d", "dist_high_250d", "z_5d", "vol_ratio",
    "ctx_vix", "ctx_vix_chg_5d", "ctx_spy_ret_1d", "ctx_spy_ret_5d", "ctx_spy_ret_20d",
    "ctx_spy_vol_20d", "ctx_rate_10y", "ctx_curve_10y_3m", "ctx_rate_10y_chg_20d",
    "ctx_sector_dispersion_5d", "ctx_breadth_1d", "ctx_dispersion_1d", "ctx_weekday",
    "is_competition", "e_sessions_since", "e_sessions_to_next", "e_last_reaction",
]
# Sunday 2026-09-27: the last completed session is Friday the 25th, the decision Monday.
NOW = pd.Timestamp("2026-09-27 12:00", tz="America/New_York")
LATEST = pd.Timestamp("2026-09-25")


def _walk(names, end=LATEST, n=340, seed=0):
    rng = np.random.default_rng(seed)
    dates = live.sessions(end - pd.Timedelta(days=int(n * 1.6) + 30), end)[-n:]
    frames = []
    for s in names:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": s, "open": open_,
            "high": np.maximum(open_, close) * 1.005, "low": np.minimum(open_, close) * 0.995,
            "close": close, "adj_close": close, "volume": rng.uniform(1e6, 2e6, n)}))
    return pd.concat(frames, ignore_index=True)


class FakePredictor:
    """Scores by momentum, so the prediction varies across names like a real one."""

    def __init__(self, features=MODEL_FEATURES):
        self._features = list(features)

    def features(self):
        return self._features

    def predict(self, frame):
        return pd.Series(frame["ret_20d"].fillna(0).to_numpy() + 0.5)


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Replace the fetchers; `world["end"]` moves the feed's last bar."""
    state = {"end": LATEST}
    membership = pd.DataFrame({"ticker": OURS + OTHERS, "symbol": OURS + OTHERS,
                               "start": pd.Timestamp("2000-01-01"), "end": pd.NaT})

    def fake_yahoo(symbols, start):
        ctx = set(symbols) == set(external.CONTEXT_SYMBOLS)
        bars = _walk(symbols, end=state["end"], n=state.get("n", 340), seed=1 if ctx else 0)
        if ctx and state.get("holiday_vix"):
            vix = bars[bars["ticker"] == "^VIX"].tail(1).assign(date=pd.Timestamp("2026-09-07"))
            bars = pd.concat([bars, vix], ignore_index=True).sort_values(["date", "ticker"])
        return bars, []

    def fake_events(symbols, now):
        ev = pd.DataFrame({"ticker": OURS[:5], "accepted": pd.Timestamp(
            "2026-08-05 16:30", tz="America/New_York")})
        return ev, {"source": "fixture", "stale": True, "path": "fixture"}

    monkeypatch.setattr(live, "yahoo_daily", fake_yahoo)
    monkeypatch.setattr(live, "load_membership", lambda: membership)
    monkeypatch.setattr(live, "load_events", fake_events)
    monkeypatch.setattr(live, "LIVE_OUT", tmp_path)
    return state


def test_a_stale_feed_raises_rather_than_scoring_old_prices(world):
    """Yahoo returning Thursday's bars on Sunday gives ordinary-looking features from old
    prices, and the model scores them as if they were Friday's."""
    world["end"] = LATEST - pd.Timedelta(days=1)
    with pytest.raises(live.StaleDataError, match="2026-09-24"):
        live.daily_scores(NOW, predictor=FakePredictor())


def test_todays_in_progress_bar_is_dropped_so_the_decision_row_reads_only_the_prior_close(world):
    """Called during the session, Yahoo's daily download ends in a partial bar for today.
    Kept, it would become the decision date's own close, the leak every backtest avoids."""
    during = pd.Timestamp("2026-09-28 10:00", tz="America/New_York")
    clean = live.daily_scores(during, predictor=FakePredictor())
    world["end"] = pd.Timestamp("2026-09-28")        # the feed now carries today's bar
    leaky = live.daily_scores(during, predictor=FakePredictor())
    assert clean.decision_date == leaky.decision_date == pd.Timestamp("2026-09-28")
    assert leaky.inputs.meta["partial_bars_dropped"] > 0
    assert leaky.inputs.daily["date"].max() == LATEST


@pytest.mark.parametrize("features", [
    MODEL_FEATURES[:-1],                                          # a column dropped
    MODEL_FEATURES[1:2] + MODEL_FEATURES[:1] + MODEL_FEATURES[2:],  # two swapped
    MODEL_FEATURES + ["sector_ret_5d"],                           # a column the frame lacks
])
def test_a_feature_frame_that_is_not_the_models_exact_columns_raises(world, features):
    """AutoGluon picks columns by name, so a missing or renamed feature is scored as NaN
    rather than refused, and a model fed NaN still returns a plausible number."""
    with pytest.raises(live.FeatureMismatchError):
        live.daily_scores(NOW, predictor=FakePredictor(features))


def test_a_dry_envelope_carries_all_30_symbols_and_cannot_name_a_real_round():
    """The kit rejects a decision missing a zero-weight symbol, and the first in-window
    upload consumes the round even when invalid. A dry run's file must be one the kit
    refuses outright: placeholder credentials and a round id no schedule contains."""
    row = {"id": "validation-2026-10-08-r1", "phase": "validation"}
    w = {t: 0.0 for t in OURS}
    decision = live.envelope(w, row)
    assert set(decision) == set(kit.contracts.FIELDS["decision"])
    assert list(decision["weights"]) == list(json.load(open(live.DECISION_TEMPLATE))["weights"])
    assert decision["round_id"] == "dryrun-validation-2026-10-08-r1"
    assert kit.contracts.is_placeholder(decision["team_token"])
    real = live.envelope(w, row, team={"team_id": "t-1", "team_token": "tok"})
    assert real["round_id"] == row["id"] and real["team_token"] == "tok"
    assert live.envelope(w, {"id": "rehearsal-2026-10-02-r1", "phase": "rehearsal-2026-10-02"})["phase"] \
        == live.DRY_RUN_PHASE


def test_a_constant_prediction_raises_instead_of_letting_vol_alone_pick_the_book(world):
    """A model fed all-NaN rows returns one value for every name; the compiler would then
    select by volatility and log it as the model's choice."""
    flat = FakePredictor()
    flat.predict = lambda frame: pd.Series(0.5, index=range(len(frame)))
    with pytest.raises(live.LiveDataError, match="constant"):
        live.daily_scores(NOW, predictor=flat)


def test_a_context_print_on_an_equity_holiday_does_not_blank_the_spy_windows(world):
    """Yahoo's ^VIX printed on Labor Day 2026. Kept, that row gives SPY a NaN return and
    ctx_spy_ret_20d / ctx_spy_vol_20d go NaN for 20 sessions, an input the model never saw."""
    world["holiday_vix"] = True
    scored = live.daily_scores(NOW, predictor=FakePredictor())
    assert scored.inputs.meta["context_off_session_dates_dropped"] == ["2026-09-07"]
    assert scored.features[["ctx_spy_ret_20d", "ctx_spy_vol_20d"]].notna().all().all()


def test_a_holiday_rolls_the_decision_to_the_next_session():
    """Good Friday 2026 is a weekday with no session: decided on the Thursday evening,
    the next session is Monday, and the latest completed close is Thursday's."""
    now = pd.Timestamp("2026-04-02 20:00", tz="America/New_York")
    assert live.latest_completed_session(now) == pd.Timestamp("2026-04-02")
    assert live.next_session(pd.Timestamp("2026-04-02")) == pd.Timestamp("2026-04-06")
    # Before the close settles, today's session is not complete yet.
    assert live.latest_completed_session(pd.Timestamp("2026-04-02 16:10", tz="America/New_York")) \
        == pd.Timestamp("2026-04-01")


# ----------------------------------------------------------------------------- the desk's market

def test_the_rule_desk_needs_every_name_on_the_latest_session(world, monkeypatch):
    """The entry is sized on the last 60 sessions' covariance and valued on the latest
    close; a name a day behind would be sized on prices the others have moved past."""
    world["n"] = 800
    daily, meta = live.fetch_closes(NOW)
    assert daily["date"].max() == LATEST and meta["sessions"] == 800
    real = live.yahoo_daily

    def one_late(symbols, start):
        bars, missing = real(symbols, start)
        late = (bars["ticker"] == "NVDA") & (bars["date"] == LATEST)
        return bars[~late], missing

    monkeypatch.setattr(live, "yahoo_daily", one_late)
    with pytest.raises(live.StaleDataError, match="NVDA"):
        live.fetch_closes(NOW)


def test_too_short_a_history_raises_rather_than_fitting_the_regime_on_less(world):
    """The regime model and its reference vol are fit on three years; on less they are
    a different model, and the exposure they set would look as ordinary as any other."""
    world["n"] = 600
    with pytest.raises(live.LiveDataError, match="sessions"):
        live.fetch_closes(NOW)


def _today_30m(day, upto, shock=1.0):
    rows = []
    t = calendar.at(day, calendar.SESSION_OPEN)
    while t + pd.Timedelta(minutes=30) <= upto:
        for k, s in enumerate(OURS):
            px = 100.0 * (shock if s == "AAPL" else 1.0)
            rows.append({"ticker": s, "start": t, "end": t + pd.Timedelta(minutes=30),
                         "open": px, "high": px, "low": px, "close": px, "volume": 1e5,
                         "source": "yahoo_30m"})
        t += pd.Timedelta(minutes=30)
    return pd.DataFrame(rows)


def test_the_live_market_serves_completed_sessions_as_closes_and_today_only_as_bars(world):
    """The desk reads daily closes for shape, regime and sigma, and the latest bar for the
    move since the prior close. Today's bars as a close would make that move zero."""
    world["n"] = 800
    daily, _ = live.fetch_closes(NOW)
    day = date(2026, 9, 28)
    deadline = calendar.at(day, calendar.ROUNDS[4][0])
    today = live.today_60m(_today_30m(day, deadline - pd.Timedelta(minutes=12)), day)
    m = live.market(daily, today, live.market_days(daily, LATEST))
    ctx = sim.RoundContext(day, 4, deadline, calendar.at(day, calendar.ROUNDS[4][1]), {}, 1e6, m)
    closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
    want = daily.pivot(index="date", columns="ticker", values="close").tail(qs.HISTORY_DAYS + 1)
    assert closes.index[-1].date() == LATEST.date() and len(closes) == qs.HISTORY_DAYS + 1
    np.testing.assert_allclose(closes[OURS].to_numpy(), want[OURS].to_numpy())
    assert ctx.recent_closes(1).index[-1] == calendar.at(day, calendar.ROUNDS[3][1])  # 11:30's close
    assert day in m.days and max(m.days) > day   # sessions ahead, for the FOMC count


def test_an_hour_missing_a_half_is_dropped_not_closed_on_the_other_half():
    """With one half missing, a 60m bar's close would be the other half's, 30 minutes off."""
    day = date(2026, 9, 28)
    bars = _today_30m(day, calendar.at(day, calendar.ROUNDS[4][1]))
    holed = bars[~((bars["ticker"] == "MSFT") & (bars["start"] == calendar.at(day, calendar.ROUNDS[2][1])
                                                   + pd.Timedelta(minutes=30)))]
    got = live.today_60m(holed, day)
    hours = got.groupby("ticker").size()
    assert hours["MSFT"] == hours["AAPL"] - 1
    assert live.fills(bars).loc[calendar.at(day, calendar.ROUNDS[2][1]), "AAPL"] == 100.0


def test_scheduled_earnings_read_as_sessions_to_the_reaction():
    """An after-close release reacts at the next open, a pre-open one that morning. A date
    with no time is read as before the open: a flag a session early costs one analyst
    call, a flag a session late is a gap the book already took."""
    days = [d.date() for d in live.sessions("2026-09-21", "2026-10-30")]
    cal = pd.DataFrame({"ticker": ["NKE", "JPM", "GS"],
                        "date": pd.to_datetime(["2026-10-01", "2026-10-13", "2026-10-13"]),
                        "side": ["amc", "bmo", "unknown"]})
    e = live.CalendarEarnings(cal, days)
    assert e.to_next(date(2026, 10, 1)) == {"NKE": 1, "JPM": 8, "GS": 8}
    assert e.to_next(date(2026, 10, 12)) == {"JPM": 1, "GS": 1}
    assert e.to_next(date(2026, 9, 21)) == {"NKE": 9}


def test_the_earnings_snapshot_fallback_never_reads_the_calendar_file(tmp_path, monkeypatch):
    """earnings_calendar_<date> sorts after every dated EDGAR snapshot, and the fallback
    globbed earnings_*: without SEC_USER_AGENT it read Yahoo's scheduled dates as EDGAR
    releases and the scorer died on a missing column."""
    from icaif import universe

    edgar = pd.DataFrame({"ticker": ["AAPL"], "accepted": [pd.Timestamp("2026-07-30 16:30", tz="America/New_York")]})
    edgar.to_parquet(tmp_path / "earnings_2026-09-27.parquet")
    pd.DataFrame({"ticker": ["AAPL"], "date": [pd.Timestamp("2026-10-29")], "side": ["amc"]}).to_parquet(
        tmp_path / "earnings_calendar_2026-09-28.parquet")
    monkeypatch.setattr(universe, "EXTERNAL", tmp_path)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    events, meta = live.load_events(["AAPL"], NOW)
    assert meta["path"].endswith("earnings_2026-09-27.parquet") and len(events) == 1


def test_archived_scores_are_the_panel_the_shadow_agent_reads(world, tmp_path):
    """The scorer runs in a child and hands its scores over as a file; a file the runner
    reads under another date or shape would leave the agent with no scores, quietly."""
    scored = live.daily_scores(NOW, predictor=FakePredictor())
    live.archive_scores(scored, tmp_path)
    s = pd.read_parquet(tmp_path / "scores.parquet")
    panel = compiler.DailyPanel(s.pivot(index="date", columns="ticker", values="pred"), sorted(OURS))
    day = scored.decision_date.date()
    got = panel.for_day(day, calendar.at(day, calendar.ROUNDS[1][0]))
    pd.testing.assert_series_equal(got.sort_index(), scored.ours.reindex(sorted(OURS)).sort_index(),
                                   check_names=False)


def test_live_edgar_adds_recent_filings_to_the_snapshot_without_doubling_any(tmp_path, monkeypatch):
    """The full EDGAR history took 4m43s live, past the scorer's watchdog, so live asks
    only for recent filings. Joined to the snapshot, a release both carry must count
    once (two copies would cluster into one quarter, but only by luck of the gap)."""
    from icaif import earnings, universe

    ny = "America/New_York"
    old = pd.Timestamp("2026-07-30 16:30", tz=ny)
    pd.DataFrame({"ticker": ["AAPL", "MSFT"], "accepted": [old, pd.Timestamp("2026-07-29 16:05", tz=ny)]}
                 ).to_parquet(tmp_path / "earnings_2026-09-01.parquet")
    calls = []

    def fake_fetch(symbols, sleep=0.12, recent_only=False):
        calls.append(recent_only)
        return pd.DataFrame({"ticker": ["AAPL", "AAPL", "AAPL"],
                             "accepted": [old, pd.Timestamp("2026-09-25 16:30", tz=ny),
                                          pd.Timestamp("2026-09-28 16:30", tz=ny)]}), []

    monkeypatch.setattr(universe, "EXTERNAL", tmp_path)
    monkeypatch.setattr(earnings, "fetch", fake_fetch)
    monkeypatch.setenv("SEC_USER_AGENT", "test test@example.com")
    events, meta = live.load_events(["AAPL", "MSFT"], NOW)
    assert calls == [True] and meta["source"] == "edgar" and not meta["stale"]
    aapl = events.loc[events["ticker"] == "AAPL", "accepted"].tolist()
    assert aapl == [old, pd.Timestamp("2026-09-25 16:30", tz=ny)]   # once each; none after NOW
    assert (events["ticker"] == "MSFT").sum() == 1


# ----------------------------------------------------------------------------- HAR vol

def _bars_30m(first="2021-07-01", last="2022-05-20", seed=4):
    """A random walk of 30m bars on the live grid, full sessions only."""
    rng = np.random.default_rng(seed)
    frames, px = [], np.full(len(OURS), 100.0)
    for d in pd.bdate_range(first, last).date:
        if d in calendar.EARLY_CLOSES:
            continue
        starts = pd.date_range(calendar.at(d, calendar.SESSION_OPEN),
                               calendar.at(d, calendar.session_close(d)), freq="30min")[:-1]
        px = px * np.exp(rng.normal(0, 0.01, len(OURS)))
        o, scale = px.copy(), np.exp(rng.normal(0, 0.4))
        for s in starts:
            c = o * np.exp(rng.normal(0, 0.004 * scale, len(OURS)))
            frames.append(pd.DataFrame({"ticker": OURS, "start": s, "end": s + pd.Timedelta(minutes=30),
                                        "open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                                        "close": c, "volume": 1.0, "source": "yahoo_30m"}))
            o = c
        px = o
    return pd.concat(frames, ignore_index=True)


def test_the_live_har_forecast_reads_no_later_bar_prefers_the_fetch_and_is_made_once_a_day(
        monkeypatch, tmp_path):
    """The live forecast joins two feeds. It must read nothing that ended after the
    deadline, take every session the fresh fetch covers from the fetch (a snapshot's
    copy of a recent day may be partial or stale), and be made once: later rounds of
    the day read the file, so a fetch failing at round 5 cannot blank the forecast."""
    from icaif import alpaca, vol
    from icaif.agents import signals

    bars = _bars_30m()
    session = date(2022, 5, 16)
    deadline = calendar.at(session, calendar.ROUNDS[1][0])
    cut = date(2022, 3, 1)
    clean_60m = alpaca.to_60m(bars[bars["end"] <= deadline])
    archive = alpaca.to_60m(bars[bars["start"].dt.date < date(2022, 4, 1)])
    # The archive's overlap with the fetch is wrong on purpose: the fetch must win it.
    overlap = archive["start"].dt.date >= cut
    archive.loc[overlap, ["open", "high", "low", "close"]] *= np.exp(
        np.random.default_rng(9).normal(0, 0.05, (int(overlap.sum()), 1)))
    monkeypatch.setattr(live, "vol_archive", lambda: archive)
    fetched = bars[bars["start"].dt.date >= cut]  # includes bars after the deadline

    got = live.vol_forecasts(deadline, tmp_path, OURS,
                             bars=lambda now: live.vol_bars(now, recent_30m=fetched))
    want = vol.forecast_next(clean_60m, deadline)
    row = got.for_day(session, deadline)
    for h in vol.HORIZONS:
        pd.testing.assert_series_equal(row[f"har_h{h}"], want[f"har_h{h}"].reindex(row.index),
                                       check_names=False)
    assert json.loads((tmp_path / "har_meta.json").read_text())["session"] == str(session)

    def offline(now):
        raise AssertionError("fetched again")

    later = calendar.at(session, calendar.ROUNDS[5][0])
    again = live.vol_forecasts(later, tmp_path, OURS, bars=offline)
    pd.testing.assert_frame_equal(again.for_day(session, later), row)
    assert isinstance(again, signals.VolForecasts)
