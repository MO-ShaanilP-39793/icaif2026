"""The live dry run, offline: every network door replaced by a small synthetic world."""

import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from icaif import compiler, data, external, kit, live

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
    dates = live.sessions(end - pd.Timedelta(days=600), end)[-n:]
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
        bars = _walk(symbols, end=state["end"], seed=1 if ctx else 0)
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


def _decide(**kw):
    kw.setdefault("predictor", FakePredictor())
    return live.decide(1, as_of=NOW, **kw)


@pytest.mark.parametrize("levers,portfolio", [
    (compiler.Levers(), None),
    (compiler.Levers(exposure=1.0, top_k=3, weighting="equal"), None),  # every name at the cap
    (compiler.Levers(exposure=1.0, top_k=4), {"weights": {"AAPL": 0.3, "MSFT": 0.3, "NVDA": 0.3}}),
])
def test_the_decision_always_passes_the_organizers_own_weight_check(world, levers, portfolio):
    """A weight like 0.1 + 0.2 fails the backend's Decimal cap and the round silently
    holds. The written file, parsed back as the backend parses it, must pass the kit."""
    decision, log = _decide(levers=levers, portfolio=portfolio)
    written = json.loads(open(log["decision_path"]).read(), parse_float=Decimal)
    kit.validate_weights(written["weights"])
    assert written == json.loads(json.dumps(decision), parse_float=Decimal)
    assert sum(decision["weights"].values()) <= levers.exposure + 1e-9


def test_a_stale_feed_raises_rather_than_scoring_old_prices(world):
    """Yahoo returning Thursday's bars on Sunday gives ordinary-looking features from old
    prices, and the model scores them as if they were Friday's."""
    world["end"] = LATEST - pd.Timedelta(days=1)
    with pytest.raises(live.StaleDataError, match="2026-09-24"):
        _decide()


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
        _decide(predictor=FakePredictor(features))


def test_the_decision_envelope_carries_all_30_symbols_and_only_the_kits_fields(world):
    """The kit rejects a decision missing a zero-weight symbol, and the first in-window
    upload consumes the round even when invalid."""
    decision, log = _decide()
    assert set(decision) == set(kit.contracts.FIELDS["decision"])
    assert list(decision["weights"]) == list(json.load(open(live.DECISION_TEMPLATE))["weights"])
    assert set(decision["weights"]) == set(OURS) and len(decision["weights"]) == 30
    assert decision["round_id"].startswith("dryrun-") and decision["phase"] == "validation"
    assert kit.contracts.is_placeholder(decision["team_token"])


def test_a_constant_prediction_raises_instead_of_letting_vol_alone_pick_the_book(world):
    """A model fed all-NaN rows returns one value for every name; the compiler would then
    select by volatility and log it as the model's choice."""
    flat = FakePredictor()
    flat.predict = lambda frame: pd.Series(0.5, index=range(len(frame)))
    with pytest.raises(live.LiveDataError, match="constant"):
        _decide(predictor=flat)


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
