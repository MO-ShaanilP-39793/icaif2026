import numpy as np
import pandas as pd
import pytest

from icaif import calendar, daily_features, data, earnings

OURS = sorted(data.load_universe())[0]
NAMES = [OURS, "BBB", "CCC", "DDD", "EEE", "OUT"]
CTX = ["SPY", "^VIX", "^TNX", "^IRX", "XLK", "XLF"]


def _walk(names, n=320, seed=0, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    frames = []
    for i, s in enumerate(names):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": s, "open": open_,
            "high": np.maximum(open_, close) * 1.005, "low": np.minimum(open_, close) * 0.995,
            "close": close, "adj_close": close, "volume": rng.uniform(1e6, 2e6, n)}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def world():
    daily = _walk(NAMES)
    ctx = _walk(CTX, seed=1)
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    mask = pd.DataFrame(True, index=dates, columns=NAMES)
    mask["OUT"] = False
    return daily, ctx, mask


def test_no_feature_on_day_d_reads_day_d_or_later(world):
    """Rewrite every bar and context value dated d or later. A decision before d's open
    has seen none of it, so every row dated d or earlier must be unchanged."""
    daily, ctx, mask = world
    d = pd.Timestamp(sorted(daily["date"].unique())[280])
    before = daily_features.build(daily, mask, ctx)
    late, late_ctx = daily.copy(), ctx.copy()
    for f in (late, late_ctx):
        rows = f["date"] >= d
        f.loc[rows, ["open", "high", "low", "close"]] *= 1.5
        f.loc[rows, "volume"] *= 3
    after = daily_features.build(late, mask, late_ctx)
    known = before.index.get_level_values("date") <= d
    pd.testing.assert_frame_equal(before[known], after[known])
    assert not before[~known].equals(after[~known])


def test_a_name_outside_the_universe_never_moves_anyone_elses_rank(world):
    daily, ctx, mask = world
    base = daily_features.build(daily, mask, ctx)
    wild = daily.copy()
    wild.loc[wild["ticker"] == "OUT", ["open", "high", "low", "close"]] *= np.linspace(1, 50, 320)[:, None]
    moved = daily_features.build(wild, mask, ctx)
    assert "OUT" not in base.index.get_level_values("ticker")
    pd.testing.assert_frame_equal(base, moved)


def test_a_zero_volume_placeholder_is_missing_not_a_flat_day(world):
    """Yahoo fills some halted or holiday rows with the last price and zero volume.
    Read as real, that is a zero return, then a catch-up jump the next day."""
    daily, ctx, mask = world
    d = pd.Timestamp(sorted(daily["date"].unique())[200])
    stale = daily.copy()
    row = (stale["ticker"] == "BBB") & (stale["date"] == d)
    stale.loc[row, "volume"] = 0
    stale.loc[row, ["open", "high", "low", "close"]] = 1e6
    f = daily_features.build(stale, mask, ctx)
    nxt = pd.Timestamp(sorted(daily["date"].unique())[201])
    assert np.isnan(f.loc[(nxt, "BBB"), "ret_1d"])


def test_labels_rank_only_within_the_days_universe(world):
    daily, _, mask = world
    lab = daily_features.build_labels(daily, mask)
    assert "OUT" not in lab.dropna(subset=["d3_label"]).index.get_level_values("ticker")
    per_day = lab["d3_label"].groupby(level="date").sum()
    counted = lab["d3_label"].groupby(level="date").count()
    assert (per_day[counted == 5] == 2).all()  # round(5 x 0.4)


def _decisions(days):
    days = pd.DatetimeIndex(days)
    return pd.DataFrame({"day": days, "deadline": [calendar.at(d.date(), daily_features.DEADLINE)
                                                   for d in days]})


SESSIONS = pd.bdate_range("2024-07-01", "2024-09-30")


def _events(*stamps):
    return pd.DataFrame({"ticker": "AAA", "accepted": pd.to_datetime(list(stamps)).tz_localize(calendar.TZ)})


def test_sessions_since_counts_from_the_first_open_that_reflects_the_release():
    ev = _events("2024-07-11 16:30", "2024-08-05 16:30")  # Thursday after close, then Monday
    got = earnings.proximity(ev, _decisions(["2024-07-11", "2024-07-12", "2024-07-15"]), SESSIONS)
    assert got["since"].isna().iloc[0]  # before the first recorded release: unknown
    assert got["since"].iloc[1:].tolist() == [0, 1]


def test_the_next_release_is_only_seen_once_its_date_would_be_announced():
    ev = _events("2024-07-11 16:30", "2024-08-05 16:30")
    near = earnings.proximity(ev, _decisions(["2024-07-26"]), SESSIONS)  # 7 sessions before
    far = earnings.proximity(ev, _decisions(["2024-07-15"]), SESSIONS)   # 16 sessions before
    assert near["to_next"].iloc[0] == 7
    assert np.isnan(far["to_next"].iloc[0])


def test_a_release_after_the_deadline_is_not_yet_in_the_past():
    """Accepted at 09:20 it moves that day's open, but a 09:10 decision hasn't seen it."""
    ev = _events("2024-07-11 16:30", "2024-08-05 09:20")
    got = earnings.proximity(ev, _decisions(["2024-08-05"]), SESSIONS)
    assert got["since"].iloc[0] == 16 and got["to_next"].iloc[0] == 0


def test_a_holiday_print_in_one_context_series_does_not_blank_spy_features():
    """Yahoo's ^VIX has bars on Memorial Day and Labor Day 2026 when SPY doesn't. On a
    union of dates that NaN SPY row blanked ctx_spy_ret_20d and ctx_spy_vol_20d for 20
    sessions, which the model never saw in training."""
    ctx = _walk(CTX, seed=2)
    holiday = pd.Timestamp(sorted(ctx["date"].unique())[200])
    ctx = ctx[~((ctx["ticker"] != "^VIX") & (ctx["date"] == holiday))]  # only VIX prints
    c = daily_features.context(ctx)
    assert holiday not in c.index
    after = c.loc[c.index > holiday, "ctx_spy_vol_20d"].iloc[:25]
    assert after.notna().all()
