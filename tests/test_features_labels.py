import numpy as np
import pandas as pd
import pytest

from icaif import calendar, data, features, labels, markets

needs_panel = pytest.mark.skipif(not data.ORGANIZER_PARQUET.exists(),
                                 reason="organizer parquet not present")


@pytest.fixture(scope="module")
def month():
    bars, _ = data.load_organizer_bars()
    lo = pd.Timestamp("2024-04-15", tz=calendar.TZ)
    hi = pd.Timestamp("2024-06-29", tz=calendar.TZ)
    return bars[(bars["start"] >= lo) & (bars["start"] < hi)].reset_index(drop=True)


@needs_panel
def test_no_feature_changes_when_everything_after_its_deadline_is_rewritten(month):
    """Rewrite every bar that ends after 11:25 on 2024-06-12 (prices x1.5, volume x3).
    Any feature at a deadline up to then that moves has read the future, however
    indirectly: through a rolling window, a same-day aggregate, or a cross-sectional
    rank taken over a peer's future."""
    cutoff = pd.Timestamp("2024-06-12 11:25", tz=calendar.TZ)
    before = features.build(month)
    changed = month.copy()
    late = changed["end"] > cutoff
    changed.loc[late, ["open", "high", "low", "close"]] *= 1.5
    changed.loc[late, "volume"] *= 3
    after = features.build(changed)

    deadlines = {r["execution"]: r["deadline"] for d in sorted(set(month["start"].dt.date))
                 for r in calendar.rounds_for(d)}
    ex = before.index.get_level_values("execution")
    known = np.array([deadlines[e] <= cutoff for e in ex])
    assert known.sum() > 30 * 7 * 20
    pd.testing.assert_frame_equal(before[known], after[known])
    assert not before[~known].equals(after[~known])  # the rewrite did reach later rows


@needs_panel
def test_a_round_one_decision_sees_yesterday_and_nothing_of_today(month):
    """At 09:10 nothing of today has traded, so the return since the prior close is 0
    for every name and today's volume is unknown, not zero-and-ranked."""
    f = features.build(month)
    first_day = f.index.get_level_values("execution").date == month["start"].min().date()
    r1 = f[(f["ctx_round"] == 1) & ~first_day]
    assert (r1["ret_1s"] == 0).all()  # every name tied, and a tie ranks at exactly 0
    assert r1["volume_today_ratio"].isna().all()


def test_a_label_reads_its_own_horizon_and_not_one_round_beyond():
    """Changing the price at round i+h must change round i's label inputs; changing
    round i+h+1 must not. An off-by-one here trains on a round the strategy never holds."""
    idx = pd.DatetimeIndex([r["execution"] for d in pd.bdate_range("2026-09-01", periods=10)
                            for r in calendar.rounds_for(d.date())])
    rng = np.random.default_rng(0)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 30)), axis=0)),
                      index=idx, columns=sorted(data.load_universe()))
    h = labels.HORIZONS["h7"]
    base = labels.components(px, h)["terminal"].iloc[0]
    at_h, beyond = px.copy(), px.copy()
    at_h.iloc[h] *= 1.1
    beyond.iloc[h + 1] *= 1.1
    assert not np.allclose(labels.components(at_h, h)["terminal"].iloc[0], base)
    np.testing.assert_allclose(labels.components(beyond, h)["terminal"].iloc[0], base)


def test_a_missing_price_on_the_path_gives_no_label_rather_than_a_label_on_a_hole():
    idx = pd.DatetimeIndex([r["execution"] for d in pd.bdate_range("2026-09-01", periods=7)
                            for r in calendar.rounds_for(d.date())])
    px = pd.DataFrame(100.0, index=idx, columns=sorted(data.load_universe()))
    px.iloc[3, 0] = np.nan
    out = labels.build(px)
    first = out.xs(idx[0], level="execution")
    assert np.isnan(first.loc[px.columns[0], "h7_label"])
    assert first["h7_label"].notna().sum() == 29


def test_labels_mark_the_top_forty_percent_at_each_decision():
    idx = pd.DatetimeIndex([r["execution"] for d in pd.bdate_range("2026-09-01", periods=10)
                            for r in calendar.rounds_for(d.date())])
    rng = np.random.default_rng(1)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 30)), axis=0)),
                      index=idx, columns=sorted(data.load_universe()))
    out = labels.build(px)
    for col in ("h7_label", "h35_label", "h35_up_label"):
        per_decision = out[col].groupby(level="execution").sum()
        counted = out[col].groupby(level="execution").count()
        assert (per_decision[counted == 30] == 12).all(), col


def test_upside_is_the_mean_of_the_best_k_fills_not_the_single_best_or_the_last():
    """At 21 rounds k = 2, so upside = mean of the two highest fills / entry - 1. Using
    the max alone rewards one spike; using the last fill makes it a terminal return."""
    h = labels.HORIZONS["h21"]
    path = np.full(h + 1, 100.0)
    path[5], path[9], path[-1] = 110.0, 106.0, 101.0
    idx = pd.date_range("2026-09-01 09:30", periods=h + 1, freq="h", tz=calendar.TZ)
    up = labels.components(pd.DataFrame({"A": path}, index=idx), h)["upside"].iloc[0, 0]
    assert up == pytest.approx(0.08)


def test_a_percentile_target_spans_zero_to_one_whatever_the_cross_section_size():
    """A pct rank tops out at 1 but bottoms at 1/n, so its mean drifts with how many
    names have a label that day. The regression target should not."""
    panel = pd.DataFrame([[3.0, 1.0, 2.0, np.nan], [1.0, 2.0, 3.0, 4.0]])
    r = labels.unit_rank(panel)
    assert r.min(axis=1).tolist() == [0.0, 0.0] and r.max(axis=1).tolist() == [1.0, 1.0]


def test_a_missing_public_open_is_filled_by_the_organizer_guess_never_by_the_last_close():
    """A last-close stand-in is a zero return then a catch-up jump: a fake path inside
    every label that spans it. A hole with no guess must stay a hole."""
    ex = pd.date_range("2025-06-02 09:30", periods=4, freq="h", tz=calendar.TZ)
    exact = pd.DataFrame({"A": [10.0, np.nan, np.nan, 13.0]}, index=ex[:4])
    guessed = pd.DataFrame({"A": [9.0, 9.5, np.nan, 12.0]}, index=ex[:4])
    earlier = pd.DataFrame({"A": [8.0]}, index=[ex[0] - pd.Timedelta(days=1)])
    out = markets.merge_fills(exact, pd.concat([earlier, guessed]))
    assert out["A"].tolist()[0] == 8.0  # before the exact span: the guess
    assert out["A"].tolist()[1:3] == [10.0, 9.5]
    assert np.isnan(out["A"].iloc[3]) and out["A"].iloc[4] == 13.0


@needs_panel
def test_no_two_features_are_the_same_column_under_another_name(month):
    """A market-relative return ranks identically to the raw return, so it adds nothing
    but a second vote for the same signal in importance-based pruning."""
    f = features.build(month).drop(columns=[c for c in features.build(month).columns
                                            if c.startswith("ctx_")])
    corr = f.corr().abs().to_numpy()
    np.fill_diagonal(corr, 0)
    assert corr.max() < 0.999


@needs_panel
def test_an_earnings_release_is_seen_from_the_first_decision_after_it(month):
    """Accepted at 09:20, a release moves that day's open, but the round-1 decision
    (deadline 09:10) hasn't seen it; round 2 (deadline 10:25) has."""
    t = sorted(data.load_universe())[0]
    events = pd.DataFrame({"ticker": t, "accepted": pd.to_datetime(
        ["2024-06-11 16:30", "2024-06-13 09:20"]).tz_localize(calendar.TZ)})
    f = features.build(month, events=events)
    at = lambda day, rnd: f[(f.index.get_level_values("execution").date == pd.Timestamp(day).date())
                            & (f["ctx_round"] == rnd)].xs(t, level="ticker").iloc[0]
    assert at("2024-06-12", 1)["e_sessions_since"] == 0  # after yesterday's close
    assert at("2024-06-13", 1)["e_sessions_since"] == 1 and at("2024-06-13", 1)["e_sessions_to_next"] == 0
    assert at("2024-06-13", 2)["e_sessions_since"] == 0


@needs_panel
def test_daily_context_on_day_d_is_the_close_of_d_minus_1(month):
    dates = pd.bdate_range("2024-03-01", "2024-07-31")
    vix = pd.Series(np.arange(len(dates), dtype=float) + 10, index=dates)
    ctx = pd.concat([pd.DataFrame({"date": dates, "ticker": s, "close": vix.to_numpy() if s == "^VIX" else 100.0})
                     for s in ["^VIX", "SPY", "^TNX", "^IRX", "XLK"]], ignore_index=True)
    f = features.build(month, ctx_daily=ctx)
    row = f[f.index.get_level_values("execution").date == pd.Timestamp("2024-06-12").date()].iloc[0]
    assert row["ctx_vix"] == vix[pd.Timestamp("2024-06-11")]
