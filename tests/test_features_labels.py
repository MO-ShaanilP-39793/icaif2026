import numpy as np
import pandas as pd
import pytest

from icaif import calendar, data, features, labels

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
    idx = pd.DatetimeIndex([r["execution"] for d in pd.bdate_range("2026-09-01", periods=5)
                            for r in calendar.rounds_for(d.date())])
    px = pd.DataFrame(100.0, index=idx, columns=sorted(data.load_universe()))
    px.iloc[3, 0] = np.nan
    out = labels.build(px)
    first = out.xs(idx[0], level="execution")
    assert np.isnan(first.loc[px.columns[0], "h7_label"])
    assert first["h7_label"].notna().sum() == 29


def test_labels_mark_the_top_thirty_percent_at_each_decision():
    idx = pd.DatetimeIndex([r["execution"] for d in pd.bdate_range("2026-09-01", periods=8)
                            for r in calendar.rounds_for(d.date())])
    rng = np.random.default_rng(1)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 30)), axis=0)),
                      index=idx, columns=sorted(data.load_universe()))
    out = labels.build(px)
    per_decision = out["h7_label"].groupby(level="execution").sum().dropna()
    assert (per_decision[per_decision > 0] == 9).all()


@needs_panel
def test_no_two_features_are_the_same_column_under_another_name(month):
    """A market-relative return ranks identically to the raw return, so it adds nothing
    but a second vote for the same signal in importance-based pruning."""
    f = features.build(month).drop(columns=[c for c in features.build(month).columns
                                            if c.startswith("ctx_")])
    corr = f.corr().abs().to_numpy()
    np.fill_diagonal(corr, 0)
    assert corr.max() < 0.999
