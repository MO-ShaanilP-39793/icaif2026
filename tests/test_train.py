import numpy as np
import pandas as pd

from icaif import train


def _ds():
    dates = pd.bdate_range("2022-12-01", "2023-02-28")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    df = pd.DataFrame({"x": 1.0, "d5_pct": 0.5}, index=idx)
    pos = {d: i for i, d in enumerate(dates)}
    df["d5_end"] = [dates[min(pos[d] + 5, len(dates) - 1)] for d in idx.get_level_values("date")]
    return train.Dataset(df, ["x"], "date", "daily")


def test_a_december_row_whose_label_runs_into_january_is_purged_from_training():
    """Cut on the decision date alone, the last days of December would stay in training
    with 5-day paths ending in the test year: the model would have seen January."""
    tr, te = train.fold_split(_ds(), "d5_pct", 2023)
    assert (pd.DatetimeIndex(tr["d5_end"]) < pd.Timestamp("2023-01-01")).all()
    last_dec = tr.index.get_level_values("date").max()
    # Dec 23 + 5 sessions = Dec 30, still 2022; Dec 26-30 all end in January and go.
    assert last_dec == pd.Timestamp("2022-12-23")
    assert te.index.get_level_values("date").min() == pd.Timestamp("2023-01-02")


def test_inner_folds_are_contiguous_blocks_with_every_decision_in_one_block():
    """A random fold puts near-copies of its held-out rows in training (neighbouring
    decisions share most of a multi-day label), so the stacker trains on leaks."""
    times = pd.Index(np.repeat(pd.bdate_range("2023-01-02", periods=80), 3))
    g = train.inner_groups(times, k=8)
    assert set(g) == set(range(8))
    per_time = pd.Series(g, index=times).groupby(level=0).nunique()
    assert (per_time == 1).all()
    assert (np.diff(pd.Series(g, index=times).groupby(level=0).first().to_numpy()) >= 0).all()


def test_a_row_whose_label_crosses_into_the_next_block_is_purged_inside_the_fold():
    """Kept, it trains on the held-out block's prices: in the first smoke run this
    inflated validation Spearman to 0.53 against a test IC of -0.01."""
    times = pd.Index(pd.bdate_range("2023-01-02", periods=80))
    ends = times + pd.offsets.BDay(5)
    keep = train.within_block(times, ends, k=8)
    g = train.inner_groups(times, k=8)
    last_of_block = pd.Series(g, index=times).groupby(g).apply(lambda s: s.index[-5:])
    assert not keep[times.isin(np.concatenate([b for b in last_of_block.iloc[:-1]]))].any()
    assert keep.sum() == 80 - 5 * 8 + 5  # the final block's tail runs past the data, no next block
