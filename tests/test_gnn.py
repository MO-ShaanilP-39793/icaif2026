import numpy as np
import pandas as pd
import pytest
import torch

from icaif import daily_features, gnn, gnn_data

NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "OUT"]
CTX = ["SPY", "^VIX", "^TNX", "^IRX", "XLK", "XLF"]
CPU = torch.device("cpu")


def _walk(names, n=320, seed=0, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    frames = []
    for s in names:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": s, "open": open_,
            "high": np.maximum(open_, close) * 1.005, "low": np.minimum(open_, close) * 0.995,
            "close": close, "adj_close": close, "volume": rng.uniform(1e6, 2e6, n)}))
    return pd.concat(frames, ignore_index=True)


def _tensors(daily, ctx):
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    mask = pd.DataFrame(True, index=dates, columns=NAMES)
    mask["OUT"] = False
    f = daily_features.build(daily, mask, ctx)
    frame = f.join(daily_features.build_labels(daily, mask), how="inner")
    return gnn_data.build(frame, daily, competition=["AAA", "BBB", "CCC"])


@pytest.fixture(scope="module")
def world():
    return _walk(NAMES), _walk(CTX, seed=1)


def test_no_model_input_for_day_d_reads_day_d_or_later(world):
    """Rewrite every price and context value dated d or later. The history window, the
    context and the correlation bias for a decision before d's open must be unchanged.
    The correlation is the easy one to get wrong: `ret[d]` is d's own close-to-close
    return, and a window ending at d instead of d-1 reads it."""
    daily, ctx = world
    before = _tensors(daily, ctx)
    d = 280
    day = before.dates[d]
    late, late_ctx = daily.copy(), ctx.copy()
    for f in (late, late_ctx):
        rows = f["date"] >= day
        f.loc[rows, ["open", "high", "low", "close"]] *= np.exp(
            np.random.default_rng(3).normal(0, 0.05, (rows.sum(), 1)))
        f.loc[rows, "volume"] *= 3
    after = _tensors(late, late_ctx)
    names = np.arange(len(before.tickers))
    np.testing.assert_array_equal(gnn_data.history(before, d, names), gnn_data.history(after, d, names))
    np.testing.assert_array_equal(before.ctx[d], after.ctx[d])
    np.testing.assert_array_equal(gnn_data.correlation(before, d, names),
                                  gnn_data.correlation(after, d, names))
    # ... and the test can fail: the next day's inputs do see the rewrite.
    assert not np.array_equal(gnn_data.correlation(before, d + 1, names),
                              gnn_data.correlation(after, d + 1, names))
    assert not np.array_equal(gnn_data.history(before, d + 1, names),
                              gnn_data.history(after, d + 1, names))


def test_a_name_outside_the_universe_is_absent_not_a_row_of_zero_ranks(world):
    daily, ctx = world
    t = _tensors(daily, ctx)
    out = t.tickers.get_loc("OUT") if "OUT" in t.tickers else None
    # OUT is never in the universe, so it has no features, no label, and is never eligible.
    assert out is None or not t.present[:, out].any()
    assert all("OUT" not in t.tickers[gnn_data.eligible(t, d, True)] for d in range(60, 300))


def test_masked_correlation_matches_pandas_pairwise_with_holes():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, (60, 5))
    r[:, 1] += r[:, 0]
    r[rng.random((60, 5)) < 0.2] = np.nan
    r[:50, 4] = np.nan                      # 10 observations: below the overlap floor
    want = pd.DataFrame(r).corr(min_periods=gnn_data.CORR_MIN_OVERLAP).fillna(0.0).to_numpy()
    np.testing.assert_allclose(gnn_data.masked_corr(r), want, atol=1e-5)
    got = gnn.masked_corr_torch(torch.tensor(r[None], dtype=torch.float32))[0].numpy()
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_a_missing_return_is_not_read_as_a_flat_day():
    """Two names that move identically are correlated 1, whatever sessions one of them
    missed. Filling its holes with 0 would read as flat days and pull that below 1."""
    r = np.random.default_rng(1).normal(0, 0.01, (60, 2))
    r[:, 1] = r[:, 0]
    r[::4, 1] = np.nan
    assert gnn_data.masked_corr(r)[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_a_validation_label_running_into_the_test_year_is_purged(world):
    """Kept, early stopping would pick the epoch that best fits the test year's first week."""
    daily, ctx = world
    t = _tensors(_walk(NAMES, n=800, start="2021-01-04"), _walk(CTX, n=800, seed=1, start="2021-01-04"))
    s = gnn_data.split_days(t, 2023)
    assert (t.label_end[s.train] < pd.Timestamp("2022-01-01")).all()
    assert (t.dates[s.val] >= pd.Timestamp("2022-01-01")).all()
    assert (t.label_end[s.val] < pd.Timestamp("2023-01-01")).all()
    assert t.dates[s.val].max() < pd.Timestamp("2022-12-26")   # the last 5 sessions go
    assert (t.dates[s.test].year == 2023).all()


def test_a_known_release_date_never_reads_as_no_release_or_vice_versa():
    """`e_sessions_to_next` is NaN when no release is within 10 sessions. Zero-filled
    raw, that would read as a release today."""
    f = pd.DataFrame({"e_sessions_to_next": [np.nan, 0, 10], "e_sessions_since": [np.nan, 3, 90],
                      "e_last_reaction": [np.nan, 9.0, -1.0]})
    e = gnn_data.encode_earnings(f)
    assert e["e_next"].tolist() == pytest.approx([0.0, 1.0, 1 / 11])
    assert e["e_since"].tolist() == pytest.approx([1.0, 0.05, 1.0])
    assert e["e_reaction"].tolist() == pytest.approx([0.0, 1.0, -0.2])


def _model(seed=0):
    torch.manual_seed(seed)
    m = gnn.CrossSectionalRanker(len(gnn_data.NAME_FEATURES), 5, gnn.Config(dropout=0.0))
    with torch.no_grad():
        m.corr_scale.fill_(0.7)             # make the correlation path matter
    return m.eval()


def _inputs(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, gnn_data.HISTORY, n, len(gnn_data.NAME_FEATURES), generator=g)
    corr = torch.rand(1, n, n, generator=g) * 2 - 1
    corr = (corr + corr.transpose(1, 2)) / 2
    return x, torch.randn(1, 5, generator=g), corr, torch.zeros(1, n, dtype=torch.bool)


def test_a_names_score_does_not_depend_on_its_position_in_the_set():
    """Sets are packed in ticker order. Anything positional would let the model learn
    the alphabet instead of the features."""
    m = _model()
    x, ctx, corr, pad = _inputs()
    perm = torch.randperm(12, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        base = m(x, ctx, corr, pad)
        moved = m(x[:, :, perm], ctx, corr[:, perm][:, :, perm], pad[:, perm])
    torch.testing.assert_close(moved, base[:, perm], atol=1e-5, rtol=1e-5)


def test_a_padded_slot_never_moves_a_real_names_score():
    """A day with 27 of the 30 names pads to 30. If the pad leaked into attention, one
    missing name would shift every other name's score."""
    m = _model()
    x, ctx, corr, pad = _inputs(n=12)
    with torch.no_grad():
        base = m(x[:, :, :9], ctx, corr[:, :9, :9], pad[:, :9])
        pad2 = pad.clone()
        pad2[:, 9:] = True
        x2 = x.clone()
        x2[:, :, 9:] = 1e3                  # garbage in the padded slots
        padded = m(x2, ctx, corr, pad2)
    torch.testing.assert_close(padded[:, :9], base, atol=1e-5, rtol=1e-5)


def _planted(T=260, N=40, seed=0):
    """Tensors where the label is the day's rank of feature 0, plus noise."""
    rng = np.random.default_rng(seed)
    F = len(gnn_data.NAME_FEATURES)
    x = rng.uniform(-0.5, 0.5, (T, N, F)).astype(np.float32)
    x[..., -1] = 1.0
    y = pd.DataFrame(x[..., 0] + rng.normal(0, 0.3, (T, N))).rank(axis=1, pct=True).to_numpy(np.float32)
    dates = pd.bdate_range("2020-01-01", periods=T)
    return gnn_data.Tensors(dates, pd.Index([f"N{i:02d}" for i in range(N)]), x,
                            np.ones((T, N), bool), rng.normal(size=(T, 5)).astype(np.float32),
                            rng.normal(0, 0.01, (T, N)).astype(np.float32), y,
                            pd.DatetimeIndex(dates), np.arange(N) < 30, [f"ctx_{i}" for i in range(5)])


def test_a_planted_signal_is_learned_so_names_and_labels_stay_aligned():
    """A gather that paired one name's features with another's label, or a history
    window off by a day, would still train and report IC near zero, which reads as
    'no signal' rather than a bug. The label here is feature 0 on the decision day."""
    t = _planted()
    cfg = gnn.Config(max_epochs=6, patience=6, batch=32, dropout=0.0, lr=3e-3)
    res = gnn.fit(t, np.arange(60, 200), np.arange(200, 255), cfg, seed=0, dev=CPU, log=lambda r: None)
    assert res.best_val_ic > 0.5
    s = gnn.score_competition(res, np.arange(200, 255))
    assert s.index.get_level_values("ticker").nunique() == 30
