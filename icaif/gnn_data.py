"""Stage 0 of the cross-sectional attention model: the daily frame as dense tensors.

The model (`icaif.gnn`) reads, for a decision day d and a set of names:
- each name's last `HISTORY` sessions of feature vectors, the rows for d-19 .. d;
- a market token carrying d's `ctx_*` columns;
- an attention bias from the names' pairwise return correlations over the
  `CORR_WINDOW` sessions before d.

Everything lives on one session axis (`dates`), so a window is an index range, and a
random 30-name subset of a day is a gather. The frame's rows for d already hold only
what was known at the close of d-1 (`daily_features.build` shifts every panel), so
the history window ending at row d is as of d-1 too.

**The returns axis is not shifted, and that is the one place a slip leaks.** `ret[t]`
is the close-to-close return *of* session t, known only at t's close. The correlation
for decision d therefore reads `ret[d-60 .. d-1]` and never `ret[d]`: including it
would put d's own close into a decision made before d's open, and the bias would look
like signal in every backtest. A test rewrites every price from d onwards and requires
all three inputs for d to be unchanged.

Absent is not zero. A name outside the day's universe has no features (they are
ranked within the universe), so its row is zero with `present` false, and the model
reads `present` as an input. Missing returns stay NaN, and correlations are computed
over the sessions both names traded: filling a hole with 0 is a flat day that never
happened, which drags every correlation towards zero for a name that was halted.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from icaif import daily_features

HISTORY = 20
CORR_WINDOW = 60
CORR_MIN_OVERLAP = 20
# Per-name columns that are already ranked to [-0.5, 0.5] within the day's universe.
RANKED = ["ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
          "mom_12_1", "vol_5d", "vol_20d", "vol_60d", "parkinson_20d", "gap_last",
          "overnight_share_20d", "volume_1d_ratio", "dist_high_20d", "dist_low_20d",
          "dist_high_250d", "z_5d", "vol_ratio"]
EARNINGS = ["e_since", "e_next", "e_reaction"]
NAME_FEATURES = RANKED + EARNINGS + ["present"]


@dataclass
class Tensors:
    dates: pd.DatetimeIndex      # T sessions, ascending
    tickers: pd.Index            # N names ever in the universe
    x: np.ndarray                # [T, N, F] float32 name features for the decision on dates[t]
    present: np.ndarray          # [T, N] bool: in dates[t]'s universe
    ctx: np.ndarray              # [T, C] float32 raw market context for the decision on dates[t]
    ret: np.ndarray              # [T, N] float32 log return OF session t (NaN where missing)
    y: np.ndarray                # [T, N] float32 label for the decision on dates[t] (NaN = none)
    label_end: pd.DatetimeIndex  # [T] session the day's label path ends at (NaT = no label)
    is_competition: np.ndarray   # [N] bool
    ctx_names: list[str]

    @property
    def feature_names(self) -> list[str]:
        return NAME_FEATURES


def encode_earnings(frame: pd.DataFrame) -> pd.DataFrame:
    """The three raw earnings columns as bounded inputs, with absence given a meaning.

    A NaN here is information, not a hole: `e_sessions_to_next` is NaN unless a release
    is within 10 sessions. Left as NaN (or naively zero-filled) it would read as "a
    release today". So `e_next` is 0 for none and rises to 1 on the day itself;
    `e_since` saturates at 60 sessions, which is also what "none on record" means.
    """
    def col(c):  # a frame built without events carries none: every name reads "none on record"
        return frame[c] if c in frame else pd.Series(np.nan, index=frame.index)

    to_next, since = col("e_sessions_to_next"), col("e_sessions_since")
    return pd.DataFrame({
        "e_since": (since.clip(0, 60) / 60).fillna(1.0),
        "e_next": ((11 - to_next.clip(0, 10)) / 11).fillna(0.0),
        "e_reaction": (col("e_last_reaction").clip(-5, 5) / 5).fillna(0.0),
    }, index=frame.index)


def build(frame: pd.DataFrame, daily: pd.DataFrame, target: str = "d5_pct",
          competition: list[str] | None = None) -> Tensors:
    """`frame`: `train.daily_dataset().frame` (features + labels, index (date, ticker)).
    `daily`: the raw daily bars it was built from, for the correlation returns."""
    close = daily_features.panels(daily)["close"]
    dates = pd.DatetimeIndex(close.index)
    tickers = pd.Index(sorted(frame.index.get_level_values("ticker").unique()))
    ti = tickers.get_indexer(frame.index.get_level_values("ticker"))
    di = dates.get_indexer(frame.index.get_level_values("date"))
    if (di < 0).any():
        raise ValueError("frame has dates the daily bars don't: the two came from different snapshots")

    T, N = len(dates), len(tickers)
    feats = pd.concat([frame[RANKED], encode_earnings(frame)], axis=1)
    x = np.zeros((T, N, len(NAME_FEATURES)), np.float32)
    x[di, ti, :-1] = feats.fillna(0.0).to_numpy(np.float32)
    x[di, ti, -1] = 1.0
    present = np.zeros((T, N), bool)
    present[di, ti] = True

    ctx_names = [c for c in frame.columns if c.startswith("ctx_")]
    ctx = (frame[ctx_names].groupby(level="date").first()
           .reindex(dates).to_numpy(np.float32))

    y = np.full((T, N), np.nan, np.float32)
    y[di, ti] = frame[target].to_numpy(np.float32)
    end_col = f"{target.split('_', 1)[0]}_end"
    label_end = pd.DatetimeIndex(frame[end_col].groupby(level="date").max().reindex(dates))

    ret = np.log(close.reindex(columns=tickers)).diff().to_numpy(np.float32)
    comp = set(competition if competition is not None else _competition())
    return Tensors(dates, tickers, x, present, ctx, ret, y, label_end,
                   np.array([t in comp for t in tickers]), ctx_names)


def _competition() -> list[str]:
    from icaif import data
    return list(data.load_universe())


def history(t: Tensors, day: int, names: np.ndarray) -> np.ndarray:
    """[HISTORY, n, F]: rows day-19 .. day for `names`, zero (present 0) before the data starts."""
    rows = np.arange(day - HISTORY + 1, day + 1)
    out = np.zeros((HISTORY, len(names), t.x.shape[2]), np.float32)
    ok = rows >= 0
    out[ok] = t.x[rows[ok]][:, names]
    return out


def masked_corr(r: np.ndarray, min_overlap: int = CORR_MIN_OVERLAP) -> np.ndarray:
    """Pairwise Pearson over the rows both columns observed; 0 where overlap is short.

    `r`: [W, n] returns with NaN holes. Equal to `pd.DataFrame(r).corr(min_periods=...)`
    with NaN set to 0, and the diagonal 1 for any name with enough data.
    """
    m = ~np.isnan(r)
    v = np.where(m, r, 0.0).astype(np.float64)
    mf = m.astype(np.float64)
    n = mf.T @ mf
    sx = v.T @ mf                  # sx[i, j] = sum of i's returns over rows j also traded
    sxx = (v * v).T @ mf
    sxy = v.T @ v
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sx.T / n
        var_i = sxx - sx * sx / n
        c = cov / np.sqrt(var_i * var_i.T)
    c = np.where((n >= min_overlap) & np.isfinite(c), c, 0.0)
    return np.clip(c, -1.0, 1.0).astype(np.float32)


def correlation(t: Tensors, day: int, names: np.ndarray) -> np.ndarray:
    """[n, n] return correlation over the CORR_WINDOW sessions strictly before `day`."""
    lo = max(0, day - CORR_WINDOW)
    return masked_corr(t.ret[lo:day][:, names])


def eligible(t: Tensors, day: int, need_label: bool) -> np.ndarray:
    """Names that can be scored on `day`: in the universe, and labelled if training."""
    ok = t.present[day]
    if need_label:
        ok = ok & ~np.isnan(t.y[day])
    return np.flatnonzero(ok)


@dataclass
class DaySplit:
    train: np.ndarray   # session indices
    val: np.ndarray
    test: np.ndarray


def split_days(t: Tensors, test_year: int, first_day: int = CORR_WINDOW) -> DaySplit:
    """Train < val year < test year, each boundary purged on the label's end.

    The validation year is the one before the test year, and it picks the epoch to
    stop at. Its rows are purged against the test year exactly as training rows are
    against it: a late-December validation label whose path runs into January would
    let early stopping choose the epoch that best fits the test year's first week.
    """
    val_start = pd.Timestamp(f"{test_year - 1}-01-01")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01")
    idx = np.arange(len(t.dates))
    labelled = np.asarray(~pd.isna(t.label_end)) & (idx >= first_day)
    end = np.asarray(t.label_end.fillna(pd.Timestamp.max))
    d = np.asarray(t.dates)
    train = idx[labelled & (end < np.datetime64(val_start))]
    val = idx[labelled & (d >= np.datetime64(val_start)) & (end < np.datetime64(test_start))]
    test = idx[labelled & (d >= np.datetime64(test_start)) & (d < np.datetime64(test_end))]
    return DaySplit(train, val, test)
