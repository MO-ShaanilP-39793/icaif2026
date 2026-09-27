"""Walk-forward training of the AutoGluon ensembles (design doc: Ensemble models tab).

Four yearly folds, each training on everything whose label ended before the test
year starts, and scoring that year out of sample:

    train < 2023 -> test 2023,  < 2024 -> 2024,  < 2025 -> 2025,  < 2026 -> 2026

**The purge** is on the label, not the row: a training row is kept only if its label
path (`<h>_end`) ended before the test year began. Cutting on the decision date alone
would keep December rows whose 5-day paths run into January, so the model would have
trained on the test year's first days.

**Inner folds** are AutoGluon's bagging folds. By default they are random rows, and
neighbouring decisions share most of a 3-5 day label, so a random fold's "held-out"
rows have near-copies in training. The stacker then learns from leaked out-of-fold
scores and looks far better than it is. Here `groups` assigns contiguous time blocks
(every row of a decision in one block), and AutoGluon splits leave-one-block-out.

Contiguous blocks are not enough on their own. A row just before a held-out block has
a label path running through that block's prices. The first smoke run showed it:
with blocks of ~3.5 days and a 5-day label, validation Spearman read 0.53 against a
test IC of -0.01. So a row whose label ends in a different block from its decision
is dropped from the fit (`within_block`), the same purge as between folds, inside
them.

The evaluation that counts is the per-decision rank IC on the test year, and for the
daily model the IC within the 30 competition names alone, re-ranked among
themselves. That is the cross-section the competition trades.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from icaif import calendar, daily_features, data, earnings, external, features, labels, markets, universe

TEST_YEARS = (2023, 2024, 2025, 2026)
INNER_BLOCKS = 8
MAX_PURGED = 0.2
COMPETITION_WEIGHT = 3.0
# Pass 1 families (design doc). Tabular foundation models join in the GPU pass.
PASS1_FAMILIES = {"GBM": {}, "CAT": {}, "XGB": {}, "NN_TORCH": {}, "FASTAI": {}, "LR": {}}
OUTPUT = data.ROOT / "output" / "ag"


@dataclass
class Dataset:
    frame: pd.DataFrame        # features + labels, index (time, ticker)
    feature_cols: list[str]
    time_level: str            # "date" (daily) or "execution" (intraday)
    name: str


def daily_dataset(train_start: str | None = None) -> Dataset:
    daily = external.load("yahoo_daily_universe")
    mask = universe.build(daily, universe.load_membership())
    events = earnings.quarterly(external.load("earnings"))
    f = daily_features.build(daily, mask, external.load("yahoo_daily_context"), events)
    lab = daily_features.build_labels(daily, mask)
    df = f.join(lab, how="inner")
    if train_start:
        df = df[df.index.get_level_values("date") >= train_start]
    return Dataset(df, list(f.columns), "date", "daily")


def intraday_dataset() -> Dataset:
    events = earnings.quarterly(external.load("earnings"))
    f = features.build(markets.intraday_info_bars(), events=events,
                       ctx_daily=external.load("yahoo_daily_context"))
    lab = labels.build(markets.label_exec_prices())
    return Dataset(f.join(lab, how="inner"), list(f.columns), "execution", "intraday")


def horizon_of(target: str) -> str:
    """'d5_up_pct' -> 'd5', 'h35_pct' -> 'h35': the label's horizon prefix."""
    return target.split("_", 1)[0]


def _year_start(year: int, like: pd.Index) -> pd.Timestamp:
    ts = pd.Timestamp(f"{year}-01-01")
    return ts.tz_localize(calendar.TZ) if getattr(like, "tz", None) is not None else ts


def fold_split(ds: Dataset, target: str, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) for one walk-forward fold, purged on each label's end."""
    df = ds.frame[ds.frame[target].notna()]
    t = df.index.get_level_values(ds.time_level)
    lo, hi = _year_start(test_year, t), _year_start(test_year + 1, t)
    end = pd.DatetimeIndex(df[f"{horizon_of(target)}_end"])
    train = df[np.asarray(end < lo)]
    test = df[np.asarray((t >= lo) & (t < hi))]
    return train, test


def inner_groups(times: pd.Index, k: int = INNER_BLOCKS) -> np.ndarray:
    """Contiguous time blocks 0..k-1, every row of one decision time in the same block."""
    order = pd.Index(times).unique().sort_values()
    block = pd.Series((np.arange(len(order)) * k) // len(order), index=order)
    return block.reindex(times).to_numpy()


def within_block(times: pd.Index, ends: pd.Index, k: int = INNER_BLOCKS) -> np.ndarray:
    """True where a row's label ends inside the block its decision is in."""
    order = pd.Index(times).unique().sort_values()
    starts = order[(np.arange(k) * len(order)) // k]
    own = starts.searchsorted(pd.Index(times), side="right") - 1
    end = starts.searchsorted(pd.Index(ends), side="right") - 1
    return own == end


def fit_fold(train: pd.DataFrame, ds: Dataset, target: str, path: Path, time_limit: int,
             presets: str = "medium_quality", families: dict | None = None):
    from autogluon.tabular import TabularPredictor

    cols = ds.feature_cols
    times = train.index.get_level_values(ds.time_level)
    keep = within_block(times, pd.DatetimeIndex(train[f"{horizon_of(target)}_end"]))
    if (~keep).mean() > MAX_PURGED:
        # Blocks barely longer than the label: most rows cross one, and what survives
        # is a fit on scraps whose validation score means nothing.
        raise ValueError(f"within-block purge would drop {(~keep).mean():.0%} of rows; "
                         f"blocks are too short for {target}'s horizon")
    train = train[keep]
    frame = train[cols + [target]].reset_index(drop=True)
    frame["_group"] = inner_groups(train.index.get_level_values(ds.time_level))
    weight = None
    if "is_competition" in cols:
        # The loss intervention from the design doc: the 30 names count more, so the
        # broad universe teaches patterns and the 30 decide which ones fit them.
        frame["_weight"] = np.where(train["is_competition"].to_numpy() > 0, COMPETITION_WEIGHT, 1.0)
        weight = "_weight"
    predictor = TabularPredictor(label=target, problem_type="regression", eval_metric="spearmanr",
                                 path=str(path), groups="_group", sample_weight=weight,
                                 verbosity=1)
    predictor.fit(frame, presets=presets, time_limit=time_limit,
                  hyperparameters=families or PASS1_FAMILIES,
                  num_bag_folds=INNER_BLOCKS, num_bag_sets=1, num_stack_levels=0,
                  dynamic_stacking=False)
    return predictor


def per_decision_ic(pred: pd.Series, truth: pd.Series, level: str) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": truth}).dropna()
    g = df.groupby(level=level)
    df = df.assign(p=g["p"].rank(), y=g["y"].rank())
    d = df - df.groupby(level=level).transform("mean")
    num = (d["p"] * d["y"]).groupby(level=level).sum()
    den = np.sqrt((d["p"] ** 2).groupby(level=level).sum() * (d["y"] ** 2).groupby(level=level).sum())
    return (num / den).dropna()


def evaluate(pred: pd.Series, test: pd.DataFrame, target: str, ds: Dataset) -> dict:
    """Test-year IC per decision; for the daily model also within the 30 alone."""
    h = horizon_of(target)
    steps = labels.DAILY_HORIZONS.get(h) or labels.HORIZONS[h] / labels.ROUNDS_PER_SESSION
    ic = per_decision_ic(pred, test[target], ds.time_level)
    days = ic.groupby(pd.DatetimeIndex(ic.index).date).mean()
    out = {"n_decisions": int(ic.size), "ic": float(days.mean()),
           "t": float(days.mean() / (days.std() / np.sqrt(len(days) / steps)))}
    if ds.name == "daily":
        ours = test.index.get_level_values("ticker").isin(list(data.load_universe()))
        ic30 = per_decision_ic(pred[ours], test.loc[ours, target], ds.time_level)
        out.update({"ic_30": float(ic30.mean()),
                    "t_30": float(ic30.mean() / (ic30.std() / np.sqrt(len(ic30) / steps)))})
    return out
