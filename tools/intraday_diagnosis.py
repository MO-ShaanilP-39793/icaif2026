"""Why the intraday model scores ~0 out of sample when the daily one scores ~0.05.

    .venv/bin/python tools/intraday_diagnosis.py [--cache DIR]

Diagnosis only: pandas and numpy on the two training datasets and the pass-1
predictions in output/preds. Nothing is fitted. Building both datasets takes ~45 s;
`--cache DIR` keeps them as parquet so a rerun takes seconds (delete DIR after a
feature or label change, or the tables describe the old frame).

Sections, cheapest question first:
  A. the two targets at round 1 on the 30 names: are they the same thing?
  B. both models against both targets on the same rows: is it the target or the model?
  C. shared features, same days and names: does the daily edge live in columns the
     intraday model also has?
  D. every intraday feature by year: which relationships reversed after 2022?
  E. the intraday model by round
  F. data red flags: rounds per day, NaN shares by year, columns tied within a decision
  G. how much independent history each model trained on, and the simplest possible
     out-of-sample check: an IC-sign composite fitted on 2016-22 and scored after.

IC throughout is the per-decision rank correlation, averaged per day, then over days.
A t-stat divides by sqrt(n_days / 5): neighbouring 5-day labels share most of a path,
so each day is not an independent observation. Pooled over years that is honest; a
single year's IC has a standard error near 0.025 at 30 names, so one year's sign
means little on its own.

Writes reports/intraday_diagnosis_feature_ic.csv (section D, per feature and year).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import data, train  # noqa: E402

PREDS = data.ROOT / "output" / "preds"
TRAIN_YEARS = range(2016, 2023)
TEST_YEARS = range(2023, 2027)
STEPS = 5  # label horizon in sessions, for the effective sample

# At round 1 nothing of today has traded, so the intraday `last` is the prior close and
# ret_Ks spans K-1 sessions: ret_1s is identically 0 there, and ret_2s is the daily
# model's ret_1d. Pairing by name alone would compare a 4-day return with a 5-day one
# and call the difference a feature gap.
SHARED = [("ret_1d", "ret_2s"), ("ret_5d", "ret_5s"), ("ret_10d", "ret_10s"),
          ("ret_20d", "ret_20s"), ("vol_5d", "vol_5d"), ("vol_20d", "vol_20d"),
          ("vol_ratio", "vol_ratio"), ("parkinson_20d", "parkinson_5d"),
          ("gap_last", "gap_prev"), ("overnight_share_20d", "overnight_share_20d"),
          ("volume_1d_ratio", "volume_1d_ratio"), ("dist_high_20d", "dist_high_20d"),
          ("dist_low_20d", "dist_low_20d"), ("z_5d", "z_5d"),
          ("e_sessions_since", "e_sessions_since"), ("e_sessions_to_next", "e_sessions_to_next")]
LONG = ["mom_12_1", "ret_120d", "ret_60d", "dist_high_250d"]
DAILY_ONLY = ["ret_2d", "ret_60d", "ret_120d", "mom_12_1", "vol_60d", "dist_high_250d",
              "e_last_reaction"]


def ic_frame(df: pd.DataFrame, cols: list[str], target: str, level: str) -> pd.DataFrame:
    """Per-decision rank IC of every column against `target` at once (decision x column).

    A column tied across a whole decision has zero variance there and gives NaN, not 0:
    a constant carries no ordering, and averaging it in as 0 would dilute the others.
    """
    sub = df[cols + [target]].copy()
    g = sub.groupby(level=level)
    ranks = g.rank()
    dev = ranks - ranks.groupby(level=level).transform("mean")
    # A feature NaN on a row must not count that row's label, or the pair is misaligned.
    y = dev[target]
    out = {}
    for c in cols:
        ok = dev[c].notna() & y.notna()
        x = dev[c].where(ok)
        yy = y.where(ok)
        # Re-centre on the rows both have; the rank means above include rows the pair lacks.
        x = x - x.groupby(level=level).transform("mean")
        yy = yy - yy.groupby(level=level).transform("mean")
        num = (x * yy).groupby(level=level).sum()
        den = np.sqrt((x ** 2).groupby(level=level).sum() * (yy ** 2).groupby(level=level).sum())
        out[c] = num / den.where(den > 0)
    return pd.DataFrame(out)


def daily_mean(ic: pd.DataFrame | pd.Series):
    """Mean over a day's decisions, indexed by a naive date (1 row a day for daily)."""
    t = pd.DatetimeIndex(ic.index)
    return ic.groupby(pd.DatetimeIndex(t.date)).mean()


def stats(days: pd.Series) -> tuple[float, float]:
    days = days.dropna()
    if len(days) < 10:
        return np.nan, np.nan
    return days.mean(), days.mean() / (days.std() / np.sqrt(len(days) / STEPS))


def split_stats(days: pd.Series) -> dict:
    yr = days.index.year
    tr_ic, tr_t = stats(days[(yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[-1])])
    te_ic, te_t = stats(days[yr >= TEST_YEARS[0]])
    return {"train_ic": tr_ic, "train_t": tr_t, "test_ic": te_ic, "test_t": te_t}


def round1_frame(intraday: pd.DataFrame) -> pd.DataFrame:
    r1 = intraday[intraday["ctx_round"] == 1].copy()
    r1.index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(r1.index.get_level_values("execution").date),
         r1.index.get_level_values("ticker")], names=["date", "ticker"])
    return r1


def load(cache: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cache:
        d_p, i_p = Path(cache) / "daily.parquet", Path(cache) / "intraday.parquet"
        if d_p.exists() and i_p.exists():
            return pd.read_parquet(d_p), pd.read_parquet(i_p)
    d, i = train.daily_dataset().frame, train.intraday_dataset().frame
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        d.to_parquet(Path(cache) / "daily.parquet")
        i.to_parquet(Path(cache) / "intraday.parquet")
    return d, i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="directory to keep the built datasets in")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    daily, intra = load(args.cache)
    ours = list(data.load_universe())
    d30 = daily[daily.index.get_level_values("ticker").isin(ours)]
    r1 = round1_frame(intra)
    both = d30.join(r1, how="inner", lsuffix="_D", rsuffix="_I")
    both = both[both["d5_pct"].notna() & both["h35_pct"].notna()]

    def col(name: str, side: str) -> str:
        return f"{name}_{side}" if f"{name}_{side}" in both else name

    # ---- A. targets ---------------------------------------------------------------
    print("A. Targets at round 1, the 30 names, same rows "
          f"({both.index.get_level_values('date').nunique()} days)")
    rows = {}
    for a, b in [("d5_pct", "h35_pct"), ("d5_terminal", "h35_terminal")]:
        ic = train.per_decision_ic(both[a], both[b], "date")
        rows[f"{a} vs {b}"] = ic.groupby(ic.index.year).mean()
    rows = pd.DataFrame(rows)
    print(rows.T.round(3).to_string())
    print("  Same terminal return; the composites differ only in the path they score "
          "(5 opens vs 35 round fills).")

    # ---- B. models x targets ------------------------------------------------------
    pd_ = pd.read_parquet(PREDS / "daily_d5_pct.parquet")
    pi = pd.read_parquet(PREDS / "intraday_h35_pct.parquet")
    pi1 = round1_frame(pi.rename(columns={"target": "h35_pct"}).assign(ctx_round=intra["ctx_round"]))
    pd30 = pd_[pd_.index.get_level_values("ticker").isin(ours)]
    m = (pd30[["pred"]].rename(columns={"pred": "daily_pred"})
         .join(pi1[["pred"]].rename(columns={"pred": "intraday_pred"}), how="inner")
         .join(both[["d5_pct", "h35_pct"]], how="inner"))
    print(f"\nB. Both models on the same rows: round 1, the 30, "
          f"{m.index.get_level_values('date').nunique()} test days (IC by year, then pooled t)")
    rows = []
    for p in ("daily_pred", "intraday_pred"):
        for t in ("d5_pct", "h35_pct"):
            ic = train.per_decision_ic(m[p], m[t], "date")
            by = ic.groupby(ic.index.year).mean()
            rows.append({"score": p, "target": t, **by.round(4).to_dict(),
                         "pooled": ic.mean(), "t": stats(ic)[1]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    corr = train.per_decision_ic(m["daily_pred"], m["intraday_pred"], "date")
    print(f"  rank corr of the two scores: {corr.mean():.3f}")

    # What the daily score leans on within the 30. A column the intraday frame lacks
    # that the daily score loads heavily on is an edge the intraday model cannot learn,
    # however long it trains. e_* are filled as "no release in the window" (99), since
    # a tree reads their NaN as a value of its own.
    fcols = [c for c in daily.columns if not c.startswith(("ctx_", "d3_", "d5_", "is_"))]
    f30 = pd30[["pred"]].join(d30[fcols])
    for c in fcols:
        if c.startswith("e_"):
            f30[c] = f30[c].fillna(99)
    loads = pd.Series({c: train.per_decision_ic(f30["pred"], f30[c], "date").mean() for c in fcols})
    shared_d = {a for a, _ in SHARED}
    loads = loads.sort_values(key=np.abs, ascending=False).head(10)
    print("  the daily score's rank corr with its own inputs, the 30, test years (top 10; "
          "* = no intraday equivalent):")
    print("  " + "  ".join(f"{c}{'' if c in shared_d else '*'} {v:+.2f}" for c, v in loads.items()))

    # ---- C. shared features ---------------------------------------------------------
    print("\nC. Shared features, round 1, the 30 names, same rows. Daily column vs d5_pct, "
          "intraday column vs h35_pct;\n   'univ' = the daily column across the ~104-name "
          "universe on the same days. agree = rank corr of the two columns.")
    dcols = [col(a, "D") for a, _ in SHARED]
    icols = [col(b, "I") for _, b in SHARED]
    ic_d = daily_mean(ic_frame(both, dcols, "d5_pct", "date"))
    ic_i = daily_mean(ic_frame(both, icols, "h35_pct", "date"))
    ic_x = daily_mean(ic_frame(both, icols, "d5_pct", "date"))
    days = both.index.get_level_values("date").unique()
    univ = daily[daily.index.get_level_values("date").isin(days)]
    ic_u = daily_mean(ic_frame(univ, [a for a, _ in SHARED], "d5_pct", "date"))
    daily_old = daily[daily.index.get_level_values("date") < "2016-01-01"]
    ic_old = daily_mean(ic_frame(daily_old, [a for a, _ in SHARED], "d5_pct", "date"))
    rows = []
    for (a, b), dc, icn in zip(SHARED, dcols, icols):
        agree = train.per_decision_ic(both[dc], both[icn], "date").mean()
        s_u, s_d, s_i, s_x = (split_stats(ic_u[a]), split_stats(ic_d[dc]),
                              split_stats(ic_i[icn]), split_stats(ic_x[icn]))
        rows.append({"daily": a, "intraday": b, "agree": agree,
                     "univ_99-15": ic_old[a].mean(),
                     "univ_16-22": s_u["train_ic"], "univ_23+": s_u["test_ic"],
                     "D30_16-22": s_d["train_ic"], "D30_23+": s_d["test_ic"],
                     "I30_16-22": s_i["train_ic"], "I30_23+": s_i["test_ic"],
                     "I30_vs_d5_23+": s_x["test_ic"]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("  e_* ICs are among names with a release inside 10 sessions only (NaN otherwise, "
          "~5 of the 30 a day),\n  so their size does not carry to the full cross-section.")
    print("  daily-only columns (intraday has no equivalent), vs d5_pct:")
    ic_do = daily_mean(ic_frame(both, [col(c, "D") for c in DAILY_ONLY], "d5_pct", "date"))
    ic_dou = daily_mean(ic_frame(univ, DAILY_ONLY, "d5_pct", "date"))
    rows = [{"feature": c, **{f"30_{k}": v for k, v in split_stats(ic_do[col(c, 'D')]).items()
                              if k.endswith("ic")},
             **{f"univ_{k}": v for k, v in split_stats(ic_dou[c]).items() if k.endswith("ic")}}
            for c in DAILY_ONLY]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # No fit at all: the long-horizon columns summed with equal weight and a sign fixed
    # in advance (momentum). If this is stable across the split, the daily edge on the 30
    # is carried by columns the intraday frame never computes.
    comp = d30[LONG].fillna(0.0).sum(axis=1)
    ic_l = train.per_decision_ic(comp, d30["d5_pct"], "date")
    ic_l = ic_l[ic_l.index.year >= TRAIN_YEARS[0]]
    tr, te = split_stats(ic_l)["train_ic"], split_stats(ic_l)["test_ic"]
    print(f"  equal-weight {' + '.join(LONG)} on the 30 vs d5_pct: 2016-22 {tr:+.4f} "
          f"(t {split_stats(ic_l)['train_t']:.2f}), 2023+ {te:+.4f} (t {split_stats(ic_l)['test_t']:.2f})")

    # ---- D. intraday feature stability ------------------------------------------------
    per_ticker = [c for c in intra.columns if not c.startswith("ctx_")
                  and not c.startswith(("h7_", "h21_", "h35_"))]
    ic_all = daily_mean(ic_frame(intra, per_ticker, "h35_pct", "execution"))
    by_year = ic_all.groupby(ic_all.index.year).mean().T
    split = pd.DataFrame({c: split_stats(ic_all[c]) for c in per_ticker}).T
    table = by_year.join(split)
    table["flip"] = (np.sign(table["train_ic"]) != np.sign(table["test_ic"])) & \
        (table["train_t"].abs() >= 2)
    table.to_csv(data.ROOT / "reports" / "intraday_diagnosis_feature_ic.csv")
    print("\nD. Intraday per-ticker features vs h35_pct, all rounds, IC by year; "
          "train = 2016-22, test = 2023-26;\n   flip = sign reversed and |train t| >= 2")
    print(table.sort_values("train_t", key=np.abs, ascending=False).round(3).to_string())

    # ---- E. by round --------------------------------------------------------------------
    rnd = intra["ctx_round"].reindex(pi.index)
    print("\nE. Intraday model (h35_pct) test IC by round and year")
    rows = {}
    for r in range(1, 8):
        sel = pi[rnd.to_numpy() == r]
        ic = train.per_decision_ic(sel["pred"], sel["target"], "execution")
        by = ic.groupby(ic.index.year).mean()
        rows[r] = {**by.to_dict(), "pooled": ic.mean(), "t": stats(daily_mean(ic))[1]}
    print(pd.DataFrame(rows).T.round(3).to_string())
    # How much the score changes between rounds of one day: little, if it is mostly
    # session features; if it churns, the extra rounds are noise the fee will charge for.
    rk = pi["pred"].groupby(level="execution").rank(pct=True)
    rk_day = pd.DataFrame({"rk": rk, "day": pd.DatetimeIndex(rk.index.get_level_values(0).date),
                           "r": rnd.to_numpy()})
    r1s = rk_day[rk_day["r"] == 1].set_index(["day", rk_day[rk_day["r"] == 1].index.get_level_values(1)])["rk"]
    r4s = rk_day[rk_day["r"] == 4].set_index(["day", rk_day[rk_day["r"] == 4].index.get_level_values(1)])["rk"]
    j = pd.DataFrame({"r1": r1s, "r4": r4s}).dropna()
    print(f"  rank corr of a name's score at round 1 vs round 4 of the same day: "
          f"{train.per_decision_ic(j['r1'], j['r4'], 'day').mean():.3f}")

    # ---- F. data red flags ------------------------------------------------------------------
    ex = intra.index.get_level_values("execution")
    per_day = pd.Series(1, index=ex).groupby(ex.date).size() / 30
    print("\nF. Data red flags")
    print(f"  decisions per day: {per_day.value_counts().to_dict()} (4-round days are half-days)")
    feats = [c for c in intra.columns if not c.startswith(("h7_", "h21_", "h35_"))]
    nan_by_year = intra[feats].isna().groupby(ex.year).mean()
    varied = nan_by_year.columns[(nan_by_year.max() - nan_by_year.min()) > 0.01]
    print("  NaN share by year, columns whose share moves by more than 1pp:")
    print(nan_by_year[varied].round(3).T.to_string())
    g = intra[feats].groupby(level="execution")
    tied = (g.nunique(dropna=True) <= 1).groupby(intra.groupby(level="execution")["ctx_round"]
                                                  .first()).mean()
    tied = tied.loc[:, (tied > 0.05).any()]
    tied = tied[[c for c in tied.columns if not c.startswith("ctx_")]]
    print("  share of decisions where a per-ticker column is tied across all 30 names, by round:")
    print(tied.round(3).T.to_string())
    n_ctx = sum(c.startswith("ctx_") for c in feats)
    print(f"  plus {n_ctx} ctx_* columns, constant within every decision by construction: "
          "they can only shift a decision's level, which a rank IC ignores, or gate "
          "interactions.")

    # ---- G. history and overfitting ------------------------------------------------------------
    print("\nG. How much independent history each model learns from (fold 2023)")
    d_tr = daily[daily.index.get_level_values("date") < "2023-01-01"]
    i_tr = intra[ex < pd.Timestamp("2023-01-01", tz=ex.tz)]
    d_days = d_tr.index.get_level_values("date").nunique()
    i_days = pd.Index(i_tr.index.get_level_values("execution").date).nunique()
    for name, rows_, n_days, names in [("daily", len(d_tr), d_days, len(d_tr) / d_days),
                                        ("intraday", len(i_tr), i_days, 30)]:
        print(f"  {name:8s} rows {rows_:>8,}  days {n_days:>5,}  names/day {names:5.0f}  "
              f"independent 5-day windows x names ~ {n_days / STEPS * names:>9,.0f}")
    wf = []
    for model in ("daily", "intraday"):
        w = pd.read_csv(data.ROOT / "reports" / f"walkforward_{model}.csv")
        w = w[w["target"].isin(["d5_pct", "h35_pct"])]
        wf.append(w[["target", "year", "best_val_spearman", "ic"]])
    print("  validation (inner leave-one-block-out, pooled Spearman) vs test IC:")
    print(pd.concat(wf).round(4).to_string(index=False))

    # An IC-sign composite: each per-ticker feature signed by its 2016-22 IC and
    # weighted by its train t, summed. No fit beyond that, so a gap between its train
    # and test IC is non-stationarity, not an ensemble memorising noise.
    w_ = split["train_t"].where(split["train_t"].abs() >= 1.0, 0.0).fillna(0.0)
    X = intra[per_ticker].fillna(0.0)
    comp = (X * w_.reindex(per_ticker).to_numpy()).sum(axis=1)
    ic_c = daily_mean(train.per_decision_ic(comp, intra["h35_pct"], "execution"))
    by = ic_c.groupby(ic_c.index.year).mean()
    print(f"\n  IC-sign composite fitted on 2016-22 intraday ICs ({int((w_ != 0).sum())} "
          "features with |t| >= 1), scored on h35_pct:")
    print("  " + "  ".join(f"{y}:{v:+.3f}" for y, v in by.items()))
    # The same recipe with signs taken from the broad daily universe 1999-2022, applied to
    # the intraday columns at round 1: does history from 104 names travel to the 30?
    old = daily[daily.index.get_level_values("date") < "2023-01-01"]
    ic_bd = daily_mean(ic_frame(old, [a for a, _ in SHARED], "d5_pct", "date"))
    wt = pd.Series({b: stats(ic_bd[a])[1] for a, b in SHARED})
    wt = wt.where(wt.abs() >= 1.0, 0.0)
    comp2 = (r1[wt.index].fillna(0.0) * wt.to_numpy()).sum(axis=1)
    ic_c2 = train.per_decision_ic(comp2, r1["h35_pct"], "date")
    by2 = ic_c2.groupby(ic_c2.index.year).mean()
    print(f"  same recipe, signs from the daily universe 1999-2022 ({int((wt != 0).sum())} shared "
          "features), intraday round 1, h35_pct:")
    print("  " + "  ".join(f"{y}:{v:+.3f}" for y, v in by2.items()))
    print(f"  pooled 2016-22 {stats(ic_c2[ic_c2.index.year < 2023])[0]:+.4f}   "
          f"2023+ {stats(ic_c2[ic_c2.index.year >= 2023])[0]:+.4f} "
          f"(t {stats(ic_c2[ic_c2.index.year >= 2023])[1]:.2f})")


if __name__ == "__main__":
    main()
