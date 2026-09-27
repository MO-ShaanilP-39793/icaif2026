"""Does the upside model still say anything once its volatility bet is divided out?

    .venv/bin/python tools/upside_residual_report.py

Reads pass 1's out-of-sample predictions (output/preds/, 2023-26) and writes
reports/upside_residual.csv, then prints the pooled table.

The upside targets (`*_up_pct`, alphaBT's Target 2) score IC ~0.2, but trailing
volatility alone ranks them at 0.13-0.20: a name that swings either way has high
"upside on fills". A model that learned only that would read as the best model we have
and would buy the most volatile names, which the drawdown and Sharpe ranks punish.
alphaBT's answer is `risk_adjusted = prob / vol^gamma`. This report asks whether what
is left after that division predicts what we earn, and whether it adds to the
composite model:

- scores: the upside prediction divided by vol^gamma (gamma 0, 0.5, 1, 1.5), both as
  the raw prediction and as its within-day percentile rank, since the raw prediction
  barely varies (~0.45-0.55) and dividing it by vol is nearly the same as ranking on
  1/vol; the upside rank residualised on the vol rank; low volatility on its own, the
  control every risk-adjusted score must beat to claim signal of its own; the
  composite prediction; and rank blends of the composite with each of those.
- outcomes, all 5 sessions from the decision day's open: the composite target, the
  plain open-to-open return, and that return over trailing 20-session vol.

Trailing vol is the std of daily log closes through the close BEFORE the decision
date (rolling, then shift(1)). Without the shift the window includes day d's close,
which moves with the outcome's first day: a leak that flatters every vol-scaled score.

IC is the per-decision rank correlation, averaged over days. Its t-stat uses n_days / 5
effective samples, because consecutive 5-day labels share four-fifths of their path.
"The 30" re-ranks the daily model among the competition names alone; a rank IC is
invariant to that re-ranking, so it is the subset that matters, not the rescale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import daily_features, data, external, train  # noqa: E402

PREDS = data.ROOT / "output" / "preds"
OUT = data.ROOT / "reports" / "upside_residual.csv"
HORIZON = 5
GAMMAS = (0.0, 0.5, 1.0, 1.5)
VOL_WINDOWS = {20: 15, 60: 45}  # window -> min_periods, as daily_features uses them
OUTCOMES = ("target_comp", "ret5", "ret5_over_vol")


def trailing_vol(close: pd.DataFrame) -> dict[int, pd.DataFrame]:
    logret = np.log(close).diff()
    return {w: logret.rolling(w, min_periods=m).std().shift(1) for w, m in VOL_WINDOWS.items()}


def forward_return(opens: pd.DataFrame) -> pd.DataFrame:
    """open[d+5] / open[d] - 1, blank where any open on the path is missing.

    Same rule as the labels: a path with a hole is not a return we could have earned,
    and filling it would pair a stale price with a catch-up jump.
    """
    ret = opens.shift(-HORIZON) / opens - 1
    hole = opens.isna().astype(float).rolling(HORIZON + 1).sum().shift(-HORIZON) > 0
    return ret.mask(hole)


def _stack(panel: pd.DataFrame, index: pd.MultiIndex) -> pd.Series:
    s = panel.stack(future_stack=True)
    s.index = s.index.set_names(index.names)
    return s.reindex(index)


def _pct(s: pd.Series, level: str) -> pd.Series:
    return s.groupby(level=level).rank(pct=True)


def _residual(y: pd.Series, x: pd.Series, level: str) -> pd.Series:
    """Per-decision OLS residual of y on x: the part of y a vol ranking can't explain."""
    df = pd.DataFrame({"y": y, "x": x})
    g = df.groupby(level=level)
    dm = df - g.transform("mean")
    beta = (dm["x"] * dm["y"]).groupby(level=level).sum() / (dm["x"] ** 2).groupby(level=level).sum()
    return dm["y"] - dm["x"] * beta.reindex(dm.index.get_level_values(level)).to_numpy()


def scores(up: pd.Series, comp: pd.Series, vol: pd.Series, level: str) -> dict[str, pd.Series]:
    """Every score for one scope and one vol window. Ranks are taken within the scope,
    so a blend on the 30 weighs the two models by their order among the 30."""
    r_up, r_comp, r_vol = _pct(up, level), _pct(comp, level), _pct(vol, level)
    s = {"comp": comp, "low_vol": -vol, "up_resid_vol": _residual(r_up, r_vol, level)}
    for g in GAMMAS:
        s[f"up/vol^{g:g}"] = up / vol ** g
        if g:
            s[f"rank_up/vol^{g:g}"] = r_up / vol ** g
    for name in [k for k in s if k.startswith(("up", "rank_up"))] + ["low_vol"]:
        s[f"comp+{name}"] = r_comp + _pct(s[name], level)
    return s


def _stats(ic: pd.Series) -> dict:
    return {"ic": ic.mean(), "t": ic.mean() / (ic.std() / np.sqrt(len(ic) / HORIZON)),
            "pos_share": (ic > 0).mean(), "n_days": len(ic)}


def evaluate(frame: pd.DataFrame, level: str, model: str, scope: str, windows) -> list[dict]:
    rows = []
    for w in windows:
        f = frame.dropna(subset=[f"vol{w}"])
        years = pd.DatetimeIndex(f.index.get_level_values(level)).year
        for sname, score in scores(f["up"], f["comp"], f[f"vol{w}"], level).items():
            for outcome in OUTCOMES:
                ic = train.per_decision_ic(score, f[outcome], level)
                ic_year = pd.DatetimeIndex(ic.index).year
                for year in sorted(set(years)) + ["2023-26"]:
                    part = ic if year == "2023-26" else ic[ic_year == year]
                    rows.append({"model": model, "scope": scope, "vol_window": w, "score": sname,
                                 "outcome": outcome, "year": year, **_stats(part)})
    return rows


def main() -> None:
    p = daily_features.panels(external.load("yahoo_daily_universe"))
    vols = trailing_vol(p["close"])
    ret5 = forward_return(p["open"])
    ours = set(data.load_universe())

    def assemble(up_path: str, comp_path: str, level: str) -> pd.DataFrame:
        up, comp = pd.read_parquet(PREDS / up_path), pd.read_parquet(PREDS / comp_path)
        f = pd.DataFrame({"up": up["pred"], "comp": comp["pred"], "target_comp": comp["target"]})
        if f["comp"].isna().any() or len(f) != len(up):
            raise ValueError(f"{up_path} and {comp_path} do not cover the same rows")
        return f

    rows = []
    # Daily: index (date, ticker); the vol and return panels share the date grid.
    d = assemble("daily_d5_up_pct.parquet", "daily_d5_pct.parquet", "date")
    for w, v in vols.items():
        d[f"vol{w}"] = _stack(v, d.index)
    d["ret5"] = _stack(ret5, d.index)
    d["ret5_over_vol"] = d["ret5"] / d["vol20"]
    _report_coverage("daily", d)
    rows += evaluate(d, "date", "daily", "universe", VOL_WINDOWS)
    rows += evaluate(d[d.index.get_level_values("ticker").isin(ours)], "date", "daily", "the 30", VOL_WINDOWS)

    # Intraday round 1 only: the 09:30 execution is the daily open, so the same daily
    # vol and 5-session return apply exactly. Other rounds fill mid-session, where a
    # daily-open return would be the wrong entry price.
    i = assemble("intraday_h35_up_pct.parquet", "intraday_h35_pct.parquet", "execution")
    ex = i.index.get_level_values("execution")
    i = i[(ex.hour == 9) & (ex.minute == 30)]
    i.index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(i.index.get_level_values("execution").date), i.index.get_level_values("ticker")],
        names=["date", "ticker"])
    for w, v in vols.items():
        i[f"vol{w}"] = _stack(v, i.index)
    i["ret5"] = _stack(ret5, i.index)
    i["ret5_over_vol"] = i["ret5"] / i["vol20"]
    _report_coverage("intraday r1", i)
    rows += evaluate(i, "date", "intraday_r1", "the 30", VOL_WINDOWS)

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUT, index=False)
    _print(table, d, i, ours)


def _report_coverage(name: str, f: pd.DataFrame) -> None:
    # A score or outcome blank on many rows would silently shrink the sample it is
    # scored on, and two scores on different samples are not comparable.
    blank = f[["vol20", "vol60", "ret5"]].isna().sum()
    print(f"{name}: {len(f)} rows, {f.index.get_level_values(0).nunique()} days; "
          f"blank vol20 {blank['vol20']}, vol60 {blank['vol60']}, ret5 {blank['ret5']}")


def _print(table: pd.DataFrame, d: pd.DataFrame, i: pd.DataFrame, ours: set) -> None:
    def xcorr(f, a, b):
        g = f[[a, b]].dropna().groupby(level="date")
        return g.apply(lambda x: x[a].rank().corr(x[b].rank())).mean()

    d30 = d[d.index.get_level_values("ticker").isin(ours)]
    print("\nmean per-day rank corr with vol20:  "
          f"daily up {xcorr(d, 'up', 'vol20'):+.2f}, comp {xcorr(d, 'comp', 'vol20'):+.2f} (universe); "
          f"up {xcorr(d30, 'up', 'vol20'):+.2f}, comp {xcorr(d30, 'comp', 'vol20'):+.2f} (the 30); "
          f"intraday r1 up {xcorr(i, 'up', 'vol20'):+.2f}, comp {xcorr(i, 'comp', 'vol20'):+.2f}")

    pooled = table[(table["year"] == "2023-26") & (table["vol_window"] == 20)]
    for (model, scope), g in pooled.groupby(["model", "scope"], sort=False):
        ic = g.pivot(index="score", columns="outcome", values="ic")[list(OUTCOMES)]
        t = g.pivot(index="score", columns="outcome", values="t")[list(OUTCOMES)]
        shown = ic.round(3).astype(str) + " (" + t.round(1).astype(str) + ")"
        shown = shown.reindex(sorted(ic.index, key=_order))
        print(f"\n{model}, {scope}, vol20, pooled 2023-26: IC (t, n_days/5)")
        print(shown.to_string())

    by_year = table[(table["vol_window"] == 20) & (table["year"] != "2023-26")
                    & table["score"].isin(["comp", "low_vol", "up/vol^0", "up_resid_vol",
                                           "rank_up/vol^1", "comp+rank_up/vol^1", "comp+low_vol"])]
    for outcome in OUTCOMES:
        wide = by_year[by_year["outcome"] == outcome].pivot_table(
            index=["model", "scope", "score"], columns="year", values="ic", sort=False)
        print(f"\nby year, outcome {outcome}, vol20")
        print(wide.round(3).to_string())

    v60 = table[(table["year"] == "2023-26") & (table["vol_window"] == 60)
                & table["score"].isin(["low_vol", "up_resid_vol", "rank_up/vol^1", "comp+rank_up/vol^1"])]
    print("\nvol60 instead of vol20, pooled 2023-26")
    print(v60.pivot_table(index=["model", "scope", "score"], columns="outcome", values="ic",
                          sort=False)[list(OUTCOMES)].round(3).to_string())


def _order(score: str) -> tuple:
    blend = score.startswith("comp+")
    base = score.removeprefix("comp+")
    head = 0 if base == "comp" else 1 if base == "low_vol" else 2 if base.startswith("up/") \
        else 3 if base.startswith("rank_up") else 4
    return (blend, head, base)


if __name__ == "__main__":
    main()
