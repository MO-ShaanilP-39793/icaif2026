"""Summarise pass 1: out-of-sample IC by model, target and year, and whether the two
models add up.

    .venv/bin/python tools/pass1_report.py

Reads output/preds/{daily,intraday}_<target>.parquet (written by train_walkforward.py)
and writes reports/pass1_summary.csv.

The comparison that decides how the scores are used is on the 30 names at round 1,
the one decision both models make. The daily model scores ~104 names; its predictions
are re-ranked within the 30. If the two models' scores are only loosely correlated,
a blend should beat either, which is the point of having both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import data, train  # noqa: E402

PREDS = data.ROOT / "output" / "preds"
PAIRS = [("d5_pct", "h35_pct"), ("d5_up_pct", "h35_up_pct")]  # same 5-day horizon


def _ic_stats(ic: pd.Series, steps: float) -> dict:
    days = ic.groupby(pd.DatetimeIndex(ic.index).date).mean()
    return {"ic": days.mean(), "t": days.mean() / (days.std() / np.sqrt(len(days) / steps)),
            "pos_share": (days > 0).mean(), "n_days": len(days)}


def main() -> None:
    ours = set(data.load_universe())
    rows = []
    loaded = {}
    for path in sorted(PREDS.glob("*.parquet")):
        if "smoke" in path.stem:
            continue
        model, target = path.stem.split("_", 1)
        p = pd.read_parquet(path)
        loaded[(model, target)] = p
        level = "date" if model == "daily" else "execution"
        steps = 5
        for year, g in list(p.groupby("year")) + [("2023-26", p)]:
            ic = train.per_decision_ic(g["pred"], g["target"], level)
            rows.append({"model": model, "target": target, "year": year, "scope": "universe",
                         **_ic_stats(ic, steps)})
            if model == "daily":
                g30 = g[g.index.get_level_values("ticker").isin(ours)]
                ic30 = train.per_decision_ic(g30["pred"], g30["target"], level)
                rows.append({"model": model, "target": target, "year": year, "scope": "the 30",
                             **_ic_stats(ic30, steps)})
    table = pd.DataFrame(rows)
    table.to_csv(data.ROOT / "reports" / "pass1_summary.csv", index=False)
    wide = table.pivot_table(index=["model", "target", "scope"], columns="year", values="ic")
    print("Out-of-sample IC (per-decision rank correlation, daily mean)")
    print(wide.round(4).to_string())
    pooled = table[table["year"] == "2023-26"].set_index(["model", "target", "scope"])
    print("\n2023-26 pooled:")
    print(pooled[["ic", "t", "pos_share", "n_days"]].round(3).to_string())

    # Round 1 on the 30 names: the decision both models make.
    for d_t, i_t in PAIRS:
        if ("daily", d_t) not in loaded or ("intraday", i_t) not in loaded:
            continue
        d = loaded[("daily", d_t)]
        d = d[d.index.get_level_values("ticker").isin(ours)]
        d.index = d.index.set_names(["day", "ticker"])
        i = loaded[("intraday", i_t)]
        ex = i.index.get_level_values("execution")
        i = i[(ex.hour == 9) & (ex.minute == 30)]
        i.index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(i.index.get_level_values("execution").date), i.index.get_level_values("ticker")],
            names=["day", "ticker"])
        both = d[["pred"]].join(i[["pred", "target"]], lsuffix="_daily", rsuffix="_intraday", how="inner")
        r = both.groupby(level="day")
        rank_d = r["pred_daily"].rank(pct=True)
        rank_i = r["pred_intraday"].rank(pct=True)
        corr = pd.DataFrame({"d": rank_d, "i": rank_i}).groupby(level="day").corr().xs("d", level=1)["i"].mean()
        print(f"\nround 1, the 30 names, {d_t} vs {i_t} (target {i_t}), {both.index.get_level_values('day').nunique()} days:")
        print(f"  rank correlation of the two models' scores: {corr:.3f}")
        for name, score in [("daily", rank_d), ("intraday", rank_i), ("blend 50/50", (rank_d + rank_i) / 2)]:
            ic = train.per_decision_ic(score, both["target"], "day")
            print(f"  {name:12s} IC {ic.mean():+.4f}  t {ic.mean() / (ic.std() / np.sqrt(len(ic) / 5)):.2f}")


if __name__ == "__main__":
    main()
