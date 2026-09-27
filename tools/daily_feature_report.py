"""Univariate signal check for the daily model, era by era.

    .venv/bin/python tools/daily_feature_report.py

Writes reports/daily_feature_ic.csv. IC is the per-day rank correlation within that
day's universe, averaged over the era. The t-stat uses n_days / horizon, because
consecutive days' 5-day labels share most of their path.

Eras split where Yahoo's survivorship changes. If a feature's IC in 2001-2010
(50-66% of members priced) differs sharply from 2023+ (95%+), old data teaches
something that no longer holds: the case for a later training start.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import daily_features, data, earnings, external, universe  # noqa: E402

ERAS = {"2001-10": ("2001", "2010"), "2011-19": ("2011", "2019"),
        "2020-22": ("2020", "2022"), "2023+": ("2023", "2026")}
TARGETS = {"d3_pct": 3, "d5_pct": 5, "d5_up_pct": 5, "d5_terminal": 5}


def ic_by_day(x: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    g = df.groupby(level="date")
    df = df.assign(x=g["x"].rank(), y=g["y"].rank())
    d = df - df.groupby(level="date").transform("mean")
    num = (d["x"] * d["y"]).groupby(level="date").sum()
    den = np.sqrt((d["x"] ** 2).groupby(level="date").sum() * (d["y"] ** 2).groupby(level="date").sum())
    return (num / den).dropna()


def main() -> None:
    daily = external.load("yahoo_daily_universe")
    ctx = external.load("yahoo_daily_context")
    membership = universe.load_membership()
    mask = universe.build(daily, membership)
    events = earnings.quarterly(external.load("earnings"))
    f = daily_features.build(daily, mask, ctx, events)
    lab = daily_features.build_labels(daily, mask)
    df = f.join(lab, how="inner")
    print(f"{len(df):,} rows, {df.index.get_level_values('date').nunique():,} days, "
          f"median {df.groupby(level='date').size().median():.0f} names a day")

    per_name = [c for c in f.columns if not c.startswith(("ctx_", "is_"))]
    dates = df.index.get_level_values("date")
    rows = []
    for era, (lo, hi) in ERAS.items():
        sel = df[(dates >= lo) & (dates <= f"{hi}-12-31")]
        for target, h in TARGETS.items():
            for col in per_name:
                ic = ic_by_day(sel[col], sel[target])
                if ic.empty:
                    continue
                rows.append({"era": era, "target": target, "feature": col, "ic": ic.mean(),
                             "t": ic.mean() / (ic.std() / np.sqrt(len(ic) / h)),
                             "coverage": sel[col].notna().mean()})
    table = pd.DataFrame(rows)
    table.to_csv(data.ROOT / "reports" / "daily_feature_ic.csv", index=False)
    for target in ("d5_pct", "d5_up_pct"):
        wide = table[table["target"] == target].pivot_table(index="feature", columns="era", values="ic")
        wide = wide[list(ERAS)]
        wide["abs_max"] = wide.abs().max(axis=1)
        print(f"\nIC vs {target} by era")
        print(wide.sort_values("abs_max", ascending=False).drop(columns="abs_max").round(4).to_string())

    rng = np.random.default_rng(0)
    recent = df[dates >= "2023"]
    shuffled = recent["d5_pct"].groupby(level="date").transform(
        lambda s: s.sample(frac=1, random_state=int(rng.integers(1 << 31))).to_numpy())
    canary = max(abs(ic_by_day(recent[c], shuffled).mean()) for c in per_name)
    print(f"\ncanary (2023+, d5 shuffled within day): max |IC| {canary:.4f}")
    cov = table[(table["target"] == "d5_pct") & (table["era"] == "2023+")].set_index("feature")["coverage"]
    print("lowest coverage 2023+:", cov.sort_values().head(4).round(3).to_dict())


if __name__ == "__main__":
    main()
