"""Univariate signal check for step 3: does any feature rank the labels at all?

    .venv/bin/python tools/feature_report.py

Writes reports/feature_ic.csv and prints it, plus two checks that should come out
boring: IC against a label shuffled within each decision (must be ~0), and agreement
of the same features computed from organizer bars and from public bars over their
overlap. If that agreement is low, a model trained on one grid is scored live on the
other and means something else there.

IC here is the per-decision rank correlation, averaged per day over the rounds we train
on (1 and 4), then across days. Its t-stat uses an effective sample of
n_days / label_horizon_days, because overlapping labels are not independent: a 3-day
label on consecutive days shares two-thirds of its path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import data, features, labels, markets  # noqa: E402

TRAIN_ROUNDS = (1, 4)
HORIZON_DAYS = {"h7": 1, "h21": 3}


def ic_by_decision(x: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    g = df.groupby(level="execution")
    df["x"] = g["x"].rank()
    df["y"] = g["y"].rank()
    d = df - df.groupby(level="execution").transform("mean")
    num = (d["x"] * d["y"]).groupby(level="execution").sum()
    den = np.sqrt((d["x"] ** 2).groupby(level="execution").sum()
                  * (d["y"] ** 2).groupby(level="execution").sum())
    return (num / den).dropna()


def summarise_ic(ic: pd.Series, horizon_days: int) -> dict:
    daily = ic.groupby(ic.index.date).mean()
    n_eff = len(daily) / horizon_days
    return {"ic": daily.mean(), "t": daily.mean() / (daily.std() / np.sqrt(n_eff)),
            "pos_share": (daily > 0).mean()}


def main() -> None:
    market = markets.research_market()
    f = features.build(market.info_bars)
    lab = labels.build(markets.label_exec_prices())
    df = f.join(lab, how="inner")
    df = df[df["ctx_round"].isin(TRAIN_ROUNDS)]
    per_ticker = [c for c in f.columns if not c.startswith("ctx_")]

    rows = []
    for h, days in HORIZON_DAYS.items():
        for target in (f"{h}_score", f"{h}_terminal"):
            for col in per_ticker:
                s = summarise_ic(ic_by_decision(df[col], df[target]), days)
                rows.append({"horizon": h, "target": target.split("_", 1)[1], "feature": col, **s})
    table = pd.DataFrame(rows)
    out = data.ROOT / "reports"
    table.to_csv(out / "feature_ic.csv", index=False)
    wide = table.pivot_table(index="feature", columns=["horizon", "target"], values="ic")
    wide["abs_max"] = wide.abs().max(axis=1)
    print("Univariate IC (per-decision rank corr, daily mean), rounds 1 and 4, 2021-2026")
    print(wide.sort_values("abs_max", ascending=False).drop(columns="abs_max").round(4).to_string())
    tmax = table.loc[table["t"].abs().idxmax()]
    print(f"\nlargest |t|: {tmax['feature']} vs {tmax['horizon']} {tmax['target']}: "
          f"IC {tmax['ic']:.4f}, t {tmax['t']:.2f}")

    rng = np.random.default_rng(0)
    shuffled = df["h7_score"].groupby(level="execution").transform(
        lambda s: s.sample(frac=1, random_state=int(rng.integers(1 << 31))).to_numpy())
    canary = [summarise_ic(ic_by_decision(df[c], shuffled), 1)["ic"] for c in per_ticker]
    print(f"\ncanary (h7 score shuffled within decision): max |IC| over features "
          f"{np.max(np.abs(canary)):.4f}")

    organizer, _ = data.load_organizer_bars()
    p60 = markets.latest_public("60m")
    lo, hi = p60["start"].min() + pd.Timedelta(days=40), organizer["end"].max()
    fo = features.build(organizer[organizer["start"] >= lo - pd.Timedelta(days=45)])
    fp = features.build(p60[(p60["start"] >= lo - pd.Timedelta(days=45)) & (p60["end"] <= hi)])
    both = fo.join(fp, how="inner", lsuffix="_org", rsuffix="_pub")
    both = both[both.index.get_level_values("execution") >= lo]
    agree = {}
    for col in per_ticker:
        for rnd in (1, 4):
            sel = both[both["ctx_round_org"] == rnd]
            agree[(col, rnd)] = sel[f"{col}_org"].corr(sel[f"{col}_pub"])
    agree = pd.Series(agree).unstack()
    agree.columns = [f"round_{c}" for c in agree.columns]
    agree.to_csv(out / "feature_grid_agreement.csv")
    print(f"\nsame feature from organizer vs public bars, {lo.date()}..{hi.date()} "
          "(correlation of centred ranks)")
    print(agree.round(3).sort_values("round_4").to_string())


if __name__ == "__main__":
    main()
