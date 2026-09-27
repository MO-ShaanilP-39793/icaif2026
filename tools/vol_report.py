"""Does the HAR forecast beat trailing vol out of sample? (build plan step 4, part 1)

    .venv/bin/python tools/vol_report.py [--save-forecasts]

Walk-forward from 2017 on `markets.intraday_info_bars()` (Alpaca's :30 grid from 2016,
then Yahoo 60m), refit each quarter on rows whose target window had ended. For the 30
names pooled, the equal-weight basket, and each name alone, it scores four forecasts of
mean daily variance over the next 1 and 3 sessions: log-HAR, and the trailing baselines
rv20 (mean RV of the last 20 sessions: the headline one), rw5 (last 5) and cc20
(variance of 20 daily close-to-close returns).

Two losses, because each flatters a different mistake. QLIKE on variance is what a
vol-scaled position pays for (it punishes a forecast that runs low harder than one that
runs high). R^2 of log RV scores ranking and level in logs; HAR enters it with its log
fit, not its bias-corrected variance, which would charge it s^2/2 of bias it does not
have. The trailing baselines enter as they are: the log of a mean runs above the mean of
logs, and that bias is part of what using trailing vol costs. `corr2_*` (squared
correlation, bias forgiven) says how much of an R^2 gap is that level offset alone.
`ratio_*` is mean realised / forecast: above 1, a vol target sized on it runs hot.

The t-stat is on the daily cross-sectional mean of the QLIKE difference, with the sample
counted as n_days / H, because overlapping H-day targets are not independent. Writes
reports/vol_forecast.csv. --save-forecasts also writes data/derived/vol_forecasts.parquet,
the forecasts later steps size and scale with.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import data, markets, vol  # noqa: E402

FIRST_TEST = "2017-01-01"
BASELINES = ("rv20", "rw5", "cc20")


def _scores(s: pd.DataFrame, h: int) -> dict:
    """QLIKE and log R^2 for every forecaster on one common sample."""
    y = s[f"y_h{h}"]
    row = {"n": len(s)}
    for m in ("har",) + BASELINES:
        row[f"qlike_{m}"] = vol.qlike(y, s[f"{m}_h{h}"]).mean()
        log_fc = s[f"harlog_h{h}"] if m == "har" else np.log(s[f"{m}_h{h}"])
        row[f"r2_{m}"] = vol.oos_r2(np.log(y), log_fc)
        row[f"corr2_{m}"] = np.corrcoef(np.log(y), log_fc)[0, 1] ** 2
        row[f"ratio_{m}"] = (y / s[f"{m}_h{h}"]).mean()
    sess = s.index.get_level_values("session")
    for base in BASELINES:
        diff = vol.qlike(y, s[f"{base}_h{h}"]) - vol.qlike(y, s[f"har_h{h}"])
        daily = diff.groupby(sess).mean()
        row[f"qlike_gain_vs_{base}_%"] = 100 * (1 - row["qlike_har"] / row[f"qlike_{base}"])
        row[f"t_vs_{base}"] = daily.mean() / (daily.std() / np.sqrt(len(daily) / h))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-forecasts", action="store_true")
    args = ap.parse_args()

    info = markets.intraday_info_bars()
    rv = vol.realised_variance(info)
    fc = vol.walk_forward(rv, first_test=FIRST_TEST)
    cc = vol.close_to_close_variance(info).stack(future_stack=True).rename("cc20")
    fc = fc.join(cc, how="left")
    for h in vol.HORIZONS:
        fc[f"cc20_h{h}"] = fc["cc20"]
    fc = fc.drop(columns="cc20")
    if args.save_forecasts:
        derived = data.ROOT / "data" / "derived"
        derived.mkdir(parents=True, exist_ok=True)
        fc.to_parquet(derived / "vol_forecasts.parquet")

    rows = []
    for h in vol.HORIZONS:
        cols = [f"har_h{h}", f"harlog_h{h}", f"y_h{h}"] + [f"{b}_h{h}" for b in BASELINES]
        # One common sample: a forecaster excused from the rows it has no value on
        # would be scored on easier days than the rest.
        d = fc[cols].dropna()
        ticker = d.index.get_level_values("ticker")
        year = pd.to_datetime(d.index.get_level_values("session")).year
        for scope, sel in (("stocks", ticker != vol.MARKET), ("market", ticker == vol.MARKET)):
            for y in sorted(set(year[sel])) + ["all"]:
                mask = sel & ((year == y) if y != "all" else True)
                rows.append({"horizon": h, "scope": scope, "year": y, **_scores(d[mask], h)})
        for t in sorted(set(ticker) - {vol.MARKET}):
            rows.append({"horizon": h, "scope": t, "year": "all", **_scores(d[ticker == t], h)})
    table = pd.DataFrame(rows)
    out = data.ROOT / "reports" / "vol_forecast.csv"
    table.to_csv(out, index=False)

    show = ["horizon", "scope", "year", "n", "r2_har", "r2_rv20", "qlike_har", "qlike_rv20",
            "qlike_gain_vs_rv20_%", "t_vs_rv20", "qlike_gain_vs_cc20_%", "ratio_har", "ratio_rv20"]
    agg = table[table["scope"].isin(["stocks", "market"])]
    print(f"Walk-forward from {FIRST_TEST}, quarterly refit. QLIKE lower is better; gain = % "
          "QLIKE reduction by HAR; t on the daily loss difference, n_eff = n_days / H.")
    print(agg[show].round(4).to_string(index=False))

    print("\nVerdict: HAR vs rv20 (trailing 20-session realised variance)")
    for h in vol.HORIZONS:
        for scope in ("stocks", "market"):
            yrs = agg[(agg["horizon"] == h) & (agg["scope"] == scope) & (agg["year"] != "all")]
            pooled = agg[(agg["horizon"] == h) & (agg["scope"] == scope)
                         & (agg["year"] == "all")].iloc[0]
            print(f"  {h}d {scope:6s}: QLIKE lower in {(yrs['qlike_har'] < yrs['qlike_rv20']).sum()}"
                  f"/{len(yrs)} years, log R^2 higher in {(yrs['r2_har'] > yrs['r2_rv20']).sum()}"
                  f"/{len(yrs)}; pooled R^2 {pooled['r2_har']:.3f} vs {pooled['r2_rv20']:.3f}, "
                  f"QLIKE gain {pooled['qlike_gain_vs_rv20_%']:.1f}% (t {pooled['t_vs_rv20']:.1f})")
        per = table[(table["horizon"] == h) & ~table["scope"].isin(["stocks", "market"])]
        print(f"  {h}d per name: HAR wins QLIKE for {(per['qlike_har'] < per['qlike_rv20']).sum()}"
              f"/{len(per)}, log R^2 for {(per['r2_har'] > per['r2_rv20']).sum()}/{len(per)}")
    print(f"\nwrote {out.relative_to(data.ROOT)}")


if __name__ == "__main__":
    main()
