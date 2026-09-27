"""Fetch Alpaca 30m bars for the 30 names since 2016, and check them against both feeds.

    .venv/bin/python tools/alpaca_report.py [--reuse]

Writes data/public/alpaca_30m_<date>.parquet and reports/alpaca_parity.json:
  - vs the organizer panel (2021-2025, after spin-off adjustment): session open, close,
    high, low. A persistent step here means an adjustment one feed has and the other lacks.
  - vs Yahoo 60m on the live grid (Nov 2023 on): the opens the competition fills at.
    This decides whether Alpaca can stand in for Yahoo as the fill price before 2023.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import alpaca, data, markets, parity  # noqa: E402

CACHE = data.ROOT / "data" / "public"


def _bps(a, b):
    return 1e4 * (a / b - 1)


def _stats(x: pd.Series) -> dict:
    a = x.abs().dropna()
    return {"n": int(a.size), "median_abs_bps": round(float(a.median()), 2),
            "p95_abs_bps": round(float(a.quantile(0.95)), 2),
            "p99_abs_bps": round(float(a.quantile(0.99)), 2),
            "share_over_10bps": round(float((a > 10).mean()), 4),
            "mean_bps": round(float(x.mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    if args.reuse:
        b30 = pd.read_parquet(sorted(glob.glob(str(CACHE / "alpaca_30m_*.parquet")))[-1])
    else:
        stamp = f"{pd.Timestamp.now():%Y-%m-%d}"
        b30 = alpaca.fetch_30m(sorted(data.load_universe()), CACHE / f"alpaca_30m_parts_{stamp}")
        path = CACHE / f"alpaca_30m_{stamp}.parquet"
        b30.to_parquet(path, index=False)
        print("->", path)
    first = b30.groupby("ticker")["start"].min()
    print(f"{len(b30):,} bars, {b30['ticker'].nunique()} tickers, "
          f"{b30['start'].min().date()} .. {b30['end'].max().date()}; "
          f"latest first bar: {first.idxmax()} {first.max().date()}")
    b60 = alpaca.to_60m(b30)

    organizer, _ = data.load_organizer_bars()
    vs_org = parity.source_parity(organizer, b30)

    y60 = markets.latest_public("60m")
    j = (b60.set_index(["start", "ticker"])[["open"]]
         .join(y60.set_index(["start", "ticker"])[["open"]], lsuffix="_alp", rsuffix="_yah", how="inner"))
    err = _bps(j["open_alp"], j["open_yah"])
    by_round = err.groupby(j.index.get_level_values("start").strftime("%H:%M")).apply(
        lambda x: round(float(x.abs().median()), 2)).to_dict()
    report = {
        "bars_30m": int(len(b30)), "window": [str(b30["start"].min()), str(b30["end"].max())],
        "hours_60m_complete": int(len(b60)),
        "vs_organizer": {k: vs_org[k] for k in ("window", "ticker_days_compared", "open_0930",
                                                 "session_close", "day_high", "day_low",
                                                 "volume_ratio_median",
                                                 "close_err_worst_tickers_median_abs_bps")},
        "vs_yahoo_60m_opens": {**_stats(err), "median_abs_bps_by_round_start": by_round},
    }
    (data.ROOT / "reports" / "alpaca_parity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
