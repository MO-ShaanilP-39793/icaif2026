"""Fetch and snapshot the free data the ensemble plan needs, then report its coverage.

    .venv/bin/python tools/enrich_data.py [--refresh-membership] [--reuse-daily] [--skip-earnings]

Writes dated snapshots to data/external/:
  - sp500_ticker_start_end_<date>.csv   point-in-time membership (fja05680/sp500, MIT)
  - yahoo_daily_universe_<date>.parquet daily bars for every symbol ever a member
  - yahoo_daily_context_<date>.parquet  VIX, SPY, sector ETFs, Treasury yield indices
  - earnings_<date>.parquet             earnings 8-K acceptance times (needs SEC_USER_AGENT)

and reports/universe_coverage.csv plus reports/enrich_missing.json: what Yahoo and
EDGAR had nothing for. The coverage table is the survivorship bias, measured.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess  # noqa: E402

import pandas as pd  # noqa: E402

from icaif import data, earnings, external, universe  # noqa: E402

MEMBERSHIP_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"


def refresh_membership() -> None:
    # curl, not httpx: Python's TLS stack fails against GitHub on this network.
    path = universe.EXTERNAL / f"sp500_ticker_start_end_{pd.Timestamp.now():%Y-%m-%d}.csv"
    subprocess.run(["curl", "-sfL", "-o", str(path), MEMBERSHIP_URL], check=True)
    print(f"membership -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-membership", action="store_true")
    ap.add_argument("--reuse-daily", action="store_true",
                    help="use the latest universe snapshot instead of refetching")
    ap.add_argument("--skip-earnings", action="store_true")
    args = ap.parse_args()

    if args.refresh_membership:
        refresh_membership()
    membership = universe.load_membership()
    symbols = sorted(set(membership["symbol"]) | set(data.load_universe()))
    if args.reuse_daily:
        daily = external.load("yahoo_daily_universe")
        missing_daily = sorted(set(symbols) - set(daily["ticker"]))
    else:
        print(f"fetching daily bars for {len(symbols)} symbols since {external.START} ...")
        daily, missing_daily = external.fetch_daily(symbols)
        print(" ->", external.save(daily, "yahoo_daily_universe"))

    ctx, missing_ctx = external.fetch_daily(external.CONTEXT_SYMBOLS)
    print(" ->", external.save(ctx, "yahoo_daily_context"))

    cov = universe.coverage(daily, membership)
    out = data.ROOT / "reports"
    cov.to_csv(out / "universe_coverage.csv")
    print("\nS&P 500 members priced by Yahoo, average per day:")
    print(cov.to_string())

    uni = universe.build(daily, membership)
    ever = sorted(uni.columns[uni.any()])
    size = uni.sum(axis=1)
    print(f"\ntraining universe: {len(ever)} symbols ever in; per day median {size.median():.0f}, "
          f"min {size.min():.0f} (from {size[size > 0].index.min().date()})")

    missing = {"daily": missing_daily, "context": missing_ctx}
    if args.skip_earnings or not os.environ.get("SEC_USER_AGENT"):
        print("\nearnings skipped: set SEC_USER_AGENT='<name> <email>' to fetch from EDGAR")
    else:
        raw, missing["earnings"] = earnings.fetch(ever)
        print(" ->", external.save(raw, "earnings"))
        ev = earnings.quarterly(raw)
        per = ev.groupby("ticker").size()
        print(f"earnings: {len(ev)} releases for {per.size} of {len(ever)} symbols, "
              f"median {per.median():.0f} each, from {ev['accepted'].min().date()}")
        # Releases cluster before the open and just after the close. Mostly mid-session
        # times mean the timestamps are being read in the wrong zone (see parse_filings).
        hour = ev["accepted"].dt.hour + ev["accepted"].dt.minute / 60
        buckets = pd.cut(hour, [0, 9.5, 16, 24], right=False,
                         labels=["before open", "in session", "after close"])
        share = buckets.value_counts(normalize=True).round(3)
        print("release timing:", share.to_dict())
        if share.get("in session", 0) > 0.3:
            print("WARNING: >30% of releases fall in session; check EDGAR's time zone")
        # About one release a quarter from a name's first price (or 2004) is expected.
        # A name far below that has history under a former CIK (earnings.FORMER_CIKS).
        first = daily.groupby("ticker")["date"].min().clip(lower=pd.Timestamp("2004-09-01"))
        expected = (daily["date"].max() - first.reindex(per.index)).dt.days / 91.3
        ratio = (per / expected).round(2)
        thin = ratio[ratio < 0.8].sort_values()
        print(f"{len(thin)} symbols with under 80% of expected releases:",
              thin.head(20).to_dict())
    (out / "enrich_missing.json").write_text(json.dumps(
        {k: {"count": len(v), "symbols": v} for k, v in missing.items()}, indent=1))
    print({k: len(v) for k, v in missing.items()}, "symbols with no data (reports/enrich_missing.json)")


if __name__ == "__main__":
    main()
