"""Fetch every event 8-K for the competition names from SEC EDGAR.

    SEC_USER_AGENT="<name> <email>" .venv/bin/python tools/filings_events.py

Writes data/external/edgar_8k_<date>.parquet: (ticker, accepted, items, amended), all
history, point in time by acceptance timestamp (icaif/filings.py). The SEC requires a
real contact in the User-Agent; there is no default.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import calendar, data, filings  # noqa: E402


def main() -> None:
    tickers = sorted(data.load_universe())
    events, missing = filings.fetch(tickers)
    path = data.ROOT / "data" / "external" / f"edgar_8k_{pd.Timestamp.now(tz=calendar.TZ):%Y-%m-%d}.parquet"
    events.to_parquet(path, index=False)
    since = events[events["accepted"] >= pd.Timestamp("2016-01-01", tz=calendar.TZ)]
    counts = since["items"].str.split(",").explode().map(lambda i: filings.ITEMS.get(i, i))
    print(f"{len(events)} event 8-Ks ({len(since)} since 2016) -> {path}")
    print(counts.value_counts().head(12).to_string())
    if missing:
        print(f"no CIK: {missing}")


if __name__ == "__main__":
    main()
