"""Snapshot every competition name's Yahoo headlines, to build a point-in-time archive.

    .venv/bin/python tools/news_archive.py

Writes data/external/news/news_<ET timestamp>.parquet and never overwrites one. Run
it before every round's deadline (09:05, 10:20, ... 15:20 ET) on trading days, and at
least once a day otherwise: a headline is only replayable from the first snapshot
that carried it, and Yahoo can't be asked later what it showed then (icaif/news.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import calendar, data, news  # noqa: E402


def main() -> None:
    now = pd.Timestamp.now(tz=calendar.TZ).floor("s")
    tickers = sorted(data.load_universe())
    frame, issues = news.fetch(tickers, now=now)
    path = news.save(frame, now)
    per = frame.groupby("ticker").size()
    fresh = frame[frame["published"] >= now - pd.Timedelta(hours=24)].groupby("ticker").size()
    print(f"{len(frame)} headlines for {per.size} names -> {path}")
    print(f"published in the last 24h: {int(fresh.sum())} across {fresh.size} names")
    if issues["empty"]:
        print(f"empty feeds: {issues['empty']}")
    if issues["failed"]:
        print(f"FAILED: {issues['failed']}")


if __name__ == "__main__":
    main()
