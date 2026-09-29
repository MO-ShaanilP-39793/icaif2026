"""Snapshot Yahoo's scheduled earnings dates for the daily universe, once a day.

    .venv/bin/python tools/earnings_calendar.py [--force]

Writes data/external/earnings_calendar_<ET date>.parquet and prints what the
competition names have coming. Run it every day before 09:10 ET: a day not
captured is lost, because Yahoo later returns what happened, not what was
announced (see icaif/earnings_calendar.py).

Names: every symbol in the training universe on any of the last 63 sessions of the
latest daily snapshot, ranked to TOP_N + BUFFER, plus the 30. The universe re-ranks
daily, so a name near rank 100 today may be in it on a decision day; missing from
the snapshot, its "sessions to next" would read NaN, a quiet "no release coming".

The file is named by the Eastern date, the calendar decisions run on. Run at 06:00
IST it is still the previous ET day, which is correct: that is when it was known.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import calendar, data, earnings_calendar, external, universe  # noqa: E402

BUFFER = 20
RECENT_SESSIONS = 63
# Above this share of names with nothing (no date and no error is normal just after
# a release), the source is probably failing rather than quiet.
MAX_EMPTY_SHARE = 0.10


def snapshot_names() -> list[str]:
    daily = external.load("yahoo_daily_universe")
    membership = universe.load_membership()
    uni = universe.build(daily, membership, top_n=universe.TOP_N + BUFFER)
    recent = uni.iloc[-RECENT_SESSIONS:]
    return sorted(set(recent.columns[recent.any()]) | set(data.load_universe()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite today's snapshot (it then records a later view of the day)")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz=calendar.TZ)
    path = universe.EXTERNAL / f"earnings_calendar_{now:%Y-%m-%d}.parquet"
    if path.exists() and not args.force:
        sys.exit(f"{path} exists; a second fetch would replace the earlier view of the day "
                 "(--force to do it anyway)")

    names = snapshot_names()
    print(f"fetching scheduled earnings for {len(names)} names ...")
    rows, issues = earnings_calendar.fetch(names, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    print(" ->", path)

    covered = rows["ticker"].nunique()
    print(f"{covered} of {len(names)} names have an upcoming date; "
          f"sides {rows['side'].value_counts().to_dict()}")
    for kind, bad in issues.items():
        if bad:
            print(f"{kind} ({len(bad)}):", ", ".join(bad[:15]) + (" ..." if len(bad) > 15 else ""))

    comp = rows[rows["ticker"].isin(data.load_universe())].groupby("ticker").head(1)
    print("\nthe 30, next release:")
    print(comp[["ticker", "date", "side"]].to_string(index=False))

    empty = (len(names) - covered) / len(names)
    if empty > MAX_EMPTY_SHARE:
        sys.exit(f"{empty:.0%} of names have no upcoming date (limit {MAX_EMPTY_SHARE:.0%}); "
                 "the snapshot is written but looks like a failing source")


if __name__ == "__main__":
    main()
