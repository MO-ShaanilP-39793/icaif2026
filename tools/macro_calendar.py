"""Snapshot the Fed's scheduled FOMC decisions.

    .venv/bin/python tools/macro_calendar.py

Writes data/external/fomc_decisions_<ET date>.json with the years the page covers
(the current one and the five before). Days outside them read "unknown" to the agents.
Refresh after the Fed publishes the next year's schedule (usually in summer).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icaif import macro  # noqa: E402


def main() -> None:
    decisions, years = macro.fetch_fomc()
    path = macro.FomcCalendar.save(decisions, years)
    per_year = decisions.groupby(decisions["date"].dt.year).size().to_dict()
    print(f"{len(decisions)} decisions, years {years[0]}-{years[-1]} {per_year} -> {path}")
    upcoming = decisions[decisions["date"] >= __import__("pandas").Timestamp.now().normalize()].head(3)
    print("next:", ", ".join(f"{d:%Y-%m-%d}{' (SEP)' if p else ''}"
                             for d, p in zip(upcoming["date"], upcoming["projections"])))


if __name__ == "__main__":
    main()
