"""Fetch a window's headlines and 8-K texts, so a real-names replay reads news.

    SEC_USER_AGENT="<name> <email>" .venv/bin/python tools/replay_sources.py --on 2026-01-21

Writes, for the 15-session window starting on that day:
- data/external/news_alpaca/news_<first>.parquet: Alpaca (Benzinga) headlines for the
  30 names, from HEADLINE_HOURS before the first round to the last (icaif/alpaca_news.py);
- data/external/edgar_texts/texts_<lo>_<hi>.parquet: every event 8-K accepted from 7
  days before the first round to the last, with what it says (Exhibit 99.1 for results).

A replay of an anonymised window needs neither: both name the company. Alpaca keys come
from .env; the SEC needs a real contact in SEC_USER_AGENT.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import alpaca_news, calendar, earnings, filings, markets, windows  # noqa: E402
from icaif.agents import observe  # noqa: E402
from icaif.live import TEXT_KEEP_CHARS  # noqa: E402

FILING_DAYS = 7   # what `recent_8k_filings` shows; texts before it are never read


def span(market, start: date, days: int = windows.WINDOW_DAYS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The first and last round deadlines of the window starting on `start`."""
    if start not in market.days:
        raise SystemExit(f"{start} is not a session in the market")
    i = market.days.index(start)
    sessions = market.days[i: i + days]
    if len(sessions) < days:
        raise SystemExit(f"only {len(sessions)} sessions from {start}; the window needs {days}")
    return (calendar.rounds_for(sessions[0])[0]["deadline"],
            calendar.rounds_for(sessions[-1])[-1]["deadline"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="the window's first session (YYYY-MM-DD)")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-texts", action="store_true")
    args = ap.parse_args()

    market = markets.research_market()
    tickers = list(market.tickers)
    first, last = span(market, date.fromisoformat(args.on))
    print(f"window {args.on}: rounds {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} ET")

    if not args.no_news:
        lo = first - pd.Timedelta(hours=observe.HEADLINE_HOURS)
        print(f"headlines {lo:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}, {len(tickers)} names")
        rows = alpaca_news.fetch(tickers, lo, last)
        path = alpaca_news.save(rows, lo, last, tickers)
        per = rows.groupby("ticker").size().reindex(tickers, fill_value=0)
        print(f"{len(rows)} rows ({rows['guid'].nunique()} stories) -> {path}")
        print(f"per name: median {per.median():.0f}, fewest {per.idxmin()} {per.min()}")

    if not args.no_texts:
        lo = first - pd.Timedelta(days=FILING_DAYS)
        events, missing = filings.fetch(tickers, recent_only=True)
        events = events[(events["accepted"] >= lo) & (events["accepted"] <= last)].copy()
        texts, errors = [], []
        with earnings._client(60) as client:
            for r in events.itertuples():
                try:
                    texts.append(filings.filing_text(client, r.cik, r.accession, r.document,
                                                     r.items)[:TEXT_KEEP_CHARS])
                except Exception as err:  # noqa: BLE001 - recorded; the codes still show
                    texts.append(None)
                    errors.append(f"{r.ticker} {r.accession}: {type(err).__name__}: {err}")
        events["text"] = texts
        path = filings.texts_path(lo, last)
        path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(path, index=False)
        print(f"{len(events)} 8-Ks, {events['text'].notna().sum()} with text -> {path}")
        if missing:
            print(f"no CIK: {missing}")
        if errors:
            print(f"{len(errors)} texts unread (codes still show): " + "; ".join(errors[:5]))


if __name__ == "__main__":
    main()
