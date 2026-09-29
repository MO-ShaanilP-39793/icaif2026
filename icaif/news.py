"""Company headlines from Yahoo Finance's public RSS feeds, archived point in time.

**Why an archive, not a fetch.** Yahoo returns only recent headlines. Asked later, it
cannot say what was on the wire on a given morning, so the only record of what an
agent could have read before a deadline is a snapshot taken before it. Each run writes
one file named by its fetch time and never overwrites one, and every row keeps
`fetched_at`. `as_of` reads only snapshots fetched by the deadline, never a headline
published before it but fetched after, which we did not in fact have.

**Why RSS.** yfinance's `Ticker.news` hits an endpoint that answers HTTP 500 on this
network (checked 2026-09-29) and turns that into an empty list: the archive would
have filled with "no news" for every name, a quiet answer in the direction that
removes a risk flag. The RSS feed is the same headlines, public and unauthenticated.
An empty result for every name raises instead.

**What a feed is.** A ticker's feed is Yahoo's related news, not only stories about
the company (AAPL's carries Nvidia stories), so relevance is the reader's job; the
archive keeps what was there.
"""

import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from icaif import calendar, data

RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
ARCHIVE = data.ROOT / "data" / "external" / "news"
COLUMNS = ["ticker", "guid", "published", "title", "summary", "link", "fetched_at"]
SUMMARY_CHARS = 400


def parse_rss(xml: str, ticker: str, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """One feed's items. An item with no parseable publish time is dropped: placed at
    an arbitrary time, it could read as known before a deadline it came after."""
    rows = []
    for item in ET.fromstring(xml).iter("item"):
        get = lambda tag: (item.findtext(tag) or "").strip()  # noqa: E731
        try:
            published = pd.Timestamp(parsedate_to_datetime(get("pubDate"))).tz_convert(calendar.TZ)
        except (TypeError, ValueError):
            continue
        rows.append({"ticker": ticker, "guid": get("guid") or get("link"),
                     "published": published, "title": get("title"),
                     "summary": get("description")[:SUMMARY_CHARS], "link": get("link"),
                     "fetched_at": fetched_at})
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch(tickers: list[str], sleep: float = 0.3, now: Optional[pd.Timestamp] = None,
          get=None) -> tuple[pd.DataFrame, dict]:
    """Every name's current feed, and which names came back empty or failed."""
    now = now if now is not None else pd.Timestamp.now(tz=calendar.TZ)
    if get is None:
        import httpx

        from icaif import net

        client = httpx.Client(verify=net.ssl_context(), timeout=20, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0"})
        get = lambda t: client.get(RSS_URL, params={"s": t, "region": "US",  # noqa: E731
                                                    "lang": "en-US"}).raise_for_status().text
    frames, empty, failed = [], [], []
    for t in tickers:
        try:
            rows = parse_rss(get(t), t, now)
        except Exception as e:  # noqa: BLE001 - one bad feed must not lose the rest
            failed.append(f"{t}: {type(e).__name__}: {e}")
            continue
        (frames.append(rows) if len(rows) else empty.append(t))
        time.sleep(sleep)
    if not frames:
        raise RuntimeError(f"no headlines for any of {len(tickers)} names; "
                           f"first failures: {failed[:3]}")
    return pd.concat(frames, ignore_index=True), {"empty": empty, "failed": failed}


def save(frame: pd.DataFrame, now: pd.Timestamp, directory: Path = ARCHIVE) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"news_{now:%Y-%m-%dT%H%M%S}.parquet"
    if path.exists():
        raise FileExistsError(f"{path} exists; a snapshot is never overwritten")
    frame.to_parquet(path, index=False)
    return path


def as_of(deadline: pd.Timestamp, per_name: int = 5, lookback_days: int = 3,
          directory: Path = ARCHIVE) -> dict[str, list[dict]]:
    """{ticker: newest headlines} from snapshots fetched by `deadline`, published by it
    and within `lookback_days`; each headline once, however many snapshots carried it."""
    deadline = pd.Timestamp(deadline)
    frames = []
    for f in sorted(directory.glob("news_*.parquet")):
        stamp = pd.Timestamp(f.stem.removeprefix("news_")).tz_localize(calendar.TZ)
        if stamp <= deadline:
            frames.append(pd.read_parquet(f))
    if not frames:
        return {}
    n = pd.concat(frames, ignore_index=True)
    n = n[(n["fetched_at"] <= deadline) & (n["published"] <= deadline)
          & (n["published"] >= deadline - pd.Timedelta(days=lookback_days))]
    n = n.sort_values("published", ascending=False).drop_duplicates(["ticker", "guid"])
    return {t: [{"published": str(r.published), "title": r.title, "summary": r.summary}
                for r in g.head(per_name).itertuples()]
            for t, g in n.groupby("ticker")}
