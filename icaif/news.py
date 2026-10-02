"""Company headlines from Yahoo Finance's public RSS feeds, archived point in time.

**Why an archive, not a fetch.** Yahoo returns only recent headlines. Asked later, it
cannot say what was on the wire on a given morning, so the only record of what an
agent could have read before a deadline is a snapshot taken before it. Each run writes
one file named by its start time and never overwrites one, and every row keeps
`fetched_at`, the moment its feed came back.

**When we had it, not when it says it was published** (`known_at`). A headline counts
from `first_seen`, the earliest fetch that carried it. The pubDate is the publisher's
claim: Yahoo's runs after the headline's first appearance in our archive for 108 of the
first 2,070 (by up to 2.2 hours), so gated on it those would vanish for hours after we
had them. And a story fetched for the first time today is news to us today, whatever
date it carries; dated by its pubDate it would read as something we had known for days.
The pubDate is still shown: it is part of what we fetched.

**Why RSS.** yfinance's `Ticker.news` hits an endpoint that answers HTTP 500 on this
network (checked 2026-09-29) and turns that into an empty list: the archive would
have filled with "no news" for every name, a quiet answer in the direction that
removes a risk flag. The RSS feed is the same headlines, public and unauthenticated.
An empty result for every name raises instead.

**What a feed is.** A ticker's feed is Yahoo's related news, not only stories about
the company (AAPL's carries Nvidia stories). `names_company` flags the headlines that
name the company, so a short list puts those first rather than filling up with
neighbours'; relevance is still the reader's call, and the flag says which is which.
"""

import functools
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from icaif import calendar, data

RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
ARCHIVE = data.ROOT / "data" / "external" / "news"
COLUMNS = ["ticker", "guid", "published", "title", "summary", "link", "fetched_at"]
SUMMARY_CHARS = 400

# What each company is called in a headline. Case-sensitive, on word boundaries: "Meta"
# the company, not "metadata". A ticker of three letters or more also counts on its
# own; a shorter one only as "(GE)", "$GE" or "NYSE:GE", or every "T" and "V" would match.
ALIASES = {
    "AAPL": ("Apple",), "AMZN": ("Amazon",), "BA": ("Boeing",),
    "BAC": ("Bank of America", "BofA"), "CAT": ("Caterpillar",), "CRM": ("Salesforce",),
    "CVX": ("Chevron",), "DIS": ("Disney",), "GE": ("GE Aerospace", "General Electric"),
    "GOOGL": ("Alphabet", "Google"), "GS": ("Goldman Sachs", "Goldman"), "INTC": ("Intel",),
    "JNJ": ("Johnson & Johnson", "J&J"), "JPM": ("JPMorgan", "JP Morgan"),
    "KO": ("Coca-Cola", "Coke"), "LLY": ("Eli Lilly", "Lilly"),
    "META": ("Meta", "Facebook", "Instagram", "WhatsApp"), "MSFT": ("Microsoft",),
    "NEE": ("NextEra",), "NKE": ("Nike",), "NVDA": ("Nvidia", "NVIDIA"), "PFE": ("Pfizer",),
    "PYPL": ("PayPal",), "T": ("AT&T",), "TMO": ("Thermo Fisher",), "TSLA": ("Tesla",),
    "UNH": ("UnitedHealth", "Optum"), "V": ("Visa",), "WMT": ("Walmart",),
    "XOM": ("Exxon", "ExxonMobil"),
}


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


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=calendar.TZ)


def fetch(tickers: list[str], sleep: float = 0.3, now: Optional[pd.Timestamp] = None,
          get=None, clock: Optional[Callable[[], pd.Timestamp]] = None,
          budget_s: Optional[float] = None) -> tuple[pd.DataFrame, dict]:
    """Every name's current feed, and which names came back empty or failed.

    Each name's rows are stamped when its own feed came back, not when the run began: a
    run of 30 feeds takes about 20 s, and a row stamped at the start would count as known
    up to 20 s before we had it. `now`, the run's start, names the file (`save`), so a
    file's name is never later than any row in it. Past `budget_s` the remaining names
    are listed as failed rather than fetched: a live round has a deadline.
    """
    clock = clock or _now
    t0 = time.monotonic()
    if get is None:
        import httpx

        from icaif import net

        client = httpx.Client(verify=net.ssl_context(), timeout=20, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0"})
        get = lambda t: client.get(RSS_URL, params={"s": t, "region": "US",  # noqa: E731
                                                    "lang": "en-US"}).raise_for_status().text
    frames, empty, failed = [], [], []
    for t in tickers:
        if budget_s is not None and time.monotonic() - t0 > budget_s:
            failed.append(f"{t}: not fetched, the {budget_s:.0f}s budget was spent")
            continue
        try:
            xml = get(t)
            rows = parse_rss(xml, t, clock())
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


def snapshot(tickers: list[str], directory: Path = ARCHIVE, **kw) -> tuple[Path, pd.DataFrame, dict]:
    """Fetch every feed now and archive it: what the scheduled job and a live round run."""
    now = _now().floor("s")
    frame, issues = fetch(tickers, now=now, **kw)
    return save(frame, now, directory), frame, issues


@functools.lru_cache(maxsize=4096)
def _read(path: str, mtime_ns: int) -> pd.DataFrame:
    # Keyed on the modification time too, so a file rewritten under the same name (a
    # restored backup) is read again rather than served from memory.
    return pd.read_parquet(path)


def known_at(deadline: pd.Timestamp, directory: Path = ARCHIVE) -> pd.DataFrame:
    """Every headline the archive had by `deadline`: one row per (ticker, guid), its
    `first_seen` (the earliest fetch that carried it), and the text of the latest fetch
    by the deadline (Yahoo edits titles). Nothing fetched after the deadline is read."""
    deadline = pd.Timestamp(deadline)
    frames = []
    for f in sorted(Path(directory).glob("news_*.parquet")):
        stamp = pd.Timestamp(f.stem.removeprefix("news_")).tz_localize(calendar.TZ)
        if stamp <= deadline:   # a file's rows are all stamped at or after its name
            frames.append(_read(str(f), f.stat().st_mtime_ns))
    if not frames:
        return pd.DataFrame(columns=[*COLUMNS, "first_seen"])
    n = pd.concat(frames, ignore_index=True)
    n = n[n["fetched_at"] <= deadline].sort_values("fetched_at", kind="stable")
    if n.empty:
        return pd.DataFrame(columns=[*COLUMNS, "first_seen"])
    first = n.groupby(["ticker", "guid"])["fetched_at"].transform("min")
    n = n.assign(first_seen=first).drop_duplicates(["ticker", "guid"], keep="last")
    return n.reset_index(drop=True)


@functools.lru_cache(maxsize=64)
def _pattern(ticker: str) -> re.Pattern:
    names = [rf"\b{re.escape(a)}(?!\w)" for a in ALIASES.get(ticker, ())]
    tick = (rf"\b{re.escape(ticker)}\b" if len(ticker) >= 3
            else rf"(?:\(|\$|:){re.escape(ticker)}\b")
    return re.compile("|".join([*names, tick]))


def names_company(ticker: str, *texts) -> bool:
    return any(_pattern(ticker).search(t or "") for t in texts)


def recent(known: pd.DataFrame, deadline: pd.Timestamp, tickers, per_name: int = 4,
           lookback_hours: float = 48) -> pd.DataFrame:
    """Up to `per_name` headlines per name in `tickers`, first seen within `lookback_hours`
    of the deadline: those naming the company first, newest first within each group.

    A cap filled newest-first would, for most names, be filled by the feed's stories
    about other companies, and the one about this name would be the one left out.
    """
    deadline = pd.Timestamp(deadline)
    k = known[known["ticker"].isin(list(tickers))
              & (known["first_seen"] >= deadline - pd.Timedelta(hours=lookback_hours))]
    if k.empty:
        return k.assign(names_it=pd.Series(dtype=bool))
    k = k.assign(names_it=[names_company(t, a, b) for t, a, b in
                           zip(k["ticker"], k["title"], k["summary"])])
    k = k.sort_values(["ticker", "names_it", "first_seen", "guid"],
                      ascending=[True, False, False, True], kind="stable")
    return k.groupby("ticker", sort=False).head(per_name).reset_index(drop=True)
