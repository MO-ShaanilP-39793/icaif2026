"""Historical headlines from Alpaca's news API (Benzinga), for replays only.

**Why a second source.** The Yahoo archive (`news.py`) starts on 2026-09-29, the day we
began fetching it, and Yahoo cannot say what was on the wire on a past morning. So a
replay of any earlier window would show no headlines at all, and the roles that read
news would be judged on a desk that never sees any. Alpaca serves Benzinga's stories
back years, each stamped and tagged with tickers, on the same free account as our bars.

**Replays only.** The kit wants news that is free and public; Benzinga's content reaches
us through a free Alpaca account but is Benzinga's. Live rounds keep reading Yahoo, so
nothing submitted depends on this feed, and replay headlines are a stand-in for the live
mix, not a copy of it. Both are disclosed.

**Known from its last edit, not its first stamp.** Alpaca returns each story as it reads
now. A story Benzinga edited after publishing shows the edited text, which nobody could
read before the edit, so a row counts from `max(created_at, updated_at)`. Most stories
are never edited; an edited one turns up later than it could have, never earlier.

**Same shape as the Yahoo archive.** Rows are written in `news.COLUMNS` with that known
time as `fetched_at`, in a directory of their own, so `news.known_at` and the desk read
them unchanged. A story tagged with several of our names is one row per name, as a Yahoo
story in several feeds is.

**Coverage is recorded, not inferred.** An empty answer for a quiet name and a span we
never fetched look the same in the rows, and the second would replay as a desk that saw
no news. `COVERAGE` lists the spans fetched; a replay checks its window lies inside them.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from icaif import calendar, net
from icaif.alpaca import _RateLimiter, _credentials
from icaif.data import ROOT
from icaif.news import COLUMNS

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
ARCHIVE = ROOT / "data" / "external" / "news_alpaca"
COVERAGE = "coverage.json"
PAGE = 50   # the API's maximum


def _fetch_symbol(client, limiter: _RateLimiter, symbol: str, start: str, end: str) -> list[dict]:
    """Every story Alpaca tags with `symbol` in [start, end], oldest first."""
    rows, token = [], None
    while True:
        params = {"symbols": symbol, "start": start, "end": end, "limit": PAGE,
                  "sort": "asc", "include_content": "false"}
        if token:
            params["page_token"] = token
        limiter.wait()
        r = client.get(NEWS_URL, params=params)
        r.raise_for_status()
        j = r.json()
        rows.extend(j.get("news") or [])
        token = j.get("next_page_token")
        if not token:
            return rows


def to_rows(stories: list[dict], tickers: list[str]) -> pd.DataFrame:
    """Alpaca stories as archive rows: one per (story, our ticker it is tagged with)."""
    want = set(tickers)
    out = []
    for s in stories:
        if not (s.get("headline") or "").strip():
            continue
        created = pd.Timestamp(s["created_at"]).tz_convert(calendar.TZ)
        updated = pd.Timestamp(s.get("updated_at") or s["created_at"]).tz_convert(calendar.TZ)
        known = max(created, updated)
        for t in sorted(set(s.get("symbols") or []) & want):
            out.append({"ticker": t, "guid": f"alpaca:{s['id']}", "published": created,
                        "title": s["headline"], "summary": s.get("summary") or "",
                        "link": s.get("url") or "", "fetched_at": known})
    if not out:
        return pd.DataFrame(columns=COLUMNS)
    return (pd.DataFrame(out, columns=COLUMNS).drop_duplicates(["ticker", "guid"])
            .sort_values(["fetched_at", "ticker"]).reset_index(drop=True))


def fetch(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Archive rows for every story tagged with one of `tickers` in [start, end]."""
    import httpx

    s, e = (pd.Timestamp(x).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") for x in (start, end))
    limiter = _RateLimiter()
    stories = {}
    with httpx.Client(headers=_credentials(), verify=net.ssl_context(), timeout=60) as client:
        for t in tickers:
            got = _fetch_symbol(client, limiter, t, s, e)
            stories.update({x["id"]: x for x in got})
            print(f"  {t}: {len(got)} stories", flush=True)
    return to_rows(list(stories.values()), tickers)


def save(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tickers: list[str],
         directory: Path = ARCHIVE) -> Path:
    """Write one file named by its earliest row and add its span to the coverage list.

    `news.known_at` skips a file whose name is after the deadline, which is safe only
    if every row in it is stamped at or after that name; the earliest row's stamp is.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    first = rows["fetched_at"].min() if len(rows) else pd.Timestamp(start).tz_convert(calendar.TZ)
    path = directory / f"news_{first.strftime('%Y-%m-%dT%H%M%S')}.parquet"
    if path.exists():
        raise FileExistsError(f"{path.name} exists; an archive file is never overwritten")
    tmp = path.with_suffix(".tmp")
    rows.to_parquet(tmp, index=False)
    tmp.replace(path)
    spans = coverage(directory)
    spans.append({"start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(),
                  "tickers": sorted(tickers), "file": path.name, "rows": int(len(rows))})
    (directory / COVERAGE).write_text(json.dumps(spans, indent=1))
    return path


def coverage(directory: Path = ARCHIVE) -> list[dict]:
    p = Path(directory) / COVERAGE
    return json.loads(p.read_text()) if p.exists() else []


def uncovered(lo: pd.Timestamp, hi: pd.Timestamp, tickers: list[str],
              directory: Path = ARCHIVE) -> Optional[str]:
    """Why [lo, hi] for `tickers` is not inside one fetched span, or None if it is."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    spans = coverage(directory)
    for s in spans:
        if (pd.Timestamp(s["start"]) <= lo and pd.Timestamp(s["end"]) >= hi
                and set(tickers) <= set(s["tickers"])):
            return None
    have = "; ".join(f"{s['start'][:10]} to {s['end'][:10]} ({len(s['tickers'])} names)" for s in spans)
    return (f"no Alpaca news fetched for {lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} on all "
            f"{len(tickers)} names (have: {have or 'nothing'}); run tools/replay_sources.py")
