"""Alpaca historical bars (free plan, SIP feed): the live :30 grid back to 2016.

The organizer panel sits on a :00 grid, so before Yahoo's 60m window (Oct 2023) every
fill at 10:30 ... 15:30 was a guess (median error 11 bps). Alpaca serves 30-minute bars
from Jan 2016 that start at 09:30 ET. Pairing them rebuilds the exact 60m bars the
competition fills at: 09:30-10:30, ..., 14:30-15:30, then 15:30-16:00.

Alpaca's own hourly bars can't be used: they sit on UTC clock hours, so they are on the
:00 grid in winter and the :30 grid in summer, and they include pre-market prints.

Prices are split-adjusted (`adjustment=split`), the same basis as the organizer panel
and Yahoo. Spin-offs (T, GE) are not in Alpaca's split adjustment, so
`data.adjust_spin_offs` applies here too; the parity report checks that no step is
left.

The free plan refuses SIP data from the last 15 minutes. History is all we fetch.
Credentials come from .env (ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY) and are never
logged.
"""

import time

import pandas as pd

from icaif import calendar, net
from icaif.data import COLUMNS, ROOT, adjust_spin_offs, regular_session

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
START = "2016-01-01"
PAGE = 10000


def _credentials() -> dict[str, str]:
    path = ROOT / ".env"
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY") if not env.get(k)]
    if missing:
        raise RuntimeError(f"missing {missing} in {path}")
    return {"APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"]}


class _RateLimiter:
    """At most `per_minute` requests across all threads (the free plan allows 200)."""

    def __init__(self, per_minute: int = 180):
        import threading
        self.gap = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.gap
        time.sleep(max(0.0, slot - now))


def _fetch_symbol(client, limiter: _RateLimiter, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Every 30m bar for one symbol. Pages hold about three weeks each, extended hours
    included, whatever `limit` says, so ten years is ~180 requests a symbol."""
    rows, token = [], None
    while True:
        params = {"symbols": symbol, "timeframe": "30Min", "start": start, "end": end,
                  "feed": "sip", "adjustment": "split", "limit": PAGE, "sort": "asc"}
        if token:
            params["page_token"] = token
        limiter.wait()
        r = client.get(BARS_URL, params=params)
        r.raise_for_status()
        j = r.json()
        rows.extend((symbol, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"])
                    for b in (j.get("bars") or {}).get(symbol, []))
        token = j.get("next_page_token")
        if not token:
            break
    return pd.DataFrame(rows, columns=["ticker", "t", "open", "high", "low", "close", "volume"])


def _canonical(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["start"] = pd.to_datetime(raw["t"], utc=True).dt.tz_convert(calendar.TZ)
    close = pd.Series([calendar.at(d, calendar.session_close(d)) for d in raw["start"].dt.date],
                      index=raw.index)
    nominal = raw["start"] + pd.Timedelta(minutes=30)
    raw["end"] = nominal.where(nominal < close, close)
    raw["volume"] = raw["volume"].astype("float64")
    raw["source"] = "alpaca_30m"
    bars, _ = regular_session(raw)
    bars, _ = adjust_spin_offs(bars)
    return bars.sort_values(["start", "ticker"]).reset_index(drop=True)[COLUMNS]


def fetch_30m(symbols: list[str], parts_dir, start: str = START, end: str | None = None,
              workers: int = 4) -> pd.DataFrame:
    """Regular-session 30m bars for `symbols`, resumable: each symbol is saved to
    `parts_dir` as it completes and skipped on a rerun, so an interrupted fetch
    (the whole set takes ~30 minutes) loses at most the symbols in flight."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import httpx

    parts = Path(parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    todo = [s for s in symbols if not (parts / f"{s}.parquet").exists()]
    print(f"alpaca: {len(symbols) - len(todo)} symbols already saved, {len(todo)} to fetch", flush=True)
    limiter = _RateLimiter()
    with httpx.Client(headers=_credentials(), verify=net.ssl_context(), timeout=120) as client, \
            ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_fetch_symbol, client, limiter, s, start, end): s for s in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            bars = _canonical(fut.result())
            bars.to_parquet(parts / f"{s}.parquet", index=False)
            print(f"  {n}/{len(todo)} {s}: {len(bars):,} regular-session bars", flush=True)
    frames = [pd.read_parquet(parts / f"{s}.parquet") for s in symbols]
    return pd.concat(frames, ignore_index=True).sort_values(["start", "ticker"]).reset_index(drop=True)


def to_60m(bars_30m: pd.DataFrame) -> pd.DataFrame:
    """Pair 30-minute bars into the live 60m grid (09:30-10:30 ... 15:30-16:00).

    A 60m bar is kept only if every 30-minute bar in it is present. With one half
    missing, its open or close would silently be the other half's, a fill price
    30 minutes off.
    """
    b = bars_30m.copy()
    day_open = b["start"].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
    slot = ((b["start"] - day_open) // pd.Timedelta(hours=1))
    b["bucket"] = day_open + slot * pd.Timedelta(hours=1)
    g = b.sort_values("start").groupby(["ticker", "bucket"])
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"), end=("end", "last"),
                n=("open", "size")).reset_index().rename(columns={"bucket": "start"})
    close = pd.Series([calendar.at(d, calendar.session_close(d)) for d in out["start"].dt.date],
                      index=out.index)
    # Expected halves come from the nominal hour (clipped to the close), not from the
    # bucket's last bar: a bucket missing its second half would otherwise look complete.
    nominal_end = (out["start"] + pd.Timedelta(hours=1)).where(
        out["start"] + pd.Timedelta(hours=1) < close, close)
    expected = ((nominal_end - out["start"]) / pd.Timedelta(minutes=30)).round().astype(int)
    out = out[out["n"] == expected].copy()
    out["end"] = nominal_end[out.index]
    out["source"] = "alpaca_60m"
    return out.sort_values(["start", "ticker"]).reset_index(drop=True)[COLUMNS]
