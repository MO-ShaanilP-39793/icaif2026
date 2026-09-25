"""Public intraday bars (Yahoo Finance via yfinance), normalised to the canonical frame.

Yahoo is the organizers' own suggested public source, and it is also the only live feed
we have: the competition supplies no prices. What it offers, measured 2026-09-25:

| interval | grid                      | history            |
| 60m      | 09:30, 10:30 ... 15:30    | ~730 days (Oct 2023) |
| 30m, 5m  | both grids reconstructible | ~60 days (Jul 2026)  |

The 60m grid is the live execution grid, so a 60m bar's open is a candidate for the
exact price a round fills at. The organizer panel is on a different grid; see data.py.
"""

from pathlib import Path

import pandas as pd

from icaif import calendar
from icaif.data import COLUMNS, ROOT, regular_session

CACHE = ROOT / "data" / "public"

_MINUTES = {"60m": 60, "30m": 30, "15m": 15, "5m": 5}


def fetch(tickers: list[str], interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    fetched_at = pd.Timestamp.now(tz=calendar.TZ)
    raw = yf.download(
        tickers, interval=interval, period=period, group_by="ticker",
        auto_adjust=False, prepost=False, progress=False, threads=True,
    )
    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            raise ValueError(f"yfinance returned no {interval} bars for {ticker}")
        t = raw[ticker].dropna(subset=["Open", "Close"])
        start = t.index.tz_convert(calendar.TZ)
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "start": start,
            "open": t["Open"].to_numpy(),
            "high": t["High"].to_numpy(),
            "low": t["Low"].to_numpy(),
            "close": t["Close"].to_numpy(),
            "volume": t["Volume"].to_numpy(dtype="float64"),
        }))
    bars = pd.concat(frames, ignore_index=True)
    bars["end"] = _end(bars["start"], _MINUTES[interval])
    bars["source"] = f"yahoo_{interval}"

    bars, _ = regular_session(completed(bars, fetched_at))
    return bars.sort_values(["start", "ticker"]).reset_index(drop=True)[COLUMNS]


def completed(bars: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Only bars that had ended by `as_of`.

    Yahoo serves the bar in progress as if it were complete, with a close that is just
    the latest trade. Live, that is a feature computed on a bar that has not ended, and
    its volume is a fraction of a real bar's. It is look-ahead in a backtest and noise
    live, and in both cases the numbers look ordinary.
    """
    return bars[bars["end"] <= as_of]


def _end(start: pd.Series, minutes: int) -> pd.Series:
    """Bar end, clipped to the session close (Yahoo's last 60m bar is 15:30-16:00)."""
    nominal = start + pd.Timedelta(minutes=minutes)
    close = pd.Series(
        [calendar.at(d, calendar.session_close(d)) for d in start.dt.date], index=start.index
    )
    return nominal.where(nominal < close, close)


def cached(tickers: list[str], interval: str, period: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch once and keep a dated copy; history beyond Yahoo's window cannot be refetched.

    60m history ends ~730 days back and 5m ~60 days back, and both windows roll forward
    daily. A snapshot not kept today is lost for good, so every fetch is stored under its
    date rather than overwriting the last one.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz=calendar.TZ).date().isoformat()
    path = CACHE / f"yahoo_{interval}_{today}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    bars = fetch(tickers, interval, period)
    bars.to_parquet(path, index=False)
    return bars
