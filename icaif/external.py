"""Free daily data beyond the organizer panel: Yahoo daily bars, including rates.

Every fetch is saved as a dated snapshot in `data/external/`, and loaders read the
latest one. Two reasons: a rerun of an experiment should see the same data, and Yahoo
revises and drops history silently (a delisted symbol simply stops returning data).

Prices are split-adjusted but **not** dividend-adjusted (`auto_adjust=False`), the
same basis as the organizer panel. A feature built on a different basis in each model
would mean slightly different things. `adj_close` is kept beside it for anything that
wants total return.
"""

import pandas as pd

from icaif.universe import EXTERNAL, latest

START = "1999-01-01"
# Market context for the market node / ctx features. XLRE starts 2015 and XLC 2018;
# a context feature from them is NaN before that, not zero.
# Rates are Cboe's Treasury yield indices on Yahoo (13-week, 5-year, 10-year, quoted in
# percent: 5.18 is 5.18%), not FRED: Python's strict TLS checks fail against FRED
# on this network, and loosening certificate checks to reach it is not worth a series
# Yahoo already carries. Like any close, a value dated d is usable from d+1's decisions.
RATE_SYMBOLS = ["^IRX", "^FVX", "^TNX"]
CONTEXT_SYMBOLS = ["^VIX", "SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
                   "XLB", "XLRE", "XLC"] + RATE_SYMBOLS


def fetch_daily(symbols: list[str], start: str = START) -> tuple[pd.DataFrame, list[str]]:
    """Long frame (date, ticker, OHLC, adj_close, volume) and the symbols Yahoo had nothing for."""
    import yfinance as yf

    raw = yf.download(symbols, start=start, interval="1d", group_by="ticker",
                      auto_adjust=False, actions=False, progress=False, threads=True)
    frames, missing = [], []
    present = set(raw.columns.get_level_values(0)) if len(raw) else set()
    for s in symbols:
        t = raw[s].dropna(subset=["Open", "Close"]) if s in present else pd.DataFrame()
        if t.empty:
            missing.append(s)
            continue
        frames.append(pd.DataFrame({
            "date": pd.DatetimeIndex(t.index).tz_localize(None).normalize(),
            "ticker": s,
            "open": t["Open"].to_numpy(), "high": t["High"].to_numpy(),
            "low": t["Low"].to_numpy(), "close": t["Close"].to_numpy(),
            "adj_close": t["Adj Close"].to_numpy(),
            "volume": t["Volume"].to_numpy(dtype="float64"),
        }))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.sort_values(["date", "ticker"]).reset_index(drop=True), missing


def save(frame: pd.DataFrame, name: str) -> str:
    path = EXTERNAL / f"{name}_{pd.Timestamp.now():%Y-%m-%d}.parquet"
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return str(path)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(latest(f"{name}_*.parquet"))
