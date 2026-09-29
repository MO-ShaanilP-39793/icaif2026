"""Scheduled earnings releases from Yahoo: the dates a live run can know in advance.

EDGAR (`earnings.py`) records a release only once it has happened, so live the
"sessions to next earnings" feature is NaN for every name, while training read the
realised next release. This module supplies the missing half: the dates companies
have announced. Two properties of the source shape it:

- **Only the date and the side of the session are real.** Yahoo's upcoming times are
  placeholders (08:00 or 16:00, and 15:00 EST once the date crosses the daylight-
  saving change), while the same names' past releases read 06:00 or 07:00. So each
  row keeps `side` (before the open or after it) and never a clock time a consumer
  might trust. That is all `earnings.reaction_session` uses anyway.
- **History cannot be rebuilt.** Asked later, Yahoo returns the realised dates, not
  what was announced on a given day. Each day's snapshot is the only point-in-time
  record, which is why the tool keeps every day's file and refuses to overwrite one.
"""

import time

import pandas as pd

from icaif import calendar

MARKET_OPEN = pd.Timestamp("09:30").time()
COLUMNS = ["ticker", "date", "side", "raw_time", "fetched_at"]


def side_of_session(times: pd.Series) -> pd.Series:
    """'bmo' before 09:30 ET, 'amc' at or after it, 'unknown' for a midnight stamp.

    Classified by the side of the open, not the hour, because the hour is a
    placeholder that shifts with daylight saving: a 16:00 EDT slot reads 15:00 EST
    in December, still after the open. Midnight is how a date with no time at all
    arrives; read as a clock it would land before the open and move the reaction a
    session early for every after-close reporter with no time posted.
    """
    local = times.dt.tz_convert(calendar.TZ)
    t = local.dt.time
    out = pd.Series("amc", index=times.index, dtype=object)
    out[t.map(lambda x: x < MARKET_OPEN)] = "bmo"
    out[t.map(lambda x: x == pd.Timestamp("00:00").time())] = "unknown"
    return out


def parse(ticker: str, frame: pd.DataFrame | None, now: pd.Timestamp) -> pd.DataFrame:
    """The upcoming releases in one `get_earnings_dates` frame (index = release time).

    Only releases after `now` are kept. A past row here is Yahoo's copy of what EDGAR
    already records with a real timestamp, and one mixed in would be a second,
    placeholder-timed version of the same event.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    times = pd.Series(pd.DatetimeIndex(frame.index).tz_convert(calendar.TZ))
    times = times[times > now].drop_duplicates().sort_values().reset_index(drop=True)
    return pd.DataFrame({
        "ticker": ticker,
        "date": times.dt.tz_localize(None).dt.normalize(),
        "side": side_of_session(times),
        "raw_time": times,
        "fetched_at": now,
    })[COLUMNS]


def fetch(tickers: list[str], limit: int = 8, sleep: float = 0.2,
          now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Upcoming releases for `tickers`, and which names had none or failed.

    A name with no upcoming date is ordinary for a day or two after it reports,
    before the next date is posted, so it is returned, not raised. An empty result
    for every name raises: it reads downstream as "nobody reports soon", the
    direction that removes a risk flag rather than adding one.
    """
    import yfinance as yf

    now = now if now is not None else pd.Timestamp.now(tz=calendar.TZ)
    frames, no_date, failed = [], [], []
    for t in tickers:
        try:
            raw = yf.Ticker(t).get_earnings_dates(limit=limit)
        except Exception as e:  # yfinance raises assorted types for an unknown symbol
            failed.append(f"{t}: {type(e).__name__}: {e}")
            continue
        rows = parse(t, raw, now)
        (frames.append(rows) if len(rows) else no_date.append(t))
        time.sleep(sleep)
    if not frames:
        raise RuntimeError(f"no upcoming earnings date for any of {len(tickers)} names; "
                           f"first failures: {failed[:3]}")
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return out, {"no_date": no_date, "failed": failed}
