"""Canonical bar frames, and the loader for the organizers' historical panel.

Every source (organizer parquet, public bars) is normalised to one long frame:

    ticker, start, end, open, high, low, close, volume, source

`start`/`end` are tz-aware New York timestamps. `end` is explicit because the two
sources sit on different grids -- organizer bars start 09:30, 10:00, 11:00 ... 15:00;
public hourly bars start 09:30, 10:30 ... 15:30 -- and "what was known at a deadline" is
a question about when a bar *ended*. Deriving it from `start` plus a fixed hour gets the
first organizer bar (30 minutes) and the last public bar (30 minutes) wrong, and each
error hands a decision half an hour of the future.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from icaif import calendar

ROOT = Path(__file__).resolve().parents[1]
ORGANIZER_PARQUET = ROOT / "data" / "hourly_market_data_2021_2026.parquet"
UNIVERSE_JSON = ROOT / "starter-kit" / "universe.json"

COLUMNS = ["ticker", "start", "end", "open", "high", "low", "close", "volume", "source"]

# Spin-offs the organizer panel is NOT adjusted for (it is split-adjusted only). On each
# ex-date the raw price drops ~20-25% because holders received shares of the new company,
# so the panel shows a crash nobody suffered: three of the worst "losses" in 2021-25 are
# fake, and a path label or a drawdown feature learns from them. Prices before the
# ex-date are multiplied by `factor`, the permanent step in organizer/Yahoo daily close
# (median of 5 days either side, Yahoo 1d fetched 2026-09-25). Yahoo's own series is
# already adjusted for all three; a scan of the full 2021-25 ratio found no other
# permanent step. Volume is left alone: a spin-off changes value per share, not shares.
CORPORATE_ACTIONS = (
    {"ticker": "T", "ex_date": "2022-04-11", "factor": 0.755287,
     "event": "Warner Bros. Discovery spin-off"},
    {"ticker": "GE", "ex_date": "2023-01-04", "factor": 0.780640,
     "event": "GE HealthCare spin-off"},
    {"ticker": "GE", "ex_date": "2024-04-02", "factor": 0.798268,
     "event": "GE Vernova spin-off"},
)
PRICE_COLUMNS = ["open", "high", "low", "close"]


def load_universe() -> dict[str, str]:
    """ticker -> sector group, from the organizers' own file rather than a copy."""
    sectors = json.loads(UNIVERSE_JSON.read_text())["sectors"]
    return {t: sector for sector, tickers in sectors.items() for t in tickers}


@dataclass
class DataIssues:
    """What loading dropped or found missing, with counts.

    Recorded rather than raised: a missing bar is a fact about the source, and the panel
    is still worth using. But a loader that drops rows quietly reads as a clean panel.
    """

    dropped_extended_hours: int = 0
    spin_off_adjusted_bars: int = 0
    short_sessions: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        n_short = sum(len(v) for v in self.short_sessions.values())
        return (f"dropped {self.dropped_extended_hours} extended-hours bars; "
                f"spin-off-adjusted {self.spin_off_adjusted_bars} bars; "
                f"{n_short} ticker-days short of a full session across "
                f"{len(self.short_sessions)} days")


def regular_session(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only bars that end within the day's regular session."""
    day = bars["start"].dt.date
    close = pd.Series(
        [calendar.at(d, calendar.session_close(d)) for d in day], index=bars.index
    )
    # `start < close` as well as `end <= close`: a half-day's 13:00 bar is clipped to
    # end at 13:00 by `_organizer_end`, so the end test alone would keep it.
    keep = ((bars["start"].dt.time >= calendar.SESSION_OPEN)
            & (bars["start"] < close) & (bars["end"] <= close))
    return bars[keep], int((~keep).sum())


def _organizer_end(start: pd.Series) -> pd.Series:
    """Organizer bars run 09:30-10:00, then clock hours, cut at the session close."""
    nxt = start.dt.floor("h") + pd.Timedelta(hours=1)
    close = pd.Series(
        [calendar.at(d, calendar.session_close(d)) for d in start.dt.date], index=start.index
    )
    return nxt.where(nxt < close, close)


def load_organizer_bars(path: Path = ORGANIZER_PARQUET) -> tuple[pd.DataFrame, DataIssues]:
    raw = pd.read_parquet(path)
    universe = load_universe()
    unknown = set(raw["ticker"]) - set(universe)
    if unknown or set(universe) - set(raw["ticker"]):
        raise ValueError(f"organizer tickers differ from universe.json: {sorted(unknown)}")
    if raw.duplicated(["timestamp_et", "ticker"]).any():
        raise ValueError("organizer panel has duplicate (timestamp, ticker) bars")
    # adj_close is dropped, not ignored by accident: it equals close on every row, and
    # the panel is already split-adjusted (no jump at NVDA 2024-06-10). If a refresh ever
    # makes them differ, the choice between them needs making deliberately.
    if not (raw["adj_close"] == raw["close"]).all():
        raise ValueError("adj_close differs from close; decide which one the panel means")

    start = raw["timestamp_et"].dt.tz_localize(calendar.TZ)
    bars = pd.DataFrame({
        "ticker": raw["ticker"],
        "start": start,
        "end": _organizer_end(start),
        "open": raw["open"],
        "high": raw["high"],
        "low": raw["low"],
        "close": raw["close"],
        "volume": raw["volume"].astype("float64"),
        "source": "organizer",
    })
    issues = DataIssues()
    bars, issues.dropped_extended_hours = regular_session(bars)
    bars, issues.spin_off_adjusted_bars = adjust_spin_offs(bars)
    issues.short_sessions = short_sessions(bars, expected_per_session=_organizer_bar_count)
    return bars.sort_values(["start", "ticker"]).reset_index(drop=True)[COLUMNS], issues


def adjust_spin_offs(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    bars = bars.copy()
    touched = 0
    for action in CORPORATE_ACTIONS:
        ex = calendar.at(pd.Timestamp(action["ex_date"]).date(), calendar.SESSION_OPEN)
        rows = (bars["ticker"] == action["ticker"]) & (bars["start"] < ex)
        bars.loc[rows, PRICE_COLUMNS] *= action["factor"]
        touched += int(rows.sum())
    return bars, touched


def _organizer_bar_count(day) -> int:
    return 4 if day in calendar.EARLY_CLOSES else 7


def short_sessions(bars: pd.DataFrame, expected_per_session) -> dict[str, list[str]]:
    counts = bars.groupby([bars["start"].dt.date, "ticker"]).size()
    short = {}
    for (day, ticker), n in counts.items():
        if n < expected_per_session(day):
            short.setdefault(day.isoformat(), []).append(ticker)
    # A ticker absent for a whole day has no row to count, so check that too.
    tickers = set(bars["ticker"])
    for day, present in bars.groupby(bars["start"].dt.date)["ticker"]:
        missing = tickers - set(present)
        if missing:
            short.setdefault(day.isoformat(), []).extend(sorted(missing))
    return short
