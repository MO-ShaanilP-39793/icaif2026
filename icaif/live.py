"""A live round's inputs from public data: the desk's market, and the daily model's scores.

Nothing here talks to Codabench (`runner` does, through the kit). Two products:

- `market`: a `sim.Market` over Yahoo bars, so the rule desk and the shadow agent run
  the backtested `Desk` unchanged. Each completed session is one bar ending at its
  close; today's 60m bars, ended by the fetch, sit after them.
- `daily_scores`: the frozen daily model's scores for the next session, built from the
  training universe's daily bars exactly as training built them. Only the shadow agent
  reads them, so a failure here never touches the submitted book.

`envelope` writes the kit's decision.json. Without team credentials it carries the
kit's placeholders and a `dryrun-` round id, which the kit's client refuses locally
(placeholder check, unknown round) before any upload could happen.

The traps a live run has and a backtest does not:

- **The decision row does not exist yet.** Features for day d are the close of d-1
  shifted one session, and `daily_features.build` shifts along the dates it is given.
  A frame whose last date is d-1 has nowhere to put d, so d's row would be silently
  missing and the latest row would be d-1's (built from d-2's close). A placeholder row
  dated d is appended first: last price, zero volume, which `panels` reads as missing.
- **Yahoo serves today's bar before it is complete.** Called during the session, the
  daily download ends in a partial bar. Anything dated after the last completed session
  is dropped before anything is computed.
- **A stale feed looks like a quiet market.** Yahoo returning Friday's bars on Tuesday
  gives ordinary-looking features from old prices, and the model scores them happily.
  The latest bar must be the last completed session on the NYSE calendar, or the run
  raises (a missed round holds, which is the cheap failure).
- **Earnings timing is thinner live than in training.** Training's
  `e_sessions_to_next` read the actual next release (within 10 sessions); EDGAR only
  records releases after they happen, so live it is NaN for every name until an
  earnings calendar is wired in. Measured on the 2026 test year with the frozen model,
  that alone takes universe IC from 0.053 to 0.043 and IC among the 30 from 0.021 to
  0.012. Without SEC_USER_AGENT the events are the last snapshot, and a release since
  then is simply missing.
- **Context off the equity grid.** A context series printing on an equity holiday
  (^VIX on Labor Day 2026) breaks `context`'s 20-row SPY windows; see `fetch_inputs`.
- **A session still trading has no close.** The market's panel holds today's bars so
  far; `quant_strategies.daily_closes` must not read the latest as today's close.

Replayed on the 2026-09-27 snapshots for 2026-08-20, the live feature frame equals
`daily_features.build`'s row for row except `e_sessions_to_next`, and scoring the
training frame reproduces output/preds/daily_d5_pct.parquet exactly.
"""

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday, sunday_to_monday)

from icaif import (alpaca, calendar, daily_features, data, earnings, external, kit, public_bars,
                   quant_strategies as qs, sim, universe)

MODEL = data.ROOT / "output" / "ag" / "daily" / "d5_pct" / "2026"
LIVE_OUT = data.ROOT / "output" / "live"
DECISION_TEMPLATE = data.ROOT / "starter-kit" / "decision.json"
DRY_RUN_PHASE = "validation"

# ~330 sessions: mom_12_1 reads the close 250 sessions before d-1, dist_high_250d wants
# 200 of 250, and the universe's dollar volume 63. Short of that, the long features are
# NaN for every name and the rank of what remains is a different model input.
HISTORY_DAYS = 480
MIN_SESSIONS = 260
# Membership spells ending within this window are fetched too: cheap, and a name the
# snapshot has not yet recorded as leaving still gets priced rather than dropped.
MEMBERSHIP_DAYS = 730
# Yahoo's daily bar for a session is final a little after the close.
SETTLE = pd.Timedelta(minutes=30)
MAX_NAN_SHARE = 0.05
MAX_MISSING_MEMBERS = 0.05


class LiveDataError(RuntimeError):
    """Live inputs a decision must not be made from. Raised, never defaulted: the
    alternative is a legal-looking decision computed from the wrong data."""


class StaleDataError(LiveDataError):
    """The feed's latest bar is older than the last completed session."""


class FeatureMismatchError(LiveDataError):
    """The live feature frame is not the frame the model was trained on."""


# ---------------------------------------------------------------------------- calendar

class _NYSEHolidays(AbstractHolidayCalendar):
    """Full-day NYSE closures. New Year on a Saturday is not observed on the Friday
    (NYSE rule), unlike the federal calendar. Unscheduled closures (a national day of
    mourning) are not here: on one, the freshness check raises rather than guessing."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def sessions(start, end) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    return days.difference(_NYSEHolidays().holidays(days.min(), days.max())) if len(days) else days


def _as_now(as_of=None) -> pd.Timestamp:
    """The wall-clock moment the decision is made, in ET. A bare date means after that
    day's close (how a replay reads it)."""
    if as_of is None:
        return pd.Timestamp.now(tz=calendar.TZ)
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        if ts == ts.normalize():
            ts = ts + pd.Timedelta(hours=23, minutes=59)
        return ts.tz_localize(calendar.TZ)
    return ts.tz_convert(calendar.TZ)


def latest_completed_session(now: pd.Timestamp) -> pd.Timestamp:
    """The last session whose close (plus settling time) is before `now`, as a naive date."""
    today = now.tz_localize(None).normalize()
    days = sessions(today - pd.Timedelta(days=14), today)
    done = [d for d in days if calendar.at(d.date(), calendar.session_close(d.date())) + SETTLE <= now]
    return done[-1]


def next_session(day: pd.Timestamp) -> pd.Timestamp:
    return sessions(day + pd.Timedelta(days=1), day + pd.Timedelta(days=14))[0]


# ---------------------------------------------------------------------- the network doors
# Module-level so tests replace them; everything else is computed from what they return.

def yahoo_daily(symbols: list[str], start: str) -> tuple[pd.DataFrame, list[str]]:
    """Split-adjusted, not dividend-adjusted daily bars: the training basis."""
    return external.fetch_daily(symbols, start=start)


def load_membership() -> pd.DataFrame:
    return universe.load_membership()


def load_events(symbols: list[str], now: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """(ticker, accepted) earnings 8-Ks known at `now`, and where they came from.

    EDGAR when SEC_USER_AGENT is set; otherwise the last snapshot, flagged stale.
    Releases after `now` are dropped: a live run cannot know them, and in a replay the
    snapshot does, which would hand the replay a better earnings column than live had.
    """
    if os.environ.get("SEC_USER_AGENT", "").strip():
        events, missing = earnings.fetch(symbols)
        meta = {"source": "edgar", "stale": False, "no_cik": missing}
    else:
        # "earnings_2*", not "earnings_*": the Yahoo calendar's earnings_calendar_<date>
        # sorts after every dated EDGAR snapshot and is a different table (scheduled
        # dates, no acceptance times).
        path = universe.latest("earnings_2*.parquet")
        events = pd.read_parquet(path)
        meta = {"source": "snapshot", "path": str(path), "stale": True,
                "note": "SEC_USER_AGENT unset: releases after the snapshot are missing"}
    events = events[events["accepted"] <= now]
    meta["latest_release"] = str(events["accepted"].max()) if len(events) else None
    return events, meta


def load_predictor(path: Path = MODEL):
    """The frozen ensemble, with torch held to one thread.

    On this Mac, predicting LightGBM and then the FastAI net in one process deadlocks:
    torch's batch_norm waits forever in an OpenMP barrier (two OpenMP runtimes, 0% CPU,
    no error). Live that is a round that silently misses its deadline and holds. One
    thread costs nothing on ~104 rows (the whole ensemble predicts in about a second).
    """
    import torch
    from autogluon.tabular import TabularPredictor

    torch.set_num_threads(1)
    return TabularPredictor.load(str(path))


# ------------------------------------------------------------------------------- inputs

@dataclass
class LiveInputs:
    now: pd.Timestamp
    latest_session: pd.Timestamp
    decision_date: pd.Timestamp
    daily: pd.DataFrame          # completed bars only, no placeholder row
    ctx: pd.DataFrame
    membership: pd.DataFrame
    meta: dict = field(default_factory=dict)


def live_symbols(membership: pd.DataFrame, day: pd.Timestamp) -> list[str]:
    """Members whose spell touches the last MEMBERSHIP_DAYS, plus the 30."""
    cut = day - pd.Timedelta(days=MEMBERSHIP_DAYS)
    recent = membership[(membership["start"] <= day) & (membership["end"].isna() | (membership["end"] >= cut))]
    return sorted(set(recent["symbol"]) | set(data.load_universe()))


def completed_only(frame: pd.DataFrame, latest: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Drop bars dated after the last completed session (Yahoo's in-progress bar)."""
    late = frame["date"] > latest
    return frame[~late].reset_index(drop=True), int(late.sum())


def check_fresh(frame: pd.DataFrame, latest: pd.Timestamp, label: str, per_symbol: bool = False) -> None:
    """The latest bar must be the last completed session.

    `per_symbol` for the context series: a single lagging series (VIX a day late) leaves
    that day's context value NaN or, worse after a pivot, a day old.
    """
    if frame.empty:
        raise StaleDataError(f"{label}: no bars at all")
    last = frame["date"].max()
    if last < latest:
        raise StaleDataError(f"{label}: latest bar is {last:%Y-%m-%d} but the last completed "
                             f"session was {latest:%Y-%m-%d}; refusing to score old prices")
    if per_symbol:
        lag = frame.groupby("ticker")["date"].max()
        lag = lag[lag < latest]
        if len(lag):
            raise StaleDataError(f"{label}: {dict(lag.dt.strftime('%Y-%m-%d'))} end before "
                                 f"{latest:%Y-%m-%d}")


def with_decision_row(frame: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Append a placeholder bar dated `day` for every ticker: last price, zero volume.

    `build` shifts every as-of-close value one row forward, so d's row needs d to exist.
    Zero volume is Yahoo's own placeholder convention, which `panels` reads as missing;
    the price only has to be non-NaN so pivots keep the date. Nothing dated d feeds any
    feature of d (the leak test in tests/test_daily_features.py).
    """
    last = frame.sort_values("date").groupby("ticker").tail(1).copy()
    last["date"] = day
    last["volume"] = 0.0
    return pd.concat([frame, last], ignore_index=True)


def fetch_inputs(now: pd.Timestamp) -> LiveInputs:
    latest = latest_completed_session(now)
    day = next_session(latest)
    membership = load_membership()
    symbols = live_symbols(membership, day)
    start = f"{day - pd.Timedelta(days=HISTORY_DAYS):%Y-%m-%d}"
    fetched_at = pd.Timestamp.now(tz=calendar.TZ)
    daily, missing = yahoo_daily(symbols, start)
    ctx, ctx_missing = yahoo_daily(list(external.CONTEXT_SYMBOLS), start)
    if ctx_missing:
        # A context series the model trained with, gone: ctx_* would be NaN or computed
        # over fewer sectors, a market it never saw, with nothing looking wrong.
        raise LiveDataError(f"Yahoo returned no context bars for {ctx_missing}")
    daily, dropped = completed_only(daily, latest)
    ctx, ctx_dropped = completed_only(ctx, latest)
    # Yahoo's ^VIX printed on Memorial Day and Labor Day 2026, when equities were shut.
    # `daily_features.context` diffs along its own dates, so that row makes SPY's return
    # NaN and ctx_spy_ret_20d / ctx_spy_vol_20d NaN for the next 20 sessions: an input
    # the model (trained through 2025, when no such row existed) never saw.
    on_grid = ctx["date"].isin(sessions(ctx["date"].min(), latest))
    ctx_off_grid = sorted(ctx.loc[~on_grid, "date"].dt.strftime("%Y-%m-%d").unique())
    ctx = ctx[on_grid].reset_index(drop=True)
    check_fresh(daily, latest, "universe daily bars")
    check_fresh(ctx, latest, "context series", per_symbol=True)
    n_sessions = daily["date"].nunique()
    if n_sessions < MIN_SESSIONS:
        raise LiveDataError(f"only {n_sessions} sessions of history; the long features need {MIN_SESSIONS}")

    members = set(membership[(membership["start"] <= day) & (membership["end"].isna() | (membership["end"] > day))]["symbol"])
    unpriced = sorted(members & set(missing))
    if len(unpriced) > MAX_MISSING_MEMBERS * max(len(members), 1):
        # The top 100 is refilled from whoever is left, so the universe keeps its size and
        # every rank shifts quietly. A handful is normal (renames, takeovers the snapshot
        # still lists); a large share is a feed problem.
        raise LiveDataError(f"{len(unpriced)} of {len(members)} current members unpriced: {unpriced[:20]}")
    return LiveInputs(now=now, latest_session=latest, decision_date=day, daily=daily, ctx=ctx,
                      membership=membership, meta={
                          "fetched_at": str(fetched_at), "history_start": start,
                          "symbols_requested": len(symbols), "symbols_missing": missing,
                          "current_members_unpriced": unpriced,
                          "partial_bars_dropped": dropped + ctx_dropped, "sessions": int(n_sessions),
                          "context_off_session_dates_dropped": ctx_off_grid})


# ------------------------------------------------------------------------------- scores

@dataclass
class Scored:
    ours: pd.Series              # the 30, kit order; NaN = missing data: no score shown, never a guess
    universe: pd.Series          # every universe name on the decision date, for the log
    features: pd.DataFrame       # the decision date's feature rows, as scored
    decision_date: pd.Timestamp
    inputs: LiveInputs
    events: pd.DataFrame         # the (ticker, accepted) releases the features saw
    events_meta: dict
    warnings: list[str]


def check_features(frame: pd.DataFrame, expected: list[str], tickers: list[str]) -> None:
    """The frame must be the model's exact columns in the model's order, and mostly filled.

    AutoGluon selects columns by name, so a renamed or missing feature is filled as NaN
    rather than refused, and a model scoring a column of NaN still returns a number.
    """
    got = list(frame.columns)
    if got != list(expected):
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        raise FeatureMismatchError(f"live features differ from the model's: missing {missing}, "
                                   f"extra {extra}, first out of order at position {first}")
    ctx_cols = [c for c in got if c.startswith("ctx_")]
    empty_ctx = [c for c in ctx_cols if frame[c].isna().all()]
    if empty_ctx:
        raise LiveDataError(f"market context missing on the decision date: {empty_ctx}")
    # Per-name columns only: e_* are sparse by design (no release within 10 sessions is
    # NaN), and ctx_* are market-wide, checked above. A missing row counts as all-NaN.
    dense = [c for c in got if not c.startswith(("e_", "ctx_")) and c != "is_competition"]
    ours = frame.reindex(tickers)[dense]
    share = float(ours.isna().to_numpy().mean())
    if share > MAX_NAN_SHARE:
        worst = ours.isna().sum(axis=1).sort_values(ascending=False).head(5)
        raise LiveDataError(f"{share:.1%} of the 30 names' features are NaN (limit {MAX_NAN_SHARE:.0%}); "
                            f"worst: {dict(worst)}")


def check_prediction(pred: pd.Series) -> None:
    p = pred.dropna()
    if len(p) < 2 or p.nunique() <= 1 or float(p.std()) < 1e-9:
        # A constant score is what a model fed all-NaN or all-identical rows returns; the
        # agent would read a model with no view as one that likes every name equally.
        raise LiveDataError(f"prediction among the 30 is constant ({p.nunique()} distinct values)")


@contextmanager
def _timed(timings: dict, stage: str):
    t = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round(time.perf_counter() - t, 3)


def daily_scores(as_of=None, *, predictor=None, inputs: Optional[LiveInputs] = None,
                 timings: Optional[dict] = None) -> Scored:
    """The frozen daily model's scores for the next session after the latest close."""
    timings = {} if timings is None else timings
    now = _as_now(as_of)
    tickers = list(data.load_universe())
    warnings: list[str] = []
    with _timed(timings, "fetch_prices"):
        inputs = inputs or fetch_inputs(now)
    day = inputs.decision_date

    with _timed(timings, "universe"):
        daily_x = with_decision_row(inputs.daily, day)
        ctx_x = with_decision_row(inputs.ctx, day)
        mask = universe.build(daily_x, inputs.membership)
        if day not in mask.index:
            raise LiveDataError(f"no universe row for the decision date {day:%Y-%m-%d}")
        names = sorted(mask.columns[mask.loc[day].to_numpy()])

    with _timed(timings, "earnings"):
        raw_events, events_meta = load_events(names, inputs.now)
        events = earnings.quarterly(raw_events)
    if events_meta.get("stale"):
        warnings.append(f"earnings from a stale snapshot ({events_meta.get('path')}); "
                        f"latest release {events_meta.get('latest_release')}")
    warnings.append("e_sessions_to_next is NaN for every name live (EDGAR records only past "
                    "releases); training saw it filled within 10 sessions of a release. "
                    "Cost on 2026: IC_30 0.021 -> 0.012 until an earnings calendar feeds it")

    with _timed(timings, "features"):
        f = daily_features.build(daily_x, mask, ctx_x, events)
        dates = f.index.get_level_values("date")
        if not (dates == day).any():
            raise LiveDataError(f"the feature frame has no rows for {day:%Y-%m-%d}")
        frame = f.xs(day, level="date")

    with _timed(timings, "load_model"):
        predictor = predictor if predictor is not None else load_predictor()
    check_features(frame, list(predictor.features()), tickers)

    with _timed(timings, "predict"):
        cols = list(predictor.features())
        pred = pd.Series(np.asarray(predictor.predict(frame[cols].reset_index(drop=True)), dtype=float),
                         index=frame.index, name="pred")
    ours = pred.reindex(tickers)
    # A name with no bar on the latest session has features from older prices mixed with
    # today's ranks. NaN it: the agent is shown no score rather than one from mixed dates.
    last_bar = inputs.daily.groupby("ticker")["date"].max().reindex(tickers)
    stale = [t for t in tickers if not last_bar.get(t) == inputs.latest_session]
    if stale:
        warnings.append(f"no bar on {inputs.latest_session:%Y-%m-%d} for {stale}: scored NaN")
        ours[stale] = np.nan
    check_prediction(ours)
    return Scored(ours=ours, universe=pred, features=frame, decision_date=day, inputs=inputs,
                  events=raw_events, events_meta=events_meta, warnings=warnings)


# ------------------------------------------------------------------- the desk's market
# The rule desk and the shadow agent run the backtested `Desk` on a `sim.Market` built
# from live bars. The desk reads daily closes (shape, regime, the trigger's sigma) and
# the latest bar (valuation, the trigger's move); nothing in it reads hourly history,
# so past sessions are one daily bar each and only today keeps its intraday bars.

# Calendar days of history: the rule reads qs.HISTORY_DAYS + 1 = 751 sessions.
CLOSE_HISTORY_DAYS = 1200
INTRADAY_PERIOD = "5d"
# Sessions ahead in the market's calendar: the desk counts sessions to the next FOMC
# decision along `market.days`, and a calendar ending today reads "no meeting ahead".
FUTURE_DAYS = 60


def yahoo_intraday(symbols: list[str], interval: str = "30m",
                   period: str = INTRADAY_PERIOD) -> pd.DataFrame:
    """Completed intraday bars (canonical frame); raises if a name has none."""
    return public_bars.fetch(symbols, interval, period)


def fetch_closes(now: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """The 30 names' daily bars through the last completed session, checked fresh.

    Every name must have the latest session's bar. The entry is sized on the last 60
    sessions' covariance and valued on the latest close, and a name a day behind
    would be sized and valued on prices the others have moved past.
    """
    tickers = sorted(data.load_universe())
    latest = latest_completed_session(now)
    start = f"{latest - pd.Timedelta(days=CLOSE_HISTORY_DAYS):%Y-%m-%d}"
    fetched_at = pd.Timestamp.now(tz=calendar.TZ)
    daily, missing = yahoo_daily(tickers, start)
    if missing:
        raise LiveDataError(f"Yahoo returned no daily bars for {missing}")
    daily, dropped = completed_only(daily, latest)
    check_fresh(daily, latest, "competition names' daily bars", per_symbol=True)
    n = int(daily["date"].nunique())
    if n < qs.HISTORY_DAYS + 1:
        # The regime fit and the 3-year reference vol would be fit on a shorter span
        # than the backtest's, a different model with nothing looking wrong.
        raise LiveDataError(f"only {n} sessions of daily history; the rule reads {qs.HISTORY_DAYS + 1}")
    return daily, {"latest_session": str(latest.date()), "sessions": n,
                   "partial_bars_dropped": dropped, "fetched_at": str(fetched_at)}


def fetch_intraday(now: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """The 30 names' 30m bars of the last few sessions, ended by `now`.

    30m, not 60m: a round's fill is the open of the bar starting at its execution
    time, and the 30m bar that starts there has ended by the next round's wake, while
    the 60m bar has not. `today_60m` pairs them into the backtest's 60m grid.
    """
    bars = public_bars.completed(yahoo_intraday(sorted(data.load_universe())), now)
    return bars, {"bars": int(len(bars)),
                  "last_end": str(bars["end"].max()) if len(bars) else None}


def today_60m(bars_30m: pd.DataFrame, day) -> pd.DataFrame:
    """`day`'s completed 60m bars (09:30-10:30 ... 15:30-16:00), paired from 30m.

    A pair missing either half is dropped, as in the backtest's `alpaca.to_60m`: with
    one half missing, its close would be the other half's, 30 minutes off.
    """
    b = bars_30m[bars_30m["start"].dt.date == day]
    if b.empty:
        return b
    out = alpaca.to_60m(b)
    out["source"] = "yahoo_30m_pairs"
    return out


def fills(bars_30m: pd.DataFrame) -> pd.DataFrame:
    """Opens of the 30m bars, by start time: a round's fill is the row at its execution."""
    return bars_30m.pivot_table(index="start", columns="ticker", values="open", aggfunc="first")


def market(daily: pd.DataFrame, intraday: Optional[pd.DataFrame], days: list) -> sim.Market:
    """A `sim.Market` the desk can read live: completed sessions as daily bars, then
    `intraday` (today's 60m bars, ended by the fetch).

    `days` is the market's calendar, history and the sessions ahead. Its execution and
    close frames are NaN: nothing executes against this market, they only give it its
    tickers and days.
    """
    tickers = sorted(data.load_universe())
    d = daily[daily["ticker"].isin(tickers)].copy()
    day = d["date"].dt.date
    opens = {x: calendar.at(x, calendar.SESSION_OPEN) for x in day.unique()}
    closes = {x: calendar.at(x, calendar.session_close(x)) for x in day.unique()}
    bars = pd.DataFrame({
        "ticker": d["ticker"].to_numpy(), "start": day.map(opens).to_numpy(),
        "end": day.map(closes).to_numpy(), "open": d["open"].to_numpy(),
        "high": d["high"].to_numpy(), "low": d["low"].to_numpy(),
        "close": d["close"].to_numpy(), "volume": d["volume"].to_numpy(), "source": "yahoo_1d"})
    bars["start"] = pd.to_datetime(bars["start"]).dt.tz_convert(calendar.TZ)
    bars["end"] = pd.to_datetime(bars["end"]).dt.tz_convert(calendar.TZ)
    if intraday is not None and len(intraday):
        late = intraday[intraday["start"].dt.date > max(closes)]
        bars = pd.concat([bars, late[data.COLUMNS]], ignore_index=True)
    execution = pd.DatetimeIndex([r["execution"] for s in days for r in calendar.rounds_for(s)])
    close_ts = pd.DatetimeIndex([calendar.at(s, calendar.session_close(s)) for s in days])
    return sim.Market(pd.DataFrame(np.nan, index=execution, columns=tickers),
                      pd.DataFrame(np.nan, index=close_ts, columns=tickers), bars,
                      issues={"live": True})


def market_days(daily: pd.DataFrame, latest: pd.Timestamp) -> list:
    """Sessions from the first daily bar to FUTURE_DAYS past the latest close."""
    return [d.date() for d in sessions(daily["date"].min(), latest + pd.Timedelta(days=FUTURE_DAYS))]


class CalendarEarnings:
    """Sessions to each name's next earnings reaction, from Yahoo's scheduled dates.

    The live stand-in for the replay's `agents.desk.EarningsCalendar` (realised EDGAR
    releases, as if announced), with the same `to_next` contract. Yahoo posts a date
    and a side of the session, not a time. A date posted with no time is read as
    before the open, the earlier of its two possible reactions: a flag a session early
    costs one analyst call, a flag a session late is a gap the book already took.
    """

    def __init__(self, cal: pd.DataFrame, days: list):
        from icaif.agents.desk import EarningsCalendar

        from datetime import time as clock

        accepted = pd.Series([calendar.at(pd.Timestamp(d).date(),
                                          clock(16, 30) if side == "amc" else clock(8, 0))
                              for d, side in zip(cal["date"], cal["side"])])
        events = pd.DataFrame({"ticker": cal["ticker"].to_numpy(), "accepted": accepted})
        self._cal = EarningsCalendar(events, days)

    def to_next(self, day) -> dict:
        return self._cal.to_next(day)


# ----------------------------------------------------------------------------- decision

def envelope(weights: dict, round_row: dict, team: Optional[dict] = None) -> dict:
    """The kit's decision.json for `round_row`, every symbol in the template's order.

    Without `team` it is a dry run's: the kit's placeholder credentials and a
    `dryrun-` round id, which the kit's client rejects locally (placeholder check, a
    round no schedule contains) before any upload. Only an armed live round passes
    `team`, the credentials the kit's session holds.
    """
    template = json.loads(DECISION_TEMPLATE.read_text())
    if team is None:
        creds = {"team_id": template["team_id"], "team_token": template["team_token"]}
        round_id = f"dryrun-{round_row['id']}"
    else:
        creds = {"team_id": team["team_id"], "team_token": team["team_token"]}
        round_id = round_row["id"]
    values = {"submission_type": "decision", **creds,
              "phase": round_row.get("phase") if round_row.get("phase") in ("validation", "official")
              else DRY_RUN_PHASE,
              "round_id": round_id, "weights": {t: weights[t] for t in template["weights"]}}
    return {k: values[k] for k in template}


def check_decision(decision: dict) -> str:
    """The JSON text of the decision, validated as the backend will parse it.

    The kit validates `Decimal(str(w))`, so the check runs on the serialised text parsed
    back with Decimal, not on the floats in memory.
    """
    if set(decision) != set(kit.contracts.FIELDS["decision"]):
        raise kit.SubmissionError(f"decision fields {sorted(decision)} are not the kit's")
    text = json.dumps(decision, indent=2)
    kit.validate_weights(json.loads(text, parse_float=Decimal)["weights"])
    return text


def _clean(x):
    """JSON-safe copy: NaN -> null, numpy -> Python, sets and tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = sorted(x) if isinstance(x, (set, frozenset)) else x
        return [_clean(v) for v in items]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    return x


def archive_scores(scored: "Scored", out: Path) -> dict:
    """Write what the shadow agent and the record need from a scoring run.

    `scores.parquet` is the 30 names' row for the decision date (what the agent sees);
    the rest is the evidence that the row was built from data fetched before the
    deadline: the inputs, the features and every universe name's score.
    """
    out.mkdir(parents=True, exist_ok=True)
    scored.ours.rename("pred").rename_axis("ticker").reset_index().assign(
        date=scored.decision_date).to_parquet(out / "scores.parquet", index=False)
    scored.inputs.daily.to_parquet(out / "prices_daily_universe.parquet", index=False)
    scored.inputs.ctx.to_parquet(out / "prices_context.parquet", index=False)
    scored.events.to_parquet(out / "earnings_events.parquet", index=False)
    scored.features.reset_index().to_parquet(out / "features.parquet", index=False)
    scored.universe.rename("pred").reset_index().to_parquet(out / "scores_universe.parquet", index=False)
    meta = _clean({"decision_date": str(scored.decision_date.date()),
                   "latest_session": str(scored.inputs.latest_session.date()),
                   "inputs": {**scored.inputs.meta, "earnings": scored.events_meta},
                   "model": str(MODEL), "universe_size": int(len(scored.universe)),
                   "warnings": scored.warnings})
    (out / "scores_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
