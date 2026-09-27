"""A live round's decision from public data and the frozen daily model: a dry run.

Nothing here talks to Codabench. `decide` fetches what a real round would fetch
(Yahoo daily bars for the training universe and the context series), builds the daily
features for the next session exactly as training did, scores them with the frozen
predictor, compiles legal weights, checks them with the organizers' own validator, and
archives every input beside the decision. The envelope it writes carries the kit's
placeholder credentials and a `dryrun-` round id, so the kit's client refuses it
locally (placeholder check, unknown round) before any upload could happen.

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

Replayed on the 2026-09-27 snapshots for 2026-08-20, the live feature frame equals
`daily_features.build`'s row for row except `e_sessions_to_next`, and scoring the
training frame reproduces output/preds/daily_d5_pct.parquet exactly.
"""

import dataclasses
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday, sunday_to_monday)

from icaif import calendar, compiler, daily_features, data, earnings, external, kit, universe

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
        path = universe.latest("earnings_*.parquet")
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
    ours: pd.Series              # the 30, kit order; NaN = missing data, which the compiler holds
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
        # compiler would then pick names by vol alone and call it the model's choice.
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
    # today's ranks. NaN it: the compiler holds a NaN (frozen), rather than trading on it.
    last_bar = inputs.daily.groupby("ticker")["date"].max().reindex(tickers)
    stale = [t for t in tickers if not last_bar.get(t) == inputs.latest_session]
    if stale:
        warnings.append(f"no bar on {inputs.latest_session:%Y-%m-%d} for {stale}: scored NaN (held)")
        ours[stale] = np.nan
    check_prediction(ours)
    return Scored(ours=ours, universe=pred, features=frame, decision_date=day, inputs=inputs,
                  events=raw_events, events_meta=events_meta, warnings=warnings)


def trailing_vol(daily: pd.DataFrame, decision_date: pd.Timestamp) -> pd.Series:
    """The 30's 20-session daily vol through the prior close: the backtest's own function.

    Called through `compiler.trailing_daily_vol` rather than recomputed, so live and the
    backtest cannot drift apart on the window, min_periods or the one-session shift.
    """
    panel = compiler.trailing_daily_vol(with_decision_row(daily, decision_date))
    day = decision_date.date()
    return panel.for_day(day, calendar.at(day, calendar.ROUNDS[1][0]))


# ----------------------------------------------------------------------------- decision

def portfolio_weights(portfolio: Optional[Mapping], tickers: list[str]) -> pd.Series:
    """Current weights from {"weights": {...}} or a flat {ticker: weight}; None is all cash.

    An unknown ticker or a negative weight raises: a typo would otherwise read as a zero
    holding, and the compiler would "buy" a name the book already owns.
    """
    if portfolio is None:
        return pd.Series(0.0, index=tickers)
    w = portfolio.get("weights", portfolio)
    unknown = sorted(set(w) - set(tickers))
    if unknown:
        raise ValueError(f"portfolio names outside the 30: {unknown}")
    s = pd.Series({t: float(w.get(t, 0.0)) for t in tickers})
    if (~np.isfinite(s)).any() or (s < 0).any() or s.sum() > 1.0 + 1e-9:
        raise ValueError("portfolio weights must be finite, non-negative and sum to at most 1")
    return s


def envelope(weights: dict, day, round_no: int) -> dict:
    """The kit's decision.json, with its placeholder credentials and a round id no
    schedule contains: the kit's client rejects both locally, before any upload."""
    template = json.loads(DECISION_TEMPLATE.read_text())
    values = {"submission_type": "decision", "team_id": template["team_id"],
              "team_token": template["team_token"], "phase": DRY_RUN_PHASE,
              "round_id": f"dryrun-{pd.Timestamp(day):%Y-%m-%d}-r{round_no}",
              "weights": {t: weights[t] for t in template["weights"]}}
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
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    return x


def decide(round_no: int, portfolio: Optional[Mapping] = None,
           levers: compiler.Levers = compiler.Levers(), *, as_of=None, predictor=None,
           inputs: Optional[LiveInputs] = None, out_dir: Optional[Path] = None,
           archive: bool = True) -> tuple[dict, dict]:
    """(decision.json dict, log record) for `round_no` of the next session. Dry run only."""
    t0 = time.perf_counter()
    timings: dict = {}
    scored = daily_scores(as_of, predictor=predictor, inputs=inputs, timings=timings)
    day = scored.decision_date.date()
    runs = {r["round"]: r for r in calendar.rounds_for(day)}
    if round_no not in runs:
        raise ValueError(f"round {round_no} does not run on {day} (rounds {sorted(runs)})")
    tickers = list(data.load_universe())

    with _timed(timings, "vol"):
        vol = trailing_vol(scored.inputs.daily, scored.decision_date).reindex(tickers)
    current = portfolio_weights(portfolio, tickers)
    with _timed(timings, "compile"):
        weights = compiler.compile_weights(scored.ours, vol, current, levers, round_no)
    with _timed(timings, "validate"):
        decision = envelope(weights, day, round_no)
        text = check_decision(decision)

    changed = [t for t in tickers if abs(weights[t] - current[t]) > compiler.HELD]
    warnings = list(scored.warnings)
    if levers.stop_sigma is not None and round_no not in levers.rebalance_rounds:
        warnings.append("stop_sigma set but the dry run has no intraday bars: stops are blind")
    now = scored.inputs.now
    deadline = runs[round_no]["deadline"]
    fetched_at = pd.Timestamp(scored.inputs.meta["fetched_at"])
    log = {
        "dry_run": True, "round": round_no, "decision_date": str(day), "now": str(now),
        "deadline": str(deadline), "execution": str(runs[round_no]["execution"]),
        "fetched_before_deadline": bool(fetched_at < deadline),
        "latest_session": str(scored.inputs.latest_session.date()),
        "inputs": {**scored.inputs.meta, "earnings": scored.events_meta},
        "model": str(MODEL), "universe_size": int(len(scored.universe)),
        "levers": dataclasses.asdict(levers),
        "scores": scored.ours.to_dict(), "vol": vol.to_dict(),
        "current_weights": current.to_dict(), "weights": weights,
        "gross": sum(weights.values()), "changed": changed,
        # The compiler's "hold": nothing moves, so a real round should not upload at all
        # (re-submitting current weights re-sizes every name at the fill and pays for it).
        "action": "trade" if changed else "hold",
        "warnings": warnings,
    }
    if archive:
        with _timed(timings, "archive"):
            out = Path(out_dir) if out_dir else LIVE_OUT / f"{now:%Y%m%dT%H%M%S}"
            out.mkdir(parents=True, exist_ok=True)
            scored.inputs.daily.to_parquet(out / "prices_daily.parquet", index=False)
            scored.inputs.ctx.to_parquet(out / "prices_context.parquet", index=False)
            scored.events.to_parquet(out / "earnings_events.parquet", index=False)
            scored.features.reset_index().to_parquet(out / "features.parquet", index=False)
            scored.universe.rename("pred").reset_index().to_parquet(out / "scores_universe.parquet", index=False)
            (out / "decision.json").write_text(text)
            log["archive"] = str(out)
            log["decision_path"] = str(out / "decision.json")
    timings["total"] = round(time.perf_counter() - t0, 3)
    log["timings"] = timings
    log = _clean(log)
    if archive:
        (Path(log["archive"]) / "log.json").write_text(json.dumps(log, indent=2))
    return decision, log
