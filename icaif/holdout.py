"""Score an agent's decisions, handed over as one JSON file, on a held-out span.

The official phase is one 15-day window that every entrant starts from $1M in cash. So
the harness scores a fresh $1M in every 15-day window starting on each trading day of
the span (`windows.py`), and the file carries one decision sequence per window: the
agent run as itself from that window's first round, as it would be in the contest.

    {"strategy": "my_agent", "suite": "holdout",
     "windows": {"2026-01-02": [{"round_id": "holdout-2026-01-02-r1", "cash": 0.25,
                                 "weights": {<all 30 symbols>: number}}, ...],
                 "2026-01-05": [...], ...}}

The file used to hold one continuous run, replayed into every window from cash. That
scored a different strategy from the one the agent is. An equal-weight buy-and-hold
written that way bought, in a March window, the weights that had drifted since 2 Jan,
not 1/30 each; and an agent that decides from its own book (a drawdown stop, profit
booking, a band around its holdings) saw a book it never had. The old shape is rejected
by name, not accepted alongside: both would rank side by side with nothing to tell
them apart.

The windows are a suite's (`suites.py`): every rolling window of the holdout span, or a
fixed set such as official4. The file names its suite, and a file without one is a
holdout file, as every file was before suites existed. A file for one suite scored as
another is rejected by name: its windows would otherwise fail one by one, or worse, a
holdout file narrowed to a stretch could pass as a different suite's.

Two kinds of error are treated differently, as the backend treats them:

- **Structural errors reject the file before anything runs**: a window that is missing
  or does not exist, a malformed, duplicate or out-of-window round, a wrong symbol set,
  or a cash weight that disagrees with the stock weights. Each is a sign the agent and
  the harness disagree about what a window or round is. Scoring the file anyway would
  produce a clean number for a run that never happened.
- **Weight-rule failures and missing rounds hold**, exactly as the backend holds
  them: over the cap, negative, a total over 1, or float dust like 0.1 + 0.2. They
  are scored as holds and listed, because the same file uploaded live would hold
  the same rounds. `strict=True` makes them fatal.

Weights are parsed as Decimals from the JSON text and never snapped to a grid. The
point is to score what would be submitted, and snapping here would quietly fix the
float dust that the backend rejects.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from icaif import calendar, kit, sim, suites
from icaif.leaderboard import independent_windows  # noqa: F401  (re-exported)
from icaif.windows import WINDOW_DAYS

HOLDOUT_START, HOLDOUT_END = (date.fromisoformat(d) for d in suites.SUITES["holdout"].span)
CASH_TOLERANCE = Decimal("1e-9")
METRICS = ("cumulative_return", "sharpe_ratio", "maximum_drawdown", "turnover")
# A systematic mistake repeats in every window; 11,000 identical lines bury the first.
MAX_LISTED_ERRORS = 25

_ROUND_ID = re.compile(r"^[a-z]+-(\d{4}-\d{2}-\d{2})-r(\d+)$")


class DecisionFileError(ValueError):
    """The file cannot be scored; the offending windows and rounds are named."""


@dataclass
class Decisions:
    strategy: str
    suite: str
    windows: dict                     # window start -> {(day, round): {ticker: Decimal}}
    spans: dict                       # window start -> its trading days, scored windows only
    skipped: list = field(default_factory=list)   # starts of windows on a degraded day
    missing: list = field(default_factory=list)   # "<window start> <round_id>", held
    invalid: list = field(default_factory=list)   # rounds that break a weight rule, held

    def strategy_fn(self, start: date):
        # A missing round returns None, which `sim.run` holds, as the backend does.
        # Defined here rather than borrowing `rankplay._Replay`: that module pulls in the
        # strategy code, and the hosted harness must ship without it.
        plan = self.windows[start]
        return lambda ctx: plan.get((ctx.day, ctx.round))


def span_days(market: sim.Market, start: date, end: date) -> list[date]:
    days = [d for d in market.days if start <= d <= end]
    if not days:
        raise DecisionFileError(f"the market has no trading days in {start}..{end}")
    # A market snapshot that stops short would score a shorter run under the same name.
    if max(market.days) < end:
        raise DecisionFileError(f"the market ends {max(market.days)}, before {end}")
    return days


def window_spans(market: sim.Market, start: date, end: date, n_days: int = WINDOW_DAYS):
    """Every n_days window starting on a trading day of the span: (scored, skipped starts).

    A window touching a degraded day is skipped and named: its stand-in prices make it
    look calmer than the market was.
    """
    days = span_days(market, start, end)
    degraded = set(market.issues.get("degraded_days", []))
    scored, skipped = {}, []
    for i in range(len(days) - n_days + 1):
        span = days[i:i + n_days]
        if degraded & {str(d) for d in span}:
            skipped.append(str(span[0]))
        else:
            scored[span[0]] = span
    return scored, skipped


def suite_spans(market: sim.Market, suite) -> tuple[dict, list]:
    """A suite's windows on this market: ({first day: its trading days}, skipped starts).

    A rolling suite skips and names a window touching a degraded day, as `window_spans`
    does. A fixed suite cannot: dropping one of four windows would rank a different suite
    under the same name. So a degraded day, a window whose sessions the market does not
    hold all of, or one that runs past its stated last session, raises.
    """
    suite = suites.get(suite)
    if not suite.fixed:
        start, end = (date.fromisoformat(d) for d in suite.span)
        return window_spans(market, start, end, suite.n_days)
    span_days(market, *(date.fromisoformat(d) for d in suite.span))
    degraded = set(market.issues.get("degraded_days", []))
    scored = {}
    for first, last in suite.windows:
        days = [d for d in market.days if first <= str(d) <= last]
        if len(days) != suite.n_days or (str(days[0]), str(days[-1])) != (first, last):
            raise DecisionFileError(
                f"suite {suite.name} window {first}..{last}: the market holds "
                f"{len(days)} trading days there, not {suite.n_days} from {first} to {last}")
        if degraded & {str(d) for d in days}:
            raise DecisionFileError(f"suite {suite.name} window {first}..{last} touches a "
                                    "degraded day; a fixed suite cannot skip a window")
        scored[days[0]] = days
    return scored, []


def suite_days(market: sim.Market, suite) -> list[date]:
    """The trading days a suite scores: its span for a rolling suite, the union of its
    windows for a fixed one. A fixed suite's span runs over a year it never scores, and
    a day count off it would report sessions no window contains."""
    suite = suites.get(suite)
    if not suite.fixed:
        return span_days(market, *(date.fromisoformat(d) for d in suite.span))
    spans, _ = suite_spans(market, suite)
    return sorted({d for days in spans.values() for d in days})


def round_id(day: date, round_no: int, phase: str = "holdout") -> str:
    return f"{phase}-{day.isoformat()}-r{round_no}"


def load_decisions(path, market: sim.Market, suite=suites.DEFAULT,
                   strict: bool = False) -> Decisions:
    """Parse and check a decisions file against the suite's windows on the market's calendar."""
    suite = suites.get(suite)
    raw = Path(path).read_text()
    doc = json.loads(raw, parse_float=Decimal, parse_constant=_reject_constant)
    if isinstance(doc, dict) and "decisions" in doc and "windows" not in doc:
        raise DecisionFileError(
            'this is the old one-run format ("decisions": [...]). Each 15-day window now '
            'has its own decisions, the agent run from cash at that window\'s first round: '
            '{"strategy": ..., "windows": {"YYYY-MM-DD": [...], ...}}. '
            "tools/holdout_template.py writes the new shape.")
    if not isinstance(doc, dict) or not isinstance(doc.get("windows"), dict):
        raise DecisionFileError('expected {"strategy": ..., "windows": {"YYYY-MM-DD": [...]}}')
    strategy = str(doc.get("strategy") or Path(path).stem)
    named = doc.get("suite", suites.DEFAULT)
    if named != suite.name:
        raise DecisionFileError(
            f"this file is for suite {named!r} and was asked to score as {suite.name!r}. "
            f"Score it as its own suite, or write the file for {suite.name!r} "
            f"(tools/holdout_template.py --suite {suite.name}). Suites: {', '.join(suites.SUITES)}")

    spans, skipped = suite_spans(market, suite)
    n_days = suite.n_days
    tickers = set(market.tickers)
    errors, windows, missing, invalid = [], {}, [], []
    given = {}
    for key, entries in doc["windows"].items():
        try:
            ws = date.fromisoformat(key)
        except ValueError:
            errors.append(f"window {key!r}: not a YYYY-MM-DD start day")
            continue
        if key in skipped:
            continue   # touches a degraded day; not scored, so not checked
        if ws not in spans:
            errors.append(f"window {key}: not one of suite {suite.name}'s windows ("
                          + (", ".join(w for w, _ in suite.windows) if suite.fixed else
                             f"a {n_days}-day window from each trading day in "
                             f"{suite.span[0]}..{suite.span[1]}; not a trading day, or too "
                             "close to the end") + ")")
            continue
        if not isinstance(entries, list):
            errors.append(f"window {key}: expected a list of decisions")
            continue
        given[ws] = entries
    for ws in spans:
        if ws not in given and not any(e.startswith(f"window {ws}") for e in errors):
            errors.append(f"window {ws}: no decisions. Every window is scored, each from "
                          "cash by its own run")

    for ws, entries in given.items():
        expected = [(d, r["round"]) for d in spans[ws] for r in calendar.rounds_for(d)]
        weights, errs, bad = _parse_rounds(entries, set(expected), tickers, f"window {ws}")
        errors.extend(errs)
        windows[ws] = weights
        invalid.extend({"window": str(ws), **b} for b in bad)
        missing.extend(f"{ws} {round_id(d, r)}" for d, r in expected if (d, r) not in weights)

    if errors:
        listed = "\n  ".join(errors[:MAX_LISTED_ERRORS])
        more = f"\n  ... and {len(errors) - MAX_LISTED_ERRORS} more" if len(errors) > MAX_LISTED_ERRORS else ""
        raise DecisionFileError(f"{len(errors)} problem(s) in {path}:\n  {listed}{more}")
    if strict and (missing or invalid):
        raise DecisionFileError(
            f"strict: {len(missing)} missing round(s), {len(invalid)} invalid round(s); "
            f"first missing {missing[:5]}, first invalid {invalid[:5]}")
    ordered = {ws: windows[ws] for ws in spans}
    return Decisions(strategy, suite.name, ordered, spans, skipped, missing, invalid)


def _parse_rounds(entries: list, valid: set, tickers: set, where_window: str):
    """One window's decisions: (weights by (day, round), structural errors, held rounds)."""
    errors, weights, invalid = [], {}, []
    for i, entry in enumerate(entries):
        rid = entry.get("round_id") if isinstance(entry, dict) else None
        where = f"{where_window} decision {i} ({rid!r})"
        m = _ROUND_ID.match(rid) if isinstance(rid, str) else None
        if m is None:
            errors.append(f"{where}: round_id is not <phase>-YYYY-MM-DD-r<n>")
            continue
        key = (date.fromisoformat(m.group(1)), int(m.group(2)))
        if key not in valid:
            errors.append(f"{where}: no such round in this window "
                          "(weekend, holiday, a round cancelled by an early close, "
                          "or outside the window's 15 days)")
            continue
        if key in weights:
            errors.append(f"{where}: a second decision for the same round")
            continue
        w = entry.get("weights")
        if not isinstance(w, dict) or set(w) != tickers:
            got = set(w) if isinstance(w, dict) else set()
            errors.append(f"{where}: weights must name exactly the 30 symbols "
                          f"(missing {sorted(tickers - got)}, extra {sorted(got - tickers)})")
            continue
        weights[key] = w

        try:
            kit.validate_weights(w)
        except kit.SubmissionError as err:
            invalid.append({"round_id": rid, "reason": str(err)})
            continue
        # Only checked on a valid round: a string or negative weight has no sum to
        # compare against, and that round holds anyway.
        cash = entry.get("cash")
        if isinstance(cash, bool) or not isinstance(cash, (int, Decimal)):
            errors.append(f"{where}: cash must be a JSON number")
            continue
        implied = Decimal(1) - sum(Decimal(v) for v in w.values())
        if abs(Decimal(cash) - implied) > CASH_TOLERANCE:
            errors.append(f"{where}: cash {cash} but the stock weights leave {implied}")
    return weights, errors, invalid


def _reject_constant(name):
    raise DecisionFileError(f"{name} is not a JSON number the backend accepts")


def market_from_frames(exec_prices: pd.DataFrame, closes: pd.DataFrame,
                       degraded_days: list[str]) -> sim.Market:
    """A replay-only Market: fills and closes, no information bars.

    A decisions file is fixed before the harness sees it, so nothing reads
    `history()`. Shipping only these two frames is what lets the hosted harness carry a
    few hundred KB of 2026 prices rather than the bar archive.
    """
    info = pd.DataFrame({"end": pd.Series([], dtype=f"datetime64[ns, {calendar.TZ}]")})
    return sim.Market(exec_prices, closes, info, issues={"degraded_days": list(degraded_days)})


def _window_row(res: sim.Result, span: list, n_missing: int) -> dict:
    return {"window_start": str(span[0]), "window_end": str(span[-1]), **res.metrics(),
            "missing_rounds": n_missing, "invalid_rounds": len(res.invalid_rounds)}


def rolling(decisions: Decisions, market: sim.Market, sizing: str = "pre_fee"):
    """A fresh $1M in every window, each running that window's own decisions.

    Returns (per-window DataFrame, skipped starts), in the shape of `rolling_runs`.
    """
    n_missing = {}
    for m in decisions.missing:
        n_missing[m.split(" ", 1)[0]] = n_missing.get(m.split(" ", 1)[0], 0) + 1
    rows = []
    for ws, span in decisions.spans.items():
        res = sim.run(decisions.strategy_fn(ws), market, span[0], len(span), sizing=sizing)
        rows.append(_window_row(res, span, n_missing.get(str(ws), 0)))
    return pd.DataFrame(rows), list(decisions.skipped)


def rolling_runs(factory, market: sim.Market, suite=suites.DEFAULT, sizing: str = "pre_fee"):
    """`rolling` for any strategy factory, one fresh instance per window of the suite.

    The leaderboard's reference strategies run here as themselves, so a stateful one
    such as a buy-and-hold starts each window in cash, as it would in the contest. A
    decisions file holds the same thing written down: one run per window.
    """
    spans, skipped = suite_spans(market, suite)
    rows = [_window_row(sim.run(factory(), market, span[0], len(span), sizing=sizing), span, 0)
            for span in spans.values()]
    return pd.DataFrame(rows), skipped


def summarise_rolling(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Per metric: mean, median, p10, p90, min, max across windows.

    Adjacent windows share 14 of 15 days, so ~150 windows are ~10 independent
    samples; the count of non-overlapping windows is stated beside the total.
    """
    m = windows_df[list(METRICS)]
    out = pd.DataFrame({
        "mean": m.mean(), "median": m.median(),
        "p10": m.quantile(0.10), "p90": m.quantile(0.90),
        "min": m.min(), "max": m.max(),
    })
    out.attrs["windows"] = len(windows_df)
    out.attrs["independent_windows"] = independent_windows(windows_df)
    return out
