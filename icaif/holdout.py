"""Score an agent's decisions, handed over as one JSON file, on a held-out span.

The agent runs somewhere else and writes, for every round, a cash weight and all 30
stock weights. The file goes through the same ledger (`sim.run`) and the same kit
metrics as every backtest in this repo. It is scored two ways:

- one continuous run from $1M across the whole span, and
- a fresh $1M in every 15-day window starting on each trading day, because the
  official phase is one such window, and a long equity curve says little about a
  15-day rank (`windows.py`).

Input shape:

    {"strategy": "my_agent",
     "decisions": [{"round_id": "holdout-2026-01-02-r1", "cash": 0.25,
                    "weights": {<all 30 symbols>: number}}, ...]}

Two kinds of error are treated differently, as the backend treats them:

- **Structural errors reject the file before anything runs**: a malformed, duplicate
  or non-existent round, a wrong symbol set, or a cash weight that disagrees with the
  stock weights. Each is a sign the agent and the harness disagree about what a round
  is. Scoring the file anyway would produce a clean number for a run that never
  happened.
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

from icaif import calendar, kit, sim
from icaif.windows import WINDOW_DAYS

HOLDOUT_START = date(2026, 1, 2)
HOLDOUT_END = date(2026, 8, 31)
CASH_TOLERANCE = Decimal("1e-9")
METRICS = ("cumulative_return", "sharpe_ratio", "maximum_drawdown", "turnover")

_ROUND_ID = re.compile(r"^[a-z]+-(\d{4}-\d{2}-\d{2})-r(\d+)$")


class DecisionFileError(ValueError):
    """The file cannot be scored; every offending round is named in the message."""


@dataclass
class Decisions:
    strategy: str
    weights: dict                     # (day, round) -> {ticker: Decimal}
    expected: list                    # every (day, round) in the span, in order
    missing: list = field(default_factory=list)   # rounds the file leaves out
    invalid: list = field(default_factory=list)   # rounds that break a weight rule

    def strategy_fn(self):
        # A missing round returns None, which `sim.run` holds, as the backend does.
        # Defined here rather than borrowing `rankplay._Replay`: that module pulls in the
        # strategy code, and the hosted harness must ship without it.
        return lambda ctx: self.weights.get((ctx.day, ctx.round))


def span_days(market: sim.Market, start: date, end: date) -> list[date]:
    days = [d for d in market.days if start <= d <= end]
    if not days:
        raise DecisionFileError(f"the market has no trading days in {start}..{end}")
    # A market snapshot that stops short would score a shorter run under the same name.
    if max(market.days) < end:
        raise DecisionFileError(f"the market ends {max(market.days)}, before {end}")
    return days


def round_id(day: date, round_no: int, phase: str = "holdout") -> str:
    return f"{phase}-{day.isoformat()}-r{round_no}"


def load_decisions(path, market: sim.Market, start: date = HOLDOUT_START,
                   end: date = HOLDOUT_END, strict: bool = False) -> Decisions:
    """Parse and check a decisions file against the market's calendar for the span."""
    raw = Path(path).read_text()
    doc = json.loads(raw, parse_float=Decimal, parse_constant=_reject_constant)
    if not isinstance(doc, dict) or not isinstance(doc.get("decisions"), list):
        raise DecisionFileError('expected {"strategy": ..., "decisions": [...]}')
    strategy = str(doc.get("strategy") or Path(path).stem)

    days = span_days(market, start, end)
    expected = [(d, r["round"]) for d in days for r in calendar.rounds_for(d)]
    valid = set(expected)
    tickers = set(market.tickers)

    errors, weights, invalid = [], {}, []
    for i, entry in enumerate(doc["decisions"]):
        rid = entry.get("round_id") if isinstance(entry, dict) else None
        where = f"decision {i} ({rid!r})"
        m = _ROUND_ID.match(rid) if isinstance(rid, str) else None
        if m is None:
            errors.append(f"{where}: round_id is not <phase>-YYYY-MM-DD-r<n>")
            continue
        key = (date.fromisoformat(m.group(1)), int(m.group(2)))
        if key not in valid:
            errors.append(f"{where}: no such round in {start}..{end} "
                          "(weekend, holiday, a round cancelled by an early close, "
                          "or outside the span)")
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

    if errors:
        raise DecisionFileError(f"{len(errors)} problem(s) in {path}:\n  " + "\n  ".join(errors))

    missing = [round_id(d, r) for d, r in expected if (d, r) not in weights]
    if strict and (missing or invalid):
        raise DecisionFileError(
            f"strict: {len(missing)} missing round(s), {len(invalid)} invalid round(s); "
            f"first missing {missing[:5]}, first invalid {invalid[:5]}")
    return Decisions(strategy, weights, expected, missing, invalid)


def _reject_constant(name):
    raise DecisionFileError(f"{name} is not a JSON number the backend accepts")


def continuous(decisions: Decisions, market: sim.Market, start: date = HOLDOUT_START,
               end: date = HOLDOUT_END, sizing: str = "pre_fee"):
    """One run from $1M over every trading day in the span. Returns (summary, Result)."""
    days = span_days(market, start, end)
    res = sim.run(decisions.strategy_fn(), market, days[0], len(days), sizing=sizing)
    degraded = [d for d in market.issues.get("degraded_days", [])
                if str(days[0]) <= d <= str(days[-1])]
    summary = {
        **res.metrics(),
        "first_day": str(days[0]), "last_day": str(days[-1]), "trading_days": len(days),
        "rounds": len(res.periods),
        "missing_rounds": len(decisions.missing),
        "invalid_rounds": len(res.invalid_rounds),
        # A stand-in price inside the run is a zero return that never happened; the
        # continuous run cannot skip it, so it is named instead.
        "degraded_days": degraded,
    }
    return summary, res


def market_from_frames(exec_prices: pd.DataFrame, closes: pd.DataFrame,
                       degraded_days: list[str]) -> sim.Market:
    """A replay-only Market: fills and closes, no information bars.

    A decisions file is fixed before the harness sees it, so nothing reads
    `history()`. Shipping only these two frames is what lets the hosted harness carry a
    few hundred KB of 2026 prices rather than the bar archive.
    """
    info = pd.DataFrame({"end": pd.Series([], dtype=f"datetime64[ns, {calendar.TZ}]")})
    return sim.Market(exec_prices, closes, info, issues={"degraded_days": list(degraded_days)})


def equity_curve(res: sim.Result) -> pd.DataFrame:
    times = ["initial"] + [str(t) for t in res.valuation_times[1:]]
    return pd.DataFrame({"time": times, "nav": res.valuation_points})


def rolling(decisions: Decisions, market: sim.Market, start: date = HOLDOUT_START,
            end: date = HOLDOUT_END, n_days: int = WINDOW_DAYS, sizing: str = "pre_fee"):
    """A fresh $1M in every `n_days` window starting on each trading day of the span.

    Returns (per-window DataFrame, skipped starts). A window touching a degraded day is
    skipped and named: its stand-in prices make it look calmer than the market was.

    Each window replays the same decisions as the continuous run. Weights resize off
    the window's own NAV, so the ledger is right, but an agent whose choices depend on
    its own holdings would have chosen differently starting from cash.
    """
    days = span_days(market, start, end)
    degraded = set(market.issues.get("degraded_days", []))
    missing = set(decisions.missing)
    rows, skipped = [], []
    for i in range(len(days) - n_days + 1):
        span = days[i:i + n_days]
        if degraded & {str(d) for d in span}:
            skipped.append(str(span[0]))
            continue
        res = sim.run(decisions.strategy_fn(), market, span[0], n_days, sizing=sizing)
        n_missing = sum(round_id(d, r["round"]) in missing
                        for d in span for r in calendar.rounds_for(d))
        rows.append({"window_start": str(span[0]), "window_end": str(span[-1]),
                     **res.metrics(), "missing_rounds": n_missing,
                     "invalid_rounds": len(res.invalid_rounds)})
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


def independent_windows(windows_df: pd.DataFrame) -> int:
    """How many of the windows could be picked without any two sharing a day."""
    n, last_end = 0, ""
    for s, e in zip(windows_df["window_start"], windows_df["window_end"]):
        if s > last_end:
            n, last_end = n + 1, e
    return n
