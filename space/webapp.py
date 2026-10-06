"""The page's only Python entry point: load the shipped prices, score one file.

The browser (Pyodide) and the build's native check call the same `score`, so a number
the page shows is a number the build already compared against `tools/holdout_eval.py`.
Everything scored goes through `icaif.holdout`; this file only loads prices and
serialises results.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from icaif import calendar, holdout, leaderboard, suites

ROOT = Path(__file__).resolve().parent


def _frame(doc: dict) -> pd.DataFrame:
    # Index as UTC nanoseconds, not a local-time string: a string with -05:00 and -04:00
    # offsets parses to object dtype, and no execution timestamp would match a row.
    # json writes floats as repr and reads them back exactly, so the page fills at the
    # build's prices bit for bit (the build checks this).
    idx = pd.to_datetime(doc["index"], unit="ns", utc=True).tz_convert(calendar.TZ)
    return pd.DataFrame(doc["data"], index=idx, columns=doc["columns"], dtype=float)


def frame_doc(df: pd.DataFrame) -> dict:
    return {"index": pd.DatetimeIndex(df.index).asi8.tolist(),
            "columns": list(df.columns), "data": df.to_numpy(dtype=float).tolist()}


# Prices ship as JSON, not CSV: the office network's download policy blocks .csv from
# the Space, which left the page stuck at "HTTP 403 fetching data/exec_prices.csv".
def load_market(root: Path = ROOT):
    meta = json.loads((root / "data" / "market.json").read_text())
    prices = json.loads((root / "data" / "prices.json").read_text())
    market = holdout.market_from_frames(_frame(prices["exec_prices"]),
                                        _frame(prices["closes"]), meta["degraded_days"])
    return market, meta


_MARKET = None


def score(path: str, start: str, end: str, strict: bool, sizing: str,
          suite: str = suites.DEFAULT) -> str:
    """JSON: {"rejected": msg} or the notes, the tables and CSV texts for download.

    `start`/`end` narrow a rolling suite's span; blank, or a fixed suite, takes the
    suite's own. A narrowed run is scored but not offered for submission.
    """
    global _MARKET
    if _MARKET is None:
        _MARKET = load_market()
    market, meta = _MARKET
    try:
        chosen = suites.get(suite)
        span = (start.strip() or chosen.span[0], end.strip() or chosen.span[1])
        if span != tuple(chosen.span):
            chosen = chosen.narrowed(*(str(date.fromisoformat(d)) for d in span))
        start_d, end_d = (date.fromisoformat(d) for d in chosen.span)
        dec = holdout.load_decisions(path, market, chosen, strict=strict)
    except (holdout.DecisionFileError, ValueError, KeyError) as err:
        # A rejected file shows the reason and no numbers: a partial score of a file
        # the harness disagrees with would read as a real result.
        return json.dumps({"rejected": str(err)})

    wins, skipped = holdout.rolling(dec, market, sizing=sizing)
    roll = holdout.summarise_rolling(wins)
    days = holdout.suite_days(market, chosen)
    report = {"strategy": dec.strategy, "suite": dec.suite, "sizing": sizing, "fills": "alpaca",
              "market_snapshot": meta["snapshot"],
              "first_day": str(days[0]), "last_day": str(days[-1]), "trading_days": len(days),
              "rounds": sum(len(w) for w in dec.windows.values()),
              "missing": dec.missing, "invalid": dec.invalid, "windows": roll.attrs["windows"],
              "independent_windows": roll.attrs["independent_windows"],
              "skipped_window_starts": skipped}
    # The entry the page would submit. Only a whole suite at the board's sizing can rank,
    # so any other run is scored but not offered for submission.
    on_board = suites.is_canonical(chosen) and sizing == leaderboard.BOARD_SIZING
    entry = leaderboard.make_entry(
        dec.strategy, leaderboard.SUBMITTED, wins, span=(start_d, end_d),
        sizing=sizing, market_snapshot=meta["snapshot"],
        decisions_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        suite=dec.suite)
    return json.dumps({
        "entry": entry if on_board else None,
        "report": report,
        "rolling": roll.reset_index(names="metric").to_dict(orient="records"),
        "windows": wins.to_dict(orient="records"),
        "files": {
            "report.json": json.dumps(report, indent=1),
            "windows.csv": wins.to_csv(index=False),
            "rolling_summary.csv": roll.to_csv(),
        },
    })


def boot() -> None:
    """Parse the shipped prices once, when the page starts, not on the first score."""
    global _MARKET
    _MARKET = load_market()
