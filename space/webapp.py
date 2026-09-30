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

from icaif import calendar, holdout, leaderboard

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


def score(path: str, start: str, end: str, strict: bool, sizing: str) -> str:
    """JSON: {"rejected": msg} or the notes, the tables and CSV texts for download."""
    global _MARKET
    if _MARKET is None:
        _MARKET = load_market()
    market, meta = _MARKET
    try:
        start_d, end_d = date.fromisoformat(start.strip()), date.fromisoformat(end.strip())
        dec = holdout.load_decisions(path, market, start_d, end_d, strict=strict)
    except (holdout.DecisionFileError, ValueError) as err:
        # A rejected file shows the reason and no numbers: a partial score of a file
        # the harness disagrees with would read as a real result.
        return json.dumps({"rejected": str(err)})

    summary, res = holdout.continuous(dec, market, start_d, end_d, sizing)
    wins, skipped = holdout.rolling(dec, market, start_d, end_d, sizing=sizing)
    roll = holdout.summarise_rolling(wins)
    report = {**summary, "strategy": dec.strategy, "sizing": sizing, "fills": "alpaca",
              "market_snapshot": meta["snapshot"], "missing": dec.missing,
              "invalid": dec.invalid, "windows": roll.attrs["windows"],
              "independent_windows": roll.attrs["independent_windows"],
              "skipped_window_starts": skipped}
    # The entry the page would submit. Only the board's own span and sizing can rank,
    # so any other run is scored but not offered for submission.
    on_board = (start_d, end_d, sizing) == (holdout.HOLDOUT_START, holdout.HOLDOUT_END,
                                            leaderboard.BOARD_SIZING)
    entry = leaderboard.make_entry(
        dec.strategy, leaderboard.SUBMITTED, summary, wins, span=(start_d, end_d),
        sizing=sizing, market_snapshot=meta["snapshot"],
        decisions_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return json.dumps({
        "entry": entry if on_board else None,
        "report": report,
        "continuous": {k: summary[k] for k in holdout.METRICS},
        "rolling": roll.reset_index(names="metric").to_dict(orient="records"),
        "windows": wins.to_dict(orient="records"),
        "files": {
            "continuous.json": json.dumps(report, indent=1),
            "equity.csv": holdout.equity_curve(res).to_csv(index=False),
            "windows.csv": wins.to_csv(index=False),
            "rolling_summary.csv": roll.to_csv(),
        },
    })


def board(texts: list[str]) -> str:
    """The leaderboard from entry JSON texts (references and submissions), as JSON.

    Ranked here rather than stored ranked: a rank depends on the whole field, so a
    stored one would go stale the moment another entry arrived.
    """
    try:
        return json.dumps(leaderboard.standings([json.loads(t) for t in texts]))
    except leaderboard.EntryError as err:
        return json.dumps({"error": str(err)})
