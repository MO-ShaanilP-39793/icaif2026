"""The page's only Python entry point: load the shipped prices, score one file.

The browser (Pyodide) and the build's native check call the same `score`, so a number
the page shows is a number the build already compared against `tools/holdout_eval.py`.
Everything scored goes through `icaif.holdout`; this file only loads prices and
serialises results.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from icaif import calendar, holdout

ROOT = Path(__file__).resolve().parent


def _frame(path: Path) -> pd.DataFrame:
    # Index as UTC nanoseconds, not a local-time string: a string with -05:00 and -04:00
    # offsets parses to object dtype, and no execution timestamp would match a row.
    # round_trip keeps every float bit-exact, so the page fills at the build's prices.
    df = pd.read_csv(path, index_col=0, float_precision="round_trip")
    df.index = pd.to_datetime(df.index, unit="ns", utc=True).tz_convert(calendar.TZ)
    return df


def write_frame(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).asi8
    out.to_csv(path)


def load_market(root: Path = ROOT):
    meta = json.loads((root / "data" / "market.json").read_text())
    market = holdout.market_from_frames(_frame(root / "data" / "exec_prices.csv"),
                                        _frame(root / "data" / "closes.csv"),
                                        meta["degraded_days"])
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
    return json.dumps({
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
