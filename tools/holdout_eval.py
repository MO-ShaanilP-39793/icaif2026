"""Score an agent's decisions file on the Jan-Aug 2026 holdout.

    .venv/bin/python tools/holdout_eval.py --decisions path/to/decisions.json
        [--start 2026-01-02] [--end 2026-08-31] [--strict] [--sizing pre_fee|post_fee]
        [--out output/holdout/<strategy>/<ts>/]

Two views, both on Alpaca :30 fills (markets.research_market):
- one continuous run from $1M over the span: continuous.json, equity.csv;
- a fresh $1M in every 15-day window starting on each trading day: windows.csv,
  rolling_summary.csv.

The file format and which errors reject it are in icaif/holdout.py. Each window replays
the same decisions from cash, so an agent that decides from its own holdings is only
approximately scored per window.
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icaif import data, holdout, markets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--start", type=date.fromisoformat, default=holdout.HOLDOUT_START)
    ap.add_argument("--end", type=date.fromisoformat, default=holdout.HOLDOUT_END)
    ap.add_argument("--strict", action="store_true",
                    help="make missing and invalid rounds fatal instead of holds")
    ap.add_argument("--sizing", choices=["pre_fee", "post_fee"], default="pre_fee")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    market = markets.research_market("alpaca")
    try:
        dec = holdout.load_decisions(args.decisions, market, args.start, args.end, args.strict)
    except holdout.DecisionFileError as err:
        sys.exit(f"rejected: {err}")

    summary, res = holdout.continuous(dec, market, args.start, args.end, args.sizing)
    wins, skipped = holdout.rolling(dec, market, args.start, args.end, sizing=args.sizing)
    roll = holdout.summarise_rolling(wins)

    out = args.out or (data.ROOT / "output" / "holdout" / dec.strategy
                       / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    report = {"strategy": dec.strategy, "decisions_file": str(args.decisions),
              "sizing": args.sizing, "fills": "alpaca", **summary,
              "missing": dec.missing, "invalid": dec.invalid,
              "windows": roll.attrs["windows"],
              "independent_windows": roll.attrs["independent_windows"],
              "skipped_window_starts": skipped}
    (out / "continuous.json").write_text(json.dumps(report, indent=1))
    holdout.equity_curve(res).to_csv(out / "equity.csv", index=False)
    wins.to_csv(out / "windows.csv", index=False)
    roll.to_csv(out / "rolling_summary.csv")

    print(f"{dec.strategy}: {summary['first_day']}..{summary['last_day']}, "
          f"{summary['trading_days']} days, {summary['rounds']} rounds")
    if dec.missing or dec.invalid:
        print(f"  HELD: {len(dec.missing)} missing round(s), {len(dec.invalid)} invalid "
              f"round(s); listed in continuous.json")
    if summary["degraded_days"]:
        print(f"  stand-in prices on {summary['degraded_days']} (inside the continuous run)")
    print("\ncontinuous run")
    for k in holdout.METRICS:
        print(f"  {k:<18} {summary[k]: .4f}")
    print(f"\n{roll.attrs['windows']} rolling 15-day windows "
          f"(~{roll.attrs['independent_windows']} independent; "
          f"{len(skipped)} skipped for degraded days)")
    print(roll.round(4).to_string())
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
