"""Score an agent's decisions file on a suite's windows (icaif/suites.py).

    .venv/bin/python tools/holdout_eval.py --decisions path/to/decisions.json
        [--suite holdout|official4] [--start 2026-01-02] [--end 2026-06-30]
        [--strict] [--sizing pre_fee|post_fee]
        [--out output/holdout/<strategy>/<ts>/]
        [--submit [--name NAME] [--author WHO] [--note TEXT]]

On Alpaca :30 fills (markets.research_market), a fresh $1M in every window of the suite,
each running that window's own decisions: report.json, windows.csv, rolling_summary.csv.
The suite defaults to the holdout (every 15-day window starting on a trading day of
Jan 2 - Jun 30 2026); --start/--end narrow a rolling suite's span, and a narrowed run is
never submitted. The file must name the suite it is scored as.

--submit also posts the result (metrics and window table, never the decisions file) to
the public leaderboard, recorded first in the private entry dataset (icaif/space_hub.py).
The entry names its suite and ranks on that suite's board only. It accepts only a whole
suite at the board's sizing, since an entry scored on other windows cannot be ranked.

The file format (one decision sequence per window) and which errors reject it are in
icaif/holdout.py; tools/holdout_template.py writes an example.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icaif import data, holdout, leaderboard, markets, space_hub, suites  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--suite", choices=sorted(suites.SUITES), default=suites.DEFAULT)
    ap.add_argument("--start", help="narrow a rolling suite's span (not submittable)")
    ap.add_argument("--end", help="narrow a rolling suite's span (not submittable)")
    ap.add_argument("--strict", action="store_true",
                    help="make missing and invalid rounds fatal instead of holds")
    ap.add_argument("--sizing", choices=["pre_fee", "post_fee"], default="pre_fee")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--submit", action="store_true", help="post the result to the HF leaderboard")
    ap.add_argument("--name", help="leaderboard name (default: the file's strategy)")
    ap.add_argument("--author", help="default: your HF username")
    ap.add_argument("--note", default="", help="one line: what this version changes")
    args = ap.parse_args()
    suite = suites.get(args.suite)
    if args.start or args.end:
        try:
            suite = suite.narrowed(args.start or suite.span[0], args.end or suite.span[1])
        except ValueError as err:
            sys.exit(str(err))
    if args.submit and not (suites.is_canonical(suite) and args.sizing == leaderboard.BOARD_SIZING):
        sys.exit(f"--submit scores only a whole suite at {leaderboard.BOARD_SIZING} sizing, "
                 "so every entry is ranked on the same windows")

    t0 = time.time()
    market = markets.research_market("alpaca")
    try:
        dec = holdout.load_decisions(args.decisions, market, suite, args.strict)
    except holdout.DecisionFileError as err:
        sys.exit(f"rejected: {err}")

    wins, skipped = holdout.rolling(dec, market, sizing=args.sizing)
    roll = holdout.summarise_rolling(wins)
    days = holdout.suite_days(market, suite)

    out = args.out or (data.ROOT / "output" / "holdout" / dec.strategy
                       / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    report = {"strategy": dec.strategy, "suite": dec.suite, "decisions_file": str(args.decisions),
              "sizing": args.sizing, "fills": "alpaca",
              "first_day": str(days[0]), "last_day": str(days[-1]), "trading_days": len(days),
              "missing": dec.missing, "invalid": dec.invalid,
              "windows": roll.attrs["windows"],
              "independent_windows": roll.attrs["independent_windows"],
              "skipped_window_starts": skipped}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    wins.to_csv(out / "windows.csv", index=False)
    roll.to_csv(out / "rolling_summary.csv")

    print(f"{dec.strategy} ({dec.suite}): {days[0]}..{days[-1]}, {len(days)} days, "
          f"{sum(len(w) for w in dec.windows.values())} decisions")
    if dec.missing or dec.invalid:
        print(f"  HELD: {len(dec.missing)} missing round(s), {len(dec.invalid)} invalid "
              f"round(s) across all windows; listed in report.json")
    print(f"\n{roll.attrs['windows']} 15-day windows, each its own run from cash "
          f"(~{roll.attrs['independent_windows']} independent; "
          f"{len(skipped)} skipped for degraded days)")
    print(roll.round(4).to_string())
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s)")

    if args.submit:
        snapshot = sorted((data.ROOT / "data" / "public").glob("alpaca_30m_2*.parquet"))[-1].name
        entry = leaderboard.make_entry(
            args.name or dec.strategy, leaderboard.SUBMITTED, wins,
            span=tuple(suite.span), sizing=args.sizing, market_snapshot=snapshot,
            author=args.author or space_hub.whoami(), note=args.note,
            decisions_sha256=hashlib.sha256(args.decisions.read_bytes()).hexdigest(),
            suite=suite.name)
        path = space_hub.submit(entry)
        print(f"submitted {entry['strategy']} as {path}; "
              f"https://huggingface.co/spaces/{space_hub.BOARD_ID}")


if __name__ == "__main__":
    main()
