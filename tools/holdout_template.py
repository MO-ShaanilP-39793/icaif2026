"""Write a decisions file naming every window and round of the holdout, at equal weights.

    .venv/bin/python tools/holdout_template.py [--out output/holdout/equal_weight_<rebalance>.json]
        [--start 2026-01-02] [--end 2026-06-30] [--rebalance once|daily|every]

It gives the agent side the exact window keys and round_ids to fill (half-days have only
rounds 1-4). Each window is its own run from cash, so:
- `once` (default) buys 1/30 each at the window's first round and holds: the board's
  ew_hold reference, and it scores identically to it;
- `daily` writes round 1 of each day only, so every other round is a missing-round hold,
  the same as baselines.EqualWeightDaily;
- `every` rebalances to 1/30 at every round.
"""

import argparse
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icaif import calendar, data, holdout, markets  # noqa: E402
from icaif import weights as W  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--start", type=date.fromisoformat, default=holdout.HOLDOUT_START)
    ap.add_argument("--end", type=date.fromisoformat, default=holdout.HOLDOUT_END)
    ap.add_argument("--rebalance", choices=["once", "daily", "every"], default="once")
    args = ap.parse_args()

    market = markets.research_market("alpaca")
    tickers = market.tickers
    w = W.safe({t: 1 / len(tickers) for t in tickers}, tickers)
    # Cash from the same Decimal text the loader will read, so the check is exact.
    cash = float(Decimal(1) - sum(Decimal(repr(v)) for v in w.values()))
    spans, skipped = holdout.window_spans(market, args.start, args.end)
    windows = {}
    for ws, span in spans.items():
        rounds = [(d, r["round"]) for d in span for r in calendar.rounds_for(d)]
        if args.rebalance == "once":
            rounds = rounds[:1]
        elif args.rebalance == "daily":
            rounds = [(d, r) for d, r in rounds if r == 1]
        windows[str(ws)] = [{"round_id": holdout.round_id(d, r), "cash": cash, "weights": w}
                            for d, r in rounds]
    name = f"equal_weight_{args.rebalance}"
    out = args.out or data.ROOT / "output" / "holdout" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"strategy": name, "windows": windows}))
    n = sum(len(v) for v in windows.values())
    print(f"wrote {len(windows)} windows, {n} decisions to {out}"
          + (f"; skipped degraded windows {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
