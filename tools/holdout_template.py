"""Write a decisions file naming every round of the holdout, filled with equal weights.

    .venv/bin/python tools/holdout_template.py [--out output/holdout/equal_weight_<rebalance>.json]
        [--start 2026-01-02] [--end 2026-08-31] [--rebalance every|daily]

It gives the agent side the exact round_ids to fill (half-days have only rounds 1-4),
and scores as an equal-weight sanity baseline. `--rebalance daily` writes round 1 only,
so every other round is a missing-round hold, the same as baselines.EqualWeightDaily.
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
    ap.add_argument("--rebalance", choices=["every", "daily"], default="every")
    args = ap.parse_args()

    market = markets.research_market("alpaca")
    tickers = market.tickers
    w = W.safe({t: 1 / len(tickers) for t in tickers}, tickers)
    # Cash from the same Decimal text the loader will read, so the check is exact.
    cash = Decimal(1) - sum(Decimal(repr(v)) for v in w.values())
    rows = []
    for d in holdout.span_days(market, args.start, args.end):
        for r in calendar.rounds_for(d):
            if args.rebalance == "daily" and r["round"] != 1:
                continue
            rows.append({"round_id": holdout.round_id(d, r["round"]),
                         "cash": float(cash), "weights": w})
    name = f"equal_weight_{args.rebalance}"
    out = args.out or data.ROOT / "output" / "holdout" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"strategy": name, "decisions": rows}))
    print(f"wrote {len(rows)} decisions to {out}")


if __name__ == "__main__":
    main()
