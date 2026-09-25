"""Rank the baseline field in every non-overlapping 15-day window.

    .venv/bin/python tools/baselines_report.py [--stride 15]

Writes reports/baselines_summary.csv (per strategy) and reports/baselines_windows.csv
(per window x strategy), and prints the summary.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, data, markets, windows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=windows.WINDOW_DAYS)
    ap.add_argument("--exposure-scan", action="store_true",
                    help="also rank inv_vol_hold at 25/50/75/100%% gross against the field")
    args = ap.parse_args()

    t0 = time.time()
    market = markets.research_market()
    starts = windows.window_starts(market, stride=args.stride)
    results = windows.run_field(baselines.FIELD, market, starts)
    summary = windows.summarise(results)

    out = data.ROOT / "reports"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "baselines_windows.csv", index=False)
    summary.to_csv(out / "baselines_summary.csv")
    print(json.dumps({"windows": len(starts), "first": str(starts[0]), "last": str(starts[-1]),
                      "market_issues": {k: v for k, v in market.issues.items()
                                        if k != "dropped_leading_days"},
                      "seconds": round(time.time() - t0, 1)}, indent=1))
    with_pct = summary.copy()
    for c in ("mean_return", "mean_mdd", "mean_turnover"):
        with_pct[c] = (with_pct[c] * 100).round(2)
    print(with_pct.round(3).to_string())

    if args.exposure_scan:
        field = {k: v for k, v in baselines.FIELD.items() if k != "inv_vol_hold"}
        rows = {}
        for g in (0.25, 0.5, 0.75, 1.0):
            f = {**field, "candidate": baselines.scaled(baselines.InverseVolHold, g)}
            rows[g] = windows.summarise(windows.run_field(f, market, starts)).loc["candidate"]
        scan = pd.DataFrame(rows).T
        scan.index.name = "gross"
        scan.to_csv(out / "exposure_scan.csv")
        print("\ninv_vol_hold by gross exposure, ranked against the rest of the field")
        print(scan.round(4).to_string())


if __name__ == "__main__":
    main()
