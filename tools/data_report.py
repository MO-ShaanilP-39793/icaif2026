"""Write reports/data_parity.json: organizer panel vs public feed, and fill-guess error.

Rerun after any refresh of either source. The numbers are what every later backtest
quietly depends on, so they are kept as a file rather than remembered.

    .venv/bin/python tools/data_report.py [--refresh]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icaif import data, parity, public_bars  # noqa: E402

REPORT = data.ROOT / "reports" / "data_parity.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch today's public bars")
    args = ap.parse_args()

    organizer, issues = data.load_organizer_bars()
    tickers = sorted(data.load_universe())
    p60 = public_bars.cached(tickers, "60m", "730d", refresh=args.refresh)
    p30 = public_bars.cached(tickers, "30m", "60d", refresh=args.refresh)

    report = {
        "organizer_issues": {
            "summary": issues.summary(),
            "short_sessions": issues.short_sessions,
            "spin_offs_adjusted": list(data.CORPORATE_ACTIONS),
        },
        "source_parity_vs_yahoo_60m": parity.source_parity(organizer, p60),
        "fill_guess_error_vs_true_half_hour_open": parity.fill_approximation(p30),
        "yahoo_60m_vs_30m_open_at_half_hour": parity.feed_self_consistency(p60, p30),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {REPORT}")
    print(issues.summary())
    sp = report["source_parity_vs_yahoo_60m"]
    for k in ("open_0930", "session_close"):
        print(k, sp[k])
    for k, v in report["fill_guess_error_vs_true_half_hour_open"].items():
        print("fill", k, v)


if __name__ == "__main__":
    main()
