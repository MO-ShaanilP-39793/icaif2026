"""Dry-run one live round: the decision it WOULD submit, written to disk, never uploaded.

    .venv/bin/python tools/live_dry_run.py                      # round 1, all cash
    .venv/bin/python tools/live_dry_run.py --portfolio book.json --round 3

Fetches fresh Yahoo daily bars and context, scores the next session with the frozen
daily model, compiles weights at default levers and validates them with the kit.
Everything fetched and the decision land in output/live/<timestamp>/.

Nothing here touches Codabench: the decision carries the kit's placeholder credentials
and a dryrun- round id, which the kit's own client refuses before any upload.
The clock starts before the heavy imports, because a live round pays for them too.
"""

import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icaif import compiler, live  # noqa: E402

BUDGET_SECONDS = 120


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--portfolio", type=Path, help='JSON {"weights": {ticker: w}}; default all cash')
    ap.add_argument("--as-of", default=None, help="replay as of this ET time (default: now)")
    ap.add_argument("--out", type=Path, default=None, help="archive dir (default output/live/<ts>)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    imports = time.perf_counter() - T0

    portfolio = json.loads(args.portfolio.read_text()) if args.portfolio else None
    levers = compiler.Levers()
    decision, log = live.decide(args.round, portfolio, levers, as_of=args.as_of, out_dir=args.out)
    total = time.perf_counter() - T0

    print(f"DRY RUN  {decision['round_id']}  decision date {log['decision_date']}  "
          f"(latest close {log['latest_session']}, deadline {log['deadline']})")
    print(f"universe {log['universe_size']} names; earnings: {log['inputs']['earnings']['source']}"
          f"{' (STALE)' if log['inputs']['earnings'].get('stale') else ''}; "
          f"action: {log['action']}; gross {log['gross']:.4f}")
    print(f"\nlevers: {log['levers']}")

    scores, vol, w = log["scores"], log["vol"], log["weights"]
    ranked = sorted(scores, key=lambda t: (scores[t] is None, -(scores[t] or 0)))
    print(f"\n{'ticker':<7}{'score':>9}{'vol20':>9}{'weight':>10}")
    # The book is chosen on score percentile / vol**gamma, so a low-vol name can be held
    # from below the top by raw score; list it too rather than hide half the book.
    shown = ranked[:args.top] + [t for t in ranked[args.top:] if w[t] > 0]
    for t in shown:
        s = "nan" if scores[t] is None else f"{scores[t]:.4f}"
        v = "nan" if vol[t] is None else f"{vol[t]:.4f}"
        print(f"{t:<7}{s:>9}{v:>9}{w[t]:>10.6f}")
    held = {t: x for t, x in w.items() if x > 0}
    print(f"\nweights ({len(held)} names): " + ", ".join(f"{t} {x:.4f}" for t, x in
                                                   sorted(held.items(), key=lambda kv: -kv[1])))

    print("\ntimings (s): imports " + f"{imports:.1f}, " +
          ", ".join(f"{k} {v:.2f}" for k, v in log["timings"].items()))
    print(f"end to end {total:.1f}s (budget ~{BUDGET_SECONDS}s)"
          + ("  OVER BUDGET" if total > BUDGET_SECONDS else ""))
    for msg in log["warnings"]:
        print(f"WARNING: {msg}")
    missing = log["inputs"]["current_members_unpriced"]
    if missing:
        print(f"note: {len(missing)} current S&P members unpriced on Yahoo: {missing}")
    print(f"\ndecision: {log['decision_path']}")


if __name__ == "__main__":
    main()
