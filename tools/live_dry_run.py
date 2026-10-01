"""Dry-run one live round: the rule desk's book it WOULD submit, written to disk, never uploaded.

    .venv/bin/python tools/live_dry_run.py                     # the next round by the ET clock, from cash
    .venv/bin/python tools/live_dry_run.py --round 1           # round 1 of the next session
    .venv/bin/python tools/live_dry_run.py --portfolio saved.json   # a book in the server's format
    .venv/bin/python tools/live_dry_run.py --as-of "2026-09-21 08:58"   # replay a past wake

One round through `runner.run_round`, in a scratch phase directory: fresh Yahoo closes,
the backtested rule desk (risk parity at the regime-blended exposure at round 1 from
cash, then hold), the hold guard and the kit's validator. The shadow agent runs on the
rule brain unless `--shadow claude`, which spends. Nothing here touches Codabench: the
file carries the kit's placeholder credentials and a dryrun- round id.

A round whose deadline has passed is replayed at its wake time, on the data a live
round would have had then. The clock starts before the heavy imports, because a live
round pays for them too.
"""

import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import calendar, data, live, runner  # noqa: E402
from icaif import portfolio as P  # noqa: E402

BUDGET_SECONDS = 120


def pick_round(now: pd.Timestamp, number):
    """The round to dry-run: `number` of the session being decided, else the next one."""
    latest = live.latest_completed_session(now)
    day = live.next_session(latest).date()
    rounds = calendar.rounds_for(day)
    if number is None:
        ahead = [r for r in rounds if r["deadline"] > now]
        if not ahead:
            day = live.next_session(pd.Timestamp(day)).date()
            ahead = calendar.rounds_for(day)
        number = ahead[0]["round"]
    if number not in {r["round"] for r in calendar.rounds_for(day)}:
        raise SystemExit(f"round {number} does not run on {day}")
    return day, number


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--portfolio", type=Path,
                    help="a portfolio response in the server's format (default: all cash)")
    ap.add_argument("--as-of", default=None, help="replay as of this ET time (default: now)")
    ap.add_argument("--out", type=Path, default=None, help="phase dir (default output/live/dryrun-<ts>)")
    ap.add_argument("--shadow", choices=["claude", "rule", "none"], default="rule")
    ap.add_argument("--no-scores", action="store_true")
    args = ap.parse_args()
    imports = time.perf_counter() - T0

    now = runner.et(args.as_of) if args.as_of else pd.Timestamp.now(tz=calendar.TZ)
    day, number = pick_round(now, args.round)
    phase = f"dryrun-{pd.Timestamp.now(tz=calendar.TZ):%Y%m%dT%H%M%S}"
    sched = runner.rehearsal_schedule([day], phase)
    row = next(r for r in sched["rounds"] if r["number"] == number)
    wake = runner.et(row["deadline"]) - pd.Timedelta(minutes=12)
    if now >= runner.et(row["deadline"]):
        print(f"{row['id']}'s deadline has passed; replaying it at its wake, {wake:%Y-%m-%d %H:%M %Z}")
        now = wake
    start, real0 = now, pd.Timestamp.now(tz=calendar.TZ)
    clock = lambda: start + (pd.Timestamp.now(tz=calendar.TZ) - real0)  # noqa: E731

    cfg = runner.Config(phase=phase, shadow=args.shadow, out=args.out, scoring=not args.no_scores,
                        window_days=15)
    book = None
    if args.portfolio:
        parsed = P.parse(json.loads(args.portfolio.read_text()))
        book = lambda: parsed  # noqa: E731
    rec = runner.run_round(cfg, row, runner.Doors(now=clock, book=book))
    total = time.perf_counter() - T0

    print(f"DRY RUN  {row['id']}  (decision date {day}, deadline {runner.et(row['deadline']):%H:%M %Z})")
    print(f"book: {rec.get('book')}")
    print(f"entered: {rec.get('entered')}")
    rule = rec.get("rule") or {}
    for role in rule.get("roles", []):
        print(f"rule desk, {role['role']}: {role['decision']} ({role['source']})")
    if rule.get("p_turbulent_next") is not None:
        print(f"p(turbulent next session) {rule['p_turbulent_next']:.4f}")
    if rule.get("action") == "trade":
        w = rule["weights"]
        print(f"\nrule's book: {rule['names']} names, gross {rule['gross']:.4f}")
        for t, x in sorted(w.items(), key=lambda kv: -kv[1]):
            print(f"  {t:<6}{x:>10.6f}   {data.load_universe()[t]}")
    sub = rec.get("submitted") or {}
    print(f"\nsubmitted: {sub.get('action')} ({sub.get('source')}): {sub.get('guard')}")
    print(f"upload: {(sub.get('upload') or {}).get('status')}; file {sub.get('file')}")
    shadow = rec.get("shadow") or {}
    print(f"shadow ({shadow.get('brain', args.shadow)}): {shadow.get('action')}"
          + (f", gross {shadow['gross']:.4f}" if shadow.get("gross") is not None else ""))
    print("\ntimings (s): imports " + f"{imports:.1f}, " +
          ", ".join(f"{k} {v:.2f}" for k, v in rec["timings"].items()))
    print(f"end to end {total:.1f}s (budget ~{BUDGET_SECONDS}s)"
          + ("  OVER BUDGET" if total > BUDGET_SECONDS else ""))
    print(f"ready {rec.get('ready_before_deadline_s')}s before the deadline")
    for e in rec.get("errors", []):
        print(f"ERROR: {e}")
    for w in rec.get("warnings", []):
        print(f"WARNING: {w}")
    print(f"\nrecord: {cfg.out / row['id'] / 'round.json'}")


if __name__ == "__main__":
    main()
