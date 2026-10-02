"""The live runner: a phase's rounds on the clock, one round, a rehearsal, the owner's switch.

    .venv/bin/python tools/live_runner.py rehearse                      # today's 7 rounds, real time, dry
    .venv/bin/python tools/live_runner.py rehearse --date 2026-09-30 --fast   # a past day, back to back
    AWS_PROFILE=dev .venv/bin/python tools/live_runner.py rehearse --date 2026-09-30 --fast --shadow gemma
    .venv/bin/python tools/live_runner.py run --phase validation        # dry, on the kit's bundled schedule
    .venv/bin/python tools/live_runner.py run --phase validation --live # the server's schedule and book
    .venv/bin/python tools/live_runner.py status --phase validation
    .venv/bin/python tools/live_runner.py journal --phase validation    # each desk's memory, as its roles read it
    .venv/bin/python tools/live_runner.py portfolio --phase validation  # read-only: does the book parse?
    .venv/bin/python tools/live_runner.py arm --phase validation        # the owner's approval to upload
    .venv/bin/python tools/live_runner.py disarm

**Nothing uploads unless the owner armed it.** `--live` reads the server's schedule and
portfolio (GETs only); an upload also needs starter-kit/.icaif/ARMED.json naming this
phase and submit mode, unexpired, and only `arm` writes it: at a terminal, after the
server's portfolio has parsed, with the phase name typed back. Without `--live` the
decision files carry the kit's placeholder credentials and a `dryrun-` round id, which
the kit refuses before any upload.

Each round runs in its own process under the watchdog (`run`), so a hang in one round
(the LightGBM/torch deadlock, a stuck fetch) is killed and the next round still runs.
Everything a round saw and decided lands in output/live/<phase>/<round_id>/, one line
per round in rounds.jsonl, each desk's journal in journal/<desk>.json, and the
scheduler's log in runner.log. A runner killed between rounds or mid-write resumes
where the last commit left it: start the same command again.

Keep the machine awake and online from 08:45 ET to 15:30 ET (18:15-01:00 IST in
October). On macOS the runner holds a caffeinate assertion while it runs; a closed lid
on battery still sleeps.
"""

import argparse
import json
import os
import socket
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import calendar, live, runner  # noqa: E402
from icaif import portfolio as P  # noqa: E402
from icaif.agents import brains  # noqa: E402
from icaif.agents.journal import Journal  # noqa: E402


def _cfg(args, phase=None) -> runner.Config:
    return runner.Config(phase=phase or args.phase, submit=args.submit, shadow=args.shadow,
                         live=getattr(args, "live", False),
                         out=Path(args.out) if getattr(args, "out", None) else None,
                         scoring=not args.no_scores,
                         gemma_model=getattr(args, "gemma_model", brains.GEMMA_DEFAULT))


def _common(p):
    p.add_argument("--submit", choices=["rule", "agent"], default="rule",
                   help="whose book is submitted (default rule; the agent shadows)")
    p.add_argument("--shadow", choices=["claude", "gemma", "rule", "none"], default="claude",
                   help="the agent desk's brain (claude: Opus 5 via the API; gemma: Gemma 3 on "
                        "Bedrock; both spend, capped per phase)")
    p.add_argument("--gemma-model", choices=brains.GEMMA_MODELS, default=brains.GEMMA_DEFAULT,
                   help="the Gemma 3 model when --shadow gemma")
    p.add_argument("--no-scores", action="store_true", help="skip the daily model (shadow only)")
    p.add_argument("--out", default=None, help="phase directory (default output/live/<phase>)")


def cmd_round(args) -> int:
    """One round in this process: the scheduler's worker, or a manual run of a round."""
    cfg = _cfg(args)
    snap = json.loads(Path(args.schedule).read_text()) if args.schedule else (
        runner.bundled_schedule() if args.phase in runner.PHASES else None)
    if snap is None:
        raise SystemExit("--schedule is required for a phase the kit's schedule does not list")
    rows = {r["id"]: r for r in snap["rounds"]}
    if args.round_id not in rows:
        raise SystemExit(f"{args.round_id} is not in the schedule")
    cfg.window_days = snap.get("window_days") or len({r["day"] for r in runner.phase_rounds(snap, cfg.phase)})
    offset = pd.Timedelta(seconds=float(snap.get("clock_offset_s", 0.0)))
    if args.as_of:
        start, real0 = runner.et(args.as_of), pd.Timestamp.now(tz=calendar.TZ)
        now = lambda: start + (pd.Timestamp.now(tz=calendar.TZ) - real0)  # noqa: E731
    else:
        now = lambda: pd.Timestamp.now(tz=calendar.TZ) + offset  # noqa: E731
    doors = runner.Doors(now=now, session=runner.kit_session if cfg.live else None)
    rec = runner.run_round(cfg, rows[args.round_id], doors)
    print(json.dumps({k: rec.get(k) for k in ("round_id", "outcome", "ready_before_deadline_s",
                                              "errors", "warnings")}, indent=1))
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args)
    if cfg.live:
        read = runner.server_schedule
    else:
        sched = runner.bundled_schedule()
        read = lambda: (sched, 0.0)  # noqa: E731
    runner.say(f"{cfg.phase}: {'LIVE (uploads only if armed)' if cfg.live else 'dry run'}, "
               f"submit {cfg.submit}, shadow {cfg.shadow}, on {socket.gethostname()}", cfg.out)
    runner.run_phase(cfg, read)
    return 0


def cmd_rehearse(args) -> int:
    """A dry phase on the standard calendar: today's rounds in real time by default."""
    first = date.fromisoformat(args.date) if args.date else pd.Timestamp.now(tz=calendar.TZ).date()
    days = [d.date() for d in live.sessions(first, first + pd.Timedelta(days=10))][: args.days]
    if not days or days[0] != first:
        raise SystemExit(f"{first} is not a session")
    phase = f"rehearsal-{first}" + (f"-{args.days}d" if args.days > 1 else "") + ("-fast" if args.fast else "")
    sched = runner.rehearsal_schedule(days, phase)
    last_deadline = runner.et(sched["rounds"][-1]["deadline"])
    if args.fast and last_deadline > pd.Timestamp.now(tz=calendar.TZ):
        raise SystemExit("--fast replays past days only: a future round's data does not exist yet")
    cfg = _cfg(args, phase)
    runner.say(f"{phase}: {len(sched['rounds'])} rounds on {[str(d) for d in days]}, "
               f"{'fast replay' if args.fast else 'real time'}, shadow {cfg.shadow}", cfg.out)
    done = runner.run_phase(cfg, lambda: (sched, 0.0), fast=args.fast)
    _report(cfg.out)
    return 0 if all(not str(d.get("outcome", "")).startswith(("missed", "no submission: the round"))
                    for d in done) else 1


def cmd_score(args) -> int:
    """The watchdog's child: score the daily model and archive what it saw."""
    scored = live.daily_scores(runner.et(args.as_of))
    meta = live.archive_scores(scored, Path(args.out))
    print(json.dumps({"decision_date": meta["decision_date"], "warnings": meta["warnings"]}))
    return 0


def cmd_portfolio(args) -> int:
    """Read-only: the server's book for a phase, its shape, and whether `parse` reads it."""
    with runner.kit_session(readonly=True) as s:
        raw = s.portfolio(args.phase)
    out = runner.LIVE_OUT / args.phase
    stamp = pd.Timestamp.now(tz=calendar.TZ)
    runner.write_private(out / f"portfolio_raw_{stamp:%Y%m%dT%H%M%S}.json",
                         json.dumps(raw, default=str, indent=1))
    print("shape (values blanked):")
    print(json.dumps(P.keys_only(raw), indent=1))
    try:
        book = P.parse(raw)
    except P.PortfolioFormatError as err:
        print(f"\nDOES NOT PARSE: {err}\nFix portfolio.parse to this shape before arming.")
        return 1
    print(f"\nparses: cash {book.cash:,.2f}, {len(book.summary()['names_held'])} names held, "
          f"nav {book.reported_nav}, as of {book.as_of}")
    return 0


def cmd_arm(args) -> int:
    """The owner's approval to upload, for one phase and submit mode, until it ends."""
    if not sys.stdin.isatty():
        raise SystemExit("arm needs a terminal: approval is typed by the owner, not piped")
    if not (runner.KIT_STATE / "credentials.json").exists():
        raise SystemExit("no team credentials in starter-kit/.icaif; register or import them first")
    sched, offset = runner.server_schedule()
    rows = runner.phase_rounds(sched, args.phase)
    if not rows:
        raise SystemExit(f"the server's schedule has no {args.phase} rounds")
    with runner.kit_session(readonly=True) as s:
        book = P.parse(s.portfolio(args.phase))
    ends = runner.et(rows[-1]["close_time"]) + pd.Timedelta(hours=1)
    print(f"{args.phase}: {len(rows)} rounds, {rows[0]['id']} .. {rows[-1]['id']}")
    print(f"server clock offset {offset:+.1f}s; book: cash {book.cash:,.2f}, "
          f"{len(book.summary()['names_held'])} names held")
    print(f"submit: the {args.submit} desk's book; armed until {ends}")
    if args.submit == "agent":
        print("WARNING: the agent's book has not passed the step-6 gate (Roadmap).")
    typed = input(f"Type '{args.phase}' to allow uploads: ").strip()
    if typed != args.phase:
        raise SystemExit("not armed")
    runner.write_private(runner.ARM_FILE, json.dumps({
        "phase": args.phase, "submit": args.submit, "expires": str(ends),
        "armed_at": str(pd.Timestamp.now(tz=calendar.TZ)), "host": socket.gethostname(),
        "user": os.environ.get("USER")}, indent=1))
    print(f"armed: {runner.ARM_FILE}")
    return 0


def cmd_disarm(args) -> int:
    runner.ARM_FILE.unlink(missing_ok=True)
    print("disarmed: no round will upload")
    return 0


def _report(out: Path) -> None:
    rows = runner.read_lines(out / "rounds.jsonl")
    if not rows:
        print("no rounds yet")
        return
    n = max(len(x["round_id"]) for x in rows) + 2
    print(f"\n{'round':<{n}}{'ready s':>8}{'rule':>7}{'upload':>11}{'shadow':>9}  outcome")
    for x in rows:
        sub = x.get("submitted", {})
        shadow = (x.get("shadow") or {}).get("action", "-")
        print(f"{x['round_id']:<{n}}{str(x.get('ready_before_deadline_s', '-')):>8}"
              f"{(x.get('rule') or {}).get('action', '-'):>7}"
              f"{sub.get('upload', {}).get('status', '-'):>11}{shadow:>9}  {x.get('outcome')}")
        for e in x.get("errors", []):
            print(f"{'':<{n}}ERROR {e}")
        for w in x.get("warnings", []):
            print(f"{'':<{n}}warn  {w[:150]}")


def cmd_status(args) -> int:
    out = runner.LIVE_OUT / args.phase
    st = runner.State(out)
    ok, why = runner.arm_status(args.phase, args.submit, pd.Timestamp.now(tz=calendar.TZ))
    print(f"{args.phase}: entry {st.data.get('entry')}; shadow spend ${st.data.get('spent_usd', 0):.2f}; "
          f"{'ARMED' if ok else 'not armed'}: {why}")
    for name, d in st.data.get("desks", {}).items():
        j = Journal.from_json(d.get("journal"))
        b = j.memory(lambda t: t, real=True)["book"]
        print(f"journal {name}: {len(j.rounds)} rounds, {b['names_held']} names held, return "
              f"{b['return_since_start']}, peak {b['peak_return_since_start']}, "
              f"{len(j.orders)} orders pending, {len(j.issues)} issues")
    _report(out)
    return 0


def cmd_journal(args) -> int:
    """Each desk's journal as its roles read it, then (`--full`) every round on record."""
    st = runner.State(Path(args.out) if args.out else runner.LIVE_OUT / args.phase)
    for name, d in st.data.get("desks", {}).items():
        if args.desk and name != args.desk:
            continue
        j = Journal.from_json(d.get("journal"))
        print(f"== {name}: {len(j.rounds)} rounds on record")
        print(json.dumps({"memory": j.memory(lambda t: t, real=True),
                          "names": j.name_fields(real=True)}, indent=1))
        if args.full:
            print(json.dumps(j.to_json(), indent=1))
        for i in j.issues:
            print(f"ISSUE {i['key']} {i['kind']}: {i['text']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("round", help="run one round now (the scheduler's worker)")
    p.add_argument("--phase", required=True)
    p.add_argument("--round-id", required=True)
    p.add_argument("--schedule", help="schedule snapshot the scheduler wrote")
    p.add_argument("--as-of", help="replay with this ET clock (fast rehearsal)")
    p.add_argument("--live", action="store_true")
    _common(p)
    p.set_defaults(fn=cmd_round)

    p = sub.add_parser("run", help="run a phase's rounds on the clock")
    p.add_argument("--phase", required=True, choices=runner.PHASES)
    p.add_argument("--live", action="store_true", help="server schedule and book; uploads only if armed")
    _common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("rehearse", help="a dry phase on the standard calendar")
    p.add_argument("--date", help="first session (default today, ET)")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--fast", action="store_true", help="past days only, rounds back to back")
    _common(p)
    p.set_defaults(fn=cmd_rehearse)

    p = sub.add_parser("score", help="(internal) score the daily model in a killable child")
    p.add_argument("--as-of", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("portfolio", help="read-only: the server's book and whether it parses")
    p.add_argument("--phase", required=True, choices=runner.PHASES)
    p.set_defaults(fn=cmd_portfolio)

    p = sub.add_parser("arm", help="the owner's approval to upload (interactive)")
    p.add_argument("--phase", required=True, choices=runner.PHASES)
    p.add_argument("--submit", choices=["rule", "agent"], default="rule")
    p.set_defaults(fn=cmd_arm)

    p = sub.add_parser("disarm", help="withdraw the approval")
    p.set_defaults(fn=cmd_disarm)

    p = sub.add_parser("status", help="a phase's entry, spend, arm and rounds")
    p.add_argument("--phase", required=True)
    p.add_argument("--submit", choices=["rule", "agent"], default="rule")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("journal", help="each desk's journal, as its roles read it")
    p.add_argument("--phase", required=True)
    p.add_argument("--desk", choices=["rule", "agent"], default=None)
    p.add_argument("--out", default=None, help="phase directory (default output/live/<phase>)")
    p.add_argument("--full", action="store_true", help="also every round on record, in full")
    p.set_defaults(fn=cmd_journal)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
