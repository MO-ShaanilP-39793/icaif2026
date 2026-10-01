"""Kill the live runner between rounds and in the middle of its writes, in a dry replay
of past sessions, and check it ends where an unbroken run does (Roadmap step 4's gate).

    .venv/bin/python tools/restart_drill.py --date 2026-09-25 --days 4    # ~5 min

Two runs of the same sessions through the real scheduler (`runner.run_phase`, fast),
each round in its own worker process, as `tools/live_runner.py rehearse --fast` runs
them, dry and with the rule as the shadow (no LLM is called):

- **unbroken**: every round once.
- **broken**: workers SIGKILLed halfway through the temp file of a write (the entry's
  decision.json, a round's state.json commit, the journal copy written after it,
  rounds.jsonl), and the scheduler stopped between rounds and started again. SIGKILL
  runs no `finally` and flushes nothing; the scheduler keeps nothing in memory between
  rounds, so stopping its loop is what killing its process does.

Then the two phase directories must hold the same entry, rounds, paper books and
journals, and each desk's journal must agree with its own paper ledger
(`journal.verify`). Writes reports/restart_drill.json.

**Yahoo is fetched once**, at the start, and every worker reads that snapshot through the
doors a live round uses (`live.yahoo_daily`, `live.yahoo_intraday`), cut at its own
clock by the same code (`fetch_closes`, `fetch_intraday`, `vol_bars`). A fast rehearsal
fetches every round; two of them would send Yahoo a few thousand requests in minutes,
enough to be throttled in the middle of a live rehearsal on the same machine, and they
would replay slightly different data, so a difference between the runs could be Yahoo's.
Yahoo serves 30m bars five sessions back, so only the last four completed sessions can
be replayed.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import calendar, data, live, runner, watchdog  # noqa: E402
from icaif.agents import journal as J  # noqa: E402

TOOL = Path(__file__).resolve()
OUT = data.ROOT / "output" / "live"
REPORT = data.ROOT / "reports" / "restart_drill.json"
# Round index (0-based, in deadline order) -> the write its first worker dies in.
KILLS = {0: "decision.json:1", 1: "state.json:1", 8: "agent.json:1", 14: "state.json:1",
         20: "rounds.jsonl:1"}
STOPS = (5, 17)   # the scheduler is stopped after this many rounds, then started again


class Stopped(BaseException):
    """The scheduler, stopped between rounds."""


def snapshot(out: Path, last_session) -> Path:
    """Yahoo's daily bars and 30m bars (5 and 60 days), fetched once for every worker."""
    tickers = sorted(data.load_universe())
    out.mkdir(parents=True, exist_ok=True)
    start = f"{pd.Timestamp(last_session) - pd.Timedelta(days=live.CLOSE_HISTORY_DAYS):%Y-%m-%d}"
    daily, missing = live.yahoo_daily(tickers, start)
    if missing:
        raise SystemExit(f"Yahoo returned no daily bars for {missing}")
    daily.to_parquet(out / "daily.parquet")
    live.yahoo_intraday(tickers, "30m", live.INTRADAY_PERIOD).to_parquet(out / "bars_5d.parquet")
    live.yahoo_intraday(tickers, "30m", live.VOL_PERIOD).to_parquet(out / "bars_60d.parquet")
    return out


def cmd_worker(args) -> int:
    """One round as `live_runner.py round` runs it, on the snapshot, killed if asked."""
    snap = Path(args.snapshot)
    daily = pd.read_parquet(snap / "daily.parquet")
    bars = {live.INTRADAY_PERIOD: pd.read_parquet(snap / "bars_5d.parquet"),
            live.VOL_PERIOD: pd.read_parquet(snap / "bars_60d.parquet")}
    live.yahoo_daily = lambda symbols, start: (daily, [])
    live.yahoo_intraday = lambda symbols, interval="30m", period=live.INTRADAY_PERIOD: bars[period]
    if args.kill_at:
        name, nth = args.kill_at.rsplit(":", 1)
        real, seen = runner.write_atomic, {"n": 0}

        def write(path, text, mode=0o644):
            if Path(path).name == name:
                seen["n"] += 1
                if seen["n"] == int(nth):
                    with open(Path(path).with_name(f".{Path(path).name}.tmp"), "w") as f:
                        f.write(text[: len(text) // 2])
                    os.kill(os.getpid(), signal.SIGKILL)
            return real(path, text, mode)

        runner.write_atomic = write
    import importlib.util

    spec = importlib.util.spec_from_file_location("live_runner", runner.TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    sys.argv = [str(runner.TOOL), *args.rest]
    return tool.main()


def run(cfg: runner.Config, sched: dict, snap: Path, broken: bool) -> dict:
    """One run of the phase; with `broken`, the kills and stops above. Returns what happened."""
    ids = [r["id"] for r in runner.phase_rounds(sched, cfg.phase)]
    log = {"workers": 0, "sigkilled": [], "stops": 0, "retried": []}
    killed: set = set()

    def spawn(argv, timeout):
        rid = argv[argv.index("--round-id") + 1]
        i = ids.index(rid)
        if broken and i in STOPS and rid not in killed and i not in log.setdefault("stopped_at", []):
            log["stopped_at"].append(i)
            log["stops"] += 1
            raise Stopped()
        kill = KILLS.get(i) if broken and rid not in killed else None
        if kill:
            killed.add(rid)
        elif rid in killed:
            log["retried"].append(rid)
        cmd = [sys.executable, str(TOOL), "worker", "--snapshot", str(snap)]
        cmd += (["--kill-at", kill] if kill else []) + ["--", *argv[2:]]
        log["workers"] += 1
        res = watchdog.run(cmd, timeout)
        if kill:
            log["sigkilled"].append({"round": rid, "write": kill, "returncode": res.returncode})
        return res

    while True:
        try:
            runner.run_phase(cfg, lambda: (sched, 0.0), fast=True, spawn=spawn)
            return log
        except Stopped:
            continue   # started again, from what the last round committed


def comparable(out: Path) -> dict:
    """The phase's committed state without clocks: what must match run to run."""
    st = json.loads((out / "state.json").read_text())
    for r in st["rounds"].values():
        r.pop("at", None)
    for d in st["desks"].values():
        for e in d["log"]:
            e.pop("latency_s", None)
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("worker")
    w.add_argument("--snapshot", required=True)
    w.add_argument("--kill-at", default=None)
    w.add_argument("rest", nargs=argparse.REMAINDER)
    ap.add_argument("--date", default=None, help="first session (default: four sessions back)")
    ap.add_argument("--days", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "worker":
        args.rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
        return cmd_worker(args)

    t0 = time.time()
    now = pd.Timestamp.now(tz=calendar.TZ)
    last = live.latest_completed_session(now)
    sessions = [d.date() for d in live.sessions(last - pd.Timedelta(days=14), last)]
    first = date.fromisoformat(args.date) if args.date else sessions[-args.days]
    days = [d for d in sessions if d >= first][: args.days]
    if len(days) < args.days or days[0] != first:
        raise SystemExit(f"{first} plus {args.days} sessions is not within the completed sessions")
    phase = f"drill-{first}-{args.days}d"
    sched = runner.rehearsal_schedule(days, phase)
    base = OUT / phase
    snap = snapshot(base / "yahoo", days[-1])
    print(f"{phase}: {len(sched['rounds'])} rounds on {[str(d) for d in days]}; Yahoo snapshot in {snap}")
    report = {"phase": phase, "days": [str(d) for d in days], "rounds": len(sched["rounds"])}
    for name, broken in (("unbroken", False), ("broken", True)):
        cfg = runner.Config(phase=phase, shadow="rule", scoring=False, out=base / name)
        if (cfg.out / "state.json").exists():
            raise SystemExit(f"{cfg.out} holds a run already; remove it to drill again")
        t1 = time.time()
        report[name] = run(cfg, sched, snap, broken)
        report[name]["seconds"] = round(time.time() - t1)
        print(f"{name}: {report[name]}")
    a, b = comparable(base / "unbroken"), comparable(base / "broken")
    report["same_state"] = a == b
    report["differs_in"] = sorted(k for k in a if a.get(k) != b.get(k))
    report["journals"] = {}
    for name in ("unbroken", "broken"):
        st = runner.State(base / name)
        for desk, paper in (("rule", "submitted"), ("agent", "agent")):
            j = J.Journal.from_json(st.data["desks"][desk]["journal"])
            problems = J.verify(j, J.paper_fills(st.data["paper"][paper]))
            disk = json.loads((base / name / "journal" / f"{desk}.json").read_text())["journal"]
            report["journals"][f"{name}/{desk}"] = {
                "rounds": len(j.rounds), "fills": sum(1 for e in j.rounds if e.get("fill")),
                "names_held": len(j.positions), "issues": len(j.issues), "disagreements": problems,
                "copy_on_disk_is_committed": disk == st.data["desks"][desk]["journal"],
                "return_since_start": j.memory(lambda t: t, True)["book"]["return_since_start"]}
        outcomes = [st.data["rounds"][r["id"]]["outcome"] for r in sched["rounds"]]
        report[name]["outcomes"] = {o: outcomes.count(o) for o in sorted(set(outcomes))}
        listed = {x["round_id"] for x in runner.read_lines(base / name / "rounds.jsonl")}
        report[name]["rounds_indexed"] = len(listed & {r["id"] for r in sched["rounds"]})
    report["seconds"] = round(time.time() - t0)
    REPORT.write_text(json.dumps(report, indent=1, default=str) + "\n")
    ok = (report["same_state"]
          and all(report[n]["rounds_indexed"] == report["rounds"] for n in ("unbroken", "broken"))
          and all(not v["disagreements"] and v["issues"] == 0 and v["copy_on_disk_is_committed"]
                  for v in report["journals"].values()))
    print(json.dumps({k: report[k] for k in ("same_state", "differs_in", "journals")}, indent=1, default=str))
    print(f"{'PASS' if ok else 'FAIL'} ({report['seconds']}s); wrote {REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
