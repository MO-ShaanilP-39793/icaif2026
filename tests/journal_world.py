"""A two-day dry phase for the journal's restart tests, run in this process or in a worker
that kills itself in the middle of a write.

    python -m tests.journal_world --out DIR --round-id ID --kill-at state.json:1

The world is test_runner's synthetic one (seeded, so every process builds the same
prices) with a shock to AAPL from 10:30 on day 1. The shadow's brain is active, so its
journal has an entry with an exclusion, an event exit and a review cut with an exit to
remember, not just a hold. `--kill-at NAME:N` sends SIGKILL to the worker halfway through
writing the temp file of its N-th write to a file called NAME: no `finally`, no flush, the
way a power cut or the watchdog leaves it.
"""

import argparse
import os
import signal
import sys
from pathlib import Path

import pandas as pd

from icaif import live, runner
from icaif.agents.schemas import EntryDecision, EventDecision, Exclusion, NameCall, ReviewDecision
from tests import test_runner as T
from tests.test_agents import Scripted

DAYS = (T.DAY1, T.DAY2)


def setup(out: Path, patch=setattr) -> dict:
    """The world, with Yahoo and the arm file pointed at it. `patch` is pytest's
    `monkeypatch.setattr` in a test, so the module globals come back afterwards."""
    daily = T._daily()
    world = {"daily": daily, "bars30": T._bars30(daily, list(DAYS), shock=("AAPL", T.DAY1, 0.8)),
             "now": None, "tmp": Path(out)}
    patch(live, "yahoo_daily", lambda symbols, start: (world["daily"], []))
    patch(live, "yahoo_intraday", lambda *a, **k: world["bars30"])
    patch(runner, "ARM_FILE", Path(out) / "kit" / "ARMED.json")
    world["sched"] = runner.rehearsal_schedule(list(DAYS), "test")
    return world


def brain() -> Scripted:
    def cut(p):
        held = [r["name"] for r in p["names"] if (r.get("weight_now") or 0) > 0]
        return ReviewDecision(action="set_exposure", exposure=0.35, reason=None, exit=held[:1],
                              rationale="the storm is persistent; trim, and sell the weakest name")
    return Scripted(
        entry=EntryDecision(shape="inverse_vol", views="none", exposure=0.5,
                            avoid=[Exclusion(name="TSLA", signal="earnings", why="reports tomorrow")],
                            rationale="a calmer start than the rule, without the reporter"),
        review=cut,
        event=lambda p: EventDecision(calls=[NameCall(name=t["name"], action="exit", reason="a gap")
                                             for t in p["triggers"]]))


def cfg(world, out: Path) -> runner.Config:
    return runner.Config(phase="test", shadow="rule", scoring=False, out=Path(out), window_days=2)


def doors(world, b=None) -> runner.Doors:
    return runner.Doors(now=lambda: world["now"], brain=lambda c, st: b or brain(),
                        score=lambda *a: {"status": "off"}, inputs=lambda c, mkt, r: ({}, {}))


def rows(world) -> list[dict]:
    return world["sched"]["rounds"]


def play(world, c, row: dict, b=None) -> dict:
    """One round at its wake, 12 minutes before the deadline, as the scheduler runs it."""
    world["now"] = runner.et(row["deadline"]) - pd.Timedelta(minutes=12)
    return runner.run_round(c, row, doors(world, b))


def kill_on_write(name: str, nth: int, patch=setattr) -> None:
    """SIGKILL this process halfway through its `nth` write of a file called `name`."""
    real, seen = runner.write_atomic, {"n": 0}

    def write(path, text, mode=0o644):
        if Path(path).name == name:
            seen["n"] += 1
            if seen["n"] == nth:
                tmp = Path(path).with_name(f".{Path(path).name}.tmp")
                with open(tmp, "w") as f:
                    f.write(text[: len(text) // 2])
                    f.flush()
                os.kill(os.getpid(), signal.SIGKILL)
        return real(path, text, mode)

    patch(runner, "write_atomic", write)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--round-id", required=True)
    ap.add_argument("--kill-at", default=None)
    args = ap.parse_args()
    world = setup(Path(args.out))
    if args.kill_at:
        name, nth = args.kill_at.rsplit(":", 1)
        kill_on_write(name, int(nth))
    row = next(r for r in rows(world) if r["id"] == args.round_id)
    play(world, cfg(world, Path(args.out) / "test"), row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
