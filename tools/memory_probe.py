"""Ask the model for a window's biggest moves, and flag a window it appears to remember.

    .venv/bin/python tools/memory_probe.py --on 2026-01-21              # prints the cost, stops
    .venv/bin/python tools/memory_probe.py --on 2026-01-21 --yes        # asks Gemini 2.5 Pro
    .venv/bin/python tools/memory_probe.py --board --yes                # the board's 8 windows
    .venv/bin/python tools/memory_probe.py --on 2024-07-24 --yes        # a control it should flag

Each window is the 15 sessions from its start, as `agent_replay --on` replays it. Two
questions a window (icaif/memprobe.py): the results reactions, told they are, and the
`LARGEST` largest daily moves by name and date alone. For each it prints the moves
beside the guesses, then sign agreement against chance, the correlation, and the
guesses' sizes, and calls the window clean, remembered or unanswered.

**Paid and cached.** Without `--yes` it prints the questions' size and the estimate and
stops. Answers are cached under output/agent/cache/ like a replay's, so a rerun (or
`--offline`) repeats them for nothing. Writes output/agent/memory_probe/<model>_<start>.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import boardrank, data, leaderboard, markets, memprobe, windows  # noqa: E402
from icaif.agents import brains  # noqa: E402
from icaif.agents.triggers import EarningsHistory, daily_panel  # noqa: E402

OUT = data.ROOT / "output" / "agent"
# Output tokens a call: the v1 runs' ~7,000 at effort high, reasoning included. A probe
# writes a short JSON, so this is the safe side; the measured cost is printed after.
EST_OUT = 7_000


def board_starts() -> list[str]:
    """The board's windows that share no day: the 8 the v2 evaluation ranks on it."""
    standing = leaderboard.standings(boardrank.fetch_entries())
    wins = standing["by_window"]
    return [wins[i]["window_start"] for i in leaderboard.disjoint_windows(wins)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", action="append", default=[], help="a window's first session (repeatable)")
    ap.add_argument("--board", action="store_true", help="add the board's non-overlapping windows")
    ap.add_argument("--model", default=brains.DEFAULT_MODEL, choices=brains.ALLOWED_MODELS)
    # The deep roles' effort: a model that thinks harder recalls more, and the PM, which
    # decides, thinks at high. Probed at medium, a window could pass that the PM remembers.
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--largest", type=int, default=memprobe.LARGEST)
    ap.add_argument("--yes", action="store_true", help="confirm spending")
    ap.add_argument("--offline", action="store_true", help="answer only from the cache")
    args = ap.parse_args()

    starts = list(args.on) + (board_starts() if args.board else [])
    if not starts:
        raise SystemExit("name a window with --on, or --board")
    market = markets.research_market()
    daily = daily_panel(market)
    path = sorted((data.ROOT / "data" / "external").glob("earnings_2*.parquet"))[-1]
    events = pd.read_parquet(path)
    reactions = EarningsHistory.from_market(events, market).table

    sets = []   # (start, kind, moves)
    for s in starts:
        on = pd.Timestamp(s).date()
        if on not in market.days:
            raise SystemExit(f"{s} is not a session in the market")
        i = market.days.index(on)
        days = market.days[i:i + windows.WINDOW_DAYS]
        if len(days) < windows.WINDOW_DAYS:
            raise SystemExit(f"window {s}: only {len(days)} sessions in the market")
        sets.append((s, "earnings", memprobe.earnings_moves(reactions, days)))
        sets.append((s, "largest", memprobe.largest_moves(daily, days, args.largest)))

    asked = [(s, k, m) for s, k, m in sets if memprobe.untested(m) is None]
    n_calls = len(asked)
    tin = sum(len(memprobe.SYSTEM) + len(json.dumps(memprobe.payload(m))) for _, _, m in asked) / 3.5
    p_in, p_out, _, _ = brains.PRICES[args.model]
    est = (tin * p_in + n_calls * EST_OUT * p_out) / 1e6
    print(f"{len(starts)} windows, {n_calls} calls to {args.model} at effort {args.effort}: "
          f"~${est:.2f} (cache hits are free)")
    for s, kind, m in sets:
        print(f"  {s} {kind}: {len(m)} moves" + ("" if memprobe.untested(m) is None else ", not asked"))
    if not (args.yes or args.offline):
        print("re-run with --yes to spend it, or --offline to use cached answers only")
        return
    problem = brains.credentials_problem(args.model)
    if problem and not args.offline:
        raise SystemExit(f"{problem}: every probe would come back unanswered")
    live = brains.make(args.model, args.effort)
    brain = brains.CachedBrain(live, OUT / "cache", offline=args.offline)

    out_dir = OUT / "memory_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts = {}
    for s in starts:
        record, scores = {"window": s, "model": args.model, "effort": args.effort, "sets": {}}, []
        for _, kind, moves in [x for x in sets if x[0] == s]:
            skip = memprobe.untested(moves)
            if skip:
                scores.append(skip)
                record["sets"][kind] = skip
                continue
            try:
                ans = brain.decide(f"memory_probe_{kind}", memprobe.SYSTEM, memprobe.payload(moves),
                                   memprobe.schema(moves), timeout=600)
                got = memprobe.attach(moves, ans)
                sc = memprobe.score(got)
            except brains.BrainError as err:
                got, sc = moves.assign(guess=float("nan")), {"lines": len(moves), "remembered": None,
                                                             "why": f"no answer: {err}"}
            scores.append(sc)
            record["sets"][kind] = {**sc, "moves": got.drop(columns="note").astype({"day": str})
                                    .to_dict(orient="records")}
            print(f"\n{s} {kind}:")
            print(got[["ticker", "day", "actual", "guess"]].to_string(index=False))
            print("  " + ", ".join(f"{k} {v}" for k, v in sc.items()))
        verdicts[s] = record["verdict"] = memprobe.verdict(scores)
        (out_dir / f"{args.model}_{s}.json").write_text(json.dumps(record, indent=1, default=str))

    print("\nwindows: " + ", ".join(f"{s} {v}" for s, v in verdicts.items()))
    print(f"{live.name}: {len(live.records)} calls, ${live.cost():.2f}; cache hits "
          f"{brain.hits}, misses {brain.misses}")


if __name__ == "__main__":
    main()
