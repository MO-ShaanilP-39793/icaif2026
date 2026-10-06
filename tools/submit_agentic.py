"""Submit a paid agent replay to the leaderboard's agentic panel.

    .venv/bin/python tools/submit_agentic.py C_grok47_free_blank_2026-01-21 \
        --name grok47_free_blank --window-choice "..." --note "..." [--dry]

Reads `output/agent/<tag>/`: the desk's row of `windows.csv` (its metrics in each window)
and the run's console log `<tag>.out` (calls and dollars, as `agent_replay` printed
them). Only metrics and the run's description are published, never the decisions or
the model's reasoning.

The panel ranks the entry against the main board's field in each window it covers
(icaif/leaderboard.py). `--window-choice` is required and shown beside the result: a
window picked for what the agent is good at ranks it higher than the board's own
windows would, and the reader has to know which it was.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import data, leaderboard, markets, space_hub  # noqa: E402

OUT = data.ROOT / "output" / "agent"
DESKS = ("desk_claude", "free_blank_claude", "free_informed_claude")


def spend(tag: str) -> tuple[str, int, float]:
    """(brain, calls, USD) from the replay's own summary line; raises if it is not there.

    Read from the run's log rather than typed: a cost entered by hand is the one number
    on the panel nobody could check against the run.
    """
    path = OUT / f"{tag}.out"
    m = re.search(r"^(\S+): (\d+) calls, \$([\d.]+);", path.read_text(), re.M)
    if not m:
        raise SystemExit(f"no spend line in {path}: was the replay paid and did it finish?")
    return m.group(1), int(m.group(2)), float(m.group(3))


def window_ends(market, starts) -> dict:
    from icaif import windows
    out = {}
    for s in starts:
        i = market.days.index(pd.Timestamp(s).date())
        out[s] = str(market.days[i + windows.WINDOW_DAYS - 1])
    return out


def code_version() -> dict:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=data.ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, cwd=data.ROOT).stdout.strip())
    return {"commit": head, "uncommitted_changes": dirty}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="an output/agent/<tag> directory from agent_replay")
    ap.add_argument("--name", required=True, help="the strategy name on the panel")
    ap.add_argument("--desk", required=True, help="what the desk is, in words")
    ap.add_argument("--window-choice", required=True, help="how the windows were picked")
    ap.add_argument("--note", default="")
    ap.add_argument("--author", help="default: your HF username")
    ap.add_argument("--dry", action="store_true", help="print the entry; submit nothing")
    args = ap.parse_args()

    res = pd.read_csv(OUT / args.tag / "windows.csv")
    mine = res[res["strategy"].str.startswith(DESKS)]
    if mine["strategy"].nunique() != 1:
        raise SystemExit(f"expected one agent desk in {args.tag}, found {sorted(mine['strategy'].unique())}")
    brain, calls, cost = spend(args.tag)
    log = pd.read_json(OUT / args.tag / "log.jsonl", lines=True)
    fallbacks = int((log["source"] == "fallback").sum())

    market = markets.research_market("alpaca")
    ends = window_ends(market, mine["window"].astype(str))
    wins = mine.assign(window_start=mine["window"].astype(str),
                       window_end=mine["window"].astype(str).map(ends))
    snapshot = sorted((data.ROOT / "data" / "public").glob("alpaca_30m_2*.parquet"))[-1].name
    agent = {"model": brain, "desk": args.desk, "calls": calls, "cost_usd": cost,
             "fallbacks": fallbacks, "window_choice": args.window_choice,
             "replay": args.tag, "code": code_version()}
    entry = leaderboard.make_entry(
        args.name, leaderboard.AGENTIC, wins, span=(wins["window_start"].min(), wins["window_end"].max()),
        sizing=leaderboard.BOARD_SIZING, market_snapshot=snapshot,
        author=args.author or ("dry-run" if args.dry else space_hub.whoami()),
        note=args.note, agent=agent)
    print(json.dumps(entry, indent=1))
    if args.dry:
        return
    path = space_hub.submit(entry)
    print(f"submitted {args.name} as {path}; https://huggingface.co/spaces/{space_hub.BOARD_ID}")


if __name__ == "__main__":
    main()
