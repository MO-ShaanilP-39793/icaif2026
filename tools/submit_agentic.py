"""Submit paid agent replays to a suite's leaderboard as one agentic entry.

    .venv/bin/python tools/submit_agentic.py TAG [TAG ...] --name NAME --desk "..."
        [--suite holdout|official4] [--window-choice "..."] [--note "..."] [--dry]

    # the v1 desk on the Earnings season's four windows, one run per window:
    .venv/bin/python tools/submit_agentic.py v1_free_gemini_2025-04-11 \\
        v1_free_gemini_2025-10-13 v1_free_gemini_2026-04-13 v1_free_gemini_2026-07-13 \\
        --suite official4 --name v1_free_gemini --desk "..." --dry

Reads, per tag, `output/agent/<tag>/windows.csv` (the desk's metrics in each window) and
`output/agent/<tag>.out` (calls and dollars, as `agent_replay` printed them). Only metrics
and the run's description are published, never the decisions or the model's reasoning.
icaif/replay_entry.py has every check, and what each one prevents.

The board is read first, as its page reads it, because the entry is checked against it:
each window's inv_vol_hold_75 must match the board's reference, and a fixed suite's
windows must all be there. On the holdout an agentic entry sits on its own panel and
`--window-choice` is required, since a window picked for the agent's strength ranks it
higher than the board's own windows would. On a fixed suite it ranks in the one field,
and the windows are the suite's, chosen before any run.

--dry prints the entry and submits nothing. Without it the entry goes to the private
dataset and the public board (`space_hub.submit`).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import boardrank, data, leaderboard, replay_entry, space_hub, suites  # noqa: E402

OUT = data.ROOT / "output" / "agent"


def code_version() -> dict:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=data.ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, cwd=data.ROOT).stdout.strip())
    return {"commit": head, "uncommitted_changes": dirty}


def load_run(tag: str) -> replay_entry.Run:
    log_path = OUT / tag / "log.jsonl"
    log = pd.read_json(log_path, lines=True) if log_path.exists() else None
    return replay_entry.read_run(tag, pd.read_csv(OUT / tag / "windows.csv"),
                                 (OUT / f"{tag}.out").read_text(), log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="output/agent/<tag> directories from agent_replay")
    ap.add_argument("--suite", choices=sorted(suites.SUITES), default=suites.DEFAULT)
    ap.add_argument("--name", required=True, help="the strategy name on the board")
    ap.add_argument("--desk", required=True, help="what the desk is, in words")
    ap.add_argument("--window-choice", help="how the windows were picked (holdout: required)")
    ap.add_argument("--note", default="")
    ap.add_argument("--author", help="default: your HF username")
    ap.add_argument("--dry", action="store_true", help="print the entry; submit nothing")
    args = ap.parse_args()

    suite = suites.get(args.suite)
    choice = args.window_choice
    if not choice:
        if not suite.fixed:
            sys.exit("--window-choice is required on the holdout: a window picked for the "
                     "agent's strength ranks it higher than the board's own would")
        choice = f"all {len(suite.windows)} windows of the {suite.title} suite, fixed before the runs"

    try:
        runs = [load_run(t) for t in args.tags]
        standing = leaderboard.boards(boardrank.fetch_entries())
        snapshot = sorted((data.ROOT / "data" / "public").glob("alpaca_30m_2*.parquet"))[-1].name
        entry = replay_entry.build_entry(
            runs, suite.name, standing, name=args.name, desk=args.desk, window_choice=choice,
            snapshot=snapshot, author=args.author or ("dry-run" if args.dry else space_hub.whoami()),
            note=args.note, code=code_version())
    except (replay_entry.ReplayError, boardrank.AnchorMismatch, leaderboard.EntryError) as err:
        sys.exit(f"not submitted: {err}")

    print(json.dumps(entry, indent=1))
    if args.dry:
        return
    path = space_hub.submit(entry)
    print(f"submitted {args.name} to {suite.name} as {path}; "
          f"https://huggingface.co/spaces/{space_hub.BOARD_ID}")


if __name__ == "__main__":
    main()
