"""Place a replay's strategies on the holdout leaderboard, window by window.

    .venv/bin/python tools/board_rank.py C_grok47_free_blank_2026-01-21
    .venv/bin/python tools/board_rank.py <tag> <tag> ... [--field 10]

Reads each `output/agent/<tag>/windows.csv` from `agent_replay` and the live board
(icaif/boardrank.py). For every window the board covers it first checks that the
replay's `inv_vol_hold_75` equals the board's reference to 1e-9 in all four metrics,
and stops if not. Then each strategy is placed against the board's whole field plus
itself (as the agentic panel does), with the entrants just ahead and behind it.
Windows outside the board's span (Jul-Sep 2026) have no field: they are listed with
the replay's own metrics, to compare against the hold and the rule only.

Several tags add up: the summary gives each strategy's mean score minus the hold's
and the rule's (`q_riskparity_entry_regime`), with the SE over the windows that share
no day, which is the v2 gate's test. Writes `output/agent/<tag>/board.csv`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import boardrank, data, leaderboard  # noqa: E402

OUT = data.ROOT / "output" / "agent"
RULE = "q_riskparity_entry_regime"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="output/agent/<tag> directories from agent_replay")
    ap.add_argument("--field", type=int, default=5, help="show the board's top N in each window")
    args = ap.parse_args()

    standing = leaderboard.standings(boardrank.fetch_entries())
    print(f"board: {standing['entrants']} entrants, {standing['windows']} windows "
          f"{standing['span'][0]}..{standing['span'][1]}")
    wins = boardrank.board_windows(standing)
    placed, outside = [], []
    for tag in args.tags:
        replay = pd.read_csv(OUT / tag / "windows.csv")
        try:
            got, off = boardrank.rank_replay(replay, standing)
        except boardrank.AnchorMismatch as err:
            raise SystemExit(f"{tag}: {err}") from err
        if len(got):
            got.insert(0, "tag", tag)
            got.to_csv(OUT / tag / "board.csv", index=False)
            placed.append(got)
        outside += [(tag, w, replay[replay["window"].astype(str) == w]) for w in off]

    cols = ["position", "of", "overall_score", "cumulative_return", "sharpe_ratio",
            "maximum_drawdown", "turnover", "source", "ahead", "behind"]
    for p in placed:
        for w, g in p.groupby("window_start", sort=True):
            print(f"\n{p['tag'].iloc[0]}  window {w}..{g['window_end'].iloc[0]}  "
                  f"({boardrank.ANCHOR} matches the board to {g['anchor_max_diff'].iloc[0]:.0e})")
            print(g.set_index("strategy")[cols].sort_values(["position", "overall_score"])
                  .round(4).to_string())
            top = wins[w]["rows"][: args.field]
            print("  board's top: " + ", ".join(f"{r['position']}. {r['strategy']} "
                                                f"({r['overall_score']:.2f})" for r in top))
    for tag, w, g in outside:
        print(f"\n{tag}  window {w}: not on the board ({standing['span'][0]}..{standing['span'][1]}); "
              "the hold and the rule are its only comparison")
        print(g.set_index("strategy")[boardrank.METRICS].round(4).to_string())

    if not placed:
        return
    # The hold and the rule are in every replay of a window, identically; two runs of one
    # desk in one window are not, and averaging either in would count that window twice.
    allp = pd.concat(placed, ignore_index=True).drop_duplicates(
        ["window_start", "strategy", *boardrank.METRICS])
    if allp.duplicated(["window_start", "strategy"]).any():
        dup = allp[allp.duplicated(["window_start", "strategy"], keep=False)]
        raise SystemExit(f"{sorted(set(dup['strategy']))} ran twice with different results in "
                         f"{sorted(set(dup['window_start']))}: which run counts is ambiguous")
    summ = allp.groupby("strategy").agg(windows=("window_start", "nunique"),
                                        mean_position=("position", "mean"),
                                        mean_score=("overall_score", "mean"))
    print("\nacross windows on the board:")
    print(summ.sort_values("mean_score").round(3).to_string())
    for ref in (boardrank.ANCHOR, RULE):
        c = boardrank.compare(allp, ref)
        if len(c):
            print(f"\nscore minus {ref}'s (lower is better; SE over windows sharing no day):")
            print(c.drop(columns="against").set_index("strategy").round(3).to_string())


if __name__ == "__main__":
    main()
