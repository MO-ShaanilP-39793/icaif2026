"""Score one of this repo's own strategies on the holdout, and submit it to the leaderboard.

    .venv/bin/python tools/submit_strategy.py model_tilt_0.5 [--dry] [--author WHO]

For strategies that live here as code, not as an agent's decisions file. Each one runs
the way the references do: a fresh instance from $1M in cash in every 15-day window.
A decisions file holds the same thing written down, one run per window, so this is a
shortcut for code we already have, not a different way of scoring.

--dry scores and prints without writing anything.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icaif import compiler, data, holdout, leaderboard, markets, space_hub  # noqa: E402
from icaif import quant_strategies as qs  # noqa: E402
from icaif.compiler import Levers  # noqa: E402


def _tilt(levers: Levers):
    def build():
        return compiler.compiled(compiler.load_daily_scores(), compiler.trailing_daily_vol(), levers)
    return build


# name -> (factory builder, what it is). The note is shown on the public board.
STRATEGIES = {
    "model_tilt_0.5": (
        _tilt(Levers(weighting="tilt", tilt=0.5, rebalance_every=5, exposure=0.75,
                     band=0.05, gamma=0.0)),
        "Inverse-vol book at 75% tilted by the daily 5-day model score (tilt 0.5, "
        "rebalanced weekly, 5% band). Best tilt of the 2023-24 lever sweep. Model "
        "predictions are walk-forward out-of-sample; run fresh in each window."),
    "q_riskparity_entry_regime": (
        lambda: qs.CANDIDATES["q_riskparity_entry_regime"],
        "The rule desk's book: risk-parity weights at 85% gross in calm markets and 30% "
        "in turbulent ones, blended by a two-state HMM fit on the three years before "
        "each window. Exposure set at entry, then held. Run fresh in each window."),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(STRATEGIES))
    ap.add_argument("--dry", action="store_true", help="score and print; submit nothing")
    ap.add_argument("--author", help="default: your HF username")
    args = ap.parse_args()

    build, note = STRATEGIES[args.name]
    factory = build()
    market = markets.research_market("alpaca")
    start, end = holdout.HOLDOUT_START, holdout.HOLDOUT_END
    wins, skipped = holdout.rolling_runs(factory, market, "holdout")
    if skipped:
        print(f"skipped windows touching degraded days: {skipped}")

    roll = holdout.summarise_rolling(wins)
    print(f"{args.name}: {start}..{end}, {len(wins)} windows, "
          f"{int(wins['invalid_rounds'].sum())} invalid rounds")
    for k in holdout.METRICS:
        print(f"  {k:<18} mean window {roll.loc[k, 'mean']: .4f}   median window {roll.loc[k, 'median']: .4f}")
    if args.dry:
        return

    snapshot = sorted((data.ROOT / "data" / "public").glob("alpaca_30m_2*.parquet"))[-1].name
    entry = leaderboard.make_entry(
        args.name, leaderboard.SUBMITTED, wins, span=(start, end),
        sizing=leaderboard.BOARD_SIZING, market_snapshot=snapshot,
        author=args.author or space_hub.whoami(), note=note)
    path = space_hub.submit(entry)
    print(f"submitted {args.name} as {path}; https://huggingface.co/spaces/{space_hub.BOARD_ID}")


if __name__ == "__main__":
    main()
