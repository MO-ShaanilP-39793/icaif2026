"""Replay the agent desk through competition windows, ranked like every other strategy.

    .venv/bin/python tools/agent_replay.py                          # rule brain: free
    .venv/bin/python tools/agent_replay.py --brain claude --max-calls 60 --yes

The rule brain is the sanity check: its desk must tie `q_riskparity_entry_regime` (the quant
candidate it stands for) in every window, or the desk's plumbing, not its judgement,
is what any LLM result would measure. The run stops if it doesn't.

**Claude replays cost money and need `--yes`.** The tool prints the call and dollar
estimate first. They are anonymised unless `--real-names` (see `agents.observe`), and
cached under output/agent/cache/, so a rerun with `--offline` repeats every answer and
calls nothing. Windows default to the confirm era's non-overlapping windows from
2025, the latest the model is least likely to have memorised; `--start` moves them.
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from icaif import baselines, compiler, data, markets, quant_strategies as qs, windows  # noqa: E402
from icaif.agents import brains  # noqa: E402
from icaif.agents.desk import DeskConfig, EarningsCalendar, desk  # noqa: E402

OUT = data.ROOT / "output" / "agent"
# Rough per-call size: ~9k input tokens (prompt + 30-name observation), ~2.5k output
# with thinking. Measured costs replace this after the first calls (brain.cost()).
EST_IN, EST_OUT = 9_000, 2_500


def load_earnings(market) -> EarningsCalendar:
    path = sorted((data.ROOT / "data" / "external").glob("earnings_2*.parquet"))[-1]
    events = pd.read_parquet(path)
    events = events[events["ticker"].isin(market.tickers)]
    return EarningsCalendar(events, market.days)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["rule", "claude"], default="rule")
    ap.add_argument("--model", default=brains.DEFAULT_MODEL, choices=brains.ALLOWED_MODELS)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--start", default=None, help="first window start (default: all for rule, 2025-01-01 for claude)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--windows", type=int, default=None, help="at most this many windows")
    ap.add_argument("--no-review", action="store_true", help="entry and events only (~15x fewer calls)")
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--real-names", action="store_true", help="do not anonymise (post-cutoff windows only)")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="answer only from the cache")
    ap.add_argument("--yes", action="store_true", help="confirm spending on a claude replay")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    t0 = time.time()
    market = markets.research_market()
    start = args.start or ("2025-01-01" if args.brain == "claude" else None)
    starts = [s for s in windows.window_starts(market) if s >= market.days[60]]
    if start:
        starts = [s for s in starts if s >= date.fromisoformat(start)]
    if args.end:
        starts = [s for s in starts if s <= date.fromisoformat(args.end)]
    if args.windows:
        starts = starts[: args.windows]
    cfg = DeskConfig(review=not args.no_review, events=not args.no_events,
                     anonymize=not args.real_names)
    earnings = load_earnings(market)
    scores = compiler.load_daily_scores()

    tag = args.tag or f"{args.brain}_{'noreview_' if args.no_review else ''}{len(starts)}w"
    log_dir = OUT / tag
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.brain == "rule":
        make = brains.RuleBrain
        live_brains = []
    else:
        per_window = 1 + (0 if args.no_review else 14) + (0 if args.no_events else 2)
        n_calls = per_window * len(starts)
        if args.max_calls:
            n_calls = min(n_calls, args.max_calls)
        p_in, p_out, _, _ = brains.PRICES[args.model]
        est = n_calls * (EST_IN * p_in + EST_OUT * p_out) / 1e6
        print(f"{len(starts)} windows, ~{n_calls} calls to {args.model} at effort "
              f"{args.effort}: ~${est:.0f} (cache hits are free)")
        if not (args.yes or args.offline):
            print("re-run with --yes to spend it, or --offline to use cached answers only")
            return
        shared = brains.ClaudeBrain(args.model, args.effort, max_calls=args.max_calls)
        cache = brains.CachedBrain(shared, OUT / "cache", offline=args.offline)
        make = lambda: cache  # noqa: E731 - one brain across windows, so the budget is global
        live_brains = [shared]

    desks = []

    def factory():
        d = desk(make, cfg, scores=scores, earnings=earnings)()
        desks.append(d)
        return d

    field = windows.run_field(baselines.FIELD, market, starts)
    cands = {"inv_vol_hold_75": baselines.scaled(baselines.InverseVolHold, 0.75),
             "q_riskparity_entry_regime": qs.CANDIDATES["q_riskparity_entry_regime"],
             f"desk_{args.brain}": factory}
    res = pd.concat([windows.rank_against_field(windows.run_field({n: f}, market, starts), field)
                     for n, f in cands.items()], ignore_index=True)
    piv = res.pivot(index="window", columns="strategy", values="overall_score")
    name = f"desk_{args.brain}"

    if args.brain == "rule" and not (piv[name] == piv["q_riskparity_entry_regime"]).all():
        bad = piv.index[piv[name] != piv["q_riskparity_entry_regime"]].tolist()
        raise SystemExit(f"rule desk differs from q_riskparity_entry_regime in {len(bad)} windows "
                         f"(first {bad[:3]}): the desk's plumbing is off; not reporting")

    log = [dict(e, window=str(w)) for w, d in zip(starts, desks) for e in d.log]
    (log_dir / "log.jsonl").write_text("\n".join(json.dumps(e, default=str) for e in log))
    res.to_csv(log_dir / "windows.csv", index=False)

    summ = windows.summarise(res).drop(columns=["invalid_rounds"])
    print(summ.round(4).to_string())
    for ref in ("inv_vol_hold_75", "q_riskparity_entry_regime"):
        d = piv[name] - piv[ref]
        print(f"{name} - {ref}: {d.mean():+.3f} (SE {d.std() / len(d) ** 0.5:.3f}), "
              f"better in {(d < 0).mean():.0%}, worse in {(d > 0).mean():.0%} of {len(d)} windows")
    lg = pd.DataFrame(log)
    if len(lg):
        print("\ndecisions by role and source:")
        print(lg.groupby(["role", "source"]).size().to_string())
        print(f"answers that differ from the rule: {(~lg['same_as_rule'].astype(bool)).sum()} "
              f"of {len(lg)}")
    for b in live_brains:
        print(f"\n{b.name}: {len(b.records)} calls, ${b.cost():.2f}; cache hits "
              f"{make().hits}, misses {make().misses}")
    print(f"\nlog: {log_dir}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
