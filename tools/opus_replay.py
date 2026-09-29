"""Let Opus decide everything, in two arms, and rank it like any other strategy.

    .venv/bin/python tools/opus_replay.py --dry                # free: plumbing + size
    .venv/bin/python tools/opus_replay.py --yes --max-cost 75  # both arms, paid

Arms (icaif/agents/free.py): **blank** gets the game's rules and the observation;
**informed** also gets our backtest evidence and the backtested book to beat. Both see
prices, model scores, macro (as z-scores and changes) and the FOMC calendar, all
anonymised per window. Windows are non-overlapping 15-session windows from 2025-01-01,
the most recent the model is least likely to have memorised.

Scored against two fields, neither containing a near-clone of the reference
(`inv_vol_hold_75`): the default field without `inv_vol_hold` (see blend_report for
why) and the active field.

Spending. `--dry` sends nothing: every question falls back to the rule, and the run
measures the real prompt sizes for the estimate. A paid run needs `--yes`, stops
calling at `--max-cost` (split evenly between arms; later questions fall back to the
rule), caches every answer under output/agent/cache_free/, and `--offline` repeats a
cached run for free. The key is read from ANTHROPIC_API_KEY or from `.env`; it is
never printed.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import baselines, calendar, compiler, data, external, macro, markets, sim, windows  # noqa: E402
from icaif import quant_strategies as qs  # noqa: E402
from icaif.agents import brains, prompts  # noqa: E402
from icaif.agents.brains import BrainError  # noqa: E402
from icaif.agents.desk import DeskConfig  # noqa: E402
from icaif.agents.free import ARMS, FreeDesk  # noqa: E402

OUT = data.ROOT / "output" / "agent"
CALLS_PER_WINDOW = 15
EST_OUT_TOKENS = 3_000  # adaptive thinking at effort high; replaced by measured usage after


class SizeProbe:
    """Records what would be sent, then declines: the desk falls back to the rule."""

    name = "dry"

    def __init__(self):
        self.sizes = []

    def decide(self, role, system, payload, schema, timeout):
        self.sizes.append(len(system) + len(json.dumps(payload, sort_keys=True)))
        raise BrainError("dry run")


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = data.ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"').strip("'")
                return
    raise SystemExit("no ANTHROPIC_API_KEY in the environment or .env; add it to .env "
                     "(gitignored) and rerun")


def warm(market, start) -> None:
    """Build the market's lazy caches once, before threads race to build them."""
    dl = calendar.at(start, calendar.ROUNDS[1][0])
    market.recent_closes(dl, 1)
    qs.daily_closes(type("C", (), {"market": market, "deadline": dl})(), 2)


def run_arm(arm, brain, market, starts, workers, **desk_kw):
    def one(i, s):
        d = FreeDesk(brain, arm, DeskConfig(anonymize=True, seed=i), **desk_kw)
        r = sim.run(d, market, s, windows.WINDOW_DAYS)
        return ({"window": s, "strategy": f"opus_{arm}", **r.metrics(),
                 "invalid_rounds": len(r.invalid_rounds)},
                [dict(e, window=str(s)) for e in d.log])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda a: one(*a), enumerate(starts)))
    return pd.DataFrame([o[0] for o in out]), [e for o in out for e in o[1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["blank", "informed", "both"], default="both")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-cost", type=float, default=75.0, help="USD, split between arms")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    market = markets.research_market()
    starts = [s for s in windows.window_starts(market) if s >= date.fromisoformat(args.start)]
    last = compiler.load_daily_scores().frame.index.max().date()
    starts = [s for s in starts if market.days[market.days.index(s) + 14] <= last][: args.windows]
    arms = ["blank", "informed"] if args.arm == "both" else [args.arm]
    desk_kw = dict(scores=compiler.load_daily_scores(),
                   context=macro.wide(external.load("yahoo_daily_context")),
                   fomc=macro.FomcCalendar.load())
    warm(market, starts[0])

    if not (args.dry or args.offline):
        probe = SizeProbe()
        run_arm("informed", probe, market, starts[:2], 2, **desk_kw)
        tokens_in = np.mean(probe.sizes) / 3.0
        p_in, p_out, _, _ = brains.PRICES[brains.DEFAULT_MODEL]
        per_call = (tokens_in * p_in + EST_OUT_TOKENS * p_out) / 1e6
        n = CALLS_PER_WINDOW * len(starts) * len(arms)
        print(f"{len(starts)} windows x {len(arms)} arm(s) = ~{n} calls to {brains.DEFAULT_MODEL} "
              f"(effort {args.effort}); ~{tokens_in:,.0f} input tokens a call; "
              f"~${n * per_call:.0f} estimated, hard cap ${args.max_cost:.0f}")
        if not args.yes:
            print("rerun with --yes to spend it")
            return
        load_key()

    rows, logs, spend = [], [], {}
    for arm in arms:
        if args.dry:
            brain = SizeProbe()
        else:
            inner = brains.ClaudeBrain(effort=args.effort, max_cost=args.max_cost / len(arms))
            brain = brains.CachedBrain(inner, OUT / "cache_free", offline=args.offline)
        res, log = run_arm(arm, brain, market, starts, args.workers, **desk_kw)
        rows.append(res)
        logs += [dict(e, arm=arm) for e in log]
        if args.dry:
            print(f"{arm}: prompt ~{np.mean(brain.sizes) / 3.0:,.0f} tokens a call "
                  f"({len(brain.sizes)} calls probed)")
        else:
            inner = brain.inner
            spend[arm] = {"calls": len(inner.records), "usd": round(inner.cost(), 2),
                          "cache_hits": brain.hits,
                          "input_tokens": sum(r["input_tokens"] for r in inner.records),
                          "output_tokens": sum(r["output_tokens"] for r in inner.records),
                          "errors": sum(1 for r in inner.records if r["error"])}
        print(f"{arm} done ({time.time() - t0:.0f}s)")

    tag = "dry" if args.dry else f"paid_{args.effort}"
    log_dir = OUT / f"opus_{tag}_{len(starts)}w"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "log.jsonl").write_text("\n".join(json.dumps(e, default=str) for e in logs))
    (log_dir / "spend.json").write_text(json.dumps(spend, indent=1))

    ref = windows.run_field({"inv_vol_hold_75": baselines.scaled(baselines.InverseVolHold, 0.75)},
                            market, starts)
    cands = pd.concat([ref, *rows], ignore_index=True)
    no_clone = {k: v for k, v in baselines.FIELD.items() if k != "inv_vol_hold"}
    ranked = []
    for fname, fdef in (("default_no_clone", no_clone), ("active", baselines.ACTIVE_FIELD)):
        field = windows.run_field(fdef, market, starts)
        for name, g in cands.groupby("strategy"):
            ranked.append(windows.rank_against_field(g, field).assign(field=fname))
    res = pd.concat(ranked, ignore_index=True)
    res.to_csv(log_dir / "windows.csv", index=False)

    rk = ["rank_cumulative_return", "rank_sharpe_ratio", "rank_maximum_drawdown", "rank_turnover"]
    for fname, g in res.groupby("field"):
        piv = g.pivot(index="window", columns="strategy", values="overall_score")
        s = g.groupby("strategy").agg(score=("overall_score", "mean"),
                                      **{r.replace("rank_", "r_"): (r, "mean") for r in rk},
                                      ret=("cumulative_return", "mean"), sharpe=("sharpe_ratio", "mean"),
                                      mdd=("maximum_drawdown", "mean"), turnover=("turnover", "mean"))
        d = piv.sub(piv["inv_vol_hold_75"], axis=0)
        s["vs_hold"], s["se"] = d.mean(), d.std() / len(d) ** 0.5
        s["better_in"] = (d < 0).mean()
        print(f"\n=== {fname} field, {len(piv)} windows (lower is better) ===")
        print(s.sort_values("score").round(4).to_string())

    lg = pd.DataFrame(logs)
    if len(lg):
        lg["action"] = lg["decision"].map(lambda d: d.get("action"))
        print("\nsources by arm:", lg.groupby(["arm", "source"]).size().to_dict())
        after = lg[(lg["day"] > 1) & (lg["action"] == "rebalance")]
        print("rebalances after day 1:", after.groupby("arm").size().to_dict(),
              "in windows:", after.groupby("arm")["window"].nunique().to_dict())
    if spend:
        print("\nspend:", json.dumps(spend))
    print(f"\nlog: {log_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
