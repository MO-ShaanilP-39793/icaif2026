"""Stage 1 of desk v3: the PM's entry, then a pure hold, on the stage-1 windows.

    .venv/bin/python tools/entry_replay.py round0                 # no LLM: holds at each gross
    .venv/bin/python tools/entry_replay.py ledgers                # code brain = the hold, all windows
    .venv/bin/python tools/entry_replay.py arm --analysts none --gross free          # prints the cost
    .venv/bin/python tools/entry_replay.py arm --analysts reports_raw --drop headlines --yes

Windows are `v3.STAGE1`: `--split select` (2025, where arms are chosen; the default) or
`--split confirm` (2026, the hold-out: scored once, for the committed choice and its
runner-up). `--split confirm` needs `--confirm-choice <commit>`, the commit that recorded
the choice, so a hold-out run can't happen by accident and its record names what it
checked. It is no gate: v3 ships either way (stage1_doe.md).

**Every score is paired by window and reported on two fields**: the modelled field
(`baselines.FIELD`) and the same without `inv_vol_hold`, whose near-copy of the reference
shifts close calls (the no-clone field, the primary one). An arm is compared with
`inv_vol_hold_75` (the bar), with inverse-vol at the arm's own gross in that window
(names apart from cash), and with cash and the rule for context. Each of the four
metric ranks is reported apart: an arm that gains two return ranks and loses two
turnover ranks scores as the hold does, but is not the same thing.

**The rescale** (`v3.BoughtBook`): each window's book bought again at 25, 50, 75 and
100% gross and scored, at no cost. A hold trades once, so this is exactly what that
gross would have done. A gross the book can't reach under the 30% cap is left blank.

**Cash floors, scored for free** (`stage1_doe.md`, round 3): each window's book capped
at every floor's gross (`v3.capped_book`), as `arm_floor_<pct>`. That assumes the PM
would pick the same names under a floor; `--cash-floor F` runs the floor for real (the
PM is told it), and `tools/doe_report.py cash` compares the two.

**The design** (`stage1_doe.md`): `--design streams16 --run N` runs row N of the streams
factorial (`v3.STREAMS16`), so a run is named by its row rather than a retyped list.

**Paid runs** print their estimate and stop without `--yes`; answers are cached under
output/agent/cache/, so `--offline` repeats a run and asks nothing. `--repeat N` asks
every question again under its own cache keys, for the noise check.
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import baselines, compiler, data, markets, quant_strategies as qs, ranking, sim, windows  # noqa: E402
from icaif.agents import brains, signals  # noqa: E402
from icaif.agents import journal as J  # noqa: E402
from icaif.agents.desk import EarningsCalendar  # noqa: E402
from icaif.agents.triggers import EarningsHistory  # noqa: E402
from icaif.agents.v3 import (CASH_FLOORS, STAGE1, STREAMS, STREAMS16, BoughtBook, HoldBrain,  # noqa: E402
                             V3Config, V3Desk, capped_book, floor_gross, parse_gross)

OUT = data.ROOT / "output" / "entry"
CACHE = data.ROOT / "output" / "agent" / "cache"
GROSSES = (0.25, 0.5, 0.75, 1.0)
HOLD = "inv_vol_hold_75"
RULE = "q_riskparity_entry_regime"
# Tokens a call. The PM and check are measured on the smoke window (2025-02-03, real
# names): a 24,000-character observation plus a 5,700-character prompt, about 8,500
# tokens in, and $0.07 and $0.04 at Pro high, which leaves about 6,000 and 3,000 tokens
# out, reasoning included. The analysts are unmeasured: a slice of the same observation
# at Flash medium. Measured costs replace these after each run (`brain.cost()`).
EST = {"pm": (8_500, 6_000), "pm_check": (10_000, 3_000), "analyst": (4_000, 3_500)}


def windows_of(market, split: str) -> list[date]:
    """The split's window starts, each checked to run its 15 sessions to its stated end:
    a session missing from the market would run a window on into the next one's days."""
    days = market.days
    out = []
    for first, last in STAGE1[split]:
        a, b = date.fromisoformat(first), date.fromisoformat(last)
        if a not in days:
            raise SystemExit(f"{a} is not a session in the market")
        i = days.index(a)
        span = days[i:i + windows.WINDOW_DAYS]
        if len(span) < windows.WINDOW_DAYS or span[-1] != b:
            raise SystemExit(f"window {a}: its 15 sessions end {span[-1] if span else None}, not {b}")
        out.append(a)
    return out


def fields(market, starts) -> dict:
    field = windows.run_field(baselines.FIELD, market, starts)
    return {"no_clone": field[field["strategy"] != "inv_vol_hold"], "default": field}


def ranked(results: pd.DataFrame, flds: dict, exclude=()) -> pd.DataFrame:
    """`results`: one row per (window, strategy) with the four metrics. Each strategy is
    ranked alone against each field, less any member named in `exclude` (a candidate is
    never ranked against its own copy)."""
    rows = []
    for fname, f in flds.items():
        f = f[~f["strategy"].isin(exclude)]
        for name, g in results.groupby("strategy", sort=False):
            r = windows.rank_against_field(g, f)
            rows.append(r.assign(field=fname))
    return pd.concat(rows, ignore_index=True)


def metrics_row(res: sim.Result, window, name) -> dict:
    return {"window": window, "strategy": name, **res.metrics(), "invalid_rounds": len(res.invalid_rounds)}


def paired(table: pd.DataFrame, a: str, b: str) -> dict:
    """a - b per window (lower is better), on the windows both have."""
    piv = table.pivot(index="window", columns="strategy", values="overall_score")
    d = (piv[a] - piv[b]).dropna()
    out = {"diff": d.mean(), "se": d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan,
           "better": int((d < 0).sum()), "worse": int((d > 0).sum()), "n": int(len(d))}
    for m in ranking.METRICS:
        col = f"rank_{m}"
        p = table.pivot(index="window", columns="strategy", values=col)
        out[col] = (p[a] - p[b]).dropna().mean()
    return out


def report(table: pd.DataFrame, arm: str, refs: list[str]) -> None:
    for fname in ("no_clone", "default"):
        t = table[table["field"] == fname]
        print(f"\n{fname} field" + ("  (primary)" if fname == "no_clone" else ""))
        print(windows.summarise(t).drop(columns=["invalid_rounds", "wins"]).round(4).to_string())
        print(f"\n  {arm} minus each reference, paired by window (negative is better):")
        for ref in refs:
            if ref not in set(t["strategy"]):
                continue
            p = paired(t, arm, ref)
            ranks = ", ".join(f"{m.split('_')[0]} {p[f'rank_{m}']:+.2f}" for m in ranking.METRICS)
            print(f"  vs {ref:28s} {p['diff']:+.3f} (SE {p['se']:.3f}), better in {p['better']}, "
                  f"worse in {p['worse']} of {p['n']}; by rank: {ranks}")


# ----------------------------------------------------------------------------- round 0

def round0(market, starts, out: Path) -> None:
    """The holds at each gross, and cash: the bar each arm must clear at its own gross."""
    rows = []
    for s in starts:
        for g in GROSSES:
            for name, fac in ((f"inv_vol_hold_{int(g * 100)}", baselines.scaled(baselines.InverseVolHold, g)),
                              (f"risk_parity_hold_{int(g * 100)}", qs.book("risk_parity", lambda g=g: qs.Fixed(g),
                                                                         qs.ENTRY_ONLY))):
                rows.append(metrics_row(sim.run(fac(), market, s, windows.WINDOW_DAYS), s, name))
        rows.append(metrics_row(sim.run(baselines.Cash(), market, s, windows.WINDOW_DAYS), s, "cash"))
    res = pd.DataFrame(rows)
    flds = fields(market, starts)
    table = pd.concat([ranked(res[res["strategy"] == "cash"], flds, exclude=("cash",)),
                       ranked(res[res["strategy"] == "inv_vol_hold_100"], flds, exclude=("inv_vol_hold",)),
                       ranked(res[~res["strategy"].isin(["cash", "inv_vol_hold_100"])], flds)],
                      ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "windows.csv", index=False)
    for fname in ("no_clone", "default"):
        t = table[table["field"] == fname]
        print(f"\n{fname} field" + ("  (primary)" if fname == "no_clone" else ""))
        print(windows.summarise(t).drop(columns=["invalid_rounds", "wins"]).round(4).to_string())
        for g in GROSSES:
            p = paired(t, f"inv_vol_hold_{int(g * 100)}", HOLD)
            if g != 0.75:
                print(f"  inv_vol_hold_{int(g * 100)} - {HOLD}: {p['diff']:+.3f} (SE {p['se']:.3f}), "
                      f"better in {p['better']}, worse in {p['worse']} of {p['n']}")


# ----------------------------------------------------------------------------- inputs

def desk_inputs(market, starts, real_names: bool) -> dict:
    from agent_replay import load_earnings_events, load_filings, real_name_sources
    from icaif import external, macro

    events = load_earnings_events(market)
    filings = load_filings(market)
    news_dir = None
    if real_names:
        filings, news_dir = real_name_sources(market, starts, filings)
    context, fomc = macro.wide(external.load("yahoo_daily_context")), macro.FomcCalendar.load()
    if fomc is None:
        raise SystemExit("no FOMC calendar in data/external; run tools/enrich_data.py")
    return dict(earnings=EarningsCalendar(events, market.days), filings=filings, news_dir=news_dir,
                scores=compiler.load_daily_scores(), universe_scores=signals.UniverseScores.load(),
                vol=signals.VolForecasts.from_bars(market.info_bars, market.tickers),
                context=context, fomc=fomc,
                earnings_history=EarningsHistory.from_market(events, market))


def ledgers(market, out: Path) -> None:
    """A desk answered in code buys the fallback hold and holds it, in every stage-1 window
    and at a sleeve's gross too, and its journal agrees with its ledger."""
    t0 = time.time()
    starts = windows_of(market, "select") + windows_of(market, "confirm")
    kw = desk_inputs(market, starts, real_names=False)
    closes = market.recent_closes(pd.Timestamp("2100-01-01", tz="America/New_York"), 10 ** 7)
    bad, wrong = [], []
    for gross, g in ((("free",), 0.75), (("fixed", 0.3), 0.3)):
        hold = baselines.scaled(baselines.InverseVolHold, g)
        for s in starts:
            d = V3Desk({"quick": HoldBrain(), "deep": HoldBrain()}, V3Config(gross=gross), **kw)
            got = sim.run(d, market, s, windows.WINDOW_DAYS)
            if not got.ledger.equals(sim.run(hold(), market, s, windows.WINDOW_DAYS).ledger):
                bad.append((gross, str(s)))
            problems = J.verify(d.journal, J.sim_fills(got, market), to_ticker=d.anon.ticker, closes=closes)
            if problems:
                wrong.append((gross, str(s), problems[:2]))
    n = 2 * len(starts)
    print(f"v3 on code vs the inverse-vol hold at its fallback gross: {n - len(bad)} of {n} runs equal "
          f"trade for trade; journal agrees with ledger in {n - len(wrong)} of {n} ({time.time() - t0:.0f}s)")
    if bad or wrong:
        raise SystemExit(f"differ: {bad[:3]}; journal: {wrong[:2]}")


# ----------------------------------------------------------------------------- an arm

def arm_name(args) -> str:
    g = (f"cf{int(round(args.cash_floor * 100))}" if args.cash_floor is not None
         else args.gross.replace(":", ""))
    streams = (f"_s16r{args.run:02d}" if args.design else
               "".join(f"-{s}" for s in sorted(args.drop)))
    return (f"v3_{args.analysts}_{g}{streams}{'_evidence' if args.evidence else ''}"
            f"{'_nocheck' if args.no_self_check else ''}{f'_r{args.repeat}' if args.repeat else ''}")


def resolve(args) -> None:
    """Turn the experiment's names (a cash floor, a design row) into the desk's settings."""
    if args.cash_floor is not None:
        if args.gross != "free":
            raise SystemExit("--cash-floor sets the gross rule; give it or --gross, not both")
        if args.cash_floor not in CASH_FLOORS:
            raise SystemExit(f"--cash-floor is one of {CASH_FLOORS}")
        if args.cash_floor > 0:
            args.gross = f"band:0:{floor_gross(args.cash_floor):g}"
    if args.design:
        if args.drop:
            raise SystemExit("--design picks the streams; give it or --drop, not both")
        if not (args.run and 1 <= args.run <= len(STREAMS16)):
            raise SystemExit(f"--design streams16 needs --run 1..{len(STREAMS16)}")
        args.drop = [s for s in STREAMS if s not in STREAMS16[args.run - 1]]


def estimate(args, n_windows: int, quick: str, deep: str) -> tuple[int, float]:
    calls, usd = 0, 0.0
    asks = [("pm", deep)] + ([] if args.no_self_check else [("pm_check", deep)])
    asks += [("analyst", quick)] * (4 if args.analysts != "none" else 0)
    for kind, model in asks:
        p_in, p_out, _, _ = brains.PRICES[model]
        tin, tout = EST[kind]
        usd += n_windows * (tin * p_in + tout * p_out) / 1e6
        calls += n_windows
    return calls, usd


def run_arm(args, market, starts, out: Path) -> None:
    cfg = V3Config(analysts=args.analysts, gross=parse_gross(args.gross),
                   streams=tuple(s for s in STREAMS if s not in set(args.drop)),
                   evidence=args.evidence, self_check=not args.no_self_check, anonymize=False)
    quick = (args.quick_model, args.quick_effort)
    deep = (args.deep_model, args.deep_effort)
    n_calls, usd = estimate(args, len(starts), quick[0], deep[0])
    print(f"{arm_name(args)}: {len(starts)} windows ({args.split}), ~{n_calls} calls "
          f"(quick {quick}, deep {deep}): ~${usd:.2f} estimated (cache hits are free)")
    if not (args.yes or args.offline):
        print("re-run with --yes to spend it, or --offline to use cached answers only")
        return
    for m in {quick[0], deep[0]}:
        problem = brains.credentials_problem(m)
        if problem and not args.offline:
            raise SystemExit(f"{problem}: every call would fail and the desk would buy the hold")
    t0 = time.time()
    kw = desk_inputs(market, starts, real_names=True)
    live = {"quick": brains.make(*quick, max_calls=args.max_calls),
            "deep": brains.make(*deep, max_calls=args.max_calls)}
    cached = {k: brains.CachedBrain(b, CACHE, offline=args.offline, repeat=args.repeat) for k, b in live.items()}
    name = arm_name(args)
    rows, entries, chain = [], [], []
    for s in starts:
        d = V3Desk(cached, cfg, **kw)
        bought = []

        def strategy(ctx, d=d, bought=bought):
            w = d(ctx)
            if w is not None:
                bought.append(w)
            return w

        res = sim.run(strategy, market, s, windows.WINDOW_DAYS)
        rows.append(metrics_row(res, s, name))
        book = bought[0] if bought else {}
        gross = float(sum(book.values()))
        hold_g = baselines.scaled(baselines.InverseVolHold, gross) if gross > compiler.HELD else baselines.Cash
        rows.append(metrics_row(sim.run(hold_g(), market, s, windows.WINDOW_DAYS), s, "inv_vol_at_arm_gross"))
        for g in GROSSES:
            try:
                bb = BoughtBook(book, g) if book else None
            except ValueError:
                continue                     # a gross this book can't reach under the cap
            if bb is not None:
                rows.append(metrics_row(sim.run(bb, market, s, windows.WINDOW_DAYS), s,
                                        f"arm_names_at_{int(g * 100)}"))
        for f in CASH_FLOORS[1:]:
            capped = capped_book(book, floor_gross(f))
            run = BoughtBook(capped) if capped else baselines.Cash()
            rows.append(metrics_row(sim.run(run, market, s, windows.WINDOW_DAYS), s,
                                    f"arm_floor_{int(round(f * 100))}"))
        for ref, fac in ((HOLD, baselines.scaled(baselines.InverseVolHold, 0.75)),
                         (RULE, qs.CANDIDATES[RULE])):
            rows.append(metrics_row(sim.run(fac(), market, s, windows.WINDOW_DAYS), s, ref))
        rows.append(metrics_row(sim.run(baselines.Cash(), market, s, windows.WINDOW_DAYS), s, "cash"))
        entries.append({"window": str(s), "entry": d.entry, "plan": d.plan, "fallbacks": dict(d.fallbacks)})
        chain += [dict(e, window=str(s)) for e in d.chain]
        print(f"  {s}: gross {gross:.3f}, {sum(v > 0 for v in book.values())} names, "
              f"{d.entry['source'] if d.entry else 'no entry'}"
              f"{' (' + d.entry['final'] + ')' if d.entry and d.entry.get('final') else ''}, "
              f"return {res.metrics()['cumulative_return']:+.4f}")
    res = pd.DataFrame(rows)
    flds = fields(market, starts)
    table = pd.concat([ranked(res[res["strategy"] == "cash"], flds, exclude=("cash",)),
                       ranked(res[res["strategy"] != "cash"], flds)], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "windows.csv", index=False)
    (out / "entries.jsonl").write_text("\n".join(json.dumps(e, default=str) for e in entries))
    (out / "chain.jsonl").write_text("\n".join(json.dumps(e, default=str) for e in chain))
    (out / "config.json").write_text(json.dumps({"arm": name, "split": args.split, "windows": [str(s) for s in starts],
                                                 "config": {k: getattr(cfg, k) for k in
                                                            ("analysts", "gross", "streams", "evidence",
                                                             "self_check", "tiers", "slots")},
                                                 "quick": quick, "deep": deep, "repeat": args.repeat,
                                                 "cash_floor": args.cash_floor or 0.0,
                                                 "design": args.design, "design_run": args.run,
                                                 "confirm_choice": args.confirm_choice}, indent=1, default=str))
    report(table, name, [HOLD, "inv_vol_at_arm_gross", RULE, "cash"])
    print("\nthe arm's names at each gross (rescaled, no calls) minus the hold:")
    for fname in ("no_clone", "default"):
        t = table[table["field"] == fname]
        cells = []
        for g in GROSSES:
            col = f"arm_names_at_{int(g * 100)}"
            if col in set(t["strategy"]):
                p = paired(t, col, HOLD)
                cells.append(f"{int(g * 100)}%: {p['diff']:+.3f} (SE {p['se']:.3f}, n {p['n']})")
        print(f"  {fname}: " + "; ".join(cells))
    falls = {}
    for e in entries:
        for k, v in e["fallbacks"].items():
            falls[k] = falls.get(k, 0) + v
    print("\nfallbacks: " + (", ".join(f"{k} {v}" for k, v in sorted(falls.items())) or "none"))
    finals = pd.Series([e["entry"]["final"] if e["entry"] else None for e in entries]).value_counts(dropna=False)
    print("entries: " + ", ".join(f"{k} {v}" for k, v in finals.items()))
    for k, b in live.items():
        c = cached[k]
        print(f"{b.name}: {len(b.records)} calls, ${b.cost():.2f}; cache hits {c.hits}, misses {c.misses}; "
              "by role: " + ", ".join(f"{r} ${v:.2f}" for r, v in sorted(brains.cost_by_role(b).items())))
    print(f"\nlog: {out}   ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["round0", "ledgers", "arm"])
    ap.add_argument("--split", choices=["select", "confirm"], default="select")
    ap.add_argument("--confirm-choice", default=None,
                    help="the commit that recorded the choice; required with --split confirm")
    ap.add_argument("--analysts", choices=["none", "reports_raw", "reports_only"], default="none")
    ap.add_argument("--gross", default="free", help="free, band:LO:HI or fixed:G")
    ap.add_argument("--cash-floor", type=float, default=None,
                    help="the least cash the PM must hold, one of v3.CASH_FLOORS (0.4 = at least 40%%)")
    ap.add_argument("--design", choices=["streams16"], default=None,
                    help="run a row of the streams factorial (stage1_doe.md, round 2)")
    ap.add_argument("--run", type=int, default=None, help="with --design: the row, 1-16")
    ap.add_argument("--drop", nargs="*", default=[], choices=list(STREAMS))
    ap.add_argument("--evidence", action="store_true", help="the PM's prompt carries our backtest findings")
    ap.add_argument("--no-self-check", action="store_true")
    ap.add_argument("--repeat", type=int, default=0, help="ask again under repeat N's cache keys")
    ap.add_argument("--windows", type=int, default=None, help="the split's first K windows only")
    ap.add_argument("--on", default=None, help="one window of the split, by its first session")
    ap.add_argument("--quick-model", default="gemini-2.5-flash", choices=brains.ALLOWED_MODELS)
    ap.add_argument("--quick-effort", default="medium")
    ap.add_argument("--deep-model", default="gemini-2.5-pro", choices=brains.ALLOWED_MODELS)
    ap.add_argument("--deep-effort", default="high")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    resolve(args)
    if args.split == "confirm" and args.command == "arm" and not args.confirm_choice:
        raise SystemExit("--split confirm scores the committed choice once: name its commit "
                         "with --confirm-choice")
    t0 = time.time()
    market = markets.research_market()
    if args.command == "ledgers":
        return ledgers(market, OUT / "ledgers")
    starts = windows_of(market, args.split)
    if args.on:
        on = date.fromisoformat(args.on)
        if on not in starts:
            raise SystemExit(f"{on} starts no {args.split} window")
        starts = [on]
    elif args.windows:
        starts = starts[: args.windows]
    if args.command == "round0":
        round0(market, starts, OUT / (args.tag or f"round0_{args.split}"))
        print(f"\n({time.time() - t0:.0f}s)")
        return
    tag = args.tag or f"{arm_name(args)}_{args.split}_{len(starts)}w"
    run_arm(args, market, starts, OUT / tag)


if __name__ == "__main__":
    main()
