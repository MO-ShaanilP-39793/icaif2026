"""Replay the agent desk through competition windows, ranked like every other strategy.

    .venv/bin/python tools/agent_replay.py                          # rule brain: free
    .venv/bin/python tools/agent_replay.py --ledgers-only           # the same check, ~3 min
    .venv/bin/python tools/agent_replay.py --brain claude --max-calls 60 --yes
    .venv/bin/python tools/agent_replay.py --brain claude --model gemini-2.5-pro --yes   # Gemini API
    .venv/bin/python tools/agent_replay.py --desk v2 --ledgers-only  # v2 on code = the hold

**`--desk v2`** (`agents/v2.py`) runs the morning chain. On the rule brain every role is
answered in code (`v2.HoldBrain`) and the desk must trade exactly as `inv_vol_hold_75`;
the run stops if it doesn't. With `--brain claude` the quick roles ask `--model` at
`--quick-effort` and the risk manager and PM ask `--deep-model` at `--deep-effort`, and
the log adds each role's calls, fallbacks, payload sizes and cost (`chain.jsonl`).

The rule brain is the sanity check: its desk must tie `q_riskparity_entry_regime` (the quant
candidate it stands for) in every window, or the desk's plumbing, not its judgement,
is what any LLM result would measure. The run stops if it doesn't. The desk reads every
signal an LLM desk would (walk-forward scores, the universe's ranking, HAR vol, earnings)
and its own journal,
so the check covers the plumbing those inputs added too. Since step 5 it reads the 8-K
snapshot as well, so every filing for a held name wakes its analyst (the rule holds).
`--ledgers-only` runs just
that check, and two stricter ones: the two ledgers equal trade for trade in every
window, not only their scores, and the desk's journal agrees with its own ledger in
every window (`journal.verify`: each fill to the cent, each held name's entry, cost and
peak), without ranking anything against the field (a fifth of the time).

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

from icaif import baselines, compiler, data, markets, quant_strategies as qs, sim, windows  # noqa: E402
from icaif.agents import brains, signals  # noqa: E402
from icaif.agents import journal as J  # noqa: E402
from icaif.agents.desk import DeskConfig, EarningsCalendar, desk  # noqa: E402
from icaif.agents.free import FreeDesk  # noqa: E402
from icaif.agents.triggers import EarningsHistory  # noqa: E402
from icaif.agents.v2 import HoldBrain, V2Config, V2Desk  # noqa: E402

OUT = data.ROOT / "output" / "agent"
# Rough per-call size: ~9k input tokens (prompt + 30-name observation), ~2.5k output
# with thinking. Measured costs replace this after the first calls (brain.cost()).
EST_IN, EST_OUT = 9_000, 2_500


def load_earnings_events(market) -> pd.DataFrame:
    path = sorted((data.ROOT / "data" / "external").glob("earnings_2*.parquet"))[-1]
    events = pd.read_parquet(path)
    return events[events["ticker"].isin(market.tickers)]


def load_earnings(market) -> EarningsCalendar:
    return EarningsCalendar(load_earnings_events(market), market.days)


# v2's calls a window and their size per role. Input: the system prompt and payload
# measured on the free run of 2026-01-21 with real names (`--desk v2 --on 2026-01-21
# --real-names`), plus what the earlier roles would write (about 2,000 characters a
# report, 2,500 a debate turn, 1,500 each for the proposal, code's compile, the review
# and the event case, 3,000 of reasons for the reflection), at 3.5 characters a token.
# Output: the six Grok 4.7 free-desk runs averaged $0.067 a call on 31,357-character
# prompts, which leaves about 7,000 output tokens a call at effort high, reasoning
# included; effort medium is taken as half, unmeasured. The earnings analyst is asked
# only on mornings a name reports (all 15 in Jan 21's window, an earnings season; about
# 10 is typical). Trigger rounds were 30 in that window and 23 a window on average over
# all 167 (v1's rule desk, the same triggers); reflection runs on mornings with something
# settled (11 there). Measured costs replace these after the first paid window.
V2_CALLS = {  # role: (calls a window, input tokens a call, output tokens a call)
    "market": (15, 1_600, 3_500), "earnings": (10, 2_400, 3_500), "news": (15, 7_000, 3_500),
    "quant": (15, 5_900, 3_500), "bull": (30, 8_500, 3_500), "bear": (30, 9_200, 3_500),
    "trader": (15, 10_700, 3_500), "risk": (15, 12_100, 7_000), "pm": (15, 12_000, 7_000),
    "event": (25, 3_000, 3_500), "event_pm": (25, 3_400, 7_000), "reflect": (12, 6_600, 7_000)}
V2_DEEP = ("risk", "pm", "event_pm", "reflect")


def v2_estimate(n_windows: int, quick: str, deep: str) -> tuple[int, float, dict]:
    """(calls, USD, USD by role) for `n_windows` windows of the v2 morning chain."""
    by_role = {}
    for role, (n, tin, tout) in V2_CALLS.items():
        p_in, p_out, _, _ = brains.PRICES[deep if role in V2_DEEP else quick]
        by_role[role] = n_windows * n * (tin * p_in + tout * p_out) / 1e6
    calls = n_windows * sum(n for n, _, _ in V2_CALLS.values())
    return calls, sum(by_role.values()), by_role


def load_filings(market) -> pd.DataFrame:
    """Every event 8-K for the 30 names (item codes and acceptance times only: a filing's
    text names the company, so replays never carry it)."""
    path = sorted((data.ROOT / "data" / "external").glob("edgar_8k_*.parquet"))[-1]
    events = pd.read_parquet(path)
    return events[events["ticker"].isin(market.tickers)].reset_index(drop=True)


def eight_k_wakes(filings: pd.DataFrame, market, starts) -> float:
    """Mean 8-K wake-ups a window: filings accepted between its entry and its last round.

    An upper bound on the analyst calls the trigger adds (two filings for one name in one
    gap are one call; a name sold out is not woken), so a cost estimate made from it is
    on the safe side.
    """
    from icaif import calendar

    n = []
    for s in starts:
        days = market.days[market.days.index(s): market.days.index(s) + windows.WINDOW_DAYS]
        lo = calendar.rounds_for(days[0])[0]["deadline"]
        hi = calendar.rounds_for(days[-1])[-1]["deadline"]
        n.append(int(((filings["accepted"] > lo) & (filings["accepted"] <= hi)).sum()))
    return float(sum(n) / max(len(n), 1))


def real_name_sources(market, starts, codes: pd.DataFrame):
    """The 8-Ks with their texts and the headline archive for a real-names replay.

    Both are fetched per window (`tools/replay_sources.py`), and a window outside what
    was fetched stops the run: it would replay as a desk shown no headlines and no
    filing text, and its score would read as the LLM ignoring news it never saw.
    """
    from icaif import alpaca_news, calendar
    from icaif import filings as F
    from icaif.agents import observe

    texts = []
    for s in starts:
        days = market.days[market.days.index(s): market.days.index(s) + windows.WINDOW_DAYS]
        first = calendar.rounds_for(days[0])[0]["deadline"]
        last = calendar.rounds_for(days[-1])[-1]["deadline"]
        gap = alpaca_news.uncovered(first - pd.Timedelta(hours=observe.HEADLINE_HOURS), last,
                                    list(market.tickers))
        if gap:
            raise SystemExit(f"window {s}: {gap}")
        try:
            texts.append(F.load_texts(first - pd.Timedelta(days=7), last))
        except FileNotFoundError as err:
            raise SystemExit(f"window {s}: {err}") from err
    texts = pd.concat(texts, ignore_index=True)
    texts = texts[texts["ticker"].isin(market.tickers)]
    print(f"real names: {len(texts)} 8-Ks with {texts['text'].notna().sum()} texts; "
          f"headlines from {alpaca_news.ARCHIVE}")
    return F.merge(codes, texts), alpaca_news.ARCHIVE


def name_of(args) -> str:
    if args.desk == "free":
        return f"free_{args.arm}_{args.brain}{'_noregime' if args.no_regime else ''}"
    if args.desk == "v2":
        return f"v2_{args.brain}"
    return (f"desk_{args.brain}{'_unanchored' if args.unanchored else ''}"
            f"{'_noevidence' if args.no_evidence else ''}{'_noregime' if args.no_regime else ''}")


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
    ap.add_argument("--real-names", action="store_true",
                    help="do not anonymise (post-cutoff windows only); reads headlines and 8-K texts")
    ap.add_argument("--on", default=None,
                    help="one window starting on this session (any session, as the leaderboard's are)")
    ap.add_argument("--desk", choices=["levered", "free", "v2"], default="levered",
                    help="levered: roles pull levers the compiler turns into weights; free: the "
                         "model writes the whole book each morning (icaif/agents/free.py); v2: "
                         "analysts, debate, trader, risk manager and PM (icaif/agents/v2.py)")
    ap.add_argument("--deep-model", default=None, choices=brains.ALLOWED_MODELS,
                    help="v2: the risk manager's and PM's model (default --model)")
    ap.add_argument("--quick-effort", default=None, help="v2: the quick roles' effort (default V2Config)")
    ap.add_argument("--deep-effort", default=None, help="v2: the deep roles' effort (default V2Config)")
    ap.add_argument("--arm", choices=["blank", "informed"], default="blank",
                    help="free desk: blank (no evidence, no rule) or informed")
    ap.add_argument("--unanchored", action="store_true",
                    help="levered desk: withhold the rule's answer (kept as the fallback only)")
    ap.add_argument("--no-evidence", action="store_true",
                    help="with --unanchored: the prompts carry no backtest findings")
    ap.add_argument("--no-regime", action="store_true",
                    help="hide the regime model's read (turbulence odds, persistence) from every role")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="answer only from the cache")
    ap.add_argument("--yes", action="store_true", help="confirm spending on a claude replay")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ledgers-only", action="store_true",
                    help="rule brain: check its ledger equals the candidate's in every window, then stop")
    args = ap.parse_args()
    if args.ledgers_only and (args.brain != "rule" or args.desk == "free"):
        raise SystemExit("--ledgers-only checks a code-answered desk (levered or v2); it takes no "
                         "--brain claude or --desk free")
    if args.desk == "v2" and (args.unanchored or args.no_evidence or args.no_regime):
        raise SystemExit("v2 fixes these: no rule shown, evidence for the risk manager only, no "
                         "regime label")
    if args.no_evidence and not args.unanchored:
        raise SystemExit("--no-evidence needs --unanchored: the anchored prompt calls the rule "
                         "the backtested one")
    if args.unanchored and args.desk != "levered":
        raise SystemExit("--unanchored is the levered desk's; the free desk's blank arm has no rule")

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
    if args.on:
        on = date.fromisoformat(args.on)
        if on not in market.days:
            raise SystemExit(f"{on} is not a session in the market")
        starts = [on]
    cfg = DeskConfig(review=not args.no_review, events=not args.no_events,
                     anonymize=not args.real_names, anchored=not args.unanchored,
                     evidence=not args.no_evidence, regime=not args.no_regime)
    events = load_earnings_events(market)
    earnings = EarningsCalendar(events, market.days)
    filings = load_filings(market)
    news_dir = None
    if args.real_names:
        filings, news_dir = real_name_sources(market, starts, filings)
    scores = compiler.load_daily_scores()
    universe_scores = signals.UniverseScores.load()
    # The walk-forward over the bars the windows trade on: forecasts from its first
    # fittable quarter (2016-07), each made before its session opened.
    har = signals.VolForecasts.from_bars(market.info_bars, market.tickers)

    tag = args.tag or f"{name_of(args)}_{'noreview_' if args.no_review else ''}{len(starts)}w"
    log_dir = OUT / tag

    v2cfg = V2Config(anonymize=not args.real_names)
    tiers = dict(v2cfg.tiers)
    if args.desk == "v2" and args.brain == "claude":
        deep = args.deep_model or args.model
        tiers = {"quick": (args.model, args.quick_effort or tiers["quick"][1]),
                 "deep": (deep, args.deep_effort or tiers["deep"][1])}
        v2cfg.tiers = tiers
    v2_brains = None
    if args.brain == "rule":
        make = brains.RuleBrain
        live_brains = []
        if args.desk == "v2":
            v2_brains = {"quick": HoldBrain(), "deep": HoldBrain()}
    elif args.desk == "v2":
        n_calls, est, by_role = v2_estimate(len(starts), tiers["quick"][0], tiers["deep"][0])
        print(f"{len(starts)} windows, ~{n_calls} calls (quick {tiers['quick']}, deep {tiers['deep']}): "
              f"~${est:.2f} (cache hits are free)")
        print("  by role: " + ", ".join(f"{r} ${c:.2f}" for r, c in by_role.items()))
        if not (args.yes or args.offline):
            print("re-run with --yes to spend it, or --offline to use cached answers only")
            return
        for m_ in {tiers["quick"][0], tiers["deep"][0]}:
            problem = brains.credentials_problem(m_)
            if problem and not args.offline:
                raise SystemExit(f"{problem}: every call would fail and the desk would hold")
        live_brains = [brains.make(*tiers[k], max_calls=args.max_calls) for k in ("quick", "deep")]
        v2_brains = {k: brains.CachedBrain(b, OUT / "cache", offline=args.offline)
                     for k, b in zip(("quick", "deep"), live_brains)}
        make = lambda: v2_brains["deep"]  # noqa: E731 - for the hit counts below
    else:
        per_window = (windows.WINDOW_DAYS if args.desk == "free" else
                      1 + (0 if args.no_review else 14)
                      + (0 if args.no_events else 2 + eight_k_wakes(filings, market, starts)))
        n_calls = per_window * len(starts)
        if args.max_calls:
            n_calls = min(n_calls, args.max_calls)
        p_in, p_out, _, _ = brains.PRICES[args.model]
        est = n_calls * (EST_IN * p_in + EST_OUT * p_out) / 1e6
        n_calls = int(round(n_calls))
        print(f"{len(starts)} windows, ~{n_calls} calls to {args.model} at effort "
              f"{args.effort}: ~${est:.0f} (cache hits are free)")
        if not (args.yes or args.offline):
            print("re-run with --yes to spend it, or --offline to use cached answers only")
            return
        problem = brains.credentials_problem(args.model)
        if problem and not args.offline:
            raise SystemExit(f"{problem}: every call would fall back to the rule, and the "
                             "replay would score the rule desk as the LLM's")
        shared = brains.make(args.model, args.effort, max_calls=args.max_calls)
        cache = brains.CachedBrain(shared, OUT / "cache", offline=args.offline)
        make = lambda: cache  # noqa: E731 - one brain across windows, so the budget is global
        live_brains = [shared]

    # Macro (the market, VIX, yields, sectors as of the prior close; FOMC timing), for every
    # desk. Until 2026-10-06 this tool never passed it, so every replay before then,
    # Run C's included, decided without the macro block its prompt describes as "may
    # include", and nothing said it was missing. A missing file raises here rather than
    # replay without it again.
    from icaif import external, macro

    context, fomc = macro.wide(external.load("yahoo_daily_context")), macro.FomcCalendar.load()
    if fomc is None:
        raise SystemExit("no FOMC calendar in data/external; run tools/enrich_data.py")
    history = EarningsHistory.from_market(events, market) if args.desk == "v2" else None

    def v2_desk():
        return V2Desk(v2_brains, v2cfg, earnings_history=history, scores=scores, earnings=earnings,
                      vol=har, filings=filings, universe_scores=universe_scores, news_dir=news_dir,
                      context=context, fomc=fomc)

    if args.ledgers_only and args.desk == "v2":
        bad, wrong, falls = [], [], {}
        closes = market.recent_closes(pd.Timestamp("2100-01-01", tz="America/New_York"), 10 ** 7)
        hold = baselines.scaled(baselines.InverseVolHold, 0.75)
        for s in starts:
            d = v2_desk()
            got = sim.run(d, market, s, windows.WINDOW_DAYS)
            if not got.ledger.equals(sim.run(hold(), market, s, windows.WINDOW_DAYS).ledger):
                bad.append(str(s))
            problems = J.verify(d.journal, J.sim_fills(got, market), to_ticker=d.anon.ticker, closes=closes)
            if problems:
                wrong.append((str(s), problems[:3]))
            for k, v in d.fallbacks.items():
                falls[k] = falls.get(k, 0) + v
        print(f"v2 on code vs inv_vol_hold_75: {len(starts) - len(bad)} of {len(starts)} windows equal "
              f"trade for trade; its journal agrees with its ledger in {len(starts) - len(wrong)} of "
              f"{len(starts)} ({time.time() - t0:.0f}s)")
        print("fallbacks: " + ", ".join(f"{k} {v}" for k, v in sorted(falls.items())))
        if bad:
            raise SystemExit(f"ledgers differ in {len(bad)} windows (first {bad[:3]})")
        if wrong:
            raise SystemExit(f"journal and ledger disagree in {len(wrong)} windows (first {wrong[:2]})")
        return

    if args.ledgers_only:
        bad, wrong, woken = [], [], []
        closes = market.recent_closes(pd.Timestamp("2100-01-01", tz="America/New_York"), 10 ** 7)
        for s in starts:
            d = desk(make, cfg, scores=scores, earnings=earnings, vol=har, filings=filings,
                     universe_scores=universe_scores, news_dir=news_dir, context=context, fomc=fomc)()
            got = sim.run(d, market, s, windows.WINDOW_DAYS)
            want = sim.run(qs.CANDIDATES["q_riskparity_entry_regime"](), market, s, windows.WINDOW_DAYS)
            if not got.ledger.equals(want.ledger):
                bad.append(str(s))
            problems = J.verify(d.journal, J.sim_fills(got, market), to_ticker=d.anon.ticker, closes=closes)
            if problems:
                wrong.append((str(s), problems[:3]))
            for e in d.log:
                if e["role"] == "event":
                    woken.append(e)
        causes = pd.Series([cause for e in woken for why in e.get("triggers", {}).values()
                            for cause in (["8-K"] if "new 8-K" in why else [])
                            + (["earnings"] if "earnings" in why else [])
                            + (["3-sigma move"] if "sigmas" in why else [])])
        print(f"rule desk vs q_riskparity_entry_regime: {len(starts) - len(bad)} of {len(starts)} "
              f"windows equal trade for trade; its journal agrees with its ledger in "
              f"{len(starts) - len(wrong)} of {len(starts)} ({time.time() - t0:.0f}s)")
        print(f"analyst calls: {len(woken)} ({len(woken) / len(starts):.1f} a window); names woken: "
              + ", ".join(f"{k} {v}" for k, v in causes.value_counts().items()))
        if bad:
            raise SystemExit(f"ledgers differ in {len(bad)} windows (first {bad[:3]})")
        if wrong:
            raise SystemExit(f"journal and ledger disagree in {len(wrong)} windows (first {wrong[:2]})")
        return

    desks = []
    desk_kw = dict(scores=scores, earnings=earnings, vol=har, filings=filings,
                   universe_scores=universe_scores, news_dir=news_dir, context=context, fomc=fomc)

    def factory():
        d = (v2_desk() if args.desk == "v2" else
             FreeDesk(make(), args.arm, cfg, **desk_kw) if args.desk == "free"
             else desk(make, cfg, **desk_kw)())
        desks.append(d)
        return d

    field = windows.run_field(baselines.FIELD, market, starts)
    cands = {"inv_vol_hold_75": baselines.scaled(baselines.InverseVolHold, 0.75),
             "q_riskparity_entry_regime": qs.CANDIDATES["q_riskparity_entry_regime"],
             name_of(args): factory}
    res = pd.concat([windows.rank_against_field(windows.run_field({n: f}, market, starts), field)
                     for n, f in cands.items()], ignore_index=True)
    piv = res.pivot(index="window", columns="strategy", values="overall_score")
    name = name_of(args)

    if args.brain == "rule" and args.desk == "v2" and not (piv[name] == piv["inv_vol_hold_75"]).all():
        bad = piv.index[piv[name] != piv["inv_vol_hold_75"]].tolist()
        raise SystemExit(f"v2 on code differs from inv_vol_hold_75 in {len(bad)} windows "
                         f"(first {bad[:3]}): the desk's plumbing is off; not reporting")
    if args.brain == "rule" and args.desk == "levered" and not (piv[name] == piv["q_riskparity_entry_regime"]).all():
        bad = piv.index[piv[name] != piv["q_riskparity_entry_regime"]].tolist()
        raise SystemExit(f"rule desk differs from q_riskparity_entry_regime in {len(bad)} windows "
                         f"(first {bad[:3]}): the desk's plumbing is off; not reporting")

    log = [dict(e, window=str(w)) for w, d in zip(starts, desks) for e in d.log]
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "log.jsonl").write_text("\n".join(json.dumps(e, default=str) for e in log))
    res.to_csv(log_dir / "windows.csv", index=False)

    summ = windows.summarise(res).drop(columns=["invalid_rounds"])
    print(summ.round(4).to_string())
    for ref in ("inv_vol_hold_75", "q_riskparity_entry_regime"):
        d = piv[name] - piv[ref]
        print(f"{name} - {ref}: {d.mean():+.3f} (SE {d.std() / len(d) ** 0.5:.3f}), "
              f"better in {(d < 0).mean():.0%}, worse in {(d > 0).mean():.0%} of {len(d)} windows")
    if args.desk == "v2":
        chain = [dict(e, window=str(w)) for w, d in zip(starts, desks) for e in d.chain]
        (log_dir / "chain.jsonl").write_text("\n".join(json.dumps(e, default=str) for e in chain))
        ch = pd.DataFrame(chain)
        ch["role_kind"] = ch["role"].where(~ch["role"].str[-1].str.isdigit(),
                                           ch["role"].str.rsplit("_", n=1).str[0])   # bull_2 -> bull
        print("\nv2 calls by role and source:")
        print(ch.groupby(["role_kind", "source"]).size().unstack(fill_value=0).to_string())
        sizes = ch.groupby("role_kind")[["system_chars", "payload_chars", "latency_s"]].agg(["mean", "max"])
        print("\nprompt sizes (characters) and latency by role:")
        print(sizes.round(1).to_string())
        falls = {}
        for d in desks:
            for k, v in d.fallbacks.items():
                falls[k] = falls.get(k, 0) + v
        print("fallbacks: " + (", ".join(f"{k} {v}" for k, v in sorted(falls.items())) or "none"))
        for b in live_brains:
            print(f"{b.name} by role: " + ", ".join(f"{r} ${c:.2f}" for r, c in
                                                     sorted(brains.cost_by_role(b).items())))
    lg = pd.DataFrame(log)
    if len(lg) and args.desk != "v2":
        print("\ndecisions by role and source:")
        print(lg.groupby(["role", "source"]).size().to_string())
        trims = [t for e in log for t in (e["decision"].get("trim") or [])]
        trims += [c for e in log for c in e["decision"].get("calls", []) if c["action"] == "trim"]
        if trims:
            print("trims by cause: " + ", ".join(
                f"{k} {v}" for k, v in pd.Series([t["cause"] for t in trims]).value_counts().items()))
        print(f"answers that differ from the rule: {(~lg['same_as_rule'].astype(bool)).sum()} "
              f"of {len(lg)}")
    caches = list(v2_brains.values()) if v2_brains and args.brain == "claude" else [make()] * len(live_brains)
    for b, c in zip(live_brains, caches):
        print(f"\n{b.name}: {len(b.records)} calls, ${b.cost():.2f}; cache hits "
              f"{c.hits}, misses {c.misses}")
    print(f"\nlog: {log_dir}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
