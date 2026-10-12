"""Stage 2, variation B: does the PM, woken by code and only able to reduce, beat holding?

    .venv/bin/python tools/chain_replay.py ledgers          # code-answered B = the hold (free)
    .venv/bin/python tools/chain_replay.py run              # prints the cost estimate and stops
    .venv/bin/python tools/chain_replay.py run --yes        # official4, the four Earnings season windows
    .venv/bin/python tools/chain_replay.py run --on 2025-04-11 --yes
    .venv/bin/python tools/chain_replay.py stage1 --overlay 0.5 --offline   # stage 1 alone, for the board

Each window runs twice from the same entry: stage 1 (`V3Desk`, entry then hold) and B
(`ChainDesk`). Both use Official's configuration (the reports-only analysts, every stream
but 8-K filings, a 10% overlay, Gemini 3.1 Pro), and the cache, so the entry is asked
once and B's only difference is what it does after entry. Any score gap is the chain's.

The windows are `suites.official4`, kept out of stage 1 for this: each opens as a
quarter's reporting starts, as the Official phase does. Four windows cannot settle a
small gap (a window's score has an SE near 0.24); what they can show is whether B's
interventions paid, one by one (`settled`), and whether it stays quiet when nothing
happens.
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

from icaif import baselines, data, markets, sim, suites, windows  # noqa: E402
from icaif.agents import brains  # noqa: E402
from icaif.agents import journal as J  # noqa: E402
from icaif.agents.chain import ChainConfig, ChainDesk, ChainHoldBrain  # noqa: E402
from icaif.agents.v3 import STREAMS, V3Config, V3Desk  # noqa: E402
from entry_replay import (CACHE, HOLD, desk_inputs, fields, metrics_row, paired, ranked,  # noqa: E402
                          report, windows_of)

OUT = data.ROOT / "output" / "chain"
MODEL = "gemini-3.1-pro-preview"
# Official's entry (stage1_doe.md): reports only, every stream but filings, 10% overlay.
OFFICIAL = dict(analysts="reports_only", gross=("free",), streams=tuple(s for s in STREAMS if s != "filings"),
                evidence=False, objective="contest", overlay=0.10, self_check=True, anonymize=False)
# Tokens a call, unmeasured until the first run: the entry's as entry_replay measured
# them; a chain wake about 5,000 in (prompt 1,300, payload 3,000-4,000) and 4,000 out
# at high thinking. Measured costs replace these (`brain.cost()`).
EST = {"entry": 0.20, "chain_call": (5_000, 4_000), "chain_calls_a_window": 30}


def starts_of(market, suite: str) -> list[date]:
    s = suites.get(suite)
    out = []
    for first, last in s.windows:
        a = date.fromisoformat(first)
        i = market.days.index(a)
        span = market.days[i:i + windows.WINDOW_DAYS]
        if len(span) < windows.WINDOW_DAYS or str(span[-1]) != last:
            raise SystemExit(f"window {a}: its 15 sessions end {span[-1] if span else None}, not {last}")
        out.append(a)
    return out


def ledgers(market, out: Path) -> None:
    """B answered in code, holding at every wake, buys the fallback and holds it in every
    window it will be judged on and every stage-1 window, and its journal agrees."""
    t0 = time.time()
    starts = starts_of(market, "official4") + windows_of(market, "select") + windows_of(market, "confirm")
    kw = desk_inputs(market, starts, real_names=False)
    closes = market.recent_closes(pd.Timestamp("2100-01-01", tz="America/New_York"), 10 ** 7)
    hold = baselines.scaled(baselines.InverseVolHold, OFFICIAL["overlay"])
    bad, wrong, wakes = [], [], 0
    for s in starts:
        d = ChainDesk({"quick": ChainHoldBrain(), "deep": ChainHoldBrain()}, ChainConfig(**OFFICIAL), **kw)
        got = sim.run(d, market, s, windows.WINDOW_DAYS)
        if not got.ledger.equals(sim.run(hold(), market, s, windows.WINDOW_DAYS).ledger):
            bad.append(str(s))
        problems = J.verify(d.journal, J.sim_fills(got, market), to_ticker=d.anon.ticker, closes=closes)
        if problems:
            wrong.append((str(s), problems[:2]))
        wakes += len(d.wakes)
    n = len(starts)
    print(f"B on code vs the inverse-vol hold at {OFFICIAL['overlay']:g}: {n - len(bad)} of {n} windows equal "
          f"trade for trade; journal agrees in {n - len(wrong)} of {n}; {wakes} wakes held "
          f"({time.time() - t0:.0f}s)")
    if bad or wrong:
        raise SystemExit(f"differ: {bad[:3]}; journal: {wrong[:2]}")


def settle(d: ChainDesk, market, end) -> list[dict]:
    """Each reduction to the window's last close: the names sold times what they did after
    the fill, in bp of the PM's book (positive: the sale saved that much). Fees apart."""
    last = market.recent_closes(end, 1).iloc[-1]
    out = []
    for iv in d.interventions:
        e = next((x for x in d.journal.rounds if x["key"] == iv["key"]), None)
        fill = (e or {}).get("fill") or {}
        for t, sold in iv["sold_pm"].items():
            px = (fill.get("prices") or {}).get(t)
            if not px:
                continue
            move = float(last[t]) / px - 1
            out.append({"day": iv["day"], "round": iv["round"], "name": t, "sold_pm": round(sold, 4),
                        "move_to_end": round(move, 4), "saved_bp": round(-sold * move * 1e4, 1),
                        "counts": iv["counts"]})
    return out


def run(args, market, starts, out: Path) -> None:
    calls = len(starts) * EST["chain_calls_a_window"]
    p_in, p_out, _, _ = brains.PRICES[MODEL]
    usd = len(starts) * EST["entry"] + calls * (EST["chain_call"][0] * p_in + EST["chain_call"][1] * p_out) / 1e6
    print(f"B on {len(starts)} windows ({', '.join(map(str, starts))}): entries (~${EST['entry']:.2f} each "
          f"unless cached) and ~{calls} PM wakes: ~${usd:.2f} estimated")
    if not (args.yes or args.offline):
        print("re-run with --yes to spend it, or --offline to use cached answers only")
        return
    problem = brains.credentials_problem(MODEL)
    if problem and not args.offline:
        raise SystemExit(f"{problem}: every call would fail and both desks would buy the hold")
    t0 = time.time()
    kw = desk_inputs(market, starts, real_names=True)
    live = {"quick": brains.make(MODEL, "medium", max_calls=args.max_calls),
            "deep": brains.make(MODEL, "high", max_calls=args.max_calls)}
    cached = {k: brains.CachedBrain(b, CACHE, offline=args.offline) for k, b in live.items()}
    rows, records, chain = [], [], []
    for s in starts:
        one = V3Desk(cached, V3Config(**OFFICIAL), **kw)
        a = sim.run(one, market, s, windows.WINDOW_DAYS)
        two = ChainDesk(cached, ChainConfig(**OFFICIAL, budget=args.budget), **kw)
        b = sim.run(two, market, s, windows.WINDOW_DAYS)
        rows += [metrics_row(a, s, "stage1"), metrics_row(b, s, "chain_b")]
        for ref, fac in ((HOLD, baselines.scaled(baselines.InverseVolHold, 0.75)),
                         ("inv_vol_hold_10", baselines.scaled(baselines.InverseVolHold, 0.10))):
            rows.append(metrics_row(sim.run(fac(), market, s, windows.WINDOW_DAYS), s, ref))
        rows.append(metrics_row(sim.run(baselines.Cash(), market, s, windows.WINDOW_DAYS), s, "cash"))
        if two.entry != one.entry:
            raise SystemExit(f"{s}: the two desks entered differently; the comparison would not be the chain's")
        end = b.valuation_times[-1]
        settled = settle(two, market, end)
        kinds = pd.Series([e["kind"] for w in two.wakes for e in w["events"]]).value_counts().to_dict()
        records.append({"window": str(s), "entry": one.entry, "plan": one.plan, "wakes": two.wakes,
                        "interventions": two.interventions, "settled": settled,
                        "fallbacks": dict(two.fallbacks), "decisions": [x for x in two.log if x["role"] == "chain_pm"]})
        chain += [dict(e, window=str(s)) for e in two.chain]
        ma, mb = a.metrics(), b.metrics()
        print(f"  {s}: stage1 return {ma['cumulative_return']:+.4f}, B {mb['cumulative_return']:+.4f}; "
              f"{len(two.wakes)} wakes ({', '.join(f'{k} {v}' for k, v in kinds.items()) or 'mornings only'}), "
              f"{len(two.interventions)} reductions, saved {sum(x['saved_bp'] for x in settled):+.0f} bp of the PM's book")
    res = pd.DataFrame(rows)
    flds = fields(market, starts)
    table = pd.concat([ranked(res[res["strategy"] == "cash"], flds, exclude=("cash",)),
                       ranked(res[res["strategy"] != "cash"], flds)], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "windows.csv", index=False)
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in records))
    (out / "chain.jsonl").write_text("\n".join(json.dumps(e, default=str) for e in chain))
    (out / "config.json").write_text(json.dumps({"windows": [str(s) for s in starts], "official": OFFICIAL,
                                                 "model": MODEL, "budget": args.budget}, indent=1, default=str))
    report(table, "chain_b", ["stage1", HOLD, "inv_vol_hold_10", "cash"])
    for fname in ("no_clone", "default"):
        t = table[table["field"] == fname]
        p = paired(t, "chain_b", "stage1")
        print(f"\n{fname}: B minus stage 1 {p['diff']:+.3f} (SE {p['se']:.3f}), better in {p['better']}, "
              f"worse in {p['worse']} of {p['n']}")
    falls = {}
    for r in records:
        for k, v in r["fallbacks"].items():
            falls[k] = falls.get(k, 0) + v
    print("fallbacks: " + (", ".join(f"{k} {v}" for k, v in sorted(falls.items())) or "none"))
    for k, br in live.items():
        c = cached[k]
        print(f"{k}: {len(br.records)} calls, ${br.cost():.2f}; cache hits {c.hits}, misses {c.misses}; by role: "
              + ", ".join(f"{r} ${v:.2f}" for r, v in sorted(brains.cost_by_role(br).items())))
    print(f"\nlog: {out}   ({time.time() - t0:.0f}s)")


AGENT = data.ROOT / "output" / "agent"
DESK = "desk_v3_stage1"


def stage1(args, market, starts) -> None:
    """Stage 1 alone at `--overlay`, written as `agent_replay` writes a run, one per window
    (output/agent/<tag>/windows.csv and <tag>.out), for `tools/submit_agentic.py`.

    The PM is never told the overlay, so its payloads, and with them its cached answers, are
    the same at every share: a run at another overlay is the same entry bought at another
    gross. `--offline` makes that a guarantee rather than a hope: an answer not in the cache
    fails, the desk falls back, and the fallback count says so.
    """
    cfg = dict(OFFICIAL, overlay=args.overlay)
    kw = desk_inputs(market, starts, real_names=True)
    pct = int(round(args.overlay * 100))
    for s in starts:
        live = {"quick": brains.make(MODEL, "medium", max_calls=args.max_calls),
                "deep": brains.make(MODEL, "high", max_calls=args.max_calls)}
        cached = {k: brains.CachedBrain(b, CACHE, offline=args.offline) for k, b in live.items()}
        d = V3Desk(cached, V3Config(**cfg), **kw)
        res = sim.run(d, market, s, windows.WINDOW_DAYS)
        rows = [metrics_row(res, s, DESK)]
        for ref, fac in ((HOLD, baselines.scaled(baselines.InverseVolHold, 0.75)),
                         (f"inv_vol_hold_{pct}", baselines.scaled(baselines.InverseVolHold, args.overlay))):
            rows.append(metrics_row(sim.run(fac(), market, s, windows.WINDOW_DAYS), s, ref))
        tag = f"v3_stage1_ov{pct}_{s}"
        out = AGENT / tag
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "windows.csv", index=False)
        (out / "entry.json").write_text(json.dumps({"entry": d.entry, "plan": d.plan}, default=str, indent=1))
        lines = [f"{MODEL}-{eff}: {len(live[k].records)} calls, ${live[k].cost():.2f}; "
                 f"cache hits {cached[k].hits}, misses {cached[k].misses}"
                 for k, eff in (("quick", "medium"), ("deep", "high"))]
        falls = ", ".join(f"{k} {v}" for k, v in sorted(d.fallbacks.items())) or "none"
        (AGENT / f"{tag}.out").write_text(f"stage 1 at overlay {args.overlay:g}, window {s}\n"
                                          + "\n".join(lines) + f"\nfallbacks: {falls}\n")
        m = res.metrics()
        print(f"{tag}: {d.entry['source']} ({d.entry.get('final')}), gross {d.entry['gross']:.4f}, "
              f"return {m['cumulative_return']:+.4f}, sharpe {m['sharpe_ratio']:.2f}; {' | '.join(lines)}; "
              f"fallbacks {falls}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["ledgers", "run", "stage1"])
    ap.add_argument("--overlay", type=float, default=OFFICIAL["overlay"],
                    help="stage1: the invested share code holds (0.5 = 50%% cash)")
    ap.add_argument("--suite", default="official4")
    ap.add_argument("--on", default=None, help="one window, by its first session")
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    market = markets.research_market()
    if args.command == "ledgers":
        return ledgers(market, OUT / "ledgers")
    starts = starts_of(market, args.suite)
    if args.on:
        on = date.fromisoformat(args.on)
        if on not in starts:
            raise SystemExit(f"{on} starts no {args.suite} window")
        starts = [on]
    if args.command == "stage1":
        return stage1(args, market, starts)
    run(args, market, starts, OUT / (args.tag or f"b_{args.suite}_{len(starts)}w_budget{args.budget}"))


if __name__ == "__main__":
    main()
