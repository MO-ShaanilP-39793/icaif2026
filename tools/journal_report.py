"""Measure the portfolio memory (Roadmap step 4): its size at every round of a 105-round
phase, and its agreement with the ledger, over the research windows.

    .venv/bin/python tools/journal_report.py                 # every window, ~3 min
    .venv/bin/python tools/journal_report.py --windows 20    # the first 20

Three desks, each run fresh through every 15-day window (105 rounds) of the research
market, the windows the rule's backtests were scored on:

- **rule**: the rule desk, what Validation submits. An entry, then holds, with a morning
  review and the odd event: the memory of a quiet book.
- **busy**: a scripted brain that pulls a lever at every chance: an entry leaving out
  three names, a review each morning that moves the exposure or sells a name, an exit
  for every name an event wakes it for, every reason about 1,000 characters long. No
  LLM writes this much; it is the most a real desk's memory has to carry.
- **worst**: no desk. Every round carries the longest answer each role's schema allows
  and trades, which no round can: the bound the budget is held to.

For each round the memory block a role would read then is rendered and measured (as
the brain sends it: `json.dumps(sort_keys=True)`), whether or not a role was asked, and
the whole observation is measured whenever one was. Each rule and busy window's journal
is then checked against the simulator's ledger (`journal.verify`). Writes
reports/journal_budget.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import compiler, data, markets, sim, windows  # noqa: E402
from icaif.agents import brains, journal as J, signals  # noqa: E402
from icaif.agents.desk import Desk, DeskConfig  # noqa: E402
from icaif.agents.observe import Anonymizer  # noqa: E402
from icaif.agents.schemas import (EntryDecision, EventDecision, Exclusion, NameCall,  # noqa: E402
                                  ReviewDecision)

OUT = data.ROOT / "reports" / "journal_budget.json"
LONG = ("The book is up since entry and the regime model reads calm, but the basket's "
        "HAR forecast has risen two days running and three names report inside the "
        "window; ") * 7


class Busy:
    """Pulls a lever at every chance the desk gives, with long reasons."""

    name = "busy"

    def __init__(self):
        self.reviews = 0

    def decide(self, role, system, payload, schema, timeout):
        names = payload["names"]
        if role == "entry":
            return EntryDecision(shape="inverse_vol", views="none", exposure=0.7, rationale=LONG[:1400],
                                 avoid=[Exclusion(name=r["name"], signal="earnings", why=LONG[:280])
                                        for r in names[:3]])
        if role == "review":
            self.reviews += 1
            gross = payload["book"]["gross"] or 0.0
            held = [r["name"] for r in names if (r.get("weight_now") or 0) > 0]
            if self.reviews % 2 and gross > 0.4:
                return ReviewDecision(action="set_exposure", exposure=round(max(0.3, gross - 0.08), 4),
                                      reason=None, exit=[], rationale=LONG[:1400])
            return ReviewDecision(action="hold", exposure=None, reason=None, exit=held[-1:],
                                  rationale=LONG[:1400])
        return EventDecision(calls=[NameCall(name=t["name"], action="exit", reason=LONG[:580])
                                    for t in payload["triggers"]])


class Measured(Desk):
    """A desk that records the memory a role would read at every round, asked or not."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sizes, self.payloads = [], []

    def _journal_open(self, ctx):
        super()._journal_open(ctx)
        view = self.journal.memory(self.anon.code, real=not self.anon.enabled)
        self.sizes.append(J.size(view))

    def _payload(self, *a, **k):
        obs = super()._payload(*a, **k)
        names = sum(J.size({k2: r[k2] for k2 in r if k2.endswith("since_entry") or k2 == "entry_day"})
                    for r in obs["names"])
        self.payloads.append({"total": J.size(obs), "memory": J.size(obs["memory"]), "name_fields": names,
                              "trimmed": "trimmed_to_fit" in obs["memory"]})
        return obs


def worst_phase(market, start) -> list[int]:
    """Every round the longest answer each role allows, and a trade: the bound."""
    anon = Anonymizer(market.tickers, 7)
    code = anon.code
    j, sizes, st = J.Journal(), [], {"day": None, "n": 0, "i": 0}
    books = ({t: 1 / 30 for t in market.tickers}, {t: 0.02 for t in market.tickers})
    long = "x" * 1500

    def rec(i, rnd):
        if i == 0:
            return {"role": "entry", "source": "brain", "decision": {
                "shape": "risk_parity", "views": "strong", "exposure": 0.95, "rationale": long,
                "avoid": [{"name": code(t), "signal": "other", "why": "y" * 300} for t in market.tickers[:8]]}}
        if rnd == 1:
            return {"role": "review", "source": "fallback", "reason": "TimeoutError: " + "z" * 300,
                    "decision": {"action": "rebalance", "exposure": 0.6, "reason": "vol_change",
                                 "exit": [code(t) for t in market.tickers[:8]], "rationale": long}}
        return {"role": "event", "source": "brain", "decision": {"calls": [
            {"name": code(t), "action": "exit" if k % 2 else "hold", "reason": "r" * 600}
            for k, t in enumerate(market.tickers)]}}

    def strategy(ctx):
        if ctx.day != st["day"]:
            st["day"], st["n"] = ctx.day, st["n"] + 1
        j.open_round(ctx, st["n"])
        sizes.append(J.size(j.memory(code, real=False)))
        w = books[st["i"] % 2]
        j.close_round([rec(st["i"], ctx.round)], w)
        st["i"] += 1
        return w

    sim.run(strategy, market, start, windows.WINDOW_DAYS)
    return sizes


def summary(per_round: list[list[int]]) -> dict:
    """Sizes over every round of every window; by round number over the full 105-round
    windows (a window with a half-day has fewer rounds, so its round numbers shift)."""
    flat = np.array([x for s in per_round for x in s], dtype=float)
    out = {"windows": len(per_round), "rounds": int(len(flat)), "max_chars": int(flat.max()),
           "median_chars": float(np.median(flat)), "p95_chars": float(np.percentile(flat, 95))}
    full = np.array([s for s in per_round if len(s) == 105], dtype=float)
    if len(full):
        by_round = full.max(axis=0)
        out.update(full_windows=int(len(full)), max_at_round=int(by_round.argmax()) + 1,
                   max_by_round=[int(x) for x in by_round],
                   median_by_round=[float(x) for x in np.median(full, axis=0)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    market = markets.research_market()
    starts = [s for s in windows.window_starts(market) if s >= market.days[60]]
    if args.windows:
        starts = starts[: args.windows]
    scores = compiler.load_daily_scores()
    har = signals.VolForecasts.from_bars(market.info_bars, market.tickers)
    from tools.agent_replay import load_earnings
    earnings = load_earnings(market)
    closes = market.recent_closes(pd.Timestamp("2100-01-01", tz="America/New_York"), 10 ** 7)
    out = {"budget_chars": J.MEMORY_MAX_CHARS, "recent_rounds": J.RECENT_ROUNDS,
           "windows": [str(starts[0]), str(starts[-1]), len(starts)], "desks": {}}
    for name, make in (("rule", brains.RuleBrain), ("busy", Busy)):
        sizes, payloads, wrong, counts = [], [], [], {"fills": 0, "closed": 0, "issues": 0, "trimmed": 0}
        for s in starts:
            d = Measured(make(), DeskConfig(anonymize=True), scores=scores, earnings=earnings, vol=har)
            res = sim.run(d, market, s, windows.WINDOW_DAYS)
            sizes.append(d.sizes)
            payloads += d.payloads
            problems = J.verify(d.journal, J.sim_fills(res, market), to_ticker=d.anon.ticker, closes=closes)
            if problems:
                wrong.append({"window": str(s), "problems": problems[:3]})
            counts["fills"] += sum(1 for e in d.journal.rounds if e.get("fill"))
            counts["closed"] += len(d.journal.closed)
            counts["issues"] += len(d.journal.issues)
            counts["trimmed"] += sum(1 for p in d.payloads if p["trimmed"])
        p = pd.DataFrame(payloads)
        out["desks"][name] = {
            "memory": summary(sizes), "journal_disagrees_with_ledger": wrong, **counts,
            "rounds_asked": int(len(p)),
            "observation_chars": {"median": float(p["total"].median()), "max": int(p["total"].max())},
            "memory_share_of_observation": {"median": float((p["memory"] / p["total"]).median()),
                                            "max": float((p["memory"] / p["total"]).max())},
            "name_fields_chars": {"median": float(p["name_fields"].median()), "max": int(p["name_fields"].max())}}
        m = out["desks"][name]["memory"]
        print(f"{name}: memory max {m['max_chars']} chars, median {m['median_chars']:.0f}, p95 "
              f"{m['p95_chars']:.0f}, over {m['rounds']} rounds of {m['windows']} windows; "
              f"observation median {p['total'].median():.0f}, max {p['total'].max()}; "
              f"journal vs ledger: {len(starts) - len(wrong)} of {len(starts)} agree "
              f"({counts['fills']} fills, {counts['closed']} names closed, {counts['issues']} issues)")
    # Its answers are the same in every window, only the prices differ: a sample will do.
    worst = [worst_phase(market, s) for s in starts[::15]]
    out["desks"]["worst"] = {"memory": summary(worst)}
    w = out["desks"]["worst"]["memory"]
    print(f"worst: memory max {w['max_chars']} chars (budget {J.MEMORY_MAX_CHARS}), median "
          f"{w['median_chars']:.0f}, over {w['windows']} windows, every 15th")
    out["seconds"] = round(time.time() - t0)
    if not args.no_write:
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {OUT} ({out['seconds']}s)")
    if any(out["desks"][k]["journal_disagrees_with_ledger"] for k in ("rule", "busy")):
        raise SystemExit("a journal disagrees with its ledger")


if __name__ == "__main__":
    main()
