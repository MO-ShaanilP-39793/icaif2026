"""Desk v2: a small trading firm each morning (the "ICAIF agent desk v2" design doc).

Four analysts report in parallel (market, earnings and events, news, quant), a bull and a
bear argue for two rounds, the trader proposes a trade list, the risk manager checks it,
and the portfolio manager (PM) decides. Only the PM's answer reaches code, and code only
turns it into weights (`tradelist.compile_trades`), refusing a list that breaks a rule.

**What code owns, as in v1.** The observation is point in time (`Desk._payload`, the
trigger tags, the journal). Nobody is shown a regime label (`regime=False`) or the rule's
answer. Only the risk manager is told our backtest evidence (`prompts_v2`). Gross above
`exposure_free` needs the risk manager's sign-off. Turnover is shown to every role that
trades and capped only against a runaway (`turnover_cap`, about five books a window): a
trade that earns its fee is the PM's call, and the turnover rank is the price it weighs.

**Deadlines.** Each stage has a slot that ends a fixed time after the chain starts
(`slots`); time a stage saves rolls forward to the next. A role is asked with the time
left in its slot as its timeout, so a slow provider call fails inside the brain, and a
replay from the cache fails the same calls (a failed call is never cached). A role whose
slot is spent is not asked. A late, failed or invalid role is skipped and logged, and the
desk goes on without it. A late, failed or invalid PM, or a PM list code refuses, means
no trade; before the book is first bought, it means the 75% inverse-vol book
(`baselines.scaled(InverseVolHold, 0.75)`, exactly what we would submit). Every skip and
fallback is counted in `fallbacks`.

**The journal** records the decision of record each round (the PM's, or the fallback),
so `memory` reads one line per morning rather than eleven; every role's call, with its
payload size, latency and answer, is in `chain`.

**Rounds 2-7** trade only on a code trigger for a held name, v1's three: results at the
next open (the last round), a move of `sigma_trigger` daily sigmas since the last close
(once a day a name), and a new 8-K (once a filing). The event analyst makes the case and
the PM decides those names alone: a list naming any other name, or setting an exposure,
is refused. Each trigger carries its tag (`triggers.TriggerTags`), so the roles know
whether the move has already happened.

**Reflection.** Each morning code settles what is now known (`_settle`): every line the
PM traded, at the close of the session after its fill, and every name held through its
results. A deep role reads those settlements beside the reasons given at the time and
writes at most four lessons into the journal. It runs in the analysts' slot, in
parallel, as of the prior close (nothing later is visible at round 1), so the chain is
no longer for it, and the debate, trader, risk manager and PM read the new lessons. The
lessons live in this desk's journal: one window's memory, never carried to the next.
"""

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from icaif import baselines, calendar, compiler, filings as F, quant_strategies as qs, sim
from icaif.agents import observe, prompts_v2
from icaif.agents.brains import BrainError
from icaif.agents.budget import TurnoverBudget, window_rounds
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.schemas import (AnalystReport, DebateTurn, EarningsReport, EventReport,
                                  PMDecision, Reflection, RiskReview, TradeList, TriggerDecision)
from icaif.agents.tradelist import TradeListError, compile_trades, gross_limit
from icaif.agents.triggers import TriggerTags, move_since_close

ANALYSTS = ("market", "earnings", "news", "quant")
DEBATE = ("bull_1", "bear_1", "bull_2", "bear_2")
ROLE_TIER = {"market": "quick", "earnings": "quick", "news": "quick", "quant": "quick",
             "bull": "quick", "bear": "quick", "trader": "quick", "risk": "deep", "pm": "deep",
             "event": "quick", "event_pm": "deep", "reflect": "deep"}
# Seconds from the chain's start at which each stage's slot ends. Grok 4.7 took 40-235 s
# a call at effort high in the v1 runs (about 7,000 output tokens with its reasoning), so
# the design doc's 7.5-minute chain would skip most roles: these slots run to 17 minutes.
# That does not fit the runner's 12-minute lead, so a live v2 shadow needs round 1 to wake
# earlier; its inputs are as of the prior close plus overnight news, so it can.
SLOTS = {"analysts": 180, "bull_1": 270, "bear_1": 360, "bull_2": 450, "bear_2": 540,
         "trader": 660, "risk": 840, "pm": 1020,
         # A trigger round runs inside the runner's 12-minute lead, after the data.
         "event": 240, "event_pm": 540}
LESSONS_SHOWN, SETTLED_SHOWN = 8, 6
HOLD_GROSS = 0.75
CORE = ("name", "weight_now", "ret_1d", "ret_5d", "ret_20d", "vol_ann_20d", "vol_ann_har_1d",
        "vol_ann_har_3d", "model_score_rank", "earnings_in_sessions", "entry_day", "entry_date",
        "gain_since_entry", "peak_gain_since_entry")


@dataclass
class V2Config(DeskConfig):
    # (model, effort) per tier; the analysts, debate and trader are quick, the risk manager
    # and the PM deep. The replay tool builds one brain per tier from these.
    tiers: dict = field(default_factory=lambda: {"quick": ("gemini-2.5-flash", "medium"),
                                                 "deep": ("gemini-2.5-pro", "high")})
    slots: dict = field(default_factory=lambda: dict(SLOTS))
    grace_s: float = 15.0          # past a slot, how long a call already running is awaited
    min_call_s: float = 5.0        # a slot with less left than this is not asked
    exposure_free: float = 0.75    # gross above this needs the risk manager's sign-off
    turnover_cap: float = 5.0      # notional / NAV a window: a runaway guard, not a budget
    earnings_horizon: int = 2      # sessions ahead the earnings analyst looks
    triggers: bool = True          # rounds 2-7: the event analyst and PM on a code trigger
    reflect: bool = True           # each morning: settle, and write lessons
    # v1's flags, fixed for v2: no regime label, no rule shown, no evidence but the risk
    # manager's, and every name's headlines read by the news analyst.
    regime: bool = False
    anchored: bool = False
    evidence: bool = False
    review: bool = False
    events: bool = False
    headline_roles: tuple = ("v2",)
    universe_roles: tuple = ("v2",)


def _size(x) -> int:
    return len(json.dumps(x, sort_keys=True, default=str))


def _role(key: str) -> str:
    """The role a call key asks: "bull_2" is the bull's second turn; "event_pm" is a role."""
    return key.rsplit("_", 1)[0] if key[-1].isdigit() else key


class V2Desk(Desk):
    def __init__(self, brains: dict, config: Optional[V2Config] = None, earnings_history=None, **kw):
        """`brains`: {"quick": brain, "deep": brain}; `earnings_history`: a
        `triggers.EarningsHistory` for the tags' past reactions (none: the tags say so)."""
        cfg = config or V2Config()
        if not isinstance(cfg, V2Config):
            raise TypeError("V2Desk takes a V2Config")
        if set(brains) != {"quick", "deep"}:
            raise ValueError("V2Desk needs a quick and a deep brain")
        super().__init__(brains["deep"], cfg, **kw)
        self.brains = brains
        self.tags = TriggerTags(earnings_history, self.earnings)
        self.chain: list[dict] = []
        self.fallbacks: Counter = Counter()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._t_chain = 0.0

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        out = super().state()
        out["v2"] = {"chain": self.chain, "fallbacks": dict(self.fallbacks)}
        return out

    def restore(self, state, tickers):
        super().restore(state, tickers)
        v2 = (state or {}).get("v2") or {}
        self.chain = list(v2.get("chain", []))
        self.fallbacks = Counter(v2.get("fallbacks", {}))

    # ------------------------------------------------------------------ calls

    # The role names, tiers and prompts a subclass asks under (desk v3 reuses these calls).
    # The prefix keys the cache and the cost by role: a v3 PM sharing "v2_pm" would be
    # handed a v2 PM's cached answer whenever their payloads matched.
    PREFIX = "v2"

    def _tier(self, role: str) -> str:
        return ROLE_TIER[role]

    def _system(self, role: str) -> str:
        return prompts_v2.SYSTEM[role]

    def _left(self, slot: str) -> float:
        return self.cfg.slots[slot] - (time.perf_counter() - self._t_chain)

    def _record(self, ctx, key, tier, source, reason, answer, payload, system, latency):
        role = _role(key)
        if source != "brain":
            self.fallbacks[f"{role}:{source}"] += 1
        self.chain.append({
            "day": self.day_no, "round": ctx.round, "role": key, "tier": tier,
            "brain": getattr(self.brains[tier], "name", "?"), "source": source, "reason": reason,
            "answer": answer.model_dump() if answer is not None else None,
            "payload_chars": _size(payload), "system_chars": len(system),
            "latency_s": round(latency, 3)})

    def _ask_many(self, ctx, asks: list, slot: str) -> dict:
        """{key: answer or None} for `asks` = [(key, payload, schema, check)], all in one
        slot, in parallel. A role that fails, runs late or answers outside its checks is
        skipped: logged, counted, and None to the roles after it."""
        out = {}
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=len(ANALYSTS))
        started = []
        for key, payload, schema, check in asks:
            role = _role(key)
            tier = self._tier(role)
            system = self._system(role)
            left = self._left(slot)
            if left < self.cfg.min_call_s:
                self._record(ctx, key, tier, "skipped", f"slot spent ({left:.0f}s left)", None,
                             payload, system, 0.0)
                out[key] = None
                continue
            fut = self._pool.submit(self.brains[tier].decide, f"{self.PREFIX}_{key}", system, payload,
                                    schema, left)
            started.append((key, tier, system, payload, check, fut, time.perf_counter()))
        for key, tier, system, payload, check, fut, t0 in started:
            source, reason, answer = "brain", None, None
            try:
                answer = fut.result(timeout=max(self._left(slot), 0.0) + self.cfg.grace_s)
                check(answer)
            except FutureTimeout:
                source, reason, answer = "late", "still running past its slot", None
            except (BrainError, ValueError, KeyError, TypeError) as err:
                source, reason, answer = "failed", f"{type(err).__name__}: {err}", None
            except Exception as err:  # noqa: BLE001 - a crash must not cost the morning
                source, reason, answer = "failed", f"unexpected {type(err).__name__}: {err}", None
            self._record(ctx, key, tier, source, reason, answer, payload, system,
                         time.perf_counter() - t0)
            out[key] = answer
        return out

    def _ask_one(self, ctx, key, payload, schema, check=lambda a: None, slot=None):
        return self._ask_many(ctx, [(key, payload, schema, check)], slot or key)[key]

    def _names_ok(self, codes) -> None:
        """An answer naming anything but the 30 is refused, as v1's `_only_tradeable`."""
        for c in codes:
            if c in self.anon.to_ticker:
                continue
            if self.ucodes is not None and self.ucodes.is_universe_name(c):
                raise ValueError(f"{c} is in universe_context only and is not tradeable")
            raise ValueError(f"{c} is not one of the 30 tradeable names")

    # ------------------------------------------------------------------ the morning

    def _decide(self, ctx, tickers):
        if ctx.round != 1:
            return self._triggered(ctx, tickers) if self.cfg.triggers else None
        value, nav = self._value(ctx, tickers)
        if not np.isfinite(nav) or nav <= 0:
            self._note = "the book cannot be valued (a held name has no price); held"
            return None
        current = value / nav
        self.book.weights = current
        self._current = current
        self.book.nav.append(nav)
        self._t_chain = self._t_round
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        rd = observe.readings(closes, None)
        tags = {}
        for t in tickers:
            tag = self.tags.earnings(ctx, t, self.cfg.earnings_horizon)
            if tag is not None:
                tags[self.anon.code(t)] = tag
        hot = [t for t in tickers if self.anon.code(t) in tags]
        obs = self._payload(closes, rd, ctx, "v2", held=list(tickers), triggered=hot)
        budget = TurnoverBudget(self.cfg.turnover_cap, self._window_rounds(ctx))
        views = self._views(obs, tags, self._new_filings(ctx), budget.view(self.journal))
        settled = self._settle(ctx) if self.cfg.reflect else []

        reports = self._analysts(ctx, views, settled)
        shared = dict(views["shared"], reports=_shown(reports), **self._learned())
        debate = {}
        for key in DEBATE:
            debate[key] = self._ask_one(ctx, key, dict(shared, debate=_shown(debate),
                                                        debate_round=int(key[-1])), DebateTurn)
        shared["debate"] = _shown(debate)

        gross = float(current.sum())
        proposal = self._ask_one(ctx, "trader", shared, TradeList)
        shared["proposal"] = proposal.model_dump() if proposal is not None else _missing(self.chain[-1])
        shared["compiled"] = self._preview(proposal, current, budget)
        risk = self._ask_one(ctx, "risk", shared, RiskReview)
        shared["risk_review"] = risk.model_dump() if risk is not None else _missing(self.chain[-1])

        def check_pm(d: PMDecision):
            if d.action == "amend" and d.trade_list is None:
                raise ValueError("amend without a trade list")
            if d.action != "amend" and d.trade_list is not None:
                raise ValueError(f"a trade list belongs to amend, not {d.action}")
            if d.action == "approve" and proposal is None:
                raise ValueError("approve with no proposal to approve")

        pm = self._ask_one(ctx, "pm", shared, PMDecision, check_pm)
        cap = min(max(self.cfg.exposure_free, gross,
                      risk.exposure_signoff if risk is not None and risk.exposure_signoff else 0.0), 1.0)
        return self._execute(ctx, tickers, current, nav, pm, proposal, cap, budget)

    def _analysts(self, ctx, views, settled: list) -> dict:
        """The four reports, and the reflection on `settled` beside them in the same slot."""
        names = lambda d: self._names_ok([x.name for x in getattr(d, "names", getattr(d, "calls", []))])  # noqa: E731
        asks = [("market", views["market"], AnalystReport, names),
                ("quant", views["quant"], AnalystReport, names)]
        if views["earnings"] is not None:
            asks.append(("earnings", views["earnings"], EarningsReport, names))
        if views["news"] is not None:
            asks.append(("news", views["news"], AnalystReport, names))
        known = {s["id"] for s in self.journal.settlements} | {s["id"] for s in settled}

        def cites(d: Reflection):
            bad = sorted({i for x in d.lessons for i in x.settlements} - known)
            if bad:
                raise ValueError(f"lessons cite settlements that do not exist: {bad}")

        if settled:
            asks.append(("reflect", self._reflect_view(views, settled), Reflection, cites))
        out = self._ask_many(ctx, asks, "analysts")
        self.journal.settlements += settled
        if out.get("reflect") is not None:
            self.journal.lessons += [{"day": self.day_no, "text": x.text, "from": list(x.settlements)}
                                     for x in out["reflect"].lessons]
        if views["earnings"] is None:
            out["earnings"] = "no name reports within the horizon"
        if views["news"] is None:
            out["news"] = "no headlines or filings to read"
        return {k: out[k] for k in ANALYSTS}

    def _views(self, obs: dict, tags: dict, new_filings: dict, turnover: dict) -> dict:
        """Each role's slice of the morning's observation: what its job reads, no more."""
        rows = obs["names"]
        core = [{k: r[k] for k in CORE if k in r} for r in rows]
        base = {"clock": obs["clock"], "book": obs["book"]}
        by_code = {r["name"]: r for r in rows}
        reporting = [dict({k: by_code[c][k] for k in CORE if k in by_code[c]}, tag=tags[c],
                          recent_8k_filings=by_code[c].get("recent_8k_filings", []))
                     for c in sorted(tags)]
        news = [{"name": r["name"], "weight_now": r.get("weight_now"), "ret_1d": r.get("ret_1d"),
                 "headlines": r.get("headlines", []), "recent_8k_filings": r.get("recent_8k_filings", [])}
                for r in rows]
        has_news = any(x["headlines"] or x["recent_8k_filings"] for x in news) or bool(new_filings)
        quant = [{k: v for k, v in r.items() if k not in ("headlines", "recent_8k_filings")} for r in rows]
        return {
            "market": dict(base, market=obs["market"], **({"macro": obs["macro"]} if "macro" in obs else {})),
            "earnings": dict(base, reporting=reporting) if reporting else None,
            "news": dict(base, names=news, new_filings=new_filings) if has_news else None,
            "quant": dict(base, names=quant, market={k: v for k, v in obs["market"].items() if "vol" in k},
                          **({"universe_context": obs["universe_context"]}
                             if "universe_context" in obs else {})),
            "shared": dict(base, memory=obs["memory"], names=core, earnings_tags=tags,
                           turnover=turnover),
        }

    def _new_filings(self, ctx) -> dict:
        """{code: 8-Ks accepted since the last morning, with their text}, real names only."""
        after, self._filings_to = self._filings_to, pd.Timestamp(ctx.deadline)
        if self.filings is None or self.anon.enabled:
            return {}
        after = after if after is not None else pd.Timestamp(ctx.deadline) - pd.Timedelta(hours=24)
        return {self.anon.code(t): observe.filing_rows(g, ctx.deadline, real=True)
                for t, g in F.new(self.filings, after, ctx.deadline).groupby("ticker")}

    def _window_rounds(self, ctx) -> int:
        """The board's divisor: the window's rounds, a missing future session counted as 7."""
        days = ctx.market.days
        i = days.index(self._start) if self._start in days else days.index(ctx.day)
        got = days[i:i + self.cfg.window_days]
        return window_rounds(got) + 7 * (self.cfg.window_days - len(got))

    def _compile(self, tl, current, budget, cap):
        uni = set(self.ucodes.to_code.values()) if self.ucodes is not None and self.ucodes.enabled else set()
        return compile_trades(tl, current, self.anon.to_ticker, budget_left=budget.left(self.journal),
                              exposure_cap=cap, universe_names=uni)

    def _preview(self, proposal, current, budget) -> dict:
        """What code would make of the trader's list, for the risk manager and the PM."""
        if proposal is None:
            return {"unavailable": "no proposal"}
        free = max(self.cfg.exposure_free, float(current.sum()))
        try:
            got = self._compile(proposal, current, budget, 1.0)
        except TradeListError as err:
            return {"refused": err.errors}
        if got is None:
            return {"hold": True}
        gross = float(got.target.sum())
        out = {"turnover": round(got.turnover, 4), "fee_bps_of_nav": round(got.turnover * 10, 2),
               "gross_after": round(gross, 4),
               "lines": [{"name": x["name"], "action": x["action"], "from": round(x["from"], 4),
                          "to": round(x["to"], 4)} for x in got.lines]}
        if gross > gross_limit(free) + 1e-12:
            out["needs_exposure_signoff"] = f"gross {gross:.4f} is above {free:.4f}"
        return out

    # ------------------------------------------------------------------ reflection

    def _learned(self) -> dict:
        j = self.journal
        return {"lessons": [{"day": x["day"], "text": x["text"]} for x in j.lessons[-LESSONS_SHOWN:]],
                "settled": [s["text"] for s in j.settlements[-SETTLED_SHOWN:]]}

    def _reflect_view(self, views, settled) -> dict:
        keys = {s["key"] for s in settled if s.get("key")}
        why = {e["key"]: [{"role": d["role"], "why": d["why"]} for d in e["decisions"]]
               for e in self.journal.rounds if e["key"] in keys}
        return dict(views["shared"], settlements=[{k: v for k, v in s.items() if k != "key"} for s in settled],
                    decisions=why, **self._learned())

    def _day_of(self, d) -> Optional[int]:
        return next((e["day"] for e in self.journal.rounds if e.get("date") == str(d)), None)

    def _settle(self, ctx) -> list[dict]:
        """Decisions whose outcome is known by this deadline and not yet settled.

        - **Trades**: each name the PM traded after the entry (at a trigger too; never a
          fallback, which no role chose), marked from its fill to the close of the next
          session: the effect is the weight traded times the name's move, in bp of NAV,
          fees apart. A sale before results that then rise is a cost here, not a dodge.
        - **Held through results**: each name the book held across its reaction session,
          with no trade in it from that day on: the weight times the reaction.

        Prices are the deadline-cut closes (`qs.daily_closes`) and the journal's own fills,
        so at round 1 the latest close is yesterday's, as if run after it.
        """
        j, out = self.journal, []
        done = {s["id"] for s in j.settlements}
        daily = qs.daily_closes(ctx, 3 * self.cfg.window_days)
        if not len(daily):
            return out
        dates = [ts.date() for ts in daily.index]
        at = {d: i for i, d in enumerate(dates)}
        hist = self.tags.history
        lines = {e["key"]: {n for d in e.get("decisions", []) if d.get("source") == "brain"
                            for n in (d.get("levers") or {}).get("lines", [])} for e in j.rounds}
        touched: dict = {}
        for e in j.rounds:
            for f in [e.get("fill")] + list(e.get("more_fills", [])):
                if f is None:
                    continue
                day = pd.Timestamp(f["at"]).tz_convert(calendar.TZ).date()
                for tk in set(f.get("bought", {})) | set(f.get("sold", {})):
                    touched.setdefault(tk, []).append(day)
                # The entry (bought from cash) is the window's base, scored by the window,
                # not line by line: thirty settlements would bury the decisions after it.
                if (f is not e.get("fill") or f.get("matched") == "adopted"
                        or not lines.get(e["key"]) or not e.get("names_held")):
                    continue
                later = [d for d in dates if d > day]
                if not later:
                    continue
                h = later[0]
                for tk in sorted(set(f.get("bought", {})) | set(f.get("sold", {}))):
                    code = self.anon.code(tk)
                    if code not in lines[e["key"]]:
                        continue
                    sid = f"{e['key']}:{code}"
                    px, close = f["prices"].get(tk), float(daily[tk].iloc[at[h]])
                    if sid in done or not px or not np.isfinite(close) or not f.get("nav_before"):
                        continue
                    dw = (f.get("bought", {}).get(tk, 0.0) - f.get("sold", {}).get(tk, 0.0)) * px / f["nav_before"]
                    move = close / px - 1.0
                    effect, fee = dw * move * 1e4, abs(dw) * sim.FEE_RATE * 1e4
                    through = hist is not None and bool(len(hist.table[
                        (hist.table["ticker"] == tk) & (hist.table["session"] > day)
                        & (hist.table["session"] <= h) & (hist.table["known_at"] <= pd.Timestamp(ctx.deadline))]))
                    hday = self._day_of(h)
                    verb = "bought" if dw > 0 else "sold"
                    out.append({
                        "id": sid, "key": e["key"], "kind": "trade", "name": code, "day": e["day"],
                        "round": e["round"], "weight": round(dw, 4), "move": round(move, 4),
                        "through_results": through, "effect_bp": round(effect, 1), "fee_bp": round(fee, 1),
                        "text": (f"{verb} {code} ({abs(dw):.1%} of NAV) on day {e['day']} r{e['round']}; "
                                 f"by day {hday}'s close it {'rose' if move > 0 else 'fell'} {abs(move):.1%}"
                                 f"{' through its results' if through else ''}: the trade "
                                 f"{'gained' if effect > 0 else 'cost'} {abs(effect):.0f} bp, fees {fee:.0f} bp")})
        if hist is None or self._start is None:
            return out
        shares = pd.Series(ctx.shares, dtype=float).reindex(daily.columns).fillna(0.0)
        for tk in daily.columns:
            if shares[tk] <= 1e-9:
                continue
            for _, r in hist.past(tk, ctx.deadline).iterrows():
                s = r["session"]
                code, k = self.anon.code(tk), self._day_of(s)
                sid = f"held:{code}:day{k}"
                if s < self._start or k is None or sid in done or s not in at or at[s] == 0:
                    continue
                if any(d >= s for d in touched.get(tk, [])):
                    continue   # traded on or after the reaction: not held through it as it stands
                prev, close = daily.iloc[at[s] - 1], daily.iloc[at[s]]
                held = shares[shares > 1e-9]
                nav = float(ctx.cash) + float((held * prev[held.index]).sum())
                w, move = shares[tk] * prev[tk] / nav, close[tk] / prev[tk] - 1.0
                effect = w * move * 1e4
                out.append({
                    "id": sid, "kind": "held_through_results", "name": code, "day": k,
                    "weight": round(float(w), 4), "move": round(float(move), 4),
                    "effect_bp": round(float(effect), 1),
                    "text": (f"held {code} ({w:.1%} of NAV) through its results on day {k}: it "
                             f"{'rose' if move > 0 else 'fell'} {abs(move):.1%}, "
                             f"{'+' if effect > 0 else '-'}{abs(effect):.0f} bp to the book")})
        return out

    # ------------------------------------------------------------------ rounds 2-7

    def _triggered(self, ctx, tickers):
        """A trigger round: the event analyst, then the PM, on the triggered names alone."""
        value, nav = self._value(ctx, tickers)
        if not np.isfinite(nav) or nav <= 0:
            return None
        current = value / nav
        self.book.weights = current
        self._current = current
        held = [t for t in tickers if current[t] > compiler.HELD]
        after, self._filings_to = self._filings_to, pd.Timestamp(ctx.deadline)
        fresh = {}
        if self.filings is not None and after is not None and held:
            fresh = {t: g for t, g in F.new(self.filings, after, ctx.deadline).groupby("ticker")
                     if t in held}
        open_ = [t for t in held if t not in self._fired]
        why, kind = {}, {}
        if self.earnings is not None and ctx.round == len(calendar.rounds_for(ctx.day)):
            nxt = self.earnings.to_next(ctx.day)
            for t in open_:
                if nxt.get(t) == 1:
                    why[t], kind[t] = "results react at the next open", ("earnings", None, 1)
        for t in open_:
            if t in why:
                continue
            _, z = move_since_close(ctx, t)
            if z is not None and abs(z) >= self.cfg.sigma_trigger:
                why[t], kind[t] = f"moved {z:+.1f} daily sigmas since the last close", ("move", None, None)
        # A price or results trigger fires once a day a name; a filing once a filing.
        self._fired |= set(why)
        for t, g in fresh.items():
            what = ", ".join(dict.fromkeys(x for items in g["items"] for x in F.labels(items)))
            why[t] = (f"{why[t]}; " if t in why else "") + f"new 8-K: {what}"
            kind.setdefault(t, ("8k", g["accepted"].max(), None))
        if not why:
            return None

        self._t_chain = self._t_round
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        obs = self._payload(closes, observe.readings(closes, None), ctx, "v2", held=held,
                            triggered=list(why))
        rows = {r["name"]: r for r in obs["names"]}
        real = not self.anon.enabled
        trig = []
        for t, w in why.items():
            code, (k, at, n) = self.anon.code(t), kind[t]
            row = rows[code]
            trig.append(dict({f: row[f] for f in CORE if f in row}, why=w,
                             tag=self.tags.tag(ctx, t, k, at=at, sessions_to=n),
                             headlines=row.get("headlines", []),
                             **({"new_8k": observe.filing_rows(fresh[t], ctx.deadline, real)}
                                if t in fresh else {})))
        codes = {x["name"] for x in trig}
        budget = TurnoverBudget(self.cfg.turnover_cap, self._window_rounds(ctx))
        base = dict({"clock": obs["clock"], "book": obs["book"], "memory": obs["memory"],
                     "triggers": trig, "turnover": budget.view(self.journal)}, **self._learned())

        def only_these(d: EventReport):
            extra = sorted({x.name for x in d.cases} - codes)
            if extra:
                raise ValueError(f"cases for names with no trigger: {extra}")

        case = self._ask_one(ctx, "event", base, EventReport, only_these)
        payload = dict(base, event_case=case.model_dump() if case is not None else _missing(self.chain[-1]))

        def check(d: TriggerDecision):
            if (d.action == "trade") != (d.trade_list is not None):
                raise ValueError("a trade needs a list, and a hold has none")
            if d.trade_list is not None:
                tl = d.trade_list
                extra = sorted({x.name for x in tl.adds + tl.cuts + tl.trims} - codes)
                if extra:
                    raise ValueError(f"lines for names with no trigger: {extra}")
                if tl.target_exposure is not None:
                    raise ValueError("a trigger trades its names; exposure is the morning's decision")

        d = self._ask_one(ctx, "event_pm", payload, TriggerDecision, check)
        reason, w = None, None
        if d is None:
            reason = f"event_pm {self.chain[-1]['source']}: {self.chain[-1]['reason']}"
        elif d.action == "trade":
            cap = min(max(self.cfg.exposure_free, float(current.sum())), 1.0)
            try:
                got = self._compile(d.trade_list, current, budget, cap)
            except TradeListError as err:
                reason = f"event_pm list refused: {err}"
                self.fallbacks["event_pm:refused"] += 1
            else:
                if got is not None:
                    w = self._take(got.weights, got.turnover, current, nav)
        self.log.append({"day": self.day_no, "round": ctx.round, "role": "event_pm",
                         "brain": getattr(self.brains["deep"], "name", "?"),
                         "source": "brain" if reason is None else "fallback", "reason": reason,
                         "decision": _record_of(d), "traded": w is not None,
                         "triggers": {x["name"]: x["why"] for x in trig},
                         "latency_s": round(time.perf_counter() - self._t_chain, 3)})
        return w

    # ------------------------------------------------------------------ execution

    def _execute(self, ctx, tickers, current, nav, pm, proposal, cap, budget):
        reason, w, tl = None, None, None
        if pm is None:
            reason = f"pm {self.chain[-1]['source']}: {self.chain[-1]['reason']}"
        elif pm.action != "hold":
            tl = proposal if pm.action == "approve" else pm.trade_list
            try:
                got = self._compile(tl, current, budget, cap)
            except TradeListError as err:
                reason = f"pm list refused: {err}"
                self.fallbacks["pm:refused"] += 1
            else:
                if got is not None:
                    w = self._take(got.weights, got.turnover, current, nav)
        held_cash = float(current.sum()) <= compiler.HELD
        if reason is not None and not self.book.entered and held_cash:
            book = baselines.scaled(baselines.InverseVolHold, HOLD_GROSS)()(ctx)
            if book is not None:
                self.fallbacks["entry:hold_book"] += 1
                turnover = float(sum(book.values()))
                w = self._take(book, turnover, current, nav)
                reason += "; bought the 75% inverse-vol book"
        elif reason is not None:
            self.fallbacks["pm:hold"] += 1
        self.log.append({"day": self.day_no, "round": ctx.round, "role": "pm",
                         "brain": getattr(self.brains["deep"], "name", "?"),
                         "source": "brain" if reason is None else "fallback", "reason": reason,
                         "decision": _record_of(pm, tl), "traded": w is not None,
                         "latency_s": round(time.perf_counter() - self._t_chain, 3)})
        return w

    def _take(self, weights: dict, turnover: float, current, nav) -> dict:
        """Submit `weights` as they are: they are already on the grid, and a second pass
        through `weights.safe` could floor a stated weight a step lower."""
        self.book.traded_notional += turnover * nav
        self.book.weights = pd.Series(weights, dtype=float).reindex(current.index).fillna(0.0)
        self.book.entered = True
        return weights


def _shown(answers: dict) -> dict:
    return {k: (v.model_dump() if hasattr(v, "model_dump") else
                {"unavailable": v if isinstance(v, str) else "skipped: late, failed or invalid"})
            for k, v in answers.items()}


def _missing(entry: dict) -> dict:
    return {"unavailable": f"{entry['source']}: {entry['reason']}"}


def _record_of(pm, tl: Optional[TradeList] = None) -> dict:
    """The PM's decision as the journal keeps it: the list it traded (the trader's, when it
    approved), compact, with the names its lines named. Those names are what a settlement
    grades: a fill also moves every other name by the drift between the close the weights
    were valued at and the open they fill at, which no role chose."""
    if pm is None:
        return {"action": "none", "rationale": ""}
    out = {"action": pm.action, "rationale": pm.rationale}
    tl = tl if tl is not None else pm.trade_list
    if tl is not None:
        out.update(adds=[f"{x.name}:{x.weight:g}" for x in tl.adds], cuts=[x.name for x in tl.cuts],
                   trims=[f"{x.name}:{x.fraction}({x.cause})" for x in tl.trims],
                   target_exposure=tl.target_exposure,
                   lines=sorted({x.name for x in tl.adds + tl.cuts + tl.trims}))
    return out


class HoldBrain:
    """Answers every v2 role in code, as a desk that only ever holds: no analyst finds
    anything, no one argues, the trader proposes nothing and the risk manager approves.
    The PM holds once the book is bought, and before that gives no answer, so the desk's
    own fallback buys the 75% inverse-vol book. A v2 desk on this brain must trade
    exactly as `inv_vol_hold_75`; any difference is the desk's plumbing."""

    name = "v2_hold"

    def decide(self, role, system, payload, schema, timeout):
        if schema is AnalystReport:
            return AnalystReport(summary="code: nothing to report", names=[])
        if schema is EarningsReport:
            return EarningsReport(summary="code: nothing to report", calls=[])
        if schema is DebateTurn:
            return DebateTurn(argument="code: no case", points=[])
        if schema is TradeList:
            return TradeList(adds=[], cuts=[], trims=[], target_exposure=None, rationale="code: hold")
        if schema is RiskReview:
            return RiskReview(verdict="approve", objections=[], exposure_signoff=None, rationale="code")
        if schema is PMDecision:
            if not payload["book"]["entered"]:
                raise BrainError("the hold brain makes no entry; the desk's fallback does")
            return PMDecision(action="hold", trade_list=None, rationale="code: hold")
        if schema is EventReport:
            return EventReport(summary="code: nothing to add", cases=[])
        if schema is TriggerDecision:
            return TriggerDecision(action="hold", trade_list=None, rationale="code: hold")
        if schema is Reflection:
            return Reflection(summary="code: no lessons", lessons=[])
        raise BrainError(f"no code answer for {schema.__name__}")
