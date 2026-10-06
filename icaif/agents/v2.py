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

Rounds 2-7 trade nothing yet: the trigger path is the next step of the build plan.
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

from icaif import baselines, compiler, filings as F, quant_strategies as qs
from icaif.agents import observe, prompts_v2
from icaif.agents.brains import BrainError
from icaif.agents.budget import TurnoverBudget, window_rounds
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.schemas import (AnalystReport, DebateTurn, EarningsReport, PMDecision,
                                  RiskReview, TradeList)
from icaif.agents.tradelist import TradeListError, compile_trades
from icaif.agents.triggers import TriggerTags

ANALYSTS = ("market", "earnings", "news", "quant")
DEBATE = ("bull_1", "bear_1", "bull_2", "bear_2")
ROLE_TIER = {"market": "quick", "earnings": "quick", "news": "quick", "quant": "quick",
             "bull": "quick", "bear": "quick", "trader": "quick", "risk": "deep", "pm": "deep"}
# Seconds from the chain's start at which each stage's slot ends. Grok 4.7 took 40-235 s
# a call at effort high in the v1 runs (about 7,000 output tokens with its reasoning), so
# the design doc's 7.5-minute chain would skip most roles: these slots run to 17 minutes.
# That does not fit the runner's 12-minute lead, so a live v2 shadow needs round 1 to wake
# earlier; its inputs are as of the prior close plus overnight news, so it can.
SLOTS = {"analysts": 180, "bull_1": 270, "bear_1": 360, "bull_2": 450, "bear_2": 540,
         "trader": 660, "risk": 840, "pm": 1020}
HOLD_GROSS = 0.75
CORE = ("name", "weight_now", "ret_1d", "ret_5d", "ret_20d", "vol_ann_20d", "vol_ann_har_1d",
        "vol_ann_har_3d", "model_score_rank", "earnings_in_sessions", "entry_day", "entry_date",
        "gain_since_entry", "peak_gain_since_entry")


@dataclass
class V2Config(DeskConfig):
    # (model, effort) per tier; the analysts, debate and trader are quick, the risk manager
    # and the PM deep. The replay tool builds one brain per tier from these.
    tiers: dict = field(default_factory=lambda: {"quick": ("grok-4.7", "medium"),
                                                 "deep": ("grok-4.7", "high")})
    slots: dict = field(default_factory=lambda: dict(SLOTS))
    grace_s: float = 15.0          # past a slot, how long a call already running is awaited
    min_call_s: float = 5.0        # a slot with less left than this is not asked
    exposure_free: float = 0.75    # gross above this needs the risk manager's sign-off
    turnover_cap: float = 5.0      # notional / NAV a window: a runaway guard, not a budget
    earnings_horizon: int = 2      # sessions ahead the earnings analyst looks
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

    def _left(self, slot: str) -> float:
        return self.cfg.slots[slot] - (time.perf_counter() - self._t_chain)

    def _record(self, ctx, key, tier, source, reason, answer, payload, system, latency):
        role = key.split("_")[0]
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
            role = key.split("_")[0]
            tier = ROLE_TIER[role]
            system = prompts_v2.SYSTEM[role]
            left = self._left(slot)
            if left < self.cfg.min_call_s:
                self._record(ctx, key, tier, "skipped", f"slot spent ({left:.0f}s left)", None,
                             payload, system, 0.0)
                out[key] = None
                continue
            fut = self._pool.submit(self.brains[tier].decide, f"v2_{key}", system, payload, schema, left)
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
            return None
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

        reports = self._analysts(ctx, views)
        shared = dict(views["shared"], reports=_shown(reports))
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

    def _analysts(self, ctx, views) -> dict:
        names = lambda d: self._names_ok([x.name for x in getattr(d, "names", getattr(d, "calls", []))])  # noqa: E731
        asks = [("market", views["market"], AnalystReport, names),
                ("quant", views["quant"], AnalystReport, names)]
        if views["earnings"] is not None:
            asks.append(("earnings", views["earnings"], EarningsReport, names))
        if views["news"] is not None:
            asks.append(("news", views["news"], AnalystReport, names))
        out = self._ask_many(ctx, asks, "analysts")
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
        if gross > free + 1e-12:
            out["needs_exposure_signoff"] = f"gross {gross:.4f} is above {free:.4f}"
        return out

    # ------------------------------------------------------------------ execution

    def _execute(self, ctx, tickers, current, nav, pm, proposal, cap, budget):
        reason, w = None, None
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
                         "decision": _record_of(pm), "traded": w is not None,
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


def _record_of(pm: Optional[PMDecision]) -> dict:
    """The PM's decision as the journal keeps it: compact levers and the rationale."""
    if pm is None:
        return {"action": "none", "rationale": ""}
    out = {"action": pm.action, "rationale": pm.rationale}
    tl = pm.trade_list
    if tl is not None:
        out.update(adds=[f"{x.name}:{x.weight:g}" for x in tl.adds], cuts=[x.name for x in tl.cuts],
                   trims=[f"{x.name}:{x.fraction}({x.cause})" for x in tl.trims],
                   target_exposure=tl.target_exposure)
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
        raise BrainError(f"no code answer for {schema.__name__}")
