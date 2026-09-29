"""The desk: a `sim.Strategy` that asks its roles, validates, falls back and executes.

One code path serves the backtest, the replay and the live round: the desk reads a
`RoundContext`, and the live runner builds one from fresh data.

The rule each role falls back to is the best entry-only candidate of
`tools/quant_report.py`: risk parity, entry exposure blended by the HMM's probability
of turbulence (0.85 calm, 0.30 turbulent), then hold, and hold through every event. A
rule desk (`RuleBrain`) reproduces that candidate trade for trade, which a test
checks, so whatever an LLM desk scores differently is the LLM's doing.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from icaif import calendar, compiler, quant_strategies as qs
from icaif import weights as W
from icaif.agents import observe, prompts
from icaif.agents.brains import BrainError, rule_event
from icaif.agents.schemas import EntryDecision, EventDecision, ReviewDecision

E_CALM, E_TURBULENT = 0.85, 0.30  # = quant_strategies.Regime's defaults


@dataclass
class DeskConfig:
    window_days: int = 15
    review: bool = True
    events: bool = True
    band: float = 0.05          # a review exposure change smaller than this is a hold
    sigma_trigger: float = 3.0  # |move since yesterday's close| in daily sigmas
    anonymize: bool = False
    seed: int = 0
    round_budget_s: float = 360.0
    timeouts: dict = field(default_factory=lambda: {"entry": 240.0, "review": 120.0,
                                                    "event": 120.0})


def rule_exposure(p_turbulent: float) -> float:
    if not np.isfinite(p_turbulent):
        return 0.5 * (E_CALM + E_TURBULENT)
    return float(E_CALM * (1 - p_turbulent) + E_TURBULENT * p_turbulent)


class EarningsCalendar:
    """Sessions until each name's next earnings reaction, within the announced horizon.

    Built from realised EDGAR releases, as training was: a date within
    `earnings.NEXT_KNOWN_SESSIONS` sessions would have been announced by then. Live, the
    Yahoo calendar supplies the same thing from announcements.
    """

    def __init__(self, events: pd.DataFrame, sessions: list):
        from icaif import earnings

        idx = pd.DatetimeIndex(pd.to_datetime(sessions))
        q = earnings.quarterly(events)
        react = earnings.reaction_session(q["accepted"], idx)
        self.pos = {d.date(): i for i, d in enumerate(idx)}
        self.react = {t: sorted(self.pos[d.date()] for d in g.dropna())
                      for t, g in react.groupby(q["ticker"])}
        self.horizon = earnings.NEXT_KNOWN_SESSIONS

    def to_next(self, day) -> dict:
        i = self.pos.get(day)
        if i is None:
            return {}
        out = {}
        for t, ps in self.react.items():
            nxt = [p - i for p in ps if 0 < p - i <= self.horizon]
            if nxt:
                out[t] = nxt[0]
        return out


class Desk:
    def __init__(self, brain, config: Optional[DeskConfig] = None,
                 scores: Optional[compiler.DailyPanel] = None,
                 earnings: Optional[EarningsCalendar] = None):
        self.brain = brain
        self.cfg = config or DeskConfig()
        self.scores, self.earnings = scores, earnings
        self.anon: Optional[observe.Anonymizer] = None
        self.book: Optional[observe.BookState] = None
        self.hmm = None
        self.day_no = 0
        self._day = None
        self._fired: set = set()
        self.journal: list[dict] = []
        self.log: list[dict] = []
        self._t_round = 0.0

    # ------------------------------------------------------------------ plumbing

    def _ask(self, role: str, payload: dict, schema: type[BaseModel], check) -> tuple:
        """(decision, source): the brain's answer if it passes `check`, else the rule's."""
        rule = schema.model_validate(payload["rule_proposal"])
        left = self.cfg.round_budget_s - (time.perf_counter() - self._t_round)
        timeout = min(self.cfg.timeouts[role], left)
        t0 = time.perf_counter()
        source, reason, decision = "brain", None, None
        if timeout < 5.0:
            source, reason = "fallback", f"round budget spent ({left:.0f}s left)"
        else:
            try:
                decision = self.brain.decide(role, prompts.SYSTEM[role], payload, schema, timeout)
                check(decision)
            except (BrainError, ValueError, KeyError, TypeError) as err:
                source, reason = "fallback", f"{type(err).__name__}: {err}"
            except Exception as err:  # noqa: BLE001 - a crash would miss the round
                source, reason = "fallback", f"unexpected {type(err).__name__}: {err}"
        if source == "fallback":
            decision = rule
        entry = {"day": self.day_no, "round": payload["clock"]["round"], "role": role,
                 "brain": getattr(self.brain, "name", "?"), "source": source,
                 "reason": reason, "decision": decision.model_dump(),
                 "same_as_rule": decision.model_dump(exclude={"rationale"}) ==
                 rule.model_dump(exclude={"rationale"}) if role != "event"
                 else [c.action for c in decision.calls] == [c.action for c in rule.calls],
                 "latency_s": round(time.perf_counter() - t0, 3)}
        self.log.append(entry)
        summary = {k: v for k, v in decision.model_dump().items() if k != "rationale"}
        why = getattr(decision, "rationale", None) or "; ".join(
            f"{c.name}: {c.reason}" for c in getattr(decision, "calls", []))
        self.journal.append({"day": self.day_no, "role": role, "source": source,
                             "decision": summary, "why": (why or "")[:240]})
        return decision, source

    def _value(self, ctx, tickers):
        shares = pd.Series(ctx.shares, dtype=float).reindex(tickers).fillna(0.0)
        last = ctx.recent_closes(1)
        px = last.iloc[-1].reindex(tickers) if len(last) else pd.Series(np.nan, index=tickers)
        value = shares * px
        return value, ctx.cash + float(value.sum())

    def _payload(self, closes, rd, ctx, **extra) -> dict:
        s = self.scores.for_day(ctx.day, ctx.deadline) if self.scores is not None else None
        obs = observe.observation(
            closes, rd, self.book, self.anon, day=self.day_no,
            window_days=self.cfg.window_days, round_no=ctx.round, scores=s,
            earnings=self.earnings.to_next(ctx.day) if self.earnings is not None else None,
            calendar_date=str(ctx.day))
        obs["memory"] = self.journal[-8:]
        obs.update(extra)
        return obs

    def _submit(self, target: pd.Series, current: pd.Series, nav: float, tickers):
        self.book.traded_notional += float((target - current).abs().sum()) * nav
        self.book.weights = target
        return W.safe(target.clip(lower=0.0, upper=W.CAP).to_dict(), tickers)

    # ------------------------------------------------------------------ the round

    def __call__(self, ctx):
        self._t_round = time.perf_counter()
        tickers = ctx.market.tickers
        if self.anon is None:
            seed = (self.cfg.seed * 1_000_003 + ctx.day.toordinal()) if self.cfg.anonymize else None
            self.anon = observe.Anonymizer(tickers, seed)
            self.book = observe.BookState(pd.Series(0.0, index=tickers), [])
        if ctx.day != self._day:
            self._day, self._fired = ctx.day, set()
            self.day_no += 1
        value, nav = self._value(ctx, tickers)
        if not np.isfinite(nav) or nav <= 0:
            return None  # cannot value the book; holding beats guessing
        current = value / nav
        self.book.weights = current
        if ctx.round == 1:
            self.book.nav.append(nav)
        if not self.book.entered:
            return self._enter(ctx, tickers, current, nav) if ctx.round == 1 else None
        if ctx.round == 1:
            return self._review(ctx, tickers, current, nav) if self.cfg.review else None
        return self._events(ctx, tickers, current, nav) if self.cfg.events else None

    def _enter(self, ctx, tickers, current, nav):
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        tail = np.log(closes).diff().iloc[1:].tail(qs.SHAPE_DAYS)
        shapes = {n: qs.SHAPES[n](tail) for n in ("inverse_vol", "risk_parity")}
        if any(s is None for s in shapes.values()):
            return None  # not enough history for a shape: cash until there is
        self.hmm = observe.fit_regime(closes)
        rd = observe.readings(closes, self.hmm)
        rule = {"shape": "risk_parity", "exposure": rule_exposure(rd.p_turbulent_next),
                "avoid": [], "rationale": "rule: risk parity at the regime-blended exposure"}
        payload = self._payload(closes, rd, ctx, rule_proposal=rule)

        def check(d: EntryDecision):
            for code in d.avoid:
                self.anon.ticker(code)
            if len(d.avoid) >= len(tickers) - 3:
                raise ValueError("avoid list leaves fewer than 4 names")

        d, _ = self._ask("entry", payload, EntryDecision, check)
        w = shapes[d.shape].reindex(tickers).fillna(0.0)
        avoid = [self.anon.ticker(c) for c in d.avoid]
        if avoid:
            w[avoid] = 0.0
            w = pd.Series(compiler._water_fill(w.to_numpy(), 1.0, W.CAP), index=tickers)
        self.book.entered = True
        return self._submit(w * d.exposure, current, nav, tickers)

    def _review(self, ctx, tickers, current, nav):
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        rd = observe.readings(closes, self.hmm)
        rule = {"action": "hold", "exposure": None, "exit": [], "rationale": "rule: hold"}
        payload = self._payload(closes, rd, ctx, rule_proposal=rule)
        held = {self.anon.code(t) for t in tickers if current[t] > compiler.HELD}

        def check(d: ReviewDecision):
            if d.action == "set_exposure" and d.exposure is None:
                raise ValueError("set_exposure without an exposure")
            unknown = set(d.exit) - held
            if unknown:
                raise ValueError(f"exit names not held: {sorted(unknown)}")

        d, _ = self._ask("review", payload, ReviewDecision, check)
        target = current.copy()
        gross = float(current.sum())
        if d.action == "set_exposure" and abs(d.exposure - gross) >= self.cfg.band and gross > 0:
            target = current * (d.exposure / gross)
        for code in d.exit:
            target[self.anon.ticker(code)] = 0.0
        if (target - current).abs().max() <= compiler.HELD:
            return None
        return self._submit(target, current, nav, tickers)

    def _events(self, ctx, tickers, current, nav):
        held = [t for t in tickers if current[t] > compiler.HELD and t not in self._fired]
        if not held:
            return None
        why = {}
        last_round = len(calendar.rounds_for(ctx.day))
        if self.earnings is not None and ctx.round == last_round:
            nxt = self.earnings.to_next(ctx.day)
            for t in held:
                if nxt.get(t) == 1:
                    why[t] = "earnings reaction at the next open"
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        bars = ctx.recent_closes(1)
        if len(bars) and len(closes) > 21 and bars.index[-1] > closes.index[-1]:
            rets = np.log(closes).diff().tail(20)
            z = np.log(bars.iloc[-1] / closes.iloc[-1]) / rets.std()
            for t in held:
                if np.isfinite(z[t]) and abs(z[t]) >= self.cfg.sigma_trigger and t not in why:
                    why[t] = f"moved {z[t]:+.1f} daily sigmas since yesterday's close"
        if not why:
            return None
        self._fired |= set(why)
        codes = [self.anon.code(t) for t in why]
        rd = observe.readings(closes, self.hmm)
        payload = self._payload(
            closes, rd, ctx, triggers=[{"name": self.anon.code(t), "why": w} for t, w in why.items()],
            rule_proposal=rule_event(codes).model_dump())

        def check(d: EventDecision):
            extra = {c.name for c in d.calls} - set(codes)
            if extra:
                raise ValueError(f"calls for names with no trigger: {sorted(extra)}")

        d, _ = self._ask("event", payload, EventDecision, check)
        target = current.copy()
        for c in d.calls:
            if c.action == "exit":
                target[self.anon.ticker(c.name)] = 0.0
        if (target - current).abs().max() <= compiler.HELD:
            return None
        return self._submit(target, current, nav, tickers)


def desk(brain_factory, config: Optional[DeskConfig] = None, **kw):
    """A `windows.run_field` factory: a fresh desk and brain per window."""
    return lambda: Desk(brain_factory(), config, **kw)
