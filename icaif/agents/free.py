"""The free desk: Opus decides the whole book each morning, with no levers in between.

The test of the question "what if we just ask Opus?". Two arms share everything but
the prompt:

- **blank**: the game's rules and the observation, nothing else. No backtest evidence,
  no proposal. What an LLM agent does left to itself.
- **informed**: the same, plus what our backtests found and the backtested book
  (`rule_proposal`) as the bar to beat.

Code keeps what must never be wrong, as in the desk: the observation is point in time,
the answer is validated (known codes, each weight at most 30%, a book summing to at
most 100%; rejected rather than rescaled), and anything invalid, late, refused or over
budget becomes the fallback: the inverse-vol book at 75% on day 1, then hold. That
fallback is the book we would submit, so a failing Opus costs nothing against it.
"""

import numpy as np
import pandas as pd

from icaif import baselines, compiler, quant_strategies as qs
from icaif import weights as W
from icaif.agents import observe
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.schemas import FreeDecision

ARMS = {"blank": "free_blank", "informed": "free_informed"}
GROSS = 0.75


class FreeDesk(Desk):
    def __init__(self, brain, arm: str, config=None, **kw):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {sorted(ARMS)}")
        cfg = config or DeskConfig(anonymize=True)
        cfg.timeouts = {**cfg.timeouts, ARMS[arm]: cfg.timeouts.get("entry", 240.0)}
        super().__init__(brain, cfg, **kw)
        self.arm, self.role = arm, ARMS[arm]

    def _rule(self, ctx, tickers) -> dict:
        """The book we would submit, exactly: `baselines.scaled(InverseVolHold, 0.75)`.

        Its own code, not a lookalike: an inverse-vol book on daily closes instead of
        hourly bars scored 0.13-0.20 worse over the same 2025-26 windows, and every
        Opus-vs-fallback difference would have carried that gap.
        """
        if self.book.entered:
            return {"action": "hold", "weights": [], "rationale": "rule: hold"}
        w = baselines.scaled(baselines.InverseVolHold, GROSS)()(ctx)
        if w is None:
            return {"action": "hold", "weights": [], "rationale": "rule: not enough history"}
        return {"action": "rebalance",
                "weights": [{"name": self.anon.code(t), "weight": float(w[t])}
                            for t in tickers if w[t] > 0],
                "rationale": "rule: inverse-vol at 75%, bought once"}

    def _decide(self, ctx, tickers):
        """Once a day, at round 1; `Desk.__call__` keeps the clock and the journal."""
        if ctx.round != 1:
            return None
        value, nav = self._value(ctx, tickers)
        if not np.isfinite(nav) or nav <= 0:
            return None
        current = value / nav
        self.book.weights = current
        self._current = current
        self.book.nav.append(nav)
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        if self.hmm is None:
            self.hmm = observe.fit_regime(closes)
        rd = observe.readings(closes, self.hmm)
        rule = self._rule(ctx, tickers)
        extra = {"rule_proposal": rule} if self.arm == "informed" else {}
        payload = self._payload(closes, rd, ctx, self.role, **extra)

        def check(d: FreeDecision):
            codes = [x.name for x in d.weights]
            for c in codes:
                self.anon.ticker(c)
            if len(set(codes)) != len(codes):
                raise ValueError("a name listed twice")
            if sum(x.weight for x in d.weights) > 1.0 + 1e-9:
                raise ValueError(f"book sums to {sum(x.weight for x in d.weights):.4f} > 1")

        d, _ = self._ask(self.role, payload, FreeDecision, check, rule=rule)
        if d.action == "hold":
            if not self.book.entered and current.sum() <= compiler.HELD:
                return None  # holding cash before entry: still not entered
            self.book.entered = True
            return None
        target = pd.Series(0.0, index=tickers)
        for x in d.weights:
            target[self.anon.ticker(x.name)] = x.weight
        target = target.map(W.floor_to_grid)
        self.book.entered = True
        if (target - current).abs().max() <= compiler.HELD:
            return None
        return self._submit(target, current, nav, tickers)
