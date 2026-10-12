"""Desk v3, stage 2, variation B: code wakes the PM, and the PM may only make the book smaller.

Stage 1 buys one book and holds it. Over a 15-session window that leaves three things
unanswered: about two thirds of the 30 names report results, a held name can break, and a
gain can be booked before the window ends. Each of them reduces the book, so after entry
this desk can only hold or reduce (`schemas.ChainDecision`): trim a name by a quarter or a
half, exit it, or lower the whole book's gross, to all cash at most.

**Who decides.** Code watches every round and wakes the PM only when one of these happens:
- `before_results`: a held name reacts to results at the next open (the day's last round:
  its fill is the last before the reaction);
- `after_results`: a held name's release is public and its reaction session is trading
  (the first round from 2 on, once a bar of the session has ended);
- `move`: a held name has moved `MOVE_SIGMAS` daily sigmas since the last close;
- `filing`: a held name filed an 8-K of a serious kind (`MATERIAL_ITEMS`);
- `condition`: one of the plan's conditions fired (`conditions.Watch`), with the PM's own
  stated action as the proposal;
- `morning`: every morning after entry, so a gain can be booked.
Events of one round reach the PM in one call. There is no analyst and no associate: B
asks whether the PM's judgement at these moments beats holding at all. A layered chain
(variation A) can only filter what reaches the PM, so it cannot rescue a PM that loses.

**What the PM sees is its own book.** Under the overlay code holds most of the NAV in
cash and the PM is never told (`V3Config.overlay`). Every weight, gross, return,
drawdown and turnover it reads is at its book's scale (`marks["scale"]`, the PM's gross
over the gross bought), and its decision is scaled back down. Shown the real book, it
would see a gross it never chose, and a drawdown a tenth of the one its conditions mean.

**What code enforces**, each refusing the whole answer, never a part:
- sell-only: a line on a name not held, a new gross not below today's, or anything that
  adds, is refused;
- scope: outside the morning, lines only on names an event named;
- the budget: `BUDGET` interventions a window; all cash in the last `ENDGAME_SESSIONS`
  sessions spends none;
- a reduction under `MIN_TRADE` of the PM's book is a hold and spends nothing.
A failed, late or refused PM holds. Every fallback is counted.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from icaif import calendar, compiler, filings as F, quant_strategies as qs, sim
from icaif import weights as W
from icaif.agents import conditions as C
from icaif.agents import observe, prompts_chain
from icaif.agents.schemas import REDUCTIONS, ChainDecision
from icaif.agents.tradelist import NUDGE
from icaif.agents.triggers import move_since_close, priced_from
from icaif.agents.v3 import FALLBACK_GROSS, ROLE_TIER, HoldBrain, V3Config, V3Desk, condition_errors, strip

BUDGET = 2
ENDGAME_SESSIONS = 3
MIN_TRADE = 0.005           # of the PM's book: below it a reduction is a hold, as in v1
MOVE_SIGMAS = 3.0           # the v1 and v2 trigger's unit and level
# 8-K items that can mean something broke. Results (2.02) have their own wake; the rest
# (agreements, votes, Reg FD, exhibits) are routine.
MATERIAL_ITEMS = ("1.03", "2.04", "2.05", "2.06", "3.01", "4.01", "4.02", "5.02")
VOL_SESSIONS = 60
CHAIN_SLOT = 600            # seconds from the round's start: inside the runner's lead
ROLES = dict(ROLE_TIER, chain_pm="deep")


@dataclass
class ChainConfig(V3Config):
    budget: int = BUDGET
    morning: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.slots = dict(self.slots, chain_pm=CHAIN_SLOT)


class ChainDesk(V3Desk):
    def __init__(self, brains: dict, config: Optional[ChainConfig] = None, earnings_history=None, **kw):
        cfg = config or ChainConfig()
        if not isinstance(cfg, ChainConfig):
            raise TypeError("ChainDesk takes a ChainConfig")
        super().__init__(brains, cfg, earnings_history=earnings_history, **kw)
        self.watch: Optional[C.Watch] = None
        self.woken: set = set()
        self.interventions: list[dict] = []
        self.wakes: list[dict] = []
        self._chain_filings_to: Optional[str] = None

    def _tier(self, role: str) -> str:
        return ROLES[role]

    def _system(self, role: str) -> str:
        return prompts_chain.SYSTEM[role] if role == "chain_pm" else super()._system(role)

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        out = super().state()
        out["chain"] = {"watch": self.watch.state() if self.watch is not None else None,
                        "woken": sorted(self.woken), "interventions": self.interventions,
                        "wakes": self.wakes, "filings_to": self._chain_filings_to}
        return out

    def restore(self, state, tickers):
        super().restore(state, tickers)
        ch = (state or {}).get("chain") or {}
        self.watch = C.Watch.from_state(ch["watch"]) if ch.get("watch") else None
        self.woken = set(ch.get("woken", []))
        self.interventions = list(ch.get("interventions", []))
        self.wakes = list(ch.get("wakes", []))
        self._chain_filings_to = ch.get("filings_to")

    # ------------------------------------------------------------------ the round

    def _decide(self, ctx, tickers):
        if not self.book.entered:
            w = super()._decide(ctx, tickers)
            if self.book.entered:
                self._arm(ctx, tickers)
            return w
        value, nav = self._value(ctx, tickers)
        if not np.isfinite(nav) or nav <= 0:
            self._note = "the book cannot be valued (a held name has no price); held"
            return None
        current = value / nav
        self.book.weights = current
        self._current = current
        if ctx.round == 1:
            self.book.nav.append(nav)
        if self.watch is None:      # restored from a stage-1 state: measured from now
            self._arm(ctx, tickers)
        return self._round(ctx, tickers, current, nav)

    def _arm(self, ctx, tickers) -> None:
        """The plan's conditions and their marks, fixed at entry (`conditions.entry_marks`)."""
        e = self.entry or {}
        if e.get("source") == "brain" and e.get("gross"):
            scale = float(e["pm_gross"]) / float(e["gross"])
        elif self.cfg.overlay is not None:
            # Code bought the fallback at the overlay's share; the PM reads it as the
            # 75% inverse-vol book it stands for.
            scale = FALLBACK_GROSS / self.cfg.overlay
        else:
            scale = 1.0
        marks = C.entry_marks(ctx, tickers, self.vol, scale=scale)
        self.watch = C.Watch((self.plan or {}).get("conditions", []), marks)
        self._chain_filings_to = marks["at"]

    def _round(self, ctx, tickers, current, nav):
        held = [t for t in tickers if current[t] > compiler.HELD]
        self._nav = nav
        rows = self.watch.check(ctx, to_ticker=self.anon.ticker, positions=self._positions(),
                                nav=self.journal.nav(), base_nav=self._base_nav(),
                                start_nav=self.journal.start_nav,
                                history=self.tags.history, filings=self.filings)
        events = self._events(ctx, held, rows)
        morning = self.cfg.morning and ctx.round == 1 and self.day_no >= 2
        if not held or (not events and not morning):
            return None
        self._t_chain = self._t_round
        wake = {"day": self.day_no, "round": ctx.round, "morning": morning,
                "events": [{"id": e["id"], "kind": e["kind"], "name": e.get("name")} for e in events]}
        self.wakes.append(wake)
        payload = self._view(ctx, tickers, current, held, events, rows, morning)
        scope = None if morning else {e["name"] for e in events if e.get("name")}
        d = self._ask_one(ctx, "chain_pm", payload, ChainDecision,
                          lambda a: self._check(a, held, scope, {e["id"] for e in events}, current))
        w, reason = None, None
        if d is None:
            reason = f"chain_pm {self.chain[-1]['source']}: {self.chain[-1]['reason']}"
            self.fallbacks["chain:hold"] += 1
        else:
            w, reason = self._execute(ctx, d, current, nav, tickers)
            self.watch.rewrite(keep=[c["id"] for c in self.watch.conditions
                                     if c["id"] not in set(d.drop_conditions)],
                               add=[c.model_dump() for c in d.add_conditions])
        wake["outcome"] = "traded" if w is not None else "held"
        if reason:
            wake["reason"] = reason
        self.log.append({"day": self.day_no, "round": ctx.round, "role": "chain_pm",
                         "brain": getattr(self.brains["deep"], "name", "?"),
                         "source": "brain" if d is not None else "fallback", "reason": reason,
                         "decision": d.model_dump() if d is not None else None,
                         "events": wake["events"], "morning": morning, "traded": w is not None,
                         "latency_s": round(time.perf_counter() - self._t_chain, 3)})
        return w

    # ------------------------------------------------------------------ what wakes the PM

    def _events(self, ctx, held, rows) -> list[dict]:
        out = []
        deadline = pd.Timestamp(ctx.deadline)
        last_round = ctx.round == len(calendar.rounds_for(ctx.day))
        for r in rows:
            if r["new"]:
                out.append({"id": r["id"], "kind": "condition", "name": r["name"],
                            "condition": {k: r[k] for k in ("kind", "name", "threshold", "item", "value")},
                            "proposal": r["action"]})
        nxt = self.earnings.to_next(ctx.day) if self.earnings is not None else {}
        hist = self.tags.history
        for t in held:
            code = self.anon.code(t)
            if last_round and nxt.get(t) == 1 and self._once(f"before:{t}:{ctx.day}"):
                out.append({"id": f"before_results:{code}", "kind": "before_results", "name": code,
                            "tag": self.tags.tag(ctx, t, "earnings", sessions_to=1)})
            if ctx.round >= 2 and hist is not None:
                acc = hist.release(t, deadline)
                p = priced_from(acc, ctx.market.days) if acc is not None else None
                if (p is not None and p.date() == ctx.day and acc > pd.Timestamp(self.watch.marks["at"])
                        and move_since_close(ctx, t)[0] is not None and self._once(f"after:{t}:{acc}")):
                    out.append({"id": f"after_results:{code}", "kind": "after_results", "name": code,
                                "tag": self.tags.tag(ctx, t, "earnings", at=acc),
                                "release": self._texts(t, acc - pd.Timedelta(hours=2), deadline, "2.02")})
            if ctx.round >= 2:
                _, z = move_since_close(ctx, t)
                if z is not None and abs(z) >= MOVE_SIGMAS and self._once(f"move:{t}:{ctx.day}"):
                    out.append({"id": f"move:{code}", "kind": "move", "name": code,
                                "tag": self.tags.tag(ctx, t, "move")})
        if self.filings is not None:
            after = pd.Timestamp(self._chain_filings_to)
            new = F.new(self.filings, after, deadline) if after < deadline else self.filings.iloc[:0]
            for t, g in new.groupby("ticker"):
                if t not in set(held):
                    continue
                bad = g[[any(i.strip() in MATERIAL_ITEMS for i in str(x).split(",")) for x in g["items"]]]
                if len(bad) and self._once(f"filing:{t}:{bad['accepted'].max()}"):
                    code = self.anon.code(t)
                    out.append({"id": f"filing:{code}", "kind": "filing", "name": code,
                                "tag": self.tags.tag(ctx, t, "8k", at=bad["accepted"].max()),
                                "filings": observe.filing_rows(bad, deadline, real=not self.anon.enabled)})
            self._chain_filings_to = deadline.isoformat()
        return out

    def _once(self, key: str) -> bool:
        if key in self.woken:
            return False
        self.woken.add(key)
        return True

    def _texts(self, t, lo, hi, item: str) -> list[dict]:
        if self.filings is None:
            return []
        f = self.filings[(self.filings["ticker"] == t) & (self.filings["accepted"] > lo)
                         & (self.filings["accepted"] <= hi)]
        f = f[[item in [i.strip() for i in str(x).split(",")] for x in f["items"]]]
        return observe.filing_rows(f, hi, real=not self.anon.enabled) if len(f) else []

    # ------------------------------------------------------------------ what the PM reads

    def _scale(self) -> float:
        return float(self.watch.marks.get("scale", 1.0))

    def _to_pm(self, nav: float) -> float:
        """Real weights times this is the PM's book: it holds `scale` times every share,
        so its P&L since the start (the entry's fee included) is `scale` times the real one.
        Scaling weights by `scale` alone would divide by the real NAV, and overstate the
        PM's gross after a rise (by 2% of it after a 3% gain at a tenth invested)."""
        k, a = self._scale(), self.journal.start_nav
        if not a or not np.isfinite(nav) or nav <= 0:
            return k
        return k * nav / (a + k * (nav - a))

    def _positions(self) -> dict:
        return {t: {"entry_price": p["entry_price"], "entry_at": p["entry_at"]}
                for t, p in self.journal.positions.items() if p.get("entry_price")}

    def _base_nav(self) -> Optional[float]:
        for e in self.journal.rounds:
            f = e.get("fill")
            if f and f.get("nav_after"):
                return float(f["nav_after"])
        return None

    def _view(self, ctx, tickers, current, held, events, rows, morning) -> dict:
        k = self._to_pm(self._nav)
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        names = sorted({e["name"] for e in events if e.get("name")})
        obs = strip(self._payload(closes, observe.readings(closes, None), ctx, "v3", held=held,
                                  triggered=[self.anon.ticker(c) for c in names]), self.cfg.streams)
        by_code = {r["name"]: r for r in obs["names"]}
        fields = self.journal.name_fields(real=not self.anon.enabled)
        book = []
        for t in held:
            code = self.anon.code(t)
            row = {"name": code, "weight": round(float(current[t]) * k, 4),
                   **{f: fields.get(t, {}).get(f) for f in ("entry_day", "gain_since_entry",
                                                          "peak_gain_since_entry")}}
            if code in names:
                src = by_code.get(code, {})
                row.update({f: src[f] for f in ("ret_1d", "ret_5d", "vol_ann_20d", "vol_ann_har_1d",
                                                "earnings_in_sessions", "headlines") if f in src})
            book.append(row)
        left = self.cfg.window_days - self.day_no
        return {
            "clock": dict(obs["clock"], sessions_left_after_today=left),
            "wake": ["morning"] * morning + [e["id"] for e in events],
            "events": events,
            "plan": {"thesis": (self.plan or {}).get("thesis", "code bought the inverse-vol book; there is no plan"),
                     "conditions": [{k_: r[k_] for k_ in ("id", "kind", "name", "threshold", "item", "action",
                                                          "value", "to_fire", "fired", "note") if r.get(k_) is not None
                                     or k_ in ("value", "fired")} for r in rows]},
            "book": {"gross": round(float(current.sum()) * k, 4), "names": book},
            "market": {f: obs["market"][f] for f in ("basket_ret_1d", "basket_ret_5d", "vol_vs_3y_median")
                       if f in obs["market"]},
            "horizon": self._horizon(closes, current * k, left),
            "budget": {"interventions_left": max(self.cfg.budget - self._spent(), 0),
                       "free": f"all cash in the last {ENDGAME_SESSIONS} sessions spends none"},
            "settled": self._settled(ctx),
            "previews": self._previews(current * k, names),
        }

    def _horizon(self, closes, book, left: int) -> dict:
        """`book`: the PM's weights."""
        k, a = self._scale(), self.journal.start_nav
        path = [float(e["nav"]) for e in self.journal.rounds if e.get("nav")]
        nav = self.journal.nav()
        out = {}
        if a and nav:
            pm = [a + k * (x - a) for x in path + [nav]]
            peak = np.maximum.accumulate(pm)
            ret = pm[-1] / a - 1
            out.update(**{"return": round(ret, 4)},
                       max_drawdown=round(float(np.max(1 - np.array(pm) / peak)), 4),
                       if_all_cash_now={"return_locked": round(ret - float(book.sum()) * sim.FEE_RATE, 4),
                                        "turnover": round(float(book.sum()), 4)})
        r = np.expm1(np.log(closes).diff().iloc[1:].tail(VOL_SESSIONS))
        w = book.reindex(r.columns).fillna(0.0)
        daily = (r.fillna(0.0) @ w).std()
        if np.isfinite(daily) and "return" in out:
            band = float(daily * np.sqrt(max(left, 0)))
            out.update(vol_ann=round(float(daily * np.sqrt(252)), 4),
                       if_held_to_end_one_sigma=[round(out["return"] - band, 4), round(out["return"] + band, 4)])
        return out

    def _previews(self, book, names) -> dict:
        """`book`: the PM's weights."""
        out = {f"exit_{c}": round(float(book[self.anon.ticker(c)]), 4) for c in names}
        out["half_the_book"] = round(float(book.sum()) / 2, 4)
        out["all_cash"] = round(float(book.sum()), 4)
        return {"turnover": out, "fee_bp_of_book": {n: round(v * sim.FEE_RATE * 1e4, 1) for n, v in out.items()}}

    def _settled(self, ctx) -> list[dict]:
        """Each earlier reduction since its fill: what the names sold have done, in bp of
        the PM's book (positive: the sale saved that much)."""
        out = []
        bars = ctx.recent_closes(1)
        last = bars.iloc[-1] if len(bars) else pd.Series(dtype=float)
        for iv in self.interventions:
            e = next((x for x in self.journal.rounds if x["key"] == iv["key"]), None)
            fill = (e or {}).get("fill")
            if not fill:
                continue
            for t, sold in iv["sold_pm"].items():
                px = fill["prices"].get(t)
                now = last.get(t, np.nan)
                if not px or not np.isfinite(now):
                    continue
                move = now / px - 1
                out.append({"day": iv["day"], "round": iv["round"], "name": self.anon.code(t),
                            "sold": round(sold, 4), "move_since": round(move, 4),
                            "saved_bp": round(-sold * move * 1e4, 1)})
        return out

    # ------------------------------------------------------------------ the decision

    def _spent(self) -> int:
        return sum(1 for iv in self.interventions if iv["counts"])

    def _check(self, d: ChainDecision, held, scope, ids: set, current) -> None:
        """Every reason the answer is refused whole (raised, so the round holds)."""
        errs = []
        codes = [x.name for x in d.lines]
        if d.action == "hold" and (d.lines or d.gross is not None):
            errs.append("a hold carries no lines and no gross")
        if d.action == "reduce" and not d.lines and d.gross is None:
            errs.append("a reduction needs lines or a gross")
        held_codes = {self.anon.code(t) for t in held}
        twice = sorted({c for c in codes if codes.count(c) > 1})
        if twice:
            errs.append(f"names listed twice: {twice}")
        not_held = sorted(set(codes) - held_codes)
        if not_held:
            errs.append(f"not in the book, so nothing to reduce: {not_held}")
        if scope is not None:
            outside = sorted(set(codes) - scope)
            if outside:
                errs.append(f"outside the morning only the event names may be reduced, not {outside}")
        today = float(current.sum()) * self._to_pm(self._nav)
        if d.gross is not None and d.gross >= today - 1e-6:
            errs.append(f"gross {d.gross:g} is not below today's {today:.4f}")
        unknown = sorted(set(d.cites) - ids)
        if unknown:
            errs.append(f"cites events that did not happen this round: {unknown}")
        ids_now = {c["id"] for c in self.watch.conditions}
        bad_drop = sorted(set(d.drop_conditions) - ids_now)
        if bad_drop:
            errs.append(f"drops conditions not in the plan: {bad_drop}")
        keep = held_codes - {x.name for x in d.lines if x.to == "exit"}
        for i, c in enumerate(d.add_conditions):
            errs += [f"new condition {i + 1} ({c.kind}): {e}" for e in condition_errors(c, keep)]
        if errs:
            raise ValueError("; ".join(errs))

    def _execute(self, ctx, d: ChainDecision, current, nav, tickers) -> tuple:
        if d.action == "hold":
            return None, None
        k = self._to_pm(nav)
        cur = current * k
        new = cur.copy()
        for x in d.lines:
            t = self.anon.ticker(x.name)
            new[t] = cur[t] * REDUCTIONS[x.to]
        if d.gross is not None and float(new.sum()) > d.gross:
            new = new * (d.gross / float(new.sum())) if new.sum() > 0 else new
        turnover = float((new - cur).abs().sum())
        if turnover < MIN_TRADE:
            self.fallbacks["chain:below_min_trade"] += 1
            return None, f"a reduction of {turnover:.4f} of the book is under {MIN_TRADE}: held"
        endgame = self.day_no > self.cfg.window_days - ENDGAME_SESSIONS and float(new.sum()) <= compiler.HELD
        if not endgame and self._spent() >= self.cfg.budget:
            self.fallbacks["chain:over_budget"] += 1
            return None, f"over budget: {self._spent()} of {self.cfg.budget} interventions spent; held"
        real = new / k
        weights = W.safe({t: v + NUDGE if v > 0 else 0.0 for t, v in real.items()}, tickers)
        w = self._take(weights, float((real - current).abs().sum()), current, nav)
        key = ctx.round_id or f"day{self.day_no}-r{ctx.round}"
        self.interventions.append({"key": key, "day": self.day_no, "round": ctx.round, "counts": not endgame,
                                   "turnover_pm": round(turnover, 4),
                                   "sold_pm": {t: float(cur[t] - new[t]) for t in tickers if cur[t] - new[t] > 1e-9}})
        return w, None


class ChainHoldBrain(HoldBrain):
    """Stage 1's hold brain, and a PM that holds at every wake: a chain desk on it must
    trade exactly as the stage-1 desk on the hold brain, which trades as the hold."""

    name = "chain_hold"

    def decide(self, role, system, payload, schema, timeout):
        if schema is ChainDecision:
            return ChainDecision(action="hold", lines=[], gross=None, drop_conditions=[],
                                 add_conditions=[], cites=[], rationale="code: hold")
        return super().decide(role, system, payload, schema, timeout)
