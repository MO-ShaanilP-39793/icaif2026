"""The desk: a `sim.Strategy` that asks its roles, validates, falls back and executes.

One code path serves the backtest, the replay and the live round: the desk reads a
`RoundContext`, and the live runner builds one from fresh data.

The rule each role falls back to is the best entry-only candidate of
`tools/quant_report.py`: risk parity, entry exposure blended by the HMM's probability
of turbulence (0.85 calm, 0.30 turbulent), then hold, and hold through every event. A
rule desk (`RuleBrain`) reproduces that candidate trade for trade, which a test
checks, so whatever an LLM desk scores differently is the LLM's doing.

The roles also read our own signals (`signals`: HAR vol, the score's rank, sessions to
earnings), and have levers built on them: the Strategist can tilt risk parity toward
the scores (Black-Litterman, at a level whose book it is shown) and name exclusions
with their cause; the Risk review can rebalance to the entry's recipe on today's
inputs, with a stated reason, at most `max_rebalances` times. The rule uses none of
them, so they change what an LLM can do and never what the fallback does.

Every round also passes through the desk's `journal` (Roadmap step 4): the book is
reconciled against it before anything is decided, each role reads it back as `memory`
and per-name entry fields, and the round's decisions go into it after. It changes what
a role is shown, never what the rule decides, so the rule desk still equals its
candidate trade for trade.

Roadmap step 5 adds news and profit booking. The Risk review and the Event analyst read
each held name's headlines (live only) and every role its recent 8-K filings, and a new
8-K for a held name wakes the Event analyst beside earnings and the 3-sigma move. Both
roles can trim a held name (a quarter or half of it, with a cause), floored at
`trim.MIN_TRIM` of NAV and capped at `max_trims` a window. The rule's own trim
(`trim.TrimRule`) is passed in only once it has won its gate; without it the rule
proposes no trim and the rule desk is still its candidate.
"""

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from icaif import calendar, compiler, quant, quant_strategies as qs, sim
from icaif import filings as F
from icaif import trim as TR
from icaif import weights as W
from icaif.agents import observe, prompts, signals
from icaif.agents.brains import BrainError, rule_event
from icaif.agents.journal import Journal
from icaif.agents.schemas import EntryDecision, EventDecision, ReviewDecision
from icaif.vol import MARKET

E_CALM, E_TURBULENT = 0.85, 0.30  # = quant_strategies.Regime's defaults


@dataclass
class DeskConfig:
    window_days: int = 15
    review: bool = True
    events: bool = True
    band: float = 0.05          # a review exposure change smaller than this is a hold
    # Rebalances a window allows, and the turnover below which one is a hold. Each pays
    # 0.1% of what it moves and a turnover rank; "occasional" is enforced, not asked.
    max_rebalances: int = 2
    rebalance_min_turnover: float = 0.02
    sigma_trigger: float = 3.0  # |move since yesterday's close| in daily sigmas
    anonymize: bool = False
    seed: int = 0
    round_budget_s: float = 360.0
    timeouts: dict = field(default_factory=lambda: {"entry": 240.0, "review": 120.0,
                                                    "event": 120.0})
    # A journal error raises (replays, tests) or is recorded and the round goes on
    # without memory (live): the memory informs a role, and a bug in it must not cost
    # the submitted book its entry.
    journal_strict: bool = True
    # Trims a window allows, rule and agent together. Each pays the fee on what it sells
    # and a turnover rank, so "part of a winner, sometimes" is enforced, not asked.
    max_trims: int = TR.MAX_TRIMS
    # The roles that read held names' headlines: the two that can act on a held name.
    headline_roles: tuple = ("review", "event")
    # The roles shown the whole universe's ranking as context. The free desk's arms are
    # an experiment on what each prompt is given, so they are not.
    universe_roles: tuple = ("entry", "review", "event")


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
                 earnings: Optional[EarningsCalendar] = None,
                 context: Optional[pd.DataFrame] = None,
                 fomc=None, filings: Optional[pd.DataFrame] = None,
                 news_dir=None, vol: Optional[signals.VolForecasts] = None,
                 trim: Optional[TR.TrimRule] = None,
                 universe_scores: Optional[signals.UniverseScores] = None):
        """`context`: `macro.wide(...)` closes; `fomc`: a `macro.FomcCalendar`;
        `filings`: `filings.fetch(...)` events (a `text` column, live); `news_dir`: the
        headline archive, read only with real names (headlines name companies); `vol`:
        HAR forecasts; `trim`: the rule's own trim, once it has won its gate (it reads
        `vol`); `universe_scores`: every universe name's score, shown as context."""
        self.brain = brain
        self.cfg = config or DeskConfig()
        self.scores, self.earnings, self.vol = scores, earnings, vol
        self.context, self.fomc, self.filings, self.news_dir = context, fomc, filings, news_dir
        self.trim_rule = trim
        self.universe_scores = universe_scores
        self.ucodes: Optional[observe.UniverseCodes] = None
        # Trims spent this window, each {ticker, day, round, fraction, role}; the window's
        # first day; and the acceptance time up to which 8-Ks have been triggered on. A
        # desk rebuilt without the last would wake the analyst on the same filing again.
        self.trims: list[dict] = []
        self._start: Optional[date] = None
        self._filings_to: Optional[pd.Timestamp] = None
        # The entry's recipe (shape, views, names kept out), what the signals read at
        # entry, and rebalances spent: a rebalance re-applies the recipe, so without it
        # a restored desk would rebuild a book the Strategist never chose.
        self.recipe: Optional[dict] = None
        self.at_entry: Optional[dict] = None
        self.rebalances = 0
        self.anon: Optional[observe.Anonymizer] = None
        self.book: Optional[observe.BookState] = None
        self.hmm = None
        self.day_no = 0
        self._day = None
        self._fired: set = set()
        self.journal = Journal()
        self.journal_errors: list[str] = []
        self._journal_error: Optional[str] = None
        self._journal_backup: Optional[dict] = None
        self._current: Optional[pd.Series] = None
        self._note: Optional[str] = None
        self.log: list[dict] = []
        self._t_round = 0.0

    # ------------------------------------------------------------------ plumbing

    def _ask(self, role: str, payload: dict, schema: type[BaseModel], check,
             rule: Optional[dict] = None) -> tuple:
        """(decision, source): the brain's answer if it passes `check`, else the rule's.

        `rule` is the fallback when the payload must not show it (the blank arm of the
        free desk); otherwise the payload's `rule_proposal` is both.
        """
        rule = schema.model_validate(rule if rule is not None else payload["rule_proposal"])
        left = self.cfg.round_budget_s - (time.perf_counter() - self._t_round)
        timeout = min(self.cfg.timeouts[role], left)
        t0 = time.perf_counter()
        source, reason, decision = "brain", None, None
        if timeout < 5.0:
            source, reason = "fallback", f"round budget spent ({left:.0f}s left)"
        else:
            try:
                decision = self.brain.decide(role, prompts.SYSTEM[role], payload, schema, timeout)
                self._only_tradeable(decision)
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
        return decision, source

    def _only_tradeable(self, decision) -> None:
        """Refuse an answer whose levers name anything but the 30.

        The universe block shows about 70 names the desk cannot trade. A role that
        avoided, exited, trimmed or weighted one of them would otherwise reach a KeyError
        deep in a lever, or, where a lever reads names leniently, a decision about a book
        that does not exist. Checked once here for every lever and role, so a lever added
        later is covered without its own check remembering to be.
        """
        names = list(getattr(decision, "exit", []) or [])
        for field_ in ("avoid", "trim", "calls", "weights"):
            names += [x.name for x in getattr(decision, field_, []) or []]
        for n in names:
            if n in self.anon.to_ticker:
                continue
            if self.ucodes is not None and self.ucodes.is_universe_name(n):
                raise ValueError(f"{n} is in universe_context only and is not tradeable; "
                                 f"levers name only the 30 in `names`")
            raise ValueError(f"{n} is not one of the 30 tradeable names")

    def _universe(self, ctx, role: str) -> Optional[dict]:
        if self.universe_scores is None or role not in self.cfg.universe_roles:
            return None
        ranks = self.universe_scores.for_day(ctx.day, ctx.deadline)
        return observe.universe_block(ranks, self.anon, self.ucodes) if len(ranks) else None

    def _value(self, ctx, tickers):
        shares = pd.Series(ctx.shares, dtype=float).reindex(tickers).fillna(0.0)
        last = ctx.recent_closes(1)
        px = last.iloc[-1].reindex(tickers) if len(last) else pd.Series(np.nan, index=tickers)
        # Only held names need a price, and a held name without one makes the book
        # unvaluable (NaN NAV, so the round holds). pandas' default sum skips the NaN
        # and values the book without that name: every weight wrong, nothing looking it.
        value = shares * px.where(shares != 0, 0.0)
        return value, ctx.cash + float(value.sum(skipna=False))

    def _macro(self, ctx) -> Optional[dict]:
        if self.context is None and self.fomc is None:
            return None
        from icaif import macro

        out = macro.readings(self.context, ctx.day, self.cfg.anonymize) if self.context is not None else {}
        if self.fomc is not None:
            out.update(self.fomc.block(ctx.day, ctx.market.days))
        return out

    def _signals(self, ctx, tickers) -> dict:
        """Today's HAR vols and score ranks per ticker and for the basket, unrounded.

        `raw_scores` rides along for the views; it is dropped before the agent sees
        anything, which reads the rank.
        """
        names = {t: {} for t in tickers}
        market, raw = {}, None
        if self.vol is not None:
            ann = signals.annualised(self.vol.for_day(ctx.day, ctx.deadline))
            for t in tickers:
                names[t].update(vol_ann_har_1d=ann.at[t, "har_h1"], vol_ann_har_3d=ann.at[t, "har_h3"])
            market = {"vol_ann_har_1d": ann.at[MARKET, "har_h1"],
                      "vol_ann_har_3d": ann.at[MARKET, "har_h3"]}
        if self.scores is not None:
            raw = self.scores.for_day(ctx.day, ctx.deadline).reindex(tickers)
            ranks = signals.score_ranks(raw)
            for t in tickers:
                names[t]["model_score_rank"] = ranks[t]
        return {"names": names, "market": market, "raw_scores": raw}

    def _payload(self, closes, rd, ctx, role: str, sig: Optional[dict] = None, previews=None,
                 held=(), triggered=(), **extra) -> dict:
        sig = sig if sig is not None else self._signals(ctx, list(closes.columns))
        obs = observe.observation(
            closes, rd, self.book, self.anon, day=self.day_no,
            window_days=self.cfg.window_days, round_no=ctx.round,
            signals={k: v for k, v in sig.items() if k != "raw_scores"},
            at_entry=self.at_entry if self.book.entered else None, previews=previews,
            earnings=self.earnings.to_next(ctx.day) if self.earnings is not None else None,
            macro=self._macro(ctx),
            filings=(F.recent(self.filings, ctx.deadline)
                     if self.filings is not None else None),
            news=self._headlines(ctx, role, held, triggered),
            calendar_date=str(ctx.day), positions=self._name_fields(),
            universe=self._universe(ctx, role))
        obs["memory"] = self._memory()
        obs.update(extra)
        return obs

    def _headlines(self, ctx, role: str, held, triggered) -> Optional[dict]:
        """{ticker: headlines} for each held name, in the roles that act on held names.

        A triggered name shows its newest few in full, naming the company first; any other
        held name its titles that name the company. Never anonymised: a headline names
        the company, which is the one thing a replay's codes exist to hide.
        """
        if (self.news_dir is None or self.cfg.anonymize or role not in self.cfg.headline_roles
                or not len(held)):
            return None
        from icaif import news

        known = news.known_at(ctx.deadline, self.news_dir)
        out = {t: [] for t in held}
        hot = [t for t in held if t in set(triggered)]
        full = news.recent(known, ctx.deadline, hot, observe.HEADLINES_TRIGGERED,
                           observe.HEADLINE_HOURS)
        rest = news.recent(known, ctx.deadline, [t for t in held if t not in set(hot)],
                           observe.HEADLINES_HELD, observe.HEADLINE_HOURS)
        rest = rest[rest["names_it"]] if len(rest) else rest
        for rows, whole in ((full, True), (rest, False)):
            for t, g in rows.groupby("ticker"):
                out[t] = observe.headline_rows(g, ctx.deadline, full=whole)
        return out

    # ------------------------------------------------------------------ the journal

    def _memory(self) -> dict:
        if self._journal_error is not None:
            why = "" if self.anon.enabled else f" ({self._journal_error[:120]})"
            return {"unavailable": f"the journal failed this round{why}; the book shown is the "
                                   f"one the round started from"}
        return self.journal.memory(self.anon.code, real=not self.anon.enabled)

    def _name_fields(self) -> Optional[dict]:
        if self._journal_error is not None:
            return None
        return self.journal.name_fields(real=not self.anon.enabled)

    def _journal_failed(self, err: Exception) -> None:
        """Back to the journal as it stood before this round: half a reconcile on disk
        would read next round as a book that changed without an order."""
        self._journal_error = f"{type(err).__name__}: {err}"
        self.journal_errors.append(self._journal_error)
        self.journal = Journal.from_json(self._journal_backup)

    def _journal_open(self, ctx) -> None:
        self._journal_error, self._current, self._note = None, None, None
        self._journal_backup = None if self.cfg.journal_strict else self.journal.to_json()
        try:
            self.journal.open_round(ctx, self.day_no)
        except Exception as err:  # noqa: BLE001 - recorded; the round goes on without memory
            if self.cfg.journal_strict:
                raise
            self._journal_failed(err)

    def _journal_close(self, w: Optional[dict], n0: int) -> None:
        if self._journal_error is not None:
            return   # restored to the last good journal; this round is not on record
        try:
            self.journal.close_round(self.log[n0:], w, self._current, note=self._note)
        except Exception as err:  # noqa: BLE001
            if self.cfg.journal_strict:
                raise
            self._journal_failed(err)

    def _submit(self, target: pd.Series, current: pd.Series, nav: float, tickers):
        self.book.traded_notional += float((target - current).abs().sum()) * nav
        self.book.weights = target
        return W.safe(target.clip(lower=0.0, upper=W.CAP).to_dict(), tickers)

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        """Everything the desk carries from one round to the next, as JSON.

        Live, each round runs in its own process. A desk rebuilt from nothing would
        re-enter a book it already holds (the entry flag), tell the agent it is day 1
        again, fire the same event twice in a day, and read the regime with a model fit
        on a later window than the one it entered on.
        """
        hmm = (None if self.hmm is None else
               {k: getattr(self.hmm, k).tolist() for k in ("mu", "sigma", "trans", "start")})
        book = (None if self.book is None else
                {"nav": [float(x) for x in self.book.nav],
                 "traded_notional": float(self.book.traded_notional),
                 "entered": bool(self.book.entered)})
        return {"day_no": self.day_no, "day": None if self._day is None else self._day.isoformat(),
                "fired": sorted(self._fired), "hmm": hmm, "book": book,
                "anon": None if self.anon is None else dict(self.anon.to_code),
                "universe_codes": None if self.ucodes is None else self.ucodes.state(),
                "recipe": self.recipe, "at_entry": self.at_entry, "rebalances": self.rebalances,
                "trims": self.trims, "start": None if self._start is None else self._start.isoformat(),
                "filings_to": None if self._filings_to is None else self._filings_to.isoformat(),
                "journal": self.journal.to_json(), "log": self.log}

    def restore(self, state: Optional[dict], tickers: list[str]) -> None:
        """Continue from `state()`; an empty state is a desk that has seen no round.

        The book's weights are not restored: every round recomputes them from the
        shares and cash it is handed, which is the book that actually exists. Nor is
        the journal trusted over it: the next round reconciles the two.
        """
        state = state or {}
        self.journal = Journal.from_json(state.get("journal"))
        self.day_no = int(state.get("day_no", 0))
        self._day = date.fromisoformat(state["day"]) if state.get("day") else None
        self._fired = set(state.get("fired", []))
        self.log = list(state.get("log", []))
        self.recipe, self.at_entry = state.get("recipe"), state.get("at_entry")
        self.rebalances = int(state.get("rebalances", 0))
        self.trims = list(state.get("trims", []))
        self._start = date.fromisoformat(state["start"]) if state.get("start") else None
        # A state from before the 8-K trigger has none: the desk then starts watching at
        # its next event round, rather than waking on every filing of the window so far.
        self._filings_to = pd.Timestamp(state["filings_to"]) if state.get("filings_to") else None
        h = state.get("hmm")
        self.hmm = None if h is None else quant.HMM2(**{k: np.asarray(v, dtype=float)
                                                        for k, v in h.items()})
        anon = state.get("anon")
        uc = state.get("universe_codes")
        if uc is not None:
            self.ucodes = observe.UniverseCodes.from_state(uc, tickers)
        elif not self.cfg.anonymize:
            self.ucodes = observe.UniverseCodes(None, tickers)
        elif anon is not None:
            # An anonymised state from before the universe block. Disabled codes here
            # would put real tickers into a replay; the window's own seed draws them.
            start = date.fromisoformat(state["start"]) if state.get("start") else None
            if start is None:
                raise ValueError("an anonymised desk state with no start day cannot code "
                                 "its universe names")
            self.ucodes = observe.UniverseCodes(self.cfg.seed * 1_000_003 + start.toordinal(), tickers)
        else:
            self.ucodes = None
        if anon is not None:
            self.anon = observe.Anonymizer.from_mapping(anon)
        elif not self.cfg.anonymize:
            self.anon = observe.Anonymizer(tickers, None)
        else:
            # Codes are drawn from the first day the desk sees, as in a fresh run.
            self.anon = self.book = None
            return
        b = state.get("book") or {}
        self.book = observe.BookState(pd.Series(0.0, index=tickers), list(b.get("nav", [])),
                                      float(b.get("traded_notional", 0.0)),
                                      bool(b.get("entered", False)))

    # ------------------------------------------------------------------ the round

    def __call__(self, ctx):
        self._t_round = time.perf_counter()
        tickers = ctx.market.tickers
        if self.anon is None:
            seed = (self.cfg.seed * 1_000_003 + ctx.day.toordinal()) if self.cfg.anonymize else None
            self.anon = observe.Anonymizer(tickers, seed)
            self.ucodes = observe.UniverseCodes(seed, tickers)
            self.book = observe.BookState(pd.Series(0.0, index=tickers), [])
        if ctx.day != self._day:
            self._day, self._fired = ctx.day, set()
            self.day_no += 1
            self._start = self._start or ctx.day
        n0 = len(self.log)
        self._journal_open(ctx)
        w = None
        try:
            w = self._decide(ctx, tickers)
        finally:
            self._journal_close(w, n0)
        return w

    def _decide(self, ctx, tickers):
        value, nav = self._value(ctx, tickers)
        if not np.isfinite(nav) or nav <= 0:
            self._note = "the book cannot be valued (a held name has no price); held"
            return None  # cannot value the book; holding beats guessing
        current = value / nav
        self.book.weights = current
        self._current = current
        if ctx.round == 1:
            self.book.nav.append(nav)
        if not self.book.entered:
            return self._enter(ctx, tickers, current, nav) if ctx.round == 1 else None
        if ctx.round == 1:
            return self._review(ctx, tickers, current, nav) if self.cfg.review else None
        return self._events(ctx, tickers, current, nav) if self.cfg.events else None

    @staticmethod
    def _tail(closes):
        return np.log(closes).diff().iloc[1:].tail(qs.SHAPE_DAYS)

    def _recipe_book(self, tail, raw_scores, recipe: dict, tickers, budget: float = 1.0,
                     base: Optional[pd.Series] = None) -> Optional[pd.Series]:
        """The book `recipe` makes of `tail` and the scores at gross `budget`, or None.

        One function for the entry and every rebalance, so a rebalance buys the book the
        Strategist chose, on today's inputs, rather than a lookalike of it.

        The names kept out are filled at `budget` itself, never at 1.0 and then scaled:
        the cap is on the book as bought. Filled at 1.0, three names would each hold the
        cap and the book land at 0.9 of its exposure (0.765 at 0.85), short of what was
        chosen with nothing saying so. None too when every name with a weight is kept
        out: there is no book to rebalance into, and cash is the exposure dial's to
        reach (set_exposure 0), not a rebalance's. With nothing kept out it is the shape
        times `budget`, the rule's own arithmetic.
        """
        base = qs.SHAPES[recipe["shape"]](tail) if base is None else base
        if base is None:
            return None
        w = base.reindex(tickers).fillna(0.0)
        if recipe["views"] != "none":
            w = signals.views_book(w, tail, raw_scores, recipe["views"])
            if w is None:
                return None
            w = w.reindex(tickers).fillna(0.0)
        out = list(recipe["excluded"])
        if not out:
            return w * budget
        w = w.copy()
        w[out] = 0.0
        if not (w > 0).any():
            return None
        return pd.Series(compiler._water_fill(w.to_numpy(), budget, W.CAP), index=tickers)

    def _exclude(self, tickers_out) -> None:
        """A name sold for a reason stays out: a later rebalance must not buy it back."""
        if self.recipe is not None:
            self.recipe["excluded"] = sorted(set(self.recipe["excluded"]) | set(tickers_out))

    @staticmethod
    def _snapshot(sig: dict) -> dict:
        """The entry day's score ranks and 3-day vols, JSON-safe, for every later round."""
        def clean(x):
            return None if x is None or not np.isfinite(x) else float(x)

        keep = ("model_score_rank", "vol_ann_har_3d")
        return {"names": {t: {k: clean(v) for k, v in f.items() if k in keep}
                          for t, f in sig["names"].items() if any(k in f for k in keep)},
                "market": {k: clean(v) for k, v in sig["market"].items() if k in keep}}

    def _enter(self, ctx, tickers, current, nav):
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        tail = self._tail(closes)
        shapes = {n: qs.SHAPES[n](tail) for n in ("inverse_vol", "risk_parity")}
        if any(s is None for s in shapes.values()):
            self._note = "not enough history for a shape; cash until there is"
            return None  # not enough history for a shape: cash until there is
        self.hmm = observe.fit_regime(closes)
        rd = observe.readings(closes, self.hmm)
        sig = self._signals(ctx, tickers)
        books = {lvl: self._recipe_book(tail, sig["raw_scores"],
                                        {"shape": "risk_parity", "views": lvl, "excluded": []},
                                        tickers, base=shapes["risk_parity"])
                 for lvl in ("light", "strong")}
        previews = {f"risk_parity_views_{lvl}": b for lvl, b in books.items() if b is not None}
        rule = {"shape": "risk_parity", "views": "none",
                "exposure": rule_exposure(rd.p_turbulent_next),
                "avoid": [], "rationale": "rule: risk parity at the regime-blended exposure"}
        payload = self._payload(closes, rd, ctx, "entry", sig=sig, previews=previews,
                                rule_proposal=rule)

        def check(d: EntryDecision):
            codes = [x.name for x in d.avoid]
            for code in codes:
                self.anon.ticker(code)
            if len(set(codes)) != len(codes):
                raise ValueError("a name excluded twice")
            if len(codes) >= len(tickers) - 3:
                raise ValueError("avoid list leaves fewer than 4 names")
            if d.views != "none" and d.shape != "risk_parity":
                raise ValueError("views tilt the risk_parity book only")
            if d.views != "none" and books[d.views] is None:
                raise ValueError(f"no {d.views} views book today: the model scores are missing")

        d, _ = self._ask("entry", payload, EntryDecision, check)
        avoid = [self.anon.ticker(x.name) for x in d.avoid]
        recipe = {"shape": d.shape, "views": d.views, "excluded": sorted(avoid)}
        # With views "none" and nothing avoided this is the rule's own path, untouched:
        # the shape as computed above, times the exposure.
        target = self._recipe_book(tail, sig["raw_scores"], recipe, tickers, budget=d.exposure,
                                   base=shapes[d.shape])
        if target is None:
            return None   # check() refuses a views level with no book, so never in practice
        self.recipe = recipe
        self.at_entry = self._snapshot(sig)
        self.book.entered = True
        # Every filing accepted by now was in the Strategist's observation; the analyst
        # wakes for the ones after it.
        self._filings_to = pd.Timestamp(ctx.deadline)
        return self._submit(target, current, nav, tickers)

    def _review(self, ctx, tickers, current, nav):
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        rd = observe.readings(closes, self.hmm)
        sig = self._signals(ctx, tickers)
        tail = self._tail(closes)
        gross = float(current.sum())
        left = self.cfg.max_rebalances - self.rebalances
        book = (self._recipe_book(tail, sig["raw_scores"], self.recipe, tickers, budget=gross)
                if self.recipe else None)
        previews = offer = None
        if book is not None and gross > compiler.HELD:
            moved = float((book - current).abs().sum())
            previews = {"rebalanced": book}
            offer = {"turnover": round(moved, 4),
                     "fee_bps_of_nav": round(moved * sim.FEE_RATE * 1e4, 2),
                     "rebalances_left": left,
                     "below_this_turnover_is_a_hold": self.cfg.rebalance_min_turnover}
        held_t = [t for t in tickers if current[t] > compiler.HELD]
        rule_cuts = self._rule_trims(ctx, current)
        rule = {"action": "hold", "exposure": None, "reason": None, "exit": [],
                "trim": [self._rule_trim_call(c) for c in rule_cuts],
                "rationale": "rule: hold" + (f", and book part of {len(rule_cuts)} winner(s) that "
                                             f"turned" if rule_cuts else "")}
        payload = self._payload(closes, rd, ctx, "review", sig=sig, previews=previews, held=held_t,
                                rebalance=offer, trim_lever=self._trim_lever(), rule_proposal=rule)
        held = {self.anon.code(t) for t in held_t}

        def without(codes):
            return dict(self.recipe, excluded=sorted(set(self.recipe["excluded"])
                                                     | {self.anon.ticker(c) for c in codes}))

        def check(d: ReviewDecision):
            if d.action == "set_exposure" and d.exposure is None:
                raise ValueError("set_exposure without an exposure")
            unknown = set(d.exit) - held
            if unknown:
                raise ValueError(f"exit names not held: {sorted(unknown)}")
            if d.action == "rebalance":
                if d.reason is None:
                    raise ValueError("rebalance without a reason")
                if offer is None:
                    raise ValueError("no rebalance on offer: the entry's book cannot be rebuilt today")
                if left <= 0:
                    raise ValueError(f"the window's {self.cfg.max_rebalances} rebalances are spent")
                if self._recipe_book(tail, sig["raw_scores"], without(d.exit), tickers) is None:
                    raise ValueError("these exits leave no name to rebalance into; to go to cash, "
                                     "set the exposure to 0")
                if d.trim:
                    raise ValueError("a rebalance rebuilds every name; a trim goes with hold or "
                                     "set_exposure")
            self._check_trims([(x.name, x.fraction) for x in d.trim], held, set(d.exit), current)

        d, _ = self._ask("review", payload, ReviewDecision, check)
        exits = [self.anon.ticker(code) for code in d.exit]
        if d.action == "rebalance":
            recipe = without(d.exit)
            target = self._recipe_book(tail, sig["raw_scores"], recipe, tickers,
                                       budget=d.exposure if d.exposure is not None else gross)
            if float((target - current).abs().sum()) < self.cfg.rebalance_min_turnover:
                return None  # drift, not a decision: no fee, and the budget is not spent
            self.rebalances += 1
            self.recipe = recipe
            return self._submit(target, current, nav, tickers)
        target = current.copy()
        if d.action == "set_exposure" and abs(d.exposure - gross) >= self.cfg.band and gross > 0:
            target = current * (d.exposure / gross)
        for t in exits:
            target[t] = 0.0
        cuts = self._cuts([(x.name, x.fraction) for x in d.trim], current)
        self._done(cuts)
        target = TR.apply(target, cuts)
        self._exclude(exits)
        if (target - current).abs().max() <= compiler.HELD:
            return None
        self._spend(cuts, ctx, "review")
        return self._submit(target, current, nav, tickers)

    def _events(self, ctx, tickers, current, nav):
        held_all = [t for t in tickers if current[t] > compiler.HELD]
        # Each filing wakes the analyst once: from the entry's deadline, or the last event
        # round's, to this one's. Overnight filings land in the day's first event round.
        after, self._filings_to = self._filings_to, pd.Timestamp(ctx.deadline)
        fresh = {}
        if self.filings is not None and after is not None and held_all:
            fresh = {t: g for t, g in F.new(self.filings, after, ctx.deadline).groupby("ticker")
                     if t in held_all}
        held = [t for t in held_all if t not in self._fired]
        if not held and not fresh:
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
        if held and len(bars) and len(closes) > 21 and bars.index[-1] > closes.index[-1]:
            rets = np.log(closes).diff().tail(20)
            z = np.log(bars.iloc[-1] / closes.iloc[-1]) / rets.std()
            for t in held:
                if np.isfinite(z[t]) and abs(z[t]) >= self.cfg.sigma_trigger and t not in why:
                    why[t] = f"moved {z[t]:+.1f} daily sigmas since yesterday's close"
        # A price or earnings trigger fires once a day per name; a filing fires once per
        # filing, so a second 8-K the same day is a second event, not a repeat.
        self._fired |= set(why)
        for t, g in fresh.items():
            what = ", ".join(dict.fromkeys(x for items in g["items"] for x in F.labels(items)))
            why[t] = (f"{why[t]}; " if t in why else "") + f"new 8-K: {what}"
        if not why:
            return None
        codes = [self.anon.code(t) for t in why]
        rd = observe.readings(closes, self.hmm)
        real = not self.anon.enabled
        triggers = [{"name": self.anon.code(t), "why": w,
                     **({"new_8k": observe.filing_rows(fresh[t], ctx.deadline, real)}
                        if t in fresh else {})} for t, w in why.items()]
        payload = self._payload(closes, rd, ctx, "event", held=held_all, triggered=list(why),
                                triggers=triggers, trim_lever=self._trim_lever(),
                                rule_proposal=rule_event(codes).model_dump())

        def check(d: EventDecision):
            extra = {c.name for c in d.calls} - set(codes)
            if extra:
                raise ValueError(f"calls for names with no trigger: {sorted(extra)}")
            names = [c.name for c in d.calls]
            if len(set(names)) != len(names):
                raise ValueError("a name called twice")
            for c in d.calls:
                if c.action == "trim" and (c.fraction is None or c.cause is None):
                    raise ValueError(f"{c.name}: a trim needs its fraction and its cause")
                if c.action != "trim" and (c.fraction is not None or c.cause is not None):
                    raise ValueError(f"{c.name}: a fraction or a cause belongs to a trim only")
            self._check_trims([(c.name, c.fraction) for c in d.calls if c.action == "trim"],
                              set(codes), set(), current)

        d, _ = self._ask("event", payload, EventDecision, check)
        # What woke the analyst, beside its answer: a replay counts calls by cause from it.
        self.log[-1]["triggers"] = {t["name"]: t["why"] for t in triggers}
        target = current.copy()
        exits = [self.anon.ticker(c.name) for c in d.calls if c.action == "exit"]
        for t in exits:
            target[t] = 0.0
        cuts = self._cuts([(c.name, c.fraction) for c in d.calls if c.action == "trim"], current)
        self._done(cuts)
        target = TR.apply(target, cuts)
        self._exclude(exits)
        if (target - current).abs().max() <= compiler.HELD:
            return None
        self._spend(cuts, ctx, "event")
        return self._submit(target, current, nav, tickers)

    # ------------------------------------------------------------------ the trim lever

    def _trim_lever(self) -> dict:
        return {"trims_left": max(self.cfg.max_trims - len(self.trims), 0),
                "fractions": dict(TR.FRACTIONS),
                "below_this_sale_of_nav_is_a_hold": TR.MIN_TRIM}

    def _cuts(self, calls, current: pd.Series) -> list[dict]:
        """The trims that trade: each named fraction of the position, where the sale is at
        least `trim.MIN_TRIM` of NAV. A smaller one is a hold: it would pay the fee and a
        turnover rank to move almost nothing, so it neither trades nor spends the budget."""
        out = []
        for code, frac in calls:
            t, f = self.anon.ticker(code), TR.FRACTIONS[frac]
            if f * float(current[t]) >= TR.MIN_TRIM:
                out.append({"ticker": t, "fraction": f})
        return out

    def _check_trims(self, calls, allowed: set, exits: set, current: pd.Series) -> None:
        names = [c for c, _ in calls]
        unknown = set(names) - allowed
        if unknown:
            raise ValueError(f"trims for names that are not held or not triggered: {sorted(unknown)}")
        if len(set(names)) != len(names):
            raise ValueError("a name trimmed twice in one answer")
        if set(names) & exits:
            raise ValueError(f"names both exited and trimmed: {sorted(set(names) & exits)}")
        n, left = len(self._cuts(calls, current)), self.cfg.max_trims - len(self.trims)
        if n > left:
            raise ValueError(f"{n} trims asked for, {max(left, 0)} left of the window's "
                             f"{self.cfg.max_trims}")

    def _done(self, cuts: list[dict]) -> None:
        """Which trims traded, beside the answer that asked for them: a trim under the
        floor is a hold, and the journal checks only the ones that sold."""
        self.log[-1]["trims_done"] = [self.anon.code(c["ticker"]) for c in cuts]

    def _spend(self, cuts: list[dict], ctx, role: str) -> None:
        self.trims += [{"ticker": c["ticker"], "day": self.day_no, "round": ctx.round,
                        "fraction": c["fraction"], "role": role} for c in cuts]

    def _rule_trims(self, ctx, current: pd.Series) -> list[dict]:
        """The rule's trims this morning: none unless it was given one (it has to win its
        gate first) and the journal it reads is whole this round."""
        if self.trim_rule is None or self.vol is None or self._journal_error is not None:
            return []
        return self.trim_rule.proposals(
            self.journal.positions, current, TR.sigma_today(self.vol, ctx), day_no=self.day_no,
            window_days=self.cfg.window_days, start=self._start,
            done={x["ticker"] for x in self.trims}, trims_left=self.cfg.max_trims - len(self.trims))

    def _rule_trim_call(self, c: dict) -> dict:
        frac = next(k for k, v in TR.FRACTIONS.items() if v == c["fraction"])
        return {"name": self.anon.code(c["ticker"]), "fraction": frac, "cause": "give_back",
                "why": (f"rule: up {c['gain']:.1%} since entry and {c['give_back']:.1%} off its "
                        f"high; expected give-back {c['egb']:.2%} over {c['sessions_left']} "
                        f"sessions beats the cost")}


def desk(brain_factory, config: Optional[DeskConfig] = None, **kw):
    """A `windows.run_field` factory: a fresh desk and brain per window."""
    return lambda: Desk(brain_factory(), config, **kw)
