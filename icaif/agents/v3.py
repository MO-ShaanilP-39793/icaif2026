"""Desk v3: a portfolio manager, a senior associate and an analyst (`v3_desk_plan.md`).

The score pays for one well-chosen book left alone: the 75% inverse-vol hold has been
the hardest entrant to beat, and every rule we tested that traded after entry lost. v3
spends its thinking on the entry, and makes every later trade climb a chain of
escalations to the one role that may trade.

**This module is stage 1: the entry, then a pure hold.** On day 1 the PM reads every
input the desk has (optionally after four analysts report), writes a free book, a thesis
and the plan's conditions, sees code's self-check of the draft (`selfcheck.report`), and
confirms or revises once. The book is then held to the window's end. The analyst and the
senior associate are stage 2.

What stage 1 calibrates is config, never a code path (`V3Config`):
- `analysts`: none, the reports beside every raw input, or the reports alone;
- `gross`: free, inside a band, or a sleeve code fixes with the PM choosing the names;
- `streams`: which inputs the observation carries (`strip`), so a leave-one-out run
  measures what each is worth;
- `evidence`: whether the PM's prompt carries our backtest findings;
- `self_check`: whether the PM sees code's report and may revise.

**What code owns, as in v1 and v2.** The observation is point in time (`Desk._payload`).
An entry is refused whole, with every reason, never clipped: the 30 names only, each at
most 30% of NAV as bought, the gross within the rule, weights of at most 6 decimals, and
conditions code can evaluate. A refused or failed entry buys the inverse-vol book at the
fallback gross (75%, or the sleeve's), which is what we would submit without an LLM, so
a failing model costs nothing against the reference. Every fallback is counted.

The stage-1 windows (`STAGE1`) are fixed here rather than in `suites`: a suite there is
built into the scorer Space and the public board, with prices shipped for its windows,
and these are a research split, not a board.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from icaif import baselines, compiler, quant_strategies as qs
from icaif import weights as W
from icaif.agents import observe, prompts_v3, selfcheck
from icaif.agents.brains import BrainError
from icaif.agents.schemas import (BOOK_ACTIONS, CONDITION_KINDS, NAME_ACTIONS, AnalystReport,
                                  PMCheck, PMEntry)
from icaif.agents.tradelist import NUDGE, on_grid
from icaif.agents.v2 import V2Config, V2Desk, _shown

# The inputs an ablation can take away, and where each lives in the observation. Earnings
# timing, returns, 20-day vol, the OU score and the two shapes' weights always stay: they
# are the bare minimum a book is chosen from.
STREAMS = ("regime", "headlines", "filings", "macro", "model_rank", "universe", "har_vol")
_HAR = ("vol_ann_har_1d", "vol_ann_har_3d")
NAME_FIELDS = {"headlines": ("headlines",), "filings": ("recent_8k_filings",),
               "model_rank": ("model_score_rank",), "har_vol": _HAR}
MARKET_FIELDS = {"regime": observe.REGIME_FIELDS, "har_vol": _HAR}
TOP_FIELDS = {"macro": ("macro",), "universe": ("universe_context",), "filings": ("new_filings",)}

ANALYSTS = ("market", "earnings", "news", "quant")
ROLE_TIER = {"market": "quick", "earnings": "quick", "news": "quick", "quant": "quick",
             "pm": "deep", "pm_check": "deep"}
# Seconds from the chain's start at which each stage's slot ends. An entry call took 38 s
# in the free desk on Gemini 2.5 Pro (high), and v3's carries more; the entry's inputs are
# as of the prior close plus overnight news, so a live entry can start well before round 1.
SLOTS = {"analysts": 180, "pm": 600, "pm_check": 1020}
FALLBACK_GROSS = 0.75
BASIC_MARKET = ("basket_ret_1d", "basket_ret_5d", "basket_ret_20d", "basket_vol_ann_ewma",
                "vol_vs_3y_median", "avg_pairwise_corr_60d")
BASIC_NAME = ("name", "vol_ann_20d", "ret_1d", "ret_5d", "ret_20d", "weight_if_inverse_vol",
              "weight_if_risk_parity", "earnings_in_sessions")
SLEEVE_SUM_TOL = 0.01

# Non-overlapping 15-session windows after Gemini 2.5's January 2025 knowledge cutoff,
# tiled from the first session of February 2025 and stepping past the four official4
# windows (stage 2's), then split by date: selection (2025) chooses the arm, and
# confirmation (2026) scores the committed choice once. Picking the best of a dozen arms
# on the windows that also judge it finds the luckiest arm, and its gate would pass on
# luck while reading as evidence.
STAGE1 = {
    "select": (("2025-02-03", "2025-02-24"), ("2025-02-25", "2025-03-17"),
               ("2025-03-18", "2025-04-07"), ("2025-05-05", "2025-05-23"),
               ("2025-05-27", "2025-06-16"), ("2025-06-17", "2025-07-09"),
               ("2025-07-10", "2025-07-30"), ("2025-07-31", "2025-08-20"),
               ("2025-08-21", "2025-09-11"), ("2025-09-12", "2025-10-02"),
               ("2025-11-03", "2025-11-21"), ("2025-11-24", "2025-12-15"),
               ("2025-12-16", "2026-01-07")),
    "confirm": (("2026-01-08", "2026-01-29"), ("2026-01-30", "2026-02-20"),
                ("2026-02-23", "2026-03-13"), ("2026-03-16", "2026-04-06"),
                ("2026-05-04", "2026-05-22"), ("2026-05-26", "2026-06-15"),
                ("2026-06-16", "2026-07-08"), ("2026-08-03", "2026-08-21"),
                ("2026-08-24", "2026-09-14")),
}


def parse_gross(text: str) -> tuple:
    """"free", "band:LO:HI" or "fixed:G" as the config's tuple."""
    parts = text.split(":")
    if parts == ["free"]:
        return ("free",)
    if parts[0] == "band" and len(parts) == 3:
        return ("band", float(parts[1]), float(parts[2]))
    if parts[0] == "fixed" and len(parts) == 2:
        return ("fixed", float(parts[1]))
    raise ValueError(f"gross must be free, band:LO:HI or fixed:G, not {text!r}")


def strip(obs: dict, streams) -> dict:
    """The observation without the inputs of every stream not in `streams`.

    A dropped stream must be absent everywhere a role could read it, including the
    `_at_entry` copies: a leave-one-out run that still carried it somewhere would
    measure nothing, and read as "this stream doesn't matter".
    """
    drop = [s for s in STREAMS if s not in set(streams)]
    out = copy.deepcopy(obs)
    for s in drop:
        for k in TOP_FIELDS.get(s, ()):
            out.pop(k, None)
        for k in MARKET_FIELDS.get(s, ()):
            for key in (k, f"{k}_at_entry"):
                out.get("market", {}).pop(key, None)
        for row in out.get("names", []):
            for k in NAME_FIELDS.get(s, ()):
                for key in (k, f"{k}_at_entry"):
                    row.pop(key, None)
    return out


@dataclass
class V3Config(V2Config):
    slots: dict = field(default_factory=lambda: dict(SLOTS))
    analysts: str = "none"           # "none" | "reports_raw" | "reports_only"
    gross: tuple = ("free",)         # ("free",) | ("band", lo, hi) | ("fixed", g)
    streams: tuple = STREAMS
    self_check: bool = True
    # The observation carries the regime model's read; `streams` decides whether a role
    # sees it. v2 hid it outright (the free desk trusted "calm" through a 3% fall); v3
    # measures it instead.
    regime: bool = True
    triggers: bool = False
    reflect: bool = False
    headline_roles: tuple = ("v3",)
    universe_roles: tuple = ("v3",)

    def __post_init__(self):
        if self.analysts not in ("none", "reports_raw", "reports_only"):
            raise ValueError(f"analysts must be none, reports_raw or reports_only, not {self.analysts!r}")
        unknown = sorted(set(self.streams) - set(STREAMS))
        if unknown:
            raise ValueError(f"unknown streams {unknown}; the streams are {', '.join(STREAMS)}")
        g = tuple(self.gross)
        ok = (g == ("free",) or (len(g) == 3 and g[0] == "band" and 0 <= g[1] <= g[2] <= 1)
              or (len(g) == 2 and g[0] == "fixed" and 0 < g[1] <= 1))
        if not ok:
            raise ValueError(f"gross must be ('free',), ('band', lo, hi) or ('fixed', g), not {g!r}")
        self.gross = g

    @property
    def fallback_gross(self) -> float:
        """A sleeve arm falls back to inverse-vol at its own gross: at 75% a fallback would
        carry the cash question into an arm built to hold it fixed."""
        return self.gross[1] if self.gross[0] == "fixed" else FALLBACK_GROSS


class V3Desk(V2Desk):
    PREFIX = "v3"

    def __init__(self, brains: dict, config: Optional[V3Config] = None, earnings_history=None, **kw):
        cfg = config or V3Config()
        if not isinstance(cfg, V3Config):
            raise TypeError("V3Desk takes a V3Config")
        super().__init__(brains, cfg, earnings_history=earnings_history, **kw)
        # The PM's plan (thesis and conditions) and the whole entry's record: the draft,
        # code's report, the check and what was traded. Stage 2 reads the plan; a replay
        # reads the record.
        self.plan: Optional[dict] = None
        self.entry: Optional[dict] = None

    def _tier(self, role: str) -> str:
        return ROLE_TIER[role]

    def _system(self, role: str) -> str:
        return (prompts_v3.SYSTEM_EVIDENCE if self.cfg.evidence else prompts_v3.SYSTEM)[role]

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        out = super().state()
        out["v3"] = {"plan": self.plan, "entry": self.entry}
        return out

    def restore(self, state, tickers):
        super().restore(state, tickers)
        v3 = (state or {}).get("v3") or {}
        self.plan, self.entry = v3.get("plan"), v3.get("entry")

    # ------------------------------------------------------------------ the entry

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
        if self.book.entered:
            return None
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        tail = self._tail(closes)
        if any(qs.SHAPES[n](tail) is None for n in ("inverse_vol", "risk_parity")):
            self._note = "not enough history for a shape; cash until there is"
            return None
        self._t_chain = self._t_round
        self.hmm = observe.fit_regime(closes)
        obs = self._payload(closes, observe.readings(closes, self.hmm), ctx, "v3", held=list(tickers))
        new = self._new_filings(ctx)
        if new:
            obs["new_filings"] = new
        obs = strip(obs, self.cfg.streams)
        obs["gross_rule"] = self._gross_rule()
        reports = self._entry_analysts(ctx, obs) if self.cfg.analysts != "none" else None
        return self._enter_book(ctx, tickers, current, nav, closes, self._pm_view(obs, reports))

    def _gross_rule(self) -> dict:
        g = self.cfg.gross
        if g[0] == "band":
            return {"rule": "band", "range": [g[1], g[2]]}
        if g[0] == "fixed":
            return {"rule": "sleeve", "gross": g[1]}
        return {"rule": "free", "range": [0.0, 1.0]}

    def _pm_view(self, obs: dict, reports: Optional[dict]) -> dict:
        if reports is None:
            return obs
        if self.cfg.analysts == "reports_raw":
            return dict(obs, reports=_shown(reports))
        return {"clock": obs["clock"], "book": obs["book"], "gross_rule": obs["gross_rule"],
                "market": {k: obs["market"][k] for k in BASIC_MARKET if k in obs["market"]},
                "names": [{k: r[k] for k in BASIC_NAME if k in r} for r in obs["names"]],
                "reports": _shown(reports)}

    def _entry_analysts(self, ctx, obs: dict) -> dict:
        """The four reports, each from its own slice of the (already stripped) observation:
        a report reading another stream could carry that stream into an ablation of it."""
        rows = obs["names"]
        base = {"clock": obs["clock"], "book": obs["book"], "gross_rule": obs["gross_rule"]}
        hist = self.tags.history
        reporting = []
        for r in rows:
            if r.get("earnings_in_sessions") is None:
                continue
            t = self.anon.ticker(r["name"])
            reporting.append(dict({k: r[k] for k in BASIC_NAME if k in r},
                                  past_earnings_reactions=(hist.summary(t, pd.Timestamp(ctx.deadline))
                                                           if hist is not None else None),
                                  **({"recent_8k_filings": r["recent_8k_filings"]}
                                     if "recent_8k_filings" in r else {})))
        news = [{"name": r["name"], "ret_1d": r.get("ret_1d"),
                 **{k: r[k] for k in ("headlines", "recent_8k_filings") if r.get(k)}} for r in rows]
        news = [x for x in news if len(x) > 2]
        quant = [{k: v for k, v in r.items() if k not in ("headlines", "recent_8k_filings")} for r in rows]
        views = {
            "market": dict(base, market=obs["market"], **({"macro": obs["macro"]} if "macro" in obs else {})),
            "earnings": dict(base, reporting=reporting) if reporting else None,
            "news": (dict(base, names=news, **({"new_filings": obs["new_filings"]} if "new_filings" in obs else {}))
                     if news or obs.get("new_filings") else None),
            "quant": dict(base, names=quant,
                          market={k: v for k, v in obs["market"].items() if "vol" in k},
                          **({"universe_context": obs["universe_context"]} if "universe_context" in obs else {})),
        }
        names_ok = lambda d: self._names_ok([x.name for x in d.names])  # noqa: E731
        asks = [(k, views[k], AnalystReport, names_ok) for k in ANALYSTS if views[k] is not None]
        out = self._ask_many(ctx, asks, "analysts")
        if views["earnings"] is None:
            out["earnings"] = "no name reports within the calendar's reach"
        if views["news"] is None:
            out["news"] = "no headlines or filings in this run"
        return {k: out[k] for k in ANALYSTS}

    def _judge(self, entry: PMEntry, tickers) -> tuple[list[str], Optional[pd.Series]]:
        """(every reason the entry is refused, the book it buys): the book is None unless
        the reasons are none. Refused whole, never repaired: a book with its bad line
        dropped, or rescaled under a cap, is one the PM never chose, and its thesis would
        describe a book that does not exist."""
        errors: list[str] = []
        codes = [x.name for x in entry.weights]
        twice = sorted({c for c in codes if codes.count(c) > 1})
        if twice:
            errors.append(f"names listed twice: {twice}")
        for c in dict.fromkeys(codes):
            try:
                self._names_ok([c])
            except ValueError as err:
                errors.append(str(err))
        if errors:
            return errors, None
        w = pd.Series(0.0, index=tickers)
        for x in entry.weights:
            w[self.anon.ticker(x.name)] = x.weight
        total, g = float(w.sum()), self.cfg.gross
        if g[0] == "fixed":
            if abs(total - 1.0) > SLEEVE_SUM_TOL:
                errors.append(f"sleeve shares sum to {total:.4f}; they must sum to 1 "
                              f"(within {SLEEVE_SUM_TOL})")
            else:
                w = w * (g[1] / total)
        else:
            off = [x.name for x in entry.weights if not on_grid(x.weight)]
            if off:
                errors.append(f"weights with more than 6 decimals: {off}")
            if total > 1.0 + 1e-9:
                errors.append(f"gross {total:.4f} is over 1")
            elif g[0] == "band" and not (g[1] - 1e-9 <= total <= g[2] + 1e-9):
                errors.append(f"gross {total:.4f} is outside the band {g[1]:g}-{g[2]:g}")
        over = [f"{self.anon.code(t)} {v:.4f}" for t, v in w.items() if v > W.CAP + 1e-12]
        if over:
            errors.append("over the 30% cap as bought: " + ", ".join(over))
        held = {c for c, v in zip(codes, (x.weight for x in entry.weights)) if v > 0}
        for i, c in enumerate(entry.conditions):
            errors += [f"condition {i + 1} ({c.kind}): {e}" for e in condition_errors(c, held)]
        return errors, (None if errors else w)

    def _enter_book(self, ctx, tickers, current, nav, closes, view: dict):
        earnings = self.earnings.to_next(ctx.day) if self.earnings is not None else None
        report = lambda b: selfcheck.report(b, closes, earnings, self.anon.code)  # noqa: E731
        draft = self._ask_one(ctx, "pm", view, PMEntry)
        rec = {"draft": draft.model_dump() if draft is not None else None,
               "draft_source": self.chain[-1]["source"], "draft_reason": self.chain[-1]["reason"]}
        final, how, book = None, None, None
        if draft is not None:
            rec["draft_errors"], book = self._judge(draft, tickers)
            if book is not None:
                final, how = draft, "draft"
        if self.cfg.self_check and draft is not None:
            shown = report(book) if book is not None else {"refused": rec["draft_errors"]}
            rec["self_check"] = shown

            def consistent(c: PMCheck):
                if (c.action == "revise") != (c.revised is not None):
                    raise ValueError("a revise needs a revised entry, and a confirm has none")

            chk = self._ask_one(ctx, "pm_check", dict(view, draft=draft.model_dump(), self_check=shown),
                                PMCheck, consistent)
            rec["check"] = chk.model_dump() if chk is not None else None
            rec["check_source"] = self.chain[-1]["source"]
            if chk is not None and chk.action == "revise":
                rec["revised_errors"], revised = self._judge(chk.revised, tickers)
                if revised is not None:
                    final, how, book = chk.revised, "revised", revised
                    rec["revised_self_check"] = report(revised)   # logged; no third look
                else:
                    self.fallbacks["pm_check:refused"] += 1
        reason, w = None, None
        if final is None:
            reason = ("pm: no usable entry"
                      + (f" ({rec['draft_source']}: {rec['draft_reason']})" if draft is None
                         else f" (refused: {'; '.join(rec['draft_errors'])[:300]})"))
            fb = baselines.scaled(baselines.InverseVolHold, self.cfg.fallback_gross)()(ctx)
            if fb is not None:
                self.fallbacks["entry:hold_book"] += 1
                w = self._take(fb, float(sum(fb.values())), current, nav)
                reason += f"; bought the {self.cfg.fallback_gross:.0%} inverse-vol book"
        else:
            self.plan = {"thesis": final.thesis, "from": how,
                         "conditions": [c.model_dump() for c in final.conditions]}
            if float(book.sum()) <= compiler.HELD:
                self.book.entered = True     # all cash, chosen: entered, nothing to buy
            else:
                weights = W.safe({t: v + NUDGE if v > 0 else 0.0 for t, v in book.items()}, tickers)
                w = self._take(weights, float(sum(weights.values())), current, nav)
        bought = w if w is not None else {}
        rec.update(source="brain" if reason is None else "fallback", reason=reason, final=how,
                   gross=round(float(sum(bought.values())), 6),
                   weights={self.anon.code(t): v for t, v in bought.items() if v > 0})
        self.entry = rec
        self.log.append({"day": self.day_no, "round": ctx.round, "role": "pm",
                         "brain": getattr(self.brains["deep"], "name", "?"),
                         "source": rec["source"], "reason": reason,
                         "decision": {"weights": [{"name": c, "weight": v} for c, v in rec["weights"].items()],
                                      "rationale": final.thesis if final is not None else "",
                                      "conditions": len(final.conditions) if final is not None else 0,
                                      "from": how},
                         "traded": w is not None})
        return w


def condition_errors(c, held: set) -> list[str]:
    """Why code could not evaluate condition `c` against a book holding `held` (codes)."""
    errs = []
    scope = CONDITION_KINDS[c.kind]
    if scope == "name":
        if c.name is None:
            errs.append("names no name")
        elif c.name not in held:
            errs.append(f"{c.name} is not in the book")
        if c.action not in NAME_ACTIONS:
            errs.append(f"a name's condition acts by {', '.join(NAME_ACTIONS)}, not {c.action}")
    else:
        if c.name is not None:
            errs.append(f"a {scope} condition names no name")
        if c.action not in BOOK_ACTIONS:
            errs.append(f"a {scope} condition acts by {', '.join(BOOK_ACTIONS)}, not {c.action}")
    if c.kind == "new_8k_item":
        if not c.item:
            errs.append("names no 8-K item")
    elif c.threshold is None or not np.isfinite(c.threshold):
        errs.append("has no threshold")
    if (c.action == "set_gross") != (c.gross is not None):
        errs.append("set_gross needs a gross, and only set_gross takes one")
    return errs


class HoldBrain:
    """Answers every v3 role in code: the analysts find nothing and the PM gives no entry,
    so the desk's fallback buys the inverse-vol book at the fallback gross. A v3 desk on
    this brain must trade exactly as that hold; any difference is the desk's plumbing."""

    name = "v3_hold"

    def decide(self, role, system, payload, schema, timeout):
        if schema is AnalystReport:
            return AnalystReport(summary="code: nothing to report", names=[])
        raise BrainError("the hold brain makes no entry; the desk's fallback does")


class BoughtBook:
    """A strategy that buys one fixed book at its first round, scaled to `gross`, and holds.

    Stage 1's rescale: a hold trades once, so the PM's book bought at another gross is
    exactly what that gross would have done, and "good names, wrong cash" can be told
    from "wrong names" without another call. `weights`: {ticker: weight of NAV}.
    """

    def __init__(self, weights: dict, gross: Optional[float] = None):
        total = float(sum(weights.values()))
        scale = 1.0 if gross is None or total <= 0 else gross / total
        self.weights = {t: v * scale for t, v in weights.items()}
        over = sorted(t for t, v in self.weights.items() if v > W.CAP + 1e-9)
        if over:
            # Capped by `weights.safe`, the book would land below `gross` with nothing
            # saying so, and score as a gross nobody chose.
            raise ValueError(f"at gross {gross:g} the book holds {over} over the 30% cap")
        self.done = False

    def __call__(self, ctx):
        if self.done:
            return None
        self.done = True
        if sum(self.weights.values()) <= compiler.HELD:
            return None
        # The nudge keeps a weight already on the grid from flooring a step lower
        # (`tradelist.NUDGE`), so at its own gross this buys exactly what the desk did.
        return W.safe({t: v + NUDGE if v > 0 else 0.0 for t, v in self.weights.items()},
                      ctx.market.tickers)


# ----------------------------------------------------------------------------- the experiment

# Stage 1's cash floors (`stage1_doe.md`, round 3): the least cash the PM must hold, so its
# gross may go up to 1 - floor. A floor is a limit, not a target.
CASH_FLOORS = (0.0, 0.20, 0.40, 0.50, 0.75, 0.90, 0.95)

# Round 2's streams: a 2^(7-3) fractional factorial of resolution IV. The factors A-G are
# the streams in this order; D-G's columns are generated (E = ABC, F = BCD, G = ACD), so
# every stream is in for 8 of the 16 runs and out for 8, and no stream's effect is
# confounded with a pair of others. Hand-typed rows could break that silently, and a
# stream's "effect" would then be partly another stream's or a pair's: the table is built
# here and a test checks both properties.
FACTORIAL_ORDER = ("har_vol", "model_rank", "headlines", "filings", "macro", "universe", "regime")


def _streams16() -> tuple:
    import itertools

    rows = []
    for a, b, c, d in itertools.product((-1, 1), repeat=4):
        signs = (a, b, c, d, a * b * c, b * c * d, a * c * d)
        rows.append(tuple(s for s, v in zip(FACTORIAL_ORDER, signs) if v > 0))
    return tuple(rows)


STREAMS16 = _streams16()   # STREAMS16[n - 1]: the streams run n includes, n = 1..16


def floor_gross(floor: float) -> float:
    """The most a cash floor lets the book hold."""
    return round(1.0 - floor, 6)


def capped_book(weights: dict, cap: float) -> dict:
    """The book scaled down to gross `cap` when it holds more; as it is otherwise.

    Round 3 scores each cash floor from the no-floor run's books, assuming the PM would
    have picked the same names under the floor (two real-floor arms check that). Scaling
    down never breaches the 30% cap.
    """
    total = float(sum(weights.values()))
    if total <= cap + 1e-12:
        return dict(weights)
    return {t: v * cap / total for t, v in weights.items()}
