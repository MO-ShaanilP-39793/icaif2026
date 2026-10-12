"""The plan's conditions, evaluated by code each round (desk v3, stage 2).

At entry the PM states when it would act, in a closed vocabulary (`schemas.Condition`).
Here code reads each one against the book and the market at a round's deadline, says
how far it is from firing, and latches it when it fires. The analyst and the associate
read these distances instead of re-arguing the entry, and a fired condition goes to the
PM whatever either of them writes (`v3_desk_plan.md`, "fired conditions bind").

**Point in time.** Prices are the bars ended by the deadline (`Market.recent_closes`,
`qs.daily_closes`), filings are those accepted by it, and an earnings gap counts only on
a reaction session whose release was public by it. Each test that rewrites the future
requires every reading of the past unchanged.

**A missing price is a hole, never a stand-in.** A name whose latest bar has no price
reads None and cannot fire: a forward-filled price is a zero move that never happened,
and a condition watching for a fall would read "no fall" through the outage.

**Fixed at entry** (`entry_marks`): each name's daily sigma, the basket's prices and the
book's scale. A sigma read later would move the goalposts as vol rises (a 2.5-sigma fall
measured in the sigma of a falling market fires later than the one the PM meant), and
live a later round cannot read the entry day's HAR forecast at all
(`VolForecasts.from_live` serves one session).

**The PM's scale.** Under the overlay the PM's book is bought at code's invested share
and the PM is never told (`V3Config.overlay`). Its book conditions ("down 3% from the
peak") were stated for the book it chose, so the book's NAV is read at the PM's scale:
cash returns nothing, so the PM's book moves `scale` times the real one, exactly. Read
at the real scale, a 3% drawdown condition on a 10%-invested book would need a 30% fall
in the names, and would never fire. Name and market conditions are scale-free.

**Latched.** A condition fires once and stays fired until the PM rewrites the plan
(`Watch.rewrite`). Re-armed on recovery, a condition sitting on its threshold would
reach the PM every round it crossed back, and a decline would mean nothing.
"""

import copy
from typing import Callable, Optional

import numpy as np
import pandas as pd

from icaif import quant_strategies as qs
from icaif.agents import observe
from icaif.agents.schemas import CONDITION_KINDS
from icaif.agents.triggers import SIGMA_DAYS, move_since_close, priced_from

# Bars read back from a deadline for a name's peak since entry: a 15-session window holds
# about 105 hourly bars, and the journal's own marks read the same depth.
LOOKBACK_BARS = 400
SIGNED = ("move_from_entry_sigma", "move_from_entry_pct", "earnings_gap_pct",
          "basket_move_from_entry_pct")
# A give-back is a share of a gain; a "gain" smaller than one daily sigma is noise, and
# half of it given back is a quiet name drifting, not a winner turning.
GIVE_BACK_MIN_SIGMAS = 1.0


def _f(x) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else float(x)


def _r(x, nd: int = 4) -> Optional[float]:
    x = _f(x)
    return None if x is None else round(x, nd)


def entry_marks(ctx, tickers, vol=None, scale: float = 1.0) -> dict:
    """What the conditions are measured from, read at the entry's deadline.

    `sigma`: each name's daily sigma, the HAR one-day forecast where `vol` has one, else
    the standard deviation of its last `SIGMA_DAYS` daily log returns (the v1 trigger's
    unit), with the source named per name. `basket`: each name's last price by the
    deadline. `scale`: the PM's gross over the gross bought (1 without an overlay).
    """
    sigma, source = {}, {}
    har = None
    if vol is not None:
        try:
            har = vol.for_day(ctx.day, ctx.deadline)["har_h1"]
        except KeyError:
            har = None
    daily = qs.daily_closes(ctx, SIGMA_DAYS + 1)
    for t in tickers:
        v = _f(har.get(t)) if har is not None else None
        if v is not None and v > 0:
            sigma[t], source[t] = float(np.sqrt(v)), "har_1d"
            continue
        sd = (_f(np.log(daily[t]).diff().std()) if t in daily.columns and len(daily) > SIGMA_DAYS
              and daily[t].notna().all() else None)
        sigma[t], source[t] = (sd, "realised_20d") if sd and sd > 0 else (None, "none")
    bars = ctx.recent_closes(1)
    basket = {t: _f(bars.iloc[-1][t]) for t in tickers} if len(bars) else {}
    return {"at": pd.Timestamp(ctx.deadline).isoformat(), "sigma": sigma, "sigma_source": source,
            "basket": basket, "scale": float(scale)}


class Watch:
    """The plan's conditions and what has become of them; JSON in `state`.

    Each condition gets an id that a rewrite never reuses (c1, c2, ...), so a latch, an
    escalation and a settlement name the condition they meant even after the plan
    changes under them.
    """

    def __init__(self, conditions: list[dict], marks: dict):
        self.marks = marks
        self.next_id = 1
        self.conditions: list[dict] = []
        self.fired: dict[str, dict] = {}        # id -> the reading that fired it
        self.peak_nav: Optional[float] = None   # the book's peak at the PM's scale, round marks
        self.base_nav: Optional[float] = None
        self.start_nav: Optional[float] = None
        self._add(conditions)

    def _add(self, conditions: list[dict]) -> None:
        for c in conditions:
            self.conditions.append(dict(copy.deepcopy(c), id=f"c{self.next_id}"))
            self.next_id += 1

    def rewrite(self, keep: list[str], add: list[dict]) -> None:
        """The PM's new plan: the ids it keeps (latches and all) and the conditions it adds.

        A dropped condition goes with its latch; an added one is new, unfired, and
        measured from the same entry marks.
        """
        unknown = sorted(set(keep) - {c["id"] for c in self.conditions})
        if unknown:
            raise ValueError(f"keeps conditions that are not in the plan: {unknown}")
        self.conditions = [c for c in self.conditions if c["id"] in set(keep)]
        self.fired = {k: v for k, v in self.fired.items() if k in set(keep)}
        self._add(add)

    def state(self) -> dict:
        return copy.deepcopy({"marks": self.marks, "next_id": self.next_id,
                              "conditions": self.conditions, "fired": self.fired,
                              "peak_nav": self.peak_nav, "base_nav": self.base_nav,
                              "start_nav": self.start_nav})

    @classmethod
    def from_state(cls, s: dict) -> "Watch":
        w = cls([], s["marks"])
        s = copy.deepcopy(s)
        w.next_id, w.conditions, w.fired = s["next_id"], s["conditions"], s["fired"]
        w.peak_nav, w.base_nav, w.start_nav = s.get("peak_nav"), s.get("base_nav"), s.get("start_nav")
        return w

    # ------------------------------------------------------------------ a round

    def check(self, ctx, *, to_ticker: Callable[[str], str], positions: dict,
              nav: Optional[float], base_nav: Optional[float], history=None,
              filings: Optional[pd.DataFrame] = None, start_nav: Optional[float] = None) -> list[dict]:
        """Every condition read at `ctx`'s deadline, newly fired ones latched.

        `positions`: {ticker: {"entry_price", "entry_at"}} for the names held (the
        journal's); `nav`: the book's NAV marked at the deadline; `base_nav`: its NAV
        just after the entry filled; `start_nav`: its NAV before the entry (cash), the
        anchor of the PM's scale, so the entry's fee scales with the book (defaults to
        `base_nav`). `history`: a `triggers.EarningsHistory`;
        `filings`: (ticker, accepted, items) 8-K events.

        Each row: the condition's id, kind, name and threshold; `value` in the
        threshold's units (None when it cannot be read); `to_fire`, how far the value
        must still move to fire (<= 0 once it has), in the same units; `fired` and
        `new` (fired at this round); `note` when the value is unreadable or the
        condition is inactive.
        """
        deadline = pd.Timestamp(ctx.deadline)
        if base_nav is not None and self.base_nav is None:
            self.base_nav = float(base_nav)
            self.start_nav = float(start_nav) if start_nav is not None else self.base_nav
        pm_nav = self._pm_nav(nav)
        if pm_nav is not None:
            self.peak_nav = pm_nav if self.peak_nav is None else max(self.peak_nav, pm_nav)
        bars = ctx.market.recent_closes(deadline, LOOKBACK_BARS)
        out = []
        for c in self.conditions:
            row = {"id": c["id"], "kind": c["kind"], "name": c.get("name"),
                   "threshold": c.get("threshold"), "item": c.get("item"), "action": c["action"]}
            value, note = self._value(c, ctx, bars, to_ticker, positions, pm_nav, history, filings)
            bad = _unreadable(c)
            row["value"], row["note"] = value, bad or note
            row["to_fire"], hit = (None, False) if bad else _distance(c, value)
            if c["id"] in self.fired:
                row.update(fired=True, new=False, fired_at=self.fired[c["id"]]["at"])
            elif hit:
                self.fired[c["id"]] = {"at": deadline.isoformat(), "value": value}
                row.update(fired=True, new=True, fired_at=deadline.isoformat())
            else:
                row.update(fired=False, new=False)
            out.append(row)
        return out

    def _pm_nav(self, nav: Optional[float]) -> Optional[float]:
        """The book's NAV as the PM's book would stand: the PM holds `scale` times every
        share, so its P&L since the start (the entry's fee included) is `scale` times the
        real one."""
        if nav is None or self.base_nav is None or not np.isfinite(nav):
            return None
        a = self.start_nav if self.start_nav is not None else self.base_nav
        return a + self.marks.get("scale", 1.0) * (nav - a)

    def _value(self, c, ctx, bars, to_ticker, positions, pm_nav, history, filings) -> tuple:
        kind, scope = c["kind"], CONDITION_KINDS[c["kind"]]
        if scope == "name":
            t = to_ticker(c["name"])
            pos = positions.get(t)
            if pos is None:
                return None, "inactive: the name is no longer held"
            if kind == "new_8k_item":
                return _new_8k(filings, t, c["item"], self.marks["at"], ctx.deadline)
            if kind == "earnings_gap_pct":
                return _gap(ctx, t, history, self.marks["at"])
            last, peak = _since(bars, t, pos)
            ep = _f(pos.get("entry_price"))
            if last is None or not ep:
                return None, "no price at the deadline"
            move = last / ep - 1.0
            if kind == "move_from_entry_pct":
                return _r(100 * move, 3), None
            sd = _f((self.marks.get("sigma") or {}).get(t))
            if kind == "move_from_entry_sigma":
                if not sd:
                    return None, "no sigma at entry"
                return _r(np.log(last / ep) / sd, 3), None
            # give_back_from_peak: the share of the peak gain since entry given back
            gain = peak / ep - 1.0
            if not sd or gain < GIVE_BACK_MIN_SIGMAS * sd:
                return None, (f"not armed: the peak gain since entry ({100 * gain:.2f}%) is "
                              f"under {GIVE_BACK_MIN_SIGMAS:g} daily sigma")
            return _r((peak - last) / (peak - ep), 3), None
        if kind in ("drawdown_from_peak_pct", "drawdown_from_entry_pct"):
            if pm_nav is None:
                return None, "the book has no NAV at the deadline"
            ref = self.peak_nav if kind == "drawdown_from_peak_pct" else self._pm_nav(self.base_nav)
            return _r(100 * (1.0 - pm_nav / ref), 3), None
        if kind == "basket_move_from_entry_pct":
            if not len(bars):
                return None, "no price at the deadline"
            last = bars.iloc[-1]
            ref = self.marks.get("basket") or {}
            moves = [float(last[t]) / ref[t] - 1.0 for t in ref
                     if ref[t] and t in last.index and np.isfinite(last[t])]
            if len(moves) < len(ref) or not ref:
                return None, "a basket name has no price at the deadline"
            return _r(100 * float(np.mean(moves)), 3), None
        # vol_ratio_above
        closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
        ratio = _f(observe.readings(closes, None).vol_ratio)
        return (_r(ratio, 3), None) if ratio is not None else (None, "too little history for the ratio")


def _since(bars: pd.DataFrame, t: str, pos: dict) -> tuple:
    """(the name's last price by the deadline, its peak since its fill), or (None, None)
    when the latest bar has no price for it."""
    if not len(bars) or t not in bars.columns:
        return None, None
    last = _f(bars[t].iloc[-1])
    if last is None:
        return None, None
    after = bars[t][bars.index > pd.Timestamp(pos["entry_at"])].dropna()
    peak = max([float(pos["entry_price"])] + ([float(after.max())] if len(after) else []))
    return last, peak


def _new_8k(filings, t, item, since, deadline) -> tuple:
    """(1 when an 8-K for `t` carrying `item` was accepted after the entry and by the
    deadline, else 0; a note)."""
    if filings is None:
        return None, "no filings feed"
    f = filings[(filings["ticker"] == t) & (filings["accepted"] > pd.Timestamp(since))
                & (filings["accepted"] <= pd.Timestamp(deadline))]
    hit = any(str(item).strip() in [x.strip() for x in str(items).split(",")] for items in f["items"])
    return (1.0 if hit else 0.0), None


def _gap(ctx, t, history, since) -> tuple:
    """The move since the last close on a reaction session to results released after the
    entry, in percent; None outside one, or before a bar of it has ended."""
    if history is None:
        return None, "no earnings history"
    acc = history.release(t, ctx.deadline)
    if acc is None or acc <= pd.Timestamp(since):
        return None, "no results since entry"
    p = priced_from(acc, ctx.market.days)
    if p is None or p.date() != ctx.day:
        return None, "not a reaction session"
    pct, _ = move_since_close(ctx, t)
    if pct is None:
        return None, "no bar of the reaction session yet"
    return _r(100 * pct, 3), None


def _unreadable(c: dict) -> Optional[str]:
    """Why a threshold has no direction code can read, or None.

    Entry's `condition_errors` refuses a missing threshold but not a sign: a drawdown
    of -3 would read as "fire at once" and a signed move of 0 as "fire on any tick".
    Guessing the PM meant 3 would be code writing the plan, so such a condition never
    fires, and its row says why to every role that reads it.
    """
    kind, th = c["kind"], c.get("threshold")
    if kind == "new_8k_item":
        return None
    if th is None or not np.isfinite(th):
        return "never fires: no threshold"
    if kind in SIGNED and th == 0:
        return "never fires: a signed threshold of 0 has no direction"
    if kind not in SIGNED and th <= 0:
        return f"never fires: {kind} takes a positive threshold, not {th:g}"
    return None


def _distance(c: dict, value) -> tuple:
    """(how far `value` must still move to fire, in the threshold's units; fired now)."""
    if value is None:
        return None, False
    kind, th = c["kind"], c["threshold"]
    if kind == "new_8k_item":
        return (0.0 if value >= 1 else 1.0), value >= 1
    if kind in SIGNED:
        gap = (value - th) if th < 0 else (th - value)
        return _r(gap, 3), gap <= 0
    # drawdowns (positive percent), give_back_from_peak and vol_ratio_above fire at or above
    gap = th - value
    return _r(gap, 3), gap <= 0
