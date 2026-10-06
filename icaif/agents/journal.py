"""Portfolio memory: one book's journal of decisions, fills and held names (Roadmap step 4).

Without it a desk carries only flags between rounds (`Desk.state`): the agent sees the
book's weights and nothing of how they came about, why, or what each decision has
earned since. It can contradict its own entry by day 3 and never know. Step 5's trim
needs more again: a winner's gain since entry and the best it has been since, so the
give-back is a number the agent reads rather than one it has to remember.

One journal per desk, so one per book: live, the rule desk's is the submitted book's
and the shadow agent's is its paper book's. Each round:

- **Reconcile** (`open_round`). The book the round starts from (the server's portfolio
  live, a paper book in a dry run, the simulator's ledger in a replay) is compared with
  what the journal expects: last round's book plus the fill of each order still open,
  sized by `sim.rebalance` at the execution's prices. A match records the fill. Anything
  else is an issue naming the names and cash that differ, and the journal then adopts
  the book as it stands. The book is the source of truth. The journal never edits it,
  and is never brought into line with it without saying so.
- **Mark.** Each held name's last price and its high since entry: its fill, then every
  bar closed after the fill by the deadline.
- **Record** (`close_round`, `set_order`). The round's decisions as the desk logged them
  (role, brain or fallback, levers, stated reason), what was submitted, and, once a
  later round sees it, the fill and the P&L since.

**Point in time.** Bars come through `Market.recent_closes` and fills through
`Market.fill_prices`, both cut at the deadline, so a journal holds only rounds that have
executed: a round's own order has no fill until a later round sees one in the book.

**What a role reads** (`memory`) is bounded: the last `RECENT_ROUNDS` rounds in full, with
quiet holds folded, earlier days one line each, and the block held under
`MEMORY_MAX_CHARS` by dropping the oldest detail first. In replays it carries codes and
day numbers, and returns rather than prices: no ticker, date or price level.
"""

import copy
import json
from typing import Callable, Optional

import numpy as np
import pandas as pd

from icaif import sim

VERSION = 1
# A share count below this is flooring residue, not a position (= portfolio.SHARE_EPS).
EPS = 1e-9
# A gap between the book and the journal's expectation larger than this, in one name's
# value or in cash, is an issue. Vendor noise sits far below it: Yahoo's :30 opens, which
# price our expectation, differ from Alpaca's, which the organizers fill at, by p99 21 bps,
# about 0.6 bps of NAV on a 3% name. A missed or unexpected fill is the whole trade.
TOLERANCE = 5e-4
# Order statuses after which a fill is expected (or may come); any other means none.
EXPECT = {"placed": "yes", "dry-run": "yes", "paper": "yes", "uploaded": "yes",
          "executed": "yes", "ambiguous": "maybe"}

RECENT_ROUNDS = 7
WHY_RECENT, WHY_SHORT, WHY_EARLIER = 300, 120, 100
KEEP_CLOSED, KEEP_ISSUES = 6, 6
MEMORY_MAX_CHARS = 6000
LOOKBACK_BARS = 400


def _f(x) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else float(x)


def _r(x, nd: int = 4) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _ts(x) -> Optional[str]:
    return None if x is None else pd.Timestamp(x).isoformat()


def size(view) -> int:
    """Characters as the brain sends them (`json.dumps(payload, sort_keys=True)`)."""
    return len(json.dumps(view, sort_keys=True))


class Journal:
    """One book's memory. Every field is JSON, so `to_json`/`from_json` round-trips it exactly."""

    def __init__(self):
        self.rounds: list[dict] = []            # one entry per round seen, oldest first
        self.positions: dict[str, dict] = {}    # ticker -> entry, cost, last and peak price
        self.closed: list[dict] = []            # names sold out, and what they made
        self.issues: list[dict] = []
        self.orders: list[dict] = []            # submitted, not yet seen filled; oldest first
        self.book: Optional[dict] = None        # the book as last reconciled
        self.marked_to: Optional[str] = None    # bars ended by this time are in the marks
        self.start_nav: Optional[float] = None
        self.peak_nav: Optional[float] = None
        # The v2 desk's reflection (agents/v2.py): each decision's outcome once code can
        # measure it, and the lessons drawn from them. Kept here because the journal is
        # what persists between live rounds; v1 desks leave both empty, and `memory`
        # never shows them, so nothing a v1 role reads changes.
        self.settlements: list[dict] = []
        self.lessons: list[dict] = []
        self._open: Optional[dict] = None       # this round's entry, until close_round

    # ------------------------------------------------------------------ persistence

    def to_json(self) -> dict:
        return copy.deepcopy({
            "version": VERSION, "rounds": self.rounds, "positions": self.positions,
            "closed": self.closed, "issues": self.issues, "orders": self.orders,
            "book": self.book, "marked_to": self.marked_to, "start_nav": self.start_nav,
            "peak_nav": self.peak_nav, "settlements": self.settlements, "lessons": self.lessons})

    @classmethod
    def from_json(cls, value) -> "Journal":
        """A journal back from `to_json`. Anything else (none, or the list the desk kept
        before step 4) is a journal with nothing on record, which says so as soon as it
        meets a book that holds names."""
        j = cls()
        if not isinstance(value, dict) or value.get("version") != VERSION:
            return j
        value = copy.deepcopy(value)
        for k in ("rounds", "positions", "closed", "issues", "orders", "book", "marked_to",
                  "start_nav", "peak_nav", "settlements", "lessons"):
            if k in value:
                setattr(j, k, value[k])
        return j

    # ------------------------------------------------------------------ the round

    def open_round(self, ctx, day_no: int) -> dict:
        """Reconcile the book this round starts from, fold in fills and marks, and open
        the round's entry. Reads nothing that ended after `ctx.deadline`."""
        if self._open is not None:   # opened and never closed: on record, not lost
            self._open["note"] = "the round did not finish"
            self._open = None
        key = ctx.round_id or f"day{day_no}-r{ctx.round}"
        tickers = list(ctx.market.tickers)
        shares = {t: float(ctx.shares.get(t, 0.0) or 0.0) for t in tickers}
        cash = float(ctx.cash)
        e = {"key": key, "day": int(day_no), "round": int(ctx.round), "date": str(ctx.day),
             "deadline": _ts(ctx.deadline), "execution": _ts(ctx.execution),
             "source": ctx.book_source, "decisions": [], "action": None}
        self._open = e
        if any(x["key"] == key for x in self.rounds):
            # Run a second time before its deadline (by hand, or a worker retried after
            # a commit): the first run's entry goes, and so does any order it left open.
            self.rounds = [x for x in self.rounds if x["key"] != key]
            self.orders = [o for o in self.orders if o["key"] != key]
            self._issue("round_rerun", [], {}, f"round {key} ran again; its earlier entry was replaced")
        if self.book is None:
            self._start(ctx, shares)
        else:
            self._reconcile(ctx, shares, cash)
        self.book = {"shares": shares, "cash": cash, "key": key, "source": ctx.book_source}
        self._mark(ctx)
        nav = self.nav()
        for pos in self.positions.values():
            pos["weight"] = _f(pos["shares"] * pos["last_price"] / nav) if nav and pos["last_price"] else None
        e.update(nav=_f(nav), cash=cash, names_held=sum(abs(s) > EPS for s in shares.values()))
        if nav is not None:
            self.start_nav = nav if self.start_nav is None else self.start_nav
            self.peak_nav = nav if self.peak_nav is None else max(self.peak_nav, nav)
        self.rounds.append(e)
        return e

    def close_round(self, decisions: list[dict], target: Optional[dict],
                    current: Optional[pd.Series] = None, note: Optional[str] = None) -> None:
        """Record what the desk decided this round. A target is an order placed: a replay
        executes it at the round's execution; live, `set_order` says what became of it."""
        e = self._open
        if e is None:
            return
        self._open = None
        e["decisions"] = [_decision(d) for d in decisions]
        e["action"] = "trade" if target is not None else "hold"
        if note:
            e["note"] = note
        if target is None:
            return
        t = {k: float(v) for k, v in target.items()}
        e.update(target=t, target_gross=float(sum(t.values())))
        if current is not None and not current.isna().any():
            e["turnover"] = float(sum(abs(t.get(k, 0.0) - float(current.get(k, 0.0)))
                                      for k in set(t) | set(current.index)))
        e["order"] = {"status": "placed", "expect": "yes"}
        self.orders.append(_order(e, t, "yes"))

    def set_order(self, key: str, status: str, why: Optional[str] = None,
                  target: Optional[dict] = None, source: Optional[str] = None) -> None:
        """What became of round `key`'s decision: uploaded, put on paper, held by the guard,
        refused. Only a status that can fill keeps an order open. `target` is a book
        another desk submitted for this one (the agent's, with this desk as fallback)."""
        e = self._entry(key)
        if e is None:
            return
        expect = EXPECT.get(status, "no")
        e["order"] = {"status": status, "expect": expect, "why": why, "source": source}
        if target is not None and target != e.get("target"):
            e["submitted_target"] = {k: float(v) for k, v in target.items()}
        tgt = e.get("submitted_target") or e.get("target")
        self.orders = [o for o in self.orders if o["key"] != key]
        if expect != "no" and tgt is not None:
            self.orders.append(_order(e, tgt, expect))
            self.orders.sort(key=lambda o: o["execution"])

    def skipped(self, key: str, round_no: int, date: str, why: str) -> None:
        """A round this book's desk did not run (no prices, no time): on record, not a gap."""
        if self._entry(key) is not None:
            return
        last = self.rounds[-1] if self.rounds else None
        day = 1 if last is None else (last["day"] if last.get("date") == date else last["day"] + 1)
        self.rounds.append({"key": key, "day": day, "round": int(round_no), "date": date,
                            "decisions": [], "action": "none", "note": why})

    # ------------------------------------------------------------------ reconcile

    def _entry(self, key: str) -> Optional[dict]:
        return next((x for x in reversed(self.rounds) if x["key"] == key), None)

    def _issue(self, kind: str, names: list, detail: dict, text: str) -> None:
        e = self._open or {}
        self.issues.append({"key": e.get("key"), "day": e.get("day"), "round": e.get("round"),
                            "date": e.get("date"), "kind": kind, "names": sorted(names),
                            "detail": {k: _f(v) if isinstance(v, float) else v
                                       for k, v in detail.items()},
                            "text": text})

    def _start(self, ctx, shares: dict) -> None:
        held = [t for t, s in shares.items() if abs(s) > EPS]
        if not held:
            return
        # A book that already holds names when the journal starts (a lost state, or a
        # desk first run mid-phase): adopted, each entry estimated at the last close.
        self._issue("no_journal", held, {"names_held": len(held)},
                    "the journal started over a book that already holds names; their "
                    "entries are unknown and priced at the last close")
        last = self._last_closes(ctx)
        for t in held:
            self._buy(t, shares[t], _f(last.get(t, np.nan)), None, ctx, estimated=True)

    @staticmethod
    def _last_closes(ctx) -> pd.Series:
        bars = ctx.market.recent_closes(ctx.deadline, 1)
        return bars.iloc[-1] if len(bars) else pd.Series(dtype=float)

    def _reconcile(self, ctx, shares: dict, cash: float) -> None:
        """Match the book against last round's book plus each open order's fill, in order."""
        tickers = list(shares)
        s0 = np.array([float(self.book["shares"].get(t, 0.0)) for t in tickers])
        chain = [(s0, float(self.book["cash"]), None, None)]
        for o in self.orders:
            ex = pd.Timestamp(o["execution"])
            if ex >= pd.Timestamp(ctx.deadline):
                break
            px = ctx.market.fill_prices(ex, ctx.deadline)
            if px is None:
                break
            s, c = chain[-1][0], chain[-1][1]
            w = np.array([o["target"].get(t, 0.0) for t in tickers])
            raw = px.reindex(tickers).to_numpy(dtype=float)
            touched = (np.abs(w) > 0) | (np.abs(s) > EPS)
            if not np.isfinite(raw[touched]).all():
                break   # that open is not in yet; the paper book waits for it as well
            # A name neither held nor ordered trades at a dummy price, as in PaperBook.settle:
            # 0 x NaN is NaN, and one unpriced bystander would blank the whole fill.
            ns, nc, _ = sim.rebalance(s, c, w, np.where(touched, raw, 1.0))
            chain.append((ns, nc, o, np.where(touched, raw, np.nan)))
        act = np.array([shares[t] for t in tickers])
        px = self._last_closes(ctx).reindex(tickers).to_numpy(dtype=float)
        for *_, p in chain[1:]:
            px = np.where(np.isfinite(p), p, px)   # the latest print of each name
        for i, t in enumerate(tickers):
            if not np.isfinite(px[i]) and t in self.positions:
                px[i] = self.positions[t].get("last_price") or np.nan
        held = np.where(np.abs(act) > EPS, act * px, 0.0)
        nav = cash + float(held.sum())
        nav = nav if np.isfinite(nav) and nav > 0 else sim.INITIAL_NAV

        def gap(s, c):
            g = np.abs(act - s)
            v = np.where(g > EPS, g * px, 0.0)
            if not np.isfinite(v).all():
                return np.inf   # a change in a name nothing can price: never "close enough"
            return max(float(v.max()) if len(v) else 0.0, abs(cash - c)) / nav

        gaps = [gap(s, c) for s, c, _, _ in chain]
        best = min(gaps)
        j = max(i for i, g in enumerate(gaps) if g <= best + 1e-12)
        if best <= TOLERANCE:
            how = "exact" if best < 1e-9 else "within tolerance"
            for k in range(1, j + 1):
                s1, c1 = (act, cash) if k == j else (chain[k][0], chain[k][1])
                self._fill(ctx, chain[k][2], chain[k - 1][0], chain[k - 1][1], s1, c1,
                           chain[k][3], tickers, how)
            self.orders = self.orders[j:]
        else:
            s, c = chain[j][0], chain[j][1]
            names = [t for i, t in enumerate(tickers) if abs(act[i] - s[i]) > EPS
                     and not abs(act[i] - s[i]) * px[i] <= TOLERANCE * nav]
            self._issue("book_differs", names,
                        {"largest_gap_of_nav": best if np.isfinite(best) else None,
                         "cash_gap_of_nav": (cash - c) / nav, "orders_open": len(self.orders)},
                        f"the book differs from the journal's expectation in {len(names)} "
                        f"names and cash by {(cash - c) / nav:+.4%} of NAV; the book stands")
            self._fill(ctx, self.orders[0] if self.orders else None, chain[0][0], chain[0][1],
                       act, cash, px, tickers, "adopted")
            for o in self.orders[1:]:
                oe = self._entry(o["key"])
                if oe is not None and oe.get("fill") is None:
                    oe["fill_status"] = "unresolved: the book differed"
            self.orders = []
        self._expire(ctx)

    def _expire(self, ctx) -> None:
        """Close orders that can no longer fill: a backtest fills at once, live within a day."""
        keep = []
        for o in self.orders:
            ex = pd.Timestamp(o["execution"])
            late = ex < pd.Timestamp(ctx.deadline) and (
                ctx.book_source == "backtest" or ex.date() < ctx.day)
            if not late:
                keep.append(o)
                continue
            e = self._entry(o["key"])
            if e is not None:
                e["fill_status"] = "never filled"
            if o["expect"] == "yes":
                self._issue("unfilled_order", [], {"order_day": o["day"], "order_round": o["round"]},
                            f"the order of {o['key']} was never seen in the book")
        self.orders = keep

    def _fill(self, ctx, order, s0, c0, s1, c1, p, tickers, how: str) -> None:
        d = np.asarray(s1, dtype=float) - np.asarray(s0, dtype=float)
        traded = np.abs(d) > EPS
        if not traded.any() and abs(c1 - c0) <= EPS:
            return
        p = np.asarray(p, dtype=float)
        notional = float(np.abs(d[traded]) @ p[traded]) if traded.any() else 0.0
        before = c0 + float(np.where(np.abs(s0) > EPS, s0 * p, 0.0).sum())
        after = c1 + float(np.where(np.abs(s1) > EPS, s1 * p, 0.0).sum())
        fill = {"at": order["execution"] if order else _ts(ctx.deadline),
                "seen": self._open["key"] if self._open else None, "matched": how,
                "notional": _f(notional), "fee": _f(sim.FEE_RATE * notional),
                "nav_before": _f(before), "nav_after": _f(after), "cash_after": float(c1),
                "turnover": _f(notional / before) if np.isfinite(before) and before > 0 else None,
                "bought": {t: float(d[i]) for i, t in enumerate(tickers) if traded[i] and d[i] > 0},
                "sold": {t: float(-d[i]) for i, t in enumerate(tickers) if traded[i] and d[i] < 0},
                "prices": {t: _f(p[i]) for i, t in enumerate(tickers) if traded[i]},
                "shares_after": {t: float(s1[i]) for i, t in enumerate(tickers) if abs(s1[i]) > EPS}}
        host = self._entry(order["key"]) if order else self._open
        if host is not None:
            if host.get("fill") is None:
                host["fill"] = fill
            else:   # one order per entry, so this is an adoption on top of a fill: keep both
                host.setdefault("more_fills", []).append(fill)
        for i, t in enumerate(tickers):
            if traded[i] and d[i] > 0:
                self._buy(t, float(d[i]), _f(p[i]), order, ctx, estimated=how == "adopted")
            elif traded[i]:
                self._sell(t, float(-d[i]), _f(p[i]), order, ctx)

    def _buy(self, t, q, price, order, ctx, estimated=False) -> None:
        at = order["execution"] if order else _ts(ctx.deadline)
        pos = self.positions.get(t)
        if pos is None:
            o = order or {"key": None, "day": (self._open or {}).get("day"),
                          "date": (self._open or {}).get("date")}
            self.positions[t] = {"entry_key": o["key"], "entry_day": o["day"], "entry_date": o["date"],
                                 "entry_at": at, "entry_price": price, "shares": q,
                                 "last_price": price, "last_at": at, "peak_price": price,
                                 "peak_at": at, "estimated": bool(estimated)}
            return
        new = pos["shares"] + q
        if pos["entry_price"] is not None and price is not None:
            pos["entry_price"] = (pos["shares"] * pos["entry_price"] + q * price) / new
        pos["shares"] = new
        pos["estimated"] = pos["estimated"] or bool(estimated)
        _print(pos, price, at)

    def _sell(self, t, q, price, order, ctx) -> None:
        pos = self.positions.get(t)
        if pos is None:
            return
        at = order["execution"] if order else _ts(ctx.deadline)
        _print(pos, price, at)
        pos["shares"] -= q
        if pos["shares"] > EPS:
            return
        why = None
        e = self._entry(order["key"]) if order else None
        if e is not None and e["decisions"]:
            d = e["decisions"][-1]
            why = f"{d['role']} ({d['source']}): {d['why'][:WHY_EARLIER]}"
        ep, here = pos["entry_price"], order or self._open or {}
        self.closed.append({
            "ticker": t, "entry_day": pos["entry_day"], "entry_date": pos["entry_date"],
            "entry_price": ep, "exit_key": here.get("key"), "exit_day": here.get("day"),
            "exit_date": here.get("date"), "exit_price": price,
            "gain": _f(price / ep - 1) if ep and price else None,
            "peak_gain": _f(pos["peak_price"] / ep - 1) if ep and pos["peak_price"] else None,
            "why": why})
        del self.positions[t]

    def _mark(self, ctx) -> None:
        """Fold every bar closed since the last round into each held name's last and peak."""
        deadline = pd.Timestamp(ctx.deadline)
        if self.positions:
            bars = ctx.market.recent_closes(deadline, LOOKBACK_BARS)
            if self.marked_to is not None:
                bars = bars[bars.index > pd.Timestamp(self.marked_to)]
            for t, pos in self.positions.items():
                if not len(bars) or t not in bars.columns:
                    continue
                col = bars[t]
                col = col[(bars.index > pd.Timestamp(pos["entry_at"])) & col.notna()]
                if len(col):
                    _print(pos, float(col.iloc[-1]), _ts(col.index[-1]))
                    _print(pos, float(col.max()), _ts(col.idxmax()))
        self.marked_to = _ts(deadline)

    def since(self) -> list[dict]:
        """Each round's P&L since it was decided (since its fill, for a trade), to the
        journal's latest mark: what the copy on disk shows beside the rounds."""
        now, out = self.nav(), []
        for e in self.rounds:
            base = (e.get("fill") or {}).get("nav_after") or e.get("nav")
            if now is not None and base:
                out.append({"key": e["key"], "pnl_usd": _f(now - base), "return": _f(now / base - 1)})
        return out

    def nav(self) -> Optional[float]:
        """Cash plus each held name at its last price: the journal's own mark of the book."""
        if self.book is None:
            return None
        v = float(self.book["cash"])
        for t, s in self.book["shares"].items():
            if abs(s) <= EPS:
                continue
            px = (self.positions.get(t) or {}).get("last_price")
            if px is None:
                return None
            v += s * px
        return v

    # ------------------------------------------------------------------ what a role reads

    def name_fields(self, real: bool) -> dict:
        """Per held name: entry day, gain since entry and its high since (live: date, price)."""
        out = {}
        for t, pos in self.positions.items():
            ep, last, peak = pos["entry_price"], pos["last_price"], pos["peak_price"]
            row = {"entry_day": pos["entry_day"],
                   "gain_since_entry": _r(last / ep - 1) if ep and last else None,
                   "peak_gain_since_entry": _r(peak / ep - 1) if ep and peak else None}
            if pos["estimated"]:
                row["entry_estimated"] = True
            if real:
                row.update(entry_date=pos["entry_date"], entry_price=_r(ep, 2))
            out[t] = row
        return out

    def memory(self, code: Callable[[str], str], real: bool,
               budget: int = MEMORY_MAX_CHARS) -> dict:
        """The block every role reads: the book as reconciled, then the rounds before this
        one, the latest in full and earlier ones by day, held under `budget` characters."""
        now = self._open
        done = [e for e in self.rounds if e is not now]
        nav_now = now.get("nav") if now is not None else self.nav()
        recent, earlier = done[-RECENT_ROUNDS:], done[:-RECENT_ROUNDS]
        view = {"book": self._book_view(nav_now, real),
                "rounds": _fold([self._full(e, code, real, nav_now) for e in recent]),
                "earlier_days": self._days(earlier, real, nav_now, recent[0] if recent else now),
                "closed": [_closed_view(c, code, real) for c in self.closed[-KEEP_CLOSED:]],
                "issues": [_issue_view(i, code, real) for i in self.issues[-KEEP_ISSUES:]]}
        return _fit(view, budget)

    def _book_view(self, nav_now, real) -> dict:
        b = self.book or {"cash": None, "shares": {}, "source": None}
        return {"source": b.get("source"),
                "cash": _r(b["cash"] / nav_now) if nav_now and b["cash"] is not None else None,
                "names_held": sum(abs(s) > EPS for s in b["shares"].values()),
                "return_since_start": _r(nav_now / self.start_nav - 1)
                if nav_now and self.start_nav else None,
                "peak_return_since_start": _r(self.peak_nav / self.start_nav - 1)
                if self.peak_nav and self.start_nav else None,
                "orders_pending": [_label(o, real) for o in self.orders],
                "issues_on_record": len(self.issues)}

    def _full(self, e, code, real, nav_now) -> dict:
        row = {"when": _label(e, real), "action": e.get("action")}
        if e["decisions"]:
            row["decided"] = [_decision_view(d, WHY_RECENT) for d in e["decisions"]]
        o = e.get("order") or {}
        if e.get("action") == "trade":
            if o.get("expect") == "no":
                row["submitted"] = f"no: {o.get('status')}" + (f", {o['why'][:80]}" if o.get("why") else "")
            elif e.get("submitted_target") is not None:
                row["submitted"] = f"the {o.get('source')} desk's book instead"
        if e.get("fill") is not None or (e.get("action") == "trade" and o.get("expect") != "no"):
            row["fill"] = _fill_view(e, code)
        base = (e.get("fill") or {}).get("nav_after") or e.get("nav")
        if base and nav_now:
            row["return_since"] = _r(nav_now / base - 1)
        if e.get("note"):
            row["note"] = e["note"][:120]
        n = sum(1 for i in self.issues if i["key"] == e["key"])
        if n:
            row["issues"] = n
        return row

    def _days(self, earlier, real, nav_now, nxt) -> list:
        days: dict = {}
        for e in earlier:
            days.setdefault(e["day"], []).append(e)
        keys, out = sorted(days), []
        for i, d in enumerate(keys):
            es = days[d]
            first = es[0].get("nav")
            later = days[keys[i + 1]][0].get("nav") if i + 1 < len(keys) else (nxt or {}).get("nav", nav_now)
            # A quiet day is one short line: nothing but its return is said about it.
            line = {"day": d}
            if i == len(keys) - 1 and (nxt or {}).get("day") == d:   # split with `rounds`
                line["rounds"] = f"r{es[0]['round']}-r{es[-1]['round']}"
            if real:
                line["date"] = es[0].get("date")
            said = [_decision_line(e, x) for e in es for x in e["decisions"]]
            if said:
                line["decisions"] = said
            trades = sum(e.get("action") == "trade" for e in es)
            if trades:
                line["trades"] = trades
                line["turnover"] = _r(sum((e["fill"].get("turnover") or 0.0) for e in es if e.get("fill")))
            line["return"] = _r(later / first - 1) if first and later else None
            n = sum(1 for x in self.issues if x["day"] == d)
            if n:
                line["issues"] = n
            out.append(line)
        return out


# ----------------------------------------------------------------------------- views

def _order(e: dict, target: dict, expect: str) -> dict:
    return {"key": e["key"], "day": e["day"], "round": e["round"], "date": e["date"],
            "execution": e["execution"], "target": target, "expect": expect}


def _print(pos: dict, price, at) -> None:
    """A price seen for a held name at `at`: its last if no later one is known, its peak if higher."""
    if price is None:
        return
    if pos["last_at"] is None or pd.Timestamp(at) >= pd.Timestamp(pos["last_at"]):
        pos["last_price"], pos["last_at"] = price, at
    if pos["peak_price"] is None or price > pos["peak_price"]:
        pos["peak_price"], pos["peak_at"] = price, at


def _label(e: dict, real: bool) -> str:
    out = f"day {e.get('day')} r{e.get('round')}"
    return f"{out} ({e['date']})" if real and e.get("date") else out


def _decision(d: dict) -> dict:
    """A desk log entry as the journal keeps it: the levers apart from the stated reason."""
    dec = dict(d.get("decision") or {})
    why = dec.pop("rationale", None)
    if "calls" in dec:
        calls = dec["calls"]
        why = "; ".join(f"{c['name']}: {c['reason']}" for c in calls)
        dec = {"calls": [{"name": c["name"], "action": c["action"],
                          **({"fraction": c.get("fraction"), "cause": c.get("cause")}
                             if c["action"] == "trim" else {})} for c in calls]}
    if "weights" in dec:   # the free desk's whole book: its size, not 30 lines a day
        ws = dec.pop("weights") or []
        dec.update(names=len(ws), gross=_r(sum(w["weight"] for w in ws)))
    out = {"role": d.get("role"), "brain": d.get("brain"), "source": d.get("source"),
           "fallback": d.get("reason"), "levers": dec, "why": why or "",
           "same_as_rule": d.get("same_as_rule")}
    if d.get("trims_done") is not None:   # a trim under the floor is a hold: not traded
        out["trimmed"] = list(d["trims_done"])
    return out


def _trim_label(name: str, fraction, cause) -> str:
    return f"{name}:{fraction}({cause})"


def _levers_view(lv: dict) -> dict:
    if "calls" in lv:
        out: dict = {}
        for c in lv["calls"]:
            label = (_trim_label(c["name"], c.get("fraction"), c.get("cause"))
                     if c["action"] == "trim" else c["name"])
            out.setdefault(c["action"], []).append(label)
        if len(out.get("hold", [])) > 5:   # a hold changes nothing; the exits are the decision
            out["hold"] = len(out["hold"])
        return out
    out = {k: _r(v) if isinstance(v, float) else v for k, v in lv.items()}
    if out.get("avoid"):
        out["avoid"] = [{"name": a["name"], "signal": a["signal"]} for a in out["avoid"]]
    if "trim" in out:   # the cause is the lever; the reason sits in `why` with the rest
        out["trim"] = [_trim_label(x["name"], x["fraction"], x["cause"]) for x in out["trim"]]
        if not out["trim"]:
            del out["trim"]
    return out


def _decision_view(d: dict, why_len: int) -> dict:
    row = {"role": d["role"], "source": d["source"], "levers": _levers_view(d["levers"]),
           "why": (d["why"] or "")[:why_len]}
    if d["source"] == "fallback" and d.get("fallback"):
        row["fallback_reason"] = d["fallback"][:100]
    return row


def _decision_line(e: dict, d: dict) -> str:
    lv = _levers_view(d["levers"])
    if "shape" in lv:
        what = f"{lv['shape']}, views {lv.get('views')}, exposure {lv.get('exposure')}"
        if lv.get("avoid"):
            what += ", avoid " + " ".join(f"{a['name']}({a['signal']})" for a in lv["avoid"])
    elif "action" in lv and "names" in lv:
        what = f"{lv['action']} {lv['names']} names, gross {lv.get('gross')}"
    elif "action" in lv:
        what = lv["action"] + (f" {lv['exposure']}" if lv.get("exposure") is not None else "")
        what += f" ({lv['reason']})" if lv.get("reason") else ""
        what += (" exit " + " ".join(lv["exit"])) if lv.get("exit") else ""
        what += (" trim " + " ".join(lv["trim"])) if lv.get("trim") else ""
    else:
        what = "; ".join(f"{a} {' '.join(ns)}" if isinstance(ns, list) else f"{a} {ns} names"
                         for a, ns in lv.items())
    src = "" if d["source"] == "brain" else " (rule fallback)"
    return f"r{e['round']} {d['role']}{src}: {what} - {(d['why'] or '')[:WHY_EARLIER]}"


def _fill_view(e: dict, code):
    f = e.get("fill")
    if f is None:
        return e.get("fill_status") or "not yet seen in the book"
    out = {"turnover": _r(f.get("turnover")),
           "fee_bps_of_nav": _r(1e4 * f["fee"] / f["nav_before"], 2)
           if f.get("fee") is not None and f.get("nav_before") else None}
    for side in ("bought", "sold"):
        names = sorted(code(t) for t in f.get(side, {}))
        if names:
            out[side] = names if len(names) <= 5 else len(names)
    if f.get("matched") != "exact":
        out["matched"] = f.get("matched")
    return out


def _closed_view(c: dict, code, real: bool) -> dict:
    row = {"name": code(c["ticker"]), "entry_day": c["entry_day"], "exit_day": c["exit_day"],
           "gain": _r(c["gain"]), "peak_gain": _r(c["peak_gain"]),
           "why": (c.get("why") or "")[:WHY_EARLIER]}
    if real:
        row.update(entry_date=c["entry_date"], exit_date=c["exit_date"])
    return row


def _issue_view(i: dict, code, real: bool) -> dict:
    return {"when": _label(i, real) if i.get("day") is not None else None, "kind": i["kind"],
            "names": [code(t) for t in i["names"][:10]],
            **{k: _r(v, 6) if isinstance(v, float) else v for k, v in i["detail"].items()}}


def _quiet(row: dict) -> bool:
    return (row.get("action") in ("hold", None) and "decided" not in row and "fill" not in row
            and "note" not in row and "issues" not in row)


def _fold(rows: list) -> list:
    """A day's consecutive quiet holds (no role asked, nothing traded) become one row."""
    out, runs = [], []
    for r in rows:
        head, _, date = r["when"].partition(" (")
        day, rnd = head.split(" r")
        if _quiet(r) and runs and runs[-1] is out[-1] and out[-1]["_day"] == day:
            out[-1]["when"] = f"{day} r{out[-1]['_from']}-r{rnd}" + (f" ({date}" if date else "")
            continue
        if _quiet(r):
            r = {**r, "_day": day, "_from": rnd}
            runs.append(r)
        out.append(r)
    for r in out:
        r.pop("_day", None)
        r.pop("_from", None)
    return out


def _span(line: dict) -> tuple:
    if "day" in line:
        return line["day"], line["day"]
    a, b = line["days"].split("-")
    return int(a), int(b)


def _count(line: dict) -> int:
    d = line.get("decisions")
    return d if isinstance(d, int) else len(d or [])


def _fit(view: dict, budget: int) -> dict:
    """Hold the block under `budget` characters, the oldest detail first, and say what went.

    A hard bound, not a target: past the steps that lose least (older decisions counted,
    older days merged, reasons shortened), the oldest rows lose their detail and then go,
    the latest round always last. Room is kept for the note saying what was cut, which
    is part of the block.
    """
    cut = []
    note = 200
    budget -= note
    days = view["earlier_days"]
    for line in days:
        if size(view) <= budget:
            break
        if isinstance(line.get("decisions"), list) and line["decisions"]:
            line["decisions"] = len(line["decisions"])
            cut.append("older decisions counted, not listed")
    while size(view) > budget and len(days) > 1:
        a, b = days[0], days[1]
        ra, rb = a.get("return"), b.get("return")
        merged = {"days": f"{_span(a)[0]}-{_span(b)[1]}", "decisions": _count(a) + _count(b)}
        trades = a.get("trades", 0) + b.get("trades", 0)
        if trades:
            merged.update(trades=trades, turnover=_r((a.get("turnover") or 0) + (b.get("turnover") or 0)))
        merged["return"] = _r((1 + ra) * (1 + rb) - 1) if ra is not None and rb is not None else None
        if a.get("issues") or b.get("issues"):
            merged["issues"] = (a.get("issues") or 0) + (b.get("issues") or 0)
        days[:2] = [merged]
        cut.append("older days merged")
    for key in ("closed", "issues"):
        while size(view) > budget and len(view[key]) > 2:
            view[key].pop(0)
            cut.append(f"older {key} dropped")
    for row in view["rounds"]:
        for d in row.get("decided", []):
            if size(view) > budget and len(d["why"]) > WHY_SHORT:
                d["why"] = d["why"][:WHY_SHORT]
                cut.append("reasons shortened")
    if size(view) > budget and days:
        view["earlier_days"] = []
        cut.append("earlier days dropped")
    rows = view["rounds"]
    for row in rows[:-1]:
        if size(view) <= budget:
            break
        for d in row.get("decided", []):
            d["why"] = d["why"][:40]
            d.pop("fallback_reason", None)
            d["levers"] = {k: (len(v) if isinstance(v, list) and len(v) > 3 else v)
                           for k, v in d["levers"].items()}
        cut.append("older rounds' detail cut")
    while size(view) > budget and len(rows) > 1:
        rows.pop(0)
        cut.append("older rounds dropped")
    if cut:
        view["trimmed_to_fit"] = sorted(set(cut))
    return view


# ----------------------------------------------------------------------------- the check

def sim_fills(result: "sim.Result", market: "sim.Market") -> list[dict]:
    """The simulator's fills in the shape `verify` reads: one per round that traded."""
    tickers, out = market.tickers, []
    for i, p in enumerate(result.periods):
        if p["traded_notional"] <= 0:
            continue
        row = result.ledger.iloc[i]
        out.append({"execution": _ts(p["execution"]), "notional": float(p["traded_notional"]),
                    "cash_after": float(row["cash"]),
                    "shares_after": {t: float(row[t]) for t in tickers if abs(row[t]) > EPS},
                    "prices": market.exec_prices.loc[p["execution"]].to_dict()})
    return out


def paper_fills(paper: dict) -> list[dict]:
    """A `PaperBook`'s fills (its JSON) in the shape `verify` reads."""
    out = []
    for f in paper.get("fills", []):
        out.append({"execution": _ts(f["execution"]), "notional": float(f["notional"]),
                    "cash_after": float(f["cash_after"]), "round_id": f.get("round_id"),
                    "shares_after": {t: float(s) for t, s in f.get("shares_after", {}).items()
                                     if abs(s) > EPS},
                    "prices": f.get("prices", {})})
    return out




def verify(journal: "Journal", fills: list[dict], *, to_ticker: Optional[Callable] = None,
           closes: Optional[pd.DataFrame] = None, issues_ok: bool = False) -> list[str]:
    """Every way the journal disagrees with the ledger it kept; empty when they agree.

    `fills` is the ledger's own record (`sim_fills`, `paper_fills`); a fill after the
    journal's last round is one it cannot have seen yet and is left out. Checked: each
    fill on record is one the ledger made, to the cent, against the round that ordered
    it, and each ledger fill is on record; a hold traded nothing; an order expected to
    fill has its fill, is still open, or says why not; the stated levers are what the
    book did (an avoided name not bought, an exit sold out, an exposure bought at its
    level); each held name's shares, entry and cost come from the ledger's fills alone,
    and its peak from those fills and `closes` (bar closes by end time) after its entry;
    and no issue is on record unless `issues_ok`. `to_ticker` maps the codes an
    anonymised desk answered in.
    """
    out: list[str] = []
    to_ticker = to_ticker or (lambda c: c)
    opened = [e for e in journal.rounds if e.get("deadline")]
    if not opened:
        return out
    horizon = max(pd.Timestamp(e["deadline"]) for e in opened)
    fills = [f for f in fills if pd.Timestamp(f["execution"]) < horizon]
    by_ex = {f["execution"]: f for f in fills}
    pending = {o["key"] for o in journal.orders}
    on_record = set()
    for e in journal.rounds:
        f, ex, o = e.get("fill"), e.get("execution"), e.get("order") or {}
        if f is not None and f["matched"] != "adopted":
            lf = by_ex.get(f["at"])
            if lf is None or f["at"] != ex:
                out.append(f"{e['key']}: a fill at {f['at']} the ledger never made for this round")
            else:
                on_record.add(ex)
                out.extend(_fill_vs_ledger(e["key"], f, lf))
                out.extend(_levers_vs_book(e, f, to_ticker))
        if e.get("action") == "hold" and ex in by_ex:
            out.append(f"{e['key']}: decided hold, and the ledger traded at {ex}")
        if (e.get("action") == "trade" and o.get("expect") == "yes" and f is None
                and e.get("fill_status") is None and e["key"] not in pending):
            out.append(f"{e['key']}: an order expected to fill, with no fill, not open, and no word why")
    out.extend(f"the ledger filled at {ex} and no round has it on record"
               for ex in by_ex if ex not in on_record)
    out.extend(_positions_vs_ledger(journal, fills, closes))
    if journal.issues and not issues_ok:
        out.extend(f"issue on record at {i['key']}: {i['kind']}: {i['text']}" for i in journal.issues)
    return out


def _fill_vs_ledger(key: str, f: dict, lf: dict) -> list[str]:
    out = []
    if not np.isclose(f["notional"], lf["notional"], rtol=1e-9, atol=1e-6):
        out.append(f"{key}: notional {f['notional']:.2f} vs the ledger's {lf['notional']:.2f}")
    if abs(f["cash_after"] - lf["cash_after"]) > 1e-6:
        out.append(f"{key}: cash after the fill {f['cash_after']:.2f} vs the ledger's {lf['cash_after']:.2f}")
    a, b = f["shares_after"], lf["shares_after"]
    bad = sorted(t for t in set(a) | set(b)
                 if abs(a.get(t, 0.0) - b.get(t, 0.0)) > 1e-9 * max(1.0, abs(b.get(t, 0.0))))
    if bad:
        out.append(f"{key}: shares after the fill differ in {bad[:5]}")
    return out


def _levers_vs_book(e: dict, f: dict, to_ticker) -> list[str]:
    """The structured decision against what its fill left in the book."""
    out, after = [], f["shares_after"]
    own = e.get("submitted_target") is None
    gross = sum((e.get("submitted_target") or e.get("target") or {}).values())
    for d in e["decisions"]:
        lv, role = d["levers"], d["role"]
        if role == "entry":
            out += [f"{e['key']}: the entry avoided {a['name']}, and the book bought it"
                    for a in lv.get("avoid", []) if after.get(to_ticker(a["name"]), 0.0) > EPS]
            if own and abs(gross - lv["exposure"]) > 31e-6:   # the 1e-6 grid, 30 names
                out.append(f"{e['key']}: entry exposure {lv['exposure']:.6f}, target gross {gross:.6f}")
        elif role == "review":
            out += [f"{e['key']}: the review exited {c}, and the book still holds it"
                    for c in lv.get("exit", []) if after.get(to_ticker(c), 0.0) > EPS]
            trimmed = d.get("trimmed", [x["name"] for x in lv.get("trim", [])])
            out += _trims_vs_book(e["key"], "review", trimmed, f, to_ticker)
            if (own and lv.get("action") == "set_exposure" and lv.get("exposure") is not None
                    and not lv.get("exit") and not lv.get("trim")
                    and abs(gross - lv["exposure"]) > 31e-6):
                out.append(f"{e['key']}: exposure set to {lv['exposure']:.6f}, target gross {gross:.6f}")
        elif role == "event":
            out += [f"{e['key']}: the analyst exited {c['name']}, and the book still holds it"
                    for c in lv.get("calls", [])
                    if c["action"] == "exit" and after.get(to_ticker(c["name"]), 0.0) > EPS]
            out += _trims_vs_book(e["key"], "analyst", d.get("trimmed", [
                c["name"] for c in lv.get("calls", []) if c["action"] == "trim"]), f, to_ticker)
    return out


def _trims_vs_book(key: str, who: str, names: list, f: dict, to_ticker) -> list[str]:
    """A trim sells part of a name: some of it sold, some of it still held. `names` are the
    trims that traded (a trim under the floor is a hold, recorded as `trimmed`)."""
    out = []
    for c in names:
        t = to_ticker(c)
        if f["shares_after"].get(t, 0.0) <= EPS:
            out.append(f"{key}: the {who} trimmed {c}, and the book sold all of it")
        elif f["sold"].get(t, 0.0) <= EPS:
            out.append(f"{key}: the {who} trimmed {c}, and the book sold none of it")
    return out


def _positions_vs_ledger(journal: "Journal", fills: list[dict], closes) -> list[str]:
    """Each held name rebuilt from the ledger's fills alone, then compared."""
    out = []
    book = (journal.book or {"shares": {}})["shares"]
    last = fills[-1]["shares_after"] if fills else {}
    held = {t for t, s in book.items() if abs(s) > EPS}
    if fills and held != set(last):
        out.append(f"names held differ from the ledger's: {sorted(held ^ set(last))[:5]}")
    pos: dict = {}
    prev: dict = {}
    for f in fills:
        for t in set(prev) | set(f["shares_after"]):
            a, b = prev.get(t, 0.0), f["shares_after"].get(t, 0.0)
            if abs(b - a) <= EPS:
                continue
            px = float(f["prices"][t])
            p = pos.get(t)
            if b <= EPS:
                pos.pop(t, None)
            elif p is None:
                pos[t] = {"at": f["execution"], "price": px, "shares": b, "prints": [px]}
            else:
                if b > a:
                    p["price"] = (a * p["price"] + (b - a) * px) / b
                p["shares"] = b
                p["prints"].append(px)
        prev = f["shares_after"]
    for t, jp in journal.positions.items():
        p = pos.get(t)
        if p is None:
            out.append(f"{t}: held in the journal with no buy in the ledger")
            continue
        if jp["entry_at"] != p["at"]:
            out.append(f"{t}: entry {jp['entry_at']} vs the ledger's first buy {p['at']}")
        if jp["entry_price"] is None or not np.isclose(jp["entry_price"], p["price"], rtol=1e-12):
            out.append(f"{t}: entry price {jp['entry_price']} vs the ledger's cost {p['price']}")
        if abs(jp["shares"] - p["shares"]) > 1e-9 * max(1.0, p["shares"]):
            out.append(f"{t}: {jp['shares']} shares vs the ledger's {p['shares']}")
        if closes is not None and t in closes.columns:
            col = closes[t]
            col = col[(col.index > pd.Timestamp(p["at"])) & (col.index <= pd.Timestamp(journal.marked_to))]
            peak = max([*col.dropna().tolist(), *p["prints"]])
            if not np.isclose(jp["peak_price"], peak, rtol=1e-12):
                out.append(f"{t}: peak {jp['peak_price']} vs {peak} from its fills and the bars after")
    return out
