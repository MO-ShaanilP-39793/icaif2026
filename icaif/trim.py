"""Profit booking as a rule (Roadmap step 5): cut part of a winner that has started to give
back, only when its expected give-back is worth the trade.

**The state** of a held name at a morning review (round 1, day 2 on) comes from the
book's own journal (`agents.journal.Journal`), so the rule reads exactly the numbers every
role is shown: `g`, the gain since its fill, and `dd = 1 - last / peak`, what it has given
back from its high-water mark (the fill and every bar close after it). `sigma` is its HAR
forecast of daily vol over the next three sessions, `n` the sessions held and `h` the
sessions left, today included.

**The rule.** Trim `fraction` of the position when the name is still a winner
(g >= a sigma sqrt n: up by a times what its volatility would move it in n sessions), it
has turned (dd >= b sigma), and its expected give-back over the h sessions left exceeds
what the trim costs, both per unit of weight sold:

    EGB > COST + rank_hit

COST is the 20 bps round trip (the entry paid half of it); `rank_hit` prices the
turnover rank a trade gives away on top of the fee. Once per name per window, at most
MAX_TRIMS a window, and never a sale under MIN_TRIM of NAV.

**The expected give-back** is where the rule stands or falls, so it comes three ways,
and the selection (`tools/trim_report.py`) picks among them by score:

- `trailing`: EGB = dd. What it has given back, it gives back again: the classic
  profit-booking belief, stated rather than estimated. With any cost below b sigma it
  never binds, so this is the plain trailing profit-take.
- `expanding`: EGB = -mu sigma sqrt h, where mu is the mean forward return per sigma
  sqrt h of every past state that met the same two conditions (`GiveBack`), over every
  window that ended before this one began.
- `rolling3y`: the same from the three years before only, so a recent regime of winners
  giving back is acted on even if the decade before says they recovered.

A positive mu (winners that turned went on to recover) gives a negative expected
give-back, and the rule does not trim: it refuses to pay for a reversal history says is
not there. Neither estimate reads a return its decision could not have seen.
"""

from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from icaif import calendar, compiler, quant_strategies as qs
from icaif import weights as W

COST = 0.002        # round trip per unit of weight: 10 bps in, 10 out
HORIZON = 3         # sigma: the HAR forecast of mean daily variance over 3 sessions
MIN_TRIM = 0.005    # of NAV: a smaller sale pays the fee and a rank for nothing
MAX_TRIMS = 3       # a window, rule and agent alike
MIN_STATES = 30     # past states an estimate needs; fewer is no estimate, and no trim
ESTIMATORS = ("trailing", "expanding", "rolling3y")
FRACTIONS = {"quarter": 0.25, "half": 0.5}


@dataclass(frozen=True)
class TrimSettings:
    a: float            # still a winner: g >= a sigma sqrt(n)
    b: float            # turned: dd >= b sigma
    fraction: float     # of the position sold, on the lever's grid
    estimator: str      # how the expected give-back is had
    rank_hit: float = 0.0

    def __post_init__(self):
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, not {self.estimator!r}")
        if self.fraction not in FRACTIONS.values():
            raise ValueError(f"fraction must be on the lever's grid {sorted(FRACTIONS.values())}")

    def label(self) -> str:
        lam = f" +{self.rank_hit * 1e4:g}bp" if self.rank_hit else ""
        return f"{self.estimator} a{self.a:g} b{self.b:g} f{self.fraction:g}{lam}"

    def to_json(self) -> dict:
        return asdict(self)


class GiveBack:
    """Every state a held name passed through in past windows, and what came after it.

    One row per (window, day k >= 2, name) of a book bought at day 1's open and held:
    zg = g / (sigma sqrt n), zdd = dd / sigma, and y, the forward return from day k's
    open to the window's last close per sigma sqrt h. `mu` reads only windows that ended
    before the asking window began, so an estimate never holds a return its decision had
    not seen: built from every window up front, a lookup that ignored the end date would
    hand each window its own future.
    """

    def __init__(self, states: pd.DataFrame):
        self.states = states
        self._mu: dict = {}

    @classmethod
    def build(cls, market, har, starts: list[date], window_days: int = 15) -> "GiveBack":
        panel = market.recent_closes(pd.Timestamp("2100-01-01", tz="UTC"), len(market.info_bars))
        var = har.panels[HORIZON].frame
        days, rows = market.days, []
        for s in starts:
            span = days[days.index(s): days.index(s) + window_days]
            entry_at = calendar.rounds_for(span[0])[0]["execution"]
            entry = market.exec_prices.loc[entry_at]
            final = market.closes.loc[calendar.at(span[-1], calendar.session_close(span[-1]))]
            bars = panel[(panel.index > entry_at)
                         & (panel.index <= calendar.at(span[-1], calendar.session_close(span[-1])))]
            peak = bars.cummax().ffill().clip(lower=entry, axis=1)
            last = bars.ffill()
            for k, d in enumerate(span[1:], start=2):
                r1 = calendar.rounds_for(d)[0]
                i = bars.index.searchsorted(r1["deadline"], side="right") - 1
                if i < 0 or pd.Timestamp(d) not in var.index:
                    continue
                sig = np.sqrt(var.loc[pd.Timestamp(d)].reindex(market.tickers))
                lst, pk = last.iloc[i].fillna(entry), peak.iloc[i].fillna(entry)
                n, h = k - 1, window_days - k + 1
                fwd = final / market.exec_prices.loc[r1["execution"]] - 1
                rows.append(pd.DataFrame({
                    "start": s, "end": span[-1], "k": k, "ticker": market.tickers,
                    "zg": ((lst / entry - 1) / (sig * np.sqrt(n))).to_numpy(),
                    "zdd": ((1 - lst / pk) / sig).to_numpy(),
                    "y": (fwd / (sig * np.sqrt(h))).to_numpy()}))
        states = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        return cls(states)

    def mu(self, a: float, b: float, before: date, years: Optional[float] = None) -> Optional[float]:
        """Mean y of past states with zg >= a and zdd >= b, from windows that ended before
        `before` (and began within `years` of it); None with fewer than MIN_STATES."""
        key = (a, b, before, years)
        if key not in self._mu:
            s = self.states
            m = (s["end"] < before) & (s["zg"] >= a) & (s["zdd"] >= b)
            if years is not None:
                m &= s["start"] >= (pd.Timestamp(before) - pd.DateOffset(years=int(years))).date()
            y = s.loc[m, "y"]
            self._mu[key] = float(y.mean()) if len(y) >= MIN_STATES else None
        return self._mu[key]


class TrimRule:
    def __init__(self, settings: TrimSettings, history: Optional[GiveBack] = None):
        if settings.estimator != "trailing" and history is None:
            raise ValueError(f"the {settings.estimator} estimator needs the GiveBack history")
        self.s, self.history = settings, history

    def expected_give_back(self, dd: float, sigma: float, h: int, start: date) -> Optional[float]:
        s = self.s
        if s.estimator == "trailing":
            return dd
        mu = self.history.mu(s.a, s.b, start, 3 if s.estimator == "rolling3y" else None)
        return None if mu is None else -mu * sigma * np.sqrt(h)

    def proposals(self, positions: dict, weights: pd.Series, sigma: pd.Series, *, day_no: int,
                  window_days: int, start: date, done: set, trims_left: int) -> list[dict]:
        """This morning's trims, largest expected give-back first, at most `trims_left`.

        `positions`: the journal's held names; `weights`: the book now; `sigma`: HAR daily
        vol per name for today. A name without a forecast, an entry the journal had to
        estimate, or one already trimmed this window is left alone.
        """
        s, out = self.s, []
        h = window_days - day_no + 1
        for t in sorted(positions):
            pos = positions[t]
            ep, last, peak = pos.get("entry_price"), pos.get("last_price"), pos.get("peak_price")
            w = float(weights.get(t, 0.0))
            n = day_no - (pos.get("entry_day") or day_no)
            sg = float(sigma.get(t, np.nan))
            if (t in done or pos.get("estimated") or not (ep and last and peak) or n < 1
                    or h < 1 or not sg > 0 or w <= compiler.HELD):
                continue
            g, dd = last / ep - 1, 1 - last / peak
            if g < s.a * sg * np.sqrt(n) or dd < s.b * sg:
                continue
            egb = self.expected_give_back(dd, sg, h, start)
            if egb is None or egb <= COST + s.rank_hit or s.fraction * w < MIN_TRIM:
                continue
            out.append({"ticker": t, "fraction": s.fraction, "egb": float(egb), "gain": float(g),
                        "give_back": float(dd), "sigma": sg, "sessions_left": h})
        out.sort(key=lambda x: (-x["egb"], x["ticker"]))
        return out[:max(trims_left, 0)]


def sigma_today(har, ctx) -> pd.Series:
    """Each name's HAR daily vol for the decision's own day (NaN where none)."""
    return np.sqrt(har.for_day(ctx.day, ctx.deadline)[f"har_h{HORIZON}"])


def apply(current: pd.Series, trims: list[dict]) -> pd.Series:
    """The book with each trim's fraction sold, every other weight as it stands: the one
    target arithmetic for the rule, its desk twin and the agent's lever."""
    target = current.copy()
    for p in trims:
        target[p["ticker"]] = target[p["ticker"]] * (1 - p["fraction"])
    return target


class TrimmedRiskParity(qs.QuantBook):
    """`q_riskparity_entry_regime`, plus the trim rule at each morning review.

    The selection's fast stand-in for the rule desk (`Desk` with `trim=`), and held to
    it trade for trade by a test: the state comes from a `Journal` kept the way the desk
    keeps its own, so a gain or a peak can't differ between the book that was scored and
    the desk that would trade it. With no trim it is the candidate, call for call.
    """

    def __init__(self, rule: TrimRule, har, window_days: int = 15):
        from icaif.agents.journal import Journal

        super().__init__("risk_parity", qs.Regime(), qs.ENTRY_ONLY)
        self.rule, self.har, self.window_days = rule, har, window_days
        self.journal = Journal()
        self.day_no, self._day, self.start = 0, None, None
        self.done: set = set()
        self.trims: list[dict] = []

    def __call__(self, ctx):
        if ctx.round != 1:
            return None
        if ctx.day != self._day:
            self._day, self.day_no = ctx.day, self.day_no + 1
            self.start = self.start or ctx.day
        self.journal.open_round(ctx, self.day_no)
        target = super().__call__(ctx)   # the entry, or the entry-only band's hold
        if target is None and self.shape is not None and self.day_no > 1:
            target = self._trim(ctx)
        self.journal.close_round([], target)
        return target

    def _trim(self, ctx) -> Optional[dict]:
        tickers = ctx.market.tickers
        # Valued as `Desk._value` values it, so the twin and the desk size a trim off the
        # same weights: an unheld name needs no price, an unpriced held one stops the round.
        shares = pd.Series(ctx.shares, dtype=float).reindex(tickers).fillna(0.0)
        last = ctx.recent_closes(1)
        px = last.iloc[-1].reindex(tickers) if len(last) else pd.Series(np.nan, index=tickers)
        value = shares * px.where(shares != 0, 0.0)
        nav = ctx.cash + float(value.sum(skipna=False))
        if not np.isfinite(nav) or nav <= 0:
            return None
        current = value / nav
        props = self.rule.proposals(self.journal.positions, current, sigma_today(self.har, ctx),
                                    day_no=self.day_no, window_days=self.window_days,
                                    start=self.start, done=self.done,
                                    trims_left=MAX_TRIMS - len(self.trims))
        if not props:
            return None
        for p in props:
            self.done.add(p["ticker"])
            self.trims.append({"day": self.day_no, **p})
        return W.safe(apply(current, props).clip(lower=0.0, upper=W.CAP).to_dict(), tickers)


def trimmed(rule: TrimRule, har, window_days: int = 15):
    """A `windows.run_field` factory: a fresh book, journal and HMM per window."""
    return lambda: TrimmedRiskParity(rule, har, window_days)
