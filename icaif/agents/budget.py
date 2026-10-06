"""The window's turnover budget (desk v2): set before the window starts, spent by each fill.

The free desk ranked 34th of 39 on turnover on Jan 21 - Feb 10, 2026: it traded about
three books' worth in 15 sessions, and nothing in it could say no. A prompt that
describes the cost of trading is not a limit. So the budget is code's: every role is
shown what is left, and a trade list that would spend more is refused whole
(`tradelist.compile_trades`).

**The unit is the kit's.** The board's turnover is the mean over the window's rounds of
traded notional / NAV before the trade (`starter-kit/kit/evaluation.py`), so a budget of
B, fully spent, scores B / rounds on the board. Buying the 75% hold spends 0.75, so a
budget below the entry's own exposure leaves no room to enter.

**Spent means filled.** What a role is told it has spent is the sum of the fills the
journal has reconciled against the book (`Journal._fill`'s `turnover`, the kit's own
ratio from the fill's shares and prices), plus the estimate for each order not yet seen
filled. Counting targets instead would charge a refused or unfilled order, and
counting only fills would let two rounds spend the same headroom while the first is in
flight. The journal is what persists between live rounds, so a restarted desk reads the
same budget with no second ledger to drift from it.
"""

from dataclasses import dataclass
from typing import Optional

from icaif import calendar


def window_rounds(days) -> int:
    """Rounds the window's sessions run (a half-day runs four), i.e. the board's divisor."""
    return sum(len(calendar.rounds_for(d)) for d in days)


class BudgetExceeded(ValueError):
    pass


@dataclass(frozen=True)
class TurnoverBudget:
    """`total`: notional / NAV the window may trade, entry included. `rounds`: the window's
    rounds (`window_rounds`), which turn a spend into the board's turnover number."""

    total: float
    rounds: int

    def __post_init__(self):
        if not (self.total > 0 and self.rounds > 0):
            raise ValueError(f"a turnover budget needs a positive total and rounds, not "
                             f"{self.total!r} over {self.rounds!r}")

    @staticmethod
    def filled(journal) -> float:
        """Every fill on record, adoptions included: the board counts what traded,
        whoever decided it."""
        out = 0.0
        for e in journal.rounds:
            for f in [e.get("fill")] + list(e.get("more_fills", [])):
                if f is not None and f.get("turnover") is not None:
                    out += float(f["turnover"])
        return out

    @staticmethod
    def pending(journal) -> float:
        """Orders placed and not yet seen filled, at the turnover their round estimated."""
        keys = {o["key"] for o in journal.orders if o.get("expect") != "no"}
        return float(sum(float(e.get("turnover") or 0.0) for e in journal.rounds
                         if e["key"] in keys and e.get("fill") is None))

    def spent(self, journal) -> float:
        return self.filled(journal) + self.pending(journal)

    def left(self, journal) -> float:
        return max(self.total - self.spent(journal), 0.0)

    def check(self, journal, turnover: float) -> None:
        """Raise if a trade of `turnover` (sum |target - current| over NAV) would overspend."""
        left = self.left(journal)
        if turnover > left + 1e-9:
            raise BudgetExceeded(f"the trade list turns over {turnover:.4f} of NAV and the "
                                 f"window has {left:.4f} of its {self.total:g} left")

    def view(self, journal, proposed: Optional[float] = None) -> dict:
        """What the trader, risk manager and PM are shown."""
        spent = self.spent(journal)
        out = {"window_budget": round(self.total, 4), "spent": round(spent, 4),
               "of_which_unfilled": round(self.pending(journal), 4),
               "left": round(max(self.total - spent, 0.0), 4),
               "board_turnover_if_no_more_trades": round(spent / self.rounds, 5),
               "board_turnover_if_all_spent": round(self.total / self.rounds, 5)}
        if proposed is not None:
            out["this_trade"] = round(proposed, 4)
        return out
