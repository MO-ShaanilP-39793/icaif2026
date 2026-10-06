"""The trade-list compiler (desk v2): a `schemas.TradeList` into weights, or a refusal.

In v2 the roles say what to change (adds, cuts, quarter or half trims, a target
exposure) and code alone writes weights, as in v1. What v1 learned stays: the desk never
repairs an answer. A list over the cap, over 100%, naming an unknown name or spending
more turnover than is left is refused whole, with every reason at once. Rescaled,
clipped or with its bad line dropped, it would trade a book no role proposed, and the
rationale logged beside it would describe a different one.

**Semantics.** Lines are exact: a cut leaves a held name at 0, a trim sells that fraction
of the position, an add sets the name to its stated weight. `target_exposure` is reached
by scaling only the names no line touches, so a stated weight is never moved by
someone else's exposure call. It is refused when the lines alone hold more than the
target, or when no name is left to scale and the target differs from the lines' total.

**Checks**, each a refusal:
- every name one of the 30 (a name shown only in the universe block is named as such),
  and each in at most one line;
- cuts and trims only of names held; an add only above the name's current weight (a
  smaller one is a sale, which is a trim's or a cut's job);
- stated numbers on the 1e-6 grid: what is submitted is floored to it (`weights.safe`),
  so a weight stated with more decimals would trade as a number no role stated;
- each line moving at least `MIN_TRADE` of NAV: a smaller trade pays the fee and a
  turnover rank for nothing;
- each name at most `weights.CAP`, gross at most `exposure_cap` (1.0 unless the caller
  holds it lower);
- the list's turnover within what the window's budget has left (`budget.TurnoverBudget`).
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import pandas as pd

from icaif import trim as TR
from icaif import weights as W
from icaif.agents.schemas import TradeList

MIN_TRADE = TR.MIN_TRIM     # of NAV, the v1 trim floor: one floor for every sale
HELD = W.GRID               # a weight below this is flooring residue, not a position


class TradeListError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("trade list refused: " + "; ".join(self.errors))


@dataclass
class Compiled:
    """What the list trades: `weights` is what is submitted (through `weights.safe`)."""

    target: pd.Series
    weights: dict
    turnover: float
    lines: list[dict] = field(default_factory=list)


def on_grid(x: float) -> bool:
    """At most six decimals, read as the organizer reads a weight (`Decimal(str(w))`)."""
    return Decimal(repr(float(x))).as_tuple().exponent >= -6


def compile_trades(tl: TradeList, current: pd.Series, to_ticker: dict, *, budget_left: float,
            exposure_cap: float = 1.0, universe_names=frozenset()) -> Optional[Compiled]:
    """The list's book, or None for a hold; raises `TradeListError` naming every problem.

    `current`: today's weights, indexed by every tradeable ticker. `to_ticker`: the codes
    the roles were shown, mapped to tickers. `budget_left`: `TurnoverBudget.left(journal)`.
    `universe_names`: codes shown only as context, so a role that names one is told why.
    """
    current = current.astype(float)
    if not (tl.adds or tl.cuts or tl.trims) and tl.target_exposure is None:
        return None
    errors: list[str] = []

    def resolve(code: str) -> Optional[str]:
        if code in to_ticker:
            return to_ticker[code]
        if code in universe_names:
            errors.append(f"{code} is in universe_context only and cannot be traded")
        else:
            errors.append(f"{code} is not one of the 30 tradeable names")
        return None

    lines = ([("add", x.name, x.weight) for x in tl.adds] + [("cut", x.name, None) for x in tl.cuts]
             + [("trim", x.name, TR.FRACTIONS[x.fraction]) for x in tl.trims])
    names = [code for _, code, _ in lines]
    twice = sorted({c for c in names if names.count(c) > 1})
    if twice:
        errors.append(f"names in more than one line: {twice}")
    for x in tl.adds:
        if not on_grid(x.weight):
            errors.append(f"{x.name}: weight {x.weight!r} has more than 6 decimals")
    if tl.target_exposure is not None and not on_grid(tl.target_exposure):
        errors.append(f"target_exposure {tl.target_exposure!r} has more than 6 decimals")
    resolved = [(kind, code, resolve(code), arg) for kind, code, arg in lines]
    if errors:
        raise TradeListError(errors)

    target = current.copy()
    for kind, code, t, arg in resolved:
        held = current[t] > HELD
        if kind in ("cut", "trim") and not held:
            errors.append(f"{code}: a {kind} of a name not held")
        elif kind == "cut":
            target[t] = 0.0
        elif kind == "trim":
            target[t] = current[t] * (1.0 - arg)
        elif arg <= current[t]:
            errors.append(f"{code}: an add to {arg:.4f} is at or below its weight {current[t]:.4f}; "
                          f"a sale is a trim or a cut")
        else:
            target[t] = arg

    touched = {t for _, _, t, _ in resolved}
    if tl.target_exposure is not None:
        rest = [t for t in target.index if t not in touched]
        fixed = float(target[[t for t in target.index if t in touched]].sum())
        need, free = tl.target_exposure - fixed, float(target[rest].sum())
        if need < -HELD:
            errors.append(f"the lines alone hold {fixed:.4f}, above the target exposure "
                          f"{tl.target_exposure:.4f}")
        elif free <= HELD:
            if abs(need) > HELD:
                errors.append(f"no name outside the lines to scale to the target exposure "
                              f"{tl.target_exposure:.4f}; the lines hold {fixed:.4f}")
        else:
            target[rest] = target[rest] * (need / free)

    for kind, code, t, _ in resolved:
        move = abs(float(target[t] - current[t]))
        if move < MIN_TRADE and not any(e.startswith(f"{code}:") for e in errors):
            errors.append(f"{code}: the {kind} moves {move:.4f} of NAV, under the {MIN_TRADE} "
                          f"minimum trade")
    over = sorted(t for t in target.index if target[t] > W.CAP + 1e-12)
    if over:
        errors.append("over the 30% cap: " + ", ".join(f"{t} {target[t]:.4f}" for t in over))
    gross = float(target.sum())
    if gross > exposure_cap + 1e-12:
        errors.append(f"gross {gross:.4f} is over the {exposure_cap:g} allowed")
    turnover = float((target - current).abs().sum())
    if not resolved and turnover < MIN_TRADE:
        errors.append(f"the exposure change turns over {turnover:.4f} of NAV, under the "
                      f"{MIN_TRADE} minimum trade")
    if turnover > budget_left + 1e-9:
        errors.append(f"the list turns over {turnover:.4f} of NAV and the window's budget has "
                      f"{budget_left:.4f} left")
    if errors:
        raise TradeListError(errors)

    weights = W.safe(target.to_dict(), list(target.index))
    # Checked above to be within the cap and the gross, so `safe` only floors to the grid.
    # A rescale here would be a book other than the one checked.
    assert max(abs(weights[t] - target[t]) for t in target.index) <= W.GRID
    return Compiled(target, weights, turnover,
                    [{"name": code, "ticker": t, "action": kind, "from": float(current[t]),
                      "to": float(target[t])} for kind, code, t, _ in resolved])
