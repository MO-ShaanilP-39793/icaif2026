"""The book a live round starts from: the organizers' portfolio, or a paper one.

**The server's format is not published.** The kit GETs `/api/v1/me/portfolio` and
hands the response to a strategy untouched; neither its docs nor its examples show a
field. `parse` therefore accepts one declared shape (and the few spellings of it
listed below) and raises on anything else, naming the keys it found. A tolerant
reader is the dangerous one here: a book misread as all cash looks like a fresh phase,
and the rule's answer to a fresh phase is to buy the whole entry again over the book
we hold. Run `tools/live_runner.py portfolio` once registration issues credentials,
and fix `parse` to what it shows before arming a live round (`arm` refuses until the
server's response parses).

**A paper book** stands in where the server's cannot: the submitted book in a dry run,
and the shadow agent's own book always. It fills each order at the open of the 30m bar
starting at the round's execution time, through `sim.rebalance`, so a shadow differs
from the submitted book by its decisions and not by a second copy of the fee rule.
Yahoo's :30 opens match Alpaca's (the organizers' vendor) to a median 0 bps, p99 21.
"""

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from icaif import data, sim

# A share count below this is flooring residue, not a position.
SHARE_EPS = 1e-9
# Accepted spellings, each meaning the same thing. Not an alias list for guessing:
# anything outside it raises, so a field that means something else is never read.
POSITION_KEYS = ("positions", "holdings")
SHARES_KEYS = ("shares", "quantity")
SYMBOL_KEYS = ("symbol", "ticker")
NAV_KEYS = ("nav", "total_value", "portfolio_value")
# pre_fee sizing leaves cash about -0.1% x notional when fully invested; more negative
# than this is not the fee, it is a book we have misread.
MIN_CASH_SHARE = -0.02


class PortfolioFormatError(ValueError):
    """The server's portfolio is not in the shape `parse` knows. Raised rather than
    read as all cash; the message names keys, never values."""


@dataclass
class Book:
    cash: float
    shares: dict                    # every ticker, zeros included
    source: str                     # "server" | "paper"
    as_of: Optional[str] = None
    reported_nav: Optional[float] = None

    @property
    def all_cash(self) -> bool:
        return all(abs(s) <= SHARE_EPS for s in self.shares.values())

    def weights(self, prices: pd.Series) -> pd.Series:
        """Weights at `prices`; NaN if a held name has no price (the book can't be valued)."""
        tickers = list(self.shares)
        sh = pd.Series(self.shares, dtype=float).reindex(tickers)
        px = prices.reindex(tickers).astype(float)
        value = sh * px.where(sh.abs() > SHARE_EPS, 0.0)
        # skipna=False: pandas' default would value a held name with no price at zero.
        nav = self.cash + float(value.sum(skipna=False))
        if not np.isfinite(nav) or nav <= 0:
            return pd.Series(np.nan, index=tickers)
        return value / nav

    def summary(self, prices: Optional[pd.Series] = None) -> dict:
        out = {"source": self.source, "cash": self.cash, "as_of": self.as_of,
               "reported_nav": self.reported_nav, "all_cash": self.all_cash,
               "names_held": sorted(t for t, s in self.shares.items() if abs(s) > SHARE_EPS)}
        if prices is not None:
            w = self.weights(prices)
            out["gross"] = None if w.isna().any() else float(w.sum())
        return out


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise PortfolioFormatError(f"{label} is not a number")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise PortfolioFormatError(f"{label} is not a number") from None
    if not math.isfinite(x):
        raise PortfolioFormatError(f"{label} is not finite")
    return x


def _one(obj: dict, keys: tuple, label: str, required: bool = True):
    found = [k for k in keys if k in obj]
    if len(found) > 1:
        raise PortfolioFormatError(f"{label}: both {found} present; which one is meant is a guess")
    if not found:
        if required:
            raise PortfolioFormatError(f"no {label} field ({'/'.join(keys)}) among keys {sorted(obj)}")
        return None
    return obj[found[0]]


def parse(response: dict, tickers: Optional[list[str]] = None) -> Book:
    """The server's portfolio as a `Book`, or `PortfolioFormatError`.

    Declared shape: {"cash": n, "positions": ..., "nav": n?, "as_of": "..."?}, with
    positions either {symbol: shares}, {symbol: {"shares": n, ...}} or
    [{"symbol": s, "shares": n, ...}]. A response nested under "portfolio" is read
    from there. Unknown symbols, negative shares and duplicate symbols raise; so does
    cash far below zero, which no fee explains.
    """
    tickers = tickers or sorted(data.load_universe())
    if not isinstance(response, dict):
        raise PortfolioFormatError("portfolio response is not an object")
    body = response.get("portfolio", response)
    if not isinstance(body, dict):
        raise PortfolioFormatError("'portfolio' is not an object")
    if "cash" not in body:
        raise PortfolioFormatError(f"no 'cash' field among keys {sorted(body)}")
    cash = _number(body["cash"], "cash")
    raw = _one(body, POSITION_KEYS, "positions")
    shares: dict = {}

    def put(symbol, n):
        if not isinstance(symbol, str) or symbol not in tickers:
            raise PortfolioFormatError(f"position symbol outside the 30: {symbol!r}")
        if symbol in shares:
            raise PortfolioFormatError(f"{symbol} listed twice")
        x = _number(n, f"{symbol} shares")
        if x < -SHARE_EPS:
            raise PortfolioFormatError(f"{symbol} has negative shares; the book is long-only")
        shares[symbol] = max(x, 0.0)

    if isinstance(raw, dict):
        for symbol, v in raw.items():
            put(symbol, _one(v, SHARES_KEYS, f"{symbol} shares") if isinstance(v, dict) else v)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                raise PortfolioFormatError("a position is not an object")
            put(_one(item, SYMBOL_KEYS, "position symbol"), _one(item, SHARES_KEYS, "position shares"))
    else:
        raise PortfolioFormatError("positions are neither an object nor a list")

    nav = _one(body, NAV_KEYS, "nav", required=False)
    nav = None if nav is None else _number(nav, "nav")
    if nav is not None and (nav <= 0 or cash < MIN_CASH_SHARE * nav):
        raise PortfolioFormatError("cash is far below zero against the reported NAV; not a fee")
    as_of = body.get("as_of") or body.get("valuation_time")
    return Book(cash=cash, shares={t: shares.get(t, 0.0) for t in tickers}, source="server",
                as_of=None if as_of is None else str(as_of), reported_nav=nav)


def keys_only(value, depth: int = 0):
    """The response's structure with every value blanked: safe to print and to share."""
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        return {k: keys_only(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [keys_only(value[0], depth + 1), f"... {len(value)} items"] if value else []
    return type(value).__name__


# ----------------------------------------------------------------------------- paper

@dataclass
class PaperBook:
    """A book we keep ourselves, filled at Yahoo's opens by the backtest's own rules."""

    cash: float = sim.INITIAL_NAV
    shares: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)   # orders not yet filled, oldest first
    fills: list = field(default_factory=list)
    traded_notional: float = 0.0

    def book(self, tickers: Optional[list[str]] = None) -> Book:
        tickers = tickers or sorted(data.load_universe())
        return Book(cash=self.cash, shares={t: float(self.shares.get(t, 0.0)) for t in tickers},
                    source="paper")

    def order(self, round_id: str, execution: pd.Timestamp, weights: dict) -> None:
        self.pending.append({"round_id": round_id, "execution": str(pd.Timestamp(execution)),
                             "weights": {t: float(w) for t, w in weights.items()}})

    def settle(self, opens: Optional[pd.DataFrame], now: pd.Timestamp) -> list[str]:
        """Fill pending orders whose execution has passed, in order; return what is left
        waiting and why.

        An order waits rather than filling at another price: a fill at the last close
        is a zero return that never happened, and the shadow's P&L would carry it. A
        name neither held nor ordered needs no price; its NaN must not poison the
        arithmetic (0 x NaN is NaN), so it trades at a dummy 1.0 for zero shares.
        """
        issues = []
        while self.pending:
            o = self.pending[0]
            ex = pd.Timestamp(o["execution"])
            if ex > now:
                break
            row = None if opens is None or ex not in opens.index else opens.loc[ex]
            tickers = sorted(set(self.shares) | set(o["weights"]))
            w = np.array([o["weights"].get(t, 0.0) for t in tickers])
            sh = np.array([float(self.shares.get(t, 0.0)) for t in tickers])
            touched = (np.abs(w) > 0) | (np.abs(sh) > SHARE_EPS)
            px = (np.array([np.nan] * len(tickers)) if row is None
                  else row.reindex(tickers).to_numpy(dtype=float))
            if not np.isfinite(px[touched]).all():
                missing = [t for t, ok, tt in zip(tickers, np.isfinite(px), touched) if tt and not ok]
                issues.append(f"order {o['round_id']} waits: no {ex:%H:%M} open for "
                              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
                break
            prices = {t: float(p) for t, p, tt in zip(tickers, px, touched) if tt}
            px = np.where(touched, px, 1.0)
            new, cash, notional = sim.rebalance(sh, self.cash, w, px)
            self.shares = {t: float(s) for t, s in zip(tickers, new)}
            self.cash = float(cash)
            self.traded_notional += notional
            # What the fill left and paid, so a journal's record of it can be checked
            # against this book (`journal.verify`) rather than against itself.
            self.fills.append({**o, "notional": notional, "cash_after": self.cash,
                               "shares_after": {t: s for t, s in self.shares.items() if abs(s) > SHARE_EPS},
                               "prices": prices})
            self.pending.pop(0)
        return issues

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Optional[dict]) -> "PaperBook":
        return cls(**value) if value else cls()
