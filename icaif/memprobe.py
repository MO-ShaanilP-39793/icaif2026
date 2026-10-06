"""The memory probe: does the model remember how a window turned out?

A replay with real names is a fair test only if the model cannot recall the window. A
model trained past a window's end may know that a name fell 9% on its results, and a
desk that "foresees" it scores a skill it does not have, at a rank that reads as
skill. Nothing in the replay would show it: the reasons the desk writes cite the
headlines it was shown, not what it remembers.

So before a window is replayed, the model is asked to estimate the window's biggest
moves by name and date, framed as a calibration exercise, with nothing else to go on.
Two sets of moves:

- **earnings**: each name's close-to-close move on its results' reaction session
  (`EarningsHistory`: the first session whose open reflects the item-2.02 release),
  told that it is a reaction to results;
- **largest**: the window's `LARGEST` largest absolute daily moves, by name and date
  only.

A model that is guessing has no edge on either: its signs agree with the actual moves
at about the rate their up/down mixes imply (a model that calls every move up, in a
window where most were up, agrees often and knows nothing), and its guesses are
uncorrelated with the moves. A set where the signs beat that rate, or the correlation
is above zero, at `ALPHA` one-sided flags the window as remembered, and it is dropped.

It is a screen, not a proof. With 10-20 moves a set, a model that recalls a few of them
passes, and its four tests flag a window the model never saw about one time in nine
(11% in simulation, 7% a set): a false flag costs a window, a missed memory costs the
evaluation. Run on a window the model
surely saw (before its training cutoff) it should flag; if it does not, it is too weak
to clear anything. A probe the model did not answer is "unanswered", never clean.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from icaif import earnings

LARGEST = 20
ALPHA = 0.05
# Fewer moves than this cannot beat chance at ALPHA however they are guessed (4 of 4
# signs right is p = 0.06 at even odds), so such a set is not asked: a window with two
# results in it is judged on its largest moves alone.
MIN_LINES = 5
SYSTEM = (
    "This is a forecasting calibration exercise. Each line names a US stock and a "
    "trading date. Give your best numeric estimate of that stock's one-day move on that "
    "date, close to close, in percent (e.g. -4.2 for a 4.2% fall). Give a number for "
    "every line, even if uncertain, and answer each line under its id.")


class Guess(BaseModel):
    id: str = Field(description="the line's id, exactly as given")
    pct: float = Field(description="estimated one-day close-to-close move, in percent")


class Guesses(BaseModel):
    guesses: list[Guess]


def earnings_moves(reactions: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    """Each name's results reaction in the window: (id, ticker, day, actual %, note).

    `reactions`: `EarningsHistory.table`. Previews filed under item 2.02 (Tesla's
    deliveries) are folded into their quarter's release (`earnings.quarterly`), and a
    name reacts once a window: two lines for one name would count its memory twice.
    """
    t = earnings.quarterly(reactions) if len(reactions) else reactions
    t = t[t["session"].isin(set(days))].sort_values(["session", "ticker"])
    t = t.drop_duplicates("ticker")
    acc = pd.to_datetime(t["accepted"]).dt.tz_convert("America/New_York")
    return pd.DataFrame({
        "id": [f"{tk} {d}" for tk, d in zip(t["ticker"], t["session"])],
        "ticker": t["ticker"].to_numpy(), "day": t["session"].to_numpy(),
        "actual": (100 * t["pct"]).round(2).to_numpy(),
        "note": [f"results released {a:%Y-%m-%d %H:%M} ET; this is the first session to trade on them"
                 for a in acc]}).reset_index(drop=True)


def largest_moves(daily: pd.DataFrame, days: list[date], n: int = LARGEST) -> pd.DataFrame:
    """The window's `n` largest absolute close-to-close moves: (id, ticker, day, actual %).

    `daily`: one official close per session (`triggers.daily_panel`). A move needs both
    closes: a missing price is a hole, never a zero move or a carried-forward one.
    """
    ret = daily.pct_change(fill_method=None)
    ret.index = [ts.date() for ts in ret.index]
    r = ret[ret.index.isin(set(days))].stack().dropna()
    top = r.abs().sort_values(ascending=False, kind="stable").head(n).index
    return pd.DataFrame({"id": [f"{t} {d}" for d, t in top], "ticker": [t for _, t in top],
                         "day": [d for d, _ in top], "actual": [round(100 * r[k], 2) for k in top],
                         "note": ""})


def payload(moves: pd.DataFrame) -> dict:
    return {"lines": [{"id": m.id, **({"note": m.note} if m.note else {})}
                      for m in moves.itertuples()]}


def attach(moves: pd.DataFrame, answer: Guesses) -> pd.DataFrame:
    """The moves with the model's guess beside each; a line it skipped is NaN.

    Matched by id, first answer wins. A skipped line is left out of the scores and
    counted, never filled: a filled guess is one the model did not make.
    """
    got = {}
    for g in answer.guesses:
        got.setdefault(g.id.strip(), g.pct)
    return moves.assign(guess=[got.get(i, np.nan) for i in moves["id"]])


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def score(moves: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Sign agreement against chance, correlation, and the guesses' sizes, for one set."""
    from scipy import stats

    a = moves.dropna(subset=["actual", "guess"])
    n = len(a)
    out = {"lines": len(moves), "answered": n}
    if n < MIN_LINES:
        return {**out, "remembered": None, "why": f"answered {n} of {len(moves)} lines"}
    sa, sg = np.sign(a["actual"].to_numpy()), np.sign(a["guess"].to_numpy())
    right = int((sa == sg).sum())
    up_a, up_g = (sa > 0).mean(), (sg > 0).mean()
    down_a, down_g = (sa < 0).mean(), (sg < 0).mean()
    chance = float(up_a * up_g + down_a * down_g)
    p_sign = _binom_sf(right, n, chance) if chance < 1 else 1.0
    r = float(np.corrcoef(a["actual"], a["guess"])[0, 1]) if a["guess"].std() > 0 else float("nan")
    if np.isfinite(r) and abs(r) < 1:
        p_corr = float(stats.t.sf(r * math.sqrt((n - 2) / (1 - r * r)), n - 2))
    else:
        p_corr = 0.0 if r == 1 else 1.0
    err = (a["guess"] - a["actual"]).abs()
    flags = [w for w, p in (("signs", p_sign), ("correlation", p_corr)) if p < alpha]
    return {**out, "sign_right": right, "sign_chance": round(chance, 3), "p_sign": round(p_sign, 4),
            "corr": round(r, 3) if np.isfinite(r) else None, "p_corr": round(p_corr, 4),
            "mean_abs_guess": round(float(a["guess"].abs().mean()), 2),
            "mean_abs_actual": round(float(a["actual"].abs().mean()), 2),
            "max_abs_guess": round(float(a["guess"].abs().max()), 2),
            "within_1pp": int((err <= 1.0).sum()),
            "remembered": bool(flags), "why": " and ".join(flags) or "no edge"}


def untested(moves: pd.DataFrame) -> dict | None:
    """Why a set is not asked (too few moves to test), or None if it is asked."""
    if len(moves) < MIN_LINES:
        return {"lines": len(moves), "asked": False, "why": f"only {len(moves)} moves"}
    return None


def verdict(scores: list[dict]) -> str:
    """A window's call from its sets. Remembered if any set flags. Unanswered if an asked
    set got too few answers to test: a failed probe clears nothing. Untested if no set
    was asked. Clean only when every asked set was tested and none flagged."""
    asked = [s for s in scores if s.get("asked", True)]
    if any(s.get("remembered") for s in asked):
        return "remembered"
    if any(s.get("remembered") is None for s in asked):
        return "unanswered"
    return "clean" if asked else "untested"
