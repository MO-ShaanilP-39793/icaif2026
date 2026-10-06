"""What each role may answer. The schema is the first guard; `Desk` checks the rest.

Every field is required and bounded, so a structured-output call either returns a
decision the desk can execute or fails validation and falls back. Bounds are the
levers' own ranges, never clamps: a clamped answer runs a book the agent did not
choose, and its rationale would describe one that never existed.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SHAPES = ("inverse_vol", "risk_parity")
EXPOSURE_MIN, EXPOSURE_MAX = 0.30, 0.95


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Exclusion(_Strict):
    """One name left out of the entry, and the signal that put it there.

    The signal is a closed set so a replay can score exclusions by cause (did leaving
    out reporters help? low scores?); `why` is the case for this name in this window.
    """

    name: str
    signal: Literal["model_score", "earnings", "volatility", "filing", "other"]
    why: str = Field(max_length=300)


class EntryDecision(_Strict):
    """The Strategist's one decision: how to enter the window."""

    shape: Literal["inverse_vol", "risk_parity"] = Field(
        description="Book shape: inverse_vol ignores correlation; risk_parity equalises "
                    "each name's share of variance.")
    views: Literal["none", "light", "strong"] = Field(
        description="How much the model scores tilt the risk_parity book "
                    "(Black-Litterman); none for inverse_vol.")
    exposure: float = Field(ge=EXPOSURE_MIN, le=EXPOSURE_MAX,
                            description="Gross weight to enter at; the rest is cash.")
    avoid: list[Exclusion] = Field(max_length=8,
                                   description="Names to leave out of the book entirely.")
    rationale: str = Field(max_length=1500)


# The trim lever's grid: the fraction of a position sold. Two steps, so a trim is a
# decision about whether, not a dial tuned to a story; the desk adds the floor and the cap.
TRIM_FRACTIONS = ("quarter", "half")
TrimCause = Literal["give_back", "news", "filing", "volatility", "earnings"]


class Trim(_Strict):
    """Part of one held name sold, how much, and the cause behind it.

    The cause is a closed set so a replay can score trims by cause (did booking a
    give-back pay? trimming into a filing?); `why` is the case for this name today.
    """

    name: str
    fraction: Literal["quarter", "half"] = Field(description="Of the position, sold.")
    cause: TrimCause
    why: str = Field(max_length=300)


class ReviewDecision(_Strict):
    """A morning review after entry. Holding is free; every trade costs turnover rank."""

    action: Literal["hold", "set_exposure", "rebalance"]
    exposure: Optional[float] = Field(
        ge=0.0, le=EXPOSURE_MAX,
        description="Required for set_exposure: the new gross weight. For rebalance, "
                    "null keeps today's gross.")
    reason: Optional[Literal["score_change", "vol_change"]] = Field(
        description="Required for rebalance: what changed since entry that the book "
                    "should follow. Null otherwise.")
    exit: list[str] = Field(max_length=8, description="Held name codes to sell outright.")
    trim: list[Trim] = Field(max_length=3, description="Held names to sell part of, with "
                                                       "hold or set_exposure; none with rebalance.")
    rationale: str = Field(max_length=1500)


class NameCall(_Strict):
    name: str
    action: Literal["hold", "exit", "trim"]
    fraction: Optional[Literal["quarter", "half"]] = Field(
        description="Required for trim: the fraction of the position sold. Null otherwise.")
    cause: Optional[TrimCause] = Field(description="Required for trim. Null otherwise.")
    reason: str = Field(max_length=600)


class EventDecision(_Strict):
    """The analyst's call on each name a trigger fired for."""

    calls: list[NameCall] = Field(max_length=30)


SCHEMAS = {"entry": EntryDecision, "review": ReviewDecision, "event": EventDecision}


class NameWeight(_Strict):
    name: str
    weight: float = Field(ge=0.0, le=0.30)


class FreeDecision(_Strict):
    """The whole decision, in the agent's hands: hold, or the full book to hold now.

    Weights are listed per name (a map would not survive strict structured output);
    names left out are sold. The desk checks the sum and the codes, and rejects rather
    than rescales a book over 100%: a rescaled book is one the agent did not choose.
    """

    action: Literal["hold", "rebalance"]
    weights: list[NameWeight] = Field(max_length=30,
                                      description="Required for rebalance: every name to hold.")
    rationale: str = Field(max_length=1500)


SCHEMAS["free"] = FreeDecision


# ----------------------------------------------------------------------------- desk v2

class AddLine(_Strict):
    """Buy one name up to a stated weight: a new name, or more of one held."""

    name: str
    weight: float = Field(ge=0.0, le=0.30,
                          description="The name's weight of NAV after the trade; above its "
                                      "current weight, at most 6 decimals.")
    why: str = Field(max_length=300)


class CutLine(_Strict):
    """Sell one held name outright."""

    name: str
    why: str = Field(max_length=300)


class TradeList(_Strict):
    """The trader's proposal, and the portfolio manager's decision (desk v2).

    Lines name what changes and nothing else, so a name no line touches keeps its weight
    unless `target_exposure` scales it. Code compiles the list into weights
    (`agents.tradelist.compile_trades`) and refuses it whole if any line is wrong: a list
    with its bad line dropped is a book nobody proposed. No lines and no exposure is a hold.
    """

    adds: list[AddLine] = Field(max_length=30)
    cuts: list[CutLine] = Field(max_length=30)
    trims: list[Trim] = Field(max_length=30)
    target_exposure: Optional[float] = Field(
        ge=0.0, le=1.0,
        description="Gross weight after the trade, reached by scaling the names no line "
                    "touches; null leaves the gross where the lines put it.")
    rationale: str = Field(max_length=1500)


SCHEMAS["trade_list"] = TradeList
