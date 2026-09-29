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


class EntryDecision(_Strict):
    """The Strategist's one decision: how to enter the window."""

    shape: Literal["inverse_vol", "risk_parity"] = Field(
        description="Book shape: inverse_vol ignores correlation; risk_parity equalises "
                    "each name's share of variance.")
    exposure: float = Field(ge=EXPOSURE_MIN, le=EXPOSURE_MAX,
                            description="Gross weight to enter at; the rest is cash.")
    avoid: list[str] = Field(max_length=8,
                             description="Name codes to leave out of the book entirely.")
    rationale: str = Field(max_length=1500)


class ReviewDecision(_Strict):
    """A morning review after entry. Holding is free; every trade costs turnover rank."""

    action: Literal["hold", "set_exposure"]
    exposure: Optional[float] = Field(
        ge=0.0, le=EXPOSURE_MAX,
        description="Required when action is set_exposure: the new gross weight.")
    exit: list[str] = Field(max_length=8, description="Held name codes to sell outright.")
    rationale: str = Field(max_length=1500)


class NameCall(_Strict):
    name: str
    action: Literal["hold", "exit"]
    reason: str = Field(max_length=600)


class EventDecision(_Strict):
    """The analyst's call on each name a trigger fired for."""

    calls: list[NameCall] = Field(max_length=30)


SCHEMAS = {"entry": EntryDecision, "review": ReviewDecision, "event": EventDecision}
