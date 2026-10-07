"""The answer schemas' free-text caps (icaif/agents/schemas.py, `prose`)."""

import typing

import pytest
from annotated_types import MaxLen
from pydantic import BaseModel, ValidationError

from icaif.agents import brains
from icaif.agents import schemas as S

# Every schema a v2 role answers in, nested ones reached through these.
V2_SCHEMAS = ("trade_list", "analyst", "earnings_report", "debate", "risk", "pm",
              "event_report", "trigger", "reflection")


def _text_fields(model, seen=None):
    """(model, field, enforced cap, stated cap) for every capped str field, recursively."""
    seen = set() if seen is None else seen
    if model in seen:
        return
    seen.add(model)
    for name, f in model.model_fields.items():
        caps = [m.max_length for m in f.metadata if isinstance(m, MaxLen)]
        if f.annotation is str and caps:
            stated = (f.json_schema_extra or {}).get("maxLength")
            yield model.__name__, name, caps[0], stated
        stack = [f.annotation]
        while stack:
            t = stack.pop()
            if isinstance(t, type) and issubclass(t, BaseModel):
                yield from _text_fields(t, seen)
            stack.extend(typing.get_args(t))


def test_every_v2_free_text_field_states_its_cap_and_allows_the_overrun():
    """A field added with a plain Field(max_length=...) would again fail a whole answer
    for one long sentence, and the desk would run on without that role, counted only as
    "failed". Each must tell the model its cap and enforce PROSE_OVERRUN times it."""
    fields = [f for k in V2_SCHEMAS for f in _text_fields(S.SCHEMAS[k])]
    assert len(fields) >= 18
    hard = [(m, n) for m, n, enforced, stated in fields
            if stated is None or enforced != int(stated * S.PROSE_OVERRUN)]
    assert not hard, f"capped at the stated length, not the overrun: {hard}"


def test_a_debate_turn_a_little_over_its_cap_is_kept_and_a_runaway_is_refused():
    """On the 2026-04-13 replay four debate turns a few characters over 2,000 were thrown
    away whole, and the trader and PM read the debate without them. The model is still
    told 2,000; only a turn past 3,000 is refused, since every later role reads it."""
    assert S.DebateTurn.model_json_schema()["properties"]["argument"]["maxLength"] == 2000
    assert len(S.DebateTurn(argument="x" * 2400, points=[]).argument) == 2400
    with pytest.raises(ValidationError, match="at most 3000 characters"):
        S.DebateTurn(argument="x" * 3001, points=[])


def test_the_v1_free_desks_rationale_keeps_its_hard_cap():
    """The free desk's four-window record was scored and posted with a hard 1,500 cap;
    loosening it after the fact would make a rerun a different desk under the same name."""
    S.FreeDecision(action="hold", weights=[], rationale="x" * 1500)
    with pytest.raises(ValidationError):
        S.FreeDecision(action="hold", weights=[], rationale="x" * 1501)


def test_gemini_is_sent_the_stated_cap_not_the_overrun():
    """Told the overrun, a model writes to it, and the overrun becomes the new cap with
    nothing left for a long day. And the answer cache is keyed on the schema's name, not
    what was sent: a changed request would replay answers given to the old one."""
    text = str(brains.gemini_schema(S.DebateTurn))
    assert "at most 2000 characters" in text and "3000" not in text
