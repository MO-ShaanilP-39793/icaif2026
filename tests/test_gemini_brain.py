import json
import json as _json

import pytest

from icaif.agents import brains
from icaif.agents import schemas as S


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body

    @property
    def text(self):
        return _json.dumps(self.body)


class _Gemini:
    """Records each request and answers with `answer` (a dict, or a full response body)."""

    def __init__(self, answer=None, *, body=None, status=200, usage=None):
        self.answer, self.body, self.status, self.usage, self.sent = answer, body, status, usage, []

    def post(self, url, json, timeout):
        self.sent.append((url, json))
        if self.body is not None:
            return _Resp(self.body, self.status)
        return _Resp({"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": _json.dumps(self.answer)}]}}],
            "usageMetadata": self.usage or {"promptTokenCount": 10, "candidatesTokenCount": 5}}, self.status)


HOLD = {"adds": [], "cuts": [], "trims": [], "target_exposure": None, "rationale": "hold"}


def _ask(fake, schema=S.TradeList, model="gemini-2.5-pro", effort="high"):
    return brains.GeminiBrain(model, effort, client=fake).decide("pm", "system", {"x": 1}, schema, 60)


def test_gemini_is_never_sent_a_tool_so_it_cannot_search_the_web():
    """Google Search grounding is one field away. In a replay it would read how the window
    turned out; live it would be a data source nobody logged or disclosed."""
    fake = _Gemini(HOLD)
    _ask(fake)
    url, body = fake.sent[0]
    assert "tools" not in body and "toolConfig" not in body
    assert "search" not in json.dumps(body).lower()
    assert "key=" not in url    # the key travels in a header, never in a URL a log could hold


def test_the_schema_gemini_decodes_carries_no_length_caps_and_pydantic_still_enforces_them():
    """Gemini refuses the trader's and PM's schemas with "too many states" while a reason
    is capped inside a 30-item list, so every v2 decision would fall back. Moved into
    words, the caps must still hold: a 2,000-character rationale is refused, not kept."""
    sent = brains.gemini_schema(S.TradeList)
    text = json.dumps(sent)
    assert "maxLength" not in text and "maxItems" not in text
    assert '"maximum": 0.3' in text      # numeric bounds stay: Gemini honours them
    assert "at most 1500 characters" in text
    with pytest.raises(brains.BrainError, match="fails TradeList"):
        _ask(_Gemini({**HOLD, "rationale": "x" * 2000}))


@pytest.mark.parametrize("body,why", [
    ({"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}, "refused"),
    ({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]}, "truncated"),
    ({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}, "PROHIBITED_CONTENT"),
    ({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "hm", "thought": True}]}}]},
     "no answer text"),
])
def test_a_refused_truncated_or_empty_answer_is_an_error_the_desk_falls_back_on(body, why):
    """Each of these returns HTTP 200. Read as an answer, a refusal or a cut-off JSON would
    be parsed as a decision the model never finished; it must fall back, and be counted."""
    brain = brains.GeminiBrain("gemini-2.5-pro", "high", client=_Gemini(body=body))
    with pytest.raises(brains.BrainError, match=why):
        brain.decide("pm", "system", {}, S.TradeList, 60)
    assert len(brain.records) == 1


def test_an_http_error_is_a_fallback_and_is_recorded():
    brain = brains.GeminiBrain("gemini-2.5-pro", "high", client=_Gemini(body={"error": "x"}, status=429))
    with pytest.raises(brains.BrainError, match="HTTP 429"):
        brain.decide("pm", "system", {}, S.TradeList, 60)
    assert brain.records[0]["error"]


def test_thinking_is_billed_as_output():
    """Gemini bills thought tokens at the output price. Counted as free, a deep role's cost
    would read at a tenth of the bill, and a replay's estimate would never be checked."""
    fake = _Gemini(HOLD, usage={"promptTokenCount": 1000, "candidatesTokenCount": 100,
                                "thoughtsTokenCount": 900})
    brain = brains.GeminiBrain("gemini-2.5-pro", "high", client=fake)
    brain.decide("pm", "system", {}, S.TradeList, 60)
    assert brain.records[0]["output_tokens"] == 1000
    assert brain.cost() == pytest.approx((1000 * 1.25 + 1000 * 10.0) / 1e6)


def test_effort_sets_an_explicit_thinking_budget_within_each_models_limit():
    """A rerun must ask at the same depth: an explicit budget, not the model's dynamic
    default, and never above what Flash accepts (24,576)."""
    fake = _Gemini(HOLD)
    _ask(fake, model="gemini-2.5-flash", effort="max")
    cfg = fake.sent[0][1]["generationConfig"]
    assert cfg["thinkingConfig"]["thinkingBudget"] == 24576
    assert cfg["maxOutputTokens"] > cfg["thinkingConfig"]["thinkingBudget"]


def test_a_model_the_organizers_ruled_out_cannot_be_asked():
    """Grok 4.7 was ruled out on 2026-10-06, and "Grok 4" read strictly excludes 4.6 too.
    The Bedrock brain stays for cached replays, but no tool or round may ask it."""
    for model in ("grok-4.7", "grok-4.6", "gemini-3.8-flash"):
        with pytest.raises(ValueError):
            brains.make(model, "high")
    assert "grok-4.7" not in brains.ALLOWED_MODELS
    assert isinstance(brains.make("gemini-2.5-flash", "medium"), brains.GeminiBrain)
