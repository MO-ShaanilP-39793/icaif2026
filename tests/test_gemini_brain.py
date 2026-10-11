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
    words, the caps must still hold: a rationale past the prose overrun (1.5 times the
    stated 1,500) is refused, not kept."""
    sent = brains.gemini_schema(S.TradeList)
    text = json.dumps(sent)
    assert "maxLength" not in text and "maxItems" not in text
    assert '"maximum": 0.3' in text      # numeric bounds stay: Gemini honours them
    assert "at most 1500 characters" in text
    assert _ask(_Gemini({**HOLD, "rationale": "x" * 2000})).rationale == "x" * 2000
    with pytest.raises(brains.BrainError, match="fails TradeList"):
        _ask(_Gemini({**HOLD, "rationale": "x" * 2251}))


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
    for model in ("grok-4.7", "grok-4.6", "gemini-3.5-flash"):   # 3.8 Flash allowed since 2026-10-11
        with pytest.raises(ValueError):
            brains.make(model, "high")
    assert "grok-4.7" not in brains.ALLOWED_MODELS
    assert isinstance(brains.make("gemini-2.5-flash", "medium"), brains.GeminiBrain)


@pytest.mark.parametrize("model", ["gemini-3.1-pro-preview", "gemini-3.8-flash"])
def test_a_gemini_3_model_is_asked_by_a_fixed_thinking_level_never_a_token_budget(model):
    """Gemini 3 reads a named level; a budget sent instead would ask at whatever depth the
    API chose, and a rerun could think differently from the run it repeats."""
    cfg = brains.GeminiBrain(model, "high").request("s", {"a": 1}, S.PMCheck)["generationConfig"]
    assert cfg["thinkingConfig"] == {"thinkingLevel": "high"}
    medium = brains.GeminiBrain(model, "medium").request("s", {}, S.PMCheck)["generationConfig"]
    assert medium["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert model in brains.ALLOWED_MODELS and model in brains.PRICES


def test_through_vertex_the_project_is_asked_and_the_api_key_is_never_read(monkeypatch):
    """On the VM the key's free tier refused every 3.1 Pro call, so the PM fell back in
    every rehearsal. With a project set, the call must go to that project on Vertex with
    the machine's token; a key sent along would reach a host with no use for it."""
    def no_key():
        raise AssertionError("the API key was read on the Vertex path")

    monkeypatch.setattr(brains, "gemini_key", no_key)
    brain = brains.GeminiBrain("gemini-3.1-pro-preview", "high", vertex_project="icaif2026")
    monkeypatch.setattr(brain, "_vertex_token", lambda: "tok")
    assert brain.url() == ("https://aiplatform.googleapis.com/v1/projects/icaif2026/locations/global"
                           "/publishers/google/models/gemini-3.1-pro-preview:generateContent")
    assert brain.headers() == {"Authorization": "Bearer tok"}


def test_without_a_project_the_gemini_api_gets_the_key_and_never_a_google_token(monkeypatch):
    monkeypatch.setattr(brains, "gemini_key", lambda: "k")
    monkeypatch.setattr(brains, "vertex_credentials", lambda: (_ for _ in ()).throw(AssertionError))
    brain = brains.GeminiBrain("gemini-3.1-pro-preview", "high", vertex_project="")
    assert brain.url().startswith("https://generativelanguage.googleapis.com/")
    assert brain.headers() == {"x-goog-api-key": "k"}


def test_the_transport_is_not_in_the_brain_name_so_cached_answers_carry_over():
    """A name that changed with the transport would re-ask, and re-pay, every cached
    question in every replay the moment a machine switched to Vertex."""
    via_key = brains.GeminiBrain("gemini-3.1-pro-preview", "high", vertex_project="")
    via_vertex = brains.GeminiBrain("gemini-3.1-pro-preview", "high", vertex_project="icaif2026")
    assert via_key.name == via_vertex.name


def test_a_machine_with_no_google_identity_is_reported_before_the_round(monkeypatch):
    """Without credentials every Vertex call fails and falls back to the rule silently; the
    preflight must name the problem instead of passing."""
    monkeypatch.setenv("GEMINI_VERTEX_PROJECT", "icaif2026")

    def none():
        raise brains.BrainError("no Google credentials for Vertex (DefaultCredentialsError: x)")

    monkeypatch.setattr(brains, "vertex_credentials", none)
    assert "no Google credentials" in brains.credentials_problem("gemini-3.1-pro-preview")
