"""The thing that answers a role's question: a rule, an LLM, or a cache of either.

**Model choice is a rule of the competition, not a preference.** The kit's
`docs/llm_and_external_data.md` allows Claude Opus 5, Sonnet 5 and Haiku 4.5 among
Anthropic's models. A newer Opus is not on that list, so `ClaudeBrain` refuses any
model outside it, and it never sends the server-side refusal fallback: that would
silently answer a round with a model the disclosures don't name.

The kit also allows "xAI Grok 4". Bedrock hosts no plain Grok 4, only 4.6 and 4.7. On
2026-10-05 the owner read the line as the Grok 4 family and `BedrockBrain` asked Grok
4.7; on 2026-10-06 the organizers ruled Grok 4.7 out. Read strictly, "Grok 4" excludes
4.6 as well, so no Grok model is allowed: `BedrockBrain` stays for replaying what was
cached, and `make` refuses it. Every Grok answer in output/agent is research only.

"Google Gemini 2.5 Pro / Flash" are named exactly, and `GeminiBrain` asks them through
the Gemini API (Anthropic models are blocked on our AWS account, which is billed through
a reseller). Their knowledge cutoff is January 2025 (Google's model pages), before every
evaluation window from Apr 2025 on.

**A replay must be reproducible.** The organizers rerun what we submit, and a backtest
of an LLM that answers differently each time is a distribution, not a number.
`CachedBrain` keys every answer by a hash of the model, role, prompt and observation,
so a rerun reads the same answers and, offline, raises rather than calling out.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel

from icaif.agents.schemas import EventDecision, NameCall

CLAUDE_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
# Our name -> Bedrock's cross-region inference profile. Bedrock serves xAI models only
# through a profile; the bare model id is refused for on-demand calls.
BEDROCK_MODELS = {"grok-4.7": "us.xai.grok-4.7"}
BEDROCK_REGION = "us-east-1"
GEMINI_MODELS = ("gemini-2.5-pro", "gemini-2.5-flash")
ALLOWED_MODELS = CLAUDE_MODELS + GEMINI_MODELS
DEFAULT_MODEL = "gemini-2.5-pro"
# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
# Grok 4.7 is Bedrock's us-east-1 standard tier (AWS Price List, 2026-10-05). xAI caches
# on its own and lists no cache-write price, so a write is charged as input, the safe side.
PRICES = {"claude-opus-5": (5.0, 25.0, 0.50, 6.25),
          "claude-sonnet-5": (2.0, 10.0, 0.20, 2.50),
          "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25),
          "grok-4.7": (2.20, 6.60, 0.55, 2.20),
          # Gemini API paid tier, prompts up to 200k tokens (2026-10-06); thinking tokens
          # are billed as output. Implicit caching has no write charge; charged as input.
          # Above 200k the price doubles, and no v2 prompt comes near it (~20k at most).
          "gemini-2.5-pro": (1.25, 10.00, 0.125, 1.25),
          "gemini-2.5-flash": (0.30, 2.50, 0.03, 0.30)}


class BrainError(RuntimeError):
    """An answer the desk cannot use: refused, truncated, timed out, over budget."""


class Brain(Protocol):
    name: str

    def decide(self, role: str, system: str, payload: dict, schema: type[BaseModel],
               timeout: float) -> BaseModel: ...


class RuleBrain:
    """Answers with the rule's proposal, which the desk computes and puts in the payload.

    Kept as a brain, not a shortcut, so the desk runs the same path for the rule as for
    Claude, and the rule desk reproducing the backtested quant candidate tests that path.
    """

    name = "rule"

    def decide(self, role, system, payload, schema, timeout):
        return schema.model_validate(payload["rule_proposal"])


def rule_event(trigger_names: list[str]) -> EventDecision:
    return EventDecision(calls=[NameCall(name=n, action="hold", fraction=None, cause=None,
                                         reason="rule: hold")
                                for n in trigger_names])


class ClaudeBrain:
    """Structured-output calls to an allowed Claude model, with usage logged per call."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "high",
                 max_calls: Optional[int] = None, client=None,
                 max_cost: Optional[float] = None):
        """`max_cost` (USD) stops calling once spent: every later question falls back to
        the rule, so an overrun ends a replay's spend, not its run."""
        if model not in CLAUDE_MODELS:
            raise ValueError(f"{model!r} is not on the competition's allowed list {CLAUDE_MODELS}")
        import threading

        self.model, self.effort, self.max_calls, self.max_cost = model, effort, max_calls, max_cost
        self._lock = threading.Lock()
        self._client = client
        self.name = f"claude:{model}:{effort}"
        self.records: list[dict] = []

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def cost(self) -> float:
        p_in, p_out, p_read, p_write = PRICES[self.model]
        return sum((r["input_tokens"] * p_in + r["output_tokens"] * p_out
                    + r["cache_read_input_tokens"] * p_read
                    + r["cache_creation_input_tokens"] * p_write) / 1e6 for r in self.records)

    def decide(self, role, system, payload, schema, timeout):
        with self._lock:
            if self.max_calls is not None and len(self.records) >= self.max_calls:
                raise BrainError(f"call budget of {self.max_calls} spent")
            if self.max_cost is not None and self.cost() >= self.max_cost:
                raise BrainError(f"dollar cap of ${self.max_cost:.2f} spent")
        kwargs = dict(
            model=self.model, max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": json.dumps(payload, sort_keys=True)}],
            output_format=schema,
        )
        if self.model != "claude-haiku-4-5":  # Haiku 4.5 takes neither effort nor adaptive
            kwargs.update(thinking={"type": "adaptive"}, output_config={"effort": self.effort})
        t0 = time.perf_counter()
        try:
            resp = self.client.with_options(timeout=timeout, max_retries=1).messages.parse(**kwargs)
        except Exception as err:  # every SDK failure is a fallback, never a crash
            with self._lock:
                self.records.append(self._record(role, None, time.perf_counter() - t0, repr(err)))
            raise BrainError(f"{type(err).__name__}: {err}") from err
        with self._lock:
            self.records.append(self._record(role, resp, time.perf_counter() - t0, None))
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise BrainError(f"refused ({cat})")
        if resp.stop_reason == "max_tokens":
            raise BrainError("answer truncated at max_tokens")
        if resp.parsed_output is None:
            raise BrainError("no parsed output")
        return resp.parsed_output

    def _record(self, role, resp, latency, error):
        u = getattr(resp, "usage", None)
        get = (lambda k: int(getattr(u, k, 0) or 0)) if u is not None else (lambda k: 0)
        return {"role": role, "model": self.model, "effort": self.effort,
                "request_id": getattr(resp, "_request_id", None),
                "stop_reason": getattr(resp, "stop_reason", None),
                "latency_s": round(latency, 2), "error": error,
                "input_tokens": get("input_tokens"), "output_tokens": get("output_tokens"),
                "cache_read_input_tokens": get("cache_read_input_tokens"),
                "cache_creation_input_tokens": get("cache_creation_input_tokens")}


class BedrockBrain:
    """Grok 4.7 through Bedrock's Converse API, with usage logged per call.

    The answer comes through Converse's `outputConfig` JSON-schema format, which, like
    Anthropic's `output_format`, constrains decoding to the schema. Not a forced tool
    call: Bedrock drops null values from a tool call's input, so every hold arrived
    without the `exposure`, `reason`, `fraction` and `cause` the schema requires as
    null, and 2 of the first 3 smoke-test answers fell back to the rule. A desk scored
    that way is the rule desk under Grok's name. The answer is still validated here,
    and one that fails is an error the desk falls back on: a repaired or coerced answer
    would be one the model did not give.

    Credentials come from boto3's chain (our AWS SSO session). An expired session fails
    every call, and each failure is a fallback to the rule, so a lapsed login looks
    like an LLM that agrees with the rule; `credentials_problem` lets callers say so.
    """

    WIRE = 2   # 1: bounded schema (broken); 2: bounds as words (`bedrock_schema`)

    def __init__(self, model: str = "grok-4.7", effort: str = "high",
                 max_calls: Optional[int] = None, client=None,
                 max_cost: Optional[float] = None):
        if model not in BEDROCK_MODELS:
            raise ValueError(f"{model!r} is not a Bedrock model we ask; known: {tuple(BEDROCK_MODELS)}")
        import threading

        self.model, self.effort, self.max_calls, self.max_cost = model, effort, max_calls, max_cost
        self._lock = threading.Lock()
        self._client = client
        # The cache keys answers by this name, not by the schema sent. Answers decoded
        # under the bounded schema (WIRE 1) are snapped to bounds; a new WIRE keeps a
        # rerun from replaying them as Grok's choices.
        self.name = f"bedrock:{model}:{effort}:wire{self.WIRE}"
        self.records: list[dict] = []

    cost = ClaudeBrain.cost

    def decide(self, role, system, payload, schema, timeout):
        with self._lock:
            if self.max_calls is not None and len(self.records) >= self.max_calls:
                raise BrainError(f"call budget of {self.max_calls} spent")
            if self.max_cost is not None and self.cost() >= self.max_cost:
                raise BrainError(f"dollar cap of ${self.max_cost:.2f} spent")
        kwargs = dict(
            modelId=BEDROCK_MODELS[self.model],
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": json.dumps(payload, sort_keys=True)}]}],
            outputConfig={"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {
                "name": schema.__name__, "schema": json.dumps(bedrock_schema(schema))}}}},
            inferenceConfig={"maxTokens": 16000},
            # Validated by xAI (an unknown effort is refused), so the setting is honoured,
            # not silently dropped as an unknown field would be.
            additionalModelRequestFields={"reasoning": {"effort": self.effort}},
        )
        t0 = time.perf_counter()
        try:
            resp = self._client_for(timeout).converse(**kwargs)
        except Exception as err:  # every SDK failure is a fallback, never a crash
            with self._lock:
                self.records.append(self._record(role, None, time.perf_counter() - t0, repr(err)))
            raise BrainError(f"{type(err).__name__}: {err}") from err
        with self._lock:
            self.records.append(self._record(role, resp, time.perf_counter() - t0, None))
        stop = resp.get("stopReason")
        if stop in ("content_filtered", "guardrail_intervened"):
            raise BrainError(f"refused ({stop})")
        if stop == "max_tokens":
            raise BrainError("answer truncated at max_tokens")
        text = "".join(b["text"] for b in resp.get("output", {}).get("message", {}).get("content", [])
                       if "text" in b)
        if not text.strip():
            raise BrainError("no answer text")
        try:
            return schema.model_validate_json(text)
        except ValueError as err:
            raise BrainError(f"answer fails {schema.__name__}: {err}") from err

    def _client_for(self, timeout):
        """A client whose read timeout is this call's share of the round.

        botocore fixes the timeout per client, and a call that outlives its round makes
        the round late, not just the answer. A client costs milliseconds against calls
        of tens of seconds, so each call gets its own. Tests inject a fake instead.
        """
        if self._client is not None:
            return self._client
        import boto3
        from botocore.config import Config

        return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION,
                            config=Config(read_timeout=max(int(timeout), 1), connect_timeout=10,
                                          retries={"max_attempts": 1, "mode": "standard"}))

    def _record(self, role, resp, latency, error):
        u = (resp or {}).get("usage") or {}
        return {"role": role, "model": self.model, "effort": self.effort,
                "request_id": (resp or {}).get("ResponseMetadata", {}).get("RequestId"),
                "stop_reason": (resp or {}).get("stopReason"),
                "latency_s": round(latency, 2), "error": error,
                "input_tokens": int(u.get("inputTokens", 0)), "output_tokens": int(u.get("outputTokens", 0)),
                "cache_read_input_tokens": int(u.get("cacheReadInputTokens", 0)),
                "cache_creation_input_tokens": int(u.get("cacheWriteInputTokens", 0))}


BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")


def bedrock_schema(schema: type[BaseModel]) -> dict:
    """The JSON schema Bedrock is sent: numeric bounds moved into the descriptions.

    Grok's constrained decoding on Bedrock snaps a bounded number to a bound: asked for
    x = 0.62 and y = 0.41 under [0.3, 0.95], it returned 0.3 and 0.3, and 0.62 and 0.41
    unbounded (2026-10-05). So the first paid free-desk run wrote every weight as 0.30,
    the cap, against its own rationale ("~94% across 19 names, inverse-vol tilted"), and
    the Strategist's exposure came back at a bound whatever it argued for. Each answer
    looked like a decision. Lengths and item counts decode correctly and stay; the
    bounds are still enforced, by pydantic, on what comes back.
    """
    def walk(node):
        if isinstance(node, dict):
            found = {k: node.pop(k) for k in BOUNDS if k in node}
            if found:
                words = {"minimum": ">=", "maximum": "<=", "exclusiveMinimum": ">",
                         "exclusiveMaximum": "<"}
                rule = " and ".join(f"{words[k]} {v}" for k, v in found.items())
                node["description"] = (node.get("description", "") + f" Must be {rule}.").strip()
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        return node

    return walk(schema.model_json_schema())


class GeminiBrain:
    """Gemini 2.5 Pro or Flash through the Gemini API's generateContent, usage logged per call.

    The answer is constrained by `responseJsonSchema`, which honours numeric bounds:
    asked for x = 0.62 and y = 0.41 under [0.3, 0.95], both models returned them
    exactly (2026-10-06), where Grok on Bedrock snapped both to 0.3. The answer is still
    validated here, and one that fails is an error the desk falls back on.

    **No tools, ever.** The request carries no `tools` field, so Google Search grounding
    cannot run: in a replay it would read the future, and live it would be a data
    source nobody logged. `effort` sets the thinking budget (`THINKING`): explicit
    numbers, so a rerun asks the same question at the same depth.

    The key comes from GEMINI_API_KEY or .env and travels in a header, never the URL,
    so no error message or log line can carry it. HTTPS goes through `net.ssl_context`.
    """

    WIRE = 1
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    THINKING = {"low": 1024, "medium": 4096, "high": 16384, "xhigh": 24576, "max": 32768}
    MAX_THINKING = {"gemini-2.5-pro": 32768, "gemini-2.5-flash": 24576}
    ANSWER_TOKENS = 16000   # room for the answer beyond the thinking budget
    REFUSED = ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
               "LANGUAGE", "IMAGE_SAFETY", "OTHER")

    def __init__(self, model: str = "gemini-2.5-pro", effort: str = "high",
                 max_calls: Optional[int] = None, client=None,
                 max_cost: Optional[float] = None):
        if model not in GEMINI_MODELS:
            raise ValueError(f"{model!r} is not a Gemini model the kit allows; known: {GEMINI_MODELS}")
        if effort not in self.THINKING:
            raise ValueError(f"effort {effort!r} is not one of {tuple(self.THINKING)}")
        import threading

        self.model, self.effort, self.max_calls, self.max_cost = model, effort, max_calls, max_cost
        self.budget = min(self.THINKING[effort], self.MAX_THINKING[model])
        self._lock = threading.Lock()
        self._client = client
        self.name = f"gemini:{model}:{effort}:wire{self.WIRE}"
        self.records: list[dict] = []

    cost = ClaudeBrain.cost

    def request(self, system: str, payload: dict, schema: type[BaseModel]) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, sort_keys=True)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": gemini_schema(schema),
                "maxOutputTokens": self.budget + self.ANSWER_TOKENS,
                "thinkingConfig": {"thinkingBudget": self.budget},
            },
        }

    def decide(self, role, system, payload, schema, timeout):
        with self._lock:
            if self.max_calls is not None and len(self.records) >= self.max_calls:
                raise BrainError(f"call budget of {self.max_calls} spent")
            if self.max_cost is not None and self.cost() >= self.max_cost:
                raise BrainError(f"dollar cap of ${self.max_cost:.2f} spent")
        t0 = time.perf_counter()
        try:
            resp = self._post(self.request(system, payload, schema), timeout)
            if resp.status_code != 200:
                raise BrainError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
        except Exception as err:  # every transport failure is a fallback, never a crash
            with self._lock:
                self.records.append(self._record(role, None, time.perf_counter() - t0, repr(err)))
            if isinstance(err, BrainError):
                raise
            raise BrainError(f"{type(err).__name__}: {err}") from err
        with self._lock:
            self.records.append(self._record(role, body, time.perf_counter() - t0, None))
        cands = body.get("candidates") or []
        if not cands:
            raise BrainError(f"refused ({(body.get('promptFeedback') or {}).get('blockReason', 'no candidates')})")
        stop = cands[0].get("finishReason")
        if stop == "MAX_TOKENS":
            raise BrainError("answer truncated at maxOutputTokens")
        if stop in self.REFUSED:
            raise BrainError(f"refused ({stop})")
        text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])
                       if not p.get("thought"))
        if not text.strip():
            raise BrainError("no answer text")
        try:
            return schema.model_validate_json(text)
        except ValueError as err:
            raise BrainError(f"answer fails {schema.__name__}: {err}") from err

    def _post(self, body: dict, timeout: float):
        if self._client is not None:
            return self._client.post(self.URL.format(model=self.model), json=body, timeout=timeout)
        import httpx

        from icaif import net

        with httpx.Client(verify=net.ssl_context(), headers={"x-goog-api-key": gemini_key()}) as c:
            return c.post(self.URL.format(model=self.model), json=body, timeout=timeout)

    def _record(self, role, body, latency, error):
        u = (body or {}).get("usageMetadata") or {}
        cached = int(u.get("cachedContentTokenCount", 0))
        return {"role": role, "model": self.model, "effort": self.effort,
                "request_id": (body or {}).get("responseId"),
                "stop_reason": (((body or {}).get("candidates") or [{}])[0]).get("finishReason"),
                "latency_s": round(latency, 2), "error": error,
                "input_tokens": int(u.get("promptTokenCount", 0)) - cached,
                "output_tokens": int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0)),
                "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}


LENGTHS = {"maxLength": "at most {} characters", "minLength": "at least {} characters",
           "maxItems": "at most {} items", "minItems": "at least {} items"}


def gemini_schema(schema: type[BaseModel]) -> dict:
    """The JSON schema Gemini is sent: length caps moved into the descriptions.

    Gemini compiles a schema into a decoding grammar and refuses one with "too many
    states": on 2026-10-06 it refused EventDecision, TradeList, PMDecision and
    TriggerDecision, each a list of up to 30 objects holding a capped reason (600
    characters), while FreeDecision, the same list without a text cap, passed. With
    both caps as words all thirteen answer schemas are accepted; with text caps alone
    the four still fail. The trader's and PM's answers are among them, so without this
    every v2 decision would be a fallback. Numeric bounds stay: Gemini honours them.
    The caps are still enforced, by pydantic, on what comes back, and an answer over
    one fails as any invalid answer does.
    """
    def walk(node):
        if isinstance(node, dict):
            found = [LENGTHS[k].format(node.pop(k)) for k in list(LENGTHS) if k in node]
            if found:
                node["description"] = (node.get("description", "") + " Must be "
                                       + " and ".join(found) + ".").strip()
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        return node

    return walk(schema.model_json_schema())


def gemini_key() -> str:
    """GEMINI_API_KEY from the environment, else from the repo's .env; raises if neither."""
    import os

    from icaif import data

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    env = data.ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise BrainError("GEMINI_API_KEY is not set (environment or .env)")
    return key


def make(model: str = DEFAULT_MODEL, effort: str = "high", *, max_calls: Optional[int] = None,
         max_cost: Optional[float] = None):
    """The paid brain for an allowed model: Gemini through its API, Claude through Anthropic.

    A model the kit does not allow is refused here, so no tool or live round can ask it,
    whatever its brain class still knows how to call (Grok, for cached replays).
    """
    if model not in ALLOWED_MODELS:
        raise ValueError(f"{model!r} is not an allowed model {ALLOWED_MODELS}")
    cls = GeminiBrain if model in GEMINI_MODELS else ClaudeBrain
    return cls(model, effort, max_calls=max_calls, max_cost=max_cost)


PAID = (ClaudeBrain, BedrockBrain, GeminiBrain)


def cost_by_role(brain) -> dict[str, float]:
    """USD per role from a paid brain's records: the v2 desk asks eleven roles on two
    tiers, and one total cannot say which of them the money went to."""
    p_in, p_out, p_read, p_write = PRICES[brain.model]
    out: dict[str, float] = {}
    for r in brain.records:
        out[r["role"]] = out.get(r["role"], 0.0) + (
            r["input_tokens"] * p_in + r["output_tokens"] * p_out
            + r["cache_read_input_tokens"] * p_read + r["cache_creation_input_tokens"] * p_write) / 1e6
    return out


def credentials_problem(model: str) -> Optional[str]:
    """Why a paid brain for `model` would fail every call, or None.

    Every failed call falls back to the rule without error, so a missing key or a
    lapsed AWS login would publish a shadow that "agrees with the rule" in every round.
    """
    import os

    if model in GEMINI_MODELS:
        try:
            gemini_key()
        except BrainError as err:
            return str(err)
        return None
    if model in BEDROCK_MODELS:
        try:
            import boto3

            creds = boto3.Session().get_credentials()
            if creds is None:
                return "no AWS credentials; run `aws sso login`"
            creds.get_frozen_credentials()  # an SSO token that has lapsed raises here
        except Exception as err:  # noqa: BLE001 - any failure here is a failed round later
            return f"AWS credentials unusable ({type(err).__name__}: {err}); run `aws sso login`"
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is unset"
    return None


class CachedBrain:
    """Wraps a brain; answers from disk when the same question was asked before."""

    def __init__(self, inner, directory: Path, offline: bool = False):
        import threading

        self.inner, self.dir, self.offline = inner, Path(directory), offline
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = f"cached({inner.name})"
        self.hits = self.misses = 0
        # The v2 analysts ask in parallel; a bare `+=` from threads can lose a count, and
        # the hits and misses are how a replay says what it paid for.
        self._lock = threading.Lock()

    @staticmethod
    def key(brain_name, role, system, payload, schema) -> str:
        blob = json.dumps({"brain": brain_name, "role": role, "system": system,
                           "payload": payload, "schema": schema.__name__}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def decide(self, role, system, payload, schema, timeout):
        k = self.key(self.inner.name, role, system, payload, schema)
        path = self.dir / f"{k}.json"
        if path.exists():
            with self._lock:
                self.hits += 1
            return schema.model_validate(json.loads(path.read_text())["answer"])
        if self.offline:
            raise BrainError(f"no cached answer for {role} ({k[:12]}) and the cache is offline")
        with self._lock:
            self.misses += 1
        answer = self.inner.decide(role, system, payload, schema, timeout)
        path.write_text(json.dumps({"role": role, "brain": self.inner.name,
                                    "answer": answer.model_dump()}, indent=1))
        return answer
