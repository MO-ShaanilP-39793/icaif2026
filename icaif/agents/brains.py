"""The thing that answers a role's question: a rule, Claude, Gemma, or a cache of any.

**Model choice is a rule of the competition, not a preference.** The kit's
`docs/llm_and_external_data.md` allows Claude Opus 5, Sonnet 5 and Haiku 4.5 among
Anthropic's models. A newer Opus is not on that list, so `ClaudeBrain` refuses any
model outside it, and it never sends the server-side refusal fallback: that would
silently answer a round with a model the disclosures don't name. The same list names
"Gemma 3" among the open-weight models, so `GemmaBrain` takes Gemma 3 IDs only.

**A replay must be reproducible.** The organizers rerun what we submit, and a backtest
of an LLM that answers differently each time is a distribution, not a number.
`CachedBrain` keys every answer by a hash of the model, role, prompt and observation,
so a rerun reads the same answers and, offline, raises rather than calling out.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel

from icaif.agents.schemas import EventDecision, NameCall

ALLOWED_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
DEFAULT_MODEL = "claude-opus-5"
# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
PRICES = {"claude-opus-5": (5.0, 25.0, 0.50, 6.25),
          "claude-sonnet-5": (2.0, 10.0, 0.20, 2.50),
          "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25)}

# Gemma 3 on Amazon Bedrock, served on demand in ap-south-1 (checked 2026-10-02). The
# console labels the 27B "PT", but its ID is the instruction-tuned model and it answers
# as one.
GEMMA_MODELS = ("google.gemma-3-27b-it", "google.gemma-3-12b-it", "google.gemma-3-4b-it")
GEMMA_DEFAULT = "google.gemma-3-27b-it"
GEMMA_REGION = "ap-south-1"
# USD per million tokens, input and output: Bedrock's on-demand standard tier in
# ap-south-1, from the AWS Price List API on 2026-10-02.
GEMMA_PRICES = {"google.gemma-3-27b-it": (0.27, 0.45),
                "google.gemma-3-12b-it": (0.11, 0.34),
                "google.gemma-3-4b-it": (0.05, 0.09)}


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
    return EventDecision(calls=[NameCall(name=n, action="hold", reason="rule: hold")
                                for n in trigger_names])


class ClaudeBrain:
    """Structured-output calls to an allowed Claude model, with usage logged per call."""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high",
                 max_calls: Optional[int] = None, client=None,
                 max_cost: Optional[float] = None):
        """`max_cost` (USD) stops calling once spent: every later question falls back to
        the rule, so an overrun ends a replay's spend, not its run."""
        if model not in ALLOWED_MODELS:
            raise ValueError(f"{model!r} is not on the competition's allowed list {ALLOWED_MODELS}")
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


ANSWER_FORMAT = """\
How to answer. Reply with one JSON object and nothing else:
{{"analysis": "<your reasoning, at most 120 words, written before you decide>", "decision": <an object matching the schema below>}}

The decision's JSON Schema:
{schema}

Use exactly these field names and every required field, only the values the schema
allows, and numbers inside its bounds. Write null where the schema allows null and
nothing applies. Use only name codes the observation shows."""

REPAIR = """\
That reply could not be used: {error}
Reply again with the corrected JSON object only, in the same format."""


def answer_format(schema: type[BaseModel]) -> str:
    """The answer format and the role's schema, appended to its system prompt."""
    return ANSWER_FORMAT.format(schema=json.dumps(schema.model_json_schema(), separators=(",", ":")))


def read_reply(text: str, schema: type[BaseModel]) -> tuple[Optional[str], BaseModel]:
    """(analysis, decision) from a model's reply: its first JSON object, fences ignored.

    Gemma wraps JSON in a markdown fence and sometimes adds a line around it. A reader
    that wanted bare JSON would turn every answer into the rule's, and the shadow would
    read as an LLM that always agrees with the rule. A bare decision object, without the
    `analysis` wrapper, is accepted too: the question was still answered. Anything else
    raises ValueError (pydantic's ValidationError is one), which the caller repairs.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("the reply holds no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("the reply's JSON is not an object")
    if isinstance(obj.get("decision"), dict):
        analysis = obj.get("analysis")
        return (analysis if isinstance(analysis, str) else None), schema.model_validate(obj["decision"])
    return None, schema.model_validate(obj)


def _short(err: Exception, n: int = 600) -> str:
    """A validation error as a model can act on it: the failing fields, no doc links."""
    lines = [x for x in str(err).splitlines() if "errors.pydantic.dev" not in x]
    return " ".join(" ".join(lines).split())[:n]


class GemmaBrain:
    """A Gemma 3 model on Amazon Bedrock, held to the role's schema by checking its reply.

    Bedrock serves Gemma without the schema-constrained output Claude's API enforces, so
    the schema travels in the prompt (`answer_format`) and every reply is validated here.
    What would otherwise fail quietly:

    - **A reply outside the schema** (an exposure of 1.2, a missing field) gets one repair:
      the model sees its own reply and the validation error, inside the same time budget.
      A second miss is a `BrainError`, which the desk answers with the rule. Never a
      clamp: that would run a book the model did not choose, under its rationale.
    - **No thinking mode.** The reply's first field is a short `analysis`, written before
      the decision it argues for. It is kept in `records` with the raw reply, because the
      rules require every response disclosed in full, not just the parsed decision.
    - **Repeatability.** Temperature 0, so the same prompt gets as nearly the same answer
      as the service allows; `CachedBrain` makes a replay exact.
    - **Time.** One attempt per request and no SDK retries: botocore's default would
      retry a read that timed out, and the call would outlive the role's timeout and,
      live, the round's deadline.

    Credentials come from the environment (`AWS_PROFILE`, a role on the host); the region
    is ap-south-1 unless `ICAIF_BEDROCK_REGION` names another.
    """

    def __init__(self, model: str = GEMMA_DEFAULT, region: Optional[str] = None,
                 max_calls: Optional[int] = None, max_cost: Optional[float] = None,
                 client=None, repairs: int = 1, max_tokens: int = 2048):
        """`max_calls` and `max_cost` (USD) stop calling once spent, as for Claude: every
        later question falls back to the rule. A repair counts as a call."""
        if model not in GEMMA_MODELS:
            raise ValueError(f"{model!r} is not a Gemma 3 model on Bedrock {GEMMA_MODELS}")
        import threading

        self.model, self.max_calls, self.max_cost = model, max_calls, max_cost
        self.region = region or os.environ.get("ICAIF_BEDROCK_REGION") or GEMMA_REGION
        self.repairs, self.max_tokens = repairs, max_tokens
        self._client = client
        self._lock = threading.Lock()
        self.name = f"bedrock:{model}"
        self.records: list[dict] = []

    def _bedrock(self, timeout: float):
        if self._client is not None:
            return self._client
        import boto3
        from botocore.config import Config

        # A client per request, because the read timeout is the time this role has left.
        return boto3.client("bedrock-runtime", region_name=self.region,
                            config=Config(connect_timeout=5, read_timeout=max(1.0, timeout),
                                          retries={"total_max_attempts": 1}))

    def cost(self) -> float:
        p_in, p_out = GEMMA_PRICES[self.model]
        return sum(r["input_tokens"] * p_in + r["output_tokens"] * p_out for r in self.records) / 1e6

    def decide(self, role, system, payload, schema, timeout):
        deadline = time.perf_counter() + timeout
        prompt = f"{system}\n\n{answer_format(schema)}"
        messages = [{"role": "user", "content": [{"text": json.dumps(payload, sort_keys=True)}]}]
        problem = None
        for attempt in range(self.repairs + 1):
            with self._lock:
                if self.max_calls is not None and len(self.records) >= self.max_calls:
                    raise BrainError(f"call budget of {self.max_calls} spent")
                if self.max_cost is not None and self.cost() >= self.max_cost:
                    raise BrainError(f"dollar cap of ${self.max_cost:.2f} spent")
            left = deadline - time.perf_counter()
            if left < 2.0:
                raise BrainError(f"{left:.1f}s left, too little for a call"
                                 + (f"; the last reply was unusable: {problem}" if problem else ""))
            text, rec = self._call(role, prompt, messages, left, attempt)
            try:
                analysis, decision = read_reply(text, schema)
            except ValueError as err:
                problem = _short(err)
                rec["error"] = f"invalid: {problem}"
                messages = messages + [{"role": "assistant", "content": [{"text": text}]},
                                       {"role": "user", "content": [{"text": REPAIR.format(error=problem)}]}]
                continue
            rec["analysis"] = analysis
            return decision
        raise BrainError(f"no usable answer after {self.repairs} repair(s): {problem}")

    def _call(self, role, system, messages, timeout, attempt) -> tuple[str, dict]:
        rec = {"role": role, "model": self.model, "attempt": attempt, "request_id": None,
               "stop_reason": None, "latency_s": None, "error": None,
               "input_tokens": 0, "output_tokens": 0, "reply": None, "analysis": None}
        t0 = time.perf_counter()
        try:
            resp = self._bedrock(timeout).converse(
                modelId=self.model, system=[{"text": system}], messages=messages,
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0})
        except Exception as err:  # every SDK or network failure is a fallback, never a crash
            rec.update(latency_s=round(time.perf_counter() - t0, 2), error=repr(err))
            with self._lock:
                self.records.append(rec)
            raise BrainError(f"{type(err).__name__}: {err}") from err
        usage = resp.get("usage") or {}
        blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        text = "".join(b.get("text", "") for b in blocks)
        stop = resp.get("stopReason")
        rec.update(request_id=(resp.get("ResponseMetadata") or {}).get("RequestId"),
                   stop_reason=stop, latency_s=round(time.perf_counter() - t0, 2),
                   input_tokens=int(usage.get("inputTokens") or 0),
                   output_tokens=int(usage.get("outputTokens") or 0), reply=text)
        with self._lock:
            self.records.append(rec)
        if stop == "max_tokens":
            rec["error"] = "truncated"
            raise BrainError("answer truncated at max_tokens")
        if stop in ("content_filtered", "guardrail_intervened"):
            rec["error"] = f"refused ({stop})"
            raise BrainError(f"refused ({stop})")
        return text, rec


class CachedBrain:
    """Wraps a brain; answers from disk when the same question was asked before."""

    def __init__(self, inner, directory: Path, offline: bool = False):
        self.inner, self.dir, self.offline = inner, Path(directory), offline
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = f"cached({inner.name})"
        self.hits = self.misses = 0

    @staticmethod
    def key(brain_name, role, system, payload, schema) -> str:
        blob = json.dumps({"brain": brain_name, "role": role, "system": system,
                           "payload": payload, "schema": schema.__name__}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def decide(self, role, system, payload, schema, timeout):
        k = self.key(self.inner.name, role, system, payload, schema)
        path = self.dir / f"{k}.json"
        if path.exists():
            self.hits += 1
            return schema.model_validate(json.loads(path.read_text())["answer"])
        if self.offline:
            raise BrainError(f"no cached answer for {role} ({k[:12]}) and the cache is offline")
        self.misses += 1
        answer = self.inner.decide(role, system, payload, schema, timeout)
        path.write_text(json.dumps({"role": role, "brain": self.inner.name,
                                    "answer": answer.model_dump()}, indent=1))
        return answer
