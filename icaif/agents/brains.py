"""The thing that answers a role's question: a rule, Claude, or a cache of either.

**Model choice is a rule of the competition, not a preference.** The kit's
`docs/llm_and_external_data.md` allows Claude Opus 5, Sonnet 5 and Haiku 4.5 among
Anthropic's models. A newer Opus is not on that list, so `ClaudeBrain` refuses any
model outside it, and it never sends the server-side refusal fallback: that would
silently answer a round with a model the disclosures don't name.

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

ALLOWED_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
DEFAULT_MODEL = "claude-opus-5"
# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
PRICES = {"claude-opus-5": (5.0, 25.0, 0.50, 6.25),
          "claude-sonnet-5": (2.0, 10.0, 0.20, 2.50),
          "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25)}


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
