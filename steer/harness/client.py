"""A thin client over the OpenAI Responses API, plus cost accounting and the ledger.

The harness only needs: send a list of input items and tool schemas, get back the
output items (to replay next turn), the function calls, any text, and token usage.
Conversations are stateless on the server (store=False); the full history, including
the opaque reasoning items, is sent again each turn so prompt caching applies.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from steer.model import ROOT

LEDGER = ROOT / "runs" / "ledger.jsonl"
LONG_CONTEXT_TOKENS = 272_000  # above this, OpenAI doubles input rates and charges 1.5x for output


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.output_tokens += other.output_tokens


@dataclass
class Reply:
    items: list                      # output items to append to the next request's input
    calls: list[dict] = field(default_factory=list)   # {"call_id", "name", "arguments"}
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    logged: list[dict] = field(default_factory=list)  # JSON-safe copy of the items, for the run log


def cost_usd(u: Usage, price: dict) -> float:
    """price: USD per million tokens for input, cached_input, cache_write, output."""
    ordinary = max(0, u.input_tokens - u.cached_tokens - u.cache_write_tokens)
    mult_in, mult_out = (2.0, 1.5) if u.input_tokens > LONG_CONTEXT_TOKENS else (1.0, 1.0)
    return (mult_in * (ordinary * price["input"] + u.cached_tokens * price["cached_input"]
                       + u.cache_write_tokens * price["cache_write"])
            + mult_out * u.output_tokens * price["output"]) / 1e6


def worst_case_usd(est_input_tokens: int, max_output_tokens: int, price: dict) -> float:
    """Upper bound for one request: every input token billed as a cache write, full output."""
    mult_in, mult_out = (2.0, 1.5) if est_input_tokens > LONG_CONTEXT_TOKENS else (1.0, 1.0)
    return (mult_in * est_input_tokens * max(price["input"], price["cache_write"])
            + mult_out * max_output_tokens * price["output"]) / 1e6


def ledger_add(stage: str, what: str, model: str, usd: float, usage: Usage) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage, "what": what,
                            "model": model, "usd": round(usd, 6), **usage.__dict__}) + "\n")


def ledger_total(stage: str) -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l and json.loads(l)["stage"] == stage)


class OpenAIClient:
    EFFORTS = ["max", "xhigh", "high", "medium"]

    def __init__(self, model: str, effort: str | None = None, max_output_tokens: int = 32_000):
        from openai import OpenAI

        load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env (see .env.example).")
        self.client = OpenAI()
        self.model, self.max_output_tokens = model, max_output_tokens
        self.efforts = [effort] if effort else list(self.EFFORTS)
        self.effort = self.efforts[0]

    def create(self, input_items: list, tools: list[dict] | None = None) -> Reply:
        from openai import APIConnectionError, APIStatusError, BadRequestError, RateLimitError

        delay = 5.0
        for attempt in range(8):
            try:
                kwargs = dict(model=self.model, input=input_items, store=False,
                              reasoning={"effort": self.effort}, max_output_tokens=self.max_output_tokens)
                if tools:
                    kwargs.update(tools=tools, parallel_tool_calls=False)
                resp = self.client.responses.create(**kwargs)
                return self._reply(resp)
            except BadRequestError as e:
                # an unsupported effort level: fall back to the next lower one
                if "effort" in str(e).lower() and self.effort in self.efforts[:-1]:
                    self.effort = self.efforts[self.efforts.index(self.effort) + 1]
                    continue
                raise
            except (APIConnectionError, RateLimitError) as e:
                err = e
            except APIStatusError as e:
                if e.status_code < 500:
                    raise
                err = e
            time.sleep(delay)
            delay = min(delay * 2, 120)
        raise RuntimeError(f"the API kept failing: {err}")

    @staticmethod
    def _reply(resp) -> Reply:
        calls, texts, logged = [], [], []
        for item in resp.output:
            d = item.model_dump(exclude_none=True)
            if d.get("type") == "reasoning":
                d.pop("encrypted_content", None)   # opaque and large; not useful in the log
            logged.append(d)
            if item.type == "function_call":
                calls.append({"call_id": item.call_id, "name": item.name, "arguments": item.arguments})
            elif item.type == "message":
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        texts.append(c.text)
        u = resp.usage
        det = getattr(u, "input_tokens_details", None)
        usage = Usage(input_tokens=u.input_tokens,
                      cached_tokens=getattr(det, "cached_tokens", 0) or 0,
                      cache_write_tokens=getattr(det, "cache_write_tokens", 0) or 0,
                      output_tokens=u.output_tokens)
        return Reply(items=list(resp.output), calls=calls, text="\n".join(texts), usage=usage, logged=logged)
