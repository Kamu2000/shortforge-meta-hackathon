"""Muse Spark 1.3 client for the Meta Model API (Chat Completions).

Role in the pipeline (declared in the README): Spark Open (Baseten)
does the first-pass candidate sweep; Spark 1.3 on the Meta Model API
makes the final picks - and, in the caption path, restores the
punctuation/capitalisation that Voice Transcribe does not return.
This client is the generic ledger-gated chat wrapper for those
calls; task-specific prompts live with their callers.

Verified facts this client is built on (dev.meta.ai docs, read
2026-10-26; the team's verified-facts sheet):

- Chat Completions is one of the API's three protocol formats and is
  OpenAI-compatible: ``POST /v1/chat/completions`` with ``model`` +
  ``messages``; the reply text is ``choices[0].message.content`` and
  token counts are in ``usage`` (``prompt_tokens`` /
  ``completion_tokens``).
- Spark 1.3 Standard pricing: $1.25 per 1M input tokens, $4.25 per
  1M output tokens ($0.15 per 1M for cached input when the usage
  block reports it). The Contributor variant is never used - it
  grants Meta training rights on the content.
- Reasoning models spend completion tokens on reasoning first (the
  lesson two sibling clients learned live: a tight ``max_tokens``
  returns HTTP 200 with empty content), so an empty reply raises
  instead of reading as an empty answer.

Spend discipline mirrors ``sam_client``: with a SpendLedger attached,
a worst-case estimate (prompt size + full ``max_tokens`` at output
rates) is pre-flight checked against the cap, and the actual cost is
recorded from the response usage block.

Written fresh in-repo on 2026-10-29. Stdlib only.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .spend_ledger import SpendLedger

MODEL_ID = "muse-spark-1.3"
API_BASE = "https://api.meta.ai/v1"
USD_PER_1M_INPUT = 1.25
USD_PER_1M_CACHED_INPUT = 0.15
USD_PER_1M_OUTPUT = 4.25


class SparkClientError(RuntimeError):
    """Any Spark request/validation failure."""


def cost_from_usage(prompt_tokens: int, completion_tokens: int,
                    cached_tokens: int = 0) -> float:
    """USD cost of a call from its token usage (Standard rates)."""
    cached = max(0, min(cached_tokens, prompt_tokens))
    fresh = prompt_tokens - cached
    return round(
        fresh * USD_PER_1M_INPUT / 1_000_000
        + cached * USD_PER_1M_CACHED_INPUT / 1_000_000
        + completion_tokens * USD_PER_1M_OUTPUT / 1_000_000,
        4,
    )


def estimate_cost_usd(prompt_chars: int, max_tokens: int) -> float:
    """Worst-case estimate: ~4 chars/token in, full budget out."""
    return cost_from_usage(prompt_chars // 4 + 1, max_tokens)


@dataclass
class ChatResult:
    raw: dict[str, Any]
    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return cost_from_usage(
            self.prompt_tokens, self.completion_tokens,
            self.extra.get("cached_tokens", 0),
        )


# transport: (method, url, headers, body_bytes) -> (status, parsed_json)
Transport = Callable[[str, str, dict, bytes], tuple[int, dict]]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - live path
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SparkClientError(f"Spark API HTTP {exc.code}: {detail}") from exc


class SparkClient:
    """Minimal Spark 1.3 chat-completions client with ledger gating."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_provider: Callable[[], str] | None = None,
        ledger: SpendLedger | None = None,
        transport: Transport | None = None,
    ):
        # Key resolution mirrors the sibling clients: lazily through
        # a provider, so no raw key sits in this object or the repo.
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self.ledger = ledger
        self._transport = transport or _urllib_transport

    def _key(self) -> str:
        if self._api_key_provider is not None:
            return self._api_key_provider()
        if self._api_key:
            return self._api_key
        raise SparkClientError(
            "no API key available: pass api_key or api_key_provider "
            "(on the team machine, the vault connector supplies it)"
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        purpose: str = "spark-chat",
        ts: str = "",
    ) -> ChatResult:
        """One chat completion, ledger-gated; raises on empty replies."""
        if not messages:
            raise SparkClientError("chat needs at least one message")
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        estimate = estimate_cost_usd(prompt_chars, max_tokens)
        if self.ledger is not None:
            # Raises SpendCapError BEFORE anything is sent if the
            # cumulative cap would be exceeded.
            self.ledger.check_can_afford(estimate)

        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        status, raw = self._transport(
            "POST",
            f"{API_BASE}/chat/completions",
            {
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            json.dumps(payload).encode("utf-8"),
        )
        if status != 200:
            raise SparkClientError(f"Spark API returned HTTP {status}: {raw}")
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SparkClientError(
                f"unexpected chat response shape: {raw!r}") from exc
        if not text:
            # Verified failure class in the sibling clients: the
            # token budget was consumed by reasoning. Fail loud so
            # the caller raises the budget instead of proceeding on
            # an empty answer.
            raise SparkClientError(
                "empty chat reply: the token budget was consumed by "
                "reasoning; raise max_tokens"
            )
        usage = raw.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        result = ChatResult(
            raw=raw,
            text=str(text),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            request_id=raw.get("id"),
            extra={"cached_tokens": int(details.get("cached_tokens") or 0)},
        )
        if self.ledger is not None:
            cost = result.cost_usd
            if cost is None:
                cost = estimate
            self.ledger.record(
                ts=ts,
                model=MODEL_ID,
                cost_usd=cost,
                purpose=purpose,
                request_id=result.request_id,
            )
        return result
