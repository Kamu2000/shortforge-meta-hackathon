"""Muse Spark Open client (Baseten) - the candidate-moment sweep.

Role in the pipeline (declared in the README and the submission):
Spark Open, served by Baseten, performs the FIRST-PASS sweep of a
long transcript and proposes candidate moments; Muse Spark 1.3 on
the Meta Model API then makes the final picks. Removing this stage
removes the sweep - at Baseten's 2M TPM there is no throughput
equivalent in the stack - which is what makes it load-bearing for
the Best Build on Meta Open Weights entry (partner-verified usage).

Facts this client is built on:
- Endpoint https://inference.baseten.co/v1, OpenAI-style
  POST /chat/completions; model id ``meta-models/muse-spark-open-1026``
  (verified in the account's /v1/models list, 2026-10-28).
- The hackathon partner credit ($100 + up to 2M TPM) covers Muse
  Spark Open ONLY. Other Baseten models bill outside the credit, so
  the model id is a module constant, never a free parameter.
- This spend draws the Baseten partner credit, NOT the Meta $150 -
  it is therefore tracked in a separate partner usage log, never in
  spend-ledger.jsonl.

Every completed call is appended to the partner usage log (model,
token usage, latency, purpose, timestamp) when a log path is set:
that log is the evidence trail for partner verification during
judging.

The transport is injectable so request/validation/logic is
unit-testable offline. The key is resolved lazily through a provider
callback so no raw key sits in this repo; on the team machine the
provider routes through the Secure Vault connector pattern.

Written fresh in-repo on 2026-10-28. Stdlib only.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MODEL_ID = "meta-models/muse-spark-open-1026"
API_BASE = "https://inference.baseten.co/v1"

SWEEP_SYSTEM_PROMPT = (
    "You are the candidate-moment sweep for a video clipping "
    "pipeline. Given timestamped transcript turns from one long "
    "video, propose the strongest short-form moments: a claim, a "
    "conflict, a number, or a question a viewer would stop for. "
    "Reply with strict JSON only: a list of objects with keys "
    "start_s, end_s, score (0-1), reason (one line)."
)


class SparkOpenError(RuntimeError):
    """Any Spark Open request/validation failure."""


@dataclass
class Candidate:
    start_s: float
    end_s: float
    score: float
    reason: str = ""


@dataclass
class SweepResult:
    candidates: list[Candidate]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# transport: (method, url, headers, body_bytes) -> (status, parsed_json)
Transport = Callable[[str, str, dict, bytes], tuple[int, dict]]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - live path
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SparkOpenError(f"Baseten HTTP {exc.code}: {detail}") from exc


def _parse_candidates(text: str) -> list[Candidate]:
    """Parse the sweep's strict-JSON reply; fail loud on anything else."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SparkOpenError(f"sweep reply is not strict JSON: {exc}") from exc
    if not isinstance(data, list):
        raise SparkOpenError("sweep reply must be a JSON list of candidates")
    out: list[Candidate] = []
    for item in data:
        if not isinstance(item, dict):
            raise SparkOpenError("candidate entries must be JSON objects")
        try:
            start = float(item["start_s"])
            end = float(item["end_s"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SparkOpenError(f"candidate missing numeric fields: {item!r}") from exc
        if end <= start:
            raise SparkOpenError(f"candidate end_s must exceed start_s: {item!r}")
        out.append(
            Candidate(
                start_s=start,
                end_s=end,
                score=max(0.0, min(1.0, score)),
                reason=str(item.get("reason", "")),
            )
        )
    return out


class SparkOpenClient:
    """Minimal Spark Open chat client for the transcript sweep."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_provider: Callable[[], str] | None = None,
        transport: Transport | None = None,
        usage_log_path: str | Path | None = None,
    ):
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self._transport = transport or _urllib_transport
        self.usage_log_path = Path(usage_log_path) if usage_log_path else None

    def _key(self) -> str:
        if self._api_key_provider is not None:
            return self._api_key_provider()
        if self._api_key:
            return self._api_key
        raise SparkOpenError(
            "no API key available: pass api_key or api_key_provider "
            "(on the team machine, the vault connector supplies it)"
        )

    def _log_call(self, entry: dict[str, Any]) -> None:
        if self.usage_log_path is None:
            return
        with self.usage_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def sweep(
        self,
        turns: list[dict[str, Any]],
        *,
        purpose: str = "candidate-moment-sweep",
        ts: str = "",
        max_tokens: int = 4000,
    ) -> SweepResult:
        """Sweep timestamped transcript turns for candidate moments.

        ``turns`` items carry at least ``start_s``, ``end_s`` and
        ``text``. Returns parsed candidates; raises SparkOpenError on
        transport, HTTP, or strict-JSON failures - a malformed sweep
        must never silently degrade into "no candidates".
        """
        if not turns:
            raise SparkOpenError("sweep needs at least one transcript turn")
        lines = []
        for t in turns:
            try:
                lines.append(
                    f"[{float(t['start_s']):.1f}-{float(t['end_s']):.1f}] {t['text']}"
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SparkOpenError(f"turn missing fields: {t!r}") from exc
        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": SWEEP_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        started = time.monotonic()
        status, raw = self._transport(
            "POST",
            f"{API_BASE}/chat/completions",
            {
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            json.dumps(payload).encode("utf-8"),
        )
        latency = round(time.monotonic() - started, 3)
        if status != 200:
            raise SparkOpenError(f"Baseten returned HTTP {status}: {raw}")
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SparkOpenError(f"unexpected chat response shape: {raw!r}") from exc
        if not text:
            # Verified live 2026-10-28: Spark Open spends completion
            # tokens on reasoning first; a tight max_tokens returns
            # HTTP 200 with null content. Fail loud so the caller
            # raises the budget instead of reading "no candidates".
            raise SparkOpenError(
                "empty sweep reply: the token budget was consumed by "
                "reasoning; raise max_tokens"
            )
        candidates = _parse_candidates(text)
        usage = raw.get("usage") or {}
        result = SweepResult(
            candidates=candidates,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=latency,
            request_id=raw.get("id"),
        )
        self._log_call(
            {
                "ts": ts,
                "provider": "baseten",
                "model": MODEL_ID,
                "purpose": purpose,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": latency,
                "request_id": result.request_id,
                "candidates": len(candidates),
            }
        )
        return result
