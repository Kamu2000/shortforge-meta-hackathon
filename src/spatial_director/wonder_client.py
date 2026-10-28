"""Muse Wonder client for the Meta Model API (Responses API).

Role in the pipeline: Wonder is the verifier's judge - the ONE
model call in the F1-F7 gate. After the deterministic checks run,
Wonder watches the full-resolution first 5 seconds of the ACTUAL
render, muted, and must be able to summarise the hook from the
visuals alone (the F5 mute-comprehension sub-check: a Short that
cannot be understood with the sound off fails the feed's majority
context). Deterministic checks arbitrate everything measurable;
Wonder judges only what geometry cannot - legibility and premise.

Verified facts this client is built on (dev.meta.ai docs, read
2026-10-26; the team's verified-facts sheet):

- Wonder (``muse-wonder-1.0``, Standard only on the API) takes
  text/image/video/audio inputs and can review an actual rendered
  clip, not just keyframes. Video goes through the Responses API
  with an ``input_video`` part referencing a ``file_id`` from
  ``POST /v1/files`` (purpose ``user_data``) - the same shape the
  SAM client uses.
- Pricing: $4.00 per 1M input tokens, $20.00 per 1M output ($0.20
  per 1M cached input when usage reports it).
- Reasoning models burn output budget on reasoning first (this
  team's first Wonder calibration call returned nothing at a tight
  cap, $0.245 of smoke), so the judge asks for a strict two-part
  reply format and an empty/unparseable reply RAISES instead of
  reading as "cannot summarise" - a transport failure must never
  masquerade as a verdict.

The reply contract is exact: ``SUMMARY: <text>`` on one line, or
``SUMMARY: CANNOT_DETERMINE`` when the premise cannot be determined
from the muted visuals. Parsing is therefore deterministic.

Spend discipline mirrors the sibling clients: a conservative
estimate is pre-flight checked against the SpendLedger cap before
upload, and the actual cost (from the response usage block) is
recorded after.

Written fresh in-repo on 2026-10-29. Stdlib only.
"""

from __future__ import annotations

import json
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .spend_ledger import SpendLedger

MODEL_ID = "muse-wonder-1.0"
API_BASE = "https://api.meta.ai/v1"
USD_PER_1M_INPUT = 4.00
USD_PER_1M_CACHED_INPUT = 0.20
USD_PER_1M_OUTPUT = 20.00
# Pre-flight estimate for one judge call. The team's overnight
# text-heavy Wonder calls cost $0.245-0.372; a 5s clip + a short
# prompt should land well under this, and the ledger records the
# actual from usage - the estimate only gates the cap.
ESTIMATE_USD = 0.40

JUDGE_PROMPT = (
    "Attached is a short vertical video clip with NO audio. Watch "
    "it and judge only what the visuals communicate. Reply in this "
    "exact format, one line:\n"
    "SUMMARY: <one or two sentences: what is this clip about, and "
    "what is the hook - the text premise a viewer would stop for?>\n"
    "If there is no readable on-screen text, or the premise cannot "
    "be determined from the visuals alone, reply exactly:\n"
    "SUMMARY: CANNOT_DETERMINE"
)


class WonderClientError(RuntimeError):
    """Any Wonder request/validation failure."""


def cost_from_usage(input_tokens: int, output_tokens: int,
                    cached_tokens: int = 0) -> float:
    """USD cost of a call from its token usage (Standard rates)."""
    cached = max(0, min(cached_tokens, input_tokens))
    fresh = input_tokens - cached
    return round(
        fresh * USD_PER_1M_INPUT / 1_000_000
        + cached * USD_PER_1M_CACHED_INPUT / 1_000_000
        + output_tokens * USD_PER_1M_OUTPUT / 1_000_000,
        4,
    )


@dataclass
class WonderJudgment:
    raw: dict[str, Any]
    summary: str = ""
    can_summarize: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    file_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return cost_from_usage(
            self.input_tokens, self.output_tokens,
            self.extra.get("cached_tokens", 0),
        )


# transport: (method, url, headers, body_bytes) -> (status, parsed_json)
Transport = Callable[[str, str, dict, bytes], tuple[int, dict]]
# upload transport: (url, headers, body_bytes) -> (status, parsed_json)
UploadTransport = Callable[[str, dict, bytes], tuple[int, dict]]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - live path
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise WonderClientError(f"Wonder API HTTP {exc.code}: {detail}") from exc


def extract_output_text(raw: dict[str, Any]) -> str:
    """Pull the output_text string out of a Responses payload."""
    for item in raw.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return str(content["text"])
    raise WonderClientError("no output_text content in Wonder response")


def parse_judgment(text: str) -> tuple[str, bool]:
    """Parse the strict reply: (summary, can_summarize).

    A reply with no SUMMARY marker, or whose summary is the
    CANNOT_DETERMINE sentinel, cannot summarise. Anything else is
    taken at face value - Wonder judged; the receipt records its
    words verbatim.
    """
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("SUMMARY:"):
            summary = line[len("SUMMARY:"):].strip()
            if not summary or "CANNOT_DETERMINE" in summary.upper():
                return "", False
            return summary, True
    return "", False


class WonderClient:
    """Minimal Wonder judge client with ledger gating."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_provider: Callable[[], str] | None = None,
        ledger: SpendLedger | None = None,
        transport: Transport | None = None,
        upload_transport: UploadTransport | None = None,
    ):
        # Key resolution mirrors the sibling clients: lazily through
        # a provider, so no raw key sits in this object or the repo.
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self.ledger = ledger
        self._transport = transport or _urllib_transport
        self._upload_transport = upload_transport or _urllib_transport

    def _key(self) -> str:
        if self._api_key_provider is not None:
            return self._api_key_provider()
        if self._api_key:
            return self._api_key
        raise WonderClientError(
            "no API key available: pass api_key or api_key_provider "
            "(on the team machine, the vault connector supplies it)"
        )

    def upload_file(self, path: str | Path) -> str:
        """Upload a media file for Responses input; returns file_id."""
        path = Path(path)
        boundary = "----wonder" + uuid.uuid4().hex
        data = path.read_bytes()
        body = b""
        body += f"--{boundary}\r\n".encode()
        body += (b'Content-Disposition: form-data; name="purpose"\r\n\r\n'
                 b"user_data\r\n")
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\n'
        ).encode()
        body += b"Content-Type: video/mp4\r\n\r\n" + data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        status, raw = self._upload_transport(
            f"{API_BASE}/files",
            {
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            body,
        )
        if status != 200 or "id" not in raw:
            raise WonderClientError(
                f"Wonder file upload failed (HTTP {status}): {raw}")
        return str(raw["id"])

    def judge_muted_hook(
        self,
        video_path: str | Path,
        *,
        max_output_tokens: int = 2000,
        purpose: str = "wonder-muted-hook-judge",
        ts: str = "",
    ) -> WonderJudgment:
        """Upload a MUTED clip and ask Wonder to summarise its hook.

        The caller guarantees the file has no audio track (the
        verifier strips it); Wonder judges visuals + text only.
        """
        if self.ledger is not None:
            # Raises SpendCapError BEFORE anything is sent if the
            # cumulative cap would be exceeded.
            self.ledger.check_can_afford(ESTIMATE_USD)
        file_id = self.upload_file(video_path)
        payload = {
            "model": MODEL_ID,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": JUDGE_PROMPT},
                        {"type": "input_video", "file_id": file_id},
                    ],
                }
            ],
            "max_output_tokens": max_output_tokens,
        }
        status, raw = self._transport(
            "POST",
            f"{API_BASE}/responses",
            {
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            json.dumps(payload).encode("utf-8"),
        )
        if status != 200:
            raise WonderClientError(
                f"Wonder API returned HTTP {status}: {raw}")
        text = extract_output_text(raw)
        if not text.strip():
            raise WonderClientError(
                "empty Wonder reply: the token budget was consumed "
                "by reasoning; raise max_output_tokens")
        summary, can = parse_judgment(text)
        usage = raw.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        judgment = WonderJudgment(
            raw=raw,
            summary=summary,
            can_summarize=can,
            input_tokens=usage.get("input_tokens",
                                   usage.get("prompt_tokens")),
            output_tokens=usage.get("output_tokens",
                                    usage.get("completion_tokens")),
            request_id=raw.get("id"),
            file_id=file_id,
            extra={"cached_tokens": int(details.get("cached_tokens") or 0)},
        )
        if self.ledger is not None:
            cost = judgment.cost_usd
            if cost is None:
                cost = ESTIMATE_USD
            self.ledger.record(
                ts=ts,
                model=MODEL_ID,
                cost_usd=cost,
                purpose=purpose,
                request_id=judgment.request_id,
            )
        return judgment
