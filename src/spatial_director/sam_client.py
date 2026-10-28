"""SAM 3.1 client for the Meta Model API (Responses API only).

Verified facts this client is built on (dev.meta.ai docs, read
2026-10-26; see the team's verified-facts sheet):

- One model, ``sam-3.1``, for image + video segmentation via
  ``POST /v1/responses``. Video input is a ``file_id`` from
  ``POST /v1/files`` (purpose ``user_data``) or a ``video_url``.
- ONE concept per request, as a bare noun phrase in ``input_text``.
  A list ("bus, bicycle") returns zero matches; commands, questions
  and negation are unreliable.
- Default concept here is ``"person"``: it is the speaker prompt with
  practitioner evidence behind it. ``"speaking person"`` has zero
  evidence and is never used as a default.
- Video pricing: $0.20 per 1,000 frames, minimum 50 frames billed per
  request; at most 15,000 frames per request; at most 16 objects per
  frame. There is no fps parameter - cost is controlled by
  pre-encoding the clip at the frame rate we want masks for.
- Output is special-token text (one record per frame: object id + box
  token + mask token). Decoding lives in the mask module, not here.

Spend discipline: when a SpendLedger is attached, every request is
pre-flight checked against the ledger's cumulative cap using the
worst-case estimate (the full frame count), and the actual billed
cost is recorded from ``usage.video_frames_processed`` when the
response reports it.

The transport is injectable so the request/validation/ledger logic is
unit-testable offline. The wire format will be reconciled against the
first live response at the spike, and a redacted real response will be
committed as the canonical fixture at that point.

Written fresh in-repo on 2026-10-28. Stdlib only.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .spend_ledger import SpendLedger

MODEL_ID = "sam-3.1"
DEFAULT_CONCEPT = "person"
MAX_FRAMES_PER_REQUEST = 15_000
MIN_BILLED_FRAMES = 50
USD_PER_1000_FRAMES = 0.20
API_BASE = "https://api.meta.ai/v1"


class SamClientError(RuntimeError):
    """Any SAM request/validation failure."""


def estimate_video_cost_usd(n_frames: int) -> float:
    """Worst-case cost of one video request at ``n_frames`` frames."""
    billed = max(int(n_frames), MIN_BILLED_FRAMES)
    return round(billed * USD_PER_1000_FRAMES / 1000.0, 4)


def validate_concept(concept: str) -> str:
    """Enforce SAM's one-bare-noun-phrase rule; return the cleaned phrase."""
    phrase = " ".join(str(concept).split()).strip()
    if not phrase:
        raise SamClientError("concept must be a non-empty noun phrase")
    if "," in phrase:
        raise SamClientError(
            "SAM takes ONE concept per request; a comma list returns "
            "zero matches - split it into separate requests"
        )
    if any(ch in phrase for ch in "?!"):
        raise SamClientError(
            "SAM concepts are bare noun phrases; commands/questions "
            "are unreliable"
        )
    if len(phrase.split()) > 4:
        raise SamClientError(
            "concept is unexpectedly long for a noun phrase; keep it "
            "to a bare noun phrase (e.g. 'person')"
        )
    return phrase


@dataclass
class SegmentationResult:
    raw: dict[str, Any]
    billed_frames: int | None = None
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float | None:
        if self.billed_frames is None:
            return None
        return estimate_video_cost_usd(self.billed_frames)


# transport: (method, url, headers, body_bytes) -> (status, parsed_json)
Transport = Callable[[str, str, dict, bytes], tuple[int, dict]]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - live path
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SamClientError(f"SAM API HTTP {exc.code}: {detail}") from exc


class SamClient:
    """Minimal SAM 3.1 video-segmentation client with ledger gating."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_provider: Callable[[], str] | None = None,
        ledger: SpendLedger | None = None,
        transport: Transport | None = None,
    ):
        # The key is resolved lazily through a provider so no raw key
        # ever sits in this object's repr or in the repo. On the team's
        # machine the provider routes through the Secure Vault
        # connector pattern; judges can pass their own key directly.
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self.ledger = ledger
        self._transport = transport or _urllib_transport

    def _key(self) -> str:
        if self._api_key_provider is not None:
            return self._api_key_provider()
        if self._api_key:
            return self._api_key
        raise SamClientError(
            "no API key available: pass api_key or api_key_provider "
            "(on the team machine, the vault connector supplies it)"
        )

    def segment_video(
        self,
        *,
        file_id: str | None = None,
        video_url: str | None = None,
        concept: str = DEFAULT_CONCEPT,
        n_frames: int,
        purpose: str = "sam-video-segmentation",
        ts: str = "",
    ) -> SegmentationResult:
        """Segment one video for one concept, ledger-gated.

        ``n_frames`` is the frame count of the (pre-encoded) clip; it
        drives both the 15k limit check and the worst-case estimate.
        """
        phrase = validate_concept(concept)
        if (file_id is None) == (video_url is None):
            raise SamClientError("pass exactly one of file_id / video_url")
        if n_frames <= 0:
            raise SamClientError("n_frames must be positive")
        if n_frames > MAX_FRAMES_PER_REQUEST:
            raise SamClientError(
                f"{n_frames} frames exceeds the "
                f"{MAX_FRAMES_PER_REQUEST}-frame per-request limit; "
                "chunk the video"
            )

        estimate = estimate_video_cost_usd(n_frames)
        if self.ledger is not None:
            # Raises SpendCapError BEFORE anything is sent if the
            # cumulative cap would be exceeded.
            self.ledger.check_can_afford(estimate)

        video_part: dict[str, Any] = {"type": "input_video"}
        if file_id is not None:
            video_part["file_id"] = file_id
        else:
            video_part["video_url"] = video_url
        payload = {
            "model": MODEL_ID,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": phrase},
                        video_part,
                    ],
                }
            ],
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
            raise SamClientError(f"SAM API returned HTTP {status}: {raw}")

        usage = raw.get("usage") or {}
        billed = usage.get("video_frames_processed")
        result = SegmentationResult(
            raw=raw,
            billed_frames=int(billed) if billed is not None else None,
            request_id=raw.get("id"),
        )
        if self.ledger is not None and result.cost_usd is not None:
            self.ledger.record(
                ts=ts,
                model=MODEL_ID,
                cost_usd=result.cost_usd,
                purpose=purpose,
                request_id=result.request_id,
            )
        return result
