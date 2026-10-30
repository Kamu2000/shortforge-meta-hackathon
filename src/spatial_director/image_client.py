"""Muse Image client for the Meta Model API (image generations).

Role in the pipeline (declared in the README): the context
graphics layer's image kind. When the graphics planner picks a
concrete visual object or concept for a spoken moment, this client
generates the picture; everything textual (headlines, numbers,
quotes) is rendered deterministically by the card renderer in
``graphics_layer`` and never passes through this model - image
models garble lettering, and a verifier-first product does not ship
numbers it cannot reproduce exactly.

Verified facts this client is built on (dev.meta.ai docs + the
team's verified-facts sheet, read 2026-10-26/28):

- ``muse-image-1.0`` on the OpenAI-compatible endpoint
  ``POST /v1/images/generations``; the reply carries the image in
  ``data[0]`` (base64 ``b64_json`` when requested; a ``url`` is the
  other documented shape). Output is WebP by default.
- Pricing is FLAT: $0.01 per image, and only successful images are
  billed - so the cost of a call is known exactly before it is
  sent, and a failed call is recorded nowhere.
- One probe call from this environment returned HTTP 200 with a
  valid WebP on 2026-10-28 (the Geographic Use Policy discrepancy
  around Muse Image in India is documented in the team's facts
  sheet; the call was served with no circumvention, exactly like
  every other API call this client makes).
- Rate limit for Muse Image is 150 requests per minute - far above
  anything the graphics layer requests (at most 3 graphics per
  Short, and the planner's bar makes image picks the minority).

Spend discipline mirrors ``spark_client`` / ``sam_client``: with a
SpendLedger attached, the flat $0.01 is pre-flight checked against
the cumulative cap before the request is sent, and recorded after a
successful decode. The Contributor model ids are never used (there
is no image Contributor id, and this client pins the model id in
code regardless - a caller cannot ask for another model).

Written fresh in-repo on 2026-10-30. Stdlib only.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .spend_ledger import SpendLedger

MODEL_ID = "muse-image-1.0"
API_BASE = "https://api.meta.ai/v1"
COST_PER_IMAGE_USD = 0.01


class ImageClientError(RuntimeError):
    """Any image request/validation failure."""


@dataclass
class ImageResult:
    raw: dict[str, Any]
    image_bytes: bytes = b""
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        """Flat per-image price; only successful images are billed."""
        return COST_PER_IMAGE_USD


# transport: (method, url, headers, body_bytes) -> (status, parsed_json)
Transport = Callable[[str, str, dict, bytes], tuple[int, dict]]
# fetch_bytes: (url) -> raw bytes, for the url-shaped reply
FetchBytes = Callable[[str], bytes]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - live path
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ImageClientError(f"Image API HTTP {exc.code}: {detail}") from exc


def _urllib_fetch(url: str) -> bytes:  # pragma: no cover - live path
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


class ImageClient:
    """Minimal Muse Image generations client with ledger gating."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_provider: Callable[[], str] | None = None,
        ledger: SpendLedger | None = None,
        transport: Transport | None = None,
        fetch_bytes: FetchBytes | None = None,
    ):
        # Key resolution mirrors the sibling clients: lazily through
        # a provider, so no raw key sits in this object or the repo.
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self.ledger = ledger
        self._transport = transport or _urllib_transport
        self._fetch_bytes = fetch_bytes or _urllib_fetch

    def _key(self) -> str:
        if self._api_key_provider is not None:
            return self._api_key_provider()
        if self._api_key:
            return self._api_key
        raise ImageClientError(
            "no API key available: pass api_key or api_key_provider "
            "(on the team machine, the vault connector supplies it)"
        )

    def generate(
        self,
        prompt: str,
        *,
        purpose: str = "muse-image-generation",
        ts: str = "",
    ) -> ImageResult:
        """Generate one image, ledger-gated; raises on any failure.

        The prompt is sent as written. Callers own prompt policy
        (the graphics layer forbids text/numbers in image prompts -
        see ``graphics_layer``); this client owns the wire, the
        flat-cost pre-flight, and the ledger record.
        """
        prompt = " ".join(str(prompt).split()).strip()
        if not prompt:
            raise ImageClientError("image prompt must be non-empty")
        if self.ledger is not None:
            # Raises SpendCapError BEFORE anything is sent if the
            # cumulative cap would be exceeded. The price is flat,
            # so the estimate is exact, not a worst case.
            self.ledger.check_can_afford(COST_PER_IMAGE_USD)

        payload = {
            "model": MODEL_ID,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        status, raw = self._transport(
            "POST",
            f"{API_BASE}/images/generations",
            {
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            json.dumps(payload).encode("utf-8"),
        )
        if status != 200:
            raise ImageClientError(f"Image API returned HTTP {status}: {raw}")
        image_bytes = self._extract_bytes(raw)
        result = ImageResult(
            raw=raw,
            image_bytes=image_bytes,
            request_id=raw.get("id"),
        )
        if self.ledger is not None:
            # Only successful images are billed, so the record is
            # written only now that an image actually exists.
            self.ledger.record(
                ts=ts,
                model=MODEL_ID,
                cost_usd=COST_PER_IMAGE_USD,
                purpose=purpose,
                request_id=result.request_id,
            )
        return result

    def _extract_bytes(self, raw: dict[str, Any]) -> bytes:
        data = raw.get("data") or []
        if not data or not isinstance(data[0], dict):
            raise ImageClientError(
                f"unexpected image response shape: {raw!r}")
        first = data[0]
        b64 = first.get("b64_json")
        if b64:
            try:
                return base64.b64decode(str(b64), validate=True)
            except Exception as exc:
                raise ImageClientError(
                    "image response carried undecodable b64_json") from exc
        url = first.get("url")
        if url:
            return self._fetch_bytes(str(url))
        raise ImageClientError(
            f"image response carried neither b64_json nor url: {raw!r}")
