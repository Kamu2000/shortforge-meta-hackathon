"""Muse Voice Transcribe client for the Meta Model API (one-shot file ASR).

Why this module exists: the strict runtime-model ruling the team
adopted on 2026-10-29 means every model the shipped pipeline calls at
runtime is a Meta model. Transcription is the one place the original
design used a non-Meta helper (word-level timing); Voice Transcribe
is the Meta-native replacement. Its known trade, documented in the
team's verified-facts sheet and Meta's own cookbook: timestamps are
TURN-level, not word-level. The caption design therefore works from
diarized turns + non-model DSP alignment, and this client is the
piece that has to be verified before that design is built on.

Verified contract (dev.meta.ai docs + Meta's meta-model-cookbook
``transcribe_file.py``/``utils.py``, read 2026-10-29):

- ``POST /v1/asr/transcribe``, credential in the ``Authorization``
  header (unlike the streaming endpoint, which handshakes in-band).
- Multipart body with exactly two parts: ``request`` - a JSON part
  ``{"mode": ..., "model": "muse-voice-transcribe-1.0",
  "audioEncoding": "WAV"}`` (plus optional ``keywords`` /
  ``languageBias`` lists) - and ``audio`` - the WAV file itself.
- Input is narrow: mono 16-bit PCM WAV at 16 or 24 kHz only; caps of
  32 MB per body and 10 minutes of audio per request.
- Modes: ``PUSH_TO_TALK`` (default; caller delimits the turn),
  ``ENDPOINTING`` (model-detected boundaries), ``DIARIZATION``
  (endpointing + speaker labels). The server IGNORES an unrecognised
  mode instead of rejecting it, so the mode is validated here.
- Response is one JSON object: ``sessionId``, ``transcript``,
  ``audioDurationMs``, and - in the speech modes - a ``turns`` array
  whose entries carry ``startMs`` (and ``endMs``), ``transcript``,
  and, in DIARIZATION only, a bare-letter ``speaker`` ("A", not
  "speaker_a").
- Pricing: $0.18 per hour of audio. No per-request minimum is
  documented (unlike SAM's 50-frame floor), so cost is computed from
  the billed duration the response reports, falling back to the
  local WAV duration.

Spend discipline mirrors ``sam_client``: with a SpendLedger attached,
the estimate (local duration at the hourly rate) is pre-flight
checked against the cap, and the actual cost is recorded from the
response's ``audioDurationMs`` when present.

Written fresh in-repo on 2026-10-29. Stdlib only.
"""

from __future__ import annotations

import json
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from .spend_ledger import SpendLedger

MODEL_ID = "muse-voice-transcribe-1.0"
API_BASE = "https://api.meta.ai/v1"
USD_PER_HOUR = 0.18
MODES = ("PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION")
DEFAULT_MODE = "DIARIZATION"
MAX_BYTES = 32 * 1024 * 1024
MAX_DURATION_S = 600.0  # 10-minute per-request cap


class VoiceTranscribeError(RuntimeError):
    """Any Voice Transcribe request/validation failure."""


def estimate_cost_usd(duration_s: float) -> float:
    """Cost of transcribing ``duration_s`` seconds at $0.18/hour."""
    return round(float(duration_s) / 3600.0 * USD_PER_HOUR, 4)


def wav_duration_s(wav_bytes: bytes) -> float:
    """Duration of a WAV byte string, from its header (stdlib wave)."""
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / float(w.getframerate())


@dataclass
class TranscriptTurn:
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    raw: dict[str, Any]
    transcript: str = ""
    turns: list[TranscriptTurn] = field(default_factory=list)
    audio_duration_ms: int | None = None
    session_id: str | None = None

    @property
    def cost_usd(self) -> float | None:
        if self.audio_duration_ms is None:
            return None
        return estimate_cost_usd(self.audio_duration_ms / 1000.0)


# transport: (url, headers, body_bytes) -> (status, parsed_json)
Transport = Callable[[str, dict, bytes], tuple[int, dict]]


def _urllib_transport(url: str, headers: dict, body: bytes):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - live path
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise VoiceTranscribeError(
            f"Voice Transcribe API HTTP {exc.code}: {detail}"
        ) from exc


def build_multipart(
    request_json: dict[str, Any], audio_name: str, audio_bytes: bytes,
    boundary: str,
) -> tuple[bytes, str]:
    """Assemble the two-part multipart body the endpoint expects.

    The params travel as a JSON *part* named ``request`` (not as form
    fields); the WAV travels as a file part named ``audio``.
    """
    b = boundary.encode("ascii")
    body = b""
    body += b"--" + b + b"\r\n"
    body += b'Content-Disposition: form-data; name="request"\r\n'
    body += b"Content-Type: application/json\r\n\r\n"
    body += json.dumps(request_json).encode("utf-8") + b"\r\n"
    body += b"--" + b + b"\r\n"
    body += (
        f'Content-Disposition: form-data; name="audio"; '
        f'filename="{audio_name}"\r\n'
    ).encode("utf-8")
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += audio_bytes + b"\r\n"
    body += b"--" + b + b"--\r\n"
    return body, f"multipart/form-data; boundary={boundary}"


def parse_response(raw: dict[str, Any]) -> TranscriptionResult:
    """Parse the one-shot response defensively.

    ``turns`` is absent outside the speech modes and ``speaker`` is
    present only in DIARIZATION, so every field is read optionally -
    the same defensive posture Meta's own cookbook CLI takes.
    """
    turns: list[TranscriptTurn] = []
    for t in raw.get("turns") or []:
        turns.append(
            TranscriptTurn(
                text=str(t.get("transcript", "")),
                start_ms=t.get("startMs"),
                end_ms=t.get("endMs"),
                speaker=t.get("speaker"),
            )
        )
    dur = raw.get("audioDurationMs")
    return TranscriptionResult(
        raw=raw,
        transcript=str(raw.get("transcript", "")),
        turns=turns,
        audio_duration_ms=int(dur) if dur is not None else None,
        session_id=raw.get("sessionId"),
    )


class VoiceTranscribeClient:
    """Minimal Voice Transcribe file client with ledger gating."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_provider: Callable[[], str] | None = None,
        ledger: SpendLedger | None = None,
        transport: Transport | None = None,
    ):
        # Key resolution mirrors SamClient: lazily through a provider,
        # so no raw key ever sits in this object or in the repo.
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self.ledger = ledger
        self._transport = transport or _urllib_transport

    def _key(self) -> str:
        if self._api_key_provider is not None:
            return self._api_key_provider()
        if self._api_key:
            return self._api_key
        raise VoiceTranscribeError(
            "no API key available: pass api_key or api_key_provider "
            "(on the team machine, the vault connector supplies it)"
        )

    def transcribe_file(
        self,
        wav_path: str | Path,
        *,
        mode: str = DEFAULT_MODE,
        keywords: list[str] | None = None,
        language_bias: list[str] | None = None,
        purpose: str = "voice-transcribe",
        ts: str = "",
    ) -> TranscriptionResult:
        """Transcribe one WAV file in a single request, ledger-gated.

        The WAV must be mono 16-bit PCM at 16 or 24 kHz (the endpoint
        accepts nothing else); duration/size caps are enforced here
        before anything is sent.
        """
        if mode not in MODES:
            # The server silently ignores an unrecognised mode rather
            # than rejecting it - a stale name would transcribe in the
            # wrong mode and look identical to success.
            raise VoiceTranscribeError(
                f"mode must be one of {MODES}, got {mode!r}"
            )
        path = Path(wav_path)
        audio_bytes = path.read_bytes()
        if len(audio_bytes) > MAX_BYTES:
            raise VoiceTranscribeError(
                f"{path.name} is {len(audio_bytes) / 1024 / 1024:.1f} MB; "
                "the per-request limit is 32 MB - split the audio"
            )
        duration_s = wav_duration_s(audio_bytes)
        if duration_s > MAX_DURATION_S:
            raise VoiceTranscribeError(
                f"{path.name} is {duration_s:.0f}s; the per-request cap "
                "is 10 minutes - split the audio"
            )

        estimate = estimate_cost_usd(duration_s)
        if self.ledger is not None:
            # Raises SpendCapError BEFORE anything is sent if the
            # cumulative cap would be exceeded.
            self.ledger.check_can_afford(estimate)

        request_json: dict[str, Any] = {
            "mode": mode,
            "model": MODEL_ID,
            "audioEncoding": "WAV",
        }
        if keywords:
            request_json["keywords"] = list(keywords)
        if language_bias:
            request_json["languageBias"] = list(language_bias)

        boundary = "----vt" + uuid.uuid4().hex
        body, content_type = build_multipart(
            request_json, path.name, audio_bytes, boundary
        )
        status, raw = self._transport(
            f"{API_BASE}/asr/transcribe",
            {
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": content_type,
            },
            body,
        )
        if status != 200:
            raise VoiceTranscribeError(
                f"Voice Transcribe API returned HTTP {status}: {raw}"
            )

        result = parse_response(raw)
        if self.ledger is not None:
            cost = result.cost_usd
            if cost is None:
                cost = estimate
            self.ledger.record(
                ts=ts,
                model=MODEL_ID,
                cost_usd=cost,
                purpose=purpose,
                request_id=result.session_id,
            )
        return result
