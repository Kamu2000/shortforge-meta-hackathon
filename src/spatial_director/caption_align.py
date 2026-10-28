"""Caption alignment: Voice Transcribe turns -> timed caption cues.

Why this module exists: the strict runtime ruling means captions are
built from Muse Voice Transcribe, whose timestamps are TURN-level
and whose text arrives with NO punctuation or capitalisation (both
verified live on clip A, 2026-10-29). Word-level timing therefore
cannot come from a model - it is reconstructed here, in three steps:

1. **Punctuation restoration (Spark 1.3).** One ledger-gated chat
   call per turn asks Muse Spark 1.3 to restore punctuation and
   capitalisation ONLY. The model's output is never trusted blindly:
   a deterministic check compares the normalised word sequence of
   the reply against the transcript's, and on ANY difference (a word
   added, dropped, "fixed", reordered) the raw transcript text is
   used instead and the fallback is recorded. The model may dress
   the words; it may not touch them.
2. **Phrase segmentation.** The (possibly restored) text splits at
   sentence enders, then at clause punctuation / a word cap, into
   caption-sized phrases.
3. **DSP timing.** The clip's audio energy profile (ffmpeg decode ->
   per-20ms RMS in dB) defines speech-active segments inside the
   turn. Words are laid uniformly over ACTIVE time (silences consume
   no words), phrase start/end come from their word spans, and each
   internal boundary snaps to the midpoint of a nearby detected
   silence - the standard caption practice of breaking in the pause.

Everything except the optional Spark call is deterministic signal
processing and string handling - no model at runtime beyond the two
Meta calls already named (Voice Transcribe upstream, Spark here).

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .spark_client import SparkClient

SILENCE_DB = -45.0       # below this a 20ms frame counts as silence
MIN_SILENCE_S = 0.15     # a gap must last this long to be a boundary
MAX_WORDS_PER_PHRASE = 8
SNAP_TOLERANCE_S = 0.35  # boundaries snap to silence within this
HOP_S = 0.02
SAMPLE_RATE = 16000

PUNCTUATION_SYSTEM_PROMPT = (
    "You restore punctuation and capitalisation in raw speech "
    "transcripts. Rules: add punctuation marks and capital letters "
    "ONLY. Do not add, remove, reorder, or change any word - "
    "including repeated words and fillers, which must stay exactly "
    "as written. Do not add quotation marks, brackets, or line "
    "breaks. Reply with the restored text only, nothing else."
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class CaptionAlignError(RuntimeError):
    """Any caption-alignment failure."""


@dataclass
class CaptionCue:
    text: str
    start_s: float
    end_s: float

    def to_srt(self, index: int) -> str:
        def stamp(t: float) -> str:
            ms = int(round(t * 1000))
            h, rem = divmod(ms, 3_600_000)
            m, rem = divmod(rem, 60_000)
            s, ms = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        return f"{index}\n{stamp(self.start_s)} --> {stamp(self.end_s)}\n{self.text}\n"


@dataclass
class EnergyProfile:
    """Per-hop RMS energy of a clip's audio, in dBFS."""

    times: np.ndarray  # hop start times, seconds
    db: np.ndarray     # RMS per hop, dBFS (silence floored at -90)
    hop_s: float = HOP_S


@dataclass
class TurnReport:
    turn_index: int
    used_fallback: bool = False
    spark_request_id: str | None = None
    spark_cost_usd: float | None = None
    note: str = ""


@dataclass
class CaptionReport:
    turns: list[TurnReport] = field(default_factory=list)
    cues: int = 0


# ---------------------------------------------------------------- energy


def energy_from_samples(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                        hop_s: float = HOP_S) -> EnergyProfile:
    """RMS-per-hop energy profile of a mono float sample array."""
    hop = max(1, int(round(sample_rate * hop_s)))
    n = len(samples) // hop
    if n == 0:
        raise CaptionAlignError("audio too short for an energy profile")
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-4))
    times = np.arange(n) * hop_s
    return EnergyProfile(times=times, db=db, hop_s=hop_s)


def load_energy(audio_path: str | Path, hop_s: float = HOP_S) -> EnergyProfile:
    """Decode a media file's audio via ffmpeg and profile its energy."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio_path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise CaptionAlignError(
            f"ffmpeg could not decode audio from {audio_path}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise CaptionAlignError(f"no audio samples decoded from {audio_path}")
    return energy_from_samples(samples, SAMPLE_RATE, hop_s)


def active_segments(profile: EnergyProfile, t0: float, t1: float,
                    *, thresh_db: float = SILENCE_DB,
                    min_silence_s: float = MIN_SILENCE_S
                    ) -> list[tuple[float, float]]:
    """Speech-active [start, end) segments inside [t0, t1].

    A frame is active when its RMS is above the silence threshold;
    active runs separated by less than ``min_silence_s`` of silence
    merge (a breath is not a boundary).
    """
    sel = (profile.times >= t0) & (profile.times < t1)
    times = profile.times[sel]
    active = profile.db[sel] > thresh_db
    segs: list[tuple[float, float]] = []
    start: float | None = None
    last_t = t0
    for t, is_active in zip(times, active):
        if is_active and start is None:
            start = float(t)
        elif not is_active and start is not None:
            segs.append((start, float(t)))
            start = None
        last_t = float(t)
    if start is not None:
        segs.append((start, last_t + profile.hop_s))
    merged: list[tuple[float, float]] = []
    for seg in segs:
        if merged and seg[0] - merged[-1][1] < min_silence_s:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return merged


# ---------------------------------------------------------------- text


def word_sequence(text: str) -> list[str]:
    """Normalised word tokens - the punctuation check's ground truth.

    Apostrophes are stripped (``it's`` and ``its`` are the same word
    dressed differently), as is all other punctuation and case.
    """
    return [t.replace("'", "") for t in _WORD_RE.findall(text.lower())]


def words_unchanged(raw: str, restored: str) -> bool:
    """True iff ``restored`` differs from ``raw`` only in dress.

    Case and punctuation are ignored; the word sequence (including
    repeated words, which ASR produces and a helpful model loves to
    "fix") must be identical.
    """
    return word_sequence(raw) == word_sequence(restored)


def split_phrases(text: str, *,
                  max_words: int = MAX_WORDS_PER_PHRASE) -> list[str]:
    """Split text into caption phrases: sentences, then clauses/caps."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())
                 if s.strip()]
    out: list[str] = []
    for sentence in sentences:
        out.extend(_split_long(sentence, max_words))
    return out


def _split_long(sentence: str, max_words: int) -> list[str]:
    if len(sentence.split()) <= max_words:
        return [sentence]
    chunks = [c.strip() for c in re.split(r"(?<=[,;:])\s+", sentence)
              if c.strip()]
    out: list[str] = []
    current = ""
    for chunk in chunks:
        words = chunk.split()
        if len(words) > max_words:
            if current:
                out.append(current)
                current = ""
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i:i + max_words]))
        elif current and len(current.split()) + len(words) > max_words:
            out.append(current)
            current = chunk
        else:
            current = (current + " " + chunk).strip()
    if current:
        out.append(current)
    return out


# ---------------------------------------------------------------- timing


def _active_to_wall(segments: list[tuple[float, float]], offset: float,
                    *, is_end: bool) -> float:
    """Map an offset into accumulated active time back to wall time.

    Boundary convention matters at segment edges: an offset exactly
    on a boundary is the END of the word that finished there (maps
    to the closing segment's end) and the START of the word that
    begins there (maps to the NEXT segment's start - the word is
    spoken after the pause, not at the end of the previous breath).
    """
    acc = 0.0
    for i, (start, end) in enumerate(segments):
        dur = end - start
        if offset < acc + dur or (is_end and offset <= acc + dur) \
                or i == len(segments) - 1:
            return start + min(offset - acc, dur)
        acc += dur
    return segments[-1][1] if segments else 0.0


def align_turn(phrases: list[str], t0: float, t1: float,
               profile: EnergyProfile | None, *,
               thresh_db: float = SILENCE_DB) -> list[CaptionCue]:
    """Time one turn's phrases over its speech-active spans.

    Words are laid uniformly over active time, so a phrase spanning
    a pause starts before it and ends after it, and boundaries fall
    where speech actually is. Each internal boundary then snaps to
    the midpoint of a detected silence gap when one is near (within
    SNAP_TOLERANCE_S) - breaks land in pauses, not mid-word.
    """
    words_per_phrase = [len(p.split()) for p in phrases]
    n_words = sum(words_per_phrase)
    if n_words == 0:
        return []
    if profile is not None:
        segments = active_segments(profile, t0, t1, thresh_db=thresh_db)
    else:
        segments = []
    if not segments:
        segments = [(t0, t1)]
    total_active = sum(e - s for s, e in segments)
    if total_active <= 0:
        segments = [(t0, t1)]
        total_active = t1 - t0

    cues: list[CaptionCue] = []
    word_idx = 0
    for phrase, count in zip(phrases, words_per_phrase):
        a_start = total_active * word_idx / n_words
        a_end = total_active * (word_idx + count) / n_words
        cues.append(CaptionCue(
            text=phrase,
            start_s=round(_active_to_wall(segments, a_start, is_end=False), 3),
            end_s=round(_active_to_wall(segments, a_end, is_end=True), 3),
        ))
        word_idx += count

    # Snap internal boundaries into nearby silence.
    gaps = [(segments[i][1], segments[i + 1][0])
            for i in range(len(segments) - 1)
            if segments[i + 1][0] - segments[i][1] >= MIN_SILENCE_S]
    for i in range(len(cues) - 1):
        boundary = cues[i].end_s
        for g0, g1 in gaps:
            mid = (g0 + g1) / 2.0
            if abs(mid - boundary) <= SNAP_TOLERANCE_S \
                    and cues[i].start_s + 0.1 < mid < cues[i + 1].end_s - 0.1:
                cues[i].end_s = round(mid, 3)
                cues[i + 1].start_s = round(mid, 3)
                break
    return cues


# ---------------------------------------------------------------- pipeline


def _turn_fields(turn: Any) -> tuple[str, float, float]:
    """(text, start_s, end_s) from a TranscriptTurn or a plain dict."""
    if isinstance(turn, dict):
        text = turn.get("text", turn.get("transcript", ""))
        start = turn.get("start_s", (turn.get("startMs") or 0) / 1000.0)
        end = turn.get("end_s", (turn.get("endMs") or 0) / 1000.0)
        return str(text), float(start), float(end)
    start_ms = getattr(turn, "start_ms", None)
    end_ms = getattr(turn, "end_ms", None)
    return (
        str(getattr(turn, "text", "")),
        (start_ms or 0) / 1000.0,
        (end_ms or 0) / 1000.0,
    )


def build_captions(turns: Sequence[Any],
                   profile: EnergyProfile | None, *,
                   spark: SparkClient | None = None,
                   purpose: str = "caption-punctuation",
                   ts: str = "") -> tuple[list[CaptionCue], CaptionReport]:
    """Full caption path: (optional) punctuation -> phrases -> cues.

    ``spark=None`` skips the punctuation pass entirely (raw VT text,
    every turn flagged as fallback). With a client, each turn gets
    one call; a reply that changes any word - or a failed call -
    falls back to the raw text for that turn, on the record.
    """
    cues: list[CaptionCue] = []
    report = CaptionReport()
    for i, turn in enumerate(turns):
        text, t0, t1 = _turn_fields(turn)
        trep = TurnReport(turn_index=i)
        restored = text
        if spark is not None and text.strip():
            try:
                result = spark.chat(
                    [
                        {"role": "system",
                         "content": PUNCTUATION_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=max(300, len(text) * 2),
                    purpose=purpose,
                    ts=ts,
                )
                trep.spark_request_id = result.request_id
                trep.spark_cost_usd = result.cost_usd
                candidate = result.text.strip()
                if words_unchanged(text, candidate):
                    restored = candidate
                else:
                    trep.used_fallback = True
                    trep.note = "spark reply changed the word sequence"
            except Exception as exc:  # a failed call must not kill captions
                trep.used_fallback = True
                trep.note = f"spark call failed: {exc}"[:200]
        else:
            trep.used_fallback = True
            trep.note = "no spark client; raw transcript text"
        phrases = split_phrases(restored)
        cues.extend(align_turn(phrases, t0, t1, profile))
        report.turns.append(trep)
    cues.sort(key=lambda c: c.start_s)
    report.cues = len(cues)
    return cues, report


def cues_to_srt(cues: Sequence[CaptionCue]) -> str:
    return "\n".join(cue.to_srt(i + 1) for i, cue in enumerate(cues))
