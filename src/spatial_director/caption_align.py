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

SILENCE_DB = -45.0       # historical absolute floor; fallback only (below)
MIN_SILENCE_S = 0.15     # a gap must last this long to be a boundary

# Silence detection is RELATIVE to the clip's own speech level, not
# an absolute dBFS floor. The reason is the render path's loudness
# normalisation (see plan_audio_normalisation): a static gain of +G
# dB shifts every hop by G, so an absolute floor would reclassify
# room tone as speech and erase real pauses from the measurement -
# the verifier would stop measuring the shipped artifact truthfully.
# A threshold anchored to the speech level shifts 1:1 with any static
# gain, leaving the segmentation (and therefore every gap/onset
# measurement) exactly invariant. Calibration on the first Holmes
# run's windows: the rule below lands within 0.06s of the old -45 dB
# absolute segmentation on all three windows (gaps 1.46/0.74/1.00
# vs 1.46/0.74/1.06 measured absolutely), and is exactly invariant
# under +6/+12 dB shifts, both verified numerically before shipping.
SPEECH_SELECT_DEPTH_DB = 35.0  # hops within this of p99 = speech reference
GAP_DEPTH_DB = 20.0            # silence threshold = speech RMS - this

# Loudness normalisation (render path): one STATIC gain per window,
# sized from the window's speech RMS, capped, with a true-peak
# ceiling applied as a limiter. The gain makes quiet openings clear
# F1's absolute -32 dBFS 0-2s floor with margin on the measured
# Holmes windows (worst case: speech RMS -28.5, opening -34.0 ->
# +8.5 dB gain -> opening at -25.5). Because gap detection is
# relative (above), the gain cannot launder pauses: F5 measures the
# same gaps before and after.
TARGET_SPEECH_RMS_DB = -20.0
MAX_GAIN_DB = 12.0
PEAK_CEILING_DBFS = -0.5
MAX_WORDS_PER_PHRASE = 8
SNAP_TOLERANCE_S = 0.35  # boundaries snap to silence within this
HOP_S = 0.02
SAMPLE_RATE = 16000

# Spark is a reasoning model: the sibling clients' live lesson is
# that a tight max_tokens is consumed by reasoning and returns an
# empty reply. The restored text is short, but the budget must carry
# the reasoning too - 2,500 is the floor that worked for Spark Open.
PUNCTUATION_MAX_TOKENS = 2500

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


def speech_reference_rms_db(profile: EnergyProfile, t0: float,
                            t1: float) -> float | None:
    """Robust speech level of a span: RMS over its loudest hops.

    The reference set is the hops within SPEECH_SELECT_DEPTH_DB of
    the span's 99th-percentile hop - in any span containing speech,
    the top of the distribution is speech, whatever the duty cycle.
    Every statistic here is a pure function of the profile, so the
    result shifts exactly 1:1 under a static gain. None when the
    span selects no hops.
    """
    sel = (profile.times >= t0) & (profile.times < t1)
    db = profile.db[sel]
    if db.size == 0:
        return None
    top = float(np.percentile(db, 99))
    loud = db[db > top - SPEECH_SELECT_DEPTH_DB]
    if loud.size == 0:
        return None
    return float(10.0 * np.log10(np.mean(10.0 ** (loud / 10.0))))


def silence_threshold_db(profile: EnergyProfile, t0: float,
                         t1: float) -> float:
    """The silence threshold for a span: speech RMS - GAP_DEPTH_DB.

    Falls back to the historical absolute SILENCE_DB when the span
    is degenerate for a relative rule (flat dynamics - a constant
    tone, digital silence - where "relative to speech" has no
    meaning and the absolute floor classifies correctly anyway).
    """
    sel = (profile.times >= t0) & (profile.times < t1)
    db = profile.db[sel]
    if db.size == 0:
        return SILENCE_DB
    spread = float(np.percentile(db, 99) - np.percentile(db, 10))
    if spread < GAP_DEPTH_DB:
        return SILENCE_DB
    ref = speech_reference_rms_db(profile, t0, t1)
    if ref is None:
        return SILENCE_DB
    return ref - GAP_DEPTH_DB


@dataclass
class NormalisationPlan:
    """One window's loudness plan: a static gain and its evidence."""

    gain_db: float
    speech_rms_db: float | None
    peak_dbfs: float
    target_db: float = TARGET_SPEECH_RMS_DB


def plan_audio_normalisation(
    profile: EnergyProfile,
    peak_dbfs: float,
    *,
    target_db: float = TARGET_SPEECH_RMS_DB,
    max_gain_db: float = MAX_GAIN_DB,
) -> NormalisationPlan:
    """Size a static gain bringing the speech RMS to ``target_db``.

    Gain only ever goes UP (a loud window is already fine; the plan
    is 0 dB, never attenuation) and never past ``max_gain_db``.
    ``peak_dbfs`` is measured by the caller on the samples the gain
    will be applied to; the caller pairs the gain with a limiter at
    PEAK_CEILING_DBFS, which on the measured Holmes windows touches
    ~1 ms of transient peaks per window.
    """
    ref = speech_reference_rms_db(
        profile, float(profile.times[0]),
        float(profile.times[-1]) + profile.hop_s)
    if ref is None:
        return NormalisationPlan(0.0, None, peak_dbfs, target_db)
    gain = min(max(target_db - ref, 0.0), max_gain_db)
    return NormalisationPlan(round(gain, 2), round(ref, 2),
                             round(float(peak_dbfs), 2), target_db)


def active_segments(profile: EnergyProfile, t0: float, t1: float,
                    *, thresh_db: float | None = None,
                    min_silence_s: float = MIN_SILENCE_S
                    ) -> list[tuple[float, float]]:
    """Speech-active [start, end) segments inside [t0, t1].

    A frame is active when its RMS is above the silence threshold;
    active runs separated by less than ``min_silence_s`` of silence
    merge (a breath is not a boundary). The threshold defaults to
    the span's RELATIVE threshold (``silence_threshold_db``); pass
    an explicit dB value to pin it.
    """
    if thresh_db is None:
        thresh_db = silence_threshold_db(profile, t0, t1)
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


# Fix round 5: the phrase map above merges silences shorter than
# MIN_SILENCE_S (0.15s), which is right for caption LAYOUT but too
# coarse to bound an alignment slice: inside one merged segment a
# speaker breathes, lists, hesitates - sub-phrase dips of
# 50-150ms that carry no words. A forced-alignment slice bounded
# by those dips (first onset to last offset of the FINE map) is
# bounded by measured speech activity, never by a word-count
# proportion - the distinction the QM6 caption defect turned on
# (see word_timing's pace gate and pipeline._caption_word_timings).
FINE_MIN_SILENCE_S = 0.05


def speech_activity_span(
    profile: EnergyProfile, t0: float, t1: float, *,
    min_silence_s: float = FINE_MIN_SILENCE_S,
) -> tuple[float, float] | None:
    """(first onset, last offset) of speech activity in [t0, t1].

    Reads the FINE segment map (sub-phrase dips split segments),
    so the span hugs the speech a caption cue actually covers
    instead of the merged phrase block. None when the span holds
    no detectable speech activity at all.
    """
    segs = active_segments(profile, t0, t1,
                           min_silence_s=min_silence_s)
    if not segs:
        return None
    return (segs[0][0], segs[-1][1])


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
               thresh_db: float | None = None) -> list[CaptionCue]:
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

    # Snap cue boundaries onto speech onsets/offsets (refinement,
    # 2026-10-29). The uniform word layout can leave a phrase
    # boundary a beat off the actual edge of the speech - real
    # words are not uniform. Abutting cues share their boundary,
    # so a boundary moves JOINTLY: when a segment edge (a speech
    # onset or offset) lies within SNAP_TOLERANCE_S of the shared
    # boundary, both cues' edges move onto it - the caption cut
    # lands on the sound's edge, not near it. The turn's outer
    # edges snap the same way. Guards keep every cue ordered and
    # non-degenerate.
    if cues:
        first = cues[0]
        near = [s for s, _e in segments
                if abs(s - first.start_s) <= SNAP_TOLERANCE_S]
        if near:
            s0 = min(near, key=lambda s: abs(s - first.start_s))
            if t0 <= s0 < first.end_s - 0.1:
                first.start_s = round(s0, 3)
        last = cues[-1]
        near = [e for _s, e in segments
                if abs(e - last.end_s) <= SNAP_TOLERANCE_S]
        if near:
            e0 = min(near, key=lambda e: abs(e - last.end_s))
            if last.start_s + 0.1 < e0 <= t1:
                last.end_s = round(e0, 3)
        seg_edges = [x for seg in segments for x in seg]
        for i in range(len(cues) - 1):
            end_i, start_next = cues[i].end_s, cues[i + 1].start_s
            if abs(end_i - start_next) < 0.002:
                # Abutting cues share the boundary: move it jointly
                # onto the nearest speech edge.
                near = [x for x in seg_edges
                        if abs(x - end_i) <= SNAP_TOLERANCE_S]
                if not near:
                    continue
                e0 = min(near, key=lambda x: abs(x - end_i))
                if cues[i].start_s + 0.1 < e0 < cues[i + 1].end_s - 0.1:
                    cues[i].end_s = round(e0, 3)
                    cues[i + 1].start_s = round(e0, 3)
            else:
                # Separated cues (a pause sits between them): each
                # edge snaps independently, never crossing the other.
                near = [x for x in seg_edges
                        if abs(x - end_i) <= SNAP_TOLERANCE_S]
                if near:
                    e0 = min(near, key=lambda x: abs(x - end_i))
                    if cues[i].start_s + 0.1 < e0 <= start_next:
                        cues[i].end_s = round(e0, 3)
                near = [x for x in seg_edges
                        if abs(x - start_next) <= SNAP_TOLERANCE_S]
                if near:
                    s0 = min(near, key=lambda x: abs(x - start_next))
                    if cues[i].end_s <= s0 < cues[i + 1].end_s - 0.1:
                        cues[i + 1].start_s = round(s0, 3)
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
                    max_tokens=PUNCTUATION_MAX_TOKENS,
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


# ------------------------------------------------------------------
# Cue-sanity guard (fix round 2, Fix D, 2026-11-02).
#
# The Voice Transcribe turns occasionally carry a near-zero span
# for a full sentence (a turn timestamp glitch). build_captions
# then lays the turn's phrases over ~0 seconds: cues like 3 words
# in 0.06s (pipeline-run-ext-qm4 short-1 cue 14) or five cues in
# 0.2-0.3s (pipeline-run-ext-qh3 short-1 cues 0-4). DSP word timing
# tolerates them (it just subdivides silence-free spans); the MMS
# CTC aligner physically cannot - it needs at least one emission
# frame per token - so one rotten cue used to fail the WHOLE
# Short's MMS alignment and silently downgrade every cue to DSP,
# and graphic spans anchored to those cues fired at wrong times.
#
# The plausibility bound, from measurement, not taste:
# * CTC floor: the MMS aligner emits one frame per 20 ms of audio
#   (16 kHz, stride 320). A cue's alignment needs at least one
#   frame per character token plus one separator frame per word,
#   so duration >= (chars + words) * 0.02 s is a hard physical
#   floor. The measured failures sit far below it (0.06 s for a
#   3-word / 8-char cue needs >= 0.22 s).
# * Rate floor: 0.05 s/word = 20 words/second = 1200 wpm is ~3x
#   the fastest sustained human speech (~0.15 s/word); no real
#   phrase is spoken faster. Every cue in the banked corpus at or
#   below 0.02 s/word belongs to a compressed-turn artifact, and
#   the fastest plausible banked cues sit at >= ~0.09 s/word.
# A cue is degenerate iff its duration is below BOTH floors'
# maximum. Repair is deterministic and word-preserving: expand
# into the neighbouring gaps first (never past a neighbour's
# span), then merge into a neighbour (previous preferred), until
# the cue passes or nothing can absorb it - merges cascade. A cue
# that still fails is left in place and RECORDED unrepairable;
# the pipeline's timing source then honestly falls back for the
# Short (the guard never invents words and never moves a cue
# across another cue's span).
CUE_MIN_SECONDS_PER_WORD = 0.05
CUE_CTC_FRAME_S = 0.02


def cue_min_plausible_s(text: str) -> float:
    """The shortest span `text` can physically occupy."""
    words = text.split()
    if not words:
        return 0.0
    chars = sum(len(w) for w in words)
    return max(len(words) * CUE_MIN_SECONDS_PER_WORD,
               (chars + len(words)) * CUE_CTC_FRAME_S)


def is_degenerate_cue(cue: CaptionCue) -> bool:
    return (cue.end_s - cue.start_s) \
        < cue_min_plausible_s(cue.text) - 1e-9


def repair_cue_spans(
    cues: Sequence[CaptionCue], *, duration_s: float | None = None,
) -> tuple[list[CaptionCue], list[dict[str, Any]]]:
    """Repair degenerate cue spans before word timing.

    Returns (repaired cues, repair records). Each record:
    {cue_index (index in the ORIGINAL list), action:
    "expanded" | "merged_into_previous" | "merged_next" |
    "unrepairable", before: [start, end], after: [start, end],
    reason}. Word sequence over the whole list is preserved
    verbatim and no new overlaps are created: expansions stop at
    the neighbours' current spans, merges only union spans.
    """
    work = [CaptionCue(text=c.text, start_s=c.start_s, end_s=c.end_s)
            for c in cues]
    origin = list(range(len(work)))  # original index per work cue
    records: list[dict[str, Any]] = []
    i = 0
    while i < len(work):
        cue = work[i]
        if not is_degenerate_cue(cue):
            i += 1
            continue
        need = cue_min_plausible_s(cue.text)
        before = [round(cue.start_s, 3), round(cue.end_s, 3)]
        reason = (f"span {cue.end_s - cue.start_s:.3f}s < plausible "
                  f"minimum {need:.3f}s for "
                  f"{len(cue.text.split())} words")
        # 1) Expand into adjacent gaps (left first, then right).
        prev_end = work[i - 1].end_s if i > 0 else 0.0
        if i + 1 < len(work):
            next_start = work[i + 1].start_s
        elif duration_s is not None:
            next_start = duration_s
        else:
            next_start = cue.end_s
        deficit = need - (cue.end_s - cue.start_s)
        take_l = min(deficit, max(0.0, cue.start_s - prev_end))
        cue.start_s -= take_l
        deficit -= take_l
        take_r = min(deficit, max(0.0, next_start - cue.end_s))
        cue.end_s += take_r
        deficit -= take_r
        if deficit <= 1e-9:
            records.append({
                "cue_index": origin[i], "action": "expanded",
                "before": before,
                "after": [round(cue.start_s, 3), round(cue.end_s, 3)],
                "reason": reason})
            i += 1
            continue
        # 2) Merge. Previous neighbour preferred: it is already
        # repaired (indices < i are final), so one merge usually
        # ends the cascade.
        if i > 0:
            prev = work[i - 1]
            merged = CaptionCue(
                text=f"{prev.text} {cue.text}".strip(),
                start_s=min(prev.start_s, cue.start_s),
                end_s=max(prev.end_s, cue.end_s))
            records.append({
                "cue_index": origin[i],
                "action": "merged_into_previous",
                "before": before,
                "after": [round(merged.start_s, 3),
                          round(merged.end_s, 3)],
                "reason": reason})
            work[i - 1] = merged
            del work[i]
            del origin[i]
            i -= 1  # re-examine the merged cue
            continue
        if i + 1 < len(work):
            nxt = work[i + 1]
            merged = CaptionCue(
                text=f"{cue.text} {nxt.text}".strip(),
                start_s=min(cue.start_s, nxt.start_s),
                end_s=max(cue.end_s, nxt.end_s))
            records.append({
                "cue_index": origin[i], "action": "merged_next",
                "before": before,
                "after": [round(merged.start_s, 3),
                          round(merged.end_s, 3)],
                "reason": reason})
            work[i] = merged
            del work[i + 1]
            del origin[i + 1]
            continue  # re-examine the merged cue at i
        records.append({
            "cue_index": origin[i], "action": "unrepairable",
            "before": before, "after": before, "reason": reason})
        i += 1
    return work, records
