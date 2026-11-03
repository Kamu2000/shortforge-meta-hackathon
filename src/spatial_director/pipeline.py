"""The pipeline: one long video in, up to N verified Shorts out.

Stages (the converged design, assembled end to end):

    ingest (ffprobe)
      -> Voice Transcribe (diarized turns; chunked at the API's
         10-minute cap)
      -> Spark Open sweep (Baseten; candidate moments, partner log)
      -> per selected moment:
           cut window -> SAM 3.1 (cached per window; the response
           is saved next to the Short and re-used on re-runs)
           -> mask_decode (primary speaker + finish stack)
           -> captions (Spark 1.3 punctuation under the word guard
              + DSP phrase timing)
           -> hook line (Spark 1.3, one call, deterministic
              fallbacks)
           -> layout (hook band + caption block from head geometry)
           -> composite render (draft 1)
           -> verifier F1-F9 + Wonder judge (receipt 1)
           -> rescue loop if draft 1 failed (ONE menu repair,
              re-verify, chained receipt)
      -> per-Short final MP4 + chained receipt, and a run summary
         (per-Short cost from ledger deltas, total cost, verdicts,
         door-to-door seconds).

Every paid call goes through the repo clients with the SpendLedger
attached; the summary's costs are ledger deltas measured around each
stage, never estimates. SAM is bought once per window - the rescue
loop works from the cached response, by construction.

CLI: ``python -m spatial_director.pipeline <video> --out <dir>
[--max-shorts N]``. On the team machine the clients resolve their
credentials through the Secure Vault surrogate pattern (the same
wiring as every live proof in the BUILDLOG); API keys can also be
supplied via the SPATIAL_DIRECTOR_META_API_KEY /
SPATIAL_DIRECTOR_BASETEN_API_KEY environment variables.

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .caption_align import (
    PEAK_CEILING_DBFS,
    build_captions,
    load_energy,
    plan_audio_normalisation,
    repair_cue_spans,
    speech_activity_span,
)
from .compositor import (
    BACKDROP_FILL,
    BACKDROP_GRADE_BRIGHTNESS,
    BACKDROP_GRADE_SATURATION,
    BACKDROP_SOURCE_MINUS_SPEAKER,
    BACKDROP_STUDIO,
    DEFAULT_BACKDROP,
    DEFAULT_CANVAS,
    read_video_frames,
    studio_seed,
    video_fps,
)
from .graphics_layer import build_graphics, plan_layout
from .image_client import ImageClient
from .layout_law import (
    CAPTION_CHUNK_MAX_WORDS,
    CAPTION_POP_S,
    EFFECT_OCCLUSION_BAND,
    SAFE_Y0,
    SAFE_Y1,
    apply_speaker_emphasis,
    apply_speaker_reframe,
    apply_stage_zoom,
    effect_note,
    hook_accent_index,
    hook_word_layers,
    plan_captions,
    plan_hook_effect,
    plan_speaker_emphasis,
    plan_speaker_reframe,
    plan_stage_zoom,
    render_hook_layer,
    to_canvas_alpha,
)
from .geometry_bundle import (
    DRAFT2_BUNDLE_REPAIRS,
    alpha_recipe_dict,
    draft_bundle_dict,
    hook_to_dict,
    write_bundle,
)
from .mask_decode import finished_alpha, parse_sam_response
from .rescue_loop import (
    RenderArtifacts,
    backdrop_entries,
    graphics_entries,
    render_composite_with_captions,
    rescue,
)
from .sam_client import API_BASE as META_API_BASE
from .sam_client import SamClient
from .sfx import mix_to_file, plan_sfx
from .spark_client import SparkClient
from .spark_open_client import Candidate, SparkOpenClient
from .spend_ledger import SpendLedger
from .verifier import HookSpec, Receipt, VerifyInputs, _content_words, verify
from .voice_transcribe import TranscriptTurn, VoiceTranscribeClient
from .word_timing import (
    MMS_SAMPLE_RATE,
    TIMING_SOURCES,
    WordTiming,
    WordTimingError,
    load_samples,
    pace_gate_failure,
    snap_word_timings,
    word_timings,
)
from .wonder_client import WonderClient

# ---------------------------------------------------------------- constants

MIN_WINDOW_S = 14.0    # a Short shorter than this is not a Short
MAX_WINDOW_S = 22.0    # memory ceiling for the finish stack at 30fps
HOOK_WINDOW_S = 6.0
# Word-timing source for caption sweeps (extension Stage 4):
# the DSP floor is the code default. CAPTION_TIMING_SOURCE in the
# environment REQUESTS a source ("dsp" | "mms"); an unrecognised
# value falls back to "dsp" rather than failing the run over a
# typo. The request is not the label: _caption_word_timings
# returns the EFFECTIVE source that actually produced a Short's
# timings, and that is what captions.json and every receipt
# carry - an MMS request the aligner cannot serve falls back to
# DSP for the whole Short, labelled "dsp", never "mms".
CAPTION_TIMING_SOURCE = os.environ.get(
    "CAPTION_TIMING_SOURCE", "dsp").strip().lower()
if CAPTION_TIMING_SOURCE not in TIMING_SOURCES:
    CAPTION_TIMING_SOURCE = "dsp"
# Canvas backdrop mode (fix round 3, 2026-11-02): the
# studio_procedural canvas is the code default - it carries no
# source pixels, so the ghost the founder rejected cannot
# exist. CANVAS_BACKDROP in the environment may select a
# surviving source-derived treatment ("source_minus_speaker" |
# "treated" | "brand" | "naive"); an unrecognised value falls
# back to the default rather than failing the run over a typo.
# The EFFECTIVE mode is declared per Short on the receipt's
# backdrop dict, together with the deterministic studio seed
# when the studio mode renders.
CANVAS_BACKDROP_MODE = os.environ.get(
    "CANVAS_BACKDROP", DEFAULT_BACKDROP).strip().lower()
if CANVAS_BACKDROP_MODE not in (
        BACKDROP_STUDIO, BACKDROP_SOURCE_MINUS_SPEAKER,
        "treated", "brand", "naive"):
    CANVAS_BACKDROP_MODE = DEFAULT_BACKDROP
VT_CHUNK_S = 540.0     # under the API's 600s / 32MB per-request caps
HOOK_MAX_TOKENS = 2500  # Spark reasons first; tight budgets return empty

HOOK_SYSTEM_PROMPT = (
    "You write ONE hook line for a short vertical video clipped "
    "from a longer talk. Rules: 3 to 7 words; at least 3 content "
    "words; the line must contain a conflict word (never, no, quit, "
    "secret, truth, wrong, mistake...), a number, or be a question. "
    "Use only facts present in the transcript excerpt - do not "
    "invent numbers, names, or claims. Reply with the hook line "
    "only, nothing else."
)


class PipelineError(RuntimeError):
    """Any pipeline stage failure."""


def _annotate_effect(receipt: Receipt) -> None:
    """Append the depth-effect line to a receipt (annotation only).

    The F1-F7 verdict is untouched: the line reports where the
    MEASURED glyph occlusion stands against the planner's 8-20%
    effect band, so a draft whose speaker never crosses the hook
    text carries that fact on its receipt instead of passing as a
    silent success (the gate run's failure mode: 0.0 everywhere).
    """
    f2 = next((c for c in receipt.checks if c.check_id == "F2"), None)
    if f2 is None:
        return
    occ = float(f2.measured.get("glyph_occlusion_mean", 0.0))
    receipt.notes.append(effect_note(occ))


# ---------------------------------------------------------------- utilities


def _probe_duration(path: str | Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(probe.stdout.strip())


def _run_ffmpeg(args: list[str], what: str) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                          capture_output=True)
    if proc.returncode != 0:
        raise PipelineError(
            f"ffmpeg failed ({what}): "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}")


def _cut_window(source: str | Path, start_s: float, length_s: float,
                out: Path) -> None:
    _run_ffmpeg(
        ["-ss", f"{start_s:.3f}", "-i", str(source),
         "-t", f"{length_s:.3f}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-c:a", "aac", str(out)],
        f"cut window at {start_s:.1f}s")


def _extract_wav_chunk(source: str | Path, start_s: float,
                       length_s: float, out: Path) -> None:
    _run_ffmpeg(
        ["-ss", f"{start_s:.3f}", "-i", str(source),
         "-t", f"{length_s:.3f}", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out)],
        f"extract WAV chunk at {start_s:.1f}s")


SAM_CACHE_TOLERANCE_S = 0.05


def _sam_cache_hit(short_dir: Path, w0: float, w1: float) -> bool:
    """True only when the cached SAM response is FOR THIS WINDOW.

    The cache lives in the per-Short directory (keyed by index), so
    a sidecar records the window the response was bought for; a
    response whose sidecar is missing or names a different window
    is never re-used - a re-run whose sweep picks different windows
    must not silently inherit another moment's masks.
    """
    resp = short_dir / "sam-response.json"
    side = short_dir / "sam-window.json"
    if not resp.exists() or not side.exists():
        return False
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
        return (
            abs(float(meta["start_s"]) - w0) <= SAM_CACHE_TOLERANCE_S
            and abs(float(meta["end_s"]) - w1) <= SAM_CACHE_TOLERANCE_S
        )
    except Exception:
        return False


def _sam_cache_store(short_dir: Path, raw: Any, w0: float,
                     w1: float) -> None:
    (short_dir / "sam-response.json").write_text(
        json.dumps(raw), encoding="utf-8")
    (short_dir / "sam-window.json").write_text(
        json.dumps({"start_s": w0, "end_s": w1}), encoding="utf-8")


def _decode_native_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """Decode a media file's audio at its native rate and channels.

    Returns (samples float32 shaped (n, channels), sample_rate,
    channels).
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    sample_rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", str(channels), "-ar", str(sample_rate),
         "-f", "f32le", "pipe:1"],
        capture_output=True)
    if proc.returncode != 0:
        raise PipelineError(f"could not decode audio from {path}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    return samples.reshape(-1, channels), sample_rate, channels


def _normalise_window_clip(raw_clip: Path, out_clip: Path,
                           work_dir: Path) -> dict[str, Any]:
    """Apply the window's loudness plan; the video stream is copied.

    One static gain (``plan_audio_normalisation``) plus a true-peak
    limiter at PEAK_CEILING_DBFS - on the measured Holmes windows
    the limiter touches ~1 ms of transient peaks per window. When
    the planned gain is negligible the raw cut is copied through
    untouched. The plan (with its evidence) is returned for the run
    summary.
    """
    profile = load_energy(raw_clip)
    samples, sample_rate, channels = _decode_native_audio(raw_clip)
    peak_dbfs = float(20.0 * np.log10(max(np.abs(samples).max(), 1e-9)))
    plan = plan_audio_normalisation(profile, peak_dbfs)
    record: dict[str, Any] = {
        "gain_db": plan.gain_db,
        "speech_rms_db": plan.speech_rms_db,
        "peak_dbfs": plan.peak_dbfs,
        "target_db": plan.target_db,
        "applied": False,
        "clipped_samples": 0,
    }
    if plan.gain_db < 0.25:
        shutil.copyfile(raw_clip, out_clip)
        return record
    gained = samples * float(10.0 ** (plan.gain_db / 20.0))
    ceiling = float(10.0 ** (PEAK_CEILING_DBFS / 20.0))
    record["clipped_samples"] = int((np.abs(gained) > ceiling).sum())
    gained = np.clip(gained, -ceiling, ceiling)
    wav_path = work_dir / "normalised-audio.wav"
    pcm = (np.clip(gained, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    _run_ffmpeg(
        ["-i", str(raw_clip), "-i", str(wav_path),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_clip)],
        "mux normalised window audio")
    record["applied"] = True
    return record


class LazyCanvasAlphas:
    """Canvas-space alphas computed on demand (LRU-cached).

    Materialising a whole window's canvas alphas at once is
    gigabytes; the verifier and the layout planner index/stream the
    sequence, so each frame is mapped through the compositor's
    speaker transform only when asked for.
    """

    def __init__(self, src_alphas: list[np.ndarray],
                 canvas: tuple[int, int] = DEFAULT_CANVAS, cap: int = 32,
                 placements: Sequence[object | None] | None = None):
        self.src = src_alphas
        self.canvas = canvas
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cap = cap
        # Layout states (extension): an optional per-frame explicit
        # speaker placement aligned with src_alphas, so the geometry
        # the verifier measures is the layout render's geometry.
        # None (everywhere) = the historic Stage mapping, unchanged.
        if placements is not None and len(placements) != len(src_alphas):
            raise ValueError("placements must align 1:1 with src_alphas")
        self.placements = placements

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        placement = (self.placements[i]
                     if self.placements is not None else None)
        a = to_canvas_alpha(self.src[i], self.canvas,
                            placement=placement).astype(np.float32)
        self.cache[i] = a
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return a

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------- selection


def normalize_window(start_s: float, end_s: float,
                     duration_s: float) -> tuple[float, float] | None:
    """Clamp a candidate span to a workable Short window.

    Too-short spans grow symmetrically to MIN_WINDOW_S; too-long
    spans keep their start and cap at MAX_WINDOW_S (the finish
    stack's memory ceiling at source frame rates). Returns None if
    nothing usable remains inside the source.
    """
    start_s = max(0.0, float(start_s))
    end_s = min(float(duration_s), float(end_s))
    if end_s <= start_s:
        return None
    length = end_s - start_s
    if length < MIN_WINDOW_S:
        mid = (start_s + end_s) / 2.0
        start_s = max(0.0, mid - MIN_WINDOW_S / 2.0)
        end_s = min(duration_s, start_s + MIN_WINDOW_S)
        start_s = max(0.0, end_s - MIN_WINDOW_S)
    elif length > MAX_WINDOW_S:
        end_s = start_s + MAX_WINDOW_S
    if end_s - start_s < 8.0:
        return None
    return round(start_s, 3), round(end_s, 3)


def _windows_overlap(s: float, e: float, cs: float, ce: float) -> bool:
    """The window-selection disqualifier: overlap beyond 50% of
    the smaller window."""
    return min(e, ce) - max(s, cs) > 0.5 * min(e - s, ce - cs)


def select_windows(candidates: Sequence[Candidate], duration_s: float,
                   max_shorts: int) -> list[tuple[Candidate, float, float]]:
    """Top-scoring, non-overlapping candidate windows, in score order."""
    ranked = sorted(candidates, key=lambda c: -c.score)
    chosen: list[tuple[Candidate, float, float]] = []
    for cand in ranked:
        if len(chosen) >= max_shorts:
            break
        window = normalize_window(cand.start_s, cand.end_s, duration_s)
        if window is None:
            continue
        s, e = window
        if any(_windows_overlap(s, e, cs, ce) for _c, cs, ce in chosen):
            continue
        chosen.append((cand, s, e))
    return chosen


# ------------------------------------------------- SAM coverage guard

# Threshold derivation (measured 2026-11-02 on cached runs; parse
# coverage = SAM-parsed frames / decoded window frames): every
# healthy window measured >= 0.991 (the lowest: McArthur graphics
# Short 2 in both runs, 650/656); the one coverage failure -
# McArthur Short 3 in BOTH graphics runs, SAM returning masks for
# only part of a split-screen window - measured 155/659 = 0.235
# both times. The threshold is the midpoint of the empty band
# between the failing and passing clusters - the maximum-margin
# separator the measured data supports: (0.235 + 0.991) / 2.
MIN_SAM_COVERAGE_FRAC = 0.613


class SamCoverageError(RuntimeError):
    """A selected window's SAM coverage cannot sustain a Short.

    Carries the measured counts so the run loop can record the
    rejection with its evidence and take the next candidate.
    """

    def __init__(self, parsed_frames: int, window_frames: int,
                 coverage: float) -> None:
        super().__init__(
            f"SAM covered {parsed_frames} of {window_frames} "
            f"window frames (coverage {coverage:.3f} below the "
            f"{MIN_SAM_COVERAGE_FRAC} floor) - the window cannot "
            "sustain a Short")
        self.parsed_frames = parsed_frames
        self.window_frames = window_frames
        self.coverage = coverage


def sam_coverage(parsed_frames: int, window_frames: int) -> float:
    """SAM parse coverage of a window: parsed / decoded frames."""
    if window_frames <= 0:
        return 0.0
    return parsed_frames / window_frames


def _next_untried_candidate(
        candidates: Sequence[Candidate], duration_s: float,
        reserved: Sequence[tuple[float, float]],
        tried: set[tuple[float, float]],
        ) -> tuple[Candidate, float, float] | None:
    """The highest-ranked candidate whose normalised window has
    not been tried and does not overlap a reserved window (the
    select_windows disqualifier) - the coverage guard's
    replacement pick. None when the pool is exhausted."""
    for cand in sorted(candidates, key=lambda c: -c.score):
        window = normalize_window(cand.start_s, cand.end_s, duration_s)
        if window is None:
            continue
        s, e = window
        if (round(s, 3), round(e, 3)) in tried:
            continue
        if any(_windows_overlap(s, e, rs, re) for rs, re in reserved):
            continue
        return cand, s, e
    return None


# Fix round 5: how far an alignment slice may reach past the
# DSP speech-activity bounds of its cue - roughly one word of
# spill in either direction. The slice is bounded by MEASURED
# activity; the margin exists because a word's acoustic edge can
# sit a few frames outside the thresholded activity map, not
# because the layout's proportions deserve a vote.
ALIGN_MARGIN_S = 0.25


@dataclass
class CaptionTimingResult:
    """One Short's word timings + their truthful provenance.

    ``source`` is the summary label: "dsp" when every cue is the
    DSP phrase split (or MMS was never requested), "mms" when
    every cue is a genuine forced alignment that passed the
    pace gate, "mms+dsp" for the honest mix. ``cue_sources``
    names each cue's producer ("mms" | "dsp_phrase");
    ``gate_records`` carries every pace-gate event (a rejected
    alignment, the one re-slice, a per-cue fallback) - the
    record the receipt and captions.json quote.
    """

    timings: list[list[WordTiming]]
    source: str
    cue_sources: list[str]
    gate_records: list[dict[str, Any]]


def _caption_word_timings(cues: Sequence[Any], energy: Any,
                          window_clip: str | Path,
                          requested_source: str,
                          log: Callable[[str], None],
                          ) -> CaptionTimingResult:
    """Per-cue word timings for one Short + truthful provenance.

    DSP (the default and the floor) is exactly the Stage 4 path:
    every cue is the DSP phrase split, labelled per cue as
    "dsp_phrase", summary "dsp".

    An MMS request aligns PER CUE (fix round 5 - the round-4
    whole-Short fallback let one bad cue silently relabel every
    cue, and the slices were the cues' own layout spans, which
    is how a uniform word grid got certified as measured
    timing):

    - The cue's slice is bounded by the DSP speech-activity
      span of its own span (caption_align.speech_activity_span)
      plus a small fixed margin - measured activity, never a
      word-count proportion.
    - The aligned result must pass the pace sanity gate
      (word_timing.pace_gate_failure): sustained pace over the
      words' own extent, and the extent against the passage's
      activity span. A failure re-slices ONCE to the exact
      activity span and realigns; a second failure (or any
      aligner failure) falls back for THAT CUE ONLY to the DSP
      phrase split, labelled "dsp_phrase", on the record.
    - The summary label reflects the mix exactly.
    """
    def dsp_cue(c: Any) -> list[WordTiming]:
        return word_timings(c.text.split(), c.start_s, c.end_s,
                            source="dsp", profile=energy)

    if requested_source != "mms":
        return CaptionTimingResult(
            timings=[dsp_cue(c) for c in cues], source="dsp",
            cue_sources=["dsp_phrase"] * len(cues),
            gate_records=[])
    try:
        samples = load_samples(window_clip)
    except Exception as exc:
        log(f"caption timing: MMS requested but the window "
            f"audio could not be decoded ({exc}); DSP floor "
            f"used and labelled")
        return CaptionTimingResult(
            timings=[dsp_cue(c) for c in cues], source="dsp",
            cue_sources=["dsp_phrase"] * len(cues),
            gate_records=[{"cue_index": None, "action": "dsp_fallback",
                           "reason": f"audio decode failed: {exc}"}])

    total_s = samples.shape[0] / MMS_SAMPLE_RATE
    out: list[list[WordTiming]] = []
    cue_sources: list[str] = []
    records: list[dict[str, Any]] = []

    def align(words: list[str], w0: float, w1: float,
              ) -> list[WordTiming]:
        i0 = max(0, int(round(w0 * MMS_SAMPLE_RATE)))
        i1 = min(samples.shape[0], int(round(w1 * MMS_SAMPLE_RATE)))
        return word_timings(words, w0, w1, source="mms",
                            samples=samples[i0:i1])

    for ci, c in enumerate(cues):
        words = c.text.split()
        act = speech_activity_span(energy, c.start_s, c.end_s)
        if act is not None:
            act_span_s = act[1] - act[0]
            w0 = max(0.0, act[0] - ALIGN_MARGIN_S)
            w1 = min(total_s, act[1] + ALIGN_MARGIN_S)
        else:
            act_span_s = None
            w0, w1 = c.start_s, c.end_s
        timings: list[WordTiming] | None = None
        try:
            timings = align(words, w0, w1)
        except WordTimingError as exc:
            records.append({
                "cue_index": ci, "action": "dsp_fallback",
                "reason": f"aligner failed on the activity-bounded "
                          f"slice [{w0:.2f}, {w1:.2f}]: {exc}"})
        if timings is not None:
            failure = pace_gate_failure(timings, act_span_s)
            if failure is not None:
                records.append({
                    "cue_index": ci, "action": "pace_gate_rejected",
                    "slice": [round(w0, 3), round(w1, 3)],
                    "reason": failure})
                timings = None
                if act is not None:
                    try:
                        retry = align(words, act[0], act[1])
                    except WordTimingError as exc:
                        records.append({
                            "cue_index": ci, "action": "dsp_fallback",
                            "reason": f"aligner failed on the DSP "
                                      f"re-slice: {exc}"})
                        retry = None
                    if retry is not None:
                        failure2 = pace_gate_failure(retry, act_span_s)
                        if failure2 is None:
                            records.append({
                                "cue_index": ci,
                                "action": "realigned_on_dsp_span",
                                "slice": [round(act[0], 3),
                                          round(act[1], 3)]})
                            timings = retry
                        else:
                            records.append({
                                "cue_index": ci,
                                "action": "dsp_fallback",
                                "reason": f"re-slice still fails the "
                                          f"pace gate: {failure2}"})
                else:
                    records.append({
                        "cue_index": ci, "action": "dsp_fallback",
                        "reason": "pace gate failed and the cue has "
                                  "no DSP activity span to re-slice "
                                  "to"})
        if timings is None:
            out.append(dsp_cue(c))
            cue_sources.append("dsp_phrase")
        else:
            out.append(timings)
            cue_sources.append("mms")
    n_mms = cue_sources.count("mms")
    if n_mms == len(cues):
        summary = "mms"
    elif n_mms == 0:
        summary = "dsp"
    else:
        summary = "mms+dsp"
    if records:
        log(f"caption timing: {n_mms}/{len(cues)} cues MMS-aligned; "
            f"{len(records)} gate/fallback record(s); effective "
            f"source {summary}")
    return CaptionTimingResult(timings=out, source=summary,
                               cue_sources=cue_sources,
                               gate_records=records)


# ---------------------------------------------------------------- hook


def propose_hook(spark: SparkClient, transcript_text: str,
                 fallback_text: str, *, ts: str = ""
                 ) -> tuple[str, dict[str, Any] | None]:
    """One Spark 1.3 call for the hook line; deterministic fallbacks.

    The reply is validated (word count + content words) before use -
    a line that breaks the hook rules is replaced by words lifted
    from the moment itself, and the fallback is on the record.
    """
    def usable(line: str) -> bool:
        words = line.split()
        return 2 <= len(words) <= 9 and len(_content_words(line)) >= 3

    call: dict[str, Any] | None = None
    try:
        result = spark.chat(
            [{"role": "system", "content": HOOK_SYSTEM_PROMPT},
             {"role": "user", "content":
              f"Transcript excerpt: {transcript_text[:1200]}"}],
            max_tokens=HOOK_MAX_TOKENS,
            purpose="pipeline-hook-line", ts=ts)
        call = {"model": "muse-spark-1.3",
                "purpose": "pipeline-hook-line",
                "cost_usd": result.cost_usd,
                "request_id": result.request_id}
        for raw_line in result.text.splitlines():
            line = raw_line.strip().strip("\"'").upper()
            if line and usable(line):
                return line, call
    except Exception:
        call = None
    fallback = " ".join(fallback_text.replace("\n", " ").split()[:7]).upper()
    if usable(fallback):
        return fallback, call
    return "WATCH THIS MOMENT", call


# ---------------------------------------------------------------- clients


def upload_media_file(path: str | Path, *, api_key_provider: Callable[[], str]
                      ) -> str:
    """Upload a media file to the Meta Model API; returns file_id.

    Mirrors the verified /v1/files multipart shape (purpose +
    file parts) that the SAM and Wonder paths both use.
    """
    path = Path(path)
    boundary = "----pipeline" + uuid.uuid4().hex
    data = path.read_bytes()
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (b'Content-Disposition: form-data; name="purpose"\r\n\r\n'
             b"user_data\r\n")
    body += f"--{boundary}\r\n".encode()
    body += (f'Content-Disposition: form-data; name="file"; '
             f'filename="{path.name}"\r\n').encode()
    body += b"Content-Type: video/mp4\r\n\r\n" + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{META_API_BASE}/files", data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key_provider()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        })

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            raise RuntimeError("redirect refused")

    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=300) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise PipelineError(f"file upload failed (HTTP {exc.code}): "
                            f"{detail}") from exc
    if "id" not in raw:
        raise PipelineError(f"file upload returned no id: {raw}")
    return str(raw["id"])


# ---------------------------------------------------------------- the run


@dataclass
class _Clients:
    sam: SamClient
    spark_open: SparkOpenClient
    spark: SparkClient
    vt: VoiceTranscribeClient
    wonder: WonderClient
    ledger: SpendLedger
    upload_file: Callable[[str | Path], str]
    # Muse Image, for the context graphics layer only; consulted
    # solely when a run enables graphics.
    image: ImageClient | None = None


def _transcribe_all(video: Path, work_dir: Path, clients: _Clients,
                    *, ts: str, log: Callable[[str], None]
                    ) -> tuple[list[TranscriptTurn], float]:
    duration = _probe_duration(video)
    work_dir = Path(work_dir) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    turns: list[TranscriptTurn] = []
    offset = 0.0
    while offset < duration:
        length = min(VT_CHUNK_S, duration - offset)
        wav = work_dir / f"vt-chunk-{int(offset)}.wav"
        _extract_wav_chunk(video, offset, length, wav)
        result = clients.vt.transcribe_file(
            wav, purpose="pipeline-voice-transcribe", ts=ts)
        for t in result.turns:
            start_ms = (t.start_ms or 0) + int(offset * 1000)
            end_ms = (t.end_ms + int(offset * 1000)
                      if t.end_ms is not None else None)
            turns.append(TranscriptTurn(
                text=t.text, start_ms=start_ms, end_ms=end_ms,
                speaker=t.speaker))
        log(f"transcribed {offset:.0f}-{offset + length:.0f}s: "
            f"{len(result.turns)} turns")
        offset += length
    turns.sort(key=lambda t: t.start_ms or 0)
    return turns, duration


def _clip_turn_text(text: str, start: float, end: float,
                    w0: float, w1: float) -> str:
    """The words of a turn that belong to the window's slice.

    Fix round 5 root cause: _local_turns clipped a turn's SPAN
    to the window but kept its FULL text. A 30s turn (91 words)
    overlapping a 14s window by its second half became a 14s
    turn carrying all 91 words, and the caption layout - which
    flows a turn's words uniformly over its active time - laid
    them out at a constant, superhuman ~6.9 words/s. Forced
    alignment then "confirmed" the grid cue by cue, and the
    receipt called it mms (QM6; QH5 carried the same defect at
    a plausible-looking 3.44 wps). The span and the text are
    now clipped together: words are apportioned over the
    turn's span by time (the same flow assumption the layout
    makes - word SELECTION only; the timings themselves are
    measured by the aligner afterwards and audited by the
    pace gate), and a word is kept when the centre of its
    proportional slot falls inside the window.
    """
    words = text.split()
    if not words or end <= start:
        return text
    if start >= w0 and end <= w1:
        return text
    n = len(words)
    span = end - start
    kept = [wd for i, wd in enumerate(words)
            if w0 <= start + (i + 0.5) * span / n < w1]
    return " ".join(kept)


def _local_turns(turns: Sequence[TranscriptTurn], w0: float, w1: float
                 ) -> tuple[list[TranscriptTurn], str]:
    """Turns overlapping the window, re-based to window-local
    seconds - span AND text clipped to the window (fix round
    5: see _clip_turn_text). A turn whose slice holds no words
    is dropped rather than carried as an empty shell."""
    local: list[TranscriptTurn] = []
    texts: list[str] = []
    for t in turns:
        start = (t.start_ms or 0) / 1000.0
        end = (t.end_ms or 0) / 1000.0
        if end <= start:
            end = start + 5.0
        if end <= w0 or start >= w1:
            continue
        clipped = _clip_turn_text(t.text, start, end, w0, w1)
        if not clipped:
            continue
        local.append(TranscriptTurn(
            text=clipped,
            start_ms=int(max(0.0, start - w0) * 1000),
            end_ms=int(min(w1 - w0, end - w0) * 1000),
            speaker=t.speaker))
        texts.append(clipped)
    return local, " ".join(texts)


def _process_short(index: int, cand: Candidate, w0: float, w1: float,
                   video: Path, turns: list[TranscriptTurn],
                   out_dir: Path, clients: _Clients, *, ts: str,
                   canvas: tuple[int, int],
                   log: Callable[[str], None],
                   graphics_enabled: bool = False) -> dict[str, Any]:
    ledger = clients.ledger
    cost_before = ledger.total_usd()
    short_dir = out_dir / f"short-{index}"
    short_dir.mkdir(parents=True, exist_ok=True)
    length = w1 - w0
    log(f"short {index}: window {w0:.1f}-{w1:.1f}s "
        f"(candidate score {cand.score:.2f}: {cand.reason})")

    raw_clip = short_dir / "window-raw.mp4"
    _cut_window(video, w0, length, raw_clip)
    # Loudness normalisation (static gain + peak limiter; the DSP's
    # relative silence threshold keeps F5's gap measurement honest
    # under the gain - see caption_align). The normalised cut is the
    # window everything downstream renders and measures.
    window_clip = short_dir / "window.mp4"
    norm = _normalise_window_clip(raw_clip, window_clip, short_dir)
    log(f"short {index}: audio normalisation gain "
        f"{norm['gain_db']:+.2f} dB (speech RMS "
        f"{norm['speech_rms_db']}, peak {norm['peak_dbfs']} dBFS, "
        f"applied {norm['applied']}, limiter touched "
        f"{norm['clipped_samples']} samples)")
    fps = video_fps(window_clip)
    frames = read_video_frames(window_clip)
    log(f"short {index}: cut {len(frames)} frames @ {fps:.2f} fps")

    # SAM, cached per window: a saved response is re-used only when
    # its sidecar proves it was bought for THIS window.
    sam_path = short_dir / "sam-response.json"
    if _sam_cache_hit(short_dir, w0, w1):
        raw_sam = json.loads(sam_path.read_text(encoding="utf-8"))
        log(f"short {index}: SAM response from cache "
            f"(window verified against sidecar)")
    else:
        if sam_path.exists():
            log(f"short {index}: cached SAM response is not for "
                f"this window; buying segmentation")
        file_id = clients.upload_file(window_clip)
        seg = clients.sam.segment_video(
            file_id=file_id, n_frames=len(frames),
            purpose=f"pipeline-sam-short{index}", ts=ts)
        raw_sam = seg.raw
        _sam_cache_store(short_dir, raw_sam, w0, w1)
        log(f"short {index}: SAM segmented "
            f"({seg.billed_frames} frames billed)")
    parsed = parse_sam_response(raw_sam)
    # SAM-coverage window guard (fix phase, 2026-11-02): when SAM
    # returns masks for only a fraction of the window, the old
    # tail-truncation below silently produced a stub Short
    # (McArthur Short 3 twice: 155/659 frames -> a 5.2s stub that
    # failed downstream). Reject the window HERE - before the
    # finish/composite spend - and let the run loop take the next
    # sweep candidate instead. The one-frame tail artifact (659
    # of 660) is coverage 0.998 and passes far above the floor.
    coverage = sam_coverage(len(parsed), len(frames))
    if coverage < MIN_SAM_COVERAGE_FRAC:
        raise SamCoverageError(len(parsed), len(frames), coverage)
    # Alignment policy: SAM may return one frame fewer than the container
    # decodes (a sub-frame tail artifact: a 22.0s window at 29.97fps is
    # 659.34 frames - ffmpeg yields 660, SAM returns 659). Both sequences
    # decode from frame 0 in order, so a surplus of DECODED frames is the
    # tail: truncate the guides to the parsed count and log it. The reverse
    # (SAM returning MORE frames than decoded) is a real misalignment and
    # still raises in finished_alpha's strict guard.
    n_parsed = len(parsed)
    if len(frames) > n_parsed:
        log(f"short {index}: aligning guides - {len(frames)} decoded "
            f"frames vs {n_parsed} SAM frames; truncating "
            f"{len(frames) - n_parsed} tail frame(s)")
        frames = frames[:n_parsed]
    # Guides stream into the finish pass as a generator: the list
    # form cost a full float64 copy of the window (~1.2 GB at 660
    # frames) that then coexisted with the finish stack's own lists
    # (the 2026-10-29 OOM - see mask_decode.finished_alpha).
    guides = (f.astype(np.float64).mean(axis=2) / 255.0 for f in frames)
    _indices, src_alphas = finished_alpha(parsed, guides)
    # Source alpha frame size BEFORE any geometry transform bakes
    # into the list (the geometry bundle's alpha provenance).
    src_frame_shape = (tuple(src_alphas[0].shape)
                       if src_alphas else None)

    # Stage composition (fix round 2, Fix A): the tall-narrow
    # Stage zoom is planned and baked FIRST, so the reframe, the
    # hook effect, and every downstream consumer work from the
    # enlarged speaker geometry. The reframe below then plans on
    # the zoomed alphas (normally identity: the zoom already
    # solved the centroid target).
    stage_zoom = plan_stage_zoom(src_alphas, canvas)
    if stage_zoom.feasible and not stage_zoom.identity:
        frames, src_alphas = apply_stage_zoom(
            frames, src_alphas, stage_zoom)
    log(f"short {index}: stage zoom {stage_zoom.note}")

    # Speaker reframing by construction (F3): place the FACE, not
    # just the text around it. The transform is PLANNED here but
    # applied after the hook is known: the depth-effect planner
    # (below) may add lift to the construction reframe so the
    # speaker's head/shoulders actually reach the hook band, and
    # the transform must be applied exactly once, at its final
    # value, so layout, render and verifier all work from the same
    # reframed geometry.
    reframe = plan_speaker_reframe(src_alphas, canvas)

    def make_canvas(alphas: list[np.ndarray]) -> LazyCanvasAlphas:
        return LazyCanvasAlphas(alphas, canvas)

    # Captions + hook for this moment.
    local_turns, excerpt = _local_turns(turns, w0, w1)
    energy = load_energy(window_clip)
    cues, cap_report = build_captions(
        local_turns, energy, spark=clients.spark,
        purpose=f"pipeline-caption-punctuation-short{index}", ts=ts)
    # Cue-sanity guard (fix round 2, Fix D): repair degenerate
    # cue spans (compressed VT turns) BEFORE word timing and
    # before graphics planning, so word timings and graphic span
    # anchors are both derived from the repaired spans. Records
    # ride captions.json and the receipt.
    cues, cue_repairs = repair_cue_spans(cues, duration_s=length)
    if cue_repairs:
        log(f"short {index}: cue-sanity guard: "
            f"{len(cue_repairs)} cue span repair(s)")
    # Word-level caption timing (extension Stage 4; MMS wired
    # live in the 2026-11-02 fix phase; per-cue provenance in
    # fix round 5): per-cue word timings behind the word_timing
    # interface - DSP floor by default, MMS forced alignment
    # when CAPTION_TIMING_SOURCE=mms is requested, each cue
    # aligned on its DSP-activity-bounded slice and held to the
    # pace sanity gate. The EFFECTIVE per-cue sources and the
    # summary label come back with the timings and are what
    # captions.json and every receipt for this Short carry.
    timing_result = _caption_word_timings(
        cues, energy, window_clip, CAPTION_TIMING_SOURCE, log)
    cue_word_timings = timing_result.timings
    timing_source = timing_result.source
    # Valley snap (fix round 4; scope narrowed in fix round 5):
    # a deterministic DSP correction of the aligner's per-word
    # jitter - each word start moves to the nearest speech-energy
    # valley within +/-150ms (see word_timing.snap_word_timings).
    # Only genuinely aligned (mms) words may move: snapping a
    # proportional DSP split would manufacture false precision.
    # The snap's own stats ride captions.json + the receipt as
    # caption_snap. An audio decode failure keeps the unsnapped
    # timings and records applied=False with the reason - the
    # record never claims a correction that did not run.
    caption_snap: dict[str, Any] | None = None
    try:
        snap_samples = load_samples(window_clip)
        cue_word_timings, caption_snap = snap_word_timings(
            cue_word_timings, snap_samples, snap_sources=("mms",))
        if caption_snap.get("applied"):
            log(f"short {index}: caption valley snap: "
                f"{caption_snap['words_snapped']}/"
                f"{caption_snap['words']} words moved, mean "
                f"{caption_snap['mean_abs_shift_ms']}ms")
    except Exception as exc:  # decode failure: keep, record
        caption_snap = {"applied": False,
                        "method": "energy_valley_snap",
                        "reason": f"{type(exc).__name__}: {exc}"}
        log(f"short {index}: caption valley snap unavailable "
            f"({exc}); unsnapped timings kept")
    (short_dir / "captions.json").write_text(json.dumps({
        "cues": [{"text": c.text, "start_s": c.start_s,
                  "end_s": c.end_s,
                  "word_timings": [w.to_dict() for w in timings]}
                 for c, timings in zip(cues, cue_word_timings)],
        "timing_source": timing_source,
        "cue_sources": timing_result.cue_sources,
        "timing_gate": timing_result.gate_records,
        "caption_snap": caption_snap,
        "cue_repairs": cue_repairs,
        "chunk_max_words": CAPTION_CHUNK_MAX_WORDS,
        "turn_reports": [
            {"turn_index": t.turn_index, "used_fallback": t.used_fallback,
             "spark_request_id": t.spark_request_id,
             "spark_cost_usd": t.spark_cost_usd, "note": t.note}
            for t in cap_report.turns],
    }, indent=2), encoding="utf-8")
    hook_text, hook_call = propose_hook(
        clients.spark, excerpt,
        cues[0].text if cues else cand.reason, ts=ts)
    log(f"short {index}: hook {hook_text!r}; {len(cues)} caption cues")

    # Depth-effect planning: aim the hook at the 8-20% occlusion
    # band, lifting the speaker via the reframe machinery when no
    # band position reaches it. The returned transform is the one
    # applied - once - before anything is laid out or rendered.
    hook_sample = min(len(src_alphas), int(HOOK_WINDOW_S * fps))
    effect = plan_hook_effect(
        hook_text, src_alphas, canvas, base_reframe=reframe,
        hook_sample=hook_sample)
    if effect.pad_bottom_px != 0:
        frames, src_alphas = apply_speaker_reframe(
            frames, src_alphas, effect.pad_bottom_px)
    log(f"short {index}: speaker reframe "
        f"{'identity' if reframe.identity else reframe.note} "
        f"(feasible {reframe.feasible}); effect: {effect.note} "
        f"(applied pad {effect.pad_bottom_px})")
    canvas_alphas = LazyCanvasAlphas(src_alphas, canvas)

    # Fix round 2/3 receipt declarations (records, not checks):
    # the Stage composition applied to this Short and the canvas
    # backdrop treatment the renders use (fix round 3: the
    # studio_procedural canvas is the default; its seed is
    # derived deterministically from the run/window identity.
    # The fix B minus-speaker declaration survives verbatim for
    # runs that select that mode; "blur" only when no frame
    # carries a speaker mask at all).
    stage_geometry = stage_zoom.declaration()
    stage_geometry["reframe_pad_bottom_px"] = reframe.pad_bottom_px
    stage_geometry["effect_pad_bottom_px"] = effect.pad_bottom_px
    maskless_frames = sum(
        1 for a in src_alphas if not bool((a > 0.5).any()))
    backdrop_seed = studio_seed(
        f"{video.name}:{w0:.3f}-{w1:.3f}:short-{index}")
    if CANVAS_BACKDROP_MODE == BACKDROP_STUDIO:
        canvas_backdrop = {
            "mode": BACKDROP_STUDIO,
            "seed": backdrop_seed,
            "drift": "ken_burns_1.12",
            "shadow": "mask_base_ellipse",
            "maskless_frames": maskless_frames,
        }
    elif CANVAS_BACKDROP_MODE == BACKDROP_SOURCE_MINUS_SPEAKER:
        canvas_backdrop = {
            "mode": ("source_minus_speaker"
                     if maskless_frames < len(src_alphas) else "blur"),
            "darken": BACKDROP_GRADE_BRIGHTNESS,
            "saturation": BACKDROP_GRADE_SATURATION,
            "drift": "ken_burns_1.12",
            "fill": BACKDROP_FILL,
            "maskless_frames": maskless_frames,
        }
    else:
        canvas_backdrop = {
            "mode": CANVAS_BACKDROP_MODE,
            "maskless_frames": maskless_frames,
        }

    # Layout from the decoded geometry.
    placement = effect.placement
    cap_layout = plan_captions(
        [(c.start_s, c.end_s) for c in cues], canvas_alphas,
        fps=fps, canvas=canvas)
    log(f"short {index}: layout hook frac "
        f"{placement.y_center_frac:.2f} (predicted occlusion "
        f"{placement.predicted_occlusion:.3f}, effect "
        f"{placement.effect_status}); caption rect "
        f"{cap_layout.rect} feasible {cap_layout.feasible}")

    # Fix round 2 (Fix E): two-tone hook - the accent word
    # (deterministic selection) in Viral Pink, the solved text
    # size from the effect plan, and per-word layers for the
    # stagger-pop entry.
    accent_index = hook_accent_index(hook_text)
    hook_style = {
        "tone": "two_tone",
        "accent_word": (hook_text.split()[accent_index]
                        if accent_index is not None else None),
        "accent_index": accent_index,
        "accent_rgb": [255, 61, 126],
        "entry": "stagger_pop",
        "text_size": placement.text_size,
    }
    caption_style = {
        "chunk_max_words": CAPTION_CHUNK_MAX_WORDS,
        "word_states": {
            "upcoming": "white_alpha_140",
            "active": "viral_pink_scale_1.10",
            "spoken": "white_full_alpha",
            "spoken_emphasis": "viral_pink",
        },
    }
    hook_spec = HookSpec(hook_text, (0.0, min(HOOK_WINDOW_S, length)),
                         placement.y_center_frac, layout_rendered=True,
                         accent_index=accent_index,
                         text_size=placement.text_size)
    hook_rgba = render_hook_layer(
        hook_text, canvas, y_center_frac=placement.y_center_frac,
        accent_index=accent_index, start_size=placement.text_size)
    hook_word_layers_list = hook_word_layers(
        hook_text, canvas, y_center_frac=placement.y_center_frac,
        accent_index=accent_index, start_size=placement.text_size)

    # Context graphics (flag-gated, default OFF - Kamlesh's
    # 2026-10-30 feature): planned from the caption phrases against
    # the reframed geometry, placed on the backdrop plane, and
    # carried through render, verify (F8) and the rescue artifacts.
    # Disabled = None everywhere below = the pre-graphics code path.
    placed_graphics = None
    placed_backdrops = None
    speaker_forward_decls = None
    speaker_forward_recipe = None
    layout_track = None
    layout_decls = None
    layout_rects = None
    graphics_report = None
    sfx_plan_events = None
    applied_sfx = None
    draft1_audio = window_clip
    if graphics_enabled:
        _glyph_mask, glyph_bbox = hook_spec.glyph(canvas)
        build = build_graphics(
            clients.spark, clients.image, cues, canvas_alphas,
            fps=fps, canvas=canvas,
            caption_rect=cap_layout.rect,
            hook_glyph_bbox=glyph_bbox,
            hook_window=hook_spec.window, duration_s=length,
            ts=ts)
        # Speaker-forward (block 2): emphasis framing for the
        # planner's marked spans, planned inside the F3/F4 limits
        # per span. A feasible warp is baked into frames/alphas -
        # like the construction reframe - and the canvas alphas
        # rebuilt, so render and verifier measure the emphasised
        # geometry as-is; an infeasible span falls back to the
        # standard framing with the reason on the record.
        if build.speaker_forward:
            decls: list[dict[str, Any]] = []
            sf_recipe_entries: list[dict[str, Any]] = []
            for span in build.speaker_forward:
                f0 = max(0, int(round(span.start_s * fps)))
                f1 = min(len(src_alphas),
                         int(round(span.end_s * fps)))
                plan = plan_speaker_emphasis(
                    src_alphas[f0:f1], canvas)
                if not plan.feasible:
                    build.dropped.append({
                        "span": span.to_dict(),
                        "reason": "speaker_forward fell back to "
                                  "standard framing: "
                                  f"{plan.note}"})
                    continue
                frames, src_alphas = apply_speaker_emphasis(
                    frames, src_alphas, plan, f0, f1)
                decls.append({
                    "span": {"start_s": span.start_s,
                             "end_s": span.end_s},
                    "scale": plan.scale,
                    "centroid_target": [
                        canvas[0] / 2.0,
                        (SAFE_Y0 + SAFE_Y1) / 2.0],
                    "placed_centroid": plan.placed_centroid,
                    "placed_head_top_min":
                        plan.placed_head_top_min,
                    "note": plan.note})
                # Geometry-bundle capture: the applied transform
                # itself (crop + pad), which the receipt-facing
                # declaration above does not carry.
                sf_recipe_entries.append({
                    "span": {"start_s": span.start_s,
                             "end_s": span.end_s},
                    "scale": plan.scale,
                    "crop": list(plan.crop),
                    "pad_bottom_px": plan.pad_bottom_px,
                    "placed_centroid": plan.placed_centroid,
                    "placed_head_top_min":
                        plan.placed_head_top_min})
            if decls:
                canvas_alphas = make_canvas(src_alphas)
                speaker_forward_decls = decls
                speaker_forward_recipe = sf_recipe_entries
                log(f"short {index}: speaker-forward emphasis "
                    f"applied on {len(decls)} span(s)")
        # Layout states (extension Stage 2): plan the layout
        # track for the placed specs through the SHARED builder
        # (graphics_layer.plan_layout - the same function the
        # rescue executors rebuild from), so the pipeline render
        # and any rescue re-render of this Short use the identical
        # track. The verifier then measures the layout-mapped
        # alphas - the geometry the render actually used.
        layout_built = plan_layout(
            build.placed, src_alphas, fps=fps, canvas=canvas,
            n_frames=len(frames)) if build.placed else None
        if layout_built is not None:
            layout_track, layout_decls, layout_rects = layout_built
            build.layout_spans = layout_decls
            if layout_track is not None:
                canvas_alphas = LazyCanvasAlphas(
                    src_alphas, canvas, placements=layout_track)
                states = [d["layout"] for d in layout_decls
                          if d.get("feasible")]
                log(f"short {index}: layout states planned "
                    f"{states}")
        placed_graphics = build.placed
        placed_backdrops = build.backdrops
        graphics_report = build.to_dict()
        if speaker_forward_decls:
            graphics_report["speaker_forward_transforms"] = \
                speaker_forward_decls
        # SFX (block 2): effects derived deterministically from
        # the placed graphics, mixed over the normalised window
        # audio. The MIX is what ships; the window clip remains
        # the speech stem the verifier's F5 measures.
        sfx_plan_events = plan_sfx(placed_graphics)
        graphics_report["sfx_plan"] = [
            e.to_dict() for e in sfx_plan_events]
        applied_sfx = []
        if sfx_plan_events:
            mix_path = short_dir / "sfx-mix.wav"
            applied_sfx = mix_to_file(
                window_clip, sfx_plan_events, mix_path)
            draft1_audio = mix_path
        (short_dir / "graphics-plan.json").write_text(
            json.dumps(graphics_report, indent=2), encoding="utf-8")
        log(f"short {index}: graphics planned "
            f"{len(build.placed)} placed, {len(build.dropped)} dropped"
            f"{' (' + build.error + ')' if build.error else ''}")

    # Draft 1 render + verify.
    draft1 = short_dir / "draft-1.mp4"
    render_kwargs: dict[str, Any] = {}
    if placed_graphics is not None:
        render_kwargs["graphics"] = graphics_entries(placed_graphics)
    if placed_backdrops is not None:
        render_kwargs["backdrop_spans"] = backdrop_entries(
            placed_backdrops)
    if layout_track is not None:
        render_kwargs["layout_track"] = layout_track
        if layout_rects:
            render_kwargs["span_caption_rects"] = layout_rects
    render_kwargs["caption_word_timings"] = cue_word_timings
    render_kwargs["caption_pop_s"] = CAPTION_POP_S
    render_kwargs["hook_word_layers"] = hook_word_layers_list
    if CANVAS_BACKDROP_MODE != DEFAULT_BACKDROP:
        render_kwargs["backdrop"] = CANVAS_BACKDROP_MODE
    if CANVAS_BACKDROP_MODE == BACKDROP_STUDIO:
        render_kwargs["backdrop_seed"] = backdrop_seed
    render_composite_with_captions(
        frames, src_alphas, hook_rgba=hook_rgba,
        hook_window=hook_spec.window, cues=cues,
        caption_rect=cap_layout.rect, audio_from=draft1_audio,
        out_path=draft1, fps=fps, canvas=canvas, **render_kwargs)
    receipt1 = verify(
        VerifyInputs(
            video_path=draft1, fps=fps, canvas_alphas=canvas_alphas,
            audio=load_energy(draft1), hook=hook_spec, cues=cues,
            caption_rect=cap_layout.rect, sam_response_path=sam_path,
            graphics=placed_graphics,
            backdrops=placed_backdrops,
            speaker_forward=speaker_forward_decls,
            layout_spans=layout_decls,
            caption_timing_source=timing_source,
            caption_cue_sources=timing_result.cue_sources,
            caption_timing_gate=timing_result.gate_records or None,
            cue_repairs=cue_repairs or None,
            stage_geometry=stage_geometry,
            canvas_backdrop=canvas_backdrop,
            hook_style=hook_style,
            caption_style=caption_style,
            caption_snap=caption_snap,
            speech_audio=energy if graphics_enabled else None,
            sfx=applied_sfx),
        wonder=clients.wonder, draft=1, ts=ts)
    _annotate_effect(receipt1)
    log(f"short {index}: draft 1 verdict {receipt1.verdict} "
        f"failed {receipt1.failed}")

    # Geometry bundle (reverify closure, 2026-11-03): record the
    # exact geometry draft 1's verifier measured with, copied from
    # the objects above — never recomputed — so the offline
    # re-verifier builds its inputs from the record instead of
    # re-deriving them. Draft 2's entry is written after the
    # rescue, when the executed repair is known.
    bundle_drafts: dict[int, dict[str, Any]] = {
        1: draft_bundle_dict(
            stage_geometry=stage_geometry,
            alpha_recipe=alpha_recipe_dict(
                src_frame_size=src_frame_shape,
                stage_geometry=stage_geometry,
                speaker_forward=(speaker_forward_recipe
                                 if speaker_forward_recipe
                                 is not None
                                 else speaker_forward_decls)),
            hook=hook_to_dict(hook_spec),
            caption_rect=cap_layout.rect,
            layout_spans=layout_decls,
            hook_slide_px_s=0.0,
            inputs_sha256=receipt1.inputs_sha256),
    }
    write_bundle(short_dir, bundle_drafts)

    # Rescue (short-circuits when draft 1 passed).
    artifacts = RenderArtifacts(
        window_clip=window_clip, render_path=draft1, frames=frames,
        src_alphas=src_alphas, canvas_alphas=canvas_alphas,
        make_canvas_alphas=make_canvas, fps=fps, hook=hook_spec,
        cues=cues, caption_rect=cap_layout.rect,
        sam_response_path=sam_path, transcript_text=excerpt,
        work_dir=short_dir, canvas=canvas, graphics=placed_graphics,
        sfx_plan=sfx_plan_events, backdrops=placed_backdrops,
        layout_track=layout_track, layout_spans=layout_decls,
        span_caption_rects=layout_rects,
        caption_word_timings=cue_word_timings,
        caption_timing_source=timing_source,
        caption_cue_sources=timing_result.cue_sources,
        caption_timing_gate=timing_result.gate_records or None,
        caption_pop_s=CAPTION_POP_S,
        cue_repairs=cue_repairs or None,
        stage_geometry=stage_geometry,
        canvas_backdrop=canvas_backdrop,
        backdrop_mode=CANVAS_BACKDROP_MODE,
        backdrop_seed=(backdrop_seed
                       if CANVAS_BACKDROP_MODE == BACKDROP_STUDIO
                       else None),
        hook_word_layers=hook_word_layers_list,
        hook_style=hook_style,
        caption_style=caption_style,
        caption_snap=caption_snap)

    captured_execution: dict[str, Any] = {}

    def verify_fn(execution) -> Any:
        captured_execution["execution"] = execution
        receipt2 = verify(
            VerifyInputs(
                video_path=execution.render_path, fps=fps,
                canvas_alphas=execution.canvas_alphas,
                audio=load_energy(execution.render_path),
                hook=execution.hook, cues=execution.cues,
                caption_rect=execution.caption_rect,
                sam_response_path=sam_path,
                graphics=execution.graphics,
                backdrops=execution.backdrops,
                speaker_forward=speaker_forward_decls,
                layout_spans=execution.layout_spans,
                caption_timing_source=execution.caption_timing_source,
                caption_cue_sources=execution.caption_cue_sources,
                caption_timing_gate=execution.caption_timing_gate,
                cue_repairs=execution.cue_repairs,
                stage_geometry=execution.stage_geometry,
                canvas_backdrop=execution.canvas_backdrop,
                hook_style=execution.hook_style,
                caption_style=execution.caption_style,
                caption_snap=execution.caption_snap,
                speech_audio=(
                    load_energy(execution.speech_audio_from)
                    if execution.speech_audio_from is not None
                    else None),
                sfx=execution.sfx_events,
                hook_slide_px_s=execution.hook_slide_px_s),
            wonder=clients.wonder, draft=2, ts=ts)
        _annotate_effect(receipt2)
        return receipt2

    chain = rescue(receipt1, artifacts, spark=clients.spark,
                   verify_fn=verify_fn, ts=ts)
    log(f"short {index}: final verdict {chain.final_verdict} "
        f"(repair: {chain.repair.status} {chain.repair.menu_id or ''})")

    # Geometry bundle, draft 2: only when the executed repair
    # touches geometry (audio-only repairs reuse draft 1's bundle
    # by design). The entry points at draft 1's alpha recipe as
    # its base and records the geometry draft 2's verifier
    # actually consumed from the captured execution.
    if (len(chain.drafts) > 1
            and chain.repair.status == "executed"
            and chain.repair.menu_id in DRAFT2_BUNDLE_REPAIRS
            and "execution" in captured_execution):
        execution = captured_execution["execution"]
        bundle_drafts[2] = draft_bundle_dict(
            stage_geometry=(execution.stage_geometry
                            or stage_geometry),
            alpha_recipe={"base_draft": 1,
                          "repair": chain.repair.menu_id},
            hook=hook_to_dict(execution.hook),
            caption_rect=execution.caption_rect,
            layout_spans=execution.layout_spans,
            hook_slide_px_s=execution.hook_slide_px_s,
            inputs_sha256=chain.drafts[1].inputs_sha256)
        write_bundle(short_dir, bundle_drafts)

    final_path = short_dir / "final.mp4"
    if Path(chain.artifact) != final_path:
        shutil.copyfile(chain.artifact, final_path)
    (short_dir / "receipt.json").write_text(chain.to_json(),
                                            encoding="utf-8")
    cost = round(ledger.total_usd() - cost_before, 4)
    report: dict[str, Any] = {
        "index": index,
        "window": {"start_s": w0, "end_s": w1},
        "candidate": {"score": cand.score, "reason": cand.reason},
        "hook_text": hook_text,
        "hook_call": hook_call,
        "layout": {
            "stage_geometry": stage_geometry,
            "backdrop": canvas_backdrop,
            "hook_y_frac": placement.y_center_frac,
            "predicted_occlusion": round(placement.predicted_occlusion, 4),
            "caption_rect": list(cap_layout.rect),
            "caption_feasible": cap_layout.feasible,
            "effect": {
                "band": list(EFFECT_OCCLUSION_BAND),
                "predicted_status": placement.effect_status,
                "lift_px": effect.lift_px,
                "geometry_ok": effect.geometry_ok,
                "note": effect.note,
            },
        },
        "audio_normalisation": norm,
        "speaker_reframe": {
            "pad_bottom_px": reframe.pad_bottom_px,
            "applied_pad_bottom_px": effect.pad_bottom_px,
            "placed_centroid": reframe.placed_centroid,
            "placed_head_top_min": reframe.placed_head_top_min,
            "feasible": reframe.feasible,
            "identity": reframe.identity,
            "note": reframe.note,
        },
        "draft_1": {"verdict": receipt1.verdict,
                    "failed_checks": receipt1.failed},
        "repair": chain.repair.to_dict(),
        "final_verdict": chain.final_verdict,
        "cost_usd": cost,
        "files": {"final": str(final_path),
                  "receipt": str(short_dir / "receipt.json")},
    }
    if graphics_report is not None:
        report["graphics"] = graphics_report
    return report


def run_pipeline(video_path: str | Path, out_dir: str | Path, *,
                 sam: SamClient, spark_open: SparkOpenClient,
                 spark: SparkClient, vt: VoiceTranscribeClient,
                 wonder: WonderClient, ledger: SpendLedger,
                 upload_file: Callable[[str | Path], str],
                 ts: str = "", max_shorts: int = 3,
                 canvas: tuple[int, int] = DEFAULT_CANVAS,
                 log: Callable[[str], None] = print,
                 image: ImageClient | None = None,
                 graphics: bool = False) -> dict[str, Any]:
    """Run the whole pipeline on one long video; returns the summary.

    ``graphics`` (default False) enables the context graphics layer
    per Short; False runs exactly the pre-graphics pipeline.
    """
    started = time.monotonic()
    video = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clients = _Clients(sam=sam, spark_open=spark_open, spark=spark,
                       vt=vt, wonder=wonder, ledger=ledger,
                       upload_file=upload_file, image=image)
    total_before = ledger.total_usd()
    summary: dict[str, Any] = {
        "source": str(video), "created": ts, "max_shorts": max_shorts,
        "shorts": [],
    }
    try:
        turns, duration = _transcribe_all(video, out, clients,
                                          ts=ts, log=log)
        shared_before_sweep = ledger.total_usd()
        sweep_turns = []
        for t in turns:
            start = (t.start_ms or 0) / 1000.0
            end = (t.end_ms or 0) / 1000.0
            if end <= start:
                end = start + 5.0
            if t.text.strip():
                sweep_turns.append({"start_s": start, "end_s": end,
                                    "text": t.text})
        sweep = spark_open.sweep(
            sweep_turns, purpose="pipeline-candidate-sweep", ts=ts)
        log(f"sweep: {len(sweep.candidates)} candidates from "
            f"{len(sweep_turns)} turns over {duration:.0f}s")
        summary["sweep"] = {
            "candidates": [
                {"start_s": c.start_s, "end_s": c.end_s,
                 "score": c.score, "reason": c.reason}
                for c in sweep.candidates],
            "request_id": sweep.request_id,
        }
        # VT is charged to the run; the sweep draws partner credit
        # (not the Meta ledger), so the shared Meta cost is VT only.
        summary["shared_cost_usd"] = round(
            shared_before_sweep - total_before, 4)
        chosen = select_windows(sweep.candidates, duration, max_shorts)
        log(f"selected {len(chosen)} windows")
        if graphics:
            summary["graphics_enabled"] = True
        # SAM-coverage guard loop (fix phase, 2026-11-02): a
        # window the guard rejects is recorded on the summary
        # with its measured coverage and replaced by the next
        # untried sweep candidate; when the pool runs out the
        # run proceeds with fewer Shorts and says so - a window
        # is never fabricated to fill the count.
        pending = list(chosen)
        reserved = [(w0, w1) for _c, w0, w1 in chosen]
        tried: set[tuple[float, float]] = set()
        rejections: list[dict[str, Any]] = []
        attempt = 0
        while pending and len(summary["shorts"]) < max_shorts:
            cand, w0, w1 = pending.pop(0)
            tried.add((round(w0, 3), round(w1, 3)))
            attempt += 1
            try:
                summary["shorts"].append(_process_short(
                    attempt, cand, w0, w1, video, turns, out,
                    clients, ts=ts, canvas=canvas, log=log,
                    graphics_enabled=graphics))
            except SamCoverageError as exc:
                rejections.append({
                    "index": attempt,
                    "window": {"start_s": w0, "end_s": w1},
                    "candidate": {"score": cand.score,
                                  "reason": cand.reason},
                    "parsed_frames": exc.parsed_frames,
                    "window_frames": exc.window_frames,
                    "sam_coverage": round(exc.coverage, 4),
                    "reason": str(exc),
                })
                log(f"short {attempt}: window rejected by the "
                    f"SAM coverage guard - {exc}")
                replacement = _next_untried_candidate(
                    sweep.candidates, duration, reserved, tried)
                if replacement is not None:
                    pending.append(replacement)
                    reserved.append((replacement[1], replacement[2]))
                    log(f"coverage guard: next candidate "
                        f"{replacement[1]:.1f}-{replacement[2]:.1f}s "
                        "queued as a replacement window")
        if rejections:
            summary["window_rejections"] = rejections
            if len(summary["shorts"]) < max_shorts:
                log(f"coverage guard: candidate pool exhausted - "
                    f"run proceeds with {len(summary['shorts'])} "
                    f"of {max_shorts} Shorts "
                    f"({len(rejections)} window(s) rejected)")
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["door_to_door_s"] = round(time.monotonic() - started, 1)
        summary["total_cost_usd"] = round(
            ledger.total_usd() - total_before, 4)
        (out / "run-summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        raise
    summary["door_to_door_s"] = round(time.monotonic() - started, 1)
    summary["total_cost_usd"] = round(ledger.total_usd() - total_before, 4)
    summary["ledger_total_usd"] = ledger.total_usd()
    (out / "run-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log(f"run complete: {len(summary['shorts'])} shorts, "
        f"${summary['total_cost_usd']:.4f} Meta spend, "
        f"{summary['door_to_door_s']:.0f}s door to door")
    return summary


# ---------------------------------------------------------------- CLI wiring


def _no_redirect_transport(method, url, headers, body):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            raise RuntimeError("redirect refused")

    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1200]
        return exc.code, {"error_body": detail}


def _vt_transport(url, headers, body):
    return _no_redirect_transport("POST", url, headers, body)


def _key_provider(env_var: str, vault_provider: str):
    """Env key if set, else the Secure Vault surrogate for the provider."""
    env_key = os.environ.get(env_var)
    if env_key:
        return lambda: env_key
    sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
    from dynamic_credentials import dynamic_credential_entry
    return lambda: str(
        dynamic_credential_entry(vault_provider)["surrogate"]).strip()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-shorts", type=int, default=3)
    parser.add_argument(
        "--graphics", action="store_true",
        help="enable the context graphics layer (timed cards/images "
             "on the backdrop plane, F8-verified; also enabled by "
             "GRAPHICS_LAYER=1). Default off.")
    parser.add_argument("--ts", default="2026-10-29")
    parser.add_argument(
        "--ledger", type=Path,
        default=Path(__file__).resolve().parents[2] / "spend-ledger.jsonl")
    args = parser.parse_args(argv)

    meta_key = _key_provider("SPATIAL_DIRECTOR_META_API_KEY",
                             "custom.meta-model-api")
    baseten_key = _key_provider("SPATIAL_DIRECTOR_BASETEN_API_KEY",
                                "custom.baseten")
    ledger = SpendLedger(args.ledger)
    repo_root = Path(__file__).resolve().parents[2]
    summary = run_pipeline(
        args.video, args.out,
        sam=SamClient(api_key_provider=meta_key, ledger=ledger,
                      transport=_no_redirect_transport),
        spark_open=SparkOpenClient(
            api_key_provider=baseten_key, transport=_no_redirect_transport,
            usage_log_path=repo_root / "partner-usage-log.jsonl"),
        spark=SparkClient(api_key_provider=meta_key, ledger=ledger,
                          transport=_no_redirect_transport),
        vt=VoiceTranscribeClient(api_key_provider=meta_key, ledger=ledger,
                                 transport=_vt_transport),
        wonder=WonderClient(api_key_provider=meta_key, ledger=ledger,
                            transport=_no_redirect_transport),
        image=ImageClient(api_key_provider=meta_key, ledger=ledger,
                          transport=_no_redirect_transport),
        ledger=ledger,
        upload_file=lambda p: upload_media_file(
            p, api_key_provider=meta_key),
        ts=args.ts, max_shorts=args.max_shorts,
        graphics=args.graphics
        or os.environ.get("GRAPHICS_LAYER") == "1")
    print(json.dumps({
        "shorts": len(summary["shorts"]),
        "verdicts": [s["final_verdict"] for s in summary["shorts"]],
        "total_cost_usd": summary["total_cost_usd"],
        "door_to_door_s": summary["door_to_door_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
