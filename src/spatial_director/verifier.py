"""The F1-F7 verifier: deterministic checks first, one judge second.

This is the gate the whole product is named for. It runs on the
ACTUAL render (never a proxy, never the plan), in the order the
Agents-102 verification session teaches and the design adopts:

1. **Deterministic checks first.** F1-F7 are measurements - pixels,
   masks, DSP energy, string patterns - each returning the measured
   value next to its threshold. A failed check names the check, the
   measurement, and the threshold it missed: feedback a repair step
   can act on, not a vibe.
2. **One Wonder call second.** Wonder watches the full-res first 5s
   of the render, muted, and must summarise the hook (F5's
   mute-comprehension sub-check). It judges legibility/premise -
   the one thing geometry cannot measure - and nothing else.

The verifier is a fixed pipeline stage, not a tool anything can
choose to skip: the receipt is issued here, in the delivery path,
and a render does not ship without one. Verdicts: all seven checks
pass on draft 1 -> ``PASSED_FIRST_DRAFT``; all pass on a later draft
-> ``PASSED_AFTER_REPAIR``; anything else -> ``FAIL`` with the
failed checks named. A PASS is a valid outcome; failures are never
manufactured, and every number in the receipt is a measurement made
by this module - none is asserted by a model.

Check definitions (the decision pack's research taxonomy as
operationalised by the converged design; every place the sources
disagree is reconciled in the BUILDLOG and in layout_law):

- F1 hook in frame 1: hook present at frame 0, rendered opacity
  >= 0.90, >= 3 content words, audio RMS (0-2s) > -32 dBFS, first
  speech onset <= 2.0s.
- F2 face/mouth obstruction: mean hook-glyph occlusion by the
  speaker <= 25% over the hook window (an alpha-weighted variant is
  measured and reported alongside - the feathered matte ghosts text
  through the cap brim at partial opacity, which the binary measure
  understates), and caption-block intersection with face pixels
  == 0 on every cue.
- F3 safe zone: hook bbox inside the central band, caption block
  inside the lower zone, face centroid inside the central band.
- F4 subject containment: >= 98% of frames with the head unclipped
  by the canvas edges, and worst-frame head-top margin >= 90px.
- F5 dead air / mute-incomprehensible: max silence gap inside the
  speech span <= 0.45s, onset <= 2.0s, captions present and covering
  the speech, and Wonder can summarise the muted hook.
- F6 open loop by 3s: hook text or the first 3s of captions carries
  a conflict word, a number, or a question.
- F7 static opening: speaker scale change >= 6% or hook-card slide
  > 120px/s in 0-1.5s, or clear inter-frame motion across 0-3s
  (mean mask IoU < 0.999 - a talking head passes on motion; a frozen
  frame fails every way it is measured).

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .caption_align import CaptionCue, EnergyProfile, active_segments
from .compositor import DEFAULT_CANVAS, hook_layer
from .layout_law import (
    CAPTION_BOTTOM_MARGIN,
    GLYPH_OCCLUSION_CAP,
    SAFE_X0,
    SAFE_X1,
    SAFE_Y0,
    SAFE_Y1,
    face_pixels,
    frame_geometry,
    hook_glyph_mask,
    rect_overlap_px,
    render_hook_layer,
)
from .temporal_smooth import mask_iou
from .wonder_client import WonderClient

# ---------------------------------------------------------------- thresholds

HOOK_OPACITY_MIN = 0.90
AUDIO_RMS_MIN_DB = -32.0
HOOK_MIN_CONTENT_WORDS = 3
SPEECH_ONSET_DEADLINE_S = 2.0
CAPTION_FACE_OVERLAP_MAX_PX = 0
CONTAINMENT_MIN_FRAC = 0.98
HEAD_MARGIN_MIN_PX = 90
MAX_WORD_GAP_S = 0.45
CAPTION_COVERAGE_MIN = 0.5
ZOOM_MIN_FRAC = 0.06
SLIDE_MIN_PX_S = 120.0
MOTION_IOU_CEILING = 0.999  # mean IoU at/above this reads as frozen
GHOSTING_DELTA = 0.05       # alpha-weighted minus geometric occlusion

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it",
    "its", "on", "at", "as", "be", "by", "for", "with",
}
_CONFLICT_WORDS = {
    "never", "no", "not", "stopped", "quit", "banned", "wrong",
    "mistake", "secret", "truth", "actually", "refused", "fired",
    "broke", "vs", "versus", "hate", "lied", "failed", "dead",
    "worst", "shocking", "hidden", "nobody", "nothing", "everything",
}
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "hundred", "thousand",
    "million", "first", "second", "third",
}
_QUESTION_STARTERS = {
    "who", "what", "why", "how", "when", "which", "where", "can",
    "does", "did", "is", "are", "would", "could",
}


# ---------------------------------------------------------------- inputs


@dataclass
class HookSpec:
    """The hook as rendered: text, window, band centre, render path."""

    text: str
    window: tuple[float, float]
    y_center_frac: float
    layout_rendered: bool = False  # True = layout_law safe-width layer

    def layer(self, canvas: tuple[int, int] = DEFAULT_CANVAS) -> np.ndarray:
        if self.layout_rendered:
            return render_hook_layer(
                self.text, canvas, y_center_frac=self.y_center_frac)
        return hook_layer(self.text, canvas, y_center_frac=self.y_center_frac)

    def glyph(self, canvas: tuple[int, int] = DEFAULT_CANVAS):
        if self.layout_rendered:
            return hook_glyph_mask(
                self.text, canvas, y_center_frac=self.y_center_frac)
        layer = self.layer(canvas)
        mask = layer[..., 3] > 0
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            raise ValueError(f"hook text rendered no glyphs: {self.text!r}")
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1)
        return mask, bbox


@dataclass
class VerifyInputs:
    video_path: str | Path
    fps: float
    canvas_alphas: Sequence[np.ndarray]  # canvas-space, dense, frame-aligned
    audio: EnergyProfile                # energy profile of the RENDER
    hook: HookSpec | None = None
    cues: Sequence[CaptionCue] = ()
    caption_rect: tuple[int, int, int, int] | None = None
    sam_response_path: str | Path | None = None
    hook_slide_px_s: float = 0.0
    canvas: tuple[int, int] = DEFAULT_CANVAS


@dataclass
class CheckResult:
    check_id: str
    name: str
    passed: bool
    measured: dict[str, Any]
    threshold: dict[str, Any]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "name": self.name,
            "pass": self.passed,
            "measured": self.measured,
            "threshold": self.threshold,
            "note": self.note,
        }


@dataclass
class Receipt:
    artifact: str
    verdict: str
    checks: list[CheckResult]
    meta_calls: list[dict[str, Any]] = field(default_factory=list)
    inputs_sha256: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    draft: int = 1
    created: str = ""

    @property
    def failed(self) -> list[str]:
        return [c.check_id for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": 1,
            "artifact": self.artifact,
            "created": self.created,
            "draft": self.draft,
            "verdict": self.verdict,
            "failed_checks": self.failed,
            "checks": [c.to_dict() for c in self.checks],
            "meta_calls": self.meta_calls,
            "inputs_sha256": self.inputs_sha256,
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------- helpers


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def extract_muted_segment(video_path: str | Path, seconds: float = 5.0,
                          out_path: str | Path | None = None) -> Path:
    """The first ``seconds`` of a render with the audio STRIPPED.

    Wonder must judge the muted hook; sending the real audio track
    would let it cheat on a check whose entire point is sound-off
    comprehension.
    """
    out = Path(out_path) if out_path else Path(
        tempfile.mkstemp(suffix=".mp4", prefix="muted-hook-")[1])
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-t", str(seconds), "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not extract the muted segment: "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}")
    return out


def _rms_db(profile: EnergyProfile, t0: float, t1: float) -> float:
    sel = (profile.times >= t0) & (profile.times < t1)
    if not sel.any():
        return -90.0
    linear = np.mean(10.0 ** (profile.db[sel] / 10.0))
    return round(float(10.0 * np.log10(max(linear, 1e-9))), 2)


def _speech_span(profile: EnergyProfile):
    """(onset, offset, max_gap) of speech, gaps measured unmerged."""
    segs = active_segments(profile, float(profile.times[0]),
                           float(profile.times[-1]) + profile.hop_s,
                           min_silence_s=0.0)
    if not segs:
        return None, None, None
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    return segs[0][0], segs[-1][1], (max(gaps) if gaps else 0.0)


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


# ---------------------------------------------------------------- the gate


def verify(inputs: VerifyInputs, *,
           wonder: WonderClient | None = None,
           draft: int = 1,
           ts: str = "") -> Receipt:
    """Run F1-F7 over one render and issue its receipt.

    All deterministic checks always run (a failure never
    short-circuits the rest - the receipt reports everything, so a
    repair step sees the full picture). The Wonder judge runs when a
    client is supplied; without one, F5 records the judge as not
    run and fails - a receipt may not claim a judgement nobody made.
    """
    cw, ch = inputs.canvas
    fps = inputs.fps
    alphas = list(inputs.canvas_alphas)
    n_frames = len(alphas)
    geos = [frame_geometry(a) for a in alphas]
    checks: list[CheckResult] = []
    meta_calls: list[dict[str, Any]] = []
    notes: list[str] = []

    def frames_in(t0: float, t1: float) -> range:
        f0 = max(0, int(round(t0 * fps)))
        f1 = min(n_frames, max(f0 + 1, int(round(t1 * fps))))
        return range(f0, f1)

    # ---- F1: hook in frame 1 -------------------------------------
    onset, offset, _ = _speech_span(inputs.audio)
    if inputs.hook is not None:
        layer = inputs.hook.layer(inputs.canvas)
        glyph_count = int((layer[..., 3] > 0).sum())
        opacity = float(layer[..., 3].max() / 255.0) if glyph_count else 0.0
        content = _content_words(inputs.hook.text)
        present_at_0 = inputs.hook.window[0] <= 0.0 and glyph_count > 0
        by_1s = inputs.hook.window[0] <= 1.0
    else:
        glyph_count, opacity, content = 0, 0.0, []
        present_at_0 = by_1s = False
    rms_0_2 = _rms_db(inputs.audio, 0.0, 2.0)
    f1_pass = bool(
        present_at_0 and by_1s and opacity >= HOOK_OPACITY_MIN
        and len(content) >= HOOK_MIN_CONTENT_WORDS
        and rms_0_2 > AUDIO_RMS_MIN_DB
        and onset is not None and onset <= SPEECH_ONSET_DEADLINE_S
    )
    checks.append(CheckResult(
        check_id="F1", name="hook in frame 1", passed=f1_pass,
        measured={
            "hook_present_at_frame_0": present_at_0,
            "hook_present_by_1s": by_1s,
            "hook_opacity": round(opacity, 3),
            "content_words": len(content),
            "audio_rms_db_0_2s": rms_0_2,
            "first_speech_onset_s": None if onset is None else round(onset, 3),
        },
        threshold={
            "opacity_min": HOOK_OPACITY_MIN,
            "content_words_min": HOOK_MIN_CONTENT_WORDS,
            "audio_rms_db_min": AUDIO_RMS_MIN_DB,
            "first_speech_onset_max_s": SPEECH_ONSET_DEADLINE_S,
        },
    ))

    # ---- F2: face/mouth obstruction -------------------------------
    occ_mean = occ_max = aw_mean = 0.0
    if inputs.hook is not None:
        glyph, _ = inputs.hook.glyph(inputs.canvas)
        glyph_n = int(glyph.sum())
        occs, aws = [], []
        for f in frames_in(*inputs.hook.window):
            a = alphas[f]
            occs.append(float(((a > 0.5) & glyph).sum()) / glyph_n)
            aws.append(float((a * glyph).sum()) / glyph_n)
        occ_mean = round(float(np.mean(occs)), 4) if occs else 0.0
        occ_max = round(float(np.max(occs)), 4) if occs else 0.0
        aw_mean = round(float(np.mean(aws)), 4) if aws else 0.0
    caption_overlap = 0
    if inputs.cues and inputs.caption_rect is not None:
        for cue in inputs.cues:
            for f in frames_in(cue.start_s, cue.end_s):
                caption_overlap = max(
                    caption_overlap,
                    rect_overlap_px(face_pixels(alphas[f]),
                                    inputs.caption_rect),
                )
    ghosting = aw_mean - occ_mean
    f2_note = ""
    if ghosting > GHOSTING_DELTA:
        f2_note = (
            f"alpha-weighted occlusion ({aw_mean}) exceeds the binary "
            f"geometric measure ({occ_mean}) by {ghosting:.3f}: the "
            "feathered matte lets glyphs ghost through the speaker's "
            "edge (observed at the cap brim) at partial opacity, "
            "which the binary measure understates. The pass/fail "
            "uses the geometric measure per the design cap; the "
            "alpha-weighted figure is recorded so the effect is "
            "visible in the receipt, not hidden by it."
        )
        notes.append(f"F2 ghosting: alpha-weighted occlusion {aw_mean} "
                     f"vs geometric {occ_mean} (delta {ghosting:.3f}).")
    if not inputs.cues:
        f2_note = (f2_note + " " if f2_note else "") + (
            "no captions rendered on this artifact; caption-face "
            "overlap is vacuously 0 - caption presence is F5's check.")
    checks.append(CheckResult(
        check_id="F2", name="face/mouth obstruction",
        passed=bool(occ_mean <= GLYPH_OCCLUSION_CAP
                    and caption_overlap <= CAPTION_FACE_OVERLAP_MAX_PX),
        measured={
            "glyph_occlusion_mean": occ_mean,
            "glyph_occlusion_max": occ_max,
            "glyph_occlusion_alpha_weighted_mean": aw_mean,
            "caption_face_overlap_max_px": caption_overlap,
        },
        threshold={
            "glyph_occlusion_max": GLYPH_OCCLUSION_CAP,
            "caption_face_overlap_max_px": CAPTION_FACE_OVERLAP_MAX_PX,
        },
        note=f2_note,
    ))

    # ---- F3: safe zone ---------------------------------------------
    violations: dict[str, Any] = {}
    if inputs.hook is not None:
        _, hb = inputs.hook.glyph(inputs.canvas)
        violations["hook_px_outside_central_band"] = {
            "left": max(0, SAFE_X0 - hb[0]), "top": max(0, SAFE_Y0 - hb[1]),
            "right": max(0, hb[2] - SAFE_X1), "bottom": max(0, hb[3] - SAFE_Y1),
        }
    if inputs.caption_rect is not None:
        x0, y0, x1, y1 = inputs.caption_rect
        violations["caption_px_outside_lower_zone"] = {
            "left": max(0, SAFE_X0 - x0), "right": max(0, x1 - SAFE_X1),
            "top": max(0, SAFE_Y0 - y0),
            "bottom": max(0, y1 - (ch - CAPTION_BOTTOM_MARGIN)),
        }
    cents = [g.face_centroid for g in geos if g.face_centroid is not None]
    centroid_in = None
    if cents:
        cx = float(np.median([c[0] for c in cents]))
        cy = float(np.median([c[1] for c in cents]))
        centroid_in = bool(SAFE_X0 <= cx <= SAFE_X1
                           and SAFE_Y0 <= cy <= SAFE_Y1)
        violations["face_centroid_median"] = [round(cx, 1), round(cy, 1)]
    f3_pass = centroid_in is not False and all(
        v == 0 for item in violations.values() if isinstance(item, dict)
        for v in item.values()
    )
    checks.append(CheckResult(
        check_id="F3", name="safe zone", passed=bool(f3_pass),
        measured={"violations": violations,
                  "face_centroid_in_central_band": centroid_in},
        threshold={
            "central_band": [SAFE_X0, SAFE_Y0, SAFE_X1, SAFE_Y1],
            "caption_bottom_max": ch - CAPTION_BOTTOM_MARGIN,
        },
    ))

    # ---- F4: subject containment ------------------------------------
    contained = 0
    margins: list[int] = []
    present_frames = 0
    for g, a in zip(geos, alphas):
        if not g.present:
            continue
        present_frames += 1
        margins.append(int(g.head_top))
        face = face_pixels(a)
        ys, xs = np.nonzero(face)
        clipped = bool(
            xs.size and (xs.min() <= 0 or xs.max() >= cw - 1 or ys.min() <= 0)
        )
        if not clipped:
            contained += 1
    contained_frac = round(contained / n_frames, 4) if n_frames else 0.0
    min_margin = min(margins) if margins else None
    checks.append(CheckResult(
        check_id="F4", name="subject containment",
        passed=bool(contained_frac >= CONTAINMENT_MIN_FRAC
                    and min_margin is not None
                    and min_margin >= HEAD_MARGIN_MIN_PX),
        measured={
            "frames_contained_frac": contained_frac,
            "head_margin_min_px": min_margin,
            "frames_with_speaker": present_frames,
            "frames_total": n_frames,
        },
        threshold={
            "frames_contained_min_frac": CONTAINMENT_MIN_FRAC,
            "head_margin_min_px": HEAD_MARGIN_MIN_PX,
        },
        note=("containment is measured on the head region (the "
              "research taxonomy's face-clipping notion): shoulders "
              "and the frame bottom are normal talking-head framing "
              "and are not counted as exits."),
    ))

    # ---- F5: dead air / mute comprehension ---------------------------
    _, _, max_gap = _speech_span(inputs.audio)
    speech_span = None
    coverage = 0.0
    if onset is not None and offset is not None:
        speech_span = offset - onset
        covered = sum(max(0.0, min(c.end_s, offset) - max(c.start_s, onset))
                      for c in inputs.cues)
        coverage = round(covered / speech_span, 3) if speech_span > 0 else 0.0
    captions_present = bool(inputs.cues) and coverage >= CAPTION_COVERAGE_MIN
    judgment = None
    judge_note = ""
    if wonder is not None:
        segment = extract_muted_segment(inputs.video_path, seconds=5.0)
        judgment = wonder.judge_muted_hook(
            segment, purpose="verifier-f5-muted-hook", ts=ts)
        meta_calls.append({
            "model": "muse-wonder-1.0",
            "purpose": "verifier-f5-muted-hook",
            "cost_usd": judgment.cost_usd,
            "request_id": judgment.request_id,
        })
    else:
        judge_note = ("Wonder judge not run (no client supplied): F5 "
                      "cannot pass on DSP alone - a receipt may not "
                      "claim a judgement nobody made.")
    can_summarize = bool(judgment and judgment.can_summarize)
    f5_pass = bool(
        max_gap is not None and max_gap <= MAX_WORD_GAP_S
        and onset is not None and onset <= SPEECH_ONSET_DEADLINE_S
        and captions_present and can_summarize
    )
    checks.append(CheckResult(
        check_id="F5", name="dead air / mute comprehension", passed=f5_pass,
        measured={
            "max_silence_gap_s": None if max_gap is None else round(max_gap, 3),
            "first_speech_onset_s": None if onset is None else round(onset, 3),
            "captions_present": bool(inputs.cues),
            "caption_coverage_of_speech": coverage,
            "wonder_can_summarize_muted_hook": can_summarize,
            "wonder_summary": judgment.summary if judgment else None,
        },
        threshold={
            "max_silence_gap_max_s": MAX_WORD_GAP_S,
            "first_speech_onset_max_s": SPEECH_ONSET_DEADLINE_S,
            "caption_coverage_min": CAPTION_COVERAGE_MIN,
            "wonder_must_summarize": True,
        },
        note=judge_note,
    ))

    # ---- F6: open loop by 3s ------------------------------------------
    early_texts = []
    if inputs.hook is not None and inputs.hook.window[0] < 3.0:
        early_texts.append(("hook", inputs.hook.text))
    for cue in inputs.cues:
        if cue.start_s < 3.0:
            early_texts.append(("caption", cue.text))
    fired: list[dict[str, str]] = []
    for source, text in early_texts:
        words = re.findall(r"[a-z0-9]+", text.lower())
        if re.search(r"\d", text):
            fired.append({"pattern": "number", "source": source,
                          "evidence": text})
        if "?" in text or (words and words[0] in _QUESTION_STARTERS):
            fired.append({"pattern": "question", "source": source,
                          "evidence": text})
        hits = sorted(set(words) & _CONFLICT_WORDS)
        if hits:
            fired.append({"pattern": "conflict", "source": source,
                          "evidence": f"{text} [{', '.join(hits)}]"})
        nums = sorted(set(words) & _NUMBER_WORDS)
        if nums and not re.search(r"\d", text):
            fired.append({"pattern": "number", "source": source,
                          "evidence": f"{text} [{', '.join(nums)}]"})
    checks.append(CheckResult(
        check_id="F6", name="open loop by 3s", passed=bool(fired),
        measured={"patterns_fired": fired},
        threshold={"any_of": ["conflict", "number", "question"],
                   "by_s": 3.0},
    ))

    # ---- F7: static opening ---------------------------------------------
    def _area(f: int) -> int:
        return int((alphas[f] > 0.5).sum())

    zoom = 0.0
    win = frames_in(0.0, 1.5)
    if len(win) >= 2:
        first = float(np.median([_area(f) for f in list(win)[:3]]))
        last = float(np.median([_area(f) for f in list(win)[-3:]]))
        zoom = round(last / first - 1.0, 4) if first > 0 else 0.0
    ious = []
    win3 = frames_in(0.0, 3.0)
    seq = list(win3)
    for a, b in zip(seq, seq[1:]):
        ious.append(mask_iou(alphas[a], alphas[b]))
    mean_iou = round(float(np.mean(ious)), 4) if ious else 1.0
    checks.append(CheckResult(
        check_id="F7", name="static opening",
        passed=bool(zoom >= ZOOM_MIN_FRAC
                    or inputs.hook_slide_px_s > SLIDE_MIN_PX_S
                    or mean_iou < MOTION_IOU_CEILING),
        measured={
            "speaker_scale_change_0_1p5s": zoom,
            "hook_slide_px_s": inputs.hook_slide_px_s,
            "mean_mask_iou_0_3s": mean_iou,
        },
        threshold={
            "zoom_min_frac": ZOOM_MIN_FRAC,
            "slide_min_px_s": SLIDE_MIN_PX_S,
            "motion_iou_ceiling": MOTION_IOU_CEILING,
        },
        note=("pass = a designed motion signal (zoom or card slide) "
              "OR measured inter-frame motion clearly above frozen "
              "(the research taxonomy fails only a truly static "
              "opening); all three measurements are reported."),
    ))

    # ---- receipt ---------------------------------------------------------
    receipt = Receipt(
        artifact=str(inputs.video_path),
        verdict="",  # set below
        checks=checks,
        meta_calls=meta_calls,
        inputs_sha256={
            "video": _sha256_file(inputs.video_path)
            if Path(inputs.video_path).exists() else None,
            "sam_response": _sha256_file(inputs.sam_response_path)
            if inputs.sam_response_path else None,
            "captions": _sha256_json(
                [{"text": c.text, "start_s": c.start_s, "end_s": c.end_s}
                 for c in inputs.cues]),
        },
        notes=notes,
        draft=draft,
        created=ts,
    )
    if not receipt.failed:
        receipt.verdict = ("PASSED_FIRST_DRAFT" if draft == 1
                           else "PASSED_AFTER_REPAIR")
    else:
        receipt.verdict = "FAIL"
    return receipt
