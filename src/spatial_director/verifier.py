"""The F1-F9 verifier (F8 only for graphics renders; F9 on every
render): deterministic checks first, one judge second.

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
- F8 context graphics (only on renders the graphics layer was
  part of): per graphic - zero overlap with the face region plus
  its 28px margin, the caption block/zone, and the hook glyph
  zone while the hook is up; pixel-measured presence inside its
  span and absence outside it (+-0.5s); across graphics, no two
  graphics whose spans overlap in time may have intersecting
  boxes. Timing is otherwise free (2026-10-31 ruling): graphics
  may appear from 0s and back-to-back - there is no hook-window
  ban and no minimum gap. No count cap: the semantic bar limits
  the number, F8 certifies what was placed.
- Block 2 (2026-10-31): F8 additionally certifies path motions
  (a travelling graphic is pixel-matched DURING its travel, and
  its zone legs measure the swept path union), backdrop spans
  (present only inside their declared span), and declared SFX
  (an effect landing in a speech gap must measurably lift the
  mix over the stem). F5 measures dead air on the pre-mix speech
  stem whenever the shipped audio carries mixed SFX, so an
  effect can never mask a gap.
- F9 platform loudness (2026-10-31): the FINAL artifact's audio,
  measured with ffmpeg's loudnorm analysis pass (EBU R128) -
  integrated loudness must sit at the platform normalisation
  target -14 LUFS +/- 1 (pass band -15.0 to -13.0) and the true
  peak at or below -1.0 dBTP. F9 runs on EVERY render, graphics
  or not: the pipeline's relative speech normalisation (speech
  RMS -> -20 dB) is a per-window levelling for F1, not a
  platform loudness guarantee, and until F9 nothing measured a
  render against the level YouTube/Shorts will normalise it to.
  The measured integrated / true peak / LRA are recorded next
  to the thresholds, like every other check.

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from .caption_align import CaptionCue, EnergyProfile, active_segments
from .compositor import (DEFAULT_CANVAS, hook_layer, hook_slide_dx_at,
                         shift_layer_x)
from .graphics_layer import (
    FACE_MARGIN_PX,
    PATH_TRAVEL_S,
    BackdropSpec,
    GraphicSpec,
)
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

# F8 (context graphics) thresholds. The overlap rules are hard
# (0 px: face + its margin, caption block/zone, hook glyph zone,
# and pairwise box collisions between time-overlapping graphics);
# presence is measured on the render's own pixels against the
# graphic's deterministic layer_at(t) state, with H.264-tolerant
# matching. Timing itself is NOT a rule since the 2026-10-31
# ruling (no hook-window ban, no minimum gap).
GRAPHIC_FACE_OVERLAP_MAX_PX = 0
GRAPHIC_CAPTION_OVERLAP_MAX_PX = 0
GRAPHIC_HOOK_GLYPH_OVERLAP_MAX_PX = 0
GRAPHIC_COLLIDING_PAIRS_MAX = 0
GRAPHIC_HOOK_ZONE_PAD_PX = 12  # mirrors placement's glyph-zone pad
GRAPHIC_MATCH_MIN = 0.75        # in-span pixel match (visible part)
GRAPHIC_ABSENT_MATCH_MAX = 0.5  # out-of-span: above this = present
GRAPHIC_SPAN_TOLERANCE_S = 0.5  # presence window vs declared span
GRAPHIC_PIXEL_TOL = 28          # per-channel abs diff counting as same
GRAPHIC_MIN_VISIBLE_PX = 50     # below this a sample is unmeasurable

# F8 SFX leg (block 2): an effect declared at an instant where the
# speech stem is at/below the floor must lift the shipped mix's
# energy at that instant by at least this much - otherwise the
# effect the receipt declares is not in the render.
SFX_GAP_FLOOR_DB = -50.0
SFX_GAP_LIFT_MIN_DB = 6.0

# F9 (platform loudness) thresholds: the level YouTube/Shorts
# normalise uploads to. Integrated loudness is gated at the
# target +/- the tolerance (pass band -15.0 to -13.0 LUFS); the
# true-peak ceiling is a hard cap. LRA is the loudnorm design
# loudness range used by both the analysis pass and the repair;
# it is recorded, not gated.
LOUDNESS_TARGET_LUFS = -14.0
LOUDNESS_TOLERANCE_LU = 1.0
TRUE_PEAK_MAX_DBTP = -1.0
LOUDNESS_LRA_LU = 11.0

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
    # Slide repair (2026-10-30, F7): (dx_start_px, duration_s) -
    # the card starts displaced by dx_start at the window start and
    # eases to its final position over duration_s. None = the
    # historic static hook. The slide speed is a designed value
    # AND a measured one: the renderer composites exactly this
    # offset function, and F7 reads the speed back off it.
    slide: tuple[float, float] | None = None

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

    @property
    def slide_px_s(self) -> float:
        """Designed slide speed in px/s; 0.0 for a static hook."""
        if not self.slide:
            return 0.0
        dx_start, duration = self.slide
        if duration <= 0:
            return 0.0
        return abs(dx_start) / duration

    def glyph_at(self, t_s: float,
                 canvas: tuple[int, int] = DEFAULT_CANVAS):
        """Glyph mask + bbox at ``t_s`` (slide-aware).

        With no slide this is exactly ``glyph()``; with a slide the
        mask is shifted by the same offset function the renderer
        composites, so F2 measures occlusion against the glyphs
        where they actually are at that instant, not where they
        end up.
        """
        mask, bbox = self.glyph(canvas)
        dx = hook_slide_dx_at(t_s, self.window, self.slide)
        if dx == 0.0:
            return mask, bbox
        shifted = shift_layer_x(mask, dx)
        x0, y0, x1, y1 = bbox
        return shifted, (x0 + int(round(dx)), y0,
                         x1 + int(round(dx)), y1)


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
    # Context graphics (2026-10-30): None = the graphics layer was
    # not part of this render (F8 does not run; the receipt is the
    # original F1-F7 shape). A list (possibly empty) = the layer
    # was enabled and these specs are what should be on the render.
    graphics: Sequence[GraphicSpec] | None = None
    # Speech stem (block 2, 2026-10-31): the render's PRE-MIX
    # speech audio profile. When the shipped audio carries mixed
    # SFX, F5's dead-air measurement MUST read this stem, never
    # the mix - an effect filling a gap must not mask dead air.
    # None = the render's audio IS the speech (historic behaviour,
    # measurement identical).
    speech_audio: EnergyProfile | None = None
    # Applied SFX declarations for this render (timestamp, recipe,
    # gain, graphic index) as mixed - present only when the
    # graphics layer's SFX were in play. F8 checks gap-landing
    # effects are measurably present in the mix.
    sfx: list[dict[str, Any]] | None = None
    # Backdrop-mode spans (block 2): the declared backdrop specs
    # for this render. F8's backdrop leg certifies each scene is
    # present inside its span and absent outside it. None = the
    # layer was not in play (an empty list still runs F8 when
    # graphics are in play).
    backdrops: Sequence[BackdropSpec] | None = None
    # Speaker-forward declarations (block 2): the emphasis
    # transforms the pipeline applied, declared for the receipt.
    # F3/F4 verify the render as-is; this section is the record,
    # not a check input.
    speaker_forward: list[dict[str, Any]] | None = None


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
    # The graphics section (per graphic: kind, span, motion, box,
    # source quote or prompt, cost) - present only on renders the
    # graphics layer was part of; None keeps the historic shape.
    graphics: list[dict[str, Any]] | None = None
    # Applied SFX declarations (block 2) - present only on renders
    # whose shipped audio mixes the graphics layer's effects.
    sfx: list[dict[str, Any]] | None = None
    # Backdrop-mode declarations (block 2): per span, the scene
    # prompt as sent, the verbatim quote, and the image cost.
    backdrops: list[dict[str, Any]] | None = None
    # Speaker-forward declarations (block 2): per applied span,
    # the emphasis transform (scale, centroid target) the
    # pipeline planned inside the F3/F4 limits.
    speaker_forward: list[dict[str, Any]] | None = None

    @property
    def failed(self) -> list[str]:
        return [c.check_id for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
        if self.graphics is not None:
            out["graphics"] = self.graphics
        if self.sfx is not None:
            out["sfx"] = self.sfx
        if self.backdrops is not None:
            out["backdrops"] = self.backdrops
        if self.speaker_forward is not None:
            out["speaker_forward"] = self.speaker_forward
        return out

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


def _profile_db_at(profile: EnergyProfile, t_s: float) -> float:
    """The profile's dB of the hop containing ``t_s``."""
    idx = int(np.searchsorted(profile.times, t_s, side="right")) - 1
    idx = min(max(idx, 0), len(profile.db) - 1)
    return float(profile.db[idx])


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


# ---------------------------------------------------------------- F9 loudness


@dataclass
class LoudnessMeasurement:
    """One ffmpeg loudnorm analysis of a media file (EBU R128).

    ``integrated_lufs`` / ``true_peak_dbtp`` / ``lra_lu`` are the
    input_* values of the analysis pass; ``threshold_lufs`` and
    ``target_offset_lu`` are carried because the two-pass repair
    (rescue_loop's NORMALIZE_LOUDNESS) feeds the full measurement
    back into its applying pass. Non-finite readings (digital
    silence measures -inf) surface as None - a receipt must stay
    strict JSON, and -inf is not. ``error`` is set when no
    measurement could be made at all (missing file, no audio
    stream, unparseable output); measuring never raises.
    """

    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    lra_lu: float | None = None
    threshold_lufs: float | None = None
    target_offset_lu: float | None = None
    error: str | None = None


_LOUDNORM_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}")


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def measure_loudness(path: str | Path) -> LoudnessMeasurement:
    """Analyse ``path``'s audio with ffmpeg's loudnorm filter.

    Single analysis pass (``print_format=json``); the JSON block
    loudnorm prints to stderr is the parse - it is structured
    output from the tool itself, not a log scrape. Any failure
    returns a measurement with ``error`` set instead of raising:
    an unmeasurable render is an F9 FAIL with the reason on the
    receipt, not a verifier crash.
    """
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-vn",
         "-af", (f"loudnorm=I={LOUDNESS_TARGET_LUFS:.0f}"
                 f":TP={TRUE_PEAK_MAX_DBTP:.0f}"
                 f":LRA={LOUDNESS_LRA_LU:.0f}:print_format=json"),
         "-f", "null", "-"],
        capture_output=True, text=True)
    blocks = _LOUDNORM_JSON_RE.findall(proc.stderr or "")
    if not blocks:
        return LoudnessMeasurement(error=(
            "ffmpeg loudnorm analysis produced no measurement "
            f"(exit {proc.returncode}; missing file, no audio "
            "stream, or undecodable audio)"))
    try:
        data = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return LoudnessMeasurement(
            error="ffmpeg loudnorm analysis output was not "
                  "parseable JSON")
    return LoudnessMeasurement(
        integrated_lufs=_finite_float(data.get("input_i")),
        true_peak_dbtp=_finite_float(data.get("input_tp")),
        lra_lu=_finite_float(data.get("input_lra")),
        threshold_lufs=_finite_float(data.get("input_thresh")),
        target_offset_lu=_finite_float(data.get("target_offset")))


def _check_loudness(inputs: VerifyInputs) -> CheckResult:
    """F9: platform loudness of the FINAL artifact's audio.

    Measured on the artifact file itself (never the plan, never
    a caller-supplied profile): the shipped mix, SFX included,
    is exactly what a platform would normalise.
    """
    m = measure_loudness(inputs.video_path)
    lo = LOUDNESS_TARGET_LUFS - LOUDNESS_TOLERANCE_LU
    hi = LOUDNESS_TARGET_LUFS + LOUDNESS_TOLERANCE_LU
    passed = bool(
        m.integrated_lufs is not None and lo <= m.integrated_lufs <= hi
        and m.true_peak_dbtp is not None
        and m.true_peak_dbtp <= TRUE_PEAK_MAX_DBTP)
    note = ""
    if m.error:
        note = f"loudness could not be measured: {m.error}"
    elif m.integrated_lufs is None:
        note = ("no measurable integrated loudness (digital "
                "silence or an empty audio stream): a silent "
                "render is not at platform loudness")
    return CheckResult(
        check_id="F9", name="platform loudness", passed=passed,
        measured={
            "integrated_lufs": (None if m.integrated_lufs is None
                                else round(m.integrated_lufs, 2)),
            "true_peak_dbtp": (None if m.true_peak_dbtp is None
                               else round(m.true_peak_dbtp, 2)),
            "lra_lu": (None if m.lra_lu is None
                       else round(m.lra_lu, 2)),
            "measurement_error": m.error,
        },
        threshold={
            "integrated_target_lufs": LOUDNESS_TARGET_LUFS,
            "integrated_min_lufs": lo,
            "integrated_max_lufs": hi,
            "true_peak_max_dbtp": TRUE_PEAK_MAX_DBTP,
        },
        note=note,
    )


# ---------------------------------------------------------------- F8


def _grab_frame(video_path: str | Path, t_s: float,
                canvas: tuple[int, int]) -> np.ndarray | None:
    """One frame of the render at ``t_s`` (RGB uint8), or None.

    Input seeking + re-encode is frame-accurate for the frame at
    the requested time; F8 samples at least 0.25s inside a span
    edge, far from any boundary ambiguity.
    """
    cw, ch = canvas
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_s):.3f}",
         "-i", str(video_path), "-frames:v", "1",
         "-s", f"{cw}x{ch}", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "pipe:1"],
        capture_output=True)
    need = cw * ch * 3
    if proc.returncode != 0 or len(proc.stdout) < need:
        return None
    return np.frombuffer(proc.stdout[:need], dtype=np.uint8) \
        .reshape(ch, cw, 3).copy()


def _graphic_match(render_rgb: np.ndarray, spec: GraphicSpec, t_s: float,
                   speaker_alpha: np.ndarray) -> float | None:
    """Fraction of the graphic's VISIBLE pixels the render matches.

    Visible = the graphic layer is opaque there AND the speaker
    (who renders in front) does not cover it. The comparison layer
    is ``spec.layer_at(t_s)`` - the same deterministic state the
    compositor drew, animation included. None when too little of
    the graphic is visible to measure honestly.
    """
    layer = spec.layer_at(t_s)
    visible = (layer[..., 3] > 127) & (speaker_alpha <= 0.5)
    if int(visible.sum()) < GRAPHIC_MIN_VISIBLE_PX:
        return None
    diff = np.abs(
        render_rgb.astype(np.int16) - layer[..., :3].astype(np.int16)
    ).max(axis=2)
    return float((diff[visible] <= GRAPHIC_PIXEL_TOL).mean())


def _backdrop_match(render_rgb: np.ndarray, layer_rgb: np.ndarray,
                    speaker_alpha: np.ndarray,
                    exclude: np.ndarray) -> float | None:
    """Fraction of TESTABLE pixels matching the declared backdrop.

    The backdrop covers the whole canvas behind everything, so a
    pixel is testable when the speaker (rendered in front) does
    not cover it and it is not excluded (caption block, hook
    glyphs, a graphic on the backdrop plane). None when too
    little is testable to measure honestly - the same floor the
    graphic match uses.
    """
    testable = (speaker_alpha <= 0.5) & ~exclude
    if int(testable.sum()) < GRAPHIC_MIN_VISIBLE_PX:
        return None
    diff = np.abs(
        render_rgb.astype(np.int16) - layer_rgb.astype(np.int16)
    ).max(axis=2)
    return float((diff[testable] <= GRAPHIC_PIXEL_TOL).mean())


def _check_graphics(inputs: VerifyInputs,
                    alphas: Sequence[np.ndarray]) -> CheckResult:
    """F8: context graphics integrity, measured on the render.

    Per graphic: zero overlap with the face region plus its
    FACE_MARGIN_PX halo (the same zone placement enforces),
    with the caption block/zone, and with the hook glyph zone
    (glyph bbox + pad, slide-aware) whenever its span overlaps
    the hook window; present (pixel match) inside its span and
    absent outside it, within the 0.5s tolerance. Across
    graphics: two graphics whose spans overlap in time must not
    have intersecting boxes. Timing is otherwise FREE
    (2026-10-31 ruling, reversing 2026-10-30): a graphic may sit
    inside the hook window and graphics may run back-to-back -
    collision, not spacing, is the discipline. There is NO count
    cap: the semantic bar, not a number, limits how many
    graphics a render carries.

    Backdrop leg (block 2): each declared backdrop-mode scene is
    pixel-matched at its span's midpoint over the pixels the
    speaker, captions, hook glyphs and graphics do not cover,
    and must NOT match at samples beyond its span edges - the
    scene exists only inside its declared span.
    """
    specs = list(inputs.graphics or [])
    cw, ch = inputs.canvas
    fps = inputs.fps
    n_frames = len(alphas)
    duration = n_frames / fps if fps else 0.0
    zone_top = ch - CAPTION_BOTTOM_MARGIN
    grabbed: dict[int, np.ndarray | None] = {}

    def frame_at(idx: int) -> np.ndarray | None:
        if idx not in grabbed:
            grabbed[idx] = _grab_frame(
                inputs.video_path, idx / fps, inputs.canvas)
        return grabbed[idx]

    def match_at(idx: int, spec: GraphicSpec) -> float | None:
        if idx < 0 or idx >= n_frames:
            return None
        rgb = frame_at(idx)
        if rgb is None:
            return None
        return _graphic_match(rgb, spec, idx / fps, alphas[idx])

    per_graphic: list[dict[str, Any]] = []
    offending: list[int] = []
    for i, spec in enumerate(specs):
        gmask = spec.mask()
        # Face + margin overlap: the union of face pixels over
        # the span's sampled frames, dilated by FACE_MARGIN_PX -
        # exactly the exclusion zone placement enforces, so F8
        # certifies the guarantee placement claims.
        f0 = max(0, int(spec.start_s * fps))
        f1 = min(n_frames, max(f0 + 1, int(np.ceil(spec.end_s * fps))))
        step = max(1, (f1 - f0) // 5)
        face_union = np.zeros((ch, cw), dtype=bool)
        for f in range(f0, f1, step):
            face_union |= face_pixels(alphas[f])
        face_overlap = 0
        if face_union.any():
            dist = ndimage.distance_transform_edt(~face_union)
            face_overlap = int(
                (gmask & (dist <= FACE_MARGIN_PX)).sum())
        caption_overlap = (
            rect_overlap_px(gmask, inputs.caption_rect)
            if inputs.caption_rect is not None else 0)
        zone_overlap = int(gmask[zone_top:, :].sum())
        # Hook glyph zone: only while the hook is actually up
        # (the span's intersection with the hook window). The
        # zone is the union of the glyphs' slide-aware bboxes
        # over the sampled frames, padded exactly as placement
        # pads it. Outside the window there are no glyphs to hit.
        hook_glyph_overlap = 0
        if inputs.hook is not None:
            w0, w1 = inputs.hook.window
            lo_t, hi_t = max(spec.start_s, w0), min(spec.end_s, w1)
            if lo_t < hi_t:
                g0 = max(0, int(lo_t * fps))
                g1 = min(n_frames,
                         max(g0 + 1, int(np.ceil(hi_t * fps))))
                gstep = max(1, (g1 - g0) // 5)
                zone_rect = None
                for f in range(g0, g1, gstep):
                    try:
                        _, gbbox = inputs.hook.glyph_at(
                            f / fps, inputs.canvas)
                    except ValueError:
                        continue  # no glyphs = no zone to hit
                    rect = (
                        max(0, gbbox[0] - GRAPHIC_HOOK_ZONE_PAD_PX),
                        max(0, gbbox[1] - GRAPHIC_HOOK_ZONE_PAD_PX),
                        min(cw, gbbox[2] + GRAPHIC_HOOK_ZONE_PAD_PX),
                        min(ch, gbbox[3] + GRAPHIC_HOOK_ZONE_PAD_PX))
                    zone_rect = rect if zone_rect is None else (
                        min(zone_rect[0], rect[0]),
                        min(zone_rect[1], rect[1]),
                        max(zone_rect[2], rect[2]),
                        max(zone_rect[3], rect[3]))
                if zone_rect is not None:
                    hook_glyph_overlap = rect_overlap_px(
                        gmask, zone_rect)
        # Presence: in-span samples sit >= 0.25s inside the edges
        # (past the fade); out-of-span samples sit TOLERANCE + 0.25s
        # beyond the edges, when the render has room there.
        lo = int(np.ceil((spec.start_s + 0.25) * fps))
        hi = int(np.floor((spec.end_s - 0.25) * fps))
        in_idx = [x for x in (lo, (lo + hi) // 2, hi)
                  if lo <= x <= hi and 0 <= x < n_frames]
        if not in_idx and n_frames:
            mid = min(n_frames - 1,
                      max(0, int((spec.start_s + spec.end_s)
                                 / 2.0 * fps)))
            in_idx = [mid]
        in_matches = [m for m in (match_at(x, spec) for x in in_idx)
                      if m is not None]
        presence = min(in_matches) if in_matches else None
        out_idx = []
        before = int(np.floor(
            (spec.start_s - GRAPHIC_SPAN_TOLERANCE_S - 0.25) * fps))
        after = int(np.ceil(
            (spec.end_s + GRAPHIC_SPAN_TOLERANCE_S + 0.25) * fps))
        if before >= 0:
            out_idx.append(before)
        if after < n_frames:
            out_idx.append(after)
        out_matches = [m for m in (match_at(x, spec) for x in out_idx)
                       if m is not None]
        absence = max(out_matches) if out_matches else None
        # Path leg (block 2): a travelling graphic is pixel-matched
        # DURING its travel - at 35% of the declared travel time,
        # deep enough in that the box is measurably displaced from
        # its rest position - against the same layer_at(t) state
        # the renderer drew. A render that parks the graphic at
        # its resting box (or anywhere off-path) mismatches there
        # even though its resting presence matches. The zone legs
        # above measure spec.mask(), which for a path spec is the
        # swept union, so a path crossing a forbidden zone fails
        # them at any point of its travel.
        path_match = None
        if spec.path is not None:
            idx_travel = int(round(
                (spec.start_s + 0.35 * PATH_TRAVEL_S) * fps))
            if 0 <= idx_travel < n_frames:
                path_match = match_at(idx_travel, spec)
        ok = bool(
            face_overlap <= GRAPHIC_FACE_OVERLAP_MAX_PX
            and caption_overlap <= GRAPHIC_CAPTION_OVERLAP_MAX_PX
            and zone_overlap <= GRAPHIC_CAPTION_OVERLAP_MAX_PX
            and hook_glyph_overlap <= GRAPHIC_HOOK_GLYPH_OVERLAP_MAX_PX
            and presence is not None and presence >= GRAPHIC_MATCH_MIN
            and (absence is None or absence <= GRAPHIC_ABSENT_MATCH_MAX)
            and (path_match is None or path_match >= GRAPHIC_MATCH_MIN)
        )
        if not ok:
            offending.append(i)
        per_graphic.append({
            "index": i,
            "kind": spec.kind,
            "motion": spec.motion,
            "path": spec.path,
            "face_overlap_px": face_overlap,
            "caption_overlap_px": caption_overlap,
            "caption_zone_overlap_px": zone_overlap,
            "hook_glyph_overlap_px": hook_glyph_overlap,
            "presence_match_min": (None if presence is None
                                   else round(presence, 3)),
            "absence_match_max": (None if absence is None
                                  else round(absence, 3)),
            "path_match": (None if path_match is None
                           else round(path_match, 3)),
            "ok": ok,
        })

    # Pairwise collision: spans overlapping in time AND boxes
    # intersecting. Placement prevents this by construction
    # (already-placed boxes are exclusions, dilated); F8 verifies
    # it on the declared specs, because a renderer or a hand-built
    # spec list owes the same guarantee. Pairs are reported with
    # the earlier-starting graphic first, so a repair can drop
    # the later one.
    collision_violations: list[dict[str, Any]] = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            if not (a.start_s < b.end_s and b.start_s < a.end_s):
                continue
            ox = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
            oy = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
            if ox > 0 and oy > 0:
                earlier, later = (
                    (i, j) if (a.start_s, i) <= (b.start_s, j)
                    else (j, i))
                collision_violations.append({
                    "earlier_index": earlier, "later_index": later,
                    "overlap_px": int(ox * oy),
                })
    # SFX leg (block 2): every declared effect that lands where
    # the speech stem is silent must be measurably present in the
    # shipped mix - the mix's energy at the event instant must
    # clear the stem's by SFX_GAP_LIFT_MIN_DB. Effects under
    # speech are ducked below it by construction; they are
    # declared on the receipt but not separately measurable, so
    # they are counted, never failed. Without a stem there is
    # nothing honest to measure against: the leg records that
    # and does not fail (the pipeline always supplies the stem
    # when SFX are in play).
    sfx_measured: dict[str, Any] | None = None
    sfx_violations: list[dict[str, Any]] = []
    if inputs.sfx is not None:
        gap_checked = 0
        if inputs.speech_audio is not None:
            for ev in inputs.sfx:
                t_ev = float(ev.get("t_s", 0.0))
                stem_db = _profile_db_at(inputs.speech_audio, t_ev)
                if stem_db > SFX_GAP_FLOOR_DB:
                    continue  # under speech: masked by design
                gap_checked += 1
                mix_db = _profile_db_at(inputs.audio, t_ev)
                if mix_db < stem_db + SFX_GAP_LIFT_MIN_DB:
                    sfx_violations.append({
                        "t_s": round(t_ev, 3),
                        "recipe": ev.get("recipe"),
                        "stem_db": round(stem_db, 2),
                        "mix_db": round(mix_db, 2),
                    })
        sfx_measured = {
            "declared": len(inputs.sfx),
            "gap_events_checked": gap_checked,
            "stem_supplied": inputs.speech_audio is not None,
            "violations": sfx_violations,
        }
    # Backdrop leg (block 2): each declared backdrop scene must be
    # MEASURABLY the backdrop inside its span (mid-span pixel
    # match against the declared treated layer) and absent
    # outside it (samples TOLERANCE + 0.25s beyond the edges must
    # NOT match it - outside the span the original treated
    # backdrop shows). Testable pixels exclude the speaker, the
    # caption block, the hook glyph bbox while the hook is up,
    # and any graphic on the plane at that instant (hero rules
    # keep graphics out of backdrop spans, but the leg does not
    # assume it). An absence sample landing inside ANOTHER
    # declared backdrop span is skipped: it measures that span's
    # scene, not this one's absence.
    backdrop_specs = list(inputs.backdrops or [])
    per_backdrop: list[dict[str, Any]] = []
    offending_backdrops: list[int] = []
    for bi, bspec in enumerate(backdrop_specs):
        def backdrop_match_at(idx: int) -> float | None:
            if idx < 0 or idx >= n_frames:
                return None
            rgb = frame_at(idx)
            if rgb is None:
                return None
            t = idx / fps
            exclude = np.zeros((ch, cw), dtype=bool)
            if inputs.caption_rect is not None:
                x0, y0, x1, y1 = inputs.caption_rect
                exclude[y0:y1, x0:x1] = True
            if inputs.hook is not None:
                w0, w1 = inputs.hook.window
                if w0 <= t < w1:
                    try:
                        _, gbbox = inputs.hook.glyph_at(
                            t, inputs.canvas)
                        exclude[gbbox[1]:gbbox[3],
                                gbbox[0]:gbbox[2]] = True
                    except ValueError:
                        pass  # no glyphs = nothing to exclude
            for spec in specs:
                if spec.start_s <= t < spec.end_s:
                    layer = spec.layer_at(t)
                    exclude |= layer[..., 3] > 127
            return _backdrop_match(
                rgb, bspec.layer_at(t), alphas[idx], exclude)

        mid_idx = int(round(
            (bspec.start_s + bspec.end_s) / 2.0 * fps))
        presence = backdrop_match_at(mid_idx)

        def inside_other(idx: int) -> bool:
            t = idx / fps
            return any(
                j != bi and o.start_s <= t < o.end_s
                for j, o in enumerate(backdrop_specs))

        out_matches: list[float] = []
        before = int(np.floor(
            (bspec.start_s - GRAPHIC_SPAN_TOLERANCE_S - 0.25) * fps))
        after = int(np.ceil(
            (bspec.end_s + GRAPHIC_SPAN_TOLERANCE_S + 0.25) * fps))
        for idx in (before, after):
            if 0 <= idx < n_frames and not inside_other(idx):
                m = backdrop_match_at(idx)
                if m is not None:
                    out_matches.append(m)
        absence = max(out_matches) if out_matches else None
        ok = bool(
            presence is not None and presence >= GRAPHIC_MATCH_MIN
            and (absence is None
                 or absence <= GRAPHIC_ABSENT_MATCH_MAX)
        )
        if not ok:
            offending_backdrops.append(bi)
        per_backdrop.append({
            "index": bi,
            "kind": bspec.kind,
            "motion": bspec.motion,
            "presence_match": (None if presence is None
                               else round(presence, 3)),
            "absence_match_max": (None if absence is None
                                  else round(absence, 3)),
            "ok": ok,
        })
    passed = not offending and not collision_violations \
        and not sfx_violations and not offending_backdrops
    return CheckResult(
        check_id="F8", name="context graphics integrity",
        passed=passed,
        measured={
            "count": len(specs),
            "graphics": per_graphic,
            "offending_indices": offending,
            "collision_violations": collision_violations,
            "backdrops": per_backdrop,
            "offending_backdrop_indices": offending_backdrops,
            "sfx": sfx_measured,
            "duration_s": round(duration, 3),
        },
        threshold={
            "face_overlap_max_px": GRAPHIC_FACE_OVERLAP_MAX_PX,
            "caption_overlap_max_px": GRAPHIC_CAPTION_OVERLAP_MAX_PX,
            "hook_glyph_overlap_max_px":
                GRAPHIC_HOOK_GLYPH_OVERLAP_MAX_PX,
            "colliding_pairs_max": GRAPHIC_COLLIDING_PAIRS_MAX,
            "presence_match_min": GRAPHIC_MATCH_MIN,
            "absence_match_max": GRAPHIC_ABSENT_MATCH_MAX,
            "span_tolerance_s": GRAPHIC_SPAN_TOLERANCE_S,
            "path_match_min": GRAPHIC_MATCH_MIN,
            "path_travel_s": PATH_TRAVEL_S,
            "sfx_gap_lift_min_db": SFX_GAP_LIFT_MIN_DB,
            "sfx_gap_floor_db": SFX_GAP_FLOOR_DB,
        },
        note=("the count is uncapped by design and the timing is "
              "free (2026-10-31 ruling, reversing 2026-10-30): "
              "graphics may start at 0s and run back-to-back; the "
              "semantic bar limits the number, and F8 certifies "
              "collision-freedom (face + margin, captions, hook "
              "glyphs, each other), presence and absence of "
              "whatever was placed. Presence is measured on the "
              "render's pixels against each graphic's "
              "deterministic layer_at(t) state."),
    )


# ---------------------------------------------------------------- the gate


def verify(inputs: VerifyInputs, *,
           wonder: WonderClient | None = None,
           draft: int = 1,
           ts: str = "") -> Receipt:
    """Run F1-F9 (F8 only when graphics are supplied; F9 on
    every render) over one render and issue its receipt.

    All deterministic checks always run (a failure never
    short-circuits the rest - the receipt reports everything, so a
    repair step sees the full picture). The Wonder judge runs when a
    client is supplied; without one, F5 records the judge as not
    run and fails - a receipt may not claim a judgement nobody made.
    """
    cw, ch = inputs.canvas
    fps = inputs.fps
    # Never materialise the alpha sequence: a 20 s clip at 24 fps is
    # 480 canvas frames (~4 GB as float32) - list()ing them OOM-killed
    # the first real proof run (2026-10-29). Sized/indexable sequences
    # (lists, lazy loaders) are indexed and streamed in place.
    alphas = inputs.canvas_alphas
    if not (hasattr(alphas, "__len__") and hasattr(alphas, "__getitem__")):
        alphas = list(alphas)
    n_frames = len(alphas)
    geos = [frame_geometry(alphas[i]) for i in range(n_frames)]
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
            if inputs.hook.slide is not None:
                # A sliding card's glyphs are elsewhere during the
                # slide: measure against the position at this
                # frame's instant, with that frame's visible glyph
                # count as the denominator (edge clipping shrinks
                # it), or the measurement would bless positions
                # the glyphs never occupied.
                glyph_f, _ = inputs.hook.glyph_at(
                    f / fps, inputs.canvas)
                glyph_n_f = int(glyph_f.sum())
                if glyph_n_f == 0:
                    continue
                occs.append(float(((a > 0.5) & glyph_f).sum())
                            / glyph_n_f)
                aws.append(float((a * glyph_f).sum()) / glyph_n_f)
            else:
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
            "glyph_occlusion_mean_max": GLYPH_OCCLUSION_CAP,
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
    # Dead air is a property of the SPEECH. When the shipped audio
    # mixes SFX over a speech stem (block 2), F5 measures the STEM
    # (inputs.speech_audio), never the mix: an effect landing in a
    # gap must not make the gap disappear from the measurement.
    # Without a stem, the render's own profile is the speech and
    # the measurement is exactly the historic one. Thresholds are
    # unchanged either way.
    if inputs.speech_audio is not None:
        f5_onset, f5_offset, max_gap = _speech_span(inputs.speech_audio)
    else:
        f5_onset, f5_offset = onset, offset
        _, _, max_gap = _speech_span(inputs.audio)
    speech_span = None
    coverage = 0.0
    if f5_onset is not None and f5_offset is not None:
        speech_span = f5_offset - f5_onset
        covered = sum(max(0.0, min(c.end_s, f5_offset)
                          - max(c.start_s, f5_onset))
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
        and f5_onset is not None and f5_onset <= SPEECH_ONSET_DEADLINE_S
        and captions_present and can_summarize
    )
    checks.append(CheckResult(
        check_id="F5", name="dead air / mute comprehension", passed=f5_pass,
        measured={
            "max_silence_gap_s": None if max_gap is None else round(max_gap, 3),
            "first_speech_onset_s": None if f5_onset is None else round(f5_onset, 3),
            "captions_present": bool(inputs.cues),
            "caption_coverage_of_speech": coverage,
            "wonder_can_summarize_muted_hook": can_summarize,
            "wonder_summary": judgment.summary if judgment else None,
            "measured_on": ("speech_stem" if inputs.speech_audio
                            is not None else "render_audio"),
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
    # Slide speed: the hook's own slide parameters when the render
    # carries them (the renderer composites exactly that offset
    # function, so the speed is read back off the render's design,
    # not asserted blindly); the legacy declared field otherwise.
    # Thresholds unchanged: SLIDE_MIN_PX_S still decides.
    slide_measured = inputs.hook_slide_px_s
    if inputs.hook is not None and inputs.hook.slide is not None:
        slide_measured = inputs.hook.slide_px_s
    checks.append(CheckResult(
        check_id="F7", name="static opening",
        passed=bool(zoom >= ZOOM_MIN_FRAC
                    or slide_measured > SLIDE_MIN_PX_S
                    or mean_iou < MOTION_IOU_CEILING),
        measured={
            "speaker_scale_change_0_1p5s": zoom,
            "hook_slide_px_s": round(slide_measured, 1),
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

    # ---- F8: context graphics (only when the layer was in play) ----
    graphics_section = None
    backdrops_section = None
    if inputs.graphics is not None or inputs.backdrops is not None:
        checks.append(_check_graphics(inputs, alphas))
        if inputs.graphics is not None:
            graphics_section = [g.to_dict() for g in inputs.graphics]
        if inputs.backdrops is not None:
            backdrops_section = [b.to_dict()
                                 for b in inputs.backdrops]

    # ---- F9: platform loudness (every render) ---------------------
    checks.append(_check_loudness(inputs))

    # ---- receipt ---------------------------------------------------------
    receipt = Receipt(
        artifact=str(inputs.video_path),
        verdict="",  # set below
        checks=checks,
        meta_calls=meta_calls,
        graphics=graphics_section,
        sfx=inputs.sfx,
        backdrops=backdrops_section,
        speaker_forward=inputs.speaker_forward,
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
