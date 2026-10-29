"""Layout law: hook + caption placement computed from head geometry.

The compositor's first proof render placed the hook band by a manual
fraction (``--hook-y-frac``) and the stills showed why that cannot
ship: at 0.37 the second hook line passed behind the speaker's cap
with by-eye glyph occlusion around half the line - beyond the
design's <= 25% verifier cap. This module replaces the manual knob
with placement *computed* from the decoded speaker geometry
(``mask_decode`` finished alpha, mapped to canvas space through the
compositor's own ``speaker_layer`` transform, so layout and render
can never disagree about where the speaker is):

- **Hook band**: candidate vertical positions are rendered with the
  compositor's real text renderer and scored against the speaker
  occupancy map over the hook window. The winner must be in the
  safe zone and under the 25% occlusion cap - and, since the
  refinement pass (2026-10-29), it must also satisfy the DEPTH
  EFFECT objective: predicted glyph occlusion inside the 8-20%
  band. The gate run's receipts showed why a cap alone is not a
  layout law: every draft measured occlusion 0.0 - the planner
  parked the hook in clear space above the head, the check passed
  trivially, and the product's signature effect (the speaker
  standing IN FRONT of the hook text) never visually happened. A
  cap is a ceiling; the band is a target. ``plan_hook_effect``
  additionally searches speaker lifts (the reframe machinery) when
  no band position can reach the target, and every placement
  carries an ``effect_status`` so a below-band result is flagged
  on the record instead of passing silently.
- **Caption block**: fixed two-line block, horizontally inside the
  safe zone, bottom edge anchored 320px above the canvas bottom,
  shifted up only if a face would enter it (the F2 rule is
  caption-intersect-face = 0, so the planner treats any overlap as
  a placement failure to solve, and reports it if it cannot).

Safe-zone constants, with provenance (the design sources disagree in
places; the reconciliation is declared in the BUILDLOG):

- Vertical central band y in [288, 1440] on 1080x1920 - from the
  retention research taxonomy the decision pack adopts (hook text,
  captions, face centroid clear of platform UI).
- Horizontal x in [60, 980]: 60px left margin from the converged
  design; the right edge keeps text clear of the platform action
  rail (the research wording) - 100px allowance on a 1080 canvas.
- Captions live in the lower zone the converged design specifies:
  block bottom edge at canvas_h - 320 = 1600. That sits below the
  research band's 1440 bottom, so F3 checks hook text and the face
  centroid against the central band and captions against the lower
  zone - each element against the zone the design assigns it.

Head geometry follows the spike bake-off's definition: the head is
the top band of the mask (topmost row + upper 32% of the mask's
vertical extent); the face region is the mask's pixels inside that
band. Pure functions over NumPy arrays - no model calls, no spend.

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .compositor import DEFAULT_CANVAS, _load_font, hook_layer, speaker_layer

# ---------------------------------------------------------------- constants

CANVAS_W, CANVAS_H = DEFAULT_CANVAS

SAFE_X0 = 60
SAFE_X1 = 980  # clear of the right action rail on a 1080-wide canvas
SAFE_Y0 = 288
SAFE_Y1 = 1440  # central band (hook text, face centroid)

CAPTION_BOTTOM_MARGIN = 320  # caption block bottom edge: H - 320 = 1600
CAPTION_FONT_SIZE = 56
CAPTION_MAX_LINES = 2
CAPTION_MIN_TOP = 1150  # the block never climbs above this y

GLYPH_OCCLUSION_CAP = 0.25
HEAD_BAND_FRAC = 0.32  # spike bake-off: head = upper 32% of the mask
ALPHA_THRESH = 0.5
DEFAULT_HOOK_FRAC = 0.30  # the design's preferred hook band centre

# The depth-effect band (refinement, 2026-10-29): predicted glyph
# occlusion the planner AIMS for. Below the low edge the speaker
# does not visibly cross the text (the gate run's failure mode:
# 0.0 everywhere); above the high edge the text starts losing
# legibility to the silhouette. The band sits strictly inside the
# verifier's 25% cap, which remains the gate's ceiling and is not
# this module's to change.
EFFECT_OCCLUSION_BAND = (0.08, 0.20)
EFFECT_OCCLUSION_TARGET = 0.14  # band centre: aim, don't hug edges

# Evaluation finding (2026-10-29, frame inspection of the first
# composite): at the cap brim the feathered matte edge lets hook
# letters GHOST THROUGH at partial opacity instead of occluding
# cleanly - the brim zone is exactly where placement decisions bite.
# Placement therefore treated a HEAD_GUARD_PX halo around the head
# band as solid speaker (``guarded_occupancy``). RETIRED from
# plan_hook's scoring by the refinement calibration (2026-10-29):
# the retuned matte cut the bleed the guard compensated for
# (541.7 -> 41.7 alpha units on the reference frame), and the
# guard made the planner predict a quantity the verifier does not
# measure - on the first refined render it predicted 0.133 where
# F2 measured 0.065, a 2x overshoot concentrated exactly at the
# grazing placements the effect band targets. The planner now
# predicts the verifier's own quantity (binary occupancy); the
# guarded map remains available for diagnostics.
HEAD_GUARD_PX = 12

# The hook renderer must fit its text INSIDE the safe zone: the
# compositor's default width budget (0.92 of canvas) lets glyphs
# reach x=43..1037, outside [SAFE_X0, SAFE_X1] - the layout path
# therefore renders at the safe width, minus a 16px guard per side
# (the renderer's stroke bleeds a few px past the measured text
# width). F3 measures the manual path's wider render honestly.
HOOK_MAX_WIDTH_FRAC = (SAFE_X1 - SAFE_X0 - 32) / CANVAS_W


# ---------------------------------------------------------------- geometry


@dataclass
class FrameGeometry:
    """One frame of canvas-space speaker geometry (alpha > thresh)."""

    present: bool
    mask_bbox: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1 excl.
    mask_area: int = 0
    head_top: int | None = None
    head_cx: float | None = None
    head_width: float | None = None
    face_box: tuple[int, int, int, int] | None = None
    face_centroid: tuple[float, float] | None = None


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def head_band_rows(alpha: np.ndarray) -> tuple[int, int] | None:
    """(top, bottom_exclusive) rows of the head band, or None if empty."""
    mask = alpha > ALPHA_THRESH
    bbox = _bbox_of(mask)
    if bbox is None:
        return None
    _, y0, _, y1 = bbox
    band_h = max(1, round((y1 - y0) * HEAD_BAND_FRAC))
    return y0, min(y1, y0 + band_h)


def face_pixels(alpha: np.ndarray) -> np.ndarray:
    """Bool canvas of the face region: mask pixels inside the head band."""
    out = np.zeros(alpha.shape, dtype=bool)
    rows = head_band_rows(alpha)
    if rows is None:
        return out
    top, bottom = rows
    out[top:bottom] = (alpha[top:bottom] > ALPHA_THRESH)
    return out


def frame_geometry(alpha: np.ndarray) -> FrameGeometry:
    """Geometry of one canvas-space finished-alpha frame."""
    mask = alpha > ALPHA_THRESH
    bbox = _bbox_of(mask)
    if bbox is None:
        return FrameGeometry(present=False)
    face = face_pixels(alpha)
    fb = _bbox_of(face)
    geo = FrameGeometry(
        present=True,
        mask_bbox=bbox,
        mask_area=int(mask.sum()),
        head_top=bbox[1],
    )
    if fb is not None:
        ys, xs = np.nonzero(face)
        geo.face_box = fb
        geo.face_centroid = (float(xs.mean()), float(ys.mean()))
        geo.head_cx = float(xs.mean())
        geo.head_width = float(fb[2] - fb[0])
    return geo


def sequence_geometry(
    alphas: Iterable[np.ndarray],
) -> list[FrameGeometry]:
    return [frame_geometry(a) for a in alphas]


def to_canvas_alpha(
    alpha: np.ndarray,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
) -> np.ndarray:
    """Map a source-resolution alpha to canvas space.

    Goes through the compositor's ``speaker_layer`` (with a blank
    frame - only its alpha transform is used) so the geometry this
    module computes is in exactly the space the render occupies.
    """
    h, w = alpha.shape
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    _, a = speaker_layer(blank, alpha, canvas)
    return a


def occupancy_map(alphas: Iterable[np.ndarray]) -> np.ndarray:
    """Mean speaker occupancy (alpha > thresh) over a frame sequence."""
    total: np.ndarray | None = None
    n = 0
    for a in alphas:
        occ = (a > ALPHA_THRESH).astype(np.float64)
        total = occ if total is None else total + occ
        n += 1
    if total is None or n == 0:
        raise ValueError("occupancy_map needs at least one frame")
    return total / n


# ---------------------------------------------------------------- hook


@dataclass
class HookPlacement:
    text: str
    y_center_frac: float
    bbox: tuple[int, int, int, int]  # glyph bbox, x0,y0,x1,y1 excl.
    glyph_count: int
    predicted_occlusion: float  # glyph fraction the speaker covers
    within_safe_zone: bool
    ok: bool  # occlusion <= cap AND within the safe zone
    # Depth-effect standing of the predicted occlusion against
    # EFFECT_OCCLUSION_BAND: "in_band" / "below_band" / "above_band".
    # A placement can be ok (gate-safe) and below_band (effect
    # absent) at once - the gate run's every draft was exactly that.
    effect_status: str = "below_band"


def effect_status(
    occlusion: float,
    band: tuple[float, float] = EFFECT_OCCLUSION_BAND,
) -> str:
    """Where an occlusion value stands against the effect band."""
    if occlusion < band[0]:
        return "below_band"
    if occlusion > band[1]:
        return "above_band"
    return "in_band"


def effect_note(
    occlusion: float,
    band: tuple[float, float] = EFFECT_OCCLUSION_BAND,
) -> str:
    """The receipt's effect line for a measured occlusion value.

    This is an annotation, never a verdict: the F1-F7 gate judges
    safety and legibility only. The line exists so that a render
    whose speaker never crosses the hook text cannot pass silently
    as a successful demonstration of the depth effect.
    """
    status = effect_status(occlusion, band)
    if status == "in_band":
        return (
            f"effect: measured hook occlusion {occlusion:.3f} is "
            f"inside the {band[0]:.2f}-{band[1]:.2f} depth-effect "
            "band - the speaker visibly crosses the hook text.")
    if status == "below_band":
        return (
            f"effect: measured hook occlusion {occlusion:.3f} is "
            f"BELOW the {band[0]:.2f}-{band[1]:.2f} depth-effect "
            "band - EFFECT BELOW TARGET: the speaker's crossing "
            "of the hook text is marginal on this draft, whatever "
            "the checks say.")
    return (
        f"effect: measured hook occlusion {occlusion:.3f} is "
        f"ABOVE the {band[0]:.2f}-{band[1]:.2f} depth-effect band "
        "(still bounded by the 0.25 F2 cap) - the speaker covers "
        "more of the hook than the effect targets.")


def render_hook_layer(
    text: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    y_center_frac: float = DEFAULT_HOOK_FRAC,
) -> np.ndarray:
    """The hook layer as the LAYOUT path renders it (safe width).

    Same renderer as the compositor, with the width budget clamped
    to the safe zone - this is the layer the compositor's
    ``--hook-auto-layout`` path composites, so what layout_law plans
    is byte-for-byte what ships. One correction on top of the raw
    renderer: it centres text on the CANVAS (x=540), but the safe
    zone is centred at x=520 - an 885px ink block centred at 540
    spans 97..982 and breaks the right rail by 2px. The layer is
    therefore shifted onto the safe zone's centre.
    """
    # The compositor's renderer shrinks for LINE COUNT, not width:
    # a single long word ("REHEARSES" at the default size) renders
    # ~1000px wide inside an 888px budget. The layout path therefore
    # steps the start size down until the INK fits the safe width -
    # placement law, enforced on pixels, not on the renderer's say-so.
    size = 170
    layer = hook_layer(
        text, canvas, y_center_frac=y_center_frac,
        max_width_frac=HOOK_MAX_WIDTH_FRAC, start_size=size,
    )
    while size > 40:
        bbox = _bbox_of(layer[..., 3] > 0)
        if bbox is None or bbox[2] - bbox[0] <= SAFE_X1 - SAFE_X0 - 32:
            break
        size = int(size * 0.9)
        layer = hook_layer(
            text, canvas, y_center_frac=y_center_frac,
            max_width_frac=HOOK_MAX_WIDTH_FRAC, start_size=size,
        )
    dx = (SAFE_X0 + SAFE_X1) // 2 - canvas[0] // 2
    if dx == 0:
        return layer
    shifted = np.zeros_like(layer)
    if dx > 0:
        shifted[:, dx:] = layer[:, :-dx]
    else:
        shifted[:, :dx] = layer[:, -dx:]
    return shifted


def hook_glyph_mask(
    text: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    y_center_frac: float = DEFAULT_HOOK_FRAC,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Glyph mask + bbox of the hook as the layout path renders it."""
    layer = render_hook_layer(text, canvas, y_center_frac=y_center_frac)
    mask = layer[..., 3] > 0
    bbox = _bbox_of(mask)
    if bbox is None:
        raise ValueError(f"hook text rendered no glyphs: {text!r}")
    return mask, bbox


def _in_central_band(bbox: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = bbox
    return (
        x0 >= SAFE_X0 and x1 <= SAFE_X1
        and y0 >= SAFE_Y0 and y1 <= SAFE_Y1
    )


def predict_occlusion(glyph_mask: np.ndarray, occupancy: np.ndarray) -> float:
    """Expected glyph fraction covered by the speaker (0..1)."""
    n = int(glyph_mask.sum())
    if n == 0:
        return 0.0
    return float((occupancy * glyph_mask).sum() / n)


def _dilate_support(support: np.ndarray, px: int) -> np.ndarray:
    """Binary dilation of a bool support by ``px`` (3x3 iterations)."""
    out = support.astype(np.float64)
    for _ in range(px):
        padded = np.pad(out, 1)
        out = np.maximum.reduce(
            [padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
             for dy in range(3) for dx in range(3)]
        )
    return out


def guarded_occupancy(
    alphas: Sequence[np.ndarray], guard_px: int = HEAD_GUARD_PX,
) -> np.ndarray:
    """Occupancy for PLANNING: geometric occupancy, with the head
    band's support dilated by ``guard_px`` and treated as solid.

    The feathered matte ghosts text through the brim zone at partial
    alpha, so a purely geometric prediction understates how much of
    a glyph placed there is visually lost. The guard makes the
    planner conservative exactly where placement decisions bite.
    """
    occ = occupancy_map(alphas)
    if guard_px <= 0:
        return occ
    head_support = np.zeros(occ.shape, dtype=bool)
    for a in alphas:
        head_support |= face_pixels(a)
    if not head_support.any():
        return occ
    return np.maximum(occ, _dilate_support(head_support, guard_px))


def plan_hook(
    text: str,
    window_alphas: Iterable[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    preferred_frac: float = DEFAULT_HOOK_FRAC,
    cap: float = GLYPH_OCCLUSION_CAP,
    fracs: Sequence[float] | None = None,
    occlusion_band: tuple[float, float] | None = EFFECT_OCCLUSION_BAND,
) -> HookPlacement:
    """Choose the hook band centre from the window's speaker geometry.

    Candidates are scored by (predicted occlusion, safe-zone
    membership) against the window's binary occupancy map - the
    same quantity the verifier's F2 measures, so a prediction is a
    calibrated estimate of the receipt's number (see the
    HEAD_GUARD_PX note for why the guarded map was retired from
    scoring). Preference order (deterministic tie-breaks
    throughout, so the same geometry always plans the same layout):

    1. safe + under the cap + occlusion INSIDE the depth-effect
       band, nearest the band's centre, then nearest the preferred
       band - the effect actually happens, at its target strength;
    2. safe + under the cap, nearest the band (from either side),
       then nearest the preferred band;
    3. safe, least occlusion;
    4. least occlusion anywhere.

    ``occlusion_band=None`` restores the pre-refinement behaviour
    (tier 2 without the band-distance term: nearest the preferred
    band among the gate-safe candidates).
    """
    if fracs is None:
        # .14-.68: the grid's lower half is where a low-seated
        # speaker's head/shoulders actually are; the pre-refinement
        # grid stopped at .56, one step short of the band on the
        # gate run's geometry (0.080 predicted at .56, the grid's
        # last candidate).
        fracs = [round(0.14 + 0.01 * i, 2) for i in range(55)]
    alphas = list(window_alphas)
    occ_map = occupancy_map(alphas)
    band = occlusion_band if occlusion_band is not None \
        else EFFECT_OCCLUSION_BAND
    scored: list[HookPlacement] = []
    for frac in fracs:
        glyph, bbox = hook_glyph_mask(text, canvas, y_center_frac=frac)
        occ = predict_occlusion(glyph, occ_map)
        safe = _in_central_band(bbox)
        scored.append(
            HookPlacement(
                text=text,
                y_center_frac=float(frac),
                bbox=bbox,
                glyph_count=int(glyph.sum()),
                predicted_occlusion=occ,
                within_safe_zone=safe,
                ok=bool(safe and occ <= cap),
                effect_status=effect_status(occ, band),
            )
        )

    def band_distance(p: HookPlacement) -> float:
        occ = p.predicted_occlusion
        if occ < band[0]:
            return band[0] - occ
        if occ > band[1]:
            return occ - band[1]
        return 0.0

    if occlusion_band is not None:
        in_band = [p for p in scored
                   if p.ok and p.effect_status == "in_band"]
        if in_band:
            target = (band[0] + band[1]) / 2.0
            return min(
                in_band,
                key=lambda p: (abs(p.predicted_occlusion - target),
                               abs(p.y_center_frac - preferred_frac),
                               p.y_center_frac),
            )
    good = [p for p in scored if p.ok]
    if good:
        if occlusion_band is None:
            return min(
                good,
                key=lambda p: (abs(p.y_center_frac - preferred_frac),
                               p.predicted_occlusion, p.y_center_frac),
            )
        return min(
            good,
            key=lambda p: (band_distance(p),
                           abs(p.y_center_frac - preferred_frac),
                           p.y_center_frac),
        )
    safe = [p for p in scored if p.within_safe_zone]
    pool = safe if safe else scored
    return min(
        pool,
        key=lambda p: (p.predicted_occlusion,
                       abs(p.y_center_frac - preferred_frac),
                       p.y_center_frac),
    )


# ---------------------------------------------------------------- captions


def caption_lines(
    text: str,
    *,
    font_size: int = CAPTION_FONT_SIZE,
    max_width: int = SAFE_X1 - SAFE_X0,
) -> list[str]:
    """Wrap one caption cue to at most CAPTION_MAX_LINES lines."""
    font = _load_font(font_size)
    probe = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(probe)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = (current + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > CAPTION_MAX_LINES:
        # A cue that cannot fit two lines is a caption_align bug (its
        # phrases are capped), but never silently drop words: keep
        # the first lines and ellipsize nothing - return all lines
        # and let the block grow check catch it downstream.
        return lines
    return lines


def caption_block_height(font_size: int = CAPTION_FONT_SIZE) -> int:
    return int(font_size * 1.25) * CAPTION_MAX_LINES


def initial_caption_rect(
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    font_size: int = CAPTION_FONT_SIZE,
) -> tuple[int, int, int, int]:
    """The default caption block: safe-width, bottom edge at H - 320."""
    cw, ch = canvas
    y1 = ch - CAPTION_BOTTOM_MARGIN
    y0 = y1 - caption_block_height(font_size)
    return (SAFE_X0, y0, SAFE_X1, y1)


@dataclass
class CaptionLayout:
    rect: tuple[int, int, int, int]  # the shared block, x0,y0,x1,y1 excl.
    per_cue_overlap_px: dict[int, int] = field(default_factory=dict)
    feasible: bool = True  # False if a face still enters the block
    # Worst per-cue overlap with the speaker's BODY (full silhouette,
    # not just the face). Not a gate quantity - F2 measures the face
    # only - but the refinement's polish target: the planner moves
    # the block off the body when a face-feasible position allows
    # it, and reports the number either way.
    per_cue_body_overlap_px: dict[int, int] = field(default_factory=dict)


def rect_overlap_px(region: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    """Pixels of a bool region inside a rect."""
    x0, y0, x1, y1 = rect
    return int(region[y0:y1, x0:x1].sum())


def _face_band(alpha: np.ndarray) -> tuple[int, np.ndarray] | None:
    """The face region of one frame, cropped to its head-band rows.

    ``face_pixels`` is zero outside the head band by construction,
    so (band_top, band_rows) carries the identical pixel set at a
    fraction of the memory. None when the frame has no speaker.
    """
    rows = head_band_rows(alpha)
    if rows is None:
        return None
    top, bottom = rows
    return top, alpha[top:bottom] > ALPHA_THRESH


def _band_overlap_px(
    band: tuple[int, np.ndarray] | None,
    rect: tuple[int, int, int, int],
) -> int:
    """``rect_overlap_px(face_pixels(frame), rect)`` from the band."""
    if band is None:
        return 0
    top, rows = band
    x0, y0, x1, y1 = rect
    lo = max(y0, top) - top
    hi = min(y1, top + rows.shape[0]) - top
    if hi <= lo:
        return 0
    return int(rows[lo:hi, x0:x1].sum())


def _face_top(band: tuple[int, np.ndarray] | None) -> int | None:
    """First canvas row containing a face pixel, or None."""
    if band is None:
        return None
    top, rows = band
    any_row = rows.any(axis=1)
    if not any_row.any():
        return None
    return top + int(np.nonzero(any_row)[0].min())


def plan_captions(
    cue_spans: Sequence[tuple[float, float]],
    alphas: Sequence[np.ndarray],
    *,
    fps: float,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    font_size: int = CAPTION_FONT_SIZE,
) -> CaptionLayout:
    """Place the shared caption block with zero face overlap.

    ``cue_spans`` are (start_s, end_s) per cue, aligned with the cue
    order the caller will render; ``alphas`` is the dense canvas
    alpha sequence. The block starts at the anchored position and
    climbs in ``shift_px`` steps only while some cue's frames put
    face pixels inside it. If it reaches CAPTION_MIN_TOP still
    overlapping, the best (least-overlap) position is kept and
    ``feasible`` is False - the verifier will fail F2 on the
    measured overlap rather than the planner hiding it.
    """
    rect = initial_caption_rect(canvas, font_size=font_size)
    # Face pixels live only inside the head band (``face_pixels``
    # zeroes everything outside it), so store just the band rows:
    # (top, bool rows) per frame. A full-canvas bool per frame cost
    # ~1.36 GB at 660 frames - one of the allocations behind the
    # 2026-10-29 OOM kills. The overlap counts and face tops computed
    # from the bands are exactly the full-canvas values.
    faces = [_face_band(a) for a in alphas]
    block_h = rect[3] - rect[1]

    def cue_frames(i: int) -> range:
        start, end = cue_spans[i]
        f0 = max(0, int(round(start * fps)))
        f1 = min(len(faces), max(f0 + 1, int(round(end * fps))))
        return range(f0, f1)

    def overlaps(r: tuple[int, int, int, int]) -> dict[int, int]:
        return {
            i: max(
                (_band_overlap_px(faces[f], r) for f in cue_frames(i)),
                default=0,
            )
            for i in range(len(cue_spans))
        }

    best_rect, best = rect, overlaps(rect)
    # Creeping up a few px at a time first slides the block INTO the
    # face band (overlap grows before it clears), so instead jump:
    # the block's bottom edge goes to the highest face top among the
    # frames of every cue that currently overlaps. A couple of jumps
    # settle it (a different cue's frames may reach higher).
    for _ in range(4):
        bad = [i for i, v in best.items() if v > 0]
        if not bad:
            break
        face_tops = [
            top
            for i in bad
            for f in cue_frames(i)
            if (top := _face_top(faces[f])) is not None
        ]
        if not face_tops:
            break
        y1 = min(face_tops)
        y0 = y1 - block_h
        if y0 < CAPTION_MIN_TOP:
            # Cannot legally clear the face: clamp to the highest
            # allowed position and keep whichever of the two
            # positions overlaps least - reported, never hidden.
            clamped = (best_rect[0], CAPTION_MIN_TOP, best_rect[2],
                       CAPTION_MIN_TOP + block_h)
            got = overlaps(clamped)
            if sum(got.values()) < sum(best.values()):
                best_rect, best = clamped, got
            break
        candidate = (best_rect[0], y0, best_rect[2], y1)
        best_rect, best = candidate, overlaps(candidate)

    def body_overlaps(r: tuple[int, int, int, int]) -> dict[int, int]:
        bx0, by0, bx1, by1 = r
        out: dict[int, int] = {}
        for i in range(len(cue_spans)):
            worst = 0
            for f in cue_frames(i):
                region = alphas[f][by0:by1, bx0:bx1] > ALPHA_THRESH
                worst = max(worst, int(region.sum()))
            out[i] = worst
        return out

    body = body_overlaps(best_rect)
    if not any(best.values()):
        # Face-feasible. Refinement (2026-10-29): also try to clear
        # the BODY - captions printed across the speaker's shirt
        # read as unfinished even though no gate measures them.
        # Step up in 40px increments (never above CAPTION_MIN_TOP,
        # never back into the face) and take the LOWEST position
        # with zero body overlap; if none exists, the face-planned
        # block stays and its body overlap is reported as-is.
        step = 1
        while True:
            cand_y1 = best_rect[3] - 40 * step
            cand_y0 = cand_y1 - block_h
            if cand_y0 < CAPTION_MIN_TOP:
                break
            candidate = (best_rect[0], cand_y0, best_rect[2], cand_y1)
            if any(overlaps(candidate).values()):
                break  # climbing further only gets closer to the face
            cand_body = body_overlaps(candidate)
            if not any(cand_body.values()):
                best_rect, body = candidate, cand_body
                break
            step += 1
    return CaptionLayout(
        rect=best_rect,
        per_cue_overlap_px=best,
        feasible=not any(v > 0 for v in best.values()),
        per_cue_body_overlap_px=body,
    )


# ---------------------------------------------------------------- speaker reframe
#
# The first full Holmes run failed F3 on all three Shorts with zero
# hook/caption pixel violations: the median face centroid sat at
# y = 1541-1554, below the central band's 1440 edge. Holmes's desk
# framing puts his face low in the source, and the compositor's
# fit-width + bottom-anchor placement preserves that faithfully -
# layout planned the TEXT around the face, but nothing placed the
# FACE. This section plans the speaker's own placement.
#
# The mechanism is the one RESCALE_SPEAKER already proved for the
# verifier's geometry: transform the SOURCE-space inputs (frames +
# alphas) and let the standard placement render them, so the
# geometry the layout predicts, the pixels the compositor draws and
# the alphas the verifier measures are the same numbers by
# construction. Under a bottom anchor, the face's canvas height is
# set by its distance from the frame's BOTTOM edge - so the
# transform is a bottom-edge adjustment of the source frame: PAD
# rows below the speaker (edge-replicated RGB, zero alpha) to lift
# the face, or CROP bottom rows to lower it. Nothing else about the
# frame changes; the head band (upper 32% of the mask) is untouched
# by a bottom pad, so predicted face geometry is exact there, and a
# small approximation for crops (the mask extent shrinks) is
# declared where it is used.

REFRAME_TARGET_FRAC = 0.78     # of the central band: aim low-centre
REFRAME_HEAD_TOP_PLAN_MIN = 150  # planning floor (verifier F4: 90)
REFRAME_IDENTITY_TOL_PX = 60   # centroid this deep in-band: no move
_F4_HEAD_MARGIN_PX = 90        # verifier F4's floor (duplicated to
#                                avoid an import cycle; verifier.py
#                                owns the constant)


@dataclass
class SpeakerReframe:
    """A planned bottom-edge transform of the source frame."""

    pad_bottom_px: int            # source rows added (<0: cropped)
    scale: float                  # resulting speaker_layer scale
    placed_centroid: tuple[float, float] | None
    placed_head_top_min: int | None
    feasible: bool                # centroid in band AND F4 margin ok
    identity: bool = False
    note: str = ""


def _placed_y(canvas: tuple[int, int], sw: int, height: float,
              y: float) -> tuple[float, float]:
    """(canvas y, scale) of source row ``y`` under speaker placement
    for a (possibly transformed) source frame of height ``height``."""
    cw, ch = canvas
    scale = min(cw / sw, ch / height)
    return ch - height * scale + y * scale, scale


def plan_speaker_reframe(
    src_alphas: Sequence[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    target_frac: float = REFRAME_TARGET_FRAC,
    head_top_plan_min: int = REFRAME_HEAD_TOP_PLAN_MIN,
) -> SpeakerReframe:
    """Plan the bottom-edge transform placing the face in the band.

    Statistics mirror the verifier's F3/F4 exactly: the median of
    per-frame face centroids, and the MINIMUM per-frame head top.
    The transform is solved against those statistics, then clamped:
    the placed head top may not fall below ``head_top_plan_min``
    (planning margin above F4's 90px floor), and a crop may not eat
    into the face (the new bottom edge stays below every frame's
    face box, plus clearance). If the clamps bind before the
    centroid reaches the band, the best clamped transform is
    returned with ``feasible=False`` - reported, never hidden.
    """
    geos = [frame_geometry(a) for a in src_alphas]
    present = [g for g in geos if g.present
               and g.face_centroid is not None]
    if not present or not src_alphas:
        return SpeakerReframe(
            0, 0.0, None, None, feasible=False, identity=True,
            note="no speaker geometry in the window; nothing to place")
    sh, sw = src_alphas[0].shape
    cw, ch = canvas
    cx = float(np.median([g.face_centroid[0] for g in present]))
    cy = float(np.median([g.face_centroid[1] for g in present]))
    ht_min = min(g.head_top for g in present
                 if g.head_top is not None)
    face_bottom_max = max(g.face_box[3] for g in present
                          if g.face_box is not None)

    def evaluate(height: float) -> tuple[float, float, float]:
        placed_cy, scale = _placed_y(canvas, sw, height, cy)
        placed_ht, _ = _placed_y(canvas, sw, height, float(ht_min))
        return placed_cy, placed_ht, scale

    cur_cy, cur_ht, cur_scale = evaluate(float(sh))
    if SAFE_Y0 + REFRAME_IDENTITY_TOL_PX <= cur_cy \
            <= SAFE_Y1 - REFRAME_IDENTITY_TOL_PX:
        return SpeakerReframe(
            0, cur_scale,
            (round((cw - sw * cur_scale) / 2.0 + cx * cur_scale, 1),
             round(cur_cy, 1)),
            int(round(cur_ht)), feasible=True, identity=True,
            note="median face centroid already inside the central band")

    target_cy = SAFE_Y0 + target_frac * (SAFE_Y1 - SAFE_Y0)
    # Solve the frame height for the target centroid (monotone in
    # height; a few secant steps converge in either scale regime).
    height = cy + (ch - target_cy) / max(cur_scale, 1e-9)
    for _ in range(10):
        placed_cy, _ht, scale = evaluate(height)
        err = placed_cy - target_cy
        if abs(err) < 0.5:
            break
        height += err / max(scale, 1e-9)
    # Clamp: head-top planning floor (growing the frame lifts the
    # head toward the canvas top; shrink height until it clears).
    for _ in range(200):
        _cy_e, placed_ht, _s = evaluate(height)
        if placed_ht >= head_top_plan_min or height <= 1.0:
            break
        height -= 1.0
    # Clamp: never crop into the face; never pad absurdly.
    height = max(height, float(face_bottom_max + 4))
    height = min(height, float(4 * sh))
    pad = int(round(height - sh))
    height = float(sh + pad)
    placed_cy, placed_ht, scale = evaluate(height)
    placed_cx = (cw - sw * scale) / 2.0 + cx * scale
    feasible = bool(SAFE_Y0 <= placed_cy <= SAFE_Y1
                    and placed_ht >= _F4_HEAD_MARGIN_PX)
    note = ""
    if pad == 0:
        identity = True
        note = ("clamps left no transform; centroid stays at "
                f"y={placed_cy:.0f}")
    else:
        identity = False
        note = (f"bottom {'pad' if pad > 0 else 'crop'} {abs(pad)} "
                f"source rows; centroid y {cur_cy:.0f} -> "
                f"{placed_cy:.0f} (target {target_cy:.0f}), head-top "
                f"min {cur_ht:.0f} -> {placed_ht:.0f}")
    if not feasible:
        note += "; INFEASIBLE within the head-top/crop clamps"
    return SpeakerReframe(
        pad, scale, (round(placed_cx, 1), round(placed_cy, 1)),
        int(round(placed_ht)), feasible=feasible, identity=identity,
        note=note)


def apply_speaker_reframe(
    frames: list[np.ndarray],
    alphas: list[np.ndarray],
    pad_bottom_px: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply a planned bottom-edge transform to frames + alphas.

    Pad: RGB rows are edge-replicated (they feed only the blurred
    backdrop under a zero alpha), alpha rows are zero. Crop: rows
    are sliced off the bottom. Frame/alpha alignment is preserved
    1:1, and ``to_canvas_alpha`` of a transformed alpha is exactly
    the geometry the compositor renders.

    The transform is applied IN PLACE, slot by slot, and the same
    list objects are returned: building fresh lists made the old and
    padded sequences coexist (~2.6 GB at a 210-row pad on 660
    frames), the largest single allocation spike left after the
    finish-stack streaming fix (2026-10-29 OOM). Callers must treat
    the inputs as consumed - both call sites (pipeline construction,
    the REFRAME_SPEAKER executor) plan from the pre-transform
    geometry first and use only the returned lists afterwards. The
    per-frame values are exactly ``np.pad`` / slice-copy of the old
    ones; crop slices are copied so the original buffers are
    released rather than pinned by views.
    """
    if pad_bottom_px == 0:
        return frames, alphas
    if pad_bottom_px > 0:
        p = pad_bottom_px
        for i, f in enumerate(frames):
            frames[i] = np.pad(f, ((0, p), (0, 0), (0, 0)), mode="edge")
        for i, a in enumerate(alphas):
            alphas[i] = np.pad(a, ((0, p), (0, 0)), mode="constant")
        return frames, alphas
    keep = frames[0].shape[0] + pad_bottom_px
    if keep <= 0:
        raise ValueError("reframe crop would remove the whole frame")
    for i, f in enumerate(frames):
        frames[i] = f[:keep].copy()
    for i, a in enumerate(alphas):
        alphas[i] = a[:keep].copy()
    return frames, alphas


# ---------------------------------------------------------------- hook effect
#
# plan_hook aims at the depth-effect band, but on some geometry no
# band position reaches it: a small or low-seated speaker's
# silhouette simply does not extend into any safe hook position.
# Settling for occlusion ~0 is how the gate run shipped three
# passing receipts with the product's signature effect absent.
# plan_hook_effect therefore searches the one remaining lever -
# lifting the speaker further with the reframe machinery (extra
# bottom pad, solved by plan_speaker_reframe's own statistics and
# clamps) - before accepting a below-band placement. Lifts are
# tried smallest first and the search stops at the first in-band
# placement; every candidate keeps the verifier-facing geometry
# honest (median face centroid inside the central band, minimum
# head top at the reframe planning floor). If no lift reaches the
# band, the best (closest-to-band) feasible placement is returned
# with its below-band status on the record - reported, never
# hidden, and never a gate failure: the gate judges safety and
# legibility, the effect line judges the demo.


@dataclass
class HookEffectPlan:
    """A hook placement plus the speaker transform it requires."""

    pad_bottom_px: int          # total transform to apply (source rows)
    lift_px: int                # pad beyond the base reframe's plan
    placement: HookPlacement
    geometry_ok: bool           # centroid in band + head-top floor held
    note: str = ""


def _transform_alpha(alpha: np.ndarray, pad_bottom_px: int) -> np.ndarray:
    """One alpha under a bottom-edge transform (apply_speaker_reframe
    semantics for a single alpha: pad = zero rows, crop = slice)."""
    if pad_bottom_px == 0:
        return alpha
    if pad_bottom_px > 0:
        return np.pad(alpha, ((0, pad_bottom_px), (0, 0)),
                      mode="constant")
    keep = alpha.shape[0] + pad_bottom_px
    if keep <= 0:
        raise ValueError("transform would remove the whole frame")
    return alpha[:keep].copy()


def plan_hook_effect(
    text: str,
    src_alphas: Sequence[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    base_reframe: SpeakerReframe | None = None,
    lift_steps: Sequence[int] = (0, 40, 80, 120, 160, 200),
    hook_sample: int | None = None,
    sample_stride: int = 2,
) -> HookEffectPlan:
    """Plan the hook's depth effect, lifting the speaker if needed.

    ``src_alphas`` are the UNTRANSFORMED source-space alphas;
    ``base_reframe`` is the pipeline's construction reframe (planned
    here when not supplied). Returns the total bottom-edge
    transform to apply plus the hook placement predicted on the
    transformed geometry, sampled exactly as the pipeline samples
    for ``plan_hook`` (every ``sample_stride``-th frame of the hook
    window).
    """
    if base_reframe is None:
        base_reframe = plan_speaker_reframe(src_alphas, canvas)
    if not src_alphas:
        placement = plan_hook(text, [np.zeros((canvas[1], canvas[0]))],
                              canvas)
        return HookEffectPlan(
            base_reframe.pad_bottom_px, 0, placement, False,
            note="no speaker geometry; effect cannot be planned")
    sh = src_alphas[0].shape[0]
    geos = [frame_geometry(a) for a in src_alphas]
    face_bottom_max = max(
        (g.face_box[3] for g in geos if g.face_box is not None),
        default=sh)
    n = hook_sample if hook_sample is not None else len(src_alphas)
    idx = list(range(0, min(n, len(src_alphas)), max(1, sample_stride)))

    def evaluate(pad: int) -> tuple[HookPlacement, bool]:
        sample = [to_canvas_alpha(_transform_alpha(src_alphas[i], pad),
                                  canvas)
                  for i in idx]
        geos_c = [frame_geometry(a) for a in sample]
        cents = [g.face_centroid for g in geos_c
                 if g.face_centroid is not None]
        tops = [g.head_top for g in geos_c if g.head_top is not None]
        geo_ok = bool(
            cents and tops
            and SAFE_Y0 <= float(np.median([c[1] for c in cents]))
            <= SAFE_Y1
            and min(tops) >= REFRAME_HEAD_TOP_PLAN_MIN)
        return plan_hook(text, sample, canvas), geo_ok

    best: HookEffectPlan | None = None
    for lift in lift_steps:
        pad = base_reframe.pad_bottom_px + int(lift)
        if lift != 0 and not (face_bottom_max + 4 <= sh + pad <= 4 * sh):
            continue  # outside the reframe clamps; not a legal lift
        placement, geo_ok = evaluate(pad)
        cand = HookEffectPlan(
            pad, int(lift), placement, geo_ok,
            note=(f"lift +{lift}px: predicted occlusion "
                  f"{placement.predicted_occlusion:.3f} "
                  f"({placement.effect_status}), geometry ok {geo_ok}"))
        if geo_ok and placement.effect_status == "in_band":
            cand.note = (
                f"in-band at lift +{lift}px: predicted occlusion "
                f"{placement.predicted_occlusion:.3f}")
            return cand
        if best is None or (
            geo_ok and not best.geometry_ok
        ) or (
            geo_ok == best.geometry_ok
            and abs(placement.predicted_occlusion
                    - EFFECT_OCCLUSION_TARGET)
            < abs(best.placement.predicted_occlusion
                  - EFFECT_OCCLUSION_TARGET)
        ):
            best = cand
    assert best is not None  # lift_steps always contains 0 in practice
    best.note = (
        f"no lift in {list(lift_steps)} reached the effect band; "
        f"best is lift +{best.lift_px}px at predicted occlusion "
        f"{best.placement.predicted_occlusion:.3f} "
        f"({best.placement.effect_status}) - effect below target, "
        "declared")
    return best


CAPTION_PILL_FILL = (8, 8, 10, 168)
CAPTION_PILL_PAD_X = 16
CAPTION_PILL_INSET_Y = 5


def caption_layer(
    text: str,
    rect: tuple[int, int, int, int],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    font_size: int = CAPTION_FONT_SIZE,
    pill: bool = True,
) -> np.ndarray:
    """Render one cue's text as an RGBA layer inside the block rect.

    Legibility treatment (refinement, 2026-10-29): each line sits
    on a subtle dark pill - a rounded rectangle at 66% opacity,
    inset inside the line's own band and clamped to the block rect,
    so the pill can never push ink outside the planned block - with
    white fill and a thin dark stroke on top. The gate outputs'
    bare stroked text washed out over bright clothing and the
    treated backdrop's bright patches; the pill holds contrast
    constant wherever the block lands. Captions render IN FRONT of
    the speaker - this layer is composited last by the caller.
    """
    cw, ch = canvas
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    lines = caption_lines(text, font_size=font_size)
    line_h = int(font_size * 1.25)
    x0, y0, x1, y1 = rect
    total_h = line_h * len(lines)
    y = y0 + max(0, ((y1 - y0) - total_h) // 2)
    for line in lines:
        w = draw.textlength(line, font=font)
        x = x0 + ((x1 - x0) - w) / 2
        if pill:
            draw.rounded_rectangle(
                [max(x0, x - CAPTION_PILL_PAD_X),
                 y + CAPTION_PILL_INSET_Y,
                 min(x1, x + w + CAPTION_PILL_PAD_X),
                 y + line_h - CAPTION_PILL_INSET_Y],
                radius=(line_h - 2 * CAPTION_PILL_INSET_Y) // 2,
                fill=CAPTION_PILL_FILL,
            )
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=max(2, font_size // 28),
            stroke_fill=(10, 10, 12, 255),
        )
        y += line_h
    return np.asarray(img)
