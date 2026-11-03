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
    *,
    placement: object | None = None,
) -> np.ndarray:
    """Map a source-resolution alpha to canvas space.

    Goes through the compositor's ``speaker_layer`` (with a blank
    frame - only its alpha transform is used) so the geometry this
    module computes is in exactly the space the render occupies.
    ``placement`` (layout states) forwards an explicit speaker
    placement, so geometry planned/measured for a layout span is
    in exactly the space that span's render occupies; None is the
    historic fit placement.
    """
    h, w = alpha.shape
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    _, a = speaker_layer(blank, alpha, canvas, placement=placement)
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
    # Fix round 2 (Fix E): the hook size this placement was solved
    # at (plan_hook_effect's size ladder start size). 170 = the
    # historic default; smaller values are recorded step-downs
    # that hold the occlusion band against the enlarged speaker.
    text_size: int = 170


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


def hook_accent_index(text: str) -> int | None:
    """The hook's ONE emphasis word (fix round 2, Fix E).

    Deterministic, reusing the caption emphasis selection
    (caption_emphasis_indices - numbers and named entities read
    off the text itself): a digit-bearing emphasis word wins,
    else the longest emphasis word (ties: first). With no
    emphasis candidates at all, the longest word wins (ties:
    first). None only for empty text.
    """
    words = text.split()
    if not words:
        return None
    emph = caption_emphasis_indices(text)
    digit = [i for i in sorted(emph)
             if any(c.isdigit() for c in words[i])]
    if digit:
        return digit[0]
    if emph:
        return max(sorted(emph), key=lambda i: len(words[i]))
    return max(range(len(words)), key=lambda i: len(words[i]))


def _hook_final_size(text: str, canvas: tuple[int, int],
                     start_size: int) -> int:
    """The start size render_hook_layer's shrink loop settles on.

    Replicates the loop's decisions (ink width vs the safe-zone
    budget) so the per-word entry layers render at exactly the
    size the resting layer uses.
    """
    size = start_size
    while size > 40:
        layer = hook_layer(text, canvas,
                           max_width_frac=HOOK_MAX_WIDTH_FRAC,
                           start_size=size)
        bbox = _bbox_of(layer[..., 3] > 0)
        if bbox is None or bbox[2] - bbox[0] <= SAFE_X1 - SAFE_X0 - 32:
            break
        size = int(size * 0.9)
    return size


def _safe_center(layer: np.ndarray,
                 canvas: tuple[int, int]) -> np.ndarray:
    """Shift a hook layer onto the safe zone's centre (x=520)."""
    dx = (SAFE_X0 + SAFE_X1) // 2 - canvas[0] // 2
    if dx == 0:
        return layer
    shifted = np.zeros_like(layer)
    if dx > 0:
        shifted[:, dx:] = layer[:, :-dx]
    else:
        shifted[:, :dx] = layer[:, -dx:]
    return shifted


def render_hook_layer(
    text: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    y_center_frac: float = DEFAULT_HOOK_FRAC,
    accent_index: int | None = None,
    start_size: int = 170,
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

    Fix round 2: ``accent_index`` paints the one emphasis word in
    Viral Pink (two-tone type); ``start_size`` is the solved hook
    size from plan_hook_effect's size ladder.
    """
    # The compositor's renderer shrinks for LINE COUNT, not width:
    # a single long word ("REHEARSES" at the default size) renders
    # ~1000px wide inside an 888px budget. The layout path therefore
    # steps the start size down until the INK fits the safe width -
    # placement law, enforced on pixels, not on the renderer's say-so.
    size = _hook_final_size(text, canvas, start_size)
    layer = hook_layer(
        text, canvas, y_center_frac=y_center_frac,
        max_width_frac=HOOK_MAX_WIDTH_FRAC, start_size=size,
        accent_index=accent_index,
    )
    return _safe_center(layer, canvas)


def hook_word_layers(
    text: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    y_center_frac: float = DEFAULT_HOOK_FRAC,
    accent_index: int | None = None,
    start_size: int = 170,
) -> list[np.ndarray]:
    """Per-word hook layers for the stagger-pop entry (Fix E).

    Word i's layer carries only word i's ink, at its resting
    position in the styled resting layer (same settled size, same
    safe-centre shift), so compositor.hook_entry_layer can compose
    the entry from exactly the pixels that come to rest.
    """
    size = _hook_final_size(text, canvas, start_size)
    layers = []
    for i in range(len(text.split())):
        layer = hook_layer(
            text, canvas, y_center_frac=y_center_frac,
            max_width_frac=HOOK_MAX_WIDTH_FRAC, start_size=size,
            accent_index=accent_index, only_word_index=i)
        layers.append(_safe_center(layer, canvas))
    return layers


def hook_glyph_mask(
    text: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    y_center_frac: float = DEFAULT_HOOK_FRAC,
    start_size: int = 170,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Glyph mask + bbox of the hook as the layout path renders it."""
    layer = render_hook_layer(text, canvas, y_center_frac=y_center_frac,
                              start_size=start_size)
    mask = layer[..., 3] > 0
    bbox = _bbox_of(mask)
    if bbox is None:
        raise ValueError(f"hook text rendered no glyphs: {text!r}")
    return mask, bbox


# ------------------------------------------------------- hook variants
# Foreground hook variants (extension Stage 5, 2026-11-01): the
# Stage hook sits BEHIND the speaker in the central band and earns
# its depth effect from partial occlusion. The variants instead
# put the hook on a solid brand bar IN FRONT of everything (the
# caption plane's side of the stack): "banner" across the top of
# the safe zone, "lower_third" across the bottom of the central
# band, just above the caption block. Because a foreground bar
# HIDES whatever it covers, its law is the opposite of Stage's:
# the text must not cover the face AT ALL - F2's variant leg
# measures text-over-face against the variant's own declared
# geometry and allows 0 px. Both bars are built inside the safe
# zone / central band, so F3's existing geometry check reads
# them unchanged.

HOOK_VARIANTS = ("stage", "banner", "lower_third")
HOOK_VARIANT_BAR_FILL = (11, 11, 12, 238)     # near-black, near-solid
HOOK_VARIANT_ACCENT = (255, 61, 126, 255)     # Viral Pink rule
HOOK_VARIANT_BANNER_H = 168
HOOK_VARIANT_LOWER_H = 150


def hook_variant_bar_rect(
    variant: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
) -> tuple[int, int, int, int] | None:
    """The declared bar geometry of a foreground variant.

    None for "stage" (no bar - the historic behind-speaker hook).
    Both bars span the safe width; banner tops at the central
    band's top edge, lower_third bottoms 24px above the caption
    block's top edge - inside the central band either way.
    """
    _cw, ch = canvas
    if variant == "banner":
        return (SAFE_X0, SAFE_Y0, SAFE_X1,
                SAFE_Y0 + HOOK_VARIANT_BANNER_H)
    if variant == "lower_third":
        caption_top = ch - CAPTION_BOTTOM_MARGIN \
            - caption_block_height()
        y1 = caption_top - 24
        return (SAFE_X0, y1 - HOOK_VARIANT_LOWER_H, SAFE_X1, y1)
    return None


def _variant_text_layer(
    text: str,
    canvas: tuple[int, int],
    bar: tuple[int, int, int, int],
) -> np.ndarray:
    """The variant's TEXT layer alone (no bar): white hook type
    centred in the bar, shrunk until the ink fits the bar with
    margin. The glyph mask F2 measures is this layer's alpha -
    the bar itself is furniture, the text is the claim."""
    bar_h = bar[3] - bar[1]
    y_frac = ((bar[1] + bar[3]) / 2.0) / canvas[1]
    layer = None
    for size in (96, 84, 72, 60, 48, 40):
        layer = hook_layer(
            text, canvas, y_center_frac=y_frac,
            max_width_frac=HOOK_MAX_WIDTH_FRAC, start_size=size)
        bbox = _bbox_of(layer[..., 3] > 0)
        if bbox is not None and bbox[3] - bbox[1] <= bar_h - 28 \
                and bbox[2] - bbox[0] <= (bar[2] - bar[0]) - 32:
            break
    assert layer is not None
    dx = (SAFE_X0 + SAFE_X1) // 2 - canvas[0] // 2
    if dx == 0:
        return layer
    shifted = np.zeros_like(layer)
    if dx > 0:
        shifted[:, dx:] = layer[:, :-dx]
    else:
        shifted[:, :dx] = layer[:, -dx:]
    return shifted


def hook_variant_layers(
    text: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    variant: str = "stage",
    *,
    y_center_frac: float = DEFAULT_HOOK_FRAC,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """(full layer, glyph mask, glyph bbox) for a hook variant.

    "stage" is exactly the historic layout-path hook: layer =
    render_hook_layer, glyphs = its ink. The foreground variants
    return the bar + text composited, with the glyph mask over
    the TEXT ink only.
    """
    if variant == "stage":
        layer = render_hook_layer(text, canvas,
                                  y_center_frac=y_center_frac)
        mask = layer[..., 3] > 0
        bbox = _bbox_of(mask)
        if bbox is None:
            raise ValueError(f"hook text rendered no glyphs: {text!r}")
        return layer, mask, bbox
    bar = hook_variant_bar_rect(variant, canvas)
    if bar is None:
        raise ValueError(f"unknown hook variant {variant!r}")
    text_layer = _variant_text_layer(text, canvas, bar)
    mask = text_layer[..., 3] > 0
    bbox = _bbox_of(mask)
    if bbox is None:
        raise ValueError(f"hook text rendered no glyphs: {text!r}")
    img = Image.new("RGBA", (canvas[0], canvas[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [bar[0], bar[1], bar[2] - 1, bar[3] - 1], radius=18,
        fill=HOOK_VARIANT_BAR_FILL)
    draw.rectangle(
        [bar[0] + 18, bar[3] - 12, bar[2] - 18, bar[3] - 6],
        fill=HOOK_VARIANT_ACCENT)
    img.alpha_composite(Image.fromarray(text_layer))
    return np.asarray(img), mask, bbox


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
    text_size: int = 170,
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
        glyph, bbox = hook_glyph_mask(text, canvas, y_center_frac=frac,
                                      start_size=text_size)
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
                text_size=text_size,
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

REFRAME_TARGET_FRAC = 0.5573   # of the central band: centroid y 930
#                                (fix round 2: the Stage target moved
#                                from low-centre 1187 to mid-frame 930
#                                per the 2026-11-02 geometry spike)
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
    target_cy = SAFE_Y0 + target_frac * (SAFE_Y1 - SAFE_Y0)
    cur_cx = (cw - sw * cur_scale) / 2.0 + cx * cur_scale
    if abs(cur_cy - target_cy) <= REFRAME_IDENTITY_TOL_PX \
            and SAFE_X0 <= cur_cx <= SAFE_X1:
        return SpeakerReframe(
            0, cur_scale,
            (round(cur_cx, 1), round(cur_cy, 1)),
            int(round(cur_ht)), feasible=True, identity=True,
            note="median face centroid already at the Stage target")

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


# ---------------------------------------------------------------- speaker-forward
#
# Block 2 (2026-10-31, Kamlesh's design): during a speaker_forward
# span the speaker is rendered larger and reframed toward the
# canvas centre - emphasis framing for direct-address / payoff
# moments. The mechanism is the compositor's own geometry, the
# same one the construction reframe uses: speaker_layer scales a
# frame by min(cw/w, ch/h) and bottom-anchors it, so a per-span
# CROP of the source frame (a digital zoom - smaller frame, bigger
# scale) plus a bottom PAD (lifts the placed frame, exactly like
# plan_speaker_reframe's lift) reframes the speaker without any
# resampling of the speaker's own pixels. The transform is baked
# into the span's frames + alphas, so to_canvas_alpha of a
# transformed alpha is exactly the geometry F3/F4 verify - and
# the planning below is solved in that measured space, inside
# the verifier's own limits. A span whose emphasis cannot
# satisfy them falls back to the standard framing.

EMPHASIS_SCALE = 1.18          # the emphasis the planner aims for
EMPHASIS_MIN_SCALE = 1.05      # below this the emphasis is a no-op
EMPHASIS_SCALE_STEP = 0.02     # descent while solving constraints
EMPHASIS_EASE_FRAMES = 8       # boundary ease, in frames (6-10)


@dataclass
class SpeakerEmphasis:
    """A planned per-span emphasis transform (source space)."""

    feasible: bool
    scale: float                       # achieved zoom (1.0 = none)
    crop: tuple[int, int, int, int]    # (x0, y0, w, h) source window
    pad_bottom_px: int                 # rows appended below the crop
    placed_centroid: tuple[float, float] | None
    placed_head_top_min: int | None
    note: str = ""


def plan_speaker_emphasis(
    src_alphas: Sequence[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    scale: float = EMPHASIS_SCALE,
    head_top_plan_min: int = REFRAME_HEAD_TOP_PLAN_MIN,
) -> SpeakerEmphasis:
    """Plan an emphasis crop+pad for one speaker_forward span.

    Statistics mirror plan_speaker_reframe (and therefore F3/F4):
    the median per-frame face centroid and the minimum head top.
    For each candidate zoom k (descending from ``scale``): the
    crop window is (sw/k, sh/k); its left edge is solved to centre
    the face horizontally (clamped into the frame); its top edge
    sits just above the face and never above the head top; the
    bottom pad is then solved so the placed centroid lands on the
    F3 band's centre. The candidate is feasible only if the
    EVALUATED placement (integer parameters, the compositor's
    own formulas) keeps the centroid inside the band on both
    axes, the placed head-top minimum clears
    ``head_top_plan_min`` (the 150px planning floor above F4's
    90px), and the face box fits inside the crop window (a zoom
    may crop the body's edges; it may never clip the face). If no
    k >= EMPHASIS_MIN_SCALE is feasible, the caller falls back
    to standard framing, with the reason in ``note``.
    """
    geos = [frame_geometry(a) for a in src_alphas]
    present = [g for g in geos if g.present
               and g.face_centroid is not None]
    if not present or not src_alphas:
        return SpeakerEmphasis(
            feasible=False, scale=1.0, crop=(0, 0, 0, 0),
            pad_bottom_px=0, placed_centroid=None,
            placed_head_top_min=None,
            note="no speaker geometry in the span; nothing to "
                 "emphasise")
    sh, sw = src_alphas[0].shape
    cw, ch = canvas
    cx = float(np.median([g.face_centroid[0] for g in present]))
    cy = float(np.median([g.face_centroid[1] for g in present]))
    ht_min = min(g.head_top for g in present
                 if g.head_top is not None)
    face_x0 = min(g.face_box[0] for g in present
                  if g.face_box is not None)
    face_x1 = max(g.face_box[2] for g in present
                  if g.face_box is not None)
    face_y0 = min(g.face_box[1] for g in present
                  if g.face_box is not None)
    face_y1 = max(g.face_box[3] for g in present
                  if g.face_box is not None)
    target = (cw / 2.0, (SAFE_Y0 + SAFE_Y1) / 2.0)

    k = float(scale)
    last_note = ""
    while k >= EMPHASIS_MIN_SCALE - 1e-9:
        w = max(1, int(round(sw / k)))
        h = max(1, int(round(sh / k)))
        s_nom = min(cw / w, ch / h)
        # Left edge: centre the face (ox is unchanged by a uniform
        # crop, since w * s_nom == sw * s when width-bound; solve
        # against the general formula anyway).
        ox_nom = (cw - w * s_nom) / 2.0
        x0 = int(round(cx - (target[0] - ox_nom) / s_nom))
        x0 = min(max(0, x0), sw - w)
        # Top edge: just above the face, never above the head top.
        y0 = min(max(0, face_y0 - 20), sh - h, ht_min)
        # Pad: solve oy so the centroid lands on the target row,
        # using the width-bound scale (a pad only lowers the scale
        # if it makes the frame height-bound, checked below).
        oy_want = target[1] - s_nom * (cy - y0)
        pad = int(round((ch - oy_want) / s_nom)) - h
        height = h + pad
        if height < face_y1 - y0 + 4:
            last_note = (f"pad {pad} would crop the face at scale "
                         f"{k:.2f}")
            k -= EMPHASIS_SCALE_STEP
            continue
        s2 = min(cw / w, ch / height)
        ox = (cw - w * s2) / 2.0
        oy = ch - height * s2
        placed_cx = ox + s2 * (cx - x0)
        placed_cy = oy + s2 * (cy - y0)
        placed_ht = oy + s2 * max(0, ht_min - y0)
        if not (SAFE_X0 <= placed_cx <= SAFE_X1
                and SAFE_Y0 <= placed_cy <= SAFE_Y1):
            last_note = (f"placed centroid ({placed_cx:.0f},"
                         f"{placed_cy:.0f}) outside the F3 band "
                         f"at scale {k:.2f}")
            k -= EMPHASIS_SCALE_STEP
            continue
        if placed_ht < head_top_plan_min:
            last_note = (f"head-top plan {placed_ht:.0f}px < "
                         f"{head_top_plan_min}px floor at scale "
                         f"{k:.2f}")
            k -= EMPHASIS_SCALE_STEP
            continue
        if face_x0 < x0 or face_x1 > x0 + w:
            last_note = (f"face box would clip the crop window "
                         f"at scale {k:.2f} (the speaker already "
                         f"fills the frame width; a zoom has "
                         f"nowhere to go)")
            k -= EMPHASIS_SCALE_STEP
            continue
        return SpeakerEmphasis(
            feasible=True, scale=round(k, 4),
            crop=(x0, y0, w, h), pad_bottom_px=pad,
            placed_centroid=(round(placed_cx, 1),
                             round(placed_cy, 1)),
            placed_head_top_min=int(round(placed_ht)),
            note=(f"emphasis x{k:.2f}: crop {(x0, y0, w, h)} + "
                  f"pad {pad}; centroid ({cx:.0f},{cy:.0f}) -> "
                  f"({placed_cx:.0f},{placed_cy:.0f}) canvas, "
                  f"head-top min {placed_ht:.0f}px >= "
                  f"{head_top_plan_min}px plan floor"))
    return SpeakerEmphasis(
        feasible=False, scale=1.0, crop=(0, 0, sw, sh),
        pad_bottom_px=0, placed_centroid=None,
        placed_head_top_min=None,
        note=f"emphasis infeasible inside the F3/F4 limits: "
             f"{last_note}; standard framing kept for this span")


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _emphasis_frame(frame: np.ndarray, x0: int, y0: int, w: int,
                    h: int, pad: int) -> np.ndarray:
    """One RGB frame cropped to the window + bottom pad/crop."""
    out = frame[y0:y0 + h, x0:x0 + w]
    if pad > 0:
        out = np.pad(out, ((0, pad), (0, 0), (0, 0)), mode="edge")
    elif pad < 0:
        out = out[:h + pad].copy()
    return out


def _emphasis_alpha(alpha: np.ndarray, x0: int, y0: int, w: int,
                    h: int, pad: int) -> np.ndarray:
    """One float alpha cropped to the window + bottom pad/crop."""
    out = alpha[y0:y0 + h, x0:x0 + w]
    if pad > 0:
        out = np.pad(out, ((0, pad), (0, 0)), mode="constant")
    elif pad < 0:
        out = out[:h + pad].copy()
    return out


def apply_speaker_emphasis(
    frames: list[np.ndarray],
    alphas: list[np.ndarray],
    emphasis: SpeakerEmphasis,
    start_idx: int,
    end_idx: int,
    *,
    ease_frames: int = EMPHASIS_EASE_FRAMES,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bake an emphasis crop+pad into frames/alphas in a span.

    Transitions ease over ``ease_frames`` at both span boundaries
    (smoothstep interpolation of window + pad from identity), so
    the framing never jumps. Frames outside the span - and
    boundary frames whose eased factor is exactly 0 - are shared
    untouched; the input lists are not mutated (new lists are
    returned, transformed frames are new arrays).
    """
    frames = list(frames)
    alphas = list(alphas)
    sh, sw = alphas[0].shape if alphas else (0, 0)
    cx0, cy0, cw_, ch_ = emphasis.crop
    n = max(0, end_idx - start_idx)
    for j in range(n):
        ramp = min(_smoothstep(j / ease_frames),
                   _smoothstep((n - 1 - j) / ease_frames))
        if ramp <= 0.0:
            continue
        x0 = int(round(cx0 * ramp))
        y0 = int(round(cy0 * ramp))
        w = int(round(sw + (cw_ - sw) * ramp))
        h = int(round(sh + (ch_ - sh) * ramp))
        pad = int(round(emphasis.pad_bottom_px * ramp))
        if (x0, y0, w, h, pad) == (0, 0, sw, sh, 0):
            continue
        i = start_idx + j
        frames[i] = _emphasis_frame(frames[i], x0, y0, w, h, pad)
        alphas[i] = _emphasis_alpha(alphas[i], x0, y0, w, h, pad)
    return frames, alphas


# ---------------------------------------------------------------- stage zoom
#
# Fix round 2 (Fix A, 2026-11-02): the default Stage composition.
# The legacy construction fit the whole 16:9 frame into the
# portrait canvas and parked the speaker low (placed mask height
# ~24-34% of the canvas, centroid ~1187-1542). The 2026-11-02
# geometry spike (extension-fix2-geometry-spike-2026-11-02.md)
# proved a ratio-preserving crop cannot enlarge the speaker -
# visible height is capped at source height x fit scale - so the
# Stage default is now a TALL-NARROW crop: the crop keeps the full
# mask height, its WIDTH is solved for the target placed scale,
# and a bottom pad solves the centroid, all against the untouched
# F3/F4 bands. Verified feasible on all banked profile windows
# (Holmes / Belew / McArthur, head-top 625-806 vs the 150 plan
# floor, containment 1.000).

STAGE_HEIGHT_FRAC_MIN = 0.58   # speaker visual height design band
STAGE_HEIGHT_FRAC_AIM = 0.62
STAGE_HEIGHT_FRAC_MAX = 0.66
STAGE_CENTROID_TARGET = (540.0, 930.0)
STAGE_CENTROID_Y_BAND = (880.0, 980.0)
STAGE_HEADROOM_PX = 24         # crop-top headroom above the head
STAGE_SCALE_STEP = 1.02        # width-ladder step (placed scale)


@dataclass
class StageZoom:
    """A planned Stage crop+pad enlarging the speaker."""

    feasible: bool
    identity: bool
    crop: tuple[int, int, int, int] | None  # x0, y0, w, h (source)
    pad_bottom_px: int
    scale: float                   # placed scale / fit scale
    placed_scale: float
    placed_centroid: tuple[float, float] | None
    placed_head_top_min: int | None
    height_frac: float | None      # planned placed mask height / ch
    note: str = ""

    def declaration(self) -> dict[str, Any]:
        """The receipt/summary record of this plan."""
        return {
            "feasible": self.feasible,
            "identity": self.identity,
            "scale": round(self.scale, 3),
            "crop": list(self.crop) if self.crop else None,
            "pad_bottom_px": self.pad_bottom_px,
            "centroid_target": [STAGE_CENTROID_TARGET[0],
                                STAGE_CENTROID_TARGET[1]],
            "placed_centroid": (
                [round(v, 1) for v in self.placed_centroid]
                if self.placed_centroid else None),
            "head_top_min": self.placed_head_top_min,
            "mask_height_frac": (
                round(self.height_frac, 3)
                if self.height_frac is not None else None),
            "note": self.note,
        }


def plan_stage_zoom(
    src_alphas: Sequence[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    head_top_plan_min: int = REFRAME_HEAD_TOP_PLAN_MIN,
) -> StageZoom:
    """Plan the Stage zoom for a window's finished alphas.

    Statistics mirror the verifier (median face centroid, minimum
    head top); candidates are scored hard-constraints first
    (centroid inside the F3 band, head top at the plan floor, face
    inside the crop), then the design bands (height frac
    0.58-0.66, centroid y 880-980), then closeness to the aims. A
    window whose fit already reaches the height band, or whose
    geometry admits no legal zoom, returns an honest identity or
    infeasible plan - the transform is never forced.
    """
    geos = [frame_geometry(a) for a in src_alphas]
    present = [g for g in geos if g.present
               and g.face_centroid is not None]
    if not present or not src_alphas:
        return StageZoom(
            feasible=False, identity=True, crop=None, pad_bottom_px=0,
            scale=1.0, placed_scale=0.0, placed_centroid=None,
            placed_head_top_min=None, height_frac=None,
            note="no speaker geometry in the window; nothing to place")
    sh, sw = src_alphas[0].shape
    cw, ch = canvas
    cx = float(np.median([g.face_centroid[0] for g in present]))
    cy = float(np.median([g.face_centroid[1] for g in present]))
    ht_min = min(g.head_top for g in present
                 if g.head_top is not None)
    fx0 = min(g.face_box[0] for g in present if g.face_box)
    fy0 = min(g.face_box[1] for g in present if g.face_box)
    fx1 = max(g.face_box[2] for g in present if g.face_box)
    fy1 = max(g.face_box[3] for g in present if g.face_box)
    mh_med = float(np.median(
        [g.mask_bbox[3] - g.mask_bbox[1] for g in present]))
    mask_bot = float(np.median([g.mask_bbox[3] for g in present]))
    s_fit = min(cw / sw, ch / sh)
    if mh_med * s_fit / ch >= STAGE_HEIGHT_FRAC_MIN:
        return StageZoom(
            feasible=True, identity=True, crop=None, pad_bottom_px=0,
            scale=1.0, placed_scale=round(s_fit, 4),
            placed_centroid=None, placed_head_top_min=None,
            height_frac=round(mh_med * s_fit / ch, 3),
            note="fit placement already reaches the Stage height band")
    y0 = min(max(0, ht_min - STAGE_HEADROOM_PX), ht_min)
    h = int(min(sh - y0, max(fy1 - y0, mask_bot - y0 + 8)))
    if h < fy1 - y0:
        return StageZoom(
            feasible=False, identity=True, crop=None, pad_bottom_px=0,
            scale=1.0, placed_scale=round(s_fit, 4),
            placed_centroid=None, placed_head_top_min=None,
            height_frac=None,
            note="face does not fit a mask-height crop")
    mh_vis = min(mh_med, float(h))
    s_max = cw / max(1, (fx1 - fx0) + 2)  # face + 1px insets
    s_hi = min(s_max, ch / h)
    target_x, target_y = STAGE_CENTROID_TARGET
    best: dict[str, Any] | None = None
    s = s_fit * STAGE_SCALE_STEP
    while s <= s_hi + 1e-9:
        w = min(int(np.ceil(cw / s)), sw)
        if w >= 2:
            s_w = cw / w
            x0 = min(max(0, int(round(cx - target_x / s_w))), sw - w)
            if fx0 >= x0 + 1 and fx1 <= x0 + w - 1:
                s_nom = min(cw / w, ch / h)
                oy_want = target_y - s_nom * (cy - y0)
                pad = int(round((ch - oy_want) / s_nom)) - h
                height = h + pad
                if height >= (fy1 - y0) + 4:
                    s2 = min(cw / w, ch / height)
                    ox = (cw - w * s2) / 2.0
                    oy = ch - height * s2
                    pcx = ox + s2 * (cx - x0)
                    pcy = oy + s2 * (cy - y0)
                    pht = oy + s2 * max(0, ht_min - y0)
                    frac = mh_vis * s2 / ch
                    hard = bool(SAFE_X0 <= pcx <= SAFE_X1
                                and SAFE_Y0 <= pcy <= SAFE_Y1
                                and pht >= head_top_plan_min)
                    inband = bool(
                        STAGE_HEIGHT_FRAC_MIN <= frac
                        <= STAGE_HEIGHT_FRAC_MAX
                        and STAGE_CENTROID_Y_BAND[0] <= pcy
                        <= STAGE_CENTROID_Y_BAND[1])
                    score = (0 if hard else 2) + (0 if inband else 1) \
                        + abs(frac - STAGE_HEIGHT_FRAC_AIM) \
                        + abs(pcy - target_y) / 4000.0
                    cand = {
                        "crop": (x0, y0, w, h), "pad": pad, "s2": s2,
                        "pcx": pcx, "pcy": pcy, "pht": pht,
                        "frac": frac, "hard": hard, "score": score}
                    if best is None or score < best["score"]:
                        best = cand
        s = round(s * STAGE_SCALE_STEP, 4)
    if best is None or not best["hard"]:
        return StageZoom(
            feasible=False, identity=True, crop=None, pad_bottom_px=0,
            scale=1.0, placed_scale=round(s_fit, 4),
            placed_centroid=None, placed_head_top_min=None,
            height_frac=None,
            note="no zoom satisfies the F3/F4 plan constraints")
    frac = best["frac"]
    note = (f"stage zoom x{best['s2'] / s_fit:.2f}: crop "
            f"{best['crop']} pad {best['pad']}; placed centroid "
            f"({best['pcx']:.0f},{best['pcy']:.0f}), mask height "
            f"frac {frac:.3f}")
    if not (STAGE_HEIGHT_FRAC_MIN <= frac <= STAGE_HEIGHT_FRAC_MAX):
        note += (" - closest feasible step to the "
                 f"{STAGE_HEIGHT_FRAC_MIN}-{STAGE_HEIGHT_FRAC_MAX} "
                 "design band")
    return StageZoom(
        feasible=True, identity=False, crop=best["crop"],
        pad_bottom_px=int(best["pad"]),
        scale=round(best["s2"] / s_fit, 4),
        placed_scale=round(best["s2"], 4),
        placed_centroid=(round(best["pcx"], 1), round(best["pcy"], 1)),
        placed_head_top_min=int(round(best["pht"])),
        height_frac=round(frac, 3), note=note)


def apply_stage_zoom(
    frames: list[np.ndarray],
    alphas: list[np.ndarray],
    zoom: StageZoom,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bake a StageZoom crop+pad into every frame/alpha.

    The resting Stage geometry: applied once, uniformly, before
    reframe/hook-effect planning, so every downstream consumer
    (emphasis, layout states, graphics placement, the verifier)
    sees the enlarged speaker. Identity/infeasible plans return
    the inputs untouched.
    """
    if zoom.identity or not zoom.feasible or zoom.crop is None:
        return frames, alphas
    x0, y0, w, h = zoom.crop
    pad = zoom.pad_bottom_px
    return (
        [_emphasis_frame(f, x0, y0, w, h, pad) for f in frames],
        [_emphasis_alpha(a, x0, y0, w, h, pad) for a in alphas],
    )


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


HOOK_TEXT_SIZE_LADDER = (170, 144, 122, 104)


@dataclass
class HookEffectPlan:
    """A hook placement plus the speaker transform it requires."""

    pad_bottom_px: int          # total transform to apply (source rows)
    lift_px: int                # pad beyond the base reframe's plan
    placement: HookPlacement
    geometry_ok: bool           # centroid in band + head-top floor held
    note: str = ""
    text_size: int = 170        # hook size the placement was solved at


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

    def evaluate(pad: int, text_size: int = 170
                 ) -> tuple[HookPlacement, bool]:
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
        return plan_hook(text, sample, canvas,
                         text_size=text_size), geo_ok

    def search(text_size: int) -> HookEffectPlan:
        best: HookEffectPlan | None = None
        for lift in lift_steps:
            pad = base_reframe.pad_bottom_px + int(lift)
            if lift != 0 and not (face_bottom_max + 4 <= sh + pad
                                  <= 4 * sh):
                continue  # outside the reframe clamps; not a legal lift
            placement, geo_ok = evaluate(pad, text_size)
            cand = HookEffectPlan(
                pad, int(lift), placement, geo_ok,
                note=(f"lift +{lift}px: predicted occlusion "
                      f"{placement.predicted_occlusion:.3f} "
                      f"({placement.effect_status}), geometry ok "
                      f"{geo_ok}"),
                text_size=text_size)
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
        assert best is not None  # lift_steps always contains 0
        best.note = (
            f"no lift in {list(lift_steps)} reached the effect band; "
            f"best is lift +{best.lift_px}px at predicted occlusion "
            f"{best.placement.predicted_occlusion:.3f} "
            f"({best.placement.effect_status}) - effect below target, "
            "declared")
        return best

    # Fix round 2 (Fix E.3): the hook SIZE is part of the geometry
    # solution against the enlarged Stage speaker - try the
    # historic size first, then step down; the largest size whose
    # lift search reaches the occlusion band wins. If no size
    # reaches it, the historic-size result stands (declared).
    fallback: HookEffectPlan | None = None
    for size in HOOK_TEXT_SIZE_LADDER:
        plan = search(size)
        if fallback is None:
            fallback = plan
        if plan.geometry_ok \
                and plan.placement.effect_status == "in_band":
            if size != HOOK_TEXT_SIZE_LADDER[0]:
                plan.note += (f"; hook size stepped down to {size} "
                              "to hold the occlusion band")
            return plan
    assert fallback is not None
    if len(HOOK_TEXT_SIZE_LADDER) > 1:
        fallback.note += ("; hook sizes "
                          f"{list(HOOK_TEXT_SIZE_LADDER)} all tried, "
                          "none reached the band")
    return fallback


CAPTION_PILL_FILL = (8, 8, 10, 168)
CAPTION_PILL_PAD_X = 16
CAPTION_PILL_INSET_Y = 5

# Caption style (extension Stage 4, 2026-11-01): emphasis words -
# numbers and named entities taken from the phrase text itself -
# render in Viral Pink #FF3D7E (the brand value graphics_layer
# carries as VIRAL_PINK), and a brief scale pop lands at each
# phrase start. The word-sweep highlight (the word being spoken
# right now, also pink) exists ONLY when real word timings back
# it (word_timing: DSP or MMS) - see caption_word_colors.
CAPTION_EMPHASIS_RGBA = (255, 61, 126, 255)
CAPTION_TEXT_RGBA = (255, 255, 255, 255)
CAPTION_POP_S = 0.15       # the pop's duration at a phrase start
CAPTION_POP_SCALE = 1.05   # ... and its peak scale
# Fix round 2 (Fix F, 2026-11-02): display chunking + three-state
# word colouring. A cue renders in chunks of at most
# CAPTION_CHUNK_MAX_WORDS words (timing from the word timings
# themselves - see rescue_loop's renderer), and with a sweep
# active the words carry three unmistakable states: UPCOMING dim,
# ACTIVE Viral Pink with a slight scale pop, SPOKEN plain white
# at full alpha. Emphasis words keep their pink ONLY in the
# spoken state - while a word is active the ACTIVE state wins,
# so the eye can always tell which word is being spoken now.
CAPTION_CHUNK_MAX_WORDS = 4
CAPTION_UPCOMING_RGBA = (255, 255, 255, 140)
CAPTION_ACTIVE_SCALE = 1.10

_NUMBER_WORD_TOKENS = {
    "zero", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion",
}
_TOKEN_STRIP = ".,;:!?\"'()[]$€£%"


def caption_emphasis_indices(text: str) -> set[int]:
    """Word indices (over ``text.split()``) that earn emphasis.

    Two classes, both read off the phrase text itself - nothing
    is looked up or invented:
    - numbers: a token containing a digit, or a number word
      ("twenty", "million", ...);
    - named entities: a capitalised token that does NOT start a
      sentence - in punctuation-restored caption text the only
      mid-sentence capitals are proper nouns and acronyms (the
      pronoun "I" is excluded by name).
    """
    out: set[int] = set()
    sentence_start = True
    for i, tok in enumerate(text.split()):
        core = tok.strip(_TOKEN_STRIP)
        lowered = core.lower()
        if any(ch.isdigit() for ch in tok) \
                or lowered in _NUMBER_WORD_TOKENS:
            out.add(i)
        elif not sentence_start and core and core != "I" and (
                (core[0].isupper() and any(c.islower() for c in core))
                or (len(core) >= 2 and core.isalpha()
                    and core.isupper())):
            out.add(i)
        if tok.rstrip("\"' )]").endswith((".", "!", "?")):
            sentence_start = True
        elif core:
            sentence_start = False
    return out


def caption_word_colors(text: str, active_index: int | None = None,
                        *, emphasis: set[int] | None = None,
                        spoken_count: int | None = None,
                        ) -> dict[int, tuple[int, int, int, int]] | None:
    """Per-word fill colours for one cue/chunk, or None for plain.

    Without a sweep (``active_index`` None - no real word timings
    back the render): emphasis words (numbers / named entities)
    are pink, everything else inherits the plain white path, and
    None comes back when nothing colours (the historic fast path).

    With a sweep (fix round 2 three-state colouring): EVERY word
    gets an explicit state colour - upcoming words dim white, the
    active word Viral Pink, spoken words full white - except that
    an emphasis word in the SPOKEN state keeps its pink. The
    active state always wins while a word is being spoken: an
    emphasis word is dim while upcoming and pink while active
    like any other word, and only turns emphasis-pink once it has
    been spoken. ``emphasis`` may carry a precomputed index set
    (chunk rendering slices the parent cue's set).
    """
    words = text.split()
    emph = emphasis if emphasis is not None \
        else caption_emphasis_indices(text)
    if (active_index is None or not (
            0 <= active_index < len(words))) \
            and spoken_count is None:
        colors = {i: CAPTION_EMPHASIS_RGBA for i in emph
                  if 0 <= i < len(words)}
        return colors or None
    # Three-state sweep colouring. With no word active at this
    # instant (a pause inside the chunk), ``spoken_count`` says
    # how many leading words are already spoken - the states hold
    # through the pause instead of flashing back to plain.
    n_spoken = active_index if (
        active_index is not None and 0 <= active_index < len(words)
    ) else int(spoken_count or 0)
    colors: dict[int, tuple[int, int, int, int]] = {}
    for i in range(len(words)):
        if active_index is not None and i == active_index \
                and 0 <= active_index < len(words):
            colors[i] = CAPTION_EMPHASIS_RGBA
        elif i < n_spoken:
            colors[i] = CAPTION_EMPHASIS_RGBA if i in emph \
                else CAPTION_TEXT_RGBA
        else:
            colors[i] = CAPTION_UPCOMING_RGBA
    return colors


def caption_layer(
    text: str,
    rect: tuple[int, int, int, int],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    font_size: int = CAPTION_FONT_SIZE,
    pill: bool = True,
    word_colors: dict[int, tuple[int, int, int, int]] | None = None,
    active_index: int | None = None,
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

    ``word_colors`` (Stage 4) maps word indices over the cue text
    to fill colours (see caption_word_colors): emphasis words and
    the word-sweep highlight. None - the default - renders exactly
    the historic single-draw white text.
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
    word_index = 0
    for line in lines:
        line_words = line.split()
        advances: list[float] | None = None
        if word_colors:
            # Per-word drawing (Stage 4 emphasis / sweep): each
            # word is drawn at its own advance so its fill can
            # differ. Advances come from the same font metrics;
            # the line is centred on their sum, and the pill hugs
            # exactly the drawn words, as on the plain path.
            advances = [
                draw.textlength(
                    wtok + (" " if k < len(line_words) - 1 else ""),
                    font=font)
                for k, wtok in enumerate(line_words)
            ]
            w = sum(advances)
        else:
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
        stroke = max(2, font_size // 28)
        if word_colors and advances is not None:
            # Fix F: the ACTIVE word also pops (scale
            # CAPTION_ACTIVE_SCALE, centred on its line band -
            # the same containment precedent as the phrase pop).
            active_font = None
            active_dy = 0
            if active_index is not None:
                active_size = max(
                    font_size + 1,
                    int(round(font_size * CAPTION_ACTIVE_SCALE)))
                active_font = _load_font(active_size)
                active_dy = (line_h - int(active_size * 1.25)) // 2
            for k, wtok in enumerate(line_words):
                if active_font is not None \
                        and word_index == active_index:
                    draw.text(
                        (x, y + active_dy),
                        wtok,
                        font=active_font,
                        fill=word_colors.get(word_index,
                                             CAPTION_TEXT_RGBA),
                        stroke_width=stroke,
                        stroke_fill=(10, 10, 12, 255),
                    )
                else:
                    draw.text(
                        (x, y),
                        wtok,
                        font=font,
                        fill=word_colors.get(word_index,
                                             CAPTION_TEXT_RGBA),
                        stroke_width=stroke,
                        stroke_fill=(10, 10, 12, 255),
                    )
                x += advances[k]
                word_index += 1
        else:
            draw.text(
                (x, y),
                line,
                font=font,
                fill=CAPTION_TEXT_RGBA,
                stroke_width=stroke,
                stroke_fill=(10, 10, 12, 255),
            )
            word_index += len(line_words)
        y += line_h
    return np.asarray(img)
