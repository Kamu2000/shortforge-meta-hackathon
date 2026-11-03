"""Layout states: per-span speaker placement for the composite.

Extension Stage 1 (2026-11-01). Until now the speaker had exactly
one placement - the compositor's fit/centre/bottom-anchor (Stage)
- plus speaker-forward emphasis, a baked crop+pad that zooms the
Stage framing. Layout states generalise the placement itself:
each graphics span may declare a ``layout`` and the speaker's
placement becomes a per-frame track of explicit placements
(crop + scale + canvas translation + alpha), eased at span
boundaries:

- ``stage``     - the shipped placement, unchanged.
- ``spotlight`` - the span's graphic is the hero: since fix
  round 4 it takes the majority region at the top of the frame
  and the speaker shrinks to a subordinate picture-in-picture
  BELOW it (``plan_spotlight_below``, solved in graphics_layer's
  hero-first Spotlight path). ``plan_spotlight`` below is the
  original top-corner PiP solver, kept as the fallback for
  spans the hero solve cannot take.
- ``split``     - the speaker moves into a bottom band (crop +
  placement solved so the face centroid stays inside the F3
  band); graphics own the top region, captions the band's lower
  zone.
- ``cutaway``   - the speaker is absent for the span; the graphic
  is full-frame.

Placements are planned from the span's SAM statistics with the
same medians the verifier measures (median face centroid, minimum
head top) and are solved INSIDE the existing F3/F4 limits - no
threshold in verifier.py changes. A plan that cannot satisfy the
limits comes back ``feasible=False`` with the reason, and the
caller keeps Stage framing for that span (the speaker-forward
fallback discipline, applied to states).

The render path is the compositor's own: ``speaker_layer`` takes
the placement, and ``layout_law.to_canvas_alpha`` maps alphas
through the identical transform, so the geometry F3/F4 will
measure is the geometry planned here - never a parallel
approximation.

A span may also declare ``camera``: ``punch_in`` (a slow zoom
about the placed face centroid) or ``drift`` (a slow canvas
translation), and a ``follow_track`` flag: when a per-frame
source-space centroid trajectory is supplied, the span's crop
centre follows its smoothed path (the tracked-reframe input;
the planner wiring that supplies trajectories is Tier C).

Written fresh in-repo on 2026-11-01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .compositor import DEFAULT_CANVAS
from .layout_law import (
    SAFE_X0,
    SAFE_X1,
    SAFE_Y0,
    SAFE_Y1,
    frame_geometry,
    initial_caption_rect,
)

LAYOUT_STATES = ("stage", "spotlight", "split", "cutaway")
CAMERA_TYPES = ("punch_in", "drift")

TRANSITION_S = 0.30          # state-boundary ease (ease-in-out)
PIP_WIDTH_PX = 430           # Spotlight PiP frame width on canvas
PIP_INSET_PX = 90            # PiP inset from the top/side edges
PIP_TARGET_CENTROID_Y = 470.0  # placed face-centroid aim (in band)
SPLIT_BAND_FRAC = 0.35       # Split: speaker band = bottom 35%
SPLIT_CENTROID_SLACK_PX = 10   # plan inside the F3 edge, not on it
_F4_HEAD_MARGIN_PX = 90      # verifier F4's floor (read-only mirror)


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------- placement


@dataclass
class SpeakerPlacement:
    """An explicit speaker placement (compositor space).

    The source frame is cropped to ``crop`` (x0, y0, w, h in source
    px; None = the whole frame), scaled by ``scale``, and pasted
    with its top-left at (``ox``, ``oy``) on the canvas; the alpha
    is multiplied by ``alpha_scale`` (Cutaway fades the speaker
    out with it). All fields are floats; the compositor rounds at
    paste time, exactly as the legacy fit path rounds its sizes.
    """

    crop: tuple[float, float, float, float] | None
    scale: float
    ox: float
    oy: float
    alpha_scale: float = 1.0

    @classmethod
    def stage(cls, frame_shape: tuple[int, int],
              canvas: tuple[int, int] = DEFAULT_CANVAS
              ) -> "SpeakerPlacement":
        """The explicit placement equal to the legacy fit.

        Computed with the legacy path's own rounding (round sizes,
        floor-centred x, bottom-anchored y), so rendering through
        the placement branch reproduces the historic pixels
        byte-for-byte (asserted in tests).
        """
        sh, sw = frame_shape
        cw, ch = canvas
        scale = min(cw / sw, ch / sh)
        nw = max(1, round(sw * scale))
        nh = max(1, round(sh * scale))
        return cls(crop=None, scale=scale, ox=(cw - nw) // 2,
                   oy=ch - nh, alpha_scale=1.0)

    def normalized(self, frame_shape: tuple[int, int]
                   ) -> "SpeakerPlacement":
        """A copy with ``crop=None`` replaced by the full frame, so
        interpolation never has to special-case it."""
        if self.crop is not None:
            return self
        sh, sw = frame_shape
        return SpeakerPlacement(
            crop=(0.0, 0.0, float(sw), float(sh)), scale=self.scale,
            ox=self.ox, oy=self.oy, alpha_scale=self.alpha_scale)

    def lerp(self, other: "SpeakerPlacement", t: float
             ) -> "SpeakerPlacement":
        """Interpolate toward ``other`` by ``t`` in [0, 1].

        Both placements must be normalized (explicit crops). The
        placed rectangle's corners are affine in the fields, so a
        frame whose endpoints keep the speaker on-canvas keeps it
        on-canvas throughout the blend - the transition-containment
        property the tests assert on real geometry.
        """
        assert self.crop is not None and other.crop is not None

        def m(a: float, b: float) -> float:
            return a + (b - a) * t

        return SpeakerPlacement(
            crop=tuple(m(a, b) for a, b in zip(self.crop, other.crop)),  # type: ignore[arg-type]
            scale=m(self.scale, other.scale),
            ox=m(self.ox, other.ox), oy=m(self.oy, other.oy),
            alpha_scale=m(self.alpha_scale, other.alpha_scale))

    def placed_point(self, x: float, y: float) -> tuple[float, float]:
        """Canvas position of a source-space point under this
        placement."""
        cx0, cy0 = (self.crop[0], self.crop[1]) if self.crop else (0.0, 0.0)
        return (self.ox + (x - cx0) * self.scale,
                self.oy + (y - cy0) * self.scale)

    def frame_box(self) -> tuple[float, float, float, float]:
        """Canvas box of the placed (cropped) frame."""
        w = self.crop[2] if self.crop else 0.0
        h = self.crop[3] if self.crop else 0.0
        return (self.ox, self.oy, self.ox + w * self.scale,
                self.oy + h * self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop": ([round(v, 2) for v in self.crop]
                     if self.crop is not None else None),
            "scale": round(self.scale, 4),
            "origin": [round(self.ox, 1), round(self.oy, 1)],
            "alpha_scale": round(self.alpha_scale, 3),
        }


# ---------------------------------------------------------------- statistics


@dataclass
class SpanStats:
    """SAM statistics of one span, in source-frame coordinates.

    The same statistics the shipped planners solve against:
    median per-frame face centroid, the centroid's y range, the
    minimum per-frame head top, and the union boxes.
    """

    n: int
    cx_med: float
    cy_med: float
    cy_min: float
    cy_max: float
    ht_min: int
    face_union: tuple[int, int, int, int]
    mask_union: tuple[int, int, int, int]

    @classmethod
    def from_alphas(cls, src_alphas: Sequence[np.ndarray]
                    ) -> "SpanStats | None":
        cents, tops, faces, masks = [], [], [], []
        for a in src_alphas:
            g = frame_geometry(a)
            if not g.present:
                continue
            if g.face_centroid is not None:
                cents.append(g.face_centroid)
                faces.append(g.face_box)
            tops.append(g.head_top)
            masks.append(g.mask_bbox)
        if not cents:
            return None
        cents_arr = np.array(cents)
        return cls(
            n=len(cents),
            cx_med=float(np.median(cents_arr[:, 0])),
            cy_med=float(np.median(cents_arr[:, 1])),
            cy_min=float(cents_arr[:, 1].min()),
            cy_max=float(cents_arr[:, 1].max()),
            ht_min=int(min(tops)),
            face_union=(min(f[0] for f in faces),
                        min(f[1] for f in faces),
                        max(f[2] for f in faces),
                        max(f[3] for f in faces)),
            mask_union=(min(m[0] for m in masks),
                        min(m[1] for m in masks),
                        max(m[2] for m in masks),
                        max(m[3] for m in masks)),
        )


# ---------------------------------------------------------------- plans


@dataclass
class LayoutPlan:
    """One span's solved layout: its resting placement + evidence.

    ``feasible`` is the planner's own verdict against the F3/F4
    limits (median centroid in band, head margin, face unclipped
    by canvas edges). Infeasible plans are returned, not raised,
    with the reason in ``note`` - the caller falls back to Stage.
    """

    state: str
    placement: SpeakerPlacement
    feasible: bool
    note: str = ""
    centroid_src: tuple[float, float] | None = None
    placed_centroid: tuple[float, float] | None = None
    head_top_min: float | None = None
    graphic_zone: tuple[int, int, int, int] | None = None
    caption_rect: tuple[int, int, int, int] | None = None

    def speaker_box(self) -> list[int]:
        return [int(round(v)) for v in self.placement.frame_box()]

    def declaration(self, start_s: float, end_s: float,
                    camera: dict[str, Any] | None = None,
                    graphic_box: Sequence[int] | None = None
                    ) -> dict[str, Any]:
        """The receipt-facing record for this span (additive fields;
        receipt_version stays 1)."""
        out: dict[str, Any] = {
            "span": [round(start_s, 3), round(end_s, 3)],
            "layout": self.state,
            "speaker_box": self.speaker_box(),
            "speaker_placement": self.placement.to_dict(),
            "placed_centroid": (
                [round(v, 1) for v in self.placed_centroid]
                if self.placed_centroid is not None else None),
            "head_top_min": (int(round(self.head_top_min))
                             if self.head_top_min is not None
                             else None),
            "feasible": self.feasible,
            "note": self.note,
        }
        if camera is not None:
            out["camera"] = camera
        if graphic_box is not None:
            out["graphic_box"] = [int(v) for v in graphic_box]
        if self.graphic_zone is not None:
            out["graphic_zone"] = [int(v) for v in self.graphic_zone]
        if self.caption_rect is not None:
            out["caption_rect"] = [int(v) for v in self.caption_rect]
        return out


def plan_stage(frame_shape: tuple[int, int],
               canvas: tuple[int, int] = DEFAULT_CANVAS) -> LayoutPlan:
    return LayoutPlan(
        state="stage",
        placement=SpeakerPlacement.stage(frame_shape, canvas)
        .normalized(frame_shape),
        feasible=True, note="the shipped placement, unchanged")


def plan_spotlight(
    src_alphas: Sequence[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    corner: str = "left",
    pip_width: int = PIP_WIDTH_PX,
    target_centroid_y: float = PIP_TARGET_CENTROID_Y,
    stats: SpanStats | None = None,
) -> LayoutPlan:
    """Solve the Spotlight PiP placement for one span.

    The PiP frame is ``pip_width`` wide on canvas, inset
    ``PIP_INSET_PX`` from the chosen top corner horizontally; its
    vertical origin is solved so the placed median face centroid
    lands on ``target_centroid_y`` (inside the F3 band by design -
    the brief's Stage-0 spike is what proved the band is reachable
    at this size on the real sources). Feasibility is evaluated
    with the verifier's own statistics: median centroid in the
    band, minimum head top at/above F4's 90px floor, the placed
    face clear of the caption rect vertically (face bottom above
    the caption zone's top - the F2 caption rule), and the face
    unclipped by the canvas side edges (the F4 containment rule).
    """
    st = stats or SpanStats.from_alphas(src_alphas)
    frame_shape = src_alphas[0].shape if src_alphas else (0, 0)
    if st is None:
        return LayoutPlan(
            state="spotlight",
            placement=SpeakerPlacement.stage(frame_shape, canvas)
            .normalized(frame_shape),
            feasible=False,
            note="no speaker geometry in the span; Stage kept")
    fh, fw = frame_shape
    cw, ch = canvas
    s = pip_width / fw
    ox = PIP_INSET_PX if corner == "left" else cw - PIP_INSET_PX - pip_width
    oy = target_centroid_y - st.cy_med * s
    placement = SpeakerPlacement(crop=None, scale=s, ox=float(ox),
                                 oy=float(oy)).normalized(frame_shape)
    cx = placement.placed_point(st.cx_med, st.cy_med)[0]
    cy = placement.placed_point(st.cx_med, st.cy_med)[1]
    ht = placement.placed_point(0.0, float(st.ht_min))[1]
    face_x0 = placement.placed_point(float(st.face_union[0]), 0.0)[0]
    face_x1 = placement.placed_point(float(st.face_union[2]), 0.0)[0]
    face_y1 = placement.placed_point(0.0, float(st.face_union[3]))[1]
    caption_top = initial_caption_rect(canvas)[1]
    problems = []
    if not (SAFE_Y0 <= cy <= SAFE_Y1):
        problems.append(f"placed centroid y {cy:.0f} outside the "
                        f"F3 band [{SAFE_Y0}, {SAFE_Y1}]")
    if not (SAFE_X0 <= cx <= SAFE_X1):
        problems.append(f"placed centroid x {cx:.0f} outside "
                        f"[{SAFE_X0}, {SAFE_X1}]")
    if ht < _F4_HEAD_MARGIN_PX:
        problems.append(f"head-top min {ht:.0f}px < F4's "
                        f"{_F4_HEAD_MARGIN_PX}px floor")
    if face_y1 > caption_top:
        problems.append(f"face bottom {face_y1:.0f} enters the "
                        f"caption zone (top {caption_top}) - the F2 "
                        f"caption rule would fail")
    if face_x0 <= 0 or face_x1 >= cw - 1:
        problems.append("placed face would clip a canvas side edge "
                        "(F4 containment)")
    return LayoutPlan(
        state="spotlight", placement=placement,
        feasible=not problems,
        note=("; ".join(problems) if problems else
              f"PiP {corner} {pip_width}px wide at ({ox}, {oy:.0f}): "
              f"centroid ({cx:.0f}, {cy:.0f}), head-top min "
              f"{ht:.0f}px, face bottom {face_y1:.0f} above caption "
              f"top {caption_top}"),
        centroid_src=(st.cx_med, st.cy_med),
        placed_centroid=(round(cx, 1), round(cy, 1)),
        head_top_min=ht)


SPOTLIGHT_PIP_FACE_MIN_PX = 180  # placed face height floor (legibility)
SPOTLIGHT_PIP_HEADROOM_FRAC = 0.30   # face-crop headroom, x face height
SPOTLIGHT_PIP_SHOULDER_FRAC = 0.90   # face-crop below-chin reach, x face h
SPOTLIGHT_PIP_SIDE_FRAC = 0.45       # face-crop side margin, x face width


def plan_spotlight_below(
    src_alphas: Sequence[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    band: tuple[int, int],
    max_width: int,
    max_area: int,
    stats: SpanStats | None = None,
) -> LayoutPlan:
    """Solve the Spotlight PiP BELOW the hero graphic (fix round 4).

    The fix-round-1..3 Spotlight put the PiP in a top corner at a
    fixed width and let the graphic take whatever space remained -
    live, that inverted the state: the QM5 hamburger span declared
    "spotlight" with a 285px graphic beside a 430px PiP. Kamlesh's
    arrangement (2026-11-02): the graphic is the hero on top, the
    speaker is the subordinate element BELOW it, captions at the
    very bottom. This solver places the PiP inside ``band`` =
    (y0, y1) - the strip between the hero box's bottom (+ gap) and
    the caption zone's top - subject to the caller's dominance
    caps: placed width at most ``max_width`` (the hero's width)
    and placed area at most ``max_area`` (a fraction of the hero's
    area, so the ordering hero > PiP holds by construction).

    Two crop candidates, in order: the full frame (most context),
    then a face-union crop (headroom + shoulders + side margins,
    clamped to the frame) for bands the full frame cannot fit at
    the face-legibility floor. The placed face centroid aims at
    the band's centre; the origin is then clamped so the PiP box
    stays inside the band, and feasibility is re-evaluated on the
    clamped geometry with the same limits plan_spotlight uses
    (F3 band, F4 head margin, face clear of the caption zone -
    here by construction, since the PiP box ends at or above the
    band's bottom) plus the legibility floor: placed face height
    >= SPOTLIGHT_PIP_FACE_MIN_PX. Infeasible returns the plan with
    the reason, exactly like the other solvers - the caller falls
    back honestly.
    """
    st = stats or SpanStats.from_alphas(src_alphas)
    frame_shape = src_alphas[0].shape if src_alphas else (0, 0)
    if st is None:
        return LayoutPlan(
            state="spotlight",
            placement=SpeakerPlacement.stage(frame_shape, canvas)
            .normalized(frame_shape),
            feasible=False,
            note="no speaker geometry in the span; Stage kept")
    fh, fw = frame_shape
    cw, _ch = canvas
    band_y0, band_y1 = int(band[0]), int(band[1])
    band_h = band_y1 - band_y0
    fx0, fy0, fx1, fy1 = (float(v) for v in st.face_union)
    face_h = max(1.0, fy1 - fy0)
    face_w = max(1.0, fx1 - fx0)
    caption_top = initial_caption_rect(canvas)[1]

    crops: list[tuple[float, float, float, float] | None] = [None]
    cx0 = max(0.0, fx0 - SPOTLIGHT_PIP_SIDE_FRAC * face_w)
    cx1 = min(float(fw), fx1 + SPOTLIGHT_PIP_SIDE_FRAC * face_w)
    cy0 = max(0.0, fy0 - SPOTLIGHT_PIP_HEADROOM_FRAC * face_h)
    cy1 = min(float(fh), fy1 + SPOTLIGHT_PIP_SHOULDER_FRAC * face_h)
    if cx1 - cx0 >= face_w and cy1 - cy0 >= face_h:
        crops.append((cx0, cy0, cx1 - cx0, cy1 - cy0))

    problems: list[str] = []
    for crop in crops:
        if crop is None:
            crop_rect = (0.0, 0.0, float(fw), float(fh))
        else:
            crop_rect = crop
        ccx0, ccy0, ccw, cch = crop_rect
        if ccw <= 0 or cch <= 0:
            continue
        s = min(band_h / cch, max_width / ccw)
        if s <= 0:
            continue
        area = ccw * s * cch * s
        if area > max_area:
            s *= (max_area / area) ** 0.5
        placed_w, placed_h = ccw * s, cch * s
        placed_face_h = face_h * s
        if placed_face_h < SPOTLIGHT_PIP_FACE_MIN_PX:
            problems.append(
                f"{'full frame' if crop is None else 'face crop'}: "
                f"placed face height {placed_face_h:.0f}px < the "
                f"{SPOTLIGHT_PIP_FACE_MIN_PX}px legibility floor")
            continue
        ox = (cw - placed_w) / 2.0
        band_cy = (band_y0 + band_y1) / 2.0
        oy = band_cy - (st.cy_med - ccy0) * s
        oy = min(max(oy, float(band_y0)), band_y1 - placed_h)
        placement = SpeakerPlacement(
            crop=crop, scale=s, ox=float(ox), oy=float(oy)) \
            .normalized(frame_shape)
        cx = placement.placed_point(st.cx_med, st.cy_med)[0]
        cy = placement.placed_point(st.cx_med, st.cy_med)[1]
        ht = placement.placed_point(0.0, float(st.ht_min))[1]
        face_y1 = placement.placed_point(0.0, fy1)[1]
        cand_problems = []
        if not (SAFE_Y0 <= cy <= SAFE_Y1):
            cand_problems.append(
                f"placed centroid y {cy:.0f} outside the F3 band "
                f"[{SAFE_Y0}, {SAFE_Y1}]")
        if not (SAFE_X0 <= cx <= SAFE_X1):
            cand_problems.append(
                f"placed centroid x {cx:.0f} outside "
                f"[{SAFE_X0}, {SAFE_X1}]")
        if ht < _F4_HEAD_MARGIN_PX:
            cand_problems.append(
                f"head-top min {ht:.0f}px < F4's "
                f"{_F4_HEAD_MARGIN_PX}px floor")
        if face_y1 > caption_top:
            cand_problems.append(
                f"face bottom {face_y1:.0f} enters the caption "
                f"zone (top {caption_top})")
        box = placement.frame_box()
        if box[0] < 0 or box[2] > cw or box[1] < band_y0 - 1 \
                or box[3] > band_y1 + 1:
            cand_problems.append(
                "PiP box escapes the band or the canvas sides")
        if cand_problems:
            problems.extend(cand_problems)
            continue
        kind = "full frame" if crop is None else "face crop"
        return LayoutPlan(
            state="spotlight", placement=placement, feasible=True,
            note=(f"PiP below hero ({kind}) "
                  f"{placed_w:.0f}x{placed_h:.0f}px at "
                  f"({ox:.0f}, {oy:.0f}) in band "
                  f"[{band_y0}, {band_y1}]: centroid "
                  f"({cx:.0f}, {cy:.0f}), head-top min {ht:.0f}px, "
                  f"placed face height {placed_face_h:.0f}px"),
            centroid_src=(st.cx_med, st.cy_med),
            placed_centroid=(round(cx, 1), round(cy, 1)),
            head_top_min=ht)
    return LayoutPlan(
        state="spotlight",
        placement=SpeakerPlacement.stage(frame_shape, canvas)
        .normalized(frame_shape),
        feasible=False,
        note=("; ".join(problems) if problems else
              "no PiP candidate fits the band below the hero"),
        centroid_src=(st.cx_med, st.cy_med))


def plan_split(
    src_alphas: Sequence[np.ndarray],
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    band_frac: float = SPLIT_BAND_FRAC,
    stats: SpanStats | None = None,
) -> LayoutPlan:
    """Solve the Split placement: speaker into the bottom band.

    A crop descends from just above the face (the emphasis
    mechanism's insight: the full frame's low face position is
    what breaks the band) until the placed crop, fitted into the
    band, lands the median face centroid inside the F3 band with
    slack to spare. The largest feasible scale wins (least crop).
    If no crop satisfies the band at this ``band_frac``, the plan
    is infeasible with the best centroid achieved in the note -
    the caller may retry a taller band or keep Stage; the F3
    threshold itself never moves.
    """
    st = stats or SpanStats.from_alphas(src_alphas)
    frame_shape = src_alphas[0].shape if src_alphas else (0, 0)
    if st is None:
        return LayoutPlan(
            state="split",
            placement=SpeakerPlacement.stage(frame_shape, canvas)
            .normalized(frame_shape),
            feasible=False,
            note="no speaker geometry in the span; Stage kept")
    fh, fw = frame_shape
    cw, ch = canvas
    band_h = round(ch * band_frac)
    band_top = ch - band_h
    fy0 = st.face_union[1]
    best: LayoutPlan | None = None
    best_cy = float("inf")
    for crop_y0 in range(max(0, fy0 - 200), fy0 + 1, 4):
        crop = (0.0, float(crop_y0), float(fw), float(fh - crop_y0))
        s = min(cw / fw, band_h / crop[3])
        oy = ch - crop[3] * s
        ox = (cw - fw * s) / 2.0
        placement = SpeakerPlacement(crop=crop, scale=s, ox=ox, oy=oy)
        cx = placement.placed_point(st.cx_med, st.cy_med)[0]
        cy = placement.placed_point(st.cx_med, st.cy_med)[1]
        ht = placement.placed_point(0.0, float(st.ht_min))[1]
        fx0 = placement.placed_point(float(st.face_union[0]), 0.0)[0]
        fx1 = placement.placed_point(float(st.face_union[2]), 0.0)[0]
        best_cy = min(best_cy, cy)
        if not (cy <= SAFE_Y1 - SPLIT_CENTROID_SLACK_PX
                and SAFE_X0 <= cx <= SAFE_X1 and ht >= _F4_HEAD_MARGIN_PX
                and fx0 > 0 and fx1 < cw - 1):
            continue
        cand = LayoutPlan(
            state="split", placement=placement, feasible=True,
            note=(f"crop y0 {crop_y0}, scale {s:.3f}: centroid "
                  f"({cx:.0f}, {cy:.0f}) inside the band with "
                  f"{SAFE_Y1 - cy:.0f}px slack, head-top min "
                  f"{ht:.0f}px"),
            centroid_src=(st.cx_med, st.cy_med),
            placed_centroid=(round(cx, 1), round(cy, 1)),
            head_top_min=ht,
            graphic_zone=(0, 0, cw, band_top),
            caption_rect=initial_caption_rect(canvas))
        if best is None or s > best.placement.scale:
            best = cand
    if best is not None:
        return best
    return LayoutPlan(
        state="split",
        placement=SpeakerPlacement.stage(frame_shape, canvas)
        .normalized(frame_shape),
        feasible=False,
        note=(f"no crop lands the face centroid inside the F3 band "
              f"at band_frac {band_frac}: best centroid y "
              f"{best_cy:.0f} vs the {SAFE_Y1} edge; Stage kept"),
        centroid_src=(st.cx_med, st.cy_med),
        graphic_zone=(0, 0, cw, band_top),
        caption_rect=initial_caption_rect(canvas))


def plan_cutaway(frame_shape: tuple[int, int],
                 canvas: tuple[int, int] = DEFAULT_CANVAS) -> LayoutPlan:
    """The Cutaway placement: Stage geometry, alpha scaled to 0.

    The transition machinery eases alpha_scale 1 -> 0 at the span
    edges, so the speaker fades out/in around the full-frame
    graphic instead of popping.
    """
    placement = SpeakerPlacement.stage(frame_shape, canvas) \
        .normalized(frame_shape)
    placement.alpha_scale = 0.0
    return LayoutPlan(
        state="cutaway", placement=placement, feasible=True,
        note="speaker absent for the span (alpha eased to 0); the "
             "graphic is full-frame")


# ---------------------------------------------------------------- camera + track


@dataclass
class LayoutDecl:
    """One span's layout declaration (what the graphics plan says)."""

    start_s: float
    end_s: float
    state: str
    camera: dict[str, Any] | None = None
    follow_track: bool = False


def validate_camera(camera: Any) -> dict[str, Any] | None:
    """Normalize a declared camera dict, or None if unusable."""
    if not isinstance(camera, dict):
        return None
    ctype = camera.get("type")
    if ctype not in CAMERA_TYPES:
        return None
    if ctype == "punch_in":
        try:
            frm, to = float(camera["from"]), float(camera["to"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0.5 <= frm <= 2.0 and 0.5 <= to <= 2.0):
            return None
        return {"type": "punch_in", "from": frm, "to": to}
    try:
        frm = [float(v) for v in camera["from"]]
        to = [float(v) for v in camera["to"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(frm) != 2 or len(to) != 2:
        return None
    return {"type": "drift", "from": frm, "to": to}


def _camera_placement(base: SpeakerPlacement, camera: dict[str, Any],
                      progress: float,
                      plan: LayoutPlan) -> SpeakerPlacement:
    """The span's placement at ``progress`` under its camera move.

    punch_in zooms about the placed face centroid (the anchor is
    invariant by construction); drift translates in canvas px.
    """
    if camera["type"] == "punch_in":
        k = camera["from"] + (camera["to"] - camera["from"]) * progress
        if plan.centroid_src is None:
            return base
        ax, ay = base.placed_point(*plan.centroid_src)
        s2 = base.scale * k
        cx0, cy0 = (base.crop[0], base.crop[1]) if base.crop else (0.0, 0.0)
        return SpeakerPlacement(
            crop=base.crop, scale=s2,
            ox=ax - (plan.centroid_src[0] - cx0) * s2,
            oy=ay - (plan.centroid_src[1] - cy0) * s2,
            alpha_scale=base.alpha_scale)
    dx = camera["from"][0] + (camera["to"][0] - camera["from"][0]) * progress
    dy = camera["from"][1] + (camera["to"][1] - camera["from"][1]) * progress
    return SpeakerPlacement(
        crop=base.crop, scale=base.scale, ox=base.ox + dx,
        oy=base.oy + dy, alpha_scale=base.alpha_scale)


def smooth_trajectory(track: Sequence[tuple[float, float]],
                      window: int) -> list[tuple[float, float]]:
    """Centered moving average of a per-frame centroid trajectory
    (the tracked-reframe smoothing: the frame moves gradually -
    smoothing beats detector precision)."""
    if window <= 1 or not track:
        return [tuple(p) for p in track]
    half = window // 2
    out = []
    for i in range(len(track)):
        seg = track[max(0, i - half):i + half + 1]
        out.append((float(np.mean([p[0] for p in seg])),
                    float(np.mean([p[1] for p in seg]))))
    return out


def build_track(
    decls: Sequence[LayoutDecl],
    plans: Sequence[LayoutPlan],
    frame_shape: tuple[int, int],
    n_frames: int,
    fps: float,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    transition_s: float = TRANSITION_S,
    track: Sequence[tuple[float, float]] | None = None,
) -> list[SpeakerPlacement | None]:
    """Per-frame speaker placements for a whole Short.

    Returns a list aligned with the frames: None where the Stage
    (legacy) placement applies untouched, an explicit placement
    elsewhere - inside spans and across the eased transition
    windows that straddle their boundaries (each boundary blends
    over ``transition_s`` centred on the boundary, smoothstep).
    Segment placements are evaluated as functions of time (camera
    progress, tracked crop), and a boundary blend evaluates BOTH
    sides at the frame's own instant, so camera motion stays
    continuous through the ease.

    ``track`` is the optional per-frame source-space centroid
    trajectory for follow_track spans (tracked reframe input).
    """
    stage = SpeakerPlacement.stage(frame_shape, canvas).normalized(
        frame_shape)
    smoothed = (smooth_trajectory(track, max(3, int(0.5 * fps)))
                if track is not None else None)

    # Segments: [start_s, end_s, placement_fn(t) -> SpeakerPlacement]
    segments: list[tuple[float, float, Any]] = []
    cursor = 0.0
    duration = n_frames / fps if fps else 0.0
    pairs = sorted(zip(decls, plans), key=lambda dp: dp[0].start_s)
    for decl, plan in pairs:
        if decl.start_s > cursor:
            segments.append((cursor, decl.start_s, lambda t: stage))
        base = plan.placement.normalized(frame_shape)
        camera = validate_camera(decl.camera) if decl.camera else None

        def span_fn(t: float, _decl=decl, _plan=plan, _base=base,
                    _camera=camera) -> SpeakerPlacement:
            span_len = max(_decl.end_s - _decl.start_s, 1e-9)
            p = min(1.0, max(0.0, (t - _decl.start_s) / span_len))
            out = _base
            if _camera is not None:
                out = _camera_placement(out, _camera, p, _plan)
            if _decl.follow_track and smoothed is not None \
                    and out.crop is not None:
                idx = min(len(smoothed) - 1, max(0, int(round(t * fps))))
                cx, cy = smoothed[idx]
                x0 = min(max(0.0, cx - out.crop[2] / 2.0),
                         max(0.0, frame_shape[1] - out.crop[2]))
                y0 = min(max(0.0, cy - out.crop[3] / 2.0),
                         max(0.0, frame_shape[0] - out.crop[3]))
                out = SpeakerPlacement(
                    crop=(x0, y0, out.crop[2], out.crop[3]),
                    scale=out.scale, ox=out.ox, oy=out.oy,
                    alpha_scale=out.alpha_scale)
            return out

        segments.append((decl.start_s, decl.end_s, span_fn))
        cursor = max(cursor, decl.end_s)
    if cursor < duration:
        segments.append((cursor, duration, lambda t: stage))
    if not segments:
        return [None] * n_frames

    def resting(i: int, t: float) -> SpeakerPlacement:
        for s0, s1, fn in segments:
            if s0 <= t < s1 or (i == n_frames - 1 and t <= s1):
                return fn(t)
        return segments[-1][2](t)

    boundaries = [seg[0] for seg in segments[1:]]
    half = transition_s / 2.0
    out: list[SpeakerPlacement | None] = []
    for i in range(n_frames):
        t = i / fps if fps else 0.0
        placement = resting(i, t)
        for b in boundaries:
            if abs(t - b) < half and transition_s > 0:
                # Evaluate both sides at this instant: the segment
                # ending at b (progress clamped to its end) and
                # the one starting at b.
                before = None
                after = None
                for s0, s1, fn in segments:
                    if abs(s1 - b) < 1e-9:
                        before = fn(t)
                    if abs(s0 - b) < 1e-9:
                        after = fn(t)
                if before is not None and after is not None:
                    f = _smoothstep((t - (b - half)) / transition_s)
                    placement = before.lerp(after, f)
                break
        if placement is stage or (placement.crop == stage.crop
                                  and placement.scale == stage.scale
                                  and placement.ox == stage.ox
                                  and placement.oy == stage.oy
                                  and placement.alpha_scale == 1.0):
            out.append(None)
        else:
            out.append(placement)
    return out
