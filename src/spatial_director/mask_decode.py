"""SAM mask decoding: Responses payload -> finished per-frame alpha.

This module is the bridge between ``sam_client`` (which returns the
raw Responses payload) and the compositor (which needs a clean,
finished alpha matte of the speaker for every frame). It owns three
jobs, in order:

1. **Parse.** SAM's output is special-token text: per frame a
   ``<Nf>`` marker, then one record per object -
   ``id<|box;x1=..;y1=..;x2=..;y2=..;w=FW;h=FH|><|mask;x=0;y=0;data=H,W,enc|>``.
   The mask raster is BOX-sized: its ``H, W`` are the box's height and
   width, they jitter a few pixels frame to frame, and the raster
   must be placed at the box origin on the full-frame canvas - never
   treated as a fixed full-frame grid. (Wire format reconciled
   against the first live response, 2026-10-28.)

2. **Decode.** The ``enc`` payload is a custom-base85 token over a
   context-coded arithmetic stream (``~`` lossless / ``!`` one_bit) -
   it is NOT raw raster bytes, and hand-rolling the codec would be
   reimplementing a published format. Decoding therefore goes through
   Meta's official ``meta-sam-parser`` library (declared dependency,
   labeled in the README): ``decode_mask_to_raster`` thresholds
   lossless coverage at >= 129 INTERNALLY and returns a binary
   raster, so foreground here is simply value > 0 in its output.

3. **Select + finish.** The spike (2026-10-29, verdict GREEN) bought
   one pipeline rule the hard way: WIRE ORDER IS NOT SPEAKER ORDER.
   On clip C the speaker is object id 1 while id 0 is a small static
   picture-in-picture figure on a background TV; Holmes footage also
   returns flickery small background detections (a shelf cap misread
   as a person). The primary speaker is therefore selected as the
   **largest-area persistent** object id: among ids present in at
   least half the frames, the one with the largest median box area.
   The selected object's canvases then run through the finish stack -
   temporal smoothing (EMA + cut reset) followed by matte refinement
   (guided filter + choke + feather) - producing the finished
   per-frame alpha the compositor consumes.

Written fresh in-repo on 2026-10-29. NumPy + meta-sam-parser.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster

from .matte_finish import refine_matte
from .temporal_smooth import ema_smooth

# <Nf> frame markers split the output text into per-frame chunks.
_FRAME_RE = re.compile(r"<(\d+)f>")
# One object record: id + box token + mask token.
_OBJECT_RE = re.compile(
    r"(\d+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);"
    r"w=(\d+);h=(\d+)\|><\|mask;x=0;y=0;data=(\d+),(\d+),([^|]+)\|>"
)

DEFAULT_MIN_PRESENCE = 0.5


class MaskDecodeError(RuntimeError):
    """Any parse/decode failure on a SAM payload."""


@dataclass
class ObjectMask:
    """One object in one frame: box + the still-encoded mask token."""

    object_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (source pixels)
    frame_w: int
    frame_h: int
    mask_h: int
    mask_w: int
    encoding: str  # "lossless" | "one_bit"
    payload: str

    @property
    def box_area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


@dataclass
class FrameMasks:
    """All objects SAM returned for one frame."""

    index: int
    objects: list[ObjectMask] = field(default_factory=list)


# ---------------------------------------------------------------- parse


def parse_output_text(text: str) -> list[FrameMasks]:
    """Parse SAM output text into per-frame object records.

    Records keep wire order within each frame (callers must not rely
    on it - see ``select_primary_id``). Frames with no objects parse
    to an empty object list rather than being dropped, so frame
    indices always line up with the source clip.
    """
    parts = _FRAME_RE.split(text)
    # parts: [prefix, idx, chunk, idx, chunk, ...]
    frames: list[FrameMasks] = []
    for i in range(1, len(parts) - 1, 2):
        index = int(parts[i])
        chunk = parts[i + 1]
        objects: list[ObjectMask] = []
        for m in _OBJECT_RE.finditer(chunk):
            (oid, x1, y1, x2, y2, fw, fh, mh, mw, enc) = m.groups()
            objects.append(
                ObjectMask(
                    object_id=int(oid),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    frame_w=int(fw),
                    frame_h=int(fh),
                    mask_h=int(mh),
                    mask_w=int(mw),
                    encoding="lossless" if enc[0] == "~" else "one_bit",
                    payload=enc,
                )
            )
        frames.append(FrameMasks(index=index, objects=objects))
    if not frames:
        raise MaskDecodeError("no <Nf> frame records found in output text")
    return frames


def extract_output_text(raw: dict[str, Any]) -> str:
    """Pull the output_text string out of a Responses payload."""
    for item in raw.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return str(content["text"])
    raise MaskDecodeError("no output_text content in SAM response")


def parse_sam_response(raw: dict[str, Any]) -> list[FrameMasks]:
    """Parse a full SAM Responses payload into per-frame records."""
    return parse_output_text(extract_output_text(raw))


# ---------------------------------------------------------------- decode


def decode_object_raster(obj: ObjectMask) -> np.ndarray:
    """Decode one object's mask token to a bool raster (mask_h, mask_w).

    Uses Meta's official parser. Its lossless path thresholds
    coverage at >= 129 internally and returns a binary raster, so
    foreground is value > 0 - do NOT threshold again at 129 here.
    """
    mask = SegmentationMask(
        encoding=obj.encoding,
        payload=obj.payload,
        width=obj.mask_w,
        height=obj.mask_h,
    )
    raw = decode_mask_to_raster(mask)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(obj.mask_h, obj.mask_w)
    return arr > 0


def place_on_canvas(obj: ObjectMask, raster: np.ndarray) -> np.ndarray:
    """Place a box-sized raster at its box origin on a full-frame canvas.

    Raster dims jitter per frame and the box can in principle hang
    off the frame edge, so the overlap region is clipped on both
    sides; the canvas is always (frame_h, frame_w).
    """
    canvas = np.zeros((obj.frame_h, obj.frame_w), dtype=bool)
    mh, mw = raster.shape
    x1, y1 = obj.box[0], obj.box[1]
    cx1, cy1 = max(x1, 0), max(y1, 0)
    cx2, cy2 = min(x1 + mw, obj.frame_w), min(y1 + mh, obj.frame_h)
    if cx2 > cx1 and cy2 > cy1:
        canvas[cy1:cy2, cx1:cx2] = raster[
            cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1
        ]
    return canvas


def decode_object_canvas(obj: ObjectMask) -> np.ndarray:
    """Decode one object straight to a full-frame bool canvas."""
    return place_on_canvas(obj, decode_object_raster(obj))


# ---------------------------------------------------------------- select


def select_primary_id(
    frames: list[FrameMasks],
    *,
    min_presence: float = DEFAULT_MIN_PRESENCE,
) -> int:
    """Pick the primary speaker: largest-area PERSISTENT object id.

    An id is persistent if it appears in at least ``min_presence`` of
    the frames; among persistent ids the largest median box area
    wins. This is the spike's rule: the speaker dominates by area in
    every clip, while background-TV figures and flickery false
    detections are small, and a merely large but transient detection
    must not steal the track. If nothing is persistent (pathological
    input), fall back to the most-present id, then largest area, so
    the function always returns a deterministic answer.
    """
    if not frames:
        raise MaskDecodeError("no frames to select a primary object from")
    areas: dict[int, list[int]] = {}
    for frame in frames:
        for obj in frame.objects:
            areas.setdefault(obj.object_id, []).append(obj.box_area)
    if not areas:
        raise MaskDecodeError("no objects in any frame")

    n = len(frames)

    def presence(oid: int) -> float:
        return len(areas[oid]) / n

    def median_area(oid: int) -> float:
        return float(statistics.median(areas[oid]))

    persistent = [oid for oid in areas if presence(oid) >= min_presence]
    pool = persistent if persistent else list(areas)
    # Largest median area first; presence then lowest id break ties
    # deterministically.
    return sorted(
        pool, key=lambda oid: (-median_area(oid), -presence(oid), oid)
    )[0]


def primary_canvases(
    frames: list[FrameMasks],
    primary_id: int | None = None,
) -> dict[int, np.ndarray]:
    """Decode the primary object of every frame to a full-frame canvas.

    Returns {frame_index: bool canvas}. Frames where the primary is
    absent (rare flicker gaps) are simply missing from the dict;
    callers that need a dense sequence should go through
    ``finished_alpha``.
    """
    if primary_id is None:
        primary_id = select_primary_id(frames)
    out: dict[int, np.ndarray] = {}
    for frame in frames:
        for obj in frame.objects:
            if obj.object_id == primary_id:
                out[frame.index] = decode_object_canvas(obj)
                break
    return out


# ---------------------------------------------------------------- finish


def finished_alpha(
    frames: list[FrameMasks],
    guides: Iterable[np.ndarray],
    *,
    primary_id: int | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Full decode -> select -> finish pipeline for one clip.

    ``guides`` are the source frames as float greyscale arrays in
    [0, 1], aligned with the frames sorted by index (the guided
    filter pulls the matte edge onto the real image edge, so each
    guide must be the frame its mask came from). A frame where the
    primary is absent contributes an empty canvas so the sequence
    stays dense and index-aligned.

    Returns (frame_indices, alphas): float alpha in [0, 1] per frame,
    temporally smoothed then matte-refined - the compositor's input.
    """
    ordered = sorted(frames, key=lambda f: f.index)
    guide_list = list(guides)
    if len(guide_list) != len(ordered):
        raise MaskDecodeError(
            f"guide count {len(guide_list)} != frame count {len(ordered)}; "
            "guides must align 1:1 with the sorted frames"
        )
    if primary_id is None:
        primary_id = select_primary_id(ordered)
    # Frame size is a property of the clip, not of one frame: take it
    # from the first object record anywhere, so a frame whose primary
    # flickered out (or which has no objects at all) still gets a
    # correctly-shaped empty canvas.
    frame_shape: tuple[int, int] | None = None
    for frame in ordered:
        if frame.objects:
            frame_shape = (frame.objects[0].frame_h, frame.objects[0].frame_w)
            break
    if frame_shape is None:
        raise MaskDecodeError("no objects in any frame; frame size unknown")
    canvases: list[np.ndarray] = []
    for frame in ordered:
        canvas = None
        for obj in frame.objects:
            if obj.object_id == primary_id:
                canvas = decode_object_canvas(obj)
                break
        if canvas is None:
            canvas = np.zeros(frame_shape, dtype=bool)
        canvases.append(canvas)
    smoothed = ema_smooth([c.astype(np.float64) for c in canvases])
    alphas = [
        refine_matte(alpha, guide)
        for alpha, guide in zip(smoothed, guide_list)
    ]
    return [f.index for f in ordered], alphas
