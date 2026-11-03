"""Temporal smoothing for per-frame speaker masks.

SAM segments each frame independently, so raw masks shimmer: the
boundary crawls a pixel or two between frames and the composite
looks alive in the wrong way. This module steadies the sequence
with an exponential moving average over time, with one guard:

- Scene/subject reset: if a new frame's mask barely overlaps the
  running state (IoU below the cut threshold), the state resets to
  the new frame instead of blending - otherwise a hard cut or a
  fast subject move would ghost the old silhouette into the new
  shot.

Pure NumPy, no model calls. Input masks are bool/float (H, W) per
frame; output is float alpha in [0, 1] per frame, ready for
matte_finish + compositing.

Written fresh in-repo on 2026-10-28.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

DEFAULT_ALPHA = 0.6  # weight on the current frame
DEFAULT_CUT_IOU = 0.30


def mask_iou(a: np.ndarray, b: np.ndarray, thresh: float = 0.5) -> float:
    """IoU of two masks binarised at ``thresh``."""
    ab = np.asarray(a) > thresh
    bb = np.asarray(b) > thresh
    union = np.logical_or(ab, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(ab, bb).sum() / union)


def mean_frame_delta(masks: list[np.ndarray]) -> float:
    """Mean absolute per-pixel change between consecutive frames.

    The jitter metric the spike reports before/after smoothing:
    lower is steadier.
    """
    if len(masks) < 2:
        return 0.0
    diffs = [
        float(np.mean(np.abs(np.asarray(masks[i], dtype=np.float64)
                             - np.asarray(masks[i - 1], dtype=np.float64))))
        for i in range(1, len(masks))
    ]
    return float(np.mean(diffs))


# ------------------------------------------------- edge-band blending


BAND_BLEND_MAX = 0.5     # strongest pull toward the previous frame
# Band-motion gates (mean |delta alpha| over the edge band),
# calibrated on the banked Holmes w3 window's banded-finished
# alphas (fix round 4 fixture measurement, 2026-11-02):
# percentiles p50 0.050, p75 0.055, p90 0.066, p99 0.086 - the
# EMA'd edge breathes at ~0.05 even in quiet talking, so a gate
# below that would blend nearly every frame. LO sits at ~p75:
# the quieter three quarters of frames get weight exactly 0
# (no ghosting on slow frames); HI sits at ~p99, the genuine
# head-turn tail, where the full blend applies.
BAND_MOTION_LO = 0.055
BAND_MOTION_HI = 0.086


def edge_band_motion(prev: np.ndarray, cur: np.ndarray) -> float:
    """Mean |cur - prev| over the two frames' edge-band union -
    how much the EDGE moved, ignoring the static interior."""
    from .matte_finish import edge_band

    band = edge_band(prev) | edge_band(cur)
    if not bool(band.any()):
        return 0.0
    delta = np.abs(np.asarray(cur, dtype=np.float64)
                   - np.asarray(prev, dtype=np.float64))
    return float(delta[band].mean())


def edge_band_temporal_blend(
    cur: np.ndarray,
    prev: np.ndarray,
    *,
    max_blend: float = BAND_BLEND_MAX,
    motion_lo: float = BAND_MOTION_LO,
    motion_hi: float = BAND_MOTION_HI,
) -> np.ndarray:
    """Steady the edge band of ``cur`` against ``prev``.

    Per-frame SAM masks crawl a pixel or two around hair and
    during head turns; the mask-level EMA steadies the bulk but
    its weight is uniform. This blend is confined to the edge
    band (the only place the crawl is visible) and scaled by
    the band's own motion: a still frame blends not at all (a
    slow frame can never ghost), a fast turn blends up to
    ``max_blend``. Interior and far exterior are returned
    untouched - identity outside the band by construction.
    """
    from .matte_finish import edge_band

    cur = np.clip(np.asarray(cur, dtype=np.float64), 0.0, 1.0)
    prev = np.clip(np.asarray(prev, dtype=np.float64), 0.0, 1.0)
    if cur.shape != prev.shape:
        return cur
    band = edge_band(prev) | edge_band(cur)
    if not bool(band.any()):
        return cur
    motion = float(np.abs(cur - prev)[band].mean())
    if motion_hi <= motion_lo:
        w = max_blend if motion > motion_lo else 0.0
    else:
        w = max_blend * min(1.0, max(0.0,
                                   (motion - motion_lo)
                                   / (motion_hi - motion_lo)))
    if w <= 0.0:
        return cur
    out = cur.copy()
    out[band] = (1.0 - w) * cur[band] + w * prev[band]
    return np.clip(out, 0.0, 1.0)


# -------------------------------------- sequence-level band gate


# Fix round 5 (Kamlesh's QM6 verdict: the edge flickers "as if
# struggling to keep up"). The pairwise blend above computes
# its weight from ONE frame pair's instantaneous band motion.
# Under continuous low-amplitude motion (ISS floating, slow
# head turns) that measure hovers inside the gate's ramp, so
# the WEIGHT itself oscillates at frame rate - blend, release,
# blend - and the edge pumps. The gate signal must be as
# temporal as the thing it gates:
#
# * the per-frame motion series is smoothed over a centred
#   BAND_GATE_SMOOTH_FRAMES window before it touches the gate;
# * the weight is a STATE with hysteresis: it tracks the
#   gate's target with an asymmetric slew - up at
#   BAND_WEIGHT_SLEW_UP per frame, down at only
#   BAND_WEIGHT_SLEW_DOWN - so it engages promptly but cannot
#   drop out (and snap the edge back) on a one-frame motion
#   dip, which is the pumping the pairwise gate suffers;
# * while the smoothed motion sits in the gate's mid-range
#   (the oscillation habitat - on the McArthur window the
#   median band motion IS the gate's low edge, so the pairwise
#   weight idles near zero exactly where the motion is
#   continuous), the target is floored at
#   BAND_GATE_MID_FLOOR_FRAC of the blend cap: the smoothing
#   is strengthened where the old gate under-blended.
#
# Measured on the banked McArthur window (fix5 matte proof):
# a two-frame blend reference was tried first and made the
# metric WORSE (the lagged reference ghosts during the head
# turn); the floored, asymmetrically-tracked target is the
# best gate-shaped variant, and a constant cap-strength blend
# (the family's supremum) bounds what any gate can win here -
# see the round-5 BUILDLOG entry for the full table.
BAND_GATE_SMOOTH_FRAMES = 9
BAND_WEIGHT_SLEW = 0.125      # symmetric slew (primitive)
BAND_WEIGHT_SLEW_UP = 0.25    # sequence tracker: engage rate
BAND_WEIGHT_SLEW_DOWN = 0.04  # sequence tracker: release rate
BAND_WEIGHT_DEADBAND = 0.01   # hold the weight inside this
BAND_GATE_MID_FLOOR_FRAC = 0.8  # mid-range target floor x cap


def _moving_average(values: Sequence[float],
                    window: int) -> list[float]:
    """Centred moving average; the window shrinks at the edges."""
    n = len(values)
    if n == 0 or window <= 1:
        return [float(v) for v in values]
    half = window // 2
    out: list[float] = []
    acc = [0.0]
    for v in values:
        acc.append(acc[-1] + float(v))
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append((acc[hi] - acc[lo]) / (hi - lo))
    return out


def band_motion_series(alphas: Sequence[np.ndarray]) -> list[float]:
    """Per-frame edge-band motion: element t (t >= 1) is
    edge_band_motion(alphas[t-1], alphas[t]); element 0 is 0."""
    out = [0.0]
    for t in range(1, len(alphas)):
        out.append(edge_band_motion(alphas[t - 1], alphas[t]))
    return out


def gate_weight_series(
    motions: Sequence[float],
    *,
    motion_lo: float = BAND_MOTION_LO,
    motion_hi: float = BAND_MOTION_HI,
    max_blend: float = BAND_BLEND_MAX,
    smooth_frames: int = BAND_GATE_SMOOTH_FRAMES,
    slew: float = BAND_WEIGHT_SLEW,
    deadband: float = BAND_WEIGHT_DEADBAND,
    floor_frac: float = 0.0,
    slew_up: float | None = None,
    slew_down: float | None = None,
) -> tuple[list[float], list[float]]:
    """(smoothed motions, blend weights) for a whole sequence.

    The target is the pairwise gate's ramp read on the SMOOTHED
    motion, floored at ``floor_frac`` x max_blend whenever the
    smoothed motion is at or above the gate's low edge (the
    mid-range strengthening). The returned weight is the
    hysteresis state that tracks the target: it rises at
    ``slew_up`` (default ``slew``) per frame and falls at
    ``slew_down`` (default ``slew``), holding inside
    ``deadband`` of the target - with the production sequence's
    asymmetric rates the weight cannot drop out on a one-frame
    dip, and with any rates its trajectory is incapable of
    oscillating at frame rate.
    """
    up = slew if slew_up is None else slew_up
    down = slew if slew_down is None else slew_down
    smoothed = _moving_average(list(motions), smooth_frames)
    weights: list[float] = []
    w = 0.0
    for s in smoothed:
        if motion_hi <= motion_lo:
            target = max_blend if s > motion_lo else 0.0
        else:
            target = max_blend * min(1.0, max(
                0.0, (s - motion_lo) / (motion_hi - motion_lo)))
        if floor_frac > 0.0 and s >= motion_lo:
            target = max(target, floor_frac * max_blend)
        delta = target - w
        if abs(delta) <= deadband:
            pass  # hold: inside the deadband the state is still
        elif delta > 0:
            w += min(delta, up)
        else:
            w -= min(-delta, down)
        weights.append(round(w, 4))
    return smoothed, weights


def edge_band_temporal_blend_sequence(
    alphas: list[np.ndarray],
    *,
    max_blend: float = BAND_BLEND_MAX,
    motion_lo: float = BAND_MOTION_LO,
    motion_hi: float = BAND_MOTION_HI,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """The band blend as a SEQUENCE pass (fix round 5).

    Same territory as edge_band_temporal_blend - the edge band
    only, identity outside it - but the weight comes from
    gate_weight_series run with the production shaping:
    9-frame-smoothed motion, the mid-range floor
    (BAND_GATE_MID_FLOOR_FRAC of the cap whenever the smoothed
    motion is at the gate's low edge or above), and the
    asymmetric hysteresis tracker (engage at
    BAND_WEIGHT_SLEW_UP, release at BAND_WEIGHT_SLEW_DOWN).
    The reference is the previous OUTPUT, as in round 4.
    Blending is applied in place: the returned list is the
    input list, steadied; previous outputs are read from the
    positions already written. Returns (alphas, stats) - the
    stats carry the weight trajectory and the smoothed motion
    series for the record.
    """
    from .matte_finish import edge_band

    n = len(alphas)
    stats: dict[str, Any] = {
        "method": "edge_band_gate_sequence",
        "frames": n, "max_blend": max_blend,
        "motion_lo": motion_lo, "motion_hi": motion_hi,
        "smooth_frames": BAND_GATE_SMOOTH_FRAMES,
        "slew_up": BAND_WEIGHT_SLEW_UP,
        "slew_down": BAND_WEIGHT_SLEW_DOWN,
        "mid_floor_frac": BAND_GATE_MID_FLOOR_FRAC,
        "weights": [], "motion_smoothed": []}
    if n < 2:
        return alphas, stats
    motions = band_motion_series(alphas)
    smoothed, weights = gate_weight_series(
        motions, motion_lo=motion_lo, motion_hi=motion_hi,
        max_blend=max_blend,
        floor_frac=BAND_GATE_MID_FLOOR_FRAC,
        slew_up=BAND_WEIGHT_SLEW_UP,
        slew_down=BAND_WEIGHT_SLEW_DOWN)
    stats["weights"] = weights
    stats["motion_smoothed"] = [round(s, 4) for s in smoothed]
    stats["frames_blended"] = 0
    for t in range(1, n):
        w = weights[t]
        if w <= 0.0:
            continue
        cur = np.clip(np.asarray(alphas[t], dtype=np.float64),
                      0.0, 1.0)
        prev = np.asarray(alphas[t - 1], dtype=np.float64)
        band = edge_band(prev) | edge_band(cur)
        if not bool(band.any()):
            continue
        out = cur.copy()
        out[band] = (1.0 - w) * cur[band] + w * prev[band]
        alphas[t] = np.clip(out, 0.0, 1.0)
        stats["frames_blended"] += 1
    return alphas, stats


# --------------------------------- mask-sequence smoothing (r6)


# Fix round 6 (Kamlesh's fix-5 QM6 verdict: flickering body
# edges + mask quality are the only open items). Rounds 4-5
# steadied the FINISH: banded refinement, then blend gates on
# the finished alphas. The round-5 sweep proved that family's
# ceiling - a constant cap-strength blend, the supremum of
# every gate policy, buys only -17.2% on the worst-2s flicker,
# because the blend can only scale the previous frame's edge
# into the current one. The untested lever is the MASKS
# THEMSELVES: SAM's per-frame rasters carry one-frame spikes
# and a +-1px boundary crawl, and the causal EMA (weight 0.6
# on the current frame) admits a spike at 60% strength and
# decays it over ~3 frames. A CENTRED window sees the spike
# coming and going: averaged over the window its peak weight
# is the centre weight (1/3 at radius 2), and the crawl's
# variance falls ~3x before the EMA ever sees it.
#
# Two guards keep the window honest:
# * IoU guard: a neighbour whose binarised mask barely
#   overlaps the centre frame's (IoU < MASK_SMOOTH_GUARD_IOU)
#   is a different shot or a jump cut, not a steadier reading
#   of the same silhouette - it is dropped from the window
#   (the EMA's own cut reset handles the causal side).
# * The window is triangular-weighted and renormalised over
#   the neighbours that survive, so a dropped neighbour never
#   darkens the frame - it only narrows the window.
# * Sequence ends are EDGE-REPLICATED (the first/last frame
#   pads its own side of the window): an unpadded window at
#   frame 0 would divide by the full window weight while
#   summing only the frames that exist, ghosting the speaker
#   translucent for the first `radius` frames of every clip.
#   A pad is a copy of the edge frame, so it always passes
#   its own guard and the edge frames keep full weight.
#
# The smoother is a streaming generator with a `radius`-frame
# lookahead: it holds only the window's canvases, so the
# finish's memory layout (mask_decode's OOM note) is intact.
MASK_SMOOTH_RADIUS = 2
MASK_SMOOTH_GUARD_IOU = 0.85
_MASK_SMOOTH_WEIGHTS = (1.0, 2.0, 3.0, 2.0, 1.0)  # radius 2


def smooth_mask_sequence(
    masks,
    *,
    radius: int = MASK_SMOOTH_RADIUS,
    guard_iou: float = MASK_SMOOTH_GUARD_IOU,
):
    """Centred temporal smoothing of a raw mask sequence.

    ``masks`` is any iterable of bool/float (H, W) masks (a
    generator is fine - the finish streams canvases through).
    Yields float64 masks in [0, 1], one per input, in order:
    each output is the triangular-weighted mean of the window
    [t - radius, t + radius], restricted to neighbours whose
    binarised IoU against the centre frame is at least
    ``guard_iou`` (the centre frame always counts, at the
    centre weight). Window positions beyond the sequence
    ends are filled by replicating the edge frame (see the
    module note above). Radius 0 yields the inputs as
    float64, untouched - the round-5 behaviour, kept one
    parameter away for the measurement harness.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    weights = _MASK_SMOOTH_WEIGHTS[
        MASK_SMOOTH_RADIUS - radius:
        MASK_SMOOTH_RADIUS + radius + 1] if radius <= MASK_SMOOTH_RADIUS \
        else None
    if weights is None:
        weights = tuple(
            float(radius + 1 - abs(d)) for d in range(-radius, radius + 1))
    if radius == 0:
        yield from _smooth_windowed(masks, 0, weights, guard_iou)
        return

    def _padded():
        it = iter(masks)
        try:
            first = next(it)
        except StopIteration:
            return
        for _ in range(radius):
            yield first
        yield first
        last = first
        for m in it:
            last = m
            yield m
        for _ in range(radius):
            yield last

    # The padded stream yields N + 2*radius outputs; skip the
    # first `radius` (head pads) and hold back the last
    # `radius` (tail pads), emitting exactly the N real ones.
    from collections import deque

    hold: deque = deque()
    skip = radius
    for out in _smooth_windowed(_padded(), radius, weights, guard_iou):
        if skip:
            skip -= 1
            continue
        hold.append(out)
        if len(hold) > radius:
            yield hold.popleft()


def _smooth_windowed(masks, radius, weights, guard_iou):
    """The centred-window pass over a (possibly padded) stream."""
    centre_w = weights[radius]

    def _float(m):
        return np.clip(np.asarray(m, dtype=np.float64), 0.0, 1.0)

    buf: list[np.ndarray] = []
    source = iter(masks)
    exhausted = False
    index = 0  # index of the next frame to emit
    while True:
        # Fill the buffer through index + radius.
        while not exhausted and len(buf) <= index + radius:
            try:
                buf.append(_float(next(source)))
            except StopIteration:
                exhausted = True
        if index >= len(buf):
            return
        centre = buf[index]
        acc = centre * centre_w
        wsum = centre_w
        for d in range(1, radius + 1):
            w = weights[radius - d]
            for j in (index - d, index + d):
                if 0 <= j < len(buf):
                    if mask_iou(buf[j], centre) >= guard_iou:
                        acc = acc + buf[j] * w
                        wsum += w
        yield np.clip(acc / wsum, 0.0, 1.0)
        index += 1
        # Drop frames no future window can reach.
        drop = index - radius - 1
        if drop > 0:
            del buf[:drop]
            index -= drop


def ema_smooth(
    masks: list[np.ndarray],
    *,
    alpha: float = DEFAULT_ALPHA,
    cut_iou: float = DEFAULT_CUT_IOU,
) -> list[np.ndarray]:
    """EMA-smooth a mask sequence with a scene-cut reset.

    ``alpha`` is the weight on the current frame (0.6 = responsive
    but steady). Returns float alphas in [0, 1], same length/shapes.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    out: list[np.ndarray] = []
    state: np.ndarray | None = None
    for mask in masks:
        m = np.clip(np.asarray(mask, dtype=np.float64), 0.0, 1.0)
        if state is None or mask_iou(state, m) < cut_iou:
            state = m.copy()
        else:
            state = alpha * m + (1.0 - alpha) * state
        out.append(np.clip(state, 0.0, 1.0))
    return out
