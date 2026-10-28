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
