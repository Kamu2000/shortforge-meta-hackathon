"""Matte finishing for SAM speaker masks: refine, choke, feather.

The composite's credibility lives at the hair/shoulder boundary. A
raw model mask pasted over the blurred background shows a visible
cut-out edge; this module finishes the matte in three steps, in the
order the design specifies:

1. Guided-filter refinement (He et al.): the alpha edge is pulled
   onto the real image edge from the source frame, so hair strands
   and shoulder lines follow the photograph, not the model's grid.
2. Choke: the matte is eroded by a few pixels (2px standard, 4px on
   high-contrast boundaries) to eat the fringe of background colour
   SAM tends to include just outside the subject.
3. Feather: a small blur softens the final alpha transition so the
   composite has no hard digital edge.

Pure NumPy (box filters via cumulative sums) - no model calls, no
provider spend; this is signal processing on masks SAM produced.
Operates on 2D float arrays; masks come in as bool/float in [0, 1]
and leave as float alpha in [0, 1].

Written fresh in-repo on 2026-10-28.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CHOKE_PX = 2
HIGH_CONTRAST_CHOKE_PX = 4
DEFAULT_FEATHER_R = 2
DEFAULT_GF_RADIUS = 8
DEFAULT_GF_EPS = 1e-3


def _box_blur(x: np.ndarray, r: int) -> np.ndarray:
    """Mean filter with radius r, edge-padded, via cumulative sums."""
    if r <= 0:
        return x.astype(np.float64, copy=True)
    k = 2 * r + 1
    padded = np.pad(x, ((r, r), (r, r)), mode="edge")
    c = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    out = (
        c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    ) / float(k * k)
    return out


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = DEFAULT_GF_RADIUS,
    eps: float = DEFAULT_GF_EPS,
) -> np.ndarray:
    """Guided filter of ``src`` using ``guide`` (both 2D, same shape).

    Output follows the guide's edges: smooth where the guide is
    smooth, sharp where the guide has a real edge.
    """
    g = np.asarray(guide, dtype=np.float64)
    p = np.asarray(src, dtype=np.float64)
    if g.shape != p.shape:
        raise ValueError(f"guide {g.shape} and src {p.shape} shapes differ")
    mean_g = _box_blur(g, radius)
    mean_p = _box_blur(p, radius)
    corr_g = _box_blur(g * g, radius)
    corr_gp = _box_blur(g * p, radius)
    var_g = corr_g - mean_g * mean_g
    cov_gp = corr_gp - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = _box_blur(a, radius)
    mean_b = _box_blur(b, radius)
    return mean_a * g + mean_b


def _erode(alpha: np.ndarray, px: int) -> np.ndarray:
    """Greyscale erosion by ``px`` pixels (3x3 min filter, repeated)."""
    out = alpha
    for _ in range(int(px)):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = np.minimum.reduce(
            [
                p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
                p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:],
                p[2:, :-2], p[2:, 1:-1], p[2:, 2:],
            ]
        )
    return out


def refine_matte(
    mask: np.ndarray,
    guide: np.ndarray,
    *,
    choke_px: int = DEFAULT_CHOKE_PX,
    feather_r: int = DEFAULT_FEATHER_R,
    gf_radius: int = DEFAULT_GF_RADIUS,
    gf_eps: float = DEFAULT_GF_EPS,
) -> np.ndarray:
    """Finish a raw SAM mask against its source frame.

    ``mask``: bool or float mask (H, W). ``guide``: the source frame
    as a float greyscale array (H, W) in [0, 1]. Returns float alpha
    in [0, 1] ready for compositing.
    """
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {m.shape}")
    if m.max() > 1.0:
        m = m / 255.0 if m.max() > 1.5 else m
    m = np.clip(m, 0.0, 1.0)
    alpha = guided_filter(guide, m, radius=gf_radius, eps=gf_eps)
    if choke_px > 0:
        alpha = _erode(alpha, choke_px)
    if feather_r > 0:
        alpha = _box_blur(alpha, feather_r)
        # A second pass approximates a gaussian falloff.
        alpha = _box_blur(alpha, feather_r)
    return np.clip(alpha, 0.0, 1.0)
