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

Refinement pass (2026-10-29, Kamlesh's QA of the gate outputs):
the shipped defaults ghosted hook letters through the cap brim -
the feathered edge carried too much alpha outside the true
silhouette. Measured on a cached Holmes frame (short-1, frame 30;
full record in the BUILDLOG), the retuned defaults below cut the
alpha mass outside the binary mask from 569 to 44 (13x) and halved
the partial-alpha halo, at a core-alpha cost of ~0.01:

- choke 2 -> 3px (eats the bright fringe SAM leaves outside the
  subject, worst at the cap brim);
- feather radius 2 -> 1 (the double box-blur pass at r=2 spread
  the transition over ~9px; r=1 keeps a soft edge over ~5px);
- guided-filter radius 8 -> 4 and eps 1e-3 -> 1e-4 (a smaller,
  stricter filter follows hair-scale edges instead of smoothing
  across them).

This module also owns ``largest_connected_component``: SAM object
rasters can carry detached islands (a background object the model
grouped with the speaker, mask noise), which the finish would
otherwise float in the foreground. Keeping only the largest
4-connected component drops the islands; what is ATTACHED to the
silhouette (the microphone against Holmes's chest) is one
component with the body and no classical filter can separate it -
declared, not hidden.

Pure NumPy (box filters via cumulative sums; component labelling
via scipy.ndimage, a declared dependency) - no model calls, no
provider spend; this is signal processing on masks SAM produced.
Operates on 2D float arrays; masks come in as bool/float in [0, 1]
and leave as float alpha in [0, 1].

Written fresh in-repo on 2026-10-28.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CHOKE_PX = 3
HIGH_CONTRAST_CHOKE_PX = 4
DEFAULT_FEATHER_R = 1
DEFAULT_GF_RADIUS = 4
DEFAULT_GF_EPS = 1e-4


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


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected component of a bool mask.

    Detached islands - background objects SAM grouped into the
    speaker's raster, one-frame mask noise - are dropped; the
    speaker's silhouette is a single connected mass in every clip
    the spike measured. An empty mask returns empty. Labelling is
    scipy.ndimage (classical CV, no model); 4-connectivity so a
    diagonal pixel-touch does not marry an island to the body.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {m.shape}")
    if not m.any():
        return m.copy()
    from scipy import ndimage

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labels, n = ndimage.label(m, structure=structure)
    if n <= 1:
        return m.copy()
    sizes = ndimage.sum_labels(m, labels, index=np.arange(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labels == largest


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
