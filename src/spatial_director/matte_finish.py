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

Fix round 4 adds ``refine_matte_banded``: the same guided finish
confined to the matte's edge band (partial-alpha pixels,
dilated), plus an optional support-erosion overshoot trim -
the global pass could redistribute alpha onto background
structure near the silhouette; the banded pass leaves the
solid interior and the far exterior exactly as decoded.

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


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a bool mask.

    SAM rasters occasionally punch a hole in the silhouette -
    a dark garment region or a specular patch the model reads
    as background for a frame or two. A hole is background-
    labelled area fully enclosed by the silhouette; the
    speaker's body has no such enclosed background, so filling
    is safe by construction and removes a class of one-frame
    interior flicker the edge machinery cannot see (it lives
    away from the edge band). scipy.ndimage classical CV.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {m.shape}")
    if not m.any():
        return m.copy()
    from scipy import ndimage

    return ndimage.binary_fill_holes(m)


def cleanup_speaker_mask(mask: np.ndarray) -> np.ndarray:
    """Full per-frame mask cleanup (fix round 6): largest
    connected component, then hole fill.

    The component pass (round 2) drops detached islands; the
    fill pass closes interior holes. Order matters: filling
    first could bridge an island to the body through a hole
    rim and marry it; component-first cannot resurrect an
    island, because the fill only adds pixels enclosed by the
    surviving component.
    """
    return fill_mask_holes(largest_connected_component(mask))


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


EDGE_BAND_LO = 0.02
EDGE_BAND_HI = 0.98
EDGE_BAND_DILATE_PX = 3


def edge_band(
    alpha: np.ndarray,
    *,
    lo: float = EDGE_BAND_LO,
    hi: float = EDGE_BAND_HI,
    dilate_px: int = EDGE_BAND_DILATE_PX,
) -> np.ndarray:
    """Bool mask of a matte's edge band.

    The band is the partial-alpha region (``lo < alpha < hi``)
    dilated by ``dilate_px``: the only territory a refinement
    may move. Solid interior (alpha >= hi) and clean exterior
    (alpha <= lo) are identity there by definition - a filter
    that runs globally can still shift both (the guided
    filter's local means reach across the edge), which is how
    faint alpha appears on background structure far from the
    silhouette.
    """
    a = np.asarray(alpha, dtype=np.float64)
    partial = (a > lo) & (a < hi)
    if dilate_px > 0 and bool(partial.any()):
        from scipy import ndimage

        return ndimage.binary_dilation(partial,
                                       iterations=int(dilate_px))
    return partial


def _erode_support(alpha: np.ndarray, px: int) -> np.ndarray:
    """The alpha's support (alpha > EDGE_BAND_LO) eroded by px.

    Binary erosion of WHERE any alpha lives - the overshoot
    trim: SAM's raster and the temporal EMA both leave a faint
    skirt of partial alpha just outside the true silhouette
    (the hair smear); removing the support's outermost ring
    removes the skirt's foothold before refinement.
    """
    support = (np.asarray(alpha, dtype=np.float64)
               > EDGE_BAND_LO).astype(np.float64)
    if px > 0:
        support = _erode(support, px)
    return support > 0.5


def refine_matte_banded(
    alpha: np.ndarray,
    guide: np.ndarray,
    *,
    pre_erode_px: int = 0,
    choke_px: int = DEFAULT_CHOKE_PX,
    feather_r: int = DEFAULT_FEATHER_R,
    gf_radius: int = DEFAULT_GF_RADIUS,
    gf_eps: float = DEFAULT_GF_EPS,
) -> np.ndarray:
    """Banded matte finish (fix round 4): refine_matte's steps,
    confined to the edge band.

    Kamlesh's QA (2026-11-02): the cutout edge is not smooth -
    distortions and slight background slivers on either side of
    the silhouette, worst around hair and when the head turns.
    The global guided filter cannot create alpha from nothing,
    but its local statistics reach several px past the edge and
    the choke/feather then redistribute what it made; confining
    the whole finish to the edge band keeps the interior solid
    and the far exterior exactly zero, so any remaining edge
    error is the mask's own, at the edge, where the temporal
    band blend (temporal_smooth) can steady it.

    Steps: optional overshoot trim (support erosion by
    ``pre_erode_px`` - evaluated on the banked fixtures, see
    mask_decode.MATTE_PRE_ERODE_PX for the verdict), guided
    filter against the source frame, choke, feather - then the
    band composite: finished values inside the band, the
    (trimmed) input everywhere else.
    """
    a = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)
    if a.ndim != 2:
        raise ValueError(f"alpha must be 2D, got shape {a.shape}")
    if pre_erode_px > 0:
        a = a * _erode_support(a, pre_erode_px)
    band = edge_band(a)
    if not bool(band.any()):
        return a
    finished = guided_filter(guide, a, radius=gf_radius, eps=gf_eps)
    if choke_px > 0:
        finished = _erode(finished, choke_px)
    if feather_r > 0:
        finished = _box_blur(finished, feather_r)
        finished = _box_blur(finished, feather_r)
    out = np.where(band, finished, a)
    return np.clip(out, 0.0, 1.0)


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
