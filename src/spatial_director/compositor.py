"""The 3-layer spatial composite at 1080x1920.

The Director's signature frame, back to front:

1. **Back** - the canvas backdrop. History: the original
   treatment (the source frame scaled to cover and blur-36'd)
   failed the founder's eye: at that radius the speaker's own
   blurred silhouette sits behind him as a readable ghost person.
   The heavy-blur TREATED backdrop replaced it; fix round 2
   (2026-11-02) made the default ``source_minus_speaker``: the
   source frame with the speaker REMOVED via the SAM mask
   (diffusion fill from the frame's own surroundings), graded
   and slowly drifting - the room stays a real, sharp, legible
   set. The founder's eye rejected that too: the diffusion fill
   reads as a grey ghost/shadow of the speaker standing in the
   room. Since fix round 3 (2026-11-02) the shipped default is
   ``studio_procedural``: a code-rendered studio canvas (brand
   near-black gradient, two soft brand glows, fine grain,
   vignette, Ken Burns drift) with a grounding shadow drawn
   under the placed speaker - no source pixels at all, so no
   ghost can exist. ``source_minus_speaker`` and treated blur
   remain selectable; brand gradient and naive scale-blur stay
   selectable. See ``studio_backdrop`` and
   ``speaker_removed_backdrop``.
2. **Graphics plane** (optional, flag-gated, 2026-10-30) - the
   context graphics layer's timed cards/images, sitting on the
   backdrop and under everything else: they are occluded by the
   hook and the speaker exactly like the backdrop is. See
   ``graphics_layer_at``.
3. **Middle** - the hook text, rendered programmatically (Pillow)
   and positioned BEHIND the speaker. This layer is the whole point:
   the speaker occludes the words, which is the depth effect editors
   build by hand in After Effects.
4. **Front** - the speaker cutout, composited with the finished
   alpha from ``mask_decode`` (SAM -> primary selection -> temporal
   smoothing -> matte refinement).

Captions are deliberately NOT part of this module: they render in
front of everything, later, in the pipeline/layout step. What this
module owns is the spatial stack itself, plus two things built on
it:

- ``render_composite_video`` - the stack animated over a clip, with
  the hook layer present only inside its timing window.
- ``export_xray_mp4`` / ``xray_frame`` - the demo diagnostic: the
  same three layers, slid apart diagonally so a viewer can SEE the
  stack separate (at separation 0 an xray frame is exactly the
  composite; as separation grows the cards fan out over a dark
  ground with labels). This replaces any interactive layer
  inspector - it is a render, not a UI.

Frames are NumPy uint8 RGB arrays; alpha is float in [0, 1]. Video
encoding shells out to FFmpeg (labeled in the README). The canvas
defaults to the Shorts format 1080x1920; tests pass a smaller
canvas through the same code paths.

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS_W = 1080
CANVAS_H = 1920
DEFAULT_CANVAS = (CANVAS_W, CANVAS_H)
DEFAULT_BLUR_RADIUS = 36
BACKDROP_SOURCE_MINUS_SPEAKER = "source_minus_speaker"
BACKDROP_STUDIO = "studio_procedural"
DEFAULT_BACKDROP = BACKDROP_STUDIO
TREATED_BLUR_RADIUS = 90
TREATED_BRIGHTNESS = 0.55
TREATED_SATURATION = 0.30
# Brand backdrop: near-black (#0B0B0C) tinted 12% toward Forge Blue
# at the top and 12% toward Viral Pink at the bottom - a gradient a
# viewer feels rather than sees.
BRAND_TOP = (15, 25, 40)
BRAND_BOTTOM = (40, 17, 26)
XRAY_GROUND = (16, 18, 22)  # dark slate the separated cards fan over

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------- layers


def background_layer(
    frame_rgb: np.ndarray,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
) -> np.ndarray:
    """Source frame scaled to COVER the canvas, then blurred."""
    cw, ch = canvas
    img = Image.fromarray(frame_rgb)
    sw, sh = img.size
    scale = max(cw / sw, ch / sh)
    resized = img.resize(
        (max(cw, round(sw * scale)), max(ch, round(sh * scale))),
        Image.BILINEAR,
    )
    left = (resized.size[0] - cw) // 2
    top = (resized.size[1] - ch) // 2
    cropped = resized.crop((left, top, left + cw, top + ch))
    if blur_radius > 0:
        from PIL import ImageFilter

        cropped = cropped.filter(ImageFilter.GaussianBlur(blur_radius))
    return np.asarray(cropped).astype(np.uint8)


def treated_background_layer(
    frame_rgb: np.ndarray,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    blur_radius: int = TREATED_BLUR_RADIUS,
    brightness: float = TREATED_BRIGHTNESS,
    saturation: float = TREATED_SATURATION,
) -> np.ndarray:
    """The shipped backdrop: cover-scale, heavy blur, dark, desat.

    The naive blur (radius 36) leaves the speaker's silhouette
    readable in the backdrop - a ghost person standing behind the
    real one. Radius 90 destroys the silhouette's edges, and the
    brightness/saturation grade pushes what remains toward a dark
    neutral wash, so the backdrop reads as atmosphere, not as a
    second picture of the subject.
    """
    from PIL import ImageEnhance

    img = Image.fromarray(
        background_layer(frame_rgb, canvas, blur_radius=blur_radius))
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return np.asarray(img).astype(np.uint8)


def brand_background_layer(
    canvas: tuple[int, int] = DEFAULT_CANVAS,
) -> np.ndarray:
    """A clean near-black brand gradient (no source pixels at all).

    Vertical blend from BRAND_TOP to BRAND_BOTTOM - Forge Blue and
    Viral Pink at 12% over near-black. Nothing of the scene (and
    therefore no ghost of the speaker) can appear in it; the cost
    is losing the room's context entirely.
    """
    cw, ch = canvas
    top = np.array(BRAND_TOP, dtype=np.float64)
    bottom = np.array(BRAND_BOTTOM, dtype=np.float64)
    t = np.linspace(0.0, 1.0, ch, dtype=np.float64)[:, None, None]
    grad = top[None, None, :] * (1.0 - t) + bottom[None, None, :] * t
    return np.repeat(grad, cw, axis=1).astype(np.uint8)


def make_backdrop(
    frame_rgb: np.ndarray,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    backdrop: str = DEFAULT_BACKDROP,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
) -> np.ndarray:
    """Dispatch on the backdrop treatment name.

    ``treated`` = heavy blur + grade of the source (the historic
    default; now the maskless-frame fallback); ``brand`` =
    near-black brand gradient (frame ignored); ``naive`` = the
    original scale + blur (kept for comparison - it is the
    treatment that ghosted). The modes that need more than the
    frame - ``source_minus_speaker`` (the frame's speaker alpha)
    and the current default ``studio_procedural`` (a seed, the
    drift progress and the placed speaker alpha for its
    grounding shadow) - are served by composite_frame /
    _canvas_backdrop; asking make_backdrop for them raises
    rather than silently substituting another treatment.
    """
    if backdrop == "treated":
        return treated_background_layer(frame_rgb, canvas)
    if backdrop == "brand":
        return brand_background_layer(canvas)
    if backdrop == "naive":
        return background_layer(frame_rgb, canvas, blur_radius=blur_radius)
    raise ValueError(f"unknown backdrop treatment: {backdrop!r}")


def _paste_placed(
    fg_canvas: np.ndarray,
    a_canvas: np.ndarray,
    fg: Image.Image,
    a: Image.Image,
    ox: int,
    oy: int,
    alpha_scale: float = 1.0,
) -> None:
    """Paste a resized (fg, alpha) pair at (ox, oy), clipped to the
    canvas. Shared by the legacy fit placement and explicit layout
    placements so both paths round and clip identically."""
    cw, ch = a_canvas.shape[1], a_canvas.shape[0]
    nw, nh = fg.size
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    dx1, dy1 = min(cw, ox + nw), min(ch, oy + nh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    fg_np = np.asarray(fg)
    a_np = np.asarray(a).astype(np.float64) / 255.0 * alpha_scale
    fg_canvas[dy0:dy1, dx0:dx1] = fg_np[sy0:sy0 + (dy1 - dy0),
                                       sx0:sx0 + (dx1 - dx0)]
    a_canvas[dy0:dy1, dx0:dx1] = a_np[sy0:sy0 + (dy1 - dy0),
                                      sx0:sx0 + (dx1 - dx0)]


# Canvas backdrop treatment (fix round 2, Fix B revised,
# 2026-11-02): the DEFAULT canvas backdrop is the window's own
# source frame with the speaker REMOVED - the SAM mask region is
# filled by multi-scale diffusion from the frame's own
# surroundings (plain signal processing, no model), so no
# silhouette or person edge survives, while everything outside
# the mask stays SHARP: the room/set remains a real, legible
# backdrop behind the enlarged cutout. The filled frame is graded
# (brightness x0.6, saturation x0.55) and drifts with the same Ken
# Burns window math as the graphics backdrop spans (zoom
# 1.00 -> 1.12, slow pan). The historic whole-frame blur
# ("treated") remains only as the per-frame fallback for frames
# carrying no speaker mask, and the receipt declares the mode
# honestly ("source_minus_speaker" vs "blur").
BACKDROP_GRADE_BRIGHTNESS = 0.6
BACKDROP_GRADE_SATURATION = 0.55
BACKDROP_FILL = "diffusion"
BACKDROP_KEN_BURNS_MAX_ZOOM = 1.12
_BACKDROP_FILL_DOWNSCALE = 6
_BACKDROP_FILL_ITERATIONS = 300


def _diffusion_fill(frame_rgb: np.ndarray,
                    known: np.ndarray) -> np.ndarray:
    """Fill the unknown region by smooth diffusion of the known.

    Solved at 1/6 resolution (iterated 4-neighbour averaging with
    known pixels pinned) and upscaled: the fill is a smooth
    interpolation of the surroundings BY CONSTRUCTION - it cannot
    contain a person edge or any high-frequency silhouette.
    """
    h, w = known.shape
    sh = max(8, h // _BACKDROP_FILL_DOWNSCALE)
    sw = max(8, w // _BACKDROP_FILL_DOWNSCALE)
    img = Image.fromarray(frame_rgb).resize((sw, sh), Image.BILINEAR)
    small = np.asarray(img).astype(np.float64)
    kimg = Image.fromarray(
        (known * 255).astype(np.uint8)).resize((sw, sh), Image.BILINEAR)
    k = (np.asarray(kimg).astype(np.float64) / 255.0) >= 0.999
    if not k.any():
        fill_mean = small.reshape(-1, 3).mean(axis=0)
        filled = np.tile(fill_mean, (sh, sw, 1))
    else:
        filled = small.copy()
        mean = small[k].mean(axis=0)
        filled[~k] = mean
        for _ in range(_BACKDROP_FILL_ITERATIONS):
            acc = filled.copy()
            acc[1:, :] += filled[:-1, :]
            acc[:-1, :] += filled[1:, :]
            acc[:, 1:] += filled[:, :-1]
            acc[:, :-1] += filled[:, 1:]
            # border handling: divide by the true neighbour count
            cnt = np.full((sh, sw), 4.0)
            cnt[0, :] -= 1
            cnt[-1, :] -= 1
            cnt[:, 0] -= 1
            cnt[:, -1] -= 1
            avg = acc / (cnt[..., None] + 1.0)  # +1: acc seeds itself
            filled = np.where(k[..., None], small, avg)
    up = Image.fromarray(
        np.clip(filled, 0, 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR)
    return np.asarray(up).astype(np.float64)


def speaker_removed_backdrop(
    frame_rgb: np.ndarray,
    alpha: np.ndarray,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    progress: float = 0.0,
) -> np.ndarray:
    """The source frame minus the speaker, graded, drifting.

    ``alpha`` is the frame's finished SAM alpha (source space).
    The mask (> 0.5, dilated past the matte edge) is diffusion-
    filled from the frame's own surroundings; outside the mask
    the frame stays sharp. Then grade + Ken Burns cover window.
    Returns a float64 canvas-sized RGB array like make_backdrop.
    """
    from PIL import ImageEnhance, ImageFilter

    h, w = alpha.shape
    mask = alpha > 0.5
    src = frame_rgb.astype(np.float64)
    if mask.any():
        dilate = max(2, int(round(0.008 * max(w, h))))
        m_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(
            ImageFilter.MaxFilter(2 * dilate + 1))
        m = np.asarray(m_img).astype(np.float64) / 255.0 > 0.0
        fill = _diffusion_fill(frame_rgb, ~m)
        feather = np.asarray(
            Image.fromarray((m * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(2))).astype(np.float64) / 255.0
        out = src * (1.0 - feather[..., None]) \
            + fill * feather[..., None]
    else:
        out = src
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(BACKDROP_GRADE_BRIGHTNESS)
    img = ImageEnhance.Color(img).enhance(BACKDROP_GRADE_SATURATION)
    # Cover-fit with drift slack, then the Ken Burns window (the
    # same window math as graphics_layer.prepare_backdrop).
    cw, ch = canvas
    slack = BACKDROP_KEN_BURNS_MAX_ZOOM
    scale = max(cw * slack / img.width, ch * slack / img.height)
    base = img.resize(
        (max(cw, round(img.width * scale)),
         max(ch, round(img.height * scale))), Image.BILINEAR)
    zoom = 1.0 + (BACKDROP_KEN_BURNS_MAX_ZOOM - 1.0) * progress
    zw = int(base.width / zoom)
    zh = int(base.height / zoom)
    pan_x = 0.20 + (0.65 - 0.20) * progress
    pan_y = 0.20 + (0.60 - 0.20) * progress
    x0 = int((base.width - zw) * pan_x)
    y0 = int((base.height - zh) * pan_y)
    win = base.crop((x0, y0, x0 + zw, y0 + zh)).resize(
        (cw, ch), Image.BILINEAR)
    return np.asarray(win).astype(np.float64)


# Studio procedural backdrop (fix round 3, 2026-11-02): the
# founder rejected the minus-speaker diffusion fill on sight -
# the filled region reads as a grey ghost/shadow of the speaker
# standing in the original room. His direction: solve it, and
# if the source background cannot be made clean, REPLACE the
# background entirely. The new default therefore uses NO source
# pixels at all: a code-rendered studio canvas - vertical
# gradient in the brand near-blacks, two soft radial glows
# (Forge Blue upper-left, Viral Pink lower-right, low alpha so
# it reads as a lit studio, not wallpaper), fine deterministic
# grain, a gentle vignette - drifting with the same Ken Burns
# window math as the other canvas backdrops. A soft elliptical
# grounding shadow is drawn on the backdrop plane under the
# PLACED speaker (derived per frame from the placed mask's
# bottom rows), so the cutout sits IN the scene instead of
# floating over it. The seed is derived deterministically from
# the run/window identity (studio_seed) and declared on the
# receipt; same seed -> byte-identical canvas. Zero model
# calls, plain numpy/PIL.
STUDIO_TOP = (16, 16, 20)        # #101014
STUDIO_BOTTOM = (11, 11, 12)     # #0B0B0C - the brand near-black
STUDIO_GLOW_BLUE = (46, 124, 246)    # Forge Blue #2E7CF6
STUDIO_GLOW_PINK = (255, 61, 126)    # Viral Pink #FF3D7E
STUDIO_GLOW_BLUE_PEAK = 0.17
STUDIO_GLOW_PINK_PEAK = 0.13
STUDIO_GRAIN_SIGMA = 2.0
STUDIO_VIGNETTE_STRENGTH = 0.26
STUDIO_SHADOW_STRENGTH = 0.42
STUDIO_DEFAULT_SEED = 0
_STUDIO_BASE_CACHE: dict[tuple[tuple[int, int], int], np.ndarray] = {}


def studio_seed(identity: str) -> int:
    """The deterministic backdrop seed for a run/window identity.

    SHA-256 of the identity string, first 8 bytes as an int -
    stable across processes and machines, so the receipt's
    declared seed fully determines the rendered canvas.
    """
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _studio_base(canvas: tuple[int, int], seed: int) -> np.ndarray:
    """The oversize studio canvas (Ken Burns slack included).

    Built once per (canvas, seed) and cached: gradient + glows
    + vignette + seeded grain at BACKDROP_KEN_BURNS_MAX_ZOOM
    scale, so the drift window can pan/zoom inside it exactly
    like the source-derived backdrops.
    """
    key = (canvas, seed)
    base = _STUDIO_BASE_CACHE.get(key)
    if base is not None:
        return base
    cw, ch = canvas
    slack = BACKDROP_KEN_BURNS_MAX_ZOOM
    bw = max(cw, round(cw * slack))
    bh = max(ch, round(ch * slack))
    top = np.array(STUDIO_TOP, dtype=np.float64)
    bottom = np.array(STUDIO_BOTTOM, dtype=np.float64)
    t = np.linspace(0.0, 1.0, bh, dtype=np.float64)[:, None, None]
    img = np.repeat(top[None, None, :] * (1.0 - t)
                    + bottom[None, None, :] * t, bw, axis=1)
    xs = np.linspace(0.0, 1.0, bw, dtype=np.float64)[None, :]
    ys = np.linspace(0.0, 1.0, bh, dtype=np.float64)[:, None]

    def _glow(cx: float, cy: float, radius: float, peak: float,
              colour: tuple[int, int, int]) -> None:
        d2 = ((xs - cx) * (bw / bh)) ** 2 + (ys - cy) ** 2
        fall = np.exp(-d2 / (2.0 * radius * radius))
        img[...] += fall[..., None] * peak \
            * np.array(colour, dtype=np.float64)[None, None, :]

    _glow(0.20, 0.13, 0.42, STUDIO_GLOW_BLUE_PEAK, STUDIO_GLOW_BLUE)
    _glow(0.84, 0.88, 0.38, STUDIO_GLOW_PINK_PEAK, STUDIO_GLOW_PINK)
    nx = np.linspace(-1.0, 1.0, bw, dtype=np.float64)[None, :]
    ny = np.linspace(-1.0, 1.0, bh, dtype=np.float64)[:, None]
    r2 = np.clip((nx ** 2 + ny ** 2) / 2.0, 0.0, 1.0)
    img *= (1.0 - STUDIO_VIGNETTE_STRENGTH * r2 ** 1.2)[..., None]
    rng = np.random.default_rng(seed)
    img += rng.normal(0.0, STUDIO_GRAIN_SIGMA, (bh, bw, 1))
    base = np.clip(img, 0, 255)
    if len(_STUDIO_BASE_CACHE) > 4:
        _STUDIO_BASE_CACHE.clear()
    _STUDIO_BASE_CACHE[key] = base
    return base


def _grounding_shadow(bg: np.ndarray,
                      placed_alpha: np.ndarray) -> np.ndarray:
    """Darken a soft ellipse under the placed speaker's base.

    Position and width come from the placed mask itself: the
    ellipse centres on the horizontal extent of the mask's
    bottom band, tucked slightly under the mask's last row, so
    the visible rim reads as the shadow the cutout casts on the
    studio floor. Backdrop plane only - the speaker layer is
    composited over it unchanged. No mask -> no shadow.
    """
    ch, cw = bg.shape[:2]
    mask = placed_alpha > 0.5
    if not mask.any():
        return bg
    rows = np.flatnonzero(mask.any(axis=1))
    base_y = int(rows[-1])
    band_top = max(0, base_y - max(8, int(0.10 * ch)))
    cols = np.flatnonzero(mask[band_top:base_y + 1].any(axis=0))
    if cols.size == 0:
        return bg
    x0, x1 = int(cols[0]), int(cols[-1])
    base_w = x1 - x0 + 1
    if base_w < 8:
        return bg
    cx = (x0 + x1) / 2.0
    rx = max(60.0, 0.62 * base_w)
    ry = max(26.0, 0.028 * ch)
    cy = base_y - 0.35 * ry
    xa = max(0, int(cx - 2.2 * rx))
    xb = min(cw, int(cx + 2.2 * rx) + 1)
    ya = max(0, int(cy - 2.6 * ry))
    yb = min(ch, int(cy + 2.6 * ry) + 1)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    fall = np.exp(-2.2 * d2)
    out = bg.copy()
    out[ya:yb, xa:xb] *= (1.0 - STUDIO_SHADOW_STRENGTH * fall)[..., None]
    return out


def studio_backdrop(canvas: tuple[int, int] = DEFAULT_CANVAS,
                    *,
                    progress: float = 0.0,
                    seed: int | None = None,
                    shadow_alpha: np.ndarray | None = None,
                    ) -> np.ndarray:
    """The procedural studio canvas at drift ``progress``.

    Returns a float64 canvas-sized RGB array like the other
    canvas backdrops. ``seed`` None uses STUDIO_DEFAULT_SEED.
    ``shadow_alpha`` is the PLACED (canvas-space) speaker alpha
    for this frame; when given, the grounding shadow is drawn
    under the speaker's base. The frame's own pixels are never
    consulted - that is the point of the mode.
    """
    if seed is None:
        seed = STUDIO_DEFAULT_SEED
    base = _studio_base(canvas, seed)
    cw, ch = canvas
    zoom = 1.0 + (BACKDROP_KEN_BURNS_MAX_ZOOM - 1.0) * progress
    zw = int(base.shape[1] / zoom)
    zh = int(base.shape[0] / zoom)
    pan_x = 0.20 + (0.65 - 0.20) * progress
    pan_y = 0.20 + (0.60 - 0.20) * progress
    x0 = int((base.shape[1] - zw) * pan_x)
    y0 = int((base.shape[0] - zh) * pan_y)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    win = img.crop((x0, y0, x0 + zw, y0 + zh)).resize(
        (cw, ch), Image.BILINEAR)
    out = np.asarray(win).astype(np.float64)
    if shadow_alpha is not None:
        out = _grounding_shadow(out, shadow_alpha)
    return out


def _canvas_backdrop(frame_rgb: np.ndarray, alpha: np.ndarray,
                     canvas: tuple[int, int], backdrop: str,
                     progress: float,
                     blur_radius: int = DEFAULT_BLUR_RADIUS,
                     seed: int | None = None,
                     placed_alpha: np.ndarray | None = None,
                     ) -> np.ndarray:
    """Dispatch the canvas backdrop for one frame.

    The studio default ignores the frame entirely (its shadow
    comes from the PLACED alpha when the caller supplies it).
    The minus-speaker mode needs the frame's alpha; a frame
    with no speaker mask falls back to the treated blur for
    that frame (recorded by the pipeline as a maskless frame).
    """
    if backdrop == BACKDROP_STUDIO:
        return studio_backdrop(canvas, progress=progress, seed=seed,
                               shadow_alpha=placed_alpha)
    if backdrop == BACKDROP_SOURCE_MINUS_SPEAKER:
        if alpha is not None and bool((alpha > 0.5).any()):
            return speaker_removed_backdrop(
                frame_rgb, alpha, canvas, progress=progress)
        return treated_background_layer(frame_rgb, canvas)
    return make_backdrop(frame_rgb, canvas, backdrop=backdrop,
                         blur_radius=blur_radius)


def speaker_layer(
    frame_rgb: np.ndarray,
    alpha: np.ndarray,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    placement: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the speaker to fit the canvas width, bottom-anchored.

    The scale is ``min(cw/sw, ch/sh)`` - fit the width, unless the
    source is so tall that the height would overflow, in which case
    fit the height instead. The speaker is centred horizontally and
    anchored to the bottom edge (talking-head framing). Returns the
    full-canvas (fg_rgb uint8, alpha float) pair; alpha is 0 wherever
    the speaker does not reach.

    ``placement`` (layout-states extension, 2026-11-01) is an
    explicit placement object (see layout_states.SpeakerPlacement:
    optional crop rect, scale, canvas origin, alpha scale). When
    given, the (cropped) frame is scaled by ``placement.scale`` and
    pasted at ``(placement.ox, placement.oy)`` instead of the fit
    placement - this is how Spotlight's PiP, Split's bottom band
    and Cutaway's absence render through the same layer function
    the verifier's geometry path (layout_law.to_canvas_alpha)
    uses, so measured geometry is rendered geometry. None keeps
    the historic fit placement byte-for-byte.
    """
    cw, ch = canvas
    if placement is not None:
        crop = getattr(placement, "crop", None)
        if crop is not None:
            cx0 = int(round(crop[0]))
            cy0 = int(round(crop[1]))
            cwd = int(round(crop[2]))
            cht = int(round(crop[3]))
            frame_rgb = frame_rgb[cy0:cy0 + cht, cx0:cx0 + cwd]
            alpha = alpha[cy0:cy0 + cht, cx0:cx0 + cwd]
        scale = float(placement.scale)
        nw = max(1, round(alpha.shape[1] * scale))
        nh = max(1, round(alpha.shape[0] * scale))
        fg = Image.fromarray(frame_rgb).resize((nw, nh),
                                               Image.BILINEAR)
        a = Image.fromarray(
            (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        ).resize((nw, nh), Image.BILINEAR)
        fg_canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        a_canvas = np.zeros((ch, cw), dtype=np.float64)
        _paste_placed(fg_canvas, a_canvas, fg, a,
                      int(round(placement.ox)),
                      int(round(placement.oy)),
                      float(getattr(placement, "alpha_scale", 1.0)))
        return fg_canvas, a_canvas
    sh, sw = alpha.shape
    scale = min(cw / sw, ch / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    fg = Image.fromarray(frame_rgb).resize((nw, nh), Image.BILINEAR)
    a = Image.fromarray(
        (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    ).resize((nw, nh), Image.BILINEAR)
    fg_canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    a_canvas = np.zeros((ch, cw), dtype=np.float64)
    x0 = (cw - nw) // 2
    y0 = ch - nh
    fg_canvas[y0 : y0 + nh, x0 : x0 + nw] = np.asarray(fg)
    a_canvas[y0 : y0 + nh, x0 : x0 + nw] = (
        np.asarray(a).astype(np.float64) / 255.0
    )
    return fg_canvas, a_canvas


def hook_layer(
    text: str,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    y_center_frac: float = 0.30,
    max_width_frac: float = 0.92,
    start_size: int = 170,
    fill: tuple[int, int, int, int] = (255, 255, 255, 255),
    accent_index: int | None = None,
    accent_fill: tuple[int, int, int, int] = (255, 61, 126, 255),
    only_word_index: int | None = None,
) -> np.ndarray:
    """Render hook text as a transparent RGBA layer at canvas size.

    Words wrap to at most two lines; the font shrinks from
    ``start_size`` until the block fits the width budget. Text is
    centred horizontally and centred vertically on
    ``y_center_frac`` of the canvas height - in the composite it
    sits behind the speaker's head/shoulders, visible at the sides,
    which is the intended occlusion effect. A dark stroke keeps the
    letterforms legible over the backdrop.

    Fix round 2 (Fix E): two-tone type - ``accent_index`` paints
    ONE word (the hook's strongest content word, selected in
    layout_law) in Viral Pink; everything else stays white with
    the dark stroke. ``only_word_index`` renders a single word and
    leaves the rest transparent: the per-word layers the
    stagger-pop entry animation composes. With neither set the
    historic single-draw path runs byte-for-byte.
    """
    cw, ch = canvas
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    max_w = int(cw * max_width_frac)
    words = text.split()

    def wrap(font: ImageFont.FreeTypeFont) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in words:
            trial = (current + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_w or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    size = start_size
    lines = wrap(_load_font(size))
    while size > 24 and len(lines) > 2:
        size = int(size * 0.9)
        lines = wrap(_load_font(size))
    font = _load_font(size)
    line_h = int(size * 1.12)
    block_h = line_h * len(lines)
    y = int(ch * y_center_frac) - block_h // 2
    styled = accent_index is not None or only_word_index is not None

    def fill_for(idx: int) -> tuple[int, int, int, int]:
        if only_word_index is not None and idx != only_word_index:
            return (0, 0, 0, 0)
        if accent_index is not None and idx == accent_index:
            return accent_fill
        return fill

    word_index = 0
    for line in lines:
        w = draw.textlength(line, font=font)
        if not styled:
            draw.text(
                ((cw - w) / 2, y),
                line,
                font=font,
                fill=fill,
                stroke_width=max(2, size // 16),
                stroke_fill=(10, 10, 12, 255),
            )
            word_index += len(line.split())
        else:
            x = (cw - w) / 2
            space_w = draw.textlength(" ", font=font)
            for wtok in line.split():
                f = fill_for(word_index)
                # A fully transparent word (a non-selected word
                # of a per-word entry layer) is not drawn at all:
                # PIL would still paint its opaque stroke,
                # ghosting the whole text into every word layer.
                if f[3] > 0:
                    draw.text(
                        (x, y),
                        wtok,
                        font=font,
                        fill=f,
                        stroke_width=max(2, size // 16),
                        stroke_fill=(10, 10, 12, 255),
                    )
                x += draw.textlength(wtok, font=font) + space_w
                word_index += 1
        y += line_h
    return np.asarray(img)


# Hook entry animation (fix round 2, Fix E): word-stagger pop-in.
# Word 0 is at full opacity from the window's first frame (F1's
# frame-0 leg measures the declared layer at full opacity - the
# resting layer - and the render honours it for word 0 too); the
# remaining words pop in across ~0.36s, each fading over 0.05s and
# settling from 1.16x to 1.0x over 0.22s. After HOOK_ENTRY_S the
# renderer uses the resting layer directly.
HOOK_ENTRY_S = 0.65
_HOOK_ENTRY_STAGGER_SPAN_S = 0.36
_HOOK_ENTRY_FADE_S = 0.05
_HOOK_ENTRY_SETTLE_S = 0.22
_HOOK_ENTRY_POP_SCALE = 1.16


def hook_entry_layer(word_layers: Sequence[np.ndarray],
                     t_rel: float) -> np.ndarray:
    """The hook layer ``t_rel`` seconds into the stagger entry.

    ``word_layers`` are the per-word canvas layers (word i's ink
    only, at its resting position). Deterministic in t_rel.
    """
    canvas_shape = word_layers[0].shape
    acc_rgb = np.zeros(canvas_shape[:2] + (3,), dtype=np.float64)
    acc_a = np.zeros(canvas_shape[:2] + (1,), dtype=np.float64)
    n = len(word_layers)
    step = (_HOOK_ENTRY_STAGGER_SPAN_S / (n - 1)) if n > 1 else 0.0
    step = min(step, 0.09)
    for i, layer in enumerate(word_layers):
        if i == 0:
            piece = layer
        else:
            local = t_rel - i * step
            if local <= 0.0:
                continue
            alpha_f = min(1.0, local / _HOOK_ENTRY_FADE_S)
            p = min(1.0, local / _HOOK_ENTRY_SETTLE_S)
            scale = 1.0 + (_HOOK_ENTRY_POP_SCALE - 1.0) * (1.0 - p) ** 2
            if alpha_f >= 1.0 and abs(scale - 1.0) < 1e-3:
                piece = layer
            else:
                ys, xs = np.nonzero(layer[..., 3] > 0)
                if ys.size == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                crop = Image.fromarray(layer[y0:y1, x0:x1])
                if abs(scale - 1.0) >= 1e-3:
                    crop = crop.resize(
                        (max(1, round(crop.width * scale)),
                         max(1, round(crop.height * scale))),
                        Image.BILINEAR)
                if alpha_f < 1.0:
                    arr = np.asarray(crop).astype(np.float64)
                    arr[..., 3] *= alpha_f
                    crop = Image.fromarray(
                        np.clip(arr, 0, 255).astype(np.uint8))
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                nx = int(round(cx - crop.width / 2.0))
                ny = int(round(cy - crop.height / 2.0))
                piece = np.zeros(canvas_shape, dtype=np.uint8)
                sx0, sy0 = max(0, -nx), max(0, -ny)
                dx0, dy0 = max(0, nx), max(0, ny)
                dx1 = min(canvas_shape[1], nx + crop.width)
                dy1 = min(canvas_shape[0], ny + crop.height)
                if dx1 > dx0 and dy1 > dy0:
                    piece[dy0:dy1, dx0:dx1] = np.asarray(crop)[
                        sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        la = piece[..., 3:4].astype(np.float64) / 255.0
        acc_rgb = piece[..., :3].astype(np.float64) * la \
            + acc_rgb * (1.0 - la)
        acc_a = la + acc_a * (1.0 - la)
    out = np.concatenate([acc_rgb, acc_a * 255.0], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- stack


def composite_from_layers(
    bg_rgb: np.ndarray,
    hook_rgba: np.ndarray | None,
    fg_rgb: np.ndarray,
    alpha: np.ndarray,
    graphics_rgba: np.ndarray | None = None,
) -> np.ndarray:
    """Blend the canvas-sized layers: back, graphics, hook, speaker.

    ``graphics_rgba`` (the context graphics plane) is optional and
    defaults to None - with it absent the blend is byte-for-byte
    the original three-layer stack.
    """
    out = bg_rgb.astype(np.float64)
    if graphics_rgba is not None:
        ga = graphics_rgba[..., 3:4].astype(np.float64) / 255.0
        out = graphics_rgba[..., :3].astype(np.float64) * ga \
            + out * (1.0 - ga)
    if hook_rgba is not None:
        ha = hook_rgba[..., 3:4].astype(np.float64) / 255.0
        out = hook_rgba[..., :3].astype(np.float64) * ha + out * (1.0 - ha)
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    out = fg_rgb.astype(np.float64) * a + out * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def composite_frame(
    frame_rgb: np.ndarray,
    alpha: np.ndarray,
    hook_rgba: np.ndarray | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    backdrop: str = DEFAULT_BACKDROP,
    graphics_rgba: np.ndarray | None = None,
    backdrop_rgb: np.ndarray | None = None,
    speaker_placement: object | None = None,
    backdrop_progress: float = 0.0,
    backdrop_seed: int | None = None,
) -> np.ndarray:
    """One composited frame from a source frame + finished alpha.

    ``backdrop_rgb`` (block 2 backdrop mode) replaces the backdrop
    layer wholesale for this frame - a generated scene treated in
    graphics_layer.prepare_backdrop. It sits exactly where the
    treated source backdrop sits: behind graphics, hook and
    speaker. None = the canvas backdrop treatment named by
    ``backdrop`` (the default is the procedural studio canvas,
    fix round 3; ``backdrop_progress`` drives its drift and
    ``backdrop_seed`` its deterministic grain). The speaker layer
    is built first because the studio backdrop's grounding
    shadow is derived from the PLACED speaker alpha.
    ``speaker_placement`` (layout states) forwards an explicit
    placement to ``speaker_layer``; None is the historic fit.
    """
    fg, a = speaker_layer(frame_rgb, alpha, canvas,
                          placement=speaker_placement)
    if backdrop_rgb is not None:
        bg = backdrop_rgb
    elif backdrop == BACKDROP_STUDIO:
        bg = studio_backdrop(canvas, progress=backdrop_progress,
                             seed=backdrop_seed, shadow_alpha=a)
    elif backdrop == BACKDROP_SOURCE_MINUS_SPEAKER:
        bg = _canvas_backdrop(frame_rgb, alpha, canvas, backdrop,
                              backdrop_progress)
    else:
        bg = make_backdrop(frame_rgb, canvas, backdrop=backdrop,
                           blur_radius=blur_radius)
    return composite_from_layers(bg, hook_rgba, fg, a,
                                 graphics_rgba=graphics_rgba)


def hook_visible_at(t_s: float, window: tuple[float, float] | None) -> bool:
    """The hook layer exists only inside its [start, end) window."""
    if window is None:
        return False
    return window[0] <= t_s < window[1]


def shift_layer_x(layer: np.ndarray, dx_px: float) -> np.ndarray:
    """Shift a canvas-sized layer horizontally by ``dx_px`` pixels.

    Zero-filled at the vacated edge, never wrapped: a slide-in
    card must enter from off-canvas geometry, not teleport glyphs
    from the far edge. Works for RGBA uint8 layers and bool masks
    alike (the shift is pure slicing).
    """
    dx = int(round(dx_px))
    if dx == 0:
        return layer
    out = np.zeros_like(layer)
    h, w = layer.shape[:2]
    if dx > 0:
        out[:, dx:] = layer[:, :w - dx]
    else:
        out[:, :w + dx] = layer[:, -dx:]
    return out


def hook_slide_dx_at(t_s: float, window: tuple[float, float] | None,
                     slide: tuple[float, float] | None) -> float:
    """Horizontal hook offset at ``t_s`` for a slide repair.

    ``slide`` is (dx_start_px, duration_s): the card starts
    displaced by dx_start at the window start and eases linearly
    to its final position over ``duration_s``. Outside the slide
    the offset is 0, so a finished slide is the static layer.
    """
    if slide is None or window is None:
        return 0.0
    dx_start, duration = slide
    if duration <= 0:
        return 0.0
    progress = (t_s - window[0]) / duration
    if progress <= 0.0:
        return dx_start
    if progress >= 1.0:
        return 0.0
    return dx_start * (1.0 - progress)


# ---------------------------------------------------------------- graphics

GRAPHIC_FADE_S = 0.2  # fade in/out ceiling for context graphics

# One graphics entry: (layer provider, start_s, end_s). The provider
# is either a canvas-sized RGBA array (a static graphic) or a
# callable t_s -> RGBA (an animated graphic - graphics_layer's
# GraphicSpec.layer_at). The compositor treats providers as opaque:
# it owns WHEN a graphic is on the backdrop plane (window + fade),
# the graphics layer owns WHAT it looks like at that instant.
GraphicEntry = tuple[object, float, float]


def graphic_alpha_scale(t_s: float, start_s: float, end_s: float,
                        fade_s: float = GRAPHIC_FADE_S) -> float:
    """A graphic's opacity envelope: 0 outside [start, end), ramping
    linearly over ``fade_s`` at both edges, 1 in between."""
    if t_s < start_s or t_s >= end_s:
        return 0.0
    if fade_s <= 0:
        return 1.0
    scale = 1.0
    if t_s < start_s + fade_s:
        scale = min(scale, (t_s - start_s) / fade_s)
    if t_s > end_s - fade_s:
        scale = min(scale, (end_s - t_s) / fade_s)
    return min(1.0, max(0.0, scale))


def graphics_layer_at(entries: Sequence[GraphicEntry] | None,
                      t_s: float,
                      canvas: tuple[int, int] = DEFAULT_CANVAS,
                      ) -> np.ndarray | None:
    """The combined graphics plane at time ``t_s``, or None.

    Entries are alpha-composited in order over a transparent canvas
    with the fade envelope applied. Graphics are placed so they
    never overlap (graphics_layer's placement law), so the order
    only matters for the fade edges. None when no entry is active -
    callers then composite exactly the pre-graphics stack.
    """
    if not entries:
        return None
    acc: np.ndarray | None = None
    for provider, start_s, end_s in entries:
        scale = graphic_alpha_scale(t_s, float(start_s), float(end_s))
        if scale <= 0.0:
            continue
        layer = provider(t_s) if callable(provider) else provider
        rgba = np.asarray(layer, dtype=np.float64)
        if scale < 1.0:
            rgba = rgba.copy()
            rgba[..., 3] *= scale
        if acc is None:
            acc = rgba
            continue
        a = rgba[..., 3:4] / 255.0
        acc_a = acc[..., 3:4] / 255.0
        out_a = a + acc_a * (1.0 - a)
        safe = np.maximum(out_a, 1e-9)
        rgb = (rgba[..., :3] * a + acc[..., :3] * acc_a * (1.0 - a)) / safe
        acc = np.concatenate(
            [np.where(out_a > 0, rgb, 0.0), out_a * 255.0], axis=2)
    if acc is None:
        return None
    return np.clip(acc, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- xray


def xray_frame(
    frame_rgb: np.ndarray,
    alpha: np.ndarray,
    hook_rgba: np.ndarray | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    separation: float = 1.0,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    backdrop: str = DEFAULT_BACKDROP,
) -> np.ndarray:
    """The layer stack, fanned apart: back -d, hook 0, speaker +d.

    At ``separation`` 0 this is exactly ``composite_frame``. As it
    grows, the three canvas-sized cards slide diagonally over a dark
    ground with outline + label each, so the demo can show the stack
    separating. Speaker/hook cards are drawn with their real alpha -
    what separates is the genuine layers, not a redraw.
    """
    cw, ch = canvas
    fg, a = speaker_layer(frame_rgb, alpha, canvas)
    bg_f = _canvas_backdrop(frame_rgb, alpha, canvas, backdrop, 0.0,
                            blur_radius=blur_radius, placed_alpha=a)
    if separation <= 0.0:
        # Byte-identical to composite_frame: the float backdrop
        # goes in unclipped (the uint8 clip below is only for
        # the fanned card path, and it costs ±1 against the
        # composite's float blend).
        return composite_from_layers(bg_f, hook_rgba, fg, a)
    bg = np.clip(bg_f, 0, 255).astype(np.uint8)

    d = int(130 * separation * (cw / CANVAS_W))
    ground = Image.new("RGB", (cw, ch), XRAY_GROUND)
    draw = ImageDraw.Draw(ground, "RGBA")

    def paste_card(rgb: np.ndarray, alpha_f: np.ndarray, dx: int, dy: int
                   ) -> None:
        card = Image.fromarray(rgb)
        mask = Image.fromarray(
            (np.clip(alpha_f, 0.0, 1.0) * 255).astype(np.uint8)
        )
        ground.paste(card, (dx, dy), mask)
        draw.rectangle(
            [dx, dy, dx + cw - 1, dy + ch - 1],
            outline=(190, 195, 205),
            width=max(2, cw // 360),
        )

    ones = np.ones((ch, cw), dtype=np.float64)
    paste_card(bg, ones, -d, -d)
    labels = ["1 · BACK — blurred source"]
    if hook_rgba is not None:
        paste_card(hook_rgba[..., :3], hook_rgba[..., 3] / 255.0, 0, 0)
        labels.append("2 · MIDDLE — hook text")
    paste_card(fg, a, d, d)
    labels.append("3 · FRONT — speaker cutout")

    # Labels are a legend, not per-card tags: the back card sits at a
    # negative offset and the middle card is transparent, so per-card
    # anchors collide at the same clamped corner and overprint (seen
    # in the first xray still). A stacked legend on dark chips, in
    # stack order, cannot collide.
    font = _load_font(max(20, cw // 28))
    y = 12
    for label in labels:
        w = draw.textlength(label, font=font)
        draw.rectangle([8, y - 4, 16 + w, y + font.size + 8],
                       fill=(0, 0, 0, 150))
        draw.text((14, y), label, font=font, fill=(255, 255, 255))
        y += font.size + 16
    return np.asarray(ground)


def separation_at(t_s: float, ramp_s: float = 1.5) -> float:
    """Xray separation ramps 0 -> 1 over ``ramp_s`` and then holds."""
    if ramp_s <= 0:
        return 1.0
    return min(1.0, max(0.0, t_s / ramp_s))


# ---------------------------------------------------------------- video


def _encode_video(
    frames: Iterator[np.ndarray],
    out_path: str | Path,
    *,
    fps: float,
    canvas: tuple[int, int],
    audio_from: str | Path | None = None,
) -> None:
    """Pipe raw RGB frames to FFmpeg (libx264/yuv420p), muxing audio."""
    cw, ch = canvas
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{cw}x{ch}", "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_from is not None:
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(out_path)]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8)
                             .tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out_path}")


def render_composite_video(
    source_frames: Sequence[np.ndarray],
    alphas: Sequence[np.ndarray],
    out_path: str | Path,
    *,
    fps: float,
    hook_rgba: np.ndarray | None = None,
    hook_window: tuple[float, float] | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    audio_from: str | Path | None = None,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    backdrop: str = DEFAULT_BACKDROP,
    graphics: Sequence[GraphicEntry] | None = None,
    speaker_placements: Sequence[object | None] | None = None,
    backdrop_seed: int | None = None,
) -> None:
    """Render the composite over a whole clip.

    ``source_frames`` / ``alphas`` are source-resolution and aligned
    1:1. The hook layer (pre-rendered once - the text is static)
    appears only inside ``hook_window`` seconds. ``graphics`` (the
    context graphics plane) defaults to None: no graphics entries,
    the original stack, byte-identical output.
    ``speaker_placements`` (layout states) is an optional per-frame
    sequence of explicit speaker placements aligned with the
    frames; entries may be None (historic fit). None for the whole
    argument = the original render, byte-identical.
    """
    if len(source_frames) != len(alphas):
        raise ValueError("source_frames and alphas must align 1:1")
    if speaker_placements is not None \
            and len(speaker_placements) != len(source_frames):
        raise ValueError("speaker_placements must align 1:1 with "
                         "the frames")

    def gen() -> Iterator[np.ndarray]:
        for i, (frame, alpha) in enumerate(zip(source_frames, alphas)):
            t = i / fps
            hook = hook_rgba if hook_visible_at(t, hook_window) \
                else None
            plane = graphics_layer_at(graphics, t, canvas) \
                if graphics else None
            yield composite_frame(
                frame, alpha, hook, canvas, blur_radius=blur_radius,
                backdrop=backdrop, graphics_rgba=plane,
                backdrop_progress=(i / len(source_frames))
                if source_frames else 0.0,
                backdrop_seed=backdrop_seed,
                speaker_placement=(speaker_placements[i]
                                   if speaker_placements is not None
                                   else None))

    _encode_video(gen(), out_path, fps=fps, canvas=canvas,
                  audio_from=audio_from)


def export_xray_mp4(
    source_frames: Sequence[np.ndarray],
    alphas: Sequence[np.ndarray],
    out_path: str | Path,
    *,
    fps: float,
    hook_rgba: np.ndarray | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    backdrop: str = DEFAULT_BACKDROP,
) -> None:
    """Render the xray diagnostic: layers separate over the first 1.5s."""
    if len(source_frames) != len(alphas):
        raise ValueError("source_frames and alphas must align 1:1")

    def gen() -> Iterator[np.ndarray]:
        for i, (frame, alpha) in enumerate(zip(source_frames, alphas)):
            yield xray_frame(
                frame, alpha, hook_rgba, canvas,
                separation=separation_at(i / fps),
                blur_radius=blur_radius, backdrop=backdrop)

    _encode_video(gen(), out_path, fps=fps, canvas=canvas)


# ---------------------------------------------------------------- io / cli


def read_video_frames(path: str | Path) -> list[np.ndarray]:
    """Decode a clip to a list of RGB uint8 frames via FFmpeg."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True,
    )
    frame_bytes = w * h * 3
    n = len(proc.stdout) // frame_bytes
    return [
        np.frombuffer(proc.stdout[i * frame_bytes:(i + 1) * frame_bytes],
                      dtype=np.uint8).reshape(h, w, 3).copy()
        for i in range(n)
    ]


def video_fps(path: str | Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    num, _, den = probe.stdout.strip().partition("/")
    return float(num) / float(den or 1)


def _load_pipeline(clip: Path, sam_response: Path,
                   max_frames: int | None = None):
    """Shared CLI wiring: frames, guides, finished alphas.

    ``max_frames`` trims the clip AND the SAM frames before the
    finish pass, so a short diagnostic render does not pay for
    finishing frames it will never show.
    """
    import json

    from .mask_decode import finished_alpha, parse_sam_response

    frames = read_video_frames(clip)
    raw = json.loads(Path(sam_response).read_text(encoding="utf-8"))
    parsed = parse_sam_response(raw)
    if max_frames is not None:
        frames = frames[:max_frames]
        keep = {f.index for f in sorted(parsed, key=lambda f: f.index)
                [:max_frames]}
        parsed = [f for f in parsed if f.index in keep]
    guides = (
        f.astype(np.float64).mean(axis=2) / 255.0 for f in frames
    )
    _, alphas = finished_alpha(parsed, guides)
    return frames, alphas


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("composite", "xray"):
        p = sub.add_parser(name)
        p.add_argument("--clip", required=True, type=Path)
        p.add_argument("--sam-response", required=True, type=Path)
        p.add_argument("--hook", default="")
        p.add_argument("--hook-y-frac", type=float, default=0.30,
                       help="hook band centre as a fraction of canvas "
                            "height (manual placement; default 0.30)")
        if name == "composite":
            p.add_argument("--hook-auto-layout", action="store_true",
                           help="compute the hook band from the decoded "
                                "head geometry via layout_law.plan_hook "
                                "(overrides --hook-y-frac)")
        p.add_argument("--out", required=True, type=Path)
        if name == "composite":
            p.add_argument("--hook-start", type=float, default=0.0)
            p.add_argument("--hook-end", type=float, default=6.0)
            p.add_argument("--audio", action="store_true",
                           help="mux the source clip's audio track")
        else:
            p.add_argument("--seconds", type=float, default=None,
                           help="render only the first N seconds")

    args = parser.parse_args(argv)
    fps = video_fps(args.clip)
    max_frames = None
    if args.cmd == "xray" and args.seconds is not None:
        max_frames = int(args.seconds * fps)
    frames, alphas = _load_pipeline(
        args.clip, args.sam_response, max_frames=max_frames)
    hook_y_frac = args.hook_y_frac
    hook_rgba = None
    if args.cmd == "composite" and args.hook_auto_layout and args.hook:
        # Layout law replaces the manual knob: plan the band from the
        # decoded geometry of the hook window itself, and render the
        # layer through layout_law so the shipped pixels are exactly
        # the planned ones (safe-width budget included).
        from .layout_law import plan_hook, render_hook_layer, to_canvas_alpha

        f0 = max(0, int(args.hook_start * fps))
        f1 = min(len(alphas), int(args.hook_end * fps))
        placement = plan_hook(
            args.hook.upper(),
            [to_canvas_alpha(a) for a in alphas[f0:max(f1, f0 + 1)]],
        )
        hook_y_frac = placement.y_center_frac
        hook_rgba = render_hook_layer(
            args.hook.upper(), y_center_frac=hook_y_frac)
        print(
            f"layout_law: hook y-frac {hook_y_frac:.2f}, predicted "
            f"glyph occlusion {placement.predicted_occlusion:.3f} "
            f"(cap 0.25), safe zone {placement.within_safe_zone}, "
            f"ok {placement.ok}"
        )
    if hook_rgba is None and args.hook:
        hook_rgba = hook_layer(args.hook.upper(), y_center_frac=hook_y_frac)

    if args.cmd == "composite":
        render_composite_video(
            frames, alphas, args.out, fps=fps, hook_rgba=hook_rgba,
            hook_window=(args.hook_start, args.hook_end),
            audio_from=args.clip if args.audio else None,
        )
    else:
        export_xray_mp4(frames, alphas, args.out, fps=fps,
                        hook_rgba=hook_rgba)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
