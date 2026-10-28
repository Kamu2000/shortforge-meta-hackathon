"""The 3-layer spatial composite at 1080x1920.

The Director's signature frame, back to front:

1. **Back** - the source frame, scaled to cover the canvas and
   blurred: the scene is still recognisably the speaker's room, but
   it reads as a backdrop, not a competing picture.
2. **Middle** - the hook text, rendered programmatically (Pillow)
   and positioned BEHIND the speaker. This layer is the whole point:
   the speaker occludes the words, which is the depth effect editors
   build by hand in After Effects.
3. **Front** - the speaker cutout, composited with the finished
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

import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS_W = 1080
CANVAS_H = 1920
DEFAULT_CANVAS = (CANVAS_W, CANVAS_H)
DEFAULT_BLUR_RADIUS = 36
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


def speaker_layer(
    frame_rgb: np.ndarray,
    alpha: np.ndarray,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the speaker to fit the canvas width, bottom-anchored.

    The scale is ``min(cw/sw, ch/sh)`` - fit the width, unless the
    source is so tall that the height would overflow, in which case
    fit the height instead. The speaker is centred horizontally and
    anchored to the bottom edge (talking-head framing). Returns the
    full-canvas (fg_rgb uint8, alpha float) pair; alpha is 0 wherever
    the speaker does not reach.
    """
    cw, ch = canvas
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
) -> np.ndarray:
    """Render hook text as a transparent RGBA layer at canvas size.

    Words wrap to at most two lines; the font shrinks from
    ``start_size`` until the block fits the width budget. Text is
    centred horizontally and centred vertically on
    ``y_center_frac`` of the canvas height - in the composite it
    sits behind the speaker's head/shoulders, visible at the sides,
    which is the intended occlusion effect. A dark stroke keeps the
    letterforms legible over the blurred background.
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
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(
            ((cw - w) / 2, y),
            line,
            font=font,
            fill=fill,
            stroke_width=max(2, size // 16),
            stroke_fill=(10, 10, 12, 255),
        )
        y += line_h
    return np.asarray(img)


# ---------------------------------------------------------------- stack


def composite_from_layers(
    bg_rgb: np.ndarray,
    hook_rgba: np.ndarray | None,
    fg_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Blend the three canvas-sized layers: back, hook, speaker."""
    out = bg_rgb.astype(np.float64)
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
) -> np.ndarray:
    """One composited frame from a source frame + finished alpha."""
    bg = background_layer(frame_rgb, canvas, blur_radius=blur_radius)
    fg, a = speaker_layer(frame_rgb, alpha, canvas)
    return composite_from_layers(bg, hook_rgba, fg, a)


def hook_visible_at(t_s: float, window: tuple[float, float] | None) -> bool:
    """The hook layer exists only inside its [start, end) window."""
    if window is None:
        return False
    return window[0] <= t_s < window[1]


# ---------------------------------------------------------------- xray


def xray_frame(
    frame_rgb: np.ndarray,
    alpha: np.ndarray,
    hook_rgba: np.ndarray | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    *,
    separation: float = 1.0,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
) -> np.ndarray:
    """The layer stack, fanned apart: back -d, hook 0, speaker +d.

    At ``separation`` 0 this is exactly ``composite_frame``. As it
    grows, the three canvas-sized cards slide diagonally over a dark
    ground with outline + label each, so the demo can show the stack
    separating. Speaker/hook cards are drawn with their real alpha -
    what separates is the genuine layers, not a redraw.
    """
    cw, ch = canvas
    bg = background_layer(frame_rgb, canvas, blur_radius=blur_radius)
    fg, a = speaker_layer(frame_rgb, alpha, canvas)
    if separation <= 0.0:
        return composite_from_layers(bg, hook_rgba, fg, a)

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
) -> None:
    """Render the composite over a whole clip.

    ``source_frames`` / ``alphas`` are source-resolution and aligned
    1:1. The hook layer (pre-rendered once - the text is static)
    appears only inside ``hook_window`` seconds.
    """
    if len(source_frames) != len(alphas):
        raise ValueError("source_frames and alphas must align 1:1")

    def gen() -> Iterator[np.ndarray]:
        for i, (frame, alpha) in enumerate(zip(source_frames, alphas)):
            hook = hook_rgba if hook_visible_at(i / fps, hook_window) \
                else None
            yield composite_frame(
                frame, alpha, hook, canvas, blur_radius=blur_radius)

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
) -> None:
    """Render the xray diagnostic: layers separate over the first 1.5s."""
    if len(source_frames) != len(alphas):
        raise ValueError("source_frames and alphas must align 1:1")

    def gen() -> Iterator[np.ndarray]:
        for i, (frame, alpha) in enumerate(zip(source_frames, alphas)):
            yield xray_frame(
                frame, alpha, hook_rgba, canvas,
                separation=separation_at(i / fps),
                blur_radius=blur_radius)

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
    guides = [
        f.astype(np.float64).mean(axis=2) / 255.0 for f in frames
    ]
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
                            "height (the layout step sets this from "
                            "head geometry; default 0.30)")
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
    hook_rgba = (
        hook_layer(args.hook.upper(), y_center_frac=args.hook_y_frac)
        if args.hook else None
    )

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
