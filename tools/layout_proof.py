"""Offline layout-states proof render (extension Stage 1).

Renders one cached Short (default: McArthur-2 short-1, the demo
hero) through the REAL shared render path with a hand-authored
layout plan built from its own caption cues:

- Spotlight (PiP top-left + centre-majority card) over the
  "cooking meat / hamburger" passage,
- Split (speaker in the bottom band, card in the top zone,
  span caption rect) over the "metal off gassing" passage,
- Cutaway (full-frame card, speaker faded out) over the closing
  "space smells like hamburger" passage,

then measures the placed geometry F3/F4-style (median centroid,
containment, head margin, transition containment, cutaway
absence, caption overlap) and writes the render, stills and a
JSON report. Cached artifacts only - no API calls, $0.

Usage:
    PYTHONPATH=src python3 tools/layout_proof.py \
        [--run-dir ~/workspace/meta-hackathon/pipeline-run-graphics-mcarthur-2] \
        [--short short-1] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue  # noqa: E402
from spatial_director.compositor import (  # noqa: E402
    DEFAULT_CANVAS,
    read_video_frames,
    video_fps,
)
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpan,
    place_prepared,
    prepare_card,
)
from spatial_director.layout_law import (  # noqa: E402
    apply_speaker_reframe,
    face_pixels,
    frame_geometry,
    plan_captions,
    plan_hook_effect,
    plan_speaker_reframe,
    render_hook_layer,
)
from spatial_director.layout_states import (  # noqa: E402
    LayoutDecl,
    build_track,
    plan_cutaway,
    plan_spotlight,
    plan_split,
)
from spatial_director.mask_decode import (  # noqa: E402
    finished_alpha,
    parse_sam_response,
)
from spatial_director.pipeline import LazyCanvasAlphas  # noqa: E402
from spatial_director.rescue_loop import (  # noqa: E402
    graphics_entries,
    render_composite_with_captions,
)

HOOK_WINDOW_S = 6.0

# Hand-authored from short-1's own captions.json (cue-aligned):
# (state, first cue index, last cue index, corner or None)
SPANS = [
    ("spotlight", 9, 11, "left"),    # 9.31-12.47 hamburger passage
    ("split", 13, 16, None),         # 13.06-16.93 off-gassing
    ("cutaway", 19, 22, None),       # 18.42-end: the closing line
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(
        Path.home() / "workspace" / "meta-hackathon"
        / "pipeline-run-graphics-mcarthur-2"))
    ap.add_argument("--short", default="short-1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    short_dir = run_dir / args.short
    out_dir = Path(args.out) if args.out else run_dir / "layout-proof"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = DEFAULT_CANVAS
    cw, ch = canvas

    window = short_dir / "window.mp4"
    fps = video_fps(window)
    frames = read_video_frames(window)
    parsed = parse_sam_response(
        json.loads((short_dir / "sam-response.json").read_text()))
    # The pipeline's alignment rule: a surplus of decoded frames is
    # the tail - truncate the guides to the parsed count.
    if len(frames) > len(parsed):
        print(f"aligning guides: {len(frames)} decoded vs "
              f"{len(parsed)} SAM frames; truncating the tail")
        frames = frames[:len(parsed)]
    guides = (f.astype(np.float64).mean(axis=2) / 255.0 for f in frames)
    _idx, src_alphas = finished_alpha(parsed, guides)
    n = len(src_alphas)
    print(f"{n} frames @ {fps:.3f} fps")

    summary = json.loads((run_dir / "run-summary.json").read_text())
    short_index = int(args.short.rsplit("-", 1)[1])
    short_rec = next(s for s in summary["shorts"]
                     if s["index"] == short_index)
    hook_text = short_rec["hook_text"]

    # The shipped construction, exactly as the pipeline does it:
    # reframe planned, hook effect may add pad, applied ONCE.
    reframe = plan_speaker_reframe(src_alphas, canvas)
    hook_sample = min(len(src_alphas), int(HOOK_WINDOW_S * fps))
    effect = plan_hook_effect(hook_text, src_alphas, canvas,
                              base_reframe=reframe,
                              hook_sample=hook_sample)
    if effect.pad_bottom_px != 0:
        frames, src_alphas = apply_speaker_reframe(
            frames, src_alphas, effect.pad_bottom_px)
    frame_shape = src_alphas[0].shape
    print(f"padded frame shape {frame_shape}, pad "
          f"{effect.pad_bottom_px}px; hook {hook_text!r}")

    cues = [CaptionCue(text=c["text"], start_s=c["start_s"],
                       end_s=c["end_s"])
            for c in json.loads(
                (short_dir / "captions.json").read_text())["cues"]]
    stage_alphas = LazyCanvasAlphas(src_alphas, canvas)
    cap_layout = plan_captions(
        [(c.start_s, c.end_s) for c in cues], stage_alphas,
        fps=fps, canvas=canvas)
    caption_rect = cap_layout.rect
    hook_rgba = render_hook_layer(
        hook_text, canvas,
        y_center_frac=effect.placement.y_center_frac)
    hook_window = (0.0, min(HOOK_WINDOW_S, n / fps))

    # ---- the layout plan -------------------------------------------
    decls: list[LayoutDecl] = []
    plans = []
    gspans: list[GraphicSpan] = []
    declarations = []
    for state, i0, i1, corner in SPANS:
        start_s, end_s = cues[i0].start_s, cues[i1].end_s
        if state == "spotlight":
            lo, hi = int(start_s * fps), int(end_s * fps)
            plan = plan_spotlight(src_alphas[lo:hi], canvas,
                                  corner=corner)
        elif state == "split":
            lo, hi = int(start_s * fps), int(end_s * fps)
            plan = plan_split(src_alphas[lo:hi], canvas)
        else:
            plan = plan_cutaway(frame_shape, canvas)
        decl = LayoutDecl(start_s, end_s, state)
        decls.append(decl)
        plans.append(plan)
        headline = cues[i1].text
        gspans.append(GraphicSpan(
            start_s=start_s, end_s=end_s, kind="card",
            headline=headline, source_quote=cues[i0].text,
            layout=state))
        declarations.append(plan.declaration(start_s, end_s))
        print(f"{state} {start_s:.2f}-{end_s:.2f}s feasible "
              f"{plan.feasible}: {plan.note}")

    track = build_track(decls, plans, frame_shape, n, fps, canvas)
    placed_alphas = LazyCanvasAlphas(src_alphas, canvas,
                                     placements=track)

    items = [(s, prepare_card(s)) for s in gspans]
    placed, dropped = place_prepared(
        items, placed_alphas, fps=fps, caption_rect=caption_rect,
        canvas=canvas)
    print(f"graphics placed {len(placed)}, dropped "
          f"{[d['reason'] for d in dropped]}")
    for spec in placed:
        print(f"  {spec.layout}: box {spec.box}")

    split_plan = plans[1]
    span_rects = [(decls[1].start_s, decls[1].end_s,
                   tuple(split_plan.caption_rect))]
    out_mp4 = out_dir / "layout-proof.mp4"
    render_composite_with_captions(
        frames, src_alphas, hook_rgba=hook_rgba,
        hook_window=hook_window, cues=cues,
        caption_rect=caption_rect, audio_from=window,
        out_path=out_mp4, fps=fps, canvas=canvas,
        graphics=graphics_entries(placed),
        layout_track=track, span_caption_rects=span_rects)
    print(f"rendered {out_mp4}")

    # ---- measurements ----------------------------------------------
    def span_frames(d: LayoutDecl):
        return range(int(d.start_s * fps), min(n, int(d.end_s * fps)))

    cut_frames = set()
    for d in decls:
        if d.state == "cutaway":
            cut_frames.update(span_frames(d))

    cents, contained, head_tops = [], 0, []
    per_span = {}
    for d, plan in zip(decls, plans):
        sc, sh_t = [], []
        for i in span_frames(d):
            g = frame_geometry(placed_alphas[i])
            if g.face_centroid is not None:
                sc.append(g.face_centroid)
            if g.head_top is not None:
                sh_t.append(g.head_top)
        per_span[d.state] = {
            "median_centroid": (
                [round(float(np.median([c[0] for c in sc])), 1),
                 round(float(np.median([c[1] for c in sc])), 1)]
                if sc else None),
            "head_top_min": int(min(sh_t)) if sh_t else None,
            "frames": len(list(span_frames(d))),
        }
    trans_frames, trans_ok, trans_absent = 0, 0, 0
    boundaries = sorted({d.start_s for d in decls}
                        | {d.end_s for d in decls})
    cap_overlap_stage = 0
    for i in range(n):
        a = placed_alphas[i]
        g = frame_geometry(a)
        t = i / fps
        if i not in cut_frames:
            if g.face_centroid is not None:
                cents.append(g.face_centroid)
            if g.head_top is not None:
                head_tops.append(g.head_top)
            fb = g.face_box
            if fb is not None and fb[0] > 0 and fb[2] < cw \
                    and fb[1] > 0:
                contained += 1
            face = face_pixels(a)
            x0, y0, x1, y1 = caption_rect
            cap_overlap_stage += int(face[y0:y1, x0:x1].sum())
        if any(abs(t - b) <= 0.15 for b in boundaries):
            trans_frames += 1
            fb = g.face_box
            if fb is None:
                # A cutaway fade frame: the speaker is below the
                # presence threshold - nothing to contain.
                trans_absent += 1
            elif fb[0] > 0 and fb[2] < cw and fb[1] > 0:
                trans_ok += 1
    cut_mid = [placed_alphas[i] for i in span_frames(decls[2])
               if abs(i / fps - (decls[2].start_s + decls[2].end_s)
                      / 2) < 0.2]
    cutaway_max_alpha = max((float(a.max()) for a in cut_mid),
                            default=0.0)

    n_measured = n - len(cut_frames)
    report = {
        "source": f"{run_dir.name}/{args.short}",
        "frames": n, "fps": fps,
        "caption_rect": list(caption_rect),
        "summary_caption_rect": short_rec["layout"]["caption_rect"],
        "declarations": declarations,
        "graphics": [s.to_dict() for s in placed],
        "dropped": dropped,
        "measurements": {
            "whole_non_cutaway": {
                "median_centroid": [
                    round(float(np.median([c[0] for c in cents])), 1),
                    round(float(np.median([c[1] for c in cents])), 1)],
                "contained_frac": round(contained / n_measured, 4),
                "head_top_min": int(min(head_tops)),
                "frames": n_measured,
            },
            "per_span": per_span,
            "transitions": {"frames": trans_frames,
                            "contained": trans_ok,
                            "face_absent_fade": trans_absent},
            "cutaway_max_alpha_mid_span": cutaway_max_alpha,
            "caption_face_overlap_px_total": cap_overlap_stage,
        },
        "render": str(out_mp4),
    }
    (out_dir / "proof-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["measurements"], indent=2))

    stills = {"stage": 3.0, "transition": decls[0].start_s - 0.07,
              "spotlight": (decls[0].start_s + decls[0].end_s) / 2,
              "split": (decls[1].start_s + decls[1].end_s) / 2,
              "cutaway": (decls[2].start_s + decls[2].end_s) / 2}
    for name, t in stills.items():
        p = out_dir / f"still-{name}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(out_mp4),
             "-frames:v", "1", str(p)],
            check=True, capture_output=True)
        print(f"still {p} @ {t:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
