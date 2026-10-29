"""Offline validation of the 2026-10-29 refinement block.

Re-renders pipeline-run-2's shorts 1 and 3 ENTIRELY from cached
artifacts (window.mp4 + sam-response.json + captions.json) with
the refined planner / matte / captions / backdrop, then verifies
the re-renders offline. ZERO paid calls: no client is constructed;
the Wonder judge is the proof pack's deterministic FakeWonder, and
the spend ledger is never opened.

Short 1's run-3 chain failed draft 1 on F5 only and was rescued by
TRIM_SILENCE. The trim executor is deterministic (the Spark part
was only the menu PICK, which is on the run-3 receipt), so this
driver replays the executor directly on the refined geometry to
give short 1 the same full chain. Short 3 passed its first draft
in run 3 and is re-verified as a first draft here.

Outputs land in the directory given by --out: refined renders,
offline receipts (JSON), before/after stills, and report.json.

Run: PYTHONPATH=src python3 tools/refinement_rerender.py \
        --run-dir <pipeline-run-2> --out <refinement-dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_director.caption_align import CaptionCue, load_energy
from spatial_director.compositor import (
    composite_frame,
    read_video_frames,
    video_fps,
)
from spatial_director.layout_law import (
    apply_speaker_reframe,
    frame_geometry,
    plan_captions,
    plan_hook_effect,
    plan_speaker_reframe,
    render_hook_layer,
    to_canvas_alpha,
)
from spatial_director.mask_decode import (
    decode_object_canvas,
    finished_alpha,
    parse_sam_response,
    select_primary_id,
)
from spatial_director.matte_finish import (
    largest_connected_component,
    refine_matte,
)
from spatial_director.pipeline import LazyCanvasAlphas, _annotate_effect
from spatial_director.rescue_loop import (
    RenderArtifacts,
    _exec_trim_silence,
    render_composite_with_captions,
)
from spatial_director.temporal_smooth import ema_smooth
from spatial_director.verifier import HookSpec, VerifyInputs, verify

HOOK_WINDOW_S = 6.0
OLD_FINISH = dict(choke_px=2, feather_r=2, gf_radius=8, gf_eps=1e-3)


class FakeJudgment:
    def __init__(self, can: bool = True):
        self.can_summarize = can
        self.summary = ("A speaker discusses a sports team's season."
                        if can else "")
        self.cost_usd = 0.0  # fake judge: nothing is billed, ever
        self.request_id = "fake-judge-no-network"


class FakeWonder:
    """Deterministic stand-in for the Wonder judge. Never calls out."""

    def judge_muted_hook(self, path, *, purpose: str = "", ts: str = ""):
        return FakeJudgment(True)


def _frame_at(video: Path, t_s: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(t_s), "-i", str(video),
         "-frames:v", "1", str(out)], check=True)


def _lcc_stats(parsed, sample_every: int = 10) -> dict:
    """What the largest-component filter drops, on sampled frames."""
    primary = select_primary_id(parsed)
    dropped = []
    multi = 0
    for frame in parsed[::sample_every]:
        obj = next((o for o in frame.objects
                    if o.object_id == primary), None)
        if obj is None:
            continue
        canvas = decode_object_canvas(obj)
        kept = largest_connected_component(canvas)
        d = int(canvas.sum() - kept.sum())
        if d:
            multi += 1
        dropped.append(d)
    return {
        "sampled_frames": len(dropped),
        "frames_with_islands": multi,
        "max_dropped_px": max(dropped) if dropped else 0,
        "mean_dropped_px": round(float(np.mean(dropped)), 1)
        if dropped else 0.0,
    }


def _matte_stills(short_dir: Path, frames, parsed, out_dir: Path,
                  tag: str, log) -> dict:
    """Before/after matte crops on frame 30, through the real stack
    (decode -> LCC -> EMA), old finish params vs the new defaults,
    composited identically (treated backdrop + planned hook is the
    caller's frame; here the plain composite isolates the matte)."""
    fi = 30
    primary = select_primary_id(parsed)
    canvases = []
    for frame in parsed[: fi + 1]:
        obj = next((o for o in frame.objects
                    if o.object_id == primary), None)
        canvas = (decode_object_canvas(obj) if obj is not None
                  else np.zeros((360, 640), dtype=bool))
        canvases.append(largest_connected_component(canvas))
    smoothed = ema_smooth(canvases)
    guide = frames[fi].astype(np.float64).mean(axis=2) / 255.0
    metrics = {}
    crops = {}
    for name, kw in (("before", OLD_FINISH), ("after", {})):
        alpha = refine_matte(smoothed[fi], guide, **kw)
        raw = canvases[fi]
        metrics[name] = {
            "bleed_alpha_outside_mask": round(
                float(alpha[~raw].sum()), 1),
            "halo_px": int(((alpha > 0.05) & (alpha < 0.95)).sum()),
        }
        comp = composite_frame(frames[fi], alpha)
        canvas_a = to_canvas_alpha(alpha)
        geo = frame_geometry(canvas_a)
        fb = geo.face_box
        crops[name] = comp
        if fb is not None:
            x0, y0, x1, y1 = fb
            head = comp[max(0, y0 - 60):y1 + 60,
                        max(0, x0 - 60):x1 + 60]
            Image.fromarray(head).save(
                out_dir / f"{tag}-matte-head-{name}.png")
            sh = comp[y1 + 120:y1 + 520, max(0, x0 - 60):x1 + 60]
            Image.fromarray(sh).save(
                out_dir / f"{tag}-matte-shoulder-{name}.png")
    return metrics


def _backdrop_stills(frame, alpha, out_dir: Path, tag: str) -> None:
    for name in ("naive", "treated", "brand"):
        comp = composite_frame(frame, alpha, backdrop=name)
        Image.fromarray(comp).save(out_dir / f"{tag}-backdrop-{name}.png")


def process_short(index: int, run_dir: Path, out_dir: Path,
                  hook_text: str, log) -> dict:
    short_dir = run_dir / f"short-{index}"
    tag = f"short{index}"
    work = out_dir / f"{tag}-work"
    work.mkdir(parents=True, exist_ok=True)
    window_clip = short_dir / "window.mp4"
    fps = video_fps(window_clip)
    frames = read_video_frames(window_clip)
    raw_sam = json.loads((short_dir / "sam-response.json").read_text())
    parsed = parse_sam_response(raw_sam)
    if len(frames) > len(parsed):
        frames = frames[:len(parsed)]
    log(f"{tag}: {len(frames)} frames @ {fps:.2f} fps; hook {hook_text!r}")

    lcc = _lcc_stats(parsed)
    log(f"{tag}: LCC stats {lcc}")

    guides = (f.astype(np.float64).mean(axis=2) / 255.0 for f in frames)
    _idx, src_alphas = finished_alpha(parsed, guides)
    log(f"{tag}: finished alpha (new matte) done")

    matte_metrics = _matte_stills(short_dir, frames, parsed, out_dir,
                                  tag, log)
    _backdrop_stills(frames[30], src_alphas[30], out_dir, tag)
    log(f"{tag}: matte metrics {matte_metrics}")

    base_reframe = plan_speaker_reframe(src_alphas)
    cues_data = json.loads((short_dir / "captions.json").read_text())
    cues = [CaptionCue(c["text"], c["start_s"], c["end_s"])
            for c in cues_data["cues"]]
    length = len(frames) / fps
    effect = plan_hook_effect(
        hook_text, src_alphas, base_reframe=base_reframe,
        hook_sample=min(len(src_alphas), int(HOOK_WINDOW_S * fps)))
    if effect.pad_bottom_px != 0:
        frames, src_alphas = apply_speaker_reframe(
            frames, src_alphas, effect.pad_bottom_px)
    placement = effect.placement
    log(f"{tag}: effect plan: pad {effect.pad_bottom_px} "
        f"(lift {effect.lift_px}), frac {placement.y_center_frac:.2f}, "
        f"predicted occ {placement.predicted_occlusion:.4f} "
        f"({placement.effect_status}); {effect.note}")
    canvas_alphas = LazyCanvasAlphas(src_alphas)
    cap_layout = plan_captions(
        [(c.start_s, c.end_s) for c in cues], canvas_alphas, fps=fps)
    body_total = sum(cap_layout.per_cue_body_overlap_px.values())
    log(f"{tag}: caption rect {cap_layout.rect} feasible "
        f"{cap_layout.feasible}; body overlap total {body_total} px")

    hook_spec = HookSpec(hook_text, (0.0, min(HOOK_WINDOW_S, length)),
                         placement.y_center_frac, layout_rendered=True)
    hook_rgba = render_hook_layer(
        hook_text, y_center_frac=placement.y_center_frac)
    draft_path = work / "refined-draft.mp4"
    render_composite_with_captions(
        frames, src_alphas, hook_rgba=hook_rgba,
        hook_window=hook_spec.window, cues=cues,
        caption_rect=cap_layout.rect, audio_from=window_clip,
        out_path=draft_path, fps=fps)
    receipt1 = verify(
        VerifyInputs(
            video_path=draft_path, fps=fps, canvas_alphas=canvas_alphas,
            audio=load_energy(draft_path), hook=hook_spec, cues=cues,
            caption_rect=cap_layout.rect,
            sam_response_path=short_dir / "sam-response.json"),
        wonder=FakeWonder(), draft=1, ts="2026-10-29-offline-refinement")
    _annotate_effect(receipt1)
    receipt1.notes.append(
        "offline refinement validation: Wonder judge stubbed "
        "(FakeWonder, can_summarize=True); zero paid calls.")
    f2 = next(c for c in receipt1.checks if c.check_id == "F2")
    log(f"{tag}: draft verdict {receipt1.verdict} failed "
        f"{receipt1.failed}; F2 measured occ "
        f"{f2.measured['glyph_occlusion_mean']} (alpha-weighted "
        f"{f2.measured['glyph_occlusion_alpha_weighted_mean']})")

    final_path = draft_path
    receipt2 = None
    if receipt1.failed == ["F5"]:
        artifacts = RenderArtifacts(
            window_clip=window_clip, render_path=draft_path,
            frames=frames, src_alphas=src_alphas,
            canvas_alphas=canvas_alphas,
            make_canvas_alphas=lambda a: LazyCanvasAlphas(a),
            fps=fps, hook=hook_spec, cues=cues,
            caption_rect=cap_layout.rect,
            sam_response_path=short_dir / "sam-response.json",
            transcript_text="", work_dir=work)
        execution = _exec_trim_silence(
            artifacts, receipt1, render_composite_with_captions)
        receipt2 = verify(
            VerifyInputs(
                video_path=execution.render_path, fps=fps,
                canvas_alphas=execution.canvas_alphas,
                audio=load_energy(execution.render_path),
                hook=execution.hook, cues=execution.cues,
                caption_rect=execution.caption_rect,
                sam_response_path=short_dir / "sam-response.json"),
            wonder=FakeWonder(), draft=2,
            ts="2026-10-29-offline-refinement")
        _annotate_effect(receipt2)
        receipt2.notes.append(
            "offline refinement validation: TRIM_SILENCE replayed "
            "deterministically from the run-3 receipt's recorded "
            "pick; Wonder judge stubbed; zero paid calls.")
        final_path = execution.render_path
        f2b = next(c for c in receipt2.checks if c.check_id == "F2")
        log(f"{tag}: after trim: verdict {receipt2.verdict} failed "
            f"{receipt2.failed}; F2 measured occ "
            f"{f2b.measured['glyph_occlusion_mean']}")

    final_out = out_dir / f"{tag}-refined.mp4"
    final_out.write_bytes(Path(final_path).read_bytes())
    (out_dir / f"{tag}-offline-receipt.json").write_text(json.dumps({
        "short": index,
        "validation": "offline refinement re-render from cached "
                      "artifacts; FakeWonder judge; zero paid calls",
        "drafts": [r.to_dict() for r in
                   ([receipt1] + ([receipt2] if receipt2 else []))],
    }, indent=2), encoding="utf-8")

    # Before/after frames at 1.0s (short 1's first trim lands at
    # 1.87s, so the timestamp is the same content in both renders).
    _frame_at(short_dir / "final.mp4", 1.0,
              out_dir / f"{tag}-frame-before.png")
    _frame_at(final_out, 1.0, out_dir / f"{tag}-frame-after.png")

    def f2_of(r):
        c = next(x for x in r.checks if x.check_id == "F2")
        return c.measured

    return {
        "short": index,
        "hook_text": hook_text,
        "before_occlusion_measured": 0.0,  # run-3 receipts, all drafts
        "predicted_occlusion": round(placement.predicted_occlusion, 4),
        "predicted_effect_status": placement.effect_status,
        "hook_y_frac": placement.y_center_frac,
        "lift_px": effect.lift_px,
        "applied_pad_bottom_px": effect.pad_bottom_px,
        "draft1": {"verdict": receipt1.verdict,
                   "failed": receipt1.failed,
                   "f2": f2_of(receipt1)},
        "draft2": ({"verdict": receipt2.verdict,
                    "failed": receipt2.failed,
                    "f2": f2_of(receipt2)} if receipt2 else None),
        "caption_rect": list(cap_layout.rect),
        "caption_body_overlap_total_px": body_total,
        "lcc": lcc,
        "matte_frame30": matte_metrics,
        "render": str(final_out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run_dir / "run-summary.json").read_text())
    hooks = {s["index"]: s["hook_text"] for s in summary["shorts"]}
    results = []
    for index in (1, 3):
        results.append(process_short(
            index, args.run_dir, args.out, hooks[index],
            log=lambda m: print(m, flush=True)))
    report = {
        "validation": "offline refinement re-render, cached artifacts "
                      "only, FakeWonder judge, zero paid calls",
        "run_dir": str(args.run_dir),
        "shorts": results,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
