"""F7 slide-repair proof on cached run artifacts (2026-10-30).

Reconstructs Holmes graphics-run short-2 (window 583.55-597.55s,
the genuinely static window whose draft 1 failed ONLY F7) from its
cached SAM response + window clip, runs the real ENGAGE_HOOK_SLIDE
executor (no model call - the repair needs none), and re-verifies
the repaired render. Prints before/after F7 measurements and
writes a JSON note next to the proof render.

Usage:
    PYTHONPATH=src python3 tools/f7_slide_proof.py \
        ~/workspace/meta-hackathon/spike-footage/pipeline-run-graphics-holmes/short-2

No SAM/Voice Transcribe/Spark calls: everything comes from cache
except the local render + verify. Audio energy is read off the
proof render itself, exactly as the pipeline does.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from spatial_director.caption_align import CaptionCue, load_energy
from spatial_director.compositor import read_video_frames, video_fps
from spatial_director.layout_law import apply_speaker_reframe
from spatial_director.mask_decode import finished_alpha, parse_sam_response
from spatial_director.pipeline import LazyCanvasAlphas
from spatial_director.rescue_loop import (
    _exec_engage_hook_slide,
    render_composite_with_captions,
)
from spatial_director.verifier import (
    HookSpec,
    Receipt,
    VerifyInputs,
    verify,
)

CANVAS = (1080, 1920)
# Short-2's certified pipeline geometry (run-summary.json).
HOOK_TEXT = "NOW OR NEVER FOR DAK?"
HOOK_Y_FRAC = 0.6
CAPTION_RECT = (60, 1460, 980, 1600)
REFRAME_PAD_PX = 195


def main() -> int:
    run_dir = Path(sys.argv[1])
    short_dir = run_dir
    raw_sam = json.loads((short_dir / "sam-response.json").read_text())
    frames = read_video_frames(short_dir / "window.mp4")
    fps = video_fps(short_dir / "window.mp4")
    parsed = parse_sam_response(raw_sam)
    if len(frames) > len(parsed):
        frames = frames[:len(parsed)]
    guides = (f.astype(np.float64).mean(axis=2) / 255.0 for f in frames)
    _indices, src_alphas = finished_alpha(parsed, guides)
    frames, src_alphas = apply_speaker_reframe(
        frames, src_alphas, REFRAME_PAD_PX)
    canvas_alphas = LazyCanvasAlphas(src_alphas, CANVAS)

    cues = [CaptionCue(c["text"], c["start_s"], c["end_s"])
            for c in json.loads(
                (short_dir / "captions.json").read_text())["cues"]]
    hook = HookSpec(HOOK_TEXT, (0.0, 6.0), HOOK_Y_FRAC,
                    layout_rendered=True)
    artifacts = _Artifacts(
        frames, src_alphas, canvas_alphas, fps, hook, cues, short_dir)
    draft1_receipt = Receipt(artifact="draft-1.mp4", verdict="FAIL",
                             checks=[])
    execution = _exec_engage_hook_slide(
        artifacts, draft1_receipt, render_composite_with_captions)
    receipt = verify(
        VerifyInputs(
            video_path=execution.render_path, fps=fps,
            canvas_alphas=execution.canvas_alphas,
            audio=load_energy(execution.render_path),
            hook=execution.hook, cues=execution.cues,
            caption_rect=execution.caption_rect,
            hook_slide_px_s=execution.hook_slide_px_s),
        wonder=None, draft=2)
    f7 = next(c for c in receipt.checks if c.check_id == "F7")
    before = {
        "speaker_scale_change_0_1p5s": 0.0027,
        "hook_slide_px_s": 0.0,
        "mean_mask_iou_0_3s": 0.999,
        "pass": False,
    }
    after = {"pass": f7.passed, **f7.measured}
    note = {
        "short": "holmes-graphics-run short-2, window 583.55-597.55s",
        "repair": "ENGAGE_HOOK_SLIDE",
        "repair_detail": execution.detail,
        "hook_slide": execution.hook.slide,
        "hook_slide_px_s": execution.hook_slide_px_s,
        "proof_render": str(execution.render_path),
        "f7_before_draft1_receipt": before,
        "f7_after_proof_render": after,
        "full_verdict_proof_render": receipt.verdict,
        "failed_checks_proof_render": receipt.failed,
        "note": ("F5 fails on the proof render by design (no Wonder "
                 "judge offline); F7 is the check under proof. Other "
                 "checks are reported as measured, not cherry-picked."),
    }
    out_json = short_dir / "f7-slide-proof.json"
    out_json.write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps(note, indent=2))
    return 0 if f7.passed else 1


class _Artifacts:
    """Minimal RenderArtifacts stand-in for the slide executor."""

    def __init__(self, frames, src_alphas, canvas_alphas, fps, hook,
                 cues, short_dir):
        self.window_clip = short_dir / "normalised-audio.wav"
        self.render_path = short_dir / "draft-1.mp4"
        self.frames = frames
        self.src_alphas = src_alphas
        self.canvas_alphas = canvas_alphas
        self.make_canvas_alphas = lambda alphas: LazyCanvasAlphas(
            alphas, CANVAS)
        self.fps = fps
        self.hook = hook
        self.cues = cues
        self.caption_rect = CAPTION_RECT
        self.sam_response_path = short_dir / "sam-response.json"
        self.transcript_text = ""
        self.work_dir = short_dir
        self.canvas = CANVAS
        self.graphics = None


if __name__ == "__main__":
    raise SystemExit(main())
