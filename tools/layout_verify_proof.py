"""Offline verifier proof for the layout-state legs (extension S3).

Runs the REAL verifier over S1's layout-proof render
(``pipeline-run-graphics-mcarthur-2/layout-proof/layout-proof.mp4``)
with the render's own declared layout track, and reports the new
Stage-3 legs' measured values:

- F8 layout legs: per declared span, the graphic's measured mask
  fraction inside its declared box, the speaker's in-box fraction
  (Spotlight/Split) and the Cutaway mid-span speaker alpha max;
- F3/F4 span-aware records (Stage-only median; per-span declared-box
  evaluation; exclusion counts).

Then a NEGATIVE pass: the Spotlight speaker box is deliberately
misdeclared (shifted off the speaker) and the same render must FAIL
F8 with that span flagged - a leg that cannot fail is not a check.

Everything is rebuilt deterministically from the cached run
artifacts exactly as tools/layout_proof.py built the render (same
hand-authored spans, same placement calls), and the rebuilt graphic
boxes are asserted against proof-report.json before verifying, so
the declarations measured here are the render's own. No API calls,
$0. Wonder is None: F5 honestly records judge-not-run; the
deliverable is the deterministic layout legs.

Usage:
    PYTHONPATH=src python3 tools/layout_verify_proof.py \
        [--run-dir ~/workspace/meta-hackathon/pipeline-run-graphics-mcarthur-2] \
        [--short short-1]
Writes layout-verify-proof.json beside the render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue, load_energy  # noqa: E402
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
    plan_hook_effect,
    plan_speaker_reframe,
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
from spatial_director.verifier import (  # noqa: E402
    HookSpec,
    VerifyInputs,
    verify,
)

HOOK_WINDOW_S = 6.0

# Identical to tools/layout_proof.py - the render's own plan.
SPANS = [
    ("spotlight", 9, 11, "left"),
    ("split", 13, 16, None),
    ("cutaway", 19, 22, None),
]


def _check(receipt: dict, cid: str) -> dict:
    return next(c for c in receipt["checks"] if c["id"] == cid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(
        Path.home() / "workspace" / "meta-hackathon"
        / "pipeline-run-graphics-mcarthur-2"))
    ap.add_argument("--short", default="short-1")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    short_dir = run_dir / args.short
    proof_dir = run_dir / "layout-proof"
    report = json.loads((proof_dir / "proof-report.json").read_text())
    canvas = DEFAULT_CANVAS

    window = short_dir / "window.mp4"
    fps = video_fps(window)
    frames = read_video_frames(window)
    parsed = parse_sam_response(
        json.loads((short_dir / "sam-response.json").read_text()))
    if len(frames) > len(parsed):
        frames = frames[:len(parsed)]
    guides = (f.astype(np.float64).mean(axis=2) / 255.0 for f in frames)
    _idx, src_alphas = finished_alpha(parsed, guides)
    n = len(src_alphas)

    summary = json.loads((run_dir / "run-summary.json").read_text())
    short_index = int(args.short.rsplit("-", 1)[1])
    short_rec = next(s for s in summary["shorts"]
                     if s["index"] == short_index)
    hook_text = short_rec["hook_text"]

    reframe = plan_speaker_reframe(src_alphas, canvas)
    hook_sample = min(len(src_alphas), int(HOOK_WINDOW_S * fps))
    effect = plan_hook_effect(hook_text, src_alphas, canvas,
                              base_reframe=reframe,
                              hook_sample=hook_sample)
    if effect.pad_bottom_px != 0:
        frames, src_alphas = apply_speaker_reframe(
            frames, src_alphas, effect.pad_bottom_px)
    frame_shape = src_alphas[0].shape

    cues = [CaptionCue(text=c["text"], start_s=c["start_s"],
                       end_s=c["end_s"])
            for c in json.loads(
                (short_dir / "captions.json").read_text())["cues"]]
    caption_rect = tuple(report["caption_rect"])

    # ---- rebuild the render's own layout plan --------------------
    decls: list[LayoutDecl] = []
    plans = []
    gspans: list[GraphicSpan] = []
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
        decls.append(LayoutDecl(start_s, end_s, state))
        plans.append(plan)
        gspans.append(GraphicSpan(
            start_s=start_s, end_s=end_s, kind="card",
            headline=cues[i1].text, source_quote=cues[i0].text,
            layout=state))

    track = build_track(decls, plans, frame_shape, n, fps, canvas)
    placed_alphas = LazyCanvasAlphas(src_alphas, canvas,
                                     placements=track)
    items = [(s, prepare_card(s)) for s in gspans]
    placed, dropped = place_prepared(
        items, placed_alphas, fps=fps, caption_rect=caption_rect,
        canvas=canvas)
    assert not dropped, f"proof rebuild dropped graphics: {dropped}"
    declared_boxes = {g["layout"]: g["box"] for g in report["graphics"]}
    for spec in placed:
        assert list(spec.box) == declared_boxes[spec.layout], (
            f"rebuilt {spec.layout} box {spec.box} != render's "
            f"declared {declared_boxes[spec.layout]}")
    print("rebuilt graphic boxes match proof-report.json exactly")

    # Declarations: the report's own records + each span's graphic box
    # (the field the pipeline's plan_layout adds at declare time).
    declarations = []
    for d in report["declarations"]:
        d = dict(d)
        d["graphic_box"] = declared_boxes[d["layout"]]
        declarations.append(d)

    def run_verify(decls_in: list[dict]) -> dict:
        inputs = VerifyInputs(
            video_path=proof_dir / "layout-proof.mp4",
            fps=fps,
            canvas_alphas=placed_alphas,
            audio=load_energy(proof_dir / "layout-proof.mp4"),
            hook=HookSpec(hook_text, (0.0, HOOK_WINDOW_S),
                          effect.placement.y_center_frac,
                          layout_rendered=True),
            cues=cues,
            caption_rect=caption_rect,
            sam_response_path=short_dir / "sam-response.json",
            graphics=placed,
            layout_spans=decls_in,
        )
        return verify(inputs, wonder=None, draft=1,
                      ts="layout-verify-proof").to_dict()

    # ---- positive pass: the render's own declarations ------------
    receipt = run_verify(declarations)
    f8 = _check(receipt, "F8")
    f3 = _check(receipt, "F3")
    f4 = _check(receipt, "F4")
    f9 = _check(receipt, "F9")
    positive = {
        "failed_checks": receipt["failed_checks"],
        "expected_non_layout_failures": {
            "F5": "Wonder is None offline - the judge honestly "
                  "records not-run and F5 cannot pass",
            "F9": "the S1 proof render carries the raw window "
                  "audio (never loudness-normalised - it is a "
                  "geometry proof): measured "
                  f"{f9['measured'].get('integrated_lufs')} LUFS / "
                  f"{f9['measured'].get('true_peak_dbtp')} dBTP",
        },
        "receipt_layout_spans": receipt.get("layout_spans"),
        "graphics_coverage": receipt.get("graphics_coverage"),
        "F8": {"passed": f8["pass"],
               "graphics": f8["measured"].get("graphics"),
               "layout_spans": f8["measured"].get("layout_spans"),
               "offending_layout_indices":
                   f8["measured"].get("offending_layout_indices")},
        "F3": {"passed": f3["pass"],
               "layout_spans": f3["measured"].get("layout_spans")},
        "F4": {"passed": f4["pass"],
               "layout": f4["measured"].get("layout")},
    }

    # ---- negative pass: misdeclared Spotlight speaker box --------
    bad = [dict(d) for d in declarations]
    bx = bad[0]["speaker_box"]
    bad[0]["speaker_box"] = [bx[0], bx[1] + 700, bx[2], bx[3] + 700]
    bad_receipt = run_verify(bad)
    bad_f8 = _check(bad_receipt, "F8")
    negative = {
        "misdeclared": {"span_index": 0, "layout": "spotlight",
                        "speaker_box": bad[0]["speaker_box"]},
        "failed_checks": bad_receipt["failed_checks"],
        "F8": {"passed": bad_f8["pass"],
               "layout_spans": bad_f8["measured"].get("layout_spans"),
               "offending_layout_indices":
                   bad_f8["measured"].get(
                       "offending_layout_indices")},
    }

    out = {"render": str(proof_dir / "layout-proof.mp4"),
           "positive": positive, "negative": negative}
    out_path = proof_dir / "layout-verify-proof.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

    ok = (f8["pass"]
          and not f8["measured"].get("offending_layout_indices")
          and not bad_f8["pass"]
          and bad_f8["measured"].get("offending_layout_indices") == [0])
    print("LAYOUT VERIFY PROOF:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
