"""Tests for the extension Stage 2 live wiring.

The load-bearing claim: a pipeline render and a rescue re-render
of the same Short use the IDENTICAL layout track, because both
come from the one shared builder (graphics_layer.plan_layout) -
the pipeline calls it on draft 1's placed specs, repairs carry
the track through untouched, and executors that rebuild frames
re-plan it from the carried specs through the same function.

Offline, synthetic geometry only; no API calls.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director import graphics_layer, pipeline, rescue_loop  # noqa: E402
from spatial_director.graphics_layer import GraphicSpec, plan_layout  # noqa: E402
from spatial_director.rescue_loop import (  # noqa: E402
    RenderArtifacts,
    _exec_drop_graphics,
    _rebuild_layout,
    _render_execution,
)
from spatial_director.verifier import CheckResult, HookSpec, Receipt  # noqa: E402

CANVAS = (1080, 1920)
SRC_SHAPE = (583, 640)
FPS = 30.0
N = 90  # 3 seconds


def person_alphas(n=N, shape=SRC_SHAPE):
    a = np.zeros(shape, dtype=np.float32)
    a[77:, 254:476] = 1.0
    return [a.copy() for _ in range(n)]


def spec(layout, start, end, box, compare=None):
    layer = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    layer[y0:y1, x0:x1, 3] = 255
    return GraphicSpec(
        kind="card", start_s=start, end_s=end, box=box,
        layer_rgba=layer, headline="TEST",
        source_quote="q", motion="static", layout=layout,
        compare=compare)


def make_specs():
    return [
        spec("spotlight", 0.3, 1.2, (216, 1103, 864, 1424)),
        # A COMPARE split: the genuine Split user (two compared
        # items beside the speaker). A non-compare split would be
        # converted to Spotlight by the fix-round-4 assist rule -
        # these wiring tests are about track identity, so the
        # fixture uses the state that legitimately stays Split.
        spec("split", 1.3, 2.2, (208, 472, 856, 793),
             compare=[{"label": "HOME", "value": 12},
                      {"label": "AWAY", "value": 9}]),
        spec("cutaway", 2.3, 3.0, (0, 718, 1080, 1201)),
    ]


def track_signature(track):
    return [None if p is None else p.to_dict() for p in track]


class FakeCanvas(list):
    """A make_canvas_alphas result that can carry placements."""

    placements = None


class PlanLayoutTests(unittest.TestCase):
    def test_builder_is_one_shared_object(self):
        self.assertIs(pipeline.plan_layout, graphics_layer.plan_layout)
        self.assertIs(rescue_loop.plan_layout,
                      graphics_layer.plan_layout)

    def test_no_layout_specs_returns_none(self):
        specs = [spec("stage", 0.3, 1.2, (216, 1103, 864, 1424))]
        self.assertIsNone(plan_layout(
            specs, person_alphas(), fps=FPS, canvas=CANVAS))

    def test_track_and_declarations(self):
        specs = make_specs()
        built = plan_layout(specs, person_alphas(), fps=FPS,
                            canvas=CANVAS)
        self.assertIsNotNone(built)
        track, decls, rects = built
        self.assertEqual(len(track), N)
        self.assertEqual([d["layout"] for d in decls],
                         ["spotlight", "split", "cutaway"])
        self.assertTrue(all(d["feasible"] for d in decls))
        self.assertEqual(decls[0]["graphic_box"],
                         [216, 1103, 864, 1424])
        self.assertEqual(decls[0]["anchor_quality"], "high")
        # Split carries the span caption rect override.
        self.assertEqual(len(rects), 1)
        self.assertEqual(rects[0][:2], (1.3, 2.2))
        # Mid-Spotlight the speaker rests in the PiP; mid-Cutaway
        # the speaker is faded out.
        self.assertIsNotNone(track[22])
        self.assertEqual(track[85].alpha_scale, 0.0)

    def test_deterministic_same_inputs_same_track(self):
        a = plan_layout(make_specs(), person_alphas(), fps=FPS,
                        canvas=CANVAS)
        b = plan_layout(make_specs(), person_alphas(), fps=FPS,
                        canvas=CANVAS)
        self.assertEqual(a[1], b[1])
        self.assertEqual(track_signature(a[0]),
                         track_signature(b[0]))

    def test_infeasible_spotlight_downgrades_honestly(self):
        # The graphic box covers BOTH PiP corners: no legal PiP
        # exists, so the span falls back to Stage on the record.
        specs = [spec("spotlight", 0.3, 2.5, (0, 300, 1080, 900))]
        built = plan_layout(specs, person_alphas(), fps=FPS,
                            canvas=CANVAS)
        track, decls, _rects = built
        self.assertIsNone(track)
        self.assertFalse(decls[0]["feasible"])
        self.assertEqual(decls[0]["downgraded_to"], "stage")
        self.assertEqual(specs[0].layout, "stage")


def make_artifacts(td, specs, built):
    track, decls, rects = built
    return RenderArtifacts(
        window_clip="window.mp4", render_path="draft-1.mp4",
        frames=[np.zeros((4, 4, 3), dtype=np.uint8)
                for _ in range(N)],
        src_alphas=person_alphas(),
        canvas_alphas=FakeCanvas(person_alphas()),
        make_canvas_alphas=lambda alphas: FakeCanvas(alphas),
        fps=FPS,
        hook=HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                      layout_rendered=True),
        cues=[], caption_rect=(60, 1460, 980, 1600),
        work_dir=td, canvas=CANVAS, graphics=specs,
        layout_track=track, layout_spans=decls,
        span_caption_rects=rects)


class RescueCarryTests(unittest.TestCase):
    def test_render_execution_carries_the_track_identically(self):
        specs = make_specs()
        built = plan_layout(specs, person_alphas(), fps=FPS,
                            canvas=CANVAS)
        captured = {}

        def fake_renderer(frames, src_alphas, **kwargs):
            captured.update(kwargs)
            return Path("out.mp4")

        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(td, specs, built)
            execution = _render_execution(
                artifacts, frames=artifacts.frames,
                src_alphas=artifacts.src_alphas,
                canvas_alphas=artifacts.canvas_alphas,
                hook=artifacts.hook, cues=[], 
                caption_rect=artifacts.caption_rect,
                audio_from=None, out_name="draft2.mp4",
                detail="carry test", renderer=fake_renderer)
        self.assertIs(captured["layout_track"], built[0])
        self.assertEqual(captured["span_caption_rects"], built[2])
        self.assertEqual(execution.layout_spans, built[1])

    def test_rebuild_reproduces_the_pipeline_track(self):
        specs = make_specs()
        built = plan_layout(specs, person_alphas(), fps=FPS,
                            canvas=CANVAS)
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(td, specs, built)
            rebuilt = _rebuild_layout(
                artifacts, specs, artifacts.src_alphas, N)
        self.assertIsNotNone(rebuilt)
        canvas_alphas, track, rects, decls = rebuilt
        self.assertEqual(decls, built[1])
        self.assertEqual(track_signature(track),
                         track_signature(built[0]))
        self.assertIs(canvas_alphas.placements, track)

    def test_drop_graphics_replans_declarations_from_kept(self):
        specs = make_specs()
        built = plan_layout(specs, person_alphas(), fps=FPS,
                            canvas=CANVAS)
        captured = {}

        def fake_renderer(frames, src_alphas, **kwargs):
            captured.update(kwargs)
            return Path("out.mp4")

        receipt = Receipt(
            artifact="draft-1.mp4", verdict="FAIL",
            checks=[CheckResult(
                "F8", "context graphics integrity", False,
                {"offending_indices": [0],
                 "collision_violations": [],
                 "offending_backdrop_indices": []}, {})])
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(td, specs, built)
            execution = _exec_drop_graphics(
                artifacts, receipt, renderer=fake_renderer)
        # The Spotlight spec (index 0) dropped: its declaration
        # is gone from the re-verify's inputs, the rest remain.
        self.assertEqual(
            [d["layout"] for d in execution.layout_spans],
            ["split", "cutaway"])
        self.assertIn("layout_track", captured)


if __name__ == "__main__":
    unittest.main()
