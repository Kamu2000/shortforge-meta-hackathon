"""Tests for the extension layout states (Stage 1, 2026-11-01).

Offline, synthetic geometry only. The load-bearing claims:

- the explicit Stage placement is the shipped placement, byte for
  byte (the regression-identical default);
- Spotlight / Split plans land the speaker inside the EXISTING
  F2/F3/F4 law when measured through the renderer's own mapping,
  and report infeasible - never cheat - on geometry that cannot
  satisfy it;
- the per-frame track eases between states, keeps the face inside
  the canvas through transitions, fades the speaker out for
  Cutaway, anchors punch-in on the face, and follows a tracked
  centroid when asked;
- the graphics schema carries layout/camera declarations through
  parse -> span -> spec -> dict unchanged, and rejects unusable
  ones.
"""

import json
import sys
import unittest

import numpy as np

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.compositor import composite_frame  # noqa: E402
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpan,
    parse_planner_reply,
    place_prepared,
    prepare_card,
)
from spatial_director.layout_law import (  # noqa: E402
    face_pixels,
    frame_geometry,
    to_canvas_alpha,
)
from spatial_director.layout_states import (  # noqa: E402
    LayoutDecl,
    SpeakerPlacement,
    SpanStats,
    build_track,
    plan_cutaway,
    plan_spotlight,
    plan_split,
    plan_stage,
    smooth_trajectory,
    validate_camera,
)

CANVAS = (1080, 1920)
CAPTION_RECT = (60, 1460, 980, 1600)
SRC_SHAPE = (583, 640)  # (h, w) - a McArthur-like padded frame


def person_alphas(n=6, shape=SRC_SHAPE):
    """A standing-person block: body x 254-476, y 77-bottom."""
    a = np.zeros(shape, dtype=np.float32)
    a[77:, 254:476] = 1.0
    return [a.copy() for _ in range(n)]


def median_centroid(placed):
    cents = [frame_geometry(p).face_centroid for p in placed]
    cents = [c for c in cents if c is not None]
    return (float(np.median([c[0] for c in cents])),
            float(np.median([c[1] for c in cents])))


# ------------------------------------------------------- Stage identity


class StageIdentityTests(unittest.TestCase):
    def test_explicit_stage_matches_legacy_mapping_bytes(self):
        rng = np.random.default_rng(7)
        for shape in [(583, 640), (257, 333), (1920, 1080), (100, 100)]:
            src = (rng.random(shape) > 0.4).astype(np.float32)
            legacy = to_canvas_alpha(src, CANVAS)
            explicit = to_canvas_alpha(
                src, CANVAS,
                placement=SpeakerPlacement.stage(shape, CANVAS))
            self.assertEqual(legacy.shape, explicit.shape)
            self.assertTrue(np.array_equal(legacy, explicit),
                            f"shape {shape} diverged")

    def test_composite_frame_stage_placement_is_identity(self):
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 255, (SRC_SHAPE[0], SRC_SHAPE[1], 3),
                             dtype=np.uint8)
        alpha = person_alphas(1)[0]
        plain = composite_frame(frame, alpha, None, CANVAS)
        placed = composite_frame(
            frame, alpha, None, CANVAS,
            speaker_placement=SpeakerPlacement.stage(SRC_SHAPE, CANVAS))
        self.assertTrue(np.array_equal(plain, placed))

    def test_plan_stage_declares_the_shipped_numbers(self):
        plan = plan_stage(SRC_SHAPE, CANVAS)
        self.assertTrue(plan.feasible)
        d = plan.declaration(0.0, 1.0)
        self.assertEqual(d["layout"], "stage")
        self.assertEqual(d["speaker_box"][2] - d["speaker_box"][0], 1080)


# ------------------------------------------------------------- Spotlight


class SpotlightTests(unittest.TestCase):
    def setUp(self):
        self.alphas = person_alphas()

    def test_plan_lands_inside_the_law(self):
        x0, y0, x1, y1 = CAPTION_RECT
        for corner in ("left", "right"):
            plan = plan_spotlight(self.alphas, CANVAS, corner=corner)
            self.assertTrue(plan.feasible, plan.note)
            placed = [to_canvas_alpha(a, CANVAS,
                                      placement=plan.placement)
                      for a in self.alphas]
            cx, cy = median_centroid(placed)
            self.assertTrue(60 <= cx <= 980, (corner, cx))
            self.assertTrue(288 <= cy <= 1440, (corner, cy))
            for p in placed:
                g = frame_geometry(p)
                self.assertGreaterEqual(g.head_top, 90)
                face = face_pixels(p)
                self.assertEqual(
                    int(face[y0:y1, x0:x1].sum()), 0,
                    "face pixel inside the caption rect")

    def test_plan_records_pip_box(self):
        plan = plan_spotlight(self.alphas, CANVAS, corner="left")
        box = plan.declaration(2.0, 5.0)["speaker_box"]
        self.assertEqual(box[2] - box[0], 430)
        self.assertEqual(box[0], 90)

    def test_extreme_right_centroid_kills_the_right_corner(self):
        stats = SpanStats(n=4, cx_med=640.0, cy_med=148.0,
                          cy_min=148.0, cy_max=148.0, ht_min=77,
                          face_union=(600, 77, 640, 193),
                          mask_union=(600, 77, 640, 583))
        right = plan_spotlight(self.alphas, CANVAS, corner="right",
                               stats=stats)
        self.assertFalse(right.feasible)
        left = plan_spotlight(self.alphas, CANVAS, corner="left",
                              stats=stats)
        self.assertTrue(left.feasible)


# ----------------------------------------------------------------- Split


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.alphas = person_alphas()

    def test_plan_lands_inside_the_law(self):
        plan = plan_split(self.alphas, CANVAS)
        self.assertTrue(plan.feasible, plan.note)
        placed = [to_canvas_alpha(a, CANVAS, placement=plan.placement)
                  for a in self.alphas]
        _cx, cy = median_centroid(placed)
        self.assertLessEqual(cy, 1440)
        for p in placed:
            self.assertGreaterEqual(frame_geometry(p).head_top, 90)
        d = plan.declaration(2.0, 5.0)
        self.assertEqual(d["graphic_zone"], [0, 0, CANVAS[0], 1248])
        self.assertEqual(d["caption_rect"], list(CAPTION_RECT))

    def test_deep_face_centroid_is_honestly_infeasible(self):
        # A thin raised arm (mask top y=0) with the head block low:
        # the crop cannot start below the mask top, so the face
        # centroid cannot be lifted into the band at any scale.
        a = np.zeros(SRC_SHAPE, dtype=np.float32)
        a[0:170, 320:324] = 1.0          # the arm, 4px wide
        a[160:583, 254:476] = 1.0        # head + body, low
        alphas = [a.copy() for _ in range(4)]
        plan = plan_split(alphas, CANVAS)
        self.assertFalse(plan.feasible)
        self.assertIn("centroid", plan.note)


# ----------------------------------------------------------------- Track


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.alphas = person_alphas()
        self.spot = plan_spotlight(self.alphas, CANVAS, corner="left")
        self.cut_plan = plan_cutaway(SRC_SHAPE, CANVAS)
        self.split_plan = plan_split(self.alphas, CANVAS)
        self.fps = 30.0
        self.n = 300  # 10 s

    def test_span_middle_is_the_plan_and_outside_is_stage(self):
        decls = [LayoutDecl(2.0, 5.0, "spotlight")]
        track = build_track(decls, [self.spot], SRC_SHAPE, self.n,
                            self.fps, CANVAS)
        self.assertIsNone(track[30])            # t=1.0: Stage
        mid = track[105]                        # t=3.5: resting PiP
        self.assertIsNotNone(mid)
        self.assertAlmostEqual(mid.ox, self.spot.placement.ox, places=6)
        self.assertAlmostEqual(mid.oy, self.spot.placement.oy, places=6)
        self.assertAlmostEqual(mid.scale, self.spot.placement.scale,
                               places=6)
        self.assertIsNone(track[270])           # t=9.0: Stage again

    def test_transition_interpolates_and_keeps_the_face_inside(self):
        decls = [LayoutDecl(2.0, 5.0, "spotlight")]
        track = build_track(decls, [self.spot], SRC_SHAPE, self.n,
                            self.fps, CANVAS)
        stage = SpeakerPlacement.stage(SRC_SHAPE, CANVAS)
        near = track[57]                        # t=1.9, inside the ease
        self.assertIsNotNone(near)
        self.assertTrue(stage.ox < near.ox < self.spot.placement.ox)
        fb = frame_geometry(self.alphas[0]).face_box
        for i, p in enumerate(track):
            q = p if p is not None else stage
            x0, y0 = q.placed_point(fb[0], fb[1])
            x1, y1 = q.placed_point(fb[2], fb[3])
            self.assertTrue(0 <= x0 and x1 <= CANVAS[0] + 0.5,
                            f"frame {i} face clipped horizontally")
            self.assertTrue(0 <= y0 and y1 <= CANVAS[1] + 0.5,
                            f"frame {i} face clipped vertically")

    def test_cutaway_fades_the_speaker_out(self):
        decls = [LayoutDecl(6.0, 8.0, "cutaway")]
        track = build_track(decls, [self.cut_plan], SRC_SHAPE, self.n,
                            self.fps, CANVAS)
        self.assertEqual(track[210].alpha_scale, 0.0)   # t=7.0
        placed = to_canvas_alpha(self.alphas[0], CANVAS,
                                 placement=track[210])
        self.assertEqual(float(placed.max()), 0.0)

    def test_punch_in_anchors_on_the_face(self):
        decl = LayoutDecl(2.0, 8.0, "spotlight",
                          camera={"type": "punch_in",
                                  "from": 1.0, "to": 1.3})
        track = build_track([decl], [self.spot], SRC_SHAPE, self.n,
                            self.fps, CANVAS)
        anchor = self.spot.placement.placed_point(
            *self.spot.centroid_src)
        start = track[66]                       # t=2.2, progress ~0
        mid = track[150]                        # t=5.0, progress 0.5
        self.assertAlmostEqual(start.scale, self.spot.placement.scale,
                               delta=0.01)
        self.assertGreater(mid.scale, start.scale)
        for p in (start, mid):
            ax, ay = p.placed_point(*self.spot.centroid_src)
            self.assertAlmostEqual(ax, anchor[0], places=3)
            self.assertAlmostEqual(ay, anchor[1], places=3)

    def test_drift_translates_the_frame(self):
        decl = LayoutDecl(2.0, 8.0, "spotlight",
                          camera={"type": "drift",
                                  "from": [0, 0], "to": [60, 0]})
        track = build_track([decl], [self.spot], SRC_SHAPE, self.n,
                            self.fps, CANVAS)
        # t=2.2: past the entry ease, drift progress ~0 (dx ~2px)
        self.assertTrue(89.0 <= track[66].ox <= 95.0)
        # t=7.5: drift progress 0.92 (dx ~55px), before the exit ease
        self.assertGreater(track[225].ox - track[66].ox, 40.0)

    def test_follow_track_moves_the_crop(self):
        n = self.n
        track_xy = [(320.0, 150.0 + i) for i in range(n)]
        decl = LayoutDecl(0.0, 10.0, "split", follow_track=True)
        track = build_track([decl], [self.split_plan], SRC_SHAPE, n,
                            self.fps, CANVAS, track=track_xy)
        self.assertIsNotNone(track[30])
        self.assertGreater(track[270].crop[1], track[30].crop[1])

    def test_smooth_trajectory_flattens_jitter(self):
        raw = [(100.0 + (10.0 if i % 2 else -10.0), 50.0)
               for i in range(31)]
        sm = smooth_trajectory(raw, 9)
        self.assertAlmostEqual(sm[15][0], 100.0, delta=3.0)
        self.assertEqual(len(sm), len(raw))


# -------------------------------------------------------- graphics schema


TRANSCRIPT = ("the cowboys and ceedee lamb are one million dollars "
              "apart per year and the crew on get up said it is "
              "criminal negligence honestly")


def span_payload(**overrides):
    span = {
        "start_s": 4.0, "end_s": 7.0, "kind": "card",
        "headline": "BIG NEWS", "facts": "",
        "prompt": "", "source_quote": "one million dollars apart",
    }
    span.update(overrides)
    return {"graphics": [span]}


class GraphicsSchemaTests(unittest.TestCase):
    def parse(self, payload):
        return parse_planner_reply(json.dumps(payload), TRANSCRIPT,
                                   duration_s=20.0)

    def test_layout_and_camera_round_trip(self):
        accepted, rejected = self.parse(span_payload(
            layout="spotlight",
            camera={"type": "punch_in", "from": 1.0, "to": 1.25}))
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0].layout, "spotlight")
        self.assertEqual(accepted[0].camera["to"], 1.25)
        d = accepted[0].to_dict()
        self.assertEqual(d["layout"], "spotlight")
        self.assertEqual(d["camera"]["type"], "punch_in")

    def test_stage_default_stays_out_of_the_dict(self):
        accepted, _ = self.parse(span_payload())
        self.assertEqual(accepted[0].layout, "stage")
        self.assertNotIn("layout", accepted[0].to_dict())
        self.assertNotIn("camera", accepted[0].to_dict())

    def test_unknown_layout_is_rejected(self):
        accepted, rejected = self.parse(span_payload(layout="orbit"))
        self.assertEqual(accepted, [])
        self.assertIn("layout", rejected[0]["reason"])

    def test_layout_on_a_hero_kind_is_rejected(self):
        accepted, rejected = self.parse(span_payload(
            kind="backdrop", headline="", prompt="a vast nebula field",
            layout="split"))
        self.assertEqual(accepted, [])
        self.assertIn("layout", rejected[0]["reason"])

    def test_unusable_camera_is_rejected(self):
        accepted, rejected = self.parse(span_payload(
            camera={"type": "orbit", "from": 0, "to": 1}))
        self.assertEqual(accepted, [])
        self.assertIn("camera", rejected[0]["reason"])

    def test_validate_camera_bounds(self):
        self.assertIsNone(validate_camera(
            {"type": "punch_in", "from": 0.2, "to": 1.2}))
        self.assertIsNone(validate_camera(
            {"type": "punch_in", "from": 1.0, "to": 2.5}))
        self.assertIsNotNone(validate_camera(
            {"type": "drift", "from": [0, 0], "to": [10, -5]}))


class GraphicsPlacementTests(unittest.TestCase):
    def place(self, layout):
        span = GraphicSpan(start_s=0.0, end_s=0.1, kind="card",
                           headline="BIG NEWS", facts="",
                           source_quote="q", layout=layout)
        alphas = [np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
                  for _ in range(4)]
        placed, dropped = place_prepared(
            [(span, prepare_card(span))], alphas, fps=30.0,
            caption_rect=CAPTION_RECT, canvas=CANVAS)
        self.assertEqual(dropped, [])
        self.assertEqual(len(placed), 1)
        return placed[0]

    def test_spotlight_spec_is_centred_and_declared(self):
        spec = self.place("spotlight")
        self.assertEqual(spec.layout, "spotlight")
        cx = (spec.box[0] + spec.box[2]) / 2
        self.assertAlmostEqual(cx, CANVAS[0] / 2, delta=2.0)
        self.assertEqual(spec.to_dict()["layout"], "spotlight")

    def test_cutaway_spec_spans_the_canvas_width(self):
        spec = self.place("cutaway")
        self.assertEqual(spec.box[2] - spec.box[0], CANVAS[0])

    def test_split_spec_stays_in_the_top_zone(self):
        spec = self.place("split")
        self.assertLessEqual(spec.box[3], round(CANVAS[1] * 0.65))


if __name__ == "__main__":
    unittest.main()
