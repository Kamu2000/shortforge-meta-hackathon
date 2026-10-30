"""Tests for layout_law's speaker-forward emphasis (block 2).

Synthetic source-space alphas; the load-bearing assertions
measure the PLACED geometry through the compositor's real
transform (``to_canvas_alpha`` + ``frame_geometry`` - the
verifier's own measurement path): the emphasis must land the
face centroid in the F3 band with F4 headroom to spare, because
F3/F4 verify the render as-is with untouched thresholds.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.layout_law import (  # noqa: E402
    SAFE_X0,
    SAFE_X1,
    SAFE_Y0,
    SAFE_Y1,
    apply_speaker_emphasis,
    frame_geometry,
    plan_speaker_emphasis,
    to_canvas_alpha,
)

CANVAS = (1080, 1920)


def block_alpha(shape, rows, cols):
    a = np.zeros(shape, dtype=np.float64)
    a[rows[0]:rows[1], cols[0]:cols[1]] = 1.0
    return a


def placed_stats(src_alphas):
    cents, tops = [], []
    for a in src_alphas:
        geo = frame_geometry(to_canvas_alpha(a, CANVAS))
        if geo.face_centroid is not None:
            cents.append(geo.face_centroid)
            tops.append(geo.head_top)
    return (float(np.median([c[0] for c in cents])),
            float(np.median([c[1] for c in cents])),
            min(tops))


class EmphasisPlanTests(unittest.TestCase):
    def test_low_right_speaker_emphasis_lands_in_band(self):
        # Holmes-like desk framing: speaker right and low; the
        # placed centroid starts below the F3 band.
        alphas = [block_alpha((360, 640), (120, 360), (300, 560))
                  for _ in range(6)]
        _cx0, cy0, _t0 = placed_stats(alphas)
        self.assertGreater(cy0, SAFE_Y1)
        plan = plan_speaker_emphasis(alphas, CANVAS)
        self.assertTrue(plan.feasible, msg=plan.note)
        self.assertAlmostEqual(plan.scale, 1.18, places=2)
        self.assertGreaterEqual(plan.placed_head_top_min, 150)
        # Apply at full strength and measure through the real
        # placement path: the measured centroid must agree with
        # the PLANNED one (the crop window clamps for a far-right
        # speaker, so the plan promises in-band and toward-centre,
        # not exact centring), inside the band on both axes, with
        # the head top above the F4 floor with margin.
        frames = [np.zeros((360, 640, 3), dtype=np.uint8)
                  for _ in range(6)]
        frames, warped = apply_speaker_emphasis(
            frames, alphas, plan, 0, 6, ease_frames=1)
        cx, cy, top = placed_stats(warped[1:5])
        self.assertTrue(SAFE_X0 <= cx <= SAFE_X1)
        self.assertTrue(SAFE_Y0 <= cy <= SAFE_Y1)
        self.assertLess(abs(cx - plan.placed_centroid[0]), 40)
        self.assertLess(abs(cy - plan.placed_centroid[1]), 40)
        self.assertLess(cx, _cx0)  # moved toward the centre
        self.assertGreaterEqual(top, 90)

    def test_close_up_speaker_is_infeasible(self):
        # The face nearly fills the frame width: any zoom >= 1.05
        # would clip it, so the span falls back to standard
        # framing with the reason recorded.
        alphas = [block_alpha((360, 640), (40, 360), (10, 630))
                  for _ in range(6)]
        plan = plan_speaker_emphasis(alphas, CANVAS)
        self.assertFalse(plan.feasible)
        self.assertIn("clip", plan.note)
        self.assertEqual(plan.scale, 1.0)

    def test_no_speaker_is_infeasible(self):
        alphas = [np.zeros((360, 640), dtype=np.float64)
                  for _ in range(3)]
        plan = plan_speaker_emphasis(alphas, CANVAS)
        self.assertFalse(plan.feasible)


class EmphasisApplyTests(unittest.TestCase):
    def test_boundaries_ease_and_outside_frames_untouched(self):
        alphas = [block_alpha((360, 640), (120, 360), (300, 560))
                  for _ in range(24)]
        frames = [np.full((360, 640, 3), 40, dtype=np.uint8)
                  for _ in range(24)]
        plan = plan_speaker_emphasis(alphas[4:20], CANVAS)
        self.assertTrue(plan.feasible, msg=plan.note)
        out_frames, out_alphas = apply_speaker_emphasis(
            frames, alphas, plan, 4, 20)
        self.assertEqual(len(out_frames), 24)
        # Outside the span: the very same objects (shared).
        self.assertIs(out_alphas[0], alphas[0])
        self.assertIs(out_alphas[23], alphas[23])
        # First span frame eases from identity: also untouched.
        self.assertIs(out_alphas[4], alphas[4])
        # Mid-span: fully warped - measurably different geometry.
        self.assertIsNot(out_alphas[12], alphas[12])
        cx_mid = frame_geometry(
            to_canvas_alpha(out_alphas[12], CANVAS)).face_centroid
        cx_raw = frame_geometry(
            to_canvas_alpha(alphas[12], CANVAS)).face_centroid
        self.assertLess(cx_mid[1], cx_raw[1] - 100)  # lifted a lot
        self.assertLess(abs(cx_mid[0] - 540), abs(cx_raw[0] - 540))


if __name__ == "__main__":
    unittest.main()
