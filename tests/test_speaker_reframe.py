"""Tests for layout_law's speaker reframing (the F3 by-construction fix).

Synthetic source-space alphas; the load-bearing assertions measure
the PLACED geometry through the compositor's real transform
(``to_canvas_alpha`` + ``frame_geometry`` - the verifier's own
measurement path), not just the planner's prediction: prediction
and measurement must agree, because the verifier will use the
measurement.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.layout_law import (  # noqa: E402
    SAFE_Y0,
    SAFE_Y1,
    apply_speaker_reframe,
    frame_geometry,
    plan_speaker_reframe,
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


class ReframePlanTests(unittest.TestCase):
    def test_low_face_pads_and_lands_in_band(self):
        # Holmes-like: 640x360 desk framing, face low in the source.
        alphas = [block_alpha((360, 640), (60, 360), (200, 440))
                  for _ in range(6)]
        frames = [np.zeros((360, 640, 3), dtype=np.uint8)
                  for _ in range(6)]
        _x, cy0, _t = placed_stats(alphas)
        self.assertGreater(cy0, SAFE_Y1)  # starts below the band
        plan = plan_speaker_reframe(alphas, CANVAS)
        self.assertFalse(plan.identity)
        self.assertGreater(plan.pad_bottom_px, 0)
        self.assertTrue(plan.feasible, plan.note)
        _nf, na = apply_speaker_reframe(frames, alphas,
                                        plan.pad_bottom_px)
        _x, cy, top_min = placed_stats(na)
        self.assertGreaterEqual(cy, SAFE_Y0)
        self.assertLessEqual(cy, SAFE_Y1)
        self.assertGreaterEqual(top_min, 90)  # F4's floor, measured
        # Prediction vs the verifier's measurement path: agree.
        self.assertAlmostEqual(cy, plan.placed_centroid[1], delta=3.0)
        self.assertAlmostEqual(top_min, plan.placed_head_top_min,
                               delta=3.0)

    def test_face_already_in_band_is_identity(self):
        # 4:3 source: the placed strip is tall, the face lands
        # mid-band with no transform.
        alphas = [block_alpha((360, 480), (30, 360), (140, 340))
                  for _ in range(4)]
        _x, cy0, _t = placed_stats(alphas)
        self.assertGreaterEqual(cy0, SAFE_Y0 + 60)
        self.assertLessEqual(cy0, SAFE_Y1 - 60)
        plan = plan_speaker_reframe(alphas, CANVAS)
        self.assertTrue(plan.identity)
        self.assertEqual(plan.pad_bottom_px, 0)
        self.assertTrue(plan.feasible)

    def test_high_face_crops_down_into_band(self):
        # Portrait source, speaker at the very top: the centroid
        # starts ABOVE the band and a bottom crop lowers it.
        alphas = [block_alpha((160, 90), (4, 60), (20, 70))
                  for _ in range(4)]
        frames = [np.zeros((160, 90, 3), dtype=np.uint8)
                  for _ in range(4)]
        _x, cy0, _t = placed_stats(alphas)
        self.assertLess(cy0, SAFE_Y0)
        plan = plan_speaker_reframe(alphas, CANVAS)
        self.assertLess(plan.pad_bottom_px, 0)
        self.assertTrue(plan.feasible, plan.note)
        _nf, na = apply_speaker_reframe(frames, alphas,
                                        plan.pad_bottom_px)
        _x, cy, top_min = placed_stats(na)
        self.assertGreaterEqual(cy, SAFE_Y0)
        self.assertLessEqual(cy, SAFE_Y1)
        self.assertGreaterEqual(top_min, 90)

    def test_pad_ceiling_clamp_reports_infeasible(self):
        # Absurd ultrawide source: the lift the centroid needs runs
        # past the 4x pad ceiling - the plan must clamp and SAY so.
        alphas = [block_alpha((360, 3840), (0, 360), (1600, 2240))
                  for _ in range(3)]
        plan = plan_speaker_reframe(alphas, CANVAS)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.pad_bottom_px, 3 * 360)
        self.assertGreater(plan.placed_centroid[1], SAFE_Y1)
        self.assertIn("INFEASIBLE", plan.note)

    def test_no_speaker_is_identity_and_infeasible(self):
        plan = plan_speaker_reframe(
            [np.zeros((90, 160)) for _ in range(3)], CANVAS)
        self.assertTrue(plan.identity)
        self.assertFalse(plan.feasible)


class ReframeApplyTests(unittest.TestCase):
    def test_pad_zero_alpha_and_edge_rgb(self):
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        alpha = np.ones((2, 3))
        nf, na = apply_speaker_reframe([frame], [alpha], 2)
        self.assertEqual(nf[0].shape, (4, 3, 3))
        self.assertEqual(na[0].shape, (4, 3))
        self.assertTrue(np.all(na[0][2:] == 0.0))       # alpha: zero
        self.assertTrue(np.all(nf[0][2:] == frame[-1]))  # RGB: edge

    def test_crop_slices_bottom(self):
        frame = np.zeros((6, 3, 3), dtype=np.uint8)
        alpha = np.ones((6, 3))
        nf, na = apply_speaker_reframe([frame], [alpha], -2)
        self.assertEqual(nf[0].shape[0], 4)
        self.assertEqual(na[0].shape[0], 4)

    def test_crop_releases_original_buffers(self):
        # The in-place transform must COPY crop slices: a view would
        # pin the untransformed buffer (and its memory) for the rest
        # of the run - the 2026-10-29 OOM fix depends on the release.
        frame = np.arange(6 * 3 * 3, dtype=np.uint8).reshape(6, 3, 3)
        alpha = np.ones((6, 3))
        nf, na = apply_speaker_reframe([frame], [alpha], -2)
        self.assertFalse(np.shares_memory(nf[0], frame))
        self.assertFalse(np.shares_memory(na[0], alpha))
        self.assertTrue(np.array_equal(nf[0], frame[:4]))
        self.assertTrue(np.array_equal(na[0], alpha[:4]))

    def test_pad_values_match_np_pad_reference(self):
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        alpha = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
        nf, na = apply_speaker_reframe(
            [frame.copy()], [alpha.copy()], 3)
        self.assertTrue(np.array_equal(
            nf[0], np.pad(frame, ((0, 3), (0, 0), (0, 0)), mode="edge")))
        self.assertTrue(np.array_equal(
            na[0], np.pad(alpha, ((0, 3), (0, 0)), mode="constant")))

    def test_zero_pad_is_identity_objects(self):
        frame = np.zeros((6, 3, 3), dtype=np.uint8)
        alpha = np.ones((6, 3))
        nf, na = apply_speaker_reframe([frame], [alpha], 0)
        self.assertIs(nf[0], frame)
        self.assertIs(na[0], alpha)


if __name__ == "__main__":
    unittest.main()
