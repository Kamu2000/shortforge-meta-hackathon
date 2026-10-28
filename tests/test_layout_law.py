"""Tests for layout_law: geometry, hook planning, caption planning.

Synthetic canvas-space alphas at the real 1080x1920 canvas: a
"speaker" is a filled rectangle (shoulders) whose top band is the
head. The load-bearing tests: the hook planner must MOVE the band
when the preferred position is occluded, and the caption planner
must keep face pixels out of the caption block - or say so.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.layout_law import (  # noqa: E402
    CAPTION_BOTTOM_MARGIN,
    GLYPH_OCCLUSION_CAP,
    SAFE_X0,
    SAFE_X1,
    caption_block_height,
    caption_layer,
    caption_lines,
    face_pixels,
    frame_geometry,
    hook_glyph_mask,
    initial_caption_rect,
    occupancy_map,
    plan_captions,
    plan_hook,
    predict_occlusion,
    to_canvas_alpha,
)

CANVAS = (1080, 1920)


def speaker_alpha(top: int, bottom: int = 1920, x0: int = 240,
                  x1: int = 840) -> np.ndarray:
    """A rectangular 'speaker' on the canvas: alpha 1 inside."""
    a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float64)
    a[top:bottom, x0:x1] = 1.0
    return a


class GeometryTests(unittest.TestCase):
    def test_empty_frame(self):
        geo = frame_geometry(np.zeros((1920, 1080)))
        self.assertFalse(geo.present)
        self.assertIsNone(geo.head_top)

    def test_speaker_geometry(self):
        geo = frame_geometry(speaker_alpha(500))
        self.assertTrue(geo.present)
        self.assertEqual(geo.head_top, 500)
        self.assertEqual(geo.mask_bbox, (240, 500, 840, 1920))
        # Head band = upper 32% of 1420px -> face box ends ~954.
        self.assertEqual(geo.face_box[1], 500)
        self.assertLess(geo.face_box[3], 1000)
        self.assertAlmostEqual(geo.head_cx, 539.5, delta=1.0)
        self.assertAlmostEqual(geo.face_centroid[1], 727, delta=30)

    def test_face_pixels_only_in_band(self):
        a = speaker_alpha(500)
        face = face_pixels(a)
        self.assertTrue(face[600, 500])
        self.assertFalse(face[1500, 500])  # shoulders are not face

    def test_to_canvas_alpha_bottom_anchored(self):
        src = np.zeros((360, 272), dtype=np.float64)
        src[100:, :] = 1.0
        canvas_a = to_canvas_alpha(src)
        self.assertEqual(canvas_a.shape, (1920, 1080))
        self.assertEqual(canvas_a[10, 500], 0.0)      # top stays empty
        self.assertGreater(canvas_a[1900, 500], 0.9)  # anchored bottom


class HookPlanningTests(unittest.TestCase):
    def test_free_band_stays_at_preferred(self):
        # Speaker low (head top at 1000): the preferred band is free.
        alphas = [speaker_alpha(1000) for _ in range(4)]
        plan = plan_hook("THE BAND NEVER REHEARSES", alphas,
                         fracs=[0.2, 0.3, 0.4])
        self.assertTrue(plan.ok)
        self.assertEqual(plan.y_center_frac, 0.3)
        self.assertLessEqual(plan.predicted_occlusion,
                             GLYPH_OCCLUSION_CAP)

    def test_occluded_preferred_band_moves(self):
        # Speaker high: the preferred 0.30 band is inside the speaker.
        alphas = [speaker_alpha(520) for _ in range(4)]
        occ_map = occupancy_map(alphas)
        glyph, _ = hook_glyph_mask("THE BAND NEVER REHEARSES",
                                   y_center_frac=0.30)
        occ_at_preferred = predict_occlusion(glyph, occ_map)
        plan = plan_hook("THE BAND NEVER REHEARSES", alphas,
                         fracs=[0.16, 0.2, 0.3, 0.44, 0.5])
        self.assertGreater(occ_at_preferred, GLYPH_OCCLUSION_CAP)
        self.assertLess(plan.predicted_occlusion, occ_at_preferred)
        self.assertTrue(plan.ok)
        self.assertNotEqual(plan.y_center_frac, 0.3)

    def test_full_occlusion_reports_not_ok(self):
        alphas = [np.ones((1920, 1080)) for _ in range(2)]
        plan = plan_hook("HOOK", alphas, fracs=[0.3])
        self.assertFalse(plan.ok)
        self.assertAlmostEqual(plan.predicted_occlusion, 1.0)

    def test_bbox_within_safe_zone_when_ok(self):
        alphas = [speaker_alpha(1000) for _ in range(2)]
        plan = plan_hook("THE BAND NEVER REHEARSES", alphas,
                         fracs=[0.3])
        x0, y0, x1, y1 = plan.bbox
        self.assertGreaterEqual(x0, SAFE_X0)
        self.assertLessEqual(x1, SAFE_X1)


class CaptionPlanningTests(unittest.TestCase):
    def test_initial_rect_anchor(self):
        rect = initial_caption_rect()
        self.assertEqual(rect[3], 1920 - CAPTION_BOTTOM_MARGIN)
        self.assertEqual(rect[0], SAFE_X0)
        self.assertEqual(rect[1], rect[3] - caption_block_height())

    def test_no_shift_when_face_is_high(self):
        alphas = [speaker_alpha(420) for _ in range(24)]
        layout = plan_captions([(0.0, 1.0)], alphas, fps=24)
        self.assertTrue(layout.feasible)
        self.assertEqual(layout.rect, initial_caption_rect())
        self.assertEqual(layout.per_cue_overlap_px[0], 0)

    def test_low_face_shifts_block_up(self):
        # Leaning speaker: head band reaches into the default block.
        alphas = [speaker_alpha(1300) for _ in range(24)]
        layout = plan_captions([(0.0, 1.0)], alphas, fps=24)
        self.assertTrue(layout.feasible)
        self.assertEqual(layout.per_cue_overlap_px[0], 0)
        self.assertLess(layout.rect[3], initial_caption_rect()[3])

    def test_unavoidable_overlap_is_reported_not_hidden(self):
        # Face top at 1270: clearing it needs the block bottom at
        # 1270, i.e. its top at 1130 - above CAPTION_MIN_TOP (1150).
        # The planner must keep the least overlap and say the layout
        # is infeasible (the verifier fails F2 on it).
        alphas = [speaker_alpha(1270) for _ in range(24)]
        layout = plan_captions([(0.0, 1.0)], alphas, fps=24)
        self.assertFalse(layout.feasible)
        self.assertGreater(layout.per_cue_overlap_px[0], 0)


class CaptionRenderTests(unittest.TestCase):
    def test_lines_wrap_to_two(self):
        lines = caption_lines("the interesting thing about it is an "
                              "eleven piece band")
        self.assertGreaterEqual(len(lines), 1)
        self.assertLessEqual(len(lines), 2)
        self.assertEqual(" ".join(lines).split(),
                         "the interesting thing about it is an eleven "
                         "piece band".split())

    def test_layer_ink_inside_rect(self):
        rect = initial_caption_rect()
        layer = caption_layer("hello world this is a caption", rect)
        ink = layer[..., 3] > 0
        self.assertGreater(ink.sum(), 200)
        ys = np.nonzero(ink.any(axis=1))[0]
        self.assertGreaterEqual(ys.min(), rect[1] - 2)
        self.assertLessEqual(ys.max(), rect[3] + 2)


if __name__ == "__main__":
    unittest.main()
