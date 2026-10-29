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
    _band_overlap_px,
    _face_band,
    _face_top,
    _transform_alpha,
    CAPTION_BOTTOM_MARGIN,
    EFFECT_OCCLUSION_BAND,
    GLYPH_OCCLUSION_CAP,
    SAFE_X0,
    SAFE_X1,
    caption_block_height,
    caption_layer,
    caption_lines,
    effect_note,
    effect_status,
    face_pixels,
    frame_geometry,
    guarded_occupancy,
    hook_glyph_mask,
    initial_caption_rect,
    occupancy_map,
    plan_captions,
    plan_hook,
    plan_hook_effect,
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


class EffectBandTests(unittest.TestCase):
    """The depth-effect objective (refinement, 2026-10-29): the
    planner must AIM at 8-20% occlusion, not merely stay under the
    25% cap - the gate run measured 0.0 on every draft because
    nearest-to-preferred among the gate-safe candidates always
    picked clear space above the head."""

    def test_status_thresholds(self):
        lo, hi = EFFECT_OCCLUSION_BAND
        self.assertEqual(effect_status(0.0), "below_band")
        self.assertEqual(effect_status(lo), "in_band")
        self.assertEqual(effect_status(hi), "in_band")
        self.assertEqual(effect_status(0.24), "above_band")
        self.assertIn("EFFECT ABSENT", effect_note(0.0))
        self.assertIn("inside", effect_note(0.14))

    def test_planner_targets_the_band_when_reachable(self):
        # Brute-force the candidate grid: if any gate-safe candidate
        # lands in the band, the planner must return an in-band one.
        alphas = [speaker_alpha(1100, x0=300, x1=800)
                  for _ in range(4)]
        occ_map = guarded_occupancy(alphas)
        reachable = False
        for i in range(55):
            frac = round(0.14 + 0.01 * i, 2)
            glyph, bbox = hook_glyph_mask(
                "THE BAND NEVER REHEARSES", y_center_frac=frac)
            occ = predict_occlusion(glyph, occ_map)
            safe = (bbox[0] >= SAFE_X0 and bbox[2] <= SAFE_X1
                    and bbox[3] <= 1440 and bbox[1] >= 288)
            if safe and occ <= GLYPH_OCCLUSION_CAP \
                    and effect_status(occ) == "in_band":
                reachable = True
        self.assertTrue(reachable)  # fixture sanity: band is reachable
        plan = plan_hook("THE BAND NEVER REHEARSES", alphas)
        self.assertEqual(plan.effect_status, "in_band")
        self.assertGreaterEqual(plan.predicted_occlusion,
                                EFFECT_OCCLUSION_BAND[0])
        self.assertLessEqual(plan.predicted_occlusion,
                             EFFECT_OCCLUSION_BAND[1])
        self.assertTrue(plan.ok)

    def test_below_band_is_flagged_not_hidden(self):
        # Speaker hugs the bottom: no safe hook position touches it.
        alphas = [speaker_alpha(1750) for _ in range(2)]
        plan = plan_hook("THE BAND NEVER REHEARSES", alphas)
        self.assertEqual(plan.predicted_occlusion, 0.0)
        self.assertEqual(plan.effect_status, "below_band")
        self.assertTrue(plan.ok)  # gate-safe AND effect-absent

    def test_band_can_be_disabled(self):
        alphas = [speaker_alpha(1000) for _ in range(4)]
        plan = plan_hook("THE BAND NEVER REHEARSES", alphas,
                         fracs=[0.2, 0.3, 0.4], occlusion_band=None)
        self.assertEqual(plan.y_center_frac, 0.3)


class HookEffectPlanTests(unittest.TestCase):
    def _source_speaker(self, rows=(90, 360), cols=(180, 470),
                        shape=(360, 640)):
        a = np.zeros(shape, dtype=np.float64)
        a[rows[0]:rows[1], cols[0]:cols[1]] = 1.0
        return [a.copy() for _ in range(6)]

    def test_in_band_without_lift_on_holmes_like_geometry(self):
        plan = plan_hook_effect(
            "ARE COWBOYS SABOTAGING THEIR OWN SEASON?",
            self._source_speaker())
        self.assertTrue(plan.geometry_ok)
        self.assertEqual(plan.placement.effect_status, "in_band")
        self.assertEqual(plan.lift_px, 0)

    def test_unreachable_band_is_declared(self):
        # Speaker squeezed to the far right, mostly outside the
        # glyph span: no lift can manufacture overlap.
        plan = plan_hook_effect(
            "THE BAND NEVER REHEARSES",
            self._source_speaker(cols=(560, 630)),
            lift_steps=(0, 40))
        self.assertEqual(plan.placement.effect_status, "below_band")
        self.assertIn("no lift", plan.note)

    def test_transform_alpha(self):
        a = np.ones((10, 4))
        padded = _transform_alpha(a, 3)
        self.assertEqual(padded.shape, (13, 4))
        self.assertEqual(padded[10:].sum(), 0.0)
        cropped = _transform_alpha(a, -4)
        self.assertEqual(cropped.shape, (6, 4))
        np.testing.assert_array_equal(_transform_alpha(a, 0), a)


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


class FaceBandEquivalenceTests(unittest.TestCase):
    """plan_captions stores head-band rows instead of full-canvas
    face arrays (2026-10-29 OOM fix). The band helpers must reproduce
    the full-canvas measurements EXACTLY - the planner's decisions
    (and therefore F2's inputs) may not drift by a single pixel."""

    def test_band_overlap_and_top_match_full_canvas(self):
        rng = np.random.default_rng(7)
        rects = [
            (60, 1300, 980, 1600), (0, 0, 1080, 1920),
            (60, 1150, 980, 1290), (300, 400, 700, 900),
            (60, 1700, 980, 1920),
        ]
        for trial in range(6):
            top = int(rng.integers(300, 1500))
            bottom = min(1920, top + int(rng.integers(200, 700)))
            a = speaker_alpha(top, bottom,
                              x0=int(rng.integers(0, 400)),
                              x1=int(rng.integers(600, 1080)))
            if trial % 2:
                a = a * rng.random(a.shape)  # feathered alpha values
            full = face_pixels(a)
            band = _face_band(a)
            for r in rects:
                self.assertEqual(
                    _band_overlap_px(band, r),
                    int(full[r[1]:r[3], r[0]:r[2]].sum()))
            rows = np.nonzero(full.any(axis=1))[0]
            self.assertEqual(_face_top(band),
                             int(rows.min()) if rows.size else None)

    def test_empty_frame_band_is_none(self):
        self.assertIsNone(_face_band(np.zeros((1920, 1080))))
        self.assertEqual(_band_overlap_px(None, (0, 0, 1080, 1920)), 0)
        self.assertIsNone(_face_top(None))


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
