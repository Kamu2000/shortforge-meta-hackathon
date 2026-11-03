"""Fix round 6 (matte round): mask-sequence smoothing and
mask cleanup (hole fill).

Kamlesh's fix-5 QM6 verdict left two open items - flickering
body edges and mask quality. Round 5's sweep proved the
finish-side blend family caps at -17.2% worst-2s flicker;
round 6 smooths the SAM masks themselves (centred, IoU-
guarded, edge-replicated) before the finish and fills
interior mask holes. (A third candidate - guided alpha
upsampling at the compositor's canvas scale - was measured
and REJECTED: every parameter set widened the canvas edge
band and lowered its gradient vs plain bilinear, on the
banked windows and on a synthetic full-res-guide test; see
the round-6 BUILDLOG entry.) These tests pin the shipped
primitives' contracts; the adopt/reject measurement lives
in the round-6 BUILDLOG entry (offline banked-window proof).
"""
import unittest

import numpy as np

from spatial_director.matte_finish import (
    cleanup_speaker_mask,
    fill_mask_holes,
)
from spatial_director.temporal_smooth import smooth_mask_sequence


def _mask(shape=(20, 20), box=(5, 5, 15, 15)):
    m = np.zeros(shape, dtype=bool)
    m[box[0]:box[2], box[1]:box[3]] = True
    return m


# ------------------------------------------------ mask smoothing


class SmoothMaskSequenceTests(unittest.TestCase):

    def test_smooth_radius_zero_is_identity(self):
        masks = [_mask(), _mask(box=(6, 5, 16, 15)), _mask()]
        out = list(smooth_mask_sequence(masks, radius=0))
        self.assertEqual(len(out), 3)
        for got, want in zip(out, masks):
            self.assertTrue(np.allclose(got, want.astype(np.float64)))

    def test_smooth_constant_sequence_unchanged(self):
        masks = [_mask() for _ in range(7)]
        out = list(smooth_mask_sequence(masks, radius=2))
        self.assertEqual(len(out), 7)
        for got in out:
            self.assertTrue(np.allclose(got, masks[0].astype(np.float64)))

    def test_smooth_attenuates_one_frame_spike(self):
        base = _mask()
        spike = base.copy()
        spike[2, 2] = True  # one-frame island pixel
        masks = [base, base, spike, base, base]
        out = list(smooth_mask_sequence(masks, radius=2))
        # The spike enters its own frame at the centre weight
        # (3/9), not at full strength, and leaks only weakly
        # into the neighbouring frames. Frame 0 sees the spike
        # at distance 2 (weight 1) with its window completed
        # by edge replication - full denominator 9, and its
        # own silhouette stays at full alpha (no ghosting).
        self.assertAlmostEqual(out[2][2, 2], 3.0 / 9.0)
        self.assertGreater(out[1][2, 2], 0.0)
        self.assertLess(out[1][2, 2], 3.0 / 9.0)
        self.assertAlmostEqual(out[0][2, 2], 1.0 / 9.0)
        self.assertAlmostEqual(out[0][10, 10], 1.0)

    def test_smooth_guard_drops_a_different_shot(self):
        centre = _mask()
        other = _mask(box=(5, 12, 15, 19))  # no overlap: IoU 0
        masks = [other, centre, other]
        out = list(smooth_mask_sequence(masks, radius=1))
        # Both neighbours fail the IoU guard, so the centre
        # frame is returned at full weight, not blended
        # toward the cut.
        self.assertTrue(np.allclose(out[1], centre.astype(np.float64)))

    def test_smooth_accepts_a_generator_and_preserves_length(self):
        masks = (_mask() for _ in range(5))
        out = list(smooth_mask_sequence(masks, radius=2))
        self.assertEqual(len(out), 5)
        self.assertTrue(all(o.shape == (20, 20) for o in out))
        self.assertTrue(
            all(0.0 <= o.min() and o.max() <= 1.0 for o in out))

    def test_smooth_rejects_negative_radius(self):
        with self.assertRaises(ValueError):
            list(smooth_mask_sequence([_mask()], radius=-1))


# ------------------------------------------------ mask cleanup


class MaskCleanupTests(unittest.TestCase):

    def test_fill_mask_holes_closes_an_interior_hole(self):
        m = _mask()
        m[9:11, 9:11] = False  # enclosed hole
        filled = fill_mask_holes(m)
        self.assertTrue(filled[9:11, 9:11].all())
        self.assertEqual(filled.sum(), m.sum() + 4)

    def test_fill_mask_holes_leaves_open_shapes_and_empty(self):
        m = _mask()
        m[5, 9] = False  # notch open to the boundary, not a hole
        self.assertFalse(fill_mask_holes(m)[5, 9])
        empty = np.zeros((8, 8), dtype=bool)
        self.assertFalse(fill_mask_holes(empty).any())

    def test_cleanup_drops_island_and_fills_hole(self):
        m = _mask()
        m[9:11, 9:11] = False
        m[0, 0] = True  # detached island
        out = cleanup_speaker_mask(m)
        self.assertFalse(out[0, 0])
        self.assertTrue(out[9:11, 9:11].all())


if __name__ == "__main__":
    unittest.main()
