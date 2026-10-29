"""Offline tests for matte_finish + temporal_smooth (synthetic masks)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.matte_finish import (  # noqa: E402
    largest_connected_component,
    refine_matte,
)
from spatial_director.temporal_smooth import (  # noqa: E402
    ema_smooth,
    mask_iou,
    mean_frame_delta,
)


def disc(shape=(64, 64), centre=(32, 32), r=14):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= r * r).astype(float)


class MatteFinishTests(unittest.TestCase):
    def test_choke_shrinks_and_feather_softens(self):
        mask = disc()
        guide = mask.copy()  # guide aligned with the mask edge
        alpha = refine_matte(mask, guide, choke_px=2, feather_r=2)
        self.assertLess(alpha.sum(), mask.sum())  # choked
        vals = alpha[(alpha > 0.01) & (alpha < 0.99)]
        self.assertGreater(len(vals), 0)  # soft transition exists
        self.assertGreaterEqual(alpha.min(), 0.0)
        self.assertLessEqual(alpha.max(), 1.0)

    def test_core_survives(self):
        mask = disc()
        alpha = refine_matte(mask, mask.copy())
        self.assertGreater(alpha[32, 32], 0.95)  # centre stays solid


class LargestComponentTests(unittest.TestCase):
    def test_detached_island_is_dropped(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True      # the body
        mask[2:5, 34:38] = True        # a detached island
        out = largest_connected_component(mask)
        self.assertEqual(int(out.sum()), 400)
        self.assertFalse(out[3, 35])

    def test_single_component_unchanged(self):
        mask = disc() > 0.5
        np.testing.assert_array_equal(
            largest_connected_component(mask), mask)

    def test_empty_mask_unchanged(self):
        mask = np.zeros((10, 10), dtype=bool)
        self.assertFalse(largest_connected_component(mask).any())

    def test_diagonal_touch_does_not_merge(self):
        # 4-connectivity: a corner-touching island is still an island.
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:10, 2:10] = True
        mask[10, 10] = True  # touches the body only at a corner
        out = largest_connected_component(mask)
        self.assertFalse(out[10, 10])

    def test_tighter_defaults_bleed_less(self):
        # The 2026-10-29 retune (choke 3 / feather 1 / gf r4 eps 1e-4)
        # must carry less alpha outside the true mask than the old
        # defaults did - that outside mass is the visible ghosting.
        mask = disc()
        rng = np.random.default_rng(3)
        guide = np.clip(mask + rng.normal(0, 0.05, mask.shape), 0, 1)
        new = refine_matte(mask, guide)
        old = refine_matte(mask, guide, choke_px=2, feather_r=2,
                           gf_radius=8, gf_eps=1e-3)
        outside = mask <= 0.5
        self.assertLess(new[outside].sum(), old[outside].sum())


class TemporalSmoothTests(unittest.TestCase):
    def _jittery_sequence(self, n=12):
        frames = []
        for i in range(n):
            cx = 24 + i + (1 if i % 2 else -1)  # steady move + 1px jitter
            frames.append(disc(centre=(cx, 32), r=12))
        return frames

    def test_smoothing_reduces_jitter(self):
        frames = self._jittery_sequence()
        smoothed = ema_smooth(frames)
        self.assertLess(mean_frame_delta(smoothed), mean_frame_delta(frames))

    def test_cut_resets_state(self):
        a = disc(centre=(16, 32), r=10)
        b = disc(centre=(48, 32), r=10)  # disjoint: a hard move/cut
        out = ema_smooth([a, a, b])
        self.assertAlmostEqual(mask_iou(out[2], b), 1.0, places=6)

    def test_empty_sequence(self):
        self.assertEqual(ema_smooth([]), [])
        self.assertEqual(mean_frame_delta([]), 0.0)


if __name__ == "__main__":
    unittest.main()
