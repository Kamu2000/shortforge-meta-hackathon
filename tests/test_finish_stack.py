"""Offline tests for matte_finish + temporal_smooth (synthetic masks)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.matte_finish import refine_matte  # noqa: E402
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
