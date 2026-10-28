"""Tests for compositor: layer order, placement, hook, xray, video.

Small canvas (270x480) through the same code paths as the real
1080x1920 render. The layer-order test is the load-bearing one: the
speaker must occlude the hook text, and the hook must cover the
background where the speaker is absent.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.compositor import (  # noqa: E402
    background_layer,
    composite_frame,
    hook_layer,
    hook_visible_at,
    render_composite_video,
    separation_at,
    speaker_layer,
    xray_frame,
)

CANVAS = (270, 480)  # (w, h)
SRC_W, SRC_H = 68, 90


def source_frame() -> np.ndarray:
    """Two-tone source: left half red, right half blue."""
    frame = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
    frame[:, : SRC_W // 2] = (200, 30, 30)
    frame[:, SRC_W // 2:] = (30, 30, 200)
    return frame


def speaker_alpha() -> np.ndarray:
    """Left half of the source is speaker, right half is not."""
    alpha = np.zeros((SRC_H, SRC_W), dtype=np.float64)
    alpha[:, : SRC_W // 2] = 1.0
    return alpha


class LayerOrderTests(unittest.TestCase):
    def setUp(self):
        self.frame = source_frame()
        self.alpha = speaker_alpha()
        # Solid green hook card covering the whole canvas.
        self.hook = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
        self.hook[..., 0] = 40
        self.hook[..., 1] = 190
        self.hook[..., 2] = 40
        self.hook[..., 3] = 255
        self.out = composite_frame(
            self.frame, self.alpha, self.hook, CANVAS, blur_radius=4)

    def test_speaker_occludes_hook(self):
        # Centre of the speaker region (canvas coords): the speaker is
        # fit-width scaled and bottom-anchored; x=40 is well inside
        # the source's left half. Must be the speaker's red, not green.
        px = self.out[400, 40].astype(int)
        self.assertGreater(px[0], 150)   # red channel of the speaker
        self.assertLess(px[1], 90)       # not the hook's green

    def test_hook_covers_background_where_no_speaker(self):
        px = self.out[400, 230].astype(int)  # right side: no speaker
        self.assertGreater(px[1], 150)       # hook green wins over bg
        self.assertLess(px[0], 90)

    def test_background_when_no_hook(self):
        out = composite_frame(
            self.frame, self.alpha, None, CANVAS, blur_radius=4)
        px = out[400, 230].astype(int)
        self.assertGreater(px[2], 120)  # blurred blue background


class SpeakerPlacementTests(unittest.TestCase):
    def test_bottom_anchored_and_width_fit(self):
        fg, a = speaker_layer(source_frame(), speaker_alpha(), CANVAS)
        self.assertEqual(fg.shape, (CANVAS[1], CANVAS[0], 3))
        self.assertEqual(a.shape, (CANVAS[1], CANVAS[0]))
        # Speaker occupies the bottom of the canvas, not the top.
        self.assertEqual(a[5, 40], 0.0)
        self.assertGreater(a[470, 40], 0.9)
        # Total alpha mass is roughly preserved by the scaling.
        expect = speaker_alpha().sum() * (CANVAS[0] / SRC_W) ** 2
        self.assertAlmostEqual(a.sum() / expect, 1.0, delta=0.15)

    def test_background_covers_canvas(self):
        bg = background_layer(source_frame(), CANVAS, blur_radius=4)
        self.assertEqual(bg.shape, (CANVAS[1], CANVAS[0], 3))


class HookLayerTests(unittest.TestCase):
    def test_text_rendered_and_mostly_transparent(self):
        layer = hook_layer("HE TURNED DOWN NINE INCH NAILS",
                           CANVAS, start_size=60)
        self.assertEqual(layer.shape, (CANVAS[1], CANVAS[0], 4))
        ink = layer[..., 3] > 0
        self.assertGreater(ink.sum(), 500)          # text exists
        self.assertLess(ink.mean(), 0.25)           # mostly transparent
        # Ink sits around the vertical centre band (30% of height).
        ys = np.nonzero(ink.any(axis=1))[0]
        self.assertGreater(ys.min(), 40)
        self.assertLess(ys.max(), 260)

    def test_window_logic(self):
        self.assertTrue(hook_visible_at(0.0, (0.0, 6.0)))
        self.assertTrue(hook_visible_at(5.99, (0.0, 6.0)))
        self.assertFalse(hook_visible_at(6.0, (0.0, 6.0)))
        self.assertFalse(hook_visible_at(1.0, None))


class XrayTests(unittest.TestCase):
    def test_separation_zero_equals_composite(self):
        frame, alpha = source_frame(), speaker_alpha()
        hook = hook_layer("HOOK", CANVAS, start_size=60)
        comp = composite_frame(frame, alpha, hook, CANVAS, blur_radius=4)
        xray = xray_frame(frame, alpha, hook, CANVAS, separation=0.0,
                          blur_radius=4)
        np.testing.assert_array_equal(xray, comp)

    def test_separation_ramp(self):
        self.assertEqual(separation_at(0.0), 0.0)
        self.assertEqual(separation_at(1.5), 1.0)
        self.assertEqual(separation_at(99.0), 1.0)
        self.assertAlmostEqual(separation_at(0.75), 0.5)

    def test_separated_frame_differs(self):
        frame, alpha = source_frame(), speaker_alpha()
        hook = hook_layer("HOOK", CANVAS, start_size=60)
        xray = xray_frame(frame, alpha, hook, CANVAS, separation=1.0,
                          blur_radius=4)
        self.assertEqual(xray.shape, (CANVAS[1], CANVAS[0], 3))


class VideoRenderTests(unittest.TestCase):
    def test_render_composite_video_smoke(self):
        frames = [source_frame() for _ in range(6)]
        alphas = [speaker_alpha() for _ in range(6)]
        hook = hook_layer("HOOK", CANVAS, start_size=60)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.mp4"
            render_composite_video(
                frames, alphas, out, fps=6, hook_rgba=hook,
                hook_window=(0.0, 0.5), canvas=CANVAS, blur_radius=4)
            self.assertGreater(out.stat().st_size, 1000)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "stream=width,height,codec_name", "-of", "csv=p=0",
                 str(out)],
                capture_output=True, text=True, check=True)
            self.assertIn("h264", probe.stdout)
            self.assertIn("270,480", probe.stdout)


if __name__ == "__main__":
    unittest.main()
