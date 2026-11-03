"""Fix round 3 (2026-11-02): studio_procedural canvas backdrop.

Kamlesh's verdict on the qh4 live output: the source-minus-
speaker diffusion fill reads as a grey ghost/shadow of the
speaker. The new DEFAULT backdrop therefore uses no source
pixels at all - a code-rendered studio canvas (brand near-
black gradient, Forge Blue / Viral Pink glows, seeded grain,
vignette, Ken Burns drift) plus a grounding shadow derived
from the PLACED mask's bottom rows. These tests pin the
contract: determinism per seed, the shadow follows the mask,
the pipeline path defaults to studio_procedural, and the
surviving modes (source_minus_speaker above all) stay
selectable through the same wiring. F1-F9 untouched.
"""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from spatial_director import compositor, pipeline
from spatial_director.compositor import (
    BACKDROP_SOURCE_MINUS_SPEAKER, BACKDROP_STUDIO, DEFAULT_BACKDROP,
    composite_frame, make_backdrop, studio_backdrop, studio_seed)
from spatial_director.rescue_loop import RenderArtifacts, _render_execution
from spatial_director.caption_align import CaptionCue
from spatial_director.verifier import HookSpec

CANVAS = (270, 480)  # small canvas, same code paths


def _u8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


class StudioDeterminismTests(unittest.TestCase):
    def test_same_seed_byte_identical(self):
        a = studio_backdrop(CANVAS, seed=1234, progress=0.35)
        b = studio_backdrop(CANVAS, seed=1234, progress=0.35)
        self.assertTrue(np.array_equal(_u8(a), _u8(b)))

    def test_different_seed_differs(self):
        a = studio_backdrop(CANVAS, seed=1)
        b = studio_backdrop(CANVAS, seed=2)
        self.assertFalse(np.array_equal(_u8(a), _u8(b)))

    def test_seed_derivation_is_stable_and_identity_bound(self):
        ident = "seg-holmes-w3.mp4:12.026-26.026:short-1"
        self.assertEqual(studio_seed(ident), studio_seed(ident))
        self.assertNotEqual(
            studio_seed(ident),
            studio_seed("seg-holmes-w3.mp4:12.026-26.026:short-2"))
        self.assertNotEqual(
            studio_seed(ident),
            studio_seed("seg-mcarthur-hamburger.mp4:16.030-30.030"
                        ":short-1"))

    def test_drift_moves_the_canvas(self):
        a = studio_backdrop(CANVAS, seed=7, progress=0.0)
        b = studio_backdrop(CANVAS, seed=7, progress=1.0)
        self.assertFalse(np.array_equal(_u8(a), _u8(b)))

    def test_canvas_is_dark_and_brand_bounded(self):
        a = studio_backdrop(CANVAS, seed=7)
        # A lit studio, not wallpaper: dark everywhere, and even
        # the glow centres stay far below mid-grey.
        self.assertLess(float(a.mean()), 70.0)
        self.assertLess(float(a.max()), 110.0)


class GroundingShadowTests(unittest.TestCase):
    def _alpha(self, rows, cols):
        a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float64)
        a[rows[0]:rows[1], cols[0]:cols[1]] = 1.0
        return a

    def _shadow_centroid_x(self, alpha):
        plain = studio_backdrop(CANVAS, seed=99)
        shad = studio_backdrop(CANVAS, seed=99, shadow_alpha=alpha)
        diff = np.clip(plain - shad, 0, None).mean(axis=2)
        self.assertGreater(float(diff.max()), 1.0)
        xs = np.arange(CANVAS[0])[None, :]
        return float((diff * xs).sum() / diff.sum())

    def test_shadow_follows_the_mask_base(self):
        left = self._alpha((150, 300), (30, 120))
        right = self._alpha((150, 300), (150, 240))
        cx_left = self._shadow_centroid_x(left)
        cx_right = self._shadow_centroid_x(right)
        self.assertLess(abs(cx_left - 75), 0.15 * CANVAS[0])
        self.assertLess(abs(cx_right - 195), 0.15 * CANVAS[0])
        self.assertLess(cx_left, cx_right)

    def test_shadow_darkens_the_base_rim(self):
        alpha = self._alpha((150, 300), (60, 210))
        plain = studio_backdrop(CANVAS, seed=99)
        shad = studio_backdrop(CANVAS, seed=99, shadow_alpha=alpha)
        # Just below the mask's last row, at its centre column:
        # darker with the shadow; a far corner is untouched.
        self.assertLess(float(shad[306, 135].mean()),
                        float(plain[306, 135].mean()))
        self.assertTrue(np.array_equal(shad[4, 4], plain[4, 4]))

    def test_no_mask_no_shadow(self):
        empty = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float64)
        plain = studio_backdrop(CANVAS, seed=99)
        shad = studio_backdrop(CANVAS, seed=99, shadow_alpha=empty)
        self.assertTrue(np.array_equal(_u8(plain), _u8(shad)))


class DefaultAndSelectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((90, 160, 3), 120, dtype=np.uint8)
        self.alpha = np.zeros((90, 160), dtype=np.float64)
        self.alpha[20:90, 50:110] = 1.0

    def test_default_backdrop_is_studio(self):
        self.assertEqual(DEFAULT_BACKDROP, BACKDROP_STUDIO)
        self.assertEqual(DEFAULT_BACKDROP, "studio_procedural")

    def test_pipeline_path_defaults_to_studio(self):
        # The module-level selection (env unset in the suite)
        # resolves to the studio default, and a bare
        # composite_frame call renders exactly the studio mode.
        self.assertEqual(pipeline.CANVAS_BACKDROP_MODE,
                         DEFAULT_BACKDROP)
        bare = composite_frame(self.frame, self.alpha, None, CANVAS,
                               backdrop_seed=5)
        explicit = composite_frame(
            self.frame, self.alpha, None, CANVAS,
            backdrop=BACKDROP_STUDIO, backdrop_seed=5)
        self.assertTrue(np.array_equal(bare, explicit))

    def test_source_minus_speaker_still_selectable(self):
        studio = composite_frame(self.frame, self.alpha, None,
                                 CANVAS, backdrop_seed=5)
        sms = composite_frame(
            self.frame, self.alpha, None, CANVAS,
            backdrop=BACKDROP_SOURCE_MINUS_SPEAKER)
        sms_again = composite_frame(
            self.frame, self.alpha, None, CANVAS,
            backdrop=BACKDROP_SOURCE_MINUS_SPEAKER)
        self.assertFalse(np.array_equal(studio, sms))
        self.assertTrue(np.array_equal(sms, sms_again))

    def test_make_backdrop_refuses_studio(self):
        # The studio mode needs a seed/progress/placed alpha;
        # make_backdrop raises rather than substituting.
        with self.assertRaises(ValueError):
            make_backdrop(self.frame, CANVAS, backdrop=BACKDROP_STUDIO)


class _RecordingRenderer:
    def __init__(self):
        self.kwargs = {}

    def __call__(self, frames, src_alphas, **kwargs):
        self.kwargs = kwargs
        out = Path(kwargs["out_path"])
        out.write_bytes(b"fake-render")
        return out


class RepairBackdropCarryTests(unittest.TestCase):
    def _artifacts(self, td, **over):
        fields = dict(
            window_clip="window.mp4",
            render_path=Path(td) / "draft1.mp4",
            frames=[np.zeros((36, 64, 3), dtype=np.uint8)],
            src_alphas=[np.zeros((36, 64), dtype=np.float64)],
            canvas_alphas=[np.zeros((CANVAS[1], CANVAS[0]))],
            make_canvas_alphas=lambda alphas: alphas,
            fps=30.0,
            hook=HookSpec("", (0.0, 0.0), 0.30),
            cues=[CaptionCue("First phrase here.", 0.2, 0.9)],
            caption_rect=(10, 300, 260, 340),
            work_dir=td,
            canvas=CANVAS)
        fields.update(over)
        return RenderArtifacts(**fields)

    def _render(self, artifacts):
        renderer = _RecordingRenderer()
        _render_execution(
            artifacts, frames=artifacts.frames,
            src_alphas=artifacts.src_alphas,
            canvas_alphas=artifacts.canvas_alphas,
            hook=artifacts.hook, cues=artifacts.cues,
            caption_rect=artifacts.caption_rect, audio_from=None,
            out_name="draft2.mp4", detail="test", renderer=renderer)
        return renderer.kwargs

    def test_studio_seed_carries_into_repairs(self):
        with tempfile.TemporaryDirectory() as td:
            kwargs = self._render(
                self._artifacts(td, backdrop_seed=4242))
            self.assertEqual(kwargs.get("backdrop_seed"), 4242)
            self.assertNotIn("backdrop", kwargs)

    def test_selected_mode_carries_into_repairs(self):
        with tempfile.TemporaryDirectory() as td:
            kwargs = self._render(self._artifacts(
                td, backdrop_mode=BACKDROP_SOURCE_MINUS_SPEAKER))
            self.assertEqual(kwargs.get("backdrop"),
                             BACKDROP_SOURCE_MINUS_SPEAKER)
            self.assertNotIn("backdrop_seed", kwargs)

    def test_legacy_artifacts_keep_the_historic_call(self):
        with tempfile.TemporaryDirectory() as td:
            kwargs = self._render(self._artifacts(td))
            self.assertNotIn("backdrop", kwargs)
            self.assertNotIn("backdrop_seed", kwargs)


class StudioBackdropDeclarationTests(unittest.TestCase):
    def test_seed_is_an_int_the_receipt_can_carry(self):
        seed = studio_seed("video.mp4:0.000-14.000:short-1")
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)


if __name__ == "__main__":
    unittest.main()
