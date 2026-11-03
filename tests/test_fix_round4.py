"""Fix round 4 tests (2026-11-02): three user-flagged items.

1. Spotlight allocation inversion: the graphic is solved as
   the hero (majority region, cover-fit), the speaker PiP is
   subordinate BELOW it; a planner split for a non-compare
   graphic converts to Spotlight in the layout assist.
2. Caption word-timing precision: valley snapping (deepest
   envelope valley within +/-150ms, rigid span shift, neighbour
   clamps, stats) + render span hygiene (100ms floor, held
   active word).
3. Matte edge refinement: guided finish confined to the edge
   band, overshoot trim, motion-scaled temporal band blend.

Offline, synthetic fixtures only; no API calls.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpec,
    _layout_assist,
    plan_layout,
    prepare_image,
)
from spatial_director.layout_states import plan_spotlight_below  # noqa: E402
from spatial_director.matte_finish import (  # noqa: E402
    edge_band,
    refine_matte_banded,
)
from spatial_director.temporal_smooth import (  # noqa: E402
    edge_band_temporal_blend,
)
from spatial_director.word_timing import (  # noqa: E402
    WordTiming,
    active_word_index_held,
    effective_word_spans,
    snap_word_timings,
)

CANVAS = (1080, 1920)
SRC_SHAPE = (583, 640)
FPS = 30.0


def person_alphas(n=360, shape=SRC_SHAPE):
    a = np.zeros(shape, dtype=np.float32)
    a[77:, 254:476] = 1.0
    return [a.copy() for _ in range(n)]


def spec(layout, start, end, box, kind="card", compare=None):
    layer = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    layer[y0:y1, x0:x1, 3] = 255
    return GraphicSpec(
        kind=kind, start_s=start, end_s=end, box=box,
        layer_rgba=layer, headline="TEST",
        source_quote="q", motion="static", layout=layout,
        compare=compare)


# ---------------------------------------------------------- item 1


class SpotlightHeroTests(unittest.TestCase):
    def test_hero_graphic_outweighs_pip_and_pip_sits_below(self):
        from PIL import Image

        from spatial_director.graphics_layer import GraphicSpan
        img = Image.new("RGB", (640, 480), (200, 120, 40))
        span = GraphicSpan(
            start_s=4.0, end_s=7.0, kind="image", headline="",
            facts="", prompt="p", source_quote="q",
            motion="static", layout="spotlight")
        content = prepare_image(img, span, box=(960, 540))
        s = spec("spotlight", 4.0, 7.0, (0, 0, 576, 768),
                 kind="image")
        s._content = content
        _track, decls, _rects = plan_layout(
            [s], person_alphas(), fps=FPS, canvas=CANVAS,
            n_frames=360)
        decl = decls[0]
        self.assertEqual(decl["layout"], "spotlight")
        self.assertTrue(decl["feasible"], decl["note"])
        self.assertIn("spotlight_hero", decl)
        gb, pb = decl["graphic_box"], decl["speaker_box"]
        ga = (gb[2] - gb[0]) * (gb[3] - gb[1])
        pa = (pb[2] - pb[0]) * (pb[3] - pb[1])
        self.assertGreater(ga, pa)
        self.assertGreaterEqual(pb[1], gb[3])
        self.assertGreaterEqual(gb[2] - gb[0], 0.55 * CANVAS[0])

    def test_pip_solver_respects_the_face_floor(self):
        # A band too short for the legibility floor: the solver
        # says so (feasible False) instead of shrinking the face
        # into illegibility.
        alphas = person_alphas(30)
        plan = plan_spotlight_below(
            alphas, CANVAS, band=(1400, 1460), max_width=430,
            max_area=430 * 545)
        self.assertFalse(plan.feasible)

    def test_noncompare_split_converts_compare_split_stays(self):
        lone = spec("split", 0.3, 3.3, (216, 1103, 864, 1424),
                    kind="image")
        _layout_assist([lone])
        self.assertEqual(lone.layout, "spotlight")
        comp = spec("split", 0.3, 3.3, (216, 1103, 864, 1424),
                    kind="card",
                    compare=[{"label": "A", "value": 1},
                             {"label": "B", "value": 2}])
        _layout_assist([comp])
        self.assertEqual(comp.layout, "split")


# ---------------------------------------------------------- item 2


def _burst_samples(burst_at=1.0, burst_s=0.4, total_s=2.0,
                   rate=16000):
    # A noise floor throughout: real speech never has exact-zero
    # stretches, and the plateau-centre valley of a synthetic
    # flat silence is a fixture artefact, not the boundary.
    rng = np.random.default_rng(11)
    x = (rng.normal(0.0, 0.004, int(total_s * rate))
         .astype(np.float32))
    i0, i1 = int(burst_at * rate), int((burst_at + burst_s) * rate)
    t = np.arange(i1 - i0) / rate
    x[i0:i1] += 0.5 * np.sin(2 * np.pi * 220 * t)
    return x


class ValleySnapTests(unittest.TestCase):
    def test_start_snaps_back_to_the_boundary_dip(self):
        # Word placed 100ms INTO the burst; the boundary dip is
        # the quiet stretch just before the burst onset, so the
        # start must leave the burst and land in that dip.
        samples = _burst_samples()
        cues = [[WordTiming("hello", 1.100, 1.300, "mms", "t")]]
        out, stats = snap_word_timings(cues, samples)
        wt = out[0][0]
        self.assertGreaterEqual(wt.start_s, 0.94)
        self.assertLessEqual(wt.start_s, 1.01)
        # Rigid shift: duration preserved (millisecond rounding).
        self.assertAlmostEqual(wt.end_s - wt.start_s, 0.200,
                               delta=0.002)
        self.assertEqual(stats["words"], 1)
        self.assertEqual(stats["words_snapped"], 1)
        self.assertGreater(stats["mean_abs_shift_ms"], 50.0)

    def test_snap_never_crosses_a_neighbour(self):
        # Dense pair: a deep dip sits beyond the second word's
        # start; word 1 may approach but never cross it.
        samples = _burst_samples(burst_at=0.20, burst_s=0.25)
        cues = [[WordTiming("a", 0.300, 0.340, "mms", "t"),
                 WordTiming("b", 0.360, 0.420, "mms", "t")]]
        out, _stats = snap_word_timings(cues, samples)
        starts = [w.start_s for w in out[0]]
        self.assertLess(starts[0], starts[1])
        self.assertLessEqual(starts[0], 0.359)
        ends = [w.end_s for w in out[0]]
        self.assertLessEqual(ends[0], starts[1])

    def test_no_valley_in_reach_keeps_the_timing(self):
        # Flat silence is ONE plateau whose valley sits at its
        # centre (~1.0s here) - far outside the word's window,
        # so nothing is in reach and the timing stands.
        samples = np.zeros(32000, dtype=np.float32)
        cues = [[WordTiming("x", 0.200, 0.300, "dsp", "t")]]
        out, stats = snap_word_timings(cues, samples)
        self.assertEqual(out[0][0].start_s, 0.200)
        self.assertEqual(stats["words_snapped"], 0)


class SpanHygieneTests(unittest.TestCase):
    def test_effective_spans_floor_cap_and_final(self):
        timings = [WordTiming("a", 0.00, 0.04, "mms", ""),
                   WordTiming("b", 0.30, 0.34, "mms", ""),
                   WordTiming("c", 0.36, 0.40, "mms", "")]
        spans = effective_word_spans(timings)
        self.assertEqual(spans[0], (0.00, 0.10))   # 100ms floor
        self.assertEqual(spans[1], (0.30, 0.36))   # capped at next
        self.assertAlmostEqual(spans[2][1], 0.46)  # final extends

    def test_active_word_holds_across_the_gap(self):
        timings = [WordTiming("a", 0.00, 0.04, "mms", ""),
                   WordTiming("b", 0.30, 0.34, "mms", "")]
        self.assertEqual(active_word_index_held(timings, 0.20), 0)
        self.assertEqual(active_word_index_held(timings, 0.35), 1)
        self.assertIsNone(active_word_index_held(timings, 0.45))
        self.assertIsNone(active_word_index_held(timings, -0.01))


# ---------------------------------------------------------- item 3


def _edge_alpha(shape=(80, 100), edge_x=50, skirt=False):
    a = np.zeros(shape, dtype=np.float64)
    a[:, :edge_x] = 1.0
    # a soft 4px transition
    for k in range(4):
        a[:, edge_x + k] = 0.8 - 0.2 * k
    if skirt:
        a[:, edge_x + 6:edge_x + 9] = 0.05
    return a


class BandedMatteTests(unittest.TestCase):
    def test_refinement_is_confined_to_the_band(self):
        alpha = _edge_alpha()
        rng = np.random.default_rng(7)
        guide = rng.random(alpha.shape)
        out = refine_matte_banded(alpha, guide)
        band = edge_band(alpha)
        # Identity outside the band, exactly.
        self.assertTrue(np.array_equal(out[~band], alpha[~band]))
        # The deep interior stays solid, the far exterior zero.
        self.assertTrue((out[:, :30] == 1.0).all())
        self.assertTrue((out[:, 70:] == 0.0).all())

    def test_banded_refinement_is_deterministic(self):
        alpha = _edge_alpha()
        guide = np.tile(np.linspace(0, 1, alpha.shape[1]),
                        (alpha.shape[0], 1))
        a = refine_matte_banded(alpha, guide)
        b = refine_matte_banded(alpha, guide)
        self.assertTrue(np.array_equal(a, b))

    def test_pre_erode_trims_the_skirt(self):
        alpha = _edge_alpha(skirt=True)
        guide = np.zeros_like(alpha)
        plain = refine_matte_banded(alpha, guide, pre_erode_px=0)
        trimmed = refine_matte_banded(alpha, guide, pre_erode_px=1)
        self.assertLess(trimmed.sum(), plain.sum())

    def test_empty_matte_passes_through(self):
        alpha = np.zeros((40, 40))
        out = refine_matte_banded(alpha, np.zeros((40, 40)))
        self.assertTrue(np.array_equal(out, alpha))


class BandBlendTests(unittest.TestCase):
    def test_still_frames_are_untouched(self):
        a = _edge_alpha()
        out = edge_band_temporal_blend(a, a.copy())
        self.assertTrue(np.array_equal(out, a))

    def test_fast_edge_blends_inside_the_band_only(self):
        prev = _edge_alpha(edge_x=50)
        cur = _edge_alpha(edge_x=56)
        out = edge_band_temporal_blend(cur, prev)
        band = edge_band(prev) | edge_band(cur)
        self.assertTrue(np.array_equal(out[~band], cur[~band]))
        self.assertFalse(np.array_equal(out[band], cur[band]))
        # The blend pulls the moved edge back toward prev: less
        # total alpha than cur alone on the vacated side.
        self.assertLess(out[:, 52:56].sum(), cur[:, 52:56].sum())


if __name__ == "__main__":
    unittest.main()
