"""Tests for Stage 4 graphics work: exit transitions (the entry
paths' mirror) and the strengthened Ken Burns drift.

Exits obey the entry discipline exactly: declared on the span,
travel capped at 25% of the span, the swept corridor validated
against the same free mask, a refusal recorded (never silent),
and the spec's mask() sweeping the exit corridor so F8's zone
legs measure it.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.graphics_layer import (  # noqa: E402
    CAPTION_BOTTOM_MARGIN,
    EXIT_PATH_KINDS,
    GraphicSpan,
    KEN_BURNS_MAX_ZOOM,
    _exit_path_legal,
    exit_path_box_at,
    exit_progress,
    parse_planner_reply,
    path_travel_s,
    place_prepared,
    prepare_card,
    prepare_image,
)

CANVAS = (1080, 1920)
TRANSCRIPT = "the cowboys are one million dollars apart this year"


def alphas_with_speaker(n=28, *, x0=600, top=900):
    a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
    a[top:, x0:] = 1.0
    return [a] * n


def span_json(**overrides):
    span = {
        "start_s": 4.0, "end_s": 7.0, "kind": "card",
        "headline": "$1M PER YEAR APART", "facts": "",
        "prompt": "", "source_quote": "one million dollars apart",
    }
    span.update(overrides)
    return {"graphics": [span]}


class ExitParseTests(unittest.TestCase):
    def parse(self, payload):
        return parse_planner_reply(json.dumps(payload), TRANSCRIPT,
                                   duration_s=20.0)

    def test_exit_path_accepted(self):
        spans, rejected = self.parse(span_json(exit_path="glide_out"))
        self.assertEqual(rejected, [])
        self.assertEqual(spans[0].exit_path, "glide_out")
        self.assertEqual(spans[0].to_dict()["exit_path"], "glide_out")

    def test_unknown_exit_path_rejected(self):
        spans, rejected = self.parse(span_json(exit_path="vanish"))
        self.assertEqual(spans, [])
        self.assertIn("unknown exit path", rejected[0]["reason"])

    def test_exit_path_on_hero_rejected(self):
        spans, rejected = self.parse(span_json(
            kind="backdrop", headline="", facts="",
            prompt="a stadium tunnel at night, floodlights ahead",
            exit_path="glide_out"))
        self.assertEqual(spans, [])
        self.assertIn("exit path on a non-graphic span",
                      rejected[0]["reason"])

    def test_all_exit_kinds_exist(self):
        self.assertEqual(set(EXIT_PATH_KINDS),
                         {"glide_out", "rise_out", "fall_out",
                          "recede"})


class ExitGeometryTests(unittest.TestCase):
    BOX = (100, 800, 570, 940)  # 470x140, centre-left

    def test_progress_window(self):
        self.assertEqual(exit_progress(6.0, 7.0, 0.9), 0.0)
        self.assertEqual(exit_progress(7.0, 7.0, 0.9), 1.0)
        mid = exit_progress(6.55, 7.0, 0.9)
        self.assertTrue(0.0 < mid < 1.0)

    def test_glide_out_leaves_by_the_entry_side(self):
        at_end = exit_path_box_at(self.BOX, "glide_out", 7.0, 7.0,
                                  CANVAS, 0.9)
        self.assertLessEqual(at_end[2], 0)  # fully off the left edge
        at_start = exit_path_box_at(self.BOX, "glide_out", 6.0, 7.0,
                                    CANVAS, 0.9)
        self.assertEqual(at_start, self.BOX)

    def test_recede_shrinks_about_the_centre(self):
        at_end = exit_path_box_at(self.BOX, "recede", 7.0, 7.0,
                                  CANVAS, 0.9)
        w = at_end[2] - at_end[0]
        self.assertAlmostEqual(w / 470, 0.7, places=2)
        self.assertEqual((at_end[0] + at_end[2]) // 2,
                         (100 + 570) // 2)

    def test_rise_out_stops_at_the_caption_floor(self):
        at_end = exit_path_box_at(self.BOX, "rise_out", 7.0, 7.0,
                                  CANVAS, 0.9)
        floor_bottom = CANVAS[1] - CAPTION_BOTTOM_MARGIN
        self.assertEqual(at_end[3], floor_bottom)

    def test_exit_legal_clean_and_blocked(self):
        free = np.ones((1920, 1080), dtype=bool)
        self.assertTrue(_exit_path_legal(free, self.BOX, "rise_out",
                                         7.0, CANVAS, 0.9))
        blocked = free.copy()
        blocked[1000:1100, 100:570] = False  # island in the sink path
        self.assertFalse(_exit_path_legal(blocked, self.BOX,
                                          "rise_out", 7.0, CANVAS, 0.9))
        self.assertTrue(_exit_path_legal(blocked, self.BOX,
                                         "fall_out", 7.0, CANVAS, 0.9))


class ExitPlacementTests(unittest.TestCase):
    def _item(self, exit_path, start=4.0, end=7.0):
        span = GraphicSpan(start_s=start, end_s=end, kind="card",
                           headline="BIG NEWS", facts="",
                           source_quote="q", exit_path=exit_path)
        return (span, prepare_card(span))

    def test_exit_survives_and_travels(self):
        rejections: list = []
        placed, dropped = place_prepared(
            [self._item("glide_out")], alphas_with_speaker(),
            fps=4.0, caption_rect=(60, 1460, 980, 1600),
            canvas=CANVAS, path_rejections=rejections)
        self.assertEqual(dropped, [])
        self.assertEqual(rejections, [])
        spec = placed[0]
        self.assertEqual(spec.exit_path, "glide_out")
        self.assertEqual(spec.to_dict()["exit_path"], "glide_out")
        # Mid-span the layer is the resting state; late in the
        # exit window it has moved toward the edge; the resting
        # layer_rgba is untouched by the exit.
        mid = spec.layer_at(5.5)
        self.assertTrue(np.array_equal(mid, spec.layer_rgba))
        travel = path_travel_s(spec.end_s - spec.start_s)
        late = spec.layer_at(spec.end_s - travel / 2)
        xs = np.nonzero((late[..., 3] > 0).any(axis=0))[0]
        rxs = np.nonzero((spec.layer_rgba[..., 3] > 0).any(axis=0))[0]
        self.assertNotEqual(int(xs.min()), int(rxs.min()))

    def test_exit_mask_sweeps_the_corridor(self):
        # No caption RECT in play (the caption ZONE below y=1600
        # still bounds the sink): rise_out's corridor runs from
        # the resting box down to the floor, all measurable.
        placed, _ = place_prepared(
            [self._item("rise_out")], alphas_with_speaker(),
            fps=4.0, caption_rect=None,
            canvas=CANVAS)
        spec = placed[0]
        swept = spec.mask()
        rows = np.nonzero(swept.any(axis=1))[0]
        # The sink corridor reaches BELOW the resting box.
        self.assertGreater(int(rows.max()), spec.box[3] - 1)

    def test_entry_and_exit_windows_do_not_overlap(self):
        # 25% cap each side: at the entry window's end the exit
        # progress is still 0 and vice versa.
        span_s = 3.0
        travel = path_travel_s(span_s)
        self.assertLessEqual(travel, 0.25 * span_s + 1e-9)
        self.assertEqual(
            exit_progress(4.0 + travel, 7.0, travel), 0.0)


class KenBurnsTests(unittest.TestCase):
    def test_zoom_constant_strengthened(self):
        self.assertEqual(KEN_BURNS_MAX_ZOOM, 1.12)

    def test_final_state_is_more_magnified_than_first(self):
        # A vertical stripe 20px wide in a 400x500 source: in the
        # rendered box its width grows with the zoom - at p=1
        # (zoom 1.12) it must measure ~1.12x its p=0 width.
        src = Image.new("RGB", (400, 500), (10, 10, 10))
        px = src.load()
        for x in range(100, 120):
            for y in range(500):
                px[x, y] = (240, 240, 240)
        span = GraphicSpan(start_s=0.0, end_s=4.0, kind="image",
                           source_quote="q")
        content = prepare_image(src, span)

        def stripe_width(img):
            arr = np.asarray(img.convert("RGB")).astype(int)
            bright = arr[..., 0] > 100
            cols = np.nonzero(bright.any(axis=0))[0]
            return int(cols.max()) - int(cols.min()) + 1

        w0 = stripe_width(content.state_fn(0.0))
        w1 = stripe_width(content.state_fn(1.0))
        self.assertGreater(w1, w0)
        self.assertAlmostEqual(w1 / w0, 1.12, delta=0.08)


if __name__ == "__main__":
    unittest.main()
