"""Tests for mask_decode: parse, decode, primary selection, finish.

The real-response tests run against the committed trimmed fixture
(first 3 frames of the first live SAM call, clip A). The synthetic
multi-object payload is fabricated with the installed parser
library's own one-bit encoder - used ONLY to manufacture fixture
bytes; everything under test (parse, select, decode, finish) goes
through the public decode path.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meta_sam_parser._mask_codec import _encode_raster  # noqa: E402

from spatial_director.mask_decode import (  # noqa: E402
    FrameMasks,
    MaskDecodeError,
    ObjectMask,
    decode_object_canvas,
    decode_object_raster,
    finished_alpha,
    parse_output_text,
    parse_sam_response,
    place_on_canvas,
    primary_canvases,
    select_primary_id,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "sam_response_clipA_trimmed.json"
)

FW, FH = 40, 30  # synthetic frame size (w, h)


def obj(oid, box, fw=FW, fh=FH):
    """A selection-only record (payload never decoded)."""
    return ObjectMask(
        object_id=oid, box=box, frame_w=fw, frame_h=fh,
        mask_h=box[3] - box[1] + 1, mask_w=box[2] - box[0] + 1,
        encoding="one_bit", payload="!",
    )


def frame(idx, objects):
    return FrameMasks(index=idx, objects=objects)


class FixtureParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.frames = parse_sam_response(cls.raw)

    def test_three_frames_one_object_each(self):
        self.assertEqual([f.index for f in self.frames], [0, 1, 2])
        for f in self.frames:
            self.assertEqual(len(f.objects), 1)
            self.assertEqual(f.objects[0].object_id, 0)

    def test_first_frame_record_fields(self):
        o = self.frames[0].objects[0]
        self.assertEqual(o.box, (0, 48, 240, 359))
        self.assertEqual((o.frame_w, o.frame_h), (272, 360))
        # data=<H>,<W>: the raster is box-sized, H first.
        self.assertEqual((o.mask_h, o.mask_w), (312, 241))
        self.assertEqual(o.encoding, "lossless")

    def test_decode_raster_and_canvas(self):
        o = self.frames[0].objects[0]
        raster = decode_object_raster(o)
        self.assertEqual(raster.shape, (312, 241))
        coverage = float(raster.mean())
        # Spike measured 0.62-0.65 coverage on clip A frames.
        self.assertGreater(coverage, 0.55)
        self.assertLess(coverage, 0.72)
        canvas = decode_object_canvas(o)
        self.assertEqual(canvas.shape, (360, 272))
        # Box sits fully inside the frame, so nothing is clipped away.
        self.assertEqual(int(canvas.sum()), int(raster.sum()))
        np.testing.assert_array_equal(canvas[48:360, 0:241], raster)


class PlacementTests(unittest.TestCase):
    def test_box_hanging_off_edge_is_clipped(self):
        o = obj(0, (-2, -1, 3, 2), fw=10, fh=10)
        raster = np.ones((4, 6), dtype=bool)  # mask_h=4, mask_w=6
        canvas = place_on_canvas(o, raster)
        self.assertEqual(canvas.shape, (10, 10))
        self.assertEqual(int(canvas.sum()), 3 * 4)  # only the overlap
        self.assertTrue(canvas[0:3, 0:4].all())


class SelectionTests(unittest.TestCase):
    def test_speaker_need_not_be_id_zero(self):
        # Clip C's lesson: id 0 is a small background-TV figure, the
        # desk speaker is id 1 and dominates by area.
        tv = (30, 2, 35, 9)          # 6x8 = 48 px
        speaker = (5, 5, 24, 29)     # 20x25 = 500 px
        frames = [frame(i, [obj(0, tv), obj(1, speaker)]) for i in range(5)]
        self.assertEqual(select_primary_id(frames), 1)

    def test_large_but_transient_loses_to_persistent(self):
        huge = (0, 0, 39, 29)        # whole frame, but only 2 of 10
        medium = (10, 8, 25, 29)     # present in all 10
        frames = []
        for i in range(10):
            objs = [obj(1, medium)]
            if i < 2:
                objs.insert(0, obj(0, huge))
            frames.append(frame(i, objs))
        self.assertEqual(select_primary_id(frames), 1)

    def test_single_object_and_wire_order_irrelevant(self):
        frames = [frame(i, [obj(7, (1, 1, 9, 9))]) for i in range(3)]
        self.assertEqual(select_primary_id(frames), 7)
        a = (28, 2, 34, 8)
        b = (4, 4, 20, 28)
        f1 = [frame(i, [obj(0, a), obj(1, b)]) for i in range(4)]
        f2 = [frame(i, [obj(1, b), obj(0, a)]) for i in range(4)]
        self.assertEqual(select_primary_id(f1), select_primary_id(f2))


def one_bit_payload(raster: np.ndarray) -> str:
    h, w = raster.shape
    return _encode_raster(raster.astype(np.uint8).ravel().tolist(), w, h)


def record(oid, box, raster, fw=FW, fh=FH) -> str:
    x1, y1, x2, y2 = box
    mh, mw = raster.shape
    return (
        f"{oid}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={fw};h={fh}|>"
        f"<|mask;x=0;y=0;data={mh},{mw},{one_bit_payload(raster)}|>"
    )


def synthetic_payload() -> dict:
    """3 frames: small TV figure (id 0) + big speaker (id 1)."""
    tv_raster = np.ones((8, 6), dtype=bool)
    chunks = []
    for i in range(3):
        speaker = np.zeros((25, 20), dtype=bool)
        speaker[2:23, 3:17] = True  # person-ish blob inside the box
        sx = 5 + i  # speaker drifts 1px/frame
        text = (
            record(0, (30, 2, 35, 9), tv_raster)
            + record(1, (sx, 5, sx + 19, 29), speaker)
        )
        chunks.append(f"<{i}f>" + text)
    return {
        "id": "resp-synthetic",
        "output": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "".join(chunks)}],
        }],
    }


class SyntheticEndToEndTests(unittest.TestCase):
    def test_parse_select_decode_finish_with_speaker_id_1(self):
        frames = parse_sam_response(synthetic_payload())
        self.assertEqual(len(frames), 3)
        self.assertEqual(select_primary_id(frames), 1)

        canvases = primary_canvases(frames)
        self.assertEqual(sorted(canvases), [0, 1, 2])
        for c in canvases.values():
            self.assertEqual(c.shape, (FH, FW))
            # The TV figure (top-right corner) is NOT in the primary.
            self.assertFalse(c[2:10, 30:36].any())

        # Guides mimic a source frame: luminance follows the subject
        # (a flat guide would degenerate the guided filter to a blur).
        guides = [
            0.15 + 0.7 * canvases[i].astype(float) for i in range(3)
        ]
        indices, alphas = finished_alpha(frames, guides)
        self.assertEqual(indices, [0, 1, 2])
        for a in alphas:
            self.assertEqual(a.shape, (FH, FW))
            self.assertGreaterEqual(a.min(), 0.0)
            self.assertLessEqual(a.max(), 1.0)
        # Speaker interior is solid; TV corner stays empty.
        self.assertGreater(alphas[1][17, 13], 0.8)
        self.assertLess(alphas[1][5, 32], 0.05)

    def test_generator_guides_bit_identical_to_list(self):
        # The 2026-10-29 OOM fix streams the finish pass: guides may
        # arrive as a one-shot generator and canvases are decoded one
        # at a time into the EMA. The math order is unchanged, so the
        # output must be BIT-identical to the list form, not merely
        # close.
        frames = parse_sam_response(synthetic_payload())
        canvases = primary_canvases(frames)
        guides = [
            0.15 + 0.7 * canvases[i].astype(float) for i in range(3)
        ]
        _, from_list = finished_alpha(frames, guides)
        _, from_gen = finished_alpha(frames, (g for g in guides))
        self.assertEqual(len(from_list), len(from_gen))
        for a, b in zip(from_list, from_gen):
            self.assertTrue(np.array_equal(a, b))

    def test_iterator_guide_count_mismatch_still_raises(self):
        frames = parse_sam_response(synthetic_payload())
        canvases = primary_canvases(frames)
        guides = [
            0.15 + 0.7 * canvases[i].astype(float) for i in range(3)
        ]
        with self.assertRaises(MaskDecodeError):
            finished_alpha(frames, iter(guides[:2]))
        with self.assertRaises(MaskDecodeError):
            finished_alpha(frames, iter(guides + guides[:1]))


if __name__ == "__main__":
    unittest.main()
