"""Tests for the deterministic layout assist (fix phase, 2026-11-02).

Measured gap (live runs pipeline-run-ext-qh2 / -qm3): the model
planner assigned NO layout state and NO speaker_forward span
live - every graphic stayed Stage in leftover side space. The
assist is the deterministic floor under that discretion:

* a placed card/image hero span >= 2.5s at Stage is tried as
  Spotlight through the existing plan_layout feasibility
  machinery (honest Stage downgrade + reason when infeasible);
* when the planner proposed no speaker_forward span, one is
  synthesized over the strongest speaker-claim cue and fed to
  the existing pipeline emphasis machinery.

Offline, synthetic geometry only; no API calls.
"""

import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue  # noqa: E402
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpec,
    _layout_assist,
    build_graphics,
    plan_layout,
)

CANVAS = (1080, 1920)
SRC_SHAPE = (583, 640)
FPS = 30.0
N = 120  # 4 seconds at 30fps


def person_alphas(n=N, shape=SRC_SHAPE):
    a = np.zeros(shape, dtype=np.float32)
    a[77:, 254:476] = 1.0
    return [a.copy() for _ in range(n)]


def spec(layout, start, end, box, kind="card"):
    layer = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    layer[y0:y1, x0:x1, 3] = 255
    return GraphicSpec(
        kind=kind, start_s=start, end_s=end, box=box,
        layer_rgba=layer, headline="TEST",
        source_quote="q", motion="static", layout=layout)


class LayoutAssistUnitTests(unittest.TestCase):
    def test_hero_span_at_floor_is_assisted(self):
        s = spec("stage", 0.3, 2.8, (216, 1103, 864, 1424))  # 2.5s
        assisted = _layout_assist([s])
        self.assertEqual(s.layout, "spotlight")
        self.assertEqual(assisted[id(s)]["requested"], "spotlight")

    def test_short_span_is_not_assisted(self):
        s = spec("stage", 0.3, 2.7, (216, 1103, 864, 1424))  # 2.4s
        self.assertEqual(_layout_assist([s]), {})
        self.assertEqual(s.layout, "stage")

    def test_declared_states_are_never_touched(self):
        # Spotlight and Cutaway declarations stand as declared.
        for layout, box in (("spotlight", (216, 1103, 864, 1424)),
                            ("cutaway", (0, 718, 1080, 1201))):
            s = spec(layout, 0.3, 3.3, box)
            self.assertEqual(_layout_assist([s]), {})
            self.assertEqual(s.layout, layout)

    def test_noncompare_split_converts_to_spotlight(self):
        # Fix round 4: split is the two-subject state; a lone
        # non-compare graphic in split is the subordinate-graphic
        # failure, so the assist converts it to Spotlight and
        # records the conversion with its reason.
        s = spec("split", 0.3, 3.3, (208, 472, 856, 793))
        assisted = _layout_assist([s])
        self.assertEqual(s.layout, "spotlight")
        self.assertEqual(assisted[id(s)]["requested"], "split")
        self.assertIn("reason", assisted[id(s)])
        self.assertEqual(s.layout_assist["requested"], "split")

    def test_compare_split_stays_split(self):
        # Two compared items beside the speaker is what Split is
        # for - a compare card keeps its genuine Split.
        s = spec("split", 0.3, 3.3, (208, 472, 856, 793))
        s.compare = [{"label": "HOME", "value": 12},
                     {"label": "AWAY", "value": 9}]
        self.assertEqual(_layout_assist([s]), {})
        self.assertEqual(s.layout, "split")

    def test_second_overlapping_hero_stays_stage(self):
        a = spec("stage", 0.3, 3.3, (216, 1103, 864, 1424))
        b = spec("stage", 1.0, 4.0, (216, 1103, 864, 1424))
        assisted = _layout_assist([a, b])
        self.assertIn(id(a), assisted)
        self.assertNotIn(id(b), assisted)
        self.assertEqual(b.layout, "stage")

    def test_hero_overlapping_declared_state_stays_stage(self):
        declared = spec("cutaway", 0.3, 3.3, (0, 718, 1080, 1201))
        hero = spec("stage", 1.0, 4.0, (216, 1103, 864, 1424))
        self.assertEqual(_layout_assist([declared, hero]), {})
        self.assertEqual(hero.layout, "stage")


class PlanLayoutAssistTests(unittest.TestCase):
    def test_hero_gets_spotlight_with_assist_record(self):
        s = spec("stage", 0.3, 3.3, (216, 1103, 864, 1424))
        built = plan_layout([s], person_alphas(), fps=FPS,
                            canvas=CANVAS)
        self.assertIsNotNone(built)
        track, decls, _rects = built
        self.assertIsNotNone(track)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["layout"], "spotlight")
        self.assertTrue(decls[0]["feasible"])
        self.assertEqual(decls[0]["layout_assist"]["requested"],
                         "spotlight")
        self.assertEqual(decls[0]["layout_assist"]["applied"],
                         "spotlight")
        self.assertEqual(s.layout, "spotlight")

    def test_infeasible_hero_downgrades_honestly_with_reason(self):
        # The graphic box covers BOTH PiP corners: no legal PiP,
        # so the assist's Spotlight request falls back to Stage
        # with the reason on the declaration.
        s = spec("stage", 0.3, 3.3, (0, 300, 1080, 900))
        built = plan_layout([s], person_alphas(), fps=FPS,
                            canvas=CANVAS)
        self.assertIsNotNone(built)
        track, decls, _rects = built
        self.assertIsNone(track)
        self.assertEqual(decls[0]["downgraded_to"], "stage")
        self.assertFalse(decls[0]["feasible"])
        assist = decls[0]["layout_assist"]
        self.assertEqual(assist["requested"], "spotlight")
        self.assertEqual(assist["applied"], "stage")
        self.assertTrue(assist["reason"])
        self.assertEqual(s.layout, "stage")

    def test_no_hero_no_states_returns_none(self):
        # Baseline behaviour with the assist present: a Short
        # whose only graphic is a short Stage card plans no
        # layout track at all (graphics-off equivalence).
        s = spec("stage", 0.3, 1.2, (216, 1103, 864, 1424))
        self.assertIsNone(plan_layout(
            [s], person_alphas(), fps=FPS, canvas=CANVAS))


# ---------------------------------------------------------------- build


class FakeChatResult:
    def __init__(self, text):
        self.text = text
        self.request_id = "chatcmpl-fake"
        self.cost_usd = 0.004


class FakeSpark:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, *, max_tokens=1500, purpose="", ts=""):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeChatResult(reply)


class FakeImageClient:
    def generate(self, prompt, *, purpose="", ts=""):
        class Result:
            cost_usd = 0.01
            request_id = "img-fake"
        img = Image.new("RGB", (200, 260), (30, 60, 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        Result.image_bytes = buf.getvalue()
        return Result()


def build_alphas(n=48):
    # build_graphics consumes CANVAS alphas (the pipeline hands
    # it the canvas-space speaker masks), not source-shaped ones.
    a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
    a[400:1500, 300:700] = 1.0
    return [a.copy() for _ in range(n)]


CUES = [
    CaptionCue("welcome back to the show everyone", 0.5, 3.0),
    CaptionCue("he scored forty two points last night", 4.0, 7.0),
    CaptionCue("and the crowd went absolutely quiet", 8.0, 10.5),
]


class BuildSpotlightAssistTests(unittest.TestCase):
    """The assist runs in build_graphics BEFORE placement, so an
    assisted hero gets the Spotlight placement branch's
    centre-stage box - not a Stage leftover box with a Spotlight
    label pinned on afterwards."""

    CARD_CUES = [CaptionCue("one million dollars apart per year",
                            4.0, 7.0)]

    def _card_build(self, start, end, cues=None, headline=None,
                    facts=None):
        payload = {"graphics": [{
            "start_s": start, "end_s": end, "kind": "card",
            "headline": headline or "$1M PER YEAR APART",
            "facts": facts or "Cowboys - CeeDee Lamb", "prompt": "",
            "source_quote": "one million dollars apart"}]}
        spark = FakeSpark([json.dumps(payload)])
        return build_graphics(
            spark, FakeImageClient(), cues or self.CARD_CUES,
            build_alphas(), fps=4.0, duration_s=20.0, canvas=CANVAS)

    def test_hero_card_is_spotlight_placed_with_marker(self):
        build = self._card_build(4.0, 7.0)  # 3.0s hero
        self.assertEqual(len(build.placed), 1)
        placed = build.placed[0]
        self.assertEqual(placed.layout, "spotlight")
        self.assertIsNotNone(placed.layout_assist)
        self.assertEqual(placed.layout_assist["requested"],
                         "spotlight")
        self.assertEqual(placed.to_dict()["layout"], "spotlight")
        self.assertIn("layout_assist", placed.to_dict())

    def test_assisted_box_is_not_the_stage_box(self):
        hero = self._card_build(4.0, 7.0).placed[0]
        # A 2.0s span (with a matching 2.0s cue, so the timing
        # snap cannot grow it past the assist floor) stays Stage.
        short_cues = [CaptionCue("one million dollars apart",
                                 4.0, 6.0)]
        short = self._card_build(
            4.0, 6.0, cues=short_cues, headline="$1M APART",
            facts="Cowboys").placed[0]
        self.assertEqual(short.layout, "stage")
        self.assertIsNone(short.layout_assist)
        self.assertNotEqual(hero.box, short.box)


class SpeakerForwardAssistTests(unittest.TestCase):
    def _build(self, payload, cues=CUES, **kwargs):
        spark = FakeSpark([json.dumps(payload)])
        return build_graphics(
            spark, FakeImageClient(), cues, build_alphas(),
            fps=4.0, duration_s=20.0, canvas=CANVAS, **kwargs)

    def test_assist_synthesizes_span_on_strongest_claim(self):
        build = self._build({"graphics": []})
        self.assertEqual(len(build.speaker_forward), 1)
        span = build.speaker_forward[0]
        # The numbered claim beats the longer plain cues.
        self.assertEqual((span.start_s, span.end_s), (4.0, 7.0))
        self.assertEqual(span.kind, "speaker_forward")
        self.assertEqual(len(build.layout_assist), 1)
        record = build.layout_assist[0]
        self.assertEqual(record["requested"], "speaker_forward")
        self.assertEqual(record["applied"], "speaker_forward")
        self.assertEqual(record["span"], [4.0, 7.0])

    def test_planner_span_means_no_assist(self):
        payload = {"graphics": [{
            "start_s": 8.0, "end_s": 10.5, "kind": "speaker_forward",
            "headline": "", "facts": "", "prompt": "",
            "source_quote": "the crowd went absolutely quiet"}]}
        build = self._build(payload)
        self.assertEqual(len(build.speaker_forward), 1)
        self.assertEqual(build.layout_assist, [])

    def test_hook_window_cue_is_ineligible(self):
        cues = [CaptionCue("he scored forty two points", 0.5, 3.0),
                CaptionCue("a plain spoken line here", 4.0, 7.0)]
        build = self._build({"graphics": []}, cues=cues,
                            hook_window=(0.0, 3.5))
        self.assertEqual(len(build.speaker_forward), 1)
        self.assertEqual(build.speaker_forward[0].start_s, 4.0)

    def test_no_eligible_cue_means_no_span_and_no_record(self):
        cues = [CaptionCue("he scored forty two points", 0.5, 3.0)]
        build = self._build({"graphics": []}, cues=cues,
                            hook_window=(0.0, 3.5))
        self.assertEqual(build.speaker_forward, [])
        self.assertEqual(build.layout_assist, [])

    def test_assist_record_is_in_plan_dict(self):
        build = self._build({"graphics": []})
        plan = build.to_dict()
        self.assertEqual(len(plan["layout_assist"]), 1)
        self.assertEqual(len(plan["speaker_forward"]), 1)


if __name__ == "__main__":
    unittest.main()
