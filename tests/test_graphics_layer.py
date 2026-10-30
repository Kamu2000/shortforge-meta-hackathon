"""Tests for the context graphics layer: planner validation (the
strict bar, enforced in code), the deterministic card renderer,
and mask-aware placement with its hard exclusions. Offline - fake
Spark / image clients, synthetic alphas; no network, no spend.
"""

import io
import json
import sys
import unittest

import numpy as np
from PIL import Image

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue  # noqa: E402
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpan,
    build_graphics,
    parse_planner_reply,
    place_prepared,
    plan_graphics,
    prepare_card,
    prepare_image,
    render_card_image,
)

CANVAS = (1080, 1920)
TRANSCRIPT = ("the cowboys and ceedee lamb are one million dollars "
              "apart per year and the crew on get up said it is "
              "criminal negligence honestly")


class FakeChatResult:
    def __init__(self, text):
        self.text = text
        self.request_id = "chatcmpl-fake"
        self.cost_usd = 0.004


class FakeSpark:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, *, max_tokens=1500, purpose="", ts=""):
        self.calls.append({"messages": messages, "purpose": purpose})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeChatResult(reply)


def span_json(**overrides):
    span = {
        "start_s": 4.0, "end_s": 7.0, "kind": "card",
        "headline": "$1M PER YEAR APART", "facts": "Cowboys - CeeDee Lamb",
        "prompt": "", "source_quote": "one million dollars apart",
    }
    span.update(overrides)
    return {"graphics": [span]}


# ---------------------------------------------------------------- parser


class PlannerParseTests(unittest.TestCase):
    def parse(self, payload, transcript=TRANSCRIPT, duration=20.0):
        return parse_planner_reply(json.dumps(payload), transcript,
                                   duration_s=duration)

    def test_valid_card_accepted(self):
        spans, rejected = self.parse(span_json())
        self.assertEqual(len(spans), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(spans[0].kind, "card")
        self.assertEqual(spans[0].headline, "$1M PER YEAR APART")

    def test_empty_list_is_valid(self):
        spans, rejected = self.parse({"graphics": []})
        self.assertEqual((spans, rejected), ([], []))

    def test_hook_window_span_rejected(self):
        spans, rejected = self.parse(span_json(start_s=2.5, end_s=5.0))
        self.assertEqual(spans, [])
        self.assertIn("hook window", rejected[0]["reason"])

    def test_non_verbatim_quote_rejected(self):
        spans, rejected = self.parse(
            span_json(source_quote="a billion dollars apart"))
        self.assertEqual(spans, [])
        self.assertIn("verbatim", rejected[0]["reason"])

    def test_invented_number_rejected(self):
        spans, rejected = self.parse(span_json(
            headline="$5M PER YEAR APART", facts=""))
        self.assertEqual(spans, [])
        self.assertIn("never states", rejected[0]["reason"])

    def test_image_prompt_with_digit_rejected(self):
        spans, rejected = self.parse(span_json(
            kind="image", headline="", facts="",
            prompt="the 1985 championship trophy on a table",
            source_quote="the crew on get up"))
        self.assertEqual(spans, [])
        self.assertIn("digit", rejected[0]["reason"])

    def test_valid_image_accepted(self):
        spans, rejected = self.parse(span_json(
            kind="image", headline="", facts="",
            prompt="a football stadium tunnel at night, empty",
            source_quote="the crew on get up"))
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, "image")

    def test_overlapping_spans_deduped(self):
        payload = {"graphics": [
            span_json()["graphics"][0],
            span_json(start_s=5.0, end_s=8.0)["graphics"][0],
            span_json(start_s=13.0, end_s=16.0)["graphics"][0],
        ]}
        spans, rejected = self.parse(payload)
        self.assertEqual(len(spans), 2)
        self.assertTrue(any("overlaps" in r["reason"] for r in rejected))

    def test_gap_under_four_seconds_rejected(self):
        payload = {"graphics": [
            span_json()["graphics"][0],                    # 4.0-7.0
            span_json(start_s=9.0, end_s=12.0)["graphics"][0],  # gap 2s
        ]}
        spans, rejected = self.parse(payload)
        self.assertEqual(len(spans), 1)
        self.assertTrue(any("clean screen" in r["reason"]
                            for r in rejected))

    def test_no_count_cap_and_no_screen_time_ceiling(self):
        # Kamlesh's ruling (2026-10-30): the count is driven purely
        # by content clearing the bar. Four well-spaced spans
        # totalling 55% of a 40s Short all stand.
        payload = {"graphics": [
            span_json(start_s=s, end_s=s + 5.5)["graphics"][0]
            for s in (4.0, 13.5, 23.0, 32.5)
        ]}
        spans, rejected = self.parse(payload, duration=40.0)
        self.assertEqual(len(spans), 4)
        self.assertEqual(rejected, [])

    def test_span_past_short_end_rejected(self):
        spans, rejected = self.parse(span_json(start_s=18.0, end_s=21.0))
        self.assertEqual(spans, [])
        self.assertIn("past the Short's end", rejected[0]["reason"])

    def test_malformed_json_rejected_not_raised(self):
        spans, rejected = parse_planner_reply("not json at all",
                                              TRANSCRIPT)
        self.assertEqual(spans, [])
        self.assertEqual(len(rejected), 1)

    def test_fenced_json_accepted(self):
        text = "```json\n" + json.dumps(span_json()) + "\n```"
        spans, _ = parse_planner_reply(text, TRANSCRIPT, duration_s=20.0)
        self.assertEqual(len(spans), 1)


class PlanGraphicsTests(unittest.TestCase):
    def test_planner_call_failure_degrades_to_empty(self):
        spark = FakeSpark([RuntimeError("HTTP 500")])
        outcome = plan_graphics(
            spark, [CaptionCue("one million dollars apart", 4.0, 7.0)],
            duration_s=20.0)
        self.assertEqual(outcome.spans, [])
        self.assertIn("planner call failed", outcome.error)

    def test_planner_call_recorded(self):
        spark = FakeSpark([json.dumps(span_json())])
        cues = [CaptionCue("one million dollars apart per year",
                           4.0, 7.0)]
        outcome = plan_graphics(spark, cues, duration_s=20.0)
        self.assertEqual(len(outcome.spans), 1)
        self.assertEqual(outcome.spark_call["purpose"], "graphics-planner")
        self.assertEqual(outcome.spark_call["cost_usd"], 0.004)


# ---------------------------------------------------------------- cards


class CardRendererTests(unittest.TestCase):
    def test_deterministic_pixels(self):
        a = render_card_image("$1M PER YEAR APART", "Cowboys - CeeDee Lamb")
        b = render_card_image("$1M PER YEAR APART", "Cowboys - CeeDee Lamb")
        self.assertEqual(a.size, b.size)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_facts_line_grows_the_card(self):
        plain = render_card_image("BIG NEWS")
        with_facts = render_card_image("BIG NEWS", "some supporting facts")
        self.assertGreater(with_facts.size[1], plain.size[1])
        self.assertEqual(plain.size[0], 470)

    def test_card_has_visible_ink(self):
        card = render_card_image("BIG NEWS", "facts here")
        alpha = np.asarray(card)[..., 3]
        self.assertGreater(int((alpha > 0).sum()), 50_000)


# ---------------------------------------------------------------- placement


def alphas_with_speaker(n=24, *, x0=600, top=900):
    """One shared canvas alpha (speaker right side), repeated."""
    a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
    a[top:, x0:] = 1.0
    return [a] * n


def alphas_full(n=24):
    a = np.ones((CANVAS[1], CANVAS[0]), dtype=np.float32)
    return [a] * n


def card_item(start=4.0, end=7.0):
    span = GraphicSpan(start_s=start, end_s=end, kind="card",
                       headline="BIG NEWS", facts="",
                       source_quote="q")
    return (span, prepare_card(span))


class PlacementTests(unittest.TestCase):
    def test_places_beside_speaker_clear_of_face_and_captions(self):
        placed, dropped = place_prepared(
            [card_item()], alphas_with_speaker(), fps=4.0,
            caption_rect=(60, 1460, 980, 1600), canvas=CANVAS)
        self.assertEqual(dropped, [])
        self.assertEqual(len(placed), 1)
        x0, y0, x1, y1 = placed[0].box
        # The law: Euclidean clearance >= 28px from the FACE box
        # (rows 900-1226, cols 600-1080 in this fixture), nothing in
        # the caption zone (y > 1600), canvas margin respected, and
        # the box sits beside the head/shoulders (centre in the
        # band rows 840-1526) rather than drifting to a far corner.
        fx0, fy0, fx1, fy1 = 600, 900, 1080, 1226
        dx = max(fx0 - x1, x0 - fx1, 0)
        dy = max(fy0 - y1, y0 - fy1, 0)
        dist = (dx * dx + dy * dy) ** 0.5 if dx and dy else max(dx, dy)
        self.assertGreaterEqual(dist, 28)
        self.assertLessEqual(y1, 1600)
        self.assertGreaterEqual(x0, 16)
        self.assertTrue(fy0 - 60 <= (y0 + y1) // 2 <= fy1 + 300)

    def test_full_frame_speaker_drops_the_graphic(self):
        placed, dropped = place_prepared(
            [card_item()], alphas_full(), fps=4.0, canvas=CANVAS)
        self.assertEqual(placed, [])
        self.assertEqual(len(dropped), 1)
        self.assertIn("no legal placement", dropped[0]["reason"])

    def test_hook_glyph_zone_excluded_during_hook_window(self):
        # The glyph zone covers everywhere a card could legally go
        # (the whole canvas above the caption zone): during the
        # hook window there is no placement left.
        glyph = (16, 16, 1064, 1584)
        placed, dropped = place_prepared(
            [card_item(3.5, 6.0)], alphas_with_speaker(n=40), fps=4.0,
            hook_glyph_bbox=glyph, hook_window=(0.0, 6.0), canvas=CANVAS)
        self.assertEqual(placed, [])
        self.assertEqual(len(dropped), 1)

    def test_hook_zone_free_after_hook_window(self):
        glyph = (16, 100, 560, 700)
        placed, dropped = place_prepared(
            [card_item(6.5, 9.5)], alphas_with_speaker(n=40), fps=4.0,
            hook_glyph_bbox=glyph, hook_window=(0.0, 6.0), canvas=CANVAS)
        self.assertEqual(dropped, [])
        self.assertEqual(len(placed), 1)

    def test_two_graphics_never_overlap(self):
        placed, dropped = place_prepared(
            [card_item(4.0, 7.0), card_item(8.0, 11.0)],
            alphas_with_speaker(n=48), fps=4.0, canvas=CANVAS)
        self.assertEqual(len(placed), 2)
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = \
            placed[0].box, placed[1].box
        overlap = (min(ax1, bx1) - max(ax0, bx0) > 0
                   and min(ay1, by1) - max(ay0, by0) > 0)
        self.assertFalse(overlap)


# ---------------------------------------------------------------- motion

COMPARE_TRANSCRIPT = ("the cowboys offered twelve million and the "
                      "raiders offered nine million last spring")


class MotionDerivationTests(unittest.TestCase):
    def parse_one(self, payload, transcript=TRANSCRIPT, duration=30.0):
        spans, rejected = parse_planner_reply(
            json.dumps({"graphics": [payload]}), transcript,
            duration_s=duration)
        self.assertEqual(rejected, [], msg=rejected)
        self.assertEqual(len(spans), 1)
        return spans[0]

    def test_number_card_counts_up(self):
        span = self.parse_one(span_json()["graphics"][0])
        self.assertEqual(span.motion, "count_up")

    def test_image_is_ken_burns(self):
        span = self.parse_one(span_json(
            kind="image", headline="", facts="",
            prompt="a football stadium tunnel at night, empty",
            source_quote="the crew on get up")["graphics"][0])
        self.assertEqual(span.motion, "ken_burns")

    def test_compare_card_builds_two_state(self):
        span = self.parse_one({
            "start_s": 4.0, "end_s": 9.0, "kind": "card",
            "headline": "SPRING OFFERS", "facts": "",
            "prompt": "",
            "compare": [{"label": "cowboys", "value": 12000000},
                        {"label": "raiders", "value": 9000000}],
            "source_quote": "offered twelve million",
        }, transcript=COMPARE_TRANSCRIPT)
        self.assertEqual(span.motion, "build_two_state")
        self.assertEqual(len(span.compare), 2)

    def test_compare_with_invented_value_rejected(self):
        spans, rejected = parse_planner_reply(json.dumps({"graphics": [{
            "start_s": 4.0, "end_s": 9.0, "kind": "card",
            "headline": "SPRING OFFERS", "facts": "", "prompt": "",
            "compare": [{"label": "cowboys", "value": 12000000},
                        {"label": "raiders", "value": 5000000}],
            "source_quote": "offered twelve million",
        }]}), COMPARE_TRANSCRIPT, duration_s=30.0)
        self.assertEqual(spans, [])
        self.assertIn("not a number the transcript states",
                      rejected[0]["reason"])

    def test_long_quote_card_reveals_short_one_static(self):
        long_span = self.parse_one(span_json(
            headline="CRIMINAL NEGLIGENCE", facts="",
            source_quote="it is criminal negligence")["graphics"][0])
        # span_json default is 4.0-7.0 (3.0s): under the reveal floor
        self.assertEqual(long_span.motion, "static")
        span = self.parse_one(span_json(
            start_s=4.0, end_s=8.5, headline="CRIMINAL NEGLIGENCE",
            facts="", source_quote="it is criminal negligence",
        )["graphics"][0])
        self.assertEqual(span.motion, "reveal")

    def test_person_image_rejected_by_blocklist(self):
        spans, rejected = parse_planner_reply(json.dumps({"graphics": [
            span_json(kind="image", headline="", facts="",
                      prompt="a portrait of the lead singer on stage",
                      source_quote="the crew on get up")["graphics"][0],
        ]}), TRANSCRIPT, duration_s=20.0)
        self.assertEqual(spans, [])
        self.assertIn("name cards only", rejected[0]["reason"])


class AnimationStateTests(unittest.TestCase):
    def test_count_up_state_differs_and_lands_on_final(self):
        span = GraphicSpan(
            start_s=4.0, end_s=7.0, kind="card",
            headline="$1M PER YEAR APART", facts="Cowboys - CeeDee Lamb",
            source_quote="q", motion="count_up")
        content = prepare_card(span)
        start_img = content.state_fn(0.0)
        mid_img = content.state_fn(0.5)
        self.assertNotEqual(start_img.tobytes(), mid_img.tobytes())
        self.assertEqual(content.state_fn(1.0).tobytes(),
                         content.final_image.tobytes())
        self.assertEqual(content.final_image.tobytes(),
                         render_card_image(
                             "$1M PER YEAR APART",
                             "Cowboys - CeeDee Lamb").tobytes())

    def test_reveal_state_shows_fewer_pixels_early(self):
        span = GraphicSpan(
            start_s=4.0, end_s=9.0, kind="card",
            headline="CRIMINAL NEGLIGENCE HONESTLY FOLKS",
            facts="the crew said so", source_quote="q", motion="reveal")
        content = prepare_card(span)
        # Text pixels carry R=255 (white headline / pink facts);
        # the card body is near-black. Early states show fewer.
        early = np.asarray(content.state_fn(0.05))[..., 0]
        final = np.asarray(content.final_image)[..., 0]
        self.assertLess(int((early > 200).sum()),
                        int((final > 200).sum()))
        self.assertEqual(content.state_fn(1.0).tobytes(),
                         content.final_image.tobytes())

    def test_compare_bars_build_in_sequence(self):
        span = GraphicSpan(
            start_s=4.0, end_s=9.0, kind="card", headline="SPRING OFFERS",
            compare=[{"label": "cowboys", "value": 12000000},
                     {"label": "raiders", "value": 9000000}],
            source_quote="q", motion="build_two_state")
        cues = [CaptionCue("the cowboys offered twelve million",
                           4.0, 6.0),
                CaptionCue("and the raiders offered nine million",
                           6.5, 9.0)]
        content = prepare_card(span, cues)
        empty = np.asarray(content.state_fn(0.0))
        full = np.asarray(content.final_image)
        self.assertNotEqual(empty.tobytes(), full.tobytes())
        self.assertEqual(content.state_fn(1.0).tobytes(),
                         full.tobytes())

    def test_ken_burns_moves_within_a_fixed_box(self):
        img = Image.new("RGB", (300, 400))
        px = img.load()
        for y in range(400):
            for x in range(300):
                px[x, y] = (x % 256, y % 256, (x + y) % 256)
        span = GraphicSpan(start_s=4.0, end_s=9.0, kind="image",
                           motion="ken_burns")
        content = prepare_image(img, span)
        self.assertEqual(content.size, (400, 500))
        self.assertEqual(content.state_fn(0.0).size, (400, 500))
        self.assertNotEqual(content.state_fn(0.0).tobytes(),
                            content.state_fn(1.0).tobytes())


# ---------------------------------------------------------------- build


class FakeImageClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.prompts = []

    def generate(self, prompt, *, purpose="", ts=""):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("image API down")

        class Result:
            cost_usd = 0.01
            request_id = "img-fake"
        img = Image.new("RGB", (200, 260), (30, 60, 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        Result.image_bytes = buf.getvalue()
        return Result()


class BuildGraphicsTests(unittest.TestCase):
    CUES = [CaptionCue("the cowboys and ceedee lamb are", 3.2, 4.5),
            CaptionCue("one million dollars apart per year", 4.5, 7.0),
            CaptionCue("and the crew on get up said so", 7.0, 9.5)]

    def test_build_places_card_and_image(self):
        payload = {"graphics": [
            span_json()["graphics"][0],
            span_json(start_s=11.5, end_s=14.5, kind="image", headline="",
                      facts="",
                      prompt="a football stadium tunnel at night",
                      source_quote="the crew on get up")["graphics"][0],
        ]}
        spark = FakeSpark([json.dumps(payload)])
        images = FakeImageClient()
        build = build_graphics(
            spark, images, self.CUES, alphas_with_speaker(n=48),
            fps=4.0, duration_s=20.0, canvas=CANVAS)
        self.assertIsNone(build.error)
        self.assertEqual(len(build.placed), 2)
        kinds = sorted(g.kind for g in build.placed)
        self.assertEqual(kinds, ["card", "image"])
        image_spec = next(g for g in build.placed if g.kind == "image")
        self.assertEqual(image_spec.cost_usd, 0.01)
        self.assertTrue(image_spec.prompt.endswith("no words"))
        self.assertEqual(build.total_cost_usd, 0.01)

    def test_image_failure_drops_only_that_graphic(self):
        payload = {"graphics": [
            span_json(start_s=7.5, end_s=10.5, kind="image", headline="",
                      facts="",
                      prompt="a football stadium tunnel at night",
                      source_quote="the crew on get up")["graphics"][0],
        ]}
        spark = FakeSpark([json.dumps(payload)])
        build = build_graphics(
            spark, FakeImageClient(fail=True), self.CUES,
            alphas_with_speaker(n=48), fps=4.0, duration_s=20.0,
            canvas=CANVAS)
        self.assertEqual(build.placed, [])
        self.assertTrue(any("image generation failed" in d["reason"]
                            for d in build.dropped))

    def test_no_image_client_drops_image_spans(self):
        payload = {"graphics": [
            span_json(kind="image", headline="", facts="",
                      prompt="a football stadium tunnel at night",
                      source_quote="the crew on get up")["graphics"][0],
        ]}
        spark = FakeSpark([json.dumps(payload)])
        build = build_graphics(
            spark, None, self.CUES, alphas_with_speaker(n=48),
            fps=4.0, duration_s=20.0, canvas=CANVAS)
        self.assertEqual(build.placed, [])
        self.assertTrue(any("no image client" in d["reason"]
                            for d in build.dropped))


if __name__ == "__main__":
    unittest.main()
