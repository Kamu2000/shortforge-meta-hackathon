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
from spatial_director.spark_client import SparkClientError  # noqa: E402
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpan,
    _find_box,
    _path_legal,
    build_graphics,
    parse_planner_reply,
    path_box_at,
    place_prepared,
    plan_graphics,
    prepare_backdrop,
    prepare_card,
    prepare_image,
    render_card_image,
    resolve_heroes,
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

    def test_path_accepted(self):
        spans, rejected = self.parse(span_json(path="glide_in"))
        self.assertEqual(len(spans), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(spans[0].path, "glide_in")

    def test_unknown_path_rejected(self):
        spans, rejected = self.parse(span_json(path="teleport_in"))
        self.assertEqual(spans, [])
        self.assertIn("unknown motion path", rejected[0]["reason"])

    def test_backdrop_accepted(self):
        spans, rejected = self.parse(span_json(
            kind="backdrop", headline="", facts="",
            prompt="a stadium tunnel at night, floodlights ahead"))
        self.assertEqual(len(spans), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(spans[0].kind, "backdrop")
        self.assertEqual(spans[0].motion, "ken_burns")

    def test_backdrop_digit_prompt_rejected(self):
        spans, rejected = self.parse(span_json(
            kind="backdrop", headline="", facts="",
            prompt="a stadium tunnel at night, gate 4 ahead"))
        self.assertEqual(spans, [])
        self.assertIn("digit", rejected[0]["reason"])

    def test_backdrop_with_headline_rejected(self):
        # Numbers/names/comparisons stay cards - a backdrop is a
        # scene, never data.
        spans, rejected = self.parse(span_json(
            kind="backdrop", headline="BIG NEWS", facts="",
            prompt="a stadium tunnel at night, floodlights ahead"))
        self.assertEqual(spans, [])
        self.assertIn("never data", rejected[0]["reason"])

    def test_speaker_forward_accepted(self):
        spans, rejected = self.parse(span_json(
            kind="speaker_forward", headline="", facts="",
            prompt=""))
        self.assertEqual(len(spans), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(spans[0].motion, "speaker_forward")

    def test_speaker_forward_with_headline_rejected(self):
        spans, rejected = self.parse(span_json(
            kind="speaker_forward", headline="BIG NEWS", facts="",
            prompt=""))
        self.assertEqual(spans, [])
        self.assertIn("framing only", rejected[0]["reason"])

    def test_empty_list_is_valid(self):
        spans, rejected = self.parse({"graphics": []})
        self.assertEqual((spans, rejected), ([], []))

    def test_hook_window_span_accepted(self):
        # 2026-10-31 ruling: no hook-window ban - a span may sit
        # inside the first 3.0s. Collision with the hook glyphs
        # is placement's and F8's job, not the parser's.
        spans, rejected = self.parse(span_json(start_s=2.5, end_s=5.0))
        self.assertEqual(len(spans), 1)
        self.assertEqual(rejected, [])

    def test_span_starting_at_zero_accepted(self):
        spans, rejected = self.parse(span_json(start_s=0.0, end_s=3.0))
        self.assertEqual(len(spans), 1)
        self.assertEqual(rejected, [])

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

    def test_overlapping_spans_accepted(self):
        # 2026-10-31 ruling: time-overlapping spans are legal -
        # placement separates them onto disjoint boxes (or drops
        # what fits nowhere legal).
        payload = {"graphics": [
            span_json()["graphics"][0],
            span_json(start_s=5.0, end_s=8.0)["graphics"][0],
            span_json(start_s=13.0, end_s=16.0)["graphics"][0],
        ]}
        spans, rejected = self.parse(payload)
        self.assertEqual(len(spans), 3)
        self.assertEqual(rejected, [])

    def test_gap_under_four_seconds_accepted(self):
        # 2026-10-31 ruling: the >= 4s clean-screen rule is gone.
        payload = {"graphics": [
            span_json()["graphics"][0],                    # 4.0-7.0
            span_json(start_s=9.0, end_s=12.0)["graphics"][0],  # gap 2s
        ]}
        spans, rejected = self.parse(payload)
        self.assertEqual(len(spans), 2)
        self.assertEqual(rejected, [])

    def test_back_to_back_spans_accepted(self):
        payload = {"graphics": [
            span_json()["graphics"][0],                    # 4.0-7.0
            span_json(start_s=7.0, end_s=10.0)["graphics"][0],
        ]}
        spans, rejected = self.parse(payload)
        self.assertEqual(len(spans), 2)
        self.assertEqual(rejected, [])

    def test_no_count_cap_and_no_screen_time_ceiling(self):
        # Kamlesh's rulings (2026-10-30, timing extended
        # 2026-10-31): the count is driven purely by content
        # clearing the bar, and timing is free. Four back-to-back
        # spans starting at 0.0s totalling 55% of a 40s Short all
        # stand.
        payload = {"graphics": [
            span_json(start_s=s, end_s=s + 5.5)["graphics"][0]
            for s in (0.0, 5.5, 11.0, 16.5)
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


class BudgetFakeSpark:
    """Scripted Spark that records the max_tokens of every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, *, max_tokens=1500, purpose="", ts=""):
        self.calls.append({"max_tokens": max_tokens, "purpose": purpose})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeChatResult(reply)


class PlannerRetryTests(unittest.TestCase):
    """Defect A (2026-10-30 Holmes graphics run): the planner
    starved at 4000 tokens on pipeline cue lists. Base budget is
    now 8000 with ONE retry at a doubled budget (cap 16000)."""

    def _cues(self):
        return [CaptionCue("one million dollars apart per year",
                           4.0, 7.0)]

    def test_base_budget_is_8000(self):
        spark = BudgetFakeSpark([json.dumps(span_json())])
        outcome = plan_graphics(spark, self._cues(), duration_s=20.0)
        self.assertEqual(len(outcome.spans), 1)
        self.assertEqual(spark.calls[0]["max_tokens"], 8000)

    def test_empty_reply_retried_once_at_doubled_budget(self):
        spark = BudgetFakeSpark([
            SparkClientError(
                "empty chat reply: the token budget was consumed "
                "by reasoning; raise max_tokens"),
            json.dumps(span_json())])
        outcome = plan_graphics(spark, self._cues(), duration_s=20.0)
        self.assertEqual(len(outcome.spans), 1)
        self.assertEqual(
            [c["max_tokens"] for c in spark.calls], [8000, 16000])
        self.assertTrue(outcome.spark_call["empty_first_attempt"])
        self.assertIsNone(outcome.error)

    def test_double_empty_degrades_gracefully(self):
        err = SparkClientError(
            "empty chat reply: the token budget was consumed "
            "by reasoning; raise max_tokens")
        spark = BudgetFakeSpark([err, SparkClientError(str(err))])
        outcome = plan_graphics(spark, self._cues(), duration_s=20.0)
        self.assertEqual(outcome.spans, [])
        self.assertIn("empty chat reply", outcome.error)
        self.assertEqual(len(spark.calls), 2)

    def test_non_empty_failure_is_not_retried(self):
        spark = BudgetFakeSpark([RuntimeError("HTTP 500")])
        outcome = plan_graphics(spark, self._cues(), duration_s=20.0)
        self.assertEqual(outcome.spans, [])
        self.assertIn("planner call failed", outcome.error)
        self.assertEqual(len(spark.calls), 1)


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

    def test_hook_window_span_placed_clear_of_glyph_zone(self):
        # A span inside the hook window is placeable (2026-10-31
        # ruling) - as long as its box stays out of the padded
        # glyph zone. The zone blocks the left-centre here; the
        # card must land somewhere disjoint from it.
        glyph = (60, 400, 700, 760)
        placed, dropped = place_prepared(
            [card_item(0.5, 3.0)], alphas_with_speaker(n=40), fps=4.0,
            hook_glyph_bbox=glyph, hook_window=(0.0, 6.0),
            canvas=CANVAS)
        self.assertEqual(dropped, [])
        self.assertEqual(len(placed), 1)
        x0, y0, x1, y1 = placed[0].box
        gx0, gy0, gx1, gy1 = glyph[0] - 12, glyph[1] - 12, \
            glyph[2] + 12, glyph[3] + 12
        overlap = (min(x1, gx1) - max(x0, gx0) > 0
                   and min(y1, gy1) - max(y0, gy0) > 0)
        self.assertFalse(overlap)

    def test_time_overlapping_graphics_placed_on_disjoint_boxes(self):
        placed, dropped = place_prepared(
            [card_item(4.0, 7.0), card_item(5.0, 8.0)],
            alphas_with_speaker(n=48), fps=4.0, canvas=CANVAS)
        self.assertEqual(dropped, [])
        self.assertEqual(len(placed), 2)
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = \
            placed[0].box, placed[1].box
        overlap = (min(ax1, bx1) - max(ax0, bx0) > 0
                   and min(ay1, by1) - max(ay0, by0) > 0)
        self.assertFalse(overlap)

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


# ---------------------------------------------------------------- paths


class PathGeometryTests(unittest.TestCase):
    BOX = (100, 800, 570, 940)
    CV = (1080, 1920)

    def test_glide_starts_off_canvas_and_lands_on_the_box(self):
        start = path_box_at(self.BOX, "glide_in", 4.0, 4.0, self.CV)
        self.assertEqual(start, (-470, 800, 0, 940))
        self.assertEqual(
            path_box_at(self.BOX, "glide_in", 4.9, 4.0, self.CV),
            self.BOX)
        mid = path_box_at(self.BOX, "glide_in", 4.45, 4.0, self.CV)
        self.assertTrue(-470 < mid[0] < 100)

    def test_glide_right_half_enters_from_the_right(self):
        box = (600, 800, 1000, 940)
        start = path_box_at(box, "glide_in", 4.0, 4.0, self.CV)
        self.assertEqual(start[0], 1080)

    def test_fall_starts_above_and_lands(self):
        start = path_box_at(self.BOX, "fall_in", 4.0, 4.0, self.CV)
        self.assertEqual(start, (100, -140, 570, 0))
        self.assertEqual(
            path_box_at(self.BOX, "fall_in", 5.5, 4.0, self.CV),
            self.BOX)

    def test_rise_starts_at_the_caption_ceiling(self):
        # Canvas 1920 tall, caption margin 320: the floor position
        # puts the box bottom exactly on y=1600.
        start = path_box_at(self.BOX, "rise_in", 4.0, 4.0, self.CV)
        self.assertEqual(start, (100, 1460, 570, 1600))
        self.assertEqual(
            path_box_at(self.BOX, "rise_in", 4.9, 4.0, self.CV),
            self.BOX)

    def test_approach_scales_about_the_centre(self):
        start = path_box_at(self.BOX, "approach", 4.0, 4.0, self.CV)
        w = start[2] - start[0]
        h = start[3] - start[1]
        self.assertAlmostEqual(w / 470, 0.7, places=2)
        self.assertAlmostEqual(h / 140, 0.7, places=2)
        self.assertEqual((start[0] + start[2]) // 2,
                         (100 + 570) // 2)
        # The swept area stays inside the final box.
        self.assertTrue(100 <= start[0] and start[2] <= 570)

    def test_path_legal_clean_and_blocked(self):
        free = np.ones((1920, 1080), dtype=bool)
        box = (100, 800, 570, 940)
        self.assertTrue(_path_legal(free, box, "glide_in", 4.0,
                                     self.CV))
        blocked = free.copy()
        blocked[800:940, 0:60] = False   # island in the corridor
        self.assertFalse(_path_legal(blocked, box, "glide_in", 4.0,
                                      self.CV))
        # The same island is not in fall_in's column: still legal.
        self.assertTrue(_path_legal(blocked, box, "fall_in", 4.0,
                                     self.CV))
        # Approach never leaves the box: legal iff the box is.
        self.assertTrue(_path_legal(blocked, box, "approach", 4.0,
                                     self.CV))


class PathPlacementTests(unittest.TestCase):
    def _item(self, path, start=4.0, end=7.0):
        span = GraphicSpan(start_s=start, end_s=end, kind="card",
                           headline="BIG NEWS", facts="",
                           source_quote="q", path=path)
        return (span, prepare_card(span))

    def test_path_survives_and_travels(self):
        rejections: list = []
        placed, dropped = place_prepared(
            [self._item("glide_in")], alphas_with_speaker(n=28),
            fps=4.0, caption_rect=(60, 1460, 980, 1600),
            canvas=CANVAS, path_rejections=rejections)
        self.assertEqual(dropped, [])
        self.assertEqual(rejections, [])
        spec = placed[0]
        self.assertEqual(spec.path, "glide_in")
        self.assertEqual(spec.to_dict()["path"], "glide_in")
        # Mid-travel the layer sits left of its resting box; at
        # the span's end it is exactly the resting layer.
        early = spec.layer_at(4.05)
        xs = np.nonzero((early[..., 3] > 0).any(axis=0))[0]
        self.assertLess(int(xs.min()), spec.box[0])
        self.assertTrue(np.array_equal(
            spec.layer_at(7.0), spec.layer_rgba))
        # The swept mask is wider than the resting box.
        swept = spec.mask()
        sxs = np.nonzero(swept.any(axis=0))[0]
        self.assertLess(int(sxs.min()), spec.box[0])
        self.assertGreater(int(sxs.max()) - int(sxs.min()),
                           spec.box[2] - spec.box[0])

    def test_path_refused_when_corridor_crosses_hook_zone(self):
        # The glyph zone covers the top-left; the card lands
        # beside the head (below the zone). fall_in's column from
        # the top edge crosses the zone; glide_in's row does not.
        glyph = (16, 16, 500, 500)
        rejections: list = []
        placed, dropped = place_prepared(
            [self._item("fall_in")], alphas_with_speaker(n=28),
            fps=4.0, caption_rect=(60, 1460, 980, 1600),
            hook_glyph_bbox=glyph, hook_window=(0.0, 10.0),
            canvas=CANVAS, path_rejections=rejections)
        self.assertEqual(dropped, [])
        self.assertEqual(len(placed), 1)
        self.assertIsNone(placed[0].path)
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0]["path"], "fall_in")
        self.assertIn("crosses a forbidden zone",
                      rejections[0]["reason"])

        rejections2: list = []
        placed2, _ = place_prepared(
            [self._item("glide_in")], alphas_with_speaker(n=28),
            fps=4.0, caption_rect=(60, 1460, 980, 1600),
            hook_glyph_bbox=glyph, hook_window=(0.0, 10.0),
            canvas=CANVAS, path_rejections=rejections2)
        self.assertEqual(placed2[0].path, "glide_in")
        self.assertEqual(rejections2, [])


# ---------------------------------------------------------------- size + centrality


class CentralityTests(unittest.TestCase):
    def test_centrality_beats_the_corner(self):
        # A fully free canvas: the old ranking (clearance capped,
        # then topmost/leftmost) pinned the box to the top-left
        # corner. Centrality must pull it to the middle instead.
        from scipy import ndimage
        free = np.ones((1920, 1080), dtype=bool)
        edt = ndimage.distance_transform_edt(free)
        box = _find_box(free, edt, (400, 300), None,
                        (16, 16, 1064, 1600))
        self.assertIsNotNone(box)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        self.assertLess(abs(cx - 540), 120)
        self.assertLess(abs(cy - 960), 160)
        self.assertNotEqual((box[0], box[1]), (16, 16))


class SizeToSpaceTests(unittest.TestCase):
    def test_card_scales_up_to_the_legal_space(self):
        placed, dropped = place_prepared(
            [card_item()], alphas_with_speaker(n=28), fps=4.0,
            caption_rect=(60, 1460, 980, 1600), canvas=CANVAS)
        self.assertEqual(dropped, [])
        spec = placed[0]
        width = spec.box[2] - spec.box[0]
        # The defect: a 470px card in a 1080px frame. The placed
        # card must claim a substantial share of the width.
        self.assertGreaterEqual(width, int(0.45 * 1080))
        alpha = spec.layer_rgba[..., 3] > 0
        ys, xs = np.nonzero(alpha)
        self.assertEqual(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1,
             int(ys.max()) + 1), spec.box)

    def test_card_re_render_at_width_keeps_type_proportional(self):
        span = GraphicSpan(start_s=0.0, end_s=3.0, kind="card",
                           headline="BIG NEWS", facts="",
                           source_quote="q")
        native = prepare_card(span)
        self.assertEqual(native.size[0], 470)
        big = prepare_card(span, width=940)
        self.assertEqual(big.size[0], 940)
        # Height grows with the type, not just the canvas: the
        # scaled card is proportionally the same card.
        self.assertGreater(big.size[1], native.size[1])

    def test_image_upscale_cap_chooses_a_smaller_region(self):
        # A 100x125 source may never render past 200x250 (2.0x).
        src = Image.new("RGB", (100, 125), (90, 120, 160))
        span = GraphicSpan(start_s=4.0, end_s=7.0, kind="image",
                           prompt="a quiet harbour at dusk",
                           source_quote="q", motion="ken_burns")
        content = prepare_image(src, span)
        self.assertEqual(content.max_size, (200, 250))
        empty = [np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
                 ] * 28
        placed, dropped = place_prepared(
            [(span, content)], empty, fps=4.0,
            caption_rect=(60, 1460, 980, 1600), canvas=CANVAS)
        self.assertEqual(dropped, [])
        box = placed[0].box
        self.assertLessEqual(box[2] - box[0], 200)
        self.assertLessEqual(box[3] - box[1], 250)


# ---------------------------------------------------------------- heroes


def _span(kind, start, end):
    return GraphicSpan(start_s=start, end_s=end, kind=kind,
                       headline="H" if kind == "card" else "",
                       prompt=("a stadium tunnel at night"
                               if kind in ("image", "backdrop")
                               else ""),
                       source_quote="q")


class HeroResolutionTests(unittest.TestCase):
    def test_later_hero_claim_drops(self):
        card = _span("card", 2.0, 5.0)
        backdrop = _span("backdrop", 4.0, 7.0)
        kept, drops = resolve_heroes([card, backdrop])
        self.assertEqual(kept, [card])
        self.assertEqual(len(drops), 1)
        self.assertIn("hero conflict", drops[0]["reason"])

    def test_card_overlapping_an_earlier_backdrop_drops(self):
        backdrop = _span("backdrop", 2.0, 5.0)
        card = _span("card", 4.0, 7.0)
        kept, drops = resolve_heroes([backdrop, card])
        self.assertEqual(kept, [backdrop])
        self.assertEqual(drops[0]["span"]["kind"], "card")

    def test_two_cards_may_overlap(self):
        a = _span("card", 2.0, 5.0)
        b = _span("card", 4.0, 7.0)
        kept, drops = resolve_heroes([a, b])
        self.assertEqual(kept, [a, b])
        self.assertEqual(drops, [])

    def test_non_overlapping_heroes_coexist(self):
        backdrop = _span("backdrop", 2.0, 5.0)
        fwd = _span("speaker_forward", 6.0, 9.0)
        kept, drops = resolve_heroes([backdrop, fwd])
        self.assertEqual(kept, [backdrop, fwd])
        self.assertEqual(drops, [])

    def test_speaker_forward_in_hook_window_drops(self):
        fwd = _span("speaker_forward", 1.0, 4.0)
        kept, drops = resolve_heroes([fwd], hook_window=(0.0, 6.0))
        self.assertEqual(kept, [])
        self.assertIn("hook window", drops[0]["reason"])


class BackdropBuildTests(unittest.TestCase):
    CUES = BuildGraphicsTests.CUES

    def test_backdrop_span_builds_a_spec(self):
        payload = {"graphics": [
            span_json(kind="backdrop", headline="", facts="",
                      prompt="a stadium tunnel at night, lit",
                      source_quote="the crew on get up")
            ["graphics"][0],
        ]}
        spark = FakeSpark([json.dumps(payload)])
        build = build_graphics(
            spark, FakeImageClient(), self.CUES,
            alphas_with_speaker(n=48), fps=4.0, duration_s=20.0,
            canvas=CANVAS)
        self.assertEqual(len(build.backdrops), 1)
        spec = build.backdrops[0]
        self.assertEqual(spec.kind, "backdrop")
        self.assertIn("no text", spec.prompt)  # suffix recorded
        self.assertEqual(spec.cost_usd, 0.01)
        layer = spec.layer_at(spec.start_s + 0.5)
        self.assertEqual(layer.shape, (CANVAS[1], CANVAS[0], 3))
        data = build.to_dict()
        self.assertEqual(len(data["backdrops"]), 1)
        self.assertEqual(data["total_cost_usd"], 0.01)

    def test_backdrop_treatment_darkens(self):
        img = Image.new("RGB", (400, 400), (200, 200, 200))
        spec = prepare_backdrop(img, CANVAS)
        layer = spec.layer_at(0.0)
        # 200 grey at x0.55 brightness, saturation-neutral.
        self.assertAlmostEqual(int(layer[0, 0, 0]), 110, delta=3)


if __name__ == "__main__":
    unittest.main()
