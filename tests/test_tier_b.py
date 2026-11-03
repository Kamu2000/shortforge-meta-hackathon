"""Stage 5 tests (extension plan, 2026-11-01): chart card,
collage, foreground hook variants.

The chart tests pin the guard both ways: values the transcript
states are accepted AND drawn; an invented value is rejected at
parse AND by the standalone validator. The variant tests pin
the F2 variant leg: a foreground variant whose text touches the
face FAILS, and Stage receipts keep their historic measured
shape (no variant keys).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director import graphics_layer as gl
from spatial_director import layout_law as ll
from spatial_director.verifier import (
    HOOK_VARIANT_TEXT_OVER_FACE_MAX_PX,
    HookSpec,
    VerifyInputs,
    verify,
)

CHART_TRANSCRIPT = (
    "sales hit twelve in january and nine in february and "
    "fifteen in march this year overall")


def chart_span(**over):
    span = {
        "kind": "card",
        "start_s": 0.0, "end_s": 3.0,
        "headline": "SALES BY MONTH",
        "source_quote": "sales hit twelve in january",
        "chart": {"type": "bar", "items": [
            {"label": "january", "value": 12},
            {"label": "february", "value": 9},
            {"label": "march", "value": 15}]},
    }
    span.update(over)
    return span


def parse(spans, transcript=CHART_TRANSCRIPT):
    return gl.parse_planner_reply(
        json.dumps({"graphics": spans}), transcript,
        duration_s=20.0)


class ChartGuardTests(unittest.TestCase):
    def test_stated_values_accepted(self):
        spans, rejected = parse([chart_span()])
        self.assertEqual(rejected, [])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].chart["type"], "bar")
        self.assertEqual(
            [it["value"] for it in spans[0].chart["items"]],
            [12, 9, 15])

    def test_line_chart_accepted(self):
        span = chart_span()
        span["chart"] = {"type": "line",
                         "items": span["chart"]["items"]}
        spans, rejected = parse([span])
        self.assertEqual(rejected, [])
        self.assertEqual(spans[0].chart["type"], "line")

    def test_invented_value_rejected(self):
        span = chart_span()
        span["chart"]["items"][1]["value"] = 99
        spans, rejected = parse([span])
        self.assertEqual(spans, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("not a number the transcript states",
                      rejected[0]["reason"])

    def test_invented_value_rejected_by_standalone_validator(self):
        chart = {"type": "bar", "items": [
            {"label": "january", "value": 12},
            {"label": "february", "value": 42}]}
        reason = gl.validate_chart_values(chart, CHART_TRANSCRIPT)
        self.assertIsNotNone(reason)
        self.assertIn("42", reason)
        ok = {"type": "bar", "items": [
            {"label": "january", "value": 12},
            {"label": "february", "value": 9}]}
        self.assertIsNone(
            gl.validate_chart_values(ok, CHART_TRANSCRIPT))

    def test_unstated_label_rejected(self):
        span = chart_span()
        span["chart"]["items"][0]["label"] = "april"
        spans, rejected = parse([span])
        self.assertEqual(spans, [])
        self.assertIn("not transcript words", rejected[0]["reason"])

    def test_unknown_chart_type_rejected(self):
        span = chart_span()
        span["chart"]["type"] = "pie"
        spans, rejected = parse([span])
        self.assertEqual(spans, [])
        self.assertIn("unknown chart type", rejected[0]["reason"])

    def test_negative_value_rejected(self):
        # A negative magnitude is not a number the transcript
        # states (stated numbers are magnitudes), so the verbatim
        # guard rejects it; the standalone validator agrees.
        transcript = ("the count fell by five in january and rose "
                      "by five in february overall")
        chart = {"type": "bar", "items": [
            {"label": "january", "value": -5},
            {"label": "february", "value": 5}]}
        span = chart_span(source_quote="the count fell by five")
        span["chart"] = chart
        spans, rejected = parse([span], transcript=transcript)
        self.assertEqual(spans, [])
        self.assertEqual(len(rejected), 1)
        self.assertIsNotNone(
            gl.validate_chart_values(chart, transcript))

    def test_chart_and_compare_rejected(self):
        span = chart_span(compare=[
            {"label": "january", "value": 12},
            {"label": "march", "value": 15}])
        spans, rejected = parse([span])
        self.assertEqual(spans, [])
        self.assertIn("same span", rejected[0]["reason"])

    def test_chart_on_image_rejected(self):
        span = chart_span(kind="image", prompt="a warehouse at dawn")
        spans, rejected = parse([span])
        self.assertEqual(spans, [])
        self.assertIn("non-card", rejected[0]["reason"])

    def test_too_few_items_rejected(self):
        span = chart_span()
        span["chart"]["items"] = span["chart"]["items"][:1]
        spans, rejected = parse([span])
        self.assertEqual(spans, [])
        self.assertIn("must list 2-6 items", rejected[0]["reason"])


class ChartRenderTests(unittest.TestCase):
    CHART = {"type": "bar", "items": [
        {"label": "january", "value": 12},
        {"label": "february", "value": 9},
        {"label": "march", "value": 15}]}

    def test_bar_chart_draws_bars_proportionally(self):
        img = gl.render_chart_card_image("SALES", self.CHART, 1.0)
        arr = np.asarray(img)
        blue = (np.abs(arr[..., :3].astype(int)
                       - np.array(gl.FORGE_BLUE)).sum(axis=-1) < 30) \
            & (arr[..., 3] > 200)
        self.assertGreater(int(blue.sum()), 2000)

    def test_bar_chart_grows_with_progress(self):
        def blue_count(p):
            img = gl.render_chart_card_image("SALES", self.CHART, p)
            arr = np.asarray(img)
            blue = (np.abs(arr[..., :3].astype(int)
                           - np.array(gl.FORGE_BLUE)).sum(axis=-1)
                    < 30) & (arr[..., 3] > 200)
            # Plot region only (below the gradient strip/headline).
            return int(blue[250:, :].sum())
        self.assertLess(blue_count(0.0), blue_count(1.0))

    def test_line_chart_draws_the_line_in_pink(self):
        chart = {"type": "line", "items": self.CHART["items"]}
        img = gl.render_chart_card_image("SALES", chart, 1.0)
        arr = np.asarray(img)
        pink = (np.abs(arr[..., :3].astype(int)
                       - np.array(gl.VIRAL_PINK)).sum(axis=-1) < 30) \
            & (arr[..., 3] > 200)
        self.assertGreater(int(pink.sum()), 100)

    def test_prepare_card_dispatches_to_chart(self):
        spans, _ = parse([chart_span()])
        content = gl.prepare_card(spans[0], cues=[])
        self.assertEqual(content.size[0], gl.CHART_CARD_WIDTH)
        self.assertIsNotNone(content.state_fn)
        mid = content.state_fn(0.5)
        self.assertEqual(mid.size, content.size)


class CollageTests(unittest.TestCase):
    def still(self, rgb, size=(300, 300)):
        return Image.new("RGB", size, rgb)

    def span(self):
        return gl.GraphicSpan(
            start_s=0.0, end_s=2.0, kind="image", source_quote="q",
            collage=True)

    def test_grid_of_three_stills(self):
        stills = [self.still((200, 30, 30)),
                  self.still((30, 200, 30)),
                  self.still((30, 30, 200))]
        content = gl.prepare_collage(stills, self.span())
        self.assertEqual(content.size, gl.COLLAGE_BOX)
        arr = np.asarray(content.final_image)
        cells = gl._collage_cells(3, gl.COLLAGE_BOX)
        for (x, y, w, h), rgb in zip(
                cells, [(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
            px = arr[y + h // 2, x + w // 2, :3]
            self.assertTrue(
                np.allclose(px, rgb, atol=12),
                f"cell centre {px} != {rgb}")

    def test_count_bounds_are_hard(self):
        with self.assertRaises(ValueError):
            gl.prepare_collage([self.still((1, 2, 3))], self.span())
        with self.assertRaises(ValueError):
            gl.prepare_collage(
                [self.still((1, 2, 3))] * 5, self.span())

    def test_resizer_keeps_aspect(self):
        stills = [self.still((9, 9, 9)), self.still((8, 8, 8))]
        content = gl.prepare_collage(stills, self.span())
        small = content.resizer(280)
        self.assertEqual(small.size, (280, 320))

    def test_parse_sets_collage_and_spec_carries_it(self):
        span = {"kind": "image", "headline": "",
                "start_s": 0.0, "end_s": 3.0,
                "source_quote": "the station has three modules",
                "prompt": "orbital station modules in sunlight",
                "collage": True}
        transcript = "the station has three modules in orbit today"
        spans, rejected = gl.parse_planner_reply(
            json.dumps({"graphics": [span]}), transcript,
            duration_s=20.0)
        self.assertEqual(rejected, [])
        self.assertTrue(spans[0].collage)
        spec = gl.GraphicSpec(
            kind="image", start_s=0.0, end_s=3.0,
            box=(10, 10, 300, 300),
            layer_rgba=np.zeros((290, 290, 4), np.uint8),
            collage=spans[0].collage)
        self.assertTrue(spec.to_dict()["collage"])


class FakeChatResult:
    def __init__(self, text):
        self.text = text
        self.cost_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.request_id = "req-fake"


class FakeSpark:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, **kw):
        return FakeChatResult(self.reply)


class CollageBuildTests(unittest.TestCase):
    TRANSCRIPT = "the station has three modules in orbit today"
    REPLY = json.dumps({"graphics": [{
        "kind": "image", "headline": "",
        "start_s": 0.0, "end_s": 2.0,
        "source_quote": "the station has three modules",
        "prompt": "orbital station modules in sunlight",
        "collage": True}]})

    def cues(self):
        from spatial_director.caption_align import CaptionCue
        return [CaptionCue(text=self.TRANSCRIPT, start_s=0.0,
                           end_s=2.0)]

    def alphas(self):
        return [np.zeros((1920, 1080), np.float32)]

    def test_builds_from_cached_stills_only(self):
        stills = [Image.new("RGB", (200, 200), (10, 20, 30)),
                  Image.new("RGB", (200, 200), (40, 50, 60))]
        build = gl.build_graphics(
            FakeSpark(self.REPLY), None, self.cues(), self.alphas(),
            fps=30.0, canvas=(1080, 1920), duration_s=2.0,
            cached_images=stills)
        self.assertEqual(len(build.placed), 1)
        self.assertTrue(build.placed[0].collage)

    def test_dropped_loudly_without_cached_stills(self):
        build = gl.build_graphics(
            FakeSpark(self.REPLY), None, self.cues(), self.alphas(),
            fps=30.0, canvas=(1080, 1920), duration_s=2.0)
        self.assertEqual(build.placed, [])
        self.assertEqual(len(build.dropped), 1)
        self.assertIn("cached stills", build.dropped[0]["reason"])


class HookVariantGeometryTests(unittest.TestCase):
    def test_bar_rects(self):
        self.assertIsNone(ll.hook_variant_bar_rect("stage"))
        banner = ll.hook_variant_bar_rect("banner")
        self.assertEqual(banner, (ll.SAFE_X0, ll.SAFE_Y0, ll.SAFE_X1,
                                  ll.SAFE_Y0 + ll.HOOK_VARIANT_BANNER_H))
        lower = ll.hook_variant_bar_rect("lower_third")
        caption_top = 1920 - ll.CAPTION_BOTTOM_MARGIN \
            - ll.caption_block_height()
        self.assertEqual(lower[3], caption_top - 24)
        # Both bars live inside the central band F3 enforces.
        self.assertGreaterEqual(lower[1], ll.SAFE_Y0)
        self.assertLessEqual(lower[3], ll.SAFE_Y1)
        self.assertLessEqual(banner[3], ll.SAFE_Y1)

    def test_variant_glyphs_live_inside_the_bar(self):
        layer, mask, bbox = ll.hook_variant_layers(
            "THE STATION RECYCLES ITS AIR", (1080, 1920), "banner")
        bar = ll.hook_variant_bar_rect("banner")
        self.assertGreaterEqual(bbox[0], bar[0])
        self.assertGreaterEqual(bbox[1], bar[1])
        self.assertLessEqual(bbox[2], bar[2])
        self.assertLessEqual(bbox[3], bar[3])
        self.assertGreater(int(mask.sum()), 1000)

    def test_stage_variant_is_the_historic_layer(self):
        layer, mask, bbox = ll.hook_variant_layers(
            "THE STATION RECYCLES ITS AIR", (1080, 1920), "stage")
        historic = ll.render_hook_layer(
            "THE STATION RECYCLES ITS AIR", (1080, 1920))
        self.assertTrue(np.array_equal(layer, historic))


class FakeJudgment:
    can_summarize = True
    summary = "A speaker explains station air recycling."
    cost_usd = 0.0
    request_id = "resp-fake"


class FakeWonder:
    def judge_muted_hook(self, path, *, purpose="", ts=""):
        return FakeJudgment()


def _make_video(path: Path) -> Path:
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=black:s=320x240:r=10:d=1",
         "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=44100:duration=1",
         "-af", "volume=2.45",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True)
    return path


def _profile(total_s=4.0, hop=0.02):
    from spatial_director.caption_align import EnergyProfile
    n = int(total_s / hop)
    times = np.arange(n) * hop
    return EnergyProfile(times=times,
                         db=np.full(n, -20.0), hop_s=hop)


def _alphas(top, n=16):
    out = []
    for i in range(n):
        a = np.zeros((1920, 1080), dtype=np.float32)
        a[top:1920, 240 + 3 * i:840 + 3 * i] = 1.0
        out.append(a)
    return out


def _f2(receipt):
    return next(c for c in receipt.checks if c.check_id == "F2")


class HookVariantVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._td = tempfile.TemporaryDirectory()
        cls.video = _make_video(Path(cls._td.name) / "render.mp4")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def inputs(self, variant, speaker_top):
        return VerifyInputs(
            video_path=self.video,
            fps=4.0,
            canvas_alphas=_alphas(speaker_top),
            audio=_profile(),
            hook=HookSpec(
                "THE STATION RECYCLES ITS AIR", (0.0, 3.0), 0.30,
                variant=variant),
            cues=[],
            caption_rect=ll.initial_caption_rect())

    def test_banner_clear_of_face_passes_with_variant_keys(self):
        receipt = verify(self.inputs("banner", speaker_top=1000),
                         wonder=FakeWonder(), ts="t")
        f2 = _f2(receipt)
        self.assertTrue(f2.passed, f2.note)
        self.assertEqual(f2.measured["hook_variant"], "banner")
        self.assertEqual(
            f2.measured["variant_text_over_face_max_px"], 0)
        self.assertEqual(
            f2.threshold["variant_text_over_face_max_px"],
            HOOK_VARIANT_TEXT_OVER_FACE_MAX_PX)

    def test_banner_text_over_face_fails(self):
        # The speaker's head band reaches into the banner zone:
        # foreground text over the face must fail the variant leg.
        receipt = verify(self.inputs("banner", speaker_top=200),
                         wonder=FakeWonder(), ts="t")
        f2 = _f2(receipt)
        self.assertFalse(f2.passed)
        self.assertGreater(
            f2.measured["variant_text_over_face_max_px"], 0)

    def test_stage_keeps_historic_measured_shape(self):
        receipt = verify(self.inputs("stage", speaker_top=1000),
                         wonder=FakeWonder(), ts="t")
        f2 = _f2(receipt)
        self.assertNotIn("hook_variant", f2.measured)
        self.assertNotIn("variant_text_over_face_max_px",
                         f2.measured)
        self.assertNotIn("variant_text_over_face_max_px",
                         f2.threshold)

    def test_the_variant_cap_is_zero(self):
        self.assertEqual(HOOK_VARIANT_TEXT_OVER_FACE_MAX_PX, 0)


if __name__ == "__main__":
    unittest.main()
