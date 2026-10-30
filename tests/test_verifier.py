"""Tests for verifier (F1-F7 + receipt) and the Wonder judge client.

The all-pass synthetic case is the load-bearing one: a coherent
fixture must be able to PASS - a verifier that fails everything is
as broken as one that passes everything. Each failure case then
breaks exactly one thing and checks the receipt names it with the
measured value.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue, EnergyProfile  # noqa: E402
from spatial_director.compositor import (  # noqa: E402
    _encode_video,
    graphics_layer_at,
)
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpec,
    _paste_layer_clipped,
    path_box_at,
    prepare_backdrop,
)
from spatial_director.layout_law import initial_caption_rect  # noqa: E402
from spatial_director.spend_ledger import SpendLedger  # noqa: E402
from spatial_director.verifier import (  # noqa: E402
    HookSpec,
    VerifyInputs,
    verify,
)
from spatial_director.wonder_client import (  # noqa: E402
    WonderClient,
    WonderClientError,
    cost_from_usage,
    parse_judgment,
)

CANVAS = (1080, 1920)
FPS = 4.0
N = 16  # 4 seconds


def make_video(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=black:s=320x240:r=10:d=1",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True)
    return path


def speaker(top=1000, x_shift=3, n=N):
    alphas = []
    for i in range(n):
        a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
        x0 = 240 + x_shift * i
        a[top:1920, x0:x0 + 600] = 1.0
        alphas.append(a)
    return alphas


def profile(spans, total_s=4.0, hop=0.02):
    n = int(total_s / hop)
    times = np.arange(n) * hop
    db = np.full(n, -80.0)
    for a, b in spans:
        db[(times >= a) & (times < b)] = -20.0
    return EnergyProfile(times=times, db=db, hop_s=hop)


class FakeJudgment:
    def __init__(self, can=True):
        self.can_summarize = can
        self.summary = "A musician explains his band never rehearses." \
            if can else ""
        self.cost_usd = 0.01
        self.request_id = "resp-wonder-test"


class FakeWonder:
    def __init__(self, can=True):
        self.can = can
        self.calls = []

    def judge_muted_hook(self, path, *, purpose="", ts=""):
        self.calls.append(str(path))
        return FakeJudgment(self.can)


def good_inputs(video, **overrides):
    inputs = VerifyInputs(
        video_path=video,
        fps=FPS,
        canvas_alphas=speaker(),
        audio=profile([(0.0, 4.0)]),
        hook=HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                      layout_rendered=True),
        cues=[CaptionCue("The band never rehearses.", 0.0, 2.0),
              CaptionCue("It is just perfect.", 2.0, 4.0)],
        caption_rect=initial_caption_rect(),
    )
    for key, value in overrides.items():
        setattr(inputs, key, value)
    return inputs


def check(receipt, check_id):
    return next(c for c in receipt.checks if c.check_id == check_id)


class VerifierPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.video = make_video(Path(cls._td.name) / "render.mp4")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_all_pass_first_draft(self):
        receipt = verify(good_inputs(self.video), wonder=FakeWonder(),
                         ts="2026-10-29")
        self.assertEqual(receipt.failed, [],
                         msg=receipt.to_json())
        self.assertEqual(receipt.verdict, "PASSED_FIRST_DRAFT")
        self.assertEqual(len(receipt.meta_calls), 1)
        self.assertEqual(receipt.meta_calls[0]["model"], "muse-wonder-1.0")

    def test_receipt_is_json_with_hashes(self):
        receipt = verify(good_inputs(self.video), wonder=FakeWonder())
        data = json.loads(receipt.to_json())
        self.assertEqual(len(data["inputs_sha256"]["video"]), 64)
        self.assertEqual(len(data["checks"]), 7)


class VerifierFailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.video = make_video(Path(cls._td.name) / "render.mp4")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_occluded_hook_fails_f2(self):
        inputs = good_inputs(
            self.video,
            canvas_alphas=speaker(top=420, x_shift=0),
            hook=HookSpec("THE BAND NEVER REHEARSES", (0.0, 3.0), 0.30,
                          layout_rendered=False),
        )
        receipt = verify(inputs, wonder=FakeWonder())
        f2 = check(receipt, "F2")
        self.assertFalse(f2.passed)
        self.assertGreater(f2.measured["glyph_occlusion_mean"], 0.25)
        self.assertIn("glyph_occlusion_alpha_weighted_mean",
                      f2.measured)
        self.assertEqual(receipt.verdict, "FAIL")

    def test_dead_air_fails_f5(self):
        inputs = good_inputs(self.video,
                             audio=profile([(0.0, 1.0), (2.5, 4.0)]))
        receipt = verify(inputs, wonder=FakeWonder())
        f5 = check(receipt, "F5")
        self.assertFalse(f5.passed)
        self.assertAlmostEqual(f5.measured["max_silence_gap_s"], 1.5,
                               delta=0.05)

    def test_missing_captions_fail_f5_not_f2(self):
        inputs = good_inputs(self.video, cues=[], caption_rect=None)
        receipt = verify(inputs, wonder=FakeWonder())
        self.assertFalse(check(receipt, "F5").passed)
        self.assertFalse(
            check(receipt, "F5").measured["captions_present"])
        f2 = check(receipt, "F2")
        self.assertTrue(f2.passed)  # vacuous overlap; presence is F5
        self.assertIn("no captions", f2.note)

    def test_judge_cannot_summarize_fails_f5(self):
        receipt = verify(good_inputs(self.video),
                         wonder=FakeWonder(can=False))
        f5 = check(receipt, "F5")
        self.assertFalse(f5.passed)
        self.assertFalse(
            f5.measured["wonder_can_summarize_muted_hook"])

    def test_no_judge_client_fails_f5(self):
        receipt = verify(good_inputs(self.video), wonder=None)
        f5 = check(receipt, "F5")
        self.assertFalse(f5.passed)
        self.assertIn("not run", f5.note)

    def test_static_opening_fails_f7(self):
        inputs = good_inputs(self.video,
                             canvas_alphas=speaker(x_shift=0))
        receipt = verify(inputs, wonder=FakeWonder())
        f7 = check(receipt, "F7")
        self.assertFalse(f7.passed)
        self.assertAlmostEqual(f7.measured["mean_mask_iou_0_3s"], 1.0)

    def test_hook_slide_passes_f7_on_static_opening(self):
        # Defect B (2026-10-30): a genuinely static window fails
        # F7 on all three measurements; the slide repair's
        # designed motion must pass it, at the speed the render
        # actually carries - not via a text change.
        hook = HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                        layout_rendered=True, slide=(-240.0, 1.2))
        inputs = good_inputs(self.video,
                             canvas_alphas=speaker(x_shift=0),
                             hook=hook)
        receipt = verify(inputs, wonder=FakeWonder())
        f7 = check(receipt, "F7")
        self.assertTrue(f7.passed)
        self.assertEqual(f7.measured["hook_slide_px_s"], 200.0)
        self.assertAlmostEqual(f7.measured["mean_mask_iou_0_3s"], 1.0)

    def test_glyph_at_follows_the_slide(self):
        from spatial_director.compositor import shift_layer_x
        hook = HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                        layout_rendered=True, slide=(-240.0, 1.2))
        static_mask, static_bbox = hook.glyph()
        at_start, _ = hook.glyph_at(0.0)
        at_end, end_bbox = hook.glyph_at(1.2)
        self.assertTrue((at_end == static_mask).all())
        self.assertEqual(end_bbox, static_bbox)
        # At t=0 the glyphs sit exactly the renderer's shift
        # function left of final (edge clipping included); the
        # mask differs from static.
        self.assertTrue(
            (at_start == shift_layer_x(static_mask, -240)).all())
        self.assertFalse((at_start == static_mask).all())
        self.assertEqual(hook.slide_px_s, 200.0)

    def test_f2_measures_the_slid_glyph_position(self):
        # Speaker overlapping the hook band: a static hook's
        # occlusion is the full overlap, but a hook sliding in
        # from the left spends its opening frames further from
        # the speaker - F2 must report the smaller slid value,
        # not the static one.
        alphas = speaker(top=500, x_shift=0)
        static_inputs = good_inputs(self.video, canvas_alphas=alphas)
        slid_hook = HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                             layout_rendered=True, slide=(-240.0, 1.2))
        slid_inputs = good_inputs(self.video, canvas_alphas=alphas,
                                  hook=slid_hook)
        static_f2 = check(verify(static_inputs, wonder=FakeWonder()),
                          "F2")
        slid_f2 = check(verify(slid_inputs, wonder=FakeWonder()), "F2")
        self.assertLess(
            slid_f2.measured["glyph_occlusion_mean"],
            static_f2.measured["glyph_occlusion_mean"])

    def test_low_head_margin_fails_f4(self):
        inputs = good_inputs(self.video,
                             canvas_alphas=speaker(top=50, x_shift=0))
        receipt = verify(inputs, wonder=FakeWonder())
        f4 = check(receipt, "F4")
        self.assertFalse(f4.passed)
        self.assertEqual(f4.measured["head_margin_min_px"], 50)


class WonderClientTests(unittest.TestCase):
    def test_parse_judgment(self):
        self.assertEqual(
            parse_judgment("SUMMARY: A man talks about his band."),
            ("A man talks about his band.", True))
        self.assertEqual(parse_judgment("SUMMARY: CANNOT_DETERMINE"),
                         ("", False))
        self.assertEqual(parse_judgment("no marker here"), ("", False))

    def test_cost_math(self):
        self.assertAlmostEqual(cost_from_usage(1_000_000, 0), 4.00)
        self.assertAlmostEqual(cost_from_usage(0, 1_000_000), 20.00)
        self.assertAlmostEqual(
            cost_from_usage(1_000_000, 0, cached_tokens=1_000_000), 0.20)

    def _client(self, raw, ledger=None):
        calls = []

        def upload(url, headers, body):
            calls.append(("upload", url))
            return 200, {"id": "file-123"}

        def transport(method, url, headers, body):
            calls.append(("responses", url, body))
            return 200, raw

        client = WonderClient(api_key="k", ledger=ledger,
                              transport=transport, upload_transport=upload)
        return client, calls

    def test_judge_flow_and_ledger(self):
        raw = {
            "id": "resp-wonder-1",
            "output": [{"content": [{
                "type": "output_text",
                "text": "SUMMARY: A musician's hook about rehearsals.",
            }]}],
            "usage": {"input_tokens": 5000, "output_tokens": 60},
        }
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "l.jsonl", cap_usd=145.0)
            client, calls = self._client(raw, ledger)
            clip = Path(td) / "muted.mp4"
            clip.write_bytes(b"fake")
            j = client.judge_muted_hook(clip, ts="2026-10-29")
            self.assertTrue(j.can_summarize)
            self.assertEqual(j.file_id, "file-123")
            self.assertAlmostEqual(j.cost_usd,
                                   cost_from_usage(5000, 60))
            entries = ledger.entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].model, "muse-wonder-1.0")
            self.assertEqual(entries[0].request_id, "resp-wonder-1")
            payload = json.loads(calls[1][2])
            self.assertEqual(payload["model"], "muse-wonder-1.0")
            parts = payload["input"][0]["content"]
            self.assertEqual(parts[1]["type"], "input_video")
            self.assertEqual(parts[1]["file_id"], "file-123")

    def test_default_upload_transport_accepts_upload_shape(self):
        # Regression: the DEFAULT upload transport must match the
        # (url, headers, body) UploadTransport shape - wiring the
        # 4-arg Responses transport in crashed the first live proof.
        import io
        from unittest import mock

        class Resp(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        payload = json.dumps({"id": "file-default"}).encode()
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "m.mp4"
            clip.write_bytes(b"x")
            client = WonderClient(api_key="k")
            with mock.patch(
                    "spatial_director.wonder_client.urllib.request.urlopen",
                    return_value=Resp(payload)) as m:
                self.assertEqual(client.upload_file(clip), "file-default")
                req = m.call_args[0][0]
                self.assertEqual(req.method, "POST")
                self.assertIn("/files", req.full_url)

    def test_empty_reply_raises(self):
        raw = {"id": "r", "output": [{"content": [
            {"type": "output_text", "text": ""}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}
        client, _ = self._client(raw)
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "m.mp4"
            clip.write_bytes(b"x")
            with self.assertRaises(WonderClientError):
                client.judge_muted_hook(clip)

    # The live failure of 2026-10-29: the response carries reasoning
    # output only - no output_text item at all - because the token
    # budget was consumed before the answer. Reasoning length is
    # stochastic, so one bounded retry at a doubled budget absorbs
    # it; both attempts are billed, so both hit the ledger.
    _EMPTY_RAW = {
        "id": "resp-wonder-empty",
        "output": [{"type": "reasoning", "content": []}],
        "usage": {"input_tokens": 4000, "output_tokens": 2000},
    }
    _GOOD_RAW = {
        "id": "resp-wonder-1",
        "output": [{"content": [{
            "type": "output_text",
            "text": "SUMMARY: A musician's hook about rehearsals.",
        }]}],
        "usage": {"input_tokens": 5000, "output_tokens": 60},
    }

    def _client_sequence(self, raws, ledger=None):
        calls = []
        queue = list(raws)

        def upload(url, headers, body):
            calls.append(("upload", url))
            return 200, {"id": "file-123"}

        def transport(method, url, headers, body):
            calls.append(("responses", url, body))
            return 200, queue.pop(0)

        client = WonderClient(api_key="k", ledger=ledger,
                              transport=transport, upload_transport=upload)
        return client, calls

    def test_empty_first_attempt_retries_at_doubled_budget(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "l.jsonl", cap_usd=145.0)
            client, calls = self._client_sequence(
                [self._EMPTY_RAW, self._GOOD_RAW], ledger)
            clip = Path(td) / "muted.mp4"
            clip.write_bytes(b"fake")
            j = client.judge_muted_hook(clip, ts="2026-10-29")
            self.assertTrue(j.can_summarize)
            responses = [c for c in calls if c[0] == "responses"]
            self.assertEqual(len(responses), 2)
            budgets = [json.loads(c[2])["max_output_tokens"]
                       for c in responses]
            self.assertEqual(budgets, [2000, 4000])
            self.assertEqual(j.extra["empty_first_attempt_request_id"],
                             "resp-wonder-empty")
            self.assertEqual(j.extra["max_output_tokens_final"], 4000)
            entries = ledger.entries()
            self.assertEqual(len(entries), 2)
            self.assertTrue(
                entries[0].purpose.endswith("-empty-attempt"))
            self.assertEqual(entries[0].request_id, "resp-wonder-empty")

    def test_second_empty_reply_still_raises_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "l.jsonl", cap_usd=145.0)
            client, _ = self._client_sequence(
                [self._EMPTY_RAW, self._EMPTY_RAW], ledger)
            clip = Path(td) / "m.mp4"
            clip.write_bytes(b"x")
            with self.assertRaises(WonderClientError):
                client.judge_muted_hook(clip)
            entries = ledger.entries()
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(
                e.purpose.endswith("-empty-attempt") for e in entries))


# ---------------------------------------------------------------- F8

F8_N = 40  # 10 seconds at 4 fps: room for multi-graphic fixtures


def f8_inputs(video, graphics):
    return VerifyInputs(
        video_path=video,
        fps=FPS,
        canvas_alphas=speaker(n=F8_N),
        audio=profile([(0.0, 10.0)], total_s=10.0),
        hook=HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                      layout_rendered=True),
        cues=[CaptionCue("The band never rehearses.", 0.0, 3.5),
              CaptionCue("It is just perfect.", 3.5, 7.0),
              CaptionCue("And it keeps going on.", 7.0, 10.0)],
        caption_rect=initial_caption_rect(),
        graphics=graphics,
    )


def solid_spec(box, start, end, color=(200, 60, 120)):
    layer = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    layer[y0:y1, x0:x1, 0] = color[0]
    layer[y0:y1, x0:x1, 1] = color[1]
    layer[y0:y1, x0:x1, 2] = color[2]
    layer[y0:y1, x0:x1, 3] = 255
    return GraphicSpec(
        kind="card", start_s=start, end_s=end, box=box,
        layer_rgba=layer, headline="TEST CARD",
        source_quote="the band never rehearses", motion="static")


def path_spec(box, start, end, path="glide_in",
              color=(200, 60, 120)):
    """A GraphicSpec whose layer travels the declared path, built
    from the same geometry the renderer uses."""
    from PIL import Image as PILImage
    w, h = box[2] - box[0], box[3] - box[1]
    img = PILImage.new("RGBA", (w, h), color + (255,))

    def state_fn(t, s, e, _img=img, _box=box, _path=path):
        return _paste_layer_clipped(
            _img, path_box_at(_box, _path, t, s, CANVAS), CANVAS)

    return GraphicSpec(
        kind="card", start_s=start, end_s=end, box=box,
        layer_rgba=state_fn(end, start, end), headline="PATH CARD",
        source_quote="the band never rehearses", motion="static",
        path=path, _state_fn=state_fn)


def render_graphics_video(path, entries, n=F8_N):
    def gen():
        for i in range(n):
            t = i / FPS
            frame = np.zeros((CANVAS[1], CANVAS[0], 3), dtype=np.uint8)
            plane = graphics_layer_at(entries, t, CANVAS)
            if plane is not None:
                a = plane[..., 3:4].astype(np.float64) / 255.0
                frame = np.clip(
                    plane[..., :3].astype(np.float64) * a,
                    0, 255).astype(np.uint8)
            yield frame

    _encode_video(gen(), path, fps=FPS, canvas=CANVAS)
    return path


class GraphicsF8Tests(unittest.TestCase):
    def _run(self, td, specs, windows=None):
        """Render a video showing each spec in ``windows`` (default:
        its declared span) and verify."""
        windows = windows or [None] * len(specs)
        entries = [
            (spec.layer_at,
             spec.start_s if w is None else w[0],
             spec.end_s if w is None else w[1])
            for spec, w in zip(specs, windows)
        ]
        video = render_graphics_video(Path(td) / "render.mp4", entries)
        return verify(f8_inputs(video, specs), wonder=FakeWonder(),
                      ts="2026-10-30")

    def test_control_graphic_passes_everything(self):
        with tempfile.TemporaryDirectory() as td:
            spec = solid_spec((700, 300, 1000, 560), 3.5, 5.5)
            receipt = self._run(td, [spec])
            self.assertEqual(receipt.failed, [], msg=receipt.to_json())
            f8 = check(receipt, "F8")
            self.assertTrue(f8.passed)
            self.assertEqual(f8.measured["count"], 1)
            data = json.loads(receipt.to_json())
            self.assertEqual(len(data["graphics"]), 1)
            self.assertEqual(data["graphics"][0]["motion"], "static")
            self.assertEqual(data["graphics"][0]["box"],
                             [700, 300, 1000, 560])

    def test_graphic_over_face_fails_f8(self):
        with tempfile.TemporaryDirectory() as td:
            spec = solid_spec((300, 1050, 700, 1350), 3.5, 5.5)
            receipt = self._run(td, [spec])
            self.assertEqual(receipt.failed, ["F8"])
            f8 = check(receipt, "F8")
            self.assertGreater(
                f8.measured["graphics"][0]["face_overlap_px"], 0)
            self.assertEqual(f8.measured["offending_indices"], [0])

    def test_graphic_over_captions_fails_f8(self):
        with tempfile.TemporaryDirectory() as td:
            spec = solid_spec((200, 1500, 800, 1700), 3.5, 5.5)
            receipt = self._run(td, [spec])
            self.assertEqual(receipt.failed, ["F8"])
            f8 = check(receipt, "F8")
            self.assertGreater(
                f8.measured["graphics"][0]["caption_overlap_px"], 0)

    def test_graphic_outside_its_span_fails_f8(self):
        with tempfile.TemporaryDirectory() as td:
            spec = solid_spec((700, 300, 1000, 560), 3.5, 5.5)
            # The render shows it at 0.5-2.0s instead of its span.
            receipt = self._run(td, [spec], windows=[(0.5, 2.0)])
            self.assertEqual(receipt.failed, ["F8"])
            f8 = check(receipt, "F8")
            self.assertLess(
                f8.measured["graphics"][0]["presence_match_min"], 0.75)

    def test_travelling_graphic_on_path_passes(self):
        with tempfile.TemporaryDirectory() as td:
            spec = path_spec((700, 300, 1000, 560), 3.5, 6.5)
            receipt = self._run(td, [spec])
            self.assertEqual(receipt.failed, [], msg=receipt.to_json())
            g = check(receipt, "F8").measured["graphics"][0]
            self.assertEqual(g["path"], "glide_in")
            self.assertGreaterEqual(g["path_match"], 0.75)

    def test_graphic_parked_off_path_fails_f8(self):
        # The render parks the graphic at its resting box for the
        # whole span while the plan declares a glide: the travel
        # sample catches it where the resting presence cannot.
        with tempfile.TemporaryDirectory() as td:
            spec = path_spec((700, 300, 1000, 560), 3.5, 6.5)
            entries = [(lambda t, _s=spec: _s.layer_rgba,
                        spec.start_s, spec.end_s)]
            video = render_graphics_video(Path(td) / "render.mp4",
                                          entries)
            receipt = verify(f8_inputs(video, [spec]),
                             wonder=FakeWonder(), ts="2026-10-30")
            self.assertEqual(receipt.failed, ["F8"])
            g = check(receipt, "F8").measured["graphics"][0]
            self.assertLess(g["path_match"], 0.75)

    def test_graphic_inside_face_margin_fails_f8(self):
        # The box never touches a face pixel, but its right edge
        # sits 20px from the face's left edge across the span
        # (face left edge is 282 at the span's first sampled
        # frame) - inside the 28px margin zone placement
        # guarantees and F8 certifies.
        with tempfile.TemporaryDirectory() as td:
            spec = solid_spec((60, 1050, 262, 1300), 3.5, 5.5)
            receipt = self._run(td, [spec])
            self.assertEqual(receipt.failed, ["F8"])
            f8 = check(receipt, "F8")
            self.assertGreater(
                f8.measured["graphics"][0]["face_overlap_px"], 0)
            self.assertEqual(f8.measured["offending_indices"], [0])

    def test_graphic_in_hook_window_clear_of_glyphs_passes(self):
        # 2026-10-31 ruling: the hook window is open to graphics;
        # only the glyph zone is protected. This box sits in the
        # top-left, clear of the glyph bbox ((116,457)-(927,700)
        # for this fixture's hook, padded by 12) and of
        # everything else.
        with tempfile.TemporaryDirectory() as td:
            spec = solid_spec((60, 60, 360, 320), 1.0, 3.5)
            receipt = self._run(td, [spec])
            self.assertEqual(receipt.failed, [], msg=receipt.to_json())
            f8 = check(receipt, "F8")
            self.assertTrue(f8.passed)
            self.assertEqual(
                f8.measured["graphics"][0]["hook_glyph_overlap_px"], 0)

    def test_graphic_over_hook_glyphs_fails_f8(self):
        # Same window, but the box sits ON the hook glyphs:
        # F8's glyph-overlap sub-check fails it even though the
        # span's presence inside the hook window is itself legal.
        with tempfile.TemporaryDirectory() as td:
            spec = solid_spec((300, 500, 700, 650), 0.5, 3.0)
            receipt = self._run(td, [spec])
            self.assertEqual(receipt.failed, ["F8"])
            f8 = check(receipt, "F8")
            self.assertGreater(
                f8.measured["graphics"][0]["hook_glyph_overlap_px"], 0)
            self.assertEqual(f8.measured["offending_indices"], [0])

    def test_colliding_pair_fails_f8(self):
        # Spans overlap in time AND boxes intersect: F8's
        # collision sub-check fails the pair. Same colour on
        # purpose - in the intersection both layers are identical
        # pixels, so presence matches for both and only the
        # collision leg can fail.
        with tempfile.TemporaryDirectory() as td:
            a = solid_spec((700, 300, 1000, 560), 3.5, 6.0)
            b = solid_spec((880, 430, 1064, 660), 4.5, 7.0)
            receipt = self._run(td, [a, b])
            self.assertEqual(receipt.failed, ["F8"])
            f8 = check(receipt, "F8")
            self.assertEqual(len(f8.measured["collision_violations"]), 1)
            v = f8.measured["collision_violations"][0]
            self.assertEqual((v["earlier_index"], v["later_index"]),
                             (0, 1))
            self.assertEqual(v["overlap_px"], 120 * 130)

    def test_overlapping_spans_disjoint_boxes_pass(self):
        with tempfile.TemporaryDirectory() as td:
            a = solid_spec((700, 300, 1000, 560), 3.5, 6.0)
            b = solid_spec((60, 300, 360, 560), 4.5, 7.0,
                           color=(60, 120, 200))
            receipt = self._run(td, [a, b])
            self.assertEqual(receipt.failed, [], msg=receipt.to_json())
            self.assertTrue(check(receipt, "F8").passed)

    def test_back_to_back_pair_passes(self):
        # Gap exactly 0.0s - legal since the 2026-10-31 ruling.
        with tempfile.TemporaryDirectory() as td:
            a = solid_spec((700, 300, 1000, 560), 3.25, 4.75)
            b = solid_spec((60, 300, 360, 560), 4.75, 6.25,
                           color=(60, 120, 200))
            receipt = self._run(td, [a, b])
            self.assertEqual(receipt.failed, [], msg=receipt.to_json())
            self.assertTrue(check(receipt, "F8").passed)

    def test_spaced_pair_passes(self):
        with tempfile.TemporaryDirectory() as td:
            a = solid_spec((700, 300, 1000, 560), 3.25, 4.75)
            b = solid_spec((60, 300, 360, 560), 8.75, 10.0,
                           color=(60, 120, 200))
            receipt = self._run(td, [a, b])
            self.assertEqual(receipt.failed, [], msg=receipt.to_json())
            self.assertTrue(check(receipt, "F8").passed)

    def test_no_graphics_means_no_f8_and_no_section(self):
        receipt = verify(good_inputs(self._dummy()), wonder=FakeWonder())
        data = json.loads(receipt.to_json())
        self.assertEqual(len(data["checks"]), 7)
        self.assertNotIn("graphics", data)

    @classmethod
    def _dummy(cls):
        if not hasattr(cls, "_dummy_path"):
            td = tempfile.TemporaryDirectory()
            cls._dummy_td = td
            cls._dummy_path = make_video(Path(td.name) / "render.mp4")
        return cls._dummy_path


# ---------------------------------------------------------------- F8 backdrop


def backdrop_spec(start=3.5, end=6.5):
    from PIL import Image as PILImage
    spec = prepare_backdrop(
        PILImage.new("RGB", (240, 300), (200, 100, 50)), CANVAS)
    spec.start_s, spec.end_s = start, end
    spec.prompt = "a stadium tunnel at night - no text"
    spec.source_quote = "the band never rehearses"
    spec.cost_usd = 0.01
    return spec


def render_backdrop_video(path, spec, *, leak=False, n=F8_N):
    """Frames: a plain dark backdrop everywhere; the declared
    treated scene inside the span (or everywhere, when leaking).
    The speaker fixture's alpha region is painted a third colour -
    those pixels are excluded from the match anyway."""
    layer = spec.layer_at((spec.start_s + spec.end_s) / 2.0)
    scene_px = tuple(int(v) for v in layer[0, 0])
    alphas = speaker(n=n)

    def gen():
        for i in range(n):
            t = i / FPS
            inside = leak or (spec.start_s <= t < spec.end_s)
            frame = np.zeros((CANVAS[1], CANVAS[0], 3),
                             dtype=np.uint8)
            frame[:, :] = scene_px if inside else (10, 10, 10)
            frame[alphas[i] > 0.5] = (250, 250, 250)
            yield frame

    _encode_video(gen(), path, fps=FPS, canvas=CANVAS)
    return path


def f8b_inputs(video, backdrops):
    inputs = f8_inputs(video, [])
    inputs.backdrops = backdrops
    return inputs


class BackdropF8Tests(unittest.TestCase):
    def test_backdrop_inside_its_span_passes(self):
        with tempfile.TemporaryDirectory() as td:
            spec = backdrop_spec()
            video = render_backdrop_video(Path(td) / "r.mp4", spec)
            receipt = verify(f8b_inputs(video, [spec]),
                             wonder=FakeWonder(), ts="2026-10-31")
            self.assertEqual(receipt.failed, [],
                             msg=receipt.to_json())
            f8 = check(receipt, "F8")
            b = f8.measured["backdrops"][0]
            self.assertGreaterEqual(b["presence_match"], 0.75)
            self.assertLessEqual(b["absence_match_max"], 0.5)
            data = json.loads(receipt.to_json())
            self.assertEqual(len(data["backdrops"]), 1)
            self.assertEqual(data["backdrops"][0]["cost_usd"], 0.01)

    def test_backdrop_leaking_outside_its_span_fails_f8(self):
        with tempfile.TemporaryDirectory() as td:
            spec = backdrop_spec()
            video = render_backdrop_video(Path(td) / "r.mp4", spec,
                                          leak=True)
            receipt = verify(f8b_inputs(video, [spec]),
                             wonder=FakeWonder(), ts="2026-10-31")
            self.assertEqual(receipt.failed, ["F8"])
            f8 = check(receipt, "F8")
            self.assertEqual(
                f8.measured["offending_backdrop_indices"], [0])
            self.assertGreater(
                f8.measured["backdrops"][0]["absence_match_max"],
                0.5)

    def test_backdrop_missing_inside_its_span_fails_f8(self):
        with tempfile.TemporaryDirectory() as td:
            spec = backdrop_spec()
            video = render_backdrop_video(Path(td) / "r.mp4", spec)
            # Verify against a DIFFERENT declared scene: the render
            # cannot match a backdrop it never showed.
            other = backdrop_spec()
            from PIL import Image as PILImage
            treated = prepare_backdrop(
                PILImage.new("RGB", (240, 300), (20, 180, 90)),
                CANVAS)
            other._state_fn = treated._state_fn
            other.layer_rgb = treated.layer_rgb
            receipt = verify(f8b_inputs(video, [other]),
                             wonder=FakeWonder(), ts="2026-10-31")
            self.assertEqual(receipt.failed, ["F8"])
            self.assertLess(
                check(receipt, "F8")
                .measured["backdrops"][0]["presence_match"], 0.75)


if __name__ == "__main__":
    unittest.main()
