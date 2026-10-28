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


if __name__ == "__main__":
    unittest.main()
