"""Tests for pipeline.py.

The selection/normalisation logic is exercised directly, and one
end-to-end smoke run drives the real ffmpeg/decoder/layout/render/
verifier code with fake API clients (fake SAM payload, fake VT
turns, fake Spark punctuation + hook, fake Spark Open sweep, fake
Wonder judge) at a small canvas so the suite stays fast. No network,
no spend.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from meta_sam_parser._mask_codec import _encode_raster  # noqa: E402

from spatial_director.caption_align import CaptionCue  # noqa: E402
from spatial_director.pipeline import (  # noqa: E402
    LazyCanvasAlphas,
    normalize_window,
    run_pipeline,
    select_windows,
)
from spatial_director.sam_client import SegmentationResult  # noqa: E402
from spatial_director.spark_client import ChatResult  # noqa: E402
from spatial_director.spark_open_client import (  # noqa: E402
    Candidate,
    SweepResult,
)
from spatial_director.spend_ledger import SpendLedger  # noqa: E402
from spatial_director.verifier import Receipt  # noqa: E402
from spatial_director.voice_transcribe import (  # noqa: E402
    TranscriptionResult,
    TranscriptTurn,
)
from spatial_director.wonder_client import WonderJudgment  # noqa: E402


def cand(start, end, score=0.5, reason="r"):
    return Candidate(start, end, score, reason)


class NormalizeWindowTests(unittest.TestCase):
    def test_short_span_grows_to_minimum(self):
        self.assertEqual(normalize_window(100, 105, 1000), (95.5, 109.5))

    def test_long_span_caps(self):
        self.assertEqual(normalize_window(100, 160, 1000), (100.0, 122.0))

    def test_clamps_to_source(self):
        self.assertEqual(normalize_window(-5, 1000, 60), (0.0, 22.0))

    def test_empty_returns_none(self):
        self.assertIsNone(normalize_window(50, 50, 1000))
        self.assertIsNone(normalize_window(1200, 1300, 1000))


class SelectWindowsTests(unittest.TestCase):
    def test_sorts_by_score_and_caps_count(self):
        chosen = select_windows(
            [cand(0, 20, 0.3), cand(100, 120, 0.9), cand(200, 220, 0.6)],
            1000, 2)
        self.assertEqual([c.start_s for c, _s, _e in chosen],
                         [100.0, 200.0])

    def test_dedupes_heavy_overlap(self):
        chosen = select_windows(
            [cand(100, 120, 0.9), cand(105, 125, 0.8)], 1000, 3)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0][0].start_s, 100.0)


class LazyCanvasAlphasTests(unittest.TestCase):
    def test_maps_and_caches(self):
        src = [np.zeros((4, 4)) for _ in range(3)]
        lazy = LazyCanvasAlphas(src, canvas=(16, 32))
        self.assertEqual(len(lazy), 3)
        first = lazy[0]
        self.assertEqual(first.shape, (32, 16))
        self.assertIs(first, lazy[0])
        self.assertEqual(len(list(lazy)), 3)


# ------------------------------------------------------- end-to-end smoke


def make_source(path: Path) -> None:
    """12s 160x90 @10fps clip: tone audio with one 1.2s hole at 4-5.2s."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=0x335577:s=160x90:r=10:d=12",
         "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=44100:duration=12",
         "-af", "volume=enable='between(t,4,5.2)':volume=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        check=True)


def sam_payload(n_frames: int, w: int = 160, h: int = 90) -> dict:
    """One speaker, box (65,2)-(95,48); the mask drifts 1px/frame
    inside its box so inter-frame IoU < 1 (the F7 motion signal).
    The box geometry is probed so the placed face centroid lands
    inside the verifier's central band (y ~1396 of [288, 1440])."""
    chunks = []
    for i in range(n_frames):
        d = i % 3
        raster = np.zeros((40, 30), dtype=bool)
        raster[2:38, 2 + d:26 + d] = True
        enc = _encode_raster(raster.astype(np.uint8).ravel().tolist(),
                             30, 40)
        chunks.append(
            f"<{i}f>0<|box;x1=65;y1=2;x2=95;y2=48;w={w};h={h}|>"
            f"<|mask;x=0;y=0;data=40,30,{enc}|>")
    return {"id": "resp-fake",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text",
                                     "text": "".join(chunks)}]}]}


class FakeSAM:
    def __init__(self):
        self.calls = 0

    def segment_video(self, *, file_id=None, video_url=None, concept="person",
                      n_frames, purpose="", ts=""):
        self.calls += 1
        return SegmentationResult(
            raw=sam_payload(n_frames), billed_frames=max(50, n_frames),
            request_id="resp-fake")


class FakeVT:
    def transcribe_file(self, wav_path, *, mode="DIARIZATION",
                        purpose="", ts=""):
        return TranscriptionResult(
            raw={},
            transcript="he never rehearsed the band played on anyway",
            turns=[TranscriptTurn(
                text="he never rehearsed the band played on anyway",
                start_ms=1500, end_ms=11000, speaker="SPEAKER_00")],
            audio_duration_ms=12000,
            session_id="sess-fake")


class FakeSpark:
    """Punctuation pass (echo the words back) + hook line + planner."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, *, max_tokens=1500, temperature=0.0,
             purpose="", ts=""):
        self.calls += 1
        system = messages[0]["content"] if messages else ""
        if "repair planner" in system:
            return ChatResult(
                raw={},
                text='{"repair": "TRIM_SILENCE", "reason": '
                     '"the pause is over the cap"}',
                prompt_tokens=100, completion_tokens=20,
                request_id="chatcmpl-fake")
        if "hook" in system.lower():
            text = "THE BAND NEVER REHEARSES"
        else:
            words = messages[-1]["content"].split(":", 1)[-1].strip()
            text = json.dumps({"text": words.capitalize() + "."})
        return ChatResult(raw={}, text=text, prompt_tokens=100,
                          completion_tokens=20,
                          request_id="chatcmpl-fake")


class FakeSweep:
    def sweep(self, turns, *, purpose="", ts="", max_tokens=4000):
        return SweepResult(
            candidates=[Candidate(0.0, 12.0, 0.9, "strong moment")],
            prompt_tokens=100, completion_tokens=50,
            request_id="sparkopen-fake")


class FakeWonder:
    def judge_muted_hook(self, path, *, purpose="", ts=""):
        return WonderJudgment(
            raw={}, summary="A speaker talks.", can_summarize=True,
            input_tokens=1000, output_tokens=100,
            request_id="resp-fake-judge")


class FakeSAMShortByOne(FakeSAM):
    """SAM returns one frame fewer than the container decodes - the
    660-vs-659 tail-artifact case from the first Holmes run."""

    def segment_video(self, *, file_id=None, video_url=None, concept="person",
                      n_frames, purpose="", ts=""):
        self.calls += 1
        return SegmentationResult(
            raw=sam_payload(n_frames - 1),
            billed_frames=max(50, n_frames - 1),
            request_id="resp-fake-short-one")


class EndToEndSmokeTests(unittest.TestCase):
    def test_full_run_with_fakes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "source.mp4"
            make_source(source)
            ledger = SpendLedger(td / "ledger.jsonl")
            sam = FakeSAM()
            summary = run_pipeline(
                source, td / "out",
                sam=sam, spark_open=FakeSweep(), spark=FakeSpark(),
                vt=FakeVT(), wonder=FakeWonder(), ledger=ledger,
                upload_file=lambda p: "file-fake",
                ts="2026-10-29", max_shorts=3,
                log=lambda _m: None)
            self.assertNotIn("error", summary)
            self.assertEqual(len(summary["shorts"]), 1)
            short = summary["shorts"][0]
            self.assertEqual(short["hook_text"], "THE BAND NEVER REHEARSES")
            # The source has a 1.2s hole at 4-5.2s: draft 1 fails F5,
            # the menu repair TRIM_SILENCE executes, the fake Wonder
            # judges draft 2 summarizable, and with the gap trimmed
            # the chain passes after repair.
            self.assertEqual(short["draft_1"]["verdict"], "FAIL")
            self.assertIn("F5", short["draft_1"]["failed_checks"])
            self.assertEqual(short["repair"]["menu_id"], "TRIM_SILENCE")
            self.assertEqual(short["final_verdict"], "PASSED_AFTER_REPAIR")
            self.assertEqual(sam.calls, 1)  # segmentation bought once
            final = Path(short["files"]["final"])
            self.assertTrue(final.exists())
            receipt = json.loads(
                Path(short["files"]["receipt"]).read_text())
            self.assertEqual(len(receipt["drafts"]), 2)
            # The fake clients never touch the ledger: run cost is
            # measured as ledger deltas, so it is exactly zero here.
            self.assertEqual(summary["total_cost_usd"], 0.0)
            self.assertTrue((td / "out" / "run-summary.json").exists())

    def test_sam_one_frame_short_aligns(self):
        # Regression for the first Holmes run stop: ffmpeg decoded 660
        # frames, SAM returned 659 (22.0s at 29.97fps = 659.34). The
        # pipeline must truncate the tail guide frame and complete,
        # not raise MaskDecodeError.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "source.mp4"
            make_source(source)
            ledger = SpendLedger(td / "ledger.jsonl")
            sam = FakeSAMShortByOne()
            logs: list[str] = []
            summary = run_pipeline(
                source, td / "out",
                sam=sam, spark_open=FakeSweep(), spark=FakeSpark(),
                vt=FakeVT(), wonder=FakeWonder(), ledger=ledger,
                upload_file=lambda p: "file-fake",
                ts="2026-10-29", max_shorts=3,
                log=logs.append)
            self.assertNotIn("error", summary)
            self.assertEqual(len(summary["shorts"]), 1)
            self.assertEqual(sam.calls, 1)
            self.assertTrue(any("aligning guides" in line
                                for line in logs))


if __name__ == "__main__":
    unittest.main()
