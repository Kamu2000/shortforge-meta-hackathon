"""Tests for the SAM-coverage window guard (fix phase, 2026-11-02).

Known defect: McArthur Short 3 failed twice - SAM returned masks
for only 155 of 659 window frames (a split-screen window), the
pipeline's tail-truncation quietly turned it into a ~5.2s stub,
and the stub failed downstream after the full finish spend. The
guard measures parse coverage (parsed frames / decoded frames)
right after SAM parsing, rejects a window below the floor BEFORE
compositing, and the run loop takes the next sweep candidate,
recording the rejection with its measured coverage.

Threshold derivation (pinned by test): passing windows measured
>= 650/656 = 0.991; the failing profile measured 155/659 = 0.235
in both McArthur graphics runs. MIN_SAM_COVERAGE_FRAC = 0.613 is
the midpoint of that empty band.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meta_sam_parser._mask_codec import _encode_raster  # noqa: E402
from spatial_director.mask_decode import parse_sam_response  # noqa: E402
from spatial_director.compositor import read_video_frames  # noqa: E402
from spatial_director.pipeline import (  # noqa: E402
    MIN_SAM_COVERAGE_FRAC,
    SamCoverageError,
    _next_untried_candidate,
    run_pipeline,
    sam_coverage,
)
from spatial_director.sam_client import SegmentationResult  # noqa: E402
from spatial_director.spark_client import ChatResult  # noqa: E402
from spatial_director.spark_open_client import (  # noqa: E402
    Candidate,
    SweepResult,
)
from spatial_director.spend_ledger import SpendLedger  # noqa: E402
from spatial_director.voice_transcribe import (  # noqa: E402
    TranscriptTurn,
    TranscriptionResult,
)
from spatial_director.wonder_client import WonderJudgment  # noqa: E402

FAILING = 155 / 659    # McArthur Short 3, both graphics runs
PASSING = 650 / 656    # lowest healthy window measured


class CoverageMathTests(unittest.TestCase):
    def test_coverage_ratio(self):
        self.assertAlmostEqual(sam_coverage(155, 659), FAILING)
        self.assertEqual(sam_coverage(660, 660), 1.0)
        self.assertEqual(sam_coverage(10, 0), 0.0)

    def test_threshold_sits_between_the_measured_clusters(self):
        self.assertGreater(MIN_SAM_COVERAGE_FRAC, FAILING)
        self.assertLess(MIN_SAM_COVERAGE_FRAC, PASSING)
        self.assertAlmostEqual(
            MIN_SAM_COVERAGE_FRAC, (FAILING + PASSING) / 2,
            places=2)

    def test_error_carries_the_measured_counts(self):
        err = SamCoverageError(155, 659, FAILING)
        self.assertEqual(err.parsed_frames, 155)
        self.assertEqual(err.window_frames, 659)
        self.assertAlmostEqual(err.coverage, FAILING)
        self.assertIn("155", str(err))


class ReplacementPickerTests(unittest.TestCase):
    CANDS = [Candidate(0.0, 15.0, 0.95, "a"),
             Candidate(15.0, 30.0, 0.90, "b"),
             Candidate(30.0, 45.0, 0.85, "c")]

    def test_picks_highest_ranked_untried_unreserved(self):
        picked = _next_untried_candidate(
            self.CANDS, 45.0,
            reserved=[(0.0, 15.0), (15.0, 30.0)],
            tried={(0.0, 15.0)})
        self.assertIsNotNone(picked)
        self.assertEqual((picked[1], picked[2]), (30.0, 45.0))

    def test_pool_exhaustion_returns_none(self):
        picked = _next_untried_candidate(
            self.CANDS, 45.0,
            reserved=[(0.0, 15.0), (15.0, 30.0), (30.0, 45.0)],
            tried={(0.0, 15.0)})
        self.assertIsNone(picked)


# ------------------------------------------------------- cached fixture

RUN_ROOTS = [
    Path.home() / "workspace" / "meta-hackathon"
    / "pipeline-run-graphics-mcarthur-2",
    Path.home() / "workspace" / "meta-hackathon" / "spike-footage"
    / "pipeline-run-graphics-holmes-6",
    Path.home() / "workspace" / "meta-hackathon" / "spike-footage"
    / "pipeline-run-graphics-belew-2",
]


def _cached_profiles():
    profiles = []
    for root in RUN_ROOTS:
        if not root.is_dir():
            continue
        for short_dir in sorted(root.glob("short-*")):
            sam = short_dir / "sam-response.json"
            win = short_dir / "window.mp4"
            if sam.is_file() and win.is_file():
                profiles.append((f"{root.name}/{short_dir.name}",
                                 sam, win))
    return profiles


@unittest.skipUnless(_cached_profiles(),
                     "cached run artifacts absent on this machine")
class CachedProfileFixtureTests(unittest.TestCase):
    """The guard against the REAL measured profiles: McArthur
    Short 3's coverage must fall below the floor; every passing
    Short's coverage must clear it."""

    def test_real_profiles(self):
        seen_failing = False
        for name, sam_path, win_path in _cached_profiles():
            parsed = parse_sam_response(
                json.loads(sam_path.read_text()))
            n_window = len(read_video_frames(win_path))
            cov = sam_coverage(len(parsed), n_window)
            if name.endswith("mcarthur-2/short-3"):
                seen_failing = True
                self.assertLess(cov, MIN_SAM_COVERAGE_FRAC, name)
                self.assertAlmostEqual(cov, FAILING, places=2)
            else:
                self.assertGreaterEqual(
                    cov, MIN_SAM_COVERAGE_FRAC, name)
        self.assertTrue(seen_failing,
                        "the McArthur short-3 fixture was not found")


# ------------------------------------------------------------- e2e fakes


def make_source(path: Path, duration: int = 45) -> None:
    """Tone source at the F9-calibrated gain (see test_pipeline)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi",
         "-i", f"color=c=0x335577:s=160x90:r=10:d={duration}",
         "-f", "lavfi",
         "-i", f"sine=frequency=440:sample_rate=44100:"
              f"duration={duration}",
         "-af", "volume=2.45",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        check=True)


def sam_payload(n_frames: int, w: int = 160, h: int = 90) -> dict:
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


class FakeSAMTruncatingFirst:
    """First window gets 20% SAM coverage; later windows full."""

    def __init__(self):
        self.calls = 0

    def segment_video(self, *, file_id=None, video_url=None,
                      concept="person", n_frames, purpose="", ts=""):
        self.calls += 1
        n = int(n_frames * 0.2) if self.calls == 1 else n_frames
        return SegmentationResult(
            raw=sam_payload(n), billed_frames=max(50, n),
            request_id="resp-fake")


class FakeVT:
    def transcribe_file(self, wav_path, *, mode="DIARIZATION",
                        purpose="", ts=""):
        return TranscriptionResult(
            raw={},
            transcript="he never rehearsed the band played on anyway",
            turns=[TranscriptTurn(
                text="he never rehearsed the band played on anyway",
                start_ms=500, end_ms=44500, speaker="SPEAKER_00")],
            audio_duration_ms=45000,
            session_id="sess-fake")


class FakeSpark:
    def chat(self, messages, *, max_tokens=1500, temperature=0.0,
             purpose="", ts=""):
        system = messages[0]["content"] if messages else ""
        if "repair planner" in system:
            return ChatResult(
                raw={}, text='{"repair": "TRIM_SILENCE", "reason": '
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
    def __init__(self, candidates):
        self._candidates = candidates

    def sweep(self, turns, *, purpose="", ts="", max_tokens=4000):
        return SweepResult(
            candidates=self._candidates,
            prompt_tokens=100, completion_tokens=50,
            request_id="sparkopen-fake")


class FakeWonder:
    def judge_muted_hook(self, path, *, purpose="", ts=""):
        return WonderJudgment(
            raw={}, summary="A speaker talks.", can_summarize=True,
            input_tokens=1000, output_tokens=100,
            request_id="resp-fake-judge")


CANDS_3 = [Candidate(0.0, 15.0, 0.95, "first"),
           Candidate(15.0, 30.0, 0.90, "second"),
           Candidate(30.0, 45.0, 0.85, "third")]


class GuardEndToEndTests(unittest.TestCase):
    def _run(self, candidates, max_shorts):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        source = root / "source.mp4"
        make_source(source)
        logs: list[str] = []
        summary = run_pipeline(
            source, root / "out",
            sam=FakeSAMTruncatingFirst(),
            spark_open=FakeSweep(candidates), spark=FakeSpark(),
            vt=FakeVT(), wonder=FakeWonder(),
            ledger=SpendLedger(root / "ledger.jsonl"),
            upload_file=lambda p: "file-fake",
            ts="2026-11-02", max_shorts=max_shorts,
            log=logs.append)
        return td, summary, logs

    def test_rejected_window_is_replaced_by_next_candidate(self):
        td, summary, _logs = self._run(CANDS_3, max_shorts=2)
        self.addCleanup(td.cleanup)
        self.assertNotIn("error", summary)
        self.assertEqual(len(summary["shorts"]), 2)
        windows = [(s["window"]["start_s"], s["window"]["end_s"])
                   for s in summary["shorts"]]
        self.assertEqual(windows, [(15.0, 30.0), (30.0, 45.0)])
        self.assertEqual(len(summary["window_rejections"]), 1)
        rec = summary["window_rejections"][0]
        self.assertEqual(rec["index"], 1)
        self.assertEqual(rec["window"],
                         {"start_s": 0.0, "end_s": 15.0})
        self.assertLess(rec["sam_coverage"], MIN_SAM_COVERAGE_FRAC)
        self.assertEqual(rec["parsed_frames"],
                         int(rec["window_frames"] * 0.2))

    def test_pool_exhaustion_proceeds_with_fewer_shorts(self):
        td, summary, logs = self._run(CANDS_3[:2], max_shorts=2)
        self.addCleanup(td.cleanup)
        self.assertNotIn("error", summary)
        self.assertEqual(len(summary["shorts"]), 1)
        self.assertEqual(len(summary["window_rejections"]), 1)
        self.assertTrue(any("candidate pool exhausted" in line
                            for line in logs))


if __name__ == "__main__":
    unittest.main()
