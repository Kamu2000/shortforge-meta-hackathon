"""Tests for the block-2 SFX layer: procedural synthesis,
ducked mixing, plan derivation from placed graphics, and the
verifier's stem discipline (F5 measures the pre-mix speech stem;
F8 checks gap-landing effects are present in the mix). Offline -
synthetic profiles and samples; no network, no spend.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue, EnergyProfile  # noqa: E402
from spatial_director.graphics_layer import GraphicSpec  # noqa: E402
from spatial_director.layout_law import initial_caption_rect  # noqa: E402
from spatial_director.sfx import (  # noqa: E402
    RECIPE_CEILING_DB,
    SAMPLE_RATE,
    SfxEvent,
    mix_sfx,
    mix_to_file,
    plan_sfx,
    synthesize,
    write_wav,
)
from spatial_director.verifier import HookSpec, VerifyInputs, verify  # noqa: E402

CANVAS = (1080, 1920)
FPS = 4.0
N = 16


def make_video(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=black:s=320x240:r=10:d=1",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True)
    return path


def speaker(n=N):
    alphas = []
    for i in range(n):
        a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
        a[1000:1920, 240 + 3 * i:840 + 3 * i] = 1.0
        alphas.append(a)
    return alphas


def profile(spans, total_s=4.0, hop=0.02, level=-20.0):
    n = int(total_s / hop)
    times = np.arange(n) * hop
    db = np.full(n, -80.0)
    for a, b in spans:
        db[(times >= a) & (times < b)] = level
    return EnergyProfile(times=times, db=db, hop_s=hop)


class FakeJudgment:
    can_summarize = True
    summary = "A musician explains his band never rehearses."
    cost_usd = 0.0
    request_id = "fake"


class FakeWonder:
    def judge_muted_hook(self, path, *, purpose="", ts=""):
        return FakeJudgment()


def spec(motion="static", path=None, start=4.0, end=7.0,
         kind="card"):
    layer = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
    return GraphicSpec(
        kind=kind, start_s=start, end_s=end, box=(60, 60, 300, 300),
        layer_rgba=layer, motion=motion, path=path,
        headline="X", source_quote="q")


def check(receipt, check_id):
    return next(c for c in receipt.checks if c.check_id == check_id)


# ---------------------------------------------------------------- synthesis


class SynthesizeTests(unittest.TestCase):
    def test_recipes_peak_normalised_and_deterministic(self):
        for recipe in RECIPE_CEILING_DB:
            w1 = synthesize(recipe)
            w2 = synthesize(recipe)
            self.assertGreater(len(w1), 0, recipe)
            self.assertAlmostEqual(float(np.max(np.abs(w1))), 1.0,
                                   places=5, msg=recipe)
            self.assertTrue(np.array_equal(w1, w2), recipe)

    def test_unknown_recipe_raises(self):
        with self.assertRaises(ValueError):
            synthesize("orchestra_hit")


class PlanTests(unittest.TestCase):
    def test_static_card_gets_a_pop(self):
        events = plan_sfx([spec()])
        self.assertEqual([(e.recipe, e.anchor) for e in events],
                         [("pop", "enter")])

    def test_count_up_gets_ticks_and_completion_ding(self):
        events = plan_sfx([spec(motion="count_up")])
        recipes = [e.recipe for e in events]
        self.assertEqual(recipes.count("tick"), 6)
        self.assertEqual(recipes[-1], "ding")
        self.assertEqual(events[-1].anchor, "count_complete")
        times = [e.t_s for e in events]
        self.assertEqual(times, sorted(times))

    def test_path_motion_gets_whoosh_and_landing_ding(self):
        events = plan_sfx([spec(path="glide_in")])
        self.assertEqual(
            [(e.recipe, round(e.t_s, 3)) for e in events],
            [("whoosh", 4.0), ("ding", 4.9)])

    def test_indices_follow_the_spec_list(self):
        events = plan_sfx([spec(), spec(start=9.0, end=11.0)])
        self.assertEqual({e.graphic_index for e in events}, {0, 1})


class MixTests(unittest.TestCase):
    def test_silence_uses_the_absolute_ceiling(self):
        speech = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        _mix, applied = mix_sfx(
            speech, SAMPLE_RATE, [SfxEvent(0.5, "pop", 0)])
        # pop ceiling is -13, silence ceiling is -16: lower wins.
        self.assertEqual(applied[0]["gain_db"], -16.0)

    def test_ducking_caps_below_loud_speech(self):
        t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        speech = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        _mix, applied = mix_sfx(
            speech, SAMPLE_RATE, [SfxEvent(0.5, "pop", 0)])
        # Speech RMS is ~-11 dBFS, so the duck cap is ~-15 dBFS:
        # clearly below the pop recipe's -13 ceiling.
        self.assertLess(applied[0]["gain_db"],
                        RECIPE_CEILING_DB["pop"] - 1.0)
        self.assertGreater(applied[0]["gain_db"], -16.5)

    def test_event_audible_in_gap_and_mix_deterministic(self):
        speech = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        events = [SfxEvent(1.0, "whoosh", 0)]
        mix1, _ = mix_sfx(speech, SAMPLE_RATE, events)
        mix2, _ = mix_sfx(speech, SAMPLE_RATE, events)
        self.assertTrue(np.array_equal(mix1, mix2))
        gap = mix1[int(1.1 * SAMPLE_RATE):int(1.3 * SAMPLE_RATE)]
        self.assertGreater(float(np.max(np.abs(gap))), 0.05)
        self.assertEqual(len(mix1), len(speech))

    def test_mix_to_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            stem = Path(td) / "stem.wav"
            write_wav(stem, np.zeros(SAMPLE_RATE * 2,
                                     dtype=np.float32))
            applied = mix_to_file(
                stem, [SfxEvent(0.5, "ding", 0)], Path(td) / "mix.wav")
            self.assertEqual(len(applied), 1)
            self.assertTrue((Path(td) / "mix.wav").exists())
            self.assertEqual(
                mix_to_file(stem, [], Path(td) / "none.wav"), [])


# ---------------------------------------------------------------- verifier


class StemDisciplineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.video = make_video(Path(cls._td.name) / "render.mp4")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def inputs(self, **overrides):
        base = VerifyInputs(
            video_path=self.video, fps=FPS, canvas_alphas=speaker(),
            audio=profile([(0.0, 4.0)]),
            hook=HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                          layout_rendered=True),
            cues=[CaptionCue("The band never rehearses.", 0.0, 2.0),
                  CaptionCue("It is just perfect.", 2.0, 4.0)],
            caption_rect=initial_caption_rect())
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def test_sfx_filled_gap_still_fails_f5_on_the_stem(self):
        # The mix (inputs.audio) is continuous - an effect fills
        # the gap audibly - but the STEM has a 1.5s hole: F5 must
        # fail on the stem while F1 (mix onset/RMS) stays clean.
        receipt = verify(
            self.inputs(
                speech_audio=profile([(0.0, 1.0), (2.5, 4.0)]),
                graphics=[],
                sfx=[{"t_s": 1.7, "recipe": "whoosh",
                      "gain_db": -16.0, "graphic_index": 0,
                      "anchor": "enter"}]),
            wonder=FakeWonder())
        f5 = check(receipt, "F5")
        self.assertFalse(f5.passed)
        self.assertAlmostEqual(f5.measured["max_silence_gap_s"], 1.5,
                               delta=0.05)
        self.assertEqual(f5.measured["measured_on"], "speech_stem")
        self.assertTrue(check(receipt, "F1").passed)

    def test_clean_stem_passes(self):
        receipt = verify(
            self.inputs(speech_audio=profile([(0.0, 4.0)]),
                        graphics=[], sfx=[]),
            wonder=FakeWonder())
        self.assertEqual(receipt.failed, [], msg=receipt.to_json())
        self.assertEqual(receipt.to_dict()["sfx"], [])

    def test_no_stem_keeps_historic_measurement(self):
        receipt = verify(
            self.inputs(audio=profile([(0.0, 1.0), (2.5, 4.0)])),
            wonder=FakeWonder())
        f5 = check(receipt, "F5")
        self.assertFalse(f5.passed)
        self.assertEqual(f5.measured["measured_on"], "render_audio")

    def test_f8_gap_event_missing_from_mix_fails(self):
        # The declared effect lands where the stem is silent, and
        # the mix is silent there too: it is not in the render.
        receipt = verify(
            self.inputs(
                audio=profile([(0.0, 1.9), (2.1, 4.0)]),
                speech_audio=profile([(0.0, 1.9), (2.1, 4.0)]),
                graphics=[],
                sfx=[{"t_s": 2.0, "recipe": "pop",
                      "gain_db": -16.0, "graphic_index": 0,
                      "anchor": "enter"}]),
            wonder=FakeWonder())
        f8 = check(receipt, "F8")
        self.assertFalse(f8.passed)
        self.assertEqual(len(f8.measured["sfx"]["violations"]), 1)
        self.assertEqual(f8.measured["sfx"]["gap_events_checked"], 1)

    def test_f8_gap_event_present_in_mix_passes(self):
        receipt = verify(
            self.inputs(
                audio=profile([(0.0, 4.0)]),
                speech_audio=profile([(0.0, 1.9), (2.1, 4.0)]),
                graphics=[],
                sfx=[{"t_s": 2.0, "recipe": "pop",
                      "gain_db": -16.0, "graphic_index": 0,
                      "anchor": "enter"}]),
            wonder=FakeWonder())
        f8 = check(receipt, "F8")
        self.assertTrue(f8.passed, msg=str(f8.measured))
        self.assertEqual(f8.measured["sfx"]["gap_events_checked"], 1)
        self.assertEqual(f8.measured["sfx"]["violations"], [])


if __name__ == "__main__":
    unittest.main()
