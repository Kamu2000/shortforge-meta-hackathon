"""Fix round 5 tests (2026-11-02): three user-flagged defects
from Kamlesh's QM6 review.

1. Caption timings were a synthetic grid wearing an "mms"
   label. Root cause, two layers: (a) _local_turns clipped a
   transcript turn's SPAN to the window but kept its FULL text
   (a 30s / 91-word turn in a 14s window -> constant ~6.9
   words/s); (b) the MMS aligner was fed each cue's own span
   and force-aligned it, so it could never contradict the
   layout, and no pace check audited the output. Fixes tested
   here: turn text is clipped with its span; alignment slices
   are bounded by DSP speech activity; a pace sanity gate
   rejects superhuman alignments (re-slice once, then an
   honestly-labelled DSP fallback per cue); per-cue provenance
   is recorded and the summary label reflects any mix; the
   valley snap never touches non-aligned (proportional / DSP)
   timings.
2. Spotlight composition: hero + PiP solved as one group
   (top offset, fixed gap, hero up to 85% of frame width),
   dominance caps intact.
3. Matte edge flicker: the temporal blend's gate signal is
   itself temporal - smoothed motion series + slew-limited
   hysteresis weights - so the weight cannot oscillate at
   frame rate.

Offline, synthetic fixtures only; the MMS aligner is stubbed
at the pipeline seam (no subprocess, no API calls).
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director import pipeline  # noqa: E402
from spatial_director.caption_align import (  # noqa: E402
    CaptionCue, energy_from_samples, speech_activity_span)
from spatial_director.voice_transcribe import TranscriptTurn  # noqa: E402
from spatial_director.word_timing import (  # noqa: E402
    WordTiming, pace_gate_failure, snap_word_timings)

SR = 16000


def burst_samples(bursts, total_s=4.0, seed=7):
    """Silence with loud noise bursts at [(t0, t1), ...]."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(total_s * SR), dtype=np.float32)
    for t0, t1 in bursts:
        a, b = int(t0 * SR), int(t1 * SR)
        x[a:b] = (rng.standard_normal(b - a) * 0.4).astype(np.float32)
    return x


# ------------------------------------------------------ defect 1a


class LocalTurnsClipTests(unittest.TestCase):
    """The turn's text is clipped with its span (root cause)."""

    def turn(self, n=10, start_ms=0, end_ms=30000):
        return TranscriptTurn(
            text=" ".join(f"w{i}" for i in range(n)),
            start_ms=start_ms, end_ms=end_ms)

    def test_tail_window_keeps_only_the_tail_words(self):
        local, excerpt = pipeline._local_turns(
            [self.turn()], 15.0, 30.0)
        self.assertEqual(len(local), 1)
        # Word centres at (i+0.5)*3s: 16.5..28.5 are w5..w9.
        self.assertEqual(local[0].text, "w5 w6 w7 w8 w9")
        self.assertEqual(excerpt, "w5 w6 w7 w8 w9")
        self.assertEqual(local[0].start_ms, 0)
        self.assertEqual(local[0].end_ms, 15000)

    def test_head_window_keeps_only_the_head_words(self):
        local, _ = pipeline._local_turns([self.turn()], 0.0, 12.0)
        self.assertEqual(local[0].text, "w0 w1 w2 w3")

    def test_turn_fully_inside_is_unchanged(self):
        t = self.turn(n=4, start_ms=5000, end_ms=9000)
        local, _ = pipeline._local_turns([t], 0.0, 20.0)
        self.assertEqual(local[0].text, "w0 w1 w2 w3")

    def test_overlap_without_word_centres_drops_the_turn(self):
        local, excerpt = pipeline._local_turns(
            [self.turn()], 14.0, 16.0)
        self.assertEqual(local, [])
        self.assertEqual(excerpt, "")


# ------------------------------------------------------ defect 1b


class PaceGateTests(unittest.TestCase):
    def timings(self, n, extent, start=0.0):
        step = extent / n
        return [WordTiming(f"w{i}", start + i * step,
                           start + (i + 1) * step, "mms")
                for i in range(n)]

    def test_two_x_compressed_alignment_is_rejected(self):
        # 8 words crammed into 0.9s of a 2.0s speech passage:
        # 8.9 words/s sustained - no human speaks this.
        reason = pace_gate_failure(self.timings(8, 0.9), 2.0)
        self.assertIsNotNone(reason)
        self.assertIn("words/s", reason)

    def test_short_extent_against_long_activity_is_rejected(self):
        # Pace alone passes (4 words over 1.0s) but the words
        # cover only half the passage's speech activity.
        reason = pace_gate_failure(self.timings(4, 1.0), 2.0)
        self.assertIsNotNone(reason)
        self.assertIn("covers less than", reason)

    def test_measured_pace_passes(self):
        self.assertIsNone(pace_gate_failure(self.timings(6, 2.0),
                                            2.2))

    def test_single_word_cue_passes_by_construction(self):
        self.assertIsNone(pace_gate_failure(self.timings(1, 0.1),
                                            2.0))


class SpeechActivitySpanTests(unittest.TestCase):
    def test_span_hugs_fine_segments_across_a_dip(self):
        samples = burst_samples([(0.2, 1.0), (1.3, 2.2)])
        profile = energy_from_samples(samples)
        span = speech_activity_span(profile, 0.0, 2.6)
        self.assertIsNotNone(span)
        self.assertAlmostEqual(span[0], 0.2, delta=0.1)
        self.assertAlmostEqual(span[1], 2.2, delta=0.1)


def _fake_aligner(mode):
    """Stub for pipeline.word_timings (the MMS seam).

    mode "compressed": words crammed into the first 45% of
    whatever slice it is handed (the forced-alignment failure
    this round fixes). mode "honest": words spread over the
    slice's middle 84%. mode "recover": compressed on the
    first call for a word set, honest on the re-slice.
    """
    calls = {"n": 0}

    def fake(words, t0, t1, *, source, samples=None,
             sample_rate=SR, **kw):
        if source != "mms":
            # The DSP fallback routes through the same seam;
            # delegate it to the real implementation.
            from spatial_director.word_timing import (
                word_timings as real_word_timings)
            return real_word_timings(words, t0, t1,
                                     source=source, **kw)
        calls["n"] += 1
        dur = t1 - t0
        n = len(words)
        if mode == "compressed" or (
                mode == "recover" and calls["n"] == 1):
            lo, hi = 0.0, 0.45 * dur
        elif mode == "recover":
            # The re-slice lands the words at their measured
            # positions across the passage (near-full spread).
            lo, hi = 0.05 * dur, 0.975 * dur
        else:
            lo, hi = 0.06 * dur, 0.90 * dur
        step = (hi - lo) / n
        return [WordTiming(w, t0 + lo + i * step,
                           t0 + lo + (i + 1) * step,
                           "mms") for i, w in enumerate(words)]

    return fake


class CaptionTimingHonestyTests(unittest.TestCase):
    """Through pipeline._caption_word_timings with a stubbed
    aligner on synthetic speech."""

    def setUp(self):
        self.samples = burst_samples([(0.2, 2.2), (2.6, 3.8)])
        self.profile = energy_from_samples(self.samples)
        self.cue_a = CaptionCue(
            "one two three four five six seven eight", 0.2, 2.2)
        self.cue_b = CaptionCue("nine ten eleven twelve", 2.6, 3.8)

    def run_timings(self, cues, mode, requested="mms"):
        with mock.patch.object(pipeline, "load_samples",
                               return_value=self.samples), \
                mock.patch.object(pipeline, "word_timings",
                                  _fake_aligner(mode)):
            return pipeline._caption_word_timings(
                cues, self.profile, "unused.wav", requested,
                lambda m: None)

    def test_compressed_alignment_falls_back_per_cue(self):
        result = self.run_timings([self.cue_a], "compressed")
        self.assertEqual(result.cue_sources, ["dsp_phrase"])
        self.assertEqual(result.source, "dsp")
        actions = [r["action"] for r in result.gate_records]
        self.assertEqual(actions, ["pace_gate_rejected",
                                   "dsp_fallback"])
        # The fallback timings are DSP-labelled, never mms.
        self.assertTrue(all(w.source == "dsp"
                            for w in result.timings[0]))

    def test_reslice_recovers_an_honest_alignment(self):
        result = self.run_timings([self.cue_a], "recover")
        self.assertEqual(result.cue_sources, ["mms"])
        self.assertEqual(result.source, "mms")
        actions = [r["action"] for r in result.gate_records]
        self.assertEqual(actions, ["pace_gate_rejected",
                                   "realigned_on_dsp_span"])

    def test_mixed_short_is_labelled_as_a_mix(self):
        # cue_a honest (8 words, wide slice), cue_b compressed:
        # a per-word-count stub - compressed for 4-word cues.
        def fake(words, t0, t1, *, source, samples=None,
                 sample_rate=SR, **kw):
            if source != "mms":
                from spatial_director.word_timing import (
                    word_timings as real_word_timings)
                return real_word_timings(words, t0, t1,
                                         source=source, **kw)
            dur = t1 - t0
            n = len(words)
            if n == 4:
                lo, hi = 0.0, 0.30 * dur
            else:
                lo, hi = 0.06 * dur, 0.90 * dur
            step = (hi - lo) / n
            return [WordTiming(w, t0 + lo + i * step,
                               t0 + lo + (i + 1) * step, "mms")
                    for i, w in enumerate(words)]

        with mock.patch.object(pipeline, "load_samples",
                               return_value=self.samples), \
                mock.patch.object(pipeline, "word_timings", fake):
            result = pipeline._caption_word_timings(
                [self.cue_a, self.cue_b], self.profile,
                "unused.wav", "mms", lambda m: None)
        self.assertEqual(result.cue_sources, ["mms", "dsp_phrase"])
        self.assertEqual(result.source, "mms+dsp")

    def test_dsp_request_never_wears_the_mms_label(self):
        result = self.run_timings([self.cue_a, self.cue_b],
                                  "honest", requested="dsp")
        self.assertEqual(result.source, "dsp")
        self.assertEqual(result.cue_sources,
                         ["dsp_phrase", "dsp_phrase"])
        self.assertEqual(result.gate_records, [])


class SnapProvenanceTests(unittest.TestCase):
    def test_snap_moves_only_aligned_words(self):
        # Noise with silence dips at [0.30,0.34] and [0.62,0.66]:
        # word starts sit ~60ms BEFORE each dip.
        samples = burst_samples([(0.0, 0.30), (0.34, 0.62),
                                 (0.66, 1.2)], total_s=1.2)
        cues = [[
            WordTiming("grid", 0.24, 0.29, "dsp"),
            WordTiming("real", 0.56, 0.61, "mms"),
        ]]
        snapped, stats = snap_word_timings(
            cues, samples, snap_sources=("mms",))
        self.assertEqual(stats["words_eligible"], 1)
        # The proportional (dsp) word is untouched...
        self.assertEqual(snapped[0][0].start_s, 0.24)
        # ...the aligned word moved toward its valley.
        self.assertGreater(snapped[0][1].start_s, 0.56)


# -------------------------------------------------------- defect 2


class SpotlightGroupTests(unittest.TestCase):
    def test_group_composition_geometry(self):
        from PIL import Image

        from spatial_director.graphics_layer import (
            SPOTLIGHT_GROUP_GAP_PX, SPOTLIGHT_GROUP_TOP_PX,
            GraphicSpan, GraphicSpec, plan_layout, prepare_image)

        img = Image.new("RGB", (1296, 972), (200, 120, 40))
        span = GraphicSpan(
            start_s=4.0, end_s=7.0, kind="image", headline="",
            facts="", prompt="p", source_quote="q",
            motion="static", layout="spotlight")
        content = prepare_image(img, span, box=(1296, 972))
        layer = np.zeros((1920, 1080, 4), dtype=np.uint8)
        layer[0:540, 0:960, 3] = 255
        s = GraphicSpec(
            kind="image", start_s=4.0, end_s=7.0,
            box=(0, 0, 960, 540), layer_rgba=layer, headline="",
            source_quote="q", motion="static", layout="spotlight")
        s._content = content

        alphas = []
        proto = np.zeros((583, 640), dtype=np.float32)
        proto[77:, 254:476] = 1.0
        alphas = [proto.copy() for _ in range(360)]
        _track, decls, _rects = plan_layout(
            [s], alphas, fps=30.0, canvas=(1080, 1920),
            n_frames=360)
        decl = decls[0]
        self.assertEqual(decl["layout"], "spotlight")
        self.assertTrue(decl["feasible"], decl["note"])
        gb, pb = decl["graphic_box"], decl["speaker_box"]
        gw, gh = gb[2] - gb[0], gb[3] - gb[1]
        pw, ph = pb[2] - pb[0], pb[3] - pb[1]
        # Group composition: offset top, fixed small gap.
        self.assertEqual(gb[1], SPOTLIGHT_GROUP_TOP_PX)
        self.assertEqual(pb[1] - gb[3], SPOTLIGHT_GROUP_GAP_PX)
        self.assertLessEqual(pb[1] - gb[3], 80)
        # Hero grown toward the group target; caps intact.
        self.assertGreaterEqual(gw, 800)
        self.assertLessEqual(pw, gw)
        self.assertLessEqual(pw * ph, 0.92 * gw * gh)
        self.assertIn("group", decl["spotlight_hero"])


# -------------------------------------------------------- defect 3


class TemporalGateTests(unittest.TestCase):
    def test_weights_are_slew_limited_under_oscillation(self):
        from spatial_director.temporal_smooth import (
            BAND_WEIGHT_SLEW, gate_weight_series)
        motions = [0.0] + [0.02, 0.12] * 30
        _smoothed, weights = gate_weight_series(motions)
        steps = [abs(weights[i] - weights[i - 1])
                 for i in range(1, len(weights))]
        self.assertLessEqual(max(steps), BAND_WEIGHT_SLEW + 1e-9)

    def test_weight_variance_collapses_vs_raw_gate(self):
        from spatial_director.temporal_smooth import (
            BAND_BLEND_MAX, BAND_MOTION_HI, BAND_MOTION_LO,
            gate_weight_series)
        motions = [0.0] + [0.045, 0.095] * 30
        raw_targets = [
            BAND_BLEND_MAX * min(1.0, max(
                0.0, (m - BAND_MOTION_LO)
                / (BAND_MOTION_HI - BAND_MOTION_LO)))
            for m in motions]
        _s, weights = gate_weight_series(motions)
        self.assertLess(float(np.std(weights)),
                        0.35 * float(np.std(raw_targets)))

    def test_sustained_motion_reaches_full_blend(self):
        from spatial_director.temporal_smooth import (
            BAND_BLEND_MAX, gate_weight_series)
        _s, weights = gate_weight_series([0.10] * 20)
        self.assertEqual(weights[-1], BAND_BLEND_MAX)

    def test_midrange_floor_strengthens_the_gate(self):
        # Continuous motion sitting inside the gate's ramp
        # (0.07, between LO 0.055 and HI 0.086): the unfloored
        # ramp target is ~0.24, but the production shaping
        # floors the target at 0.8 x cap = 0.40 - the pairwise
        # gate idles near zero in exactly this regime on the
        # real McArthur window (median band motion = LO).
        from spatial_director.temporal_smooth import (
            BAND_GATE_MID_FLOOR_FRAC, BAND_BLEND_MAX,
            gate_weight_series)
        _s, plain = gate_weight_series([0.07] * 30)
        self.assertLess(plain[-1], 0.30)
        _s, floored = gate_weight_series(
            [0.07] * 30, floor_frac=BAND_GATE_MID_FLOOR_FRAC,
            slew_up=0.25, slew_down=0.04)
        self.assertAlmostEqual(
            floored[-1], BAND_GATE_MID_FLOOR_FRAC * BAND_BLEND_MAX,
            places=3)
        # And the release is slow: three still frames in, the
        # weight is still engaged, where the instantaneous
        # pairwise target is already 0 - the blend cannot drop
        # out and snap the edge on a momentary motion dip.
        _s, rel = gate_weight_series(
            [0.07] * 20 + [0.0] * 3,
            floor_frac=BAND_GATE_MID_FLOOR_FRAC,
            slew_up=0.25, slew_down=0.04)
        self.assertGreater(rel[-1], 0.10)

    def test_still_sequence_is_identity(self):
        from spatial_director.temporal_smooth import (
            edge_band_temporal_blend_sequence)
        base = np.zeros((40, 60))
        base[10:30, 20:40] = 1.0
        frames = [base.copy() for _ in range(12)]
        out, stats = edge_band_temporal_blend_sequence(
            [f.copy() for f in frames])
        self.assertEqual(stats["frames_blended"], 0)
        for f in out:
            self.assertTrue(np.array_equal(f, base))


# ------------------------------------------------ receipt additivity


class ReceiptProvenanceTests(unittest.TestCase):
    def test_cue_sources_field_is_additive(self):
        from spatial_director.verifier import Receipt
        r = Receipt(artifact="x", verdict="PASS", checks=[])
        self.assertNotIn("caption_cue_sources", r.to_dict())
        r2 = Receipt(artifact="x", verdict="PASS", checks=[],
                     caption_cue_sources=["mms", "dsp_phrase"],
                     caption_timing_gate=[{"cue_index": 1,
                                           "action": "dsp_fallback",
                                           "reason": "x"}])
        d = r2.to_dict()
        self.assertEqual(d["caption_cue_sources"],
                         ["mms", "dsp_phrase"])
        self.assertEqual(d["caption_timing_gate"][0]["action"],
                         "dsp_fallback")
        self.assertEqual(d["receipt_version"], 1)


if __name__ == "__main__":
    unittest.main()
