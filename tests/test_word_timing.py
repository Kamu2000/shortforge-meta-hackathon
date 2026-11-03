"""Tests for word_timing (extension Stage 4): the DSP floor, the
MMS adapter's fail-loud contract, and the dispatcher's honesty.

The load-bearing property is the LABEL: a timing is labelled with
the source that actually produced it. MMS timings are never
fabricated - when the aligner cannot run, source="mms" raises.
"""

import sys
import unittest
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import EnergyProfile  # noqa: E402
from spatial_director.word_timing import (  # noqa: E402
    MMSUnavailableError,
    WordTiming,
    WordTimingError,
    _word_spans_from_alignment,
    active_word_index,
    dsp_word_timings,
    word_timings,
)


def profile(spans, total_s=6.0, hop=0.02):
    n = int(total_s / hop)
    times = np.arange(n) * hop
    db = np.full(n, -80.0)
    for a, b in spans:
        db[(times >= a) & (times < b)] = -20.0
    return EnergyProfile(times=times, db=db, hop_s=hop)


class DspWordTimingTests(unittest.TestCase):
    def test_words_stay_on_speech_across_a_pause(self):
        # Speech 0-2s, silence 2-3s, speech 3-5s: no word may be
        # timed inside the pause, and the split covers both
        # speech islands in order.
        prof = profile([(0.0, 2.0), (3.0, 5.0)], total_s=5.0)
        out = dsp_word_timings(["alpha", "beta", "gamma", "delta"],
                               0.0, 5.0, prof)
        self.assertEqual([w.word for w in out],
                         ["alpha", "beta", "gamma", "delta"])
        self.assertTrue(all(w.source == "dsp" for w in out))
        for w in out:
            self.assertFalse(2.0 < w.start_s < 3.0 and
                             w.end_s < 3.0,
                             f"word inside pause: {w}")
            self.assertFalse(w.start_s >= 2.0 and w.end_s <= 3.0,
                             f"word inside pause: {w}")
        starts = [w.start_s for w in out]
        self.assertEqual(starts, sorted(starts))
        self.assertAlmostEqual(out[0].start_s, 0.0, places=2)
        self.assertAlmostEqual(out[-1].end_s, 5.0, places=2)

    def test_split_is_proportional_to_word_length(self):
        prof = profile([(0.0, 10.0)], total_s=10.0)
        out = dsp_word_timings(["a", "abcdefghij"], 0.0, 10.0, prof)
        d0 = out[0].end_s - out[0].start_s
        d1 = out[1].end_s - out[1].start_s
        self.assertAlmostEqual(d1 / d0, 10.0, delta=0.6)

    def test_uniform_fallback_without_profile_is_labelled(self):
        out = dsp_word_timings(["one", "two"], 1.0, 3.0, None)
        self.assertEqual(out[0].start_s, 1.0)
        self.assertEqual(out[-1].end_s, 3.0)
        self.assertIn("uniform", out[0].detail)

    def test_empty_words_and_bad_span(self):
        self.assertEqual(dsp_word_timings([], 0.0, 1.0, None), [])
        with self.assertRaises(WordTimingError):
            dsp_word_timings(["x"], 2.0, 1.0, None)


class FakeSpan(NamedTuple):
    start: int
    end: int
    score: float = 1.0


class AlignmentShapeTests(unittest.TestCase):
    def test_word_groups_become_word_spans(self):
        # The real torchaudio shape: one TokenSpan list per word.
        groups = [[FakeSpan(0, 4), FakeSpan(5, 9)],
                  [FakeSpan(10, 14)]]
        out = _word_spans_from_alignment(groups, ["hi", "there"])
        self.assertEqual(out, [(0, 9), (10, 14)])

    def test_group_count_mismatch_is_a_hard_error(self):
        groups = [[FakeSpan(0, 4)]]
        with self.assertRaises(WordTimingError):
            _word_spans_from_alignment(groups, ["hi", "there"])

    def test_no_spans_is_a_hard_error(self):
        with self.assertRaises(WordTimingError):
            _word_spans_from_alignment(None, ["hi"])


class MmsNormaliseTests(unittest.TestCase):
    def test_normalise_keeps_only_dictionary_chars(self):
        from spatial_director.word_timing import _mms_normalise
        self.assertEqual(_mms_normalise("Don't"), "don't")
        self.assertEqual(_mms_normalise("28,000"), "")
        self.assertEqual(_mms_normalise("ISS,"), "iss")


class MmsContractTests(unittest.TestCase):
    def test_mms_without_samples_is_an_error_not_a_guess(self):
        with self.assertRaises(WordTimingError):
            word_timings(["hello"], 0.0, 1.0, source="mms")

    def test_mms_unavailable_raises_never_substitutes(self):
        # Subprocess contract (fix phase, 2026-11-02): the aligner
        # runs under the isolated venv interpreter, so "cannot
        # run" is exercised by pointing the interpreter override
        # at a path that does not exist - the call must raise
        # MMSUnavailableError, never return DSP numbers.
        import os
        from unittest import mock
        samples = np.zeros(16000, dtype=np.float32)
        with mock.patch.dict(
                os.environ,
                {"SPATIAL_DIRECTOR_MMS_PYTHON":
                 "/nonexistent/mms-venv/bin/python"}):
            with self.assertRaises(MMSUnavailableError):
                word_timings(["hello", "world"], 0.0, 1.0,
                             source="mms", samples=samples)

    def test_unknown_source_is_an_error(self):
        with self.assertRaises(WordTimingError):
            word_timings(["hello"], 0.0, 1.0, source="whisper")


class MmsSubprocessTests(unittest.TestCase):
    """The subprocess boundary (fix phase, 2026-11-02): the main
    interpreter has no torch, so mms_word_timings shells into the
    isolated venv. subprocess.run is mocked here - the response
    file protocol, error mapping, and response validation are
    what these tests pin; the REAL end-to-end path is proven by
    tests/test_mms_pipeline_proof.py on cached audio."""

    def _run_with_payload(self, payload, words=("hello", "world")):
        import json as _json
        import os
        from pathlib import Path as _Path
        from types import SimpleNamespace
        from unittest import mock

        from spatial_director import word_timing as wt

        def fake_run(cmd, **kwargs):
            if payload is not None:
                _Path(cmd[3]).write_text(_json.dumps(payload))
            return SimpleNamespace(returncode=0, stderr="",
                                   stdout="")

        samples = np.zeros(16000, dtype=np.float32)
        with mock.patch.dict(
                os.environ,
                {"SPATIAL_DIRECTOR_MMS_PYTHON": "/bin/true"}):
            with mock.patch.object(wt.subprocess, "run", fake_run):
                return wt.mms_word_timings(list(words), samples,
                                           offset_s=2.0)

    def test_ok_response_becomes_mms_labelled_timings(self):
        out = self._run_with_payload(
            {"status": "ok", "word_spans": [[0, 25], [30, 49]],
             "n_frames": 50, "total_s": 1.0})
        self.assertEqual([t.word for t in out], ["hello", "world"])
        self.assertTrue(all(t.source == "mms" for t in out))
        # Frame spans convert against emission frames + offset.
        self.assertEqual((out[0].start_s, out[0].end_s), (2.0, 2.5))
        self.assertEqual((out[1].start_s, out[1].end_s), (2.6, 2.98))

    def test_word_error_maps_to_word_timing_error(self):
        with self.assertRaises(WordTimingError) as cm:
            self._run_with_payload(
                {"status": "error", "error_type": "word",
                 "message": "aligner span mismatch"})
        self.assertNotIsInstance(cm.exception, MMSUnavailableError)

    def test_unavailable_error_maps_to_mms_unavailable(self):
        with self.assertRaises(MMSUnavailableError):
            self._run_with_payload(
                {"status": "error", "error_type": "unavailable",
                 "message": "bundle could not be loaded"})

    def test_missing_response_is_unavailable(self):
        with self.assertRaises(MMSUnavailableError):
            self._run_with_payload(None)

    def test_span_count_mismatch_is_a_hard_error(self):
        with self.assertRaises(WordTimingError):
            self._run_with_payload(
                {"status": "ok", "word_spans": [[0, 25]],
                 "n_frames": 50, "total_s": 1.0})

    def test_out_of_bounds_span_is_a_hard_error(self):
        with self.assertRaises(WordTimingError):
            self._run_with_payload(
                {"status": "ok", "word_spans": [[0, 25], [30, 99]],
                 "n_frames": 50, "total_s": 1.0})

    def test_unrepresentable_word_fails_before_subprocess(self):
        import os
        from unittest import mock

        from spatial_director import word_timing as wt

        def forbidden_run(cmd, **kwargs):  # pragma: no cover
            raise AssertionError("subprocess must not be launched")

        samples = np.zeros(16000, dtype=np.float32)
        with mock.patch.dict(
                os.environ,
                {"SPATIAL_DIRECTOR_MMS_PYTHON": "/bin/true"}):
            with mock.patch.object(wt.subprocess, "run",
                                   forbidden_run):
                with self.assertRaises(WordTimingError):
                    wt.mms_word_timings(["hello", "28,000"], samples)

    def test_timeout_is_unavailable_not_a_hang(self):
        import os
        import subprocess as _sp
        from unittest import mock

        from spatial_director import word_timing as wt

        def slow_run(cmd, **kwargs):
            raise _sp.TimeoutExpired(cmd, kwargs.get("timeout", 0))

        samples = np.zeros(16000, dtype=np.float32)
        with mock.patch.dict(
                os.environ,
                {"SPATIAL_DIRECTOR_MMS_PYTHON": "/bin/true"}):
            with mock.patch.object(wt.subprocess, "run", slow_run):
                with self.assertRaises(MMSUnavailableError):
                    wt.mms_word_timings(["hello"], samples)


class ActiveWordTests(unittest.TestCase):
    def test_active_word_and_pause(self):
        timings = [
            WordTiming("one", 0.0, 0.5, "dsp"),
            WordTiming("two", 0.8, 1.3, "dsp"),
        ]
        self.assertEqual(active_word_index(timings, 0.2), 0)
        self.assertEqual(active_word_index(timings, 1.0), 1)
        self.assertIsNone(active_word_index(timings, 0.6))
        self.assertIsNone(active_word_index(timings, 2.0))


if __name__ == "__main__":
    unittest.main()
