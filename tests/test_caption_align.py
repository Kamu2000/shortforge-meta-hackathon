"""Tests for caption_align + the Spark 1.3 client it rides on.

Load-bearing properties: (1) the punctuation pass may dress words
but never change them - a reply that alters the word sequence falls
back to the raw transcript; (2) phrase timing comes from DSP energy,
so words consume ACTIVE time only and boundaries snap into pauses;
(3) the Spark client is ledger-gated and fails loud on empty replies
(the reasoning-burn failure class).
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import (  # noqa: E402
    EnergyProfile,
    active_segments,
    align_turn,
    build_captions,
    cues_to_srt,
    energy_from_samples,
    split_phrases,
    words_unchanged,
)
from spatial_director.spark_client import (  # noqa: E402
    SparkClient,
    SparkClientError,
    cost_from_usage,
)
from spatial_director.spend_ledger import SpendLedger, SpendCapError  # noqa: E402
from spatial_director.voice_transcribe import TranscriptTurn  # noqa: E402


def profile_from_spans(spans, total_s=20.0, hop=0.02, speech_db=-20.0,
                       silence_db=-80.0):
    """Synthetic energy profile: speech in ``spans``, silence elsewhere."""
    n = int(total_s / hop)
    times = np.arange(n) * hop
    db = np.full(n, silence_db)
    for a, b in spans:
        db[(times >= a) & (times < b)] = speech_db
    return EnergyProfile(times=times, db=db, hop_s=hop)


class WordGuardTests(unittest.TestCase):
    def test_punctuation_only_passes(self):
        raw = "the the interesting thing about it its an 11 piece band"
        restored = "The the interesting thing about it: it's an 11-piece band."
        # Hyphenation splits a token; apostrophe attaches - normalise:
        self.assertTrue(words_unchanged(
            raw, "The the interesting thing about it, it's an 11 piece band."))

    def test_word_change_fails(self):
        raw = "we dont even rehearse or anything"
        self.assertFalse(words_unchanged(
            raw, "We don't even rehearse or anything else."))
        self.assertFalse(words_unchanged(
            raw, "We do not even rehearse or anything."))
        # The classic ASR "cleanup": dropping a duplicated word.
        self.assertFalse(words_unchanged("the the band", "The band."))


class PhraseSplitTests(unittest.TestCase):
    def test_sentence_split(self):
        text = "It is an 11 piece band. We never rehearse. Ever."
        self.assertEqual(split_phrases(text),
                         ["It is an 11 piece band.", "We never rehearse.",
                          "Ever."])

    def test_long_sentence_splits_at_commas_and_cap(self):
        text = ("we show up and we do the show, we don't even rehearse "
                "or anything, and every time it is just perfect")
        phrases = split_phrases(text)
        self.assertGreater(len(phrases), 1)
        for p in phrases:
            self.assertLessEqual(len(p.split()), 8)
        self.assertEqual(" ".join(phrases).split(), text.split())


class EnergyTests(unittest.TestCase):
    def test_active_segments_merge_short_gaps(self):
        prof = profile_from_spans([(1.0, 3.0), (3.1, 5.0), (8.0, 9.0)])
        segs = active_segments(prof, 0.0, 20.0)
        self.assertEqual(len(segs), 2)  # 0.1s gap merged, 3s gap not
        self.assertAlmostEqual(segs[0][0], 1.0, delta=0.03)
        self.assertAlmostEqual(segs[0][1], 5.0, delta=0.03)

    def test_energy_from_samples(self):
        sr = 16000
        samples = np.concatenate([
            np.zeros(sr // 2, dtype=np.float32),
            np.full(sr // 2, 0.5, dtype=np.float32),
        ])
        prof = energy_from_samples(samples, sr)
        self.assertLess(prof.db[2], -60.0)     # silence half
        self.assertGreater(prof.db[-2], -10.0)  # loud half


class AlignTests(unittest.TestCase):
    def test_words_consume_active_time_only(self):
        # Speech 0-2s and 4-6s (2s pause between). 4 words, 2 phrases.
        prof = profile_from_spans([(0.0, 2.0), (4.0, 6.0)], total_s=6.0)
        cues = align_turn(["one two", "three four"], 0.0, 6.0, prof)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0].start_s, 0.0, delta=0.05)
        self.assertAlmostEqual(cues[0].end_s, 2.0, delta=0.4)
        # Second phrase lives entirely after the pause.
        self.assertGreaterEqual(cues[1].start_s, 3.6)
        self.assertAlmostEqual(cues[1].end_s, 6.0, delta=0.05)

    def test_no_profile_falls_back_to_uniform(self):
        cues = align_turn(["one two three four"], 1.0, 5.0, None)
        self.assertAlmostEqual(cues[0].start_s, 1.0)
        self.assertAlmostEqual(cues[0].end_s, 5.0)

    def test_cues_ordered_inside_turn(self):
        prof = profile_from_spans([(0.0, 10.0)], total_s=10.0)
        phrases = split_phrases("One two three. Four five six. Seven.")
        cues = align_turn(phrases, 0.0, 10.0, prof)
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a.end_s, b.start_s + 0.001)
            self.assertGreater(b.end_s, b.start_s)


class FakeSparkTransport:
    def __init__(self, text, status=200):
        self.text = text
        self.status = status
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        if self.status != 200:
            return self.status, {"error": "nope"}
        return 200, {
            "id": "chatcmpl-test-1",
            "choices": [{"message": {"content": self.text}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40},
        }


class SparkClientTests(unittest.TestCase):
    def test_cost_math(self):
        self.assertAlmostEqual(cost_from_usage(1_000_000, 0), 1.25)
        self.assertAlmostEqual(cost_from_usage(0, 1_000_000), 4.25)
        self.assertAlmostEqual(
            cost_from_usage(1_000_000, 0, cached_tokens=1_000_000), 0.15)

    def test_chat_shape_and_ledger(self):
        ft = FakeSparkTransport("Hello, world.")
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "l.jsonl", cap_usd=145.0)
            client = SparkClient(api_key="k", ledger=ledger, transport=ft)
            result = client.chat([{"role": "user", "content": "hi"}],
                                 ts="2026-10-29", purpose="test")
            self.assertEqual(result.text, "Hello, world.")
            self.assertEqual(result.request_id, "chatcmpl-test-1")
            entries = ledger.entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].model, "muse-spark-1.3")
            self.assertAlmostEqual(entries[0].cost_usd,
                                   cost_from_usage(120, 40))

    def test_empty_reply_raises(self):
        ft = FakeSparkTransport("")
        client = SparkClient(api_key="k", transport=ft)
        with self.assertRaises(SparkClientError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_cap_blocks_before_sending(self):
        ft = FakeSparkTransport("x")
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "l.jsonl", cap_usd=0.0001)
            client = SparkClient(api_key="k", ledger=ledger, transport=ft)
            with self.assertRaises(SpendCapError):
                client.chat([{"role": "user", "content": "hi"}],
                            max_tokens=4000)
        self.assertEqual(ft.calls, [])


class BuildCaptionsTests(unittest.TestCase):
    def setUp(self):
        self.turn = TranscriptTurn(
            text="the the interesting thing about it its an 11 piece band",
            start_ms=60, end_ms=10000, speaker="A")
        self.prof = profile_from_spans([(0.0, 10.0)], total_s=10.0)

    def test_punctuation_applied_when_words_unchanged(self):
        ft = FakeSparkTransport(
            "The the interesting thing about it, it's an 11 piece band.")
        spark = SparkClient(api_key="k", transport=ft)
        cues, report = build_captions([self.turn], self.prof, spark=spark)
        self.assertFalse(report.turns[0].used_fallback)
        self.assertEqual(report.turns[0].spark_request_id, "chatcmpl-test-1")
        self.assertTrue(cues)
        self.assertTrue(cues[0].text.startswith("The the interesting"))

    def test_fallback_when_spark_changes_words(self):
        ft = FakeSparkTransport("The interesting thing about the band.")
        spark = SparkClient(api_key="k", transport=ft)
        cues, report = build_captions([self.turn], self.prof, spark=spark)
        self.assertTrue(report.turns[0].used_fallback)
        self.assertIn("word sequence", report.turns[0].note)
        self.assertTrue(cues[0].text.startswith("the the interesting"))

    def test_no_spark_uses_raw_text(self):
        cues, report = build_captions([self.turn], self.prof, spark=None)
        self.assertTrue(report.turns[0].used_fallback)
        self.assertTrue(cues)

    def test_srt_output(self):
        cues, _ = build_captions([self.turn], self.prof, spark=None)
        srt = cues_to_srt(cues)
        self.assertIn(" --> ", srt)
        self.assertIn("the the interesting", srt)


if __name__ == "__main__":
    unittest.main()
