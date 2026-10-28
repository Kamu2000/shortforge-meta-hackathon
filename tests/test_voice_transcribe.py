"""Offline tests for voice_transcribe (fake transport, tiny WAVs)."""

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.spend_ledger import SpendLedger, SpendCapError  # noqa: E402
from spatial_director.voice_transcribe import (  # noqa: E402
    VoiceTranscribeClient,
    VoiceTranscribeError,
    estimate_cost_usd,
    parse_response,
    wav_duration_s,
)


def make_wav(path: Path, seconds: float = 2.0, rate: int = 24000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


CANNED = {
    "sessionId": "sess-test-123",
    "transcript": "Hello there. General Kenobi.",
    "audioDurationMs": 2000,
    "turns": [
        {"speaker": "A", "startMs": 0, "endMs": 900,
         "transcript": "Hello there."},
        {"speaker": "B", "startMs": 1100, "endMs": 2000,
         "transcript": "General Kenobi."},
    ],
}


class FakeTransport:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload if payload is not None else CANNED
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, body))
        return self.status, self.payload


class CostMathTests(unittest.TestCase):
    def test_hourly_rate(self):
        self.assertEqual(estimate_cost_usd(3600.0), 0.18)
        self.assertEqual(estimate_cost_usd(20.0), 0.001)  # spike clip A
        self.assertEqual(estimate_cost_usd(0.0), 0.0)

    def test_wav_duration(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_wav(Path(td) / "a.wav", seconds=3.0)
            self.assertAlmostEqual(wav_duration_s(p.read_bytes()), 3.0)


class ParseTests(unittest.TestCase):
    def test_turns_and_fields(self):
        r = parse_response(CANNED)
        self.assertEqual(r.session_id, "sess-test-123")
        self.assertEqual(r.audio_duration_ms, 2000)
        self.assertEqual(len(r.turns), 2)
        self.assertEqual(r.turns[0].speaker, "A")
        self.assertEqual(r.turns[1].start_ms, 1100)
        self.assertEqual(r.cost_usd, estimate_cost_usd(2.0))

    def test_push_to_talk_shape_has_no_turns(self):
        r = parse_response({"transcript": "one turn only",
                            "audioDurationMs": 5000})
        self.assertEqual(r.turns, [])
        self.assertIsNone(r.session_id)


class ClientTests(unittest.TestCase):
    def _client(self, transport, ledger=None):
        return VoiceTranscribeClient(
            api_key="test-key", ledger=ledger, transport=transport)

    def test_multipart_shape_and_parse(self):
        ft = FakeTransport()
        with tempfile.TemporaryDirectory() as td:
            wav = make_wav(Path(td) / "clip.wav", seconds=2.0)
            result = self._client(ft).transcribe_file(wav)
        url, headers, body = ft.calls[0]
        self.assertTrue(url.endswith("/asr/transcribe"))
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertIn(b'name="request"', body)
        self.assertIn(b'"mode": "DIARIZATION"', body)
        self.assertIn(b'"audioEncoding": "WAV"', body)
        self.assertIn(b'name="audio"; filename="clip.wav"', body)
        self.assertEqual(result.transcript, CANNED["transcript"])
        self.assertEqual(len(result.turns), 2)

    def test_unknown_mode_rejected_before_sending(self):
        ft = FakeTransport()
        with tempfile.TemporaryDirectory() as td:
            wav = make_wav(Path(td) / "clip.wav")
            with self.assertRaises(VoiceTranscribeError):
                self._client(ft).transcribe_file(wav, mode="DIARIZE")
        self.assertEqual(ft.calls, [])  # nothing was sent

    def test_non_200_raises(self):
        ft = FakeTransport(status=400, payload={"error": "bad audio"})
        with tempfile.TemporaryDirectory() as td:
            wav = make_wav(Path(td) / "clip.wav")
            with self.assertRaises(VoiceTranscribeError):
                self._client(ft).transcribe_file(wav)

    def test_ledger_records_actual_cost(self):
        ft = FakeTransport()
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "ledger.jsonl", cap_usd=145.0)
            wav = make_wav(Path(td) / "clip.wav", seconds=2.0)
            self._client(ft, ledger).transcribe_file(
                wav, ts="2026-10-29", purpose="vt-test")
            entries = ledger.entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].model, "muse-voice-transcribe-1.0")
            self.assertEqual(entries[0].request_id, "sess-test-123")
            # Actual cost from response audioDurationMs (2s), not estimate.
            self.assertEqual(entries[0].cost_usd, estimate_cost_usd(2.0))

    def test_ledger_cap_blocks_before_sending(self):
        ft = FakeTransport()
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "ledger.jsonl", cap_usd=0.0005)
            wav = make_wav(Path(td) / "clip.wav", seconds=20.0)
            with self.assertRaises(SpendCapError):
                self._client(ft, ledger).transcribe_file(wav)
        self.assertEqual(ft.calls, [])


if __name__ == "__main__":
    unittest.main()
