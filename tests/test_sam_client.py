"""Offline unit tests for the SAM client (fake transport, no network)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director import sam_client  # noqa: E402
from spatial_director.spend_ledger import SpendCapError, SpendLedger  # noqa: E402


def fake_transport_factory(captured, billed_frames=600, status=200):
    def transport(method, url, headers, body):
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = json.loads(body.decode("utf-8"))
        return status, {
            "id": "resp_test_123",
            "usage": {"video_frames_processed": billed_frames},
            "output": [],
        }
    return transport


class SamClientTests(unittest.TestCase):
    def test_default_concept_is_person(self):
        self.assertEqual(sam_client.DEFAULT_CONCEPT, "person")

    def test_validate_concept_rejects_list_and_question(self):
        with self.assertRaises(sam_client.SamClientError):
            sam_client.validate_concept("bus, bicycle")
        with self.assertRaises(sam_client.SamClientError):
            sam_client.validate_concept("find the person?")
        self.assertEqual(sam_client.validate_concept("  person "), "person")

    def test_estimate_uses_minimum_billing(self):
        self.assertEqual(sam_client.estimate_video_cost_usd(30), 0.01)
        self.assertEqual(sam_client.estimate_video_cost_usd(1000), 0.20)

    def test_segment_video_payload_and_ledger_record(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SpendLedger(Path(tmp) / "l.jsonl", cap_usd=145.0)
            client = sam_client.SamClient(
                api_key="test-key",
                ledger=ledger,
                transport=fake_transport_factory(captured),
            )
            result = client.segment_video(
                file_id="file_abc", n_frames=600, ts="2026-10-28")
            payload = captured["payload"]
            self.assertEqual(payload["model"], "sam-3.1")
            content = payload["input"][0]["content"]
            self.assertEqual(content[0]["text"], "person")
            self.assertEqual(content[1]["file_id"], "file_abc")
            self.assertEqual(result.billed_frames, 600)
            self.assertEqual(result.cost_usd, 0.12)
            self.assertEqual(ledger.total_usd(), 0.12)

    def test_ledger_cap_blocks_before_transport_is_called(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SpendLedger(Path(tmp) / "l.jsonl", cap_usd=0.05)
            client = sam_client.SamClient(
                api_key="test-key",
                ledger=ledger,
                transport=fake_transport_factory(captured),
            )
            with self.assertRaises(SpendCapError):
                client.segment_video(
                    file_id="file_abc", n_frames=600, ts="2026-10-28")
            self.assertEqual(captured, {})  # nothing was sent

    def test_frame_limit_enforced(self):
        client = sam_client.SamClient(
            api_key="test-key", transport=fake_transport_factory({}))
        with self.assertRaises(sam_client.SamClientError):
            client.segment_video(file_id="f", n_frames=15001)


if __name__ == "__main__":
    unittest.main()
