"""Offline tests for the Spark Open sweep client (fake transport)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.spark_open_client import (  # noqa: E402
    MODEL_ID,
    SparkOpenClient,
    SparkOpenError,
)

TURNS = [
    {"start_s": 0.0, "end_s": 6.0, "text": "Welcome back to the show."},
    {"start_s": 6.0, "end_s": 15.0, "text": "Nobody tells you this: 71 percent of podcasts die at episode three."},
    {"start_s": 15.0, "end_s": 24.0, "text": "The fix costs nothing and takes ten minutes."},
]


def fake_transport_factory(reply_text, status=200, usage=None):
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, json.loads(body.decode("utf-8"))))
        return status, {
            "id": "chatcmpl-test-1",
            "choices": [{"message": {"content": reply_text}}],
            "usage": usage or {"prompt_tokens": 210, "completion_tokens": 64},
        }

    return transport, calls


class SweepTests(unittest.TestCase):
    def test_payload_and_parse(self):
        reply = json.dumps(
            [{"start_s": 6.0, "end_s": 15.0, "score": 0.91, "reason": "stat hook"}]
        )
        transport, calls = fake_transport_factory(reply)
        client = SparkOpenClient(api_key="test-key", transport=transport)
        result = client.sweep(TURNS, ts="2026-10-28")
        self.assertEqual(len(calls), 1)
        method, url, headers, payload = calls[0]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertIn("71 percent", payload["messages"][1]["content"])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].score, 0.91)
        self.assertEqual(result.prompt_tokens, 210)
        self.assertEqual(result.request_id, "chatcmpl-test-1")

    def test_usage_log_written(self):
        reply = json.dumps([{"start_s": 0, "end_s": 6, "score": 0.4}])
        transport, _ = fake_transport_factory(reply)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "partner-usage-log.jsonl"
            client = SparkOpenClient(
                api_key="k", transport=transport, usage_log_path=log
            )
            client.sweep(TURNS, ts="2026-10-28")
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["provider"], "baseten")
            self.assertEqual(entry["model"], MODEL_ID)
            self.assertEqual(entry["candidates"], 1)

    def test_null_content_fails_loud(self):
        # Live-verified failure mode: reasoning burns the token
        # budget and the API returns content=None with HTTP 200.
        def transport(method, url, headers, body):
            return 200, {
                "id": "chatcmpl-null",
                "choices": [{"message": {"content": None}}],
                "usage": {"prompt_tokens": 279, "completion_tokens": 400},
            }

        client = SparkOpenClient(api_key="k", transport=transport)
        with self.assertRaises(SparkOpenError):
            client.sweep(TURNS)

    def test_non_json_reply_fails_loud(self):
        transport, _ = fake_transport_factory("Here are some moments...")
        client = SparkOpenClient(api_key="k", transport=transport)
        with self.assertRaises(SparkOpenError):
            client.sweep(TURNS)

    def test_http_error_fails(self):
        transport, _ = fake_transport_factory("x", status=500)
        client = SparkOpenClient(api_key="k", transport=transport)
        with self.assertRaises(SparkOpenError):
            client.sweep(TURNS)

    def test_empty_turns_rejected(self):
        transport, _ = fake_transport_factory("[]")
        client = SparkOpenClient(api_key="k", transport=transport)
        with self.assertRaises(SparkOpenError):
            client.sweep([])

    def test_missing_key_rejected(self):
        transport, _ = fake_transport_factory("[]")
        client = SparkOpenClient(transport=transport)
        with self.assertRaises(SparkOpenError):
            client.sweep(TURNS)


if __name__ == "__main__":
    unittest.main()
