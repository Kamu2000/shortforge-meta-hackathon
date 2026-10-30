"""Tests for the Muse Image client: wire shape, flat-cost ledger
gating, and the failure classes. All offline - a fake transport
stands in for the API; no key, no network, no spend.
"""

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.image_client import (  # noqa: E402
    COST_PER_IMAGE_USD,
    MODEL_ID,
    ImageClient,
    ImageClientError,
)
from spatial_director.spend_ledger import SpendCapError, SpendLedger  # noqa: E402

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def ok_transport(calls):
    def transport(method, url, headers, body):
        calls.append({"method": method, "url": url,
                      "headers": headers,
                      "body": json.loads(body.decode("utf-8"))})
        return 200, {
            "id": "img-test-1",
            "data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}],
        }
    return transport


class GenerateTests(unittest.TestCase):
    def test_posts_generation_and_decodes_b64(self):
        calls = []
        client = ImageClient(api_key="test-key",
                             transport=ok_transport(calls))
        result = client.generate("a brass desk lamp on a dark stage")
        self.assertEqual(result.image_bytes, PNG_BYTES)
        self.assertEqual(result.request_id, "img-test-1")
        self.assertEqual(result.cost_usd, COST_PER_IMAGE_USD)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertTrue(call["url"].endswith("/v1/images/generations"))
        self.assertEqual(call["body"]["model"], MODEL_ID)
        self.assertEqual(call["body"]["prompt"],
                         "a brass desk lamp on a dark stage")
        self.assertEqual(call["headers"]["Authorization"],
                         "Bearer test-key")

    def test_model_id_is_pinned_not_caller_chosen(self):
        # There is no model parameter on generate(): a caller cannot
        # steer the client at another (e.g. Contributor) model id.
        import inspect

        from spatial_director import image_client
        params = inspect.signature(image_client.ImageClient.generate)
        self.assertNotIn("model", params.parameters)

    def test_empty_prompt_rejected_before_any_call(self):
        calls = []
        client = ImageClient(api_key="k", transport=ok_transport(calls))
        with self.assertRaises(ImageClientError):
            client.generate("   ")
        self.assertEqual(calls, [])

    def test_url_shaped_reply_fetched(self):
        def transport(method, url, headers, body):
            return 200, {"id": "img-2",
                         "data": [{"url": "https://example.test/i.webp"}]}

        fetched = []

        def fetch(url):
            fetched.append(url)
            return b"webp-bytes"

        client = ImageClient(api_key="k", transport=transport,
                             fetch_bytes=fetch)
        result = client.generate("a lighthouse at dusk")
        self.assertEqual(result.image_bytes, b"webp-bytes")
        self.assertEqual(fetched, ["https://example.test/i.webp"])

    def test_http_error_raises_and_ledgers_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "ledger.jsonl")

            def transport(method, url, headers, body):
                return 500, {"error": "boom"}

            client = ImageClient(api_key="k", transport=transport,
                                 ledger=ledger)
            with self.assertRaises(ImageClientError):
                client.generate("a lighthouse at dusk")
            self.assertEqual(ledger.entries(), [])

    def test_malformed_reply_raises(self):
        def transport(method, url, headers, body):
            return 200, {"data": [{"revised_prompt": "only"}]}

        client = ImageClient(api_key="k", transport=transport)
        with self.assertRaises(ImageClientError):
            client.generate("a lighthouse at dusk")

    def test_no_key_raises(self):
        client = ImageClient(transport=ok_transport([]))
        with self.assertRaises(ImageClientError):
            client.generate("a lighthouse at dusk")


class LedgerTests(unittest.TestCase):
    def test_success_records_flat_cost(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.jsonl"
            ledger = SpendLedger(path)
            client = ImageClient(api_key="k",
                                 transport=ok_transport([]),
                                 ledger=ledger)
            client.generate("a lighthouse at dusk",
                            purpose="graphics-image-test", ts="2026-10-30")
            entries = ledger.entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].model, MODEL_ID)
            self.assertEqual(entries[0].cost_usd, 0.01)
            self.assertEqual(entries[0].purpose, "graphics-image-test")
            self.assertEqual(entries[0].request_id, "img-test-1")
            self.assertEqual(ledger.total_usd(), 0.01)

    def test_cap_blocks_the_call_before_it_is_sent(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = SpendLedger(Path(td) / "ledger.jsonl", cap_usd=0.005)
            calls = []
            client = ImageClient(api_key="k",
                                 transport=ok_transport(calls),
                                 ledger=ledger)
            with self.assertRaises(SpendCapError):
                client.generate("a lighthouse at dusk")
            self.assertEqual(calls, [])  # never sent


if __name__ == "__main__":
    unittest.main()
