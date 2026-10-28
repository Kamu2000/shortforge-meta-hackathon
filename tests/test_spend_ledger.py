"""Offline unit tests for the cumulative spend ledger."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.spend_ledger import SpendCapError, SpendLedger  # noqa: E402


class SpendLedgerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ledger.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_ledger_totals_zero(self):
        ledger = SpendLedger(self.path, cap_usd=145.0)
        self.assertEqual(ledger.total_usd(), 0.0)
        self.assertEqual(ledger.remaining_usd(), 145.0)

    def test_record_and_reload_persists_total(self):
        ledger = SpendLedger(self.path, cap_usd=145.0)
        ledger.record(ts="2026-10-28", model="muse-wonder-1.0",
                      cost_usd=0.372, purpose="concept run")
        ledger.record(ts="2026-10-28", model="muse-image-1.0",
                      cost_usd=0.01, purpose="probe")
        self.assertEqual(ledger.total_usd(), 0.382)
        reloaded = SpendLedger(self.path, cap_usd=145.0)
        self.assertEqual(reloaded.total_usd(), 0.382)
        self.assertEqual(len(reloaded.entries()), 2)

    def test_check_can_afford_blocks_over_cap(self):
        ledger = SpendLedger(self.path, cap_usd=1.0)
        ledger.record(ts="2026-10-28", model="sam-3.1", cost_usd=0.90,
                      purpose="spike")
        with self.assertRaises(SpendCapError):
            ledger.check_can_afford(0.20)
        ledger.check_can_afford(0.10)  # exactly at cap is allowed

    def test_record_over_cap_raises_and_appends_nothing(self):
        ledger = SpendLedger(self.path, cap_usd=0.50)
        with self.assertRaises(SpendCapError):
            ledger.record(ts="2026-10-28", model="sam-3.1",
                          cost_usd=0.60, purpose="too big")
        self.assertEqual(ledger.entries(), [])
        self.assertFalse(self.path.exists())

    def test_negative_cost_rejected(self):
        ledger = SpendLedger(self.path, cap_usd=145.0)
        with self.assertRaises(ValueError):
            ledger.record(ts="2026-10-28", model="sam-3.1",
                          cost_usd=-1.0, purpose="bad")


if __name__ == "__main__":
    unittest.main()
