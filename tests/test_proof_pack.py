"""Unit tests for the proof-pack harness's SCORING logic only.

The pack itself (tools/proof_pack.py) is a standalone runner, not
part of the suite; what belongs under test here is the pure
label-vs-outcome arithmetic: per-check TP/FP/FN/TN, precision and
recall (including the undefined-denominator cases), exact-set
agreement, and the findings list. Cases are synthetic dicts - no
verify() calls, no fixtures, so these stay fast.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "proof_pack",
    Path(__file__).resolve().parents[1] / "tools" / "proof_pack.py")
proof_pack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(proof_pack)

score_cases = proof_pack.score_cases


def case(cid, expected, actual):
    return {"id": cid, "expected_failed": expected,
            "actual_failed": actual}


class ScoreCasesTests(unittest.TestCase):
    def test_perfect_agreement_scores_one(self):
        scores = score_cases([
            case("a", {"F1"}, {"F1"}),
            case("b", set(), set()),
            case("c", {"F5", "F1"}, {"F1", "F5"}),
        ])
        f1 = scores["per_check"]["F1"]
        self.assertEqual((f1["tp"], f1["fp"], f1["fn"], f1["tn"]),
                         (2, 0, 0, 1))
        self.assertEqual(f1["precision"], 1.0)
        self.assertEqual(f1["recall"], 1.0)
        self.assertEqual(scores["agreement"],
                         {"exact_set_matches": 3, "total_cases": 3})
        self.assertEqual(scores["findings"], [])

    def test_false_positive_lowers_precision_not_recall(self):
        scores = score_cases([
            case("a", {"F2"}, {"F2", "F3"}),  # F3 flagged in error
            case("b", {"F2"}, {"F2"}),
        ])
        f3 = scores["per_check"]["F3"]
        self.assertEqual((f3["tp"], f3["fp"], f3["fn"]), (0, 1, 0))
        self.assertEqual(f3["precision"], 0.0)
        self.assertIsNone(f3["recall"])  # no labelled F3 positives
        f2 = scores["per_check"]["F2"]
        self.assertEqual(f2["precision"], 1.0)
        self.assertEqual(f2["recall"], 1.0)

    def test_false_negative_lowers_recall_not_precision(self):
        scores = score_cases([
            case("a", {"F7"}, set()),  # verifier missed the fault
            case("b", {"F7"}, {"F7"}),
        ])
        f7 = scores["per_check"]["F7"]
        self.assertEqual((f7["tp"], f7["fp"], f7["fn"]), (1, 0, 1))
        self.assertEqual(f7["precision"], 1.0)
        self.assertEqual(f7["recall"], 0.5)

    def test_undefined_metrics_are_none_not_zero(self):
        # A check with no positives of either kind: precision and
        # recall are undefined and must surface as None.
        scores = score_cases([case("a", {"F1"}, {"F1"})])
        f4 = scores["per_check"]["F4"]
        self.assertEqual((f4["tp"], f4["fp"], f4["fn"], f4["tn"]),
                         (0, 0, 0, 1))
        self.assertIsNone(f4["precision"])
        self.assertIsNone(f4["recall"])
        # A check with labelled positives but no flags at all:
        # precision undefined, recall a real 0.0.
        scores = score_cases([case("a", {"F6"}, set())])
        f6 = scores["per_check"]["F6"]
        self.assertIsNone(f6["precision"])
        self.assertEqual(f6["recall"], 0.0)

    def test_findings_name_the_disagreeing_case_and_direction(self):
        scores = score_cases([
            case("ok", {"F1"}, {"F1"}),
            case("missed", {"F4"}, set()),
            case("extra", set(), {"F5"}),
            case("both-ways", {"F2"}, {"F3"}),
        ])
        self.assertEqual(scores["agreement"]["exact_set_matches"], 1)
        by_id = {f["case"]: f for f in scores["findings"]}
        self.assertEqual(set(by_id), {"missed", "extra", "both-ways"})
        self.assertEqual(by_id["missed"]["false_negative_checks"],
                         ["F4"])
        self.assertEqual(by_id["missed"]["false_positive_checks"], [])
        self.assertEqual(by_id["extra"]["false_positive_checks"],
                         ["F5"])
        self.assertEqual(by_id["both-ways"]["false_positive_checks"],
                         ["F3"])
        self.assertEqual(by_id["both-ways"]["false_negative_checks"],
                         ["F2"])

    def test_agreement_requires_exact_set_match(self):
        # Same check count, different sets: not agreement.
        scores = score_cases([case("a", {"F1", "F5"}, {"F1"})])
        self.assertEqual(scores["agreement"]["exact_set_matches"], 0)


if __name__ == "__main__":
    unittest.main()
