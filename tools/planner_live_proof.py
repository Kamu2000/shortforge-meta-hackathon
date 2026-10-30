#!/usr/bin/env python3
"""Live proof for the Defect A fix (2026-10-30).

The Holmes graphics proof run starved the graphics planner on all
three Shorts: at PLANNER_MAX_TOKENS=4000 the Spark 1.3 reply came
back empty ("the token budget was consumed by reasoning") and zero
graphics were planned. The fix raised the budget to 8000 with one
retry-on-empty at doubled budget (cap 16000).

This script makes ONE live planner call through the exact pipeline
path - same SparkClient construction as pipeline.main, same
plan_graphics entry point - against the cached caption cues of the
run's short-2 (23 cues, 14.0s: a real pipeline-sized cue list, the
same input that starved). It prints the outcome and writes
planner-live-proof.json beside the cached artifacts.

Cost: one Spark 1.3 call (two only if the first reply is empty and
the retry fires). Ledgered through the repo SpendLedger like every
pipeline call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spatial_director.caption_align import CaptionCue  # noqa: E402
from spatial_director.graphics_layer import plan_graphics  # noqa: E402
from spatial_director.pipeline import (  # noqa: E402
    _key_provider,
    _no_redirect_transport,
)
from spatial_director.spark_client import SparkClient  # noqa: E402
from spatial_director.spend_ledger import SpendLedger  # noqa: E402

SHORT_DIR = Path(
    "/home/hatch/workspace/meta-hackathon/spike-footage/"
    "pipeline-run-graphics-holmes/short-2")
TS = "2026-10-30"


def ledger_total(ledger: SpendLedger) -> float:
    total = 0.0
    for line in ledger.path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            total += float(json.loads(line).get("cost_usd", 0.0))
    return total


def main() -> int:
    raw = json.loads((SHORT_DIR / "captions.json").read_text(
        encoding="utf-8"))
    cues = [CaptionCue(text=c["text"], start_s=float(c["start_s"]),
                       end_s=float(c["end_s"])) for c in raw["cues"]]
    duration_s = max(c.end_s for c in cues)

    ledger = SpendLedger(REPO / "spend-ledger.jsonl")
    before = ledger_total(ledger)
    meta_key = _key_provider("SPATIAL_DIRECTOR_META_API_KEY",
                             "custom.meta-model-api")
    spark = SparkClient(api_key_provider=meta_key, ledger=ledger,
                        transport=_no_redirect_transport)

    outcome = plan_graphics(spark, cues, duration_s=duration_s, ts=TS)
    after = ledger_total(ledger)

    proof = {
        "short": "holmes-graphics-run short-2 (cached captions.json)",
        "cues": len(cues),
        "duration_s": duration_s,
        "error": outcome.error,
        "spans_returned": len(outcome.spans),
        "spans": [s.to_dict() for s in outcome.spans],
        "planner_rejected": outcome.rejected,
        "planner_call": outcome.spark_call,
        "call_cost_usd": round(after - before, 4),
        "ledger_cumulative_before": round(before, 4),
        "ledger_cumulative_after": round(after, 4),
    }
    out_path = SHORT_DIR / "planner-live-proof.json"
    out_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))
    print(f"proof written: {out_path}", file=sys.stderr)
    # Success = the planner answered (no error). Zero spans with no
    # error would be a legitimate empty plan, reported as measured.
    return 0 if outcome.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
