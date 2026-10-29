#!/usr/bin/env python3
"""Matrix D — SAM prompt variants + primary-selection stress (benchmark).

Two halves:

1. **Selection stress (offline, zero paid calls).** Every cached SAM
   response (spike clips A/B/C + pipeline run-1/run-2 windows) is run
   through the shipped ``select_primary_id`` (largest-area persistent
   id) and the pick is recorded with per-id presence + median box
   area. The known trap is clip C: the true speaker is id 1 and id 0
   is a small picture-in-picture figure on a background TV.

2. **Prompt variants (paid SAM calls, ledger-gated).** On two windows
   whose shipped ``person`` responses are cached (Holmes run-3
   short-2 window, 655 frames; Belew clip A, 480 frames), re-segment
   with variant concepts and compare against the cached baseline:
   primary presence, picked id, decode errors, temporal stability
   (mean consecutive-frame primary IoU) and mask agreement with the
   baseline (mean per-frame primary IoU). Variants are bare noun
   phrases per SAM's one-concept rule: ``man``, ``speaker`` and (on
   Holmes only) ``man speaking``.

Variant raw responses land in
``~/workspace/meta-hackathon/benchmark-matrix-d-raw/`` (too large for
the repo); the metrics land in ``tools/benchmark_matrix_d_report.json``.

Usage:
    python3 tools/benchmark_matrix_d.py [--skip-variants]

Written fresh in-repo on 2026-10-29 for the Oct 30 benchmark matrix
(pulled forward to Oct 29). NumPy + meta-sam-parser via mask_decode.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")

import numpy as np  # noqa: E402

from spatial_director.mask_decode import (  # noqa: E402
    MaskDecodeError,
    decode_object_canvas,
    parse_sam_response,
    select_primary_id,
)
from spatial_director.pipeline import upload_media_file  # noqa: E402
from spatial_director.sam_client import SamClient, SegmentationResult  # noqa: E402
from spatial_director.spend_ledger import SpendLedger  # noqa: E402

TS = "2026-10-29"
SPIKE = Path("/home/hatch/workspace/meta-hackathon/spike-footage")
RAW_OUT = Path("/home/hatch/workspace/meta-hackathon/benchmark-matrix-d-raw")
REPORT = REPO / "tools" / "benchmark_matrix_d_report.json"
LEDGER = REPO / "spend-ledger.jsonl"

CACHED_RESPONSES = [
    ("spike-clipA-belew", SPIKE / "sam-response-clipA-raw.json", 0),
    ("spike-clipB-holmes", SPIKE / "sam-response-clipB-raw.json", None),
    ("spike-clipC-holmes-tv-trap", SPIKE / "sam-response-clipC-raw.json", 1),
    ("run3-short1", SPIKE / "pipeline-run-2/short-1/sam-response.json", None),
    ("run3-short2", SPIKE / "pipeline-run-2/short-2/sam-response.json", None),
    ("run3-short3", SPIKE / "pipeline-run-2/short-3/sam-response.json", None),
    ("run1-short1", SPIKE / "pipeline-run-1/short-1/sam-response.json", None),
    ("run1-short2", SPIKE / "pipeline-run-1/short-2/sam-response.json", None),
    ("run1-short3", SPIKE / "pipeline-run-1/short-3/sam-response.json", None),
]

VARIANT_JOBS = [
    {
        "window": "holmes-short2-window",
        "clip": SPIKE / "pipeline-run-2/short-2/window.mp4",
        "baseline": SPIKE / "pipeline-run-2/short-2/sam-response.json",
        "n_frames": 655,
        "variants": ["man", "speaker", "man speaking"],
    },
    {
        "window": "belew-clipA",
        "clip": SPIKE / "clipA_belew.mp4",
        "baseline": SPIKE / "sam-response-clipA-raw.json",
        "n_frames": 480,
        "variants": ["man", "speaker"],
    },
]


def _no_redirect_transport(method, url, headers, body):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            raise RuntimeError("redirect refused")

    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        return exc.code, {"error_body": detail}


def _meta_key():
    from dynamic_credentials import dynamic_credential_entry

    return lambda: str(dynamic_credential_entry(
        "custom.meta-model-api")["surrogate"]).strip()


# ---------------------------------------------------------------- analysis


def analyse(frames, *, sample_step: int = 1) -> dict[str, Any]:
    """Selection outcome + per-id stats + decoded primary canvases.

    Selection and per-id stats come from box records (exact, all
    frames, no raster decode). Raster metrics (decode errors,
    temporal stability, IoU-vs-baseline) decode the PICKED primary's
    masks on a uniform frame sample (every ``sample_step``-th frame)
    - the meta-sam-parser codec costs ~0.4s/mask, and the first two
    versions of this tool spent 20+ min decoding masks no planned
    metric needed. The sampling is stated wherever the numbers are
    reported; presence and the pick itself are never sampled.
    """
    n = len(frames)
    areas: dict[int, list[int]] = {}
    for f in frames:
        for o in f.objects:
            areas.setdefault(o.object_id, []).append(o.box_area)
    per_id = {
        str(oid): {
            "presence": round(len(v) / n, 4),
            "median_box_area": float(statistics.median(v)),
        }
        for oid, v in sorted(areas.items())
    }
    decode_errors = 0
    decoded_objects = 0
    try:
        picked = select_primary_id(frames)
    except MaskDecodeError:
        picked = None
    canvases: dict[int, np.ndarray] = {}
    shape = None
    for f in frames:
        for o in f.objects:
            if shape is None:
                shape = (o.frame_h, o.frame_w)
            if picked is None or o.object_id != picked:
                continue
            if f.index % sample_step != 0:
                continue
            decoded_objects += 1
            try:
                canvases[f.index] = decode_object_canvas(o)
            except Exception:
                decode_errors += 1
                continue
    if shape is None:
        shape = (0, 0)
    # presence is box-based (exact, all frames - never sampled)
    present = (len(areas[picked]) if picked in areas else 0)
    # temporal stability: IoU between successive SAMPLED canvases
    idxs = sorted({f.index for f in frames})
    sampled = sorted(canvases)
    ious = []
    for a, b in zip(sampled, sampled[1:]):
        ea, eb = canvases[a], canvases[b]
        union = np.logical_or(ea, eb).sum()
        if union:
            ious.append(float(np.logical_and(ea, eb).sum() / union))
    return {
        "n_frames": n,
        "n_ids": len(areas),
        "per_id": per_id,
        "picked_primary_id": picked,
        "primary_presence": round(present / n, 4) if n else 0.0,
        "primary_objects_decoded": decoded_objects,
        "decode_errors": decode_errors,
        "decode_scope": f"picked-primary masks on every "
                        f"{sample_step}-th frame (uniform sample); "
                        f"selection + presence are box-based, all frames",
        "sample_step": sample_step,
        "mean_sampled_step_iou": (
            round(float(statistics.mean(ious)), 4) if ious else None),
        "_canvases": canvases,
        "_shape": shape,
        "_idxs": idxs,
    }


def iou_vs_baseline(variant: dict, baseline: dict) -> float | None:
    """Mean IoU between two analyses' primary canvases on shared
    sampled frames (both sides sampled at the same step)."""
    step = baseline["sample_step"]
    idxs = [i for i in baseline["_idxs"] if i % step == 0]
    shape = baseline["_shape"]
    vals = []
    for i in idxs:
        a = baseline["_canvases"].get(i)
        b = variant["_canvases"].get(i)
        ea = a if a is not None else np.zeros(shape, dtype=bool)
        eb = b if b is not None else np.zeros(shape, dtype=bool)
        union = np.logical_or(ea, eb).sum()
        vals.append(
            float(np.logical_and(ea, eb).sum() / union) if union else 1.0)
    return round(float(statistics.mean(vals)), 4) if vals else None


def public(a: dict) -> dict:
    return {k: v for k, v in a.items() if not k.startswith("_")}


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-variants", action="store_true")
    args = ap.parse_args()

    ledger = SpendLedger(LEDGER)
    print(f"ledger before: ${ledger.total_usd():.4f}")

    report: dict[str, Any] = {"matrix": "D", "date": TS}

    # ---- half 1 (runs FIRST): paid prompt variants on two windows.
    # Ordered ahead of the offline stress half so the paid evidence
    # lands even if the CPU-bound decode work runs long.
    variants_out = []
    if not args.skip_variants:
        RAW_OUT.mkdir(parents=True, exist_ok=True)
        sam = SamClient(api_key_provider=_meta_key(), ledger=ledger,
                        transport=_no_redirect_transport)
        for job in VARIANT_JOBS:
            base_frames = parse_sam_response(
                json.loads(job["baseline"].read_text()))
            base = analyse(base_frames, sample_step=5)
            brow = {"window": job["window"], "concept": "person",
                    **public(base)}
            variants_out.append(brow)
            print(f"baseline {job['window']} 'person': picked="
                  f"{base['picked_primary_id']} presence="
                  f"{base['primary_presence']} step_iou="
                  f"{base['mean_sampled_step_iou']}")
            print(f"ledger before {job['window']} variants: "
                  f"${ledger.total_usd():.4f}")
            file_id = upload_media_file(job["clip"],
                                        api_key_provider=_meta_key())
            for concept in job["variants"]:
                raw_path = RAW_OUT / (
                    f"{job['window']}--{concept.replace(' ', '_')}.json")
                if raw_path.exists():
                    # Resume path: a previous invocation already paid
                    # for and saved this response (raws are written
                    # BEFORE parsing, precisely so a crash mid-analysis
                    # never re-buys a call). Reconstruct the result
                    # from the file; no API call, no ledger entry.
                    saved = json.loads(raw_path.read_text())
                    billed = (saved.get("usage") or {}).get(
                        "video_frames_processed")
                    result = SegmentationResult(
                        raw=saved, billed_frames=billed,
                        request_id=saved.get("id"))
                    print(f"--- SAM '{concept}' on {job['window']}: "
                          f"reusing saved raw response (no new call)",
                          flush=True)
                else:
                    print(f"--- SAM '{concept}' on {job['window']} "
                          f"({job['n_frames']} frames) ...", flush=True)
                    result = sam.segment_video(
                        file_id=file_id, concept=concept,
                        n_frames=job["n_frames"],
                        purpose=f"benchmark-matrix-d-variant-{concept}",
                        ts=TS)
                    raw_path.write_text(json.dumps(result.raw),
                                        encoding="utf-8")
                try:
                    frames = parse_sam_response(result.raw)
                except MaskDecodeError:
                    # SAM returned no frame records at all: the
                    # concept matched NOTHING in the clip (verified
                    # live for 'speaker' on Belew: billed 480 frames,
                    # output_text empty). That is a result, not a
                    # crash - record it as zero detections.
                    empty = {
                        "n_frames": result.billed_frames,
                        "n_ids": 0, "per_id": {},
                        "picked_primary_id": None,
                        "primary_presence": 0.0,
                        "primary_objects_decoded": 0,
                        "decode_errors": 0,
                        "decode_scope": "no detections returned",
                        "sample_step": 5,
                        "mean_sampled_step_iou": None,
                        "_canvases": {}, "_shape": base["_shape"],
                        "_idxs": base["_idxs"],
                    }
                    row = {"window": job["window"], "concept": concept,
                           "billed_frames": result.billed_frames,
                           "cost_usd": result.cost_usd,
                           "request_id": result.request_id,
                           "raw_response": str(raw_path),
                           "zero_detections": True,
                           **public(empty),
                           "mean_iou_vs_person_baseline":
                               iou_vs_baseline(empty, base)}
                    variants_out.append(row)
                    print(f"    ZERO DETECTIONS for '{concept}' "
                          f"(billed {result.billed_frames} frames, "
                          f"cost=${result.cost_usd})")
                    continue
                a = analyse(frames, sample_step=5)
                row = {"window": job["window"], "concept": concept,
                       "billed_frames": result.billed_frames,
                       "cost_usd": result.cost_usd,
                       "request_id": result.request_id,
                       "raw_response": str(raw_path),
                       **public(a),
                       "mean_iou_vs_person_baseline": iou_vs_baseline(
                           a, base)}
                variants_out.append(row)
                print(f"    picked={a['picked_primary_id']} presence="
                      f"{a['primary_presence']} ids={a['n_ids']} "
                      f"decode_errors={a['decode_errors']} "
                      f"iou_vs_baseline={row['mean_iou_vs_person_baseline']}"
                      f" cost=${result.cost_usd}")
            print(f"ledger after {job['window']} variants: "
                  f"${ledger.total_usd():.4f}")
    report["prompt_variants"] = variants_out

    # ---- half 2: offline selection stress over every cached response
    stress = []
    for name, path, expected in CACHED_RESPONSES:
        if not path.exists():
            stress.append({"case": name, "missing": str(path)})
            print(f"stress {name}: MISSING {path}")
            continue
        frames = parse_sam_response(json.loads(path.read_text()))
        a = analyse(frames, sample_step=15)
        row = {"case": name, **public(a)}
        if expected is not None:
            row["expected_speaker_id"] = expected
            row["pick_correct"] = a["picked_primary_id"] == expected
        stress.append(row)
        print(f"stress {name}: ids={a['n_ids']} picked="
              f"{a['picked_primary_id']} presence="
              f"{a['primary_presence']} decode_errors={a['decode_errors']}"
              + (f" expected={expected} correct={row.get('pick_correct')}"
                 if expected is not None else ""))
    report["selection_stress"] = stress

    report["ledger_total_usd"] = ledger.total_usd()

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {REPORT}")
    print(f"ledger after: ${ledger.total_usd():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
