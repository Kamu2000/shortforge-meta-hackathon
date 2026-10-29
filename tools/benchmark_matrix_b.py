#!/usr/bin/env python3
"""Matrix B — Spark Open: Baseten vs Fireworks (benchmark, not shipped code).

Runs the IDENTICAL candidate-moment sweep workload the pipeline uses
(same system prompt imported from ``spark_open_client``, same turns
fixture, same temperature 0.2, same 4000-token budget with the same
one bounded retry at 8000 on an empty reply) against both partner
providers, 3 repetitions each:

- Baseten:   https://inference.baseten.co/v1,
             model ``meta-models/muse-spark-open-1026``
- Fireworks: https://api.fireworks.ai/inference/v1,
             model ``accounts/fireworks/models/muse-spark-open``

Every completed call is appended to ``partner-usage-log.jsonl`` in
the shipped log format with the ``provider`` field distinguishing
the two (plus a ``rep`` field). Spend draws the two $100 partner
credit pools, NOT the Meta ledger - the only Meta-ledger spend here
is the one-time Voice Transcribe fixture regeneration (~$0.05),
because the pipeline does not persist its sweep turns; the turns are
generated ONCE and frozen to ``tools/benchmark_holmes_turns.json``
so both providers see byte-identical input.

Quality metric: overlap with the production Baseten picks recorded
in the run-3 summary (``pipeline-run-2/run-summary.json``). A
benchmark candidate matches a production candidate when their time
intervals overlap by >= 50% of the shorter interval.

Usage:
    python3 tools/benchmark_matrix_b.py [--skip-fixture]

Written fresh in-repo on 2026-10-29 for the Oct 30 benchmark matrix
(pulled forward to Oct 29). Stdlib + the shipped package only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")

from spatial_director.spark_open_client import (  # noqa: E402
    SWEEP_SYSTEM_PROMPT,
    SparkOpenError,
    _parse_candidates,
)
from spatial_director.spend_ledger import SpendLedger  # noqa: E402
from spatial_director.voice_transcribe import VoiceTranscribeClient  # noqa: E402

TS = "2026-10-29"
HOLMES_SOURCE = Path(
    "/home/hatch/workspace/meta-hackathon/spike-footage/holmes/source.mp4")
PRODUCTION_SUMMARY = Path(
    "/home/hatch/workspace/meta-hackathon/spike-footage/"
    "pipeline-run-2/run-summary.json")
FIXTURE = REPO / "tools" / "benchmark_holmes_turns.json"
REPORT = REPO / "tools" / "benchmark_matrix_b_report.json"
PARTNER_LOG = REPO / "partner-usage-log.jsonl"
LEDGER = REPO / "spend-ledger.jsonl"
VT_CHUNK_S = 540.0

PROVIDERS = {
    "baseten": {
        "base": "https://inference.baseten.co/v1",
        "model": "meta-models/muse-spark-open-1026",
        "vault": "custom.baseten",
    },
    "fireworks": {
        "base": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/muse-spark-open",
        "vault": "custom.fireworks",
    },
}


def _no_redirect_transport(method, url, headers, body):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            raise RuntimeError("redirect refused")

    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        return exc.code, {"error_body": detail}


def _vt_transport(url, headers, body):
    """VoiceTranscribeClient's transport shape: (url, headers, body)."""
    return _no_redirect_transport("POST", url, headers, body)


def _key_provider(vault_provider: str):
    from dynamic_credentials import dynamic_credential_entry

    return lambda: str(
        dynamic_credential_entry(vault_provider)["surrogate"]).strip()


# ---------------------------------------------------------------- fixture


def make_fixture(ledger: SpendLedger) -> list[dict[str, Any]]:
    """Regenerate the Holmes VT turns once and freeze them as the fixture.

    Mirrors pipeline._transcribe_all (540s chunks, offset-corrected
    turns) and the sweep-turn shaping in pipeline.run_pipeline.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(HOLMES_SOURCE)],
        capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())
    vt = VoiceTranscribeClient(
        api_key_provider=_key_provider("custom.meta-model-api"),
        ledger=ledger, transport=_vt_transport)
    work = Path("/tmp/matrix-b-vt")
    work.mkdir(parents=True, exist_ok=True)
    turns: list[dict[str, Any]] = []
    offset = 0.0
    while offset < duration:
        length = min(VT_CHUNK_S, duration - offset)
        wav = work / f"vt-chunk-{int(offset)}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}",
             "-t", f"{length:.3f}", "-i", str(HOLMES_SOURCE),
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            check=True)
        result = vt.transcribe_file(
            wav, purpose="benchmark-matrix-b-fixture", ts=TS)
        for t in result.turns:
            start = ((t.start_ms or 0) + int(offset * 1000)) / 1000.0
            end_ms = (t.end_ms + int(offset * 1000)
                      if t.end_ms is not None else None)
            end = end_ms / 1000.0 if end_ms is not None else start + 5.0
            if end <= start:
                end = start + 5.0
            if t.text.strip():
                turns.append({"start_s": start, "end_s": end,
                              "text": t.text})
        print(f"fixture: transcribed {offset:.0f}-{offset + length:.0f}s")
        offset += length
    turns.sort(key=lambda t: t["start_s"])
    FIXTURE.write_text(json.dumps({
        "source": str(HOLMES_SOURCE),
        "created": TS,
        "note": "Holmes full-source Voice Transcribe turns, generated "
                "once for Matrix B so both providers sweep identical "
                "input. VT turn segmentation is not bit-deterministic; "
                "this frozen file is the benchmark's fixed input.",
        "turns": turns,
    }, indent=2), encoding="utf-8")
    print(f"fixture: {len(turns)} turns frozen to {FIXTURE}")
    return turns


# ---------------------------------------------------------------- one call


def sweep_once(provider: str, turns: list[dict[str, Any]], rep: int,
               ) -> dict[str, Any]:
    """One sweep call with the shipped client's exact payload + retry."""
    cfg = PROVIDERS[provider]
    lines = [f"[{float(t['start_s']):.1f}-{float(t['end_s']):.1f}] "
             f"{t['text']}" for t in turns]
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SWEEP_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    key = _key_provider(cfg["vault"])()
    budgets = [4000, 8000]
    first_empty_id = None
    for attempt, budget in enumerate(budgets):
        payload["max_tokens"] = budget
        started = time.monotonic()
        status, raw = _no_redirect_transport(
            "POST", f"{cfg['base']}/chat/completions",
            {"Authorization": f"Bearer {key}",
             "Content-Type": "application/json"},
            json.dumps(payload).encode("utf-8"))
        latency = round(time.monotonic() - started, 3)
        if status != 200:
            return {"provider": provider, "rep": rep, "ok": False,
                    "http_status": status,
                    "error": str(raw)[:500], "latency_s": latency}
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {"provider": provider, "rep": rep, "ok": False,
                    "http_status": status,
                    "error": f"unexpected response shape: {str(raw)[:300]}",
                    "latency_s": latency}
        if text:
            candidates = _parse_candidates(text)
            usage = raw.get("usage") or {}
            entry = {
                "ts": TS,
                "provider": provider,
                "model": cfg["model"],
                "purpose": "benchmark-matrix-b-sweep",
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_s": latency,
                "request_id": raw.get("id"),
                "candidates": len(candidates),
                "rep": rep,
            }
            if first_empty_id is not None:
                entry["empty_first_attempt_request_id"] = first_empty_id
                entry["max_tokens_final"] = budget
            with PARTNER_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            return {
                "provider": provider, "rep": rep, "ok": True,
                "latency_s": latency,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "request_id": raw.get("id"),
                "empty_first_attempt": first_empty_id is not None,
                "max_tokens_final": budget,
                "candidates": [
                    {"start_s": c.start_s, "end_s": c.end_s,
                     "score": c.score, "reason": c.reason}
                    for c in candidates],
            }
        if attempt == 0:
            first_empty_id = raw.get("id")
    return {"provider": provider, "rep": rep, "ok": False,
            "empty_reply": True, "latency_s": latency,
            "error": "empty reply at both 4000 and 8000 token budgets"}


# ---------------------------------------------------------------- quality


def _overlap_match(a: dict, b: dict) -> bool:
    lo = max(a["start_s"], b["start_s"])
    hi = min(a["end_s"], b["end_s"])
    if hi <= lo:
        return False
    shorter = min(a["end_s"] - a["start_s"], b["end_s"] - b["start_s"])
    return shorter > 0 and (hi - lo) / shorter >= 0.5


def top3_overlap(cands: list[dict], reference_top3: list[dict]) -> int:
    top3 = sorted(cands, key=lambda c: -c["score"])[:3]
    hits = 0
    for ref in reference_top3:
        if any(_overlap_match(ref, c) for c in top3):
            hits += 1
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fixture", action="store_true",
                    help="reuse the frozen turns fixture if present")
    args = ap.parse_args()

    ledger = SpendLedger(LEDGER)
    print(f"ledger before: ${ledger.total_usd():.4f}")

    if args.skip_fixture and FIXTURE.exists():
        turns = json.loads(FIXTURE.read_text())["turns"]
        print(f"fixture: reused {len(turns)} frozen turns")
    else:
        turns = make_fixture(ledger)
    print(f"ledger after fixture: ${ledger.total_usd():.4f}")

    production = json.loads(PRODUCTION_SUMMARY.read_text())
    prod_candidates = production["sweep"]["candidates"]
    prod_top3 = sorted(prod_candidates, key=lambda c: -c["score"])[:3]
    print(f"production reference: {len(prod_candidates)} candidates, "
          f"top-3 starts {[c['start_s'] for c in prod_top3]}")

    results: list[dict[str, Any]] = []
    for provider in ("baseten", "fireworks"):
        for rep in (1, 2, 3):
            print(f"--- {provider} rep {rep} ...", flush=True)
            try:
                r = sweep_once(provider, turns, rep)
            except SparkOpenError as exc:
                r = {"provider": provider, "rep": rep, "ok": False,
                     "error": f"SparkOpenError: {exc}"[:500]}
            except Exception as exc:  # transport-level, recorded not hidden
                r = {"provider": provider, "rep": rep, "ok": False,
                     "error": f"{type(exc).__name__}: {exc}"[:500]}
            if r.get("ok"):
                r["top3_overlap_vs_production"] = top3_overlap(
                    r["candidates"], prod_top3)
                r["top_score"] = max(
                    (c["score"] for c in r["candidates"]), default=None)
                r["n_candidates"] = len(r["candidates"])
                print(f"    ok: {r['n_candidates']} candidates, "
                      f"latency {r['latency_s']}s, "
                      f"tokens {r['prompt_tokens']}+"
                      f"{r['completion_tokens']}, "
                      f"top3-overlap {r['top3_overlap_vs_production']}/3")
            else:
                print(f"    FAILED: {r.get('error')}")
            results.append(r)

    report = {
        "matrix": "B",
        "date": TS,
        "workload": "pipeline candidate-moment sweep (SWEEP_SYSTEM_PROMPT "
                    "imported from spark_open_client), Holmes full-source "
                    "VT turns fixture, temperature 0.2, max_tokens 4000 "
                    "with one bounded retry at 8000 on empty reply",
        "turns_fixture": str(FIXTURE),
        "n_turns": len(turns),
        "production_reference": {
            "source": str(PRODUCTION_SUMMARY),
            "n_candidates": len(prod_candidates),
            "top3": prod_top3,
        },
        "results": results,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {REPORT}")
    print(f"ledger after: ${ledger.total_usd():.4f} (partner calls draw "
          "partner credit, not the ledger)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
