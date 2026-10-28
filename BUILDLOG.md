# Build log

Dated record of the work in this repository, written as it happens during the event (October 26 – November 4, 2026).

## 2026-10-27 — Setup and verification

- Meta Model API key connected and verified live: the account's model list returns 9 models, including `muse-wonder-1.0`, `sam-3.1`, `muse-spark-1.3`, and `muse-voice-transcribe-1.0`.
- Baseten partner credit verified live: `meta-models/muse-spark-open-1026` is available for scoring workloads.
- Billing guardrail set before any paid call: the $150 event credit is the hard ceiling (harness-enforced cap at $145, target spend inside the event window, zero spend from the team's own funds).

## 2026-10-28 — Repository created

- Public repository created empty and seeded with README (event disclosure, labeled open-source resources), BUILDLOG, and .gitignore (secrets and local config excluded from the first commit).
- Code modules are rewritten fresh inside this repository from this point on, in small dated commits.

## 2026-10-28 (evening) — Concept locked; spend ledger lands

- Concept locked by the founder: the **Verifiable Spatial Director** (long video → spatially composited Shorts + post-render F1–F7 verifier + one repair max + PASS/FAIL receipt), Startup Track, "Build for Your World".
- `src/spatial_director/spend_ledger.py` implemented: cumulative, append-only JSONL ledger with a hard cap of $145 against the $150 credit grant (the platform Spend Limit is alert-only, so the cap is enforced in code — every paid call is pre-flight checked, and recording a call that would pass the cap raises instead of appending). Opening entries total **$0.872** (three Muse Wonder decision calls + one Muse Image probe, all 2026-10-28); `spend-ledger.jsonl` at the repo root is the ledger file. This makes the Oct 27 billing-guardrail entry above accurate as of today: the harness-enforced cap now exists in code, with unit tests.
- `src/spatial_director/sam_client.py` written fresh: Responses API, one bare noun phrase per request enforced, default concept `person`, 15,000-frame limit, worst-case cost estimate checked against the ledger before any request is sent, actual billed frames recorded from the response usage block. Wire format to be reconciled against the first live response at the spike.
- README rewritten to the converged design: claimed Meta stack is SAM 3.1 + Muse Wonder + Muse Spark Open (Baseten) only; a model is claimed only once it has a logged call in the shipped pipeline.
- Honest status: the Day-1 SAM spike (planned for today) has **not** run yet — it is waiting on test footage. The slip is declared here rather than back-filled; the spike runs as soon as footage lands.
- Staff runtime-model guidance tightened again this evening (a third strict statement named Whisper as not allowed at runtime); the strict-fallback decision point is Oct 29, 12:00 IST, scoped in the team's plan.
