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

## 2026-10-28 (late evening) - Spark Open client + first live sweep

- `src/spatial_director/spark_open_client.py` written fresh: Baseten chat-completions client for the candidate-moment sweep (Spark Open proposes, Spark 1.3 final-picks - the split is declared in the README). Model id is a constant: the partner credit covers Spark Open only. Every call appends to `partner-usage-log.jsonl` (separate from the Meta spend ledger; this is partner credit, not the $150).
- First live verification, 2026-10-28: call 1 returned HTTP 200 with null content - Spark Open spent all 400 completion tokens on reasoning (the same failure class as the first Wonder calibration call). Call 2 at a 2,500-token budget returned 3 parsed candidates, top score 0.96 on the stat hook, 4.8s latency. Client hardened accordingly: default max_tokens 4,000 and an empty reply raises instead of reading as "no candidates". 18/18 tests pass.
- Both verification calls are recorded in `partner-usage-log.jsonl` with request ids - this log is the evidence trail for partner verification of the Open Weights entry.

## 2026-10-28 (night) - finish stack built ahead of the spike

- `src/spatial_director/matte_finish.py` and `src/spatial_director/temporal_smooth.py` written fresh: guided-filter matte refinement + choke + feather, and EMA temporal smoothing with a scene-cut reset (IoU guard). Pure NumPy - these modules consume decoded mask arrays, so they do not depend on the SAM wire format and could be built and unit-tested on synthetic masks tonight (23/23 tests) while the spike waits on footage. When the spike lands, they take real masks unchanged.

## 2026-10-28 (night) - FIRST LIVE SAM CALL: the spike is unblocked

- Footage problem solved without waiting: two Creative Commons (CC BY) sources were verified in YouTube's license metadata and logged in the team's rights log - an Adrian Belew interview (Ultimate Guitar) and a Mark Holmes commentary. Three 20s spike clips were cut: A (Belew, talking head with a head tilt, t=100-120s), B (Holmes, hand gestures crossing the chest, t=168-188s), C (Holmes, hand-over-mouth/chin occlusion, t=726-746s - substituted for the held-object case, which neither CC source contains; the substitution is declared in the rights log).
- First live SAM 3.1 call on clip A: file upload HTTP 200, Responses HTTP 200 in 36s. usage.video_frames_processed = 480 (exactly the clip's 480 frames); billed $0.096, recorded in spend-ledger.jsonl (cumulative $0.968 of the $145 cap).
- Wire format reconciled on first contact: output_text carries one record per frame (`<Nf>` markers 0-479, all frames present), exactly ONE object (id 0) in every frame - the single speaker, tracked. Box + base85 mask tokens as documented. A trimmed real response (first 3 frame records) is committed at tests/fixtures/sam_response_clipA_trimmed.json as the canonical fixture for the mask decoder.
- Remaining spike work: decode masks, run clips B + C, head-geometry bake-off, GREEN/YELLOW/RED verdict.

## 2026-10-28 (night) - clip A masks decoded and verified

- All 480 masks from the first SAM response decoded with Meta's official `meta-sam-parser` library (the lossless payload is a context-coded arithmetic stream with a custom base85 alphabet - not plain base85 raster bytes; the scaffold's A2 assumption is falsified by live data and the official parser is the correct decoder; it will be declared as a dependency and labeled in the README helpers table).
- Measured on clip A: mask coverage 0.62-0.65 of the box raster across all frames; frame-to-frame mask IoU mean 0.997, min 0.981 - raw SAM output is already highly stable on clean talking-head footage. Visual overlay check (frames 0/105/479): the silhouette follows cap brim, face and shoulders with no background spill.
- Note: mask raster dimensions vary slightly per frame (the raster is box-sized and the box jitters by a few px) - compositing must place each raster at its box origin, not assume a fixed grid.
- Remaining spike work: clips B + C calls + decode, head-geometry bake-off, GREEN/YELLOW/RED verdict.
