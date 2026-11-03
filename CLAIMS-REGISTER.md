# Claims register — AI Spatial Director

Every claim that may appear in the demo, README, one-pager, or
submission is listed here with its evidence. Statuses: **PROVEN**
(measured, source named) · **DECISION** (a locked choice, stated as
such) · **PENDING** (no evidence yet — may not appear anywhere) ·
**BANNED** (quarantined by the judge panel, Round 1 — must never
ship).

Opened 2026-10-29; this is the converged register folded in at the
Nov 2 repo gate (the 2026-10-29 register plus its 2026-10-31 addendum,
including Update U1, with the gate rulings applied). Owner: Kai. Grep
the quarantine list again at the Nov 4 pre-submit audit.

## PROVEN — measured, with source

| # | Claim | Evidence |
|---|---|---|
| 1 | One Director run turns one long video into up to 3 finished Shorts, each with a chained PASS/FAIL receipt | Pipeline runs on Holmes (CC-BY); receipts in repo `receipts/` |
| 2 | Latest full F1–F7 run: 2 Shorts PASSED_AFTER_REPAIR, 1 PASSED_FIRST_DRAFT (Holmes run 4, refined build) | `receipts/receipt-pipeline-holmes4-short{1,2,3}.json`; see A16 |
| 3 | Complete passing runs cost **$0.5070–$0.7223 per run** and **$0.169–$0.241 per finished Short**, fully loaded on an API-cost-only basis (per-Short calls + shared full-source transcription; infrastructure is in no ledger and no infra-inclusive figure is claimed). Framing rule: the $0.4407 pipeline-run-1 was the pre-fix run whose 3 finals all FAILED verification — it must never be cited as the cost of a passing run | Per-run run-summaries + ledger attribution, tabulated in the economics draft §1–§3 (2026-10-31); Belew figures per the reconciliation section below |
| 4 | Cumulative event spend **$21.9867** of the $150 grant (850 ledger lines; final merge 2026-11-04 — every run ledger from both build lines unioned, deduplicated by request id where the API returned one, calls logged without a request id counted at logged multiplicity), $145 hard cap enforced in `spend_ledger.py`. Model split: SAM 3.1 $11.7224 · Muse Wonder $4.7677 · Muse Spark 1.3 $3.2519 · Muse Voice Transcribe $2.1047 · Muse Image $0.1400 | Direct aggregation of repo `spend-ledger.jsonl`, re-performed 2026-11-04 |
| 5 | SAM 3.1 tracks the speaker in 100% of frames across 3 test clips; 1,679 masks decoded, zero errors; consecutive-frame IoU means 0.980–0.995 | Spike record, BUILDLOG 2026-10-29 ~01:30 IST |
| 6 | SAM 3.1 video costs exactly $0.20 per 1,000 frames | Billed calls: 480 fr $0.096 / 600 fr $0.120 / 599 fr $0.1198 |
| 7 | The verifier runs the deterministic checks **F1–F7** on the actual render, then one Wonder judgement of the muted 5s hook (the F5 semantic leg — Wonder is a core component, not an add-on); at most ONE repair, picked by Spark 1.3 from a fixed menu, on cached artifacts. **F9** (every render) checks integrated loudness −14 LUFS ±1 and true peak ≤ −1.0 dBTP, with NORMALIZE_LOUDNESS in the repair menu. When a Short carries graphics, **F8** also runs — see the F8 statement in the graphics section below | `verifier.py`, `rescue_loop.py`; receipts carry per-check measured values + every Meta call |
| 8 | Speaker reframing + loudness normalisation fixed F3/F1 by construction: post-fix runs pass F1 and F3 on every window | holmes2-short1 + holmes3 receipts vs run-1 FAIL anatomy |
| 9 | The bounded-memory rewrite changed no output: the 14s window's draft-1 render is bit-identical pre/post fix (sha256 b4a69a75…), all F1–F7 values identical; 22s-window peak RSS 4,631 MB → 3,529 MB. (Suite figure at that fix was a point-in-time 148/148; the current suite figure is #10) | BUILDLOG 2026-10-29 (OOM diagnosis entry) |
| 10 | Test suite: **626/626 passing** | Final merged suite run at the 2026-11-04 finalize (the closure build measured 622; the 4 bundled-evidence fallback tests ported from the submission line at finalize brought it to 626) |
| 11 | Every runtime model in the shipped pipeline is a Meta model: SAM 3.1, Muse Voice Transcribe, Muse Spark 1.3, Muse Wonder, Muse Image (Meta Model API) + Muse Spark Open (Baseten) | Pipeline wiring in `pipeline.py`; `spend-ledger.jsonl` + `partner-usage-log.jsonl` show ≥1 logged call per model |
| 12 | Beachhead enumeration: 48 channels measured by a stated method; **1 fully verified** (Erin Talks Money, 444k subs) after the single-speaker spot-check. Widen-or-publish remains the founder's call; no larger count may be stated | `beachhead-enumeration-2026-10-29.md` + raw JSONL (build records) |
| 13 | Voice Transcribe returns turn-level timestamps only, no punctuation — captions are phrase-timed via a Spark punctuation pass + non-model DSP alignment | First live VT call 2026-10-29; `caption_align.py` |
| 24 | Final production batch on the round-6 code (2026-11-03): 10 receipted Shorts — **3 PASSED_AFTER_REPAIR, 7 FAIL** — plus the geometry-bundle proof run (PASSED_AFTER_REPAIR). Every receipt ships, FAILs included | `receipts/receipt-pipeline-fix6-*.json`, `receipts/receipt-pipeline-hero7-*.json`, `receipts/receipt-pipeline-bundleproof-qm6seg-short1.json` (verdicts read from the receipt files) |
| 25 | The shipped re-verifier carries both verification lines merged: per-draft **geometry bundles** (inputs rebuilt from the run's recorded declarations, trusted only after the bundle's inputs hash verifies; F8 NOT-APPLICABLE for no-graphics drafts; a final FAIL after an executed repair is a consistent chain) and the **bundled-evidence fallback** (basename resolution against the shipped `evidence/` tree, every fallback reported as a RESOLVED line, never silent). The bundle-proof receipt re-verifies **ok:true** offline. Two superseded fix-5-era receipts re-verify ok:false with a proven cause (evidence decoded by replaced matte code) — documented in BUILDLOG, kept as history | `tools/reverify.py`, `tests/test_reverify.py` (42 tests incl. the 4 ported fallback cases), full suite #10; BUILDLOG 2026-11-03 closure + 2026-11-04 finalize entries. **Correction (2026-11-04, later in the finalize):** the bundled end-to-end example originally shipped the Holmes run-4 evidence set; fresh-clone checks on two machines showed that receipt's geometry legs re-measure with deltas under the round-6 finish (recorded F2 glyph occlusion 0.1246 → 0.1316 recomputed) — the same proven superseded-finish cause as the fix-5 pair, the finish math having changed in rounds 4–6 after that receipt was measured. The Holmes run-4 receipts join the fix-5 pair in that documented category (chains still verify). The bundled set is now the round-6 QM6 showcase Short (`evidence/qm6-fix6-short1/`), which re-verifies ok:true end-to-end from a fresh clone, and the proof pack's committed-receipt case follows the same example |

## Benchmarks — 2026-10-29 afternoon (measured)

| # | Claim | Evidence |
|---|---|---|
| A14 | Provider comparison (Matrix B): Baseten serves the shipped Spark Open sweep at 8.71s mean vs Fireworks 40.94s on identical input, with quality parity (top score 0.96 both providers). Baseten = primary, Fireworks = proven fallback | `benchmark-matrix-b-results-2026-10-29.md`; `partner-usage-log.jsonl` |
| A15 | SAM prompt (Matrix D): 'person' is the right prompt — 'speaker' returns zero/near-zero detections (a trap), 'man' is equivalent but gendered, 'man speaking' slightly worse; the largest-area persistent-id selection rule is correct on 9/9 cached stress cases including the background-TV trap. A minimum-area plausibility floor applies, so a sole tiny object fails loud instead of being picked silently | `benchmark-matrix-d-results-2026-10-29.md` |
| A16 | Holmes run 4 on the refined build (paid, 2026-10-29 evening): 3/3 PASS — short 1 PASSED_AFTER_REPAIR (draft 1 FAIL F5 only → TRIM_SILENCE), short 2 PASSED_AFTER_REPAIR (same), short 3 PASSED_FIRST_DRAFT. F2 glyph occlusion draft 1 → final: 0.1246 → 0.1276 / 0.1298 → 0.1321 / 0.1325 — all inside the 0.08–0.20 band. Run cost $0.6097 | `receipts/receipt-pipeline-holmes4-short{1,2,3}.json`; BUILDLOG 2026-10-29 evening |
| A17 | Belew F1–F7 leg: the first attempt (2026-10-29) completed short 1 only (PASSED_AFTER_REPAIR, `receipts/receipt-pipeline-belew1-short1.json`) before a runtime restart killed short 2 — not a pipeline fault. **The main-session retry completed 3/3 PASSED_AFTER_REPAIR at $0.6507** (per-Short $0.2435 / $0.1894 / $0.1568 + $0.0610 shared transcription), receipts `receipts/receipt-pipeline-belew2-short{1,2,3}.json`. The retry figure is the citable Belew F1–F7 cost; see the reconciliation section for the first attempt's spend | BUILDLOG 2026-10-29 night Belew-complete entry; run-summary for pipeline-run-belew-2 |

## DECISIONS — locked choices, stated as choices (not results)

| # | Claim | Source |
|---|---|---|
| 14 | Beachhead: single-speaker long-form podcasters + talking-head educators, 20k–500k subs, ≥2 long videos/week, currently paying for an OpusClip-class clipper | Locked by Kamlesh 2026-10-29 00:37 IST |
| 15 | Director price: 30 credits per run ($2.90 Basic / $1.90 founding) | Locked 2026-10-29 00:37 IST |
| 16 | Strict runtime posture: Meta models only at runtime — adopted before it was confirmed by name in CV's 2026-10-29 06:11 IST ruling (Whisper/MediaPipe banned at runtime). MediaPipe appears only as an offline bake-off comparator, never in shipped code | Decision doc LOCKED 2026-10-29; Discord monitor log 07:23 sweep |
| 17 | Demo voiceover is ElevenLabs Eric (neutral), chosen 2026-11-03 after the founder compared it against his own recorded takes and judged it better; his takes are retained as backup | Founder decision 2026-11-03, superseding the 2026-10-29 lock |

## PENDING — may not appear anywhere until the evidence exists

| # | Claim-in-waiting | Evidence due |
|---|---|---|
| 18 | Any A/B preference figure vs an OpusClip-class incumbent. **PRE-REGISTERED DECISION THRESHOLD — not a result:** the claim may ship only if a blind A/B measures a preference of at least ≥68% among ~50 strangers for the Director output; below that threshold the claim is deleted. No blind A/B has run; no preference figure exists | Blind A/B execution (kit built 2026-10-31); recruitment blocked on creator outreach |
| 19 | Verifier precision/recall per rule. SYNTHETIC HALF DONE: **44 labelled cases, exact label/verdict agreement 44/44, per-rule precision/recall 1.000** (fault injections + controls, incl. the F9 legs), plus the receipt re-verifier proof section **4/4** — valid ONLY as "the verifier catches the injected faults it was designed to catch" (`tools/proof_pack.py`). Real-render accuracy still pending the founder's blind pass | Blind pass over real renders |
| 20 | Creator demand: switch/pay verdicts — still pending, none received. Outreach state (2026-11-03/04): 75 creators emailed at the locked 30-credit price (74 delivered); replies are coming in; ONE creator with 278,000 subscribers is open to a private test run under written consent — private-only terms: no public use of his name, likeness or feedback without separate written approval. The ONLY public form permitted is the aggregate: 'a creator with 278,000 subscribers is open to testing it, after the sample and the full disclosures on the models' | Outreach waves 1–3 (2026-11-03) + reply sweep records; verdict forms follow any consented run |
| 21 | SAM on a held-object case — **PROVEN 2026-11-01, no longer pending.** On the NASA ISS interview (CC-BY, and a US federal work), the speaker holds a microphone throughout her answers; in the McArthur re-run's two F1–F9-passing Shorts, SAM's speaker track is present and contained in **657/657 and 650/650 frames (F4 containment 1.0)**, microphone in hand — the held object never breaks the track. The clip-C hand-over-mouth substitute is retired | `receipts/receipt-pipeline-graphics-mcarthur2-short{1,2}.json` (F4 measured legs); first exercised in the run-1 receipts (G6), proven on passing receipts in G7 |
| 22 | Per-run cost typicality beyond the measured runs. Cost figures may be cited per measured run (with the run named), never as typical | Cohort runs (blocked on outreach consent) |
| 23 | Door-to-door timing typicality (~56–95 min per 3-Short run across measured runs; cite per run, never as typical) | Cohort runs (blocked on outreach consent) |

## Graphics layer — rule state and live evidence

**F8 as amended (Kamlesh ruling 2026-10-31 01:34 IST).** The
minimum-gap rule between graphics and the 0–3s hook-window ban were
REMOVED: timing is free — graphics may start at 0.0s and run
back-to-back or overlapping in time. F8's spatial discipline is at
0 px and stricter than the original F8: face region PLUS its 28px
margin = 0 px overlap; caption block and caption zone = 0 px; hook
glyph zone while a span overlaps the hook window = 0 px; pairwise
collision between time-overlapping graphics = 0 pairs. Presence
in-span / absence out-of-span (±0.5s) is pixel-measured against each
graphic's declared layer. Block 2 added three further F8 legs —
motion-path fidelity (mid-travel pixel match ≥ 0.75), SFX audibility
against the pre-mix speech stem, and backdrop containment. Guards
unchanged throughout: verbatim source-quote guard, numeric-value
guard, named people get typographic cards only, digit-free image
prompts with face/likeness blocklist, span bounds 2.5–6.0s, at most
ONE repair (DROP_GRAPHICS for F8 failures). No issued receipt is
contradicted by the amendment: placement enforced the wider zones
from the first graphics build, and the two removed timing legs only
relax verdicts.

- **G1 — First graphics proof run (2026-10-30):** 3 finished, 2
  passing; short 2 FAILED (F7) after its one repair; run cost
  $0.5785. The planner starved on all 3 Shorts in this run
  (token-budget defect, since fixed), so this run evidences the
  F8/repair path, NOT graphics placement.
- **G2 — Holmes graphics run 2 (2026-10-30, original timing rules):**
  3/3 PASSED_AFTER_REPAIR; graphics placed on 2 of 3 Shorts — card
  "CD and the Cowboys" (short 1), card "Dak Prescott" (short 2), no
  graphic placed on short 3; run cost $0.7223.
- **G3 — Holmes graphics run 3 (2026-10-31, current F8 rules):**
  short 1 PASSED_AFTER_REPAIR (F8 passed on both drafts, zero
  collisions); short 2 PASSED_FIRST_DRAFT — F8 passed with one
  graphic: kind `image`, motion `ken_burns`, all spatial overlaps
  0 px, presence match 0.966. That image is the **first live Muse
  Image placement inside the pipeline** ($0.01). Shorts 1–2 cost
  $0.4089 (ledger-attributed). **Short 3: NO RESULT** — killed
  mid-verification by a VM restart (~03:09 IST, 2026-10-31); a draft
  exists, there is no final and no receipt. Short 3 must never be
  recorded as a pass.
- **G4 — Holmes graphics run, block-2 code (2026-10-31, `holmes6`):**
  **3/3 passing F1–F8** (shorts 1–2 PASSED_AFTER_REPAIR via F5
  TRIM_SILENCE; short 3 PASSED_FIRST_DRAFT), run cost $0.6193,
  receipts `receipts/receipt-pipeline-graphics-holmes6-short{1,2,3}.json`.
  Live-proven on real renders: context cards under the size-to-space
  placement law (short 1 carried THREE cards back-to-back, spans
  0.08–3.52 / 3.52–6.74 / 7.60–10.36s, zero collisions) and
  procedural SFX (pop ×3 on short 1, ding on shorts 2–3, declared
  on the receipts and measured by the F8 SFX leg against the speech
  stem).
- **G5 — Belew graphics run (2026-10-31, block-2 code):** **0/3 —
  all three Shorts FAIL**, run cost $0.6572 (run-summary verified),
  receipts `receipts/receipt-pipeline-graphics-belew2-short{1,2,3}.json`
  (FAIL evidence, kept deliberately). Anatomy: F1–F4, F6, F7 and F8
  PASS on all three final drafts; **F5 FAIL on all three** (real
  source pauses: maximum speech gaps 1.16 / 0.94 / 0.80s vs the
  0.45s cap); **F9 FAIL on all three** (the NORMALIZE_LOUDNESS
  repair landed integrated loudness in band at −14.58…−14.61 LUFS
  but true peak overshot to +0.11 / +0.11 / +0.13 dBTP vs the
  −1 dBTP cap). Draft 1s failed F5+F9 (short 1) and F5+F8+F9
  (shorts 2–3); Spark picked the loudness repair, so with one repair
  max the F5 TRIM_SILENCE never ran. Graphics placement itself
  worked (3 graphics per Short, F8 clean on final drafts), with live
  firsts: two **count-up cards** on short 3 ("12 weeks of rehearsal
  time", "After 17 days") with tick SFX keyed to the count, and two
  **Muse Image Ken Burns stills** on short 1. Flagged nuance,
  unexplained: shorts 2–3 failed F8 on draft 1 but pass F8 on the
  final draft although the loudness repair copies the video stream
  untouched — this is not explained by the repair and is recorded
  here rather than smoothed over. **Consequence: the graphics
  default-on flip is NOT met** (it required a Belew 3/3 pass on
  F1–F8 under the same code); the layer stays opt-in (`--graphics` /
  `GRAPHICS_LAYER=1`).
- **G6 — McArthur graphics run 1 (2026-11-01, NASA ISS interview,
  CC-BY + US federal work):** **0/3**, run cost $0.7858, receipts
  `receipts/receipt-pipeline-graphics-mcarthur1-short{1,2,3}.json`
  (FAIL evidence). Shorts 1–2 passed F1–F8 completely on both
  drafts and failed **F9 only**, on the NORMALIZE_LOUDNESS
  true-peak fault (post-repair peaks +0.52 / +0.59 dBTP vs the
  −1 cap; short 1's integrated also missed the floor by 0.14 LU).
  Short 3 failed F8+F9: the sweep picked the split-screen intro
  window, SAM returned 155 of 659 frames, and the render truncated
  to a 5.2s stub. Live firsts: **backdrop mode** (short 2, galaxy
  field, 15.3–17.86s), **speaker-forward spans** (shorts 1–2), and
  the **held-object case exercised** (speaker + hand-held
  microphone tracked, F4 containment 1.0).
- **G7 — McArthur graphics re-run (2026-11-01 evening, after the
  NORMALIZE_LOUDNESS rework, remote head e3ab608):** **2/3
  PASSED_AFTER_REPAIR under the full F1–F9 set — the first
  all-passing outputs under all nine checks**, run cost $0.7781,
  receipts `receipts/receipt-pipeline-graphics-mcarthur2-short{1,2,3}.json`.
  Short 1 (21.9s): draft 1 failed F9 only (−22.59 LUFS /
  −0.19 dBTP) → repaired to **−13.91 LUFS / −1.53 dBTP**, all
  checks green. Short 2 (21.6s): −21.74 / −0.76 → **−14.49 /
  −1.42**, all green; backdrop mode selected again (galaxy
  field, 14.52–17.41s) and a speaker-forward span (19.33–22.0s).
  Short 3 FAIL on **F8 only** — the same split-screen window
  produced the same 5.1s stub (its F9 now passes on draft 2:
  −14.37 / −1.05); the loudness fault is closed on all three,
  and the remaining defect is the unbuilt SAM-coverage guard on
  window selection, recorded here rather than smoothed over.
- **Hero modes — wording rule.** **Backdrop mode and
  speaker-forward spans are now LIVE-PROVEN** (McArthur runs,
  G6/G7: galaxy-field backdrops on short 2 of both runs;
  speaker-forward spans on run-1 shorts 1–2 and run-2 short 2,
  declared on the receipts with their transforms). **Motion
  paths remain built and test-verified only** — the planner has
  still never selected one on a live run, and they may not be
  stated as proven on a real render. Count-up cards and Ken
  Burns image stills ARE live-proven (G5), and context cards +
  procedural SFX are live-proven (G4, G5).

## Belew cost reconciliation (gate ruling, 2026-11-01)

Two Belew cost figures are **citable**, both verified against their
run summaries:

- **F1–F7 retry (pipeline-run-belew-2): $0.6507** — per-Short
  $0.2435 / $0.1894 / $0.1568 + $0.0610 shared transcription; 3/3
  PASSED_AFTER_REPAIR; receipts in `receipts/`.
- **Graphics run (pipeline-run-graphics-belew-2): $0.6572** —
  per-Short $0.1970 / $0.2092 / $0.1900 + shared transcription; 0/3
  FAIL (G5); receipts in `receipts/` as FAIL evidence.

**Excluded: the $0.4042 ledger block.** `spend-ledger.jsonl` lines
143–156 sum to exactly $0.4042 (cumulative ledger $4.4821 → $4.8863):
three Voice Transcribe lines ($0.027 + $0.027 + $0.007 = $0.0610,
a full-source transcription), short-1 calls (SAM $0.0936, caption
punctuation $0.0077, hook $0.0096, Wonder $0.0200, repair planner
$0.0029, Wonder $0.0229), and short-2 calls (SAM $0.1056, hook
$0.0083, a $0.0446 `verifier-f5-muted-hook-empty-attempt`, Wonder
$0.0245, repair planner $0.0035). The block is described two
incompatible ways in the build records: BUILDLOG's first Belew entry
and the original A17 attribute it to the first Belew attempt — and
the surviving `receipt-pipeline-belew1-short1.json` does record
calls matching lines 149–151 to the cent (Wonder $0.0200, repair
$0.0029, Wonder $0.0229) — while the economics draft (§8 item 1)
describes the same lines as an unattributed attempt with no
surviving summary or receipts, attributes belew-1 differently in its
§2/§3 ($0.3923 finished + $0.1382 abandoned), and its §7(b)
cross-check figures (Wonder $0.0384 + repair $0.0027) do not match
the surviving belew1-short1 receipt. The two descriptions cannot
both be correct, and this register does not adjudicate between them.
**Ruling: the $0.4042 block is excluded as unattributed/untraceable
for citation purposes.** No Belew cost beyond the two run-summary
figures above may be cited, and $0.4042 may not be cited as the cost
of any named run. The block remains counted inside the cumulative
ledger total (#4), which is a ledger sum, not a per-run claim. The
economics draft's belew-1 table row is likewise not carried into
this register's citable figures.

## BANNED — quarantine (judge panel Round 1; grep at Nov 1 + Nov 4)

`68%` · `45 cents` · `$0.42` · `38.2%` · `34.1%` · `14 seconds` · `~55%` ·
`412 channels` · "cheaper than OpusClip" (outright — our figures are
costs, the incumbent's published figures are prices; no comparison of
that form may be made) · any manufactured verifier failure · any
unconsented creator footage.

Quarantine note (gate ruling on finding Q1): the threshold string in
claim #18 is the single permitted exception — it appears only inside
#18's PENDING row, explicitly labeled as a PRE-REGISTERED DECISION
THRESHOLD, never as a measured result. Everywhere else, the strings
above appear only in this BANNED list itself.

## Compliance lines the submission must carry verbatim-in-substance

- Demo opens with one line stating what was pre-existing (ShortForge,
  the product) vs built during the event (the Director in this repo).
  Repo-side, the stronger rule applied throughout: anything
  pre-existing was EXCLUDED from this repo, not merely disclosed
  (pre-kickoff cutoff 12:24 PM ET, October 26, 2026).
- Muse Image: staff confirmed it is unavailable in India and that
  the unavailability will not count against the entry; calls from
  India succeeded (probe 2026-10-28, $0.01; live in-pipeline
  placements 2026-10-31, $0.01 each). Muse Image is a working
  component of the graphics layer with live placements recorded on
  receipts in this repo (G3, G5) — it is not described as
  non-load-bearing. The layer degrades gracefully to code-rendered
  cards when no image call is made, but Image placements are real
  and claimed as such.
- Credit-funded API calls are training-eligible (staff-confirmed, no
  zero-retention) — event work only ever went through credits, never
  ShortForge core IP or unconsented footage.
- Muse Spark Open provenance: staff confirmed
  `meta-models/muse-spark-open` on Baseten counts for Open Weights
  via the team's own account; every Spark Open call is logged in
  `partner-usage-log.jsonl`. Spark Open runs one pipeline stage
  (candidate moment scoring); it is not the pipeline's primary model.
- Demo footage is CC-BY (Adrian Belew interview, Ultimate Guitar;
  Mark Holmes commentary), credited in the README and demo credits,
  per the rights log maintained with the build records.
