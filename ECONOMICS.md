# ECONOMICS — Verifiable Spatial Director

Staged for the submission at the Nov 2 repo gate (2026-11-01). Every
figure below is traced to a named receipt, a run-summary, or a ledger
aggregation actually performed on the repo's `spend-ledger.jsonl`
(method in §7). Anything excluded is listed in §8 with the reason, and
is used nowhere in the tables. Figures are consistent with
`CLAIMS-REGISTER.md` (claims #3 and #4 and the Belew cost
reconciliation).

All cost figures are **API-cost-only**: the per-Short calls plus the
run's shared full-source transcription. Infrastructure is recorded in
no ledger and no infra-inclusive figure is claimed (§8, item 5).

## 1. Headline figures (verified)

- **Cost per finished Short** across the complete passing runs
  tabulated in §2 (fully loaded as defined above):
  **$0.169–$0.241**.
- **Cost per Director run** (3 Shorts from one long video) across the
  same runs: **$0.5070–$0.7223**.
- **Cost per finished minute of output** across the same runs:
  **$0.646–$0.803**.
- **Marginal cost per minute of source video** across the same runs:
  **$0.0296–$0.0422** — set against OpusClip's published retail
  **price** of $0.10 per minute of source in §6 (a cost and a price;
  the two are never conflated).
- **Price per run (locked 2026-10-29): 30 credits = $2.90 Basic /
  $1.90 founding.** Gross margin on API cost at the measured run
  costs: **75.1–82.5% (Basic)**, **62.0–73.3% (founding)** (§5).
- Total event API spend (all R&D, not just production):
  **$11.0330 across 401 ledgered calls** (§4; updated 2026-11-01
  evening — the gate figure was $9.4691 / 348), of the $150 grant,
  under the $145 hard cap enforced in `spend_ledger.py`.

The two block-2 graphics runs of 2026-10-31 are tabulated in §2
alongside the earlier runs (register entries G4 and G5 in
`CLAIMS-REGISTER.md`). The Holmes graphics run (holmes6, $0.6193,
3/3 passing F1–F8) is a complete passing run and sits inside every
range above. The Belew graphics run ($0.6572) FAILED all three
Shorts and is excluded from every range — its cost is disclosed in
§2, not folded into any passing figure. The two McArthur runs
(2026-11-01, NASA ISS source) are likewise tabulated in §2 and
likewise outside every §1 range: run 1 ($0.7858) produced zero
passing Shorts (F9 repair fault), and the re-run ($0.7781)
produced the first two F1–F9-passing Shorts but is not a
*complete* passing run (short 3 failed F8), so the ranges —
computed on complete passing runs only — do not move.

## 2. Per-run economics

Definitions: a *finished Short* = a `final.mp4` plus a chained receipt
with a recorded final verdict. *Passing* = `PASSED_FIRST_DRAFT` or
`PASSED_AFTER_REPAIR`. A *complete passing run* = a run that finished
all 3 Shorts with all 3 passing; the §1 ranges are computed on those
runs only. Costs are the run totals recorded in each run's
`run-summary.json` (per-Short costs plus shared full-source Voice
Transcribe), except the run marked †, whose summary was never written
and whose costs are ledger-attributed (§7). Durations are ffprobe
measurements of the finished `final.mp4` files. Sources: Holmes
commentary, 1026.554 s (17.109 min); Belew interview, 1219.210 s
(20.320 min); McArthur ISS interview, 1035.053 s (17.251 min).

| Run (source) | Run cost | Finished / passing | $ / finished Short | $ / finished minute of output | $ / minute of source | Wall-clock |
|---|---|---|---|---|---|---|
| pipeline-run-1 (Holmes, pre-fix) | $0.4407 | 3 / 0 | $0.1469 | $0.5925 | $0.0258 | 57.4 min |
| pipeline-run-2 (Holmes) | $0.5070 | 3 / 3 | $0.1690 | $0.6834 | $0.0296 | 60.1 min |
| pipeline-run-3 (Holmes, "run 4") | $0.6097 | 3 / 3 | $0.2032 | $0.7474 | $0.0356 | 56.0 min |
| pipeline-run-belew-2 (Belew) | $0.6507 | 3 / 3 | $0.2169 | $0.6459 | $0.0320 | 81.5 min |
| pipeline-run-graphics-holmes (graphics, 1st) | $0.5785 | 3 / 2 | $0.1928 | $0.7666 | $0.0338 | 80.2 min |
| pipeline-run-graphics-holmes-2 | $0.7223 | 3 / 3 | $0.2408 | $0.7784 | $0.0422 | 94.9 min |
| pipeline-run-graphics-holmes-3 (new F8 rules) † | $0.4089 | 2 / 2 | $0.2045 | $0.7483 | $0.0239 | — (§8) |
| pipeline-run-graphics-holmes-6 (Holmes, graphics, block-2) | $0.6193 | 3 / 3 | $0.2064 | $0.8030 | $0.0362 | 72.0 min |
| pipeline-run-graphics-belew-2 (Belew, graphics, block-2) — **FAILED** | $0.6572 | 3 / 0 | $0.2191 | — | $0.0323 | 66.1 min |
| pipeline-run-graphics-mcarthur-1 (McArthur, graphics) — **FAILED (F9)** | $0.7858 | 3 / 0 | $0.2619 | — | $0.0456 | 47.7 min |
| pipeline-run-graphics-mcarthur-2 (McArthur, graphics, F9 repair reworked) | $0.7781 | 3 / 2 | $0.2594 | $0.9598 | $0.0451 | 49.4 min |

**The first Belew attempt is not in the table.** Its spend is a
contiguous block of the ledger — lines 143–156, totalling exactly
**$0.4042**: a full-source transcription ($0.027 + $0.027 + $0.007 =
$0.0610), short-1 calls (SAM $0.0936, caption punctuation $0.0077,
hook $0.0096, Wonder $0.0200, repair planner $0.0029, Wonder
$0.0229), and short-2 calls (SAM $0.1056, hook $0.0083, a $0.0446
`verifier-f5-muted-hook-empty-attempt`, Wonder $0.0245, repair
planner $0.0035). Short 1 finished **PASSED_AFTER_REPAIR**, and its
receipt survives in this repo (`receipts/receipt-pipeline-belew1-short1.json`);
the receipt's recorded calls match ledger lines 149–151 to the cent
(Wonder $0.0200, repair planner $0.0029, Wonder $0.0229). The attempt
did not complete as a run — a runtime restart killed short 2, which
was not a pipeline fault. The block is therefore excluded from every
per-Short and per-minute figure in this document **because those
figures are computed on complete passing runs only** — not because
the spend is untraceable: the full $0.4042 is counted inside the §4
ledger total. No per-Short split of the block is claimed. The citable
Belew run costs are the F1–F7 retry (**$0.6507**, tabulated above)
and the 2026-10-31 graphics run (**$0.6572**, 0/3 FAIL —
tabulated above; `CLAIMS-REGISTER.md` G5).

Notes on the imperfect rows — reported, not hidden:

- **pipeline-run-1** is the pre-fix run: all three finals FAILED
  verification (F3/F1 anatomy, since fixed by construction). $0.4407
  bought zero passing Shorts; it is in the table because the spend is
  real, and it must never be cited as the cost of a passing run.
- **pipeline-run-graphics-holmes** (first graphics proof): short 2
  FAILED (F7) after its one repair. On a passing-only basis that run
  is $0.5785 / 2 = **$0.2893 per passing Short**.
- **pipeline-run-graphics-belew-2** (block-2 graphics, 2026-10-31)
  FAILED all three Shorts: F5 on all three (real source pauses —
  maximum speech gaps 1.16 / 0.94 / 0.80s against the 0.45s cap) and
  F9 on all three (the NORMALIZE_LOUDNESS repair landed integrated
  loudness in band, −14.58…−14.61 LUFS, but true peak overshot to
  +0.11 / +0.11 / +0.13 dBTP against the −1 dBTP cap); Spark's one
  repair went to loudness, so the F5 silence trim never ran. Its
  $0.6572 is real spend, disclosed in the table and counted in the §4
  ledger total; it is excluded from every §1 range, and no
  per-finished-minute figure is stated for it because it produced
  zero passing Shorts. Its receipts are kept in this repo as FAIL
  evidence (`CLAIMS-REGISTER.md` G5).
- **The McArthur runs** (2026-11-01, NASA ISS interview — CC-BY
  and a US federal work). Run 1 ($0.7858): shorts 1–2 passed
  F1–F8 on both drafts and failed F9 only, on the
  NORMALIZE_LOUDNESS true-peak fault (post-repair peaks +0.52 /
  +0.59 dBTP); short 3's window was the split-screen intro, SAM
  returned 155 of 659 frames, and the render truncated to a 5.2s
  stub (F8+F9 fail). After the repair rework (BUILDLOG
  2026-11-01), the re-run ($0.7781) produced **the first two
  F1–F9-passing Shorts** (both PASSED_AFTER_REPAIR; draft-2 F9
  −13.91 LUFS / −1.53 dBTP and −14.49 / −1.42); short 3 failed
  F8 only on the same split-screen window (its F9 passes on
  draft 2). The re-run's $0.9598 per finished minute is computed
  on all finished output including the 5.1s stub; on the two
  passing Shorts' own recorded per-Short costs ($0.4748 over
  43.52 s of output, shared transcription excluded) it is
  $0.6546 per finished minute. Neither McArthur run enters the
  §1 ranges (run 1: zero passing; re-run: not a complete passing
  run). Receipts for both runs are in this repo (register G6/G7).
- **pipeline-run-graphics-holmes-3 †**: shorts 1–2 finished and
  passed under the current F8 ruleset (short 1 PASSED_AFTER_REPAIR,
  short 2 PASSED_FIRST_DRAFT with the run's first live Muse Image
  placement); **short 3 was killed mid-verification by a VM restart
  on 2026-10-31 (~03:09 IST)** — draft rendered, no final, no
  receipt. The row's $0.4089 excludes short 3's $0.1437 of partial
  spend; the run's total ledgered spend was $0.5526.

## 3. Per-Short detail (from the run summaries)

Per-Short `cost_usd` exactly as recorded in each `run-summary.json`;
the shared full-source transcription ($0.0513 Holmes / $0.0610 Belew
/ $0.0518 McArthur)
is additional to these and is included in the §2 run totals. In every
summary, per-Short costs + shared cost reconcile to the recorded run
total to the cent (re-verified for this staging; the two McArthur
summaries re-verified 2026-11-01 evening).

| Run | Short 1 | Short 2 | Short 3 |
|---|---|---|---|
| pipeline-run-1 | $0.0579 (FAIL) | $0.1791 (FAIL) | $0.1524 (FAIL) |
| pipeline-run-2 | $0.1549 (repaired) | $0.1749 (repaired) | $0.1259 (first draft) |
| pipeline-run-3 | $0.1836 (repaired) | $0.2498 (repaired) | $0.1250 (first draft) |
| pipeline-run-belew-2 | $0.2435 (repaired) | $0.1894 (repaired) | $0.1568 (repaired) |
| pipeline-run-graphics-holmes | $0.1500 (repaired) | $0.1706 (FAIL) | $0.2066 (repaired) |
| pipeline-run-graphics-holmes-2 | $0.2257 (repaired) | $0.1984 (repaired) | $0.2469 (repaired) |
| pipeline-run-graphics-holmes-3 † (ledger) | $0.1961 (repaired) | $0.1615 (first draft) | — (killed, $0.1437) |
| pipeline-run-graphics-mcarthur-1 | $0.2191 (FAIL) | $0.2730 (FAIL) | $0.2419 (FAIL) |
| pipeline-run-graphics-mcarthur-2 | $0.2164 (repaired) | $0.2584 (repaired) | $0.2515 (FAIL) |

The first Belew attempt has no run-summary and no per-Short row here:
the only per-Short evidence for it is the surviving short-1 receipt
(§2), and no split of its $0.4042 ledger block is claimed. The Belew
graphics run's per-Short costs ($0.1970 / $0.2092 / $0.1900 plus the
shared transcription) are recorded in `CLAIMS-REGISTER.md`'s Belew
reconciliation.

## 4. Where the money went — ledger model split

Aggregation of the repo's `spend-ledger.jsonl`, **performed
2026-11-01 on the gate ledger (348 lines, $9.4691) and
re-performed the same evening after the two McArthur runs: all
401 lines, cumulative $11.0330**. The ledger covers all event spend — production runs
(including failed, partial and killed work), the Oct 29 spike,
benchmarks, proof packs, calibration calls and one-off probes — i.e.
the R&D total, not production cost alone.

| Model | Calls | Cost | Share |
|---|---|---|---|
| SAM 3.1 (segmentation) | 51 | $5.6488 | 51.2% |
| Muse Wonder (verifier judge) | 82 | $2.7182 | 24.6% |
| Muse Spark 1.3 (hooks, captions, planning, repairs) | 207 | $1.4082 | 12.8% |
| Muse Voice Transcribe | 48 | $1.1278 | 10.2% |
| Muse Image (graphics stills) | 13 | $0.1300 | 1.2% |
| **Total** | **401** | **$11.0330** | 100% |

(Shares are rounded to one decimal and may not sum to exactly
100.0%.) Muse Spark Open runs on partner credits (Baseten/Fireworks)
and is logged separately — see §8, item 4.

## 5. Price and gross margin

Locked price (2026-10-29): **30 credits per Director run** — $2.90 at
the Basic plan, $1.90 founding-member. Margin below is revenue minus
measured API cost of the run, divided by revenue (recomputed from the
run costs for this staging; values unchanged from the draft). It is
an **API-cost gross margin**: compute/infrastructure is not in any
ledger (§8, item 5) and no infra-inclusive margin is claimed.

| Complete passing run | Run cost | Basic $2.90: gross $ / margin | Founding $1.90: gross $ / margin |
|---|---|---|---|
| pipeline-run-2 | $0.5070 | $2.3930 / 82.5% | $1.3930 / 73.3% |
| pipeline-run-3 | $0.6097 | $2.2903 / 79.0% | $1.2903 / 67.9% |
| pipeline-run-graphics-holmes-6 | $0.6193 | $2.2807 / 78.6% | $1.2807 / 67.4% |
| pipeline-run-belew-2 | $0.6507 | $2.2493 / 77.6% | $1.2493 / 65.8% |
| pipeline-run-graphics-holmes-2 | $0.7223 | $2.1777 / 75.1% | $1.1777 / 62.0% |

## 6. Comparison line — cost vs price, labelled as such

OpusClip's published pricing is **$0.10 per minute of source video**
($15/mo for 150 minutes Starter; $29 for 300 minutes Pro; 1 credit =
1 minute) — as reported in the gap-research report of 2026-10-31
(`spatial-director-gap-research`, citing OpusClip's own site via
shortshort.io; not independently re-verified for this document).

Set side by side, with the kinds kept distinct:

- The Director's measured **cost** per minute of source, on the
  complete passing runs in §2: **$0.0296–$0.0422**.
- OpusClip's published **price** per minute of source: **$0.10**.
- The Director's own **price**: $2.90 per run ≈ **$0.143–$0.170 per
  source-minute** on these two sources.

A cost and a price are different kinds of figure, and the products
differ; the numbers above are presented for scale only, and no
favourable cost-versus-price comparison against the incumbent is
made or implied.

## 7. Method — how each figure was produced

- Run totals, per-Short costs, verdicts and `door_to_door_s`: read
  directly from each run's `run-summary.json` in the build records.
  Cross-check performed: in every run-summary carrying per-Short
  costs (eight at this staging, the two McArthur summaries
  included), Σ per-Short costs +
  shared cost = recorded run total, exactly (re-verified for this
  staging).
- Durations: `ffprobe` on each finished `final.mp4` (and on the two
  source files for source durations).
- Ledger split (§4): programmatic aggregation of all 401 lines of
  `spend-ledger.jsonl` by `model`, summing `cost_usd`, performed
  2026-11-01 (gate aggregation at 348 lines; re-performed the
  same evening at 401 lines after the McArthur runs).
- † Run (graphics-holmes-3): no `run-summary.json` exists. Costs were
  attributed from contiguous ledger blocks identified by the purpose
  tags (`pipeline-sam-shortN`, `pipeline-caption-punctuation-shortN`,
  `pipeline-voice-transcribe`, `verifier-f5-muted-hook`, …) in append
  order. Anchor, re-verified on the current ledger for this staging:
  the first 259 ledger lines sum to exactly $7.4218, the
  `ledger_total_usd` recorded at the end of graphics-holmes-2, so
  lines 260–288 are the graphics-holmes-3 block, and they sum to
  exactly the run's ledger delta, $0.5526.
- First Belew attempt: the contiguous block at ledger lines 143–156
  sums to exactly $0.4042 (cumulative ledger $4.4821 → $4.8863 across
  the block). The attribution is anchored by the surviving short-1
  receipt in this repo, whose three recorded verifier/repair calls
  (Wonder $0.0200, repair planner $0.0029, Wonder $0.0229) match
  ledger lines 149–151 to the cent.
- Why receipts alone understate cost: a receipt's `meta_calls` record
  only the verifier/repair calls (e.g. run-3 short 1: $0.0395); SAM,
  transcription and hook/planner calls are ledgered separately. All
  per-Short costs here therefore come from summaries/ledger, never
  from receipt subtotals.

## 8. Excluded — and why (used nowhere above)

1. **The first Belew attempt's $0.4042 ledger block** (lines
   143–156) is excluded from the per-Short and per-minute figures
   because those figures are computed on **complete passing runs
   only**. The block is identified, not untraceable: it is the first
   Belew attempt (§2, §7), anchored by the surviving short-1 receipt.
   It remains counted inside the §4 cumulative total, and no
   per-Short split of it is claimed.
2. **Wall-clock for the first Belew attempt and
   pipeline-run-graphics-holmes-3** — no `door_to_door_s` was ever
   recorded for either; not estimated.
3. **Per-run model-by-model dollar splits** (other than the global
   §4 split) — summaries record totals only and receipts record only
   verifier/repair calls; a per-run split would require
   re-attributing every ledger line and is not claimed.
4. **Muse Spark Open dollar cost** — `partner-usage-log.jsonl`
   records 23 calls as of 2026-11-01 (Baseten 20, Fireworks 3;
   96,536 prompt + 58,066 completion tokens) with **no USD field**;
   the calls bill against partner credits. Excluded from every dollar
   figure above.
5. **Infrastructure cost** (VM/Cloud Run time, storage, bandwidth) —
   recorded in no ledger. All margins in §5 are API-cost margins,
   labelled as such; no fully-loaded margin is claimed.
6. **Ledger completeness caveat** — the shipped clients do not ledger
   empty replies (one such call is visible precisely because it was
   hand-logged: the $0.0446 empty attempt inside the Belew block,
   ledger line 154). A billing-page reconciliation is scheduled for
   Nov 3; until it lands, ledger totals are the best available
   record, stated as such.
