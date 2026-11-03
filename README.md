# AI Spatial Director — ShortForge's Meta AI Developer Hackathon entry

Team **ShortForge**'s entry to the **Meta Global AI Developer Hackathon** (Meta × Cerebral Valley, October 26 – November 4, 2026).

- **Track:** Startup Track
- **Problem statement:** Build for Your World

## What this is

The **AI Spatial Director**: give it one long talking-head video and it returns up to **3 finished vertical Shorts** (1080×1920) — and every Short ships with a measured **PASS/FAIL receipt** proving it was checked after rendering.

**The problem it solves.** Long single-speaker videos — podcasts, interviews, talks, lessons — are where the best short-form material lives, but turning one into a Short that holds attention is manual, skilled work: find the moment, cut it, frame the speaker, write a hook that lands in the first seconds, caption it without covering the face, add visual context for what is being described, keep the sound clean. Generic auto-clippers centre-crop and hope: hooks get buried, captions sit on faces, and quiet or broken renders ship silently because nothing ever checks the output file. The Director does the skilled parts as a pipeline — and then proves, per Short, that the rendered file was measured.

**How a run works, stage by stage:**

1. **Transcribe.** Muse Voice Transcribe transcribes the source (turn-level); a Muse Spark 1.3 pass restores caption punctuation under a word-sequence guard, and phrase timing is aligned for the captions.
2. **Pick the moments.** Muse Spark Open (Meta's open-weights model, served by Baseten) scores the transcript and sweeps for the strongest candidate moments.
3. **Cut the speaker out.** SAM 3.1 segments the speaker in every frame of the chosen window; the matte is finished in code — component and hole cleanup, then temporal smoothing across the mask sequence — so the cutout edge holds still instead of flickering.
4. **Direct the stage.** The Short is composed as a layered stage, not a crop: a treated backdrop; the hook text *behind* the speaker — the speaker occludes part of the lettering, and that occlusion is what makes the depth read (the planner targets an 8–20% glyph-occlusion band, hard cap 25%); the speaker cutout in front; captions always in front, never over the face.
5. **Add the graphics.** The graphics layer (next section) plans visual context for what is being said — generated stills, context cards, backdrop scenes — and places each one against the SAM masks under a strict placement law.
6. **Verify the render.** A **post-render verifier (checks F1–F9)** measures the *actual rendered file* — deterministic geometry and audio checks first, plus one Muse Wonder judgement of the muted opening seconds.
7. **Repair at most once.** On a FAIL, Muse Spark 1.3 picks **at most one repair** from a fixed menu, the repair runs on cached artifacts, and the render is re-verified.
8. **Receipt.** The receipt records every check with its measured value against its threshold, the repair (if any), every Meta model call with its request id, input hashes, and the run's cost. A FAIL that cannot be repaired in one pass is reported as a FAIL — the pipeline never manufactures a pass.

## The graphics layer — the Short shows what the speaker is talking about

The graphics layer is the Director's visual imagination, and one of the most distinctive parts of the build. A talking head alone is a weak visual; when the speaker describes something, the Short *shows* it — beside the speaker, behind the speaker, in the stage — at the moment it is said.

**What it makes.** A Muse Spark 1.3 planner reads the transcript and, for specific spoken moments, plans a graphic of one of four kinds:

- **Generated stills (Muse Image)** — a picture of the thing being described. When Megan McArthur recounts that the space station smelled of cooking meat — "like hamburger" — after a spacewalk, the run generates that visual and composites it into her Short.
- **Context cards (rendered in code)** — figures, names and short context as clean typeset cards in the brand palette.
- **Collages** — treatments assembled from stills.
- **Backdrop scenes (Muse Image)** — full-bleed generated scenes that take over the backdrop behind the speaker for a span.

Each span is anchored to a verbatim quote from the transcript, and procedural sound effects synthesized in code (NumPy) can be tied to a span's appearance. Runs enable the layer with `--graphics` (or `GRAPHICS_LAYER=1`); the graphics runs whose receipts ship in `receipts/` — Holmes, Belew, McArthur, and the geometry-bundle proof run — all ran with it on.

**The laws it runs under.** The graphics are attractive precisely because they are constrained:

- **Nothing invented.** Every graphic traces to a verbatim quote from the transcript. A numeric guard validates any value a card displays against what the speaker actually said — a card may show "$1M" only if the transcript supports it ("one million"); an invented value is rejected before rendering.
- **Text never goes through the image model.** Image models garble lettering, so words and numbers are rendered deterministically in code (Pillow), never generated. Image prompts are digit-free and carry a face/likeness blocklist — this layer never generates a face of a real person.
- **Placement is mask-aware.** Graphics are placed against the SAM speaker masks under a zero-tolerance law: the face region plus a 28 px margin is never touched, the caption block and caption zone are excluded, a graphic may cover at most 4% of the speaker, and beside-the-head placements rank first. A graphic that cannot be placed legally is dropped, with the reason recorded — never forced in.
- **Verified like everything else.** Check F8 pixel-measures each graphic's presence inside its span and absence outside it against its declared asset and box, checks motion-path fidelity and SFX audibility against the speech stem, and confirms backdrop containment. If the graphics are what broke a Short, DROP_GRAPHICS is on the repair menu.

Live placements are on record in `receipts/` — the Holmes graphics run (2026-10-31), the Belew graphics run's Ken Burns-treated stills (2026-10-31), and both McArthur runs' stills and full-bleed backdrop scenes (2026-11-01) — and the geometry-bundle proof run verifies a graphics-bearing receipt end-to-end (see *Keyless verification*). The layer degrades gracefully: context cards are rendered in code and need no image call.

## Model roster and model map

Every runtime model in this pipeline is a Meta model. The roster below matches `spend-ledger.jsonl` (Meta Model API spend) and `partner-usage-log.jsonl` (partner-credit usage) in this repository — a model is listed only because the shipped pipeline has logged calls to it.

| Stage | Model | Provider |
|---|---|---|
| Source transcription (turn-level) | Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Meta Model API |
| Caption punctuation pass | Muse Spark 1.3 (`muse-spark-1.3`) | Meta Model API |
| Candidate moment scoring / sweep | **Muse Spark Open** (open weights) | **Baseten** |
| Hook line writing | Muse Spark 1.3 | Meta Model API |
| Speaker segmentation, per frame | SAM 3.1 (`sam-3.1`) | Meta Model API |
| Graphics planning (which cards/stills, where) | Muse Spark 1.3 | Meta Model API |
| Graphics stills and backdrop scenes | Muse Image (`muse-image-1.0`) | Meta Model API |
| Post-render verification — F5 muted-hook judgement | Muse Wonder (`muse-wonder-1.0`) | Meta Model API |
| Repair selection (one repair, fixed menu) | Muse Spark 1.3 | Meta Model API |

### Muse Wonder is a core component

Wonder is not an add-on or a demo flourish: it is the semantic leg of the verifier. Check F5 fails a Short if its opening seconds cannot be understood with the sound off, and that judgement is made by Muse Wonder watching the actual rendered hook, muted. Every verified Short in `receipts/` carries a recorded Wonder call (model + request id) for this leg.

### Muse Spark Open — open-weights provenance

The candidate-sweep stage runs **Muse Spark Open** (Meta's open-weights model) served by **Baseten**. Meta staff confirmed during the event that `meta-models/muse-spark-open` on Baseten counts for the Open Weights prize, via the team's own account. Every Spark Open call is logged in `partner-usage-log.jsonl` (provider, model id as served, token counts, latency, request id); those calls bill against partner credits, so they carry no USD figure in `spend-ledger.jsonl`. Spark Open is one stage of the pipeline (moment scoring); the rest of the roster above runs on the Meta Model API. Fireworks AI also served Spark Open during our provider benchmark and is the proven fallback host (see `partner-usage-log.jsonl`, 2026-10-29 benchmark entries).

### Muse Image — disclosure

Event staff confirmed that Muse Image is **unavailable in India** and that this unavailability **will not count against the entry**. Both halves of the picture are stated plainly: Muse Image is unavailable in India per staff, **and** our runs did place Muse Image graphics live — a pipeline placement on 2026-10-31 (Holmes graphics run 3), Ken Burns-treated image stills in the Belew graphics run (2026-10-31), and image stills plus full-bleed backdrop scenes in both McArthur runs (2026-11-01) — with the calls in `spend-ledger.jsonl` ($0.01 per image) and the placements recorded on the receipts in `receipts/`. The graphics layer degrades gracefully: context cards are rendered in code and need no image call.

## The verifier — checks F1–F9

The verifier measures the rendered file, never the plan. Receipts for Shorts without graphics carry the F1–F7 + F9 shape; F8 runs when a Short carries graphics.

| Check | What it measures |
|---|---|
| F1 | Hook presence: hook-text opacity ≥ 0.90 at frame 0, and speech audio present (RMS above −32 dBFS) |
| F2 | Captions never overlap the face (0 px), and hook-glyph occlusion by the speaker ≤ 25% (the planner targets an 8–20% band) |
| F3 | Safe-zone containment: hook, captions and face inside the canvas safe zones; face centroid inside the central band |
| F4 | Speaker containment: ≥ 98% of frames contained, head-top margin ≥ 90 px |
| F5 | No dead air: maximum speech gap ≤ 0.45 s (measured on the pre-mix speech stem), **and** Muse Wonder must be able to summarise the muted opening hook |
| F6 | Hook content: a conflict, number, or question lands by 3 s |
| F7 | Early motion: zoom ≥ 6% or a card slide in the opening seconds, with motion consistency (IoU) |
| F8 | **Context-graphics integrity** (graphics Shorts only), as amended 2026-10-31: **no timing legs** — no hook-window ban and no minimum gap between graphics; graphics may start at 0 s and run back-to-back. Spatial legs are all zero-tolerance: face region **plus its 28 px margin** = 0 px overlap; caption block and caption zone = 0 px; hook glyph zone while a span overlaps the hook window = 0 px; pairwise collision between time-overlapping graphics = 0 pairs. Each graphic's presence in-span and absence out-of-span is pixel-measured against its declared asset and box. Block-2 legs: motion-path fidelity (mid-travel pixel match ≥ 0.75), SFX audibility against the speech stem, and backdrop containment |
| F9 | Loudness of every render: integrated loudness −14 LUFS ± 1, true peak ≤ −1.0 dBTP (ffmpeg loudnorm measurement) |

Repairs (at most one per Short, chosen by Muse Spark 1.3 as strict JSON from this fixed menu; an out-of-menu choice leaves the draft-1 FAIL standing): TRIM_SILENCE, REPOSITION_HOOK, REFIT_HOOK_INK, RESCALE_SPEAKER, REGENERATE_HOOK_TEXT, REFRAME_SPEAKER, ENGAGE_HOOK_SLIDE, DROP_GRAPHICS, NORMALIZE_LOUDNESS.

## Live figures (final, 2026-11-04)

- **Cumulative Meta Model API spend: $21.9867** across 850 ledgered calls (`spend-ledger.jsonl`; final merge 2026-11-04 — every run ledger from both build lines unioned and deduplicated by request id where the API returned one; the few calls logged without a request id, Muse Image stills among them, are counted at their logged multiplicity), against a **$145 hard cap enforced in code** (`src/spatial_director/spend_ledger.py`) — the entry was built entirely on the $150 event grant; no personal funds were spent. Model split: SAM 3.1 $11.7224 · Muse Wonder $4.7677 · Muse Spark 1.3 $3.2519 · Muse Voice Transcribe $2.1047 · Muse Image $0.1400.
- **Test suite: 626/626 passing** (`PYTHONPATH=src python3 -m unittest discover -s tests`; final merged suite, 2026-11-04).
- **Final production batch (round-6 code, 2026-11-03):** 10 receipted Shorts — **3 PASSED_AFTER_REPAIR, 7 FAIL** — plus the geometry-bundle proof run (PASSED_AFTER_REPAIR). All 11 receipts are in `receipts/`; the FAILs ship too.
- **Verifier proof pack: 44/44** labelled cases in exact agreement, plus the receipt re-verifier section **4/4** (`tools/proof_pack.py`).
- **Cost per finished Short: $0.169–$0.241** across complete passing runs, fully loaded on an API-cost-only basis (per-Short calls plus the run's shared full-source transcription; infrastructure is not in any ledger and no infra-inclusive figure is claimed). Full per-run table and method: `CLAIMS-REGISTER.md`.

## Setup and run (fresh clone)

Requirements: Python ≥ 3.11, FFmpeg on PATH.

```bash
pip install -e .          # numpy, Pillow, scipy, meta-sam-parser
cp .env.example .env      # then fill in your own keys — see below
```

Keys (never committed; `.env` is git-ignored):

- `SPATIAL_DIRECTOR_META_API_KEY` — a Meta Model API key from the developer console at dev.meta.ai. Used for SAM 3.1, Muse Spark 1.3, Muse Voice Transcribe, Muse Wonder and Muse Image.
- `SPATIAL_DIRECTOR_BASETEN_API_KEY` — a Baseten API key (app.baseten.co → Settings → API Keys). Used for Muse Spark Open (candidate sweep).
- `GRAPHICS_LAYER=1` (or pass `--graphics`) — enables the context-graphics layer.

Run the pipeline on a long talking-head video:

```bash
PYTHONPATH=src python3 -m spatial_director.pipeline <video.mp4> --out <run-dir> --max-shorts 3 [--graphics] [--ledger <path-to-spend-ledger.jsonl>]
```

Each Short lands in the run directory as `final.mp4` with its `receipt.json` beside it, plus a `run-summary.json` for the run.

### Keyless verification — no API key, no model calls

Anyone can check that a receipt tells the truth about its render, offline, with zero API calls. One scope limit, stated plainly: receipts record the artifact paths of the machine that issued them, and **run artifacts are not shipped in this repository — with one exception**. The complete run directory of the example Short below is bundled at **`evidence/holmes4-short1/`** (Holmes run 4, Short 1: both draft artifacts, captions, the SAM response, the window clip and the audio stems, 26 MB), so that one receipt verifies end-to-end from a fresh clone:

```bash
python3 tools/reverify.py receipts/receipt-pipeline-holmes4-short1.json --evidence-dir evidence/holmes4-short1
```

(The same command with no `--evidence-dir` also works for this receipt: when a stored path does not exist on your machine, the tool resolves the artifact by basename against the bundled `evidence/` tree and adopts that directory as the evidence directory. Every such fallback is printed as a `RESOLVED` line naming the path actually used — never silent — and a same-name substituted file is still caught by the hash check.)

For every other receipt in `receipts/`, the chain verifies when that Short's artifacts are supplied with `--evidence-dir` (or `--artifact-dir`); without the artifact files the tool reports `CHAIN MISSING: artifact file not found on disk` and fails, rather than passing a receipt it could not check.

What the tool delivers, exactly: it recomputes each draft's recorded input hashes (artifact, SAM response, caption list) and — in evidence mode — re-measures the deterministic check legs field-by-field against the receipt's recorded values. With only a receipt and its artifact (no evidence directory), it verifies the hash chain and the artifact-only legs (F9 loudness, F1 audio) and reports the rest NOT-REPRODUCIBLE. The Wonder legs are verified as *recorded* (model + request id), never re-judged. It prints its own boundary honestly: it is hash verification plus deterministic re-measurement, not a bit-identical re-render. Model-judgement legs and repairs it cannot reproduce are reported as NOT-REPRODUCIBLE with the reason named, never silently passed.

Geometry bundles (added 2026-11-03): the pipeline also writes a per-draft `geometry-bundle.json` beside each new receipt — the recorded stage geometry, layout spans, hook style and caption rect the verifier measured from, hash-chained to the receipt's own input hashes. The re-verifier rebuilds its measurement inputs from the bundle (a file bundle is trusted only after its `inputs_hash` verifies against the receipt), so the geometry legs re-measure from the run's own recorded declarations. A draft with no graphics and no backdrops reports F8 as NOT-APPLICABLE — excluded, not failed. The hash chain also accepts a final FAIL after an executed repair as a consistent chain: a repair that honestly did not fix the Short is a truthful chain, not a broken one.

Cost of verifying: evidence mode is heavy — it decodes the SAM response and rebuilds the measured frames, peaking above 3 GB RAM for the bundled Short and taking minutes per Short, so run it on an otherwise idle machine (on a small shared VM it will not complete alongside other heavy jobs).

The offline test suite needs no keys either:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

The full suite is a multi-minute run on a small VM (two of the larger modules alone took ~5 minutes in the Nov 2 verification pass); budget accordingly.

## Standalone statement — pre-existing vs built during the event

**This repository contains only work built during the event (October 26 – November 4, 2026).**

- **Pre-existing:** ShortForge is an existing private product (an AI short-form video studio) operated as a hosted service. Its codebase is **not** in this repository. Anything pre-existing was **excluded** from this repository, not merely disclosed: no pre-event code, scaffolding, or assets were copied in. The pre-kickoff cutoff applied throughout is **12:24 PM ET, October 26, 2026** — work written before it does not appear here; pre-kickoff scaffolding was rewritten from scratch during the event, not carried over.
- **Built during the event:** everything in this repository — pipeline, verifier, graphics layer, tools, tests, receipts — was written between October 26 and November 4, 2026, for this hackathon. The dated commit history and `BUILDLOG.md` are the record of that work.
- The event component runs standalone from this repository; it does not require the pre-existing ShortForge product.

## Demo footage — CC-BY attribution

The demonstration and test footage is Creative Commons Attribution (CC BY) licensed, verified in the source metadata on 2026-10-28, and credited here and in the demo credits:

- **Adrian Belew interview** — "Adrian Belew speaks up on Nine Inch Nails & Talking Heads", Ultimate Guitar ("On The Record"): https://www.youtube.com/watch?v=8H678YcwCl4 — CC BY.
- **Mark Holmes commentary** — "Cowboys and Ceedee only One Million away", Mark Holmes: https://www.youtube.com/watch?v=MQjeUxN6l4c — CC BY.
- **Megan McArthur ISS interview** — NASA astronaut Megan McArthur aboard the International Space Station, uploaded by "Kids Space Adventure": https://www.youtube.com/watch?v=yhwfDX-qzl4 — listed CC BY (verified by full metadata extraction 2026-11-01); the content is NASA-produced (end card nasa.gov/hubble, @NASAHubble) and therefore a US federal work in the public domain independently of the listing license. Credited to NASA / Megan McArthur wherever it appears.

Attribution is required wherever this footage appears (demo, receipts, README); the rights log is maintained with the build records. No unlicensed or unconsented footage is used anywhere in this entry, including for evaluation.

## Non-model libraries (helpers)

The pipeline calls Meta models for every model task at runtime. The following are non-model libraries and tools — no third-party model runs at runtime:

| Dependency | What it is | Used for |
|---|---|---|
| NumPy | Array library | Mask math, matte finishing, temporal smoothing, compositing, procedural SFX synthesis |
| SciPy (ndimage) | Scientific library | Connected-component labelling for matte cleanup (classical CV, not a model) |
| Pillow | Imaging library | Hook-text and context-card rendering, frame handling |
| meta-sam-parser | Licensed open-source parser library for the SAM segmentation protocol (PyPI) — a non-AI decoding library, not a model | Decoding SAM 3.1 mask payloads into rasters; the library thresholds coverage internally on its lossless path, and our `mask_decode` module wraps it and never re-thresholds |
| FFmpeg | Media tool (invoked as a subprocess) | Rendering, loudness measurement (loudnorm), audio/video processing |

## Claims and evidence

Every claim made for this entry — in the demo, this README, or the submission — is registered with its evidence in **`CLAIMS-REGISTER.md`**, including what is still pending and what is banned from use. `BUILDLOG.md` carries the dated build record. Receipts for the verified runs are in `receipts/`.
