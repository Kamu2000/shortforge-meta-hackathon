# ShortForge — Meta AI Developer Hackathon

Team **ShortForge**'s entry to the **Meta Global AI Developer Hackathon** (Meta × Cerebral Valley, October 26 – November 4, 2026).

- **Track:** Startup Track
- **Problem statement:** Build for Your World

## What this is

The **Verifiable Spatial Director**: give it a long talking-head video and it returns Shorts where the speaker stands *in front of* the hook text — and every Short ships with a measured PASS/FAIL receipt proving it was checked.

How it works: the source is transcribed and its strongest moments are scored and selected; SAM 3.1 segments the speaker per frame; a 3-layer composite (blurred background / hook text / speaker cutout in front) is rendered; a post-render verifier then checks the *actual rendered file* — deterministic geometry checks first, then one Muse Wonder pass over the opening seconds of the render — and on a FAIL it attempts one repair and re-verifies. The receipt records every check, its measurement against its threshold, the repair (if any), every Meta model call, and the run's cost.

Meta models claimed for this entry (each must have a logged call in the shipped pipeline before it is claimed — see `spend-ledger.jsonl`):

- **SAM 3.1** (Meta Model API) — per-frame speaker segmentation
- **Muse Wonder** (Meta Model API) — post-render verification of the real render
- **Muse Spark Open** (open weights, served via Baseten) — transcript moment scoring

This README is updated as the build lands during the event; BUILDLOG.md carries the dated record.

## Disclosure — pre-existing vs built during the event

- **Pre-existing:** ShortForge is an existing private product (an AI short-form video studio) operated as a hosted service. Its codebase is **not** in this repository, and no part of it was copied, imported, or adapted here.
- **Built during the event:** everything in this repository was written from scratch between October 26 and November 4, 2026, for this hackathon. The dated commit history and BUILDLOG.md are the record of that work.
- **Open-source resources used (labeled):** Python standard library and FFmpeg (invoked as a subprocess), plus the declared dependencies below. No other third-party code is vendored into this repo.

| Dependency | What it is | Used for |
|---|---|---|
| NumPy | Array library | Mask math, matte finishing, temporal smoothing, compositing |
| Pillow | Imaging library | Hook-text rendering and frame handling |
| meta-sam-parser | **Meta's official parser library** for the SAM segmentation protocol (PyPI) | Decoding SAM 3.1 mask payloads into rasters — the payload is a custom-base85 token over a context-coded arithmetic stream, and the library's lossless path thresholds coverage internally; our `mask_decode` module wraps it and never re-thresholds |

## Running and verifying

The event component in this repository runs standalone; it does not require the pre-existing ShortForge product. Setup and full demo instructions are added alongside the pipeline code as it lands during the event. Current state: see BUILDLOG.md.

What can be verified today (offline, no API key needed):

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

This exercises the cumulative spend ledger (including its hard cap); the SAM and Voice Transcribe clients' request construction and ledger gating with fake transports; mask decoding, canvas placement, and primary-speaker selection against a trimmed real SAM response fixture plus synthetic multi-object payloads; and the finish stack and compositor layer order on synthetic frames.

Spend accountability: `spend-ledger.jsonl` at the repository root is the cumulative, append-only record of every paid Meta API call made for this entry, with the cap enforced in code (`src/spatial_director/spend_ledger.py`).

## Credentials

This repository contains **no API keys or secrets**. The code reads credentials from environment variables / local configuration that is git-ignored and never committed.
