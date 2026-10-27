# ShortForge — Meta AI Developer Hackathon

Team **ShortForge**'s entry to the **Meta Global AI Developer Hackathon** (Meta × Cerebral Valley, October 26 – November 4, 2026).

- **Track:** Startup Track
- **Problem statement:** Build for Your World

## What this is

An autonomous AI director that turns long-form video into publish-ready short-form clips, built on Meta's models — Muse Spark, Muse Wonder, SAM 3.1, Muse Voice Transcribe, and Muse Spark Open (served via Baseten). The director transcribes a source video, scores and selects the strongest moments, segments the speaker, composites graphics in depth behind the subject, and critiques the edit before rendering.

This README is updated as the build lands during the event; BUILDLOG.md carries the dated record.

## Disclosure — pre-existing vs built during the event

- **Pre-existing:** ShortForge is an existing private product (an AI short-form video studio) operated as a hosted service. Its codebase is **not** in this repository, and no part of it was copied, imported, or adapted here.
- **Built during the event:** everything in this repository was written from scratch between October 26 and November 4, 2026, for this hackathon. The dated commit history and BUILDLOG.md are the record of that work.
- **Open-source resources used (labeled):** Python standard library, NumPy, Pillow, and FFmpeg (invoked as a subprocess). No other third-party code is vendored into this repo.

## Running and verifying

Setup and demo instructions are added alongside the code as it lands during the event. Current state: see BUILDLOG.md.

## Credentials

This repository contains **no API keys or secrets**. The code reads credentials from environment variables / local configuration that is git-ignored and never committed.
