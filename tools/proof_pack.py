#!/usr/bin/env python3
"""Verifier proof pack: labelled fault-injection cases for F1-F8.

Standalone runner (NOT part of the unit suite). It builds a set of
labelled cases - for each deterministic check, fault injections
designed to fail ONLY that check, plus unmodified controls that
should pass everything - runs ``verifier.verify`` on each with a
FAKE Wonder judge (deterministic, no network, no API keys, zero paid
calls), and scores the verifier's per-check outcomes against the
labels: per-rule precision and recall.

Labels encode the SHIPPED design intent (the check definitions in
verifier.py's module docstring and layout_law's constants). Where a
fault cannot be isolated to one check because two checks share a
sub-condition by design (e.g. speech onset is a sub-condition of
both F1 and F5), the case is labelled with BOTH expected failures
and the coupling is documented on the case - the scoring takes the
label set as-is; nothing is tuned to make the numbers prettier.

A case where the verifier disagrees with its label is a FINDING.
Findings are recorded verbatim in the report, never tuned away.

Run:  python3 tools/proof_pack.py [--out PATH] [--only PREFIX]
Writes tools/proof_pack_report.json and prints a markdown summary.

Written fresh in-repo on 2026-10-29 for the Oct 30 benchmark matrix
(Matrix C) and claims-register PENDING #19.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spatial_director.caption_align import CaptionCue, EnergyProfile  # noqa: E402
from spatial_director.compositor import (  # noqa: E402
    _encode_video,
    graphics_layer_at,
)
from spatial_director.graphics_layer import GraphicSpec  # noqa: E402
from spatial_director.layout_law import initial_caption_rect  # noqa: E402
from spatial_director.verifier import HookSpec, VerifyInputs, verify  # noqa: E402

CANVAS = (1080, 1920)
FPS = 4.0
N = 16  # 4 seconds at 4 fps - mirrors tests/test_verifier.py fixtures
CHECKS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
RUN_DATE = "2026-10-30"


# ---------------------------------------------------------------- fixtures
# Self-contained mirrors of the tests/test_verifier.py fixture patterns
# (this runner must not import the test suite).


def make_video(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=black:s=320x240:r=10:d=1",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True)
    return path


def speaker_frames(n: int = N, *, top: int = 1000, x0: int = 240,
                   width: int = 600, shift: int = 3,
                   blank: set[int] | None = None,
                   top_overrides: dict[int, int] | None = None,
                   x0_overrides: dict[int, int] | None = None,
                   ) -> list[np.ndarray]:
    """Canvas-space speaker alphas: a solid block from ``top`` to the
    frame bottom, drifting right by ``shift`` px/frame (motion so F7
    passes unless the case removes it)."""
    blank = blank or set()
    top_overrides = top_overrides or {}
    x0_overrides = x0_overrides or {}
    frames = []
    for i in range(n):
        a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
        if i not in blank:
            t = top_overrides.get(i, top)
            x = x0_overrides.get(i, x0 + shift * i)
            a[t:CANVAS[1], x:x + width] = 1.0
        frames.append(a)
    return frames


def profile(spans, total_s: float = 4.0, hop: float = 0.02,
            level: float = -20.0, floor: float = -80.0) -> EnergyProfile:
    n = int(total_s / hop)
    times = np.arange(n) * hop
    db = np.full(n, floor)
    for a, b in spans:
        db[(times >= a) & (times < b)] = level
    return EnergyProfile(times=times, db=db, hop_s=hop)


class FakeJudgment:
    def __init__(self, can: bool = True):
        self.can_summarize = can
        self.summary = ("A musician explains his band never rehearses."
                        if can else "")
        self.cost_usd = 0.0  # fake judge: nothing is billed, ever
        self.request_id = "fake-judge-no-network"


class FakeWonder:
    """Deterministic stand-in for the Wonder judge. Never calls out."""

    def __init__(self, can: bool = True):
        self.can = can
        self.calls: list[str] = []

    def judge_muted_hook(self, path, *, purpose: str = "", ts: str = ""):
        self.calls.append(str(path))
        return FakeJudgment(self.can)


GOOD_HOOK = HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.30,
                     layout_rendered=True)
GOOD_CUES = [CaptionCue("The band never rehearses.", 0.0, 2.0),
             CaptionCue("It is just perfect.", 2.0, 4.0)]
PLAIN_HOOK = HookSpec("THE BAND PLAYS SONGS", (0.0, 3.0), 0.30,
                      layout_rendered=True)
PLAIN_CUES = [CaptionCue("The band plays songs together.", 0.0, 2.0),
              CaptionCue("They sound really good tonight.", 2.0, 4.0)]


def base_inputs(video: Path, **overrides) -> VerifyInputs:
    """The all-pass control input set (the suite's good fixture)."""
    inputs = VerifyInputs(
        video_path=video,
        fps=FPS,
        canvas_alphas=speaker_frames(),
        audio=profile([(0.0, 4.0)]),
        hook=GOOD_HOOK,
        cues=GOOD_CUES,
        caption_rect=initial_caption_rect(),
    )
    for key, value in overrides.items():
        setattr(inputs, key, value)
    return inputs


# ---------------------------------------------------------------- scoring


def score_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-check precision/recall from labelled vs actual fail sets.

    Each case dict carries ``id``, ``expected_failed`` and
    ``actual_failed`` (iterables of check ids). A check outcome is a
    "positive" when the check FAILS (the verifier flags a fault):
    TP = labelled fail and verifier failed it, FP = verifier failed
    a check the label says should pass, FN = labelled fail the
    verifier passed. Precision/recall are None when their
    denominator is 0 (no such events - not silently 1.0 or 0.0).
    """
    per_check: dict[str, dict[str, Any]] = {}
    for cid in CHECKS:
        tp = fp = fn = tn = 0
        for case in cases:
            expected = cid in set(case["expected_failed"])
            actual = cid in set(case["actual_failed"])
            if expected and actual:
                tp += 1
            elif actual:
                fp += 1
            elif expected:
                fn += 1
            else:
                tn += 1
        per_check[cid] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "recall": (tp / (tp + fn)) if (tp + fn) else None,
        }
    findings = []
    exact = 0
    for case in cases:
        exp, act = set(case["expected_failed"]), set(case["actual_failed"])
        if exp == act:
            exact += 1
        else:
            findings.append({
                "case": case["id"],
                "expected_failed": sorted(exp),
                "actual_failed": sorted(act),
                "false_positive_checks": sorted(act - exp),
                "false_negative_checks": sorted(exp - act),
            })
    return {
        "per_check": per_check,
        "agreement": {"exact_set_matches": exact, "total_cases": len(cases)},
        "findings": findings,
    }


# ---------------------------------------------------------------- cases
# Each case: id, target check(s), what was injected, the label
# (expected_failed), and a coupling note where isolation is imperfect
# BY DESIGN. Builders receive the shared dummy video path (verify()
# reads only structured inputs + extracts a muted segment from the
# video file; the file's own content is never measured).

Case = dict[str, Any]


def _case(cid: str, target: str, description: str,
          expected: set[str], build: Callable[[Path], tuple],
          coupling: str = "") -> Case:
    return {"id": cid, "target": target, "description": description,
            "expected_failed": sorted(expected), "coupling": coupling,
            "build": build}


def _std(video: Path, wonder_can: bool = True, **overrides):
    return base_inputs(video, **overrides), FakeWonder(can=wonder_can)


# ---------------------------------------------------------------- F8 fixtures

F8_N = 40  # 10 seconds at 4 fps - room for the spacing-rule cases
F8_CUES = [CaptionCue("The band never rehearses.", 0.0, 3.5),
           CaptionCue("It is just perfect.", 3.5, 7.0),
           CaptionCue("And it keeps going on.", 7.0, 10.0)]


def _f8_spec(box, start, end, color=(200, 60, 120)):
    layer = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    layer[y0:y1, x0:x1, 0] = color[0]
    layer[y0:y1, x0:x1, 1] = color[1]
    layer[y0:y1, x0:x1, 2] = color[2]
    layer[y0:y1, x0:x1, 3] = 255
    return GraphicSpec(
        kind="card", start_s=start, end_s=end, box=box,
        layer_rgba=layer, headline="PROOF CARD",
        source_quote="the band never rehearses", motion="static")


def _f8(shared_video: Path, name: str, specs, windows=None):
    """An F8 case: a custom-rendered video (black + the graphic
    plane) plus the 40-frame good fixture declared with ``specs``.
    ``windows`` overrides where the render actually SHOWS each
    graphic (default: its declared span)."""
    windows = windows or [None] * len(specs)
    entries = [
        (spec.layer_at,
         spec.start_s if w is None else w[0],
         spec.end_s if w is None else w[1])
        for spec, w in zip(specs, windows)
    ]
    video = shared_video.with_name(name)

    def gen():
        for i in range(F8_N):
            t = i / FPS
            frame = np.zeros((CANVAS[1], CANVAS[0], 3), dtype=np.uint8)
            plane = graphics_layer_at(entries, t, CANVAS)
            if plane is not None:
                a = plane[..., 3:4].astype(np.float64) / 255.0
                frame = np.clip(
                    plane[..., :3].astype(np.float64) * a,
                    0, 255).astype(np.uint8)
            yield frame

    _encode_video(gen(), video, fps=FPS, canvas=CANVAS)
    inputs = VerifyInputs(
        video_path=video, fps=FPS,
        canvas_alphas=speaker_frames(F8_N),
        audio=profile([(0.0, 10.0)], total_s=10.0),
        hook=GOOD_HOOK, cues=F8_CUES,
        caption_rect=initial_caption_rect(), graphics=specs)
    return inputs, FakeWonder()


CASES: list[Case] = [
    # ---- controls: unmodified, must pass everything ----------------
    _case("control-baseline", "control",
          "The suite's all-pass fixture, unmodified.",
          set(), lambda v: _std(v)),
    _case("control-question-hook", "control",
          "Question hook ('WHY DID HE QUIT?') and faster speaker drift "
          "(6px/frame). F6 fires via the question pattern.",
          set(), lambda v: _std(
              v,
              canvas_alphas=speaker_frames(shift=6),
              hook=HookSpec("WHY DID HE QUIT?", (0.0, 3.0), 0.30,
                            layout_rendered=True))),
    _case("control-slide-only-motion", "control",
          "Frozen speaker (no drift, no zoom) but the hook card slides "
          "at 200px/s: F7 must pass on the slide signal alone.",
          set(), lambda v: _std(
              v, canvas_alphas=speaker_frames(shift=0),
              hook_slide_px_s=200.0)),
    _case("control-numeric-hook-breath-gap", "control",
          "Numeric hook ('3 SONGS HE NEVER SANG'), speaker top=900, "
          "hook lower (y=0.35), and a 0.2s breath gap in the audio at "
          "0.8-1.0s (inside the 0.45s F5 allowance).",
          set(), lambda v: _std(
              v,
              canvas_alphas=speaker_frames(top=900),
              audio=profile([(0.0, 0.8), (1.0, 4.0)]),
              hook=HookSpec("3 SONGS HE NEVER SANG", (0.0, 3.0), 0.35,
                            layout_rendered=True))),

    # ---- F1: hook in frame 1 ---------------------------------------
    _case("f1-late-hook", "F1",
          "Hook window starts at 1.5s instead of frame 0: not present "
          "at frame 0 and not present by 1s. (F6 still sees the hook "
          "text - its window opens at 1.5s < 3s - so F6 passes.)",
          {"F1"}, lambda v: _std(
              v, hook=HookSpec("HE NEVER REHEARSES", (1.5, 4.0), 0.30,
                               layout_rendered=True))),
    _case("f1-short-hook-text", "F1",
          "Hook text is a single content word ('STOP', 1 < 3 minimum). "
          "Captions still carry the F6 conflict word, isolating F1.",
          {"F1"}, lambda v: _std(
              v, hook=HookSpec("STOP", (0.0, 3.0), 0.30,
                               layout_rendered=True))),
    _case("f1-quiet-opening", "F1",
          "Speech present from 0s but uniformly quiet (-38 dBFS): the "
          "0-2s RMS sits below the -32 dBFS floor while onset, gaps "
          "and coverage (F5's sub-conditions) all stay legal - F5 "
          "has no loudness sub-check by design.",
          {"F1"}, lambda v: _std(
              v, audio=profile([(0.0, 4.0)], level=-38.0))),

    # ---- F2: face/mouth obstruction --------------------------------
    _case("f2-hook-behind-speaker", "F2",
          "Speaker raised (top=420) so the layout-path hook at "
          "y=0.30 sits behind the speaker: glyph occlusion far above "
          "the 25% cap. Drift kept at 3px/frame so F7 still passes.",
          {"F2"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(top=420),
              hook=HookSpec("HE NEVER REHEARSES", (0.0, 3.0),
                            0.30, layout_rendered=True)),
          coupling="First construction used the compositor-path "
                   "hook ('THE BAND NEVER REHEARSES', "
                   "layout_rendered=False): its ink is budgeted at "
                   "0.92 x canvas and centred on the canvas, so it "
                   "genuinely exits the safe zone (measured 14px "
                   "left, 52px right) and F3 CORRECTLY failed "
                   "alongside F2 - a fixture error, not a verifier "
                   "error (25/26 agreement on that run; disagreement "
                   "recorded in BUILDLOG). Rebuilt on the layout "
                   "path, whose ink is safe-width-clamped by "
                   "construction, to isolate F2."),
    _case("f2-caption-over-face", "F2",
          "Caption block moved up onto the face (rect y 1000-1140, "
          "across the head band): caption-face overlap > 0 while the "
          "rect stays inside F3's lower-zone bounds (top >= 288, "
          "bottom <= 1600), isolating the failure to F2.",
          {"F2"}, lambda v: _std(
              v, caption_rect=(60, 1000, 980, 1140))),
    _case("f2-partial-occlusion", "F2",
          "Layout hook centred at y=0.52 so the speaker (top=1000) "
          "covers roughly the lower half of the glyphs: occlusion "
          "above the 25% cap but far from total - the mid-range the "
          "cap exists for.",
          {"F2"}, lambda v: _std(
              v, hook=HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.52,
                               layout_rendered=True))),

    # ---- F3: safe zone ----------------------------------------------
    _case("f3-face-centroid-low", "F3",
          "Speaker sunk (top=1500): the median face centroid lands "
          "below the central band's 1440px bottom edge. Captions "
          "carry no block rect (rect=None) so the low face cannot "
          "trip F2's caption-overlap sub-check - documented input "
          "state, rect is optional in VerifyInputs.",
          {"F3"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(top=1500),
              caption_rect=None)),
    _case("f3-hook-above-band", "F3",
          "Hook centred at y=0.10: the glyph bbox top exits the "
          "central band's 288px top edge.",
          {"F3"}, lambda v: _std(
              v, hook=HookSpec("HE NEVER REHEARSES", (0.0, 3.0), 0.10,
                               layout_rendered=True))),
    _case("f3-caption-below-zone", "F3",
          "Caption block pushed down (rect y 1560-1760): its bottom "
          "edge passes the 1600px lower-zone limit by 160px. The "
          "face (rows 1000-1294) never enters the block, so F2 "
          "stays clean.",
          {"F3"}, lambda v: _std(
              v, caption_rect=(60, 1560, 980, 1760))),

    # ---- F4: subject containment ------------------------------------
    _case("f4-head-margin-one-frame", "F4",
          "One frame (index 7) has the speaker at top=40: the "
          "worst-frame head-top margin is 40px < 90px while every "
          "frame stays contained and the MEDIAN centroid (F3) is "
          "unmoved. That frame's occlusion contribution is 1/12 of "
          "the hook window, under F2's cap.",
          {"F4"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(
                  top_overrides={7: 40}))),
    _case("f4-speaker-clipped-right", "F4",
          "Speaker block spans x 480-1080+ (width 600 from x0=480): "
          "the face touches the right canvas edge on every frame, "
          "so the contained fraction is 0. Head-top margin (1000) "
          "and median centroid (~x 790, in band) stay legal.",
          {"F4"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(x0=480))),
    _case("f4-speaker-absent-half", "F4",
          "Speaker absent for the second half (frames 8-15 blank): "
          "contained frames are 8 of 16 total (the denominator is "
          "all frames by design), so the contained fraction is "
          "0.50 < 0.98 while present-frame margins stay legal.",
          {"F4"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(
                  blank=set(range(8, 16))))),

    # ---- F5: dead air / mute comprehension ---------------------------
    _case("f5-dead-air-gap", "F5",
          "A 1.5s silence gap inside the speech span (speech "
          "0-1s and 2.5-4s): max gap 1.5s > 0.45s. The 0-2s RMS "
          "stays above F1's floor (half the window is full-level "
          "speech), isolating the failure to F5.",
          {"F5"}, lambda v: _std(
              v, audio=profile([(0.0, 1.0), (2.5, 4.0)]))),
    _case("f5-judge-cannot-summarize", "F5",
          "Fake Wonder judge returns can_summarize=False: the muted "
          "hook cannot be comprehended, F5's judge sub-check fails. "
          "Every deterministic sub-condition stays legal.",
          {"F5"}, lambda v: _std(v, wonder_can=False)),
    _case("f5-caption-coverage-low", "F5",
          "Only one 1s caption cue over a 4s speech span: coverage "
          "0.25 < 0.50 minimum. The cue carries the F6 conflict "
          "word and sits clear of the face, isolating F5.",
          {"F5"}, lambda v: _std(
              v, cues=[CaptionCue("The band never rehearses.",
                                  0.0, 1.0)])),
    _case("f5-late-onset-coupled", "F5",
          "Speech does not start until 2.5s (deadline 2.0s).",
          {"F1", "F5"}, lambda v: _std(
              v, audio=profile([(2.5, 4.0)])),
          coupling="Speech onset is a sub-condition of BOTH F1 and "
                   "F5 in the shipped design, so a late onset cannot "
                   "fail one without the other (F1 also loses its "
                   "0-2s RMS sub-check: the window is silent). The "
                   "label names both; isolation is impossible by "
                   "design, not by fixture weakness."),

    # ---- F6: open loop by 3s -----------------------------------------
    _case("f6-no-open-loop", "F6",
          "Plain hook and plain early captions: no conflict word, "
          "number, or question anywhere in the first 3s.",
          {"F6"}, lambda v: _std(v, hook=PLAIN_HOOK, cues=PLAIN_CUES)),
    _case("f6-loop-after-3s", "F6",
          "The conflict word exists but only in a caption starting "
          "at exactly 3.0s ('It never gets old.', 3.0-4.0s) - the "
          "open loop arrives too late. Caption coverage stays 0.75 "
          "so F5 is unaffected.",
          {"F6"}, lambda v: _std(
              v, hook=PLAIN_HOOK,
              cues=[PLAIN_CUES[0],
                    CaptionCue("It never gets old.", 3.0, 4.0)])),
    _case("f6-lexicon-near-miss", "F6",
          "Hook 'THE BAND QUITS AT MIDNIGHT': 'quits' reads as a "
          "conflict word to a human but is NOT in the shipped fixed "
          "lexicon ('quit' is). Labelled FAIL per the shipped "
          "semantics - a boundary case documenting lexicon "
          "exactness, recorded as a design observation.",
          {"F6"}, lambda v: _std(
              v, hook=HookSpec("THE BAND QUITS AT MIDNIGHT",
                               (0.0, 3.0), 0.30, layout_rendered=True),
              cues=PLAIN_CUES)),

    # ---- F7: static opening ------------------------------------------
    _case("f7-frozen-speaker", "F7",
          "Speaker perfectly still (no drift): mean mask IoU over "
          "0-3s is 1.0, no zoom, no slide - a frozen opening.",
          {"F7"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(shift=0))),
    _case("f7-absent-opening-coupled", "F7",
          "No speaker at all in the first 3s (frames 0-11 blank, "
          "speaker enters at frame 12).",
          {"F4", "F7"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(
                  blank=set(range(0, 12)))),
          coupling="An absent speaker is simultaneously not "
                   "contained (F4's contained fraction counts absent "
                   "frames in its denominator) and a frozen opening "
                   "(blank-vs-blank IoU is 1.0). The two failures "
                   "are the same physical fault seen by two checks; "
                   "isolation is impossible by design."),
    _case("f7-subthreshold-motion", "F7",
          "The speaker moves exactly once - a single 1px shift at "
          "frame 6. Mean IoU over the 11 frame pairs in 0-3s is "
          "~0.9997, at/above the 0.999 frozen ceiling: real motion, "
          "below the designed detection floor.",
          {"F7"}, lambda v: _std(
              v, canvas_alphas=speaker_frames(
                  shift=0, x0_overrides={i: 241
                                         for i in range(6, 16)}))),

    # ---- F8: context graphics (2026-10-30) ---------------------------
    # F8 measures the render's own pixels, so these cases render
    # CUSTOM videos (the shared 320x240 dummy cannot carry a
    # graphic): black frames + the graphic plane, in a sibling file
    # of the same tempdir, on the 10s/40-frame good fixture.
    _case("control-graphics", "control",
          "One graphic placed legally (clear of face, captions and "
          "the hook window), present exactly inside its span: F8 "
          "passes and F1-F7 are untouched.",
          set(), lambda v: _f8(
              v, "f8-control.mp4",
              [_f8_spec((700, 300, 1000, 560), 3.5, 5.5)])),
    _case("control-graphics-spaced-pair", "control",
          "Two graphics 4.0s of clean screen apart (the kept "
          "spacing ruling, exactly at its floor): both pass.",
          set(), lambda v: _f8(
              v, "f8-control-spaced.mp4",
              [_f8_spec((700, 300, 1000, 560), 3.25, 4.75),
               _f8_spec((60, 300, 360, 560), 8.75, 10.0,
                        color=(60, 120, 200))])),
    _case("f8-graphic-over-face", "F8",
          "Graphic box overlapping the face band (rows 1050-1350 "
          "across the head): F8's face-overlap sub-check fails; "
          "placement quantities only, so F1-F7 stay clean.",
          {"F8"}, lambda v: _f8(
              v, "f8-over-face.mp4",
              [_f8_spec((300, 1050, 700, 1350), 3.5, 5.5)])),
    _case("f8-graphic-over-captions", "F8",
          "Graphic box overlapping the caption block AND the lower "
          "caption zone (rows 1500-1700): F8's caption-overlap "
          "sub-check fails.",
          {"F8"}, lambda v: _f8(
              v, "f8-over-captions.mp4",
              [_f8_spec((200, 1500, 800, 1700), 3.5, 5.5)])),
    _case("f8-graphic-outside-span", "F8",
          "Graphic declared at 3.5-5.5s but the render shows it at "
          "0.5-2.0s: in-span presence match ~0 - F8's timing "
          "sub-check fails (pixel-measured, not spec-trusted).",
          {"F8"}, lambda v: _f8(
              v, "f8-outside-span.mp4",
              [_f8_spec((700, 300, 1000, 560), 3.5, 5.5)],
              windows=[(0.5, 2.0)])),
    _case("f8-graphic-in-hook-window", "F8",
          "Graphic declared at 2.0-4.0s: it intrudes on the first "
          "3.0s hook window, which the kept ruling forbids - F8 "
          "fails on the declared span alone.",
          {"F8"}, lambda v: _f8(
              v, "f8-hook-window.mp4",
              [_f8_spec((700, 300, 1000, 560), 2.0, 4.0)])),
    _case("f8-spacing-violation", "F8",
          "Two individually-legal graphics only 1.25s of clean "
          "screen apart: the kept >=4s spacing ruling fails them "
          "as a pair - F8's spacing sub-check.",
          {"F8"}, lambda v: _f8(
              v, "f8-spacing.mp4",
              [_f8_spec((700, 300, 1000, 560), 3.25, 4.75),
               _f8_spec((60, 300, 360, 560), 6.0, 7.5,
                        color=(60, 120, 200))])),
]


# ---------------------------------------------------------------- runner


def run_cases(only: str = "") -> list[dict[str, Any]]:
    """Run every labelled case through verify() with the fake judge."""
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        video = make_video(Path(td) / "render.mp4")
        for case in CASES:
            if only and not case["id"].startswith(only):
                continue
            inputs, wonder = case["build"](video)
            entry: dict[str, Any] = {
                "id": case["id"],
                "target": case["target"],
                "description": case["description"],
                "coupling": case["coupling"],
                "expected_failed": case["expected_failed"],
                "wonder_can_summarize": wonder.can,
            }
            try:
                receipt = verify(inputs, wonder=wonder, ts=RUN_DATE)
                entry["actual_failed"] = receipt.failed
                entry["verdict"] = receipt.verdict
                entry["checks"] = {
                    c.check_id: {
                        "passed": c.passed,
                        "measured": c.measured,
                    }
                    for c in receipt.checks
                }
            except Exception as exc:  # a crash is a finding, verbatim
                entry["actual_failed"] = None
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
            status = ("ERROR" if "error" in entry else
                      "agree" if entry["actual_failed"]
                      == entry["expected_failed"] else "DISAGREE")
            print(f"[{status}] {entry['id']}: expected "
                  f"{entry['expected_failed']} actual "
                  f"{entry.get('actual_failed')}", file=sys.stderr)
    return results


def markdown_summary(report: dict[str, Any]) -> str:
    s = report["scores"]
    lines = [
        f"# Verifier proof pack - {report['generated']}",
        "",
        f"{report['case_count']} labelled cases "
        f"({report['fault_cases']} fault injections, "
        f"{report['control_cases']} controls), run offline against "
        "verifier.verify with a fake Wonder judge. Zero paid calls.",
        "",
        f"Exact label/verdict set agreement: "
        f"{s['agreement']['exact_set_matches']}/"
        f"{s['agreement']['total_cases']} cases.",
        "",
        "| Check | TP | FP | FN | TN | Precision | Recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid in CHECKS:
        m = s["per_check"][cid]
        prec = "-" if m["precision"] is None else f"{m['precision']:.3f}"
        rec = "-" if m["recall"] is None else f"{m['recall']:.3f}"
        lines.append(f"| {cid} | {m['tp']} | {m['fp']} | {m['fn']} "
                     f"| {m['tn']} | {prec} | {rec} |")
    lines += ["", "## Findings (verifier vs label disagreements)", ""]
    if s["findings"]:
        for f in s["findings"]:
            lines.append(
                f"- **{f['case']}**: expected failed "
                f"{f['expected_failed']}, verifier failed "
                f"{f['actual_failed']} "
                f"(false positives: {f['false_positive_checks']}, "
                f"false negatives: {f['false_negative_checks']})")
    else:
        lines.append("- None: the verifier agreed with every label.")
    for entry in report["cases"]:
        if "error" in entry:
            lines.append(f"- **{entry['id']}**: verify() raised "
                         f"{entry['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name(
                            "proof_pack_report.json"))
    parser.add_argument("--only", default="",
                        help="run only case ids with this prefix")
    args = parser.parse_args(argv)

    results = run_cases(only=args.only)
    scored = [r for r in results if r.get("actual_failed") is not None]
    report = {
        "generated": RUN_DATE,
        "runner": "tools/proof_pack.py",
        "judge": "FakeWonder (deterministic, no network, zero paid calls)",
        "case_count": len(results),
        "fault_cases": sum(1 for r in results if r["expected_failed"]),
        "control_cases": sum(1 for r in results
                             if not r["expected_failed"]),
        "scores": score_cases(scored),
        "cases": results,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(markdown_summary(report))
    print(f"\nReport written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
