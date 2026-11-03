#!/usr/bin/env python3
"""Standalone receipt re-verifier + hash chain for the Spatial Director.

Anyone — a judge, a creator, us on a fresh clone — can run this
against a receipt + its video artifact, with NO API keys and NO
model calls, to check that the receipt tells the truth about the
render. It does two things, and only those two things:

1. **Hash-chain verification.** Every draft's ``inputs_sha256``
   block is recomputed from the files on disk: the artifact file's
   own hash (``inputs_sha256.video`` is the hash of the artifact
   at verify time), the SAM response file, and the caption list.
   The drafts are also checked structurally: sequential numbering,
   the top-level artifact is the last draft's artifact, and the
   repair record sits between the failing and passing drafts it
   claims to connect. A computed chain hash (each draft's artifact
   hash folded into the previous) is printed so a published value
   can be compared out-of-band — receipts record the per-draft
   hashes, not a chain head, so the chain itself is a fingerprint
   this tool computes, not a value it compares.
2. **Deterministic re-measurement.** The verifier's own
   ``verify()`` is re-run on inputs rebuilt from the receipt +
   the run's evidence files (window clip, SAM response, captions,
   trimmed stem where one exists), and every recorded measured
   value is compared against the recomputed value within a stated
   tolerance. Per field: MATCH / MISMATCH / NOT-REPRODUCIBLE
   (with the reason — never a guess).

HONEST BOUNDARY (printed by every run): the Wonder judgement legs
(F5's muted-hook summary) are model calls and CANNOT be re-run
keylessly. The tool verifies that the receipt RECORDS them (a
muse-wonder-1.0 call with a request id in the draft's meta_calls)
and that the deterministic legs agree; it does not re-judge.
The promise is hash-verification + deterministic re-measurement,
NEVER bit-identical re-rendering: Voice Transcribe and SAM are
not bit-deterministic, so a re-render would differ even when the
receipt is honest. What is re-measured here is the artifact file
itself, which does not change.

Exit behaviour: exit 0 only if the chain verifies AND every
reproducible leg matches. NOT-REPRODUCIBLE legs do not block
exit 0 (they are declared, not hidden); any MISMATCH, any broken
chain link, or a Wonder leg claimed without a recorded model
call exits non-zero with a readable report.

Evidence model: the receipt alone does not carry everything the
verifier measured from — hook layout, SAM masks, the speech stem
live in the run directory. ``--evidence-dir`` points at the
Short's directory (the one holding captions.json, sam-response.json,
window.mp4, trimmed-audio.wav); run-summary.json is picked up from
that directory or its parent when present. With only a receipt +
artifact and no evidence, the tool still verifies the hash chain
and the artifact-only legs (F9 loudness, F1 audio) and reports
the rest NOT-REPRODUCIBLE. Receipts from the F1–F7 era (no F8/F9
checks) verify the same way; checks the receipt predates are
reported as ADVISORY (computed, not recorded), never as failures.

Stored artifact paths are absolute paths from the machine that
issued the receipt. When one does not exist on this machine, the
artifact is resolved by BASENAME: the ``--artifact-dir``, then
the ``--evidence-dir``, then the repository's bundled ``evidence/``
tree (the one shipped evidence set — see the README's keyless
verification section). Every such fallback is REPORTED in the
output with the path actually used, never silent; a swapped file
under the same name is still caught by the hash check. When no
``--evidence-dir`` is given and a bundled directory that resolved
an artifact also holds the full evidence file set (captions.json,
sam-response.json, window.mp4), it is adopted as the evidence
directory — also reported — so the shipped set re-measures in
full from a fresh clone with no flags at all.

Geometry bundles (2026-11-03 closure): when the evidence
directory holds a ``geometry-bundle.json`` written by the
pipeline at verify time — or the receipt itself records the
geometry declarations a bundle is made of (stage geometry,
layout spans, hook style, speaker-forward spans, the summary's
hook band and caption rect) — the verifier's inputs are built
FROM THE BUNDLE: the recorded stage zoom, effect pad, emphasis
spans and layout placements are replayed with the pipeline's own
functions, and the hook / caption rect / layout spans are the
recorded ones. File bundles are trusted only after their
``inputs_hash`` verifies against the receipt draft's recorded
input hashes. A draft with no graphics and no backdrops reports
F8 as NOT-APPLICABLE (excluded from findings) instead of
NOT-REPRODUCIBLE.

No threshold in this file is a copy of a verifier threshold: the
comparison reuses ``verifier.verify`` itself, so thresholds can
never drift between the gate and this tool. No verifier behaviour
is changed by this tool — it only reads receipts and artifacts.

Run:  python3 tools/reverify.py RECEIPT.json [--artifact FILE]
       [--evidence-dir DIR] [--artifact-dir DIR] [--json]
Zero paid/live API calls. Written 2026-10-31 (shortlist item 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spatial_director.caption_align import (  # noqa: E402
    CaptionCue,
    active_segments,
    load_energy,
)
from spatial_director.compositor import video_fps  # noqa: E402
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicSpan,
    GraphicSpec,
    _paste_layer,
    _paste_layer_clipped,
    path_box_at,
    prepare_card,
)
from spatial_director import geometry_bundle as gb  # noqa: E402
from spatial_director.layout_law import (  # noqa: E402
    apply_speaker_emphasis,
    apply_speaker_reframe,
    apply_stage_zoom,
    plan_captions,
    plan_hook_effect,
    plan_speaker_emphasis,
    plan_speaker_reframe,
    to_canvas_alpha,
)
from spatial_director.mask_decode import (  # noqa: E402
    finished_alpha,
    parse_sam_response,
)
from spatial_director.rescue_loop import (  # noqa: E402
    RESCALE_FACTOR,
    TRIM_KEEP_SILENCE_S,
)
from spatial_director.verifier import (  # noqa: E402
    MAX_WORD_GAP_S,
    HookSpec,
    VerifyInputs,
    _sha256_json,
    verify,
)
from spatial_director.verifier import DEFAULT_CANVAS  # noqa: E402

HOOK_WINDOW_S = 6.0

# Comparison tolerances, stated in every report. Recomputation uses
# the verifier's own code on the same artifact, so genuine values
# agree to rounding; the tolerances only absorb JSON rounding and
# H.264 frame-grab wobble. A tampered value (the proof pack edits
# an occlusion by 0.17) clears every tolerance by an order of
# magnitude.
TOL_RATIO = 0.005
TOL_DB = 0.25
TOL_SEC = 0.02
TOL_PRESENCE = 0.02
TOL_CENTROID = 0.1

# Measured keys that are NOT re-measured: Wonder's judgement is
# recorded, never re-judged; the hook slide speed is a declared
# design value the renderer composites (F7 reads it back off the
# design), not a pixel measurement.
RECORDED_ONLY_KEYS = {
    "wonder_can_summarize_muted_hook",
    "wonder_summary",
}
DECLARED_KEYS = {
    "hook_slide_px_s",
}


# ---------------------------------------------------------------- hashing


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def chain_hash(artifact_hashes: Sequence[str]) -> str:
    """Folded chain fingerprint: h[i] = sha256(h[i-1] + hash[i])."""
    cur = "0" * 64
    for ah in artifact_hashes:
        cur = hashlib.sha256((cur + ah).encode("utf-8")).hexdigest()
    return cur


# ---------------------------------------------------------------- evidence


def _ffprobe_json(path: str | Path, entries: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def artifact_duration(path: str | Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def artifact_canvas(path: str | Path) -> tuple[int, int]:
    data = _ffprobe_json(path, "stream=width,height")
    streams = data.get("streams") or []
    if streams:
        return (int(streams[0]["width"]), int(streams[0]["height"]))
    return DEFAULT_CANVAS


def bundled_evidence_dirs(receipt_path: str | Path) -> list[Path]:
    """Candidate directories of the bundled ``evidence/`` tree.

    Two roots, in order: the tree adjacent to the receipt file
    (``receipts/``'s sibling — where a clone of this repo keeps
    it) and the tree at this tool's own repo root. Each root
    itself and each immediate subdirectory (the shipped layout is
    ``evidence/<set-name>/<files>``) is a candidate, sorted for
    determinism.
    """
    roots: list[Path] = []
    rp = Path(receipt_path)
    if rp.parent.parent != rp.parent:
        roots.append(rp.parent.parent / "evidence")
    roots.append(REPO / "evidence")
    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        cands = [root] + sorted(
            d for d in root.iterdir() if d.is_dir())
        for cand in cands:
            if cand not in seen:
                seen.add(cand)
                dirs.append(cand)
    return dirs


def resolve_artifact_detailed(
        stored: str, *, override: Path | None,
        artifact_dir: Path | None, evidence_dir: Path | None,
        bundled_dirs: Sequence[Path] = ()) \
        -> tuple[Path | None, str]:
    """Resolve a receipt's stored artifact path; report HOW.

    Returns (path, via) with via one of "override", "stored",
    "artifact-dir", "evidence-dir", "bundled-evidence", "missing".
    Stored paths are absolute paths from the machine that issued
    the receipt; a fresh clone has the same files elsewhere. An
    explicit override always wins over the stored path (that is
    what makes a swapped-artifact check possible when the original
    file still exists). The basename fallback is deliberate and
    reported by the caller: a swapped file with the same name is
    still caught by the hash check.
    """
    if override is not None and override.exists():
        return override, "override"
    p = Path(stored)
    if p.exists():
        return p, "stored"
    for label, base in (("artifact-dir", artifact_dir),
                        ("evidence-dir", evidence_dir)):
        if base is not None:
            cand = base / p.name
            if cand.exists():
                return cand, label
    for base in bundled_dirs:
        cand = base / p.name
        if cand.exists():
            return cand, "bundled-evidence"
    return None, "missing"


def resolve_artifact(stored: str, *, override: Path | None,
                     artifact_dir: Path | None,
                     evidence_dir: Path | None,
                     bundled_dirs: Sequence[Path] = ()) -> Path | None:
    path, _via = resolve_artifact_detailed(
        stored, override=override, artifact_dir=artifact_dir,
        evidence_dir=evidence_dir, bundled_dirs=bundled_dirs)
    return path


def find_run_summary(evidence_dir: Path | None) -> dict[str, Any]:
    """The Short's run-summary entry, if one is on disk.

    run-summary.json lives at the RUN level (parent of the Short
    directory) and holds a ``shorts`` list; each entry carries the
    layout and reframe the pipeline planned. A file that already
    IS a short report (has ``layout``) is returned as-is. Matching
    is by the entry's recorded receipt/final path living in the
    evidence directory, falling back to the ``short-N`` directory
    name. A run-level dict is never returned as if it were a short
    entry — its keys would silently miss every lookup.
    """
    for base in (evidence_dir,
                 evidence_dir.parent if evidence_dir else None):
        if base is None:
            continue
        cand = base / "run-summary.json"
        if not cand.exists():
            continue
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if "layout" in data:
            return data
        shorts = data.get("shorts", [])
        if not shorts or evidence_dir is None:
            continue
        for entry in shorts:
            files = entry.get("files", {})
            for key in ("receipt", "final"):
                p = files.get(key)
                if p and Path(p).parent == evidence_dir:
                    return entry
        m = re.match(r"short-(\d+)$", evidence_dir.name)
        if m:
            idx = int(m.group(1))
            for entry in shorts:
                if entry.get("index") == idx:
                    return entry
    return {}


# ---------------------------------------------------------------- hook text


def recover_hook_text(draft: dict[str, Any]) -> str | None:
    """Recover the hook text from the receipt itself.

    F6 records the hook as evidence when it fires on the hook
    (question/conflict patterns carry the raw text; number
    patterns append a bracketed word list, which is stripped).
    Fallback: F5's Wonder summary quotes the on-screen hook text.
    """
    for check in draft.get("checks", []):
        if check.get("id") != "F6":
            continue
        for pat in check.get("measured", {}).get("patterns_fired", []):
            if pat.get("source") == "hook":
                ev = pat.get("evidence", "")
                return re.sub(r"\s*\[[^\]]*\]\s*$", "", ev) or None
    for check in draft.get("checks", []):
        if check.get("id") != "F5":
            continue
        summary = check.get("measured", {}).get("wonder_summary") or ""
        m = re.search(r'on-screen text\s+"([^"]+)"', summary)
        if m:
            return m.group(1)
        m = re.search(r'hook is the on-screen text question\s+"([^"]+)"',
                      summary)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------- trim replay


def trim_removals(window_clip: str | Path) -> list[tuple[float, float]]:
    """Recompute TRIM_SILENCE's removals exactly as the repair does.

    Same profile, same segmenter, same keep rule — no parsing of
    the repair's rounded log string, so frame boundaries reproduce
    exactly instead of to 2dp.
    """
    profile = load_energy(window_clip)
    segs = active_segments(
        profile, float(profile.times[0]),
        float(profile.times[-1]) + profile.hop_s, min_silence_s=0.0)
    removals: list[tuple[float, float]] = []
    for (a0, a1), (b0, _b1) in zip(segs, segs[1:]):
        gap = b0 - a1
        if gap > MAX_WORD_GAP_S:
            keep = min(TRIM_KEEP_SILENCE_S, gap / 2.0)
            removals.append((a1 + keep / 2.0, b0 - keep / 2.0))
    return removals


def _removed_before(removals: Sequence[tuple[float, float]],
                    t: float) -> float:
    return sum(max(0.0, min(t, r1) - r0)
               for r0, r1 in removals if t > r0)


def _map_time(removals: Sequence[tuple[float, float]], t: float,
              *, is_end: bool) -> float:
    for r0, r1 in removals:
        if r0 < t < r1:
            return r0 - _removed_before(removals, r0)
    return t - _removed_before(removals, t)


def remap_cues(cues: Sequence[CaptionCue],
               removals: Sequence[tuple[float, float]]) -> list[CaptionCue]:
    out: list[CaptionCue] = []
    for cue in cues:
        s = _map_time(removals, cue.start_s, is_end=False)
        e = _map_time(removals, cue.end_s, is_end=True)
        if e - s >= 0.2:
            out.append(CaptionCue(cue.text, round(s, 3), round(e, 3)))
    return out


def keep_indices(n_frames: int, fps: float,
                 removals: Sequence[tuple[float, float]]) -> list[int]:
    return [i for i in range(n_frames)
            if not any(r0 <= i / fps < r1 for r0, r1 in removals)]


# ---------------------------------------------------------------- lazy alphas


class LazyAlphas:
    """Canvas-space alphas computed on demand (LRU-cached).

    The verifier indexes alphas repeatedly but never needs them
    all at once; materialising a 660-frame 1080x1920 float32
    sequence is ~5.5 GB. Same pattern as the pipeline's
    LazyCanvasAlphas, reimplemented here so this tool does not
    import the pipeline module.
    """

    def __init__(self, src_alphas: list[np.ndarray],
                 canvas: tuple[int, int], cap: int = 32,
                 placements: list | None = None):
        self.src = src_alphas
        self.canvas = canvas
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cap = cap
        # Layout states: an optional per-frame explicit speaker
        # placement aligned with src_alphas (the pipeline's
        # LazyCanvasAlphas carries the same), so bundle-path
        # geometry is measured in the layout render's space.
        if placements is not None \
                and len(placements) != len(src_alphas):
            raise ValueError("placements must align 1:1 with "
                             "src_alphas")
        self.placements = placements

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        placement = (self.placements[i]
                     if self.placements is not None else None)
        a = to_canvas_alpha(self.src[i], self.canvas,
                            placement=placement).astype(np.float32)
        self.cache[i] = a
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return a

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------- reconstruction


def _iter_window_guides(window_clip: Path, n_frames: int):
    """Source frames as float greyscale arrays in [0, 1], streamed.

    The finish stack consumes guides exactly once, in order, so a
    one-shot generator is enough — the frames are never all held.
    """
    data = _ffprobe_json(window_clip, "stream=width,height")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"could not probe {window_clip}")
    w, h = int(streams[0]["width"]), int(streams[0]["height"])
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(window_clip),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE)
    need = w * h * 3
    yielded = 0
    try:
        while yielded < n_frames:
            buf = proc.stdout.read(need) if proc.stdout else b""
            if len(buf) < need:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            yield frame.astype(np.float64).mean(axis=2) / 255.0
            yielded += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def load_src_alphas(evidence_dir: Path,
                    cache: dict | None = None) -> list[np.ndarray]:
    """Decode the SAM response to source-space alphas (cached).

    The decode is the slow step; it is shared across a chain's
    drafts and across the reframe planning that needs the same
    alphas. The cache holds the UNREFRAMED list — the pad is
    applied per call, so one decode serves every pad.
    """
    key = str(evidence_dir)
    if cache is not None and key in cache:
        return cache[key]
    sam_path = evidence_dir / "sam-response.json"
    window_clip = evidence_dir / "window.mp4"
    if not sam_path.exists():
        raise FileNotFoundError(f"no sam-response.json in {evidence_dir}")
    if not window_clip.exists():
        raise FileNotFoundError(f"no window.mp4 in {evidence_dir}")
    raw = json.loads(sam_path.read_text(encoding="utf-8"))
    parsed = parse_sam_response(raw)
    _idx, src_alphas = finished_alpha(
        parsed, _iter_window_guides(window_clip, len(parsed)))
    src_alphas = list(src_alphas)
    if cache is not None:
        cache[key] = src_alphas
    return src_alphas


def decode_canvas_alphas(evidence_dir: Path, canvas: tuple[int, int],
                          pad_bottom_px: int,
                          keep: list[int] | None,
                          cache: dict | None = None) -> LazyAlphas:
    """Rebuild the canvas alpha sequence the verifier measured.

    Applies the construction reframe pad to the decoded alphas,
    then drops the frames a TRIM_SILENCE repair removed. Raises
    on missing/corrupt evidence — the caller converts that to
    NOT-REPRODUCIBLE.
    """
    src_alphas = load_src_alphas(evidence_dir, cache)
    if pad_bottom_px != 0:
        _f, src_alphas = apply_speaker_reframe(
            [], list(src_alphas), pad_bottom_px)
    if keep is not None:
        src_alphas = [src_alphas[i] for i in keep]
    return LazyAlphas(src_alphas, canvas)


# ---------------------------------------------------------------- geometry bundles
#
# The closure (reverify-geometry-closure-spec-2026-11-03.md): when
# a draft carries a geometry bundle — written by the pipeline at
# verify time, or backfilled below from the receipt's own recorded
# declarations — VerifyInputs are built FROM THE BUNDLE (placement,
# spans, caption rect, hook) instead of recomputing the reframe /
# effect / layout from decoded alphas. The transforms the bundle
# names are replayed with the pipeline's own functions (the same
# deterministic-replay discipline as TRIM_SILENCE below); the
# geometry DECISIONS are always the recorded ones.


class BundleError(Exception):
    """A bundle exists but its recorded geometry cannot be built.

    Carries the checks whose legs the failure takes out; the
    caller declares those NOT-REPRODUCIBLE rather than comparing
    against known-wrong inputs."""

    def __init__(self, reason: str,
                 checks: Sequence[str] = ("F2", "F3", "F4", "F7",
                                          "F8")):
        super().__init__(reason)
        self.checks = tuple(checks)


def _rescale_alphas(alphas: list[np.ndarray],
                    scale: float) -> list[np.ndarray]:
    """RESCALE_SPEAKER's alpha transform, replayed exactly.

    Mirrors the alpha half of rescue_loop._scaled_speaker_inputs
    (same PIL path, same bottom-anchored pad) — the repair's own
    math, applied to the bundle-path alphas at the recorded
    RESCALE_FACTOR.
    """
    from PIL import Image
    out: list[np.ndarray] = []
    for alpha in alphas:
        h, w = alpha.shape
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (w - nw) // 2, h - nh
        small = np.asarray(
            Image.fromarray(
                (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
            ).resize((nw, nh), Image.BILINEAR)
        ).astype(np.float64) / 255.0
        padded = np.zeros((h, w), dtype=np.float64)
        padded[top:top + nh, left:left + nw] = small
        out.append(padded)
    return out


# The emphasis planner's note format (layout_law
# .plan_speaker_emphasis): receipt-era declarations record the
# applied crop + pad ONLY inside this note, so recovering them
# from it is recovery of recorded values, not a recomputation.
_EMPHASIS_NOTE_RE = re.compile(
    r"^emphasis x(\d+\.\d+): crop "
    r"\((-?\d+), (-?\d+), (\d+), (\d+)\) \+ pad (-?\d+);")


def _emphasis_from_record(decl: dict[str, Any]):
    """Rebuild the applied SpeakerEmphasis from the record.

    Preference order: (1) the transform itself (crop + pad),
    carried by bundles the pipeline writes; (2) the same values
    recovered from the planner's recorded note (receipt-era
    declarations), cross-checked against the recorded scale;
    (3) None — the caller falls back to re-planning.
    """
    from spatial_director.layout_law import SpeakerEmphasis
    scale = decl.get("scale")
    crop = decl.get("crop")
    pad = decl.get("pad_bottom_px")
    if crop is None and decl.get("note"):
        m = _EMPHASIS_NOTE_RE.match(str(decl["note"]))
        if (m and scale is not None
                and abs(float(m.group(1)) - float(scale)) < 0.005):
            crop = [int(m.group(i)) for i in (2, 3, 4, 5)]
            pad = int(m.group(6))
    if crop is None or pad is None or scale is None:
        return None
    placed = decl.get("placed_centroid")
    head_top = decl.get("placed_head_top_min")
    return SpeakerEmphasis(
        feasible=True, scale=float(scale),
        crop=tuple(int(v) for v in crop),
        pad_bottom_px=int(pad),
        placed_centroid=(tuple(float(v) for v in placed)
                         if placed else None),
        placed_head_top_min=(float(head_top)
                             if head_top is not None else None),
        note="replayed from the recorded transform")


def replay_speaker_forward(src_alphas: list[np.ndarray],
                           decls: Sequence[dict[str, Any]], *,
                           fps: float,
                           canvas: tuple[int, int]
                           ) -> list[np.ndarray]:
    """Replay the recorded speaker-forward emphasis spans.

    The applied transform is reconstructed from the record
    wherever the record carries it (bundle field, or the
    planner's note on receipt-era declarations) and re-applied
    as-is. Where the record does not carry it, the plan is
    RE-SOLVED on the bundle-path alphas (the planner is a
    deterministic function of the span's alphas, called with
    the pipeline's own arguments) and CROSS-CHECKED against
    the recorded declaration (scale, placed centroid, placed
    head-top) before anything is applied: a replay that
    diverges from the record raises instead of measuring
    against invented geometry.
    """
    out = list(src_alphas)
    for decl in decls:
        s, e = float(decl["span"][0]), float(decl["span"][1])
        f0 = max(0, int(round(s * fps)))
        f1 = min(len(out), int(round(e * fps)))
        if f1 <= f0:
            raise BundleError(
                f"speaker-forward span [{s}, {e}] covers no frames")
        plan = _emphasis_from_record(decl)
        if plan is None:
            plan = plan_speaker_emphasis(out[f0:f1], canvas)
            if not plan.feasible:
                raise BundleError(
                    "speaker-forward replay: emphasis "
                    "infeasible on the bundle-path alphas, but "
                    "the receipt records an applied span at "
                    f"[{s}, {e}]")
            rec_centroid = decl.get("placed_centroid")
            got_centroid = (list(plan.placed_centroid)
                            if plan.placed_centroid else None)
            if (plan.scale != decl.get("scale")
                    or got_centroid != rec_centroid
                    or plan.placed_head_top_min
                    != decl.get("placed_head_top_min")):
                raise BundleError(
                    "speaker-forward replay diverges from the "
                    f"recorded declaration for span [{s}, {e}]: "
                    f"replanned scale {plan.scale} / centroid "
                    f"{got_centroid} / head-top "
                    f"{plan.placed_head_top_min} vs recorded "
                    f"{decl.get('scale')} / {rec_centroid} / "
                    f"{decl.get('placed_head_top_min')}")
        # apply_speaker_emphasis transforms frames alongside
        # alphas; the frames are discarded here, so one shared
        # dummy stands in for all of them (it is only ever read
        # by slicing/padding, never mutated).
        h, w = out[0].shape
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        _frames, out = apply_speaker_emphasis(
            [dummy] * len(out), out, plan, f0, f1)
    return out


def build_bundle_alphas(evidence_dir: Path, bundle: dict[str, Any], *,
                        canvas: tuple[int, int], fps: float,
                        keep: list[int] | None,
                        alpha_cache: dict | None = None
                        ) -> LazyAlphas:
    """Build a draft's canvas alphas from its geometry bundle.

    Replays, in the pipeline's order: SAM decode -> Stage zoom
    (recorded crop + pad) -> hook-effect pad (the recorded total)
    -> speaker-forward emphasis (recorded spans, cross-checked)
    -> the repair's own replay (TRIM keep / RESCALE / REFRAME)
    -> the layout placement track rebuilt from the recorded
    declarations. Raises BundleError on anything the record
    cannot support.
    """
    recipe = bundle.get("alpha_recipe") or {}
    try:
        src = list(load_src_alphas(evidence_dir, alpha_cache))
    except (FileNotFoundError, RuntimeError, ValueError,
            json.JSONDecodeError, OSError) as exc:
        raise BundleError(f"canvas alphas unavailable: {exc}")
    if not src:
        raise BundleError("SAM decode produced no alphas")
    recorded_size = recipe.get("src_frame_size")
    if recorded_size is not None \
            and list(src[0].shape) != list(recorded_size):
        raise BundleError(
            f"source alpha frame size {list(src[0].shape)} does "
            f"not match the bundle's recorded {recorded_size}: "
            "the evidence does not belong to this bundle")
    zoom = gb.stage_zoom_from_recipe(recipe)
    if zoom is not None:
        _f, src = apply_stage_zoom([], src, zoom)
    pad = int(recipe.get("effect_pad_bottom_px") or 0)
    if pad != 0:
        _f, src = apply_speaker_reframe([], src, pad)
    if recipe.get("speaker_forward"):
        src = replay_speaker_forward(
            src, recipe["speaker_forward"], fps=fps, canvas=canvas)
    repair = recipe.get("repair")
    if repair == "RESCALE_SPEAKER":
        src = _rescale_alphas(src, RESCALE_FACTOR)
    elif repair == "REFRAME_SPEAKER":
        # The executor re-plans from the receipt's measured
        # centroid with fixed repair settings; replaying the same
        # planner on the same alphas with the same settings is
        # the deterministic replay (cf. TRIM_SILENCE's removals).
        plan = plan_speaker_reframe(
            src, canvas, target_frac=0.5, head_top_plan_min=100)
        if plan.identity or plan.pad_bottom_px == 0:
            raise BundleError(
                "REFRAME_SPEAKER replay lands on the current "
                "placement; the repair's reframed geometry cannot "
                "be reproduced from the bundle")
        _f, src = apply_speaker_reframe([], src, plan.pad_bottom_px)
    if keep is not None:
        src = [src[i] for i in keep]
    placements = None
    if bundle.get("layout_spans"):
        try:
            placements = gb.placements_from_declarations(
                bundle["layout_spans"], src[0].shape, len(src),
                fps, canvas)
        except ValueError as exc:
            raise BundleError(str(exc))
    return LazyAlphas(src, canvas, placements=placements)


def _backfill_draft_bundle(receipt: dict[str, Any],
                           run_summary: dict[str, Any]
                           ) -> dict[str, Any] | None:
    """Draft 1's bundle, backfilled from the receipt's records.

    Receipts issued before bundles existed still RECORDED the
    geometry their verifier consumed: the draft's stage_geometry
    declaration (the StageZoom declaration plus the applied
    pads), its layout_spans declarations, its hook_style (text
    size + accent), its speaker-forward declarations, and the
    run-summary's layout (hook band + caption rect). The backfill
    copies those records into the bundle shape — nothing is
    recomputed. Returns None when the records are absent (the
    pre-geometry receipt era), leaving the legacy path in charge.
    """
    drafts = receipt.get("drafts") or []
    if not drafts and receipt.get("checks") is not None:
        drafts = [receipt]
    if not drafts:
        return None
    draft = drafts[0]
    stage_geometry = draft.get("stage_geometry")
    if not stage_geometry:
        return None
    layout = run_summary.get("layout", {})
    hook_text = recover_hook_text(draft)
    y_frac = layout.get("hook_y_frac")
    if hook_text is None or y_frac is None:
        return None
    hook_style = draft.get("hook_style") or {}
    recipe = gb.alpha_recipe_dict(
        src_frame_size=None, stage_geometry=stage_geometry,
        speaker_forward=draft.get("speaker_forward"))
    return {
        "stage_geometry": stage_geometry,
        "alpha_recipe": recipe,
        "hook": {
            "text": hook_text,
            "window": [0.0, HOOK_WINDOW_S],
            "y_center_frac": float(y_frac),
            "layout_rendered": True,
            "accent_index": hook_style.get("accent_index"),
            "text_size": int(hook_style.get("text_size", 170)),
            "slide": None,
            "variant": "stage",
        },
        "caption_rect": layout.get("caption_rect"),
        "layout_spans": draft.get("layout_spans"),
        "hook_slide_px_s": 0.0,
        "inputs_hash": None,
        "backfilled": True,
    }


def load_geometry_bundles(evidence_dir: Path | None,
                          receipt: dict[str, Any],
                          run_summary: dict[str, Any]
                          ) -> tuple[dict[int, dict], list[str]]:
    """All per-draft geometry bundles for a receipt.

    A bundle FILE wins when present; each entry's inputs_hash is
    verified against its receipt draft before it is trusted (a
    mismatch is reported and the entry dropped, never used).
    Without a file, draft 1's bundle is backfilled from the
    receipt's recorded declarations when they exist.
    """
    notes: list[str] = []
    if evidence_dir is not None:
        raw = gb.read_bundle_file(evidence_dir)
        if raw:
            drafts = receipt.get("drafts") or []
            trusted: dict[int, dict] = {}
            for idx, bundle in raw.items():
                draft = (drafts[idx - 1]
                         if 0 < idx <= len(drafts) else None)
                recorded = (draft.get("inputs_sha256")
                            if draft else None)
                if draft is not None and gb.check_inputs_hash(
                        bundle, recorded):
                    trusted[idx] = bundle
                else:
                    notes.append(
                        f"draft {idx}: geometry bundle inputs_hash "
                        "does not match the receipt's recorded "
                        "input hashes — bundle not trusted")
            if trusted:
                notes.append("geometry bundle: pipeline record "
                             "(inputs_hash verified)")
                return trusted, notes
    backfilled = _backfill_draft_bundle(receipt, run_summary)
    if backfilled is not None:
        notes.append("geometry bundle: backfilled from the "
                     "receipt's recorded declarations (no bundle "
                     "file in the evidence directory)")
        return {1: backfilled}, notes
    return {}, notes


def resolve_draft_bundle(bundles: dict[int, dict], draft_index: int,
                         repair_menu: str | None
                         ) -> dict[str, Any] | None:
    """The bundle governing one draft.

    A draft-2 entry recorded against draft 1's base recipe is
    merged over it (its own hook / caption rect / layout spans,
    draft 1's alpha transforms plus the repair replay). An
    audio-only repair (NORMALIZE_LOUDNESS) has no entry of its
    own by design: draft 1's bundle governs draft 2 unchanged.
    """
    entry = bundles.get(draft_index)
    if entry is not None:
        recipe = entry.get("alpha_recipe") or {}
        base_idx = recipe.get("base_draft")
        if base_idx is not None and base_idx in bundles:
            merged = dict(bundles[base_idx])
            merged.update(entry)
            merged_recipe = dict(
                bundles[base_idx].get("alpha_recipe") or {})
            merged_recipe["repair"] = recipe.get("repair")
            merged["alpha_recipe"] = merged_recipe
            return merged
        return entry
    if draft_index > 1 and repair_menu in (None,
                                          "NORMALIZE_LOUDNESS"):
        return bundles.get(1)
    return None


def rebuild_graphics(draft: dict[str, Any],
                     cues: Sequence[CaptionCue]) -> list[GraphicSpec] | None:
    """Rebuild GraphicSpecs from the receipt's graphics section.

    Card content is re-rendered by the same code the pipeline used
    (``prepare_card`` at the placed box width, same layer closure),
    so F8's pixel legs measure against identical layers. Image
    graphics cannot be rebuilt — the generated asset is not
    recorded in the receipt — and raise, which the caller reports
    as NOT-REPRODUCIBLE for F8 rather than faking a match.
    """
    entries = draft.get("graphics")
    if entries is None:
        return None
    specs: list[GraphicSpec] = []
    for entry in entries:
        if entry.get("kind") != "card":
            raise ValueError(
                f"cannot rebuild kind {entry.get('kind')!r}: the "
                "generated asset is not recorded in the receipt")
        span = GraphicSpan(
            start_s=float(entry["span"][0]),
            end_s=float(entry["span"][1]),
            kind="card",
            headline=entry.get("headline", ""),
            facts=entry.get("facts", ""),
            source_quote=entry.get("source_quote", ""),
            compare=entry.get("compare"),
            motion=entry.get("motion", "static"),
            path=entry.get("path"),
        )
        box = tuple(int(v) for v in entry["box"])
        content = prepare_card(span, cues, width=box[2] - box[0])

        def layer_fn(t: float, start: float, end: float,
                     _content=content, _box=box,
                     _path=span.path) -> np.ndarray:
            from PIL import Image
            if end > start:
                p = min(1.0, max(0.0, (t - start) / (end - start)))
            else:
                p = 1.0
            img = _content.state_fn(p) if _content.state_fn \
                else _content.final_image
            if _path is None:
                return _paste_layer(img, _box, DEFAULT_CANVAS)
            cur = path_box_at(_box, _path, t, start, DEFAULT_CANVAS)
            cur_size = (cur[2] - cur[0], cur[3] - cur[1])
            if cur_size != img.size:
                img = img.resize(cur_size, Image.BILINEAR)
            return _paste_layer_clipped(img, cur, DEFAULT_CANVAS)

        specs.append(GraphicSpec(
            kind="card", start_s=span.start_s, end_s=span.end_s,
            box=box,
            layer_rgba=layer_fn(span.end_s, span.start_s, span.end_s),
            source_quote=span.source_quote,
            headline=span.headline, facts=span.facts,
            compare=span.compare, motion=span.motion,
            cost_usd=float(entry.get("cost_usd", 0.0)),
            _state_fn=layer_fn, path=span.path))
    return specs


class Reconstruction:
    """Everything rebuilt for one draft, plus what could not be."""

    def __init__(self):
        self.inputs: VerifyInputs | None = None
        self.unreproducible: dict[str, str] = {}  # check id -> reason
        self.notes: list[str] = []
        self.bundle_used: bool = False


def reconstruct_draft(receipt: dict[str, Any], draft: dict[str, Any],
                      artifact: Path, evidence_dir: Path | None,
                      repair_menu: str | None,
                      run_summary: dict[str, Any],
                      alpha_cache: dict | None = None) -> Reconstruction:
    """Rebuild the VerifyInputs the verifier measured this draft with.

    Anything that cannot be rebuilt honestly lands in
    ``unreproducible`` with its reason; nothing is guessed. In
    particular: a repair other than TRIM_SILENCE / NORMALIZE_LOUDNESS
    / DROP_GRAPHICS changes hook geometry, speaker scale, or
    graphics in ways the receipt does not record, so the legs it
    touches are declared NOT-REPRODUCIBLE instead of compared
    against wrong inputs.
    """
    out = Reconstruction()
    fps = video_fps(artifact)
    canvas = artifact_canvas(artifact)
    duration = artifact_duration(artifact)

    # ---- geometry bundle ------------------------------------------
    bundles, bundle_notes = load_geometry_bundles(
        evidence_dir, receipt, run_summary)
    out.notes.extend(bundle_notes)
    bundle = resolve_draft_bundle(
        bundles, int(draft.get("draft", 1)), repair_menu)
    if bundle is not None:
        out.bundle_used = True
        out.notes.append(
            "geometry bundle governs this draft's reconstruction")

    # ---- hook text + layout -------------------------------------
    hook_text = recover_hook_text(draft)
    hook = None
    if bundle is not None:
        try:
            recorded = gb.hook_from_dict(bundle["hook"])
            hook = HookSpec(
                recorded.text,
                (recorded.window[0],
                 min(recorded.window[1], duration)),
                recorded.y_center_frac,
                layout_rendered=recorded.layout_rendered,
                slide=recorded.slide, variant=recorded.variant,
                accent_index=recorded.accent_index,
                text_size=recorded.text_size)
            hook_text = hook.text
        except (KeyError, TypeError, ValueError) as exc:
            out.notes.append(
                f"geometry bundle hook unreadable: {exc}")
            bundle = None
            out.bundle_used = False
    if hook is None and hook_text is not None:
        y_frac = None
        layout = run_summary.get("layout", {})
        if layout.get("hook_y_frac") is not None:
            y_frac = float(layout["hook_y_frac"])
        if y_frac is None:
            out.unreproducible.setdefault(
                "F1", "hook layout (y fraction) is not in the receipt "
                      "and no run-summary.json was found in the "
                      "evidence directory")
            out.unreproducible.setdefault(
                "F2", "hook glyph mask needs the hook layout; see F1")
            out.unreproducible.setdefault(
                "F3", "hook bbox needs the hook layout; see F1")
        else:
            hook = HookSpec(hook_text, (0.0, min(HOOK_WINDOW_S, duration)),
                            y_frac, layout_rendered=True)
    elif hook is None:
        for cid in ("F1", "F2", "F3", "F6"):
            out.unreproducible.setdefault(
                cid, "hook text is not recoverable from this receipt "
                     "(no F6 hook evidence, no Wonder summary quote)")
    # F8's hook-glyph leg needs the hook too.
    if hook is None:
        out.unreproducible.setdefault(
            "F8", "hook glyph zone needs the hook layout; see F1")

    # ---- cues ----------------------------------------------------
    cues: list[CaptionCue] = []
    captions_path = (evidence_dir / "captions.json") \
        if evidence_dir else None
    if captions_path is not None and captions_path.exists():
        try:
            raw = json.loads(captions_path.read_text(encoding="utf-8"))
            cues = [CaptionCue(c["text"], float(c["start_s"]),
                               float(c["end_s"]))
                    for c in raw.get("cues", [])]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            out.notes.append(f"captions.json unreadable: {exc}")
    if not cues:
        for cid in ("F2", "F5", "F6"):
            out.unreproducible.setdefault(
                cid, "no captions.json in the evidence directory")
    draft_cues = cues

    # ---- TRIM_SILENCE replay -------------------------------------
    removals: list[tuple[float, float]] = []
    keep: list[int] | None = None
    if repair_menu == "TRIM_SILENCE":
        window_clip = (evidence_dir / "window.mp4") \
            if evidence_dir else None
        if window_clip is not None and window_clip.exists():
            removals = trim_removals(window_clip)
            draft_cues = remap_cues(cues, removals)
        else:
            out.unreproducible.setdefault(
                "F5", "TRIM_SILENCE repair: window.mp4 missing, "
                      "cue retime cannot be replayed")
    elif bundle is None and repair_menu not in (
            None, "NORMALIZE_LOUDNESS", "DROP_GRAPHICS"):
        # The repair changed inputs the receipt does not record.
        # (With a geometry bundle, the changed inputs ARE recorded
        # — the bundle governs, and these legs compare.)
        if repair_menu in ("REPOSITION_HOOK", "REFIT_HOOK_INK",
                           "REGENERATE_HOOK_TEXT"):
            for cid in ("F1", "F2", "F3"):
                out.unreproducible.setdefault(
                    cid, f"repair {repair_menu} changed the hook "
                         "geometry/text, which the receipt does not "
                         "record; glyph legs cannot be replayed")
        if repair_menu in ("RESCALE_SPEAKER", "REFRAME_SPEAKER"):
            for cid in ("F3", "F4", "F7"):
                out.unreproducible.setdefault(
                    cid, f"repair {repair_menu} changed the speaker "
                         "geometry, which the receipt does not record")
        if repair_menu == "ENGAGE_HOOK_SLIDE":
            for cid in ("F2",):
                out.unreproducible.setdefault(
                    cid, "ENGAGE_HOOK_SLIDE changed glyph positions "
                         "during the slide; the slide parameters are "
                         "not recorded in the receipt")

    # ---- alphas ---------------------------------------------------
    alphas = None
    if bundle is not None and evidence_dir is not None:
        try:
            if removals:
                # Frame count before trimming, from the SAM parse
                # (the same keep computation the legacy path uses).
                sam_path = evidence_dir / "sam-response.json"
                raw = json.loads(sam_path.read_text(encoding="utf-8"))
                n_parsed = len(parse_sam_response(raw))
                keep = keep_indices(n_parsed, fps, removals)
            alphas = build_bundle_alphas(
                evidence_dir, bundle, canvas=canvas, fps=fps,
                keep=keep, alpha_cache=alpha_cache)
        except BundleError as exc:
            for cid in exc.checks:
                out.unreproducible.setdefault(cid, str(exc))
    elif evidence_dir is not None:
        try:
            # Pad: run-summary's applied value when present, else
            # recompute the construction reframe + depth-effect
            # lift from the decoded alphas themselves (both are
            # deterministic functions of hook + geometry).
            pad = None
            sr = run_summary.get("speaker_reframe", {})
            if sr.get("applied_pad_bottom_px") is not None:
                pad = int(sr["applied_pad_bottom_px"])
            if removals:
                # Frame count before trimming, from the SAM parse.
                sam_path = evidence_dir / "sam-response.json"
                raw = json.loads(sam_path.read_text(encoding="utf-8"))
                n_parsed = len(parse_sam_response(raw))
                keep = keep_indices(n_parsed, fps, removals)
            if pad is None:
                # Plan the pad from the decoded alphas themselves
                # (same deterministic functions the pipeline used).
                # The decode is cached, so the alpha build below
                # reuses these exact arrays instead of decoding
                # twice. The effect's placement also recovers the
                # hook layout when no run-summary carries it.
                src = load_src_alphas(evidence_dir, alpha_cache)
                reframe = plan_speaker_reframe(src, canvas)
                if hook_text is not None:
                    effect = plan_hook_effect(
                        hook_text, src, canvas,
                        base_reframe=reframe,
                        hook_sample=min(len(src),
                                        int(HOOK_WINDOW_S * fps)))
                    pad = int(effect.pad_bottom_px)
                    out.notes.append(
                        "reframe pad recomputed from the decoded "
                        "geometry (no run-summary entry for this Short)")
                    if hook is None:
                        try:
                            y_frac = float(
                                effect.placement.y_center_frac)
                        except (AttributeError, TypeError, ValueError):
                            y_frac = None
                        if y_frac is not None:
                            hook = HookSpec(
                                hook_text,
                                (0.0, min(HOOK_WINDOW_S, duration)),
                                y_frac, layout_rendered=True)
                            # The layout-missing reasons no longer apply.
                            for cid in list(out.unreproducible):
                                if "hook layout" in out.unreproducible[cid]:
                                    del out.unreproducible[cid]
                            out.notes.append(
                                "hook layout recovered from the "
                                "recomputed depth-effect placement")
                else:
                    # Hook unknown, so the depth-effect lift cannot
                    # be recomputed: use the construction pad alone
                    # and say so. F4/F7 stay comparable; hook legs
                    # remain NOT-REPRODUCIBLE.
                    pad = int(reframe.pad_bottom_px)
                    out.notes.append(
                        "reframe pad from the construction reframe "
                        "only (hook unknown; depth-effect lift not "
                        "recomputed)")
            alphas = decode_canvas_alphas(
                evidence_dir, canvas, pad, keep, cache=alpha_cache)
        except (FileNotFoundError, RuntimeError, ValueError,
                json.JSONDecodeError, OSError) as exc:
            for cid in ("F2", "F3", "F4", "F7", "F8"):
                out.unreproducible.setdefault(
                    cid, f"canvas alphas unavailable: {exc}")
    else:
        for cid in ("F2", "F3", "F4", "F7", "F8"):
            out.unreproducible.setdefault(
                cid, "no evidence directory: SAM masks unavailable")

    if bundle is None and draft.get("speaker_forward"):
        for cid in ("F3", "F4"):
            out.unreproducible.setdefault(
                cid, "speaker-forward emphasis is baked into this "
                     "render; the tool does not re-apply it")
    if draft.get("backdrops"):
        out.unreproducible.setdefault(
            "F8", "backdrop spans present: the treated backdrop "
                  "image asset is not recorded in the receipt")

    # ---- audio + stem ---------------------------------------------
    try:
        audio = load_energy(artifact)
    except Exception as exc:  # unreadable audio: F1/F5/F9 degrade
        out.notes.append(f"artifact audio unreadable: {exc}")
        audio = None
    speech_audio = None
    f5_measured = next(
        (c.get("measured", {}) for c in draft.get("checks", [])
         if c.get("id") == "F5"), {})
    if f5_measured.get("measured_on") == "speech_stem":
        stem_path = None
        if repair_menu == "TRIM_SILENCE":
            stem_path = (evidence_dir / "trimmed-audio.wav") \
                if evidence_dir else None
        elif evidence_dir is not None:
            stem_path = evidence_dir / "window.mp4"
        if stem_path is not None and stem_path.exists():
            try:
                speech_audio = load_energy(stem_path)
            except Exception as exc:
                out.unreproducible.setdefault(
                    "F5", f"speech stem unreadable: {exc}")
        else:
            out.unreproducible.setdefault(
                "F5", "F5 was measured on a speech stem that is not "
                      "in the evidence directory")

    # ---- graphics --------------------------------------------------
    graphics = None
    if draft.get("graphics") is not None:
        try:
            graphics = rebuild_graphics(draft, cues)
        except ValueError as exc:
            out.unreproducible.setdefault("F8", str(exc))
    backdrops = [] if draft.get("backdrops") is not None else None
    if draft.get("backdrops"):
        # Non-empty backdrops already marked NOT-REPRODUCIBLE above.
        backdrops = None

    if audio is None:
        for cid in ("F1", "F5", "F9"):
            out.unreproducible.setdefault(
                cid, "artifact audio could not be decoded")
        return out
    if alphas is None and not out.unreproducible.get("F2"):
        for cid in ("F2", "F3", "F4", "F7"):
            out.unreproducible.setdefault(
                cid, "canvas alphas unavailable")

    caption_rect = None
    if bundle is not None and bundle.get("caption_rect") is not None:
        caption_rect = tuple(int(v) for v in bundle["caption_rect"])
    if caption_rect is None and alphas is not None and draft_cues:
        try:
            caption_rect = plan_captions(
                [(c.start_s, c.end_s) for c in draft_cues],
                alphas, fps=fps, canvas=canvas).rect
        except Exception as exc:
            out.notes.append(f"caption layout failed: {exc}")
    if caption_rect is None:
        out.unreproducible.setdefault(
            "F2", "caption block could not be laid out")
        out.unreproducible.setdefault(
            "F3", "caption block could not be laid out")

    f7_recorded = next(
        (c.get("measured", {}) for c in draft.get("checks", [])
         if c.get("id") == "F7"), {})
    sam_file = (evidence_dir / "sam-response.json") \
        if evidence_dir else None
    if sam_file is not None and not sam_file.exists():
        # verify() hashes this path unconditionally when set; an
        # absent file must be None, not a crash.
        sam_file = None
    out.inputs = VerifyInputs(
        video_path=artifact, fps=fps,
        canvas_alphas=alphas if alphas is not None else [],
        audio=audio, hook=hook, cues=draft_cues,
        caption_rect=caption_rect,
        sam_response_path=sam_file,
        hook_slide_px_s=float(
            bundle.get("hook_slide_px_s", 0.0) if bundle is not None
            else f7_recorded.get("hook_slide_px_s", 0.0)),
        canvas=canvas, graphics=graphics,
        speech_audio=speech_audio,
        sfx=draft.get("sfx"),
        backdrops=backdrops,
        speaker_forward=draft.get("speaker_forward"),
        layout_spans=(bundle.get("layout_spans")
                      if bundle is not None else None))
    return out


# ---------------------------------------------------------------- chain


def verify_chain(receipt: dict[str, Any],
                 resolved: dict[int, Path | None]) -> dict[str, Any]:
    """Structural + hash verification of the chained receipt."""
    findings: list[dict[str, Any]] = []
    drafts = receipt.get("drafts", [])
    if not drafts and receipt.get("checks") is not None:
        drafts = [receipt]
    chain_hashes: list[str] = []
    for i, draft in enumerate(drafts, start=1):
        if draft.get("draft", i) != i:
            findings.append({
                "kind": "structure",
                "status": "FAIL",
                "detail": f"draft {i} is numbered "
                          f"{draft.get('draft')!r}, expected {i}",
            })
        art = resolved.get(i)
        recorded = (draft.get("inputs_sha256") or {}).get("video")
        if art is None:
            findings.append({
                "kind": "hash", "draft": i, "status": "MISSING",
                "detail": "artifact file not found on disk",
            })
            continue
        actual = sha256_file(art)
        chain_hashes.append(actual)
        if recorded is None:
            findings.append({
                "kind": "hash", "draft": i, "status": "NOT-RECORDED",
                "detail": "receipt records no artifact hash for "
                          "this draft (pre-hash receipts)",
            })
        elif actual != recorded:
            findings.append({
                "kind": "hash", "draft": i, "status": "MISMATCH",
                "detail": f"artifact hash {actual[:16]}... != "
                          f"recorded {recorded[:16]}...: the file "
                          "on disk is not the file the receipt "
                          "was issued for",
            })
        sam_recorded = (draft.get("inputs_sha256") or {}).get(
            "sam_response")
        if sam_recorded is not None:
            sam_path = None
            # Resolved against the artifact's directory.
            cand = art.parent / "sam-response.json"
            if cand.exists():
                sam_path = cand
            if sam_path is not None:
                actual_sam = sha256_file(sam_path)
                if actual_sam != sam_recorded:
                    findings.append({
                        "kind": "hash", "draft": i, "status": "MISMATCH",
                        "detail": "sam-response.json hash differs "
                                  "from the recorded input hash",
                    })
    # Top-level artifact == last draft's artifact (by resolved file).
    if drafts and receipt.get("artifact") is not None:
        last = resolved.get(len(drafts))
        top = resolved.get("top")
        if last is not None and top is not None and last != top:
            findings.append({
                "kind": "structure", "status": "FAIL",
                "detail": "top-level artifact is not the last "
                          "draft's artifact",
            })
    # Repair consistency.
    repair = receipt.get("repair") or {}
    status = repair.get("status")
    final_verdict = receipt.get("final_verdict") or (
        drafts[-1].get("verdict") if drafts else None)
    if drafts:
        last_verdict = drafts[-1].get("verdict")
        if final_verdict is not None and final_verdict != last_verdict:
            findings.append({
                "kind": "structure", "status": "FAIL",
                "detail": f"final verdict {final_verdict!r} != last "
                          f"draft verdict {last_verdict!r}",
            })
    if status == "executed":
        target = repair.get("target_check")
        if len(drafts) < 2:
            findings.append({
                "kind": "structure", "status": "FAIL",
                "detail": "repair executed but the chain has "
                          "fewer than 2 drafts",
            })
        else:
            first_failed = drafts[0].get("failed_checks", [])
            last_failed = drafts[-1].get("failed_checks", [])
            if target and target not in first_failed:
                findings.append({
                    "kind": "structure", "status": "FAIL",
                    "detail": f"repair targets {target} but draft 1 "
                              "did not fail it",
                })
            if target and target in last_failed:
                findings.append({
                    "kind": "structure", "status": "FAIL",
                    "detail": f"repair targets {target} but the "
                              "final draft still fails it",
                })
        # An executed repair does NOT guarantee a pass: the
        # pipeline's own derivation (rescue_loop.rescue) sets the
        # final verdict to FAIL when draft 2 still fails any
        # check. The structural demand is therefore consistency
        # with the verdicts an executed repair can produce — the
        # target rules above do the integrity work (the repair
        # must have fixed the check it claimed to target).
        if final_verdict not in ("PASSED_AFTER_REPAIR", "FAIL"):
            findings.append({
                "kind": "structure", "status": "FAIL",
                "detail": "repair executed but final verdict is "
                          f"{final_verdict!r}, not a verdict an "
                          "executed repair can produce "
                          "(PASSED_AFTER_REPAIR or FAIL)",
            })
    elif status == "not_needed":
        if final_verdict != "PASSED_FIRST_DRAFT":
            findings.append({
                "kind": "structure", "status": "FAIL",
                "detail": "no repair but final verdict is "
                          f"{final_verdict!r}, not PASSED_FIRST_DRAFT",
            })
    return {
        "findings": findings,
        "chain_hash": chain_hash(chain_hashes),
        "ok": not any(f["status"] in ("FAIL", "MISMATCH")
                      for f in findings),
    }


# ---------------------------------------------------------------- comparison


def _tol_for(check_id: str, field: str) -> float | None:
    """Tolerance for a measured field, or None for exact compare."""
    if field in ("glyph_occlusion_mean", "glyph_occlusion_max",
                 "glyph_occlusion_alpha_weighted_mean",
                 "frames_contained_frac", "caption_coverage_of_speech",
                 "hook_opacity", "mean_mask_iou_0_3s",
                 "speaker_scale_change_0_1p5s"):
        return TOL_RATIO
    if field in ("audio_rms_db_0_2s", "integrated_lufs",
                 "true_peak_dbtp", "lra_lu"):
        return TOL_DB
    if field in ("first_speech_onset_s", "max_silence_gap_s",
                 "duration_s"):
        return TOL_SEC
    if field in ("presence_match_min", "absence_match_max",
                 "path_match"):
        return TOL_PRESENCE
    if field in ("face_centroid_median",):
        return TOL_CENTROID
    return None


def _values_match(recorded: Any, recomputed: Any, tol: float | None) -> bool:
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return recorded is recomputed
    if isinstance(recorded, (int, float)) \
            and isinstance(recomputed, (int, float)):
        if tol is None:
            return recorded == recomputed
        return abs(float(recorded) - float(recomputed)) <= tol
    if isinstance(recorded, list) and isinstance(recomputed, list):
        if len(recorded) != len(recomputed):
            return False
        return all(_values_match(r, c, tol)
                   for r, c in zip(recorded, recomputed))
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        if set(recorded) != set(recomputed):
            return False
        return all(_values_match(recorded[k], recomputed[k], tol)
                   for k in recorded)
    return recorded == recomputed


def _skip_fields(check_id: str, draft: dict[str, Any],
                 inputs: VerifyInputs | None,
                 repair_menu: str | None,
                 bundle_used: bool = False) -> set[str]:
    """Measured fields that cannot be honestly compared for this draft.

    Derived from the rebuilt inputs' state, not from reasons:
    a field is skipped only when its measurement input is absent
    or was changed by a repair the receipt does not record. With
    a geometry bundle the repair's changed inputs ARE recorded
    (and were rebuilt from the record), so the repair-based and
    speaker-forward skips do not apply.
    """
    skip: set[str] = set()
    if inputs is None:
        return {"*"}
    hook_ok = inputs.hook is not None
    alphas_ok = len(inputs.canvas_alphas) > 0
    cues_ok = bool(inputs.cues)
    rect_ok = inputs.caption_rect is not None
    if not hook_ok:
        if check_id == "F1":
            skip |= {"hook_present_at_frame_0", "hook_present_by_1s",
                     "hook_opacity", "content_words"}
        if check_id == "F2":
            skip |= {"glyph_occlusion_mean", "glyph_occlusion_max",
                     "glyph_occlusion_alpha_weighted_mean"}
        if check_id == "F3":
            skip |= {"hook_px_outside_central_band"}
        if check_id == "F6":
            skip |= {"*"}
        if check_id == "F8":
            skip |= {"hook_glyph_overlap_px"}
    if not alphas_ok:
        if check_id in ("F2", "F4", "F7", "F8"):
            skip |= {"*"}
        if check_id == "F3":
            skip |= {"face_centroid_median",
                     "face_centroid_in_central_band"}
    if not cues_ok:
        if check_id == "F2":
            skip |= {"caption_face_overlap_max_px"}
        if check_id == "F5":
            skip |= {"captions_present", "caption_coverage_of_speech"}
        if check_id == "F6":
            skip |= {"*"}
    if not rect_ok:
        if check_id == "F2":
            skip |= {"caption_face_overlap_max_px"}
        if check_id == "F3":
            skip |= {"caption_px_outside_lower_zone"}
        if check_id == "F8":
            skip |= {"caption_overlap_px", "caption_zone_overlap_px"}
    if check_id == "F5":
        rec = next((c.get("measured", {}) for c in draft.get("checks", [])
                    if c.get("id") == "F5"), {})
        if rec.get("measured_on") == "speech_stem" \
                and inputs.speech_audio is None:
            skip |= {"max_silence_gap_s", "first_speech_onset_s",
                     "caption_coverage_of_speech", "measured_on"}
    if check_id == "F8":
        if inputs.graphics is None and draft.get("graphics") is not None:
            skip |= {"*"}
        if draft.get("backdrops"):
            skip |= {"backdrops"}
    if not bundle_used and draft.get("speaker_forward"):
        if check_id == "F3":
            skip |= {"face_centroid_median",
                     "face_centroid_in_central_band"}
        if check_id == "F4":
            skip |= {"*"}
    if not bundle_used and repair_menu in (
            "REPOSITION_HOOK", "REFIT_HOOK_INK"):
        if check_id == "F1":
            skip |= {"hook_present_at_frame_0", "hook_present_by_1s",
                     "hook_opacity"}
        if check_id == "F2":
            skip |= {"glyph_occlusion_mean", "glyph_occlusion_max",
                     "glyph_occlusion_alpha_weighted_mean"}
        if check_id == "F3":
            skip |= {"hook_px_outside_central_band"}
        if check_id == "F8":
            skip |= {"hook_glyph_overlap_px"}
    if not bundle_used and repair_menu == "REGENERATE_HOOK_TEXT":
        if check_id in ("F1", "F2", "F3", "F6"):
            skip |= {"*"}
        if check_id == "F8":
            skip |= {"hook_glyph_overlap_px"}
    if not bundle_used and repair_menu in (
            "RESCALE_SPEAKER", "REFRAME_SPEAKER"):
        if check_id == "F3":
            skip |= {"face_centroid_median",
                     "face_centroid_in_central_band"}
        if check_id == "F4":
            skip |= {"*"}
        if check_id == "F7":
            skip |= {"speaker_scale_change_0_1p5s",
                     "mean_mask_iou_0_3s"}
    if not bundle_used and repair_menu == "ENGAGE_HOOK_SLIDE":
        if check_id == "F2":
            skip |= {"glyph_occlusion_mean", "glyph_occlusion_max",
                     "glyph_occlusion_alpha_weighted_mean"}
        if check_id == "F8":
            skip |= {"hook_glyph_overlap_px"}
    return skip


def _compare_f3_violations(recorded: dict[str, Any],
                           recomputed: dict[str, Any],
                           skip: set[str]) -> list[dict[str, Any]]:
    """Per-subkey F3 comparison: the violations block mixes legs
    with different inputs (hook bbox, caption block, face
    centroid), so a whole-dict compare would let one missing
    input poison the others."""
    out: list[dict[str, Any]] = []
    for key, rec_val in recorded.items():
        if key in skip:
            out.append({"field": f"violations.{key}",
                        "status": "NOT-REPRODUCIBLE",
                        "recorded": rec_val, "recomputed": None})
            continue
        new_val = recomputed.get(key)
        if isinstance(rec_val, dict) and isinstance(new_val, dict):
            ok = all(_values_match(rec_val[k], new_val.get(k), None)
                     for k in rec_val) \
                and set(rec_val) == set(new_val)
        else:
            ok = _values_match(rec_val, new_val,
                               _tol_for("F3", key))
        out.append({"field": f"violations.{key}",
                    "status": "MATCH" if ok else "MISMATCH",
                    "recorded": rec_val, "recomputed": new_val})
    return out


def _compare_f8_graphics(recorded: dict[str, Any],
                         recomputed: dict[str, Any],
                         skip: set[str]) -> list[dict[str, Any]]:
    """Per-graphic F8 comparison (nested lists need index alignment)."""
    out: list[dict[str, Any]] = []
    rec_g = recorded.get("graphics", [])
    new_g = recomputed.get("graphics", [])
    if len(rec_g) != len(new_g):
        out.append({"field": "graphics", "status": "MISMATCH",
                    "recorded": f"{len(rec_g)} graphics",
                    "recomputed": f"{len(new_g)} graphics"})
        return out
    for i, (r, n) in enumerate(zip(rec_g, new_g)):
        for key in r:
            if key in ("index",):
                continue
            if key == "hook_glyph_overlap_px" \
                    and "hook_glyph_overlap_px" in skip:
                out.append({"field": f"graphics[{i}].{key}",
                            "status": "NOT-REPRODUCIBLE",
                            "recorded": r[key], "recomputed": None})
                continue
            if key in ("caption_overlap_px", "caption_zone_overlap_px") \
                    and key in skip:
                out.append({"field": f"graphics[{i}].{key}",
                            "status": "NOT-REPRODUCIBLE",
                            "recorded": r[key], "recomputed": None})
                continue
            tol = _tol_for("F8", key)
            out.append({
                "field": f"graphics[{i}].{key}",
                "status": "MATCH" if _values_match(r[key], n.get(key),
                                                      tol)
                else "MISMATCH",
                "recorded": r[key], "recomputed": n.get(key)})
    for key in ("offending_indices", "collision_violations",
                "offending_backdrop_indices"):
        if key in recorded or key in recomputed:
            out.append({
                "field": key,
                "status": "MATCH" if _values_match(
                    recorded.get(key), recomputed.get(key), None)
                else "MISMATCH",
                "recorded": recorded.get(key),
                "recomputed": recomputed.get(key)})
    if "backdrops" in skip:
        out.append({"field": "backdrops", "status": "NOT-REPRODUCIBLE",
                    "recorded": recorded.get("backdrops"),
                    "recomputed": None})
    elif "backdrops" in recorded or "backdrops" in recomputed:
        rec_bd = recorded.get("backdrops")
        new_bd = recomputed.get("backdrops")
        # An old receipt that predates the backdrop leg records
        # nothing; a recomputed empty list is the same statement.
        if rec_bd is None and new_bd in (None, []):
            status = "MATCH"
        else:
            status = "MATCH" if _values_match(
                rec_bd, new_bd, TOL_PRESENCE) else "MISMATCH"
        out.append({
            "field": "backdrops", "status": status,
            "recorded": rec_bd, "recomputed": new_bd})
    if "sfx" in recorded or "sfx" in recomputed:
        rec_sfx = recorded.get("sfx")
        new_sfx = recomputed.get("sfx")
        if rec_sfx is None and new_sfx is None:
            status = "MATCH"
        else:
            status = "MATCH" if _values_match(
                rec_sfx, new_sfx, None) else "MISMATCH"
        out.append({
            "field": "sfx", "status": status,
            "recorded": rec_sfx, "recomputed": new_sfx})
    return out


def compare_draft(draft: dict[str, Any], recomputed_checks: list[dict],
                  inputs: VerifyInputs | None,
                  repair_menu: str | None,
                  unreproducible: dict[str, str],
                  bundle_used: bool = False) -> dict[str, Any]:
    """Compare one draft's recorded checks against a fresh verify()."""
    new_by_id = {c["id"]: c for c in recomputed_checks}
    results: list[dict[str, Any]] = []
    advisory: list[str] = []
    for recorded_check in draft.get("checks", []):
        cid = recorded_check["id"]
        rec_measured = recorded_check.get("measured", {})
        if cid == "F8" and not draft.get("graphics") \
                and not draft.get("backdrops"):
            # No graphics and no backdrops on this draft: F8 has
            # nothing to certify. NOT-APPLICABLE (excluded from
            # the findings), never NOT-REPRODUCIBLE.
            results.append({
                "check": cid, "status": "NOT-APPLICABLE",
                "reason": "draft carries no graphics and no "
                          "backdrops; F8 has nothing to certify",
                "fields": []})
            continue
        skip = _skip_fields(cid, draft, inputs, repair_menu,
                            bundle_used=bundle_used)
        if "*" in skip:
            results.append({
                "check": cid, "status": "NOT-REPRODUCIBLE",
                "reason": unreproducible.get(
                    cid, "measurement input unavailable"),
                "fields": []})
            continue
        new_check = new_by_id.get(cid)
        if new_check is None:
            results.append({
                "check": cid, "status": "NOT-REPRODUCIBLE",
                "reason": "fresh verify() produced no such check",
                "fields": []})
            continue
        new_measured = new_check["measured"]
        fields: list[dict[str, Any]] = []
        for key, rec_val in rec_measured.items():
            if key in RECORDED_ONLY_KEYS:
                continue  # handled by the Wonder-record check below
            if key in DECLARED_KEYS:
                fields.append({"field": key, "status": "DECLARED",
                               "recorded": rec_val,
                               "recomputed": new_measured.get(key)})
                continue
            if key in skip:
                fields.append({"field": key,
                               "status": "NOT-REPRODUCIBLE",
                               "recorded": rec_val, "recomputed": None})
                continue
            if cid == "F8" and key in ("graphics", "offending_indices",
                                      "collision_violations",
                                      "offending_backdrop_indices",
                                      "backdrops", "sfx"):
                continue  # handled by the nested comparison below
            if cid == "F3" and key == "violations":
                continue  # handled by the nested comparison below
            tol = _tol_for(cid, key)
            fields.append({
                "field": key,
                "status": "MATCH" if _values_match(
                    rec_val, new_measured.get(key), tol) else "MISMATCH",
                "recorded": rec_val,
                "recomputed": new_measured.get(key)})
        if cid == "F8":
            fields.extend(_compare_f8_graphics(
                rec_measured, new_measured, skip))
        if cid == "F3":
            fields.extend(_compare_f3_violations(
                rec_measured.get("violations", {}),
                new_measured.get("violations", {}), skip))
        # Wonder-record check: a claimed judgement needs its call.
        if cid == "F5":
            claimed = rec_measured.get("wonder_can_summarize_muted_hook")
            calls = draft.get("meta_calls", [])
            has_wonder = any(
                c.get("model") == "muse-wonder-1.0"
                and c.get("purpose") == "verifier-f5-muted-hook"
                and c.get("request_id")
                for c in calls)
            if claimed:
                fields.append({
                    "field": "wonder_judgement",
                    "status": "RECORDED" if has_wonder else "NOT-RECORDED",
                    "recorded": "wonder call recorded"
                    if has_wonder else "no wonder call recorded",
                    "recomputed": None})
            else:
                fields.append({
                    "field": "wonder_judgement",
                    "status": "NOT-CLAIMED",
                    "recorded": "judge not claimed",
                    "recomputed": None})
        status = "MATCH"
        if any(f["status"] == "MISMATCH" for f in fields):
            status = "MISMATCH"
        elif any(f["status"] == "NOT-RECORDED" for f in fields):
            status = "NOT-RECORDED"
        elif any(f["status"] == "NOT-REPRODUCIBLE" for f in fields):
            status = "PARTIAL"
        results.append({"check": cid, "status": status, "fields": fields})
    recorded_ids = {c["id"] for c in draft.get("checks", [])}
    for cid, new_check in new_by_id.items():
        if cid not in recorded_ids:
            advisory.append(
                f"{cid} computed by this tool but not recorded in "
                f"this receipt (receipt predates the check)")
    return {"checks": results, "advisory": advisory}


# ---------------------------------------------------------------- report


BOUNDARY = (
    "BOUNDARY: Wonder judgement legs (F5's muted-hook summary) are "
    "model calls and cannot be re-run keylessly — this tool verifies "
    "they are RECORDED (model id + request id in the draft's "
    "meta_calls), it does not re-judge. The promise is "
    "hash-verification + deterministic re-measurement, NEVER "
    "bit-identical re-rendering: Voice Transcribe and SAM are not "
    "bit-deterministic, so a re-render would differ even when the "
    "receipt is honest. What is re-measured here is the artifact "
    "file itself, which does not change."
)


def _resolve_all(receipt: dict[str, Any], drafts: list,
                 single: bool, receipt_path: str | Path, *,
                 artifact: Path | None, evidence_dir: Path | None,
                 artifact_dir: Path | None):
    """Resolve every draft artifact (plus the top-level one).

    Returns (resolved, resolutions, evidence_dir). ``resolutions``
    records every NON-stored resolution — a fallback is part of
    the report, never silent. When no evidence dir was given and
    a resolved artifact's directory holds the full evidence file
    set (captions.json, sam-response.json, window.mp4), that
    directory is adopted as the evidence dir so a shipped evidence
    set re-measures in full; the adoption is recorded too.
    """
    bundled = bundled_evidence_dirs(receipt_path)
    if evidence_dir is None and artifact is not None:
        evidence_dir = artifact.parent
    resolved: dict[Any, Path | None] = {}
    resolutions: list[dict[str, Any]] = []

    def _one(what: str, stored: str, override: Path | None):
        path, via = resolve_artifact_detailed(
            stored, override=override, artifact_dir=artifact_dir,
            evidence_dir=evidence_dir, bundled_dirs=bundled)
        if via not in ("stored", "missing"):
            resolutions.append({
                "what": what, "stored": stored,
                "resolved": str(path), "via": via})
        return path

    for i, draft in enumerate(drafts, start=1):
        resolved[i] = _one(
            f"draft {i} artifact", draft.get("artifact", ""),
            artifact if i == len(drafts) else None)
    if not single and receipt.get("artifact"):
        resolved["top"] = _one("top-level artifact",
                               receipt["artifact"], artifact)
    if evidence_dir is None:
        for i in range(1, len(drafts) + 1):
            art = resolved.get(i)
            if art is not None and all(
                    (art.parent / name).exists() for name in
                    ("captions.json", "sam-response.json",
                     "window.mp4")):
                evidence_dir = art.parent
                resolutions.append({
                    "what": "evidence directory",
                    "stored": "(none given)",
                    "resolved": str(art.parent), "via": "adopted"})
                break
    return resolved, resolutions, evidence_dir


def reverify(receipt_path: str | Path, *,
             artifact: Path | None = None,
             evidence_dir: Path | None = None,
             artifact_dir: Path | None = None) -> dict[str, Any]:
    """Full re-verification of one receipt. Returns the report dict."""
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    drafts = receipt.get("drafts", [])
    single = False
    if not drafts and receipt.get("checks") is not None:
        drafts = [receipt]
        single = True
    repair = receipt.get("repair") or {}
    repair_menu = repair.get("menu_id")
    repair_status = repair.get("status")

    # Resolve artifacts first: the chain check needs every draft.
    resolved, resolutions, evidence_dir = _resolve_all(
        receipt, drafts, single, receipt_path, artifact=artifact,
        evidence_dir=evidence_dir, artifact_dir=artifact_dir)
    chain = verify_chain(receipt, resolved)

    run_summary: dict[str, Any] = {}
    if evidence_dir is not None:
        run_summary = find_run_summary(evidence_dir)

    draft_reports: list[dict[str, Any]] = []
    alpha_cache: dict = {}
    for i, draft in enumerate(drafts, start=1):
        art = resolved.get(i)
        menu = repair_menu if i > 1 else None
        if art is None:
            draft_reports.append({
                "draft": i, "status": "MISSING-ARTIFACT",
                "checks": [], "advisory": [],
                "notes": ["artifact file not found on disk"],
            })
            continue
        rec = reconstruct_draft(receipt, draft, art, evidence_dir,
                                menu, run_summary,
                                alpha_cache=alpha_cache)
        if rec.inputs is None:
            draft_reports.append({
                "draft": i, "status": "NOT-REPRODUCIBLE",
                "checks": [{"check": c["id"],
                            "status": "NOT-REPRODUCIBLE",
                            "reason": rec.unreproducible.get(c["id"], ""),
                            "fields": []}
                           for c in draft.get("checks", [])],
                "advisory": [],
                "notes": rec.notes,
            })
            continue
        fresh = verify(rec.inputs, wonder=None, draft=i)
        fresh_checks = [c.to_dict() for c in fresh.checks]
        comp = compare_draft(draft, fresh_checks, rec.inputs, menu,
                             rec.unreproducible,
                             bundle_used=rec.bundle_used)
        status = "MATCH"
        if any(c["status"] == "MISMATCH" for c in comp["checks"]):
            status = "MISMATCH"
        elif any(c["status"] == "NOT-RECORDED" for c in comp["checks"]):
            status = "NOT-RECORDED"
        elif any(c["status"] in ("PARTIAL", "NOT-REPRODUCIBLE")
                 for c in comp["checks"]):
            status = "PARTIAL"
        draft_reports.append({
            "draft": i, "status": status, "checks": comp["checks"],
            "advisory": comp["advisory"], "notes": rec.notes,
        })

    ok = chain["ok"] and all(
        d["status"] in ("MATCH", "PARTIAL") for d in draft_reports)
    return {
        "receipt": str(receipt_path),
        "final_verdict": receipt.get("final_verdict"),
        "chain": chain,
        "drafts": draft_reports,
        "resolutions": resolutions,
        "evidence_dir": str(evidence_dir) if evidence_dir else None,
        "boundary": BOUNDARY,
        "ok": ok,
    }


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Receipt: {report['receipt']}")
    lines.append(f"Final verdict: {report['final_verdict']}")
    for r in report.get("resolutions", []):
        lines.append(
            f"  RESOLVED {r['what']}: stored {r['stored']!r} -> "
            f"{r['resolved']} (via {r['via']})")
    if report.get("evidence_dir"):
        lines.append(f"  Evidence dir: {report['evidence_dir']}")
    lines.append(f"Chain hash: {report['chain']['chain_hash']}")
    for f in report["chain"]["findings"]:
        lines.append(f"  CHAIN {f['status']}: {f['detail']}")
    if not report["chain"]["findings"]:
        lines.append("  CHAIN OK: structure + per-draft hashes verify")
    for d in report["drafts"]:
        lines.append(f"Draft {d['draft']}: {d['status']}")
        for note in d.get("notes", []):
            lines.append(f"  note: {note}")
        for c in d["checks"]:
            lines.append(f"  {c['check']}: {c['status']}"
                         + (f" ({c['reason']})" if c.get("reason")
                            else ""))
            for fld in c.get("fields", []):
                if fld["status"] in ("MISMATCH", "NOT-RECORDED"):
                    lines.append(
                        f"    ! {fld['field']}: recorded="
                        f"{fld['recorded']!r} recomputed="
                        f"{fld['recomputed']!r}")
                elif fld["status"] == "NOT-REPRODUCIBLE":
                    lines.append(
                        f"    ~ {fld['field']}: not reproducible "
                        f"(recorded={fld['recorded']!r})")
        for adv in d.get("advisory", []):
            lines.append(f"  advisory: {adv}")
    lines.append("")
    lines.append(report["boundary"])
    lines.append("")
    lines.append("VERDICT: " + ("PASS — receipt verifies"
                                if report["ok"] else "FAIL — see above"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-verify a Spatial Director receipt against its "
                    "artifact, keylessly and offline.")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--artifact", type=Path, default=None,
                        help="override the final artifact path")
    parser.add_argument("--evidence-dir", type=Path, default=None,
                        help="Short directory holding captions.json, "
                             "sam-response.json, window.mp4, etc.")
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="directory to resolve stored artifact "
                             "paths by basename")
    parser.add_argument("--json", action="store_true",
                        help="print the machine-readable report")
    args = parser.parse_args(argv)
    report = reverify(
        args.receipt, artifact=args.artifact,
        evidence_dir=args.evidence_dir, artifact_dir=args.artifact_dir)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
