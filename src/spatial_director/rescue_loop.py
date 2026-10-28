"""The rescue loop: one bounded, menu-driven repair per failed Short.

This is the second half of the verifier's contract. The verifier
(``verifier.py``) issues a receipt whose failed checks carry the
measured value next to the threshold - feedback that is specific and
machine-actionable, per the Agents-102 verification session. This
module acts on exactly one such failure, exactly once:

1. **Plan (Spark 1.3, ledger-gated).** Spark receives the failed
   checks with their measurements and the FIXED repair menu below,
   and picks ONE repair. The menu is code, not prompt suggestion:
   the reply is parsed strictly, and a pick outside the menu - or a
   menu entry that does not apply to any failed check - is discarded
   by construction, so the planner cannot invent a repair the system
   does not know how to execute (the Agents-102 allowlist pattern).
2. **Execute from cached artifacts.** The repair re-renders from the
   render's cached SAM masks, captions and layout inputs. There is
   deliberately NO SAM client anywhere in this module: segmentation
   is never re-bought for a repair. New Spark/Wonder calls are
   allowed (the planner itself, a hook regeneration, the re-verify
   judge) and are ledger-logged by their clients.
3. **Re-verify.** The repaired render goes through the full F1-F7
   verifier again (draft 2), judge included.
4. **Chain.** The emitted receipt chains draft 1 -> repair (menu id,
   Spark's stated reason, the planner call's cost + request id) ->
   draft 2, with the final verdict. ONE repair max, hard-coded by
   structure: this function plans once, executes once, verifies
   once, and returns. A render that passes draft 1
   (``PASSED_FIRST_DRAFT``) never enters the loop at all.

Repairs are expressed only through parameters the verifier itself
re-measures from (``HookSpec`` text / band / renderer, caption cues
and block, the frame set, the speaker scale) - never through pixels
the receipt could not reproduce. A repair whose executor finds
nothing to change raises :class:`RepairNotApplicable` and the draft-1
FAIL stands, on the record.

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from .caption_align import CaptionCue, active_segments, load_energy
from .compositor import (
    DEFAULT_CANVAS,
    composite_frame,
    hook_visible_at,
)
from .layout_law import (
    GLYPH_OCCLUSION_CAP,
    SAFE_X0,
    SAFE_X1,
    SAFE_Y0,
    SAFE_Y1,
    caption_layer,
    hook_glyph_mask,
    occupancy_map,
    plan_captions,
    plan_hook,
    predict_occlusion,
)
from .spark_client import SparkClient
from .verifier import MAX_WORD_GAP_S, HookSpec, Receipt, _content_words

# ---------------------------------------------------------------- the menu

TRIM_SILENCE = "TRIM_SILENCE"
REPOSITION_HOOK = "REPOSITION_HOOK"
REFIT_HOOK_INK = "REFIT_HOOK_INK"
RESCALE_SPEAKER = "RESCALE_SPEAKER"
REGENERATE_HOOK_TEXT = "REGENERATE_HOOK_TEXT"


@dataclass(frozen=True)
class RepairSpec:
    menu_id: str
    target_checks: tuple[str, ...]
    description: str


REPAIR_MENU: dict[str, RepairSpec] = {
    TRIM_SILENCE: RepairSpec(
        TRIM_SILENCE, ("F5",),
        "Cut the over-long silence pause(s) out of the render and "
        "retime the captions onto the tightened cut."),
    REPOSITION_HOOK: RepairSpec(
        REPOSITION_HOOK, ("F2",),
        "Re-plan the hook band (and the caption block if it is the "
        "overlap source) against a stricter glyph-occlusion target."),
    REFIT_HOOK_INK: RepairSpec(
        REFIT_HOOK_INK, ("F3",),
        "Re-render the hook through the layout path so its ink fits "
        "inside the safe zone."),
    RESCALE_SPEAKER: RepairSpec(
        RESCALE_SPEAKER, ("F4",),
        "Re-render with the speaker scaled down so the head clears "
        "the canvas edges with margin."),
    REGENERATE_HOOK_TEXT: RepairSpec(
        REGENERATE_HOOK_TEXT, ("F1", "F6", "F7"),
        "Replace the hook line with a new one for the same moment "
        "(more content words / an open-loop pattern / a stronger "
        "opening line)."),
}

# Executor constants.
TRIM_KEEP_SILENCE_S = 0.30   # pause left in the cut after a trim
REPOSITION_CAP_FRACTION = 0.5  # of the F2 cap: the stricter target
RESCALE_FACTOR = 0.92
PLANNER_MAX_TOKENS = 2500    # Spark reasons first; tight budgets return empty
HOOK_GEN_MAX_TOKENS = 2500

PLANNER_SYSTEM_PROMPT = (
    "You are the repair planner for a verified video pipeline. A "
    "render failed one or more fixed checks; each failure names the "
    "measured value and the threshold it missed. Choose exactly ONE "
    "repair from the menu you are given - the menu is fixed and "
    "anything outside it will be discarded. Prefer the repair that "
    "addresses the failure with the largest measured miss. Reply "
    "with strict JSON only: {\"repair\": \"<MENU_ID>\", \"reason\": "
    "\"<one line: which measured value this repair changes, and "
    "how>\"}."
)

HOOK_GEN_SYSTEM_PROMPT = (
    "You write ONE replacement hook line for a short vertical "
    "video, for the same moment and speaker. Rules: 3 to 7 words; "
    "at least 3 content words; the line must contain a conflict "
    "word (never, no, quit, secret, truth, wrong, mistake...), a "
    "number, or be a question. Plain words from the moment itself - "
    "do not invent facts the transcript does not support. Reply "
    "with the hook line only, nothing else."
)


class RepairNotApplicable(RuntimeError):
    """The chosen repair has nothing to change on this render."""


# ---------------------------------------------------------------- artifacts


@dataclass
class RenderArtifacts:
    """Everything a repair may consume - all of it cached at render
    time. Note what is NOT here: no SAM client and no way to buy new
    segmentation. ``sam_response_path`` is the cached response the
    draft was decoded from, carried only so re-issued receipts can
    hash the same input."""

    window_clip: str | Path          # the cut source window (has audio)
    render_path: str | Path          # the draft-1 render being rescued
    frames: list[np.ndarray]         # source frames of the window
    src_alphas: list[np.ndarray]     # finished alpha, source resolution
    canvas_alphas: Sequence[np.ndarray]  # canvas-space, frame-aligned
    make_canvas_alphas: Callable[[list[np.ndarray]], Sequence[np.ndarray]]
    fps: float
    hook: HookSpec
    cues: list[CaptionCue]
    caption_rect: tuple[int, int, int, int] | None
    sam_response_path: str | Path | None = None
    transcript_text: str = ""        # the moment's transcript (hook regen)
    work_dir: str | Path = "."       # repaired renders land here
    canvas: tuple[int, int] = DEFAULT_CANVAS


@dataclass
class RepairExecution:
    """One executed repair: the new render + the exact inputs the
    re-verify must measure it with."""

    render_path: Path
    hook: HookSpec
    cues: list[CaptionCue]
    caption_rect: tuple[int, int, int, int] | None
    canvas_alphas: Sequence[np.ndarray]
    detail: str = ""


@dataclass
class RepairRecord:
    status: str  # not_needed / executed / no_repair_available /
    #              not_applicable / planner_failed
    menu_id: str | None = None
    target_check: str | None = None
    spark_reason: str | None = None
    spark_call: dict[str, Any] | None = None
    extra_calls: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "menu_id": self.menu_id,
            "target_check": self.target_check,
            "spark_reason": self.spark_reason,
            "spark_call": self.spark_call,
            "extra_calls": self.extra_calls,
            "detail": self.detail,
        }


@dataclass
class ChainedReceipt:
    artifact: str
    final_verdict: str
    drafts: list[Receipt]
    repair: RepairRecord
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_version": 1,
            "artifact": self.artifact,
            "created": self.created,
            "final_verdict": self.final_verdict,
            "repair": self.repair.to_dict(),
            "drafts": [d.to_dict() for d in self.drafts],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------- rendering


def _scaled_speaker_inputs(
    frames: list[np.ndarray],
    alphas: list[np.ndarray],
    scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Shrink the speaker inside its own frame (bottom-anchored).

    The compositor's speaker placement fits the frame to the canvas
    width, so a repair cannot ask it for a smaller speaker directly.
    Instead the frame + alpha are scaled down and padded back to the
    source size - RGB edge-padded (it only feeds the blurred
    backdrop), alpha zero-padded - after which the standard
    placement lands the speaker at ``scale`` of its former size,
    still centred and bottom-anchored, and ``to_canvas_alpha`` of
    the padded alpha reproduces exactly the geometry rendered.
    """
    if scale >= 1.0:
        return frames, alphas
    out_frames: list[np.ndarray] = []
    out_alphas: list[np.ndarray] = []
    for frame, alpha in zip(frames, alphas):
        h, w = alpha.shape
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (w - nw) // 2, h - nh
        small_rgb = np.asarray(
            Image.fromarray(frame).resize((nw, nh), Image.BILINEAR))
        small_a = np.asarray(
            Image.fromarray(
                (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
            ).resize((nw, nh), Image.BILINEAR)
        ).astype(np.float64) / 255.0
        padded_rgb = np.pad(
            small_rgb,
            ((top, 0), (left, w - nw - left), (0, 0)), mode="edge")
        padded_a = np.zeros((h, w), dtype=np.float64)
        padded_a[top:top + nh, left:left + nw] = small_a
        out_frames.append(padded_rgb.astype(np.uint8))
        out_alphas.append(padded_a)
    return out_frames, out_alphas


def render_composite_with_captions(
    frames: list[np.ndarray],
    src_alphas: list[np.ndarray],
    *,
    hook_rgba: np.ndarray | None,
    hook_window: tuple[float, float] | None,
    cues: Sequence[CaptionCue],
    caption_rect: tuple[int, int, int, int] | None,
    audio_from: str | Path | None,
    out_path: str | Path,
    fps: float,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    speaker_scale: float = 1.0,
) -> Path:
    """The shared render path: composite + captions IN FRONT.

    Draft 1 (pipeline) and every repair render go through this one
    function, so a repair differs from the draft only in the inputs
    its executor changed - never in the renderer.
    """
    from .compositor import _encode_video

    if speaker_scale != 1.0:
        frames, src_alphas = _scaled_speaker_inputs(
            frames, src_alphas, speaker_scale)
    cue_layers: dict[int, np.ndarray] = {}
    if caption_rect is not None:
        for i, cue in enumerate(cues):
            cue_layers[i] = caption_layer(cue.text, caption_rect, canvas)

    def gen():
        for i, (frame, alpha) in enumerate(zip(frames, src_alphas)):
            t = i / fps
            hook = hook_rgba if hook_visible_at(t, hook_window) else None
            out = composite_frame(frame, alpha, hook, canvas)
            for j, cue in enumerate(cues):
                if cue.start_s <= t < cue.end_s:
                    cap = cue_layers[j].astype(np.float64)
                    ca = cap[..., 3:4] / 255.0
                    out = np.clip(
                        cap[..., :3] * ca
                        + out.astype(np.float64) * (1.0 - ca),
                        0, 255).astype(np.uint8)
                    break
            yield out

    _encode_video(gen(), out_path, fps=fps, canvas=canvas,
                  audio_from=audio_from)
    return Path(out_path)


# ---------------------------------------------------------------- planning


@dataclass
class _Plan:
    menu_id: str
    target_check: str
    reason: str
    spark_call: dict[str, Any]


def _parse_planner_reply(text: str) -> tuple[str | None, str]:
    """Strict-parse the planner's JSON; (menu_id, reason) or (None, _)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, ""
    if not isinstance(data, dict):
        return None, ""
    menu_id = data.get("repair")
    if not isinstance(menu_id, str):
        return None, ""
    return menu_id.strip().upper(), str(data.get("reason", ""))


def plan_repair(
    receipt: Receipt,
    artifacts: RenderArtifacts,
    spark: SparkClient,
    *,
    ts: str = "",
) -> _Plan | RepairRecord:
    """Ask Spark for ONE repair; enforce the menu in code.

    Returns a _Plan to execute, or a RepairRecord explaining why no
    repair will run (planner failure / pick outside the menu / pick
    that targets no failed check).
    """
    failed = [c for c in receipt.checks if not c.passed]
    failed_ids = [c.check_id for c in failed]
    menu = {
        mid: {"targets": list(spec.target_checks),
              "description": spec.description}
        for mid, spec in REPAIR_MENU.items()
        if any(t in failed_ids for t in spec.target_checks)
    }
    if not menu:
        return RepairRecord(
            status="no_repair_available",
            detail=f"no menu entry targets the failed checks {failed_ids}")
    user = json.dumps({
        "failed_checks": [
            {"id": c.check_id, "measured": c.measured,
             "threshold": c.threshold}
            for c in failed
        ],
        "repair_menu": menu,
        "current_hook": artifacts.hook.text,
        "moment_transcript": artifacts.transcript_text[:800],
    }, default=str)
    try:
        result = spark.chat(
            [{"role": "system", "content": PLANNER_SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            max_tokens=PLANNER_MAX_TOKENS,
            purpose="rescue-repair-planner",
            ts=ts,
        )
    except Exception as exc:
        return RepairRecord(
            status="planner_failed",
            detail=f"spark planner call failed: {exc}"[:300])
    spark_call = {
        "model": "muse-spark-1.3",
        "purpose": "rescue-repair-planner",
        "cost_usd": result.cost_usd,
        "request_id": result.request_id,
    }
    menu_id, reason = _parse_planner_reply(result.text)
    if menu_id is None:
        return RepairRecord(
            status="no_repair_available", spark_call=spark_call,
            detail="planner reply was not strict JSON with a repair id")
    spec = REPAIR_MENU.get(menu_id)
    if spec is None:
        return RepairRecord(
            status="no_repair_available", menu_id=menu_id,
            spark_reason=reason, spark_call=spark_call,
            detail=f"planner picked {menu_id!r}, which is not in the "
                   "fixed repair menu; discarded by construction")
    targets = [t for t in spec.target_checks if t in failed_ids]
    if not targets:
        return RepairRecord(
            status="no_repair_available", menu_id=menu_id,
            spark_reason=reason, spark_call=spark_call,
            detail=f"planner picked {menu_id}, which does not target "
                   f"any failed check {failed_ids}; discarded")
    return _Plan(menu_id=menu_id, target_check=targets[0],
                 reason=reason, spark_call=spark_call)


# ---------------------------------------------------------------- executors


def _hook_window_frames(artifacts: RenderArtifacts) -> list[np.ndarray]:
    f0 = max(0, int(round(artifacts.hook.window[0] * artifacts.fps)))
    f1 = min(len(artifacts.canvas_alphas),
             max(f0 + 1, int(round(artifacts.hook.window[1]
                                   * artifacts.fps))))
    return [artifacts.canvas_alphas[i] for i in range(f0, f1)]


def _render_execution(
    artifacts: RenderArtifacts,
    *,
    frames: list[np.ndarray],
    src_alphas: list[np.ndarray],
    canvas_alphas: Sequence[np.ndarray],
    hook: HookSpec,
    cues: list[CaptionCue],
    caption_rect: tuple[int, int, int, int] | None,
    audio_from: str | Path | None,
    out_name: str,
    detail: str,
    renderer: Callable[..., Path],
    speaker_scale: float = 1.0,
) -> RepairExecution:
    hook_rgba = hook.layer(artifacts.canvas) if hook.text else None
    out_path = Path(artifacts.work_dir) / out_name
    renderer(
        frames, src_alphas,
        hook_rgba=hook_rgba, hook_window=hook.window,
        cues=cues, caption_rect=caption_rect,
        audio_from=audio_from, out_path=out_path,
        fps=artifacts.fps, canvas=artifacts.canvas,
        speaker_scale=speaker_scale,
    )
    return RepairExecution(
        render_path=out_path, hook=hook, cues=cues,
        caption_rect=caption_rect, canvas_alphas=canvas_alphas,
        detail=detail)


def _exec_trim_silence(artifacts: RenderArtifacts, receipt: Receipt,
                       renderer, **_) -> RepairExecution:
    profile = load_energy(artifacts.window_clip)
    segs = active_segments(
        profile, float(profile.times[0]),
        float(profile.times[-1]) + profile.hop_s, min_silence_s=0.0)
    removals: list[tuple[float, float]] = []
    for (a0, a1), (b0, _b1) in zip(segs, segs[1:]):
        gap = b0 - a1
        if gap > MAX_WORD_GAP_S:
            keep = min(TRIM_KEEP_SILENCE_S, gap / 2.0)
            removals.append((a1 + keep / 2.0, b0 - keep / 2.0))
    if not removals:
        raise RepairNotApplicable(
            "F5 failed but the render audio has no silence gap over "
            f"the {MAX_WORD_GAP_S}s cap to trim (the failure must be "
            "captions, onset, or the judge - none of which this "
            "repair changes)")

    duration = len(artifacts.frames) / artifacts.fps

    def removed_before(t: float) -> float:
        return sum(max(0.0, min(t, r1) - r0)
                   for r0, r1 in removals if t > r0)

    def map_time(t: float, *, is_end: bool) -> float:
        for r0, r1 in removals:
            if r0 < t < r1:
                return r0 - removed_before(r0)
        return t - removed_before(t)

    keep_idx = [
        i for i in range(len(artifacts.frames))
        if not any(r0 <= i / artifacts.fps < r1 for r0, r1 in removals)
    ]
    if not keep_idx:
        raise RepairNotApplicable("trim would remove every frame")
    frames = [artifacts.frames[i] for i in keep_idx]
    src_alphas = [artifacts.src_alphas[i] for i in keep_idx]
    canvas_alphas = artifacts.make_canvas_alphas(src_alphas)

    # Trimmed audio: decode the window audio, cut the same ranges,
    # write a WAV for the encoder to mux.
    samples = _decode_samples(artifacts.window_clip)
    sr = 16000
    kept: list[np.ndarray] = []
    cursor = 0.0
    for r0, r1 in removals + [(duration, duration)]:
        kept.append(samples[int(cursor * sr):int(r0 * sr)])
        cursor = max(cursor, r1)
    trimmed = np.concatenate(kept) if kept else samples[:0]
    audio_path = Path(artifacts.work_dir) / "trimmed-audio.wav"
    _write_wav(audio_path, trimmed, sr)

    cues: list[CaptionCue] = []
    for cue in artifacts.cues:
        s = map_time(cue.start_s, is_end=False)
        e = map_time(cue.end_s, is_end=True)
        if e - s >= 0.2:
            cues.append(CaptionCue(cue.text, round(s, 3), round(e, 3)))
    layout = plan_captions(
        [(c.start_s, c.end_s) for c in cues], canvas_alphas,
        fps=artifacts.fps, canvas=artifacts.canvas)
    hook = HookSpec(
        artifacts.hook.text,
        (artifacts.hook.window[0],
         min(artifacts.hook.window[1], len(frames) / artifacts.fps)),
        artifacts.hook.y_center_frac,
        layout_rendered=artifacts.hook.layout_rendered)
    detail = (
        "trimmed " + ", ".join(
            f"{r1 - r0:.2f}s at {r0:.2f}s" for r0, r1 in removals)
        + f"; {len(artifacts.frames)} -> {len(frames)} frames")
    return _render_execution(
        artifacts, frames=frames, src_alphas=src_alphas,
        canvas_alphas=canvas_alphas, hook=hook, cues=cues,
        caption_rect=layout.rect, audio_from=audio_path,
        out_name="draft2-trim-silence.mp4", detail=detail,
        renderer=renderer)


def _decode_samples(media_path: str | Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(media_path),
         "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        capture_output=True)
    if proc.returncode != 0:
        raise RepairNotApplicable(
            f"could not decode audio from {media_path} for the trim")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _exec_reposition_hook(artifacts: RenderArtifacts, receipt: Receipt,
                          renderer, **_) -> RepairExecution:
    window = _hook_window_frames(artifacts)
    stricter = GLYPH_OCCLUSION_CAP * REPOSITION_CAP_FRACTION
    placement = plan_hook(
        artifacts.hook.text, window, artifacts.canvas,
        cap=stricter)
    caption_rect = artifacts.caption_rect
    cues = artifacts.cues
    changed: list[str] = []
    new_hook = artifacts.hook
    if placement.y_center_frac != artifacts.hook.y_center_frac:
        new_hook = HookSpec(
            artifacts.hook.text, artifacts.hook.window,
            placement.y_center_frac, layout_rendered=True)
        changed.append(
            f"hook band {artifacts.hook.y_center_frac:.2f} -> "
            f"{placement.y_center_frac:.2f} (predicted occlusion "
            f"{placement.predicted_occlusion:.3f} at the stricter "
            f"{stricter:.3f} target)")
    f2 = next((c for c in receipt.checks if c.check_id == "F2"), None)
    caption_overlap = (
        f2.measured.get("caption_face_overlap_max_px", 0) if f2 else 0)
    if caption_overlap:
        layout = plan_captions(
            [(c.start_s, c.end_s) for c in cues],
            artifacts.canvas_alphas, fps=artifacts.fps,
            canvas=artifacts.canvas)
        if layout.rect != caption_rect and layout.feasible:
            caption_rect = layout.rect
            changed.append(f"caption block re-planned to {layout.rect}")
    if not changed:
        raise RepairNotApplicable(
            "re-planning at the stricter occlusion target lands on "
            "the same hook band and the caption block cannot move - "
            "no placement change exists for this render")
    return _render_execution(
        artifacts, frames=artifacts.frames,
        src_alphas=artifacts.src_alphas,
        canvas_alphas=artifacts.canvas_alphas, hook=new_hook,
        cues=cues, caption_rect=caption_rect,
        audio_from=artifacts.window_clip,
        out_name="draft2-reposition-hook.mp4",
        detail="; ".join(changed), renderer=renderer)


def _exec_refit_hook_ink(artifacts: RenderArtifacts, receipt: Receipt,
                         renderer, **_) -> RepairExecution:
    spec = artifacts.hook
    _glyph, bbox = spec.glyph(artifacts.canvas)

    def inside(bb) -> bool:
        return (bb[0] >= SAFE_X0 and bb[2] <= SAFE_X1
                and bb[1] >= SAFE_Y0 and bb[3] <= SAFE_Y1)

    if inside(bbox):
        raise RepairNotApplicable(
            "the hook ink is already inside the safe zone; the F3 "
            "failure is the caption block or the face centroid, "
            "which re-rendering the hook cannot change")
    window = _hook_window_frames(artifacts)
    placement = plan_hook(spec.text, window, artifacts.canvas)
    candidate = HookSpec(spec.text, spec.window, placement.y_center_frac,
                         layout_rendered=True)
    _g, cand_bbox = candidate.glyph(artifacts.canvas)
    best = candidate if inside(cand_bbox) else None
    if best is None:
        occ = occupancy_map(window)
        scored = []
        for i in range(43):
            frac = round(0.14 + 0.01 * i, 2)
            mask, bb = hook_glyph_mask(spec.text, artifacts.canvas,
                                       y_center_frac=frac)
            if inside(bb):
                scored.append((predict_occlusion(mask, occ), frac))
        if not scored:
            raise RepairNotApplicable(
                "no band centre puts this hook's ink inside the safe "
                "zone at any size the layout renderer will draw")
        scored.sort()
        best = HookSpec(spec.text, spec.window,
                        scored[0][1], layout_rendered=True)
    if (best.y_center_frac == spec.y_center_frac
            and best.layout_rendered == spec.layout_rendered):
        raise RepairNotApplicable(
            "the layout-path refit reproduces the draft's hook "
            "exactly - nothing to change")
    return _render_execution(
        artifacts, frames=artifacts.frames,
        src_alphas=artifacts.src_alphas,
        canvas_alphas=artifacts.canvas_alphas, hook=best,
        cues=artifacts.cues, caption_rect=artifacts.caption_rect,
        audio_from=artifacts.window_clip,
        out_name="draft2-refit-hook-ink.mp4",
        detail=f"hook re-rendered through the layout path at band "
               f"{best.y_center_frac:.2f}; ink bbox inside the safe "
               "zone", renderer=renderer)


def _exec_rescale_speaker(artifacts: RenderArtifacts, receipt: Receipt,
                          renderer, **_) -> RepairExecution:
    frames, src_alphas = _scaled_speaker_inputs(
        artifacts.frames, artifacts.src_alphas, RESCALE_FACTOR)
    canvas_alphas = artifacts.make_canvas_alphas(src_alphas)
    return _render_execution(
        artifacts, frames=frames, src_alphas=src_alphas,
        canvas_alphas=canvas_alphas, hook=artifacts.hook,
        cues=artifacts.cues, caption_rect=artifacts.caption_rect,
        audio_from=artifacts.window_clip,
        out_name="draft2-rescale-speaker.mp4",
        detail=f"speaker re-rendered at {RESCALE_FACTOR} scale "
               "(inputs pre-shrunk; placement unchanged)",
        renderer=renderer, speaker_scale=1.0)


def _exec_regenerate_hook(artifacts: RenderArtifacts, receipt: Receipt,
                          renderer, *, spark: SparkClient, ts: str,
                          record: RepairRecord) -> RepairExecution:
    failed = [c.check_id for c in receipt.checks if not c.passed]
    try:
        result = spark.chat(
            [{"role": "system", "content": HOOK_GEN_SYSTEM_PROMPT},
             {"role": "user", "content":
              f"Failed checks: {', '.join(failed)}. Current hook: "
              f"{artifacts.hook.text!r}. Moment transcript: "
              f"{artifacts.transcript_text[:800]}"}],
            max_tokens=HOOK_GEN_MAX_TOKENS,
            purpose="rescue-regenerate-hook",
            ts=ts,
        )
    except Exception as exc:
        raise RepairNotApplicable(
            f"hook regeneration call failed: {exc}"[:300]) from exc
    record.extra_calls.append({
        "model": "muse-spark-1.3",
        "purpose": "rescue-regenerate-hook",
        "cost_usd": result.cost_usd,
        "request_id": result.request_id,
    })
    line = ""
    for raw_line in result.text.splitlines():
        if raw_line.strip():
            line = raw_line.strip().strip("\"'").upper()
            break
    words = line.split()
    if not (2 <= len(words) <= 9) or len(_content_words(line)) < 3:
        raise RepairNotApplicable(
            f"regenerated hook {line!r} violates the hook rules "
            "(word count / content words); discarded")
    if line == artifacts.hook.text.strip().upper():
        raise RepairNotApplicable(
            "Spark regenerated the same hook line; nothing to change")
    window = _hook_window_frames(artifacts)
    placement = plan_hook(line, window, artifacts.canvas)
    new_hook = HookSpec(line, artifacts.hook.window,
                        placement.y_center_frac, layout_rendered=True)
    return _render_execution(
        artifacts, frames=artifacts.frames,
        src_alphas=artifacts.src_alphas,
        canvas_alphas=artifacts.canvas_alphas, hook=new_hook,
        cues=artifacts.cues, caption_rect=artifacts.caption_rect,
        audio_from=artifacts.window_clip,
        out_name="draft2-regenerate-hook.mp4",
        detail=f"hook {artifacts.hook.text!r} -> {line!r} at band "
               f"{placement.y_center_frac:.2f}",
        renderer=renderer)


_EXECUTORS: dict[str, Callable[..., RepairExecution]] = {
    TRIM_SILENCE: _exec_trim_silence,
    REPOSITION_HOOK: _exec_reposition_hook,
    REFIT_HOOK_INK: _exec_refit_hook_ink,
    RESCALE_SPEAKER: _exec_rescale_speaker,
    REGENERATE_HOOK_TEXT: _exec_regenerate_hook,
}


# ---------------------------------------------------------------- the loop


def rescue(
    receipt: Receipt,
    artifacts: RenderArtifacts,
    *,
    spark: SparkClient,
    verify_fn: Callable[[RepairExecution], Receipt],
    renderer: Callable[..., Path] = render_composite_with_captions,
    ts: str = "",
) -> ChainedReceipt:
    """Run the bounded rescue for one failed receipt.

    Structure is the cap: this function plans at most once, executes
    at most once, and verifies at most once. ``verify_fn`` is the
    caller's closure that re-runs the full verifier (draft=2) on a
    RepairExecution - the loop never touches SAM, and segmentation
    is never re-bought.
    """
    if not receipt.failed:
        return ChainedReceipt(
            artifact=receipt.artifact, final_verdict="PASSED_FIRST_DRAFT",
            drafts=[receipt],
            repair=RepairRecord(
                status="not_needed",
                detail="draft 1 passed all checks; the loop never runs"),
            created=ts)
    plan = plan_repair(receipt, artifacts, spark, ts=ts)
    if isinstance(plan, RepairRecord):
        return ChainedReceipt(
            artifact=receipt.artifact, final_verdict="FAIL",
            drafts=[receipt], repair=plan, created=ts)
    record = RepairRecord(
        status="executed", menu_id=plan.menu_id,
        target_check=plan.target_check, spark_reason=plan.reason,
        spark_call=plan.spark_call)
    executor = _EXECUTORS[plan.menu_id]
    try:
        execution = executor(
            artifacts, receipt, renderer,
            spark=spark, ts=ts, record=record)
    except RepairNotApplicable as exc:
        record.status = "not_applicable"
        record.detail = str(exc)
        return ChainedReceipt(
            artifact=receipt.artifact, final_verdict="FAIL",
            drafts=[receipt], repair=record, created=ts)
    record.detail = execution.detail
    draft2 = verify_fn(execution)
    final = "PASSED_AFTER_REPAIR" if not draft2.failed else "FAIL"
    return ChainedReceipt(
        artifact=str(execution.render_path), final_verdict=final,
        drafts=[receipt, draft2], repair=record, created=ts)
