"""Geometry bundles: the render-time geometry record for reverify.

The render-time verifier measures geometry from the pipeline's
in-memory objects (SAM alphas at source resolution, the exact
stage-zoom / reframe / emphasis transforms, the layout placements).
An offline re-verifier rebuilding those inputs from evidence files
alone has to RECOMPUTE them, and the recomputation is deterministic
but not identical — measured fields diverge beyond tolerance
(reverify-geometry-closure-spec-2026-11-03.md).

The closure: at verify time the pipeline writes
``geometry-bundle.json`` into the Short's evidence directory, one
entry per draft whose geometry differs (draft 1 always; draft 2
when the repair touches geometry — audio-only repairs reuse draft
1's bundle). Every value in a bundle is COPIED from the objects
the verifier actually measured with, never recomputed. The
offline tool then builds VerifyInputs from the bundle and replays
only the deterministic transforms the bundle names (the same
replay discipline the tool already uses for TRIM_SILENCE).

Each draft bundle carries an ``inputs_hash``: a hash of the
bundle's geometry sections folded with the receipt draft's own
``inputs_sha256`` block, so a bundle that does not belong to the
receipt it sits next to is detected, not trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .layout_law import StageZoom
from .layout_states import (
    LayoutDecl,
    LayoutPlan,
    SpeakerPlacement,
    build_track,
)
from .verifier import HookSpec, _sha256_json

BUNDLE_FILENAME = "geometry-bundle.json"
BUNDLE_VERSION = 1

# Repairs that change the geometry a draft is verified with (the
# spec's list). Draft 2 gets its own bundle entry when one of
# these executed; audio-only repairs reuse draft 1's bundle.
GEOMETRY_REPAIRS = frozenset({
    "REPOSITION_HOOK",
    "REFIT_HOOK_INK",
    "REGENERATE_HOOK_TEXT",
    "RESCALE_SPEAKER",
    "REFRAME_SPEAKER",
    "ENGAGE_HOOK_SLIDE",
    "DROP_GRAPHICS",
})

# TRIM_SILENCE changes the frame sequence (a geometry input), so
# it also earns a draft-2 entry; its alpha divergence is replayed
# by the re-verifier's existing trim replay, not recomputed here.
DRAFT2_BUNDLE_REPAIRS = GEOMETRY_REPAIRS | {"TRIM_SILENCE"}


# ---------------------------------------------------------------- hooks


def hook_to_dict(hook: HookSpec) -> dict[str, Any]:
    """The bundle record of the HookSpec the verifier consumed."""
    return {
        "text": hook.text,
        "window": [float(hook.window[0]), float(hook.window[1])],
        "y_center_frac": float(hook.y_center_frac),
        "layout_rendered": bool(hook.layout_rendered),
        "accent_index": hook.accent_index,
        "text_size": int(hook.text_size),
        "slide": ([float(hook.slide[0]), float(hook.slide[1])]
                  if hook.slide is not None else None),
        "variant": hook.variant,
    }


def hook_from_dict(d: Mapping[str, Any]) -> HookSpec:
    """Rebuild the recorded HookSpec from its bundle record."""
    slide = d.get("slide")
    return HookSpec(
        text=d["text"],
        window=(float(d["window"][0]), float(d["window"][1])),
        y_center_frac=float(d["y_center_frac"]),
        layout_rendered=bool(d.get("layout_rendered", False)),
        slide=(tuple(float(v) for v in slide)
               if slide is not None else None),
        variant=d.get("variant", "stage"),
        accent_index=d.get("accent_index"),
        text_size=int(d.get("text_size", 170)),
    )


# ---------------------------------------------------------------- recipes


def normalize_speaker_forward(
        decls: Sequence[Mapping[str, Any]] | None,
        ) -> list[dict[str, Any]] | None:
    """Normalise speaker-forward declarations to the bundle shape.

    The pipeline's declarations carry the span as a
    ``{"start_s", "end_s"}`` dict; the bundle stores it as a
    ``[start_s, end_s]`` pair. Only the fields the replay
    cross-checks are kept.
    """
    if not decls:
        return None
    out: list[dict[str, Any]] = []
    for decl in decls:
        span = decl["span"]
        if isinstance(span, Mapping):
            pair = [float(span["start_s"]), float(span["end_s"])]
        else:
            pair = [float(span[0]), float(span[1])]
        entry: dict[str, Any] = {
            "span": pair,
            "scale": decl.get("scale"),
            "placed_centroid": decl.get("placed_centroid"),
            "placed_head_top_min": decl.get("placed_head_top_min"),
        }
        # The applied transform itself, when the source carries
        # it (the pipeline's bundle capture does; receipt-era
        # declarations carry it only inside the planner's note,
        # which is preserved here for the replayer to recover).
        if decl.get("crop") is not None:
            entry["crop"] = list(decl["crop"])
        if decl.get("pad_bottom_px") is not None:
            entry["pad_bottom_px"] = int(decl["pad_bottom_px"])
        if decl.get("note"):
            entry["note"] = decl["note"]
        out.append(entry)
    return out


def alpha_recipe_dict(
        *,
        src_frame_size: Sequence[int] | None,
        stage_geometry: Mapping[str, Any] | None,
        speaker_forward: Sequence[Mapping[str, Any]] | None,
        ) -> dict[str, Any]:
    """The alpha-transform recipe the verifier's alphas came from.

    ``stage_geometry`` is the pipeline's assembled declaration
    (the StageZoom declaration plus the reframe/effect pads) —
    copied, not recomputed. The effect pad is the TOTAL transform
    the pipeline applied after the zoom (plan_hook_effect's pad,
    applied via apply_speaker_reframe); the construction reframe's
    own pad is a planning input to it, never applied separately.
    """
    sg = stage_geometry or {}
    applied = bool(sg.get("feasible") and not sg.get("identity")
                   and sg.get("crop"))
    return {
        "src_frame_size": (list(src_frame_size)
                           if src_frame_size is not None else None),
        "stage_zoom": {
            "applied": applied,
            "crop": sg.get("crop"),
            "pad_bottom_px": int(sg.get("pad_bottom_px") or 0),
        },
        "effect_pad_bottom_px": int(sg.get("effect_pad_bottom_px")
                                    or 0),
        "speaker_forward": normalize_speaker_forward(speaker_forward),
    }


def stage_zoom_from_recipe(recipe: Mapping[str, Any]) -> StageZoom | None:
    """The StageZoom to replay from a recipe, or None if unapplied.

    apply_stage_zoom reads only ``crop`` and ``pad_bottom_px``
    (plus the feasible/identity gates), so those are the fields
    reconstructed; the planning statistics are not needed to
    replay a transform that was already solved at render time.
    """
    zoom = recipe.get("stage_zoom") or {}
    if not zoom.get("applied") or not zoom.get("crop"):
        return None
    return StageZoom(
        feasible=True, identity=False,
        crop=tuple(int(v) for v in zoom["crop"]),
        pad_bottom_px=int(zoom.get("pad_bottom_px") or 0),
        scale=1.0, placed_scale=0.0, placed_centroid=None,
        placed_head_top_min=None, height_frac=None,
        note="geometry bundle replay")


# ---------------------------------------------------------------- drafts


def _normalized(obj: Any) -> Any:
    """JSON round-trip, so a hash computed at write time is over
    exactly the structure a reader loads back from disk."""
    return json.loads(json.dumps(obj))


def inputs_hash_for(core: Mapping[str, Any],
                    inputs_sha256: Mapping[str, Any] | None) -> str:
    """The bundle's binding hash: geometry sections folded with
    the receipt draft's recorded input hashes."""
    return _sha256_json({
        "geometry": core,
        "inputs_sha256": dict(inputs_sha256 or {}),
    })


def draft_bundle_dict(
        *,
        stage_geometry: Mapping[str, Any] | None,
        alpha_recipe: Mapping[str, Any],
        hook: Mapping[str, Any],
        caption_rect: Sequence[int] | None,
        layout_spans: Sequence[Mapping[str, Any]] | None,
        hook_slide_px_s: float,
        inputs_sha256: Mapping[str, Any] | None,
        ) -> dict[str, Any]:
    """Assemble one draft's bundle (with its ``inputs_hash``)."""
    core: dict[str, Any] = {
        "stage_geometry": dict(stage_geometry)
        if stage_geometry is not None else None,
        "alpha_recipe": dict(alpha_recipe),
        "hook": dict(hook),
        "caption_rect": (list(caption_rect)
                         if caption_rect is not None else None),
        "layout_spans": ([dict(d) for d in layout_spans]
                         if layout_spans is not None else None),
        "hook_slide_px_s": float(hook_slide_px_s),
    }
    core = _normalized(core)
    bundle = dict(core)
    bundle["inputs_hash"] = inputs_hash_for(core, inputs_sha256)
    return bundle


def check_inputs_hash(bundle: Mapping[str, Any],
                      inputs_sha256: Mapping[str, Any] | None) -> bool:
    """True when the bundle's stored hash matches this receipt
    draft's recorded input hashes (i.e. the bundle belongs to
    the receipt it was found with)."""
    stored = bundle.get("inputs_hash")
    if not stored:
        return False
    core = {k: v for k, v in bundle.items()
            if k not in ("inputs_hash", "backfilled")}
    return inputs_hash_for(core, inputs_sha256) == stored


# ---------------------------------------------------------------- file io


def write_bundle(short_dir: str | Path,
                 drafts: Mapping[int, Mapping[str, Any]]) -> Path:
    """Write geometry-bundle.json for a Short's drafts."""
    path = Path(short_dir) / BUNDLE_FILENAME
    payload = {
        "version": BUNDLE_VERSION,
        "drafts": {str(k): v for k, v in sorted(drafts.items())},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_bundle_file(evidence_dir: str | Path) -> dict[int, dict]:
    """Load a bundle file's per-draft entries ({} when absent or
    unreadable — the caller decides what absence means)."""
    path = Path(evidence_dir) / BUNDLE_FILENAME
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[int, dict] = {}
    for key, value in (raw.get("drafts") or {}).items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------- layout


def placements_from_declarations(
        layout_spans: Sequence[Mapping[str, Any]] | None,
        frame_shape: tuple[int, int],
        n_frames: int,
        fps: float,
        canvas: tuple[int, int],
        ) -> list | None:
    """Rebuild the per-frame layout placement track from the
    recorded declarations.

    The track is a pure function of the span declarations
    (layout_states.build_track): each feasible non-Stage span's
    resting placement is copied from its recorded
    ``speaker_placement`` and the boundaries are eased exactly as
    the render eased them. Returns None when no feasible
    non-Stage span is declared (the historic Stage mapping).

    Raises ValueError when a span carries a camera move: a
    punch-in's anchor (the plan's source centroid) is not part of
    the declaration, so that track cannot be reproduced from the
    record alone and must not be approximated.
    """
    decls: list[LayoutDecl] = []
    plans: list[LayoutPlan] = []
    for d in layout_spans or []:
        if not isinstance(d, Mapping):
            continue
        if d.get("layout") in (None, "stage"):
            continue
        if d.get("feasible") is False:
            continue
        span = d.get("span")
        placement = d.get("speaker_placement")
        if not (isinstance(span, (list, tuple)) and len(span) == 2
                and isinstance(placement, Mapping)):
            continue
        if d.get("camera") is not None:
            raise ValueError(
                "layout span carries a camera move; the placement "
                "track cannot be reproduced from declarations")
        crop = placement.get("crop")
        origin = placement.get("origin") or (0.0, 0.0)
        plans.append(LayoutPlan(
            state=d["layout"],
            placement=SpeakerPlacement(
                crop=(tuple(float(v) for v in crop)
                      if crop is not None else None),
                scale=float(placement.get("scale", 1.0)),
                ox=float(origin[0]), oy=float(origin[1]),
                alpha_scale=float(placement.get("alpha_scale", 1.0))),
            feasible=True,
            placed_centroid=(tuple(float(v)
                                   for v in d["placed_centroid"])
                             if d.get("placed_centroid") else None),
            head_top_min=d.get("head_top_min"),
            graphic_zone=(tuple(int(v) for v in d["graphic_zone"])
                          if d.get("graphic_zone") else None),
            caption_rect=(tuple(int(v) for v in d["caption_rect"])
                          if d.get("caption_rect") else None),
        ))
        decls.append(LayoutDecl(
            start_s=float(span[0]), end_s=float(span[1]),
            state=d["layout"], camera=None))
    if not decls:
        return None
    return build_track(decls, plans, frame_shape, n_frames, fps,
                       canvas)
