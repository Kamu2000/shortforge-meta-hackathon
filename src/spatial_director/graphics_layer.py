"""The context graphics layer: timed graphics on the backdrop plane.

Kamlesh's direction (2026-10-30): the treated backdrop leaves a
lot of calm, empty space behind the speaker - use it. When the
speaker names a concrete thing, a number, a comparison or a quote,
a graphic illustrating THAT point appears on the backdrop plane at
the moment it is spoken, and leaves when the moment passes. This is
the AI Producer feature set (motion graphics / B-roll) adapted to
the Director's format: no full-frame cutaway (the speaker is the
retention), no decoration without a spoken referent.

The layer's discipline mirrors the rest of the build:

- **Selection is a model's, validation is code's.** Spark 1.3 reads
  the Short's caption phrases and proposes graphic spans under a
  strict bar (see the system prompt): semantic selection - concrete
  named things, numbers/statistics, comparisons, visual concepts -
  never keyword decoration. Every proposal is then validated
  deterministically: the source quote must appear verbatim (as a
  word sequence) in the transcript; any number on a card must be a
  number the transcript actually states (compared as VALUES, so a
  card may render the spoken "one million" as "$1M" but may not
  invent "$5M"); image prompts carry no digits at all; named
  PEOPLE get typographic name cards only - an image prompt may
  never depict a face or likeness. Spacing discipline is also
  code (Kamlesh's rulings, 2026-10-30): NO count cap and NO
  screen-time ceiling - the number of graphics is whatever
  genuinely clears the semantic bar, each justified by its source
  quote. Timing freedom is also his (rulings reversed
  2026-10-31): NO minimum gap between graphics and NO
  hook-window ban - spans may start at 0.0s and run back-to-back
  or overlapping in time, for better engagement. What carries
  the discipline instead is collision: placement keeps every
  graphic clear of the face (+margin), the caption zone, the
  hook glyph zone while the hook is up, and every other placed
  graphic; a span with no collision-free placement is dropped.
  Both collision rules are ALSO verifier rules (F8), so a render
  whose graphics touch what they must not fails certification,
  not just planning. Span durations are content-driven
  (span-law amendment, Kamlesh 2026-11-01, extension build): a
  1.0s floor, NO fixed ceiling - a span is snapped to its anchor
  passage's DSP phrase boundaries (-0.1s / +0.3s) and may not
  outlive that passage; text cards additionally carry a
  legibility floor derived from their own words. An empty list
  is a valid, common outcome - most speech earns no graphic.
- **Text never goes through the image model.** Cards (headlines,
  numbers, quotes, comparisons) are rendered by PIL in the brand
  palette - deterministic pixels the verifier can reproduce. Muse
  Image is used only for concrete visual objects/concepts, with a
  no-text suffix appended to the prompt before it is sent.
- **Animation is deterministic and in-box.** Cards animate inside
  their fixed placement box (the box never moves, so placement
  exclusions and F8 are unaffected): number cards count up to the
  stated value across the span, comparison cards build their two
  bars keyed to when each item is spoken, quote/name cards reveal
  line by line when the span allows. Images get a subtle Ken Burns
  drift (zoom 1.00 -> 1.12 across the span). Every state is a pure
  function of time - ``GraphicSpec.layer_at(t)`` reproduces the
  exact pixels the compositor drew - and the motion type
  (count_up / build_two_state / reveal / ken_burns / static) is
  recorded on the plan and on the receipt's graphics section.
- **Placement is measured, not guessed.** From the cached canvas
  alphas, per span: the face region (plus a margin), the caption
  zone, the hook glyph zone during the hook window, and any pixel
  the speaker occupies more than rarely are HARD exclusions. A span
  with no legal placement is dropped, with the reason on the
  record - never squeezed in.

Nothing here runs by default: the pipeline wires this module in
only behind its --graphics flag.

Written fresh in-repo on 2026-10-30.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .caption_align import CaptionCue
from .compositor import DEFAULT_CANVAS, _load_font
from .image_client import ImageClient
from .layout_law import (
    CAPTION_BOTTOM_MARGIN,
    face_pixels,
    initial_caption_rect,
)
from .layout_states import (
    LAYOUT_STATES,
    LayoutDecl,
    SpanStats,
    build_track,
    plan_cutaway,
    plan_spotlight,
    plan_spotlight_below,
    plan_split,
    validate_camera,
)
from .spark_client import SparkClient, SparkClientError

# ---------------------------------------------------------------- constants

MIN_SPAN_S = 1.0
# Span law (amendment, Kamlesh 2026-11-01, extension build only -
# the filed baseline's receipts were issued under 2.5-6.0s and do
# not change): the floor is 1.0s and there is NO fixed ceiling.
# A span's duration is set by the speech it illustrates: after
# the timing snap (below) a span may not outlive its anchor
# passage (+0.3s tail). Text cards get a legibility floor from
# their own text instead of a fixed bound.
SPAN_MIN_S = MIN_SPAN_S
SNAP_START_PAD_S = 0.1       # span starts this far before its anchor
SNAP_END_PAD_S = 0.3         # ... and ends this far after it
LEGIBILITY_S_PER_WORD = 0.35  # of headline+facts, for card spans
LEGIBILITY_MIN_S = 1.5
# Entry/motion travel is capped at this fraction of the span, so
# even a 1.0s span keeps a measurable presence core (F8's
# presence legs are unchanged).
PATH_TRAVEL_MAX_FRAC = 0.25
# Timing freedom (Kamlesh's ruling, 2026-10-31, reversing
# 2026-10-30): there is NO minimum gap between graphics and NO
# hook-window ban - spans may start at 0.0s and overlap in time.
# Collision, not spacing, is the discipline: see placement + F8.
REVEAL_MIN_SPAN_S = 3.5      # shorter quote/name cards stay static
# Path motions (block 2, 2026-10-31): a graphic may TRAVEL to its
# placed box over this long at the span's start - glide/rise/fall
# from off-canvas, or an approach scale settle. The path is code-
# computed from the placed box, validated against the same
# forbidden zones as static placement, declared on the plan and
# the receipt, and measured by F8.
PATH_TRAVEL_S = 0.9
PATH_KINDS = ("glide_in", "rise_in", "fall_in", "approach")
# Exit transitions (extension Stage 4, 2026-11-01): the mirror of
# the entry paths. A graphic may LEAVE by the edge it entered
# (glide_out / rise_out / fall_out) or shrink-settle away
# (recede, the approach mirror). The exit travel occupies the
# LAST path_travel_s of the span - the same 25%-of-span cap - so
# entry core and exit never meet and the presence core between
# them is at least half the span.
EXIT_PATH_KINDS = ("glide_out", "rise_out", "fall_out", "recede")
PLANNER_MAX_TOKENS = 8000  # Spark reasons first; 2500 starved on the
# preview prompt and 4000 starved on pipeline cue lists (empty reply
# on all 3 Holmes graphics Shorts, 2026-10-30). One retry at a
# doubled budget follows the Wonder/Spark-Open precedent (cap 16000).
PLANNER_MAX_TOKENS_CAP = 16000

FACE_MARGIN_PX = 28          # hard exclusion halo around the face
FREE_OCCUPANCY_MAX = 0.04    # speaker present >4% of span frames = occupied
CANVAS_MARGIN_PX = 16        # graphics keep this far from the canvas edges
PLACED_GAP_PX = 10           # graphics never crowd each other
GRID_STEP_PX = 8             # placement search resolution

FORGE_BLUE = (46, 124, 246)
VIRAL_PINK = (255, 61, 126)
NEAR_BLACK = (11, 11, 12)
CARD_WIDTH = 470
CARD_PAD = 26
CARD_RADIUS = 26
COMPARE_CARD_WIDTH = 560
IMAGE_BOX = (400, 500)       # portrait box for generated images
KEN_BURNS_MAX_ZOOM = 1.12    # across the whole span, never more
# (strengthened 1.06 -> 1.12 in extension Stage 4: at 1.06 the
# drift read as near-static on the gate renders; 1.12 with the
# widened pan below is still slow - it crosses its travel over
# the whole span, and the backdrop legs measure the same
# deterministic state function either way.)
IMAGE_PROMPT_SUFFIX = " - no text, no letters, no numbers, no words"
# An image may never depict a person: named people are name cards.
# The planner prompt carries the rule; this blocklist is the
# deterministic backstop for the obvious phrasings.
IMAGE_PROMPT_BLOCKLIST = (
    "portrait", "headshot", "likeness", "face of", "facial",
    "a person's", "his face", "her face",
)

PLANNER_SYSTEM_PROMPT = (
    "You select context graphics for a short vertical video "
    "clipped from a talk. A graphic appears on the empty backdrop "
    "behind the speaker at the moment its topic is spoken. Select "
    "SEMANTICALLY, not by keyword: a graphic is earned only by a "
    "concrete named thing (product, gear, place, work, team), a "
    "number or statistic, a comparison between two stated values, "
    "or a visual concept the words genuinely picture. Reply with "
    "strict JSON only: {\"graphics\": [{\"start_s\": number, "
    "\"end_s\": number, \"kind\": \"card\", \"image\", "
    "\"backdrop\" or \"speaker_forward\", "
    "\"headline\": \"...\", \"facts\": \"...\", \"prompt\": \"...\", "
    "\"compare\": [{\"label\": \"...\", \"value\": number}, "
    "{\"label\": \"...\", \"value\": number}] or null, "
    "\"layout\": \"stage\", \"spotlight\", \"split\" or "
    "\"cutaway\" (optional), "
    "\"source_quote\": \"...\"}]}. Rules: there is NO cap on the "
    "number of graphics - the count is driven purely by the "
    "content, as many as genuinely clear the bar and no more; "
    "when in doubt, leave it out. An "
    "empty list is a valid and common answer - most speech earns "
    "no graphic. Optionally, a card or image may carry \"path\": "
    "one of \"glide_in\" (slides in from the nearer side edge), "
    "\"rise_in\" (rises from below), \"fall_in\" (drops from "
    "above) or \"approach\" (scales up and settles into place) - "
    "use a path when the moment lands with emphasis; omit it for "
    "a quiet in-place appearance. A graphic may likewise carry "
    "\"exit_path\": one of \"glide_out\", \"rise_out\", "
    "\"fall_out\" or \"recede\" for a travelled exit in the "
    "span's last moments; omit it for a plain fade-out. A path whose travel would cross "
    "the speaker, the captions or another graphic is refused by "
    "the placer and the graphic appears in place instead. kind \"card\" is for numbers, quotes, names and "
    "facts: headline at most 6 words, facts at most 12 words, both "
    "copied from the transcript - never invent a number, name or "
    "claim. For a comparison between two stated values, use kind "
    "\"card\" with \"compare\": the two items' labels (transcript "
    "words) and their stated values. A series of stated values "
    "may instead carry \"chart\": {\"type\": \"bar\" or \"line\", "
    "\"items\": [{\"label\": ..., \"value\": ...}]} - 2 to 6 "
    "items, every value copied verbatim from the transcript; "
    "the chart draws only those values, so an unstated value "
    "gets the span rejected. An \"image\" span may carry "
    "\"collage\": true when the moment genuinely shows several "
    "stills at once (a grid of 2-4 related images). A named PERSON is always a "
    "card (headline = their name, facts = their role in the "
    "speaker's words) - never an image: no AI faces, no likenesses. "
    "kind \"image\" is only for a concrete visual object or "
    "concept that is not a person: prompt describes the picture in "
    "one sentence with NO text, NO numbers, NO letters, NO people. "
    "kind \"backdrop\" is for when the speech pictures a SCENE or "
    "place rather than an object (a stadium tunnel at night, a "
    "harbour at dusk): the generated image becomes the backdrop "
    "itself for that span, darkened behind the speaker - prompt "
    "follows the image rules exactly (one sentence, NO text, NO "
    "numbers, NO people). Backdrop is for scenes and concepts, "
    "NEVER for numbers, names or comparisons - those stay cards. "
    "Backdrop is scene-setting only: never place one under the "
    "Short's hook line or under its single most important claim. "
    "A card or image may also carry \"layout\", the framing the "
    "span asks for, assigned by the graphic's ROLE in the span: "
    "\"spotlight\" when the graphic IS the point of the span - a "
    "hero number, a comparison, a key still: it takes centre "
    "stage at majority scale and the speaker shrinks to a "
    "picture-in-picture corner, and it is the span's only "
    "graphic. \"split\" for sustained parallel content: the "
    "graphic holds the top of the frame while the speaker keeps "
    "talking in the bottom band. \"cutaway\" for the single "
    "strongest visual claim of the Short, shown full-frame with "
    "the speaker absent - keep it short and use it at most once. "
    "Omit \"layout\" (the default, \"stage\") for ordinary "
    "graphics that sit in free space beside the speaker, for "
    "tiny or secondary graphics, and whenever the speaker's "
    "presence matters more than the graphic. Backdrop and "
    "speaker_forward spans never carry a layout - they are "
    "framing already. "
    "kind \"speaker_forward\" marks a direct-address or payoff "
    "moment: the speaker is framed larger and nearer the centre "
    "for that span. It carries no headline, facts or prompt - "
    "only the timing and the source_quote. ONE HERO AT ANY "
    "INSTANT: "
    "a card/image span, a backdrop span and a speaker_forward "
    "span must never overlap in time, and a span carrying a "
    "layout state is a hero claim too - it must not overlap any "
    "other span. When hero claims overlap, "
    "the later-starting one is dropped by the pipeline, so "
    "choose the single strongest treatment for each moment. "
    "Heroes may chain back-to-back without limit. "
    "Each span is the seconds where the topic is spoken (use the "
    "caption timings given). Span lengths are content-driven: "
    "at least 1 second, with NO fixed maximum - in code each "
    "span is snapped to its anchor phrase's boundaries and may "
    "not outlive the passage it illustrates, so anchor every "
    "graphic to the full passage it illustrates, not just its "
    "first phrase. A card's words must stay readable inside "
    "its span (roughly a third of a second per word of headline "
    "and facts): a wordy card needs a longer anchor passage or "
    "fewer words. Graphics "
    "may appear from the very start of the Short (0s), "
    "back-to-back, or overlapping in time - no gap between "
    "graphics is required. Placement keeps each graphic clear of "
    "the speaker's face, the captions, the hook text and the "
    "other graphics, and drops any graphic that cannot be placed "
    "without colliding - so propose wherever the content earns "
    "one, at whatever density the content supports. "
    "source_quote is the exact transcript words the graphic "
    "illustrates."
)


# ---------------------------------------------------------------- numbers

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
_NUMBER_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_SUFFIX_MULT = {"k": 1e3, "m": 1e6, "b": 1e9}
_DIGIT_RE = re.compile(r"\$?(\d[\d,]*(?:\.\d+)?)\s?([kmb])?", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?\s?[kmbKMB]?%?")


def numeric_values(text: str) -> set[float]:
    """Every numeric VALUE a text states, digits or words.

    "$1M", "1 million" and "one million" all canonicalise to
    1e6 - the card guard compares values, so reformatting a spoken
    number is allowed and inventing one is not.
    """
    values: set[float] = set()
    for m in _DIGIT_RE.finditer(text):
        v = float(m.group(1).replace(",", ""))
        mult = _SUFFIX_MULT.get((m.group(2) or "").lower(), 1.0)
        values.add(round(v * mult, 6))
    words = _word_seq(text)
    i = 0
    while i < len(words):
        if words[i] in _NUMBER_WORDS or words[i] in _NUMBER_SCALES \
                or words[i] == "hundred":
            total, current, seen = 0.0, 0.0, False
            j = i
            while j < len(words) and (
                    words[j] in _NUMBER_WORDS
                    or words[j] in _NUMBER_SCALES
                    or words[j] in ("hundred", "and")):
                w = words[j]
                if w in _NUMBER_WORDS:
                    current += _NUMBER_WORDS[w]
                    seen = True
                elif w == "hundred":
                    current = max(current, 1.0) * 100.0
                    seen = True
                elif w in _NUMBER_SCALES:
                    total += max(current, 1.0) * _NUMBER_SCALES[w]
                    current = 0.0
                    seen = True
                j += 1
            if seen and total + current > 0:
                values.add(round(total + current, 6))
            i = max(j, i + 1)
        else:
            i += 1
    return values


def _parse_number_token(token: str) -> tuple[float, dict[str, Any]]:
    """(value, format) of one numeric token like '$1.5M' or '10'."""
    m = re.fullmatch(
        r"(\$?)(\d[\d,]*(?:\.\d+)?)\s?([kmbKMB]?)(%?)", token.strip())
    if m is None:
        raise ValueError(f"not a number token: {token!r}")
    prefix, digits, suffix, percent = m.groups()
    mult = _SUFFIX_MULT.get(suffix.lower(), 1.0)
    value = float(digits.replace(",", "")) * mult
    fmt = {
        "prefix": prefix, "suffix": suffix, "percent": percent,
        "decimals": len(digits.split(".")[1]) if "." in digits else 0,
        "commas": "," in digits, "mult": mult,
    }
    return value, fmt


def _format_count(current: float, fmt: dict[str, Any],
                  template: str) -> str:
    """Format a counting value in the template token's own style."""
    if fmt["mult"] > 1:
        scaled = current / fmt["mult"]
        if fmt["decimals"]:
            body = f"{scaled:.{fmt['decimals']}f}"
        elif scaled < 10:
            body = f"{scaled:.2f}".rstrip("0").rstrip(".")
        else:
            body = f"{scaled:.0f}"
    elif fmt["decimals"]:
        body = f"{current:.{fmt['decimals']}f}"
    else:
        body = f"{round(current):,}" if fmt["commas"] \
            else f"{round(current):.0f}"
    return f"{fmt['prefix']}{body}{fmt['suffix']}{fmt['percent']}"


def _ease_out_cubic(p: float) -> float:
    return 1.0 - (1.0 - p) ** 3


# ---------------------------------------------------------------- specs


@dataclass
class GraphicSpan:
    """One planner-proposed graphic span (pre-placement)."""

    start_s: float
    end_s: float
    kind: str  # "card" | "image" | "backdrop" | "speaker_forward"
    headline: str = ""
    facts: str = ""
    prompt: str = ""
    source_quote: str = ""
    compare: list[dict[str, Any]] | None = None
    motion: str = "static"
    # Block 2: a path motion the planner selected for this graphic
    # (one of PATH_KINDS) or None for an in-place appearance.
    path: str | None = None
    # Stage 4: the exit transition (one of EXIT_PATH_KINDS) or
    # None for an in-place disappearance (the historic fade only).
    exit_path: str | None = None
    # Tier B (Stage 5): a code-drawn chart on a card span -
    # {"type": "bar"|"line", "items": [{"label", "value"}, ...]}
    # with EVERY value verbatim from the transcript (the numeric
    # guard, enforced at parse time like compare's).
    chart: dict[str, Any] | None = None
    # Tier B (Stage 5): this image span is a COLLAGE - its
    # content is a grid of 2-4 stills the CALLER supplies from
    # its cache (prepare_collage); the planner never invents the
    # stills, and build_graphics drops the span on the record
    # when fewer than 2 cached stills are available.
    collage: bool = False
    # Extension (2026-11-01): the layout state this span asks the
    # compositor for (layout_states.LAYOUT_STATES; "stage" = the
    # shipped behaviour) and an optional camera move dict
    # ({type: punch_in|drift, from, to}). Card/image kinds only -
    # heroes (backdrop, speaker_forward) ARE framing already.
    layout: str = "stage"
    camera: dict[str, Any] | None = None
    # Timing snap (E2): the anchor passage's quality, set by
    # snap_span_to_anchor - "high" when the source quote was
    # found verbatim in a clean caption passage, "low" when the
    # anchor is noisy (garbled turn text) or had to be inferred
    # from timing overlap. The receipt flags the low ones.
    anchor_quality: str = "high"
    # Fix phase (2026-11-02): set by the deterministic layout
    # assist when IT assigned this span's layout ({"span": [...],
    # "requested": "spotlight"}); None for planner-declared
    # layouts. Carried onto the placed spec so plan_layout can
    # put the assist record on the receipt declaration.
    layout_assist: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start_s": self.start_s, "end_s": self.end_s,
            "kind": self.kind, "headline": self.headline,
            "facts": self.facts, "prompt": self.prompt,
            "source_quote": self.source_quote,
            "compare": self.compare, "motion": self.motion,
            "path": self.path, "exit_path": self.exit_path,
            "chart": self.chart, "collage": self.collage,
        }
        if self.layout != "stage":
            out["layout"] = self.layout
        if self.layout_assist is not None:
            out["layout_assist"] = self.layout_assist
        if self.camera is not None:
            out["camera"] = self.camera
        if self.anchor_quality != "high":
            out["anchor_quality"] = self.anchor_quality
        return out


# layer_at signature: (t_s, start_s, end_s) -> canvas RGBA layer.
LayerFn = Callable[[float, float, float], np.ndarray]


@dataclass
class GraphicSpec:
    """One placed graphic: everything the compositor renders and
    the verifier measures. ``layer_rgba`` is the RESTING state
    (the content at progress 1 in the placed box - the layer at
    the span's end, unless an exit transition has carried it off
    by then); ``layer_at(t)`` reproduces the exact
    animated state at any time inside the span - animation is a
    pure function of time, rendered from the same code the
    compositor calls, so F8 measures real pixels against the same
    deterministic states. ``cost_usd`` is what producing the
    graphic cost (the image call for kind "image", 0.0 for a PIL
    card)."""

    kind: str
    start_s: float
    end_s: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 exclusive
    layer_rgba: np.ndarray
    source_quote: str = ""
    prompt: str = ""
    headline: str = ""
    facts: str = ""
    compare: list[dict[str, Any]] | None = None
    # Tier B (Stage 5): the chart declaration this card renders
    # (values verbatim, guarded at plan time) and whether an
    # image spec's content is a multi-still collage.
    chart: dict[str, Any] | None = None
    collage: bool = False
    motion: str = "static"
    cost_usd: float = 0.0
    # Block 2: the declared travel path (PATH_KINDS) or None. When
    # set, layer_at(t) during the travel window returns the layer
    # at the path's position for t, and mask() is the SWEPT union
    # over the whole path - the superset F8's zone legs measure.
    path: str | None = None
    # Stage 4: the declared exit transition (EXIT_PATH_KINDS) or
    # None. Mirrors ``path``: layer_at(t) during the exit window
    # (the span's last travel budget) returns the layer at the
    # exit's position for t, and mask() sweeps it too.
    # ``layer_rgba`` stays the RESTING state (content at progress
    # 1 in the placed box) even with an exit declared - at the
    # literal end instant the graphic has left, and the resting
    # state is the declaration the zone legs measure against.
    exit_path: str | None = None
    # Extension: the layout state + camera this spec was placed
    # under - the declaration Stage 3's verifier legs measure
    # declared-vs-actual against.
    layout: str = "stage"
    camera: dict[str, Any] | None = None
    anchor_quality: str = "high"
    # Fix phase (2026-11-02): the layout-assist record carried
    # from the planner span (see GraphicSpan.layout_assist).
    layout_assist: dict[str, Any] | None = None
    _state_fn: LayerFn | None = field(default=None, repr=False,
                                       compare=False)
    # Fix round 2 (Fix C): the prepared content this spec was
    # placed from, retained so plan_layout can RE-FIT the card
    # (resize/reposition via the content's resizer) when a
    # Spotlight PiP would otherwise collide with it. None for
    # hand-built specs - those cannot be re-rendered at a new
    # size and downgrade honestly instead.
    _content: Any = field(default=None, repr=False, compare=False)

    def layer_at(self, t_s: float) -> np.ndarray:
        """The graphic's canvas layer at time ``t_s``."""
        if self._state_fn is not None:
            return self._state_fn(t_s, self.start_s, self.end_s)
        return self.layer_rgba

    def mask(self) -> np.ndarray:
        """Bool canvas of the graphic's opaque pixels.

        Without a path this is the final state - the superset every
        in-box animated state stays inside. With a path, the box
        itself travels, so the mask is the SWEPT union: the opaque
        pixels of layer_at sampled across the travel window plus
        the final state. F8's zone legs measure this mask, so a
        path that crosses a forbidden zone at any point is a
        violation even though the resting box is legal.
        """
        base = self.layer_rgba[..., 3] > 0
        if self._state_fn is None or (
                self.path is None and self.exit_path is None):
            return base
        swept = base
        steps = 12
        travel = path_travel_s(self.end_s - self.start_s)
        if self.path is not None:
            for k in range(steps + 1):
                t = self.start_s + travel * k / steps
                if t > self.end_s:
                    break
                swept = swept | (self.layer_at(t)[..., 3] > 0)
        if self.exit_path is not None:
            for k in range(steps + 1):
                t = self.end_s - travel + travel * k / steps
                if t < self.start_s:
                    continue
                swept = swept | (self.layer_at(t)[..., 3] > 0)
        return swept

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "span": [round(self.start_s, 3), round(self.end_s, 3)],
            "box": [int(v) for v in self.box],
            "motion": self.motion,
            "cost_usd": round(float(self.cost_usd), 4),
            "source_quote": self.source_quote,
        }
        if self.kind == "card":
            out["headline"] = self.headline
            out["facts"] = self.facts
            if self.compare:
                out["compare"] = self.compare
            if self.chart:
                out["chart"] = self.chart
        else:
            out["prompt"] = self.prompt
            if self.collage:
                out["collage"] = True
        if self.path is not None:
            out["path"] = self.path
        if self.exit_path is not None:
            out["exit_path"] = self.exit_path
        if self.layout != "stage":
            out["layout"] = self.layout
        if self.layout_assist is not None:
            out["layout_assist"] = self.layout_assist
        if self.camera is not None:
            out["camera"] = self.camera
        if self.anchor_quality != "high":
            out["anchor_quality"] = self.anchor_quality
        return out


@dataclass
class GraphicsBuild:
    """The outcome of planning + generation + placement."""

    placed: list[GraphicSpec] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    planner_call: dict[str, Any] | None = None
    planner_rejected: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    # Block 2: graphics whose declared PATH was refused at
    # placement (the graphic itself was placed, in place) - the
    # refusal and its reason are on the record, never silent.
    path_rejections: list[dict[str, Any]] = field(default_factory=list)
    # Block 2 heroes: prepared backdrop-mode spans (generated
    # scenes on the backdrop layer) and the speaker_forward spans
    # the pipeline applies as emphasis framing. Hero conflicts
    # resolved by resolve_heroes land in ``dropped`` like any
    # other drop.
    backdrops: list[BackdropSpec] = field(default_factory=list)
    speaker_forward: list[GraphicSpan] = field(default_factory=list)
    # Extension: per-span layout declarations (layout_states
    # LayoutPlan.declaration records) for spans placed under a
    # non-Stage state - filled by the caller that planned the
    # layout track, quoted into the plan JSON for Stage 3.
    layout_spans: list[dict[str, Any]] = field(default_factory=list)
    # Fix phase (2026-11-02): deterministic layout-assist records
    # for speaker_forward spans the assist synthesized in
    # build_graphics. (Spotlight assists are recorded on the
    # layout declarations instead - see plan_layout.)
    layout_assist: list[dict[str, Any]] = field(default_factory=list)
    # The Short's duration, for the coverage declaration (span
    # law: coverage is DECLARED on the record, never capped).
    duration_s: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return round(
            sum(g.cost_usd for g in self.placed)
            + sum(b.cost_usd for b in self.backdrops), 4)

    def to_dict(self) -> dict[str, Any]:
        """The graphics plan JSON (written per Short by the pipeline
        and quoted in receipts' neighbourhood): what was placed,
        what was dropped and why, what the planner was told no
        about, and what it all cost."""
        return {
            "placed": [g.to_dict() for g in self.placed],
            "dropped": self.dropped,
            "planner_rejected": self.planner_rejected,
            "planner_call": self.planner_call,
            "error": self.error,
            "path_rejections": self.path_rejections,
            "backdrops": [b.to_dict() for b in self.backdrops],
            "speaker_forward": [s.to_dict()
                                for s in self.speaker_forward],
            "layout_assist": self.layout_assist,
            "layout_spans": self.layout_spans,
            "coverage": graphics_coverage(self.placed,
                                          self.duration_s),
            "total_cost_usd": self.total_cost_usd,
        }


# ---------------------------------------------------------------- planner

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _word_seq(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _contains_word_seq(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    return any(haystack[i:i + n] == needle
               for i in range(len(haystack) - n + 1))


def _reject(span: Any, reason: str) -> dict[str, Any]:
    return {"span": span if isinstance(span, dict) else str(span),
            "reason": reason}


def _derive_motion(span: GraphicSpan) -> str:
    """The motion a span's content earns - derived in code, never
    chosen by the planner, so the recorded type is auditable."""
    if span.kind in ("image", "backdrop"):
        return "ken_burns"
    if span.kind == "speaker_forward":
        return "speaker_forward"
    if span.compare:
        return "build_two_state"
    if _TOKEN_RE.search(span.headline) or _TOKEN_RE.search(span.facts):
        return "count_up"
    if span.end_s - span.start_s >= REVEAL_MIN_SPAN_S:
        return "reveal"
    return "static"


def parse_planner_reply(
    text: str,
    transcript_text: str,
    *,
    duration_s: float | None = None,
) -> tuple[list[GraphicSpan], list[dict[str, Any]]]:
    """Strict-parse + validate the planner's JSON.

    Returns (accepted spans, rejected records). Validation is the
    product rule, not a courtesy: quotes verbatim, numbers by
    value, people never images. Timing is NOT policed here beyond
    the span-length bounds (2026-10-31 ruling): spans may start at
    0.0s and overlap each other freely - collision avoidance is
    placement's job and collision verification is F8's.
    Anything malformed is rejected with its reason - never repaired
    into existence.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return [], [_reject(text[:200], "planner reply was not JSON")]
    items = data.get("graphics") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], [_reject(text[:200],
                            "planner reply has no graphics list")]

    transcript_words = _word_seq(transcript_text)
    transcript_values = numeric_values(transcript_text)
    accepted: list[GraphicSpan] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            rejected.append(_reject(item, "span is not an object"))
            continue
        try:
            start = float(item["start_s"])
            end = float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            rejected.append(_reject(item, "span times missing/invalid"))
            continue
        kind = str(item.get("kind", "")).strip().lower()
        headline = str(item.get("headline", "")).strip()
        facts = str(item.get("facts", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        quote = str(item.get("source_quote", "")).strip()
        compare = item.get("compare")
        chart = item.get("chart")
        collage = bool(item.get("collage", False))
        path = item.get("path")
        if path is not None:
            path = str(path).strip().lower() or None
        exit_path = item.get("exit_path")
        if exit_path is not None:
            exit_path = str(exit_path).strip().lower() or None
        layout = str(item.get("layout", "stage")).strip().lower() \
            or "stage"
        camera = item.get("camera")
        if camera is not None:
            camera = validate_camera(camera)
            if camera is None:
                rejected.append(_reject(
                    item, "camera declaration is not a usable "
                          "{type: punch_in|drift, from, to} move"))
                continue
        reason = None
        if kind not in ("card", "image", "backdrop",
                        "speaker_forward"):
            reason = f"unknown kind {kind!r}"
        elif layout not in LAYOUT_STATES:
            reason = f"unknown layout state {layout!r}"
        elif layout != "stage" and kind not in ("card", "image"):
            reason = ("layout state on a hero span - backdrop and "
                      "speaker_forward are framing already")
        elif camera is not None and kind not in ("card", "image"):
            reason = "camera move on a non-graphic span"
        elif path is not None and path not in PATH_KINDS:
            reason = f"unknown motion path {path!r}"
        elif path is not None and kind not in ("card", "image"):
            reason = "motion path on a non-graphic span"
        elif exit_path is not None and exit_path not in EXIT_PATH_KINDS:
            reason = f"unknown exit path {exit_path!r}"
        elif exit_path is not None and kind not in ("card", "image"):
            reason = "exit path on a non-graphic span"
        elif end <= start:
            reason = "span end is not after its start"
        elif end - start < SPAN_MIN_S:
            reason = (f"span length {end - start:.2f}s below the "
                      f"{SPAN_MIN_S}s floor (span-law amendment: "
                      f"durations are content-driven - 1.0s floor, "
                      f"no fixed ceiling, the anchor passage is "
                      f"the bound; the ceiling and the card "
                      f"legibility floor are enforced at the "
                      f"timing snap, where the anchor is known)")
        elif duration_s is not None and end > duration_s + 0.05:
            reason = f"span ends at {end:.2f}s past the Short's end"
        elif not _contains_word_seq(transcript_words, _word_seq(quote)):
            reason = "source_quote is not a verbatim transcript sequence"
        elif kind == "card" and not headline:
            reason = "card without a headline"
        elif kind == "card" and any(
                round(v, 6) not in transcript_values
                for v in numeric_values(headline + " " + facts)):
            reason = "card carries a number the transcript never states"
        elif kind == "image" and not prompt:
            reason = "image without a prompt"
        elif kind == "image" and re.search(r"\d", prompt):
            reason = ("image prompt contains a digit (numbers never "
                      "go through the image model)")
        elif kind == "image" and any(
                phrase in prompt.lower()
                for phrase in IMAGE_PROMPT_BLOCKLIST):
            reason = ("image prompt depicts a person/face - named "
                      "people are typographic name cards only")
        elif kind == "image" and not (8 <= len(prompt) <= 300):
            reason = "image prompt length outside [8, 300] chars"
        elif kind == "backdrop" and not prompt:
            reason = "backdrop without a prompt"
        elif kind == "backdrop" and re.search(r"\d", prompt):
            reason = ("backdrop prompt contains a digit (numbers "
                      "never go through the image model)")
        elif kind == "backdrop" and any(
                phrase in prompt.lower()
                for phrase in IMAGE_PROMPT_BLOCKLIST):
            reason = ("backdrop prompt depicts a person/face - "
                      "named people are typographic name cards only")
        elif kind == "backdrop" and not (8 <= len(prompt) <= 300):
            reason = "backdrop prompt length outside [8, 300] chars"
        elif kind == "backdrop" and (headline or facts):
            reason = ("backdrop span carries headline/facts - a "
                      "backdrop is a scene, never data (numbers, "
                      "names and comparisons stay cards)")
        elif kind == "speaker_forward" and (headline or facts
                                            or prompt):
            reason = ("speaker_forward carries headline/facts/"
                      "prompt - it is framing only; the verbatim "
                      "quote is its whole content")
        if reason is None and compare is not None:
            reason = _validate_compare(compare, transcript_words,
                                       transcript_values)
            if reason is None and kind != "card":
                reason = "compare data on a non-card span"
        if reason is None and chart is not None:
            if compare is not None:
                reason = "chart and compare on the same span"
            else:
                reason = _validate_chart(chart, transcript_words,
                                         transcript_values)
            if reason is None and kind != "card":
                reason = "chart data on a non-card span"
        if reason is None and collage and kind != "image":
            reason = "collage on a non-image span"
        if reason is not None:
            rejected.append(_reject(item, reason))
            continue
        span = GraphicSpan(
            start_s=round(start, 3), end_s=round(end, 3), kind=kind,
            headline=headline, facts=facts, prompt=prompt,
            source_quote=quote,
            compare=compare if compare else None,
            chart=chart if chart else None,
            collage=collage,
            path=path, exit_path=exit_path,
            layout=layout, camera=camera)
        span.motion = _derive_motion(span)
        accepted.append(span)

    accepted.sort(key=lambda s: (s.start_s, s.end_s))
    # No spacing or overlap policing (2026-10-31 ruling): spans
    # may be back-to-back or overlap in time, and there is
    # deliberately NO count cap and NO screen-time ceiling - the
    # semantic bar is the only limit on how many graphics a Short
    # carries. Placement separates whatever overlaps in time onto
    # disjoint boxes, or drops what fits nowhere legal.
    return accepted, rejected


# ---------------------------------------------------------------- span law


def legibility_floor_s(span: GraphicSpan) -> float:
    """A card's minimum dwell, derived from its own text.

    ~0.35s per word of headline+facts, minimum 1.5s: text may
    not flash unreadably. Non-card spans (images, backdrops,
    speaker_forward) carry no text and use the plain 1.0s floor.
    """
    if span.kind != "card":
        return SPAN_MIN_S
    words = (len(_word_seq(span.headline))
             + len(_word_seq(span.facts)))
    return max(LEGIBILITY_MIN_S, LEGIBILITY_S_PER_WORD * words)


def span_law_violation(
    span: GraphicSpan,
    anchor: tuple[float, float] | None = None,
) -> str | None:
    """The amended span law as one checkable rule, or None.

    Floor 1.0s for everything; the legibility floor for text
    cards; and, when the anchor passage is known, the semantic
    ceiling: the span may start at most SNAP_START_PAD_S before
    the passage and end at most SNAP_END_PAD_S after it - a span
    may not outlive the speech it illustrates.
    """
    dur = span.end_s - span.start_s
    if dur < SPAN_MIN_S - 1e-9:
        return (f"span length {dur:.2f}s is below the "
                f"{SPAN_MIN_S}s floor")
    floor = legibility_floor_s(span)
    if dur < floor - 1e-9:
        return (f"card span {dur:.2f}s is below its legibility "
                f"floor {floor:.2f}s (~{LEGIBILITY_S_PER_WORD}s "
                f"per word of headline+facts, minimum "
                f"{LEGIBILITY_MIN_S}s) - the text would flash "
                f"unreadably")
    if anchor is not None:
        a0, a1 = anchor
        if span.start_s < a0 - SNAP_START_PAD_S - 1e-6 \
                or span.end_s > a1 + SNAP_END_PAD_S + 1e-6:
            return (f"span {span.start_s:.2f}-{span.end_s:.2f}s "
                    f"outlives its anchor passage "
                    f"{a0:.2f}-{a1:.2f}s (+{SNAP_END_PAD_S}s tail "
                    f"allowed) - a graphic may not outlive the "
                    f"speech it illustrates")
    return None


def find_anchor_passage(
    quote: str,
    cues: Sequence[CaptionCue],
) -> tuple[float, float, str] | None:
    """The DSP phrase passage a source quote anchors to.

    Caption cues ARE the DSP-timed phrases, so the anchor
    passage is the shortest run of consecutive cues whose joined
    words contain the quote's word sequence verbatim: its
    boundaries are real phrase boundaries by construction.
    Returns (start_s, end_s, joined_text) or None.
    """
    qw = _word_seq(quote)
    if not qw:
        return None
    best: tuple[float, float, str] | None = None
    for i in range(len(cues)):
        words: list[str] = []
        texts: list[str] = []
        for j in range(i, len(cues)):
            words += _word_seq(cues[j].text)
            texts.append(cues[j].text)
            if _contains_word_seq(words, qw):
                cand = (cues[i].start_s, cues[j].end_s,
                        " ".join(texts))
                if best is None \
                        or cand[1] - cand[0] < best[1] - best[0]:
                    best = cand
                break
    return best


def _anchor_garbled(text: str) -> bool:
    """A deterministic garble signal: a word repeated back-to-back.

    Voice Transcribe's noisy turns characteristically stutter
    ("like, like hamburger"); clean turns do not repeat a word
    consecutively. This is a flag, not a judgement - the graphic
    still ships, the receipt just says the anchor was noisy.
    """
    words = _word_seq(text)
    return any(a == b for a, b in zip(words, words[1:]))


def snap_span_to_anchor(
    span: GraphicSpan,
    cues: Sequence[CaptionCue],
    *,
    duration_s: float | None = None,
) -> tuple[GraphicSpan | None, str | None]:
    """Snap one span to its anchor passage and enforce the law.

    Returns (snapped span, None) or (None, rejection reason).
    The snapped span starts SNAP_START_PAD_S before the passage
    and ends SNAP_END_PAD_S after it (clamped to the Short), so
    it satisfies the anchor ceiling by construction; what can
    still reject it is the floor / legibility floor when the
    passage itself is too short for the graphic's text. The
    anchor's quality is stamped on the span: "low" when the
    quote was not found in the cues (the anchor fell back to
    timing overlap, or to the proposal itself when nothing
    overlaps) or the passage text is garbled.
    """
    found = find_anchor_passage(span.source_quote, cues)
    if found is not None:
        a0, a1, atext = found
        quality = "low" if _anchor_garbled(atext) else "high"
    else:
        overlapping = [c for c in cues
                       if c.start_s < span.end_s
                       and span.start_s < c.end_s]
        if overlapping:
            a0 = min(c.start_s for c in overlapping)
            a1 = max(c.end_s for c in overlapping)
        else:
            a0, a1 = span.start_s, span.end_s
        quality = "low"
    start = max(0.0, a0 - SNAP_START_PAD_S)
    end = a1 + SNAP_END_PAD_S
    if duration_s is not None:
        end = min(end, duration_s)
    snapped = dataclasses.replace(
        span, start_s=round(start, 3), end_s=round(end, 3),
        anchor_quality=quality)
    snapped.motion = _derive_motion(snapped)
    violation = span_law_violation(snapped, anchor=(a0, a1))
    if violation is not None:
        return None, violation
    return snapped, None


def snap_spans_to_anchors(
    spans: list[GraphicSpan],
    cues: Sequence[CaptionCue],
    *,
    duration_s: float | None = None,
) -> tuple[list[GraphicSpan], list[dict[str, Any]]]:
    """Snap every span; law-breakers come back as rejections."""
    kept: list[GraphicSpan] = []
    rejected: list[dict[str, Any]] = []
    for span in spans:
        snapped, reason = snap_span_to_anchor(
            span, cues, duration_s=duration_s)
        if reason is not None or snapped is None:
            rejected.append(_reject(span.to_dict(),
                                    reason or "span failed the "
                                              "span law"))
            continue
        kept.append(snapped)
    kept.sort(key=lambda s: (s.start_s, s.end_s))
    return kept, rejected


def graphics_coverage(specs: Sequence[GraphicSpec],
                      duration_s: float) -> dict[str, Any]:
    """Per-Short graphics coverage: the union of placed spans.

    Declared, never capped (span-law amendment): the number is
    on the plan and the receipt so density is visible and
    auditable, not policed.
    """
    spans = sorted((g.start_s, g.end_s) for g in specs)
    covered = 0.0
    cur: list[float] | None = None
    for s0, s1 in spans:
        if cur is None:
            cur = [s0, s1]
        elif s0 <= cur[1]:
            cur[1] = max(cur[1], s1)
        else:
            covered += cur[1] - cur[0]
            cur = [s0, s1]
    if cur is not None:
        covered += cur[1] - cur[0]
    return {
        "graphics_s": round(covered, 3),
        "short_s": round(float(duration_s), 3),
        "coverage_frac": (round(covered / duration_s, 4)
                          if duration_s > 0 else 0.0),
    }


HERO_KINDS = ("backdrop", "speaker_forward")


def resolve_heroes(
    spans: list[GraphicSpan],
    hook_window: tuple[float, float] | None = None,
) -> tuple[list[GraphicSpan], list[dict[str, Any]]]:
    """ONE HERO AT ANY INSTANT (block 2, amended 2026-11-01):
    enforce hero exclusivity.

    A moment in the Short gets exactly one hero treatment: a
    card/image graphic, a backdrop scene, a speaker_forward
    emphasis, or none. Backdrop and speaker_forward spans are
    EXCLUSIVE heroes: neither may overlap any other span, and no
    other span may overlap them. Card/image-vs-card/image overlaps
    keep the 2026-10-31 collision rules (placement separates them
    spatially; F8 verifies) - they are not hero conflicts.
    Overlapping hero claims resolve by dropping the LATER-starting
    claim (ties: the later one in plan order), with the reason
    recorded. A speaker_forward span overlapping the hook window
    also drops: the hook's geometry is planned pre-emphasis, and
    emphasis is for payoff moments, not the hook.

    Amendment (span law, 2026-11-01): one-hero-per-span becomes
    one hero at any instant. A card/image span carrying a
    non-Stage LAYOUT state is a hero claim in the same sense -
    it owns the span's framing (in Spotlight the hero is the
    span's only graphic) - so it must not overlap ANY other
    span, and no other span may overlap it. Back-to-back
    chaining stays unlimited: only true overlaps conflict.
    """

    def is_framing(span: GraphicSpan) -> bool:
        return span.kind in HERO_KINDS or span.layout != "stage"

    def overlaps(a_start: float, a_end: float,
                 b_start: float, b_end: float) -> bool:
        return a_start < b_end and b_start < a_end

    kept: list[GraphicSpan] = []
    drops: list[dict[str, Any]] = []
    order = sorted(range(len(spans)),
                   key=lambda i: (spans[i].start_s, i))
    for i in order:
        span = spans[i]
        if span.kind == "speaker_forward" \
                and hook_window is not None \
                and overlaps(span.start_s, span.end_s,
                             hook_window[0], hook_window[1]):
            drops.append({
                "span": span.to_dict(),
                "reason": "speaker_forward overlaps the hook "
                          "window - hook geometry is planned "
                          "pre-emphasis, and emphasis is for "
                          "payoff moments, not the hook"})
            continue
        conflict = None
        for other in kept:
            if not overlaps(span.start_s, span.end_s,
                            other.start_s, other.end_s):
                continue
            if is_framing(span) or is_framing(other):
                conflict = other
                break
        if conflict is not None:
            drops.append({
                "span": span.to_dict(),
                "reason": f"hero conflict: overlaps the "
                          f"{conflict.kind} span at "
                          f"{conflict.start_s:.2f}-{conflict.end_s:.2f}s"
                          f" - one hero at any instant, and the "
                          f"later-starting claim drops"})
            continue
        kept.append(span)
    kept_set = set(map(id, kept))
    return [s for s in spans if id(s) in kept_set], drops


CHART_TYPES = ("bar", "line")
CHART_MIN_ITEMS = 2
CHART_MAX_ITEMS = 6


def _validate_chart(chart: Any, transcript_words: list[str],
                    transcript_values: set[float]) -> str | None:
    """A chart needs 2-6 items: transcript labels and VERBATIM
    stated values - the numeric guard by value, unchanged from
    compare's. The chart draws ONLY these values (its axis is
    the baseline at zero plus the values themselves as labels),
    so a value the transcript does not state is an invention
    and the span is rejected pre-render. Returns the rejection
    reason, or None."""
    if not isinstance(chart, dict):
        return "chart is not an object"
    if chart.get("type") not in CHART_TYPES:
        return f"unknown chart type {chart.get('type')!r}"
    items = chart.get("items")
    if not isinstance(items, list) or not (
            CHART_MIN_ITEMS <= len(items) <= CHART_MAX_ITEMS):
        return (f"chart must list {CHART_MIN_ITEMS}-{CHART_MAX_ITEMS} "
                "items")
    labels: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            return "chart item is not an object"
        label = str(entry.get("label", "")).strip()
        if not label or not _contains_word_seq(
                transcript_words, _word_seq(label)):
            return f"chart label {label!r} is not transcript words"
        if label.lower() in labels:
            return f"chart label {label!r} is duplicated"
        labels.add(label.lower())
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            return "chart item value is not a number"
        if round(value, 6) not in transcript_values:
            return (f"chart value {value} is not a number the "
                    "transcript states")
        if value < 0:
            return (f"chart value {value} is negative - a chart "
                    "draws magnitudes from a zero baseline and "
                    "cannot draw a negative value honestly")
    return None


def validate_chart_values(chart: Any, transcript_text: str) -> str | None:
    """The chart guard as a standalone check (plan parse uses it;
    the proof pack exercises it directly): rejection reason or
    None when every chart value is verbatim in the transcript."""
    return _validate_chart(chart, _word_seq(transcript_text),
                           numeric_values(transcript_text))


def _validate_compare(compare: Any, transcript_words: list[str],
                      transcript_values: set[float]) -> str | None:
    """A comparison needs exactly two items: transcript labels and
    stated values. Returns the rejection reason, or None."""
    if not isinstance(compare, list) or len(compare) != 2:
        return "compare must list exactly two items"
    for entry in compare:
        if not isinstance(entry, dict):
            return "compare item is not an object"
        label = str(entry.get("label", "")).strip()
        if not label or not _contains_word_seq(
                transcript_words, _word_seq(label)):
            return f"compare label {label!r} is not transcript words"
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            return "compare item value is not a number"
        if round(value, 6) not in transcript_values:
            return (f"compare value {value} is not a number the "
                    "transcript states")
    return None


@dataclass
class PlannerOutcome:
    spans: list[GraphicSpan] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    spark_call: dict[str, Any] | None = None
    error: str | None = None


# parse_planner_reply's two TOP-LEVEL rejection reasons: the reply
# itself was not usable JSON (as opposed to individual spans failing
# validation, which a retry cannot fix and must not trigger one).
_MALFORMED_REPLY_REASONS = (
    "planner reply was not JSON",
    "planner reply has no graphics list",
)


def _reply_was_malformed(spans: list[GraphicSpan],
                         rejected: list[dict[str, Any]]) -> bool:
    """True when a planner reply failed at the JSON level.

    A valid reply whose spans all fail validation is NOT malformed
    (spans empty, rejections carry per-span reasons) - retrying
    would just buy the same validation verdict twice.
    """
    if spans:
        return False
    return any(r.get("reason") in _MALFORMED_REPLY_REASONS
               for r in rejected)


def plan_graphics(
    spark: SparkClient,
    cues: Sequence[CaptionCue],
    *,
    duration_s: float,
    ts: str = "",
) -> PlannerOutcome:
    """Ledger-gated Spark call(s) proposing graphic spans.

    A planner failure degrades to an empty outcome with the error
    on the record: graphics are an additive layer, and a render
    must never die because an optional enhancement could not be
    planned. (The verifier still certifies whatever WAS placed.)

    Empty replies get ONE retry at a doubled budget (cap
    PLANNER_MAX_TOKENS_CAP), following the Wonder judge and
    Spark-Open sweep precedent: reasoning models burn the budget on
    reasoning first, and a tight budget returns empty stochastically.
    A second empty reply degrades like any other planner failure.
    Non-empty failures (HTTP errors, cap errors) are NOT retried.

    Malformed replies (the text arrived but is not usable JSON -
    fix phase, 2026-11-02) get the same single doubled-budget
    retry as empty replies: both are completion-shape failures a
    fresh sample can plausibly fix. Malformed at every budget
    degrades to an empty outcome with the parse rejections AND an
    error on the record - never an exception into the Short.
    """
    transcript = " ".join(c.text for c in cues)
    timing = "\n".join(
        f"[{c.start_s:.2f}-{c.end_s:.2f}] {c.text}" for c in cues)
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content":
         f"Short length: {duration_s:.1f}s. Caption phrases "
         f"with timings:\n{timing}"}]
    budgets = [PLANNER_MAX_TOKENS]
    doubled = min(PLANNER_MAX_TOKENS * 2, PLANNER_MAX_TOKENS_CAP)
    if doubled > budgets[0]:
        budgets.append(doubled)
    result = None
    empty_first = False
    malformed_first = False
    spans: list[GraphicSpan] = []
    rejected: list[dict[str, Any]] = []
    last_failure: str | None = None
    for budget in budgets:
        try:
            result = spark.chat(
                messages, max_tokens=budget,
                purpose="graphics-planner", ts=ts)
        except SparkClientError as exc:
            if "empty chat reply" not in str(exc):
                return PlannerOutcome(
                    error=f"planner call failed: {exc}"[:300])
            empty_first = True
            result = None
            last_failure = "empty"
            continue
        except Exception as exc:
            return PlannerOutcome(error=f"planner call failed: {exc}"[:300])
        cand_spans, cand_rejected = parse_planner_reply(
            result.text, transcript, duration_s=duration_s)
        if _reply_was_malformed(cand_spans, cand_rejected):
            # Malformed JSON (fix phase, 2026-11-02): the reply
            # arrived but is not usable JSON - the same failure
            # class as an empty reply (a truncated/unbalanced
            # completion), so it earns the same single retry at
            # the next budget instead of degrading immediately.
            malformed_first = True
            last_failure = "malformed"
            spans, rejected = cand_spans, cand_rejected
            result = None
            continue
        spans, rejected = cand_spans, cand_rejected
        last_failure = None
        break
    if result is None:
        if last_failure == "malformed":
            return PlannerOutcome(
                spans=[], rejected=rejected,
                error="planner reply was not JSON at every budget "
                      "(base and doubled retry); no graphics "
                      "planned")
        return PlannerOutcome(
            error="planner call failed: empty chat reply at every "
                  "budget (base and doubled retry); no graphics "
                  "planned")
    call = {"model": "muse-spark-1.3", "purpose": "graphics-planner",
            "cost_usd": result.cost_usd,
            "request_id": result.request_id,
            "empty_first_attempt": empty_first,
            "malformed_first_attempt": malformed_first}
    # Timing snap (E2): every accepted span is re-timed onto its
    # anchor passage's DSP phrase boundaries and the amended
    # span law is enforced there - the planner proposes, the
    # snap disposes, exactly like placement.
    spans, snap_rejected = snap_spans_to_anchors(
        spans, cues, duration_s=duration_s)
    rejected = rejected + snap_rejected
    return PlannerOutcome(spans=spans, rejected=rejected,
                          spark_call=call)


# ---------------------------------------------------------------- cards


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int,
          ) -> list[list[str]]:
    """Wrap text to lines of WORDS (line structure is layout)."""
    lines: list[list[str]] = []
    current: list[str] = []
    for word in text.split():
        trial = " ".join(current + [word])
        if draw.textlength(trial, font=font) <= max_w or not current:
            current.append(word)
        else:
            lines.append(current)
            current = [word]
    if current:
        lines.append(current)
    return lines


def _card_layout(headline: str, facts: str, width: int,
                 font_scale: float = 1.0):
    """(head_lines, fact_lines, height) - the card's fixed layout.

    Layout is computed from the FINAL text and never changes across
    animation states: count-up swaps a token inside its line,
    reveal hides whole lines, but the box is constant.
    ``font_scale`` grows the type with the card's width when the
    card is re-rendered at a placed size (block 2 size-to-space);
    at 1.0 the layout is byte-identical to the original renderer.
    """
    probe = Image.new("RGBA", (8, 8))
    draw = ImageDraw.Draw(probe)
    head_font = _load_font(round(50 * font_scale))
    fact_font = _load_font(round(38 * font_scale))
    max_w = width - 2 * CARD_PAD
    head_lines = _wrap(draw, headline.upper(), head_font, max_w)[:3]
    fact_lines = _wrap(draw, facts, fact_font, max_w)[:2] if facts else []
    head_h = int(50 * 1.18 * font_scale) * len(head_lines)
    fact_h = int(38 * 1.18 * font_scale) * len(fact_lines)
    height = (CARD_PAD + 10 + 16 + head_h
              + (12 + fact_h if fact_lines else 0) + CARD_PAD)
    return head_lines, fact_lines, height


def _draw_card(head_lines: list[list[str]], fact_lines: list[list[str]],
               height: int, width: int, *,
               reveal_count: int | None = None,
               font_scale: float = 1.0) -> Image.Image:
    """Draw a card from word-lines; ``reveal_count`` caps how many
    lines (headline first, then facts) are visible."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, width - 1, height - 1], radius=CARD_RADIUS,
        fill=NEAR_BLACK + (242,), outline=FORGE_BLUE + (255,), width=3)
    strip = Image.new("RGBA", (width - 2 * CARD_PAD, 10), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(strip)
    for x in range(strip.size[0]):
        t = x / max(1, strip.size[0] - 1)
        color = tuple(
            round(FORGE_BLUE[i] * (1 - t) + VIRAL_PINK[i] * t)
            for i in range(3)) + (255,)
        sdraw.line([(x, 0), (x, 9)], fill=color)
    img.alpha_composite(strip, (CARD_PAD, CARD_PAD))
    head_font = _load_font(round(50 * font_scale))
    fact_font = _load_font(round(38 * font_scale))
    shown = 0
    total = len(head_lines) + len(fact_lines)
    limit = total if reveal_count is None else max(0, reveal_count)
    y = CARD_PAD + 10 + 16
    for line in head_lines:
        if shown < limit:
            draw.text((CARD_PAD, y), " ".join(line), font=head_font,
                      fill=(255, 255, 255, 255))
        shown += 1
        y += int(50 * 1.18 * font_scale)
    if fact_lines:
        y += 12
        for line in fact_lines:
            if shown < limit:
                draw.text((CARD_PAD, y), " ".join(line), font=fact_font,
                          fill=VIRAL_PINK + (255,))
            shown += 1
            y += int(38 * 1.18 * font_scale)
    return img


def render_card_image(headline: str, facts: str = "", *,
                      width: int = CARD_WIDTH,
                      font_scale: float = 1.0) -> Image.Image:
    """One brand-palette card, final state (deterministic pixels)."""
    head_lines, fact_lines, height = _card_layout(
        headline, facts, width, font_scale)
    return _draw_card(head_lines, fact_lines, height, width,
                      font_scale=font_scale)


def _count_up_image(headline: str, facts: str, p: float,
                    width: int, font_scale: float = 1.0) -> Image.Image:
    """The card at count-up progress ``p``: the first number token
    (headline first, then facts) eased from 0 to its stated value;
    every other word is already final."""
    head_lines, fact_lines, height = _card_layout(
        headline, facts, width, font_scale)
    eased = _ease_out_cubic(min(1.0, max(0.0, p)))

    def animate(lines: list[list[str]]) -> tuple[list[list[str]], bool]:
        out: list[list[str]] = []
        done = False
        for line in lines:
            new_line = []
            for word in line:
                m = _TOKEN_RE.fullmatch(word) if not done else None
                if m is not None:
                    try:
                        value, fmt = _parse_number_token(word)
                    except ValueError:
                        new_line.append(word)
                        continue
                    if p >= 0.999:
                        new_line.append(word)
                    else:
                        new_line.append(
                            _format_count(value * eased, fmt, word))
                    done = True
                else:
                    new_line.append(word)
            out.append(new_line)
        return out, done

    head_lines, done = animate(head_lines)
    if not done:
        fact_lines, _ = animate(fact_lines)
    return _draw_card(head_lines, fact_lines, height, width,
                      font_scale=font_scale)


def render_compare_card_image(
    headline: str,
    items: list[dict[str, Any]],
    progresses: tuple[float, float],
    *,
    width: int = COMPARE_CARD_WIDTH,
    font_scale: float = 1.0,
) -> Image.Image:
    """A two-bar comparison card at the given build progresses.

    Bar lengths are proportional to the stated values; each bar
    grows (ease-out) inside its own window. Before its window a bar
    shows its label over an empty track - the comparison builds as
    the speaker names each side.
    """
    probe = Image.new("RGBA", (8, 8))
    draw = ImageDraw.Draw(probe)
    head_font = _load_font(round(46 * font_scale))
    label_font = _load_font(round(36 * font_scale))
    max_w = width - 2 * CARD_PAD
    head_lines = _wrap(draw, headline.upper(), head_font, max_w)[:2]
    head_h = int(46 * 1.18 * font_scale) * len(head_lines)
    row_h = int(36 * 1.18 * font_scale) + 34 + 14
    height = CARD_PAD + 10 + 16 + head_h + 10 + row_h * 2 + CARD_PAD
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, width - 1, height - 1], radius=CARD_RADIUS,
        fill=NEAR_BLACK + (242,), outline=FORGE_BLUE + (255,), width=3)
    strip = Image.new("RGBA", (width - 2 * CARD_PAD, 10), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(strip)
    for x in range(strip.size[0]):
        t = x / max(1, strip.size[0] - 1)
        color = tuple(
            round(FORGE_BLUE[i] * (1 - t) + VIRAL_PINK[i] * t)
            for i in range(3)) + (255,)
        sdraw.line([(x, 0), (x, 9)], fill=color)
    img.alpha_composite(strip, (CARD_PAD, CARD_PAD))
    y = CARD_PAD + 10 + 16
    for line in head_lines:
        draw.text((CARD_PAD, y), " ".join(line), font=head_font,
                  fill=(255, 255, 255, 255))
        y += int(46 * 1.18 * font_scale)
    y += 10
    values = [float(it["value"]) for it in items]
    peak = max(values) if values and max(values) > 0 else 1.0
    bar_area_w = width - 2 * CARD_PAD - round(118 * font_scale)
    colors = [FORGE_BLUE, VIRAL_PINK]
    for i, item in enumerate(items[:2]):
        label = str(item["label"]).upper()
        draw.text((CARD_PAD, y), label, font=label_font,
                  fill=(235, 235, 240, 255))
        y += int(36 * 1.18 * font_scale)
        track = [CARD_PAD, y, CARD_PAD + bar_area_w, y + 34]
        draw.rounded_rectangle(track, radius=17,
                               fill=(255, 255, 255, 36))
        p = progresses[i] if i < len(progresses) else 1.0
        frac = (values[i] / peak) * (1.0 - (1.0 - min(1.0, max(0.0, p))) ** 2)
        if frac > 0.005:
            draw.rounded_rectangle(
                [CARD_PAD, y, CARD_PAD + max(34.0, bar_area_w * frac),
                 y + 34],
                radius=17, fill=colors[i] + (255,))
        value = values[i]
        vtext = f"{value:,.0f}" if value == int(value) else f"{value:g}"
        draw.text((CARD_PAD + bar_area_w + 14, y - 2), vtext,
                  font=label_font, fill=(255, 255, 255, 255))
        y += 34 + 14
    return img


CHART_CARD_WIDTH = 560
CHART_PLOT_H = 250


def _format_value(value: float) -> str:
    """A stated value as chart text: thousands separators for
    integers, trimmed decimals otherwise - formatting only, the
    VALUE is always the one the transcript stated."""
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def render_chart_card_image(
    headline: str,
    chart: dict[str, Any],
    progress: float = 1.0,
    *,
    width: int = CHART_CARD_WIDTH,
    font_scale: float = 1.0,
) -> Image.Image:
    """A code-drawn chart card (bar or line) from verbatim values.

    The guard already ran (every value is one the transcript
    states); this renderer draws ONLY those values: bar heights /
    point heights are proportional to them against the largest,
    each value is printed at its own bar/point, and the only
    axis is the zero baseline - there are no tick values to
    invent. ``progress`` animates the build (bars grow, the line
    draws left to right); at 1.0 the card is final.
    """
    items = chart["items"]
    values = [float(it["value"]) for it in items]
    labels = [str(it["label"]).upper() for it in items]
    peak = max(values) if values and max(values) > 0 else 1.0
    p = min(1.0, max(0.0, progress))
    probe = Image.new("RGBA", (8, 8))
    draw = ImageDraw.Draw(probe)
    head_font = _load_font(round(46 * font_scale))
    value_font = _load_font(round(32 * font_scale))
    max_w = width - 2 * CARD_PAD
    head_lines = _wrap(draw, headline.upper(), head_font, max_w)[:2]
    head_h = int(46 * 1.18 * font_scale) * len(head_lines)
    plot_h = round(CHART_PLOT_H * font_scale)
    # Category labels: shrink until every label fits its slot.
    n = len(items)
    slot_w = max_w / n
    label_font = None
    for size in (28, 24, 20, 16):
        cand = _load_font(round(size * font_scale))
        if all(draw.textlength(lab, font=cand) <= slot_w - 6
               for lab in labels):
            label_font = cand
            break
    if label_font is None:
        label_font = _load_font(round(16 * font_scale))
    label_h = int(28 * 1.18 * font_scale)
    height = (CARD_PAD + 10 + 16 + head_h + 12 + plot_h + 8
              + label_h + CARD_PAD)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, width - 1, height - 1], radius=CARD_RADIUS,
        fill=NEAR_BLACK + (242,), outline=FORGE_BLUE + (255,), width=3)
    strip = Image.new("RGBA", (width - 2 * CARD_PAD, 10), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(strip)
    for x in range(strip.size[0]):
        t = x / max(1, strip.size[0] - 1)
        color = tuple(
            round(FORGE_BLUE[i] * (1 - t) + VIRAL_PINK[i] * t)
            for i in range(3)) + (255,)
        sdraw.line([(x, 0), (x, 9)], fill=color)
    img.alpha_composite(strip, (CARD_PAD, CARD_PAD))
    y = CARD_PAD + 10 + 16
    for line in head_lines:
        draw.text((CARD_PAD, y), " ".join(line), font=head_font,
                  fill=(255, 255, 255, 255))
        y += int(46 * 1.18 * font_scale)
    y += 12
    plot_top = y
    plot_base = y + plot_h
    plot_x0 = CARD_PAD
    plot_x1 = width - CARD_PAD
    # The zero baseline - the chart's only axis. Bars/points use
    # inner_h, reserving the top strip for the value labels so the
    # tallest bar's label never climbs into the headline.
    inner_h = plot_h - round(48 * font_scale)
    draw.line([(plot_x0, plot_base), (plot_x1, plot_base)],
              fill=(235, 235, 240, 200), width=3)
    eased = _ease_out_cubic(p)
    if chart["type"] == "bar":
        bar_w = min(110 * font_scale, slot_w * 0.55)
        for i, value in enumerate(values):
            cx = plot_x0 + slot_w * (i + 0.5)
            h = inner_h * (value / peak) * eased
            if h > 1:
                draw.rounded_rectangle(
                    [cx - bar_w / 2, plot_base - h,
                     cx + bar_w / 2, plot_base],
                    radius=min(10, bar_w / 2),
                    fill=FORGE_BLUE + (255,))
            vtext = _format_value(value)
            tw = draw.textlength(vtext, font=value_font)
            draw.text((cx - tw / 2, plot_base - h - 40 * font_scale),
                      vtext, font=value_font,
                      fill=(255, 255, 255, 255))
    else:  # line
        pts = []
        for i, value in enumerate(values):
            cx = plot_x0 + slot_w * (i + 0.5) if n > 1 else \
                (plot_x0 + plot_x1) / 2
            cy = plot_base - inner_h * (value / peak)
            pts.append((cx, cy))
        reach = eased * (n - 1)
        full = int(reach)
        for i in range(min(full, n - 1)):
            draw.line([pts[i], pts[i + 1]], fill=VIRAL_PINK + (255,),
                      width=max(3, round(5 * font_scale)))
        if full < n - 1 and n > 1:
            frac = reach - full
            x0, y0 = pts[full]
            x1, y1 = pts[full + 1]
            draw.line([(x0, y0), (x0 + (x1 - x0) * frac,
                                  y0 + (y1 - y0) * frac)],
                      fill=VIRAL_PINK + (255,),
                      width=max(3, round(5 * font_scale)))
        shown = full + 1 if p > 0 else 0
        for i in range(min(shown, n)):
            cx, cy = pts[i]
            r = max(4, round(7 * font_scale))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=FORGE_BLUE + (255,),
                         outline=(255, 255, 255, 255), width=2)
            vtext = _format_value(values[i])
            tw = draw.textlength(vtext, font=value_font)
            draw.text((cx - tw / 2, cy - 44 * font_scale), vtext,
                      font=value_font, fill=(255, 255, 255, 255))
    ly = plot_base + 8
    for i, lab in enumerate(labels):
        cx = plot_x0 + slot_w * (i + 0.5)
        tw = draw.textlength(lab, font=label_font)
        draw.text((cx - tw / 2, ly), lab, font=label_font,
                  fill=(235, 235, 240, 255))
    return img


def _prepare_chart_card(span: GraphicSpan,
                        width: int | None = None) -> GraphicContent:
    assert span.chart is not None
    w = CHART_CARD_WIDTH if width is None else int(width)
    fs = w / CHART_CARD_WIDTH
    final = render_chart_card_image(span.headline, span.chart, 1.0,
                                    width=w, font_scale=fs)
    state = _quantized(
        lambda p: render_chart_card_image(span.headline, span.chart,
                                          p, width=w, font_scale=fs))
    return GraphicContent(
        size=final.size, final_image=final, state_fn=state,
        resizer=lambda target_w: _prepare_chart_card(  # noqa: E731
            span, width=target_w))


# ---------------------------------------------------------------- content


@dataclass
class GraphicContent:
    """Prepared content for one span: its fixed box size, its final
    state, and (when animated) a progress -> box-sized RGBA state
    function. States are pure functions of span progress - the same
    image is produced whenever the same progress is asked for.

    Block 2 size-to-space: ``resizer`` re-prepares the content at
    another width (cards re-render with scaled type, so text stays
    crisp; images re-fit from the source), and ``max_size`` caps
    the box a resizer may be asked for - for images, twice the
    generated source's dimensions, so a placement never upscales
    a bitmap past 2.0x.
    """

    size: tuple[int, int]
    final_image: Image.Image
    state_fn: Callable[[float], Image.Image] | None = None
    cost_usd: float = 0.0
    max_size: tuple[int, int] | None = None
    resizer: Callable[[int], "GraphicContent"] | None = None
    # Fix round 4: re-fit to an explicit (w, h) box - the Spotlight
    # hero solve sizes image content to the REGION it must own
    # (cover-fit into the region box), which the width-only
    # resizer cannot express (it derives height from the native
    # box aspect). Only content whose rendering is a cover-fit
    # into its box (images) provides one; cards/charts keep their
    # aspect and are sized by width.
    box_resizer: Callable[[int, int], "GraphicContent"] | None = None


def _quantized(state_fn: Callable[[float], Image.Image],
               ) -> Callable[[float], Image.Image]:
    cache: dict[float, Image.Image] = {}

    def wrapped(p: float) -> Image.Image:
        key = round(min(1.0, max(0.0, p)), 2)
        if key not in cache:
            cache[key] = state_fn(key)
        return cache[key]

    return wrapped


def prepare_card(span: GraphicSpan,
                 cues: Sequence[CaptionCue] = (),
                 width: int | None = None) -> GraphicContent:
    """Render a card span's content (final state + animation).

    ``width`` re-renders the card at a placed size with the type
    scaled to match (block 2 size-to-space); the default is the
    native CARD_WIDTH / COMPARE_CARD_WIDTH rendering, unchanged.
    """
    if span.chart:
        return _prepare_chart_card(span, width=width)
    if span.compare:
        return _prepare_compare_card(span, cues, width=width)
    w = CARD_WIDTH if width is None else int(width)
    fs = w / CARD_WIDTH
    final = render_card_image(span.headline, span.facts, width=w,
                              font_scale=fs)
    resizer = lambda target_w: prepare_card(span, cues,  # noqa: E731
                                            width=target_w)
    if span.motion == "count_up":
        state = _quantized(
            lambda p: _count_up_image(span.headline, span.facts, p,
                                      w, font_scale=fs))
        return GraphicContent(size=final.size, final_image=final,
                              state_fn=state, resizer=resizer)
    if span.motion == "reveal":
        head_lines, fact_lines, _ = _card_layout(
            span.headline, span.facts, w, fs)
        total = len(head_lines) + len(fact_lines)

        def reveal(p: float) -> Image.Image:
            # Lines land one by one; all are in by p = 0.8.
            count = int(min(total, (p / 0.8) * total + 1e-9)) \
                if p > 0 else 0
            head, fact, height = _card_layout(
                span.headline, span.facts, w, fs)
            return _draw_card(head, fact, height, w,
                              reveal_count=count, font_scale=fs)

        return GraphicContent(size=final.size, final_image=final,
                              state_fn=_quantized(reveal),
                              resizer=resizer)
    return GraphicContent(size=final.size, final_image=final,
                          resizer=resizer)


def _compare_triggers(span: GraphicSpan,
                      cues: Sequence[CaptionCue]) -> tuple[float, float]:
    """Span-progress fractions at which each compared item is first
    spoken: the first cue inside the span whose words contain the
    item's label. Fallbacks (start / midpoint) keep the build
    deterministic when a label cannot be located."""
    assert span.compare is not None
    span_len = max(span.end_s - span.start_s, 1e-9)
    fracs: list[float] = []
    for item in span.compare:
        label_words = _word_seq(str(item["label"]))
        hit = None
        for cue in cues:
            if cue.end_s <= span.start_s or cue.start_s >= span.end_s:
                continue
            if _contains_word_seq(_word_seq(cue.text), label_words):
                hit = max(cue.start_s, span.start_s)
                break
        fracs.append(0.0 if hit is None else (hit - span.start_s)
                     / span_len)
    if len(fracs) == 2:
        if fracs[0] == 0.0 and fracs[1] == 0.0:
            fracs[1] = 0.5
        elif fracs[1] <= fracs[0]:
            fracs[1] = min(0.9, fracs[0] + 0.3)
    return fracs[0], (fracs[1] if len(fracs) > 1 else 0.5)


def _prepare_compare_card(span: GraphicSpan,
                          cues: Sequence[CaptionCue],
                          width: int | None = None) -> GraphicContent:
    assert span.compare is not None
    headline = span.headline
    items = span.compare
    w = COMPARE_CARD_WIDTH if width is None else int(width)
    fs = w / COMPARE_CARD_WIDTH
    final = render_compare_card_image(headline, items, (1.0, 1.0),
                                      width=w, font_scale=fs)
    fracs = _compare_triggers(span, cues)
    span_len = max(span.end_s - span.start_s, 1e-9)
    bar_window = min(0.45, 0.7 / span_len)  # ~0.7s of growth, in p units

    def state(p: float) -> Image.Image:
        progresses = tuple(
            min(1.0, max(0.0, (p - f) / bar_window)) if p >= f else 0.0
            for f in fracs)
        return render_compare_card_image(headline, items, progresses,
                                         width=w, font_scale=fs)

    return GraphicContent(
        size=final.size, final_image=final,
        state_fn=_quantized(state),
        resizer=lambda target_w: _prepare_compare_card(  # noqa: E731
            span, cues, width=target_w))


def _cover_fit(image_rgb: Image.Image,
               size: tuple[int, int]) -> Image.Image:
    bw, bh = size
    sw, sh = image_rgb.size
    scale = max(bw / sw, bh / sh)
    resized = image_rgb.resize(
        (max(bw, round(sw * scale)), max(bh, round(sh * scale))),
        Image.BILINEAR)
    left = (resized.size[0] - bw) // 2
    top = (resized.size[1] - bh) // 2
    return resized.crop((left, top, left + bw, top + bh))


def _framed(image_rgb: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Box-sized RGBA: rounded corners + Forge Blue border."""
    fitted = image_rgb.convert("RGBA")
    mask = Image.new("L", size, 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                            radius=24, fill=255)
    fitted.putalpha(mask)
    draw = ImageDraw.Draw(fitted)
    draw.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=24,
                           outline=FORGE_BLUE + (255,), width=3)
    return fitted


def prepare_image(image_rgb: Image.Image, span: GraphicSpan,
                  box: tuple[int, int] = IMAGE_BOX) -> GraphicContent:
    """Ken Burns content: zoom eases 1.00 -> 1.12 across the span
    with a slow pan, all inside the fixed box. ``box`` is the
    placed size (block 2 size-to-space re-fits from the source at
    the placed box; the default is the native IMAGE_BOX)."""
    bw, bh = box
    base = _cover_fit(image_rgb, (round(bw * 1.12), round(bh * 1.12)))

    def state(p: float) -> Image.Image:
        zoom = 1.0 + (KEN_BURNS_MAX_ZOOM - 1.0) * min(1.0, max(0.0, p))
        win_w = min(base.size[0], round(bw * 1.12 / zoom))
        win_h = min(base.size[1], round(bh * 1.12 / zoom))
        slack_x = base.size[0] - win_w
        slack_y = base.size[1] - win_h
        x = int(slack_x * (0.20 + 0.45 * p))
        y = int(slack_y * (0.20 + 0.40 * p))
        crop = base.crop((x, y, x + win_w, y + win_h)).resize(
            (bw, bh), Image.BILINEAR)
        return _framed(crop, box)

    src_w, src_h = image_rgb.size
    aspect = IMAGE_BOX[1] / IMAGE_BOX[0]
    return GraphicContent(
        size=box, final_image=state(1.0), state_fn=_quantized(state),
        max_size=(2 * src_w, 2 * src_h),
        resizer=lambda target_w: prepare_image(  # noqa: E731
            image_rgb, span,
            box=(int(target_w), max(1, round(target_w * aspect)))),
        box_resizer=lambda target_w, target_h: prepare_image(  # noqa: E731
            image_rgb, span,
            box=(int(target_w), int(target_h))))


# ---------------------------------------------------------------- collage

COLLAGE_BOX = (560, 640)
COLLAGE_PAD = 14
COLLAGE_GAP = 10


def _collage_cells(n: int, box: tuple[int, int],
                     ) -> list[tuple[int, int, int, int]]:
    """Cell rects (x, y, w, h) for ``n`` stills inside ``box``:
    2 = a stacked pair, 3 = two over one wide, 4 = a 2x2 grid."""
    bw, bh = box
    x0, y0 = COLLAGE_PAD, COLLAGE_PAD
    iw, ih = bw - 2 * COLLAGE_PAD, bh - 2 * COLLAGE_PAD
    g = COLLAGE_GAP
    if n == 2:
        h = (ih - g) // 2
        return [(x0, y0, iw, h), (x0, y0 + h + g, iw, ih - h - g)]
    if n == 3:
        h = (ih - g) // 2
        w = (iw - g) // 2
        return [(x0, y0, w, h), (x0 + w + g, y0, iw - w - g, h),
                (x0, y0 + h + g, iw, ih - h - g)]
    w = (iw - g) // 2
    h = (ih - g) // 2
    return [(x0, y0, w, h), (x0 + w + g, y0, iw - w - g, h),
            (x0, y0 + h + g, w, ih - h - g),
            (x0 + w + g, y0 + h + g, iw - w - g, ih - h - g)]


def prepare_collage(images: Sequence[Image.Image], span: GraphicSpan,
                    box: tuple[int, int] = COLLAGE_BOX) -> GraphicContent:
    """A collage card: 2-4 stills in a grid on a brand panel.

    The stills are SUPPLIED by the caller from its cache - this
    function generates nothing and calls nothing. Each cell is
    cover-fit and framed exactly like a single image still, so a
    collage reads as a set of the same stills, grouped. Static
    (no state function): a grid of photographs does not animate.
    """
    stills = [im.convert("RGB") for im in images]
    if not 2 <= len(stills) <= 4:
        raise ValueError(
            f"a collage takes 2-4 stills, got {len(stills)}")
    bw, bh = box
    panel = Image.new("RGBA", box, (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        [0, 0, bw - 1, bh - 1], radius=CARD_RADIUS,
        fill=NEAR_BLACK + (242,), outline=FORGE_BLUE + (255,), width=3)
    for (cx, cy, cw_, ch_), still in zip(
            _collage_cells(len(stills), box), stills):
        cell = _framed(_cover_fit(still, (cw_, ch_)), (cw_, ch_))
        panel.alpha_composite(cell, (cx, cy))
    min_w = min(im.size[0] for im in stills)
    min_h = min(im.size[1] for im in stills)
    aspect = COLLAGE_BOX[1] / COLLAGE_BOX[0]
    return GraphicContent(
        size=box, final_image=panel,
        max_size=(2 * min_w, 2 * min_h),
        resizer=lambda target_w: prepare_collage(  # noqa: E731
            stills, span,
            box=(int(target_w), max(1, round(target_w * aspect)))))


# ---------------------------------------------------------------- backdrop

# Backdrop-mode treatment (block 2): the generated scene replaces
# the backdrop layer for its span, treated so the speaker still
# pops - darkened, desaturated, slightly blurred, with the same
# slow Ken Burns drift images get. The speaker layer, hook and
# captions are structurally untouched (the visual sits BEHIND
# the speaker, on the layer the treated source backdrop uses).
BACKDROP_BRIGHTNESS = 0.55
BACKDROP_SATURATION = 0.55
BACKDROP_BLUR_PX = 2.0


@dataclass
class BackdropSpec:
    """One backdrop-mode span: a generated scene as the backdrop.

    Mirrors GraphicSpec's timing/state contract (``layer_at(t)``
    via an (t, start, end) state function) so retiming across a
    TRIM_SILENCE repair works with dataclasses.replace exactly
    like graphics. ``layer_at`` returns a full-canvas RGB frame.
    """

    start_s: float
    end_s: float
    layer_rgb: np.ndarray          # final state, canvas RGB
    prompt: str = ""
    source_quote: str = ""
    cost_usd: float = 0.0
    kind: str = "backdrop"
    motion: str = "ken_burns"
    _state_fn: LayerFn | None = None

    def layer_at(self, t_s: float) -> np.ndarray:
        if self._state_fn is not None:
            return self._state_fn(t_s, self.start_s, self.end_s)
        return self.layer_rgb

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "start_s": self.start_s,
            "end_s": self.end_s, "motion": self.motion,
            "prompt": self.prompt,
            "source_quote": self.source_quote,
            "cost_usd": self.cost_usd,
        }


def prepare_backdrop(image_rgb: Image.Image,
                     canvas: tuple[int, int]) -> BackdropSpec:
    """Treat a generated scene into a backdrop state function.

    Cover-fit at 1.12x (the Ken Burns window needs the slack),
    then the backdrop treatment: brightness x0.55, saturation
    x0.55, slight blur. The drift reuses the image motion: zoom
    eases 1.00 -> 1.12 with a slow pan, all inside the canvas.
    Span timing is the caller's (the spec is built by
    build_graphics with the span's times); this helper returns a
    spec with placeholder times the caller replaces.
    """
    from PIL import ImageEnhance, ImageFilter
    cw, ch = canvas
    base = _cover_fit(image_rgb, (round(cw * 1.12), round(ch * 1.12)))
    base = ImageEnhance.Brightness(base).enhance(BACKDROP_BRIGHTNESS)
    base = ImageEnhance.Color(base).enhance(BACKDROP_SATURATION)
    base = base.filter(ImageFilter.GaussianBlur(BACKDROP_BLUR_PX))

    def state(p: float) -> np.ndarray:
        zoom = 1.0 + (KEN_BURNS_MAX_ZOOM - 1.0) * min(1.0, max(0.0, p))
        win_w = min(base.size[0], round(cw * 1.12 / zoom))
        win_h = min(base.size[1], round(ch * 1.12 / zoom))
        slack_x = base.size[0] - win_w
        slack_y = base.size[1] - win_h
        x = int(slack_x * (0.20 + 0.45 * p))
        y = int(slack_y * (0.20 + 0.40 * p))
        crop = base.crop((x, y, x + win_w, y + win_h)).resize(
            (cw, ch), Image.BILINEAR)
        return np.asarray(crop.convert("RGB"))

    def state_fn(t: float, start: float, end: float) -> np.ndarray:
        p = min(1.0, max(0.0, (t - start) / (end - start))) \
            if end > start else 1.0
        return state(p)

    return BackdropSpec(start_s=0.0, end_s=0.0,
                        layer_rgb=state(1.0), _state_fn=state_fn)


def _paste_layer(content_image: Image.Image,
                 box: tuple[int, int, int, int],
                 canvas: tuple[int, int]) -> np.ndarray:
    layer = Image.new("RGBA", (canvas[0], canvas[1]), (0, 0, 0, 0))
    layer.alpha_composite(content_image, (box[0], box[1]))
    return np.asarray(layer)


# ---------------------------------------------------------------- paths

APPROACH_START_SCALE = 0.7   # approach settles 0.7 -> 1.0 into the box
PATH_SAMPLES = 17            # swept-area validation / mask resolution


def path_travel_s(span_s: float) -> float:
    """A span's entry-travel budget: PATH_TRAVEL_S, capped at
    PATH_TRAVEL_MAX_FRAC of the span (span-law amendment) so a
    short span keeps a measurable presence core."""
    return min(PATH_TRAVEL_S,
               PATH_TRAVEL_MAX_FRAC * max(0.0, span_s))


def path_progress(t_s: float, start_s: float,
                  travel_s: float = PATH_TRAVEL_S) -> float:
    """Eased travel progress at ``t_s``: 0 at the span's start,
    1 once ``travel_s`` has elapsed, ease-out throughout."""
    if t_s <= start_s:
        return 0.0
    if travel_s <= 0:
        return 1.0
    return _ease_out_cubic(min(1.0, (t_s - start_s) / travel_s))


def path_box_at(box: tuple[int, int, int, int], path: str,
                t_s: float, start_s: float,
                canvas: tuple[int, int],
                travel_s: float = PATH_TRAVEL_S
                ) -> tuple[int, int, int, int]:
    """The graphic's box at time ``t_s`` under its declared path.

    Pure geometry: glide enters from the nearer vertical edge at
    the box's own height; fall enters from the top edge at the
    box's own column; approach scales about the box centre (its
    swept area is inside the final box by construction). Rise
    enters from below - but the canvas bottom is unreachable: the
    caption zone is a hard exclusion no path may cross, so rise
    starts at the floor of the legal backdrop, its box bottom
    resting on the caption zone's ceiling, and rises from there.
    At progress 1 every path returns the placed box itself.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cw, ch = canvas
    q = path_progress(t_s, start_s, travel_s)
    if path == "glide_in":
        start_x = -w if (x0 + x1) / 2.0 < cw / 2.0 else cw
        x = int(round(start_x + (x0 - start_x) * q))
        return (x, y0, x + w, y1)
    if path == "rise_in":
        floor_y = ch - CAPTION_BOTTOM_MARGIN - h
        start_y = max(y0, floor_y)
        y = int(round(start_y + (y0 - start_y) * q))
        return (x0, y, x1, y + h)
    if path == "fall_in":
        y = int(round(-h + (y0 + h) * q))
        return (x0, y, x1, y + h)
    if path == "approach":
        k = APPROACH_START_SCALE + (1.0 - APPROACH_START_SCALE) * q
        nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        return (cx - nw // 2, cy - nh // 2,
                cx - nw // 2 + nw, cy - nh // 2 + nh)
    return box


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def exit_progress(t_s: float, end_s: float,
                  travel_s: float = PATH_TRAVEL_S) -> float:
    """Exit progress at ``t_s``: 0 until the exit window opens
    (``end_s - travel_s``), 1 at the span's end, smoothstep
    between - an exit accelerates gently away instead of snapping
    off."""
    if travel_s <= 0:
        return 0.0
    if t_s <= end_s - travel_s:
        return 0.0
    if t_s >= end_s:
        return 1.0
    return _smoothstep((t_s - (end_s - travel_s)) / travel_s)


def exit_path_box_at(box: tuple[int, int, int, int], exit_path: str,
                     t_s: float, end_s: float,
                     canvas: tuple[int, int],
                     travel_s: float = PATH_TRAVEL_S
                     ) -> tuple[int, int, int, int]:
    """The graphic's box at time ``t_s`` under its declared exit.

    The exact mirror of path_box_at: glide_out leaves by the same
    side glide_in enters from, fall_out rises off the top,
    rise_out sinks back to the caption floor (the canvas bottom
    is unreachable - the caption zone is a hard exclusion),
    recede shrinks about the centre to APPROACH_START_SCALE. At
    progress 0 every exit returns the placed box itself.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cw, ch = canvas
    q = exit_progress(t_s, end_s, travel_s)
    if q <= 0.0:
        return box
    if exit_path == "glide_out":
        start_x = -w if (x0 + x1) / 2.0 < cw / 2.0 else cw
        x = int(round(x0 + (start_x - x0) * q))
        return (x, y0, x + w, y1)
    if exit_path == "rise_out":
        floor_y = ch - CAPTION_BOTTOM_MARGIN - h
        target_y = max(y0, floor_y)
        y = int(round(y0 + (target_y - y0) * q))
        return (x0, y, x1, y + h)
    if exit_path == "fall_out":
        y = int(round(y0 + (-h - y0) * q))
        return (x0, y, x1, y + h)
    if exit_path == "recede":
        k = 1.0 - (1.0 - APPROACH_START_SCALE) * q
        nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        return (cx - nw // 2, cy - nh // 2,
                cx - nw // 2 + nw, cy - nh // 2 + nh)
    return box


def _paste_layer_clipped(content_image: Image.Image,
                         box: tuple[int, int, int, int],
                         canvas: tuple[int, int]) -> np.ndarray:
    """Paste a box-sized RGBA at a possibly off-canvas box.

    Path travel positions the box partially or fully outside the
    canvas; only the intersection is drawn. On a transparent
    ground the visible result is the content itself, clipped -
    identical to _paste_layer for in-canvas boxes.
    """
    cw, ch = canvas
    layer = np.zeros((ch, cw, 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(cw, x1), min(ch, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return layer
    src = np.asarray(content_image)
    layer[cy0:cy1, cx0:cx1] = src[cy0 - y0:cy1 - y0,
                                  cx0 - x0:cx1 - x0]
    return layer


def _box_sweep_legal(free: np.ndarray,
                     boxes: list[tuple[int, int, int, int]],
                     canvas: tuple[int, int]) -> bool:
    """Every on-canvas pixel of every sampled box inside ``free``;
    a box fully off-canvas at a sample constrains nothing there."""
    cw, ch = canvas
    for bx0, by0, bx1, by1 in boxes:
        cx0, cy0 = max(0, bx0), max(0, by0)
        cx1, cy1 = min(cw, bx1), min(ch, by1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if not bool(free[cy0:cy1, cx0:cx1].all()):
            return False
    return True


def _path_legal(free: np.ndarray, box: tuple[int, int, int, int],
                path: str, start_s: float,
                canvas: tuple[int, int],
                travel_s: float = PATH_TRAVEL_S) -> bool:
    """Every on-canvas pixel of the swept entry path inside
    ``free`` (samples the travel at PATH_SAMPLES eased positions,
    plus the resting box, which placement already proved)."""
    boxes = [path_box_at(box, path, start_s + travel_s * k
                         / PATH_SAMPLES, start_s, canvas, travel_s)
             for k in range(PATH_SAMPLES + 1)]
    return _box_sweep_legal(free, boxes, canvas)


def _exit_path_legal(free: np.ndarray, box: tuple[int, int, int, int],
                     exit_path: str, end_s: float,
                     canvas: tuple[int, int],
                     travel_s: float = PATH_TRAVEL_S) -> bool:
    """The exit mirror of _path_legal: every on-canvas pixel of
    the swept exit (sampled across the span's last travel budget)
    inside ``free``."""
    start = end_s - travel_s
    boxes = [exit_path_box_at(box, exit_path,
                              start + travel_s * k / PATH_SAMPLES,
                              end_s, canvas, travel_s)
             for k in range(PATH_SAMPLES + 1)]
    return _box_sweep_legal(free, boxes, canvas)


# ---------------------------------------------------------------- placement


def _span_frame_indices(start_s: float, end_s: float, fps: float,
                        n_frames: int, cap: int = 16) -> list[int]:
    f0 = max(0, int(start_s * fps))
    f1 = min(n_frames, max(f0 + 1, int(end_s * fps)))
    idx = list(range(f0, f1))
    if len(idx) > cap:
        step = len(idx) / cap
        idx = [idx[int(i * step)] for i in range(cap)]
    return idx


def _find_box(free: np.ndarray, edt: np.ndarray,
              size: tuple[int, int],
              face_bbox: tuple[int, int, int, int] | None,
              region: tuple[int, int, int, int],
              *, require_beside: bool = False
              ) -> tuple[int, int, int, int] | None:
    """Best legal top-left position for a (w, h) graphic, or None.

    Legal = every pixel of the box inside the free mask. Ranking is
    deterministic: beside-the-head first (vertically in the
    head/shoulder band, horizontally adjacent to the face - the
    brief's target region; clearance is already guaranteed by the
    free mask itself), then CENTRALITY - distance from the box
    centre to the centroid of the free region, in 24px buckets so
    near-equal centrality falls through - then greatest clearance
    (EDT at the box centre, capped), then smallest horizontal gap
    to the face, and only as final tiebreakers topmost, then
    leftmost. (Block 2: graphics used to hug corners because
    clearance + topmost/leftmost decided almost every placement;
    centrality now outranks both.) A non-beside position still
    wins over dropping the graphic when no beside position is
    legal.
    """
    w, h = size
    rx0, ry0, rx1, ry1 = region
    if w > rx1 - rx0 or h > ry1 - ry0:
        return None
    integral = np.zeros((free.shape[0] + 1, free.shape[1] + 1),
                        dtype=np.int64)
    integral[1:, 1:] = np.cumsum(
        np.cumsum(free.astype(np.int64), axis=0), axis=1)

    def box_free(x: int, y: int) -> bool:
        s = (integral[y + h, x + w] - integral[y, x + w]
             - integral[y + h, x] + integral[y, x])
        return s == w * h

    free_ys, free_xs = np.nonzero(free)
    centroid = None
    if len(free_xs):
        centroid = (float(free_xs.mean()), float(free_ys.mean()))

    best: tuple | None = None
    best_key: tuple | None = None
    for y in range(ry0, ry1 - h + 1, GRID_STEP_PX):
        for x in range(rx0, rx1 - w + 1, GRID_STEP_PX):
            if not box_free(x, y):
                continue
            clearance = float(edt[y + h // 2, x + w // 2])
            clearance = min(clearance, 80.0)
            beside, gap = 1, 10_000
            if face_bbox is not None:
                fx0, fy0, fx1, fy1 = face_bbox
                band_lo, band_hi = fy0 - 60, fy1 + 300
                in_band = band_lo <= y + h // 2 <= band_hi
                if x + w <= fx0:
                    gap = fx0 - (x + w)
                elif x >= fx1:
                    gap = x - fx1
                else:
                    gap = 0
                beside = 0 if (in_band and gap > 0) else 1
            centrality = 0
            if centroid is not None:
                dx = x + w / 2.0 - centroid[0]
                dy = y + h / 2.0 - centroid[1]
                centrality = int((dx * dx + dy * dy) ** 0.5 // 24)
            if require_beside and beside != 0:
                continue
            key = (beside, centrality, -clearance, gap, y, x)
            if best_key is None or key < best_key:
                best_key = key
                best = (x, y, x + w, y + h)
    return best


# Size-to-space ladder (block 2): try the largest legal rendering
# first - the aim is a graphic occupying a substantial share of
# the canvas width whenever a legal region allows - then descend
# geometrically until a legal box exists or the legibility floor.
GRAPHIC_TARGET_WIDTH_FRAC = 0.60   # aim: 60% of canvas width
GRAPHIC_LADDER_STEP = 0.85         # geometric descent per retry
GRAPHIC_MIN_WIDTH_PX = 160         # below this, text is illegible


def _candidate_widths(content: GraphicContent,
                      canvas_w: int) -> list[int]:
    native_w = content.size[0]
    start = max(native_w,
                int(round(GRAPHIC_TARGET_WIDTH_FRAC * canvas_w)))
    if content.max_size is not None:
        start = min(start, content.max_size[0])
    floor = max(GRAPHIC_MIN_WIDTH_PX, int(round(native_w * 0.45)))
    widths: list[int] = []
    w = start
    while w >= floor:
        widths.append(w)
        w = int(w * GRAPHIC_LADDER_STEP)
    if floor <= native_w <= start and native_w not in widths:
        widths.append(native_w)
        widths.sort(reverse=True)
    return widths or [native_w]


def _place_sized(free: np.ndarray, edt: np.ndarray,
                 content: GraphicContent,
                 face_bbox: tuple[int, int, int, int] | None,
                 region: tuple[int, int, int, int],
                 canvas_w: int
                 ) -> tuple[tuple[int, int, int, int],
                            GraphicContent] | None:
    """(box, content-at-that-size) for the largest legal size.

    Two passes over the size ladder: beside-the-head placements
    at any ladder size first (the brief's top soft preference
    outranks raw size), then any legal placement. Content without
    a resizer is tried at its native size only. A size whose
    height would exceed the content's upscale cap (images: 2.0x
    the source) is skipped - a smaller legal region that fits the
    achievable size wins instead.
    """
    prepared: list[tuple[int, GraphicContent]] = []
    for w in _candidate_widths(content, canvas_w):
        if w == content.size[0]:
            sized = content
        elif content.resizer is not None:
            sized = content.resizer(w)
        else:
            continue
        if (content.max_size is not None
                and sized.size[1] > content.max_size[1]):
            continue
        prepared.append((w, sized))
    if face_bbox is not None:
        for _w, sized in prepared:
            box = _find_box(free, edt, sized.size, face_bbox, region,
                            require_beside=True)
            if box is not None:
                return box, sized
    for _w, sized in prepared:
        box = _find_box(free, edt, sized.size, face_bbox, region)
        if box is not None:
            return box, sized
    return None


SPLIT_GRAPHIC_ZONE_FRAC = 0.65  # Split: graphics own the top 65%


def _box_all_free(free: np.ndarray,
                  box: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > free.shape[1] or y1 > free.shape[0]:
        return False
    return bool(free[y0:y1, x0:x1].all())


def _place_spotlight(free: np.ndarray, content: GraphicContent,
                     canvas: tuple[int, int], caption_top: int
                     ) -> tuple[tuple[int, int, int, int],
                                GraphicContent] | None:
    """The Spotlight target box: centre-majority, centred in the
    free band above the caption zone.

    Deterministic by design (the state IS the graphic at centre
    stage): the content is sized to the ladder's target width and
    centred horizontally; vertically it starts at the band centre
    and slides - down first, then up, in 8px steps - to the first
    position whose every pixel is inside the caller's free mask.
    Against layout-aware alphas that means clear of the PiP
    speaker (which owns a top corner, so the slide is what makes
    centre stage reachable at all), the face margin, the hook
    zone and the caption zone. None when no position is legal or
    the content cannot reach majority width; the caller falls
    back to the standard search and the declared state stands on
    the record for the verifier to measure.
    """
    cw, _ch = canvas
    target_w = int(round(GRAPHIC_TARGET_WIDTH_FRAC * cw))
    if content.resizer is not None and target_w != content.size[0]:
        sized = content.resizer(target_w)
    else:
        sized = content
    if content.max_size is not None \
            and sized.size[1] > content.max_size[1]:
        return None
    w, h = sized.size
    if w < int(0.55 * cw) or h > caption_top - CANVAS_MARGIN_PX:
        return None
    x0 = (cw - w) // 2
    y_mid = CANVAS_MARGIN_PX + (caption_top - CANVAS_MARGIN_PX - h) // 2
    y_max = caption_top - h
    candidates = [y_mid]
    step = 8
    k = step
    while y_mid + k <= y_max:
        candidates.append(y_mid + k)
        k += step
    k = step
    while y_mid - k >= CANVAS_MARGIN_PX:
        candidates.append(y_mid - k)
        k += step
    for y0 in candidates:
        box = (x0, y0, x0 + w, y0 + h)
        if _box_all_free(free, box):
            return box, sized
    return None


def _place_cutaway(content: GraphicContent,
                   canvas: tuple[int, int]
                   ) -> tuple[tuple[int, int, int, int],
                              GraphicContent] | None:
    """The Cutaway box: the content at full-frame scale, centred.

    No free-mask search - during a cutaway span the speaker is
    absent by declaration, so the Stage occupancy the free mask
    encodes does not apply; Stage 3's absence leg is what verifies
    the state. The content is sized to the canvas width (or, if
    that overflows the height, to fit the height), centred both
    ways over the treated backdrop.
    """
    cw, ch = canvas
    sized = content
    if content.resizer is not None:
        sized = content.resizer(cw)
        if sized.size[1] > ch:
            sized = content.resizer(
                max(GRAPHIC_MIN_WIDTH_PX,
                    int(cw * ch / sized.size[1])))
    w, h = sized.size
    if w > cw or h > ch:
        return None
    x0, y0 = (cw - w) // 2, (ch - h) // 2
    return (x0, y0, x0 + w, y0 + h), sized


def _spec_layers(content: GraphicContent,
                 box: tuple[int, int, int, int],
                 canvas: tuple[int, int],
                 path: str | None,
                 exit_path: str | None,
                 travel: float) -> tuple[LayerFn, np.ndarray]:
    """(layer_fn, resting layer) for a placed spec.

    Factored out of place_prepared (fix round 2) so the Spotlight
    refit in plan_layout rebuilds a re-fitted spec's layers
    through exactly the same code the original placement used.
    """
    def layer_fn(t: float, start: float, end: float,
                 _content: GraphicContent = content,
                 _box: tuple[int, int, int, int] = box,
                 _canvas: tuple[int, int] = canvas,
                 _path: str | None = path,
                 _exit: str | None = exit_path,
                 _travel: float = travel) -> np.ndarray:
        if end > start:
            p = min(1.0, max(0.0, (t - start) / (end - start)))
        else:
            p = 1.0
        img = _content.state_fn(p) if _content.state_fn \
            else _content.final_image
        # Entry and exit windows are disjoint by the travel
        # cap (25% of span each), so at most one applies.
        if _exit is not None and t > end - _travel:
            cur = exit_path_box_at(_box, _exit, t, end,
                                   _canvas, _travel)
        elif _path is not None:
            cur = path_box_at(_box, _path, t, start, _canvas,
                              _travel)
        else:
            return _paste_layer(img, _box, _canvas)
        cur_size = (cur[2] - cur[0], cur[3] - cur[1])
        if cur_size != img.size:
            img = img.resize(cur_size, Image.BILINEAR)
        return _paste_layer_clipped(img, cur, _canvas)

    img = content.state_fn(1.0) if content.state_fn \
        else content.final_image
    return layer_fn, _paste_layer(img, box, canvas)


def place_prepared(
    items: list[tuple[GraphicSpan, GraphicContent]],
    canvas_alphas: Sequence[np.ndarray],
    *,
    fps: float,
    caption_rect: tuple[int, int, int, int] | None = None,
    hook_glyph_bbox: tuple[int, int, int, int] | None = None,
    hook_window: tuple[float, float] | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    path_rejections: list[dict[str, Any]] | None = None,
) -> tuple[list[GraphicSpec], list[dict[str, Any]]]:
    """Place prepared graphics or drop them, with reasons.

    This function owns ONLY the geometry: the hard exclusions and
    the search. A span whose graphic fits nowhere legal is dropped
    with the reason - the layer never trades the exclusions for
    coverage. A span whose declared PATH fits nowhere legal keeps
    its placement but loses the path (recorded in
    ``path_rejections`` when a list is supplied): the swept area
    is validated against the same free mask as the resting box,
    plus the swept areas of already-placed path graphics whose
    spans overlap in time.
    """
    cw, ch = canvas
    caption_top = ch - CAPTION_BOTTOM_MARGIN
    placed: list[GraphicSpec] = []
    dropped: list[dict[str, Any]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    n_frames = len(canvas_alphas)

    for span, content in items:
        idx = _span_frame_indices(span.start_s, span.end_s, fps,
                                  n_frames)
        if not idx:
            dropped.append({"span": span.to_dict(),
                            "reason": "span has no frames to place "
                                      "against"})
            continue
        samples = [canvas_alphas[i] for i in idx]
        occupancy = np.mean(
            [(a > 0.5).astype(np.float64) for a in samples], axis=0)
        face_union = np.zeros((ch, cw), dtype=bool)
        for a in samples:
            face_union |= face_pixels(a)
        face_bbox = None
        if face_union.any():
            ys, xs = np.nonzero(face_union)
            face_bbox = (int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1)
        forbidden = np.zeros((ch, cw), dtype=bool)
        if face_union.any():
            dist = ndimage.distance_transform_edt(~face_union)
            forbidden |= dist <= FACE_MARGIN_PX
        forbidden[caption_top:, :] = True
        if caption_rect is not None:
            x0, y0, x1, y1 = caption_rect
            forbidden[y0:y1, x0:x1] = True
        if (hook_glyph_bbox is not None and hook_window is not None
                and span.start_s < hook_window[1]
                and span.end_s > hook_window[0]):
            hx0, hy0, hx1, hy1 = hook_glyph_bbox
            forbidden[max(0, hy0 - 12):hy1 + 12,
                      max(0, hx0 - 12):hx1 + 12] = True
        for bx0, by0, bx1, by1 in placed_boxes:
            forbidden[max(0, by0 - PLACED_GAP_PX):by1 + PLACED_GAP_PX,
                      max(0, bx0 - PLACED_GAP_PX):bx1 + PLACED_GAP_PX] \
                = True
        # Swept areas of already-placed PATH graphics whose spans
        # overlap this one in time: a travelling graphic owns its
        # whole corridor, not just its resting box. (mask() sweeps
        # entry and exit travel alike.)
        for prev in placed:
            if prev.path is None and prev.exit_path is None:
                continue
            if not (prev.start_s < span.end_s
                    and span.start_s < prev.end_s):
                continue
            swept = prev.mask()
            if swept.any():
                dist = ndimage.distance_transform_edt(~swept)
                forbidden |= dist <= PLACED_GAP_PX
        free = (occupancy <= FREE_OCCUPANCY_MAX) & ~forbidden
        edt = ndimage.distance_transform_edt(free)
        region = (CANVAS_MARGIN_PX, CANVAS_MARGIN_PX,
                  cw - CANVAS_MARGIN_PX, caption_top)
        if span.layout == "cutaway":
            sized = _place_cutaway(content, canvas)
        elif span.layout == "spotlight":
            sized = _place_spotlight(free, content, canvas,
                                     caption_top)
            if sized is None:
                sized = _place_sized(free, edt, content, face_bbox,
                                     region, cw)
        elif span.layout == "split":
            sized = _place_sized(
                free, edt, content, face_bbox,
                (region[0], region[1], region[2],
                 round(ch * SPLIT_GRAPHIC_ZONE_FRAC)), cw)
        else:
            sized = _place_sized(free, edt, content, face_bbox,
                                 region, cw)
        if sized is None:
            dropped.append({
                "span": span.to_dict(),
                "reason": "no legal placement: every candidate "
                          "position hits the face margin, the "
                          "caption zone, the hook zone, or speaker "
                          "occupancy"})
            continue
        box, sized_content = sized
        if sized_content is not content:
            # Re-preparation at the placed size must not lose the
            # generation cost the caller recorded on the content.
            sized_content.cost_usd = content.cost_usd
            content = sized_content

        # Path validation: the declared travel must stay inside
        # the same free mask at every sampled instant. A refused
        # path never costs the graphic its placement - it appears
        # in place, and the refusal is recorded.
        path = span.path
        travel = path_travel_s(span.end_s - span.start_s)
        if path is not None and not _path_legal(
                free, box, path, span.start_s, canvas, travel):
            if path_rejections is not None:
                path_rejections.append({
                    "span": span.to_dict(), "path": path,
                    "reason": "declared path crosses a forbidden "
                              "zone somewhere along its travel "
                              "(face + margin, caption zone, hook "
                              "glyph zone, speaker occupancy, or "
                              "another graphic's swept area); the "
                              "graphic is placed in place instead"})
            path = None
        # Exit validation (Stage 4): the mirror discipline - the
        # swept exit must stay inside the same free mask. A
        # refused exit never costs the placement either.
        exit_path = span.exit_path
        if exit_path is not None and not _exit_path_legal(
                free, box, exit_path, span.end_s, canvas, travel):
            if path_rejections is not None:
                path_rejections.append({
                    "span": span.to_dict(), "path": exit_path,
                    "phase": "exit",
                    "reason": "declared exit path crosses a "
                              "forbidden zone somewhere along its "
                              "travel (face + margin, caption "
                              "zone, hook glyph zone, speaker "
                              "occupancy, or another graphic's "
                              "swept area); the graphic is placed "
                              "in place instead"})
            exit_path = None

        layer_fn, resting = _spec_layers(content, box, canvas, path,
                                         exit_path, travel)

        placed.append(GraphicSpec(
            kind=span.kind, start_s=span.start_s, end_s=span.end_s,
            box=box,
            layer_rgba=resting,
            source_quote=span.source_quote, prompt=span.prompt,
            headline=span.headline, facts=span.facts,
            compare=span.compare, chart=span.chart,
            collage=span.collage,
            motion=span.motion,
            cost_usd=content.cost_usd, _state_fn=layer_fn,
            _content=content,
            path=path, exit_path=exit_path,
            layout=span.layout, camera=span.camera,
            anchor_quality=span.anchor_quality,
            layout_assist=span.layout_assist))
        placed_boxes.append(box)
    return placed, dropped


# ---------------------------------------------------------------- build


def build_graphics(
    spark: SparkClient,
    image_client: ImageClient | None,
    cues: Sequence[CaptionCue],
    canvas_alphas: Sequence[np.ndarray],
    *,
    fps: float,
    caption_rect: tuple[int, int, int, int] | None = None,
    hook_glyph_bbox: tuple[int, int, int, int] | None = None,
    hook_window: tuple[float, float] | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    duration_s: float,
    ts: str = "",
    cached_images: Sequence[Image.Image] | None = None,
) -> GraphicsBuild:
    """Plan, generate, and place one Short's context graphics.

    Every failure mode lands on the record as a drop or the build's
    error field; this function raises only on programming errors.
    Spend is one planner call plus one flat-$0.01 image call per
    image span the (uncapped) planner earned.
    """
    outcome = plan_graphics(spark, cues, duration_s=duration_s, ts=ts)
    build = GraphicsBuild(
        planner_call=outcome.spark_call,
        planner_rejected=outcome.rejected,
        error=outcome.error, duration_s=duration_s)
    # One hero per moment (block 2): backdrop and speaker_forward
    # spans are exclusive heroes - resolve overlaps BEFORE any
    # generation spend, dropping the later-starting claim.
    hero_spans, hero_drops = resolve_heroes(
        outcome.spans, hook_window=hook_window)
    build.dropped.extend(hero_drops)
    # Deterministic layout assist (fix phase, 2026-11-02):
    # hero card/image spans the planner left at Stage are
    # assigned Spotlight HERE - before generation and placement -
    # so the Spotlight placement branch sizes them centre-stage
    # like planner-declared spans, and the marker rides onto the
    # placed spec for plan_layout's declaration record.
    _layout_assist(hero_spans)
    items: list[tuple[GraphicSpan, GraphicContent]] = []
    for span in hero_spans:
        if span.kind == "speaker_forward":
            # No generation: the pipeline applies the emphasis
            # framing to the speaker layer itself.
            build.speaker_forward.append(span)
            continue
        if span.kind == "card":
            items.append((span, prepare_card(span, cues)))
            continue
        if span.collage:
            # Tier B: a collage is built from CACHED stills the
            # caller supplies - never from fresh generation. Too
            # few stills = a recorded drop, not a smaller grid
            # pretending to be the planned one.
            stills = list(cached_images or [])[:4]
            if len(stills) < 2:
                build.dropped.append({
                    "span": span.to_dict(),
                    "reason": "collage span but fewer than 2 "
                              "cached stills were supplied"})
                continue
            items.append((span, prepare_collage(stills, span)))
            continue
        if image_client is None:
            build.dropped.append({
                "span": span.to_dict(),
                "reason": f"{span.kind} span but no image client "
                          f"is wired"})
            continue
        sent_prompt = span.prompt + IMAGE_PROMPT_SUFFIX
        try:
            result = image_client.generate(
                sent_prompt, purpose="graphics-layer-image", ts=ts)
            image = Image.open(io.BytesIO(result.image_bytes))
            image.load()
        except Exception as exc:
            build.dropped.append({
                "span": span.to_dict(),
                "reason": f"image generation failed: {exc}"[:300]})
            continue
        # The receipt records the prompt exactly as sent (suffix
        # included) - provenance is what was asked of the model.
        span = GraphicSpan(
            start_s=span.start_s, end_s=span.end_s, kind=span.kind,
            headline=span.headline, facts=span.facts,
            prompt=sent_prompt, source_quote=span.source_quote,
            compare=span.compare, motion=span.motion,
            path=span.path, exit_path=span.exit_path,
            layout=span.layout, camera=span.camera,
            anchor_quality=span.anchor_quality)
        if span.kind == "backdrop":
            spec = prepare_backdrop(image.convert("RGB"), canvas)
            spec = dataclasses.replace(
                spec, start_s=span.start_s, end_s=span.end_s,
                prompt=span.prompt, source_quote=span.source_quote,
                cost_usd=result.cost_usd)
            build.backdrops.append(spec)
            continue
        content = prepare_image(image.convert("RGB"), span)
        content.cost_usd = result.cost_usd
        items.append((span, content))
    placed, dropped = place_prepared(
        items, canvas_alphas, fps=fps, caption_rect=caption_rect,
        hook_glyph_bbox=hook_glyph_bbox, hook_window=hook_window,
        canvas=canvas, path_rejections=build.path_rejections)
    build.placed = placed
    build.dropped.extend(dropped)
    _speaker_forward_assist(build, cues, hook_window=hook_window)
    return build


def _speaker_forward_assist(
    build: GraphicsBuild,
    cues: Sequence[CaptionCue],
    *,
    hook_window: tuple[float, float] | None,
) -> None:
    """Deterministic speaker-forward assist (fix phase, 2026-11-02).

    The emphasis half of the layout assist: the planner proposed
    speaker_forward spans in test fixtures but never live (both
    2026-11-01/02 extension runs recorded speaker_forward: []),
    so a Short whose planner offered no emphasis gets one
    synthesized span over its strongest speaker-claim cue, fed
    through the EXISTING pipeline emphasis machinery - feasibility
    (plan_speaker_emphasis) and its honest fallback recording are
    unchanged; this only guarantees the machinery is exercised.

    "Strongest claim" is deterministic: a cue carrying a stated
    number or a question outranks plain speech, then the longest
    cue wins, then the earliest. Cues overlapping the hook window
    or any placed graphic/backdrop are ineligible (emphasis is a
    payoff treatment and one-hero-at-any-instant holds), as are
    cues under the 1.0s framing-span floor. The synthesis is
    recorded on build.layout_assist; nothing is invented when no
    cue qualifies - the build simply carries no emphasis span.
    """
    if build.speaker_forward or not cues:
        return
    heroes = ([(g.start_s, g.end_s) for g in build.placed]
              + [(b.start_s, b.end_s) for b in build.backdrops])

    def claim_key(cue: CaptionCue) -> tuple[int, float, float]:
        text = cue.text or ""
        strong = bool(numeric_values(text)) or "?" in text
        return (1 if strong else 0, cue.end_s - cue.start_s,
                -cue.start_s)

    for cue in sorted(cues, key=claim_key, reverse=True):
        if cue.end_s - cue.start_s < 1.0:
            continue
        if hook_window is not None and _spans_overlap(
                cue.start_s, cue.end_s,
                hook_window[0], hook_window[1]):
            continue
        if any(_spans_overlap(cue.start_s, cue.end_s, h0, h1)
               for h0, h1 in heroes):
            continue
        build.speaker_forward.append(GraphicSpan(
            start_s=cue.start_s, end_s=cue.end_s,
            kind="speaker_forward", source_quote=cue.text))
        build.layout_assist.append({
            "span": [round(cue.start_s, 3), round(cue.end_s, 3)],
            "requested": "speaker_forward",
            "applied": "speaker_forward",
            "reason": "planner proposed no speaker_forward span; "
                      "strongest speaker-claim cue selected "
                      "deterministically (number/question claim, "
                      "then longest, then earliest); pipeline "
                      "emphasis feasibility and fallback recording "
                      "apply unchanged"})
        return


# ---------------------------------------------------------------- layout track


def _boxes_intersect(a: Sequence[int], b: Sequence[int]) -> bool:
    return (min(a[2], b[2]) - max(a[0], b[0]) > 0
            and min(a[3], b[3]) - max(a[1], b[1]) > 0)


LAYOUT_ASSIST_MIN_SPAN_S = 2.5


def _spans_overlap(a_start: float, a_end: float,
                   b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) > 0


def _layout_assist(objs: Sequence[Any],
                   ) -> dict[int, dict[str, Any]]:
    """Deterministic layout assist (fix phase, 2026-11-02).

    Measured gap (live runs pipeline-run-ext-qh2 / -qm3): the
    model planner assigned NO layout state live - every placed
    graphic stayed "stage" and got leftover side space (a 400px
    image at x=48 on the McArthur run), because the prompt's
    "when in doubt, leave it stage" guidance always won. The
    states themselves are proven (S0-S3); what was missing is a
    deterministic floor under the planner's discretion.

    A card/image span still at "stage" whose span is at least
    LAYOUT_ASSIST_MIN_SPAN_S is a hero span - its graphic IS the
    span's primary content (backdrop and speaker_forward spans
    are framing, never graphics here) - so it is assigned
    "spotlight" and marked (``layout_assist`` on the object).
    The assignment happens in build_graphics, BEFORE placement,
    so an assisted graphic gets the Spotlight placement branch's
    centre-stage box exactly like a planner-declared one; the
    existing plan_layout feasibility machinery then decides:
    feasible -> the declaration carries the assist record with
    applied="spotlight"; infeasible -> the honest Stage
    downgrade it already performs, with the record's
    applied="stage" and the planner's reason. plan_layout also
    runs this over placed specs directly, covering specs that
    never passed through build_graphics. No extra model call;
    same inputs, same assignment, so pipeline and rescue
    re-plans agree.

    One hero at any instant still holds: a span that overlaps a
    span already owning framing (planner-declared non-Stage, or
    an earlier assisted span) is left at Stage. Planner-declared
    cutaway/spotlight spans are never touched - the model keeps
    its freedom where it actually exercised it.

    Fix round 4 adds ONE exception to "declared states are never
    touched": a planner-declared SPLIT for a NON-compare graphic.
    Split is the two-subject state - its geometry gives the
    speaker a same-weight band beside the graphic. For a single
    card/image that is exactly the subordinate-graphic failure
    Kamlesh flagged on the QM5 output (a 285px graphic beside a
    430px PiP inside a span declared "spotlight"; the split form
    of the same inversion is a planner split on a lone graphic).
    A non-compare split is therefore converted to Spotlight (the
    hero geometry: graphic on top, PiP below) and the conversion
    is recorded in layout_assist with the reason, like every
    other assist action. A COMPARE card keeps its genuine Split -
    two compared items beside the speaker is what Split is for.
    """
    assisted: dict[int, dict[str, Any]] = {}
    occupied = [(s.start_s, s.end_s) for s in objs
                if s.layout != "stage"]
    for obj in objs:
        if obj.layout != "split":
            continue
        if obj.kind not in ("card", "image"):
            continue
        if getattr(obj, "compare", None):
            continue
        record = {
            "span": [round(obj.start_s, 3), round(obj.end_s, 3)],
            "requested": "split",
            "reason": "deterministic layout assist (fix round 4): "
                      "split requested for a non-compare graphic - "
                      "a lone graphic in split sits beside a "
                      "same-weight speaker and reads subordinate; "
                      "converted to spotlight (graphic hero on "
                      "top, speaker PiP below)"}
        obj.layout = "spotlight"
        obj.layout_assist = record
        assisted[id(obj)] = record
    for obj in objs:
        if obj.layout != "stage":
            continue
        if obj.kind not in ("card", "image"):
            continue
        if obj.end_s - obj.start_s < LAYOUT_ASSIST_MIN_SPAN_S - 1e-9:
            continue
        if any(_spans_overlap(obj.start_s, obj.end_s, o0, o1)
               for o0, o1 in occupied):
            continue
        record = {
            "span": [round(obj.start_s, 3), round(obj.end_s, 3)],
            "requested": "spotlight"}
        obj.layout = "spotlight"
        obj.layout_assist = record
        assisted[id(obj)] = record
        occupied.append((obj.start_s, obj.end_s))
    return assisted


SPOTLIGHT_REFIT_GAP_PX = 40  # card top >= PiP bottom + this gap


def _refit_spotlight_card(
    spec: GraphicSpec,
    pip_plan: Any,
    specs: Sequence[GraphicSpec],
    canvas: tuple[int, int],
) -> dict[str, Any] | None:
    """Re-fit a Spotlight span's card around a feasible PiP.

    Fix round 2 (Fix C): placement runs against Stage geometry,
    so a centre-stage card (648px wide) used to collide with BOTH
    PiP corners and the span downgraded to Stage - Spotlight was
    requested 4x live and refused 4x. The refit sizes/positions
    the card FOR the PiP before feasibility is declared: first at
    full size in the band fully below the PiP (top >= PiP bottom
    + 40), then down the width ladder (the ladder's floor is the
    legibility floor - a card is never shrunk below it). The
    spec's box/layers are rebuilt through _spec_layers, so the
    render, the declaration and F8 all see the re-fitted box.
    Returns the refit record, or None when no legal fit exists
    (the span then downgrades honestly, as before). Specs without
    retained content, or with a declared travel path, cannot be
    re-fitted - a baked layer cannot be re-rendered at a new
    size, and a path's swept area was validated for the old box.
    """
    content = spec._content
    if content is None or spec.path is not None \
            or spec.exit_path is not None:
        return None
    cw, ch = canvas
    caption_top = ch - CAPTION_BOTTOM_MARGIN
    pip_box = pip_plan.speaker_box()
    others = [s.box for s in specs if s is not spec]

    def clear(box: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = box
        if x0 < CANVAS_MARGIN_PX or x1 > cw - CANVAS_MARGIN_PX \
                or y0 < CANVAS_MARGIN_PX or y1 > caption_top:
            return False
        if _boxes_intersect(box, pip_box):
            return False
        for ob in others:
            inflated = (ob[0] - PLACED_GAP_PX, ob[1] - PLACED_GAP_PX,
                        ob[2] + PLACED_GAP_PX, ob[3] + PLACED_GAP_PX)
            if _boxes_intersect(box, inflated):
                return False
        return True

    def apply(sized: Any, box: tuple[int, int, int, int],
              strategy: str) -> dict[str, Any]:
        record = {"from_box": [int(v) for v in spec.box],
                  "to_box": [int(v) for v in box],
                  "strategy": strategy,
                  "pip_box": [int(v) for v in pip_box]}
        if sized is not content:
            sized.cost_usd = content.cost_usd
        travel = path_travel_s(spec.end_s - spec.start_s)
        layer_fn, resting = _spec_layers(sized, box, canvas, None,
                                         None, travel)
        spec.box = box
        spec.layer_rgba = resting
        spec._state_fn = layer_fn
        spec._content = sized
        return record

    w0 = spec.box[2] - spec.box[0]
    h0 = spec.box[3] - spec.box[1]
    y_below = pip_box[3] + SPOTLIGHT_REFIT_GAP_PX
    # (a) Full size below the PiP: keep the placed x, then centred.
    for x0 in (spec.box[0], (cw - w0) // 2):
        cand = (x0, y_below, x0 + w0, y_below + h0)
        if clear(cand):
            return apply(content, cand, "below_pip_full_size")
    # (b) The width ladder below the PiP, centred.
    if content.resizer is not None:
        for w in _candidate_widths(content, cw):
            if w >= w0:
                continue
            sized = content.resizer(w)
            sw_, sh_ = sized.size
            x0 = (cw - sw_) // 2
            cand = (x0, y_below, x0 + sw_, y_below + sh_)
            if clear(cand):
                return apply(sized, cand, "below_pip_resized")
    return None


SPOTLIGHT_HERO_GAP_PX = 40       # hero bottom -> PiP top gap
SPOTLIGHT_HERO_MIN_WIDTH_FRAC = 0.55   # hero width target floor
SPOTLIGHT_HERO_AREA_DOMINANCE = 0.92   # PiP area <= hero area x this
SPOTLIGHT_HERO_IMAGE_ASPECT = 0.75     # hero box h/w for images
SPOTLIGHT_HERO_MIN_BAND_PX = 200       # smallest PiP band worth solving
# Fix round 5 group composition (Kamlesh's QM6 verdict: the
# round-4 solve pinned the hero to the frame's top edge at its
# native width and left the PiP whatever band remained - a
# 345px dead band and a speaker shrunk past need). Hero + PiP
# are now composed as ONE group filling the band above the
# caption zone: the group starts SPOTLIGHT_GROUP_TOP_PX down,
# hero and PiP are separated by SPOTLIGHT_GROUP_GAP_PX, the
# hero may grow to SPOTLIGHT_GROUP_HERO_WIDTH_FRAC of the
# frame width (cover-fit), and among the feasible widths the
# winner maximises the PiP's placed height first and the
# hero's area second - the speaker gets every pixel the caps
# allow before the graphic takes more.
SPOTLIGHT_GROUP_TOP_PX = 100           # group top offset
SPOTLIGHT_GROUP_GAP_PX = 48            # hero bottom -> PiP top
SPOTLIGHT_GROUP_HERO_WIDTH_FRAC = 0.85  # hero width ceiling
# The group's hero floor: widths at or above this fraction are
# the preferred pool - the speaker is maximised among them,
# and narrower heroes are only used when no wide hero solves
# (the brief's ordering: hero first, then the PiP fills what
# remains under the caps). 0.74 x 1080 = the 800px floor.
SPOTLIGHT_GROUP_HERO_MIN_WIDTH_FRAC = 0.74
# Spans that could overlap the hook window keep the placement
# place_prepared validated against the hook glyph zone (plan_layout
# receives no hook inputs, so it must not move a graphic into the
# zone it never saw). Hero-first applies to the normal mid-Short
# spans - the QM5 hamburger span starts at 10.02s.
SPOTLIGHT_HERO_MIN_START_S = 3.0


def _apply_spotlight_hero(
    spec: GraphicSpec,
    sized: Any,
    box: tuple[int, int, int, int],
    pip_box: Sequence[int],
    canvas: tuple[int, int],
) -> dict[str, Any]:
    """Rebuild a spec's layers at its hero box (the refit's apply
    discipline, shared shape): the render, the declaration and F8
    all see the hero box because the spec itself moves."""
    content = spec._content
    record = {"from_box": [int(v) for v in spec.box],
              "to_box": [int(v) for v in box],
              "strategy": "hero_first",
              "pip_box": [int(v) for v in pip_box]}
    if sized is not content:
        sized.cost_usd = content.cost_usd
    travel = path_travel_s(spec.end_s - spec.start_s)
    layer_fn, resting = _spec_layers(sized, box, canvas, None,
                                     None, travel)
    spec.box = box
    spec.layer_rgba = resting
    spec._state_fn = layer_fn
    spec._content = sized
    return record


def _solve_spotlight_hero(
    spec: GraphicSpec,
    span_alphas: Sequence[np.ndarray],
    specs: Sequence[GraphicSpec],
    canvas: tuple[int, int],
) -> tuple[Any, dict[str, Any]] | None:
    """Hero-first Spotlight solve (fix round 4; group
    composition fix round 5): (plan, record).

    Solves the GRAPHIC box first - the majority region at the top
    of the frame - then the speaker PiP below it via
    plan_spotlight_below, under the dominance caps (PiP no wider
    than the hero, PiP area <= SPOTLIGHT_HERO_AREA_DOMINANCE x
    hero area). Round 4 pinned the hero to the frame's top edge
    at the ladder's first width and stranded a dead band above
    a shrunken PiP (Kamlesh's QM6 verdict); round 5 composes
    hero + PiP as one group: the group starts
    SPOTLIGHT_GROUP_TOP_PX into the frame, the hero may grow to
    SPOTLIGHT_GROUP_HERO_WIDTH_FRAC of the canvas (widths from
    the existing candidate ladder, whose cap is the content's
    max_size and whose floor is the legibility floor), the gap
    is SPOTLIGHT_GROUP_GAP_PX, and among the feasible widths
    the winner maximises the PiP's placed height, then the
    hero's area. Images are cover-fit to the region box via the
    content's box_resizer (the crop is the region - declared on
    the record); cards/charts keep their aspect through the
    width resizer, so their legibility floor is the ladder's.

    Returns None when the hero arrangement is not available -
    no retained content, a declared travel path (its swept area
    was validated for the placed box), a span inside the hook
    window, no speaker geometry, or no width/PiP pair solves -
    and the caller runs the historic corner/refit logic, whose
    honest downgrade still waits at the end of that path.
    """
    content = spec._content
    if content is None or spec.path is not None \
            or spec.exit_path is not None:
        return None
    if spec.start_s < SPOTLIGHT_HERO_MIN_START_S:
        return None
    cw, _ch = canvas
    caption_top = initial_caption_rect(canvas)[1]
    stats = SpanStats.from_alphas(span_alphas)
    if stats is None:
        return None
    others = [s.box for s in specs if s is not spec]

    def clear(box: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = box
        if x0 < CANVAS_MARGIN_PX or x1 > cw - CANVAS_MARGIN_PX \
                or y0 < CANVAS_MARGIN_PX or y1 > caption_top:
            return False
        for ob in others:
            inflated = (ob[0] - PLACED_GAP_PX, ob[1] - PLACED_GAP_PX,
                        ob[2] + PLACED_GAP_PX, ob[3] + PLACED_GAP_PX)
            if _boxes_intersect(box, inflated):
                return False
        return True

    min_w = int(round(SPOTLIGHT_HERO_MIN_WIDTH_FRAC * cw))
    hero_cap = int(round(SPOTLIGHT_GROUP_HERO_WIDTH_FRAC * cw))
    ladder = _candidate_widths(content, cw)
    # The ladder never proposes above the content's native width
    # (its start is max(native, 60% of canvas)); the group target
    # is a real candidate whenever the content's own upscale cap
    # (max_size) permits it - a cover-fit image or a re-rendered
    # card can both be produced at the cap width.
    extra = {hero_cap} if content.max_size is None \
        or hero_cap <= content.max_size[0] else set()
    # The ladder's 0.85 steps are too coarse for the group
    # optimum: the PiP's placed height is non-monotone in the
    # hero width (a wider hero shrinks the band the PiP scales
    # into), and on the QM6 geometry the best width (~860) sits
    # between ladder rungs (648 and the 918 cap). A dense sweep
    # at 0.94 steps covers the range; every candidate is one
    # cheap solver call on shared span stats.
    dense = set()
    w = hero_cap
    while w >= min_w:
        dense.add(w)
        w = int(w * 0.94)
    cand = sorted({x for x in ladder if min_w <= x <= hero_cap}
                  | extra | dense, reverse=True)
    if not cand:
        cand = [ladder[0]] if ladder else []
    top = SPOTLIGHT_GROUP_TOP_PX
    hero_floor = int(round(
        SPOTLIGHT_GROUP_HERO_MIN_WIDTH_FRAC * cw))
    feas: list[tuple[tuple[int, int], int, Any, Any,
                     tuple[int, int, int, int],
                     tuple[int, int]]] = []
    for w in cand:
        if content.max_size is not None and w > content.max_size[0]:
            continue
        if content.box_resizer is not None:
            h = max(1, int(round(
                w * SPOTLIGHT_HERO_IMAGE_ASPECT)))
            sized = content.box_resizer(w, h)
        elif content.resizer is not None:
            if w == content.size[0]:
                sized = content
            else:
                sized = content.resizer(w)
            h = sized.size[1]
        else:
            if w != content.size[0]:
                continue
            sized = content
            h = content.size[1]
        if content.max_size is not None \
                and sized.size[1] > content.max_size[1]:
            continue
        x0 = (cw - w) // 2
        hero = (x0, top, x0 + w, top + h)
        band = (hero[3] + SPOTLIGHT_GROUP_GAP_PX, caption_top)
        if band[1] - band[0] < SPOTLIGHT_HERO_MIN_BAND_PX:
            continue
        if not clear(hero):
            continue
        plan = plan_spotlight_below(
            span_alphas, canvas, band=band, max_width=w,
            max_area=int(w * h * SPOTLIGHT_HERO_AREA_DOMINANCE),
            stats=stats)
        if not plan.feasible:
            continue
        pb = plan.speaker_box()
        # The dominance cap must hold on the DECLARED (rounded)
        # boxes too, not only in the solver's floats: a rounded
        # PiP box can sit a fraction of a percent over the cap.
        if (pb[2] - pb[0]) * (pb[3] - pb[1]) \
                > SPOTLIGHT_HERO_AREA_DOMINANCE * w * h:
            continue
        key = (pb[3] - pb[1], w * h)  # PiP height, then hero area
        feas.append((key, w, sized, plan, hero, band))
    if not feas:
        return None
    # Hero floor first (the brief's ordering): maximise the
    # speaker among heroes at or above the floor; narrower
    # heroes only when no wide hero solved at all.
    pool = [e for e in feas if e[1] >= hero_floor] or feas
    _key, _w, sized, plan, hero, band = max(
        pool, key=lambda e: e[0])
    record = _apply_spotlight_hero(
        spec, sized, hero, plan.speaker_box(), canvas)
    record["fit"] = ("cover" if content.box_resizer is not None
                     else "native_aspect")
    record["band"] = [int(band[0]), int(band[1])]
    record["group"] = {
        "top_px": top, "gap_px": SPOTLIGHT_GROUP_GAP_PX,
        "hero_width_frac_cap": SPOTLIGHT_GROUP_HERO_WIDTH_FRAC,
        "hero_min_width_frac": SPOTLIGHT_GROUP_HERO_MIN_WIDTH_FRAC}
    return plan, record


def plan_layout(
    specs: Sequence[GraphicSpec],
    src_alphas: Sequence[np.ndarray],
    *,
    fps: float,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    n_frames: int | None = None,
) -> tuple[list[Any] | None, list[dict[str, Any]],
           list[tuple[float, float, tuple[int, int, int, int]]]] | None:
    """Plan the layout track for placed specs (Stage 2 wiring).

    THE shared builder: the pipeline calls it on draft 1's placed
    specs, and any rescue executor that rebuilds frames/alphas
    calls it again on the carried specs - same function, same
    inputs, same track, so a pipeline render and a rescue
    re-render of the same Short can never diverge in layout.

    Returns None when no spec declares a non-Stage layout.
    Otherwise (track, declarations, span_caption_rects): the
    per-frame placement track (None when every declared plan was
    infeasible), the receipt-facing declarations for every
    attempted non-Stage span (feasible or not), and the caption
    rect overrides for Split spans.

    A plan that cannot satisfy the existing F3/F4 limits - or
    whose PiP would collide with the span's own graphic - falls
    back to Stage for that span, honestly: the spec's ``layout``
    is downgraded in place and the declaration records
    ``feasible: False`` with the reason (the speaker-forward
    fallback discipline, applied to states).

    Spotlight is solved HERO-FIRST since fix round 4
    (_solve_spotlight_hero): the graphic takes the majority
    region at the top of the frame and the speaker PiP is solved
    below it (plan_spotlight_below), with the dominance recorded
    as ``spotlight_hero`` on the declaration. The historic
    corner-PiP logic remains as the fallback when the hero solve
    is unavailable (no retained content, a declared path, a span
    inside the hook window, unsolvable geometry): its corner is
    chosen deterministically, left first, then right - the first
    corner that is both feasible and clear of the graphic box
    wins - with the Fix C refit behind it.

    The deterministic layout assist (_layout_assist) runs first:
    qualifying Stage hero spans are tried as Spotlight, so the
    early return below now means "no span declared a state AND
    no hero span qualified for the assist".
    """
    assisted = _layout_assist(specs)
    for s in specs:
        # Spans assisted in build_graphics carry their record on
        # the placed spec (the layout is already "spotlight", so
        # _layout_assist above left them alone).
        if s.layout_assist is not None:
            assisted.setdefault(id(s), s.layout_assist)
    targets = [g for g in specs if g.layout != "stage"]
    if not targets:
        return None
    n = n_frames if n_frames is not None else len(src_alphas)
    frame_shape = src_alphas[0].shape
    decls: list[LayoutDecl] = []
    plans: list[Any] = []
    declarations: list[dict[str, Any]] = []
    span_rects: list[tuple[float, float,
                           tuple[int, int, int, int]]] = []
    for spec in targets:
        f0 = max(0, int(round(spec.start_s * fps)))
        f1 = min(len(src_alphas),
                 max(f0 + 1, int(round(spec.end_s * fps))))
        plan: Any = None
        refit_record: dict[str, Any] | None = None
        hero_record: dict[str, Any] | None = None
        if spec.layout == "spotlight":
            # Fix round 4: hero-first - the graphic's box is
            # solved as the majority region and the PiP goes
            # BELOW it (Kamlesh's arrangement). Only when that
            # solve is unavailable does the historic corner-PiP
            # logic (with the Fix C refit) run.
            hero = _solve_spotlight_hero(
                spec, src_alphas[f0:f1], specs, canvas)
            if hero is not None:
                plan, hero_record = hero
            if plan is None:
                chosen = None
                feasible_plans: list[Any] = []
                for corner in ("left", "right"):
                    cand = plan_spotlight(src_alphas[f0:f1], canvas,
                                          corner=corner)
                    if not cand.feasible:
                        continue
                    feasible_plans.append(cand)
                    if not _boxes_intersect(cand.speaker_box(),
                                            spec.box):
                        chosen = cand
                        break
                if chosen is not None:
                    plan = chosen
                elif feasible_plans:
                    # Fix C: before downgrading, try to re-fit the
                    # card around each feasible PiP.
                    for cand in feasible_plans:
                        refit_record = _refit_spotlight_card(
                            spec, cand, specs, canvas)
                        if refit_record is not None:
                            plan = cand
                            break
                    if refit_record is None:
                        plan = feasible_plans[-1]
                        plan.feasible = False
                        plan.note += ("; both PiP corners collide "
                                      "with the span's graphic box "
                                      "- Stage kept")
                else:
                    plan = plan_spotlight(src_alphas[f0:f1], canvas,
                                          corner="left")
        elif spec.layout == "split":
            plan = plan_split(src_alphas[f0:f1], canvas)
        else:  # cutaway
            plan = plan_cutaway(frame_shape, canvas)
        declaration = plan.declaration(
            spec.start_s, spec.end_s, camera=spec.camera,
            graphic_box=[int(v) for v in spec.box])
        declaration["anchor_quality"] = spec.anchor_quality
        if refit_record is not None:
            declaration["spotlight_refit"] = refit_record
        if hero_record is not None:
            declaration["spotlight_hero"] = hero_record
        assist = assisted.get(id(spec))
        if not plan.feasible:
            declaration["downgraded_to"] = "stage"
            if assist is not None:
                declaration["layout_assist"] = {
                    **assist, "applied": "stage",
                    "reason": plan.note}
            declarations.append(declaration)
            spec.layout = "stage"
            continue
        if assist is not None:
            declaration["layout_assist"] = {
                **assist, "applied": "spotlight"}
        declarations.append(declaration)
        decls.append(LayoutDecl(spec.start_s, spec.end_s,
                                spec.layout, camera=spec.camera))
        plans.append(plan)
        if plan.caption_rect is not None:
            span_rects.append((spec.start_s, spec.end_s,
                               tuple(plan.caption_rect)))
    track = (build_track(decls, plans, frame_shape, n, fps, canvas)
             if decls else None)
    return track, declarations, span_rects
