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
  quote - but at least 4s of clean screen between graphics
  ("sparse seconds") and no span touching the first 3.0s hook
  window; spans run 2.5-6s anchored to caption-phrase timing.
  Both spacing rules are ALSO verifier rules (F8), so a render
  that breaks them fails certification, not just planning. An
  empty list is a valid, common outcome - most speech earns no
  graphic.
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
  drift (zoom 1.00 -> 1.06 across the span). Every state is a pure
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
from .layout_law import CAPTION_BOTTOM_MARGIN, face_pixels
from .spark_client import SparkClient, SparkClientError

# ---------------------------------------------------------------- constants

HOOK_EXCLUSION_END_S = 3.0   # spans may not touch the first 3.0s
MIN_SPAN_S = 2.5
MAX_SPAN_S = 6.0
MIN_GAP_S = 4.0              # clean screen between graphics ("sparse
                             # seconds" - a kept ruling, also an F8 rule)
REVEAL_MIN_SPAN_S = 3.5      # shorter quote/name cards stay static
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
KEN_BURNS_MAX_ZOOM = 1.06    # across the whole span, never more
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
    "\"end_s\": number, \"kind\": \"card\" or \"image\", "
    "\"headline\": \"...\", \"facts\": \"...\", \"prompt\": \"...\", "
    "\"compare\": [{\"label\": \"...\", \"value\": number}, "
    "{\"label\": \"...\", \"value\": number}] or null, "
    "\"source_quote\": \"...\"}]}. Rules: there is NO cap on the "
    "number of graphics - the count is driven purely by the "
    "content, as many as genuinely clear the bar and no more; "
    "when in doubt, leave it out. An "
    "empty list is a valid and common answer - most speech earns "
    "no graphic. kind \"card\" is for numbers, quotes, names and "
    "facts: headline at most 6 words, facts at most 12 words, both "
    "copied from the transcript - never invent a number, name or "
    "claim. For a comparison between two stated values, use kind "
    "\"card\" with \"compare\": the two items' labels (transcript "
    "words) and their stated values. A named PERSON is always a "
    "card (headline = their name, facts = their role in the "
    "speaker's words) - never an image: no AI faces, no likenesses. "
    "kind \"image\" is only for a concrete visual object or "
    "concept that is not a person: prompt describes the picture in "
    "one sentence with NO text, NO numbers, NO letters, NO people. "
    "Each span is the seconds where the topic is spoken (use the "
    "caption timings given), lasts 2.5 to 6 seconds, starts at or "
    "after 3.0s (the opening hook window is forbidden), and leaves "
    "at least 4 seconds of clean screen before the next graphic. "
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
    kind: str  # "card" | "image"
    headline: str = ""
    facts: str = ""
    prompt: str = ""
    source_quote: str = ""
    compare: list[dict[str, Any]] | None = None
    motion: str = "static"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s, "end_s": self.end_s,
            "kind": self.kind, "headline": self.headline,
            "facts": self.facts, "prompt": self.prompt,
            "source_quote": self.source_quote,
            "compare": self.compare, "motion": self.motion,
        }


# layer_at signature: (t_s, start_s, end_s) -> canvas RGBA layer.
LayerFn = Callable[[float, float, float], np.ndarray]


@dataclass
class GraphicSpec:
    """One placed graphic: everything the compositor renders and
    the verifier measures. ``layer_rgba`` is the FINAL state (the
    layer at the span's end); ``layer_at(t)`` reproduces the exact
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
    motion: str = "static"
    cost_usd: float = 0.0
    _state_fn: LayerFn | None = field(default=None, repr=False,
                                       compare=False)

    def layer_at(self, t_s: float) -> np.ndarray:
        """The graphic's canvas layer at time ``t_s``."""
        if self._state_fn is not None:
            return self._state_fn(t_s, self.start_s, self.end_s)
        return self.layer_rgba

    def mask(self) -> np.ndarray:
        """Bool canvas of the graphic's opaque pixels (final state -
        the superset every animated state stays inside)."""
        return self.layer_rgba[..., 3] > 0

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
        else:
            out["prompt"] = self.prompt
        return out


@dataclass
class GraphicsBuild:
    """The outcome of planning + generation + placement."""

    placed: list[GraphicSpec] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    planner_call: dict[str, Any] | None = None
    planner_rejected: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def total_cost_usd(self) -> float:
        return round(sum(g.cost_usd for g in self.placed), 4)

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
    if span.kind == "image":
        return "ken_burns"
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
    value, people never images, frequency discipline enforced.
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
        reason = None
        if kind not in ("card", "image"):
            reason = f"unknown kind {kind!r}"
        elif start < HOOK_EXCLUSION_END_S:
            reason = (f"span starts at {start:.2f}s, inside the "
                      f"first {HOOK_EXCLUSION_END_S:.1f}s hook window")
        elif end <= start:
            reason = "span end is not after its start"
        elif not (MIN_SPAN_S <= end - start <= MAX_SPAN_S):
            reason = (f"span length {end - start:.2f}s outside "
                      f"[{MIN_SPAN_S}, {MAX_SPAN_S}]s")
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
        if reason is None and compare is not None:
            reason = _validate_compare(compare, transcript_words,
                                       transcript_values)
            if reason is None and kind != "card":
                reason = "compare data on a non-card span"
        if reason is not None:
            rejected.append(_reject(item, reason))
            continue
        span = GraphicSpan(
            start_s=round(start, 3), end_s=round(end, 3), kind=kind,
            headline=headline, facts=facts, prompt=prompt,
            source_quote=quote,
            compare=compare if compare else None)
        span.motion = _derive_motion(span)
        accepted.append(span)

    accepted.sort(key=lambda s: (s.start_s, s.end_s))
    deduped: list[GraphicSpan] = []
    for span in accepted:
        if any(span.start_s < kept.end_s and kept.start_s < span.end_s
               for kept in deduped):
            rejected.append(_reject(
                span.to_dict(), "span overlaps an earlier accepted span"))
            continue
        deduped.append(span)
    # Spacing discipline (kept rulings): >= MIN_GAP_S of clean
    # screen between graphics. There is deliberately NO count cap
    # and NO screen-time ceiling - the semantic bar is the only
    # limit on how many graphics a Short carries.
    kept: list[GraphicSpan] = []
    for span in deduped:
        if kept and span.start_s - kept[-1].end_s < MIN_GAP_S:
            rejected.append(_reject(
                span.to_dict(),
                f"less than {MIN_GAP_S:.0f}s of clean screen after "
                "the previous graphic"))
            continue
        kept.append(span)
    return kept, rejected


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
    for budget in budgets:
        try:
            result = spark.chat(
                messages, max_tokens=budget,
                purpose="graphics-planner", ts=ts)
            break
        except SparkClientError as exc:
            if "empty chat reply" not in str(exc):
                return PlannerOutcome(
                    error=f"planner call failed: {exc}"[:300])
            empty_first = True
            result = None
        except Exception as exc:
            return PlannerOutcome(error=f"planner call failed: {exc}"[:300])
    if result is None:
        return PlannerOutcome(
            error="planner call failed: empty chat reply at every "
                  "budget (base and doubled retry); no graphics "
                  "planned")
    call = {"model": "muse-spark-1.3", "purpose": "graphics-planner",
            "cost_usd": result.cost_usd,
            "request_id": result.request_id,
            "empty_first_attempt": empty_first}
    spans, rejected = parse_planner_reply(
        result.text, transcript, duration_s=duration_s)
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


def _card_layout(headline: str, facts: str, width: int):
    """(head_lines, fact_lines, height) - the card's fixed layout.

    Layout is computed from the FINAL text and never changes across
    animation states: count-up swaps a token inside its line,
    reveal hides whole lines, but the box is constant.
    """
    probe = Image.new("RGBA", (8, 8))
    draw = ImageDraw.Draw(probe)
    head_font = _load_font(50)
    fact_font = _load_font(38)
    max_w = width - 2 * CARD_PAD
    head_lines = _wrap(draw, headline.upper(), head_font, max_w)[:3]
    fact_lines = _wrap(draw, facts, fact_font, max_w)[:2] if facts else []
    head_h = int(50 * 1.18) * len(head_lines)
    fact_h = int(38 * 1.18) * len(fact_lines)
    height = (CARD_PAD + 10 + 16 + head_h
              + (12 + fact_h if fact_lines else 0) + CARD_PAD)
    return head_lines, fact_lines, height


def _draw_card(head_lines: list[list[str]], fact_lines: list[list[str]],
               height: int, width: int, *,
               reveal_count: int | None = None) -> Image.Image:
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
    head_font = _load_font(50)
    fact_font = _load_font(38)
    shown = 0
    total = len(head_lines) + len(fact_lines)
    limit = total if reveal_count is None else max(0, reveal_count)
    y = CARD_PAD + 10 + 16
    for line in head_lines:
        if shown < limit:
            draw.text((CARD_PAD, y), " ".join(line), font=head_font,
                      fill=(255, 255, 255, 255))
        shown += 1
        y += int(50 * 1.18)
    if fact_lines:
        y += 12
        for line in fact_lines:
            if shown < limit:
                draw.text((CARD_PAD, y), " ".join(line), font=fact_font,
                          fill=VIRAL_PINK + (255,))
            shown += 1
            y += int(38 * 1.18)
    return img


def render_card_image(headline: str, facts: str = "", *,
                      width: int = CARD_WIDTH) -> Image.Image:
    """One brand-palette card, final state (deterministic pixels)."""
    head_lines, fact_lines, height = _card_layout(headline, facts, width)
    return _draw_card(head_lines, fact_lines, height, width)


def _count_up_image(headline: str, facts: str, p: float,
                      width: int) -> Image.Image:
    """The card at count-up progress ``p``: the first number token
    (headline first, then facts) eased from 0 to its stated value;
    every other word is already final."""
    head_lines, fact_lines, height = _card_layout(headline, facts, width)
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
    return _draw_card(head_lines, fact_lines, height, width)


def render_compare_card_image(
    headline: str,
    items: list[dict[str, Any]],
    progresses: tuple[float, float],
    *,
    width: int = COMPARE_CARD_WIDTH,
) -> Image.Image:
    """A two-bar comparison card at the given build progresses.

    Bar lengths are proportional to the stated values; each bar
    grows (ease-out) inside its own window. Before its window a bar
    shows its label over an empty track - the comparison builds as
    the speaker names each side.
    """
    probe = Image.new("RGBA", (8, 8))
    draw = ImageDraw.Draw(probe)
    head_font = _load_font(46)
    label_font = _load_font(36)
    max_w = width - 2 * CARD_PAD
    head_lines = _wrap(draw, headline.upper(), head_font, max_w)[:2]
    head_h = int(46 * 1.18) * len(head_lines)
    row_h = int(36 * 1.18) + 34 + 14
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
        y += int(46 * 1.18)
    y += 10
    values = [float(it["value"]) for it in items]
    peak = max(values) if values and max(values) > 0 else 1.0
    bar_area_w = width - 2 * CARD_PAD - 118
    colors = [FORGE_BLUE, VIRAL_PINK]
    for i, item in enumerate(items[:2]):
        label = str(item["label"]).upper()
        draw.text((CARD_PAD, y), label, font=label_font,
                  fill=(235, 235, 240, 255))
        y += int(36 * 1.18)
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


# ---------------------------------------------------------------- content


@dataclass
class GraphicContent:
    """Prepared content for one span: its fixed box size, its final
    state, and (when animated) a progress -> box-sized RGBA state
    function. States are pure functions of span progress - the same
    image is produced whenever the same progress is asked for."""

    size: tuple[int, int]
    final_image: Image.Image
    state_fn: Callable[[float], Image.Image] | None = None
    cost_usd: float = 0.0


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
                 cues: Sequence[CaptionCue] = ()) -> GraphicContent:
    """Render a card span's content (final state + animation)."""
    if span.compare:
        return _prepare_compare_card(span, cues)
    final = render_card_image(span.headline, span.facts)
    if span.motion == "count_up":
        state = _quantized(
            lambda p: _count_up_image(span.headline, span.facts, p,
                                      CARD_WIDTH))
        return GraphicContent(size=final.size, final_image=final,
                              state_fn=state)
    if span.motion == "reveal":
        head_lines, fact_lines, _ = _card_layout(
            span.headline, span.facts, CARD_WIDTH)
        total = len(head_lines) + len(fact_lines)

        def reveal(p: float) -> Image.Image:
            # Lines land one by one; all are in by p = 0.8.
            count = int(min(total, (p / 0.8) * total + 1e-9)) \
                if p > 0 else 0
            head, fact, height = _card_layout(
                span.headline, span.facts, CARD_WIDTH)
            return _draw_card(head, fact, height, CARD_WIDTH,
                              reveal_count=count)

        return GraphicContent(size=final.size, final_image=final,
                              state_fn=_quantized(reveal))
    return GraphicContent(size=final.size, final_image=final)


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
                          cues: Sequence[CaptionCue]) -> GraphicContent:
    assert span.compare is not None
    headline = span.headline
    items = span.compare
    final = render_compare_card_image(headline, items, (1.0, 1.0))
    fracs = _compare_triggers(span, cues)
    span_len = max(span.end_s - span.start_s, 1e-9)
    bar_window = min(0.45, 0.7 / span_len)  # ~0.7s of growth, in p units

    def state(p: float) -> Image.Image:
        progresses = tuple(
            min(1.0, max(0.0, (p - f) / bar_window)) if p >= f else 0.0
            for f in fracs)
        return render_compare_card_image(headline, items, progresses)

    return GraphicContent(size=final.size, final_image=final,
                          state_fn=_quantized(state))


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
                  ) -> GraphicContent:
    """Ken Burns content: zoom eases 1.00 -> 1.06 across the span
    with a slow pan, all inside the fixed box."""
    bw, bh = IMAGE_BOX
    base = _cover_fit(image_rgb, (round(bw * 1.12), round(bh * 1.12)))

    def state(p: float) -> Image.Image:
        zoom = 1.0 + (KEN_BURNS_MAX_ZOOM - 1.0) * min(1.0, max(0.0, p))
        win_w = min(base.size[0], round(bw * 1.12 / zoom))
        win_h = min(base.size[1], round(bh * 1.12 / zoom))
        slack_x = base.size[0] - win_w
        slack_y = base.size[1] - win_h
        x = int(slack_x * (0.25 + 0.35 * p))
        y = int(slack_y * (0.25 + 0.30 * p))
        crop = base.crop((x, y, x + win_w, y + win_h)).resize(
            (bw, bh), Image.BILINEAR)
        return _framed(crop, IMAGE_BOX)

    return GraphicContent(size=IMAGE_BOX, final_image=state(1.0),
                          state_fn=_quantized(state))


def _paste_layer(content_image: Image.Image,
                 box: tuple[int, int, int, int],
                 canvas: tuple[int, int]) -> np.ndarray:
    layer = Image.new("RGBA", (canvas[0], canvas[1]), (0, 0, 0, 0))
    layer.alpha_composite(content_image, (box[0], box[1]))
    return np.asarray(layer)


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
              region: tuple[int, int, int, int]
              ) -> tuple[int, int, int, int] | None:
    """Best legal top-left position for a (w, h) graphic, or None.

    Legal = every pixel of the box inside the free mask. Ranking is
    deterministic: beside-the-head first (vertically in the
    head/shoulder band, horizontally adjacent to the face - the
    brief's target region; clearance is already guaranteed by the
    free mask itself), then greatest clearance (EDT at the box
    centre, capped), then smallest horizontal gap to the face,
    then topmost, then leftmost. A non-beside position still wins
    over dropping the graphic when no beside position is legal.
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
            key = (beside, -clearance, gap, y, x)
            if best_key is None or key < best_key:
                best_key = key
                best = (x, y, x + w, y + h)
    return best


def place_prepared(
    items: list[tuple[GraphicSpan, GraphicContent]],
    canvas_alphas: Sequence[np.ndarray],
    *,
    fps: float,
    caption_rect: tuple[int, int, int, int] | None = None,
    hook_glyph_bbox: tuple[int, int, int, int] | None = None,
    hook_window: tuple[float, float] | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
) -> tuple[list[GraphicSpec], list[dict[str, Any]]]:
    """Place prepared graphics or drop them, with reasons.

    This function owns ONLY the geometry: the hard exclusions and
    the search. A span whose graphic fits nowhere legal is dropped
    with the reason - the layer never trades the exclusions for
    coverage.
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
        free = (occupancy <= FREE_OCCUPANCY_MAX) & ~forbidden
        edt = ndimage.distance_transform_edt(free)
        region = (CANVAS_MARGIN_PX, CANVAS_MARGIN_PX,
                  cw - CANVAS_MARGIN_PX, caption_top)
        box = _find_box(free, edt, content.size, face_bbox, region)
        if box is None:
            dropped.append({
                "span": span.to_dict(),
                "reason": "no legal placement: every candidate "
                          "position hits the face margin, the "
                          "caption zone, the hook zone, or speaker "
                          "occupancy"})
            continue

        def layer_fn(t: float, start: float, end: float,
                     _content: GraphicContent = content,
                     _box: tuple[int, int, int, int] = box,
                     _canvas: tuple[int, int] = canvas) -> np.ndarray:
            if end > start:
                p = min(1.0, max(0.0, (t - start) / (end - start)))
            else:
                p = 1.0
            img = _content.state_fn(p) if _content.state_fn \
                else _content.final_image
            return _paste_layer(img, _box, _canvas)

        placed.append(GraphicSpec(
            kind=span.kind, start_s=span.start_s, end_s=span.end_s,
            box=box,
            layer_rgba=layer_fn(span.end_s, span.start_s, span.end_s),
            source_quote=span.source_quote, prompt=span.prompt,
            headline=span.headline, facts=span.facts,
            compare=span.compare, motion=span.motion,
            cost_usd=content.cost_usd, _state_fn=layer_fn))
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
        error=outcome.error)
    items: list[tuple[GraphicSpan, GraphicContent]] = []
    for span in outcome.spans:
        if span.kind == "card":
            items.append((span, prepare_card(span, cues)))
            continue
        if image_client is None:
            build.dropped.append({
                "span": span.to_dict(),
                "reason": "image span but no image client is wired"})
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
            compare=span.compare, motion=span.motion)
        content = prepare_image(image.convert("RGB"), span)
        content.cost_usd = result.cost_usd
        items.append((span, content))
    placed, dropped = place_prepared(
        items, canvas_alphas, fps=fps, caption_rect=caption_rect,
        hook_glyph_bbox=hook_glyph_bbox, hook_window=hook_window,
        canvas=canvas)
    build.placed = placed
    build.dropped.extend(dropped)
    return build
