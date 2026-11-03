"""Word-level caption timing behind ONE interface, two sources.

Voice Transcribe returns TURN-level timestamps only (verified -
there is no word mode), so word timings for the caption word
sweep must be reconstructed. Two implementations sit behind the
``word_timings`` dispatcher, and every timing carries its source:

- ``"dsp"`` (the floor - ships regardless): pure signal
  processing, the same non-model class as caption_align's phrase
  timing. The phrase's speech-active segments (onset/pause
  segmentation of the energy profile) constrain a proportional
  word split: the words share the ACTIVE time in proportion to
  their character lengths, silences consume no words, and word
  boundaries land on speech, never inside a pause.
- ``"mms"``: Meta's open-weights MMS forced aligner (the
  torchaudio MMS_FA bundle, facebook/mms-1b-fl102 - part of the
  MMS-1B family) run on our own compute. It is fed the Voice
  Transcribe transcript text and the audio; it generates NO
  transcript text, it only places the given words in time via
  CTC forced alignment. Event posture (staff answer, 2026-11-01):
  open-source Meta models count as Meta models however they run,
  including on own compute; the README names the model and where
  it runs when this source ships.

  Runtime shape (fix phase, 2026-11-02): the shipped pipeline's
  interpreter has NO torch, so the aligner runs in a subprocess
  under the isolated MMS venv (tools/mms_align.py under
  MMS_VENV_PYTHON; override with the SPATIAL_DIRECTOR_MMS_PYTHON
  environment variable). Samples and words go over as files, word
  frame-spans come back as JSON, and every failure mode maps to
  the same exceptions the in-process design raised - the honesty
  rule below is unchanged by the process boundary.

HONESTY RULE (load-bearing): a timing is labelled with the source
that actually produced it. ``word_timings(source="mms")`` RAISES
``MMSUnavailableError`` when the runtime or the model cannot be
made to work - it never falls back to DSP numbers wearing an MMS
label, and nothing in this module ever invents a timing. Callers
that want a floor ask for ``"dsp"`` explicitly.

Written fresh in-repo on 2026-11-01 (extension Stage 4).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .caption_align import (
    EnergyProfile,
    _active_to_wall,
    active_segments,
)

# The MMS forced-alignment bundle: model id as served by
# torchaudio's pipeline registry (weights: facebook/mms-1b-fl102,
# the forced-alignment member of the MMS-1B family).
MMS_MODEL_ID = "facebook/mms-1b-fl102 (torchaudio MMS_FA)"
MMS_SAMPLE_RATE = 16000

TIMING_SOURCES = ("dsp", "mms")

# The isolated MMS runtime (fix phase, 2026-11-02): torch +
# torchaudio live in their own venv so the shipped pipeline's
# interpreter stays torch-free. The entry script lives in the
# repo (tools/mms_align.py) and is executed BY this interpreter.
MMS_VENV_PYTHON = (Path.home() / "workspace" / "meta-hackathon"
                   / "mms-venv" / "bin" / "python")
MMS_PYTHON_ENV = "SPATIAL_DIRECTOR_MMS_PYTHON"
MMS_ENTRY_SCRIPT = (Path(__file__).resolve().parents[2]
                    / "tools" / "mms_align.py")
# Model load dominates (tens of seconds cold); an alignment of a
# caption cue is seconds after that. The bound exists so a wedged
# subprocess fails the caption step loudly instead of hanging a
# run - it is a hang guard, not a performance target.
MMS_SUBPROCESS_TIMEOUT_S = 600


def _mms_python() -> Path:
    """The interpreter that runs the MMS entry script."""
    override = os.environ.get(MMS_PYTHON_ENV, "").strip()
    return Path(override) if override else MMS_VENV_PYTHON


class WordTimingError(RuntimeError):
    """A word-timing failure that must surface, never be hidden."""


class MMSUnavailableError(WordTimingError):
    """The MMS aligner cannot run here (deps/model/runtime).

    Raised instead of returning timings: the caller asked for MMS
    timings and there are none. Substituting another source's
    numbers under the MMS label would be fabrication.
    """


@dataclass
class WordTiming:
    word: str
    start_s: float
    end_s: float
    source: str  # "dsp" | "mms" - the implementation that made it
    detail: str = ""

    def to_dict(self) -> dict:
        return {"word": self.word, "start_s": round(self.start_s, 3),
                "end_s": round(self.end_s, 3), "source": self.source}


# ---------------------------------------------------------------- DSP


def dsp_word_timings(words: Sequence[str], t0: float, t1: float,
                     profile: EnergyProfile | None, *,
                     thresh_db: float | None = None
                     ) -> list[WordTiming]:
    """Word timings from DSP: proportional split over active time.

    The phrase's words share the speech-active time inside
    [t0, t1] in proportion to their character lengths (longer
    words take longer to say - the standard proportional split),
    with the active-segment map doing the constraining: a word
    boundary computed inside a pause is pushed to the speech on
    the far side of it, exactly as caption_align's phrase layout
    maps active offsets to wall time. Without an energy profile
    the split is uniform over the wall interval - still pure DSP,
    and the ``detail`` field says which flavour produced it.
    """
    words = [str(w) for w in words]
    if not words:
        return []
    if t1 <= t0:
        raise WordTimingError(
            f"word timing span is empty or inverted: [{t0}, {t1}]")
    if profile is not None:
        segments = active_segments(profile, t0, t1,
                                   thresh_db=thresh_db)
    else:
        segments = []
    if segments:
        total_active = sum(e - s for s, e in segments)
        detail = "active-segment proportional split"
    else:
        segments = [(t0, t1)]
        total_active = t1 - t0
        detail = "uniform split (no usable energy profile)"
    weights = [max(1, len(w)) for w in words]
    total_weight = float(sum(weights))
    out: list[WordTiming] = []
    acc = 0.0
    for word, weight in zip(words, weights):
        a_start = total_active * acc / total_weight
        a_end = total_active * (acc + weight) / total_weight
        out.append(WordTiming(
            word=word,
            start_s=round(_active_to_wall(segments, a_start,
                                          is_end=False), 3),
            end_s=round(_active_to_wall(segments, a_end,
                                        is_end=True), 3),
            source="dsp", detail=detail))
        acc += weight
    return out


# ---------------------------------------------------------------- MMS


def load_samples(audio_path: str | Path,
                 sample_rate: int = MMS_SAMPLE_RATE) -> np.ndarray:
    """Decode a media file's audio to mono float32 samples (ffmpeg).

    The same decode pattern caption_align.load_energy uses - no
    model involved, just the samples the aligner will hear.
    """
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio_path),
         "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
        capture_output=True)
    if proc.returncode != 0:
        raise WordTimingError(
            f"ffmpeg could not decode audio from {audio_path}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise WordTimingError(f"no audio samples decoded from {audio_path}")
    return samples


def _load_mms():
    """Import the MMS runtime or raise MMSUnavailableError.

    Lazy by design: the shipped pipeline must import this module
    (and run the DSP floor) on machines with no torch at all.
    """
    try:
        import torch  # noqa: F401
        import torchaudio
    except ImportError as exc:
        raise MMSUnavailableError(
            "MMS forced aligner unavailable: torch/torchaudio are "
            f"not installed in this environment ({exc}). The DSP "
            "source is the shipped floor; install torch + "
            "torchaudio to enable source='mms'.") from exc
    return torchaudio


def _mms_align_in_process(words: Sequence[str], samples: np.ndarray,
                          sample_rate: int = MMS_SAMPLE_RATE,
                          ) -> tuple[list[tuple[int, int]], int, float]:
    """The aligner itself: (word frame-spans, n_frames, total_s).

    Runs ONLY inside the MMS venv interpreter (invoked by
    tools/mms_align.py); it imports torch in-process. Kept apart
    from mms_word_timings so the process boundary is the only
    difference between the verified S4 integration and the shipped
    subprocess path - the alignment math is byte-identical.
    """
    torchaudio = _load_mms()
    import torch

    try:
        bundle = torchaudio.pipelines.MMS_FA
        model = bundle.get_model(with_star=False)
        model.eval()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
    except Exception as exc:
        raise MMSUnavailableError(
            "MMS forced aligner unavailable: the MMS_FA bundle "
            f"could not be loaded ({type(exc).__name__}: {exc}). "
            "First use downloads the model weights; an offline or "
            "interrupted fetch lands here, loudly.") from exc

    # The MMS dictionary is romanised letters + apostrophe. A word
    # that normalises to nothing (a bare number token, a symbol)
    # cannot be aligned - that is a loud failure, never a skipped
    # word with an interpolated time.
    normalised = [_mms_normalise(w) for w in words]
    for word, norm in zip(words, normalised):
        if not norm:
            raise WordTimingError(
                f"MMS aligner cannot represent the word {word!r} "
                "in its dictionary (romanised letters only) - "
                "refusing to invent its timing")
    waveform = torch.tensor(np.asarray(samples, dtype=np.float32))
    waveform = waveform.reshape(1, -1)
    with torch.inference_mode():
        emission, _ = model(waveform)
        tokens = tokenizer(normalised)
        spans = aligner(emission[0], tokens)
    word_spans = _word_spans_from_alignment(spans, words)
    n_frames = int(emission.shape[1])
    total_s = waveform.shape[1] / float(sample_rate)
    return word_spans, n_frames, total_s


def mms_word_timings(words: Sequence[str], samples: np.ndarray,
                     sample_rate: int = MMS_SAMPLE_RATE, *,
                     offset_s: float = 0.0) -> list[WordTiming]:
    """Word timings from Meta's MMS forced aligner, own compute.

    CTC forced alignment of the GIVEN words (the Voice Transcribe
    transcript - the aligner writes no text of its own) against
    the audio, executed in a subprocess under the isolated MMS
    venv (see the module docstring): this interpreter has no
    torch, so samples + words cross as files and word frame-spans
    return as JSON.

    Failure semantics are the in-process design's, unchanged:
    MMSUnavailableError when the runtime or model cannot run
    (missing interpreter, wedged or crashed subprocess,
    unloadable bundle), WordTimingError when a word cannot be
    represented or the returned alignment does not cover exactly
    the words it was fed - a partial alignment is never padded
    out with guesses, and no failure path returns DSP numbers.
    """
    words = [str(w) for w in words]
    if not words:
        return []
    if sample_rate != MMS_SAMPLE_RATE:
        raise WordTimingError(
            f"MMS aligner needs {MMS_SAMPLE_RATE} Hz audio, got "
            f"{sample_rate} Hz - resample first, do not approximate")
    # The representability pre-check stays in THIS process: an
    # unrepresentable word is a WordTimingError regardless of
    # whether the aligner runtime exists.
    for word in words:
        if not _mms_normalise(word):
            raise WordTimingError(
                f"MMS aligner cannot represent the word {word!r} "
                "in its dictionary (romanised letters only) - "
                "refusing to invent its timing")
    python = _mms_python()
    if not python.is_file():
        raise MMSUnavailableError(
            "MMS forced aligner unavailable: the isolated MMS "
            f"interpreter does not exist at {python} (set "
            f"{MMS_PYTHON_ENV} to override). The DSP source is "
            "the shipped floor.")
    if not MMS_ENTRY_SCRIPT.is_file():
        raise MMSUnavailableError(
            "MMS forced aligner unavailable: the entry script "
            f"is missing at {MMS_ENTRY_SCRIPT}.")
    samples = np.asarray(samples, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="mms-align-") as td:
        td = Path(td)
        samples_path = td / "samples.npy"
        request_path = td / "request.json"
        response_path = td / "response.json"
        np.save(samples_path, samples)
        request_path.write_text(json.dumps(
            {"words": words, "sample_rate": int(sample_rate)}))
        # The verified weights live in the venv's own torch-home
        # (downloaded once during the S4 verification). Point the
        # child at it when the caller has no TORCH_HOME of their
        # own: otherwise torchaudio looks in ~/.cache/torch, finds
        # no model, and stalls on a ~1.3 GB re-download.
        child_env = dict(os.environ)
        if not child_env.get("TORCH_HOME"):
            # NB: the venv's bin/python is a symlink chain to the
            # system interpreter - the venv root is the UNRESOLVED
            # path's grandparent, never Path.resolve()'s.
            torch_home = python.parents[1] / "torch-home"
            if torch_home.is_dir():
                child_env["TORCH_HOME"] = str(torch_home)
        try:
            proc = subprocess.run(
                [str(python), str(MMS_ENTRY_SCRIPT),
                 str(request_path), str(response_path),
                 str(samples_path)],
                capture_output=True, text=True, env=child_env,
                timeout=MMS_SUBPROCESS_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise MMSUnavailableError(
                "MMS forced aligner unavailable: the alignment "
                f"subprocess exceeded {MMS_SUBPROCESS_TIMEOUT_S}s "
                "and was killed - failing loudly rather than "
                "hanging the run.") from exc
        except OSError as exc:
            raise MMSUnavailableError(
                "MMS forced aligner unavailable: could not launch "
                f"{python} ({exc}).") from exc
        payload = None
        if response_path.is_file():
            try:
                payload = json.loads(response_path.read_text())
            except (OSError, json.JSONDecodeError):
                payload = None
        if not isinstance(payload, dict) or "status" not in payload:
            tail = (proc.stderr or "").strip()[-300:]
            raise MMSUnavailableError(
                "MMS forced aligner unavailable: the alignment "
                f"subprocess exited {proc.returncode} without a "
                f"usable response. stderr tail: {tail!r}")
        if payload["status"] != "ok":
            message = str(payload.get("message", ""))[:400]
            if payload.get("error_type") == "word":
                raise WordTimingError(message)
            raise MMSUnavailableError(message)
        raw_spans = payload.get("word_spans")
        n_frames = payload.get("n_frames")
        total_s = payload.get("total_s")
    # The response is validated with the same strictness the
    # in-process path applied to the aligner's own output: exact
    # word coverage, in-bounds frames, monotone placement.
    if (not isinstance(raw_spans, list)
            or len(raw_spans) != len(words)
            or not isinstance(n_frames, int) or n_frames <= 0
            or not isinstance(total_s, (int, float))
            or not total_s or float(total_s) <= 0):
        raise WordTimingError(
            "MMS alignment response does not cover exactly the "
            f"{len(words)} words fed to it - refusing to pad or "
            "guess the missing placements")
    word_spans: list[tuple[int, int]] = []
    prev_start = -1
    for pair in raw_spans:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, int) for v in pair)):
            raise WordTimingError(
                f"MMS alignment response carries a malformed "
                f"word span {pair!r} - refusing to guess")
        f0, f1 = int(pair[0]), int(pair[1])
        if not (0 <= f0 <= f1 <= n_frames) or f0 < prev_start:
            raise WordTimingError(
                f"MMS alignment response span ({f0}, {f1}) is "
                f"out of bounds or non-monotone over {n_frames} "
                "frames - refusing to guess")
        prev_start = f0
        word_spans.append((f0, f1))
    out: list[WordTiming] = []
    for word, (f0, f1) in zip(words, word_spans):
        out.append(WordTiming(
            word=word,
            start_s=round(offset_s + float(total_s) * f0 / n_frames,
                          3),
            end_s=round(offset_s + float(total_s) * f1 / n_frames,
                        3),
            source="mms", detail=f"forced alignment, {MMS_MODEL_ID}"))
    return out


_MMS_CHARS = set("abcdefghijklmnopqrstuvwxyz'")


def _mms_normalise(word: str) -> str:
    """A transcript word in the MMS dictionary's alphabet."""
    return "".join(c for c in word.lower() if c in _MMS_CHARS)


def _word_spans_from_alignment(spans, words: Sequence[str]):
    """Per-word (start_frame, end_frame) from aligner output.

    torchaudio's MMS aligner returns one list of TokenSpans per
    word fed to the tokenizer (verified against torchaudio
    2.4.1's MMS_FA bundle). Each TokenSpan carries .start / .end
    in emission frames. A group count that disagrees with the
    word count is a hard error: guessing word boundaries is how
    fabricated timings happen.
    """
    if spans is None:
        raise WordTimingError("MMS aligner returned no spans")
    groups = list(spans)
    out: list[tuple[int, int]] = []
    for group in groups:
        toks = list(group)
        if not toks or not hasattr(toks[0], "start"):
            raise WordTimingError(
                "MMS aligner span group is not token spans - "
                "unrecognised aligner output shape")
        out.append((int(toks[0].start), int(toks[-1].end)))
    if len(out) != len(words):
        raise WordTimingError(
            f"MMS aligner produced {len(out)} word groups for "
            f"{len(words)} transcript words - refusing to pad or "
            "guess the missing placements")
    return out


# --------------------------------------------------------- pace gate


# Fix round 5 (Kamlesh's QM6 caption verdict): an alignment whose
# words run at a sustained pace no speaker sustains is not a
# measurement - it is a synthetic grid wearing the aligner's
# label (the shipped QM6/QH5 captions ran at a CONSTANT
# ~6.9 / ~3.44 words per second because each cue's span came
# from the uniform word layout, and CTC crammed the words into
# whatever slice it was handed). The gate below is the sanity
# check that was missing: it judges an ALIGNED cue against the
# DSP speech-activity span of its own passage.
PACE_GATE_MAX_WPS = 4.5        # sustained pace ceiling, words/sec
PACE_GATE_MIN_SPAN_FRAC = 0.60  # aligned extent vs DSP activity span


def cue_pace(n_words: int, span_s: float) -> float:
    """Words per second over a span; inf for a non-positive span."""
    if span_s <= 0:
        return float("inf")
    return n_words / span_s


def pace_gate_failure(timings: Sequence["WordTiming"],
                      activity_span_s: float | None) -> str | None:
    """Why ``timings`` fail the pace sanity gate, or None.

    Two legs, both measured on the ALIGNED words themselves:
    (1) sustained pace over the words' own extent (first start
    to last end) above PACE_GATE_MAX_WPS - sustained human
    speech stays below ~4 wps even when fast; a "measurement"
    above the gate is a compressed grid; (2) the words' extent
    covering less than PACE_GATE_MIN_SPAN_FRAC of the DSP
    speech-activity span of the cue's passage - the aligner
    claims the passage's words occupy a fraction of the time
    the passage's speech demonstrably occupies. A single word
    has no measurable pace and defines its own extent, so it
    passes by construction (its placement is the aligner's
    direct measurement of one token).
    """
    if len(timings) < 2:
        return None
    extent = timings[-1].end_s - timings[0].start_s
    pace = cue_pace(len(timings), extent)
    if pace > PACE_GATE_MAX_WPS:
        return (f"sustained pace {pace:.2f} words/s over the "
                f"aligned extent {extent:.2f}s exceeds the "
                f"{PACE_GATE_MAX_WPS} words/s sanity ceiling")
    if activity_span_s is not None and activity_span_s > 0 \
            and extent < PACE_GATE_MIN_SPAN_FRAC * activity_span_s:
        return (f"aligned extent {extent:.2f}s covers less than "
                f"{PACE_GATE_MIN_SPAN_FRAC:.0%} of the DSP "
                f"speech-activity span {activity_span_s:.2f}s")
    return None


# ------------------------------------------------------- valley snapping


SNAP_WINDOW_S = 0.150      # a start may move at most this far
SNAP_HOP_S = 0.005         # envelope resolution (the measurement's)
SNAP_SMOOTH_HOPS = 3       # centred moving average over the RMS hops
MIN_ACTIVE_WORD_S = 0.100  # render floor for a word's active span


def speech_envelope(samples: np.ndarray, sample_rate: int,
                    hop_s: float = SNAP_HOP_S,
                    smooth_hops: int = SNAP_SMOOTH_HOPS
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(times, env): RMS energy per hop, lightly smoothed.

    The envelope both the valley snap (below) and its acceptance
    measurement read: 5ms hops of the speech audio, RMS per hop,
    centred moving average of ``smooth_hops`` - deterministic pure
    DSP, no model involved.
    """
    x = np.asarray(samples, dtype=np.float64)
    hop = max(1, int(round(sample_rate * hop_s)))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0), np.zeros(0)
    frames = x[:n * hop].reshape(n, hop)
    env = np.sqrt(np.mean(frames * frames, axis=1))
    if smooth_hops > 1 and n >= smooth_hops:
        kernel = np.ones(smooth_hops) / smooth_hops
        env = np.convolve(env, kernel, mode="same")
    times = (np.arange(n) + 0.5) * hop_s
    return times, env


def _valleys(times: np.ndarray,
             env: np.ndarray) -> list[tuple[float, float]]:
    """(time, depth) of the envelope's local minima.

    Plateau-aware: a run of equal values counts once, at its
    centre - flat silence is one valley, not a run of candidates.
    """
    out: list[tuple[float, float]] = []
    n = len(env)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and env[j + 1] == env[i]:
            j += 1
        left = env[i - 1] if i > 0 else None
        right = env[j + 1] if j + 1 < n else None
        is_min = (left is None or env[i] <= left) \
            and (right is None or env[i] <= right) \
            and ((left is not None and env[i] < left)
                 or (right is not None and env[i] < right)
                 or (left is None and right is None))
        if is_min:
            out.append((float(times[(i + j) // 2]), float(env[i])))
        i = j + 1
    return out


def _valley_times(times: np.ndarray, env: np.ndarray) -> list[float]:
    """Times of the envelope's local minima (see ``_valleys``)."""
    return [t for t, _v in _valleys(times, env)]


def snap_word_timings(
    cue_timings: Sequence[Sequence[WordTiming]],
    samples: np.ndarray,
    sample_rate: int = MMS_SAMPLE_RATE,
    *,
    window_s: float = SNAP_WINDOW_S,
    smooth_hops: int = SNAP_SMOOTH_HOPS,
    snap_sources: Sequence[str] | None = None,
) -> tuple[list[list[WordTiming]], dict[str, Any]]:
    """Snap word starts to the speech envelope's valleys.

    Fix round 4 (Kamlesh's caption-precision flag): the aligners
    place words with per-word jitter (measured on the banked
    QM5/QH4 drafts: median |error| 110/137ms against the speech's
    own energy valleys, a quarter to a third of words beyond
    150ms) - no constant offset, so no shift can fix it; each
    start must move to ITS boundary. A word's acoustic boundary
    is the energy dip the previous word's energy falls into -
    the DEEPEST envelope valley within +/-``window_s`` (the
    micro-minima inside a vowel are shallower than the boundary
    dip; depth is what makes a valley the boundary - this is
    the valley definition the baseline measurement used, and
    the snap mirrors it). A start with no valley in reach
    keeps its timing. The whole span shifts rigidly (duration
    preserved), starts are clamped strictly inside
    (prev_start, next_start) of the sequence - words never
    reorder, never cross a neighbour - and an end pushed past
    the next word's (snapped) start is capped at it.

    The timings keep their source label: snapping is a
    deterministic DSP correction of the aligner's placements,
    declared in each timing's detail and in the returned stats
    (which the pipeline records on captions.json + the receipt
    as ``caption_snap``). Pure function of (timings, samples).

    ``snap_sources`` (fix round 5): when given, only words whose
    source is in it may MOVE. Snapping exists to correct a real
    aligner's per-word jitter; applied to a proportional split
    it would manufacture false precision - a synthetic grid's
    words would land on real valleys and read as measured.
    Immovable words keep their starts (delta 0) but still act
    as neighbour-clamp anchors for the words that do move.
    """
    flat: list[tuple[int, int, WordTiming]] = []
    for ci, timings in enumerate(cue_timings):
        for wi, wt in enumerate(timings):
            flat.append((ci, wi, wt))
    stats: dict[str, Any] = {
        "applied": True, "method": "energy_valley_snap",
        "window_s": window_s, "hop_s": SNAP_HOP_S,
        "smooth_hops": smooth_hops,
        "words": len(flat), "words_snapped": 0,
        "mean_abs_shift_ms": 0.0, "max_abs_shift_ms": 0.0}
    if snap_sources is not None:
        stats["snap_sources"] = list(snap_sources)
        stats["words_eligible"] = sum(
            1 for _ci, _wi, wt in flat if wt.source in snap_sources)
    if not flat:
        return [list(t) for t in cue_timings], stats
    times, env = speech_envelope(samples, sample_rate,
                                 smooth_hops=smooth_hops)
    valleys = _valleys(times, env) if len(env) else []
    if not valleys:
        stats["applied"] = False
        stats["reason"] = "no energy valleys in the speech envelope"
        return [list(t) for t in cue_timings], stats
    import bisect
    valley_ts = [t for t, _v in valleys]
    new_starts: list[float] = []
    deltas: list[float] = []
    for k, (_ci, _wi, wt) in enumerate(flat):
        s = wt.start_s
        lo = (new_starts[-1] + 0.001) if new_starts else 0.0
        hi = (flat[k + 1][2].start_s - 0.001) \
            if k + 1 < len(flat) else float("inf")
        delta = 0.0
        if snap_sources is not None and wt.source not in snap_sources:
            new_starts.append(s)
            deltas.append(0.0)
            continue
        j = bisect.bisect_left(valley_ts, s - window_s)
        best: float | None = None
        best_depth = 0.0
        while j < len(valleys) and valley_ts[j] <= s + window_s:
            t_j, v_j = valleys[j]
            if best is None or v_j < best_depth - 1e-9 \
                    or (abs(v_j - best_depth) <= 1e-9
                        and abs(t_j - s) < abs(best - s)):
                best, best_depth = t_j, v_j
            j += 1
        if best is not None:
            delta = best - s
        new_start = s + delta
        if new_start < lo or new_start > hi:
            new_start = min(max(new_start, lo), hi) \
                if lo <= hi else s
            delta = new_start - s
        new_starts.append(new_start)
        deltas.append(delta)
    moved = [abs(d) for d in deltas if abs(d) >= 0.001]
    stats["words_snapped"] = len(moved)
    if moved:
        stats["mean_abs_shift_ms"] = round(
            float(np.mean(moved)) * 1000.0, 1)
        stats["max_abs_shift_ms"] = round(max(moved) * 1000.0, 1)
    out: list[list[WordTiming]] = [[] for _ in cue_timings]
    for k, (ci, _wi, wt) in enumerate(flat):
        end = wt.end_s + deltas[k]
        if k + 1 < len(flat):
            end = min(end, new_starts[k + 1])
        out[ci].append(WordTiming(
            word=wt.word, start_s=round(new_starts[k], 3),
            end_s=round(max(end, new_starts[k] + 0.001), 3),
            source=wt.source,
            detail=(wt.detail + " + valley snap (fix round 4)"
                    if wt.detail else "valley snap (fix round 4)")))
    return out, stats


# --------------------------------------------------------- span hygiene


def effective_word_spans(
    timings: Sequence[WordTiming],
) -> list[tuple[float, float]]:
    """Render-effective (start, end) per word (fix round 4).

    CTC spans can be 40ms slivers: at 30fps the highlight lands
    for a single frame and reads as a flicker, not a word. The
    effective end extends each word to at least
    MIN_ACTIVE_WORD_S (capped at the next word's start, so spans
    never overlap); the final word extends to start +
    MIN_ACTIVE_WORD_S unconditionally. Starts never move - this
    is a display-span rule, not a timing correction.
    """
    out: list[tuple[float, float]] = []
    n = len(timings)
    for i, wt in enumerate(timings):
        end = wt.end_s
        if i + 1 < n:
            end = max(end, min(wt.start_s + MIN_ACTIVE_WORD_S,
                               timings[i + 1].start_s))
        else:
            end = max(end, wt.start_s + MIN_ACTIVE_WORD_S)
        out.append((wt.start_s, end))
    return out


def active_index_from_spans(spans: Sequence[tuple[float, float]],
                            t_s: float) -> int | None:
    """The held active index over effective spans.

    The active word is the LAST word that has started, held
    until the next word starts (no dark gap inside a chunk);
    None before the first word's start or at/after the final
    word's effective end.
    """
    if not spans or t_s < spans[0][0] or t_s >= spans[-1][1]:
        return None
    idx: int | None = None
    for i, (s, _e) in enumerate(spans):
        if s <= t_s:
            idx = i
        else:
            break
    return idx


def active_word_index_held(timings: Sequence[WordTiming],
                           t_s: float) -> int | None:
    """``active_index_from_spans`` over a cue's effective spans."""
    return active_index_from_spans(effective_word_spans(timings),
                                   t_s)


def active_word_index(timings: Sequence[WordTiming],
                      t_s: float) -> int | None:
    """Index of the word being spoken at ``t_s``, or None.

    The word-sweep highlight's only input: an index a real timing
    source produced for this instant. Outside every word's span
    (a pause, before the first word, after the last) there is no
    active word and the sweep shows nothing - it never guesses
    the nearest word.
    """
    for i, wt in enumerate(timings):
        if wt.start_s <= t_s < wt.end_s:
            return i
    return None


# ---------------------------------------------------------------- dispatch


def word_timings(words: Sequence[str], t0: float, t1: float, *,
                 source: str = "dsp",
                 profile: EnergyProfile | None = None,
                 samples: np.ndarray | None = None,
                 sample_rate: int = MMS_SAMPLE_RATE,
                 thresh_db: float | None = None) -> list[WordTiming]:
    """The one interface. ``source`` names the implementation and
    the returned timings are labelled with it - truthfully.

    - ``"dsp"``: needs the span [t0, t1] and (optionally) the
      energy profile.
    - ``"mms"``: needs ``samples`` covering [t0, t1]'s audio
      (the caller slices the turn's samples; timings come back
      offset by t0). Raises MMSUnavailableError when the aligner
      cannot run - NEVER substitutes DSP numbers.
    """
    if source == "dsp":
        return dsp_word_timings(words, t0, t1, profile,
                                thresh_db=thresh_db)
    if source == "mms":
        if samples is None:
            raise WordTimingError(
                "source='mms' requires the turn's audio samples")
        return mms_word_timings(words, samples, sample_rate,
                                offset_s=t0)
    raise WordTimingError(
        f"unknown word-timing source {source!r}; "
        f"known: {', '.join(TIMING_SOURCES)}")
