"""Procedural sound effects for the context graphics layer.

Block 2 (2026-10-31): graphics events get sound - an entrance
whoosh under a gliding card, ticks under a counting number, a soft
ding when a card lands. The discipline is the same as the rest of
the graphics layer:

- **Synthesis is code, never assets.** Every recipe is a closed-form
  NumPy waveform (band-swept noise, pitch-dropping sine bursts,
  decaying harmonic pings) generated with a fixed seed - the same
  recipe at the same offset produces the same samples on every run,
  so a receipt's declaration can be re-derived and re-measured.
  No library sounds, no downloads, nothing to license.
- **Events are derived, not chosen.** ``plan_sfx`` reads the placed
  graphics and derives each effect from the graphic's own motion:
  a path motion earns an entrance whoosh and a landing ding, a
  count-up earns ticks across the count plus a completion ding, a
  two-state build earns a pop per state, a reveal earns one soft
  ding, a static card earns a pop, an image's Ken Burns earns a
  soft whoosh. Nothing is placed by hand and nothing is invented.
- **Ducking is measured, not hoped for.** At mix time each event's
  peak is capped against the speech RMS around it (and against an
  absolute ceiling in silence), so an effect supports the speech
  and never talks over it. The gain that was ACTUALLY applied is
  what the receipt declares.
- **F5 never hears the mix.** The verifier measures dead air on
  the pre-mix speech stem (``verifier.VerifyInputs.speech_audio``);
  this module only ever produces the shipped mix. An effect that
  fills a speech gap makes the render SOUND continuous while F5
  still fails the gap on the stem - by design, and proven in the
  proof pack.

Audio is mono float32 at the pipeline's DSP rate (16 kHz), the
same representation the trim repair already works in.

Written fresh in-repo on 2026-10-31.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import subprocess
import wave

import numpy as np

from .graphics_layer import PATH_TRAVEL_S, GraphicSpec

SAMPLE_RATE = 16000

# Recipe peak ceilings (dBFS) before ducking: the loudest an effect
# of that kind may ever be in the mix.
RECIPE_CEILING_DB = {
    "whoosh": -15.0,
    "pop": -13.0,
    "tick": -19.0,
    "ding": -17.0,
    "click": -19.0,
}
# An event's peak may not exceed the speech RMS around it minus
# this headroom; in silence the absolute ceiling applies instead.
DUCK_HEADROOM_DB = 4.0
DUCK_SILENCE_CEILING_DB = -16.0
_SPEECH_FLOOR_DB = -55.0     # below this the stem counts as silent
_MIX_PEAK_MAX = 0.99         # global safety ceiling of the mix
_SEED = 20261031             # fixed: synthesis is deterministic


@dataclass
class SfxEvent:
    """One intended effect, derived from a placed graphic."""

    t_s: float
    recipe: str
    graphic_index: int
    anchor: str = "enter"    # enter / land / count / state / reveal

    def to_dict(self) -> dict[str, Any]:
        return {"t_s": round(self.t_s, 3), "recipe": self.recipe,
                "graphic_index": self.graphic_index,
                "anchor": self.anchor}


# ---------------------------------------------------------------- synthesis


def _env(n: int, kind: str) -> np.ndarray:
    """Amplitude envelopes: bump (smooth in/out) or decay."""
    t = np.linspace(0.0, 1.0, max(1, n), dtype=np.float64)
    if kind == "bump":
        return np.sin(np.pi * t) ** 2
    if kind == "decay":
        return np.exp(-5.0 * t)
    raise ValueError(f"unknown envelope {kind!r}")


def synthesize(recipe: str, sample_rate: int = SAMPLE_RATE,
               ) -> np.ndarray:
    """One effect's waveform (float64, peak-normalised to 1.0).

    The caller scales by the ducked gain; normalising here keeps
    every recipe's shape independent of its loudness.
    """
    rng = np.random.default_rng(_SEED)
    if recipe == "whoosh":
        dur, n = 0.45, int(0.45 * sample_rate)
        noise = rng.standard_normal(n)
        # Band-swept noise: FFT filter whose Gaussian passband
        # centre sweeps 400 -> 2400 Hz across the event.
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        centre = 400.0 + (2400.0 - 400.0) * np.linspace(0, 1, n)
        # A time-varying centre cannot be one static filter; use
        # the mean trajectory as a swept approximation via three
        # overlapping static bands cross-faded in time.
        out = np.zeros(n)
        bands = [(400.0, 0.0, 0.45), (1200.0, 0.3, 0.75),
                 (2400.0, 0.6, 1.0)]
        for centre_hz, t0, t1 in bands:
            gain = np.exp(-0.5 * ((freqs - centre_hz) / 350.0) ** 2)
            band = np.fft.irfft(spec * gain, n)
            w = np.zeros(n)
            i0, i1 = int(t0 * n), int(t1 * n)
            w[i0:i1] = np.sin(
                np.linspace(0, np.pi, max(1, i1 - i0))) ** 2
            out += band * w
        wave_ = out * _env(n, "bump")
    elif recipe == "pop":
        dur, n = 0.09, int(0.09 * sample_rate)
        t = np.arange(n) / sample_rate
        freq = 340.0 - 170.0 * (t / dur)      # pitch drops as it lands
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        wave_ = np.sin(phase) * _env(n, "decay")
    elif recipe == "tick":
        dur, n = 0.03, int(0.03 * sample_rate)
        t = np.arange(n) / sample_rate
        wave_ = np.sin(2.0 * np.pi * 2100.0 * t) * _env(n, "decay")
    elif recipe == "ding":
        dur, n = 0.50, int(0.50 * sample_rate)
        t = np.arange(n) / sample_rate
        wave_ = (np.sin(2.0 * np.pi * 880.0 * t)
                 + 0.45 * np.sin(2.0 * np.pi * 1318.0 * t)) \
            * _env(n, "decay")
    elif recipe == "click":
        dur, n = 0.04, int(0.04 * sample_rate)
        t = np.arange(n) / sample_rate
        wave_ = np.sin(2.0 * np.pi * 1750.0 * t) * _env(n, "decay")
    else:
        raise ValueError(f"unknown SFX recipe {recipe!r}")
    peak = float(np.max(np.abs(wave_))) if wave_.size else 0.0
    if peak > 0:
        wave_ = wave_ / peak
    return wave_.astype(np.float64)


# ---------------------------------------------------------------- planning


def plan_sfx(specs: Sequence[GraphicSpec]) -> list[SfxEvent]:
    """Derive the intended effects from placed graphics.

    Deterministic and purely a function of the specs (kind, motion,
    path, span) - a repair that changes the graphic list re-derives
    its effects through this same function, so a dropped graphic's
    sounds can never survive it.
    """
    events: list[SfxEvent] = []
    for i, spec in enumerate(specs):
        start, end = spec.start_s, spec.end_s
        span = max(0.0, end - start)
        if spec.path is not None:
            events.append(SfxEvent(start, "whoosh", i, "enter"))
            events.append(SfxEvent(
                start + PATH_TRAVEL_S, "ding", i, "land"))
        elif spec.motion == "ken_burns":
            events.append(SfxEvent(start, "whoosh", i, "enter"))
        elif spec.motion == "count_up":
            for k in range(6):
                events.append(SfxEvent(
                    start + 0.06 + (0.72 * span) * k / 5.0,
                    "tick", i, "count"))
            events.append(SfxEvent(
                start + 0.82 * span, "ding", i, "count_complete"))
        elif spec.motion == "build_two_state":
            events.append(SfxEvent(start, "pop", i, "state_1"))
            events.append(SfxEvent(
                start + 0.5 * span, "pop", i, "state_2"))
        elif spec.motion == "reveal":
            events.append(SfxEvent(start + 0.12, "ding", i, "reveal"))
        else:
            events.append(SfxEvent(start, "pop", i, "enter"))
    events.sort(key=lambda e: (e.t_s, e.graphic_index))
    return events


# ---------------------------------------------------------------- mixing


def _local_speech_rms_db(speech: np.ndarray, sample_rate: int,
                         t_s: float, dur_s: float) -> float:
    """Speech RMS (dBFS) in a padded window around an event."""
    i0 = max(0, int((t_s - 0.15) * sample_rate))
    i1 = min(len(speech), int((t_s + dur_s + 0.15) * sample_rate))
    if i1 <= i0:
        return -90.0
    seg = speech[i0:i1].astype(np.float64)
    rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    if rms <= 0:
        return -90.0
    return 20.0 * float(np.log10(rms))


def mix_sfx(speech: np.ndarray, sample_rate: int,
            events: Sequence[SfxEvent],
            ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Mix the intended effects under the speech stem.

    Returns (mix float32, applied declarations). Each event is
    peak-capped by ducking: its peak may reach at most the local
    speech RMS minus DUCK_HEADROOM_DB, or DUCK_SILENCE_CEILING_DB
    in silence, and never its recipe ceiling. A final global
    safety scale keeps the mix under _MIX_PEAK_MAX; applied gains
    are the peaks as they actually landed in the mix (dBFS), which
    is what the receipt declares.
    """
    mix = speech.astype(np.float64).copy()
    placed: list[tuple[SfxEvent, float, float]] = []  # event, peak_db, lin
    for event in events:
        wave_ = synthesize(event.recipe, sample_rate)
        dur_s = len(wave_) / sample_rate
        local_db = _local_speech_rms_db(speech, sample_rate,
                                        event.t_s, dur_s)
        if local_db <= _SPEECH_FLOOR_DB:
            cap_db = DUCK_SILENCE_CEILING_DB
        else:
            cap_db = local_db - DUCK_HEADROOM_DB
        peak_db = min(RECIPE_CEILING_DB[event.recipe], cap_db)
        lin = 10.0 ** (peak_db / 20.0)
        i0 = int(round(event.t_s * sample_rate))
        if i0 >= len(mix):
            continue
        i1 = min(len(mix), i0 + len(wave_))
        mix[i0:i1] += wave_[:i1 - i0] * lin
        placed.append((event, peak_db, lin))
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    global_db = 0.0
    if peak > _MIX_PEAK_MAX:
        factor = _MIX_PEAK_MAX / peak
        mix *= factor
        global_db = 20.0 * float(np.log10(factor))
    applied = [{
        "t_s": round(event.t_s, 3),
        "recipe": event.recipe,
        "gain_db": round(peak_db + global_db, 2),
        "graphic_index": event.graphic_index,
        "anchor": event.anchor,
    } for event, peak_db, _lin in placed]
    return mix.astype(np.float32), applied


# ---------------------------------------------------------------- io


def decode_audio_samples(media_path: str | Path) -> np.ndarray:
    """Decode a media file's audio to mono float32 at SAMPLE_RATE
    (the same representation caption_align's DSP works in)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(media_path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not decode audio from {media_path}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def write_wav(path: str | Path, samples: np.ndarray,
              sample_rate: int = SAMPLE_RATE) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return Path(path)


def mix_to_file(speech_path: str | Path, events: Sequence[SfxEvent],
                out_path: str | Path,
                ) -> list[dict[str, Any]]:
    """Decode the stem, mix the events over it, write the mix WAV.

    Returns the applied declarations for the receipt. With no
    events this still writes nothing and returns [] - callers use
    the stem itself as the render audio in that case.
    """
    if not events:
        return []
    speech = decode_audio_samples(speech_path)
    mix, applied = mix_sfx(speech, SAMPLE_RATE, events)
    write_wav(out_path, mix, SAMPLE_RATE)
    return applied
