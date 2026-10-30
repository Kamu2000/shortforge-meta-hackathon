"""The pipeline: one long video in, up to N verified Shorts out.

Stages (the converged design, assembled end to end):

    ingest (ffprobe)
      -> Voice Transcribe (diarized turns; chunked at the API's
         10-minute cap)
      -> Spark Open sweep (Baseten; candidate moments, partner log)
      -> per selected moment:
           cut window -> SAM 3.1 (cached per window; the response
           is saved next to the Short and re-used on re-runs)
           -> mask_decode (primary speaker + finish stack)
           -> captions (Spark 1.3 punctuation under the word guard
              + DSP phrase timing)
           -> hook line (Spark 1.3, one call, deterministic
              fallbacks)
           -> layout (hook band + caption block from head geometry)
           -> composite render (draft 1)
           -> verifier F1-F7 + Wonder judge (receipt 1)
           -> rescue loop if draft 1 failed (ONE menu repair,
              re-verify, chained receipt)
      -> per-Short final MP4 + chained receipt, and a run summary
         (per-Short cost from ledger deltas, total cost, verdicts,
         door-to-door seconds).

Every paid call goes through the repo clients with the SpendLedger
attached; the summary's costs are ledger deltas measured around each
stage, never estimates. SAM is bought once per window - the rescue
loop works from the cached response, by construction.

CLI: ``python -m spatial_director.pipeline <video> --out <dir>
[--max-shorts N]``. On the team machine the clients resolve their
credentials through the Secure Vault surrogate pattern (the same
wiring as every live proof in the BUILDLOG); API keys can also be
supplied via the SPATIAL_DIRECTOR_META_API_KEY /
SPATIAL_DIRECTOR_BASETEN_API_KEY environment variables.

Written fresh in-repo on 2026-10-29.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .caption_align import (
    PEAK_CEILING_DBFS,
    build_captions,
    load_energy,
    plan_audio_normalisation,
)
from .compositor import DEFAULT_CANVAS, read_video_frames, video_fps
from .graphics_layer import build_graphics
from .image_client import ImageClient
from .layout_law import (
    EFFECT_OCCLUSION_BAND,
    apply_speaker_reframe,
    effect_note,
    plan_captions,
    plan_hook_effect,
    plan_speaker_reframe,
    render_hook_layer,
    to_canvas_alpha,
)
from .mask_decode import finished_alpha, parse_sam_response
from .rescue_loop import (
    RenderArtifacts,
    graphics_entries,
    render_composite_with_captions,
    rescue,
)
from .sam_client import API_BASE as META_API_BASE
from .sam_client import SamClient
from .spark_client import SparkClient
from .spark_open_client import Candidate, SparkOpenClient
from .spend_ledger import SpendLedger
from .verifier import HookSpec, Receipt, VerifyInputs, _content_words, verify
from .voice_transcribe import TranscriptTurn, VoiceTranscribeClient
from .wonder_client import WonderClient

# ---------------------------------------------------------------- constants

MIN_WINDOW_S = 14.0    # a Short shorter than this is not a Short
MAX_WINDOW_S = 22.0    # memory ceiling for the finish stack at 30fps
HOOK_WINDOW_S = 6.0
VT_CHUNK_S = 540.0     # under the API's 600s / 32MB per-request caps
HOOK_MAX_TOKENS = 2500  # Spark reasons first; tight budgets return empty

HOOK_SYSTEM_PROMPT = (
    "You write ONE hook line for a short vertical video clipped "
    "from a longer talk. Rules: 3 to 7 words; at least 3 content "
    "words; the line must contain a conflict word (never, no, quit, "
    "secret, truth, wrong, mistake...), a number, or be a question. "
    "Use only facts present in the transcript excerpt - do not "
    "invent numbers, names, or claims. Reply with the hook line "
    "only, nothing else."
)


class PipelineError(RuntimeError):
    """Any pipeline stage failure."""


def _annotate_effect(receipt: Receipt) -> None:
    """Append the depth-effect line to a receipt (annotation only).

    The F1-F7 verdict is untouched: the line reports where the
    MEASURED glyph occlusion stands against the planner's 8-20%
    effect band, so a draft whose speaker never crosses the hook
    text carries that fact on its receipt instead of passing as a
    silent success (the gate run's failure mode: 0.0 everywhere).
    """
    f2 = next((c for c in receipt.checks if c.check_id == "F2"), None)
    if f2 is None:
        return
    occ = float(f2.measured.get("glyph_occlusion_mean", 0.0))
    receipt.notes.append(effect_note(occ))


# ---------------------------------------------------------------- utilities


def _probe_duration(path: str | Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(probe.stdout.strip())


def _run_ffmpeg(args: list[str], what: str) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                          capture_output=True)
    if proc.returncode != 0:
        raise PipelineError(
            f"ffmpeg failed ({what}): "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}")


def _cut_window(source: str | Path, start_s: float, length_s: float,
                out: Path) -> None:
    _run_ffmpeg(
        ["-ss", f"{start_s:.3f}", "-i", str(source),
         "-t", f"{length_s:.3f}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-c:a", "aac", str(out)],
        f"cut window at {start_s:.1f}s")


def _extract_wav_chunk(source: str | Path, start_s: float,
                       length_s: float, out: Path) -> None:
    _run_ffmpeg(
        ["-ss", f"{start_s:.3f}", "-i", str(source),
         "-t", f"{length_s:.3f}", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out)],
        f"extract WAV chunk at {start_s:.1f}s")


SAM_CACHE_TOLERANCE_S = 0.05


def _sam_cache_hit(short_dir: Path, w0: float, w1: float) -> bool:
    """True only when the cached SAM response is FOR THIS WINDOW.

    The cache lives in the per-Short directory (keyed by index), so
    a sidecar records the window the response was bought for; a
    response whose sidecar is missing or names a different window
    is never re-used - a re-run whose sweep picks different windows
    must not silently inherit another moment's masks.
    """
    resp = short_dir / "sam-response.json"
    side = short_dir / "sam-window.json"
    if not resp.exists() or not side.exists():
        return False
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
        return (
            abs(float(meta["start_s"]) - w0) <= SAM_CACHE_TOLERANCE_S
            and abs(float(meta["end_s"]) - w1) <= SAM_CACHE_TOLERANCE_S
        )
    except Exception:
        return False


def _sam_cache_store(short_dir: Path, raw: Any, w0: float,
                     w1: float) -> None:
    (short_dir / "sam-response.json").write_text(
        json.dumps(raw), encoding="utf-8")
    (short_dir / "sam-window.json").write_text(
        json.dumps({"start_s": w0, "end_s": w1}), encoding="utf-8")


def _decode_native_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """Decode a media file's audio at its native rate and channels.

    Returns (samples float32 shaped (n, channels), sample_rate,
    channels).
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    sample_rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", str(channels), "-ar", str(sample_rate),
         "-f", "f32le", "pipe:1"],
        capture_output=True)
    if proc.returncode != 0:
        raise PipelineError(f"could not decode audio from {path}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    return samples.reshape(-1, channels), sample_rate, channels


def _normalise_window_clip(raw_clip: Path, out_clip: Path,
                           work_dir: Path) -> dict[str, Any]:
    """Apply the window's loudness plan; the video stream is copied.

    One static gain (``plan_audio_normalisation``) plus a true-peak
    limiter at PEAK_CEILING_DBFS - on the measured Holmes windows
    the limiter touches ~1 ms of transient peaks per window. When
    the planned gain is negligible the raw cut is copied through
    untouched. The plan (with its evidence) is returned for the run
    summary.
    """
    profile = load_energy(raw_clip)
    samples, sample_rate, channels = _decode_native_audio(raw_clip)
    peak_dbfs = float(20.0 * np.log10(max(np.abs(samples).max(), 1e-9)))
    plan = plan_audio_normalisation(profile, peak_dbfs)
    record: dict[str, Any] = {
        "gain_db": plan.gain_db,
        "speech_rms_db": plan.speech_rms_db,
        "peak_dbfs": plan.peak_dbfs,
        "target_db": plan.target_db,
        "applied": False,
        "clipped_samples": 0,
    }
    if plan.gain_db < 0.25:
        shutil.copyfile(raw_clip, out_clip)
        return record
    gained = samples * float(10.0 ** (plan.gain_db / 20.0))
    ceiling = float(10.0 ** (PEAK_CEILING_DBFS / 20.0))
    record["clipped_samples"] = int((np.abs(gained) > ceiling).sum())
    gained = np.clip(gained, -ceiling, ceiling)
    wav_path = work_dir / "normalised-audio.wav"
    pcm = (np.clip(gained, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    _run_ffmpeg(
        ["-i", str(raw_clip), "-i", str(wav_path),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_clip)],
        "mux normalised window audio")
    record["applied"] = True
    return record


class LazyCanvasAlphas:
    """Canvas-space alphas computed on demand (LRU-cached).

    Materialising a whole window's canvas alphas at once is
    gigabytes; the verifier and the layout planner index/stream the
    sequence, so each frame is mapped through the compositor's
    speaker transform only when asked for.
    """

    def __init__(self, src_alphas: list[np.ndarray],
                 canvas: tuple[int, int] = DEFAULT_CANVAS, cap: int = 32):
        self.src = src_alphas
        self.canvas = canvas
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cap = cap

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        a = to_canvas_alpha(self.src[i], self.canvas).astype(np.float32)
        self.cache[i] = a
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return a

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------- selection


def normalize_window(start_s: float, end_s: float,
                     duration_s: float) -> tuple[float, float] | None:
    """Clamp a candidate span to a workable Short window.

    Too-short spans grow symmetrically to MIN_WINDOW_S; too-long
    spans keep their start and cap at MAX_WINDOW_S (the finish
    stack's memory ceiling at source frame rates). Returns None if
    nothing usable remains inside the source.
    """
    start_s = max(0.0, float(start_s))
    end_s = min(float(duration_s), float(end_s))
    if end_s <= start_s:
        return None
    length = end_s - start_s
    if length < MIN_WINDOW_S:
        mid = (start_s + end_s) / 2.0
        start_s = max(0.0, mid - MIN_WINDOW_S / 2.0)
        end_s = min(duration_s, start_s + MIN_WINDOW_S)
        start_s = max(0.0, end_s - MIN_WINDOW_S)
    elif length > MAX_WINDOW_S:
        end_s = start_s + MAX_WINDOW_S
    if end_s - start_s < 8.0:
        return None
    return round(start_s, 3), round(end_s, 3)


def select_windows(candidates: Sequence[Candidate], duration_s: float,
                   max_shorts: int) -> list[tuple[Candidate, float, float]]:
    """Top-scoring, non-overlapping candidate windows, in score order."""
    ranked = sorted(candidates, key=lambda c: -c.score)
    chosen: list[tuple[Candidate, float, float]] = []
    for cand in ranked:
        if len(chosen) >= max_shorts:
            break
        window = normalize_window(cand.start_s, cand.end_s, duration_s)
        if window is None:
            continue
        s, e = window
        overlap = any(
            min(e, ce) - max(s, cs) > 0.5 * min(e - s, ce - cs)
            for _c, cs, ce in chosen)
        if overlap:
            continue
        chosen.append((cand, s, e))
    return chosen


# ---------------------------------------------------------------- hook


def propose_hook(spark: SparkClient, transcript_text: str,
                 fallback_text: str, *, ts: str = ""
                 ) -> tuple[str, dict[str, Any] | None]:
    """One Spark 1.3 call for the hook line; deterministic fallbacks.

    The reply is validated (word count + content words) before use -
    a line that breaks the hook rules is replaced by words lifted
    from the moment itself, and the fallback is on the record.
    """
    def usable(line: str) -> bool:
        words = line.split()
        return 2 <= len(words) <= 9 and len(_content_words(line)) >= 3

    call: dict[str, Any] | None = None
    try:
        result = spark.chat(
            [{"role": "system", "content": HOOK_SYSTEM_PROMPT},
             {"role": "user", "content":
              f"Transcript excerpt: {transcript_text[:1200]}"}],
            max_tokens=HOOK_MAX_TOKENS,
            purpose="pipeline-hook-line", ts=ts)
        call = {"model": "muse-spark-1.3",
                "purpose": "pipeline-hook-line",
                "cost_usd": result.cost_usd,
                "request_id": result.request_id}
        for raw_line in result.text.splitlines():
            line = raw_line.strip().strip("\"'").upper()
            if line and usable(line):
                return line, call
    except Exception:
        call = None
    fallback = " ".join(fallback_text.replace("\n", " ").split()[:7]).upper()
    if usable(fallback):
        return fallback, call
    return "WATCH THIS MOMENT", call


# ---------------------------------------------------------------- clients


def upload_media_file(path: str | Path, *, api_key_provider: Callable[[], str]
                      ) -> str:
    """Upload a media file to the Meta Model API; returns file_id.

    Mirrors the verified /v1/files multipart shape (purpose +
    file parts) that the SAM and Wonder paths both use.
    """
    path = Path(path)
    boundary = "----pipeline" + uuid.uuid4().hex
    data = path.read_bytes()
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (b'Content-Disposition: form-data; name="purpose"\r\n\r\n'
             b"user_data\r\n")
    body += f"--{boundary}\r\n".encode()
    body += (f'Content-Disposition: form-data; name="file"; '
             f'filename="{path.name}"\r\n').encode()
    body += b"Content-Type: video/mp4\r\n\r\n" + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{META_API_BASE}/files", data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key_provider()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        })

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            raise RuntimeError("redirect refused")

    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=300) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise PipelineError(f"file upload failed (HTTP {exc.code}): "
                            f"{detail}") from exc
    if "id" not in raw:
        raise PipelineError(f"file upload returned no id: {raw}")
    return str(raw["id"])


# ---------------------------------------------------------------- the run


@dataclass
class _Clients:
    sam: SamClient
    spark_open: SparkOpenClient
    spark: SparkClient
    vt: VoiceTranscribeClient
    wonder: WonderClient
    ledger: SpendLedger
    upload_file: Callable[[str | Path], str]
    # Muse Image, for the context graphics layer only; consulted
    # solely when a run enables graphics.
    image: ImageClient | None = None


def _transcribe_all(video: Path, work_dir: Path, clients: _Clients,
                    *, ts: str, log: Callable[[str], None]
                    ) -> tuple[list[TranscriptTurn], float]:
    duration = _probe_duration(video)
    work_dir = Path(work_dir) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    turns: list[TranscriptTurn] = []
    offset = 0.0
    while offset < duration:
        length = min(VT_CHUNK_S, duration - offset)
        wav = work_dir / f"vt-chunk-{int(offset)}.wav"
        _extract_wav_chunk(video, offset, length, wav)
        result = clients.vt.transcribe_file(
            wav, purpose="pipeline-voice-transcribe", ts=ts)
        for t in result.turns:
            start_ms = (t.start_ms or 0) + int(offset * 1000)
            end_ms = (t.end_ms + int(offset * 1000)
                      if t.end_ms is not None else None)
            turns.append(TranscriptTurn(
                text=t.text, start_ms=start_ms, end_ms=end_ms,
                speaker=t.speaker))
        log(f"transcribed {offset:.0f}-{offset + length:.0f}s: "
            f"{len(result.turns)} turns")
        offset += length
    turns.sort(key=lambda t: t.start_ms or 0)
    return turns, duration


def _local_turns(turns: Sequence[TranscriptTurn], w0: float, w1: float
                 ) -> tuple[list[TranscriptTurn], str]:
    """Turns overlapping the window, re-based to window-local seconds."""
    local: list[TranscriptTurn] = []
    texts: list[str] = []
    for t in turns:
        start = (t.start_ms or 0) / 1000.0
        end = (t.end_ms or 0) / 1000.0
        if end <= start:
            end = start + 5.0
        if end <= w0 or start >= w1:
            continue
        local.append(TranscriptTurn(
            text=t.text,
            start_ms=int(max(0.0, start - w0) * 1000),
            end_ms=int(min(w1 - w0, end - w0) * 1000),
            speaker=t.speaker))
        texts.append(t.text)
    return local, " ".join(texts)


def _process_short(index: int, cand: Candidate, w0: float, w1: float,
                   video: Path, turns: list[TranscriptTurn],
                   out_dir: Path, clients: _Clients, *, ts: str,
                   canvas: tuple[int, int],
                   log: Callable[[str], None],
                   graphics_enabled: bool = False) -> dict[str, Any]:
    ledger = clients.ledger
    cost_before = ledger.total_usd()
    short_dir = out_dir / f"short-{index}"
    short_dir.mkdir(parents=True, exist_ok=True)
    length = w1 - w0
    log(f"short {index}: window {w0:.1f}-{w1:.1f}s "
        f"(candidate score {cand.score:.2f}: {cand.reason})")

    raw_clip = short_dir / "window-raw.mp4"
    _cut_window(video, w0, length, raw_clip)
    # Loudness normalisation (static gain + peak limiter; the DSP's
    # relative silence threshold keeps F5's gap measurement honest
    # under the gain - see caption_align). The normalised cut is the
    # window everything downstream renders and measures.
    window_clip = short_dir / "window.mp4"
    norm = _normalise_window_clip(raw_clip, window_clip, short_dir)
    log(f"short {index}: audio normalisation gain "
        f"{norm['gain_db']:+.2f} dB (speech RMS "
        f"{norm['speech_rms_db']}, peak {norm['peak_dbfs']} dBFS, "
        f"applied {norm['applied']}, limiter touched "
        f"{norm['clipped_samples']} samples)")
    fps = video_fps(window_clip)
    frames = read_video_frames(window_clip)
    log(f"short {index}: cut {len(frames)} frames @ {fps:.2f} fps")

    # SAM, cached per window: a saved response is re-used only when
    # its sidecar proves it was bought for THIS window.
    sam_path = short_dir / "sam-response.json"
    if _sam_cache_hit(short_dir, w0, w1):
        raw_sam = json.loads(sam_path.read_text(encoding="utf-8"))
        log(f"short {index}: SAM response from cache "
            f"(window verified against sidecar)")
    else:
        if sam_path.exists():
            log(f"short {index}: cached SAM response is not for "
                f"this window; buying segmentation")
        file_id = clients.upload_file(window_clip)
        seg = clients.sam.segment_video(
            file_id=file_id, n_frames=len(frames),
            purpose=f"pipeline-sam-short{index}", ts=ts)
        raw_sam = seg.raw
        _sam_cache_store(short_dir, raw_sam, w0, w1)
        log(f"short {index}: SAM segmented "
            f"({seg.billed_frames} frames billed)")
    parsed = parse_sam_response(raw_sam)
    # Alignment policy: SAM may return one frame fewer than the container
    # decodes (a sub-frame tail artifact: a 22.0s window at 29.97fps is
    # 659.34 frames - ffmpeg yields 660, SAM returns 659). Both sequences
    # decode from frame 0 in order, so a surplus of DECODED frames is the
    # tail: truncate the guides to the parsed count and log it. The reverse
    # (SAM returning MORE frames than decoded) is a real misalignment and
    # still raises in finished_alpha's strict guard.
    n_parsed = len(parsed)
    if len(frames) > n_parsed:
        log(f"short {index}: aligning guides - {len(frames)} decoded "
            f"frames vs {n_parsed} SAM frames; truncating "
            f"{len(frames) - n_parsed} tail frame(s)")
        frames = frames[:n_parsed]
    # Guides stream into the finish pass as a generator: the list
    # form cost a full float64 copy of the window (~1.2 GB at 660
    # frames) that then coexisted with the finish stack's own lists
    # (the 2026-10-29 OOM - see mask_decode.finished_alpha).
    guides = (f.astype(np.float64).mean(axis=2) / 255.0 for f in frames)
    _indices, src_alphas = finished_alpha(parsed, guides)

    # Speaker reframing by construction (F3): place the FACE, not
    # just the text around it. The transform is PLANNED here but
    # applied after the hook is known: the depth-effect planner
    # (below) may add lift to the construction reframe so the
    # speaker's head/shoulders actually reach the hook band, and
    # the transform must be applied exactly once, at its final
    # value, so layout, render and verifier all work from the same
    # reframed geometry.
    reframe = plan_speaker_reframe(src_alphas, canvas)

    def make_canvas(alphas: list[np.ndarray]) -> LazyCanvasAlphas:
        return LazyCanvasAlphas(alphas, canvas)

    # Captions + hook for this moment.
    local_turns, excerpt = _local_turns(turns, w0, w1)
    energy = load_energy(window_clip)
    cues, cap_report = build_captions(
        local_turns, energy, spark=clients.spark,
        purpose=f"pipeline-caption-punctuation-short{index}", ts=ts)
    (short_dir / "captions.json").write_text(json.dumps({
        "cues": [{"text": c.text, "start_s": c.start_s, "end_s": c.end_s}
                 for c in cues],
        "turn_reports": [
            {"turn_index": t.turn_index, "used_fallback": t.used_fallback,
             "spark_request_id": t.spark_request_id,
             "spark_cost_usd": t.spark_cost_usd, "note": t.note}
            for t in cap_report.turns],
    }, indent=2), encoding="utf-8")
    hook_text, hook_call = propose_hook(
        clients.spark, excerpt,
        cues[0].text if cues else cand.reason, ts=ts)
    log(f"short {index}: hook {hook_text!r}; {len(cues)} caption cues")

    # Depth-effect planning: aim the hook at the 8-20% occlusion
    # band, lifting the speaker via the reframe machinery when no
    # band position reaches it. The returned transform is the one
    # applied - once - before anything is laid out or rendered.
    hook_sample = min(len(src_alphas), int(HOOK_WINDOW_S * fps))
    effect = plan_hook_effect(
        hook_text, src_alphas, canvas, base_reframe=reframe,
        hook_sample=hook_sample)
    if effect.pad_bottom_px != 0:
        frames, src_alphas = apply_speaker_reframe(
            frames, src_alphas, effect.pad_bottom_px)
    log(f"short {index}: speaker reframe "
        f"{'identity' if reframe.identity else reframe.note} "
        f"(feasible {reframe.feasible}); effect: {effect.note} "
        f"(applied pad {effect.pad_bottom_px})")
    canvas_alphas = LazyCanvasAlphas(src_alphas, canvas)

    # Layout from the decoded geometry.
    placement = effect.placement
    cap_layout = plan_captions(
        [(c.start_s, c.end_s) for c in cues], canvas_alphas,
        fps=fps, canvas=canvas)
    log(f"short {index}: layout hook frac "
        f"{placement.y_center_frac:.2f} (predicted occlusion "
        f"{placement.predicted_occlusion:.3f}, effect "
        f"{placement.effect_status}); caption rect "
        f"{cap_layout.rect} feasible {cap_layout.feasible}")

    hook_spec = HookSpec(hook_text, (0.0, min(HOOK_WINDOW_S, length)),
                         placement.y_center_frac, layout_rendered=True)
    hook_rgba = render_hook_layer(
        hook_text, canvas, y_center_frac=placement.y_center_frac)

    # Context graphics (flag-gated, default OFF - Kamlesh's
    # 2026-10-30 feature): planned from the caption phrases against
    # the reframed geometry, placed on the backdrop plane, and
    # carried through render, verify (F8) and the rescue artifacts.
    # Disabled = None everywhere below = the pre-graphics code path.
    placed_graphics = None
    graphics_report = None
    if graphics_enabled:
        _glyph_mask, glyph_bbox = hook_spec.glyph(canvas)
        build = build_graphics(
            cues, canvas_alphas, fps=fps, canvas=canvas,
            caption_rect=cap_layout.rect,
            hook_glyph_bbox=glyph_bbox,
            hook_window=hook_spec.window, duration_s=length,
            spark=clients.spark, image_client=clients.image, ts=ts,
            purpose_prefix=f"pipeline-graphics-short{index}")
        placed_graphics = build.specs
        graphics_report = build.to_dict()
        (short_dir / "graphics-plan.json").write_text(
            json.dumps(graphics_report, indent=2), encoding="utf-8")
        log(f"short {index}: graphics planned "
            f"{len(build.specs)} placed, {len(build.dropped)} dropped"
            f"{' (' + build.error + ')' if build.error else ''}")

    # Draft 1 render + verify.
    draft1 = short_dir / "draft-1.mp4"
    render_kwargs: dict[str, Any] = {}
    if placed_graphics is not None:
        render_kwargs["graphics"] = graphics_entries(placed_graphics)
    render_composite_with_captions(
        frames, src_alphas, hook_rgba=hook_rgba,
        hook_window=hook_spec.window, cues=cues,
        caption_rect=cap_layout.rect, audio_from=window_clip,
        out_path=draft1, fps=fps, canvas=canvas, **render_kwargs)
    receipt1 = verify(
        VerifyInputs(
            video_path=draft1, fps=fps, canvas_alphas=canvas_alphas,
            audio=load_energy(draft1), hook=hook_spec, cues=cues,
            caption_rect=cap_layout.rect, sam_response_path=sam_path,
            graphics=placed_graphics),
        wonder=clients.wonder, draft=1, ts=ts)
    _annotate_effect(receipt1)
    log(f"short {index}: draft 1 verdict {receipt1.verdict} "
        f"failed {receipt1.failed}")

    # Rescue (short-circuits when draft 1 passed).
    artifacts = RenderArtifacts(
        window_clip=window_clip, render_path=draft1, frames=frames,
        src_alphas=src_alphas, canvas_alphas=canvas_alphas,
        make_canvas_alphas=make_canvas, fps=fps, hook=hook_spec,
        cues=cues, caption_rect=cap_layout.rect,
        sam_response_path=sam_path, transcript_text=excerpt,
        work_dir=short_dir, canvas=canvas, graphics=placed_graphics)

    def verify_fn(execution) -> Any:
        receipt2 = verify(
            VerifyInputs(
                video_path=execution.render_path, fps=fps,
                canvas_alphas=execution.canvas_alphas,
                audio=load_energy(execution.render_path),
                hook=execution.hook, cues=execution.cues,
                caption_rect=execution.caption_rect,
                sam_response_path=sam_path,
                graphics=execution.graphics),
            wonder=clients.wonder, draft=2, ts=ts)
        _annotate_effect(receipt2)
        return receipt2

    chain = rescue(receipt1, artifacts, spark=clients.spark,
                   verify_fn=verify_fn, ts=ts)
    log(f"short {index}: final verdict {chain.final_verdict} "
        f"(repair: {chain.repair.status} {chain.repair.menu_id or ''})")

    final_path = short_dir / "final.mp4"
    if Path(chain.artifact) != final_path:
        shutil.copyfile(chain.artifact, final_path)
    (short_dir / "receipt.json").write_text(chain.to_json(),
                                            encoding="utf-8")
    cost = round(ledger.total_usd() - cost_before, 4)
    report: dict[str, Any] = {
        "index": index,
        "window": {"start_s": w0, "end_s": w1},
        "candidate": {"score": cand.score, "reason": cand.reason},
        "hook_text": hook_text,
        "hook_call": hook_call,
        "layout": {
            "hook_y_frac": placement.y_center_frac,
            "predicted_occlusion": round(placement.predicted_occlusion, 4),
            "caption_rect": list(cap_layout.rect),
            "caption_feasible": cap_layout.feasible,
            "effect": {
                "band": list(EFFECT_OCCLUSION_BAND),
                "predicted_status": placement.effect_status,
                "lift_px": effect.lift_px,
                "geometry_ok": effect.geometry_ok,
                "note": effect.note,
            },
        },
        "audio_normalisation": norm,
        "speaker_reframe": {
            "pad_bottom_px": reframe.pad_bottom_px,
            "applied_pad_bottom_px": effect.pad_bottom_px,
            "placed_centroid": reframe.placed_centroid,
            "placed_head_top_min": reframe.placed_head_top_min,
            "feasible": reframe.feasible,
            "identity": reframe.identity,
            "note": reframe.note,
        },
        "draft_1": {"verdict": receipt1.verdict,
                    "failed_checks": receipt1.failed},
        "repair": chain.repair.to_dict(),
        "final_verdict": chain.final_verdict,
        "cost_usd": cost,
        "files": {"final": str(final_path),
                  "receipt": str(short_dir / "receipt.json")},
    }
    if graphics_report is not None:
        report["graphics"] = graphics_report
    return report


def run_pipeline(video_path: str | Path, out_dir: str | Path, *,
                 sam: SamClient, spark_open: SparkOpenClient,
                 spark: SparkClient, vt: VoiceTranscribeClient,
                 wonder: WonderClient, ledger: SpendLedger,
                 upload_file: Callable[[str | Path], str],
                 ts: str = "", max_shorts: int = 3,
                 canvas: tuple[int, int] = DEFAULT_CANVAS,
                 log: Callable[[str], None] = print,
                 image: ImageClient | None = None,
                 graphics: bool = False) -> dict[str, Any]:
    """Run the whole pipeline on one long video; returns the summary.

    ``graphics`` (default False) enables the context graphics layer
    per Short; False runs exactly the pre-graphics pipeline.
    """
    started = time.monotonic()
    video = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clients = _Clients(sam=sam, spark_open=spark_open, spark=spark,
                       vt=vt, wonder=wonder, ledger=ledger,
                       upload_file=upload_file, image=image)
    total_before = ledger.total_usd()
    summary: dict[str, Any] = {
        "source": str(video), "created": ts, "max_shorts": max_shorts,
        "shorts": [],
    }
    try:
        turns, duration = _transcribe_all(video, out, clients,
                                          ts=ts, log=log)
        shared_before_sweep = ledger.total_usd()
        sweep_turns = []
        for t in turns:
            start = (t.start_ms or 0) / 1000.0
            end = (t.end_ms or 0) / 1000.0
            if end <= start:
                end = start + 5.0
            if t.text.strip():
                sweep_turns.append({"start_s": start, "end_s": end,
                                    "text": t.text})
        sweep = spark_open.sweep(
            sweep_turns, purpose="pipeline-candidate-sweep", ts=ts)
        log(f"sweep: {len(sweep.candidates)} candidates from "
            f"{len(sweep_turns)} turns over {duration:.0f}s")
        summary["sweep"] = {
            "candidates": [
                {"start_s": c.start_s, "end_s": c.end_s,
                 "score": c.score, "reason": c.reason}
                for c in sweep.candidates],
            "request_id": sweep.request_id,
        }
        # VT is charged to the run; the sweep draws partner credit
        # (not the Meta ledger), so the shared Meta cost is VT only.
        summary["shared_cost_usd"] = round(
            shared_before_sweep - total_before, 4)
        chosen = select_windows(sweep.candidates, duration, max_shorts)
        log(f"selected {len(chosen)} windows")
        if graphics:
            summary["graphics_enabled"] = True
        for i, (cand, w0, w1) in enumerate(chosen, start=1):
            summary["shorts"].append(_process_short(
                i, cand, w0, w1, video, turns, out, clients,
                ts=ts, canvas=canvas, log=log,
                graphics_enabled=graphics))
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["door_to_door_s"] = round(time.monotonic() - started, 1)
        summary["total_cost_usd"] = round(
            ledger.total_usd() - total_before, 4)
        (out / "run-summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        raise
    summary["door_to_door_s"] = round(time.monotonic() - started, 1)
    summary["total_cost_usd"] = round(ledger.total_usd() - total_before, 4)
    summary["ledger_total_usd"] = ledger.total_usd()
    (out / "run-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log(f"run complete: {len(summary['shorts'])} shorts, "
        f"${summary['total_cost_usd']:.4f} Meta spend, "
        f"{summary['door_to_door_s']:.0f}s door to door")
    return summary


# ---------------------------------------------------------------- CLI wiring


def _no_redirect_transport(method, url, headers, body):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            raise RuntimeError("redirect refused")

    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1200]
        return exc.code, {"error_body": detail}


def _vt_transport(url, headers, body):
    return _no_redirect_transport("POST", url, headers, body)


def _key_provider(env_var: str, vault_provider: str):
    """Env key if set, else the Secure Vault surrogate for the provider."""
    env_key = os.environ.get(env_var)
    if env_key:
        return lambda: env_key
    sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
    from dynamic_credentials import dynamic_credential_entry
    return lambda: str(
        dynamic_credential_entry(vault_provider)["surrogate"]).strip()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-shorts", type=int, default=3)
    parser.add_argument(
        "--graphics", action="store_true",
        help="enable the context graphics layer (timed cards/images "
             "on the backdrop plane, F8-verified; also enabled by "
             "GRAPHICS_LAYER=1). Default off.")
    parser.add_argument("--ts", default="2026-10-29")
    parser.add_argument(
        "--ledger", type=Path,
        default=Path(__file__).resolve().parents[2] / "spend-ledger.jsonl")
    args = parser.parse_args(argv)

    meta_key = _key_provider("SPATIAL_DIRECTOR_META_API_KEY",
                             "custom.meta-model-api")
    baseten_key = _key_provider("SPATIAL_DIRECTOR_BASETEN_API_KEY",
                                "custom.baseten")
    ledger = SpendLedger(args.ledger)
    repo_root = Path(__file__).resolve().parents[2]
    summary = run_pipeline(
        args.video, args.out,
        sam=SamClient(api_key_provider=meta_key, ledger=ledger,
                      transport=_no_redirect_transport),
        spark_open=SparkOpenClient(
            api_key_provider=baseten_key, transport=_no_redirect_transport,
            usage_log_path=repo_root / "partner-usage-log.jsonl"),
        spark=SparkClient(api_key_provider=meta_key, ledger=ledger,
                          transport=_no_redirect_transport),
        vt=VoiceTranscribeClient(api_key_provider=meta_key, ledger=ledger,
                                 transport=_vt_transport),
        wonder=WonderClient(api_key_provider=meta_key, ledger=ledger,
                            transport=_no_redirect_transport),
        image=ImageClient(api_key_provider=meta_key, ledger=ledger,
                          transport=_no_redirect_transport),
        ledger=ledger,
        upload_file=lambda p: upload_media_file(
            p, api_key_provider=meta_key),
        ts=args.ts, max_shorts=args.max_shorts,
        graphics=args.graphics
        or os.environ.get("GRAPHICS_LAYER") == "1")
    print(json.dumps({
        "shorts": len(summary["shorts"]),
        "verdicts": [s["final_verdict"] for s in summary["shorts"]],
        "total_cost_usd": summary["total_cost_usd"],
        "door_to_door_s": summary["door_to_door_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
