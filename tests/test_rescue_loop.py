"""Tests for rescue_loop: the bounded, menu-driven repair.

Everything runs on fakes - a scripted Spark planner, a scripted
verifier, a recording renderer - plus one real audio file for the
TRIM_SILENCE executor (its gap detection is DSP, and the test proves
it finds and cuts a real pause). The properties under test are the
design's hard rules: the menu is enforced in code, ONE repair max,
no SAM anywhere in the loop, and the chained receipt's shape.
"""

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue  # noqa: E402
from spatial_director.graphics_layer import GraphicSpec  # noqa: E402
from spatial_director.layout_law import (  # noqa: E402
    frame_geometry,
    to_canvas_alpha,
)
from spatial_director.rescue_loop import (  # noqa: E402
    ENGAGE_HOOK_SLIDE,
    REFRAME_SPEAKER,
    REPAIR_MENU,
    RenderArtifacts,
    _exec_engage_hook_slide,
    rescue,
)
from spatial_director.verifier import (  # noqa: E402
    CheckResult,
    HookSpec,
    Receipt,
)

CANVAS = (1080, 1920)
FPS = 4.0


# ---------------------------------------------------------------- fakes


class FakeChatResult:
    def __init__(self, text, request_id="chatcmpl-fake"):
        self.text = text
        self.request_id = request_id
        self._cost = 0.01

    @property
    def cost_usd(self):
        return self._cost


class FakeSpark:
    """Scripted Spark: replies in order, records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, *, max_tokens=1500, purpose="", ts=""):
        self.calls.append({"messages": messages, "purpose": purpose})
        if not self.replies:
            raise AssertionError("FakeSpark: no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeChatResult(reply, f"chatcmpl-fake-{len(self.calls)}")


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def __call__(self, frames, src_alphas, *, hook_rgba, hook_window,
                 cues, caption_rect, audio_from, out_path, fps,
                 canvas=CANVAS, speaker_scale=1.0):
        self.calls.append({
            "n_frames": len(frames), "hook_window": hook_window,
            "cues": list(cues), "caption_rect": caption_rect,
            "audio_from": str(audio_from), "speaker_scale": speaker_scale,
        })
        Path(out_path).write_bytes(b"fake-render")
        return Path(out_path)


class GraphicsRenderer(FakeRenderer):
    """Accepts the graphics kwarg the shared render path passes
    when a render carries graphics, and records the entries."""

    def __call__(self, frames, src_alphas, *, graphics=None, **kwargs):
        self.graphics_seen = graphics
        return super().__call__(frames, src_alphas, **kwargs)


class FakeVerify:
    def __init__(self, receipts):
        self.receipts = list(receipts)
        self.executions = []

    def __call__(self, execution):
        self.executions.append(execution)
        return self.receipts.pop(0)


def make_receipt(failed=(), measured=None, draft=1):
    measured = measured or {}
    checks = []
    for cid in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        checks.append(CheckResult(
            check_id=cid, name=cid, passed=cid not in failed,
            measured=measured.get(cid, {}), threshold={}))
    return Receipt(
        artifact="draft1.mp4",
        verdict="FAIL" if failed else "PASSED_FIRST_DRAFT",
        checks=checks, draft=draft)


def write_wav(path: Path, segments, sr=16000):
    """segments: list of (duration_s, amplitude) - tone or silence."""
    chunks = []
    for dur, amp in segments:
        n = int(dur * sr)
        if amp > 0:
            t = np.arange(n) / sr
            chunks.append((amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    samples = np.concatenate(chunks)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def canvas_with_speaker(n=24, rows=(1200, 1920), cols=(240, 840)):
    out = []
    for _ in range(n):
        a = np.zeros((CANVAS[1], CANVAS[0]), dtype=np.float32)
        a[rows[0]:rows[1], cols[0]:cols[1]] = 1.0
        out.append(a)
    return out


def make_artifacts(work_dir: Path, *, window_clip="window.mp4",
                   canvas=None, n_frames=12, fps=FPS, hook=None,
                   cues=None):
    canvas = canvas if canvas is not None else canvas_with_speaker(n_frames)
    frames = [np.zeros((36, 64, 3), dtype=np.uint8)
              for _ in range(n_frames)]
    src_alphas = [np.zeros((36, 64), dtype=np.float64)
                  for _ in range(n_frames)]
    return RenderArtifacts(
        window_clip=window_clip,
        render_path=Path(work_dir) / "draft1.mp4",
        frames=frames, src_alphas=src_alphas,
        canvas_alphas=canvas,
        make_canvas_alphas=lambda alphas: canvas[:len(alphas)],
        fps=fps,
        hook=hook or HookSpec("THE BAND NEVER REHEARSES", (0.0, 6.0),
                              0.30, layout_rendered=True),
        cues=cues if cues is not None else [
            CaptionCue("First phrase here.", 0.2, 0.9),
            CaptionCue("After the pause now.", 1.9, 2.6),
        ],
        caption_rect=(60, 1460, 980, 1600),
        sam_response_path=None,
        transcript_text="it's an 11 piece band and we never rehearse",
        work_dir=work_dir,
    )


# ---------------------------------------------------------------- tests


class ShortCircuitTests(unittest.TestCase):
    def test_passing_receipt_never_enters_the_loop(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            spark = FakeSpark([])
            verify = FakeVerify([])
            chain = rescue(make_receipt(()), artifacts,
                           spark=spark, verify_fn=verify,
                           renderer=FakeRenderer(), ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "PASSED_FIRST_DRAFT")
            self.assertEqual(chain.repair.status, "not_needed")
            self.assertEqual(len(chain.drafts), 1)
            self.assertEqual(spark.calls, [])
            self.assertEqual(verify.executions, [])


class MenuEnforcementTests(unittest.TestCase):
    def test_pick_outside_the_menu_is_discarded(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            spark = FakeSpark([
                '{"repair": "DELETE_AND_RETRY", "reason": "trust me"}'])
            verify = FakeVerify([])
            chain = rescue(make_receipt(("F5",)), artifacts,
                           spark=spark, verify_fn=verify,
                           renderer=FakeRenderer(), ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "FAIL")
            self.assertEqual(chain.repair.status, "no_repair_available")
            self.assertEqual(chain.repair.menu_id, "DELETE_AND_RETRY")
            self.assertEqual(verify.executions, [])

    def test_pick_for_a_passing_check_is_discarded(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            spark = FakeSpark([
                '{"repair": "REPOSITION_HOOK", "reason": "move it"}'])
            chain = rescue(make_receipt(("F5",)), artifacts,
                           spark=spark, verify_fn=FakeVerify([]),
                           renderer=FakeRenderer(), ts="2026-10-29")
            self.assertEqual(chain.repair.status, "no_repair_available")
            self.assertIn("does not target", chain.repair.detail)

    def test_unparseable_planner_reply_means_no_repair(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            spark = FakeSpark(["I think you should trim the silence."])
            chain = rescue(make_receipt(("F5",)), artifacts,
                           spark=spark, verify_fn=FakeVerify([]),
                           renderer=FakeRenderer(), ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "FAIL")
            self.assertEqual(chain.repair.status, "no_repair_available")

    def test_planner_call_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            spark = FakeSpark([RuntimeError("HTTP 500")])
            chain = rescue(make_receipt(("F5",)), artifacts,
                           spark=spark, verify_fn=FakeVerify([]),
                           renderer=FakeRenderer(), ts="2026-10-29")
            self.assertEqual(chain.repair.status, "planner_failed")
            self.assertEqual(chain.final_verdict, "FAIL")

    def test_menu_covers_the_designed_mapping(self):
        self.assertEqual(REPAIR_MENU["TRIM_SILENCE"].target_checks, ("F5",))
        self.assertEqual(REPAIR_MENU["REPOSITION_HOOK"].target_checks, ("F2",))
        self.assertEqual(REPAIR_MENU["REFIT_HOOK_INK"].target_checks, ("F3",))
        self.assertEqual(REPAIR_MENU["RESCALE_SPEAKER"].target_checks, ("F4",))
        self.assertEqual(REPAIR_MENU["REGENERATE_HOOK_TEXT"].target_checks,
                         ("F1", "F6"))
        self.assertEqual(REPAIR_MENU["DROP_GRAPHICS"].target_checks,
                         ("F8",))
        # Defect B (2026-10-30): text cannot fix a motion failure;
        # F7 belongs to the slide repair alone.
        self.assertEqual(REPAIR_MENU[ENGAGE_HOOK_SLIDE].target_checks,
                         ("F7",))


class SlideRenderer(FakeRenderer):
    """FakeRenderer that also records the hook_slide kwarg."""

    def __call__(self, frames, src_alphas, *, hook_slide=None, **kwargs):
        self.hook_slide_seen = hook_slide
        return super().__call__(frames, src_alphas, **kwargs)


class EngageHookSlideTests(unittest.TestCase):
    """Defect B (2026-10-30 Holmes graphics run, short-2): a
    genuinely static window failed F7 alone and the text repair
    left every F7 measurement identical. The slide repair must
    produce designed motion by construction, at 200 px/s."""

    def test_slide_uses_room_and_reports_200_px_s(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            renderer = SlideRenderer()
            execution = _exec_engage_hook_slide(
                artifacts, make_receipt(("F7",)), renderer)
            dx, duration = execution.hook.slide
            self.assertEqual(execution.hook_slide_px_s, 200.0)
            self.assertEqual(renderer.hook_slide_seen, (dx, duration))
            self.assertAlmostEqual(abs(dx) / duration, 200.0)
            self.assertLessEqual(abs(dx), 240.0)
            self.assertEqual(execution.hook.text,
                             artifacts.hook.text)
            self.assertIn("200 px/s", execution.detail)

    def test_slide_direction_follows_the_roomier_side(self):
        # A hook whose glyphs hug the left edge must slide in
        # from the right (positive dx); the executor reads the
        # glyph bbox, not a fixed direction.
        with tempfile.TemporaryDirectory() as td:
            hook = HookSpec("THE BAND NEVER REHEARSES", (0.0, 6.0),
                            0.30, layout_rendered=True)
            _mask, bbox = hook.glyph()
            artifacts = make_artifacts(Path(td), hook=hook)
            execution = _exec_engage_hook_slide(
                artifacts, make_receipt(("F7",)), SlideRenderer())
            dx, _duration = execution.hook.slide
            left_room, right_room = bbox[0], 1080 - bbox[2]
            if left_room >= right_room:
                self.assertLess(dx, 0)
            else:
                self.assertGreater(dx, 0)

    def test_slide_repair_flows_through_rescue_loop(self):
        # Model picks ENGAGE_HOOK_SLIDE for an F7-only failure;
        # the loop executes it and the execution carries the
        # slide speed for the re-verify.
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            spark = FakeSpark([
                '{"repair": "ENGAGE_HOOK_SLIDE", '
                '"reason": "static opening needs designed motion"}',
            ])
            verify = FakeVerify([make_receipt((), draft=2)])
            chain = rescue(make_receipt(("F7",)), artifacts,
                           spark=spark, verify_fn=verify,
                           renderer=SlideRenderer(), ts="2026-10-30")
            self.assertEqual(chain.repair.menu_id, ENGAGE_HOOK_SLIDE)
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertEqual(verify.executions[0].hook_slide_px_s, 200.0)


class TrimSilenceTests(unittest.TestCase):
    def _run(self, td, wav_segments, verify_receipts):
        wav = write_wav(Path(td) / "window.wav", wav_segments)
        artifacts = make_artifacts(Path(td), window_clip=wav)
        spark = FakeSpark([
            '{"repair": "TRIM_SILENCE", "reason": "gap 0.68 over cap"}'])
        renderer = FakeRenderer()
        verify = FakeVerify(verify_receipts)
        chain = rescue(
            make_receipt(("F5",), {"F5": {"max_silence_gap_s": 0.68}}),
            artifacts, spark=spark, verify_fn=verify,
            renderer=renderer, ts="2026-10-29")
        return chain, renderer, verify, spark

    def test_trim_executes_and_repair_passes(self):
        with tempfile.TemporaryDirectory() as td:
            chain, renderer, verify, spark = self._run(
                td, [(1.0, 0.4), (0.8, 0.0), (1.2, 0.4)],
                [make_receipt((), draft=2)])
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertEqual(chain.repair.status, "executed")
            self.assertEqual(chain.repair.menu_id, "TRIM_SILENCE")
            self.assertEqual(chain.repair.target_check, "F5")
            self.assertEqual(len(chain.drafts), 2)
            # The 0.8s gap loses 0.5s: frames at t=1.25 and t=1.50
            # (fps=4) fall inside the (1.15, 1.65) removal.
            self.assertEqual(renderer.calls[0]["n_frames"], 10)
            retimed = {c.text: (c.start_s, c.end_s)
                       for c in renderer.calls[0]["cues"]}
            self.assertEqual(retimed["After the pause now."], (1.4, 2.1))
            self.assertEqual(retimed["First phrase here."], (0.2, 0.9))
            self.assertEqual(len(verify.executions), 1)
            self.assertEqual(len(spark.calls), 1)

    def test_one_repair_cap_second_failure_stands(self):
        with tempfile.TemporaryDirectory() as td:
            chain, renderer, verify, spark = self._run(
                td, [(1.0, 0.4), (0.8, 0.0), (1.2, 0.4)],
                [make_receipt(("F5",), draft=2)])
            self.assertEqual(chain.final_verdict, "FAIL")
            self.assertEqual(len(chain.drafts), 2)
            self.assertEqual(len(verify.executions), 1)
            self.assertEqual(len(spark.calls), 1)
            self.assertEqual(len(renderer.calls), 1)

    def test_trim_not_applicable_without_a_gap(self):
        with tempfile.TemporaryDirectory() as td:
            chain, renderer, verify, _ = self._run(
                td, [(3.0, 0.4)], [make_receipt((), draft=2)])
            self.assertEqual(chain.final_verdict, "FAIL")
            self.assertEqual(chain.repair.status, "not_applicable")
            self.assertEqual(renderer.calls, [])
            self.assertEqual(verify.executions, [])


class OtherExecutorTests(unittest.TestCase):
    def test_reposition_hook_moves_the_band(self):
        with tempfile.TemporaryDirectory() as td:
            canvas = canvas_with_speaker(24, rows=(620, 1500))
            hook = HookSpec("THE BAND NEVER REHEARSES", (0.0, 6.0),
                            0.37, layout_rendered=True)
            artifacts = make_artifacts(
                Path(td), canvas=canvas, n_frames=24, hook=hook)
            spark = FakeSpark([
                '{"repair": "REPOSITION_HOOK", "reason": "occlusion"}'])
            renderer = FakeRenderer()
            verify = FakeVerify([make_receipt((), draft=2)])
            chain = rescue(
                make_receipt(("F2",), {
                    "F2": {"glyph_occlusion_mean": 0.41,
                           "caption_face_overlap_max_px": 0}}),
                artifacts, spark=spark, verify_fn=verify,
                renderer=renderer, ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertNotEqual(
                verify.executions[0].hook.y_center_frac, 0.37)
            self.assertEqual(len(renderer.calls), 1)

    def test_refit_hook_ink_uses_the_layout_path(self):
        with tempfile.TemporaryDirectory() as td:
            hook = HookSpec("THE BAND NEVER REHEARSES", (0.0, 6.0),
                            0.30, layout_rendered=False)
            artifacts = make_artifacts(Path(td), hook=hook)
            _g, bbox = hook.glyph()
            self.assertTrue(bbox[0] < 60 or bbox[2] > 980)
            spark = FakeSpark([
                '{"repair": "REFIT_HOOK_INK", "reason": "ink overflow"}'])
            verify = FakeVerify([make_receipt((), draft=2)])
            chain = rescue(make_receipt(("F3",)), artifacts,
                           spark=spark, verify_fn=verify,
                           renderer=FakeRenderer(), ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            new_hook = verify.executions[0].hook
            self.assertTrue(new_hook.layout_rendered)
            _g, nb = new_hook.glyph()
            self.assertTrue(60 <= nb[0] and nb[2] <= 980
                            and 288 <= nb[1] and nb[3] <= 1440)

    def test_regenerate_hook_text(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            spark = FakeSpark([
                '{"repair": "REGENERATE_HOOK_TEXT", "reason": "no loop"}',
                "HE QUIT THE BAND TODAY",
            ])
            verify = FakeVerify([make_receipt((), draft=2)])
            chain = rescue(make_receipt(("F6",)), artifacts,
                           spark=spark, verify_fn=verify,
                           renderer=FakeRenderer(), ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertEqual(verify.executions[0].hook.text,
                             "HE QUIT THE BAND TODAY")
            self.assertEqual(len(spark.calls), 2)
            self.assertEqual(len(chain.repair.extra_calls), 1)
            self.assertEqual(
                chain.repair.extra_calls[0]["purpose"],
                "rescue-regenerate-hook")

    def test_rescale_speaker_renders_smaller(self):
        from spatial_director.layout_law import to_canvas_alpha
        with tempfile.TemporaryDirectory() as td:
            artifacts = make_artifacts(Path(td))
            src = []
            for _ in artifacts.src_alphas:
                a = np.zeros((36, 64))
                a[6:36, 10:54] = 1.0
                src.append(a)
            artifacts.src_alphas = src
            artifacts.make_canvas_alphas = (
                lambda alphas: [to_canvas_alpha(a) for a in alphas])
            artifacts.canvas_alphas = artifacts.make_canvas_alphas(src)
            spark = FakeSpark([
                '{"repair": "RESCALE_SPEAKER", "reason": "head clipped"}'])
            renderer = FakeRenderer()
            verify = FakeVerify([make_receipt((), draft=2)])
            chain = rescue(make_receipt(("F4",)), artifacts,
                           spark=spark, verify_fn=verify,
                           renderer=renderer, ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            shrunk = verify.executions[0].canvas_alphas
            area = float((shrunk[0] > 0.5).sum())
            full = float((artifacts.canvas_alphas[0] > 0.5).sum())
            self.assertLess(area, full)


class ChainShapeTests(unittest.TestCase):
    def test_chained_receipt_shape_and_no_sam(self):
        with tempfile.TemporaryDirectory() as td:
            wav = write_wav(Path(td) / "window.wav",
                            [(1.0, 0.4), (0.8, 0.0), (1.2, 0.4)])
            artifacts = make_artifacts(Path(td), window_clip=wav)
            spark = FakeSpark([
                '{"repair": "TRIM_SILENCE", "reason": "gap"}'])
            verify = FakeVerify([make_receipt((), draft=2)])
            chain = rescue(make_receipt(("F5",)), artifacts,
                           spark=spark, verify_fn=verify,
                           renderer=FakeRenderer(), ts="2026-10-29")
            data = json.loads(chain.to_json())
            self.assertEqual(data["chain_version"], 1)
            self.assertEqual(data["final_verdict"], "PASSED_AFTER_REPAIR")
            self.assertEqual(len(data["drafts"]), 2)
            for d in data["drafts"]:
                self.assertEqual(d["receipt_version"], 1)
                self.assertEqual(len(d["checks"]), 7)
            rep = data["repair"]
            self.assertEqual(rep["menu_id"], "TRIM_SILENCE")
            self.assertEqual(rep["target_check"], "F5")
            self.assertEqual(rep["spark_reason"], "gap")
            self.assertEqual(rep["spark_call"]["cost_usd"], 0.01)
            self.assertTrue(rep["spark_call"]["request_id"])
            self.assertIn("trimmed", rep["detail"])

    def test_module_has_no_sam_recourse(self):
        import spatial_director.rescue_loop as rl
        source = Path(rl.__file__).read_text(encoding="utf-8")
        self.assertNotIn("segment_video", source)
        self.assertNotIn("SamClient", source)
        self.assertNotIn("sam_client", source)


def low_face_artifacts(work_dir: Path, n_frames: int = 12):
    """Artifacts whose speaker sits low (the Holmes F3 anatomy):
    source 64x36, mask rows 10-36 - the placed face centroid lands
    below the central band under the real canvas transform."""
    artifacts = make_artifacts(work_dir, n_frames=n_frames)
    src = []
    for _ in range(n_frames):
        a = np.zeros((36, 64), dtype=np.float64)
        a[10:36, 20:44] = 1.0
        src.append(a)
    artifacts.src_alphas = src
    artifacts.frames = [np.zeros((36, 64, 3), dtype=np.uint8)
                        for _ in range(n_frames)]
    artifacts.canvas_alphas = [to_canvas_alpha(a) for a in src]
    artifacts.make_canvas_alphas = (
        lambda alphas: [to_canvas_alpha(a) for a in alphas])
    return artifacts


def f3_receipt_for(artifacts, centroid_y=None):
    if centroid_y is None:
        cents = [frame_geometry(a).face_centroid
                 for a in artifacts.canvas_alphas]
        cx = float(np.median([c[0] for c in cents]))
        centroid_y = float(np.median([c[1] for c in cents]))
    else:
        cx = 531.6
    measured = {"F3": {
        "violations": {
            "hook_px_outside_central_band": {
                "left": 0, "top": 0, "right": 0, "bottom": 0},
            "caption_px_outside_lower_zone": {
                "left": 0, "right": 0, "top": 0, "bottom": 0},
            "face_centroid_median": [cx, centroid_y]},
        "face_centroid_in_central_band": False}}
    return make_receipt(("F3",), measured=measured)


class ReframeSpeakerRepairTests(unittest.TestCase):
    def test_menu_targets_f3(self):
        self.assertEqual(REPAIR_MENU[REFRAME_SPEAKER].target_checks,
                         ("F3",))

    def test_reframe_moves_centroid_into_band(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = low_face_artifacts(Path(td))
            receipt = f3_receipt_for(artifacts)
            cy = receipt.checks[2].measured["violations"][
                "face_centroid_median"][1]
            self.assertGreater(cy, 1440)  # the failure is real
            spark = FakeSpark([
                '{"repair": "REFRAME_SPEAKER", "reason": '
                '"centroid below the band"}'])
            verify = FakeVerify([make_receipt((), draft=2)])
            renderer = FakeRenderer()
            chain = rescue(receipt, artifacts, spark=spark,
                           verify_fn=verify, renderer=renderer,
                           ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertEqual(chain.repair.menu_id, REFRAME_SPEAKER)
            self.assertEqual(chain.repair.target_check, "F3")
            self.assertEqual(len(renderer.calls), 1)
            execution = verify.executions[0]
            cents = [frame_geometry(a).face_centroid
                     for a in execution.canvas_alphas]
            new_cy = float(np.median([c[1] for c in cents]))
            self.assertGreaterEqual(new_cy, 288)
            self.assertLessEqual(new_cy, 1440)
            self.assertIn("reframed", chain.repair.detail)

    def test_reframe_refused_when_centroid_already_in_band(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = low_face_artifacts(Path(td))
            receipt = f3_receipt_for(artifacts, centroid_y=1200.0)
            spark = FakeSpark([
                '{"repair": "REFRAME_SPEAKER", "reason": "try it"}'])
            verify = FakeVerify([])
            chain = rescue(receipt, artifacts, spark=spark,
                           verify_fn=verify, renderer=FakeRenderer(),
                           ts="2026-10-29")
            self.assertEqual(chain.final_verdict, "FAIL")
            self.assertEqual(chain.repair.status, "not_applicable")
            self.assertEqual(verify.executions, [])


# ---------------------------------------------------------------- graphics


def _spec(start, end, headline="CARD"):
    return GraphicSpec(
        kind="card", start_s=start, end_s=end, box=(0, 0, 10, 10),
        layer_rgba=np.zeros((1920, 1080, 4), dtype=np.uint8),
        headline=headline)


def f8_receipt(measured):
    receipt = make_receipt(())
    receipt.checks.append(CheckResult(
        check_id="F8", name="context graphics integrity", passed=False,
        measured=measured, threshold={}))
    receipt.verdict = "FAIL"
    return receipt


class DropGraphicsTests(unittest.TestCase):
    def _run(self, td, graphics, measured):
        artifacts = make_artifacts(Path(td))
        artifacts.graphics = graphics
        spark = FakeSpark([
            '{"repair": "DROP_GRAPHICS", "reason": "f8 named it"}'])
        renderer = GraphicsRenderer()
        verify = FakeVerify([make_receipt((), draft=2)])
        chain = rescue(f8_receipt(measured), artifacts, spark=spark,
                       verify_fn=verify, renderer=renderer,
                       ts="2026-10-30")
        return chain, renderer, verify

    def test_drops_exactly_the_offending_graphic(self):
        with tempfile.TemporaryDirectory() as td:
            specs = [_spec(3.5, 5.5, "A"), _spec(9.5, 11.5, "B"),
                     _spec(15.5, 17.5, "C")]
            chain, renderer, verify = self._run(
                td, specs, {"offending_indices": [1],
                            "spacing_violations": []})
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertEqual(chain.repair.menu_id, "DROP_GRAPHICS")
            kept = verify.executions[0].graphics
            self.assertEqual([g.headline for g in kept], ["A", "C"])
            self.assertEqual(len(renderer.graphics_seen), 2)

    def test_spacing_violation_drops_the_later_graphic(self):
        with tempfile.TemporaryDirectory() as td:
            specs = [_spec(3.5, 5.5, "A"), _spec(7.0, 9.0, "B")]
            chain, _r, verify = self._run(
                td, specs,
                {"offending_indices": [],
                 "spacing_violations": [{"earlier_index": 0,
                                         "later_index": 1,
                                         "gap_s": 1.5}]})
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertEqual(
                [g.headline for g in verify.executions[0].graphics],
                ["A"])

    def test_unnamed_offenders_drop_everything(self):
        with tempfile.TemporaryDirectory() as td:
            specs = [_spec(3.5, 5.5, "A")]
            chain, _r, verify = self._run(td, specs, {})
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            self.assertEqual(verify.executions[0].graphics, [])

    def test_trim_silence_retimes_and_drops_collapsed_graphics(self):
        with tempfile.TemporaryDirectory() as td:
            wav = write_wav(Path(td) / "window.wav",
                            [(1.0, 0.4), (0.8, 0.0), (1.2, 0.4)])
            artifacts = make_artifacts(Path(td), window_clip=wav)
            artifacts.graphics = [
                _spec(0.2, 0.9, "BEFORE"),     # untouched by the cut
                _spec(1.2, 1.6, "IN GAP"),     # inside the removal
                _spec(1.9, 2.6, "AFTER"),      # shifts 0.5s earlier
            ]
            spark = FakeSpark([
                '{"repair": "TRIM_SILENCE", "reason": "gap"}'])
            verify = FakeVerify([make_receipt((), draft=2)])
            chain = rescue(
                make_receipt(("F5",), {"F5": {"max_silence_gap_s": 0.68}}),
                artifacts, spark=spark, verify_fn=verify,
                renderer=GraphicsRenderer(), ts="2026-10-30")
            self.assertEqual(chain.final_verdict, "PASSED_AFTER_REPAIR")
            kept = verify.executions[0].graphics
            self.assertEqual([g.headline for g in kept],
                             ["BEFORE", "AFTER"])
            after = kept[1]
            self.assertAlmostEqual(after.start_s, 1.4)
            self.assertAlmostEqual(after.end_s, 2.1)


if __name__ == "__main__":
    unittest.main()
