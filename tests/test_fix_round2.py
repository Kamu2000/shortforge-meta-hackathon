"""Fix round 2 tests (2026-11-02): Stage zoom (A), minus-speaker
backdrop (B), Spotlight refit (C), cue-sanity guard (D), hook
style (E), caption chunking + three-state sweep (F).

Offline throughout: synthetic geometry, banked caption fixtures,
and a skip-guarded MMS engagement proof over the same fixtures.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import (  # noqa: E402
    CaptionCue, cue_min_plausible_s, is_degenerate_cue,
    repair_cue_spans)
from spatial_director.compositor import (  # noqa: E402
    BACKDROP_SOURCE_MINUS_SPEAKER, composite_frame, hook_entry_layer,
    speaker_removed_backdrop, treated_background_layer)
from spatial_director.graphics_layer import (  # noqa: E402
    GraphicContent, GraphicSpan, GraphicSpec, plan_layout,
    snap_span_to_anchor)
from spatial_director.layout_law import (  # noqa: E402
    CAPTION_EMPHASIS_RGBA, CAPTION_TEXT_RGBA, CAPTION_UPCOMING_RGBA,
    caption_word_colors, frame_geometry, hook_accent_index,
    plan_stage_zoom, apply_stage_zoom, render_hook_layer,
    hook_word_layers, to_canvas_alpha)
from spatial_director.word_timing import WordTiming  # noqa: E402

CANVAS = (1080, 1920)
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def block(shape, rows, cols, value=1.0):
    a = np.zeros(shape, dtype=np.float64)
    a[rows[0]:rows[1], cols[0]:cols[1]] = value
    return a


# ---------------------------------------------------------------- A


class StageZoomTests(unittest.TestCase):
    def holmes_alphas(self, n=6):
        # 640x360 desk framing like the banked Holmes windows:
        # a narrower head (cols 200..400) over wider shoulders -
        # frame_geometry reads the face from the head band, and a
        # full-width rectangle would cap the zoom unrealistically.
        out = []
        for _ in range(n):
            a = block((360, 640), (160, 346), (150, 450))
            a[73:160, 200:400] = 1.0
            out.append(a)
        return out

    def test_zoom_enlarges_into_the_design_bands(self):
        alphas = self.holmes_alphas()
        plan = plan_stage_zoom(alphas, CANVAS)
        self.assertTrue(plan.feasible, plan.note)
        self.assertFalse(plan.identity)
        self.assertGreater(plan.scale, 1.5)
        self.assertGreaterEqual(plan.height_frac, 0.55)
        self.assertLessEqual(plan.height_frac, 0.70)
        self.assertTrue(880 <= plan.placed_centroid[1] <= 980,
                        plan.placed_centroid)
        self.assertGreaterEqual(plan.placed_head_top_min, 150)
        # Applied: the measured placed geometry lands in the bands.
        _f, za = apply_stage_zoom(
            [np.zeros((360, 640, 3), dtype=np.uint8)
             for _ in alphas], alphas, plan)
        cents, heights = [], []
        for a in za:
            g = frame_geometry(to_canvas_alpha(a, CANVAS))
            cents.append(g.face_centroid)
            heights.append((g.mask_bbox[3] - g.mask_bbox[1]) / 1920)
        cy = float(np.median([c[1] for c in cents]))
        self.assertTrue(860 <= cy <= 1000, cy)
        self.assertGreater(float(np.median(heights)), 0.50)

    def test_already_large_speaker_is_identity(self):
        # Portrait source whose fit already reaches the band.
        alphas = [block((1280, 720), (100, 1100), (200, 520))
                  for _ in range(4)]
        plan = plan_stage_zoom(alphas, CANVAS)
        self.assertTrue(plan.feasible)
        self.assertTrue(plan.identity, plan.note)

    def test_no_speaker_is_honest_identity(self):
        plan = plan_stage_zoom([np.zeros((360, 640)) for _ in range(3)],
                               CANVAS)
        self.assertFalse(plan.feasible)
        self.assertTrue(plan.identity)


# ---------------------------------------------------------------- B


class MinusSpeakerBackdropTests(unittest.TestCase):
    def setUp(self):
        # A frame with a hard person edge INSIDE the mask (the
        # ghost source) and sharp texture OUTSIDE it.
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (90, 120, 160)
        frame[::8, :] = (200, 210, 220)   # sharp horizontal lines
        frame[:, ::8] = (200, 210, 220)
        frame[60:330, 220:420] = (20, 20, 20)
        frame[60:330, 320:420] = (240, 240, 240)  # the hard edge
        self.frame = frame
        self.alpha = block((360, 640), (60, 330), (220, 420))

    def test_fill_kills_the_person_edge_outside_stays_sharp(self):
        bg = speaker_removed_backdrop(self.frame, self.alpha, CANVAS)
        self.assertEqual(bg.shape, (1920, 1080, 3))
        # The backdrop is a cover-fit drift window, so measure the
        # SOURCE-space treatment via a same-size canvas instead.
        small = speaker_removed_backdrop(self.frame, self.alpha,
                                         (640, 360))
        # Inside the filled region the hard edge is gone: local
        # horizontal gradient where the edge was is a small
        # fraction of the original step (230 levels).
        region = small[150:250, 330:410]
        grad = np.abs(np.diff(region[:, :, 0], axis=1)).max()
        self.assertLess(grad, 0.35 * 230)
        # Outside the mask the sharp lines survive: the corner's
        # local contrast is a multiple of the treated (blurred)
        # backdrop's - sharp set vs smear, measured relatively.
        corner = small[0:60, 0:200]
        tb = treated_background_layer(self.frame, (640, 360))
        self.assertGreater(corner.std(), 3.0 * tb[0:60, 0:200].std())
        self.assertGreater(corner.std(), 10.0)

    def test_maskless_frame_falls_back_to_treated(self):
        # The sms contract (fix round 2): a frame with no speaker
        # mask falls back to the treated blur. Since fix round 3
        # the canvas DEFAULT is studio_procedural, so the mode is
        # selected explicitly here.
        empty = np.zeros((360, 640))
        out = composite_frame(self.frame, empty, None, CANVAS,
                              backdrop=BACKDROP_SOURCE_MINUS_SPEAKER)
        treated = treated_background_layer(self.frame, CANVAS)
        fg_free = (10, 10)  # far from any speaker (there is none)
        self.assertTrue(np.allclose(
            out[fg_free], treated[fg_free], atol=1.0))

    def test_drift_moves_the_window(self):
        a = speaker_removed_backdrop(self.frame, self.alpha, CANVAS,
                                     progress=0.0)
        b = speaker_removed_backdrop(self.frame, self.alpha, CANVAS,
                                     progress=1.0)
        self.assertFalse(np.allclose(a, b))


# ---------------------------------------------------------------- C


def person_alphas(n=120, shape=(583, 640)):
    a = np.zeros(shape, dtype=np.float32)
    a[77:, 254:476] = 1.0
    return [a.copy() for _ in range(n)]


def card_content(w=648, h=394):
    img = Image.new("RGBA", (w, h), (30, 30, 40, 255))

    def resizer(nw):
        nh = max(1, round(h * nw / w))
        return GraphicContent(
            size=(nw, nh),
            final_image=img.resize((nw, nh)),
            resizer=resizer)
    return GraphicContent(size=(w, h), final_image=img,
                          resizer=resizer)


class SpotlightRefitTests(unittest.TestCase):
    def spec(self, box, content=None, layout="spotlight"):
        layer = np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
        x0, y0, x1, y1 = box
        layer[y0:y1, x0:x1, 3] = 255
        return GraphicSpec(
            kind="card", start_s=0.3, end_s=3.3, box=box,
            layer_rgba=layer, headline="TEST", source_quote="q",
            layout=layout, _content=content)

    def test_qh3_case_now_fits_below_the_pip(self):
        # The live refusal: 648px centred card colliding with both
        # PiP corners. With content retained, the refit moves the
        # card below the PiP and the span goes Spotlight.
        s = self.spec((216, 611, 864, 1005), content=card_content())
        built = plan_layout([s], person_alphas(), fps=30.0,
                            canvas=CANVAS)
        self.assertIsNotNone(built)
        track, decls, _rects = built
        self.assertIsNotNone(track)
        self.assertEqual(decls[0]["layout"], "spotlight")
        self.assertTrue(decls[0]["feasible"])
        self.assertIn("spotlight_refit", decls[0])
        pip = decls[0]["speaker_box"]
        self.assertGreaterEqual(s.box[1], pip[3] + 40)
        self.assertEqual(s.layout, "spotlight")

    def test_impossible_card_still_downgrades_honestly(self):
        # A card too tall for the below-PiP band even at its
        # smallest ladder size, with no resizer: no legal fit.
        content = GraphicContent(
            size=(648, 1250),
            final_image=Image.new("RGBA", (648, 1250),
                                  (30, 30, 40, 255)))
        s = self.spec((216, 300, 864, 1550), content=content)
        built = plan_layout([s], person_alphas(), fps=30.0,
                            canvas=CANVAS)
        _track, decls, _rects = built
        self.assertEqual(decls[0]["downgraded_to"], "stage")
        self.assertNotIn("spotlight_refit", decls[0])
        self.assertEqual(s.layout, "stage")


# ---------------------------------------------------------------- D


def load_fixture(name):
    fx = json.loads((FIXTURES / f"cue_sanity_{name}.json").read_text())
    cues = [CaptionCue(text=c["text"], start_s=c["start_s"],
                       end_s=c["end_s"]) for c in fx["cues"]]
    return cues, fx["duration_s"]


def word_seq(cues):
    return [w for c in cues for w in c.text.split()]


class CueSanityTests(unittest.TestCase):
    def test_detection_both_ways(self):
        self.assertTrue(is_degenerate_cue(
            CaptionCue("and it was.", 13.78, 13.84)))
        self.assertFalse(is_degenerate_cue(
            CaptionCue("and it was.", 13.0, 13.9)))
        self.assertGreater(
            cue_min_plausible_s("and it was."), 0.2)

    def test_expand_into_gaps(self):
        cues = [CaptionCue("hello world this is", 0.0, 2.0),
                CaptionCue("and it was.", 2.1, 2.16),
                CaptionCue("more speech follows here", 4.0, 6.0)]
        out, records = repair_cue_spans(cues, duration_s=8.0)
        self.assertEqual(len(out), 3)
        self.assertEqual(records[0]["action"], "expanded")
        self.assertFalse(is_degenerate_cue(out[1]))
        self.assertGreaterEqual(out[1].start_s, out[0].end_s)
        self.assertLessEqual(out[1].end_s, out[2].start_s)
        self.assertEqual(word_seq(out), word_seq(cues))

    def test_merge_when_no_gap(self):
        cues = [CaptionCue("hello world this is", 0.0, 2.09),
                CaptionCue("and it was.", 2.1, 2.16),
                CaptionCue("more speech follows", 2.17, 4.0)]
        out, records = repair_cue_spans(cues, duration_s=6.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(records[0]["action"], "merged_into_previous")
        self.assertEqual(word_seq(out), word_seq(cues))
        self.assertFalse(any(is_degenerate_cue(c) for c in out))

    def test_unrepairable_is_recorded_not_hidden(self):
        # No neighbours and no gap anywhere: nothing can absorb
        # or expand the cue, so it stays - on the record.
        cues = [CaptionCue("trapped words here now", 0.0, 0.05)]
        out, records = repair_cue_spans(cues, duration_s=0.05)
        self.assertEqual(records[0]["action"], "unrepairable")
        self.assertTrue(is_degenerate_cue(out[0]))

    def test_qm4_fixture_repairs_both_degenerate_cues(self):
        cues, dur = load_fixture("qm4")
        out, records = repair_cue_spans(cues, duration_s=dur)
        # Two cues are degenerate on the measured bound: cue 14
        # (3 words in 0.06s - the headline case) expands into its
        # gaps to exactly its plausible minimum, and cue 0 (5
        # words / 30 chars in 0.68s vs the 0.70s CTC floor) merges
        # forward. Both repairs are bound-driven, not tuned.
        self.assertEqual(len(records), 2)
        by_index = {r["cue_index"]: r for r in records}
        self.assertEqual(by_index[14]["action"], "expanded")
        self.assertEqual(by_index[14]["before"], [13.78, 13.84])
        self.assertEqual(by_index[0]["action"], "merged_next")
        self.assertEqual(len(out), 14)
        self.assertFalse(any(is_degenerate_cue(c) for c in out))
        self.assertEqual(word_seq(out), word_seq(cues))

    def test_qh3_fixture_all_plausible_words_preserved(self):
        cues, dur = load_fixture("qh3")
        out, records = repair_cue_spans(cues, duration_s=dur)
        self.assertTrue(records)
        self.assertFalse(any(is_degenerate_cue(c) for c in out),
                         [(c.text, c.start_s, c.end_s) for c in out
                          if is_degenerate_cue(c)])
        self.assertEqual(word_seq(out), word_seq(cues))
        # The Parsons run merged into one physically real cue.
        merged = [c for c in out if "Parsons" in c.text]
        self.assertTrue(any(c.end_s - c.start_s > 4.0 for c in merged))

    def test_span_reanchors_to_repaired_cues(self):
        cues, dur = load_fixture("qh3")
        repaired, _ = repair_cue_spans(cues, duration_s=dur)
        span = GraphicSpan(start_s=0.2, end_s=0.3, kind="card",
                           source_quote="Micah Parsons, no matter what")
        raw, _r = snap_span_to_anchor(span, cues, duration_s=dur)
        fixed, _f = snap_span_to_anchor(span, repaired,
                                        duration_s=dur)
        # Against the rotten cues the passage is so short the
        # span law rejects it outright (or pins it under 1s);
        # against the repaired cues it covers the spoken passage.
        self.assertTrue(raw is None or raw.end_s < 1.0)
        self.assertIsNotNone(fixed)
        self.assertGreater(fixed.end_s, 5.0)  # the spoken passage


# ---------------------------------------------------------------- E


class HookStyleTests(unittest.TestCase):
    def test_accent_prefers_the_number_word(self):
        self.assertEqual(
            hook_accent_index("Cowboys only $1M apart this year"), 2)

    def test_accent_falls_back_to_longest_word(self):
        self.assertEqual(hook_accent_index("the quick brown fox"), 1)

    def test_two_tone_layer_paints_one_word_pink(self):
        layer = render_hook_layer(
            "HELLO BRAVE WORLD", CANVAS, y_center_frac=0.3,
            accent_index=1)
        rgb = layer[:, :, :3].astype(int)
        alpha = layer[:, :, 3] > 0
        pink = alpha & (rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 110) \
            & (rgb[:, :, 2] > 90)
        self.assertGreater(pink.sum(), 100)
        ys, xs = np.nonzero(pink)
        # The pink ink is the MIDDLE word: its centre sits in the
        # middle third of the whole hook's ink bbox.
        all_ys, all_xs = np.nonzero(alpha)
        ink_cx = (all_xs.min() + all_xs.max()) / 2.0
        pink_cx = (xs.min() + xs.max()) / 2.0
        third = (all_xs.max() - all_xs.min()) / 3.0
        self.assertLess(abs(pink_cx - ink_cx), third)

    def test_entry_starts_with_word_zero_only(self):
        text = "ONE TWO THREE"
        layers = hook_word_layers(text, CANVAS, y_center_frac=0.3,
                                  accent_index=None)
        self.assertEqual(len(layers), 3)
        at0 = hook_entry_layer(layers, 0.0)
        self.assertEqual(int((at0[:, :, 3] > 0).sum()),
                         int((layers[0][:, :, 3] > 0).sum()))
        mid = hook_entry_layer(layers, 0.3)
        self.assertGreater(int((mid[:, :, 3] > 0).sum()),
                           int((at0[:, :, 3] > 0).sum()))
        # Settled entry == the resting styled layer's ink (the
        # words come to rest exactly where the layer declares).
        resting = render_hook_layer(text, CANVAS, y_center_frac=0.3)
        rest_ink = int((resting[:, :, 3] > 0).sum())
        late = hook_entry_layer(layers, 0.64)
        late_ink = int((late[:, :, 3] > 0).sum())
        self.assertGreaterEqual(late_ink, int(0.92 * rest_ink))
        self.assertLessEqual(late_ink, int(1.08 * rest_ink))


# ---------------------------------------------------------------- F


class CaptionStateTests(unittest.TestCase):
    TEXT = "the $5 million run"  # emphasis: $5 (digit), million

    def test_three_states_and_emphasis_collision_rule(self):
        # Active word 0: upcoming emphasis words are DIM, not pink.
        colors = caption_word_colors(self.TEXT, 0)
        self.assertEqual(colors[0], CAPTION_EMPHASIS_RGBA)  # active
        self.assertEqual(colors[1], CAPTION_UPCOMING_RGBA)
        self.assertEqual(colors[2], CAPTION_UPCOMING_RGBA)
        # Active word 2: spoken emphasis ($5) is pink, spoken
        # plain word is white, active is pink.
        colors = caption_word_colors(self.TEXT, 2)
        self.assertEqual(colors[0], CAPTION_TEXT_RGBA)
        self.assertEqual(colors[1], CAPTION_EMPHASIS_RGBA)
        self.assertEqual(colors[2], CAPTION_EMPHASIS_RGBA)
        self.assertEqual(colors[3], CAPTION_UPCOMING_RGBA)

    def test_pause_holds_states_via_spoken_count(self):
        colors = caption_word_colors(self.TEXT, None, spoken_count=1)
        self.assertEqual(colors[0], CAPTION_TEXT_RGBA)
        self.assertEqual(colors[1], CAPTION_UPCOMING_RGBA)

    def test_no_sweep_is_the_historic_emphasis_only(self):
        colors = caption_word_colors(self.TEXT, None)
        self.assertEqual(colors, {1: CAPTION_EMPHASIS_RGBA,
                                  2: CAPTION_EMPHASIS_RGBA})

    def test_chunked_render_shows_chunks_in_their_windows(self):
        from spatial_director.rescue_loop import \
            render_composite_with_captions
        import tempfile
        canvas = (240, 320)
        frames = [np.full((32, 32, 3), 120, dtype=np.uint8)
                  for _ in range(20)]
        alphas = [np.zeros((32, 32)) for _ in range(20)]
        words = [f"w{i}" for i in range(8)]
        cue = CaptionCue(" ".join(words), 0.2, 1.8)
        timings = [WordTiming(w, 0.2 + 0.2 * i, 0.4 + 0.2 * i, "dsp")
                     for i, w in enumerate(words)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "chunked.mp4"
            render_composite_with_captions(
                frames, alphas, hook_rgba=None, hook_window=None,
                cues=[cue], caption_rect=(10, 200, 230, 300),
                audio_from=None, out_path=out, fps=10.0,
                canvas=canvas, caption_word_timings=[timings])
            from spatial_director.compositor import read_video_frames
            rendered = read_video_frames(out)
        self.assertEqual(len(rendered), 20)
        region = (slice(200, 300), slice(10, 230))

        def ink(frame):
            return float(np.abs(
                frame[region].astype(int) - 120).sum())
        self.assertGreater(ink(rendered[5]), 1000.0)   # chunk 1
        self.assertGreater(ink(rendered[15]), 1000.0)  # chunk 2
        # Different chunk text in the two windows.
        diff = np.abs(rendered[5][region].astype(int)
                      - rendered[15][region].astype(int)).sum()
        self.assertGreater(diff, 1000.0)


# ---------------------------------------------------------------- D proof


@unittest.skipUnless(
    (Path.home() / "workspace/meta-hackathon/mms-venv/bin/python").exists()
    and (Path.home() / "workspace/meta-hackathon/pipeline-run-ext-qh3/short-1/window.mp4").exists(),
    "MMS venv or fixture windows unavailable")
class CueSanityMmsProofTests(unittest.TestCase):
    def test_both_fixture_windows_engage_mms_after_repair(self):
        from spatial_director.caption_align import load_energy
        from spatial_director.pipeline import _caption_word_timings
        base = Path.home() / "workspace/meta-hackathon"
        for name, run in (("qh3", "pipeline-run-ext-qh3"),
                          ("qm4", "pipeline-run-ext-qm4")):
            cues, dur = load_fixture(name)
            repaired, _records = repair_cue_spans(cues, duration_s=dur)
            window = base / run / "short-1/window.mp4"
            # Fix round 5: _caption_word_timings returns a
            # CaptionTimingResult with per-cue provenance. The
            # round-2 premise - a repaired Short comes out pure
            # "mms" - was itself the defect's signature: these
            # banked layouts were built by the old _local_turns
            # (turn span clipped, full text kept), so some cues
            # carry more words than their audio holds, and the
            # pace gate now refuses those cues honestly -
            # measured 2026-11-02: qh3 keeps 7 mms + 3 dsp
            # (rejected cues at 4.6-5.8 wps); qm4 keeps only
            # 3 mms + 11 dsp (its layout is the most compressed
            # of the banked runs - one cue aligns at 13.3 wps).
            # Both fixtures are PRE-FIX artifacts; the fixed
            # caption path's clean engagement is proven on the
            # QM6 reconstruction in the round-5 acceptance.
            # What this proof asserts is the HONESTY contract:
            # MMS genuinely engages (at least one cue aligns),
            # every cue carries a provenance, and the summary
            # label is exactly the mix the per-cue record
            # implies - never purer.
            result = _caption_word_timings(
                repaired, load_energy(window), window, "mms",
                lambda m: None)
            self.assertEqual(len(result.cue_sources), len(repaired))
            self.assertIn("mms", result.cue_sources,
                          f"{name}: MMS never engaged")
            n_mms = result.cue_sources.count("mms")
            expected = ("mms" if n_mms == len(repaired) else
                        "dsp" if n_mms == 0 else "mms+dsp")
            self.assertEqual(result.source, expected,
                             f"{name}: label {result.source} != "
                             f"per-cue record {result.cue_sources}")


if __name__ == "__main__":
    unittest.main()
