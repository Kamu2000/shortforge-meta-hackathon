"""Tests for spatial_director.geometry_bundle — pure logic,
synthetic fixtures only. The end-to-end behaviour (real receipt +
real evidence) is covered by manual reverify runs reported in the
BUILDLOG, not by this suite.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spatial_director import geometry_bundle as gb  # noqa: E402
from spatial_director.layout_states import (  # noqa: E402
    LayoutDecl,
    LayoutPlan,
    SpeakerPlacement,
    build_track,
)
from spatial_director.verifier import HookSpec  # noqa: E402

STAGE_GEOMETRY = {
    "feasible": True, "identity": False, "scale": 2.735,
    "crop": [208, 66, 234, 294], "pad_bottom_px": 3,
    "centroid_target": [540.0, 930.0],
    "placed_centroid": [540.2, 930.8], "head_top_min": 660,
    "mask_height_frac": 0.618, "note": "stage zoom x2.74",
    "reframe_pad_bottom_px": 0, "effect_pad_bottom_px": 12,
}

INPUTS_SHA = {"video": "a" * 64, "sam_response": "b" * 64,
              "captions": "c" * 64}


def _draft_bundle(**over):
    kw = dict(
        stage_geometry=STAGE_GEOMETRY,
        alpha_recipe=gb.alpha_recipe_dict(
            src_frame_size=(360, 640),
            stage_geometry=STAGE_GEOMETRY,
            speaker_forward=None),
        hook=gb.hook_to_dict(HookSpec(
            "WHY SPACEWALK SMELLS LIKE HAMBURGER?", (0.0, 6.0), 0.4,
            layout_rendered=True, accent_index=3, text_size=144)),
        caption_rect=(60, 1460, 980, 1600),
        layout_spans=None,
        hook_slide_px_s=0.0,
        inputs_sha256=INPUTS_SHA)
    kw.update(over)
    return gb.draft_bundle_dict(**kw)


class TestHookRoundTrip(unittest.TestCase):
    def test_hook_dict_round_trip(self):
        hook = HookSpec("HELLO WORLD?", (0.0, 5.5), 0.42,
                        layout_rendered=True, slide=(-240.0, 1.2),
                        accent_index=1, text_size=122)
        back = gb.hook_from_dict(gb.hook_to_dict(hook))
        self.assertEqual(back.text, hook.text)
        self.assertEqual(back.window, hook.window)
        self.assertEqual(back.y_center_frac, hook.y_center_frac)
        self.assertEqual(back.layout_rendered, hook.layout_rendered)
        self.assertEqual(back.slide, hook.slide)
        self.assertEqual(back.accent_index, hook.accent_index)
        self.assertEqual(back.text_size, hook.text_size)
        self.assertEqual(back.variant, hook.variant)


class TestRecipe(unittest.TestCase):
    def test_recipe_copies_stage_geometry(self):
        recipe = gb.alpha_recipe_dict(
            src_frame_size=(360, 640),
            stage_geometry=STAGE_GEOMETRY, speaker_forward=None)
        self.assertEqual(recipe["src_frame_size"], [360, 640])
        self.assertEqual(recipe["stage_zoom"], {
            "applied": True, "crop": [208, 66, 234, 294],
            "pad_bottom_px": 3})
        self.assertEqual(recipe["effect_pad_bottom_px"], 12)
        self.assertIsNone(recipe["speaker_forward"])

    def test_recipe_identity_zoom_not_applied(self):
        recipe = gb.alpha_recipe_dict(
            src_frame_size=(360, 640),
            stage_geometry={"feasible": True, "identity": True,
                            "crop": None, "pad_bottom_px": 0,
                            "effect_pad_bottom_px": 0},
            speaker_forward=None)
        self.assertFalse(recipe["stage_zoom"]["applied"])
        self.assertIsNone(gb.stage_zoom_from_recipe(recipe))

    def test_stage_zoom_replay_fields(self):
        recipe = gb.alpha_recipe_dict(
            src_frame_size=(360, 640),
            stage_geometry=STAGE_GEOMETRY, speaker_forward=None)
        zoom = gb.stage_zoom_from_recipe(recipe)
        self.assertIsNotNone(zoom)
        self.assertEqual(zoom.crop, (208, 66, 234, 294))
        self.assertEqual(zoom.pad_bottom_px, 3)
        self.assertTrue(zoom.feasible)
        self.assertFalse(zoom.identity)

    def test_speaker_forward_normalisation(self):
        decls = [{"span": {"start_s": 9.866, "end_s": 11.929},
                  "scale": 1.18, "centroid_target": [540.0, 864.0],
                  "placed_centroid": [540.2, 865.6],
                  "placed_head_top_min": 474, "note": "emphasis"}]
        out = gb.normalize_speaker_forward(decls)
        self.assertEqual(out[0]["span"], [9.866, 11.929])
        self.assertEqual(out[0]["scale"], 1.18)
        self.assertEqual(out[0]["placed_centroid"], [540.2, 865.6])
        self.assertEqual(out[0]["placed_head_top_min"], 474)
        # The planner's note is preserved: receipt-era
        # declarations record the applied crop + pad only
        # inside it, and the replayer recovers them from there.
        self.assertEqual(out[0]["note"], "emphasis")
        self.assertNotIn("crop", out[0])
        self.assertIsNone(gb.normalize_speaker_forward(None))
        self.assertIsNone(gb.normalize_speaker_forward([]))

    def test_speaker_forward_transform_passthrough(self):
        decls = [{"span": [1.0, 3.0], "scale": 1.18,
                  "crop": [10, 9, 227, 284], "pad_bottom_px": 40,
                  "placed_centroid": [540.2, 865.6],
                  "placed_head_top_min": 474}]
        out = gb.normalize_speaker_forward(decls)
        self.assertEqual(out[0]["crop"], [10, 9, 227, 284])
        self.assertEqual(out[0]["pad_bottom_px"], 40)


class TestBundleFileRoundTrip(unittest.TestCase):
    def test_write_read_round_trip_and_hash(self):
        bundle = _draft_bundle()
        with tempfile.TemporaryDirectory() as td:
            gb.write_bundle(td, {1: bundle})
            raw = json.loads(
                (Path(td) / gb.BUNDLE_FILENAME).read_text())
            self.assertEqual(raw["version"], gb.BUNDLE_VERSION)
            loaded = gb.read_bundle_file(td)
        self.assertEqual(set(loaded), {1})
        self.assertEqual(loaded[1], bundle)
        self.assertTrue(gb.check_inputs_hash(loaded[1], INPUTS_SHA))

    def test_hash_detects_wrong_receipt(self):
        bundle = _draft_bundle()
        self.assertFalse(gb.check_inputs_hash(
            bundle, {"video": "z" * 64}))

    def test_hash_detects_tampered_bundle(self):
        bundle = _draft_bundle()
        bundle["caption_rect"] = [0, 0, 10, 10]
        self.assertFalse(gb.check_inputs_hash(bundle, INPUTS_SHA))

    def test_missing_file_reads_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(gb.read_bundle_file(td), {})


class TestPlacementsFromDeclarations(unittest.TestCase):
    def _plan(self):
        placement = SpeakerPlacement(
            crop=(0.0, 0.0, 234.0, 297.0), scale=2.3704,
            ox=262.7, oy=756.0, alpha_scale=1.0)
        return LayoutPlan(
            state="spotlight", placement=placement, feasible=True,
            placed_centroid=(541.0, 948.7), head_top_min=815.0)

    def test_track_matches_direct_build(self):
        plan = self._plan()
        decl = plan.declaration(1.0, 2.0,
                                graphic_box=[135, 100, 945, 708])
        shape = (297, 234)
        direct = build_track(
            [LayoutDecl(1.0, 2.0, "spotlight")], [plan], shape,
            90, 30.0, (1080, 1920))
        rebuilt = gb.placements_from_declarations(
            [decl], shape, 90, 30.0, (1080, 1920))
        self.assertEqual(len(direct), len(rebuilt))
        for want, got in zip(direct, rebuilt):
            self.assertEqual(want is None, got is None)
            if want is None:
                continue
            # The declaration rounds (crop 2dp, scale 4dp, origin
            # 1dp), so the rebuilt track agrees to rounding.
            self.assertAlmostEqual(want.scale, got.scale, places=3)
            self.assertAlmostEqual(want.ox, got.ox, places=0)
            self.assertAlmostEqual(want.oy, got.oy, places=0)
            self.assertAlmostEqual(
                want.alpha_scale, got.alpha_scale, places=2)

    def test_no_feasible_spans_returns_none(self):
        decl = self._plan().declaration(1.0, 2.0)
        decl["feasible"] = False
        self.assertIsNone(gb.placements_from_declarations(
            [decl], (297, 234), 90, 30.0, (1080, 1920)))
        stage = dict(decl, layout="stage", feasible=True)
        self.assertIsNone(gb.placements_from_declarations(
            [stage], (297, 234), 90, 30.0, (1080, 1920)))
        self.assertIsNone(gb.placements_from_declarations(
            None, (297, 234), 90, 30.0, (1080, 1920)))

    def test_camera_span_declines(self):
        decl = self._plan().declaration(1.0, 2.0)
        decl["camera"] = {"type": "drift", "from": [0, 0],
                          "to": [10, 0]}
        with self.assertRaises(ValueError):
            gb.placements_from_declarations(
                [decl], (297, 234), 90, 30.0, (1080, 1920))


if __name__ == "__main__":
    unittest.main()
