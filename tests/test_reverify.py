"""Tests for tools/reverify.py — pure logic, synthetic fixtures only.

No workspace run directories, no API keys, no paid calls. The
end-to-end behaviour (real receipt + real artifact) is covered by
the proof-pack reverify cases and by manual runs reported in the
BUILDLOG, not by this suite.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from spatial_director.caption_align import CaptionCue, EnergyProfile  # noqa: E402
from spatial_director.verifier import VerifyInputs  # noqa: E402

import reverify as rv  # noqa: E402


def _draft(checks, **kw):
    d = {"draft": 1, "artifact": "/nonexistent/a.mp4",
         "verdict": "PASSED_FIRST_DRAFT", "failed_checks": [],
         "checks": checks, "meta_calls": [],
         "inputs_sha256": {}, "notes": []}
    d.update(kw)
    return d


def _check(cid, measured, passed=True):
    return {"id": cid, "name": cid, "pass": passed,
            "measured": measured, "threshold": {}, "note": ""}


def _inputs(**kw):
    base = dict(
        video_path="/nonexistent/a.mp4", fps=30.0, canvas_alphas=[],
        audio=EnergyProfile(times=np.array([0.0]),
                            db=np.array([-20.0])),
        hook=None, cues=[], caption_rect=None,
        sam_response_path=None, hook_slide_px_s=0.0,
        canvas=(1080, 1920), graphics=None, speech_audio=None,
        sfx=None, backdrops=None, speaker_forward=None)
    base.update(kw)
    return VerifyInputs(**base)


class TestHashing(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(b"abc")
            path = fh.name
        self.assertEqual(
            rv.sha256_file(path),
            hashlib.sha256(b"abc").hexdigest())

    def test_chain_hash_is_order_sensitive(self):
        a = rv.chain_hash(["x", "y"])
        b = rv.chain_hash(["y", "x"])
        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 64)


class TestChain(unittest.TestCase):
    def _receipt(self, drafts, repair=None, top=None):
        return {"chain_version": 1,
                "artifact": top if top is not None
                else (drafts[-1]["artifact"] if drafts else None),
                "final_verdict": drafts[-1]["verdict"] if drafts else None,
                "repair": repair or {"status": "not_needed"},
                "drafts": drafts}

    def test_clean_single_draft_chain(self):
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(b"video-bytes")
            path = Path(fh.name)
        h = hashlib.sha256(b"video-bytes").hexdigest()
        d = _draft([], inputs_sha256={"video": h})
        d["artifact"] = str(path)
        rep = self._receipt([d])
        out = rv.verify_chain(rep, {1: path, "top": path})
        self.assertTrue(out["ok"])
        self.assertEqual(out["findings"], [])

    def test_swapped_artifact_flagged_on_hash(self):
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(b"different-bytes")
            path = Path(fh.name)
        d = _draft([], inputs_sha256={
            "video": hashlib.sha256(b"original-bytes").hexdigest()})
        d["artifact"] = str(path)
        rep = self._receipt([d])
        out = rv.verify_chain(rep, {1: path, "top": path})
        self.assertFalse(out["ok"])
        self.assertTrue(any(f["status"] == "MISMATCH"
                            for f in out["findings"]))

    def test_missing_artifact_reported(self):
        d = _draft([], inputs_sha256={"video": "0" * 64})
        rep = self._receipt([d])
        out = rv.verify_chain(rep, {1: None, "top": None})
        self.assertTrue(any(f["status"] == "MISSING"
                            for f in out["findings"]))

    def test_repair_consistency(self):
        d1 = _draft([], verdict="FAIL", failed_checks=["F5"])
        d2 = _draft([], verdict="PASSED_AFTER_REPAIR")
        d1["draft"] = 1
        d2["draft"] = 2
        rep = self._receipt(
            [d1, d2],
            repair={"status": "executed", "menu_id": "TRIM_SILENCE",
                    "target_check": "F5"})
        rep["final_verdict"] = "PASSED_AFTER_REPAIR"
        out = rv.verify_chain(rep, {1: None, 2: None, "top": None})
        # Only the missing-artifact findings; no structure failures.
        self.assertFalse(any(f["kind"] == "structure"
                             and f["status"] == "FAIL"
                             for f in out["findings"]))

    def test_repair_target_not_failed_is_structural_fail(self):
        d1 = _draft([], verdict="FAIL", failed_checks=["F1"])
        d2 = _draft([], verdict="PASSED_AFTER_REPAIR")
        d1["draft"] = 1
        d2["draft"] = 2
        rep = self._receipt(
            [d1, d2],
            repair={"status": "executed", "menu_id": "TRIM_SILENCE",
                    "target_check": "F5"})
        rep["final_verdict"] = "PASSED_AFTER_REPAIR"
        out = rv.verify_chain(rep, {1: None, 2: None, "top": None})
        self.assertTrue(any("did not fail it" in f["detail"]
                            for f in out["findings"]))


class TestHookRecovery(unittest.TestCase):
    def test_f6_question_evidence(self):
        d = _draft([_check("F6", {"patterns_fired": [
            {"pattern": "question", "source": "hook",
             "evidence": "DID COWBOYS RISK IT?"}]})])
        self.assertEqual(rv.recover_hook_text(d), "DID COWBOYS RISK IT?")

    def test_f6_number_bracket_stripped(self):
        d = _draft([_check("F6", {"patterns_fired": [
            {"pattern": "number", "source": "hook",
             "evidence": "A MILLION APART? [million]"}]})])
        self.assertEqual(rv.recover_hook_text(d), "A MILLION APART?")

    def test_wonder_summary_fallback(self):
        d = _draft([_check("F5", {
            "wonder_summary": 'A man speaks. The hook is the '
            'on-screen text "HELLO WORLD?"'})])
        self.assertEqual(rv.recover_hook_text(d), "HELLO WORLD?")

    def test_no_evidence_is_none(self):
        self.assertIsNone(rv.recover_hook_text(_draft([])))


class TestTrimReplay(unittest.TestCase):
    def _profile(self):
        # 10s at 0.02 hop: speech 0-3s, gap 3-4.2s, speech 4.2-10s.
        times = np.arange(0.0, 10.0, 0.02)
        db = np.where(((times >= 3.0) & (times < 4.2)),
                      -90.0, -20.0)
        return EnergyProfile(times=times, db=db, hop_s=0.02)

    def test_removals_match_repair_rule(self):
        # Monkeypatch load_energy via a stub profile through the
        # module's own function is overkill; instead verify the
        # remap math directly on a known removal.
        removals = [(3.15, 4.05)]  # keep 0.30 of a 1.2s gap
        cues = [CaptionCue("early", 1.0, 2.0),
                CaptionCue("late", 5.0, 6.0)]
        out = rv.remap_cues(cues, removals)
        self.assertEqual(out[0].start_s, 1.0)
        # 5.0 - 0.9 removed = 4.1
        self.assertAlmostEqual(out[1].start_s, 4.1, places=2)
        self.assertEqual(rv.keep_indices(10, 1.0, removals),
                         [0, 1, 2, 3, 5, 6, 7, 8, 9])

    def test_cue_inside_removal_is_clamped(self):
        removals = [(3.15, 4.05)]
        cues = [CaptionCue("mid", 3.3, 3.9)]
        out = rv.remap_cues(cues, removals)
        # Start and end both map to the removal start (3.15);
        # the cue collapses below the 0.2s floor and drops.
        self.assertEqual(out, [])


class TestCompare(unittest.TestCase):
    def test_all_match(self):
        draft = _draft([_check("F1", {
            "hook_present_at_frame_0": True,
            "audio_rms_db_0_2s": -20.0})])
        new = [_check("F1", {
            "hook_present_at_frame_0": True,
            "audio_rms_db_0_2s": -20.01})]
        out = rv.compare_draft(draft, new, _inputs(hook=object()),
                               None, {})
        self.assertEqual(out["checks"][0]["status"], "MATCH")

    def test_tampered_value_is_mismatch(self):
        # Tamper the RECORDED value the way the proof pack does:
        # a 6.86 dB audio difference must not hide in tolerance.
        draft2 = _draft([_check("F1", {"audio_rms_db_0_2s": -17.72})])
        new2 = [_check("F1", {"audio_rms_db_0_2s": -24.58})]
        out2 = rv.compare_draft(draft2, new2, _inputs(), None, {})
        self.assertEqual(out2["checks"][0]["status"], "MISMATCH")

    def test_wonder_recorded_vs_not_recorded(self):
        base = {"wonder_can_summarize_muted_hook": True,
                "wonder_summary": "x",
                "max_silence_gap_s": 0.3,
                "first_speech_onset_s": 0.1,
                "captions_present": True,
                "caption_coverage_of_speech": 1.0}
        draft = _draft([_check("F5", dict(base))],
                       meta_calls=[{"model": "muse-wonder-1.0",
                                    "purpose": "verifier-f5-muted-hook",
                                    "request_id": "resp_1"}])
        new = [_check("F5", dict(base, wonder_can_summarize_muted_hook=False,
                                 wonder_summary=None))]
        out = rv.compare_draft(
            draft, new,
            _inputs(cues=[CaptionCue("x", 0.0, 1.0)]), None, {})
        fields = {f["field"]: f["status"]
                  for f in out["checks"][0]["fields"]}
        self.assertEqual(fields["wonder_judgement"], "RECORDED")
        draft["meta_calls"] = []
        out2 = rv.compare_draft(
            draft, new,
            _inputs(cues=[CaptionCue("x", 0.0, 1.0)]), None, {})
        fields2 = {f["field"]: f["status"]
                   for f in out2["checks"][0]["fields"]}
        self.assertEqual(fields2["wonder_judgement"], "NOT-RECORDED")
        self.assertEqual(out2["checks"][0]["status"], "NOT-RECORDED")

    def test_f8_nested_graphics_compare(self):
        rec_g = [{"index": 0, "kind": "card", "motion": "static",
                  "face_overlap_px": 0, "presence_match_min": 0.95,
                  "ok": True}]
        draft = _draft([_check("F8", {
            "count": 1, "graphics": rec_g, "offending_indices": [],
            "collision_violations": []})],
            graphics=[{"kind": "card"}])
        new = [_check("F8", {
            "count": 1, "graphics": [dict(rec_g[0])],
            "offending_indices": [], "collision_violations": [],
            "backdrops": [], "sfx": None, "duration_s": 12.0})]
        inputs = _inputs(canvas_alphas=[np.zeros((2, 2))],
                         graphics=[object()],
                         caption_rect=(0, 0, 10, 10))
        out = rv.compare_draft(draft, new, inputs, None, {})
        self.assertEqual(out["checks"][0]["status"], "MATCH")


class TestResolve(unittest.TestCase):
    def test_basename_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "final.mp4"
            p.write_bytes(b"x")
            got = rv.resolve_artifact("/old/path/final.mp4",
                                      override=None, artifact_dir=None,
                                      evidence_dir=Path(td))
            self.assertEqual(got, p)

    def test_explicit_override_wins_over_stored_path(self):
        # The swapped-artifact case depends on this: when the
        # caller names a different file, the stored path must not
        # shadow it merely because it still exists.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            stored = td / "orig.mp4"
            stored.write_bytes(b"orig")
            swapped = td / "other.mp4"
            swapped.write_bytes(b"other")
            got = rv.resolve_artifact(str(stored), override=swapped,
                                      artifact_dir=None,
                                      evidence_dir=td)
            self.assertEqual(got, swapped)


class TestRunSummaryShape(unittest.TestCase):
    def test_run_level_shorts_list_matched_by_receipt_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            short_dir = td / "short-1"
            short_dir.mkdir()
            (td / "run-summary.json").write_text(json.dumps({
                "shorts": [
                    {"index": 1,
                     "layout": {"hook_y_frac": 0.59},
                     "files": {"receipt": str(short_dir / "receipt.json")}},
                    {"index": 2,
                     "layout": {"hook_y_frac": 0.62},
                     "files": {"receipt": str(td / "short-2"
                                               / "receipt.json")}},
                ]}))
            got = rv.find_run_summary(short_dir)
            self.assertEqual(got["layout"]["hook_y_frac"], 0.59)

    def test_short_report_returned_as_is(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "run-summary.json").write_text(json.dumps({
                "layout": {"hook_y_frac": 0.5}}))
            self.assertEqual(
                rv.find_run_summary(td)["layout"]["hook_y_frac"], 0.5)


class TestF3NestedCompare(unittest.TestCase):
    def test_hook_subkey_skipped_others_compared(self):
        # Hook unknown: the hook-bbox subkey is NOT-REPRODUCIBLE
        # while the caption and centroid subkeys still compare.
        draft = _draft([_check("F3", {"violations": {
            "hook_px_outside_central_band": {"left": 0, "top": 0,
                                             "right": 0, "bottom": 0},
            "caption_px_outside_lower_zone": {"left": 0, "top": 0,
                                              "right": 0, "bottom": 0},
            "face_centroid_median": [500.0, 1187.0]},
            "face_centroid_in_central_band": True})])
        new = [_check("F3", {"violations": {
            "caption_px_outside_lower_zone": {"left": 0, "top": 0,
                                              "right": 0, "bottom": 0},
            "face_centroid_median": [500.0, 1187.0]},
            "face_centroid_in_central_band": True})]
        # Only the hook is missing: caption block + centroid stay
        # comparable, the hook-bbox subkey alone is skipped.
        inputs = _inputs(caption_rect=(60, 1460, 980, 1600),
                         canvas_alphas=[np.zeros((2, 2))])
        out = rv.compare_draft(draft, new, inputs, None, {})
        fields = {f["field"]: f["status"]
                  for f in out["checks"][0]["fields"]}
        self.assertEqual(
            fields["violations.hook_px_outside_central_band"],
            "NOT-REPRODUCIBLE")
        self.assertEqual(
            fields["violations.caption_px_outside_lower_zone"], "MATCH")


if __name__ == "__main__":
    unittest.main()
