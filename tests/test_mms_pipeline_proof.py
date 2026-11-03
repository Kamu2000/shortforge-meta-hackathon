"""Offline end-to-end proof of the MMS pipeline path (fix phase).

The unit tests mock the subprocess boundary; THIS test runs the
real thing: the shipped pipeline's own word_timing entry
(``word_timings(source="mms")``) from the torch-free main
interpreter, through the subprocess into the isolated MMS venv,
on cached McArthur audio from the pipeline-run-ext-qm3 run -
the same window whose DSP-labelled captions Kamlesh QA'd.

Skips (never fails) when the venv, the entry script, or the
cached run artifacts are absent - e.g. a fresh clone - so the
suite stays green everywhere while still proving the path on
this machine.

Measured at proof time (2026-11-02): see extension-build-state /
the fix-phase report for the recorded timings.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.word_timing import (  # noqa: E402
    MMS_SAMPLE_RATE,
    MMS_VENV_PYTHON,
    _mms_normalise,
    load_samples,
    word_timings,
)

REPO = Path(__file__).resolve().parents[1]
SHORT_DIR = (Path.home() / "workspace" / "meta-hackathon"
             / "pipeline-run-ext-qm3" / "short-1")
AUDIO = SHORT_DIR / "normalised-audio.wav"
CAPTIONS = SHORT_DIR / "captions.json"


def _proof_cue() -> dict | None:
    """The first caption cue the aligner can fully represent,
    with at least five words (a real phrase, not a fragment)."""
    if not CAPTIONS.is_file():
        return None
    cues = json.loads(CAPTIONS.read_text())["cues"]
    for cue in cues:
        words = cue["text"].split()
        if len(words) >= 5 and all(_mms_normalise(w) for w in words):
            return cue
    return None


@unittest.skipUnless(
    MMS_VENV_PYTHON.is_file()
    and (REPO / "tools" / "mms_align.py").is_file()
    and AUDIO.is_file() and _proof_cue() is not None,
    "MMS venv / entry script / cached McArthur artifacts absent")
class MmsPipelineProofTests(unittest.TestCase):
    def test_word_timings_mms_end_to_end_on_cached_audio(self):
        cue = _proof_cue()
        words = cue["text"].split()
        samples = load_samples(AUDIO)
        i0 = int(round(cue["start_s"] * MMS_SAMPLE_RATE))
        i1 = int(round(cue["end_s"] * MMS_SAMPLE_RATE))
        timings = word_timings(
            words, cue["start_s"], cue["end_s"], source="mms",
            samples=samples[i0:i1])
        # Exactly the cue's words, all labelled by their producer.
        self.assertEqual([t.word for t in timings], words)
        self.assertTrue(all(t.source == "mms" for t in timings))
        # Inside the cue window (small CTC tolerance), monotone,
        # positive-length spans - placements, not fabrications.
        prev = cue["start_s"] - 0.05
        for t in timings:
            self.assertGreaterEqual(t.start_s, cue["start_s"] - 0.05)
            self.assertLessEqual(t.end_s, cue["end_s"] + 0.05)
            self.assertGreaterEqual(t.start_s, prev - 0.001)
            self.assertGreater(t.end_s, t.start_s)
            prev = t.start_s
        # The detail names the model that produced them.
        self.assertIn("mms-1b", timings[0].detail)


if __name__ == "__main__":
    unittest.main()
