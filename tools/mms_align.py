#!/usr/bin/env python3
"""MMS forced-alignment entry point for the isolated venv.

Executed BY the MMS venv interpreter (torch + torchaudio live
there; the shipped pipeline's interpreter has neither) from
``spatial_director.word_timing.mms_word_timings``:

    mms_align.py <request.json> <response.json> <samples.npy>

request.json: {"words": [...], "sample_rate": 16000}
samples.npy:  float32 mono samples for exactly those words' span.

The response is always written (unless the process dies before
main runs - the caller treats a missing/unusable response as
MMSUnavailableError):

    {"status": "ok", "word_spans": [[f0, f1], ...],
     "n_frames": int, "total_s": float}
    {"status": "error",
     "error_type": "unavailable" | "word", "message": str}

error_type mirrors the caller's exception taxonomy:
MMSUnavailableError -> "unavailable", any other WordTimingError
-> "word"; an unexpected runtime failure is reported as
"unavailable" (the aligner could not be made to work).

No transcript text is created here - the words come from the
caller (the Voice Transcribe transcript) and are only placed in
time. Written 2026-11-02 (fix phase) alongside the subprocess
refactor of word_timing.mms_word_timings.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.word_timing import (  # noqa: E402
    MMSUnavailableError,
    WordTimingError,
    _mms_align_in_process,
)


def main(argv: list[str]) -> int:
    request_path, response_path, samples_path = (
        Path(argv[1]), Path(argv[2]), Path(argv[3]))
    request = json.loads(request_path.read_text())
    samples = np.load(samples_path)
    try:
        spans, n_frames, total_s = _mms_align_in_process(
            [str(w) for w in request["words"]], samples,
            int(request["sample_rate"]))
        payload = {
            "status": "ok",
            "word_spans": [[int(a), int(b)] for a, b in spans],
            "n_frames": int(n_frames),
            "total_s": float(total_s),
        }
    except MMSUnavailableError as exc:
        payload = {"status": "error", "error_type": "unavailable",
                   "message": str(exc)}
    except WordTimingError as exc:
        payload = {"status": "error", "error_type": "word",
                   "message": str(exc)}
    except Exception as exc:  # runtime failure inside the aligner
        payload = {"status": "error", "error_type": "unavailable",
                   "message": "MMS forced aligner unavailable: "
                              f"runtime failure in the alignment "
                              f"subprocess ({type(exc).__name__}: "
                              f"{exc})"}
    response_path.write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
