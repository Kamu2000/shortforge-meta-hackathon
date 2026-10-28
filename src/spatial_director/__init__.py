"""Verifiable Spatial Director.

Built from scratch during the Meta Global AI Developer Hackathon
(Oct 26 - Nov 4, 2026), Startup Track, "Build for Your World".

Pipeline: a long talking-head video is transcribed and scored, the
strongest moments are selected, SAM 3.1 segments the speaker per frame,
and a 3-layer composite places the hook text *behind* the speaker
cutout. The rendered Short is then checked by a post-render verifier
(deterministic geometry checks plus one Muse Wonder pass over the
actual render); on a FAIL, one repair is attempted and the clip is
re-verified. Every clip ships with a PASS/FAIL receipt.

Modules land in dated steps during the event; see BUILDLOG.md.

spend_ledger   Cumulative Meta API spend ledger with a hard cap. This
               is the enforcement behind the BUILDLOG billing claim:
               every paid call checks the ledger before it is sent,
               and the ledger refuses any call that would push the
               cumulative total past the cap.
sam_client     SAM 3.1 client (Responses API). Default concept prompt
               is the bare noun "person" - the only speaker prompt with
               practitioner evidence behind it.
"""

__version__ = "0.1.0"

__all__ = ["spend_ledger", "sam_client"]
