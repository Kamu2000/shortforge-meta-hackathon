"""Cumulative spend ledger with a hard cap for Meta Model API calls.

Why this exists: the hackathon's $150 credit grant is the entire API
budget (zero spend from the team's own funds), and the platform Spend
Limit is a notification only - it does NOT stop billing. So the cap is
enforced here, in code, cumulatively:

- The ledger is an append-only JSONL file. Every entry names the model,
  the cost in USD, what the call was for, and (when known) the request
  id, so the ledger can be reconciled against the Billing page.
- Before any paid call, the caller must run ``check_can_afford`` with
  the call's estimated cost. If cumulative spend + estimate would pass
  the cap, SpendCapError is raised and the call is never sent.
- ``record`` re-checks the cap before appending, so an actual cost that
  comes in higher than its estimate still cannot push the ledger past
  the cap silently - it raises instead, and the caller must surface it.

Default cap: $145.00 against the $150 grant. The $5 buffer covers
billing lag and token-count uncertainty; usage beyond the grant would
bill the team's card, which the event plan forbids.

Stdlib only. Written fresh in-repo on 2026-10-28.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CAP_USD = 145.00


class SpendCapError(RuntimeError):
    """Raised when a call would push cumulative spend past the cap."""


@dataclass(frozen=True)
class LedgerEntry:
    ts: str  # ISO date or datetime, caller's clock
    model: str
    cost_usd: float
    purpose: str
    request_id: str | None = None


class SpendLedger:
    """Append-only cumulative ledger backed by a JSONL file."""

    def __init__(self, path: str | Path, cap_usd: float = DEFAULT_CAP_USD):
        self.path = Path(path)
        self.cap_usd = float(cap_usd)

    # -- reading -----------------------------------------------------

    def entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        out: list[LedgerEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append(
                LedgerEntry(
                    ts=str(raw["ts"]),
                    model=str(raw["model"]),
                    cost_usd=float(raw["cost_usd"]),
                    purpose=str(raw["purpose"]),
                    request_id=raw.get("request_id"),
                )
            )
        return out

    def total_usd(self) -> float:
        return round(sum(e.cost_usd for e in self.entries()), 4)

    def remaining_usd(self) -> float:
        return round(self.cap_usd - self.total_usd(), 4)

    # -- enforcement ---------------------------------------------------

    def check_can_afford(self, estimated_cost_usd: float) -> None:
        """Raise SpendCapError if cumulative + estimate exceeds the cap.

        Callers run this BEFORE sending a paid request. An estimate of
        0 is always allowed (free reads are not gated).
        """
        if estimated_cost_usd < 0:
            raise ValueError("estimated cost cannot be negative")
        projected = round(self.total_usd() + float(estimated_cost_usd), 4)
        if projected > self.cap_usd:
            raise SpendCapError(
                f"projected cumulative spend ${projected:.2f} exceeds cap "
                f"${self.cap_usd:.2f} (current ${self.total_usd():.2f} + "
                f"estimate ${estimated_cost_usd:.2f}); refusing the call"
            )

    def record(
        self,
        *,
        ts: str,
        model: str,
        cost_usd: float,
        purpose: str,
        request_id: str | None = None,
    ) -> LedgerEntry:
        """Append one completed call, re-checking the cap first.

        If the actual cost would pass the cap, SpendCapError is raised
        and NOTHING is appended - the caller must surface the breach
        rather than let the ledger quietly exceed its ceiling.
        """
        cost = round(float(cost_usd), 4)
        if cost < 0:
            raise ValueError("cost cannot be negative")
        self.check_can_afford(cost)
        entry = LedgerEntry(
            ts=ts, model=model, cost_usd=cost, purpose=purpose,
            request_id=request_id,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry
