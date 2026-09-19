"""The defender's console: what the human at the detector can see and do during a round.

Deliberately narrow. The console holds the detector host, anonymised telemetry (flags,
scores, near misses) and a gate. It holds NO ground truth, no attempt outcomes, no agent
log and no reference to the agent, so the defender works the way a real one does: from their
own telemetry, not from knowing which accounts are bots. Account ids are anonymised per round
so "bot-007" versus "h-0042" cannot leak a label.

The audience scoreboard (scoreboard/feed.py) shows the truth-labelled results. Keep it off the
defender's screen while patching, or the label leak is back.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable

from detector.detector import Verdict
from faceoff.host import DetectorHost, DetectorVersion, PatchResult

NEAR_MISSES = 10


@dataclass(frozen=True)
class FlagRow:
    account: str  # anonymised
    score: float
    top_feature: str
    fired: tuple[str, ...]  # every feature at or below its budget share


@dataclass(frozen=True)
class Telemetry:
    attempt: int
    version: str
    scored_accounts: int
    flagged: tuple[FlagRow, ...]
    near_misses: tuple[FlagRow, ...]  # highest-scoring accounts that were NOT flagged: what may be slipping


def make_anonymiser(round_id: str) -> Callable[[str], str]:
    def anon(account: str) -> str:
        return "a-" + hashlib.blake2b(f"{round_id}:{account}".encode(), digest_size=4).hexdigest()

    return anon


def make_telemetry(attempt: int, version: DetectorVersion, verdicts: dict[str, Verdict],
                   anon: Callable[[str], str]) -> Telemetry:
    def row(v: Verdict) -> FlagRow:
        fired = tuple(sorted(f for f, p in v.pvalues.items() if p <= version.alpha))
        return FlagRow(anon(v.account_id), round(v.score, 3), v.top_feature, fired)

    ranked = sorted(verdicts.values(), key=lambda v: (-v.score, v.account_id))
    flagged = [row(v) for v in ranked if v.flagged]
    near = [row(v) for v in ranked if not v.flagged][:NEAR_MISSES]
    return Telemetry(attempt, version.id, len(verdicts), tuple(flagged), tuple(near))


class DefenderConsole:
    def __init__(self, host: DetectorHost, telemetry: list[Telemetry], gate: threading.Event,
                 waiting: threading.Event) -> None:
        self._host, self._telemetry, self._gate, self._waiting = host, telemetry, gate, waiting

    def status(self) -> dict[str, Any]:
        v = self._host.current
        return {
            "version": v.id, "label": v.label, "features": list(v.features), "alpha": v.alpha,
            "fpr_budget": v.config.fpr_budget, "settings": v.config.to_json(),
            "pending_patches": [label for label, _ in self._host.pending],
            "frozen": self._host.frozen, "waiting_for_you": self._waiting.is_set(),
            "attempts_scored": len(self._telemetry),
        }

    def latest(self) -> Telemetry | None:
        return self._telemetry[-1] if self._telemetry else None

    def history(self) -> list[Telemetry]:
        return list(self._telemetry)

    def versions(self) -> list[DetectorVersion]:
        return list(self._host.versions)

    def patch(self, label: str, **changes: Any) -> PatchResult:
        """Change the detector. Live: applies at once (the run in flight is unaffected).
        In fair-play mode: queued until the current attempt has been scored."""
        return self._host.patch(label, **changes)

    def ready(self) -> None:
        """Release the gate (fair-play mode): the next attempt may begin."""
        self._gate.set()
