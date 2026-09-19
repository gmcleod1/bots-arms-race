"""Runs the detector on an attempt and decides what the agent may learn from it.

The agent sees what a real attacker sees: which of its own accounts were suspended
and how much of the boost survived. It never sees scores, p-values, which signal
fired, or anything about humans' verdicts. The defender's view (precision, recall,
humans flagged) is kept separately in `AttemptOutcome` for the scoreboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.sandbox import CommitResult
from detector.metrics import Confusion, confusion
from simulator.bots import engagement_delivered

FEEDBACK_KEYS = frozenset({
    "attempt", "attempts_remaining", "accounts_created", "accounts_suspended", "suspended_ids",
    "engagement_delivered_total", "engagement_kept_after_suspensions", "actions_executed",
    "actions_rejected", "actions_dropped_by_limit", "note",
})


class _Detector(Protocol):
    def score(self, events): ...


@dataclass
class AttemptOutcome:
    attempt: int
    commit: CommitResult
    flagged: set[str]  # every account the detector flagged, humans included (defender's view)
    suspended: list[str]  # the agent's accounts among them
    confusion: Confusion  # defender-side: precision and recall reported separately
    engagement_total: int
    engagement_kept: int
    feedback: dict[str, Any]  # the only thing the agent is told


class Referee:
    def __init__(self, detector: _Detector) -> None:
        self.detector = detector

    def evaluate(self, attempt: int, commit: CommitResult, attempts_remaining: int) -> AttemptOutcome:
        verdicts = self.detector.score(commit.events)
        flagged = {a for a, v in verdicts.items() if v.flagged}
        mine = set(commit.farm.accounts)
        suspended = sorted(mine & flagged)
        survivors = sorted(mine - flagged)
        total = engagement_delivered(commit.events, commit.farm)
        kept = engagement_delivered(commit.events, commit.farm, by=survivors)
        feedback = {
            "attempt": attempt,
            "attempts_remaining": attempts_remaining,
            "accounts_created": len(mine),
            "accounts_suspended": len(suspended),
            "suspended_ids": suspended,
            "engagement_delivered_total": total,
            "engagement_kept_after_suspensions": kept,
            "actions_executed": len(commit.executed) - len(mine),  # creates are not activity
            "actions_rejected": commit.rejected,
            "actions_dropped_by_limit": commit.dropped,
            "note": "The platform has been reset. Your accounts are gone; the humans are unchanged."
            if attempts_remaining else "That was your last attempt.",
        }
        assert set(feedback) <= FEEDBACK_KEYS
        return AttemptOutcome(
            attempt=attempt, commit=commit, flagged=flagged, suspended=suspended,
            confusion=confusion(flagged, commit.ground_truth),
            engagement_total=total, engagement_kept=kept, feedback=feedback,
        )
