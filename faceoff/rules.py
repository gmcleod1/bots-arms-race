"""The win condition (decision 6.5): a measurable verdict per attempt, and a round tally.

PROPOSED, not yet confirmed by Garfield. The attacker wins an attempt by keeping at least
`attacker_min_kept` engagement after suspensions (a share of what a canonical naive farm
delivers). The detector wins the attempt otherwise, but only if it stayed within
`max_human_fpr`: a detector that flags everyone forfeits, which is what stops "ban everyone"
from being a winning patch. That is the false-positive stance with an explicit ceiling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent.referee import AttemptOutcome
from agent.reference_plans import naive
from agent.sandbox import Budget, Sandbox
from agent.tools import ToolBox
from simulator.bots import engagement_delivered
from simulator.world import World


@dataclass(frozen=True)
class WinCondition:
    attacker_min_kept: int  # engagement kept after suspensions that wins an attempt for the attacker
    max_human_fpr: float = 0.10  # share of humans flagged above which the detector forfeits
    naive_engagement: int = 0  # what the canonical naive farm delivers here, for context

    @classmethod
    def from_baseline(cls, naive_engagement: int, kept_fraction: float = 0.10,
                      max_human_fpr: float = 0.10) -> "WinCondition":
        return cls(max(1, round(kept_fraction * naive_engagement)), max_human_fpr, naive_engagement)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "WinCondition":
        return cls(**data)


@dataclass(frozen=True)
class Judgement:
    winner: str  # "attacker" or "detector"
    reason: str
    human_fpr: float
    kept: int


def judge(outcome: AttemptOutcome, rule: WinCondition) -> Judgement:
    fpr = outcome.confusion.fpr or 0.0
    kept = outcome.engagement_kept
    if fpr > rule.max_human_fpr:
        return Judgement("attacker", f"detector flagged {fpr:.1%} of humans, over its {rule.max_human_fpr:.0%} limit",
                         fpr, kept)
    if kept >= rule.attacker_min_kept:
        return Judgement("attacker", f"farm kept {kept:,} engagement, at or above the {rule.attacker_min_kept:,} needed",
                         fpr, kept)
    return Judgement("detector", f"farm kept only {kept:,} of the {rule.attacker_min_kept:,} needed", fpr, kept)


def tally(judgements: list[Judgement]) -> dict[str, Any]:
    a = sum(1 for j in judgements if j.winner == "attacker")
    d = len(judgements) - a
    return {"attacker": a, "detector": d, "leader": "attacker" if a > d else "detector" if d > a else "draw"}


def naive_baseline(world: World, bots: int = 40) -> int:
    """Engagement the canonical naive farm delivers on this world's top targets (no detector needed)."""
    box = ToolBox(Sandbox(world, Budget(), 0, 1), lambda s: {})
    targets = [t["account"] for t in box.call("list_targets", {"limit": 3})["targets"]]
    box.call("create_accounts", {"count": bots})
    result = box.call("apply_profile", naive(targets))
    if "error" in result:
        raise ValueError(result["error"])
    commit = box.sb.commit()
    return engagement_delivered(commit.events, commit.farm)
