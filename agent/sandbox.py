"""One attempt's staging area, budgets and public view of the platform.

The sandbox is the agent's only boundary with the simulator: it holds a FRESH
`Platform` per attempt, stages actions, enforces hard budgets, and exposes only
public information about humans. It has no reference to the detector, ground truth
or scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right

import numpy as np

from agent.actions import Action, execute
from simulator.bots import Farm
from simulator.events import Event
from simulator.platform import Platform
from simulator.world import World


class BudgetError(ValueError):
    """A hard budget stopped the request."""


@dataclass(frozen=True)
class Budget:
    """Same limits every round, for both sides of the comparison."""

    max_attempts: int = 5
    max_accounts: int = 60
    max_actions: int = 60_000  # per attempt, executed actions after profile expansion
    max_tokens: int = 400_000  # LLM tokens across the whole round
    wall_clock_s: float = 1_800.0
    max_calls_per_attempt: int = 40


@dataclass(frozen=True)
class WorldView:
    """What is publicly visible about the humans: their posts and how active they are."""

    horizon: int  # seconds of history, and the end of the attempt window
    human_posts: tuple[Event, ...]  # time-ordered
    post_ts: tuple[int, ...]
    human_events: tuple[Event, ...]

    @classmethod
    def from_world(cls, world: World) -> "WorldView":
        events = tuple(world.events)
        posts = tuple(e for e in events if e.action == "post")
        return cls(events[-1].sim_ts, posts, tuple(p.sim_ts for p in posts), events)

    @property
    def days(self) -> float:
        return self.horizon / 86_400


@dataclass(frozen=True)
class CommitResult:
    events: list[Event]
    ground_truth: dict[str, str]
    farm: Farm
    executed: list[Action]
    rejected: int
    dropped: int


class Sandbox:
    def __init__(self, world: World, budget: Budget, seed: int, attempt: int) -> None:
        self.world = world
        self.view = WorldView.from_world(world)
        self.budget = budget
        self.attempt = attempt
        self.rng = np.random.default_rng([seed, attempt])
        self.accounts: list[str] = []
        self.targets: set[str] = set()
        self._staged: list[Action] = []
        self._actions_used = 0
        self._dropped = 0

    @property
    def actions_remaining(self) -> int:
        return self.budget.max_actions - self._actions_used

    def create_accounts(self, count: int, distinct_ips: int = 1) -> list[str]:
        if not 1 <= count:
            raise BudgetError("count must be at least 1")
        if len(self.accounts) + count > self.budget.max_accounts:
            raise BudgetError(
                f"account budget: {len(self.accounts)} exist, {count} more would exceed "
                f"the limit of {self.budget.max_accounts}"
            )
        new = [f"bot-{len(self.accounts) + i + 1:03d}" for i in range(count)]
        for i, acct in enumerate(new):
            self._staged.append(Action(0, "create", acct, f"198.51.100.{1 + (i % max(1, distinct_ips))}"))
        self.accounts += new
        return new

    def stage(self, actions: list[Action], dropped: int = 0) -> int:
        """Stage expanded actions, trimming to the action budget. Returns how many were kept."""
        room = max(0, self.actions_remaining)
        kept = actions[:room]
        self._dropped += dropped + (len(actions) - len(kept))
        self._staged += kept
        self._actions_used += len(kept)
        return len(kept)

    def human_posts_before(self, ts: int) -> int:
        return bisect_right(self.view.post_ts, ts)

    def commit(self) -> CommitResult:
        """Run everything staged on a fresh platform and return the resulting events."""
        platform = Platform(self.world)
        executed, rejected = execute(platform, self._staged)
        return CommitResult(
            events=platform.events_until(10**12),
            ground_truth=dict(platform.ground_truth),
            farm=Farm(tuple(self.accounts), tuple(sorted(self.targets)), ()),
            executed=executed,
            rejected=rejected,
            dropped=self._dropped,
        )
