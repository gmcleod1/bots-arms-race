"""Scripted bots. Round 0 of the face-off is a naive farm; the autonomous agent comes later.

`Script` collects timed actions and replays them in time order through the real
`Platform`, so anything built on it obeys the same rules a bot would: the clock
only moves forward and targets must exist.

The naive farm is one operator running a script with no evasion. Many accounts post
canned text on a fixed clock around the clock, and all of them like and follow the same
target humans in lockstep, from one IP. Its purpose is to inflate those targets'
engagement, so `engagement_delivered` measures what it achieved: the thing the detector
is protecting.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable

import numpy as np

from simulator import content
from simulator.content import WritingStyle
from simulator.events import Event
from simulator.platform import Platform

PLAIN_STYLE = WritingStyle(0.0, (), False, "", (), 0.0)  # no typos, no personal quirks


class Script:
    def __init__(self, platform: Platform) -> None:
        self.p = platform
        self.actions: list[tuple[int, int, partial]] = []

    def _add(self, ts: int, fn: partial) -> None:
        self.actions.append((ts, len(self.actions), fn))

    def create(self, acct: str, ts: int, ip: str = "198.51.100.1") -> None:
        self._add(ts, partial(self.p.create_account, acct, ip, ts))

    def post(self, acct: str, ts: int, text: str) -> None:
        self._add(ts, partial(self.p.post, acct, ts, text))

    def like(self, acct: str, ts: int, post_id: str) -> None:
        self._add(ts, partial(self.p.like, acct, ts, post_id))

    def comment(self, acct: str, ts: int, post_id: str, text: str) -> None:
        self._add(ts, partial(self.p.comment, acct, ts, post_id, text))

    def follow(self, acct: str, ts: int, target: str) -> None:
        self._add(ts, partial(self.p.follow, acct, ts, target))

    def run(self) -> None:
        for _, _, fn in sorted(self.actions, key=lambda a: (a[0], a[1])):
            fn()


@dataclass(frozen=True)
class NaiveFarmConfig:
    n_bots: int = 40
    ip: str = "198.51.100.10"  # one farm location
    start: int = 0  # the farm exists from the start of the window
    post_interval: int = 1_800  # every bot posts on a fixed 30 minute clock
    stagger: int = 3  # seconds between one bot's clock and the next
    n_canned: int = 5  # size of the shared pool of messages
    n_targets: int = 3  # humans whose engagement the farm inflates
    react_delay: int = 20  # seconds after a target posts before the first bot likes it
    react_spacing: int = 1  # seconds between successive bots liking the same post


@dataclass(frozen=True)
class Farm:
    accounts: tuple[str, ...]
    targets: tuple[str, ...]  # human accounts being boosted
    canned: tuple[str, ...]  # the shared messages every bot cycles through


def naive_farm(
    platform: Platform,
    rng: np.random.Generator,
    config: NaiveFarmConfig = NaiveFarmConfig(),
    prefix: str = "nf",
) -> Farm:
    """Build and run a naive farm on `platform` for the length of its human history."""
    human = platform.human_events
    until = human[-1].sim_ts
    posts_by: dict[str, int] = {}
    for e in human:
        if e.action == "post":
            posts_by[e.account_id] = posts_by.get(e.account_id, 0) + 1
    eligible = sorted(a for a, n in posts_by.items() if n >= 20)
    picks = sorted(int(i) for i in rng.choice(len(eligible), size=config.n_targets, replace=False))
    targets = tuple(eligible[i] for i in picks)
    canned = tuple(
        content.compose(rng, "post", int(rng.integers(8)), PLAIN_STYLE) for _ in range(config.n_canned)
    )
    accounts = tuple(f"{prefix}-{i:03d}" for i in range(config.n_bots))

    s = Script(platform)
    for i, acct in enumerate(accounts):
        s.create(acct, config.start, config.ip)
        for target in targets:
            s.follow(acct, config.start + 1 + i, target)
        t, k = config.start + i * config.stagger, 0
        while t <= until:
            s.post(acct, t, canned[k % len(canned)])
            t, k = t + config.post_interval, k + 1
    target_set = set(targets)
    for e in human:
        if e.action == "post" and e.account_id in target_set:
            for i, acct in enumerate(accounts):
                s.like(acct, e.sim_ts + config.react_delay + i * config.react_spacing, e.post_id)
    s.run()
    return Farm(accounts, targets, canned)


def engagement_delivered(events: Iterable[Event], farm: Farm, by: Iterable[str] | None = None) -> int:
    """Likes, comments and follows the farm's accounts gave its targets (optionally only `by` accounts)."""
    events = list(events)
    actors = set(farm.accounts if by is None else by)
    targets = set(farm.targets)
    target_posts = {e.post_id for e in events if e.action == "post" and e.account_id in targets}
    return sum(
        1
        for e in events
        if e.account_id in actors
        and (
            (e.action in ("like", "comment") and e.target_id in target_posts)
            or (e.action == "follow" and e.target_id in targets)
        )
    )
