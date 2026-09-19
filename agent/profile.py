"""Behavior profiles: the knobs the agent turns, and the actions they expand into.

Every field is a lever the agent may set; none has an opinion about what is a good
setting. Defaults are deliberately the plain, unevasive choices (one canned pool, a
fixed-ish rhythm, no camouflage) so any evasion is something the agent chose.

What is NOT a knob, and so is outside the agent's possibility space here: it cannot
change human accounts, read or alter the detector, or see scores. Bots cannot engage
each other's posts through a profile (their post ids do not exist until the attempt
runs); `schedule_actions` is limited to human posts and accounts for the same reason.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, fields
from typing import Any, Sequence

import numpy as np

from agent.actions import Action
from agent.sandbox import WorldView
from simulator import content
from simulator.bots import PLAIN_STYLE

RECENCY_WINDOW = 72 * 3600
N_COMMUNITIES = len(content.NOUNS)


class ProfileError(ValueError):
    """The agent supplied an argument outside what the platform tool accepts."""


def _num(name: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProfileError(f"{name} must be a number")
    if not lo <= value <= hi:
        raise ProfileError(f"{name} must be between {lo} and {hi}, got {value}")
    return float(value)


@dataclass(frozen=True)
class Timing:
    sessions_per_day: float = 4.0  # how often a bot starts a burst of activity (Poisson)
    events_per_session: float = 4.0  # average actions in a burst (geometric)
    pause_median_s: float = 25.0  # typical pause between actions inside a burst
    pause_sigma: float = 1.1  # 0 = identical pauses; higher = heavier-tailed
    awake_hours: float = 16.0  # length of the daily window bursts may start in; 24 = never sleeps
    wake_hour_utc: float = 7.0  # when that window opens (UTC)
    tz_spread_hours: float = 0.0  # each bot's window shifts by a random amount within +/- this
    fixed_interval_s: float | None = None  # if set: one action on an exact clock, ignoring bursts


@dataclass(frozen=True)
class Content:
    text_mode: str = "canned"  # canned = shared pool; spun = pool with word swaps; fresh = new text each time
    pool_size: int = 5  # messages in the shared pool (canned, spun)
    typo_rate: float = 0.0  # chance per word (4+ letters) of a random slip
    style: str = "plain"  # plain, or human = each bot gets a personal typo/casing/emoji style (fresh mode)
    post_share: float = 0.3  # share of actions that are posts; the rest like a recent human post


@dataclass(frozen=True)
class Engagement:
    targets: tuple[str, ...] = ()  # human accounts whose engagement the farm inflates
    react_probability: float = 0.0  # chance each bot likes each new post by a target
    react_delay_median_s: float = 30.0  # how long after the post
    react_delay_sigma: float = 0.0  # 0 = every bot waits exactly the median
    follow_targets: bool = False
    follow_delay_median_s: float = 60.0


@dataclass(frozen=True)
class Profile:
    timing: Timing = field(default_factory=Timing)
    content: Content = field(default_factory=Content)
    engagement: Engagement = field(default_factory=Engagement)

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> "Profile":
        unknown = set(args) - {"timing", "content", "engagement"}
        if unknown:
            raise ProfileError(f"unknown profile section(s): {sorted(unknown)}")
        return cls(
            timing=_build(Timing, args.get("timing", {}), _TIMING_RANGES),
            content=_build(Content, args.get("content", {}), _CONTENT_RANGES),
            engagement=_build(Engagement, args.get("engagement", {}), _ENGAGEMENT_RANGES),
        )


_TIMING_RANGES = {
    "sessions_per_day": (0.05, 48), "events_per_session": (1, 40), "pause_median_s": (1, 7_200),
    "pause_sigma": (0, 3), "awake_hours": (2, 24), "wake_hour_utc": (0, 23.99),
    "tz_spread_hours": (0, 12), "fixed_interval_s": (30, 86_400),
}
_CONTENT_RANGES = {"pool_size": (1, 50), "typo_rate": (0, 0.2), "post_share": (0, 1)}
_ENGAGEMENT_RANGES = {
    "react_probability": (0, 1), "react_delay_median_s": (1, 172_800), "react_delay_sigma": (0, 3),
    "follow_delay_median_s": (1, 432_000),
}
_CHOICES = {"text_mode": ("canned", "spun", "fresh"), "style": ("plain", "human")}


def _build(cls, raw: dict[str, Any], ranges: dict[str, tuple[float, float]]):
    if not isinstance(raw, dict):
        raise ProfileError(f"{cls.__name__.lower()} must be an object")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ProfileError(f"unknown {cls.__name__.lower()} field(s): {sorted(unknown)}")
    values: dict[str, Any] = {}
    for name, value in raw.items():
        if name in ranges:
            if name == "fixed_interval_s" and value is None:
                values[name] = None
                continue
            number = _num(name, value, *ranges[name])
            values[name] = int(number) if name == "pool_size" else number
        elif name in _CHOICES:
            if value not in _CHOICES[name]:
                raise ProfileError(f"{name} must be one of {_CHOICES[name]}, got {value!r}")
            values[name] = value
        elif name == "targets":
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise ProfileError("targets must be a list of account ids")
            values[name] = tuple(value)
        elif name == "follow_targets":
            if not isinstance(value, bool):
                raise ProfileError("follow_targets must be true or false")
            values[name] = value
    return cls(**values)


def expected_actions(profile: Profile, n_accounts: int, days: float) -> float:
    """Rough size of an expansion, used to refuse absurd profiles before generating them."""
    t = profile.timing
    per_account = (days * 86_400 / t.fixed_interval_s) if t.fixed_interval_s else (
        days * t.sessions_per_day * t.events_per_session * min(1.0, t.awake_hours / 24 + 0.05)
    )
    return n_accounts * per_account


def _times(t: Timing, rng: np.random.Generator, horizon: int, offset_hours: float) -> list[int]:
    start = (t.wake_hour_utc + offset_hours) % 24

    def awake(ts: float) -> bool:
        return t.awake_hours >= 24 or ((ts / 3600.0 - start) % 24) < t.awake_hours

    out: list[int] = []
    if t.fixed_interval_s:
        ts = float(rng.uniform(0, t.fixed_interval_s))
        while ts <= horizon:
            if awake(ts):
                out.append(int(ts))
            ts += t.fixed_interval_s
        return out
    rate = t.sessions_per_day / 86_400
    ts = 0.0
    while True:
        ts += float(rng.exponential(1.0 / rate))
        if ts > horizon:
            return sorted(out)
        if not awake(ts):
            continue
        n = int(rng.geometric(1.0 / t.events_per_session))
        pauses = (
            rng.lognormal(math.log(t.pause_median_s), t.pause_sigma, n - 1)
            if t.pause_sigma > 0 else np.full(n - 1, t.pause_median_s)
        )
        for when in ts + np.concatenate(([0.0], np.cumsum(pauses))):
            if when <= horizon:
                out.append(int(when))


def _slip(rng: np.random.Generator, text: str, rate: float) -> str:
    if rate <= 0:
        return text
    words = text.split(" ")
    for i, w in enumerate(words):
        core = w.strip(".,!?:;")
        if len(core) >= 4 and rng.random() < rate:
            words[i] = w.replace(core, content._random_typo(rng, core.lower()))
    return " ".join(words)


_SWAPPABLE = [ADJ for ADJ in content.ADJECTIVES + content.FEELINGS if " " not in ADJ]


def _spin(rng: np.random.Generator, text: str) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        core = w.strip(".,!?:;")
        if core.lower() in _SWAPPABLE and rng.random() < 0.5:
            words[i] = w.replace(core, _SWAPPABLE[int(rng.integers(len(_SWAPPABLE)))])
    return " ".join(words)


def expand(
    profile: Profile,
    accounts: Sequence[str],
    view: WorldView,
    rng: np.random.Generator,
    max_actions: int,
) -> tuple[list[Action], int]:
    """Turn a profile into timed actions for `accounts`. Returns (actions, dropped_by_cap)."""
    t, c, e = profile.timing, profile.content, profile.engagement
    pool = [
        content.compose(rng, "post", int(rng.integers(N_COMMUNITIES)), PLAIN_STYLE)
        for _ in range(c.pool_size)
    ]
    actions: list[Action] = []
    for acct in accounts:
        offset = float(rng.uniform(-t.tz_spread_hours, t.tz_spread_hours)) if t.tz_spread_hours else 0.0
        style = content.make_style(rng) if c.style == "human" else PLAIN_STYLE
        community = int(rng.integers(N_COMMUNITIES))
        for ts in _times(t, rng, view.horizon, offset):
            like_target = None
            if rng.random() >= c.post_share:
                like_target = _recent_post(view, rng, ts)
            if like_target is not None:
                actions.append(Action(ts, "like", acct, like_target))
                continue
            if c.text_mode == "fresh":
                text = content.compose(rng, "post", community, style)
            else:
                text = pool[int(rng.integers(len(pool)))]
                text = _spin(rng, text) if c.text_mode == "spun" else text
            actions.append(Action(ts, "post", acct, None, _slip(rng, text, c.typo_rate)))
    targets = set(e.targets)
    if targets and e.react_probability > 0:
        for post in view.human_posts:
            if post.account_id not in targets:
                continue
            for acct in accounts:
                if rng.random() < e.react_probability:
                    delay = _delay(rng, e.react_delay_median_s, e.react_delay_sigma)
                    if post.sim_ts + delay <= view.horizon:
                        actions.append(Action(post.sim_ts + delay, "like", acct, post.post_id))
    if targets and e.follow_targets:
        for acct in accounts:
            for target in sorted(targets):
                actions.append(Action(_delay(rng, e.follow_delay_median_s, e.react_delay_sigma or 0.5),
                                      "follow", acct, target))
    actions.sort(key=lambda a: a.ts)
    dropped = max(0, len(actions) - max_actions)
    return actions[:max_actions], dropped


def _delay(rng: np.random.Generator, median: float, sigma: float) -> int:
    return int(rng.lognormal(math.log(median), sigma)) if sigma > 0 else int(median)


def _recent_post(view: WorldView, rng: np.random.Generator, ts: int) -> str | None:
    lo = bisect_left(view.post_ts, ts - RECENCY_WINDOW)
    hi = bisect_right(view.post_ts, ts)
    if hi <= lo:
        return None
    return view.human_posts[int(rng.integers(lo, hi))].post_id
