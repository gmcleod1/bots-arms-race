"""Who humans follow and what they engage with.

Follow targets are weighted by popularity (heavy-tailed) and boosted when the
target is in the same community, which produces homophily. Likes and comments
prefer recent posts by accounts already followed, falling back to exploring
the wider feed.
"""
from __future__ import annotations

import math

import numpy as np

# (action, weight). "post" here is the intended kind; a like/comment/follow with
# nothing valid to target becomes a post instead.
# Follows are rare next to likes; at 10% a human filled their whole community within days.
ACTION_WEIGHTS = (("post", 0.25), ("like", 0.53), ("comment", 0.20), ("follow", 0.02))
HOMOPHILY = 20.0  # same-community accounts are this much likelier to be picked
RECENCY_WINDOW = 72 * 3600  # seconds a post stays engageable
RECENCY_TAU = 24 * 3600.0  # engagement decays with post age on this scale
EXPLORE_POOL = 300  # how many recent global posts the "explore" feed considers
FEED_DEPTH = 5  # newest posts per followee that can be engaged with (a feed shows the latest)


def draw_kinds(rng: np.random.Generator, n: int) -> list[str]:
    names, weights = zip(*ACTION_WEIGHTS)
    return [str(k) for k in rng.choice(names, size=n, p=weights)]


def pick_weights(i: int, communities: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Unnormalised chance that account i picks each account; never itself."""
    w = popularity * np.where(communities == communities[i], HOMOPHILY, 1.0)
    w[i] = 0.0
    return w


def initial_followees(rng: np.random.Generator, i: int, communities, popularity) -> list[int]:
    n = len(communities)
    if n < 2:
        return []
    k = int(np.clip(round(rng.lognormal(math.log(8.0), 0.6)), 2, min(60, n - 1)))
    w = pick_weights(i, communities, popularity)
    return [int(x) for x in rng.choice(n, size=k, replace=False, p=w / w.sum())]


def pick_follow(rng, i: int, communities, popularity, already: set[int]) -> int | None:
    w = pick_weights(i, communities, popularity)
    if already:
        w[list(already)] = 0.0
    total = w.sum()
    return int(rng.choice(len(w), p=w / total)) if total > 0 else None


def pick_post(rng, ts: int, me: str, followees: set[str], posts_by_author, recent) -> int | None:
    """Event id of a post to like or comment on, or None if nothing is engageable."""
    cutoff = ts - RECENCY_WINDOW
    cands: list[tuple[int, int]] = []  # (event_id, post_ts)
    for f in sorted(followees):  # sorted: set iteration order must not affect the draw
        seen = 0
        for post_ts, pid in reversed(posts_by_author.get(f, ())):
            if post_ts < cutoff or seen == FEED_DEPTH:
                break
            cands.append((pid, post_ts))
            seen += 1
    if cands:
        ages = ts - np.array([t for _, t in cands], dtype=float)
        w = np.exp(-ages / RECENCY_TAU)
        return cands[int(rng.choice(len(cands), p=w / w.sum()))][0]
    pool = [(pid, a) for pid, a, t in recent[-EXPLORE_POOL:] if a != me and t >= cutoff]
    return pool[int(rng.integers(len(pool)))][0] if pool else None
