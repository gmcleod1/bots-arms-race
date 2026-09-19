"""Increment 3 of human behavior: a community-structured social graph.

Engagement now has real targets: likes and comments point at earlier posts,
follows point at real accounts. People mostly engage inside their community
and a few accounts are far more popular than the rest. That structure is what
the coordination detector (M2) has to see through.
"""
from collections import Counter

from simulator.world import generate_world
from tests.helpers import cached_world

N, DAYS = 200, 14


def _world():
    return cached_world(7, N, DAYS)


def _posts(w):
    return {f"p-{e.event_id}": e for e in w.events if e.action == "post"}


def test_like_and_comment_targets_are_real_earlier_posts_by_someone_else():
    w = _world()
    posts = _posts(w)
    checked = 0
    for e in w.events:
        if e.action in ("like", "comment"):
            assert e.target_id in posts, e
            assert posts[e.target_id].sim_ts <= e.sim_ts
            assert posts[e.target_id].account_id != e.account_id
            checked += 1
    assert checked > 1000


def test_follow_targets_are_real_other_accounts_and_never_repeat():
    w = _world()
    seen = set()
    n = 0
    for e in w.events:
        if e.action == "follow":
            assert e.target_id in w.ground_truth and e.target_id != e.account_id
            assert (e.account_id, e.target_id) not in seen
            assert e.target_id in w.followees[e.account_id]
            seen.add((e.account_id, e.target_id))
            n += 1
    assert n > 100


def test_plain_posts_have_no_target():
    assert all(e.target_id is None for e in _world().events if e.action == "post")


def test_everyone_starts_with_followees():
    w = _world()
    assert min(len(f) for f in w.followees.values()) >= 2
    assert all(a not in f for a, f in w.followees.items())


def test_engagement_is_concentrated_in_own_community():
    w = _world()
    posts = _posts(w)
    same = total = 0
    for e in w.events:
        if e.action in ("like", "comment"):
            author = posts[e.target_id].account_id
        elif e.action == "follow":
            author = e.target_id
        else:
            continue
        total += 1
        same += w.profiles[e.account_id].community == w.profiles[author].community
    share = same / total
    assert 0.55 <= share <= 0.95  # random mixing across 8 communities would be about 0.125


def test_popularity_is_heavy_tailed():
    w = _world()
    posts = _posts(w)
    likes = Counter(posts[e.target_id].account_id for e in w.events if e.action == "like")
    counts = sorted((likes.get(a, 0) for a in w.ground_truth), reverse=True)
    top_decile = sum(counts[: N // 10]) / sum(counts)
    assert top_decile >= 0.2  # uniform popularity would give 0.1


def test_action_mix_is_mostly_engagement():
    w = _world()
    mix = Counter(e.action for e in w.events)
    n = len(w.events)
    assert 0.15 <= mix["post"] / n <= 0.40
    assert mix["like"] > mix["comment"] > mix["follow"] > 0


def test_graph_generation_is_deterministic():
    a = generate_world(seed=3, n_humans=60, days=5)
    b = generate_world(seed=3, n_humans=60, days=5)
    assert a.followees == b.followees
    assert [(e.action, e.target_id) for e in a.events] == [(e.action, e.target_id) for e in b.events]
