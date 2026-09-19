"""Increment 1 of human behavior: a circadian sleep schedule.

Real people are quiet at night in their own timezone. These tests pin that,
plus determinism now that events actually exist.
"""
from collections import Counter

from simulator.events import serialize_events
from simulator.world import generate_world, serialize_ground_truth
from tests.helpers import cached_world

N, DAYS = 200, 14


def _local_hour(world, event):
    tz = world.profiles[event.account_id].tz_offset_hours
    return ((event.sim_ts / 3600) + tz) % 24


def test_humans_generate_events_and_are_labeled_human():
    w = cached_world(7, N, DAYS)
    assert len(w.events) > 0
    assert len(w.ground_truth) == N
    assert set(w.ground_truth.values()) == {"human"}


def test_events_are_ordered_contiguous_and_in_horizon():
    w = generate_world(seed=7, n_humans=50, days=3)
    assert [e.event_id for e in w.events] == list(range(len(w.events)))
    ts = [e.sim_ts for e in w.events]
    assert ts == sorted(ts)
    assert 0 <= ts[0] and ts[-1] < 3 * 86400


def test_activity_is_low_in_the_dead_of_night_local_time():
    w = cached_world(7, N, DAYS)
    assert len({p.tz_offset_hours for p in w.profiles.values()}) > 1, "need mixed timezones"
    night = sum(1 for e in w.events if 2 <= _local_hour(w, e) < 5)
    assert night / len(w.events) < 0.03


def test_activity_is_concentrated_in_waking_hours():
    w = cached_world(7, N, DAYS)
    by_hour = Counter(int(_local_hour(w, e)) for e in w.events)
    busiest = max(by_hour.values())
    quietest = min(by_hour.get(h, 0) for h in range(24))
    assert busiest > 8 * max(quietest, 1)


def test_daily_volume_is_plausible():
    w = cached_world(7, N, DAYS)
    per_human_day = len(w.events) / (N * DAYS)
    assert 5 <= per_human_day <= 40


def test_same_seed_is_byte_identical_with_humans_and_other_seed_is_not():
    a = generate_world(seed=99, n_humans=40, days=3)
    b = generate_world(seed=99, n_humans=40, days=3)
    c = generate_world(seed=100, n_humans=40, days=3)
    assert len(a.events) > 0
    assert serialize_events(a.events) == serialize_events(b.events)
    assert serialize_ground_truth(a.ground_truth) == serialize_ground_truth(b.ground_truth)
    assert serialize_events(a.events) != serialize_events(c.events)


def test_adding_humans_does_not_change_existing_humans_events():
    """Per-account streams: human h-0000..h-0029 look identical in a bigger world."""
    small = generate_world(seed=5, n_humans=30, days=3)
    big = generate_world(seed=5, n_humans=60, days=3)

    def times(world, acct):
        return [e.sim_ts for e in world.events if e.account_id == acct]

    for acct in ("h-0000", "h-0013", "h-0029"):
        assert times(small, acct) == times(big, acct)
