"""Determinism and leakage guards for the simulator foundation.

The whole project rests on these: a round can only be replayed and a
measurement can only be trusted if the same seed gives the same world and
labels never travel in the event stream the detector reads.
"""
import dataclasses

from simulator.events import ACTIONS, Event, serialize_events
from simulator.rng import SeedTree
from simulator.world import generate_world, serialize_ground_truth


def test_same_seed_gives_byte_identical_event_log():
    a = generate_world(seed=1234)
    b = generate_world(seed=1234)
    assert serialize_events(a.events) == serialize_events(b.events)
    assert serialize_ground_truth(a.ground_truth) == serialize_ground_truth(b.ground_truth)


def test_account_stream_ignores_which_other_accounts_exist():
    """Adding an account must not shift another account's random draws."""
    alone = SeedTree(42).for_account("acct-A").random(5)
    tree = SeedTree(42)
    tree.for_account("acct-B").random(100)  # draw from a neighbour first
    with_neighbour = tree.for_account("acct-A").random(5)
    assert list(alone) == list(with_neighbour)


def test_streams_differ_by_account_and_by_seed():
    assert list(SeedTree(42).for_account("acct-A").random(5)) != list(
        SeedTree(42).for_account("acct-B").random(5)
    )
    assert list(SeedTree(42).for_account("acct-A").random(5)) != list(
        SeedTree(43).for_account("acct-A").random(5)
    )


def test_event_has_no_label_field():
    """Leakage guard: ground truth lives in a separate file, never on an event."""
    fields = {f.name for f in dataclasses.fields(Event)}
    assert fields == {"event_id", "sim_ts", "account_id", "action", "target_id", "text", "ip"}
    assert not {"label", "is_bot", "kind", "ground_truth"} & fields


def test_event_is_immutable_and_validates_action():
    e = Event(0, 0, "acct-A", "post", None, "hello", "10.0.0.1")
    try:
        e.sim_ts = 5  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Event must be frozen")
    assert "post" in ACTIONS
    try:
        Event(1, 0, "acct-A", "teleport", None, None, "10.0.0.1")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown action must be rejected")
