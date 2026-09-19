"""The platform API the red agent will call (M4).

Humans are pre-generated; bots act through this surface only. It validates
everything a real platform would (targets exist, no time travel, no self-
engagement), stamps bot accounts as bots in ground truth, and never lets bot
activity alter the human event log.
"""
import pytest

from simulator.events import serialize_events
from simulator.platform import Platform, PlatformError
from simulator.world import generate_world

DAY = 86_400


@pytest.fixture
def world():
    return generate_world(seed=11, n_humans=40, days=3)


@pytest.fixture
def platform(world):
    return Platform(world)


def _human_post(world, before=None):
    return next(e for e in world.events if e.action == "post" and (before is None or e.sim_ts <= before))


def test_created_accounts_are_labeled_bot_and_ids_cannot_collide(platform):
    platform.create_account("b-001", ip="198.51.100.7", sim_ts=100)
    assert platform.ground_truth["b-001"] == "bot"
    with pytest.raises(PlatformError):
        platform.create_account("b-001", ip="198.51.100.7", sim_ts=100)
    with pytest.raises(PlatformError):
        platform.create_account("h-0000", ip="198.51.100.7", sim_ts=100)


def test_bot_actions_become_valid_events_with_fresh_ids(world, platform):
    hp = _human_post(world)
    t = hp.sim_ts + 60
    platform.create_account("b-001", ip="198.51.100.7", sim_ts=t)
    platform.create_account("b-002", ip="198.51.100.7", sim_ts=t)
    own = platform.post("b-001", t, "hello there")
    like = platform.like("b-002", t + 1, own.post_id)
    cmt = platform.comment("b-002", t + 2, hp.post_id, "nice one")
    fol = platform.follow("b-002", t + 3, "b-001")
    assert own.action == "post" and own.target_id is None
    assert (like.action, like.target_id) == ("like", own.post_id)
    assert (cmt.action, cmt.target_id, cmt.text) == ("comment", hp.post_id, "nice one")
    assert (fol.action, fol.target_id) == ("follow", "b-001")
    ids = [own.event_id, like.event_id, cmt.event_id, fol.event_id]
    assert len(set(ids)) == 4 and min(ids) >= len(world.events)
    assert all(e.ip == "198.51.100.7" for e in (own, like, cmt, fol))


def test_rejections(world, platform):
    hp = _human_post(world)
    t = hp.sim_ts + 60
    platform.create_account("b-001", ip="198.51.100.7", sim_ts=t)
    own = platform.post("b-001", t + 10, "hello there")
    bad = [
        lambda: platform.post("ghost", t + 20, "x"),  # unknown account
        lambda: platform.post("b-001", t + 5, "x"),  # time travel behind the clock
        lambda: platform.post("b-001", t + 20, ""),  # empty text
        lambda: platform.like("b-001", t + 20, own.post_id),  # self-engagement
        lambda: platform.like("b-001", t + 20, "p-99999999"),  # no such post
        lambda: platform.comment("b-001", t + 20, hp.post_id, ""),
        lambda: platform.follow("b-001", t + 20, "b-001"),  # self-follow
        lambda: platform.follow("b-001", t + 20, "nobody"),
    ]
    for call in bad:
        with pytest.raises(PlatformError):
            call()
    platform.follow("b-001", t + 30, "h-0001")
    with pytest.raises(PlatformError):
        platform.follow("b-001", t + 31, "h-0001")  # already following


def test_cannot_act_before_account_creation_or_on_posts_from_the_future(world, platform):
    late = max((e for e in world.events if e.action == "post"), key=lambda e: e.sim_ts)
    platform.create_account("b-001", ip="198.51.100.7", sim_ts=1_000)
    with pytest.raises(PlatformError):
        platform.post("b-001", 500, "before I existed")
    with pytest.raises(PlatformError):
        platform.like("b-001", 2_000, late.post_id)  # that post does not exist yet at t=2000


def test_bot_activity_never_alters_the_human_log(world, platform):
    before = serialize_events(world.events)
    platform.create_account("b-001", ip="198.51.100.7", sim_ts=10)
    for i in range(20):
        platform.post("b-001", 20 + i, f"bot post {i}")
    assert serialize_events(platform.human_events) == before


def test_stream_is_time_ordered_and_merges_both(world, platform):
    platform.create_account("b-001", ip="198.51.100.7", sim_ts=10)
    platform.post("b-001", 50, "early bot post")
    stream = platform.events_until(DAY)
    keys = [(e.sim_ts, e.event_id) for e in stream]
    assert keys == sorted(keys)
    assert all(e.sim_ts <= DAY for e in stream)
    assert any(e.account_id == "b-001" for e in stream)
    assert any(e.account_id.startswith("h-") for e in stream)


def _scripted_run(seed):
    platform = Platform(generate_world(seed=seed, n_humans=40, days=3))
    hp = _human_post(platform.world)
    t = hp.sim_ts + 60
    for i in range(3):
        platform.create_account(f"b-{i}", ip="198.51.100.7", sim_ts=t)
    for i in range(3):  # actions must be issued in time order: the clock only moves forward
        platform.post(f"b-{i}", t + i, f"same text {i}")
    for i in range(3):
        platform.like(f"b-{i}", t + 10 + i, hp.post_id)
    return serialize_events(platform.events_until(3 * DAY)), dict(platform.ground_truth)


def test_replaying_the_same_actions_is_byte_identical():
    a, b = _scripted_run(11), _scripted_run(11)
    assert a == b
    assert a != _scripted_run(12)
