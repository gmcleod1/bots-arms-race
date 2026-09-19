"""M3: the naive bot farm, and round 0 of the face-off.

A naive farm is one operator running a script with no evasion: many accounts posting
canned text on a fixed clock around the clock, all liking and following the same
targets in lockstep, all from one IP. This is the baseline the detector must beat
before the autonomous agent gets a turn.
"""
import numpy as np
import pytest

from detector.metrics import average_precision, confusion
from simulator.bots import NaiveFarmConfig, engagement_delivered, naive_farm
from simulator.events import serialize_events
from simulator.platform import Platform

BUDGET = 0.05


def _farm(world, seed=1, **overrides):
    platform = Platform(world)
    farm = naive_farm(platform, np.random.default_rng(seed), NaiveFarmConfig(**overrides))
    return platform, farm


@pytest.fixture(scope="session")
def round0(eval_world):
    platform, farm = _farm(eval_world)
    return {"platform": platform, "farm": farm, "events": platform.events_until(10**9)}


@pytest.fixture(scope="session")
def round0_verdicts(detector, round0):
    return detector.score(round0["events"])


def _flagged(verdicts):
    return {a for a, v in verdicts.items() if v.flagged}


def test_farm_accounts_are_bots_from_one_location(round0):
    farm, platform = round0["farm"], round0["platform"]
    assert len(farm.accounts) == 40
    assert {platform.ground_truth[a] for a in farm.accounts} == {"bot"}
    ips = {e.ip for e in round0["events"] if e.account_id in set(farm.accounts)}
    assert len(ips) == 1


def test_the_farm_posts_a_shared_pool_of_canned_text_on_a_fixed_clock(round0):
    farm = round0["farm"]
    mine = [e for e in round0["events"] if e.account_id == farm.accounts[0] and e.action == "post"]
    assert len({e.text for e in mine}) == len(farm.canned) == 5
    gaps = {b.sim_ts - a.sim_ts for a, b in zip(mine, mine[1:])}
    assert gaps == {1_800}


def test_every_bot_engages_every_target_post_and_the_farm_reports_what_it_delivered(round0):
    farm, events = round0["farm"], round0["events"]
    target_posts = [e for e in events if e.action == "post" and e.account_id in set(farm.targets)]
    likes = [e for e in events if e.action == "like" and e.account_id in set(farm.accounts)
             and e.target_id in {p.post_id for p in target_posts}]
    assert len(likes) == len(target_posts) * len(farm.accounts)
    delivered = engagement_delivered(events, farm)
    assert delivered == len(likes) + len(farm.accounts) * len(farm.targets)  # likes + follows
    assert delivered > 1_000


def test_the_farm_is_deterministic_and_never_alters_the_humans(eval_world):
    a, b = _farm(eval_world, seed=3), _farm(eval_world, seed=3)
    assert serialize_events(a[0].events_until(10**9)) == serialize_events(b[0].events_until(10**9))
    assert serialize_events(a[0].human_events) == serialize_events(eval_world.events)
    c = _farm(eval_world, seed=4)
    assert serialize_events(c[0].events_until(10**9)) != serialize_events(a[0].events_until(10**9))


def test_round0_the_detector_catches_the_whole_farm_within_its_false_positive_budget(
    round0, round0_verdicts
):
    truth = round0["platform"].ground_truth
    c = confusion(_flagged(round0_verdicts), truth)
    assert c.recall == 1.0
    assert c.fpr <= BUDGET + 0.025
    assert 0.0 < c.precision < 1.0  # the price of the stance, reported not hidden
    assert c.humans_flagged_per_10k <= (BUDGET + 0.025) * 10_000


def test_round0_the_farm_delivers_almost_nothing_once_flagged_accounts_are_removed(
    round0, round0_verdicts
):
    farm, events = round0["farm"], round0["events"]
    survivors = [a for a in farm.accounts if a not in _flagged(round0_verdicts)]
    total = engagement_delivered(events, farm)
    assert engagement_delivered(events, farm, by=survivors) / total <= 0.05


def test_round0_every_bot_trips_more_than_one_signal_family(detector, round0, round0_verdicts):
    """A naive farm is not a single point of failure for the attacker's luck: all three
    families see it, so patching or fooling any one of them would not save it."""
    alpha = detector.alpha
    family = {"t_": "timing", "c_": "coordination", "x_": "content"}
    for a in round0["farm"].accounts:
        fired = {family[f[:2]] for f, p in round0_verdicts[a].pvalues.items() if p <= alpha}
        assert len(fired) >= 2, (a, fired)


def test_ranking_quality_is_reported_alongside_the_flags(round0, round0_verdicts):
    scores = {a: v.score for a, v in round0_verdicts.items()}
    assert average_precision(scores, round0["platform"].ground_truth) > 0.6


def test_a_small_farm_is_harder_but_still_caught(detector, eval_world):
    platform, farm = _farm(eval_world, n_bots=8)
    flagged = detector.flag(platform.events_until(10**9))
    caught = len(set(farm.accounts) & flagged) / len(farm.accounts)
    assert caught >= 0.75
