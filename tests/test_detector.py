"""End-to-end detector: calibrate on clean history, evaluate on a fresh world.

The stance under test: the detector is tuned against an explicit false-positive
budget on humans, and everything it reports about performance is precision and
recall separately (never accuracy).
"""
import dataclasses

import numpy as np
import pytest

from detector.config import DetectorConfig, build_detector
from detector.detector import Detector
from detector.metrics import confusion
from tests.helpers import cached_world

BUDGET = 0.05


@pytest.fixture(scope="session")
def human_verdicts(detector, eval_world):
    return detector.score(eval_world.events)


@pytest.fixture(scope="session")
def bot_verdicts(detector, bot_world):
    return detector.score(bot_world["events"])


def _flagged(verdicts):
    return {a for a, v in verdicts.items() if v.flagged}


def test_using_an_unfitted_detector_is_an_error(eval_world):
    with pytest.raises(RuntimeError):
        Detector().score(eval_world.events[:10])


def test_reference_too_small_to_resolve_the_budget_is_refused():
    small = cached_world(7, 200, 14)
    with pytest.raises(ValueError, match="too small"):
        Detector(fpr_budget=0.01).fit(small.events)


def test_humans_only_baseline_stays_within_the_false_positive_budget(human_verdicts, eval_world):
    flagged = _flagged(human_verdicts)
    c = confusion(flagged, eval_world.ground_truth)
    assert c.tp == 0 and c.recall is None  # no bots exist, so recall is undefined
    assert c.precision == 0.0  # anything flagged is a human, so precision is defined and zero
    assert c.fpr <= BUDGET + 0.025  # budget plus about 3 standard errors at n = 800
    assert c.humans_flagged_per_10k == pytest.approx(c.fpr * 10_000)


def test_hotel_guests_are_not_flagged_more_than_anyone_else(human_verdicts, eval_world):
    hotel = {a for a, p in eval_world.profiles.items() if p.hotel is not None}
    assert len(hotel) > 80
    rate = len(hotel & _flagged(human_verdicts)) / len(hotel)
    assert rate <= BUDGET + 0.06  # about 3 standard errors at n = 130


def test_the_ip_field_has_no_influence(detector, eval_world, human_verdicts):
    swapped = [dataclasses.replace(e, ip="192.0.2.1") for e in eval_world.events]
    assert {a: v.score for a, v in detector.score(swapped).items()} == {
        a: v.score for a, v in human_verdicts.items()
    }


def test_scoring_is_deterministic(detector, eval_world, human_verdicts):
    again = detector.score(eval_world.events)
    assert {a: (v.score, v.flagged, v.top_feature) for a, v in again.items()} == {
        a: (v.score, v.flagged, v.top_feature) for a, v in human_verdicts.items()
    }


def test_every_flag_is_explained_by_a_feature_that_beat_its_share_of_the_budget(detector, bot_verdicts):
    alpha = BUDGET / len(detector.features)
    flagged = _flagged(bot_verdicts)
    assert flagged
    for v in (bot_verdicts[a] for a in flagged):
        assert v.top_feature in detector.features
        assert v.pvalues[v.top_feature] <= alpha
        assert v.score == pytest.approx(-np.log10(min(v.pvalues.values())))


@pytest.mark.parametrize(
    "group, minimum_recall, feature_prefix",
    [
        ("fixed", 1.0, "t_"),
        ("jitter", 1.0, "t_"),
        ("lockstep", 1.0, "c_"),
        ("copy", 0.9, "x_"),
        ("typo_copy", 0.8, "x_"),
        ("novel_typo", 1.0, "x_shared_oov"),
    ],
)
def test_each_naive_pattern_is_caught_and_attributed_to_the_right_signal(
    group, minimum_recall, feature_prefix, bot_world, bot_verdicts
):
    accts = bot_world["groups"][group]
    caught = [a for a in accts if a in bot_verdicts and bot_verdicts[a].flagged]
    assert len(caught) / len(accts) >= minimum_recall
    assert all(bot_verdicts[a].top_feature.startswith(feature_prefix) for a in caught)


def test_precision_and_recall_with_bots_present(bot_world, bot_verdicts):
    truth = bot_world["platform"].ground_truth
    c = confusion(_flagged(bot_verdicts), truth)
    assert c.recall >= 0.8
    assert c.fpr <= BUDGET + 0.025
    # Reported separately, with the cost of the stance made visible:
    assert c.precision is not None and 0.0 < c.precision <= 1.0
    assert c.humans_flagged_per_10k <= (BUDGET + 0.025) * 10_000


def test_known_gap_same_targets_hours_apart_evades_the_default_detector(bot_world, bot_verdicts):
    """Documented blind spot of the DEFAULT detector: the 60 s co-action window misses slow lockstep.
    The opt-in `target_overlap` feature closes it (next test) at a small cost, so it ships off."""
    accts = bot_world["groups"]["slow_lockstep"]
    caught = [a for a in accts if a in bot_verdicts and bot_verdicts[a].flagged]
    assert len(caught) / len(accts) <= 0.5


@pytest.fixture(scope="session")
def overlap_verdicts(ref_world, bot_world):
    det = build_detector(DetectorConfig(target_overlap=True), ref_world.events)
    return det, det.score(bot_world["events"])


def test_target_overlap_closes_the_slow_lockstep_gap_and_gets_the_credit(bot_world, overlap_verdicts):
    """0/12 caught by default, 12/12 with `target_overlap=true` (which ignores time). Every catch is
    attributed to it: the window features still cannot see these bots."""
    det, verdicts = overlap_verdicts
    accts = bot_world["groups"]["slow_lockstep"]
    assert all(verdicts[a].flagged for a in accts)
    for a in accts:
        fired = {f for f, p in verdicts[a].pvalues.items() if p <= det.alpha}
        assert "c_target_overlap" in fired


def test_target_overlap_keeps_the_human_false_positive_rate_near_budget(bot_world, overlap_verdicts):
    _, verdicts = overlap_verdicts
    c = confusion(_flagged(verdicts), bot_world["platform"].ground_truth)
    assert c.fpr <= BUDGET + 0.025 and c.recall >= 0.8


def test_accounts_without_enough_evidence_get_no_verdict(bot_world, bot_verdicts):
    # 12-event lockstep bots have no timing or content evidence, only coordination
    v = bot_verdicts[bot_world["groups"]["lockstep"][0]]
    assert set(v.pvalues) == {"c_coaction_max", "c_lockstep_degree"}
