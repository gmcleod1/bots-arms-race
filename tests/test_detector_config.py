"""The detector as data: configs, patches, caching, and a patch that actually works.

A live patch is a new DetectorConfig. These tests pin that configs validate, round-trip
through JSON (so every version can be recorded and rebuilt), reuse cached reference work,
stay honestly calibrated whatever the setting, and that the budget is a real lever with a
measurable price in humans flagged. They also record a measured NEGATIVE result: widening
the co-action window does not reliably close the slow-lockstep blind spot (the opt-in
`target_overlap` feature does, see test_detector.py).
"""
import numpy as np
import pytest

from detector.config import ConfigError, DetectorConfig, build_detector
from detector.detector import Detector
from detector.metrics import confusion
from simulator.platform import Platform
from tests import botkit


def _flagged(det, events):
    return {a for a, v in det.score(events).items() if v.flagged}


def test_default_config_reproduces_the_plain_detector(duel_ref, duel_eval):
    plain = Detector(fpr_budget=0.05)
    plain.fit(duel_ref.events)
    built = build_detector(DetectorConfig(), duel_ref.events)
    a, b = plain.score(duel_eval.events), built.score(duel_eval.events)
    assert {k: v.score for k, v in a.items()} == {k: v.score for k, v in b.items()}


def test_configs_validate_and_round_trip_through_json():
    base = DetectorConfig()
    assert DetectorConfig.from_json(base.to_json()) == base
    patched = base.patched(coordination_window=3_600, disabled_features=["x_shared_oov"])
    assert DetectorConfig.from_json(patched.to_json()) == patched
    assert base.diff(patched) == {"coordination_window": [60, 3_600], "disabled_features": [[], ["x_shared_oov"]]}
    for bad in ({"warp": 1}, {"fpr_budget": 0}, {"fpr_budget": 0.9}, {"coordination_window": 0},
                {"content_jaccard": 0.1}, {"timing_min_events": 0}, {"disabled_features": ["nope"]},
                {"disabled_features": ["t_regularity", "t_no_sleep", "c_coaction_max", "c_lockstep_degree",
                                       "x_dup_rate", "x_shared_oov"]},
                {"target_overlap": "yes"}):
        with pytest.raises(ConfigError):
            base.patched(**bad)


def test_disabling_a_feature_shares_the_budget_among_the_rest(duel_ref, duel_eval):
    full = build_detector(DetectorConfig(), duel_ref.events)
    less = build_detector(DetectorConfig(disabled_features=("x_shared_oov",)), duel_ref.events)
    assert len(full.features) == 6 and len(less.features) == 5
    assert less.alpha == pytest.approx(0.05 / 5) and "x_shared_oov" not in less.features
    assert all("x_shared_oov" not in v.pvalues for v in less.score(duel_eval.events).values())


def test_a_patch_that_touches_one_signal_reuses_the_others_from_cache(duel_ref):
    cache = {}
    v1 = build_detector(DetectorConfig(), duel_ref.events, cache)
    v2 = build_detector(DetectorConfig(coordination_window=3_600), duel_ref.events, cache)
    timing1, coord1, content1 = v1.signals
    timing2, coord2, content2 = v2.signals
    assert timing2 is timing1 and content2 is content1  # reused, not recomputed
    assert coord2 is not coord1 and coord2.window == 3_600


def test_a_fully_disabled_signal_is_not_built_or_run(duel_ref):
    det = build_detector(DetectorConfig(disabled_features=("x_dup_rate", "x_shared_oov")), duel_ref.events)
    assert [s.name for s in det.signals] == ["timing", "coordination"]


def _slow_lockstep_world(duel_eval):
    platform = Platform(duel_eval)
    script = botkit.Script(platform)
    bots = botkit.lockstep_bots(script, np.random.default_rng(4), platform, n=12, k=12,
                                spread=7_200, prefix="sl")
    script.run()
    return bots, platform.events_until(10**9), platform.ground_truth


def test_recalibrating_at_any_window_keeps_the_human_false_positive_rate_near_budget(duel_ref, duel_eval):
    """Widening the co-action window is NOT a reliable fix for slow lockstep (measured: 2/12 bots at
    60 s, 0/12 at 15 to 30 min, 7/12 at 1 h, 0/12 at 3 h) because ordinary human co-action grows
    faster than the bots' signal. What a patch can promise is honest calibration: whatever the
    window, the detector is re-fitted on clean history at that window, so humans stay within budget."""
    bots, events, truth = _slow_lockstep_world(duel_eval)
    cache = {}
    for window in (60, 900, 3_600):
        det = build_detector(DetectorConfig(coordination_window=window), duel_ref.events, cache)
        assert confusion(_flagged(det, events), truth).fpr <= 0.05 + 0.04, window


def test_raising_the_budget_is_a_real_lever_and_it_costs_humans(duel_ref, duel_eval):
    """The stated stance in numbers: accept more false positives to catch more bots.

    Measured on slow lockstep with `target_overlap` off (the default). With it on the budget buys
    nothing here: 12/12 at 5%. Every other pattern tried (24/7 jitter, partial-overlap bots) also
    either sits at the p-value floor or deep in the human bulk, so no natural marginal bot is left."""
    bots, events, truth = _slow_lockstep_world(duel_eval)
    cache = {}
    tight = build_detector(DetectorConfig(fpr_budget=0.05), duel_ref.events, cache)
    loose = build_detector(DetectorConfig(fpr_budget=0.15), duel_ref.events, cache)
    c_tight, c_loose = (confusion(_flagged(d, events), truth) for d in (tight, loose))
    assert c_loose.tp > c_tight.tp  # more bots caught
    assert c_loose.fp > c_tight.fp  # and more humans flagged: the price
    assert c_loose.fpr <= 0.15 + 0.05


def test_target_overlap_is_off_by_default_and_a_patch_turns_it_on(duel_ref, duel_eval):
    """Off by default: the default detector is exactly v1 (six features, budget/6). A record written
    before the feature existed has no `target_overlap` key and rebuilds that same detector, which is
    what keeps rounds recorded against v1 replayable. Turning it on adds one feature and shrinks
    every feature's share of the budget to budget/7."""
    old_record = {k: v for k, v in DetectorConfig().to_json().items() if k != "target_overlap"}
    assert DetectorConfig.from_json(old_record) == DetectorConfig() and not DetectorConfig().target_overlap
    cache = {}
    off = build_detector(DetectorConfig(), duel_ref.events, cache)
    on = build_detector(DetectorConfig().patched(target_overlap=True), duel_ref.events, cache)
    assert "c_target_overlap" not in off.features and len(off.features) == 6
    assert "c_target_overlap" in on.features and len(on.features) == 7
    assert off.alpha == pytest.approx(0.05 / 6) and on.alpha == pytest.approx(0.05 / 7)
    assert all("c_target_overlap" not in v.pvalues for v in off.score(duel_eval.events).values())
    assert DetectorConfig().diff(DetectorConfig().patched(target_overlap=True)) == {"target_overlap": [False, True]}
    timing_off, _, content_off = off.signals
    timing_on, _, content_on = on.signals
    assert timing_on is timing_off and content_on is content_off  # only coordination is recomputed
