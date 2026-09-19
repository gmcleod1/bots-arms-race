"""Each signal fires on its own pattern, and stays quiet for humans.

Raw values only (higher = more bot-like). Calibration into p-values and the
false-positive budget are tested in test_detector.py.
"""
import numpy as np
import pytest

from simulator.platform import Platform
from tests import botkit


def _vals(features, feature, accts):
    return [features[feature][a] for a in accts if a in features[feature]]


def _human_pct(human_features, feature, q):
    return float(np.percentile(list(human_features[feature].values()), q))


# ---- timing -------------------------------------------------------------

def test_timing_humans_are_bursty_and_sleep(human_features):
    reg = list(human_features["t_regularity"].values())
    ent = list(human_features["t_no_sleep"].values())
    assert len(reg) > 700  # nearly everyone has enough events to judge
    assert -0.65 <= float(np.median(reg)) <= -0.3
    assert float(np.median(ent)) < 0.95


def test_fixed_interval_bots_are_perfectly_regular(bot_world, bot_features, human_features):
    v = _vals(bot_features, "t_regularity", bot_world["groups"]["fixed"])
    assert len(v) == 6 and min(v) > 0.95
    assert min(v) > max(human_features["t_regularity"].values())


def test_jittered_bots_are_caught_by_regularity_and_never_sleeping(bot_world, bot_features, human_features):
    reg = _vals(bot_features, "t_regularity", bot_world["groups"]["jitter"])
    ent = _vals(bot_features, "t_no_sleep", bot_world["groups"]["jitter"])
    assert min(reg) > _human_pct(human_features, "t_regularity", 99.5)
    assert min(ent) > _human_pct(human_features, "t_no_sleep", 99.5)


def test_timing_needs_enough_events(bot_world, bot_features):
    sparse = bot_world["groups"]["lockstep"]  # 12 events each
    assert not any(a in bot_features["t_regularity"] for a in sparse)


# ---- coordination -------------------------------------------------------

def test_lockstep_bots_form_a_clique_humans_do_not(bot_world, bot_features, human_features):
    deg = _vals(bot_features, "c_lockstep_degree", bot_world["groups"]["lockstep"])
    co = _vals(bot_features, "c_coaction_max", bot_world["groups"]["lockstep"])
    assert min(deg) >= 11 and min(co) >= 10
    assert min(deg) > max(human_features["c_lockstep_degree"].values())
    assert min(co) > max(human_features["c_coaction_max"].values())


def test_same_targets_hours_apart_is_not_lockstep(bot_world, bot_features):
    slow = _vals(bot_features, "c_lockstep_degree", bot_world["groups"]["slow_lockstep"])
    fast = _vals(bot_features, "c_lockstep_degree", bot_world["groups"]["lockstep"])
    assert max(slow) < 5  # inside the 60 s window only by chance
    assert max(slow) < min(fast)


# ---- content ------------------------------------------------------------

def test_copy_paste_bots_repost_earlier_text(bot_world, bot_features, human_features):
    v = _vals(bot_features, "x_dup_rate", bot_world["groups"]["copy"])
    assert float(np.median(v)) >= 0.85
    assert float(np.median(v)) > _human_pct(human_features, "x_dup_rate", 99)


def test_small_typo_mutations_do_not_hide_a_copy(bot_world, bot_features):
    v = _vals(bot_features, "x_dup_rate", bot_world["groups"]["typo_copy"])
    assert float(np.median(v)) >= 0.7


def test_a_shared_novel_misspelling_is_a_fingerprint(bot_world, bot_features, human_features):
    v = _vals(bot_features, "x_shared_oov", bot_world["groups"]["novel_typo"])
    assert min(v) >= 0.9
    assert min(v) > max(human_features["x_shared_oov"].values())


def test_content_signal_ignores_unique_text(bot_world, bot_features, human_features):
    """Timing bots write unique posts: content features stay in the human range."""
    for group in ("fixed", "jitter"):
        dup = _vals(bot_features, "x_dup_rate", bot_world["groups"][group])
        assert max(dup) <= _human_pct(human_features, "x_dup_rate", 99.5)


def test_the_original_author_is_not_blamed_for_being_copied(bot_world, fitted_signals, eval_world, human_features):
    """Isolated on purpose: in the full bot world thousands of extra bot posts could, by
    chance alone, resemble something a human wrote later. Here only the copiers are added."""
    content = next(s for s in fitted_signals if s.name == "content")
    victim = bot_world["victim"]
    platform = Platform(eval_world)
    script = botkit.Script(platform)
    posts = [e for e in eval_world.events if e.account_id == victim and e.action == "post"
             and len(e.text.split()) >= 6][:8]
    copiers = botkit.copy_human_posts(script, np.random.default_rng(0), posts)
    script.run()
    feats = content.compute(platform.events_until(10**9))["x_dup_rate"]
    assert feats[victim] == pytest.approx(human_features["x_dup_rate"][victim])
    assert min(feats[a] for a in copiers) >= 0.8


def test_signals_are_deterministic(fitted_signals, eval_world):
    for s in fitted_signals:
        assert s.compute(eval_world.events) == s.compute(eval_world.events)
