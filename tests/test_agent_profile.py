"""The behavior knobs the red agent turns, and what they generate.

The agent chooses every value; nothing here decides a tactic. These tests only pin
that each knob does what its description says, so the agent's reasoning about
feedback is grounded in a tool surface that behaves as documented.
"""
import math
from collections import Counter

import numpy as np
import pytest

from agent.profile import Profile, ProfileError, expand
from agent.sandbox import WorldView
from simulator import content
from tests.helpers import cached_world

DAY = 86_400


@pytest.fixture(scope="module")
def view():
    return WorldView.from_world(cached_world(11, 60, 5))


def _gen(view, args=None, n=6, seed=0, max_actions=10**6):
    profile = Profile.from_args(args or {})
    accounts = [f"bot-{i:03d}" for i in range(n)]
    actions, dropped = expand(profile, accounts, view, np.random.default_rng(seed), max_actions)
    return actions, dropped, accounts


def _by_account(actions, kinds=("post", "like")):
    out = {}
    for a in actions:
        if a.kind in kinds:
            out.setdefault(a.account, []).append(a)
    return out


def test_defaults_and_unknown_or_out_of_range_arguments():
    assert Profile.from_args({}).timing.awake_hours == 16.0
    for bad in (
        {"timing": {"sessions_per_day": -1}},
        {"timing": {"awake_hours": 30}},
        {"content": {"text_mode": "telepathy"}},
        {"content": {"post_share": 1.5}},
        {"engagement": {"react_probability": 2}},
        {"timing": {"warp_speed": 9}},
        {"nonsense": {}},
    ):
        with pytest.raises(ProfileError):
            Profile.from_args(bad)


def test_fixed_interval_is_a_perfect_clock(view):
    actions, _, _ = _gen(view, {"timing": {"fixed_interval_s": 1800, "awake_hours": 24}})
    for acts in _by_account(actions).values():
        gaps = {b.ts - a.ts for a, b in zip(acts, acts[1:])}
        assert gaps <= {1799, 1800, 1801}  # int truncation only


def test_awake_hours_24_never_sleeps_and_a_short_window_is_respected(view):
    busy, _, _ = _gen(view, {"timing": {"awake_hours": 24, "sessions_per_day": 12}}, n=10)
    hours = Counter((a.ts // 3600) % 24 for a in busy if a.kind in ("post", "like"))
    p = np.array([hours.get(h, 0) for h in range(24)], dtype=float)
    p /= p.sum()
    assert -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(24) > 0.98

    short, _, _ = _gen(
        view, {"timing": {"awake_hours": 6, "wake_hour_utc": 10, "sessions_per_day": 12,
                          "events_per_session": 1}}, n=10)
    starts = [(a.ts // 3600) % 24 for a in short if a.kind in ("post", "like")]
    assert starts and all(10 <= h < 16 for h in starts)


def test_tz_spread_staggers_each_accounts_window(view):
    args = {"timing": {"awake_hours": 6, "wake_hour_utc": 12, "tz_spread_hours": 8,
                       "sessions_per_day": 10, "events_per_session": 1}}
    actions, _, _ = _gen(view, args, n=12)
    firsts = {min((a.ts // 3600) % 24 for a in acts) for acts in _by_account(actions).values()}
    assert len(firsts) >= 4  # windows are spread out, not one shared clock


def test_pause_sigma_controls_how_bursty_the_gaps_are(view):
    def cv(sigma):
        # rare sessions, so gaps under 5 minutes are (almost all) pauses inside a session
        actions, _, _ = _gen(view, {"timing": {"pause_sigma": sigma, "pause_median_s": 60,
                                               "events_per_session": 20, "awake_hours": 24,
                                               "sessions_per_day": 2}}, n=30)
        gaps = np.concatenate([np.diff([a.ts for a in acts]) for acts in _by_account(actions).values()])
        gaps = gaps[gaps < 300]  # only pauses inside a session; one long outlier would swamp the CV
        return gaps.std() / gaps.mean()

    assert cv(0.0) < 0.5
    assert cv(2.0) > cv(0.0) + 0.5


def test_canned_reuses_a_small_pool_fresh_is_unique_and_spun_varies_the_pool(view):
    def posts(args):
        actions, _, _ = _gen(view, {"content": {"post_share": 1.0, **args}, "timing": {"sessions_per_day": 8}}, n=6)
        return [a.text for a in actions if a.kind == "post"]

    canned = posts({"text_mode": "canned", "pool_size": 4})
    assert len(set(canned)) == 4
    fresh = posts({"text_mode": "fresh"})
    assert len(set(fresh)) / len(fresh) > 0.95
    spun = posts({"text_mode": "spun", "pool_size": 4, "typo_rate": 0.05})
    assert len(set(spun)) > 4


def test_typo_rate_injects_unfamiliar_words(view):
    vocab = content.vocabulary()

    def oov(rate):
        actions, _, _ = _gen(view, {"content": {"post_share": 1.0, "text_mode": "fresh",
                                               "typo_rate": rate}}, n=6)
        words = [w for a in actions if a.kind == "post" for w in a.text.lower().split()]
        return sum(1 for w in words if w.strip(".,!?;:") not in vocab) / len(words)

    assert oov(0.15) > oov(0.0) + 0.02


def test_post_share_zero_means_every_event_is_a_like_on_a_real_earlier_human_post(view):
    actions, _, _ = _gen(view, {"content": {"post_share": 0.0}, "timing": {"sessions_per_day": 8}}, n=4)
    likes = [a for a in actions if a.kind == "like"]
    assert likes and not any(a.kind == "post" for a in actions)
    posts = {p.post_id: p for p in view.human_posts}
    assert all(a.target in posts and posts[a.target].sim_ts <= a.ts for a in likes)


def test_reactions_follow_target_posts_with_the_chosen_probability_and_delay(view):
    target = next(iter({p.account_id for p in view.human_posts}))
    their_posts = [p for p in view.human_posts if p.account_id == target]
    eng = {"targets": [target], "react_probability": 1.0, "react_delay_median_s": 20,
           "react_delay_sigma": 0.0, "follow_targets": True}
    args = {"engagement": eng, "timing": {"sessions_per_day": 0.05}}
    actions, _, accounts = _gen(view, args, n=5)
    likes = {(a.account, a.target): a.ts for a in actions if a.kind == "like"}
    for p in their_posts:
        for acct in accounts:
            assert likes[(acct, p.post_id)] == p.sim_ts + 20
    assert {a.account for a in actions if a.kind == "follow" and a.target == target} == set(accounts)

    eng["react_probability"] = 0.5
    half, _, _ = _gen(view, args, n=40)
    n_likes = sum(1 for a in half if a.kind == "like" and a.target in {p.post_id for p in their_posts})
    assert 0.35 < n_likes / (40 * len(their_posts)) < 0.65


def test_a_slow_reaction_delay_spreads_likes_over_hours(view):
    target = next(iter({p.account_id for p in view.human_posts}))
    args = {"engagement": {"targets": [target], "react_probability": 1.0, "react_delay_median_s": 7200,
                           "react_delay_sigma": 1.0}, "timing": {"sessions_per_day": 0.05}}
    actions, _, _ = _gen(view, args, n=20)
    first_post = min((p for p in view.human_posts if p.account_id == target), key=lambda p: p.sim_ts)
    delays = [a.ts - first_post.sim_ts for a in actions if a.kind == "like" and a.target == first_post.post_id]
    assert len(delays) == 20 and np.std(delays) > 1_000


def test_expansion_is_deterministic_and_respects_the_action_cap(view):
    a1, _, _ = _gen(view, {"timing": {"awake_hours": 24, "sessions_per_day": 20}}, seed=3)
    a2, _, _ = _gen(view, {"timing": {"awake_hours": 24, "sessions_per_day": 20}}, seed=3)
    assert a1 == a2
    capped, dropped, _ = _gen(view, {"timing": {"awake_hours": 24, "sessions_per_day": 20}}, seed=3, max_actions=100)
    assert len(capped) == 100 and dropped == len(a1) - 100
    assert [a.ts for a in capped] == sorted(a.ts for a in capped)  # cap keeps the earliest
