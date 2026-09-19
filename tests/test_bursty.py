"""Increment 2 of human behavior: bursty, heavy-tailed session timing.

Goh-Barabasi burstiness B = (sigma - mu) / (sigma + mu) of inter-event times:
about 0 for a memoryless (Poisson) process, toward 1 for very bursty activity.
Human communication data typically sits well above 0. The metric lives here in
the test on purpose; the detector (M2) will own its own implementation.
"""
import numpy as np

from tests.helpers import cached_world

N, DAYS = 200, 14


def burstiness(gaps) -> float:
    gaps = np.asarray(gaps, dtype=float)
    mu, sigma = gaps.mean(), gaps.std()
    return float((sigma - mu) / (sigma + mu))


def _gaps_by_account(world):
    times: dict[str, list[int]] = {}
    for e in world.events:
        times.setdefault(e.account_id, []).append(e.sim_ts)
    return {a: np.diff(t) for a, t in times.items() if len(t) >= 30}


def test_metric_reads_near_zero_for_a_memoryless_process():
    rng = np.random.default_rng(0)
    assert abs(burstiness(rng.exponential(60.0, 5000))) < 0.05


def test_humans_are_bursty_not_poisson():
    w = cached_world(7, N, DAYS)
    b = [burstiness(g) for g in _gaps_by_account(w).values()]
    assert len(b) > 0.9 * N, "most humans should have enough events to measure"
    assert 0.35 <= float(np.median(b)) <= 0.9
    # 5th percentile, not the minimum: the minimum swings 0.06 to 0.17 across seeds
    # (one extreme profile), while the 5th percentile holds at 0.26 to 0.33.
    assert float(np.percentile(b, 5)) > 0.2, "few humans should look memoryless"


def test_activity_comes_in_sessions_separated_by_long_silences():
    w = cached_world(7, N, DAYS)
    gaps = np.concatenate(list(_gaps_by_account(w).values()))
    assert (gaps < 300).mean() > 0.5  # most events follow another within 5 minutes
    assert (gaps > 3600).mean() > 0.10  # but there are many silences over an hour


def test_within_session_gaps_are_heavy_tailed():
    w = cached_world(7, N, DAYS)
    gaps = np.concatenate(list(_gaps_by_account(w).values()))
    short = gaps[gaps < 3600]
    assert short.std() > short.mean()  # coefficient of variation > 1: heavier than exponential


def test_humans_differ_from_each_other():
    w = cached_world(7, N, DAYS)
    b = [burstiness(g) for g in _gaps_by_account(w).values()]
    assert np.std(b) > 0.03  # a population, not one template stamped 200 times
