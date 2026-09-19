"""Increment 5 of human behavior: hotel-style shared-IP accounts.

Some real people legitimately share one IP (a hotel, a campus, carrier-grade
NAT). They are co-located, so they share a timezone and a daily rhythm, but
they are strangers: different interests, no social ties, no lockstep. This is
the trap for anyone who reaches for IP blocking or a naive same-time signal.
"""
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from simulator.world import HOTEL_FRACTION
from tests.helpers import cached_world

N, DAYS = 200, 14


def _world():
    return cached_world(7, N, DAYS)


def _ip_groups(w):
    ip_of = {}
    for e in w.events:
        ip_of.setdefault(e.account_id, e.ip)
    groups = defaultdict(list)
    for acct, ip in ip_of.items():
        groups[ip].append(acct)
    return groups


def _hotels(w):
    return {ip: accts for ip, accts in _ip_groups(w).items() if len(accts) > 1}


def test_a_realistic_share_of_humans_sit_behind_shared_ips():
    w = _world()
    behind = sum(len(a) for a in _hotels(w).values())
    assert 0.6 * HOTEL_FRACTION <= behind / N <= 1.5 * HOTEL_FRACTION
    assert len(_hotels(w)) >= 2
    assert max(len(a) for a in _hotels(w).values()) >= 8


def test_everyone_else_has_their_own_ip():
    w = _world()
    sizes = Counter(len(a) for a in _ip_groups(w).values())
    assert sizes[1] > 0.7 * N


def test_hotel_guests_are_all_human_and_share_a_timezone():
    w = _world()
    for ip, accts in _hotels(w).items():
        assert {w.ground_truth[a] for a in accts} == {"human"}
        assert len({w.profiles[a].tz_offset_hours for a in accts}) == 1, ip


def test_hotel_guests_are_strangers_not_a_community():
    w = _world()
    for ip, accts in _hotels(w).items():
        assert len({w.profiles[a].community for a in accts}) >= 3, ip

    def density(pairs):
        pairs = list(pairs)
        edges = sum((b in w.followees[a]) + (a in w.followees[b]) for a, b in pairs)
        return edges / len(pairs)

    overall = density(combinations(sorted(w.ground_truth), 2))
    inside = density(p for accts in _hotels(w).values() for p in combinations(sorted(accts), 2))
    assert inside <= 1.5 * overall  # sharing an IP creates no follow ties


def test_hotel_guests_do_not_act_in_lockstep():
    w = _world()
    times = defaultdict(list)
    for e in w.events:
        times[e.account_id].append(e.sim_ts)
    near = total = 0
    for accts in _hotels(w).values():
        for a, b in combinations(sorted(accts), 2):
            tb = np.array(times[b])
            for t in times[a]:
                total += 1
                i = np.searchsorted(tb, t)
                gaps = [abs(tb[j] - t) for j in (i - 1, i) if 0 <= j < len(tb)]
                near += bool(gaps) and min(gaps) <= 2
    assert total > 10_000
    assert near / total < 0.02  # shared timezone gives shared daytime, not shared seconds


def test_hotel_guests_behave_like_everyone_else_in_volume():
    w = _world()
    per = Counter(e.account_id for e in w.events)
    guests = {a for accts in _hotels(w).values() for a in accts}
    g = np.mean([per[a] for a in guests])
    o = np.mean([per[a] for a in w.ground_truth if a not in guests])
    assert 0.7 <= g / o <= 1.4
