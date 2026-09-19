"""Synthetic human behavior.

Built in increments, each with its own distribution test:
  1. circadian sleep schedule                  <- done
  2. bursty, heavy-tailed session timing       <- done
  3. community-structured social graph         <- done (social.py)
  4. organic content with natural typos        <- done (content.py)
  5. hotel-style shared-IP accounts            <- done (world.py assigns IP and timezone)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulator.content import WritingStyle, make_style

SECONDS_PER_DAY = 86_400
SLEEP_FLOOR = 0.02  # a sliver of night activity: insomnia, night shifts, other timezones

# Whole-hour UTC offsets and rough population weights: Americas, Europe, Asia-Pacific.
_TZ_OFFSETS = (-8, -7, -6, -5, 0, 1, 2, 5, 8, 9)
_TZ_WEIGHTS = (0.14, 0.05, 0.10, 0.20, 0.08, 0.14, 0.06, 0.08, 0.08, 0.07)


@dataclass(frozen=True)
class HumanProfile:
    tz_offset_hours: int
    wake_hour: float  # local hour the person usually wakes
    awake_hours: float  # length of the waking day
    session_rate_per_hour: float  # peak session starts/hour while awake
    mean_session_events: float  # average actions per session
    gap_median_seconds: float  # typical pause between actions inside a session
    gap_sigma: float  # log-scale spread of those pauses (heavy tail)
    community: int  # which friend group / interest cluster they belong to
    popularity: float  # relative pull as someone to follow (heavy-tailed)
    style: WritingStyle  # how they write: typo habits, casing, punctuation, emoji
    hotel: int | None  # shared-IP group (hotel, campus, carrier NAT), or None for their own IP


def make_profile(
    rng: np.random.Generator,
    n_communities: int = 8,
    hotel_fraction: float = 0.0,
    n_hotels: int = 3,
) -> HumanProfile:
    tz = int(rng.choice(_TZ_OFFSETS, p=_TZ_WEIGHTS))
    wake = float(np.clip(rng.normal(7.0, 1.2), 4.0, 11.0))
    sleep_len = float(np.clip(rng.normal(7.5, 1.0), 5.5, 10.0))
    session_rate = float(rng.lognormal(mean=math.log(0.3), sigma=0.5))
    session_events = float(max(1.5, rng.lognormal(mean=math.log(4.0), sigma=0.4)))
    gap_median = float(rng.lognormal(mean=math.log(25.0), sigma=0.4))
    gap_sigma = float(np.clip(rng.normal(1.1, 0.2), 0.6, 1.6))
    community = int(rng.integers(n_communities))
    popularity = float(rng.lognormal(mean=0.0, sigma=1.0))
    style = make_style(rng)
    in_hotel = bool(rng.random() < hotel_fraction)
    group = int(rng.integers(n_hotels))  # always drawn, so draw count is constant
    return HumanProfile(
        tz, wake, 24.0 - sleep_len, session_rate, session_events, gap_median, gap_sigma,
        community, popularity, style, group if in_hotel else None,
    )


def circadian_multiplier(profile: HumanProfile, local_hour: float) -> float:
    """Activity level in [SLEEP_FLOOR, 1]: near zero asleep, a smooth hump while awake."""
    since_wake = (local_hour - profile.wake_hour) % 24
    if since_wake >= profile.awake_hours:
        return SLEEP_FLOOR
    return 0.6 + 0.4 * math.sin(math.pi * since_wake / profile.awake_hours)


def generate_timestamps(profile: HumanProfile, rng: np.random.Generator, days: int) -> list[int]:
    """Sessions start by a circadian-thinned Poisson process; actions inside a session
    are separated by heavy-tailed (lognormal) pauses. Returns sorted sim_ts in seconds."""
    horizon = days * SECONDS_PER_DAY
    lam_max = profile.session_rate_per_hour / 3600.0  # per second; multiplier never exceeds 1
    out: list[int] = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / lam_max)
        if t >= horizon:
            break
        local_hour = ((t / 3600.0) + profile.tz_offset_hours) % 24
        if rng.random() >= circadian_multiplier(profile, local_hour):
            continue
        n_events = int(rng.geometric(1.0 / profile.mean_session_events))
        pauses = rng.lognormal(math.log(profile.gap_median_seconds), profile.gap_sigma, n_events - 1)
        for ts in t + np.concatenate(([0.0], np.cumsum(pauses))):
            if ts < horizon:
                out.append(int(ts))
    out.sort()
    return out
