"""World generation.

`World` holds simulator-internal state (profiles, graph, ground truth). The
detector and the red agent never receive a `World`, only the event stream.
Ground truth stays here, separate from events.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np

from simulator import content, social
from simulator.events import Event, post_id_of
from simulator.humans import HumanProfile, generate_timestamps, make_profile
from simulator.rng import SeedTree

N_COMMUNITIES = 8  # fixed, so a human's profile never depends on population size
HOTEL_FRACTION = 0.15  # share of humans who sit behind a shared IP
HOTEL_TZ = (-5, 1, 9)  # each hotel is in one place, so all its guests share a timezone
N_HOTELS = len(HOTEL_TZ)


@dataclass
class World:
    seed: int
    events: list[Event] = field(default_factory=list)
    ground_truth: dict[str, str] = field(default_factory=dict)  # account_id -> "human" | "bot"
    profiles: dict[str, HumanProfile] = field(default_factory=dict)  # internal, test-visible only
    followees: dict[str, set[str]] = field(default_factory=dict)  # final follow graph


def human_id(i: int) -> str:
    return f"h-{i:04d}"


def generate_world(seed: int, n_humans: int = 0, days: int = 7) -> World:
    tree = SeedTree(seed)  # every random draw must go through this tree
    world = World(seed=seed)
    ids = [human_id(i) for i in range(n_humans)]
    kinds: dict[str, list[str]] = {}
    stamped: list[tuple[int, str, int]] = []  # (sim_ts, account_id, index into kinds[account])
    for acct in ids:
        rng = tree.for_account(acct)
        profile = make_profile(rng, N_COMMUNITIES, HOTEL_FRACTION, N_HOTELS)  # fixed draw order: profile, times, kinds
        if profile.hotel is not None:  # co-located strangers: same clock, different people
            profile = replace(profile, tz_offset_hours=HOTEL_TZ[profile.hotel])
        world.profiles[acct] = profile
        world.ground_truth[acct] = "human"
        times = generate_timestamps(profile, rng, days)
        kinds[acct] = social.draw_kinds(rng, len(times))
        stamped.extend((ts, acct, j) for j, ts in enumerate(times))
    stamped.sort()  # (sim_ts, account_id, j): a total order, independent of loop order

    communities = np.array([world.profiles[a].community for a in ids])
    popularity = np.array([world.profiles[a].popularity for a in ids])
    index = {a: i for i, a in enumerate(ids)}
    following: dict[str, set[int]] = {}
    for acct in ids:
        i = index[acct]
        graph_rng = tree.for_account(f"{acct}#graph")
        following[acct] = set(social.initial_followees(graph_rng, i, communities, popularity))
    engage_rngs = {a: tree.for_account(f"{a}#engage") for a in ids}
    text_rngs = {a: tree.for_account(f"{a}#text") for a in ids}

    ip = {
        a: f"203.0.113.{world.profiles[a].hotel + 1}"  # TEST-NET-3: obviously synthetic, shared
        if world.profiles[a].hotel is not None
        else f"10.{(i >> 8) & 255}.{i & 255}.1"
        for i, a in enumerate(ids)
    }
    posts_by_author: dict[str, list[tuple[int, int]]] = {}  # author -> [(ts, event_id)]
    recent: list[tuple[int, str, int]] = []  # (event_id, author, ts), global, time-ordered
    for n, (ts, acct, j) in enumerate(stamped):
        rng = engage_rngs[acct]
        i = index[acct]
        action, target = kinds[acct][j], None
        if action in ("like", "comment"):
            names = {ids[x] for x in following[acct]}
            pid = social.pick_post(rng, ts, acct, names, posts_by_author, recent)
            if pid is None:
                action = "post"
            else:
                target = post_id_of(pid)
        elif action == "follow":
            pick = social.pick_follow(rng, i, communities, popularity, following[acct])
            if pick is None:
                action = "post"
            else:
                following[acct].add(pick)
                target = ids[pick]
        if action == "post":
            posts_by_author.setdefault(acct, []).append((ts, n))
            recent.append((n, acct, ts))
        text = None
        if action in ("post", "comment"):
            profile = world.profiles[acct]
            text = content.compose(text_rngs[acct], action, profile.community, profile.style)
        world.events.append(Event(n, ts, acct, action, target, text, ip[acct]))

    world.followees = {a: {ids[x] for x in following[a]} for a in ids}
    return world


def serialize_ground_truth(ground_truth: dict[str, str]) -> bytes:
    return json.dumps(ground_truth, sort_keys=True, separators=(",", ":")).encode("utf-8")
