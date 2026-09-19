"""Coordination: accounts that should not know each other moving in lockstep.

Two accounts co-act when they engage the same target (like/comment on the same
post, or follow the same account) within WINDOW seconds. Counting distinct
shared targets per pair separates a farm from coincidence: one shared like is
chance, a dozen is a script. Sharing an IP or a timezone creates no co-action.

  c_coaction_max      most distinct targets this account co-acted on with any one other account
  c_lockstep_degree   how many other accounts it co-acted with on 2 or more targets
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from detector.signals import Features
from simulator.events import Event

WINDOW = 60  # seconds
MIN_TARGET_EVENTS = 10


class CoordinationSignal:
    name = "coordination"
    features = ("c_coaction_max", "c_lockstep_degree")

    def fit(self, events: Iterable[Event]) -> None:
        pass  # stateless

    def compute(self, events: Iterable[Event]) -> Features:
        by_target: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        n_events: Counter[str] = Counter()
        for e in events:
            if e.action in ("like", "comment"):
                key = ("engage", e.target_id)
            elif e.action == "follow":
                key = ("follow", e.target_id)
            else:
                continue
            by_target[key].append((e.sim_ts, e.account_id))
            n_events[e.account_id] += 1

        shared: Counter[tuple[str, str]] = Counter()  # (a, b) -> distinct targets co-acted on
        for acts in by_target.values():
            if len(acts) < 2:
                continue
            acts.sort()
            pairs: set[tuple[str, str]] = set()
            for i, (t, a) in enumerate(acts):
                j = i + 1
                while j < len(acts) and acts[j][0] - t <= WINDOW:
                    b = acts[j][1]
                    if a != b:
                        pairs.add((a, b) if a < b else (b, a))
                    j += 1
            shared.update(pairs)

        best: Counter[str] = Counter()
        degree: Counter[str] = Counter()
        for (a, b), count in shared.items():
            for x in (a, b):
                best[x] = max(best[x], count)
                if count >= 2:
                    degree[x] += 1

        out: Features = {f: {} for f in self.features}
        for acct, n in n_events.items():
            if n >= MIN_TARGET_EVENTS:
                out["c_coaction_max"][acct] = float(best.get(acct, 0))
                out["c_lockstep_degree"][acct] = float(degree.get(acct, 0))
        return out
