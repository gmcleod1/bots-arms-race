"""Coordination: accounts that should not know each other moving in lockstep.

Two accounts co-act when they engage the same target (like/comment on the same
post, or follow the same account) within WINDOW seconds. Counting distinct
shared targets per pair separates a farm from coincidence: one shared like is
chance, a dozen is a script. Sharing an IP or a timezone creates no co-action.

  c_coaction_max      most distinct targets this account co-acted on with any one other account
  c_lockstep_degree   how many other accounts it co-acted with on 2 or more targets
  c_target_overlap    the largest cosine similarity between this account's set of engaged
                      targets and any other account's, with NO time window (opt-in: `overlap=True`)

The first two only see accounts moving within WINDOW seconds of each other. Slow lockstep
(the same targets, hours apart) evades them, and widening the window does not fix it because
ordinary human co-action grows faster than the bots' signal. `c_target_overlap` ignores time
and asks who picks the same targets: bots boosting one set of posts share almost all of their
targets, humans share few (their reach spreads over a community).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from detector.signals import Features
from simulator.events import Event

WINDOW = 60  # seconds
MIN_TARGET_EVENTS = 10
# A target with more engagers than this is left out of pair counting: it cannot single out a
# clique and its pair count grows with the square of its audience. Known limit: a farm bigger
# than this, all on one target, hides that target from this feature (the others still see it).
MAX_ENGAGERS = 1_000


class CoordinationSignal:
    name = "coordination"

    def __init__(
        self, window: int = WINDOW, min_target_events: int = MIN_TARGET_EVENTS, overlap: bool = False
    ) -> None:
        self.window = window
        self.min_target_events = min_target_events
        self.overlap = overlap  # OFF by default: it costs a little stealth-plan recall (see CLAUDE.md)

    @property
    def features(self) -> tuple[str, ...]:
        base = ("c_coaction_max", "c_lockstep_degree")
        return base + ("c_target_overlap",) if self.overlap else base

    @property
    def params(self) -> dict:
        return {"window": self.window, "min_target_events": self.min_target_events, "overlap": self.overlap}

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
                while j < len(acts) and acts[j][0] - t <= self.window:
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
            if n >= self.min_target_events:
                out["c_coaction_max"][acct] = float(best.get(acct, 0))
                out["c_lockstep_degree"][acct] = float(degree.get(acct, 0))
        if self.overlap:
            out["c_target_overlap"] = self._target_overlap(by_target)
        return out

    def _target_overlap(self, by_target: dict[tuple[str, str], list[tuple[int, str]]]) -> dict[str, float]:
        """Each eligible account's largest cosine similarity to another eligible account.

        Eligible = engaged at least `min_target_events` DISTINCT targets, so a sparse account
        cannot look similar to everyone by sharing its one target."""
        audience = {key: {a for _, a in acts} for key, acts in by_target.items()}
        distinct: Counter[str] = Counter()
        for accts in audience.values():
            distinct.update(accts)
        eligible = {a for a, n in distinct.items() if n >= self.min_target_events}

        shared: Counter[tuple[str, str]] = Counter()
        for accts in audience.values():
            if len(accts) > MAX_ENGAGERS:
                continue
            group = sorted(accts & eligible)
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    shared[(a, b)] += 1

        best: dict[str, float] = dict.fromkeys(eligible, 0.0)
        for (a, b), n in shared.items():
            cosine = n / (distinct[a] * distinct[b]) ** 0.5
            if cosine > best[a]:
                best[a] = cosine
            if cosine > best[b]:
                best[b] = cosine
        return best
