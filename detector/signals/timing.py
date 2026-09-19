"""Timing and rhythm.

Humans are bursty (sessions, then silence) and they sleep. Bots tend to be
regular, and even with added jitter they often never rest. Two features:

  t_regularity  negative Goh-Barabasi burstiness of inter-event gaps. About +1 for a
                fixed interval, about 0 for a Poisson process, negative for bursty humans.
  t_no_sleep    normalised entropy of the hour-of-day histogram (UTC). Near 1 for an
                account active around the clock. The detector does not know timezones;
                a person's own sleep still shows up as a quiet stretch in their histogram.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from detector.signals import Features
from simulator.events import Event

MIN_EVENTS = 30


class TimingSignal:
    name = "timing"
    features = ("t_regularity", "t_no_sleep")

    def __init__(self, min_events: int = MIN_EVENTS) -> None:
        self.min_events = min_events

    @property
    def params(self) -> dict:
        return {"min_events": self.min_events}

    def fit(self, events: Iterable[Event]) -> None:
        pass  # stateless

    def compute(self, events: Iterable[Event]) -> Features:
        times: dict[str, list[int]] = defaultdict(list)
        for e in events:
            times[e.account_id].append(e.sim_ts)
        out: Features = {f: {} for f in self.features}
        for acct, ts in times.items():
            if len(ts) < self.min_events:
                continue
            arr = np.sort(np.asarray(ts, dtype=np.int64))
            gaps = np.diff(arr).astype(float)
            mu, sigma = gaps.mean(), gaps.std()
            burstiness = -1.0 if mu + sigma == 0 else (sigma - mu) / (sigma + mu)
            out["t_regularity"][acct] = float(-burstiness)
            counts = np.bincount((arr // 3600) % 24, minlength=24)
            p = counts[counts > 0] / counts.sum()
            out["t_no_sleep"][acct] = float(-(p * np.log(p)).sum() / math.log(24))
        return out
