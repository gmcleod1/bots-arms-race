"""Behavioral signals. Each turns an event stream into per-account raw features.

Contract: `compute(events)` returns {feature_name: {account_id: raw_value}}, where a
HIGHER value is MORE bot-like. An account with too little evidence for a feature is
simply absent from that feature's dict (no evidence is not evidence of a bot).
Signals read events only, never labels, never `ip`.

Windows: every count-based feature depends on how much data it sees, so calibrate
(`fit`) and score on windows of the same length and scale.
"""
from __future__ import annotations

from typing import Iterable, Protocol

from simulator.events import Event

Features = dict[str, dict[str, float]]


class Signal(Protocol):
    name: str
    features: tuple[str, ...]

    def fit(self, events: Iterable[Event]) -> None: ...

    def compute(self, events: Iterable[Event]) -> Features: ...


def default_signals() -> list[Signal]:
    from detector.signals.content import ContentSignal
    from detector.signals.coordination import CoordinationSignal
    from detector.signals.timing import TimingSignal

    return [TimingSignal(), CoordinationSignal(), ContentSignal()]
