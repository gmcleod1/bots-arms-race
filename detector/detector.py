"""The v1 detector: calibrate each behavioral feature on clean history, flag by p-value.

Every feature's raw value (higher = more bot-like) is turned into an empirical
p-value against reference humans: the chance a reference human looks at least
this bot-like. An account is flagged if ANY feature's p-value is at or below its
share of the false-positive budget (Bonferroni: budget / number of features), so
the expected share of humans flagged stays near the budget. That is the stated
stance made explicit: choose the false-positive rate, then see what it buys.

The ranking score is -log10(smallest p-value). Every verdict records which
feature fired, so a catch can be explained ("regular timing", "lockstep").

Inputs are events only. Ground truth never enters here; it lives in metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from detector.signals import Signal, default_signals
from simulator.events import Event


@dataclass(frozen=True)
class Verdict:
    account_id: str
    score: float  # -log10(min p-value); higher = more suspicious
    flagged: bool
    top_feature: str  # the feature with the smallest p-value
    pvalues: dict[str, float] = field(default_factory=dict)  # only features with enough evidence


class Detector:
    def __init__(self, signals: list[Signal] | None = None, fpr_budget: float = 0.05) -> None:
        self.signals = signals if signals is not None else default_signals()
        self.fpr_budget = fpr_budget
        self._reference: dict[str, np.ndarray] = {}  # feature -> sorted raw values of reference humans

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(f for s in self.signals for f in s.features)

    @property
    def alpha(self) -> float:
        """Each feature's share of the false-positive budget."""
        return self.fpr_budget / len(self.features)

    def fit(self, reference_events: Iterable[Event]) -> None:
        """Calibrate on clean history: same window length and scale as what will be scored."""
        events = list(reference_events)
        reference: dict[str, np.ndarray] = {}
        for signal in self.signals:
            signal.fit(events)
            for feature, values in signal.compute(events).items():
                reference[feature] = np.sort(np.fromiter(values.values(), dtype=float))
        for feature in self.features:
            n = len(reference.get(feature, ()))
            if n + 1 < 1.0 / self.alpha:
                raise ValueError(
                    f"reference too small: {feature!r} has {n} accounts, but a per-feature "
                    f"budget of {self.alpha:.4f} needs at least {int(1 / self.alpha) - 1}"
                )
        self._reference = reference

    def p_value(self, feature: str, raw: float) -> float:
        ref = self._reference[feature]
        at_least_as_extreme = len(ref) - int(np.searchsorted(ref, raw, side="left"))
        return (at_least_as_extreme + 1) / (len(ref) + 1)

    def score(self, events: Iterable[Event]) -> dict[str, Verdict]:
        """A verdict per account with enough evidence for at least one feature."""
        if not self._reference:
            raise RuntimeError("Detector.fit() must be called before scoring")
        events = list(events)
        pvalues: dict[str, dict[str, float]] = {}
        for signal in self.signals:
            for feature, values in signal.compute(events).items():
                for acct, raw in values.items():
                    pvalues.setdefault(acct, {})[feature] = self.p_value(feature, raw)
        out = {}
        for acct, ps in pvalues.items():
            top = min(ps, key=lambda f: (ps[f], f))  # ties broken by name: deterministic
            out[acct] = Verdict(
                account_id=acct,
                score=float(-np.log10(ps[top])),
                flagged=ps[top] <= self.alpha,
                top_feature=top,
                pvalues=ps,
            )
        return out

    def flag(self, events: Iterable[Event]) -> set[str]:
        return {a for a, v in self.score(events).items() if v.flagged}
