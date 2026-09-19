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


def compute_reference(signal: Signal, events: list[Event]) -> dict[str, np.ndarray]:
    """Fit `signal` on clean events and return each feature's sorted reference distribution."""
    signal.fit(events)
    return {
        feature: np.sort(np.fromiter(values.values(), dtype=float))
        for feature, values in signal.compute(events).items()
    }


class Detector:
    def __init__(
        self,
        signals: list[Signal] | None = None,
        fpr_budget: float = 0.05,
        disabled: Iterable[str] = (),
    ) -> None:
        self.signals = signals if signals is not None else default_signals()
        self.fpr_budget = fpr_budget
        self.disabled = frozenset(disabled)
        known = {f for s in self.signals for f in s.features}
        if self.disabled - known:
            raise ValueError(f"cannot disable unknown feature(s): {sorted(self.disabled - known)}")
        if not self.features:
            raise ValueError("at least one feature must stay enabled")
        self._reference: dict[str, np.ndarray] = {}  # feature -> sorted raw values of reference humans

    @classmethod
    def from_fitted(
        cls,
        signals: list[Signal],
        reference: dict[str, np.ndarray],
        fpr_budget: float,
        disabled: Iterable[str] = (),
    ) -> "Detector":
        """Assemble a detector from signals that are already fitted, with their reference
        distributions already computed. This is what makes a live patch quick: only the
        signals a patch touched need recomputing."""
        d = cls(signals, fpr_budget, disabled)
        d._check_reference_size(reference)
        d._reference = reference
        return d

    @property
    def features(self) -> tuple[str, ...]:
        """The enabled features."""
        return tuple(f for s in self.signals for f in s.features if f not in self.disabled)

    @property
    def alpha(self) -> float:
        """Each enabled feature's share of the false-positive budget."""
        return self.fpr_budget / len(self.features)

    def _check_reference_size(self, reference: dict[str, np.ndarray]) -> None:
        for feature in self.features:
            n = len(reference.get(feature, ()))
            if n + 1 < 1.0 / self.alpha:
                raise ValueError(
                    f"reference too small: {feature!r} has {n} accounts, but a per-feature "
                    f"budget of {self.alpha:.4f} needs at least {int(1 / self.alpha) - 1}"
                )

    def fit(self, reference_events: Iterable[Event]) -> None:
        """Calibrate on clean history: same window length and scale as what will be scored."""
        events = list(reference_events)
        reference: dict[str, np.ndarray] = {}
        for signal in self.signals:
            if all(f in self.disabled for f in signal.features):
                continue  # nothing this signal produces is used
            reference.update(compute_reference(signal, events))
        self._check_reference_size(reference)
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
            if all(f in self.disabled for f in signal.features):
                continue
            for feature, values in signal.compute(events).items():
                if feature in self.disabled:
                    continue
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
