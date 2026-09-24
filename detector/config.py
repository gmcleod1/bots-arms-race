"""A detector as data: everything a live patch can change, and a builder that reuses work.

A patch is a new `DetectorConfig`. Because the config is plain JSON, every version can
be recorded and rebuilt exactly, which is what lets a recorded face-off be replayed.
Patches are limited to these parameters on purpose: the defender tunes the detector,
they do not write arbitrary code mid-round.

`target_overlap` switches the `c_target_overlap` feature on. It is OFF by default, so the default
detector is exactly v1 and any record written before the feature existed (no such key) rebuilds
the detector it was played against. A defender turns it on with a patch: `target_overlap=true`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterable

from detector.detector import Detector, compute_reference
from detector.signals import Signal
from detector.signals.content import ContentSignal
from detector.signals.coordination import CoordinationSignal
from detector.signals.timing import TimingSignal
from simulator.events import Event

# (signal name, params as JSON) -> (fitted signal, its reference distributions)
ReferenceCache = dict[tuple[str, str], tuple[Signal, dict[str, Any]]]


class ConfigError(ValueError):
    """A patch asked for something the detector does not allow."""


@dataclass(frozen=True)
class DetectorConfig:
    fpr_budget: float = 0.05
    coordination_window: int = 60  # seconds within which two accounts count as acting together
    coordination_min_target_events: int = 10
    timing_min_events: int = 30
    content_jaccard: float = 0.75
    content_min_texts: int = 5
    content_oov_min_accounts: int = 3
    disabled_features: tuple[str, ...] = ()
    target_overlap: bool = False  # add c_target_overlap: catches same-target lockstep at any spacing

    def signals(self) -> list[Signal]:
        return [
            TimingSignal(self.timing_min_events),
            CoordinationSignal(self.coordination_window, self.coordination_min_target_events, self.target_overlap),
            ContentSignal(self.content_jaccard, self.content_min_texts, self.content_oov_min_accounts),
        ]

    def patched(self, **changes: Any) -> "DetectorConfig":
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise ConfigError(f"unknown detector setting(s) {sorted(unknown)}; allowed: {sorted(known)}")
        if "disabled_features" in changes:
            changes["disabled_features"] = tuple(sorted(changes["disabled_features"]))
        new = replace(self, **changes)
        new.validate()
        return new

    def validate(self) -> None:
        if not 0 < self.fpr_budget <= 0.5:
            raise ConfigError("fpr_budget must be in (0, 0.5]")
        if not 1 <= self.coordination_window <= 7 * 86_400:
            raise ConfigError("coordination_window must be 1 second to 7 days")
        for name in ("coordination_min_target_events", "timing_min_events", "content_min_texts",
                     "content_oov_min_accounts"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if not isinstance(self.target_overlap, bool):
            raise ConfigError("target_overlap must be true or false")
        if not 0.3 <= self.content_jaccard <= 1.0:
            raise ConfigError("content_jaccard must be between 0.3 and 1.0")
        every = {f for s in self.signals() for f in s.features}
        if set(self.disabled_features) - every:
            raise ConfigError(f"unknown feature(s): {sorted(set(self.disabled_features) - every)}")
        if not every - set(self.disabled_features):
            raise ConfigError("at least one feature must stay enabled")

    def diff(self, other: "DetectorConfig") -> dict[str, list[Any]]:
        """{setting: [self value, other value]} for everything that differs."""
        a, b = self.to_json(), other.to_json()  # lists, not tuples: JSON-friendly for the feed
        return {k: [a[k], b[k]] for k in a if a[k] != b[k]}

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["disabled_features"] = list(self.disabled_features)
        return d

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DetectorConfig":
        data = dict(data)
        data["disabled_features"] = tuple(data.get("disabled_features", ()))
        return cls(**data)


def build_detector(
    config: DetectorConfig,
    reference_events: Iterable[Event],
    cache: ReferenceCache | None = None,
) -> Detector:
    """Deterministically build the detector a config describes, reusing cached reference work."""
    config.validate()
    events = list(reference_events) if not isinstance(reference_events, list) else reference_events
    signals: list[Signal] = []
    reference: dict[str, Any] = {}
    disabled = set(config.disabled_features)
    for signal in config.signals():
        if all(f in disabled for f in signal.features):
            continue  # a fully disabled signal is not built or run at all
        key = (signal.name, json.dumps(signal.params, sort_keys=True))
        if cache is not None and key in cache:
            fitted, ref = cache[key]
        else:
            fitted, ref = signal, compute_reference(signal, events)
            if cache is not None:
                cache[key] = (fitted, ref)
        signals.append(fitted)
        reference.update(ref)
    present = {f for s in signals for f in s.features}
    return Detector.from_fitted(signals, reference, config.fpr_budget, disabled & present)
