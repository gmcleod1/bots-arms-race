"""The live detector: versioned, hot-patchable, and honest about which version scored what.

The defender patches by giving a label and settings to change. A patch builds a new
`DetectorVersion` (reusing cached reference work, so it takes seconds, not a full refit)
and swaps it in atomically: a scoring run in progress finishes on the version it started
with, and the next run uses the new one. Every scoring run is recorded with its version,
so a round can be replayed exactly.

Fair-play mode (`pause_edits`): between `freeze()` and `thaw()` patches are validated and
queued but not applied, which turns a round into turn-based play. `thaw()` applies them
in order.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from detector.config import ConfigError, DetectorConfig, ReferenceCache, build_detector
from detector.detector import Detector, Verdict
from simulator.events import Event


@dataclass(frozen=True)
class DetectorVersion:
    id: str  # "v1", "v2", ...
    number: int
    label: str
    config: DetectorConfig
    diff: dict[str, list[Any]]  # what changed from the previous version
    created_at: float
    build_seconds: float
    features: tuple[str, ...]  # the enabled features this version tests
    alpha: float  # each enabled feature's share of the false-positive budget


@dataclass(frozen=True)
class ScoreRecord:
    version_id: str
    at: float
    n_events: int


@dataclass(frozen=True)
class PatchResult:
    queued: bool  # True: accepted but waiting for thaw()
    version: DetectorVersion | None  # None while queued
    queue_position: int = 0


class DetectorHost:
    def __init__(
        self,
        reference_events: Iterable[Event],
        config: DetectorConfig = DetectorConfig(),
        pause_edits: bool = False,
        clock: Callable[[], float] = time.time,
        on_event: Callable[..., None] | None = None,
    ) -> None:
        self._reference = list(reference_events)
        self._cache: ReferenceCache = {}
        self._clock = clock
        self._on_event = on_event or (lambda *a, **k: None)
        self.pause_edits = pause_edits
        self._lock = threading.RLock()  # guards the fields below
        self._build_lock = threading.Lock()  # one patch builds at a time
        self._frozen = False
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self.versions: list[DetectorVersion] = []
        self.score_log: list[ScoreRecord] = []
        self._detector: Detector
        self._install(config, "initial detector", DetectorConfig().diff(config) if config != DetectorConfig() else {})

    # -- reading -----------------------------------------------------------

    @property
    def current(self) -> DetectorVersion:
        with self._lock:
            return self.versions[-1]

    @property
    def pending(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            return list(self._pending)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def score(self, events: Iterable[Event]) -> tuple[dict[str, Verdict], DetectorVersion]:
        """Score on whichever version is live right now; return it with the version used."""
        with self._lock:
            version, detector = self.versions[-1], self._detector
        events = list(events)
        verdicts = detector.score(events)
        with self._lock:
            self.score_log.append(ScoreRecord(version.id, self._clock(), len(events)))
        return verdicts, version

    # -- patching ----------------------------------------------------------

    def patch(self, label: str, **changes: Any) -> PatchResult:
        """Apply (or queue, when frozen) a change. Raises ConfigError for an invalid patch,
        in which case nothing changes."""
        if not changes:
            raise ConfigError("a patch must change at least one setting")
        with self._lock:
            base = self._latest_config()
            base.patched(**changes)  # validate now, so a bad patch fails fast even when queued
            if self._frozen:
                self._pending.append((label, dict(changes)))
                self._on_event("detector_patch_queued", label=label, changes=changes,
                               position=len(self._pending))
                return PatchResult(True, None, len(self._pending))
        return PatchResult(False, self._apply(label, changes))

    def freeze(self) -> None:
        if self.pause_edits:
            with self._lock:
                self._frozen = True

    def thaw(self) -> list[DetectorVersion]:
        """Unfreeze and apply everything queued, in order. Returns the versions created."""
        with self._lock:
            self._frozen = False
            queued, self._pending = self._pending, []
        return [self._apply(label, changes) for label, changes in queued]

    def _latest_config(self) -> DetectorConfig:
        config = self.versions[-1].config
        for _, changes in self._pending:
            config = config.patched(**changes)
        return config

    def _apply(self, label: str, changes: dict[str, Any]) -> DetectorVersion:
        with self._build_lock:
            new = self.versions[-1].config.patched(**changes)
            return self._install(new, label, self.versions[-1].config.diff(new))

    def _install(self, config: DetectorConfig, label: str, diff: dict[str, list[Any]]) -> DetectorVersion:
        started = self._clock()
        detector = build_detector(config, self._reference, self._cache)
        took = self._clock() - started
        with self._lock:
            n = len(self.versions) + 1
            version = DetectorVersion(f"v{n}", n, label, config, diff, self._clock(), took,
                                      detector.features, detector.alpha)
            self.versions.append(version)
            self._detector = detector  # the atomic swap: scoring in flight keeps its own reference
        self._on_event("detector_patched", version=version.id, label=label, diff=diff,
                       config=config.to_json(), build_seconds=round(took, 2))
        return version

    # -- recording ---------------------------------------------------------

    def history(self) -> dict[str, Any]:
        with self._lock:
            return {
                "versions": [
                    {"id": v.id, "number": v.number, "label": v.label, "config": v.config.to_json(),
                     "diff": v.diff, "created_at": v.created_at}
                    for v in self.versions
                ],
                "scores": [{"version": r.version_id, "at": r.at, "n_events": r.n_events} for r in self.score_log],
            }
