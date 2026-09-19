"""The face-off runner: the agent, the hot-patchable detector, the rules and the scoreboard.

`FaceOff.start()` runs the red agent on a worker thread while the defender patches through
`console`. Everything needed to replay the round without the model is written to the round
directory: manifest.json, per-attempt traces, detector-timeline.json (every version and
which one scored what), outcomes.json, and the audience feed (events.jsonl, state.json).
The agent's log stays sealed until the round ends.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.agent import RedAgent, RoundResult
from agent.llm import LLM
from agent.log import SealedLog
from agent.referee import AttemptOutcome, Referee
from agent.sandbox import Budget, CommitResult
from detector.config import DetectorConfig
from faceoff.console import DefenderConsole, Telemetry, make_anonymiser, make_telemetry
from faceoff.host import DetectorHost
from faceoff.rules import WinCondition, judge, naive_baseline, tally
from scoreboard.feed import Feed
from simulator.world import World, generate_world

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class WorldSpec:
    """Enough to rebuild both worlds exactly, for replay."""

    ref_seed: int
    eval_seed: int
    n_humans: int
    days: int

    def build(self) -> tuple[World, World]:
        return (generate_world(self.ref_seed, self.n_humans, self.days),
                generate_world(self.eval_seed, self.n_humans, self.days))


class _FaceOffReferee:
    """Scores each attempt on whichever detector version is live, then judges and publishes it."""

    def __init__(self, run: "FaceOff") -> None:
        self.run = run

    def evaluate(self, attempt: int, commit: CommitResult, attempts_remaining: int) -> AttemptOutcome:
        r = self.run
        r.feed.emit("scoring_started", attempt=attempt)
        verdicts, version = r.host.score(commit.events)
        outcome = Referee.evaluate_verdicts(attempt, commit, verdicts, attempts_remaining)
        outcome.detector_version = version.id
        judgement = judge(outcome, r.rule)
        r.judgements.append(judgement)
        r.telemetry.append(make_telemetry(attempt, version, verdicts, r.anon))
        c = outcome.confusion
        r.feed.emit(
            "attempt_scored", attempt=attempt, detector_version=version.id,
            winner=judgement.winner, reason=judgement.reason,
            bots_created=len(commit.farm.accounts), bots_caught=len(outcome.suspended),
            humans_flagged=c.fp, humans_total=c.fp + c.tn,
            humans_flagged_per_10k=round(c.humans_flagged_per_10k or 0.0),
            precision=c.precision, recall=c.recall,
            engagement_total=outcome.engagement_total, engagement_kept=outcome.engagement_kept,
        )
        r.record_attempt(outcome, judgement)
        r.host.thaw()  # fair-play mode: patches queued during the attempt apply now
        if r.gate_between_attempts and attempts_remaining > 0:
            r.waiting.set()
            r.feed.emit("waiting_for_defender", attempt=attempt)
            r.gate.wait(r.gate_timeout_s)  # the agent's launch call blocks here until the defender is ready
            r.gate.clear()
            r.waiting.clear()
        return outcome


class FaceOff:
    def __init__(
        self,
        llm: LLM,
        spec: WorldSpec,
        out_dir: Path,
        budget: Budget = Budget(),
        config: DetectorConfig = DetectorConfig(),
        pause_edits: bool = False,
        gate_between_attempts: bool = False,
        kept_fraction: float = 0.10,
        max_human_fpr: float = 0.10,
        gate_timeout_s: float | None = None,
        seed: int = 0,
        round_id: str | None = None,
        worlds: tuple[World, World] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.llm, self.spec, self.out, self.budget = llm, spec, out_dir, budget
        self.config, self.pause_edits, self.seed = config, pause_edits, seed
        self.gate_between_attempts = gate_between_attempts
        self.kept_fraction, self.max_human_fpr = kept_fraction, max_human_fpr
        self.gate_timeout_s = gate_timeout_s  # None waits for the defender indefinitely
        self.round_id = round_id or time.strftime("%Y%m%d-%H%M%S")
        self._clock = clock
        self.ref_world, self.eval_world = worlds or spec.build()
        self.anon = make_anonymiser(self.round_id)
        self.telemetry: list[Telemetry] = []
        self.judgements: list = []
        self.gate, self.waiting = threading.Event(), threading.Event()
        self._records: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.result: RoundResult | None = None
        self.feed: Feed
        self.host: DetectorHost
        self.console: DefenderConsole
        self.log: SealedLog

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "FaceOff":
        self.out.mkdir(parents=True, exist_ok=True)
        self.feed = Feed(self.out, self._clock)
        self.rule = WinCondition.from_baseline(naive_baseline(self.eval_world), self.kept_fraction,
                                               self.max_human_fpr)
        self.feed.emit("round_started", round_id=self.round_id, attempts_planned=self.budget.max_attempts,
                       pause_edits=self.pause_edits, win_condition=self.rule.to_json())
        self.host = DetectorHost(self.ref_world.events, self.config, self.pause_edits, self._clock,
                                 on_event=lambda t, **d: self.feed.emit(t, **d))
        self.console = DefenderConsole(self.host, self.telemetry, self.gate, self.waiting)
        self.log = SealedLog(self.out / "agent-log.jsonl")
        self._write_manifest()
        self._agent = RedAgent(self.llm, self.eval_world, _FaceOffReferee(self), self.budget, self.seed,
                               self.log, trace_dir=self.out, hooks=self._hook)
        self._thread = threading.Thread(target=self._run, name="red-agent", daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> RoundResult:
        assert self._thread is not None, "call start() first"
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("the round is still running")
        if self._error is not None:
            raise self._error
        assert self.result is not None
        return self.result

    def run(self) -> RoundResult:
        return self.start().join()

    # -- internals ---------------------------------------------------------

    def _hook(self, kind: str, **data: Any) -> None:
        if kind == "attempt_start":
            self.host.freeze()  # fair-play mode: no patches land mid-attempt
            self.feed.emit("attempt_started", attempt=data["attempt"])
        elif kind == "agent_call":
            self.feed.emit("agent_activity", attempt=data["attempt"], calls=data["calls"])
        elif kind == "attempt_launched":
            self.feed.emit("attempt_launched", attempt=data["attempt"], bots=data["bots"], actions=data["actions"])

    def _run(self) -> None:
        try:
            self.result = self._agent.run()
            self._write_records()
            self.feed.emit("round_finished", stop_reason=self.result.stop_reason,
                           tokens_used=self.result.tokens_used, tally=tally(self.judgements))
        except BaseException as exc:  # reported on the feed, then re-raised from join()
            self._error = exc
            self.feed.emit("round_failed", error=f"{type(exc).__name__}: {exc}")
            self.log.unseal()

    def record_attempt(self, outcome: AttemptOutcome, judgement) -> None:
        c = outcome.confusion
        self._records.append({
            "attempt": outcome.attempt, "detector_version": outcome.detector_version,
            "accounts": list(outcome.commit.farm.accounts), "targets": list(outcome.commit.farm.targets),
            "dropped": outcome.commit.dropped, "rejected": outcome.commit.rejected,
            "feedback": outcome.feedback, "flagged": sorted(outcome.flagged),
            "confusion": {"tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn},
            "engagement_total": outcome.engagement_total, "engagement_kept": outcome.engagement_kept,
            "judgement": {"winner": judgement.winner, "reason": judgement.reason},
        })

    def _write_manifest(self) -> None:
        manifest = {
            "manifest_version": MANIFEST_VERSION, "round_id": self.round_id,
            "worlds": {"ref_seed": self.spec.ref_seed, "eval_seed": self.spec.eval_seed,
                       "n_humans": self.spec.n_humans, "days": self.spec.days},
            "budget": self.budget.__dict__, "seed": self.seed, "pause_edits": self.pause_edits,
            "gate_between_attempts": self.gate_between_attempts,
            "win_condition": self.rule.to_json(), "initial_detector": self.config.to_json(),
            "model": type(self.llm).__name__,
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    def _write_records(self) -> None:
        (self.out / "outcomes.json").write_text(json.dumps(self._records, indent=1, sort_keys=True), encoding="utf-8")
        (self.out / "detector-timeline.json").write_text(
            json.dumps(self.host.history(), indent=1, sort_keys=True), encoding="utf-8")
