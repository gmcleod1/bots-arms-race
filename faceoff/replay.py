"""Replay a recorded round without the model, and check it reproduces.

Rebuilds both worlds from the manifest, rebuilds every detector version from its recorded
config, re-executes each attempt's trace on a fresh platform, scores it with the version
that scored it live, and compares everything the round recorded: the agent's feedback, the
flagged accounts, the confusion counts, engagement, and the audience state. Any difference is
reported; nothing is assumed to match.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.actions import execute, load_trace
from agent.referee import Referee
from agent.sandbox import CommitResult
from detector.config import DetectorConfig, ReferenceCache, build_detector
from faceoff.round import WorldSpec
from faceoff.rules import WinCondition, judge
from scoreboard.feed import read_events, replay_state
from simulator.bots import Farm
from simulator.platform import Platform


@dataclass
class ReplayReport:
    attempts_checked: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def replay_round(round_dir: Path) -> ReplayReport:
    manifest = json.loads((round_dir / "manifest.json").read_text(encoding="utf-8"))
    timeline = json.loads((round_dir / "detector-timeline.json").read_text(encoding="utf-8"))
    recorded = json.loads((round_dir / "outcomes.json").read_text(encoding="utf-8"))
    rule = WinCondition.from_json(manifest["win_condition"])
    ref_world, eval_world = WorldSpec(**manifest["worlds"]).build()

    cache: ReferenceCache = {}
    detectors = {
        v["id"]: build_detector(DetectorConfig.from_json(v["config"]), ref_world.events, cache)
        for v in timeline["versions"]
    }
    report = ReplayReport()

    def check(label: str, got: Any, want: Any) -> None:
        if got != want:
            report.mismatches.append(f"{label}: replayed {got!r}, recorded {want!r}")

    for rec in recorded:
        k = rec["attempt"]
        platform = Platform(eval_world)
        executed, rejected = execute(platform, load_trace(round_dir / f"attempt-{k}.jsonl"))
        commit = CommitResult(
            events=platform.events_until(10**12), ground_truth=dict(platform.ground_truth),
            farm=Farm(tuple(rec["accounts"]), tuple(rec["targets"]), ()),
            executed=executed, rejected=rejected, dropped=rec["dropped"],
        )
        verdicts = detectors[rec["detector_version"]].score(commit.events)
        outcome = Referee.evaluate_verdicts(k, commit, verdicts, rec["feedback"]["attempts_remaining"])
        c = outcome.confusion
        check(f"attempt {k} feedback", outcome.feedback, rec["feedback"])
        check(f"attempt {k} flagged accounts", sorted(outcome.flagged), rec["flagged"])
        check(f"attempt {k} confusion", {"tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn}, rec["confusion"])
        check(f"attempt {k} engagement", (outcome.engagement_total, outcome.engagement_kept),
              (rec["engagement_total"], rec["engagement_kept"]))
        j = judge(outcome, rule)
        check(f"attempt {k} judgement", {"winner": j.winner, "reason": j.reason}, rec["judgement"])
        report.attempts_checked += 1

    events = read_events(round_dir / "events.jsonl")
    check("audience state rebuilt from events", replay_state(events),
          json.loads((round_dir / "state.json").read_text(encoding="utf-8")))
    scored = [e for e in events if e["type"] == "attempt_scored"]
    check("attempts in feed vs outcomes", [e["attempt"] for e in scored], [r["attempt"] for r in recorded])
    return report
