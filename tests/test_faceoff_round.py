"""The face-off end to end, with a scripted model and a small real detector.

Pins: live patching while the agent works, the turn-based fair-play mode, that the
defender's console cannot see labels, that the audience feed never carries the agent's
content, and that a recorded round replays to the same outcome (and that the replay check
can actually fail).
"""
import json
import threading
import time

import pytest

from agent.llm import ScriptedLLM, scripted
from agent.reference_plans import naive, reference_plans
from agent.sandbox import Budget, Sandbox
from agent.tools import ToolBox
from faceoff.replay import replay_round
from faceoff.round import FaceOff, WorldSpec
from scoreboard.feed import read_events

SPEC = WorldSpec(ref_seed=301, eval_seed=302, n_humans=250, days=6)
MARKER = "SECRET-AGENT-MARKER-4471"


@pytest.fixture(scope="module")
def targets(duel_eval):
    box = ToolBox(Sandbox(duel_eval, Budget(), 0, 1), lambda s: {})
    return [t["account"] for t in box.call("list_targets", {"limit": 3})["targets"]]


def _attempt(plan, bots=30, first=None):
    calls = [scripted(("platform_overview", {}), text=f"my plan is {MARKER}") if first is None else first,
             scripted(("create_accounts", {"count": bots}), ("apply_profile", plan)),
             scripted(("launch_farm", {}))]
    return calls


def _round(tmp_path, duel_ref, duel_eval, script, name="round", **kwargs):
    llm = ScriptedLLM(script)
    fo = FaceOff(llm, SPEC, tmp_path / name, Budget(max_attempts=kwargs.pop("attempts", 2)),
                 worlds=(duel_ref, duel_eval), round_id=name, **kwargs)
    return fo, llm


def test_a_round_runs_scores_publishes_and_unseals(tmp_path, duel_ref, duel_eval, targets):
    quiet = reference_plans(targets)["H F with reactions at 5%, 12 h median"]
    fo, _ = _round(tmp_path, duel_ref, duel_eval,
                   _attempt(naive(targets)) + _attempt(quiet) + [scripted(("finish", {}))], attempts=3)
    result = fo.run()
    assert result.stop_reason == "agent_finished" and len(result.outcomes) == 2

    state = json.loads((fo.out / "state.json").read_text())
    assert state["round"]["status"] == "finished" and len(state["attempts"]) == 2
    first = state["attempts"][0]
    assert first["detector_version"] == "v1" and first["winner"] == "detector"  # naive farm is stopped
    assert first["bots_caught"] >= 0.9 * first["bots_created"] and first["engagement_kept"] == 0
    assert {"precision", "recall", "humans_flagged_per_10k"} <= set(first) and "accuracy" not in first
    assert state["tally"]["attacker"] + state["tally"]["detector"] == 2
    for name in ("manifest.json", "events.jsonl", "state.json", "outcomes.json", "detector-timeline.json",
                 "agent-log.jsonl", "attempt-1.jsonl", "attempt-2.jsonl"):
        assert (fo.out / name).exists(), name
    assert not fo.log.sealed


def test_a_live_patch_while_the_agent_is_thinking_scores_the_next_attempt(tmp_path, duel_ref, duel_eval, targets):
    holder = {}

    def defender_patches_now(messages):
        holder["fo"].console.patch("accept more false positives", fpr_budget=0.15)  # from "another thread"
        return scripted(("platform_overview", {}))

    script = _attempt(naive(targets)) + [defender_patches_now] + _attempt(naive(targets))[1:]
    fo, _ = _round(tmp_path, duel_ref, duel_eval, script)
    holder["fo"] = fo
    fo.run()
    outcomes = json.loads((fo.out / "outcomes.json").read_text())
    assert [o["detector_version"] for o in outcomes] == ["v1", "v2"]  # patched mid-round, no waiting
    timeline = json.loads((fo.out / "detector-timeline.json").read_text())
    assert [s["version"] for s in timeline["scores"]] == ["v1", "v2"]


def test_fair_play_queues_patches_during_an_attempt_and_gates_the_next_one(tmp_path, duel_ref, duel_eval, targets):
    holder = {}
    seen = {}

    def defender_tries_mid_attempt(messages):
        seen["queued"] = holder["fo"].console.patch("mid-attempt", fpr_budget=0.15).queued
        seen["current_during"] = holder["fo"].console.status()["version"]
        return scripted(("platform_overview", {}))

    script = [defender_tries_mid_attempt] + _attempt(naive(targets))[1:] + _attempt(naive(targets))
    fo, _ = _round(tmp_path, duel_ref, duel_eval, script, pause_edits=True, gate_between_attempts=True)
    holder["fo"] = fo
    fo.start()
    deadline = time.time() + 60
    while not fo.console.status()["waiting_for_you"] and time.time() < deadline:
        time.sleep(0.05)
    assert fo.console.status()["waiting_for_you"], "the round should pause for the defender after attempt 1"
    status = fo.console.status()
    assert status["version"] == "v2" and status["pending_patches"] == []  # queued patch applied at the boundary
    fo.console.patch("between attempts", coordination_window=300)  # the defender's own turn
    fo.console.ready()
    fo.join(timeout=120)

    assert seen == {"queued": True, "current_during": "v1"}  # nothing landed mid-attempt
    outcomes = json.loads((fo.out / "outcomes.json").read_text())
    assert [o["detector_version"] for o in outcomes] == ["v1", "v3"]  # attempt 2 saw both patches
    types = [e["type"] for e in read_events(fo.out / "events.jsonl")]
    assert types.index("waiting_for_defender") < len(types) - 1 - types[::-1].index("attempt_started")


def test_the_defender_console_sees_telemetry_but_no_labels(tmp_path, duel_ref, duel_eval, targets):
    fo, _ = _round(tmp_path, duel_ref, duel_eval, _attempt(naive(targets)) + [scripted(("finish", {}))])
    fo.run()
    t = fo.console.latest()
    assert t and t.flagged and t.near_misses and t.version == "v1"
    ids = [r.account for r in t.flagged + t.near_misses]
    assert all(i.startswith("a-") and "bot" not in i and "h-" not in i for i in ids)  # anonymised
    assert fo.anon("bot-001") == fo.anon("bot-001") != fo.anon("bot-002")
    other = FaceOff(ScriptedLLM([]), SPEC, tmp_path / "other", round_id="other", worlds=(duel_ref, duel_eval))
    assert other.anon("bot-001") != fo.anon("bot-001")  # per-round mapping
    # the console holds no truth, outcomes, agent log or agent
    held = {type(v).__name__ for v in vars(fo.console).values()}
    assert not held & {"AttemptOutcome", "SealedLog", "RedAgent", "World", "Platform", "CommitResult"}
    assert not any("truth" in name or "label" in name for name in vars(fo.console))
    assert not hasattr(t.flagged[0], "is_bot") and not hasattr(t.flagged[0], "label")


def test_the_audience_feed_never_carries_the_agents_content(tmp_path, duel_ref, duel_eval, targets):
    fo, _ = _round(tmp_path, duel_ref, duel_eval, _attempt(naive(targets)) + [scripted(("finish", {}))])
    fo.run()
    feed_text = (fo.out / "events.jsonl").read_text() + (fo.out / "state.json").read_text()
    assert MARKER not in feed_text and "bot-001" not in feed_text
    assert MARKER in (fo.out / "agent-log.jsonl").read_text()  # it is in the log, which unsealed at the end


def test_a_failure_is_reported_on_the_feed_and_raised(tmp_path, duel_ref, duel_eval):
    def boom(messages):
        raise RuntimeError("model unavailable")

    fo, _ = _round(tmp_path, duel_ref, duel_eval, [boom])
    with pytest.raises(RuntimeError, match="model unavailable"):
        fo.run()
    assert json.loads((fo.out / "state.json").read_text())["round"]["status"] == "failed"
    assert not fo.log.sealed


def test_a_recorded_round_replays_to_the_same_outcome_and_the_check_can_fail(tmp_path, duel_ref, duel_eval, targets):
    holder = {}

    def patch_between(messages):
        holder["fo"].console.patch("accept more false positives", fpr_budget=0.12)
        return scripted(("platform_overview", {}))

    quiet = reference_plans(targets)["H F with reactions at 5%, 12 h median"]
    script = _attempt(naive(targets)) + [patch_between] + _attempt(quiet)[1:]
    fo, llm = _round(tmp_path, duel_ref, duel_eval, script)
    holder["fo"] = fo
    fo.run()
    calls_during_round = llm.calls

    report = replay_round(fo.out)
    assert report.ok, report.mismatches
    assert report.attempts_checked == 2 and llm.calls == calls_during_round  # replay never asked the model

    # The check is not a rubber stamp: tamper with a trace and it must notice.
    trace = fo.out / "attempt-2.jsonl"
    lines = trace.read_text().splitlines()
    trace.write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    tampered = replay_round(fo.out)
    assert not tampered.ok and any("attempt 2" in m for m in tampered.mismatches)
