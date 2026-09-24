"""The win condition and the scoreboard feed."""
import json
import threading
from types import SimpleNamespace

import pytest

from detector.metrics import Confusion
from faceoff.rules import Judgement, WinCondition, judge, naive_baseline, tally
from scoreboard.feed import Feed, initial_state, read_events, reduce, replay_state


def _outcome(kept, fp=10, tn=990, tp=20, fn=0):
    return SimpleNamespace(engagement_kept=kept, confusion=Confusion(tp, fp, fn, tn))


RULE = WinCondition.from_baseline(naive_engagement=10_000, kept_fraction=0.10, max_human_fpr=0.10)


def test_the_rule_is_built_from_a_measured_baseline():
    assert RULE.attacker_min_kept == 1_000 and RULE.max_human_fpr == 0.10 and RULE.naive_engagement == 10_000
    assert WinCondition.from_json(RULE.to_json()) == RULE


def test_the_attacker_wins_an_attempt_by_keeping_enough_engagement():
    j = judge(_outcome(kept=1_000), RULE)
    assert j.winner == "attacker" and "1,000" in j.reason
    assert judge(_outcome(kept=999), RULE).winner == "detector"


def test_a_detector_that_flags_too_many_humans_forfeits_even_if_it_stops_the_farm():
    ban_everyone = _outcome(kept=0, fp=150, tn=850)  # 15% of humans flagged
    j = judge(ban_everyone, RULE)
    assert j.winner == "attacker" and "limit" in j.reason and j.human_fpr == pytest.approx(0.15)
    assert judge(_outcome(kept=0, fp=100, tn=900), RULE).winner == "detector"  # exactly at the limit is fine


def test_tally_and_leader():
    a, d = Judgement("attacker", "", 0, 0), Judgement("detector", "", 0, 0)
    assert tally([a, d, d]) == {"attacker": 1, "detector": 2, "leader": "detector"}
    assert tally([a, d])["leader"] == "draw" and tally([a, a])["leader"] == "attacker"


def test_the_naive_baseline_is_measured_on_the_world(duel_eval):
    assert naive_baseline(duel_eval, bots=20) > 500


# ---- feed ---------------------------------------------------------------

def _run_feed(tmp_path):
    feed = Feed(tmp_path)
    feed.emit("round_started", round_id="r1", attempts_planned=2, pause_edits=False,
              win_condition=RULE.to_json())
    feed.emit("detector_patched", version="v1", label="initial detector", diff={}, config={})
    feed.emit("attempt_started", attempt=1)
    feed.emit("agent_activity", attempt=1, calls=3)
    feed.emit("attempt_launched", attempt=1, bots=40, actions=1000)
    feed.emit("attempt_scored", attempt=1, detector_version="v1", winner="detector", bots_caught=40,
              bots_created=40, humans_flagged=9, engagement_total=5000, engagement_kept=0)
    feed.emit("detector_patch_queued", label="looser", changes={"fpr_budget": 0.1}, position=1)
    feed.emit("detector_patched", version="v2", label="looser", diff={"fpr_budget": [0.05, 0.1]}, config={})
    feed.emit("attempt_started", attempt=2)
    feed.emit("attempt_scored", attempt=2, detector_version="v2", winner="attacker", bots_caught=5,
              bots_created=40, humans_flagged=20, engagement_total=900, engagement_kept=700)
    feed.emit("round_finished", stop_reason="max_attempts")
    return feed


def test_the_state_is_a_pure_function_of_the_events(tmp_path):
    feed = _run_feed(tmp_path)
    state = feed.state
    assert replay_state(read_events(tmp_path / "events.jsonl")) == state  # rebuildable from the log
    assert json.loads((tmp_path / "state.json").read_text()) == state
    assert state["round"]["status"] == "finished" and state["current"] is None
    assert state["detector"]["version"] == "v2" and state["detector"]["queued"] == []
    assert [p["version"] for p in state["detector"]["patches"]] == ["v1", "v2"]
    assert state["tally"] == {"attacker": 1, "detector": 1, "leader": "draw"}
    assert [a["detector_version"] for a in state["attempts"]] == ["v1", "v2"]


def test_events_are_sequenced_versioned_and_append_only(tmp_path):
    _run_feed(tmp_path)
    events = read_events(tmp_path / "events.jsonl")
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert {e["v"] for e in events} == {1} and all(e["t"] > 0 for e in events)


def test_the_phase_follows_the_attempt(tmp_path):
    feed = Feed(tmp_path)
    feed.emit("attempt_started", attempt=1)
    assert feed.state["current"]["phase"] == "agent_planning"
    feed.emit("attempt_launched", attempt=1, bots=10, actions=50)
    assert feed.state["current"]["phase"] == "scoring"
    feed.emit("attempt_scored", attempt=1, detector_version="v1", winner="detector")
    assert feed.state["current"]["phase"] == "between"
    feed.emit("waiting_for_defender", attempt=1)
    assert feed.state["current"]["phase"] == "waiting_for_defender"


def test_attempt_started_label_is_optional_and_carried_into_current(tmp_path):
    feed = Feed(tmp_path)
    feed.emit("attempt_started", attempt=1, label="15% reaction stealth")
    assert feed.state["current"]["label"] == "15% reaction stealth"
    feed.emit("attempt_started", attempt=2)  # a real live-agent round: no label passed
    assert feed.state["current"]["label"] is None


def test_concurrent_emitters_never_corrupt_the_log(tmp_path):
    feed = Feed(tmp_path)
    threads = [threading.Thread(target=lambda i=i: [feed.emit("agent_activity", attempt=1, calls=i) for _ in range(25)])
               for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    events = read_events(tmp_path / "events.jsonl")
    assert sorted(e["seq"] for e in events) == list(range(200))


def test_a_failed_round_is_reported_not_left_hanging(tmp_path):
    feed = Feed(tmp_path)
    feed.emit("round_started", round_id="r", attempts_planned=1)
    feed.emit("round_failed", error="boom")
    assert feed.state["round"]["status"] == "failed" and feed.state["round"]["stop_reason"] == "boom"
