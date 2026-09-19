"""The round runner: budgets, feedback, the sealed log, and replay.

The model is scripted, so these tests cost no tokens and pin the harness itself:
what stops the agent, what it is told, and that a round can be reproduced from its
trace without asking the model anything again.
"""
import json
from dataclasses import dataclass

import pytest

from agent.actions import load_trace
from agent.agent import RedAgent
from agent.llm import ScriptedLLM, scripted
from agent.log import SealedError, SealedLog
from agent.referee import FEEDBACK_KEYS, Referee
from agent.sandbox import Budget
from detector.detector import Verdict
from simulator.events import serialize_events
from simulator.platform import Platform
from agent.actions import execute
from tests.helpers import cached_world

LOUD = {"timing": {"awake_hours": 24, "sessions_per_day": 6, "events_per_session": 6}, "content": {"post_share": 1.0}}
QUIET = {"timing": {"awake_hours": 8, "sessions_per_day": 1, "events_per_session": 2}, "content": {"post_share": 1.0}}


class StubDetector:
    """Flags any bot account with 60+ events. Stands in for the real detector in loop tests."""

    def score(self, events):
        counts = {}
        for e in events:
            counts[e.account_id] = counts.get(e.account_id, 0) + 1
        return {a: Verdict(a, 3.0, n >= 60, "t_regularity", {"t_regularity": 0.001})
                for a, n in counts.items() if a.startswith("bot-")}


@pytest.fixture(scope="module")
def world():
    return cached_world(11, 60, 5)


def _agent(world, script, log=None, clock=None, trace_dir=None, **budget):
    llm = ScriptedLLM(script)
    kwargs = {"clock": clock} if clock else {}
    agent = RedAgent(llm, world, Referee(StubDetector()), Budget(**budget), seed=2, log=log,
                     trace_dir=trace_dir, **kwargs)
    return agent, llm


def _attempt(profile, n=8, tokens=1_000):
    return [scripted(("platform_overview", {}), tokens=tokens),
            scripted(("create_accounts", {"count": n}), ("apply_profile", profile), tokens=tokens),
            scripted(("launch_farm", {}), tokens=tokens)]


def test_two_attempts_adapt_to_feedback_and_the_conversation_is_well_formed(world):
    agent, llm = _agent(world, _attempt(LOUD) + _attempt(QUIET) + [scripted(("finish", {"reason": "done"}))],
                        max_attempts=3)
    result = agent.run()
    assert result.stop_reason == "agent_finished" and len(result.outcomes) == 2
    loud, quiet = result.outcomes
    assert len(loud.suspended) == 8 and len(quiet.suspended) == 0  # the agent's change of approach worked
    assert quiet.engagement_kept >= 0

    final = llm.seen[-1]  # history as the last call saw it
    for i, m in enumerate(final):
        if m["role"] == "assistant":
            uses = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
            nxt = final[i + 1]
            results = [b["tool_use_id"] for b in nxt["content"]]
            assert nxt["role"] == "user" and sorted(uses) == sorted(results)  # all results, one message
    # The first model call of attempt 2 (call index 3) already saw attempt 1's feedback, as the
    # result of its launch_farm call.
    first_of_attempt_2 = llm.seen[3]
    feedback = [json.loads(b["content"]) for m in first_of_attempt_2
                if m["role"] == "user" and isinstance(m["content"], list)
                for b in m["content"] if "accounts_suspended" in b["content"]]
    assert len(feedback) == 1 and feedback[0]["attempt"] == 1 and feedback[0]["accounts_suspended"] == 8


def test_feedback_is_limited_to_what_a_real_attacker_can_observe(world):
    agent, _ = _agent(world, _attempt(LOUD) + [scripted(("finish", {}))], max_attempts=2)
    fb = agent.run().outcomes[0].feedback
    assert set(fb) <= FEEDBACK_KEYS
    blob = json.dumps(fb).lower()
    assert not any(w in blob for w in ("score", "pvalue", "t_regularity", "precision", "recall", "h-0"))
    assert all(a.startswith("bot-") for a in fb["suspended_ids"])


def test_the_defenders_view_reports_precision_and_recall_separately(world):
    agent, _ = _agent(world, _attempt(LOUD) + [scripted(("finish", {}))], max_attempts=2)
    c = agent.run().outcomes[0].confusion
    assert c.recall == 1.0 and c.precision == 1.0 and c.fp == 0 and not hasattr(c, "accuracy")


def test_token_budget_is_a_hard_stop_between_calls(world):
    # Checked between calls: 2,000 used < 2,500, so a third call is allowed (and it launches);
    # the budget then stops the round before a fourth. Overshoot is at most one call.
    agent, llm = _agent(world, _attempt(LOUD, tokens=1_000) + [scripted(("finish", {}))], max_tokens=2_500)
    result = agent.run()
    assert result.stop_reason == "token_budget" and llm.calls == 3 and result.tokens_used == 3_000
    assert len(result.outcomes) == 1

    # A tighter budget stops the round before the agent ever launches.
    agent, llm = _agent(world, _attempt(LOUD, tokens=1_000), max_tokens=1_500)
    result = agent.run()
    assert result.stop_reason == "token_budget" and llm.calls == 2 and result.outcomes == []


def test_wall_clock_is_a_hard_stop(world):
    ticks = iter(range(0, 10_000, 100))
    agent, llm = _agent(world, _attempt(LOUD), clock=lambda: next(ticks), wall_clock_s=250)
    assert agent.run().stop_reason == "wall_clock" and llm.calls < 3


def test_attempts_are_capped_and_no_model_call_follows_the_last_launch(world):
    agent, llm = _agent(world, _attempt(QUIET) + _attempt(QUIET), max_attempts=2)
    result = agent.run()
    assert result.stop_reason == "max_attempts" and len(result.outcomes) == 2 and llm.calls == 6
    assert result.outcomes[-1].feedback["note"] == "That was your last attempt."


def test_calls_per_attempt_are_capped(world):
    agent, _ = _agent(world, [scripted(("platform_overview", {}))] * 3, max_calls_per_attempt=3)
    assert agent.run().stop_reason == "call_budget"


def test_an_agent_that_only_talks_is_nudged_then_stopped(world):
    agent, llm = _agent(world, [scripted(text="thinking...")] * 3)
    assert agent.run().stop_reason == "agent_stalled" and llm.calls == 3


def test_a_refusal_ends_the_round_cleanly(world):
    agent, _ = _agent(world, [scripted(stop_reason="refusal")])
    assert agent.run().stop_reason == "refusal"


def test_thinking_blocks_are_passed_back_verbatim(world):
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    first = scripted(("platform_overview", {}), extra_blocks=[thinking])
    agent, llm = _agent(world, [first, scripted(("finish", {}))])
    agent.run()
    history = llm.seen[1]
    assert history[1] == {"role": "assistant", "content": first.content}  # unedited, in order
    assert history[1]["content"][0] is thinking


def test_the_log_is_sealed_during_the_round_and_readable_after(world, tmp_path):
    log = SealedLog(tmp_path / "agent-log.jsonl")
    probe = {}

    def peek(messages):
        try:
            log.read()
        except SealedError:
            probe["sealed_mid_round"] = True
        return scripted(("finish", {}))

    agent, _ = _agent(world, [peek], log=log)
    agent.run()
    assert probe == {"sealed_mid_round": True} and not log.sealed
    kinds = [r["kind"] for r in log.read()]
    assert kinds[0] == "round_start" and kinds[-1] == "round_end" and "assistant" in kinds
    assert len((tmp_path / "agent-log.jsonl").read_text().splitlines()) == len(kinds)


def test_a_round_replays_from_its_trace_without_the_model(world, tmp_path):
    agent, llm = _agent(world, _attempt(LOUD) + [scripted(("finish", {}))], trace_dir=tmp_path, max_attempts=2)
    outcome = agent.run().outcomes[0]
    platform = Platform(world)
    executed, rejected = execute(platform, load_trace(tmp_path / "attempt-1.jsonl"))
    assert rejected == 0
    replayed = platform.events_until(10**12)
    assert serialize_events(replayed) == serialize_events(outcome.commit.events)
    assert StubDetector().score(replayed).keys() == StubDetector().score(outcome.commit.events).keys()
    assert llm.calls == 4  # the replay itself made no model calls
