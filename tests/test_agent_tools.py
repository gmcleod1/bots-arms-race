"""The tool boundary: what the agent can reach, and the hard limits it cannot pass.

The tools are the only door between the agent and the simulator. These tests pin
that the door shows only public information, refuses bad input as data (never an
exception into the conversation), and enforces every budget as a hard stop.
"""
import json

import pytest

from agent.sandbox import Budget, Sandbox
from agent.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolBox
from simulator.events import serialize_events
from tests.helpers import cached_world

FORBIDDEN = ("ground_truth", "label", "is_bot", "score", "pvalue", "p_value", "verdict", "detector", "flagged")


@pytest.fixture(scope="module")
def world():
    return cached_world(11, 60, 5)


def _box(world, **budget):
    launched = []
    sb = Sandbox(world, Budget(**budget), seed=1, attempt=1)
    return ToolBox(sb, lambda s: launched.append(s) or {"attempt": 1}), sb, launched


def test_every_schema_has_a_handler_and_a_valid_shape():
    assert {t["name"] for t in TOOL_SCHEMAS} == TOOL_NAMES
    for t in TOOL_SCHEMAS:
        assert t["description"] and t["input_schema"]["type"] == "object"
    assert TOOL_NAMES == {"platform_overview", "sample_public_posts", "list_targets", "create_accounts",
                          "apply_profile", "schedule_actions", "launch_farm", "finish"}


def test_public_tools_reveal_only_public_human_information(world):
    box, sb, _ = _box(world)
    outputs = [box.call("platform_overview", {}), box.call("sample_public_posts", {"n": 8}),
               box.call("list_targets", {"limit": 5})]
    blob = json.dumps(outputs).lower()
    assert not any(word in blob for word in FORBIDDEN)
    ov = outputs[0]
    assert abs(sum(ov["hourly_activity_utc"]) - 1) < 0.01 and ov["window_days"] > 4
    assert all(p["account"].startswith("h-") for p in outputs[1]["posts"])
    assert 0 < len(outputs[2]["targets"]) <= 5


def test_tools_return_errors_as_data_and_never_raise(world):
    box, _, _ = _box(world)
    for name, args in (("nope", {}), ("create_accounts", {}), ("create_accounts", {"count": "many"}),
                       ("apply_profile", {"timing": {"awake_hours": 99}}),
                       ("apply_profile", {"bogus": 1}), ("schedule_actions", {"actions": []}),
                       ("launch_farm", {}), ("sample_public_posts", {"n": None})):
        out = box.call(name, args)
        assert isinstance(out, dict)
    assert "error" in box.call("apply_profile", {"timing": {"awake_hours": 99}})
    assert "error" in box.call("launch_farm", {})  # nothing staged


def test_account_budget_is_a_hard_stop(world):
    box, sb, _ = _box(world, max_accounts=10)
    assert len(box.call("create_accounts", {"count": 6})["created"]) == 6
    assert "error" in box.call("create_accounts", {"count": 5})
    assert len(sb.accounts) == 6
    assert len(box.call("create_accounts", {"count": 4})["created"]) == 4
    assert "error" in box.call("create_accounts", {"count": 1})


def test_action_budget_is_a_hard_stop_and_extra_actions_never_execute(world):
    box, sb, _ = _box(world, max_actions=300)
    box.call("create_accounts", {"count": 5})
    out = box.call("apply_profile", {"timing": {"awake_hours": 24, "sessions_per_day": 3, "events_per_session": 4}})
    assert out["actions_staged"] <= 300 and out["actions_remaining"] >= 0
    commit = sb.commit()
    assert len(commit.executed) - 5 <= 300  # minus the five create actions


def test_absurd_profiles_are_refused_before_they_are_generated(world):
    box, _, _ = _box(world, max_actions=1_000)
    box.call("create_accounts", {"count": 30})
    out = box.call("apply_profile", {"timing": {"sessions_per_day": 48, "events_per_session": 40, "awake_hours": 24}})
    assert "error" in out and "lower the activity" in out["error"]


def test_targets_must_be_public_humans_and_accounts_must_be_yours(world):
    box, _, _ = _box(world)
    box.call("create_accounts", {"count": 2})
    assert "error" in box.call("apply_profile", {"engagement": {"targets": ["bot-001"], "react_probability": 1}})
    assert "error" in box.call("apply_profile", {"accounts": ["h-0001"], "content": {"post_share": 1}})


def test_schedule_actions_validates_each_action_and_stages_the_valid_ones(world):
    box, sb, _ = _box(world)
    box.call("create_accounts", {"count": 1})
    post = sb.view.human_posts[5]
    out = box.call("schedule_actions", {"actions": [
        {"account": "bot-001", "at_s": post.sim_ts + 5, "kind": "like", "target": post.post_id},
        {"account": "bot-001", "at_s": post.sim_ts + 6, "kind": "post", "text": "hello"},
        {"account": "bot-001", "at_s": post.sim_ts - 1, "kind": "like", "target": post.post_id},  # before the post
        {"account": "bot-001", "at_s": 10, "kind": "post", "text": ""},
        {"account": "h-0001", "at_s": 10, "kind": "post", "text": "impersonation"},
        {"account": "bot-001", "at_s": 10, "kind": "follow", "target": "bot-001"},
    ]})
    assert out["actions_staged"] == 2 and out["rejected"] == 4


def test_launch_runs_the_farm_labels_bots_and_leaves_humans_untouched(world):
    box, sb, launched = _box(world)
    box.call("create_accounts", {"count": 4})
    box.call("apply_profile", {"timing": {"awake_hours": 24}, "content": {"post_share": 1.0}})
    out = box.call("launch_farm", {})
    assert out == {"attempt": 1} and launched == [sb]
    commit = sb.commit()
    assert {commit.ground_truth[a] for a in sb.accounts} == {"bot"}
    human = [e for e in commit.events if e.account_id.startswith("h-")]
    assert serialize_events(human) == serialize_events(world.events)


def test_after_launch_or_finish_every_tool_refuses(world):
    box, _, _ = _box(world)
    box.call("create_accounts", {"count": 1})
    box.call("apply_profile", {"content": {"post_share": 1.0}})
    box.call("launch_farm", {})
    assert "error" in box.call("create_accounts", {"count": 1})
    box2, _, _ = _box(world)
    box2.call("finish", {"reason": "done"})
    assert box2.finished and "error" in box2.call("platform_overview", {})
