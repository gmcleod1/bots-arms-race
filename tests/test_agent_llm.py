"""The real `anthropic` client against a local mock server.

No API key is needed: the SDK talks to an in-process transport that checks the
request it sends and returns canned Messages responses. That verifies request
shape (model, adaptive thinking, effort, fallbacks, betas, no removed parameters),
response parsing, token accounting, and that thinking blocks survive a round trip.
It cannot verify how the live model behaves.
"""
import json

import anthropic
import httpx2 as httpx  # the 1.x SDK uses httpx2

from agent.llm import FALLBACK_BETA, MODEL, AnthropicLLM
from agent.tools import TOOL_SCHEMAS

THINKING = {"type": "thinking", "thinking": "", "signature": "sig-xyz"}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "platform_overview", "input": {}}


def _reply(content, stop="tool_use", usage=None):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": usage or {"input_tokens": 100, "output_tokens": 40,
                               "cache_creation_input_tokens": 10, "cache_read_input_tokens": 5}}


def _llm(replies, seen):
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "headers": dict(request.headers),
                     "body": json.loads(request.content)})
        return httpx.Response(200, json=queue.pop(0))

    client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return AnthropicLLM(client=client)


def test_request_shape_for_claude_opus_5():
    seen = []
    llm = _llm([_reply([TOOL_USE])], seen)
    llm.respond("system prompt", [{"role": "user", "content": "go"}], TOOL_SCHEMAS)
    req = seen[0]
    body = req["body"]
    assert req["path"].endswith("/v1/messages")
    assert body["model"] == "claude-opus-5" == MODEL
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in req["headers"]["anthropic-beta"]
    assert body["tools"] == TOOL_SCHEMAS and body["system"] and body["max_tokens"] >= 8_000
    assert body["cache_control"] == {"type": "ephemeral"}
    for removed in ("temperature", "top_p", "top_k", "budget_tokens"):  # rejected with a 400 on Opus 5
        assert removed not in json.dumps(body)
    assert body["messages"][-1] != {"role": "assistant"}  # no prefill


def test_response_is_parsed_and_tokens_are_counted_across_all_usage_fields():
    seen = []
    llm = _llm([_reply([THINKING, {"type": "text", "text": "plan"}, TOOL_USE])], seen)
    r = llm.respond("s", [{"role": "user", "content": "go"}], TOOL_SCHEMAS)
    assert r.stop_reason == "tool_use" and r.text == "plan" and r.tokens == 155
    assert [(c.id, c.name, c.input) for c in r.tool_calls] == [("toolu_1", "platform_overview", {})]


def test_thinking_blocks_survive_a_round_trip_unchanged():
    seen = []
    llm = _llm([_reply([THINKING, TOOL_USE]), _reply([{"type": "text", "text": "done"}], stop="end_turn")], seen)
    messages = [{"role": "user", "content": "go"}]
    first = llm.respond("s", messages, TOOL_SCHEMAS)
    messages += [{"role": "assistant", "content": first.content},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]}]
    llm.respond("s", messages, TOOL_SCHEMAS)
    echoed = seen[1]["body"]["messages"][1]["content"]
    assert echoed[0]["type"] == "thinking" and echoed[0]["signature"] == "sig-xyz"
    assert echoed[1]["type"] == "tool_use" and echoed[1]["id"] == "toolu_1"


def test_a_refusal_is_reported_not_raised():
    seen = []
    r = _llm([_reply([], stop="refusal")], seen).respond("s", [{"role": "user", "content": "go"}], TOOL_SCHEMAS)
    assert r.stop_reason == "refusal" and r.tool_calls == ()
