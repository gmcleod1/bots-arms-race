"""The model behind the agent, behind a small interface so the harness can be tested.

`AnthropicLLM` is the real thing: Claude Opus 5, adaptive thinking, and server-side
refusal fallbacks (the platform default recommended for this model). `ScriptedLLM`
replays canned responses so the harness (budgets, feedback, sealed log, replay) is
tested without spending tokens.

Conversation rules honoured here: the assistant's `content` is appended back to
the history verbatim, unedited and append-only, so thinking blocks and their
signatures stay valid; and every tool_result for one assistant turn goes in one
user message (the loop in agent.py does that).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: list[Any]  # opaque: appended to the history exactly as received
    tool_calls: tuple[ToolCall, ...]
    text: str
    stop_reason: str  # end_turn, tool_use, max_tokens, refusal, ...
    tokens: int  # input + output (+ cache) tokens this call used


class LLM(Protocol):
    def respond(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(
        self,
        client: Any = None,
        model: str = MODEL,
        max_tokens: int = 16_000,
        effort: str = "high",
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or a logged-in profile
        self.client, self.model, self.max_tokens, self.effort = client, model, max_tokens, effort

    def respond(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        blocks = list(response.content)
        calls = tuple(
            ToolCall(b.id, b.name, dict(b.input)) for b in blocks if getattr(b, "type", None) == "tool_use"
        )
        text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text")
        u = response.usage
        tokens = sum(int(getattr(u, k, 0) or 0) for k in (
            "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return LLMResponse(blocks, calls, text, str(response.stop_reason), tokens)


class ScriptedLLM:
    """Replays responses in order. An entry may be a callable taking the message history."""

    def __init__(self, script: list[LLMResponse | Callable[[list[dict[str, Any]]], LLMResponse]]) -> None:
        self._script = list(script)
        self.calls = 0
        self.seen: list[list[dict[str, Any]]] = []

    def respond(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.seen.append(list(messages))
        if self.calls >= len(self._script):
            raise AssertionError("ScriptedLLM was called more times than scripted")
        entry = self._script[self.calls]
        self.calls += 1
        return entry(messages) if callable(entry) else entry


def scripted(*calls: tuple[str, dict[str, Any]], tokens: int = 1_000, text: str = "",
             stop_reason: str | None = None, extra_blocks: list[Any] | None = None) -> LLMResponse:
    """Build a canned response containing the given (tool name, input) calls."""
    tool_calls = tuple(ToolCall(f"toolu_{i}_{name}", name, args) for i, (name, args) in enumerate(calls))
    content: list[Any] = list(extra_blocks or [])
    if text:
        content.append({"type": "text", "text": text})
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tool_calls]
    return LLMResponse(content, tool_calls, text, stop_reason or ("tool_use" if tool_calls else "end_turn"), tokens)
