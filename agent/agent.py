"""The round runner: attempts, hard budgets, feedback, sealed log, replayable traces.

One conversation carries across attempts, so what the agent learned stays with it. Hard
stops are enforced here, between model calls: total tokens, wall-clock, attempts, and
model calls per attempt. Each attempt runs on a fresh platform and the full window, so
the detector sees the window length it was calibrated on.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.actions import save_trace
from agent.llm import LLM, LLMResponse
from agent.log import SealedLog
from agent.prompts import SYSTEM, briefing
from agent.referee import AttemptOutcome, Referee
from agent.sandbox import Budget, Sandbox
from agent.tools import TOOL_SCHEMAS, ToolBox
from simulator.world import World

NUDGE = "Call launch_farm to run this attempt, or finish to end the round."


@dataclass
class RoundResult:
    outcomes: list[AttemptOutcome] = field(default_factory=list)
    stop_reason: str = ""
    tokens_used: int = 0
    calls: int = 0
    elapsed_s: float = 0.0


class RedAgent:
    def __init__(
        self,
        llm: LLM,
        world: World,
        referee: Referee,
        budget: Budget = Budget(),
        seed: int = 0,
        log: SealedLog | None = None,
        clock: Callable[[], float] = time.monotonic,
        trace_dir: Path | None = None,
        hooks: Callable[..., None] | None = None,
    ) -> None:
        self.llm, self.world, self.referee, self.budget = llm, world, referee, budget
        self.seed, self.log, self.clock, self.trace_dir = seed, log or SealedLog(), clock, trace_dir
        # Progress signals only (attempt start, model call count, launch): never content or reasoning.
        self._hooks = hooks or (lambda *a, **k: None)

    def run(self) -> RoundResult:
        result = RoundResult()
        start = self.clock()
        messages: list[dict[str, Any]] = [{"role": "user", "content": briefing(self.budget)}]
        self.log.write("round_start", budget=self.budget.__dict__, seed=self.seed)
        try:
            result.stop_reason = self._attempts(messages, result, start)
        finally:
            result.elapsed_s = self.clock() - start
            self.log.write("round_end", stop_reason=result.stop_reason, tokens=result.tokens_used)
            self.log.unseal()  # the round is over: the defender may now read what the agent did
        return result

    def _attempts(self, messages: list[dict[str, Any]], result: RoundResult, start: float) -> str:
        for attempt in range(1, self.budget.max_attempts + 1):
            sandbox = Sandbox(self.world, self.budget, self.seed, attempt)
            self._hooks("attempt_start", attempt=attempt)
            toolbox = ToolBox(sandbox, lambda sb, n=attempt: self._launch(sb, n, result))
            calls = nudges = 0
            while not toolbox.launched:
                stop = self._hard_stop(result, start, calls)
                if stop:
                    return stop
                response = self.llm.respond(SYSTEM, messages, TOOL_SCHEMAS)
                calls += 1
                result.calls += 1
                self._hooks("agent_call", attempt=attempt, calls=result.calls)
                result.tokens_used += response.tokens
                self.log.write("assistant", attempt=attempt, text=response.text, stop=response.stop_reason,
                               calls=[(c.name, c.input) for c in response.tool_calls], tokens=response.tokens)
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "refusal":
                    return "refusal"
                if not response.tool_calls:
                    nudges += 1
                    if nudges > 2:
                        return "agent_stalled"
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                messages.append({"role": "user", "content": self._run_tools(toolbox, response)})
                if toolbox.finished:
                    return "agent_finished"
        return "max_attempts"

    def _hard_stop(self, result: RoundResult, start: float, calls: int) -> str | None:
        if result.tokens_used >= self.budget.max_tokens:
            return "token_budget"
        if self.clock() - start >= self.budget.wall_clock_s:
            return "wall_clock"
        if calls >= self.budget.max_calls_per_attempt:
            return "call_budget"
        return None

    def _run_tools(self, toolbox: ToolBox, response: LLMResponse) -> list[dict[str, Any]]:
        # Launch and finish run last, so anything staged in the same turn is included.
        ordered = sorted(response.tool_calls, key=lambda c: c.name in ("launch_farm", "finish"))
        blocks = []
        for call in ordered:
            out = toolbox.call(call.name, call.input)
            self.log.write("tool", name=call.name, input=call.input, output=_short(out))
            blocks.append({
                "type": "tool_result", "tool_use_id": call.id,
                "content": json.dumps(out), **({"is_error": True} if "error" in out else {}),
            })
        return blocks

    def _launch(self, sandbox: Sandbox, attempt: int, result: RoundResult) -> dict[str, Any]:
        commit = sandbox.commit()
        self._hooks("attempt_launched", attempt=attempt, bots=len(sandbox.accounts),
                    actions=len(commit.executed) - len(sandbox.accounts))
        outcome = self.referee.evaluate(attempt, commit, self.budget.max_attempts - attempt)
        result.outcomes.append(outcome)
        if self.trace_dir is not None:
            save_trace(commit.executed, self.trace_dir / f"attempt-{attempt}.jsonl")
        self.log.write("attempt", attempt=attempt, feedback=outcome.feedback,
                       precision=outcome.confusion.precision, recall=outcome.confusion.recall,
                       humans_flagged=outcome.confusion.fp)
        return outcome.feedback


def _short(out: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(out)
    return out if len(text) < 2_000 else {"truncated": text[:2_000]}
