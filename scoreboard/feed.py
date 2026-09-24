"""The scoreboard feed: what the audience (and any overlay) can watch.

Two files in the round directory, both plain JSON so anything can read them:

  events.jsonl  append-only, one event per line: {"v": 1, "seq": n, "t": unix seconds, "type": ..., ...}
  state.json    the current picture, derived from the events by `reduce` (pure, so it can
                always be rebuilt from events.jsonl)

Schema version 1. The feed carries OUTCOMES only: which detector version scored an attempt,
bots caught, humans wrongly flagged, engagement inflated. It never carries the agent's
reasoning, tool inputs or log; those stay sealed until the round ends. `attempt_started` may
also carry an optional `label` (e.g. "15% reaction stealth"): a production caption set by a
scripted demo script, never by the agent, and absent on a real live-agent round.

For a stream overlay: serve or copy `state.json` and poll it, or tail `events.jsonl`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1


def initial_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "round": {"id": None, "status": "waiting", "attempts_planned": 0, "started_at": None,
                  "finished_at": None, "stop_reason": None, "pause_edits": False},
        "win_condition": None,
        "detector": {"version": None, "label": None, "patches": [], "queued": []},
        "current": None,
        "attempts": [],
        "tally": {"attacker": 0, "detector": 0, "leader": "draw"},
        "last_event_seq": -1,
    }


def _leader(t: dict[str, Any]) -> str:
    return "attacker" if t["attacker"] > t["detector"] else "detector" if t["detector"] > t["attacker"] else "draw"


def reduce(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Fold one event into the state (in place) and return it."""
    kind, t = event["type"], event["t"]
    state["last_event_seq"] = event["seq"]
    if kind == "round_started":
        state["round"].update(id=event["round_id"], status="running", attempts_planned=event["attempts_planned"],
                              started_at=t, pause_edits=event.get("pause_edits", False))
        state["win_condition"] = event.get("win_condition")
    elif kind == "detector_patched":
        d = state["detector"]
        d["version"], d["label"] = event["version"], event["label"]
        d["patches"].append({"version": event["version"], "label": event["label"], "diff": event["diff"], "at": t})
        if event["label"] in d["queued"]:
            d["queued"].remove(event["label"])
    elif kind == "detector_patch_queued":
        state["detector"]["queued"].append(event["label"])
    elif kind == "attempt_started":
        state["current"] = {"attempt": event["attempt"], "phase": "agent_planning", "model_calls": 0,
                            "label": event.get("label")}
    elif kind == "agent_activity":
        if state["current"]:
            state["current"]["model_calls"] = event["calls"]
    elif kind == "attempt_launched":
        if state["current"]:
            state["current"].update(phase="scoring", bots=event["bots"], actions=event["actions"])
    elif kind == "attempt_scored":
        record = {k: v for k, v in event.items() if k not in ("v", "seq", "t", "type")}
        state["attempts"].append(record)
        state["tally"][event["winner"]] += 1
        state["tally"]["leader"] = _leader(state["tally"])
        if state["current"]:
            state["current"]["phase"] = "between"
    elif kind == "waiting_for_defender":
        if state["current"]:
            state["current"]["phase"] = "waiting_for_defender"
    elif kind == "round_finished":
        state["round"].update(status="finished", finished_at=t, stop_reason=event["stop_reason"])
        state["current"] = None
    elif kind == "round_failed":
        state["round"].update(status="failed", finished_at=t, stop_reason=event.get("error", "error"))
        state["current"] = None
    return state


def replay_state(events: list[dict[str, Any]]) -> dict[str, Any]:
    state = initial_state()
    for e in events:
        reduce(state, e)
    return state


def read_events(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Feed:
    def __init__(self, directory: Path, clock: Callable[[], float] = time.time) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._state = initial_state()
        self.events_path = directory / "events.jsonl"
        self.state_path = directory / "state.json"
        self.events_path.write_text("", encoding="utf-8")
        self._write_state()

    def emit(self, type: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            event = {"v": SCHEMA_VERSION, "seq": self._seq, "t": self._clock(), "type": type, **data}
            self._seq += 1
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
            reduce(self._state, event)
            self._write_state()
            return event

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def _write_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, sort_keys=True, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)  # readers never see a half-written file
