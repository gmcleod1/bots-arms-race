"""The event record every other component reads.

This schema is the contract between the simulator, the detector (M2) and the
red agent (M4). Labels are deliberately absent: ground truth lives in a
separate file keyed by account_id, so the detector cannot read it by accident.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

ACTIONS = ("post", "like", "follow", "comment")


def post_id_of(event_id: int) -> str:
    """The id other events use as `target_id` to engage with the post created by `event_id`."""
    return f"p-{event_id}"


@dataclass(frozen=True)
class Event:
    event_id: int
    sim_ts: int  # simulated seconds since world start
    account_id: str
    action: str
    target_id: str | None  # post or account acted on; None for a plain post
    text: str | None  # post/comment body; None for like/follow
    ip: str  # present so tests can prove the detector ignores it

    @property
    def post_id(self) -> str:
        """Id other events use as `target_id` to like or comment on this post."""
        if self.action != "post":
            raise ValueError(f"only a post has a post_id, not {self.action!r}")
        return post_id_of(self.event_id)

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}, expected one of {ACTIONS}")
        has_body = self.action in ("post", "comment")
        if has_body and not self.text:
            raise ValueError(f"{self.action} needs non-empty text")
        if not has_body and self.text is not None:
            raise ValueError(f"{self.action} must not carry text")


def serialize_events(events: Iterable[Event]) -> bytes:
    """Canonical JSONL: fixed key order, one event per line, ordered by (sim_ts, event_id)."""
    ordered = sorted(events, key=lambda e: (e.sim_ts, e.event_id))
    lines = [
        json.dumps(asdict(e), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for e in ordered
    ]
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
