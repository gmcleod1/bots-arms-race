"""The one execution path for anything a bot does: a flat, replayable list of actions.

Tools stage `Action`s; committing a sandbox and replaying a saved trace both go
through `execute`, so a round can be reproduced from its trace without asking the
LLM anything again.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from simulator.platform import Platform, PlatformError

KINDS = ("create", "post", "like", "comment", "follow")


class Action(NamedTuple):
    ts: int
    kind: str
    account: str
    target: str | None = None  # like/comment: post id; follow: account id; create: the ip address
    text: str | None = None


def execute(platform: Platform, actions: list[Action]) -> tuple[list[Action], int]:
    """Apply actions in time order (creates first at equal times). Returns (executed, rejected)."""
    order = sorted(range(len(actions)), key=lambda i: (actions[i].ts, actions[i].kind != "create", i))
    executed: list[Action] = []
    rejected = 0
    for i in order:
        a = actions[i]
        try:
            if a.kind == "create":
                platform.create_account(a.account, a.target or "198.51.100.1", a.ts)
            elif a.kind == "post":
                platform.post(a.account, a.ts, a.text)
            elif a.kind == "like":
                platform.like(a.account, a.ts, a.target)
            elif a.kind == "comment":
                platform.comment(a.account, a.ts, a.target, a.text)
            elif a.kind == "follow":
                platform.follow(a.account, a.ts, a.target)
            else:
                raise PlatformError(f"unknown action kind {a.kind!r}")
        except PlatformError:
            rejected += 1
            continue
        executed.append(a)
    return executed, rejected


def save_trace(actions: list[Action], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in actions:
            f.write(json.dumps(a._asdict(), sort_keys=True, ensure_ascii=True) + "\n")


def load_trace(path: Path) -> list[Action]:
    with path.open(encoding="utf-8") as f:
        return [Action(**json.loads(line)) for line in f if line.strip()]
