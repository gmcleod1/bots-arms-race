"""The tool surface the red agent works through. This IS the possibility space.

The agent may turn any knob listed here and nothing else: it cannot touch human
accounts, read or change the detector, or see scores. Every tool returns plain data
and reports problems as {"error": ...}; nothing here raises into the conversation.
What the public tools reveal is only what any user of the platform could see.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from agent.actions import Action
from agent.profile import Profile, ProfileError, expand, expected_actions
from agent.sandbox import BudgetError, Sandbox

_TIMING = {
    "type": "object",
    "description": "When bots act. Bursts ('sessions') start at random; each holds several actions.",
    "properties": {
        "sessions_per_day": {"type": "number", "description": "Bursts per bot per day, 0.05 to 48. Default 4."},
        "events_per_session": {"type": "number", "description": "Average actions per burst, 1 to 40. Default 4."},
        "pause_median_s": {"type": "number", "description": "Typical pause between actions in a burst, seconds, 1 to 7200. Default 25."},
        "pause_sigma": {"type": "number", "description": "Spread of those pauses, 0 to 3. 0 = identical pauses; higher = heavier-tailed. Default 1.1."},
        "awake_hours": {"type": "number", "description": "Length of the daily window in which bursts may start, 2 to 24 (24 = around the clock). Default 16."},
        "wake_hour_utc": {"type": "number", "description": "UTC hour the window opens, 0 to 23.99. Default 7."},
        "tz_spread_hours": {"type": "number", "description": "Shift each bot's window by a random amount within +/- this many hours, 0 to 12. Default 0."},
        "fixed_interval_s": {"type": ["number", "null"], "description": "If set (30 to 86400), each bot acts on an exact clock of this many seconds instead of bursts."},
    },
}
_CONTENT = {
    "type": "object",
    "description": "What bots post.",
    "properties": {
        "text_mode": {"type": "string", "enum": ["canned", "spun", "fresh"], "description": "canned: every bot draws from one shared pool. spun: the pool with random word swaps. fresh: newly written text each time. Default canned."},
        "pool_size": {"type": "integer", "description": "Messages in the shared pool (canned, spun), 1 to 50. Default 5."},
        "typo_rate": {"type": "number", "description": "Chance per word of 4+ letters of a random misspelling, 0 to 0.2. Default 0."},
        "style": {"type": "string", "enum": ["plain", "human"], "description": "plain: no quirks. human: each bot gets its own casing, punctuation, emoji and misspelling habits (fresh mode). Default plain."},
        "post_share": {"type": "number", "description": "Share of actions that are posts, 0 to 1. The rest like a recent human post. Default 0.3."},
    },
}
_ENGAGEMENT = {
    "type": "object",
    "description": "The farm's purpose: inflating engagement for target accounts.",
    "properties": {
        "targets": {"type": "array", "items": {"type": "string"}, "description": "Human account ids to boost (see list_targets)."},
        "react_probability": {"type": "number", "description": "Chance each bot likes each new post by a target, 0 to 1. Default 0."},
        "react_delay_median_s": {"type": "number", "description": "Seconds after the post that bots typically like it, 1 to 172800. Default 30."},
        "react_delay_sigma": {"type": "number", "description": "Spread of that delay, 0 to 3. 0 = every bot waits exactly the median. Default 0."},
        "follow_targets": {"type": "boolean", "description": "Bots follow the targets. Default false."},
        "follow_delay_median_s": {"type": "number", "description": "Typical seconds before following, 1 to 432000. Default 60."},
    },
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "platform_overview",
        "description": "Public facts about the platform this attempt: length of the window, how many accounts are visible, how active people are, the mix of actions, when in the day (UTC) people are active, and your remaining limits.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "sample_public_posts",
        "description": "A random sample of public posts by ordinary users, with author, time (seconds into the window) and text. Anyone can read these.",
        "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "description": "How many, 1 to 30. Default 10."}}},
    },
    {
        "name": "list_targets",
        "description": "Public accounts that post often enough to be worth boosting, with their post counts.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "description": "How many, 1 to 30. Default 10."}}},
    },
    {
        "name": "create_accounts",
        "description": "Create bot accounts for this attempt. Counts against your account limit. Returns their ids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "How many accounts to create."},
                "distinct_ips": {"type": "integer", "description": "Spread the accounts across this many IP addresses. Default 1."},
            },
            "required": ["count"],
        },
    },
    {
        "name": "apply_profile",
        "description": "Give accounts a behavior program covering the whole attempt window. You can call it several times with different accounts and settings; actions accumulate against your action limit. Unset fields take plain defaults.",
        "input_schema": {
            "type": "object",
            "properties": {
                "accounts": {"type": "array", "items": {"type": "string"}, "description": "Account ids to program. Default: all your accounts."},
                "timing": _TIMING,
                "content": _CONTENT,
                "engagement": _ENGAGEMENT,
            },
        },
    },
    {
        "name": "schedule_actions",
        "description": "Schedule individual actions for your accounts, for tactics the profile cannot express. Up to 200 per call. Likes and comments must target a public human post that already exists at that time; follows must target a public human account.",
        "input_schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "account": {"type": "string"},
                            "at_s": {"type": "integer", "description": "Seconds into the window."},
                            "kind": {"type": "string", "enum": ["post", "like", "comment", "follow"]},
                            "text": {"type": "string", "description": "For post and comment."},
                            "target": {"type": "string", "description": "Post id for like/comment, account id for follow."},
                        },
                        "required": ["account", "at_s", "kind"],
                    },
                }
            },
            "required": ["actions"],
        },
    },
    {
        "name": "launch_farm",
        "description": "Run everything you have staged for the whole window. Returns what you can observe afterwards. The platform is then reset for the next attempt: your accounts are gone, the humans are unchanged, and what you learned carries over.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finish",
        "description": "End the round now, without using the remaining attempts.",
        "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}},
    },
]
TOOL_NAMES = frozenset(t["name"] for t in TOOL_SCHEMAS)


class ToolBox:
    def __init__(self, sandbox: Sandbox, on_launch: Callable[[Sandbox], dict[str, Any]]) -> None:
        self.sb = sandbox
        self._on_launch = on_launch
        self.launched = False
        self.finished = False
        self._posts = {p.post_id: p for p in sandbox.view.human_posts}
        self._humans = sorted({e.account_id for e in sandbox.view.human_events})

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.launched or self.finished:
            return {"error": "this attempt is already over; wait for the next one"}
        handler = getattr(self, f"_{name}", None) if name in TOOL_NAMES else None
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return handler(**(args or {}))
        except (ProfileError, BudgetError, ValueError) as exc:
            return {"error": str(exc)}
        except TypeError as exc:  # missing or unexpected argument
            return {"error": f"bad arguments: {exc}"}

    # -- public information ------------------------------------------------

    def _platform_overview(self) -> dict[str, Any]:
        v, b = self.sb.view, self.sb.budget
        mix = Counter(e.action for e in v.human_events)
        hours = Counter((e.sim_ts // 3600) % 24 for e in v.human_events)
        total = sum(hours.values())
        return {
            "attempt": self.sb.attempt,
            "window_days": round(v.days, 2),
            "public_accounts": len(self._humans),
            "actions_per_account_per_day": round(len(v.human_events) / len(self._humans) / v.days, 1),
            "action_mix": {k: round(n / len(v.human_events), 3) for k, n in sorted(mix.items())},
            "hourly_activity_utc": [round(hours.get(h, 0) / total, 4) for h in range(24)],
            "your_accounts": len(self.sb.accounts),
            "limits": {
                "accounts": b.max_accounts,
                "actions_remaining": self.sb.actions_remaining,
            },
        }

    def _sample_public_posts(self, n: int = 10) -> dict[str, Any]:
        n = max(1, min(30, int(n)))
        posts = self.sb.view.human_posts
        idx = self.sb.rng.choice(len(posts), size=min(n, len(posts)), replace=False)
        return {"posts": [
            {"post_id": posts[int(i)].post_id, "account": posts[int(i)].account_id,
             "at_s": posts[int(i)].sim_ts, "text": posts[int(i)].text}
            for i in sorted(idx)
        ]}

    def _list_targets(self, limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(30, int(limit)))
        counts = Counter(p.account_id for p in self.sb.view.human_posts)
        top = sorted(((n, a) for a, n in counts.items() if n >= 20), reverse=True)[:limit]
        return {"targets": [{"account": a, "posts": n} for n, a in top]}

    # -- building the farm -------------------------------------------------

    def _create_accounts(self, count: int, distinct_ips: int = 1) -> dict[str, Any]:
        ids = self.sb.create_accounts(int(count), max(1, int(distinct_ips)))
        return {"created": ids, "accounts_total": len(self.sb.accounts)}

    def _apply_profile(self, accounts: list[str] | None = None, **sections: Any) -> dict[str, Any]:
        profile = Profile.from_args(sections)
        accounts = list(self.sb.accounts if accounts is None else accounts)
        if not accounts:
            return {"error": "no accounts: create some first"}
        unknown = [a for a in accounts if a not in self.sb.accounts]
        if unknown:
            return {"error": f"not your accounts: {unknown[:3]}"}
        bad_targets = [t for t in profile.engagement.targets if t not in self._humans]
        if bad_targets:
            return {"error": f"targets must be public human accounts, not: {bad_targets[:3]}"}
        est = expected_actions(profile, len(accounts), self.sb.view.days)
        if est > 3 * max(1, self.sb.actions_remaining):
            return {"error": f"this profile would generate roughly {int(est):,} actions, far over your "
                             f"remaining {self.sb.actions_remaining:,}; lower the activity"}
        actions, dropped = expand(profile, accounts, self.sb.view, self.sb.rng, self.sb.actions_remaining)
        kept = self.sb.stage(actions, dropped)
        self.sb.targets |= set(profile.engagement.targets)
        return {"accounts": len(accounts), "actions_staged": kept,
                "actions_dropped_by_limit": dropped, "actions_remaining": self.sb.actions_remaining}

    def _schedule_actions(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(actions, list) or not 1 <= len(actions) <= 200:
            return {"error": "provide between 1 and 200 actions"}
        staged: list[Action] = []
        problems: list[str] = []
        for i, a in enumerate(actions):
            try:
                staged.append(self._check_action(a))
            except ValueError as exc:
                problems.append(f"action {i}: {exc}")
        kept = self.sb.stage(staged)
        out: dict[str, Any] = {"actions_staged": kept, "actions_remaining": self.sb.actions_remaining}
        if problems:
            out["rejected"] = len(problems)
            out["reasons"] = problems[:5]
        return out

    def _check_action(self, a: dict[str, Any]) -> Action:
        acct, kind, at = a.get("account"), a.get("kind"), a.get("at_s")
        if acct not in self.sb.accounts:
            raise ValueError(f"{acct!r} is not your account")
        if kind not in ("post", "like", "comment", "follow"):
            raise ValueError(f"unknown kind {kind!r}")
        if isinstance(at, bool) or not isinstance(at, int) or not 0 <= at <= self.sb.view.horizon:
            raise ValueError(f"at_s must be an integer between 0 and {self.sb.view.horizon}")
        target, text = a.get("target"), a.get("text")
        if kind in ("post", "comment") and not (isinstance(text, str) and text.strip()):
            raise ValueError(f"{kind} needs text")
        if kind in ("like", "comment"):
            post = self._posts.get(target)
            if post is None or post.sim_ts > at:
                raise ValueError(f"no public human post {target!r} exists at that time")
        if kind == "follow" and target not in self._humans:
            raise ValueError(f"{target!r} is not a public human account")
        return Action(at, kind, acct, target if kind != "post" else None, text if kind in ("post", "comment") else None)

    # -- ending the attempt ------------------------------------------------

    def _launch_farm(self) -> dict[str, Any]:
        if not self.sb.accounts:
            return {"error": "create accounts before launching"}
        if self.sb.actions_remaining == self.sb.budget.max_actions:
            return {"error": "nothing is staged; apply_profile or schedule_actions first"}
        self.launched = True
        return self._on_launch(self.sb)

    def _finish(self, reason: str = "") -> dict[str, Any]:
        self.finished = True
        return {"status": "round ended", "reason": str(reason)[:200]}
