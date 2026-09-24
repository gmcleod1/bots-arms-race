# Code overview

A plain-language walkthrough of how the pieces fit together. For terse, load-bearing facts and
measured numbers (the kind you'd want at your fingertips while editing), see `CLAUDE.md` — this
file is the "how does this actually work" companion to that one. `README.md` has the concept and
motivation; this file has the mechanics.

## The mental model

Everything happens inside one process, on a fake platform with a known ground truth:

```
simulator/  → a fake social platform with real humans and, during a round, bot accounts
detector/   → the blue team: reads only events (never labels), flags accounts as bot-like
agent/      → the red team: an entity (scripted or a real LLM) that operates bot accounts
faceoff/    → the referee: runs rounds, scores them, lets a human patch the detector live
scoreboard/ → publishes the outcome (never the bots' identities) for an audience to watch
scripts/    → entry points that wire the above together for a specific purpose
```

Nothing here talks to a real social platform. `simulator.Platform` is the entire "internet" for
this project — every action bots take goes through it, and it's the only thing that knows which
accounts are actually bots.

## Data flow, end to end

1. **`simulator/world.py`** generates a `World`: several hundred synthetic humans with
   circadian/bursty posting schedules, a follow graph across 8 communities, realistic-ish text,
   and ~15% of them sharing an IP (like hotel guests) specifically to prove the detector doesn't
   cheat by looking at IPs. Every random draw goes through `simulator/rng.py`'s `SeedTree`, keyed
   per account, so the whole thing is exactly reproducible from a seed — this is what makes
   replay possible later.
2. **`simulator/platform.py`**'s `Platform` wraps a `World` and is the only way anything (bots)
   can add events to it. It enforces realism (no time travel, can't engage your own post, a
   target must already exist) and it — not the caller — labels every account it creates as
   `"bot"` in `ground_truth`. Human events never change once the world is generated.
3. Whoever is playing red (a scripted plan, or a real LLM) calls **`agent/tools.py`**'s
   `ToolBox` — the *entire* set of actions available: create accounts, set a behavior profile
   (timing/content/engagement knobs), or schedule raw actions, then `launch_farm`. This tool
   surface **is** the possibility space; there's no way to touch the detector, see scores, or
   act on humans. If you want a new evasion tactic to be possible, it has to become a new knob
   here — that's a deliberate design constraint (`CLAUDE.md`), not an oversight.
4. Launching produces a `CommitResult` full of `Event`s — indistinguishable in shape from human
   events. This full window of events (humans + this attempt's bots) is what the **detector**
   sees.
5. **`detector/`** turns events into a verdict per account, entirely unsupervised:
   - `detector/signals/{timing,coordination,content}.py` each compute raw "how bot-like is
     this account on this axis" numbers (0 or more features each). Coordination, for instance,
     looks at how many *other* accounts engage the same targets close in time
     (`c_coaction_max`, `c_lockstep_degree`), plus an opt-in time-free variant
     (`c_target_overlap`) that instead asks "do these two accounts share almost the exact same
     set of targets, whenever they acted."
   - `detector/detector.py`'s `Detector.fit()` calibrates each feature against a clean
     (human-only) reference world: for every feature it stores the *sorted* human values, so
     later it can ask "what fraction of reference humans were at least this extreme?" — that
     fraction is the feature's p-value for a given account.
   - An account is flagged if **any single feature's** p-value falls at or below
     `fpr_budget / (number of enabled features)` — a Bonferroni correction, so turning on more
     features doesn't silently increase the false-positive rate. This is why `c_target_overlap`
     being optional matters: enabling it shrinks every other feature's share of the budget too.
   - Nothing here is a trained model. It's explicitly a calibrated-statistics ensemble, chosen
     because there were no labeled bots to train on until the agent existed (`M3`).
6. **`faceoff/round.py`**'s `FaceOff` is the referee that ties it all together for a *live*
   session: it runs the red side (`agent/agent.py`'s `RedAgent`) on a background thread, one
   attempt at a time, each on a freshly-reset platform (so detection is judged per full window,
   not mid-stream). A `DetectorHost` (`faceoff/host.py`) holds the *current* detector version
   and lets a human hot-patch its config (new false-positive budget, new window size, toggle
   `target_overlap`, ...) in seconds, without touching signal code — a patch is just a new
   `DetectorConfig`, and each version is timestamped so a recorded round can be replayed exactly
   against whichever version scored which attempt.
7. **Fairness is enforced by what different consumers are allowed to see, not by hiding data
   after the fact.** `faceoff/console.py`'s `DefenderConsole` (what a human patching live would
   see) gets anonymised account IDs, scores, and near-misses — never which accounts are
   *actually* bots. `agent/log.py`'s `SealedLog` keeps the agent's own reasoning unreadable
   until the round ends. `scoreboard/feed.py`'s `Feed` — the one thing published to an audience
   — carries only outcomes (bots caught, humans flagged, engagement) and, optionally, a
   production caption a scripted demo sets for itself; it never carries the agent's reasoning or
   ground truth either. Three different audiences, three different views of the same round, all
   fed by the same underlying event stream.
8. **`scoreboard/`** turns that outcome stream into something watchable. `feed.py` appends
   events to `events.jsonl` and folds them into `state.json` via a pure `reduce()` function —
   pure meaning `state.json` can always be rebuilt from scratch by replaying `events.jsonl`
   through the same function, which is exactly what `replay_state()` (used by both tests and
   `replay_round.py`) does. `server.py` is a tiny stdlib HTTP server that serves *only*
   `dashboard.html`, `state.json`, and `events.jsonl` — nothing else in a round's folder is
   reachable over HTTP. `dashboard.html` is a single self-contained page: no build step, no
   framework, just hand-rolled SVG charts and a 1-second poll loop.

## Determinism and replay

Because every random draw is seeded and keyed, and because a `FaceOff` records everything it
did (`manifest.json`, per-attempt action traces, `detector-timeline.json`, the full feed), a
finished round can be reproduced exactly without ever calling a model again:
`faceoff/replay.py`'s `replay_round()` rebuilds both worlds from the manifest, rebuilds every
detector version from its recorded config, re-executes each attempt's trace on a fresh
`Platform`, re-scores it, and diffs everything (feedback, flagged accounts, confusion counts,
engagement, judgement, and the entire audience `state.json`) against what was recorded. A test
in `tests/test_faceoff_round.py` deliberately corrupts a trace to prove this check can actually
fail, not just always report success.

## The two "attacker" implementations

`agent/llm.py` defines one small interface (`LLM.respond(system, messages, tools) → LLMResponse`)
with two implementations:

- **`ScriptedLLM`** replays a canned list of responses. `agent/reference_plans.py`'s named plans
  (A through H — naive, through various combinations of human-style text/timing/reactions) exist
  *only* for this: measuring the harness and producing repeatable demo rounds. The real agent
  never sees these; they're not part of its prompt or tools.
- **`AnthropicLLM`** calls the real Claude Opus 5 API with adaptive thinking and server-side
  refusal fallbacks. Same interface, so `agent/agent.py`'s `RedAgent` round-running logic
  (budgets, attempts, feedback, sealed logging) doesn't know or care which one it's talking to.

`scripts/escalation_round.py` is a third thing built on top of `ScriptedLLM`: a *fixed* 4-attempt
arc using named plans, plus an `attempt_labels` dict passed into `FaceOff` purely so the
scoreboard can caption each attempt with a human-readable name (e.g. "G — reactions at 15%, 6h
median"). That's a production annotation only — it flows through the same generic
`attempt_started` feed event as everything else, and a real live-agent round simply never sets it.

## Where to look if you want to change something

| Want to... | Start here |
|---|---|
| Add a new detection signal | `detector/signals/` — implement `fit`/`compute` per the `Signal` protocol in `detector/signals/__init__.py`, add it to `default_signals()` |
| Add a new evasion tactic for the agent | `agent/tools.py` (the tool schema) + `agent/profile.py` (how it expands into actions) — the agent can only do what's a tool |
| Change the win condition | `faceoff/rules.py` (`WinCondition`, `judge`) |
| Change what the defender console can see | `faceoff/console.py` |
| Change the scoreboard's look | `scoreboard/dashboard.html` (single file, hand-rolled SVG charts, CSS custom properties for theming) |
| Add a new scripted demo scenario | `agent/reference_plans.py` for the plan shapes, a new `scripts/*.py` modeled on `scripts/escalation_round.py` to sequence them |
| Understand a measured number cited in code comments | `CLAUDE.md` — it's the terse, fact-dense reference; this file is the narrative one |
