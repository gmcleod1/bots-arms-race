# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this directory.

## Project Overview

Bots Arms Race is a self-contained red team vs. blue team build. A behavioral bot detector (blue) is attacked by an autonomous agent running an adversarial bot farm (red). The agent picks its own evasion tactics instead of following a scripted plan. Each escalation is filmed for the **Garfield McLeod | Security** YouTube channel.

`README.md` holds the full concept, origin problem, and open questions. Read it first.

**Status:** M0 to M4 done (2026-09-19); M4's live-model path is UNVERIFIED because no API credentials existed (see "Red agent" below). Decisions are logged in `intent.md` section 12, `simulator/` has humans, the platform API and the scripted naive bot farm (`simulator/bots.py`), and `detector/` v1 has all three signals plus the calibrated ensemble. Round 0 is recorded in `results/round-0.md` (regenerate with `.venv/Scripts/python scripts/round0.py`, about 80 s). `agent/` has the autonomous red agent; `scoreboard/` does not exist yet. The roadmap is milestones M0 to M6 in `C:\Users\netge\.claude\plans\pasted-content-id-0be3-bots-arms-smooth-scone.md`; next is M5 (live face-off harness: rolling windows, detector hot-reload, scoreboard), gated on confirming 6.3 and 6.5 with Garfield. Optional follow-ups: a live agent run once credentials exist, and a learned ensemble now that labeled bots exist. Scoring/presentation (6.3), agent autonomy (6.4), and win condition (6.5) are proposed but not yet accepted, so confirm them with Garfield before building `agent/` or `scoreboard/`.

## Design Constraints

- **Behavioral detection, not network signals.** IP and VPN blocking is out of scope. The detector works on behavior.
- **Signal families:** timing and rhythm, cross-account coordination, content fingerprinting. All three go into v1, built as independent modules behind one interface.
- **Bias toward false positives.** The stance is that the detector should accept more false positives, assuming a fast human appeals process. Report precision and recall separately, not just accuracy.
- **The red agent is autonomous.** Never hardcode its evasion tactics. It observes and decides. Manual check-in points are set by Garfield.
- **Live face-off, fair by construction.** Garfield sees only defender-side telemetry during a round. The agent's action log is sealed until the round ends. Every agent action and every detector hot-reload is timestamped and versioned so rounds replay from the recorded trace.
- **Self-contained environment.** Default to a sandboxed, simulated platform with known ground truth. Never point the agent at a real platform or a third-party service's public surface without asking first.

## Layout

| Directory | Role |
|-----------|------|
| `simulator/` | BUILT. Synthetic humans (circadian, bursty, community graph, typo'd content, shared-IP hotels), the `Platform` API bots act through, and `bots.py` (`Script` scheduler + scripted `naive_farm`, `engagement_delivered`) |
| `scripts/`, `results/` | `scripts/round0.py` writes the measured round-0 snapshot to `results/round-0.md` |
| `detector/` | BUILT (v1). `signals/{timing,coordination,content}.py`, `detector.py` (calibrated p-value ensemble), `metrics.py`, `minhash.py`. Reads events only |
| `agent/` | BUILT (M4). `tools.py` (the possibility space), `profile.py` (behavior knobs), `sandbox.py` (per-attempt budgets and public view), `referee.py` (what the agent may learn), `agent.py` (round runner), `llm.py` (Claude Opus 5 adapter + scripted fake), `log.py` (sealed log), `actions.py` (replayable traces), `reference_plans.py` (measurement only) |
| `scoreboard/` | Not built. Scoring and presentation of each round (M5) |

## Simulator contracts (tests enforce these; do not break them)

- **Run tests:** `.\.venv\Scripts\python -m pytest -q` from this directory. `requirements-dev.txt` recreates the venv.
- **Labels never appear in events.** `Event` has no label field. Ground truth is `World.ground_truth` / `Platform.ground_truth`. The detector and agent must only ever receive events, never a `World`.
- **Ordering key is `(sim_ts, event_id)`.** Bot event ids continue after the last human id, so id order alone is not time order.
- **Seed streams are keyed by account id** (`SeedTree.for_account`), not creation order, so adding accounts never shifts another account's draws. Every random draw goes through `SeedTree`, in a fixed order.
- **Bots never alter the human log.** `Platform` labels its accounts `bot` itself and rejects time travel, self-engagement, and targets that do not exist yet.
- **`ip` is on every event only to prove the detector ignores it.** About 15% of humans share an IP (hotel groups): same timezone, different communities, no lockstep.
- **Determinism tests must call `generate_world` directly.** Read-only tests share worlds through `tests/helpers.cached_world`. Never use it to compare two runs.

## Detector v1 design (tests enforce the contracts; the limits are known, not bugs)

- **Six features, higher = more bot-like:** `t_regularity`, `t_no_sleep` (timing); `c_coaction_max`, `c_lockstep_degree` (coordination, same target within 60 s); `x_dup_rate`, `x_shared_oov` (content). Each signal returns `{feature: {account: raw}}` and omits accounts without enough evidence.
- **Ensemble = calibrated p-values, not a trained model.** `Detector.fit(clean_reference_events)` builds each feature's human distribution; a feature's p-value is the share of reference humans at least as extreme; an account is flagged when any p-value is at most `fpr_budget / 6` (Bonferroni). Score is `-log10(min p)`. It is untrained because there were no labeled bots until M3; a learned ensemble is a candidate once M3 exists.
- **Calibrate and score on the same window length and scale.** Count-based features (coordination, content) grow with data. Reference must be big enough to resolve the budget or `fit` raises. M5 will need rolling windows.
- **Measured baseline (reference seed 101, 800 humans x 10 days):** human FPR tracks the budget on fresh seeds (budget 5% gave 4.25% and 5.1%; 2% gave 2.0% and 2.1%). Naive patterns (fixed interval, jitter, lockstep, copy-paste, mutated copy, shared novel typo) are all caught at a 2% budget.
- **Known limits:** (1) same-target lockstep spread over hours evades the 60 s window (0/12 caught; `test_known_gap_same_targets_hours_apart_evades_v1` documents it, retire it when patched). (2) The score saturates at `-log10(1/(n_ref+1))`, so top-tier accounts tie: 9 of 800 humans sit at the same ceiling as caught bots and average precision is only 0.68 despite full recall on caught patterns. (3) `x_shared_oov` is the noisiest feature and produces about half the human false positives, partly because the simulator's typo variants collide.
- **Content signal attributes duplication to the later author.** The first occurrence of a text is never a duplicate, so a copied human is not blamed. Verified in isolation; in a crowded bot world chance similarity can still flip a human's later post.
- **Near-duplicate search is a local MinHash/LSH (`minhash.py`)**, not `datasketch`: stable crc32 shingles and seeded permutations keep it byte-deterministic, and LSH candidates are verified with exact Jaccard.
- **Simulator text was enriched during M2.** With 10 short templates, 43% of human posts already had a near-duplicate among 20k posts, which made content detection meaningless. Posts are now an opening clause plus connector plus closing clause; the human near-duplicate rate at Jaccard 0.7 is about 0.1%.
- **Round 0 (40 naive bots among 800 humans, budget 5%):** recall 100%, precision 54% (34 humans flagged, 4.2% FPR); at a 1% budget recall is still 100% with 82% precision. Each signal family alone catches all 40, and even a 4-bot farm is caught, so a naive farm is far below the detector's ability and gives the agent no useful gradient. The farm delivered 5,200 engagement to its 3 targets and 0 of it survived flagging.
- **`Verdict.top_feature` is an arbitrary tie-break when several features hit the p-value floor** (alphabetical). Attribute by the full set of features with p at or below `detector.alpha`, as `scripts/round0.py` does.
- **Full suite takes about 2.5 minutes** (large shared fixtures: two 800-human worlds and a bot world).

## Red agent (M4)

- **Run it:** `.venv/Scripts/python scripts/run_agent.py --dry-run` (scripted model, no key, about 50 s) or without `--dry-run` for a live round (Claude Opus 5; needs `ANTHROPIC_API_KEY` or `ant auth login`; spends up to `--max-tokens`, default 400k). Artifacts go to `rounds/<id>/` (git-ignored): sealed `agent-log.jsonl`, replayable `attempt-N.jsonl`, defender-side `summary.md`.
- **What was and was not verified.** Verified: the whole harness with a scripted model, and the real `anthropic` SDK against a mock HTTP transport (request shape, thinking-block round trip, token accounting, refusal). NOT verified: how the live model behaves. No API key or `ant` login existed when this was built, so no live call was ever made. The prompts in `agent/prompts.py` are untested against the real model.
- **Model settings (per the claude-api skill):** `claude-opus-5`, adaptive thinking, `output_config.effort` (default high), top-level `cache_control`, and server-side refusal fallbacks (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`), enabled by default and reported to Garfield. No temperature/top_p/top_k/budget_tokens/prefill (all rejected on Opus 5). The assistant's content is appended back verbatim so thinking blocks stay valid.
- **The tools ARE the possibility space.** The agent can only turn the knobs in `tools.py` / `profile.py` (timing, text mode, typo/style, reaction probability and delay, follow, raw scheduled actions, distinct IPs). It cannot touch humans, the detector, or scores. Bots cannot engage each other's posts (ids do not exist until the attempt runs). Any new evasion tactic requires a new knob, so widen this surface deliberately.
- **A round = several full-window attempts on a reset platform.** The agent is told only which of its accounts were suspended, engagement delivered, and engagement kept (`referee.FEEDBACK_KEYS`, enforced by an assert and a test). The defender's view (precision, recall reported separately, humans flagged) stays in `AttemptOutcome`. Mid-window bans need rolling-window calibration and are M5.
- **Hard budgets** (`sandbox.Budget`, same every round): attempts, accounts, actions per attempt, tokens, wall-clock, model calls per attempt. Token and time checks happen between calls, so overshoot is at most one call.
- **Sealed log:** `SealedLog.read()` raises until the round ends. The seal is an in-process guard; do not open the JSONL file mid-round.
- **Measured possibility space (scripted plans vs the real detector, 40 bots, budget 5%; run-to-run variation in the stealthy row was 9 to 11 of 40):** plain settings: 40/40 suspended, 0 of 33,168 engagement kept. Human-looking text and timing with lockstep reactions: still 40/40 (coordination fires, plus `t_no_sleep`). Reactions at 5% with a 12 h median: about 10/40 suspended, roughly 1,200 of 1,660 engagement kept. Stealth is possible but costs about 95% of the yield, which is the trade-off the agent must discover. `x_shared_oov` still flags a couple of even the stealthy bots.
- **Full suite takes about 3.5 minutes.**

## Workspace Conventions

- Any GitHub repo for this project goes under the `gmcleod-security` org.
- If this becomes a numbered YouTube episode series, check `../youtube-content/youtube-content-calendar.md` for exact repo names and specs before creating anything.
- `../stream-overlay/` holds the stream overlay work. Coordinate with it for any on-screen presentation.
