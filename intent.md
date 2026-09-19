# Intent: Bots Arms Race

> Outline only. Every `✍️` is yours to write. The prompts and pointers are there to steer you, not to be answered by Claude. Write badly first, tighten later. Two passes: (1) fill it for yourself, (2) rewrite the parts you'd show a hiring manager or put in a video description.

Status: outline, 2026-09-19. Prose sections still yours to fill. Decisions 6.1, 6.2 and 5a are logged in section 12; 6.3 to 6.5 are proposed, not yet accepted.

---

## 1. One-sentence intent

✍️ Finish this: *"I'm building ______ to prove ______ so that ______."*

Working draft to react to (rewrite in your voice): *I build a behavioral bot detector, then face an autonomous AI agent, on camera, that tries to get a bot farm past it.*

Prompt: if a hiring manager read only this line, what would they remember? Try it in three versions (technical, plain-English, YouTube title) and keep the strongest.

## 2. Why this project, for me

The point of this file. Be specific and honest, not aspirational.

✍️ Fill in the table. "Evidence" is the artifact someone can look at, not a claim.

| Skill I want to build or show | Where it shows up in this project | Evidence (repo, write-up, video, metric) |
|---|---|---|
| Behavioral detection / anomaly detection | | |
| Detection engineering (rules → measured performance) | | |
| Adversarial thinking / red-team autonomy | | |
| Threat hunting mindset (hypothesis → data → finding) | | |
| Measurement (precision/recall, not accuracy) | | |
| Python / data work (Jupyter, simulation) | | |
| Communicating technical work (video, writing) | | |
| Lead-level: scoping, tradeoffs, decisions under uncertainty | | |

Pointer: reread the JD mapping in `../youtube-content/threat-hunter-track.md` and pick the 3 to 4 rows where this project is your *strongest* evidence. Cut the rest. A project that claims everything proves nothing.

## 3. Career positioning

✍️ Answer each in 2 to 4 sentences:

- What role am I aiming at with this (Lead Threat Hunter or similar), and what gap in my current story does this close?
- What does this project show that malware analysis and DFIR work doesn't already show?
- What would a skeptical interviewer ask about it, and what's my answer? (Write 3 questions. E.g. "Isn't this just a toy on synthetic data?")
- Which 2 or 3 resume/LinkedIn bullets should exist when this is done? Write them now as targets, with placeholders for numbers.

## 4. The problem and my thesis

✍️ In your own words, no copying from the README:

- The problem (bot farms, why IP blocking fails). Aim for 5 sentences max.
- The thesis: why behavior is the right layer to detect on.
- Where the thesis could be wrong. (Best hiring signal in the whole file: name the weakness before someone else does.)

Pointers to sharpen it:
- Industry term to learn and use: **coordinated inauthentic behavior**. Search it, plus "coordinated behavior detection" (e.g. Pacheco et al., *Uncovering Coordinated Networks on Social Media*).
- **OWASP Automated Threats to Web Applications** (OAT handbook): a standard vocabulary for bot abuse. Map your farm's behaviors to it.
- **MITRE ATLAS**: ATT&CK's counterpart for adversarial ML. Relevant to the red agent's evasion.
- Look at how existing bot-scoring work frames features (Botometer / OSoMe from Indiana University) so you can say what's different about your approach.

## 5. Stance and constraints (already decided, so defend them)

These come from the README and `CLAUDE.md`. ✍️ For each, write *why* in one or two sentences, as if arguing with someone who disagrees.

- Behavioral signals only. IP/VPN blocking is out of scope. Why?
- Bias toward false positives, paired with a fast appeals process. Why, and what does it cost?
- The red agent is autonomous. Its tactics are never hardcoded. Why does that matter for the results being credible?
- Self-contained, simulated environment with known ground truth. Why not test on a real platform?

Pointer: the false-positive stance is your most opinionated position. Interviewers will probe it. Have a numeric answer ready (e.g. "at X% FPR, Y humans per 10k get flagged, and appeals must clear in Z hours").

## 5a. The face-off (core format)

The show is a head-to-head: I build and run the bot detector, and an autonomous AI agent tries to get its bot farm past it, on camera. That makes me a participant, not a narrator, and it changes what has to be true.

✍️ Answer each:

- **Who acts when?** Do I ship a detector, then the agent gets a round, then I patch, and so on (turn-based)? Or do we run at the same time with live patching? What's the rule that stops me from reading the agent's logs and hard-fixing exactly what it did?
- **What's fair?** What does each side know about the other? Do I see the agent's tactics only through the detector's data, the same way a real defender would? Does the agent see anything beyond the verdicts it gets back?
- **Resources.** Time box, compute or token budget, and number of attempts per round. Write the same limit for both sides, or write down why they differ.
- **Live or recorded?** What breaks on camera when an agent is autonomous and slow or unpredictable? What's the fallback if a round stalls or the agent does something dull?
- **What can the audience follow?** A viewer can't read agent logs in real time. What's the one number or visual that shows who's winning at a glance? (Ties into 6.3.)
- **What if I lose?** Losing rounds is good content, but only if the reason is instructive. Write down what you'd want the viewer to learn from an agent win.
- **Where do I add value as the human?** The detector is your work and the agent is the opponent. What's the human-driven part (choosing signals, interpreting a miss, patching) that you'd want a hiring manager to watch you do?

Pointers:
- This is a purple-team-style exercise with a defender under time pressure. Look at how CTF attack/defense formats score both sides (service uptime plus successful attacks) for scoring ideas.
- Decide early whether the agent can be *replayed* deterministically (seeded simulator, logged actions). Reproducibility is what lets you say "here is exactly how it beat v2" without hand-waving.

## 6. Open decisions (these gate all code)

Per `CLAUDE.md`, no code directories until decisions 6.1 and 6.2 are answered. For each: ✍️ list options, pick one, write the tradeoff you accepted, and date it.

### 6.1 Detector v1 signals
Options: timing/rhythm, cross-account coordination, content fingerprinting, or a mix.
Questions to answer:
- Which signal is easiest to get a *real* result from on simulated data, and which is easiest to fool?
- Does starting with one signal give a better story (clear before/after per escalation) than starting with all three?
- What features would each signal actually compute? (e.g. inter-event time entropy, burstiness, graph similarity between accounts, n-gram or edit-distance overlap)

### 6.2 Environment
Options: sandboxed simulation (the default per `CLAUDE.md`), real small platform, an existing service's public surface.
Questions to answer:
- What does the simulator need to model for the signals in 6.1 to mean anything? (Session structure, sleep schedules, social graph, content generation.)
- How do I make the "human" population realistic enough that beating the detector isn't trivial? This is the hardest and most credible part of the project.
- How do I keep ground-truth labels honest?

Note: pointing the agent at a real platform or a third party's public surface needs your explicit decision first. Write that decision down here, even if it's "no".

### 6.3 Scoring and presentation
Options: live dashboard, scoreboard, episodic reveals.
Coordinate with `../stream-overlay/` if any of this goes on screen.

### 6.4 Agent autonomy
- What can the agent do with no check-in? What forces a stop? (Budget, number of rounds, a category of action.) In a face-off, check-ins also happen on camera, so decide which ones are part of the show.
- What does it observe about the detector? (Score only? Verdicts? Nothing?) This choice defines how realistic the adversary is.

### 6.5 Win condition
Fixed end, or runs indefinitely as content? What would "the attacker wins" or "the detector wins" mean in measurable terms?

## 7. Measurement plan

✍️ Define success before building anything.

- Precision and recall reported *separately*, per round. Why never just accuracy?
- Which curve or table goes in every episode? (Precision-recall curve is a good default with heavy class imbalance.)
- What's the cost of a false positive vs. a missed bot in this system, in your own units?
- How will you avoid fooling yourself? (Detector tuned on the same data the agent learned from, leakage between rounds, the agent overfitting to one detector version.)

Pointer: read up on **Goodhart's law** and **adversarial robustness evaluation** (adaptive attackers are the standard, not fixed ones). Being able to explain why your evaluation is honest is a lead-level skill.

## 8. Safety and scope

✍️ One paragraph each:

- What can the agent touch? What is it explicitly forbidden from touching?
- What would make you stop or shelve the project?
- Anything in the write-up or video that could be a how-to for running a real bot farm, and how you'll handle that. (Show the detection and the *pattern* of evasion, not a deployable tool.)

## 9. Audience and marketing

✍️ Who is this for, and what should each one walk away with?

| Audience | What they should think after | Where they'll see it |
|---|---|---|
| Hiring managers / hunt leads | | GitHub README, LinkedIn |
| Security practitioners | | YouTube, write-ups |
| General tech audience | | YouTube, shorts |
| Trust & safety / ML people | | Blog post, conference talk idea |

Questions:
- What's the one image or moment that makes someone click? (The detector catching something the agent thought it hid? The agent inventing a tactic you didn't expect?)
- What's the episode structure? (Round N: detector state → agent adapts → what got through → patch.)
- Does this fit the numbered series in `../youtube-content/youtube-content-calendar.md`, or is it standalone? Check before naming any repo under `gmcleod-security`.

## 10. Milestones

✍️ Fill in after section 6 is decided. Suggested shape, but change it:

1. Decisions 6.1 to 6.5 answered and dated in this file.
2. Simulator with labeled humans, before any bot exists.
3. Detector v1 on the humans and a *naive* bot, with a measured baseline.
4. Red agent v1. First round, first result.
5. Escalation rounds, each with a metrics snapshot.
6. Public write-up and video.

For each milestone: what artifact proves it's done?

## 11. Reading and reference list

Keep adding. Put a one-line takeaway next to each, not just a link.

- ✍️ OWASP OAT handbook:
- ✍️ MITRE ATLAS:
- ✍️ Coordinated behavior detection paper(s):
- ✍️ Timing/keystroke/behavioral biometrics work:
- ✍️ Adversarial ML evaluation (adaptive attacks):
- ✍️ Anything from real platform integrity / trust & safety reports:

## 12. Decision log

Append-only. Date every entry.

| Date | Decision | Alternatives rejected | Why |
|---|---|---|---|
| 2026-09-19 | 6.1: Detector v1 uses all three signals (timing, coordination, content fingerprinting) | Timing only (recommended by Claude for scope); timing + coordination | ✍️ |
| 2026-09-19 | 6.2: Sandboxed, seeded simulation with ground-truth labels. Agent never touches a real platform or third-party surface | Tiny self-hosted real platform | ✍️ |
| 2026-09-19 | 5a: Face-off is simultaneous with live detector patching. Fallback: `--pause-edits` turns a round turn-based | Turn-based rounds (recommended by Claude for fairness and replay) | ✍️ |
| 2026-09-19 | Scope: full build roadmap, simulator through video. Plan: `C:\Users\netge\.claude\plans\pasted-content-id-0be3-bots-arms-smooth-scone.md` | Decisions-only; decisions + simulator only | ✍️ |

| 2026-09-19 | M4 built on the plan's PROPOSED 6.4 defaults, not separately confirmed by you (you said "do m4"): no check-in inside a round; the agent observes only which of its own accounts were suspended and how much engagement survived; hard budgets (attempts, accounts, actions, tokens, wall-clock, model calls). One deviation: a round is several full-window attempts on a reset platform, with verdicts per attempt, instead of delayed mid-window bans, so the detector always scores the window length it was calibrated on | Mid-window delayed bans (needs rolling-window calibration, M5) | ✍️ |

Still open, with proposed defaults from the plan (accept or change, then log): 6.3 live scoreboard plus episodic reveals; 6.4 as built in M4 (see the row above; confirm or change); 6.5 measurable win condition with thresholds X, R, F set after the M3 baseline, fixed round count.
