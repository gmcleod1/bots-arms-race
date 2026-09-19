# Bots Arms Race

A self-contained red team vs. blue team build. I build a bot detection system, then build an autonomous agent that runs an adversarial bot farm against it. The agent chooses its own evasion tactics instead of following a scripted plan. Each round is filmed as the detector and the attacker escalate.

Status: concept stage, decisions made (2026-09-19). Detector v1 uses all three signals, the environment is a sandboxed seeded simulation, and the face-off is simultaneous with live patching. Nothing built yet. See `intent.md` section 12.

## Origin problem

Social platforms, TikTok especially, are flooded with AI bots and bot farms. Farms often run many phones from one location to boost numbers and fake engagement.

Platforms can't just IP-ban them. Bad actors use VPNs and rotating IPs, and some high-traffic locations (hotels) legitimately share one IP across many real users. Network-level blocking is too blunt.

The more promising angle is behavioral detection. A farm can spoof its network identity cheaply, but it is much harder to fake the messy, irregular shape of real human behavior at scale.

Three signal families:

1. **Timing and rhythm.** Humans are irregular. Bots often aren't, even with added jitter.
2. **Coordination.** Accounts that shouldn't know each other moving in lockstep.
3. **Content fingerprinting.** Similar phrasing, cadence, and typos repeated across accounts.

## Why platforms under-detect

The gap is incentives, not technology.

- Bots inflate engagement, DAU, and time on platform, which drive ad revenue and investor confidence.
- Enforcement is expensive. False positives ban real humans and require a well-staffed appeals process, so detection gets tuned conservatively.

Stance: that tradeoff hurts community health. A platform hollowed out by bots loses user trust, and people disengage even from genuine content. A good detector should accept more false positives, paired with a fast, well-funded human appeals process.

## The concept

- **Blue team:** a behavioral bot detector.
- **Red team:** an autonomous agent operating a bot farm. It decides for itself how to evade detection, which keeps the contest unpredictable and closer to how real adversaries behave.
- **Environment:** self-contained. No video hosting, no social platform, no moderation team. Just an escalating cat-and-mouse loop between detector and attacker.
- **Content angle:** spectacular, hard, technical builds that are fun to watch (the cybersecurity take on the Mr. Beast / Linus Tech Tips style).

Where it sits in security: adversarial machine learning, behavioral and anomaly detection, red vs. blue dynamics, and trust and safety (cybersecurity applied to human behavior and content).

## Open questions

- [x] Detector v1 signals: all three (timing, coordination, content fingerprinting). Decided 2026-09-19.
- [x] Agent environment: sandboxed simulation. Decided 2026-09-19.
- [ ] How to score and present it: live dashboard, scoreboard, episodic reveals of each evasion and patch?
- [ ] How much autonomy the agent gets before a manual check-in.
- [ ] Clear win condition, or run indefinitely as ongoing content?

## Shelved idea (for reference)

A TikTok-style recommender for long-form video essays with credibility as a ranking signal. Every upload is shown to 100 real people as a cold-start test, and mass downvotes trigger expert human review instead of auto-removal. Shelved because it stacks AI moderation, an expert review panel, per-upload compute, and long-form video hosting. That is too heavy a cost structure for a solo build.
