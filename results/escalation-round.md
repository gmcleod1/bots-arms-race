# M6 escalation round: naive to stealthy, with a live patch

Recorded with `scripts/escalation_round.py` (scripted agent, no API key). Same worlds as round 0
(ref seed 101, eval seed 202; 800 humans x 10 days), so these numbers are directly comparable to
`results/round-0.md`. Round id `20260924-114236-escalation`; replayed clean with
`scripts/replay_round.py` (`REPRODUCED`, all 4 attempts matched).

Win condition (confirmed as proposed): the farm wins an attempt by keeping at least 10% of round
0's naive-farm engagement on its top 3 targets (3,317 of 33,169); the detector forfeits any
attempt where it flags more than 10% of humans. Detector starts on v1 (the default: no
`target_overlap`); the defender patches `target_overlap=true` live at the start of attempt 3.

| Attempt | Plan | Detector | Bots caught | Recall | Precision | Humans flagged | Engagement kept | Winner |
|---|---|---|---|---|---|---|---|---|
| 1 | A — naive, fixed clock, canned text, lockstep likes | v1 | 40/40 | 100.0% | 53.3% | 35 | 0 / 33,169 | detector |
| 2 | E — human text + timing, reactions still lockstep | v1 | 40/40 | 100.0% | 50.0% | 40 | 0 / 32,975 | detector |
| 3 | G — reactions at 15%, 6 h median | v2 (`target_overlap` on) | 38/40 | 95.0% | 53.5% | 33 | 229 / 4,816 | detector |
| 4 | H — reactions at 5%, 12 h median | v2 | 8/40 | 20.0% | 19.0% | 34 | 1,279 / 1,614 | detector |

Round winner: detector, 4-0 by the win condition. But attempt 4 is the real finding, not a clean
defender victory.

## What actually happened at the finale

The win condition only asks whether the farm kept 10% of round 0's engagement on its top 3
targets. By that yardstick the detector "won" every attempt, including attempt 4. But attempt 4's
detection quality collapsed: recall fell from 95% (attempt 3) to 20%, precision to 19% — worse
than round 0's naive baseline (54%) despite `target_overlap` being on. Plan H's bots spread their
5% reaction rate over 65-196 distinct targets each, which dilutes each pair's cosine overlap to
0.06-0.11 — below the human ceiling — so the new feature that closed the slow-lockstep gap in
attempt 3 cannot see this farm at all, and it cost a little of every other feature's budget
getting there (Bonferroni share shrinks from `budget/6` to `budget/7`). The farm kept 1,279 of
1,614 engagement delivered; it just didn't reach the 3,317-engagement bar on a 40-bot attempt.

The honest read: the round's win condition measures engagement denied, not detection accuracy,
and at the finale those two diverge. A slightly larger or slightly more patient farm running
Plan H's shape would clear 3,317 without the detector improving at all. That gap — bots that
dilute their target-overlap footprint below any human's — is unsolved going into the next
episode.

## Reproduce

```
.venv/Scripts/python scripts/escalation_round.py
.venv/Scripts/python scripts/replay_round.py rounds/<round-id>
```
