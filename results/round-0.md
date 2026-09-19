# Round 0: detector vs. naive bot farm

Setup: detector calibrated on 800 clean humans over 10 days (seed 101); evaluated on a fresh world of 800 humans (seed 202) with a 40-bot naive farm added (4.8% of accounts). The farm posts 5 canned messages on a fixed 30-minute clock, around the clock, from one IP, and all bots like and follow the same 3 target humans in lockstep.

## Precision and recall at each false-positive budget

| Budget | Recall | Precision | Human FPR | Humans flagged per 10k | Bots caught | Humans flagged |
|---|---|---|---|---|---|---|
| 1% | 100.0% | 81.6% | 1.1% | 112 | 40/40 | 9/800 |
| 2% | 100.0% | 71.4% | 2.0% | 200 | 40/40 | 16/800 |
| 5% | 100.0% | 54.1% | 4.2% | 425 | 40/40 | 34/800 |
| 10% | 100.0% | 36.4% | 8.8% | 875 | 40/40 | 70/800 |

Ranking quality (average precision over the score): 0.833

## Which signal family catches the farm (budget 5%, each family alone)

| Family | Bots caught | Humans flagged |
|---|---|---|
| timing | 40/40 | 13/800 |
| coordination | 40/40 | 4/800 |
| content | 40/40 | 19/800 |

Signal families that fire per caught bot: {3: 40} (key = number of families, value = bots)
Features that fired (p at or below its budget share), bots per feature: {'t_regularity': 40, 't_no_sleep': 40, 'c_coaction_max': 40, 'c_lockstep_degree': 40, 'x_dup_rate': 40}
(Several features reach the same p-value floor on these bots, so one 'top feature' would be an arbitrary tie-break; the list above is the honest attribution.)

## What the farm achieved

- Engagement delivered to its 3 targets (likes plus follows): 5,200
- Delivered by bots that were never flagged: 0 (0.0%)
- Target humans wrongly flagged because of the boost: 0 of 3

## Detection as the farm gets smaller (budget 5%)

| Bots | Caught | Recall |
|---|---|---|
| 4 | 4/4 | 100% |
| 8 | 8/8 | 100% |
| 16 | 16/16 | 100% |
| 40 | 40/40 | 100% |
