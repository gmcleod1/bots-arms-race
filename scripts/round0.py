"""Round 0: the detector against a naive bot farm. Writes results/round-0.md.

Run from the project root:  .venv/Scripts/python scripts/round0.py
Every number in the report is measured here, none is typed in.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.detector import Detector  # noqa: E402
from detector.metrics import average_precision, confusion  # noqa: E402
from simulator.bots import NaiveFarmConfig, engagement_delivered, naive_farm  # noqa: E402
from simulator.platform import Platform  # noqa: E402
from simulator.world import generate_world  # noqa: E402

REF_SEED, EVAL_SEED, FARM_SEED = 101, 202, 1
N_HUMANS, DAYS, BUDGET = 800, 10, 0.05
FAMILIES = {"t_": "timing", "c_": "coordination", "x_": "content"}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def main() -> None:
    ref = generate_world(REF_SEED, N_HUMANS, DAYS)
    world = generate_world(EVAL_SEED, N_HUMANS, DAYS)
    detector = Detector(fpr_budget=BUDGET)
    detector.fit(ref.events)
    features = len(detector.features)

    def run(n_bots: int):
        platform = Platform(world)
        farm = naive_farm(platform, np.random.default_rng(FARM_SEED), NaiveFarmConfig(n_bots=n_bots))
        events = platform.events_until(10**9)
        return platform, farm, events, detector.score(events)

    def flagged_at(verdicts, budget, only=None):
        alpha = budget / features
        return {
            a for a, v in verdicts.items()
            if any(p <= alpha for f, p in v.pvalues.items() if only is None or f.startswith(only))
        }

    platform, farm, events, verdicts = run(40)
    truth = platform.ground_truth
    lines = [
        "# Round 0: detector vs. naive bot farm",
        "",
        f"Setup: detector calibrated on {N_HUMANS} clean humans over {DAYS} days (seed {REF_SEED}); "
        f"evaluated on a fresh world of {N_HUMANS} humans (seed {EVAL_SEED}) with a {len(farm.accounts)}-bot "
        f"naive farm added ({len(farm.accounts) / (N_HUMANS + len(farm.accounts)):.1%} of accounts). "
        f"The farm posts {len(farm.canned)} canned messages on a fixed {NaiveFarmConfig().post_interval // 60}-minute "
        f"clock, around the clock, from one IP, and all bots like and follow the same "
        f"{len(farm.targets)} target humans in lockstep.",
        "",
        "## Precision and recall at each false-positive budget",
        "",
        "| Budget | Recall | Precision | Human FPR | Humans flagged per 10k | Bots caught | Humans flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in (0.01, 0.02, 0.05, 0.10):
        c = confusion(flagged_at(verdicts, b), truth)
        lines.append(
            f"| {b:.0%} | {pct(c.recall)} | {pct(c.precision)} | {pct(c.fpr)} | "
            f"{c.humans_flagged_per_10k:,.0f} | {c.tp}/{c.tp + c.fn} | {c.fp}/{c.fp + c.tn} |"
        )
    scores = {a: v.score for a, v in verdicts.items()}
    lines += [
        "",
        f"Ranking quality (average precision over the score): {average_precision(scores, truth):.3f}",
        "",
        f"## Which signal family catches the farm (budget {BUDGET:.0%}, each family alone)",
        "",
        "| Family | Bots caught | Humans flagged |",
        "|---|---|---|",
    ]
    for prefix, name in FAMILIES.items():
        fl = flagged_at(verdicts, BUDGET, only=prefix)
        lines.append(
            f"| {name} | {len(fl & set(farm.accounts))}/{len(farm.accounts)} | "
            f"{len(fl - set(farm.accounts))}/{N_HUMANS} |"
        )
    caught = flagged_at(verdicts, BUDGET) & set(farm.accounts)
    fam_count = Counter(
        len({FAMILIES[f[:2]] for f, p in verdicts[a].pvalues.items() if p <= detector.alpha}) for a in caught
    )
    lines += [
        "",
        f"Signal families that fire per caught bot: {dict(sorted(fam_count.items()))} "
        f"(key = number of families, value = bots)",
        "Features that fired (p at or below its budget share), bots per feature: "
        f"{dict(Counter(f for a in caught for f, p in verdicts[a].pvalues.items() if p <= detector.alpha))}",
        "(Several features reach the same p-value floor on these bots, so one 'top feature' "
        "would be an arbitrary tie-break; the list above is the honest attribution.)",
    ]
    total = engagement_delivered(events, farm)
    survivors = [a for a in farm.accounts if a not in caught]
    kept = engagement_delivered(events, farm, by=survivors)
    targets_flagged = [t for t in farm.targets if t in flagged_at(verdicts, BUDGET)]
    lines += [
        "",
        "## What the farm achieved",
        "",
        f"- Engagement delivered to its {len(farm.targets)} targets (likes plus follows): {total:,}",
        f"- Delivered by bots that were never flagged: {kept:,} ({kept / total:.1%})",
        f"- Target humans wrongly flagged because of the boost: {len(targets_flagged)} of {len(farm.targets)}",
        "",
        "## Detection as the farm gets smaller (budget 5%)",
        "",
        "| Bots | Caught | Recall |",
        "|---|---|---|",
    ]
    for n in (4, 8, 16, 40):
        p2, f2, _, v2 = run(n)
        got = len(flagged_at(v2, BUDGET) & set(f2.accounts))
        lines.append(f"| {n} | {got}/{n} | {got / n:.0%} |")
    out = Path(__file__).resolve().parents[1] / "results" / "round-0.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
