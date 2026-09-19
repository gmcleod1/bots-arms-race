"""Evaluation metrics. Ground truth lives here, on the scoring side, never in the detector.

Precision and recall are always reported separately, and there is deliberately
no accuracy: with a few bots among thousands of humans it rewards flagging nothing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Confusion:
    tp: int  # bots flagged
    fp: int  # humans flagged (each one needs an appeals process)
    fn: int  # bots missed
    tn: int  # humans left alone

    @property
    def precision(self) -> float | None:
        """None when nothing was flagged: undefined, not perfect."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        bots = self.tp + self.fn
        return self.tp / bots if bots else None

    @property
    def fpr(self) -> float | None:
        humans = self.fp + self.tn
        return self.fp / humans if humans else None

    @property
    def humans_flagged_per_10k(self) -> float | None:
        return None if self.fpr is None else self.fpr * 10_000


def confusion(flagged: set[str], truth: dict[str, str]) -> Confusion:
    unknown = flagged - truth.keys()
    if unknown:
        raise KeyError(f"flagged accounts missing from ground truth: {sorted(unknown)[:3]}")
    tp = sum(1 for a in flagged if truth[a] == "bot")
    fp = len(flagged) - tp
    bots = sum(1 for v in truth.values() if v == "bot")
    return Confusion(tp=tp, fp=fp, fn=bots - tp, tn=(len(truth) - bots) - fp)


def pr_curve(scores: dict[str, float], truth: dict[str, str]) -> list[tuple[float, float | None, float | None]]:
    """(threshold, precision, recall) flagging every account with score >= threshold.

    Accounts with no score count as 0. Tied scores enter together, so the curve never
    pretends to order accounts the detector could not tell apart.
    """
    full = {a: scores.get(a, 0.0) for a in truth}
    points = []
    for threshold in sorted(set(full.values()), reverse=True):
        c = confusion({a for a, s in full.items() if s >= threshold}, truth)
        points.append((threshold, c.precision, c.recall))
    return points


def average_precision(scores: dict[str, float], truth: dict[str, str]) -> float:
    """Area under the precision-recall curve (step-wise). 0.0 if there are no bots."""
    total, prev_recall = 0.0, 0.0
    for _, precision, recall in pr_curve(scores, truth):
        if recall is None or precision is None:
            continue
        total += precision * (recall - prev_recall)
        prev_recall = recall
    return total
