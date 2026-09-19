"""Metrics report precision and recall separately, never accuracy.

With a few bots among thousands of humans, "accuracy" rewards a detector that
flags nothing. Every number here is chosen to keep the false-positive stance
(more flagged humans, paired with appeals) visible.
"""
import math

import pytest

from detector.metrics import Confusion, average_precision, confusion, pr_curve

TRUTH = {"h1": "human", "h2": "human", "h3": "human", "h4": "human", "b1": "bot", "b2": "bot"}


def test_confusion_counts_and_rates():
    c = confusion({"b1", "h1"}, TRUTH)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 3)
    assert c.precision == 0.5 and c.recall == 0.5
    assert c.fpr == 0.25
    assert c.humans_flagged_per_10k == 2500.0


def test_no_accuracy_field_exists():
    assert not hasattr(Confusion, "accuracy")
    assert not hasattr(confusion(set(), TRUTH), "accuracy")


def test_flagging_nothing_gives_undefined_precision_not_a_perfect_score():
    c = confusion(set(), TRUTH)
    assert c.precision is None and c.recall == 0.0 and c.fp == 0


def test_flagging_everything_gives_full_recall_and_bad_precision():
    c = confusion(set(TRUTH), TRUTH)
    assert c.recall == 1.0 and c.precision == pytest.approx(2 / 6) and c.fpr == 1.0


def test_flagged_accounts_missing_from_truth_are_an_error():
    with pytest.raises(KeyError):
        confusion({"ghost"}, TRUTH)


def test_pr_curve_orders_by_score_and_ties_move_together():
    scores = {"b1": 3.0, "h1": 2.0, "b2": 2.0, "h2": 0.5}  # h3, h4 unscored -> 0
    curve = pr_curve(scores, TRUTH)
    assert [t for t, _, _ in curve] == [3.0, 2.0, 0.5, 0.0]
    assert curve[0][1:] == (1.0, 0.5)  # only b1 flagged
    assert curve[1][1:] == (pytest.approx(2 / 3), 1.0)  # h1 and b2 tie, enter together
    assert curve[-1][2] == 1.0


def test_average_precision_perfect_and_worst():
    perfect = {"b1": 9.0, "b2": 8.0, "h1": 1.0}
    worst = {"h1": 9.0, "h2": 8.0, "h3": 7.0, "h4": 6.0, "b1": 1.0, "b2": 0.5}
    assert average_precision(perfect, TRUTH) == pytest.approx(1.0)
    assert average_precision(worst, TRUTH) < 0.4
    assert not math.isnan(average_precision({}, TRUTH))
