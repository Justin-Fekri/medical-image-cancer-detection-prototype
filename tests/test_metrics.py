"""Metric correctness, checked against cases with a known answer."""

import numpy as np

from src.metrics import (
    auroc,
    best_threshold_for_sensitivity,
    classification_report,
    sweep_thresholds,
)


def test_auroc_is_one_for_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(labels, scores) == 1.0


def test_auroc_is_zero_when_ranking_is_inverted():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert auroc(labels, scores) == 0.0


def test_auroc_is_half_for_a_constant_score():
    # Every score tied means the ranking carries no information.
    labels = np.array([0, 1, 0, 1])
    scores = np.full(4, 0.5)
    assert auroc(labels, scores) == 0.5


def test_auroc_handles_a_single_class():
    assert auroc(np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9])) == 0.5


def test_confusion_matrix_matches_hand_computation():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.1, 0.05])

    report = classification_report(labels, scores, threshold=0.5)

    assert (report.true_positives, report.false_negatives) == (3, 1)
    assert (report.false_positives, report.true_negatives) == (1, 3)
    assert report.sensitivity == 0.75
    assert report.specificity == 0.75
    assert report.accuracy == 0.75
    assert report.precision == 0.75


def test_lowering_the_threshold_cannot_reduce_sensitivity():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)

    sensitivities = [r.sensitivity for r in sweep_thresholds(labels, scores)]
    assert sensitivities == sorted(sensitivities, reverse=True)


def test_suggested_threshold_meets_the_sensitivity_floor():
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.linspace(0.0, 0.6, 50), np.linspace(0.4, 1.0, 50)])

    threshold = best_threshold_for_sensitivity(labels, scores, min_sensitivity=0.90)
    assert classification_report(labels, scores, threshold).sensitivity >= 0.90


def test_suggested_threshold_falls_back_when_floor_is_unreachable():
    # The one positive scores 0.01, below every threshold the sweep considers,
    # so no candidate reaches the floor and the function returns its default.
    labels = np.array([0, 1])
    scores = np.array([0.9, 0.01])
    assert best_threshold_for_sensitivity(labels, scores, min_sensitivity=0.99) == 0.5
