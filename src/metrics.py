"""Threshold-aware metrics.

A single accuracy number hides what matters for a screening prototype, so
every report here is expressed at a stated threshold, and the threshold
sweep makes the sensitivity / specificity trade-off explicit.
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ClassificationReport:
    """Metrics computed at one specific decision threshold."""

    threshold: float
    accuracy: float
    sensitivity: float  # recall on malignant; the costly error to miss
    specificity: float  # recall on benign
    precision: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"threshold={self.threshold:.2f}  acc={self.accuracy:.4f}  "
            f"sens={self.sensitivity:.4f}  spec={self.specificity:.4f}  "
            f"prec={self.precision:.4f}  f1={self.f1:.4f}"
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_report(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> ClassificationReport:
    """Confusion matrix and derived rates at ``threshold``."""
    labels = np.asarray(labels).astype(int)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    precision = _safe_divide(tp, tp + fp)
    sensitivity = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * sensitivity, precision + sensitivity)

    return ClassificationReport(
        threshold=float(threshold),
        accuracy=_safe_divide(tp + tn, len(labels)),
        sensitivity=sensitivity,
        specificity=_safe_divide(tn, tn + fp),
        precision=precision,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Equivalent to the Mann-Whitney U statistic normalized by the number of
    positive-negative pairs, with ties given averaged ranks. Returns 0.5
    when one class is absent, since ranking is undefined there.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(probabilities, dtype=float)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    # Average the ranks inside each group of tied scores.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    rank_sum_positive = ranks[labels == 1].sum()
    return float((rank_sum_positive - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sweep_thresholds(
    labels: np.ndarray, probabilities: np.ndarray, steps: int = 19
) -> list[ClassificationReport]:
    """Report metrics across evenly spaced thresholds in ``(0, 1)``."""
    grid = np.linspace(0.05, 0.95, steps)
    return [classification_report(labels, probabilities, t) for t in grid]


def best_threshold_for_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_sensitivity: float = 0.90,
) -> float:
    """Highest specificity among thresholds meeting a sensitivity floor.

    Screening tools are usually tuned to catch positives, accepting more
    false alarms; this picks the least noisy threshold that still clears
    the requested sensitivity, falling back to 0.5 if none does.
    """
    candidates = [
        report
        for report in sweep_thresholds(labels, probabilities, steps=91)
        if report.sensitivity >= min_sensitivity
    ]
    if not candidates:
        return 0.5
    return max(candidates, key=lambda r: r.specificity).threshold
