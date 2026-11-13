"""Metrics and threshold-selection helpers for imbalanced anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Container for imbalance-aware anomaly metrics."""

    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float


def select_percentile_threshold(normal_scores: np.ndarray, percentile: float = 99.0) -> float:
    """Return a high-tail threshold learned from normal-only anomaly scores."""

    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    scores = np.asarray(normal_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("normal_scores cannot be empty")
    return float(np.percentile(scores, percentile))


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Compute metrics appropriate for rare-event anomaly detection.

    anomaly_scores should be larger for more anomalous observations.
    PR-AUC is emphasized because accuracy can be misleading under severe class imbalance.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if anomaly_scores is None:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        anomaly_scores = np.asarray(anomaly_scores, dtype=float)
        pr_auc = average_precision_score(y_true, anomaly_scores)
        roc_auc = roc_auc_score(y_true, anomaly_scores)

    return EvaluationResult(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pr_auc=float(pr_auc),
        roc_auc=float(roc_auc),
    )


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Pretty-print the per-class report without hiding zero-division cases."""

    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
