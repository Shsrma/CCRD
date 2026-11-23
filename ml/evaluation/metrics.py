"""
Comprehensive evaluation metrics module for fraud detection models.
Calculates threshold metrics, curve metrics, and probability calibration scores.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
)


def evaluate_fraud_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for binary fraud classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Hard binary predictions at threshold.
        y_prob: Predicted fraud probabilities [0.0, 1.0].

    Returns:
        Dictionary of formatted metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    # Rate calculations
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "matthews_corrcoef": round(mcc, 4),
        "brier_score": round(brier, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "total_evaluated": int(len(y_true)),
        "actual_fraud_count": int(np.sum(y_true)),
        "predicted_fraud_count": int(np.sum(y_pred)),
    }
