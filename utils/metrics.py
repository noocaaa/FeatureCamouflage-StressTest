"""
utils/metrics.py  —  Evaluation metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def fix_scores(scores):
    """Flatten scores: some PyGOD detectors return (N, 2) where col -1 is the anomaly score."""
    scores = np.asarray(scores)
    if scores.ndim == 2:
        scores = scores[:, -1]
    return scores

def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Return AUC-ROC and Average Precision. NaN if only one class present."""
    scores = fix_scores(scores)
    if len(np.unique(y_true)) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    return {
        "auc": roc_auc_score(y_true, scores),
        "ap":  average_precision_score(y_true, scores),
    }