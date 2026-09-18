"""Official SHM MAPE-derived metric."""

from __future__ import annotations

import numpy as np


def shm_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute percentage error as a fraction."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must have matching non-empty shapes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be finite")
    if np.any(truth == 0):
        raise ValueError("MAPE is undefined for zero targets")
    return float(np.mean(np.abs(truth - prediction) / np.abs(truth)))


def shm_mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the official ``max(0, 1 - MAPE)`` score."""
    return max(0.0, 1.0 - shm_mape(y_true, y_pred))
