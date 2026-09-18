"""Official metric helpers for rail corrugation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

RAIL_LABELS = ["Normal", "Side I", "Side II"]


def rail_macro_f1(y_true, y_pred) -> float:
    """Return 3-class macro F1 over the official rail labels."""
    truth = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must have matching non-empty shapes")
    unknown = (set(truth) | set(pred)) - set(RAIL_LABELS)
    if unknown:
        raise ValueError(f"Unknown rail labels: {sorted(unknown)}")
    return float(f1_score(truth, pred, labels=RAIL_LABELS, average="macro", zero_division=0))
