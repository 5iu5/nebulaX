"""Model loading and training helpers for rail corrugation."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import extract_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("rail_model.pkl")
SEED = 42


def build_model() -> RandomForestClassifier:
    """Return the model configuration selected in the rail notebooks."""
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED)


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Rail model artifact not found: {model_path}")
    return joblib.load(model_path)


def save_model(model, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def fit_from_recordings(recordings: list[tuple[str, pd.DataFrame, str]]):
    rows = []
    labels = []
    for _filename, data, label in recordings:
        rows.append(extract_features(data))
        labels.append(label)
    X = pd.DataFrame(rows)
    model = build_model()
    model.fit(X, labels)
    return model
