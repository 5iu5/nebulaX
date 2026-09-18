"""Inference utilities for rail corrugation classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from .data import csv_files, read_recording
from .features import extract_features
from .metrics import RAIL_LABELS
from .models import DEFAULT_MODEL_PATH, load_model


@dataclass(frozen=True)
class RailPrediction:
    file_id: str
    prediction: str
    probabilities: dict[str, float]
    features: dict[str, float]

    @property
    def confidence(self) -> float:
        return max(self.probabilities.values()) if self.probabilities else float("nan")


def _features_for_model(features: dict[str, float], model) -> pd.DataFrame:
    frame = pd.DataFrame([features])
    names = list(getattr(model, "feature_names_in_", frame.columns))
    for name in names:
        if name not in frame.columns:
            frame[name] = 0.0
    return frame[names].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def predict_dataframe(data: pd.DataFrame, *, file_id: str, model=None) -> RailPrediction:
    estimator = load_model() if model is None else model
    features = extract_features(data)
    X = _features_for_model(features, estimator)
    prediction = str(estimator.predict(X)[0])
    probabilities: dict[str, float]
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)[0]
        probabilities = {str(label): float(prob) for label, prob in zip(estimator.classes_, probs, strict=False)}
        for label in RAIL_LABELS:
            probabilities.setdefault(label, 0.0)
    else:
        probabilities = {label: 0.0 for label in RAIL_LABELS}
        probabilities[prediction] = 1.0
    return RailPrediction(file_id=file_id, prediction=prediction, probabilities=probabilities, features=features)


def predict_source(source: str | Path | BinaryIO, *, file_id: str | None = None, model=None) -> RailPrediction:
    data = read_recording(source)
    name = file_id or getattr(source, "name", "uploaded.csv")
    return predict_dataframe(data, file_id=Path(str(name)).name, model=model)


def predict_path(path: str | Path, *, model=None) -> RailPrediction:
    path = Path(path)
    return predict_source(path, file_id=path.name, model=model)


def predict_paths(paths: list[str | Path], *, model_path: str | Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    model = load_model(model_path)
    rows = []
    for path in paths:
        result = predict_path(path, model=model)
        rows.append({"file_id": result.file_id, "prediction": result.prediction})
    return pd.DataFrame(rows, columns=["file_id", "prediction"])


def predict_directory(path: str | Path, *, model_path: str | Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    return predict_paths(csv_files(path), model_path=model_path)


def validate_predictions(predictions: pd.DataFrame, *, expected_file_ids: set[str] | None = None, expected_rows: int | None = None) -> None:
    if list(predictions.columns) != ["file_id", "prediction"]:
        raise ValueError("rail_predictions.csv must have columns exactly: file_id,prediction")
    if expected_rows is not None and len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} predictions, got {len(predictions)}")
    if predictions["file_id"].isna().any() or predictions["prediction"].isna().any():
        raise ValueError("Predictions contain missing values")
    bad = sorted(set(predictions["prediction"]) - set(RAIL_LABELS))
    if bad:
        raise ValueError(f"Invalid rail labels: {bad}")
    if expected_file_ids is not None and set(predictions["file_id"]) != expected_file_ids:
        missing = sorted(expected_file_ids - set(predictions["file_id"]))
        extra = sorted(set(predictions["file_id"]) - expected_file_ids)
        raise ValueError(f"File ID mismatch; missing={missing}, extra={extra}")
