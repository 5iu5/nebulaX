"""Reusable SHM inference without CLI or Streamlit dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from .data import load_signal
from .features import cumulative_damage, rainflow_cycles
from .models import DEFAULT_METADATA_PATH, PhysicsDamageModel


@dataclass
class DamagePrediction:
    file_id: str
    prediction: float
    cycles: pd.DataFrame
    cumulative: pd.DataFrame


def predict_source(
    source: str | Path | BinaryIO | TextIO,
    *,
    file_id: str | None = None,
    model: PhysicsDamageModel | None = None,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> DamagePrediction:
    """Predict one CSV source and retain plot-ready rainflow details."""
    fitted = model or PhysicsDamageModel.load(metadata_path)
    signal = load_signal(source)
    cycles = rainflow_cycles(signal)
    prediction = fitted.predict_cycles(cycles)
    name = file_id or Path(str(source)).name
    curve = cumulative_damage(cycles, m=fitted.m, c=fitted.c, signal_length=len(signal))
    return DamagePrediction(file_id=name, prediction=prediction, cycles=cycles, cumulative=curve)


def predict_paths(paths: list[Path], *, metadata_path: str | Path = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """Predict a collection of files in deterministic input order."""
    model = PhysicsDamageModel.load(metadata_path)
    rows = []
    for path in paths:
        result = predict_source(path, file_id=path.name, model=model)
        rows.append({"file_id": result.file_id, "prediction": result.prediction})
    return pd.DataFrame(rows, columns=["file_id", "prediction"])


def validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected_file_ids: set[str] | None = None,
    expected_rows: int | None = None,
) -> None:
    """Validate the exact SHM submission schema and values."""
    if list(predictions.columns) != ["file_id", "prediction"]:
        raise ValueError("SHM output columns must be exactly: file_id,prediction")
    if predictions.empty or predictions.isna().any().any():
        raise ValueError("SHM output must be non-empty and contain no missing values")
    if predictions["file_id"].duplicated().any():
        raise ValueError("SHM output contains duplicate file_id values")
    if not predictions["file_id"].map(lambda value: isinstance(value, str) and Path(value).suffix.lower() == ".csv").all():
        raise ValueError("Every file_id must include its .csv extension")
    numeric = pd.to_numeric(predictions["prediction"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.any(numeric <= 0):
        raise ValueError("Every prediction must be one finite positive float")
    if expected_rows is not None and len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(predictions)}")
    if expected_file_ids is not None and set(predictions["file_id"]) != expected_file_ids:
        missing = sorted(expected_file_ids - set(predictions["file_id"]))
        extra = sorted(set(predictions["file_id"]) - expected_file_ids)
        raise ValueError(f"Output file IDs do not match inputs; missing={missing}, extra={extra}")
