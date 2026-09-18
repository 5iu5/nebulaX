"""End-to-end segmentation and classification for the Door subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "door_final_model.joblib"
DEFAULT_FEATURE_PATH = PACKAGE_ROOT / "door_final_features.csv"
GAP_THRESHOLD_SECONDS = 1.0
SIGNALS = [
    "Motor current(mA)",
    "Motor Voltage(10mV)",
    "Motor electrodynamic force",
    "Door leaf position",
]
REQUIRED_COLUMNS = ["Datetime", *SIGNALS]
OUTPUT_COLUMNS = ["start_time", "end_time", "prediction"]
LABELS = {0: "Normal", 1: "Abnormal resistance"}


@dataclass
class DoorResult:
    predictions: pd.DataFrame
    segments: pd.DataFrame
    features: pd.DataFrame
    probabilities: np.ndarray | None


def parse_door_datetime(value: object) -> pd.Timestamp:
    """Parse the dataset's year-month-day-hour-minute-second-millisecond value."""
    parts = str(value).split("-")
    if len(parts) != 7:
        return pd.NaT
    try:
        year, month, day, hour, minute, second, millisecond = map(int, parts)
        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=millisecond * 1000,
        )
    except (TypeError, ValueError):
        return pd.NaT


def load_stream(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load and validate one continuous Door telemetry stream."""
    if hasattr(source, "seek"):
        source.seek(0)
    data = pd.read_csv(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"Missing required Door columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Door telemetry is empty")

    data = data.copy()
    data["timestamp"] = data["Datetime"].map(parse_door_datetime)
    invalid_timestamps = int(data["timestamp"].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"Door telemetry contains {invalid_timestamps} invalid timestamps")
    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Door timestamps must be in chronological order")

    for signal in SIGNALS:
        data[signal] = pd.to_numeric(data[signal], errors="coerce")
    invalid_signals = int(data[SIGNALS].isna().sum().sum())
    if invalid_signals:
        raise ValueError(f"Door telemetry contains {invalid_signals} missing or non-numeric signal values")
    return data


def segment_stream(data: pd.DataFrame, gap_threshold: float = GAP_THRESHOLD_SECONDS) -> pd.DataFrame:
    """Split consecutive cycles at timestamp gaps greater than the learned rule."""
    gaps = data["timestamp"].diff().dt.total_seconds()
    gap_indices = data.index[gaps > gap_threshold].tolist()
    starts = [0, *gap_indices]
    ends = [index - 1 for index in gap_indices] + [len(data) - 1]
    segments = pd.DataFrame({"start_idx": starts, "end_idx": ends})
    segments["n_rows"] = segments["end_idx"] - segments["start_idx"] + 1
    segments["start_timestamp"] = segments["start_idx"].map(data["timestamp"])
    segments["end_timestamp"] = segments["end_idx"].map(data["timestamp"])
    segments["duration_seconds"] = (segments["end_timestamp"] - segments["start_timestamp"]).dt.total_seconds()
    if (segments["n_rows"] <= 0).any() or (segments["duration_seconds"] < 0).any():
        raise ValueError("Invalid Door cycle boundaries detected")
    return segments


def _statistics(values: pd.Series, prefix: str) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_range": float(np.max(array) - np.min(array)),
    }


def extract_features(data: pd.DataFrame, segments: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reproduce the whole-cycle statistics used to train the saved model."""
    rows = []
    for segment in segments.itertuples(index=False):
        cycle = data.loc[int(segment.start_idx) : int(segment.end_idx)]
        row: dict[str, float] = {}
        for signal in SIGNALS:
            row.update(_statistics(cycle[signal], signal))
        rows.append(row)
    features = pd.DataFrame(rows)
    missing = [feature for feature in feature_names if feature not in features]
    if missing:
        raise ValueError(f"Feature generation is missing model features: {', '.join(missing)}")
    features = features[feature_names]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Generated Door features contain non-finite values")
    return features


def load_artifact(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_path: str | Path = DEFAULT_FEATURE_PATH,
):
    """Load the trusted local model and its exact ordered feature list."""
    feature_names = pd.read_csv(feature_path)["feature"].tolist()
    if not feature_names:
        raise ValueError("Door model feature list is empty")
    model = joblib.load(model_path)
    return model, feature_names


def predict_stream(
    source: str | Path | BinaryIO,
    model=None,
    feature_names: list[str] | None = None,
) -> DoorResult:
    """Segment and classify a complete Door stream."""
    if model is None or feature_names is None:
        model, feature_names = load_artifact()
    data = load_stream(source)
    segments = segment_stream(data)
    features = extract_features(data, segments, feature_names)
    encoded = model.predict(features)
    try:
        labels = [LABELS[int(value)] for value in encoded]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Door model returned an unsupported class") from exc
    predictions = pd.DataFrame(
        {
            "start_time": segments["start_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3],
            "end_time": segments["end_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3],
            "prediction": labels,
        },
        columns=OUTPUT_COLUMNS,
    )
    probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    return DoorResult(predictions, segments, features, probabilities)


def predictions_csv(result: DoorResult) -> bytes:
    """Serialize predictions using the required submission schema."""
    if result.predictions.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError("Unexpected Door prediction columns")
    if not set(result.predictions["prediction"]).issubset(set(LABELS.values())):
        raise ValueError("Invalid Door prediction label")
    return result.predictions.to_csv(index=False, lineterminator="\n").encode()
