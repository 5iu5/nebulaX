"""Feature extraction and cumulative fatigue-damage inference for SHM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import joblib
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "shm_final_model.joblib"
DEFAULT_FEATURE_PATH = PACKAGE_ROOT / "shm_final_features.csv"
OUTPUT_COLUMNS = ["file_id", "prediction"]
MAX_FFT_POINTS = 200_000


@dataclass
class SHMResult:
    predictions: pd.DataFrame
    features: pd.DataFrame


def load_signal(source: str | Path | BinaryIO) -> np.ndarray:
    """Read the first CSV column and retain finite stress samples."""
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        values = pd.read_csv(source, usecols=[0]).iloc[:, 0].to_numpy(dtype=np.float64)
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        raise ValueError("SHM input must contain a numeric signal in its first CSV column") from exc
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("SHM input contains no finite signal samples")
    return values


def _zero_crossing_rate(values: np.ndarray) -> float:
    centred = values - np.mean(values)
    if len(centred) < 2:
        return 0.0
    return float(np.mean(centred[:-1] * centred[1:] < 0))


def _spectral_features(values: np.ndarray) -> dict[str, float]:
    if len(values) > MAX_FFT_POINTS:
        values = values[np.linspace(0, len(values) - 1, MAX_FFT_POINTS).astype(int)]
    values = values - np.mean(values)
    power = np.abs(np.fft.rfft(values)) ** 2
    if len(power) <= 1 or power.sum() == 0:
        return {
            "fft_dominant_bin": 0.0,
            "fft_spectral_centroid": 0.0,
            "fft_low_power_ratio": 0.0,
            "fft_high_power_ratio": 0.0,
        }
    frequency = np.fft.rfftfreq(len(values), d=1.0)
    power[0] = 0.0
    total = power.sum()
    split = max(1, len(power) // 4)
    return {
        "fft_dominant_bin": float(np.argmax(power)),
        "fft_spectral_centroid": float((frequency * power).sum() / total),
        "fft_low_power_ratio": float(power[:split].sum() / total),
        "fft_high_power_ratio": float(power[-split:].sum() / total),
    }


def extract_features(values: np.ndarray, file_id: str) -> dict[str, float | str]:
    """Reproduce the 27 statistical and spectral training features."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No valid samples in {file_id}")
    mean = np.mean(values)
    std = np.std(values)
    absolute = np.abs(values)
    rms = np.sqrt(np.mean(values * values))
    differences = np.diff(values)
    if not len(differences):
        differences = np.array([0.0])
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    output: dict[str, float | str] = {
        "filename": file_id,
        "n_samples": len(values),
        "mean": mean,
        "std": std,
        "min": np.min(values),
        "max": np.max(values),
        "range": np.ptp(values),
        "median": np.median(values),
        "q01": quantiles[0],
        "q05": quantiles[1],
        "q25": quantiles[2],
        "q75": quantiles[3],
        "q95": quantiles[4],
        "q99": quantiles[5],
        "iqr": quantiles[3] - quantiles[2],
        "abs_mean": np.mean(absolute),
        "rms": rms,
        "energy_per_sample": np.mean(values * values),
        "crest_factor": np.max(absolute) / (rms + 1e-12),
        "zero_crossing_rate": _zero_crossing_rate(values),
        "diff_mean": np.mean(differences),
        "diff_std": np.std(differences),
        "diff_abs_mean": np.mean(np.abs(differences)),
        "diff_max_abs": np.max(np.abs(differences)),
    }
    output.update(_spectral_features(values))
    return output


def load_artifact(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_path: str | Path = DEFAULT_FEATURE_PATH,
):
    """Load the trusted local model and ordered feature contract."""
    features = pd.read_csv(feature_path)["feature"].tolist()
    if not features:
        raise ValueError("SHM model feature list is empty")
    return joblib.load(model_path), features


def predict_files(
    sources: Iterable[tuple[str, str | Path | BinaryIO]],
    model=None,
    feature_names: list[str] | None = None,
) -> SHMResult:
    """Predict one non-negative cumulative-damage value per signal file."""
    sources = list(sources)
    names = [name for name, _ in sources]
    if not sources:
        raise ValueError("No SHM CSV files supplied")
    if len(names) != len(set(names)):
        raise ValueError("SHM input filenames must be unique")
    if model is None or feature_names is None:
        model, feature_names = load_artifact()

    rows = [extract_features(load_signal(source), name) for name, source in sources]
    features = pd.DataFrame(rows)
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"SHM feature generation is missing: {', '.join(missing)}")
    matrix = features[feature_names].replace([np.inf, -np.inf], np.nan)
    if matrix.isna().any().any():
        raise ValueError("SHM features contain missing or invalid values")
    predictions = np.maximum(model.predict(matrix), 0.0)
    output = pd.DataFrame({"file_id": features["filename"], "prediction": predictions}, columns=OUTPUT_COLUMNS).sort_values("file_id")
    return SHMResult(output.reset_index(drop=True), features)


def predictions_csv(result: SHMResult) -> bytes:
    """Serialize predictions in the required submission format."""
    if result.predictions.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError("Unexpected SHM prediction columns")
    if not np.isfinite(result.predictions["prediction"]).all() or (result.predictions["prediction"] < 0).any():
        raise ValueError("SHM predictions must be finite and non-negative")
    return result.predictions.to_csv(index=False, lineterminator="\n").encode()
