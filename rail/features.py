"""Feature extraction for the shipped rail corrugation model.

This module is the notebook implementation from ``11_finalPrediction.ipynb`` factored into
reusable functions for training, CLI inference, and Streamlit inference.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

FS = 10_000
BANDS = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 5000)]
SIDE_I_POSITIONS = (1, 3, 5, 7)
SIDE_II_POSITIONS = (2, 4, 6, 8)
PAIR_POSITIONS = ((1, 2), (3, 4), (5, 6), (7, 8))


def _position_pattern(position: int) -> str:
    return rf"position\s*{position}(\D|$)"


def _contains_position(column: str, position: int) -> bool:
    return re.search(_position_pattern(position), column.lower()) is not None


def channel_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    """Identify vibration/shock columns and side-specific vibration subsets."""
    columns = [str(c) for c in data.columns]
    vib_cols = [c for c in columns if "vibration" in c.lower()]
    shock_cols = [c for c in columns if "shock" in c.lower()]
    if not vib_cols:
        # Fallback for minimally labelled files: speed + alternating vibration/shock pairs.
        numeric = columns[1:]
        vib_cols = numeric[0::2]
        shock_cols = numeric[1::2]
    side1_vib = [c for c in vib_cols if any(_contains_position(c, p) for p in SIDE_I_POSITIONS)]
    side2_vib = [c for c in vib_cols if any(_contains_position(c, p) for p in SIDE_II_POSITIONS)]
    if not side1_vib or not side2_vib:
        half = len(vib_cols) // 2
        side1_vib = vib_cols[:half]
        side2_vib = vib_cols[half:]
    return {"vibration": vib_cols, "shock": shock_cols, "side1_vibration": side1_vib, "side2_vibration": side2_vib}


def _finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if np.isfinite(value) else default


def signal_features(x: np.ndarray) -> dict[str, float]:
    values = np.asarray(x, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Cannot extract features from an empty signal")
    return {
        "rms": _finite(np.sqrt(np.mean(values**2))),
        "std": _finite(np.std(values)),
        "mean_abs": _finite(np.mean(np.abs(values))),
        "peak": _finite(np.max(np.abs(values))),
        "kurtosis": _finite(kurtosis(values, nan_policy="omit")),
        "skew": _finite(skew(values, nan_policy="omit")),
    }


def spectral_features(x: np.ndarray) -> dict[str, float]:
    values = np.asarray(x, dtype=float).ravel()
    nperseg = min(2048, len(values))
    if nperseg < 8:
        raise ValueError("Signal too short for rail spectral features")
    f, psd = welch(values, fs=FS, nperseg=nperseg)
    features = {"dominant_freq": _finite(f[int(np.argmax(psd))])}
    for low, high in BANDS:
        mask = (f >= low) & (f < high)
        features[f"power_{low}_{high}"] = _finite(np.trapezoid(psd[mask], f[mask]) if mask.any() else 0.0)
    return features


def _safe_ratio(num: float, den: float) -> float:
    return _finite(num / den) if abs(den) > 1e-12 else 0.0


def axle_rms(data: pd.DataFrame, vib_cols: list[str]) -> dict[str, float]:
    return {c: _finite(np.sqrt(np.mean(np.asarray(data[c], dtype=float) ** 2))) for c in vib_cols}


def base_features(data: pd.DataFrame) -> dict[str, float]:
    groups = channel_groups(data)
    side1_vib = groups["side1_vibration"]
    side2_vib = groups["side2_vibration"]
    s1_time = signal_features(data[side1_vib].to_numpy())
    s2_time = signal_features(data[side2_vib].to_numpy())
    s1_spec = spectral_features(data[side1_vib].mean(axis=1).to_numpy())
    s2_spec = spectral_features(data[side2_vib].mean(axis=1).to_numpy())

    features: dict[str, float] = {}
    features.update({f"side1_{k}": v for k, v in s1_time.items()})
    features.update({f"side2_{k}": v for k, v in s2_time.items()})
    features.update({f"side1_{k}": v for k, v in s1_spec.items()})
    features.update({f"side2_{k}": v for k, v in s2_spec.items()})
    features["rms_diff"] = _finite(s1_time["rms"] - s2_time["rms"])
    features["rms_ratio"] = _safe_ratio(s1_time["rms"], s2_time["rms"])
    return features


def spatial_features(data: pd.DataFrame) -> dict[str, float]:
    groups = channel_groups(data)
    vib_cols = groups["vibration"]
    rms = axle_rms(data, vib_cols)
    features: dict[str, float] = {}

    for p1, p2 in PAIR_POSITIONS:
        cols1 = [c for c in vib_cols if _contains_position(c, p1)]
        cols2 = [c for c in vib_cols if _contains_position(c, p2)]
        v1 = _finite(np.mean([rms[c] for c in cols1])) if cols1 else 0.0
        v2 = _finite(np.mean([rms[c] for c in cols2])) if cols2 else 0.0
        features[f"pair_{p1}_{p2}_diff"] = _finite(v1 - v2)
        features[f"pair_{p1}_{p2}_ratio"] = _safe_ratio(v1, v2)

    for car in range(1, 9):
        car_suffix = f"car {car}"
        side1 = [rms[c] for c in vib_cols if c.lower().endswith(car_suffix) and any(_contains_position(c, p) for p in SIDE_I_POSITIONS)]
        side2 = [rms[c] for c in vib_cols if c.lower().endswith(car_suffix) and any(_contains_position(c, p) for p in SIDE_II_POSITIONS)]
        features[f"car_{car}_diff"] = _finite((np.mean(side1) if side1 else 0.0) - (np.mean(side2) if side2 else 0.0))
    return features


def extract_features(data: pd.DataFrame) -> dict[str, float]:
    features = base_features(data)
    features.update(spatial_features(data))
    return features


def feature_frame(recordings: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for file_id, data in recordings:
        row = extract_features(data)
        row["file_id"] = file_id
        rows.append(row)
    return pd.DataFrame(rows)
