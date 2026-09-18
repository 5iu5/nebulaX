"""Data loading for headerless, single-channel SHM stress histories."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

PathLike = str | Path


def load_signal(source: PathLike | BinaryIO | TextIO) -> np.ndarray:
    """Load one finite numeric stress channel from a headerless CSV."""
    frame = pd.read_csv(source, header=None)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one stress column, found {frame.shape[1]}")
    try:
        signal = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Stress values must all be numeric") from exc
    if signal.size < 2:
        raise ValueError("At least two stress samples are required")
    if not np.isfinite(signal).all():
        raise ValueError("Stress signal contains NaN or infinite values")
    return signal


def load_labels(path: PathLike) -> pd.DataFrame:
    """Load and validate the training filename/damage table."""
    labels = pd.read_csv(path)
    if list(labels.columns) != ["filename", "damage"]:
        raise ValueError("Labels must have exactly: filename,damage")
    if labels.empty or labels["filename"].duplicated().any():
        raise ValueError("Labels must be non-empty with unique filenames")
    labels["damage"] = pd.to_numeric(labels["damage"], errors="raise")
    if not np.isfinite(labels["damage"]).all() or (labels["damage"] <= 0).any():
        raise ValueError("Damage labels must be finite and positive")
    return labels


def csv_files(input_path: PathLike) -> list[Path]:
    """Return one CSV or all CSV files in a directory, sorted by name."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got {path.name}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"), key=lambda p: p.name.lower())
    if not files:
        raise ValueError(f"No CSV files found in {path}")
    return files
