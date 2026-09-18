"""Data utilities for rail corrugation files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "PS3" / "02_Datasets" / "Rail_Corrugation"
TRAIN_DIR = DATASET_DIR / "Train"
LABELS_PATH = DATASET_DIR / "Train_Labels.csv"


def csv_files(path: str | Path) -> list[Path]:
    """Return one or more CSV paths in deterministic order."""
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got {source}")
        return [source]
    if not source.exists():
        raise FileNotFoundError(source)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() == ".csv")
    if not paths:
        raise ValueError(f"No CSV files found under {source}")
    return paths


def read_recording(path_or_buffer) -> pd.DataFrame:
    """Read a 10,000 x 129 rail recording."""
    data = pd.read_csv(path_or_buffer)
    if data.shape[1] < 3:
        raise ValueError("Rail recording must contain speed plus vibration/shock channels")
    return data


def read_labels(path: str | Path = LABELS_PATH) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"filename", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Rail labels missing columns: {sorted(missing)}")
    return labels


def iter_training_files(labels: pd.DataFrame | None = None, train_dir: str | Path = TRAIN_DIR) -> Iterable[tuple[Path, str]]:
    """Yield labelled training recordings. This intentionally does not touch Test/."""
    table = read_labels() if labels is None else labels
    base = Path(train_dir)
    for row in table.itertuples(index=False):
        yield base / row.filename, row.label
