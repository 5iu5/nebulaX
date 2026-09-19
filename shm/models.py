"""Two-parameter S-N/Miner model, with the exponent fixed by validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import rainflow_log_moment

DEFAULT_METADATA_PATH = Path(__file__).with_name("model_metadata.json")


@dataclass(frozen=True)
class PhysicsDamageModel:
    """Predict fatigue damage as ``D = S(m) / C``."""

    m: float
    c: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.m) or self.m <= 0:
            raise ValueError("m must be finite and positive")
        if not np.isfinite(self.c) or self.c <= 0:
            raise ValueError("C must be finite and positive")

    @classmethod
    def fit(cls, log_moments: np.ndarray, damage: np.ndarray, *, m: float = 5.0) -> "PhysicsDamageModel":
        """Fit C by the geometric-mean estimator for relative error."""
        log_s = np.asarray(log_moments, dtype=float)
        targets = np.asarray(damage, dtype=float)
        if log_s.shape != targets.shape or log_s.ndim != 1 or log_s.size == 0:
            raise ValueError("log moments and damage must be matching non-empty vectors")
        if not np.isfinite(log_s).all() or not np.isfinite(targets).all() or np.any(targets <= 0):
            raise ValueError("Training values must be finite and damage must be positive")
        log_c = float(np.mean(log_s - np.log(targets)))
        return cls(m=float(m), c=float(np.exp(log_c)))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_METADATA_PATH) -> "PhysicsDamageModel":
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(m=float(metadata["m"]), c=float(metadata["C"]))

    def save(self, path: str | Path, **metadata: object) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": "rainflow_miner_power_law", "m": self.m, "C": self.c, **metadata}
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def predict_log_moment(self, log_moment: float) -> float:
        prediction = float(np.exp(float(log_moment) - np.log(self.c)))
        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError("Model produced a non-finite or non-positive prediction")
        return prediction

    def predict_cycles(self, cycles: pd.DataFrame) -> float:
        return self.predict_log_moment(rainflow_log_moment(cycles, self.m))
