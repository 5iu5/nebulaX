from __future__ import annotations

import numpy as np
import pandas as pd

from rail.features import extract_features
from rail.inference import validate_predictions
from rail.metrics import rail_macro_f1


def _synthetic_recording(n: int = 256) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {"speed_signal": np.arange(n) % 2}
    for car in range(1, 9):
        for position in range(1, 9):
            data[f"position {position} vibration car {car}"] = rng.normal(size=n)
            data[f"position {position} shock car {car}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_extract_features_matches_shipped_model_schema_subset():
    features = extract_features(_synthetic_recording())
    for key in ["side1_rms", "side2_rms", "side1_power_100_250", "pair_1_2_diff", "car_8_diff"]:
        assert key in features
        assert np.isfinite(features[key])


def test_validate_predictions_schema_and_labels():
    predictions = pd.DataFrame({"file_id": ["Test1.csv"], "prediction": ["Normal"]})
    validate_predictions(predictions, expected_file_ids={"Test1.csv"}, expected_rows=1)


def test_rail_macro_f1():
    assert rail_macro_f1(["Normal", "Side I", "Side II"], ["Normal", "Side I", "Side II"]) == 1.0
