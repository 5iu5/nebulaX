"""Regression tests for the notebook-derived Door inference pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from door.inference import OUTPUT_COLUMNS, load_artifact, parse_door_datetime, predict_stream, predictions_csv

ROOT = Path(__file__).resolve().parents[2]
DOOR = ROOT / "door"
TEST_STREAM = ROOT / "PS3/02_Datasets/Door/Test.csv"


def test_timestamp_parser():
    assert parse_door_datetime("2023-7-5-0-0-3-760") == pd.Timestamp("2023-07-05 00:00:03.760")
    assert pd.isna(parse_door_datetime("invalid"))


def test_inference_reproduces_notebook_predictions():
    result = predict_stream(TEST_STREAM)
    expected = pd.read_csv(DOOR / "door_predictions.csv")
    actual = pd.read_csv(pd.io.common.BytesIO(predictions_csv(result)))
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.columns.tolist() == OUTPUT_COLUMNS
    assert len(actual) == 38
    assert set(actual.prediction) <= {"Normal", "Abnormal resistance"}


def test_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Datetime": ["2023-7-5-0-0-0-0"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Missing required Door columns"):
        predict_stream(bad)


def test_saved_model_and_features_are_compatible():
    model, features = load_artifact()
    assert len(features) == 19
    assert list(model.feature_names_in_) == features
