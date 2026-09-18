"""Regression tests for notebook-derived SHM inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shm.inference import OUTPUT_COLUMNS, load_artifact, predict_files, predictions_csv

ROOT = Path(__file__).resolve().parents[2]
SHM = ROOT / "shm"
TEST_DIR = ROOT / "PS3/02_Datasets/SHM/Test"


def test_inference_reproduces_notebook_predictions():
    files = sorted(TEST_DIR.glob("*.csv"))
    result = predict_files([(path.name, path) for path in files])
    expected = pd.read_csv(SHM / "shm_predictions.csv")
    actual = pd.read_csv(pd.io.common.BytesIO(predictions_csv(result)))
    assert actual.columns.tolist() == OUTPUT_COLUMNS
    assert actual.file_id.tolist() == expected.file_id.tolist()
    np.testing.assert_allclose(actual.prediction, expected.prediction, rtol=0, atol=2e-15)
    assert len(actual) == 16
    assert (actual.prediction >= 0).all()


def test_rejects_signal_without_numeric_samples(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("stress\nnot-a-number\n")
    with pytest.raises(ValueError, match="numeric signal"):
        predict_files([(bad.name, bad)])


def test_saved_model_and_features_are_compatible():
    model, features = load_artifact()
    assert len(features) == 27
    assert list(model.feature_names_in_) == features
