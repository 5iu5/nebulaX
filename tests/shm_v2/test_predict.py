import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shm_v2.inference import validate_predictions
from shm_v2.metrics import shm_mape_score
from shm_v2.predict import run_prediction


def test_official_metric_worked_example() -> None:
    truth = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
    prediction = np.array([0.15, 0.28, 0.55, 0.68, 0.85])
    assert shm_mape_score(truth, prediction) == pytest.approx(0.85, abs=0.001)


def test_end_to_end_predict_directory_validates_before_write(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.Series([0.0, 4.0, 0.0]).to_csv(input_dir / "test01.csv", index=False, header=False)
    pd.Series([0.0, 2.0, 0.0]).to_csv(input_dir / "test02.csv", index=False, header=False)

    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(json.dumps({"m": 5.0, "C": 32.0}), encoding="utf-8")
    output = tmp_path / "shm_predictions.csv"

    run_prediction(input_dir, output, metadata_path=metadata, expected_count=2)
    frame = pd.read_csv(output)

    validate_predictions(frame, expected_file_ids={"test01.csv", "test02.csv"}, expected_rows=2)
    assert frame["file_id"].tolist() == ["test01.csv", "test02.csv"]
    assert frame["prediction"].tolist() == pytest.approx([1.0, 1.0 / 32.0])


def test_invalid_expected_count_is_not_written(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.Series([0.0, 4.0, 0.0]).to_csv(input_dir / "test01.csv", index=False, header=False)
    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(json.dumps({"m": 5.0, "C": 32.0}), encoding="utf-8")
    output = tmp_path / "shm_predictions.csv"

    with pytest.raises(ValueError, match="Expected 16 rows"):
        run_prediction(input_dir, output, metadata_path=metadata, expected_count=16)
    assert not output.exists()
