import copy
import numpy as np
import pandas as pd
import pytest
from acv.models import Model
from acv.features import percentiles, ordered, LINEAR_FEATURES, TREE_FEATURES
from acv.inference import rank_case, predictions_csv, prediction_zip
from acv.metrics import rank_score


@pytest.mark.parametrize("rank,score", [(1, 1.0), (2, 0.875), (8, 0.125)])
def test_official_score(rank, score):
    cars = [f"{i:02d}" for i in range(1, 9)]
    assert rank_score(cars, cars[rank - 1]) == score
    assert rank_score(cars, "99") == 0


def test_missing_car_neutral_not_dropped():
    s = percentiles(pd.Series({"01": 3.0, "02": 1.0, "03": np.nan}))
    assert s["03"] == 0.5 and ordered(s) == ["01", "03", "02"]


def test_no_identifier_features():
    assert len(LINEAR_FEATURES) == 12 and len(TREE_FEATURES) <= 32
    assert not set(LINEAR_FEATURES + TREE_FEATURES) & {"car", "filename", "train", "model_family", "time"}


def test_baseline_and_submission(features, baseline):
    r = rank_case(features, baseline)
    assert r.ranked_cars[0] == "03"
    assert predictions_csv([r]).decode().startswith("file_id,ranked_cars\ncase.xlsx,03|")
    assert predictions_csv([r]) == predictions_csv([rank_case(features, baseline)])
    import io, zipfile

    with zipfile.ZipFile(io.BytesIO(prediction_zip(predictions_csv([r])))) as z:
        assert z.namelist() == ["acv_predictions.csv"]
        assert z.read("acv_predictions.csv") == predictions_csv([r])
    with pytest.raises(ValueError, match="Duplicate input"):
        predictions_csv([r, r])


@pytest.mark.parametrize("name,params", [("linear_ranker", {"C": 0.1}), ("healthy_residual", {"alpha": 1.0, "aggregation": "mean"})])
def test_learned_model_provenance_and_roundtrip(features, name, params, tmp_path):
    import joblib

    data = {"training_a.xlsx": features, "training_b.xlsx": copy.deepcopy(features)}
    labels = {n: "03" for n in data}
    m = Model(name, params).fit(data, labels)
    assert m.training_files == sorted(data) and "test.xlsx" not in m.training_hashes
    scores = m.predict(features)
    assert scores.index.tolist() == features.case.cars and np.isfinite(scores).all()
    path = tmp_path / "m.joblib"
    joblib.dump(m, path)
    np.testing.assert_allclose(joblib.load(path).predict(features), scores)


def test_evaluator_rejects_fold_overlap(features, tmp_path):
    from acv.evaluate import Evaluator

    ev = Evaluator({"a": features, "b": features}, {"a": "03", "b": "03"}, {"output_root": str(tmp_path)})
    with pytest.raises(AssertionError, match="leakage"):
        ev.predict("thermal", {"aggregation": "mean"}, ["a"], ["a"])


def test_bootstrap_reproducible(features, baseline):
    from acv.robustness import bootstrap_stability

    a = bootstrap_stability(features, baseline, samples=20)
    b = bootstrap_stability(features, baseline, samples=20)
    assert a == b and a["cars"]["03"]["top1_frequency"] == 1.0


def test_neural_shared_encoder_equivariance():
    torch = pytest.importorskip("torch")
    from acv.neural import TemporalMIL

    torch.manual_seed(42)
    m = TemporalMIL().eval()
    x = torch.randn(4, 2, 9, 120)
    mask = torch.ones(4, 2, 120)
    perm = [2, 0, 3, 1]
    with torch.no_grad():
        a = m(x, mask)
        b = m(x[perm], mask[perm])
    torch.testing.assert_close(a[perm], b)


def test_artifact_checksum_refuses_tampering(tmp_path):
    import json
    from acv.inference import load_artifact

    (tmp_path / "model.joblib").write_bytes(b"not a model")
    (tmp_path / "manifest.json").write_text(json.dumps({"model_sha256": "wrong"}))
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(tmp_path)
