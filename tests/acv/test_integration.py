import io
import pytest
from acv.common import config, ROOT
from acv.prepare import training_data, cached_case
from acv.features import ordered, percentiles, thermal_scores


@pytest.mark.integration
def test_six_workbooks_and_frozen_baseline():
    data, labels = training_data(config())
    ranks = []
    for n, f in data.items():
        r = ordered(percentiles(thermal_scores(f), f.table.available))
        ranks.append(r.index(labels[n]) + 1)
        assert len(r) == 8
    assert ranks == [1, 1, 1, 2, 1, 1]
    rich = data["acv_case_04.xlsx"]
    assert rich.case.metadata["unavailable_cars"] == ["05", "06", "07", "08"]
    assert rich.case.metadata["dominant_interval_seconds"] == 10


@pytest.mark.integration
def test_test_schema_only():
    f = cached_case(ROOT / "PS3/02_Datasets/ACV/Test/acv_test_case.xlsx", config())
    assert f.case.metadata["rows"] == 9082 and len(f.case.cars) == 8


@pytest.mark.integration
def test_app_upload_service_cli_parity(tmp_path):
    from acv.inference import rank_case, predictions_csv, load_artifact
    from acv.validate_submission import validate

    artifact = ROOT / "artifacts/acv/final"
    if not artifact.exists():
        pytest.skip("Final artifact not trained yet")
    p = ROOT / "PS3/02_Datasets/ACV/Test/acv_test_case.xlsx"
    bundle = load_artifact(artifact)
    a = rank_case(p, bundle)
    b = rank_case(io.BytesIO(p.read_bytes()), bundle, file_id=p.name)
    assert predictions_csv([a]) == predictions_csv([b])
    output = tmp_path / "acv_predictions.csv"
    output.write_bytes(predictions_csv([a]))
    assert validate(p, output) == {"valid": True, "files": 1}
