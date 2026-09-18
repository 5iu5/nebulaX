"""Run the real Streamlit script in its in-process test harness (no sockets)."""
import json
from pathlib import Path
import pytest
from acv.common import ROOT

@pytest.mark.integration
def test_actual_app_upload_export_and_cli_parity(tmp_path):
    artifact=ROOT/"artifacts/acv/final"
    if not artifact.exists():pytest.skip("Final artifact not trained")
    from streamlit.testing.v1 import AppTest
    from acv.inference import load_artifact,rank_case,predictions_csv
    source=ROOT/"PS3/02_Datasets/ACV/Test/acv_test_case.xlsx"
    script=f'''
import io
import runpy
from pathlib import Path
from unittest.mock import patch
source=Path({str(source)!r})
upload=io.BytesIO(source.read_bytes())
upload.name=source.name
with patch("streamlit.file_uploader",return_value=[upload]):
    runpy.run_path({str(ROOT/"app/main.py")!r},run_name="__main__")
'''
    at=AppTest.from_string(script,default_timeout=90).run()
    assert not at.exception
    button=next(b for b in at.button if b.label=="Analyze uploaded files")
    button.click().run(timeout=90)
    assert not at.exception
    assert [m.label for m in at.metric]==["First car to inspect","Cars ranked","Usable cooling data · top car"]
    assert at.metric[1].value=="8"
    assert len(at.get("download_button"))==2
    expected=predictions_csv([rank_case(source,load_artifact(artifact))])
    export=ROOT/"outputs/acv/app_export"
    assert (export/"acv_predictions.csv").read_bytes()==expected
    assert json.loads((export/"provenance.json").read_text())["source"]=="streamlit_app"
    # Record the verification mechanism honestly; this is not a live-browser recording.
    (export/"verification.json").write_text(json.dumps({"method":"Streamlit AppTest", "upload_widget":"injected actual workbook bytes", "analysis_button":"clicked", "cli_parity":True, "live_browser_verified":False},indent=2))
