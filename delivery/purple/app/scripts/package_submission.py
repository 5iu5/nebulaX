"""Package only app-generated, schema-validated predictions; never raw data."""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acv.common import digest
from acv.validate_submission import validate

def main():
    p=argparse.ArgumentParser();p.add_argument("--team",default="purple");p.add_argument("--stage-only",action="store_true",help="Stage files but explicitly mark a missing demo as incomplete");a=p.parse_args()
    if Path(a.team).name!=a.team or a.team in [".",".."]:raise ValueError("Team name must be a single folder name")
    export=ROOT/"outputs/acv/app_export"
    provenance=json.loads((export/"provenance.json").read_text())
    if provenance["source"]!="streamlit_app":raise ValueError("Generate final predictions through the app")
    validate(ROOT/"PS3/02_Datasets/ACV/Test",export/"acv_predictions.csv")
    if provenance["model_sha256"]!=digest(ROOT/"artifacts/acv/final/model.joblib"):
        raise ValueError("App export came from a different model")
    expected={p.name for p in (ROOT/"PS3/02_Datasets/ACV/Test").glob("*.xlsx")}
    if set(provenance["input_hashes"])!=expected:raise ValueError("Input provenance must cover every test file")
    for name,sha in provenance["input_hashes"].items():
        if digest(ROOT/"PS3/02_Datasets/ACV/Test"/name)!=sha:raise ValueError("Input provenance mismatch")
    with zipfile.ZipFile(export/"predictions.zip") as archive:
        if archive.namelist()!=["acv_predictions.csv"] or archive.read("acv_predictions.csv")!=(export/"acv_predictions.csv").read_bytes():
            raise ValueError("Prediction ZIP must contain exactly the validated app CSV")
    video=ROOT/"outputs/acv/demo_video.mp4"
    duration=None
    if video.exists():
        if not shutil.which("ffprobe"):raise ValueError("Install ffmpeg/ffprobe to verify the demo duration")
        duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(video)],text=True))
        if not 0<duration<=180:raise ValueError("The demo must be nonempty and no longer than three minutes")
    elif not a.stage_only:raise ValueError("Record the app demo before packaging the final submission")
    team=ROOT/"delivery"/a.team;app=team/"app";app.mkdir(parents=True,exist_ok=True)
    for name in ["acv","app","configs",".streamlit","artifacts/acv/final","scripts","docs","tests"]:
        shutil.copytree(ROOT/name,app/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["pyproject.toml","predict.py","requirements.lock","requirements-catboost.txt","README_ACV.md"]:
        shutil.copy2(ROOT/name,app/name)
    shutil.copy2(export/"predictions.zip",team/"predictions.zip")
    optional=team/"Optional_Items";optional.mkdir(exist_ok=True)
    shutil.copy2(ROOT/"docs/ACV_METHOD.md",optional/"write_up.md")
    validation=optional/"ACV"/"validation";validation.mkdir(parents=True,exist_ok=True)
    for name in ["selection.json","robustness.json","audit.json","rich_diagnostics.json"]:
        source=ROOT/"outputs/acv"/name
        if source.exists():shutil.copy2(source,validation/name)
    shutil.copytree(ROOT/"outputs/acv/evaluation",validation/"evaluation",dirs_exist_ok=True)
    for name in ["provenance.json","verification.json"]:
        if (export/name).exists():shutil.copy2(export/name,validation/name)
    if (ROOT/"outputs/acv/tests.xml").exists():shutil.copy2(ROOT/"outputs/acv/tests.xml",validation/"tests.xml")
    if video.exists():shutil.copy2(video,team/"demo_video.mp4")
    (team/"DELIVERY_STATUS.json").write_text(json.dumps({"complete":video.exists(),"missing":[] if video.exists() else ["Required live app demo video"],"team":a.team,"demo_seconds":duration},indent=2))
    (team/"START_HERE.txt").write_text("ACV submission for "+a.team+".\nOpen app/README_ACV.md for setup and run instructions.\nThe prediction archive was generated through the included app.\n")
    print(team)
if __name__=="__main__":main()
