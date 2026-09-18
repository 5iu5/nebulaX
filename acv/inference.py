from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import io
import json
import zipfile
import joblib
import pandas as pd
from .common import digest
from .data import load_case, Case
from .features import extract_features, Features, percentiles, ordered


@dataclass
class Ranking:
    file_id: str
    ranked_cars: list[str]
    table: pd.DataFrame
    features: Features
    warnings: list[str]


def load_artifact(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if digest(path / "model.joblib") != manifest["model_sha256"]:
        raise ValueError("Model artifact checksum mismatch")
    # Only load trusted, locally trained artifacts. Uploaded files are always parsed as XLSX.
    return joblib.load(path / "model.joblib")


def rank_case(source, artifact, file_id=None):
    bundle = load_artifact(artifact) if isinstance(artifact, (str, Path)) else artifact
    f = source if isinstance(source, Features) else extract_features(source if isinstance(source, Case) else load_case(source, file_id))
    scores = percentiles(bundle["baseline"].predict(f), f.table.available)
    if bundle.get("challenger") is not None:
        other = percentiles(bundle["challenger"].predict(f), f.table.available)
        w = bundle["weight"]
        scores = (1 - w) * scores + w * other
    cars = ordered(scores)
    table = f.table.copy()
    table["score"] = scores
    table["evidence"] = table.available.map({True: "Available", False: "Unavailable"})
    table = table.loc[cars]
    table.insert(0, "rank", range(1, len(cars) + 1))
    warnings = []
    missing = table.index[~table.available].tolist()
    if missing:
        warnings.append("No usable cooling evidence for cars " + ", ".join(missing) + ". They retain a neutral score, not a healthy diagnosis.")
    if table.usable_hours.max() < 1:
        warnings.append("Less than one hour of usable cooling data; ranking stability is limited.")
    warnings.append("Scores express relative evidence, not calibrated leakage probabilities.")
    return Ranking(f.case.file_id, cars, table, f, warnings)


def predictions_csv(results):
    names = [r.file_id for r in results]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate input filenames would create ambiguous submission rows")
    rows = []
    for r in results:
        if len(r.ranked_cars) != len(set(r.ranked_cars)) or set(r.ranked_cars) != set(r.features.case.cars):
            raise ValueError("Ranking must contain every header car exactly once")
        rows.append({"file_id": r.file_id, "ranked_cars": "|".join(r.ranked_cars)})
    return pd.DataFrame(rows, columns=["file_id", "ranked_cars"]).sort_values("file_id").to_csv(index=False, lineterminator="\n").encode()


def write_predictions(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(predictions_csv(results))
    return path


def prediction_zip(csv_bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("acv_predictions.csv", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, csv_bytes)
    return buf.getvalue()
