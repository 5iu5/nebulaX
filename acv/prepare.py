from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import joblib
from .common import config, digest, save_json
from .data import load_case
from .features import extract_features

DATA_VERSION = "4"


def cached_case(path, cfg):
    path = Path(path)
    from . import data, features

    version = digest(data.__file__) + digest(features.__file__) + DATA_VERSION
    key = digest(path)
    root = Path(cfg["output_root"]) / "cache" / key
    manifest = root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("version") == version:
        return joblib.load(root / "features.joblib")
    print(f"Preparing {path.name}", flush=True)
    case = load_case(path)
    result = extract_features(case)
    root.mkdir(parents=True, exist_ok=True)
    case.frame.reset_index().to_parquet(root / "telemetry.parquet", index=False)
    result.table.to_parquet(root / "features.parquet")
    joblib.dump(result, root / "features.joblib", compress=3)
    save_json(manifest, {"version": version, **case.metadata, "file_id": case.file_id})
    return result


def training_data(cfg):
    root = Path(cfg["data_root"])
    labels = pd.read_csv(root / "Train_Labels.csv", dtype=str).set_index("filename").faulty_car.to_dict()
    data = {p.name: cached_case(p, cfg) for p in sorted((root / "Train").glob("*.xlsx"))}
    if set(data) != set(labels):
        raise ValueError("Training files and labels do not match.")
    for name, car in labels.items():
        if car not in data[name].case.cars:
            raise ValueError(f"Unknown faulty car {car} in {name}")
    return data, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/acv.yaml")
    cfg = config(p.parse_args().config)
    data, labels = training_data(cfg)
    report = [{"file_id": name, "faulty_car": labels[name], **f.case.metadata} for name, f in data.items()]
    save_json(Path(cfg["output_root"]) / "prepared.json", report)
    print(f"Prepared {len(data)} training cases. Test excluded from training cache.")


if __name__ == "__main__":
    main()
