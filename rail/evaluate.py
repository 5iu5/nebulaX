"""Cross-validation for the rail corrugation model on Train/ only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from .data import TRAIN_DIR, iter_training_files, read_labels, read_recording
from .features import extract_features
from .metrics import rail_macro_f1
from .models import build_model, save_model

SEED = 42


def build_training_frame(train_dir: str | Path = TRAIN_DIR) -> pd.DataFrame:
    rows = []
    for path, label in iter_training_files(read_labels(), train_dir):
        data = read_recording(path)
        row = extract_features(data)
        row["filename"] = path.name
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_features(features: pd.DataFrame) -> dict:
    X = features.drop(columns=["filename", "label"])
    y = features["label"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    scores = cross_val_score(build_model(), X, y, cv=cv, scoring=lambda estimator, Xt, yt: rail_macro_f1(yt, estimator.predict(Xt)))
    return {
        "metric": "macro_f1",
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "fold_scores": [float(s) for s in scores],
        "n_splits": int(len(scores)),
        "class_counts": y.value_counts().to_dict(),
    }


def run(output: str | Path = "outputs/rail/evaluation.json", model_output: str | Path | None = None) -> dict:
    features = build_training_frame()
    result = evaluate_features(features)
    result.update({"ts": datetime.now(timezone.utc).isoformat(), "subsystem": "rail", "model": "RandomForestClassifier"})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    if model_output:
        X = features.drop(columns=["filename", "label"])
        y = features["label"]
        model = build_model().fit(X, y)
        save_model(model, model_output)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate rail model with repeated stratified CV")
    parser.add_argument("--output", default="outputs/rail/evaluation.json")
    parser.add_argument("--model-output", default=None)
    args = parser.parse_args(argv)
    result = run(args.output, args.model_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
