"""Command-line inference for the SHM subsystem."""

import argparse
from pathlib import Path

from .inference import load_artifact, predict_files, predictions_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="SHM CSV file or directory")
    parser.add_argument("--output", required=True, help="Output shm_predictions.csv path")
    parser.add_argument("--model", default=None, help="Optional trained model.joblib path")
    parser.add_argument("--features", default=None, help="Optional ordered feature-list CSV path")
    args = parser.parse_args()

    path = Path(args.input)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files or not all(file.is_file() for file in files):
        raise ValueError("No SHM CSV input files found")
    artifact = load_artifact(args.model, args.features) if args.model and args.features else load_artifact()
    result = predict_files([(file.name, file) for file in files], *artifact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(predictions_csv(result))
    print(f"Wrote {len(result.predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
