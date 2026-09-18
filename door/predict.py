"""Command-line inference for the Door subsystem."""

import argparse
from pathlib import Path

from .inference import load_artifact, predict_stream, predictions_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Continuous Door telemetry CSV")
    parser.add_argument("--output", required=True, help="Output door_predictions.csv path")
    parser.add_argument("--model", default=None, help="Optional trained model.joblib path")
    parser.add_argument("--features", default=None, help="Optional ordered feature-list CSV path")
    args = parser.parse_args()

    artifact = load_artifact(args.model, args.features) if args.model and args.features else load_artifact()
    result = predict_stream(args.input, *artifact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(predictions_csv(result))
    print(f"Wrote {len(result.predictions)} segments to {output}")


if __name__ == "__main__":
    main()
