"""Command-line rail prediction entry point.

This is the only rail module intended to be pointed at the held-out Test/ directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import csv_files
from .inference import predict_paths, validate_predictions
from .models import DEFAULT_MODEL_PATH


def run_prediction(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    expected_count: int | None = None,
) -> Path:
    paths = csv_files(input_path)
    predictions = predict_paths(paths, model_path=model_path)
    validate_predictions(
        predictions,
        expected_file_ids={path.name for path in paths},
        expected_rows=expected_count if expected_count is not None else len(paths),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    predictions.to_csv(temporary, index=False)
    temporary.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict rail corrugation class")
    parser.add_argument("--input", required=True, help="A rail CSV or directory of CSV files")
    parser.add_argument("--output", default="rail_predictions.csv", help="Destination CSV")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Rail model artifact")
    parser.add_argument("--expected-count", type=int, default=None, help="Optional required output row count")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = run_prediction(args.input, args.output, model_path=args.model, expected_count=args.expected_count)
    print(f"Wrote validated rail predictions to {destination}")


if __name__ == "__main__":
    main()
