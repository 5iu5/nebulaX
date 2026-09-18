"""Command-line prediction entry point; the only SHM v2 module that reads Test/."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import csv_files
from .inference import predict_paths, validate_predictions
from .models import DEFAULT_METADATA_PATH


def run_prediction(
    input_path: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    expected_count: int | None = None,
) -> Path:
    """Predict, validate in memory, then atomically write the output CSV."""
    paths = csv_files(input_path)
    predictions = predict_paths(paths, metadata_path=metadata_path)
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
    parser = argparse.ArgumentParser(description="Predict cumulative SHM fatigue damage")
    parser.add_argument("--input", required=True, help="A headerless stress CSV or directory of CSV files")
    parser.add_argument("--output", default="shm_predictions.csv", help="Destination CSV")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="Fitted m/C metadata JSON")
    parser.add_argument("--expected-count", type=int, default=None, help="Optional required output row count")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = run_prediction(
        args.input,
        args.output,
        metadata_path=args.metadata,
        expected_count=args.expected_count,
    )
    print(f"Wrote validated SHM predictions to {destination}")


if __name__ == "__main__":
    main()
