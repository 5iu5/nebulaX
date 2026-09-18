"""CLI uses the identical inference service as the submission app."""

import argparse
from pathlib import Path
from acv.inference import load_artifact, rank_case, write_predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="artifacts/acv/final")
    a = p.parse_args()
    path = Path(a.input)
    files = sorted(path.glob("*.xlsx")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No XLSX input files")
    model = load_artifact(a.model)
    results = [rank_case(f, model) for f in files]
    write_predictions(results, a.output)
    for r in results:
        print(r.file_id, "|".join(r.ranked_cars))


if __name__ == "__main__":
    main()
