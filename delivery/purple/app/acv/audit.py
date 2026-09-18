import argparse
from pathlib import Path
from .common import config, save_json
from .prepare import cached_case

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/acv.yaml")
    p.add_argument("--include-test", action="store_true", help="Schema/quality audit only; no test ranking")
    args = p.parse_args(); cfg = config(args.config)
    files = sorted((Path(cfg["data_root"])/"Train").glob("*.xlsx"))
    if args.include_test: files += sorted((Path(cfg["data_root"])/"Test").glob("*.xlsx"))
    rows = []
    for path in files:
        features = cached_case(path,cfg)
        row = {"file_id": path.name, **features.case.metadata,
               "quality": features.table[["available", "usable_hours", "missing_fraction", "invalid_fraction"]].to_dict("index")}
        rows.append(row)
        print(path.name, row["rows"], row["schema"], "unavailable:", row["unavailable_cars"], flush=True)
    save_json(Path(cfg["output_root"])/"audit.json", rows)

if __name__ == "__main__": main()
