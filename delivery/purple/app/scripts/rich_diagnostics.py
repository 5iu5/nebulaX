"""Diagnostic-only circuit and peer pressure comparisons for rich training cases."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from acv.common import config, save_json
from acv.prepare import training_data


def main():
    data, _ = training_data(config())
    report = {
        "interpretation": "Diagnostic only: one rich case cannot validate a pressure fault predictor.",
        "units": "Unspecified source units; differences are comparative, with no absolute operating limits.",
        "peer_rule": "At least two usable peers with the same simultaneous compressor states; candidate circuit on.",
        "cases": {},
    }
    for name, features in data.items():
        case = features.case
        if case.metadata["schema"] != "rich":
            continue
        frame = case.frame
        state1 = frame.pivot(columns="car", values="compressor1")
        state2 = frame.pivot(columns="car", values="compressor2")
        cars = {}
        for car in case.cars:
            row = features.table.loc[car]
            values = {key: row[key] for key in [
                "pressure_low_asymmetry", "pressure_high_asymmetry",
                "compressor_duty_imbalance", "both_compressors_hours",
            ]}
            compatible = state1.eq(state1[car], axis=0) & state2.eq(state2[car], axis=0)
            for circuit in [1, 2]:
                for side in ["low", "high"]:
                    key = f"p{circuit}_{side}"
                    pressure = frame.pivot(columns="car", values=key)
                    peers = pressure.where(compatible).drop(columns=car)
                    median = peers.median(axis=1).where(peers.count(axis=1) >= 2)
                    states = state1 if circuit == 1 else state2
                    residual = (pressure[car] - median).where(states[car].eq(1))
                    values[f"{key}_peer_residual_median"] = residual.median()
                    values[f"{key}_comparison_hours"] = residual.notna().sum() / 120
            cars[car] = values
        report["cases"][name] = {"source_hash": case.metadata["source_hash"], "cars": cars}
    path = ROOT / "outputs/acv/rich_diagnostics.json"
    save_json(path, report)
    print(path)


if __name__ == "__main__":
    main()
