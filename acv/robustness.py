from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import config, save_json
from .data import Case
from .features import extract_features, percentiles, ordered
from .inference import rank_case
from .prepare import training_data
from .metrics import case_result, summary
from .evaluate import Evaluator


def subset_case(case, start, end):
    f = case.frame.loc[(case.frame.index >= start) & (case.frame.index < end)].copy()
    n = case.native.loc[(case.native.index >= start) & (case.native.index < end)].copy()
    return Case(case.file_id, case.cars, f, n, case.metadata.copy())


def bootstrap_stability(features, bundle, samples=200, seed=42):
    """Synchronized hourly block bootstrap; exact for frozen mean thermal model.

    For learned models, bootstrap the mean hourly model evidence, explicitly
    labelled separately from the full-case estimator to avoid false confidence.
    """
    case = features.case
    rng = np.random.default_rng(seed)
    cars = case.cars
    if bundle.get("challenger") is None and bundle["baseline"].params["aggregation"] == "mean":
        f = case.native
        # Fallback to standalone error only for cars without any peer evidence.
        signal = f.peer.copy()
        for c in cars:
            mask = f.car.eq(c)
            if not signal.loc[mask].notna().any():
                signal.loc[mask] = f.loc[mask, "err"]
        temp = pd.DataFrame({"car": f.car, "signal": signal, "block": f.index.floor("h")})
        grouped = temp.groupby(["block", "car"]).signal.agg(["sum", "count"])
        sums = grouped["sum"].unstack("car").reindex(columns=cars).fillna(0)
        counts = grouped["count"].unstack("car").reindex(index=sums.index, columns=cars).fillna(0)
        sums, counts = sums.to_numpy(), counts.to_numpy()
        keep = counts.sum(1) > 0
        sums, counts = sums[keep], counts[keep]
        method = "Synchronized hourly bootstrap of the full-case thermal mean"

        def score(idx):
            num = sums[idx].sum(0)
            den = counts[idx].sum(0)
            return percentiles(pd.Series(np.divide(num, den, out=np.full(len(cars), np.nan), where=den > 0), index=cars))

        nblocks = len(sums)
    else:
        values = []
        for start in sorted(case.frame.index.floor("h").unique()):
            chunk = subset_case(case, start, start + pd.Timedelta(hours=1))
            if chunk.frame.eligible.sum() < 84:
                continue
            r = rank_case(extract_features(chunk), bundle)
            values.append(r.table.score.reindex(cars).to_numpy())
        values = np.asarray(values)
        nblocks = len(values)
        method = "Synchronized hourly bootstrap of mean hourly model evidence (not a calibrated full-case confidence interval)"

        def score(idx):
            return pd.Series(values[idx].mean(0), index=cars)

    if not nblocks:
        return {"method": method, "samples": 0, "cars": {}, "warning": "No usable blocks"}
    ranks = {c: [] for c in cars}
    for _ in range(samples):
        ranking = ordered(score(rng.integers(0, nblocks, size=nblocks)))
        for c in cars:
            ranks[c].append(ranking.index(c) + 1)
    return {
        "method": method,
        "samples": samples,
        "blocks": nblocks,
        "seed": seed,
        "cars": {
            c: {
                "top1_frequency": float(np.mean(np.array(rs) == 1)),
                "median_rank": float(np.median(rs)),
                "rank_p05": float(np.quantile(rs, 0.05)),
                "rank_p95": float(np.quantile(rs, 0.95)),
            }
            for c, rs in ranks.items()
        },
    }


def stress_tests(cfg, selection, include_bootstrap=True):
    data, labels = training_data(cfg)
    ev = Evaluator(data, labels, cfg)
    report = {"grouped": {}, "truncation": [], "bootstrap": {}, "ablations": {}}
    for group in ["train", "model_family"]:
        groups = {n: f.case.metadata[group] for n, f in data.items()}
        rows = []
        for value in sorted(set(groups.values())):
            test = [n for n in data if groups[n] == value]
            train = [n for n in data if n not in test]
            # Frozen final hyperparameters: this is a stress check, not another unbiased model-selection estimate.
            pred = ev.apply(selection, train, test)
            rows.extend([{**r, "held_out_group": value, "training_files": train} for r in ev.rows(pred)])
        report["grouped"][group] = {"summary": summary(rows), "cases": rows, "interpretation": "Frozen-hyperparameter stress test"}
    # Full-data fit used only for descriptive perturbation tests; never headline validation.
    base = ev.get_model("thermal", selection["baseline_params"], list(data))
    challenger = None if selection["model"] == "thermal" else ev.get_model(selection["model"], selection["params"], list(data))
    bundle = {"baseline": base, "challenger": challenger, "weight": selection["weight"]}
    for name, f in data.items():
        for hours in [6, 12, 24]:
            start = f.case.frame.index.min()
            chunk = subset_case(f.case, start, start + pd.Timedelta(hours=hours))
            result = rank_case(extract_features(chunk), bundle)
            report["truncation"].append(
                {**case_result(name, result.ranked_cars, labels[name]), "hours": hours, "interpretation": "Descriptive training-case perturbation"}
            )
        if include_bootstrap:
            report["bootstrap"][name] = bootstrap_stability(f, bundle, cfg["bootstrap_samples"])
    # Fixed ablations: distinguish thermal evidence from telemetry-quality shortcuts.
    for mode in ["thermal_native", "steady_only", "no_peer", "quality_only"]:
        rows = []
        for name, f in data.items():
            if mode == "thermal_native":
                s = f.table.native_thermal
            elif mode == "no_peer":
                s = f.table.err_mean
            elif mode == "quality_only":
                s = f.table.invalid_fraction
            else:
                s = f.case.frame.loc[f.case.frame.steady].groupby("car").peer.mean().reindex(f.case.cars)
            rows.append(case_result(name, ordered(percentiles(s, f.table.available)), labels[name]))
        report["ablations"][mode] = {"summary": summary(rows), "cases": rows, "interpretation": "Exploratory diagnostic; not used for model selection"}
    save_json(Path(cfg["output_root"]) / "robustness.json", report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/acv.yaml")
    p.add_argument("--selection", default="outputs/acv/selection.json")
    p.add_argument("--skip-bootstrap", action="store_true")
    a = p.parse_args()
    stress_tests(config(a.config), json.loads(Path(a.selection).read_text()), not a.skip_bootstrap)


if __name__ == "__main__":
    main()
