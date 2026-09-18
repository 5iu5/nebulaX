"""Train-only evaluation and final metadata fitting for the analytical SHM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import load_labels, load_signal
from .features import cache_cycle_table, rainflow_log_moment
from .metrics import shm_mape, shm_mape_score
from .models import PhysicsDamageModel

FIXED_M = 5.0
SEED = 42


def _cycles_for(filename: str, train_dir: Path, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"{Path(filename).stem}.parquet"
    if cache_path.exists():
        cycles = pd.read_parquet(cache_path)
        required = {"range", "mean", "count", "start_index", "end_index", "sigma_a"}
        if required.issubset(cycles.columns):
            return cycles
    return cache_cycle_table(load_signal(train_dir / filename), cache_path)


def _cycle_descriptors(cycles: pd.DataFrame) -> list[float]:
    usable = cycles.loc[(cycles["sigma_a"] > 0) & (cycles["count"] > 0)]
    weights = usable["count"].to_numpy(dtype=float)
    amplitude = usable["sigma_a"].to_numpy(dtype=float)
    means = usable["mean"].to_numpy(dtype=float)
    weighted_mean = float(np.average(means, weights=weights))
    weighted_std = float(np.sqrt(np.average(np.square(means - weighted_mean), weights=weights)))
    return [
        float(np.log(weights.sum())),
        weighted_mean,
        weighted_std,
        *np.quantile(amplitude, [0.25, 0.50, 0.75, 0.90, 0.99]).tolist(),
        float(np.log(amplitude.max())),
    ]


def _two_c_record(log_s: np.ndarray, damage: np.ndarray, descriptors: np.ndarray) -> dict[str, object]:
    log_y = np.log(damage)
    residual = log_s - log_y

    oracle_groups = KMeans(n_clusters=2, random_state=SEED, n_init=50).fit_predict(residual.reshape(-1, 1))
    oracle_predictions = np.empty_like(damage)
    for group in (0, 1):
        members = oracle_groups == group
        log_c = float(np.mean(residual[members]))
        oracle_predictions[members] = np.exp(log_s[members] - log_c)

    scaled = StandardScaler().fit_transform(descriptors)
    feature_groups = KMeans(n_clusters=2, random_state=SEED, n_init=50).fit_predict(scaled)
    feature_predictions = np.empty_like(damage)
    for group in (0, 1):
        members = feature_groups == group
        log_c = float(np.mean(residual[members]))
        feature_predictions[members] = np.exp(log_s[members] - log_c)

    feature_loo_predictions = np.empty_like(damage)
    for held_out in range(len(damage)):
        training = np.arange(len(damage)) != held_out
        scaler = StandardScaler().fit(descriptors[training])
        clustering = KMeans(n_clusters=2, random_state=SEED, n_init=50).fit(scaler.transform(descriptors[training]))
        held_out_group = int(clustering.predict(scaler.transform(descriptors[[held_out]]))[0])
        same_group = clustering.labels_ == held_out_group
        log_c = float(np.mean(log_s[training][same_group] - log_y[training][same_group]))
        feature_loo_predictions[held_out] = np.exp(log_s[held_out] - log_c)

    return {
        "question": "Do the undocumented line/AW0/AW4 conditions justify separate C values?",
        "one_c_in_sample_mape": shm_mape(damage, np.exp(log_s - residual.mean())),
        "target_residual_oracle": {
            "mape": shm_mape(damage, oracle_predictions),
            "group_sizes": np.bincount(oracle_groups).tolist(),
            "status": "rejected: groups are derived from target residuals and would leak labels for an unseen file",
        },
        "signal_descriptor_groups": {
            "in_sample_mape": shm_mape(damage, feature_predictions),
            "loo_mape": shm_mape(damage, feature_loo_predictions),
            "group_sizes": np.bincount(feature_groups).tolist(),
            "descriptors": "cycle count, cycle-mean mean/std, amplitude quantiles 25/50/75/90/99%, maximum amplitude",
        },
        "decision": "Rejected. File IDs are random and no line/load labels are supplied; signal-only grouping worsened both in-sample and LOO MAPE.",
    }


def evaluate(
    train_dir: str | Path,
    labels_path: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, object]:
    train = Path(train_dir)
    cache = Path(cache_dir)
    labels = load_labels(labels_path)
    expected = set(labels["filename"])
    actual = {path.name for path in train.glob("*.csv")}
    if expected != actual:
        raise ValueError(f"Training files and labels differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    log_moments = []
    descriptors = []
    for filename in labels["filename"]:
        cycles = _cycles_for(filename, train, cache)
        log_moments.append(rainflow_log_moment(cycles, FIXED_M))
        descriptors.append(_cycle_descriptors(cycles))

    log_s = np.asarray(log_moments)
    damage = labels["damage"].to_numpy(dtype=float)
    final_model = PhysicsDamageModel.fit(log_s, damage, m=FIXED_M)
    in_sample_predictions = np.array([final_model.predict_log_moment(value) for value in log_s])

    loo_predictions = np.empty_like(damage)
    for held_out in range(len(damage)):
        training = np.arange(len(damage)) != held_out
        fold_model = PhysicsDamageModel.fit(log_s[training], damage[training], m=FIXED_M)
        loo_predictions[held_out] = fold_model.predict_log_moment(log_s[held_out])

    loo_errors = np.abs(loo_predictions - damage) / damage
    result: dict[str, object] = {
        "method": "ASTM E1049 rainflow + Miner's rule D=S(m)/C",
        "config": {"m": FIXED_M, "m_selection": "fixed after nested LOO selected 5.00 in all 64 folds", "c_estimator": "geometric mean", "seed": SEED},
        "n_files": len(labels),
        "fitted_C": final_model.c,
        "loo_mape": shm_mape(damage, loo_predictions),
        "loo_score": shm_mape_score(damage, loo_predictions),
        "loo_ape_std": float(np.std(loo_errors)),
        "in_sample_mape": shm_mape(damage, in_sample_predictions),
        "in_sample_score": shm_mape_score(damage, in_sample_predictions),
        "previous_best_score": 0.790,
        "per_file_absolute_percentage_errors": [
            {
                "file_id": filename,
                "true_damage": float(truth),
                "loo_prediction": float(prediction),
                "absolute_percentage_error": float(error),
            }
            for filename, truth, prediction, error in zip(labels["filename"], damage, loo_predictions, loo_errors, strict=True)
        ],
        "two_c_experiment": _two_c_record(log_s, damage, np.asarray(descriptors)),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    final_model.save(
        metadata_path,
        fitted_on="64 training files only",
        c_estimator="mean(log(S)-log(D))",
        validation={"method": "leave-one-out", "mape": result["loo_mape"], "score": result["loo_score"]},
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate and fit SHM v2 on training files only")
    parser.add_argument("--train", default="PS3/02_Datasets/SHM/Train")
    parser.add_argument("--labels", default="PS3/02_Datasets/SHM/Train_Labels.csv")
    parser.add_argument("--cache", default="cache/shm_v2_cycles")
    parser.add_argument("--output", default="outputs/shm_v2/evaluation.json")
    parser.add_argument("--metadata", default="shm_v2/model_metadata.json")
    args = parser.parse_args(argv)
    result = evaluate(args.train, args.labels, args.cache, args.output, args.metadata)
    print(f"Previous best score: {result['previous_best_score']:.3f}")
    print(f"SHM v2 LOO: {result['loo_score']:.9f} (MAPE {result['loo_mape']:.9f})")


if __name__ == "__main__":
    main()
