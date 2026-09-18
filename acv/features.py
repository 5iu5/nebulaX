from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import Case

LINEAR_FEATURES = [
    "peer_mean",
    "peer_p90",
    "peer_p95",
    "err_mean",
    "err_p90",
    "positive_error",
    "peer_gt_05",
    "peer_gt_1",
    "peer_long_1",
    "slope_5",
    "manual_fraction",
    "mode_disagreement",
]
TREE_FEATURES = LINEAR_FEATURES + [
    "err_median",
    "err_p95",
    "peer_gt_2",
    "peer_long_05",
    "peer_long_2",
    "slope_15",
    "startup_error",
    "steady_error",
    "cooling_fraction",
    "full_fraction",
    "half_fraction",
    "late_change",
    "daily_max",
    "block15_p90",
    "block60_p90",
    "block180_p90",
    "ambient_error",
    "block60_mean",
    "recovery_15",
    "positive_peer",
]
QUALITY_FEATURES = ["missing_fraction", "invalid_fraction", "peer_coverage", "usable_hours"]


@dataclass
class Features:
    case: Case
    table: pd.DataFrame
    blocks: dict


def longest_minutes(mask, index):
    if not len(mask):
        return 0.0
    a = np.asarray(mask.fillna(False), dtype=bool)
    breaks = index.to_series().diff().dt.total_seconds().ne(30).to_numpy()
    best = run = 0
    for positive, gap in zip(a, breaks, strict=True):
        if gap:
            run = 0
        run = run + 1 if positive else 0
        best = max(best, run)
    return best * 0.5


def aggregate(series, kind="mean"):
    valid = series.dropna()
    if valid.empty:
        return np.nan
    if kind == "mean":
        return float(valid.mean())
    counts = series.resample("60min").count()
    means = series.resample("60min").mean().where(counts >= 84).dropna()
    if means.empty:
        return float(valid.mean())
    if kind == "block_mean":
        return float(means.mean())
    if kind == "block_tail":
        return float(0.5 * means.mean() + 0.5 * means.quantile(0.9))
    raise ValueError(f"Unknown aggregation {kind}")


def extract_features(case):
    rows, blocks = [], {}
    all_modes = case.frame.pivot(columns="car", values="mode")
    for car in case.cars:
        f = case.frame.loc[case.frame.car.eq(car)].copy()
        raw = case.native.loc[case.native.car.eq(car)]
        e, p = f.err, f.peer
        usable = f.eligible
        row = {
            "car": car,
            "available": bool(usable.any()),
            "usable_hours": float(usable.sum() / 120),
            "missing_fraction": float(f.cabin_missing.mean()),
            "invalid_fraction": float((f.cabin_invalid | f.validity_invalid | f.target_invalid).mean()),
            "peer_coverage": float(p.notna().sum() / max(1, usable.sum())),
            "native_thermal": float(raw.peer.mean()) if raw.peer.notna().any() else float(raw.err.mean()),
            "err_mean": e.mean(),
            "err_median": e.median(),
            "err_p90": e.quantile(0.9),
            "err_p95": e.quantile(0.95),
            "peer_mean": p.mean(),
            "peer_p90": p.quantile(0.9),
            "peer_p95": p.quantile(0.95),
            "positive_error": e.clip(lower=0).mean(),
            "positive_peer": p.clip(lower=0).mean(),
            "slope_5": f.slope_5.mean(),
            "slope_15": f.slope_15.mean(),
            "startup_error": e.where(f.since_transition.lt(300)).mean(),
            "steady_error": e.where(f.steady).mean(),
            "cooling_fraction": usable.mean(),
            "full_fraction": f["mode"].eq("Full Cooling").mean(),
            "half_fraction": f["mode"].eq("Half Cooling").mean(),
            "manual_fraction": f.control.eq("Manual Control").where(f.control.notna()).mean(),
            "invalid_events": f.invalid_events.sum(),
            "mode_events": f.mode_events.sum(),
            "ambient_error": (f.cabin - f.ambient_shared).where(usable).mean(),
        }
        other_modes = all_modes.drop(columns=car)
        disagreement = other_modes.ne(all_modes[car], axis=0).where(other_modes.notna()).mean(axis=1)
        row["mode_disagreement"] = disagreement.where(f["mode"].notna()).mean()
        for val, name in [(0.5, "05"), (1.0, "1"), (2.0, "2")]:
            row[f"peer_gt_{name}"] = float((p > val).sum() / p.count()) if p.count() else np.nan
            row[f"peer_long_{name}"] = longest_minutes(p > val, f.index) if p.count() else np.nan
        half = f.index.min() + (f.index.max() - f.index.min()) / 2
        row["late_change"] = p.loc[p.index > half].mean() - p.loc[p.index <= half].mean()
        row["daily_max"] = p.groupby(p.index.date).mean().max()
        # Temperature improvement 15 minutes after a cooling episode begins.
        starts = f.eligible & f.since_transition.eq(0)
        recovery = f.cabin - f.cabin.shift(-30)
        row["recovery_15"] = recovery.where(starts & f.episode.eq(f.episode.shift(-30))).mean()
        blocks[car] = {}
        for minutes in [15, 60, 180]:
            means = p.resample(f"{minutes}min").mean()
            coverage = p.resample(f"{minutes}min").count() / (minutes * 2)
            means = means.where(coverage >= 0.7)
            blocks[car][minutes] = pd.DataFrame({"mean": means, "coverage": coverage})
            row[f"block{minutes}_mean"] = means.mean()
            row[f"block{minutes}_p90"] = means.quantile(0.9)
        # Diagnostic-only rich features; never in production predictor catalogue.
        c1, c2 = f.compressor1.eq(1), f.compressor2.eq(1)
        both = c1 & c2
        row["pressure_low_asymmetry"] = (f.p1_low - f.p2_low).abs().where(both).median()
        row["pressure_high_asymmetry"] = (f.p1_high - f.p2_high).abs().where(both).median()
        row["compressor_duty_imbalance"] = abs(f.compressor1.mean() - f.compressor2.mean())
        for k in [1, 2]:
            for side in ["low", "high"]:
                row[f"pressure_{k}_{side}_on"] = f[f"p{k}_{side}"].where(f[f"compressor{k}"].eq(1)).median()
        row["both_compressors_hours"] = both.sum() / 120
        rows.append(row)
    table = pd.DataFrame(rows).set_index("car")
    return Features(case, table, blocks)


def thermal_scores(features, aggregation="mean"):
    if aggregation == "mean":
        return features.table.native_thermal.copy()
    return pd.Series({c: aggregate(features.case.frame.loc[features.case.frame.car.eq(c), "peer"], aggregation) for c in features.case.cars})


def percentiles(scores, available=None):
    scores = pd.Series(scores, dtype=float).copy()
    if available is not None:
        scores = scores.where(pd.Series(available).reindex(scores.index).fillna(False))
    good = scores.dropna()
    if len(good) <= 1:
        out = pd.Series(0.5, index=scores.index)
        if len(good):
            out.loc[good.index] = 1.0
        return out
    return ((scores.rank(method="average") - 1) / (len(good) - 1)).fillna(0.5)


def ordered(scores):
    return sorted(scores.index, key=lambda c: (-float(scores[c]), str(c)))
