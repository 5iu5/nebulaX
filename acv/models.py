from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor, LogisticRegression
from .features import LINEAR_FEATURES, TREE_FEATURES, QUALITY_FEATURES, aggregate, thermal_scores

AGGREGATIONS = ["mean", "block_mean", "block_tail"]
# Production uses the validated, position-agnostic thermal baseline only.
# Challenger implementations remain below for reproducibility of historical experiments,
# but are not exposed as selectable production models.
MODEL_NAMES = ["thermal"]


def candidates(name):
    if name == "thermal":
        return [{"aggregation": a} for a in AGGREGATIONS]
    if name == "healthy_residual":
        return [{"alpha": x, "aggregation": a} for x, a in itertools.product([0.1, 1.0, 10.0], AGGREGATIONS)]
    if name == "linear_ranker":
        return [{"C": x} for x in [0.01, 0.1, 1.0]]
    if name == "catboost_ranker":
        return [dict(depth=d, iterations=i, l2_leaf_reg=l) for d, i, l in itertools.product([2, 3], [100, 300], [10, 30])]
    if name == "tcn_mil":
        return [{"max_epochs": 100}]
    raise ValueError(name)


def behaviour_frame(features):
    f = features.case.frame
    out = f[["peer_err", "ambient_shared", "target", "mode", "control", "since_transition"]].copy()
    out["since_transition"] = out.since_transition.clip(upper=10800) / 3600
    out["ambient_cooling"] = out.ambient_shared * f.eligible.astype(float)
    out["mode"] = out["mode"].fillna("unknown").astype(str)
    out["control"] = out.control.fillna("unknown").astype(str)
    return out


@dataclass
class Model:
    name: str
    params: dict
    estimator: object = None
    scaler: object = None
    feature_names: list = field(default_factory=list)
    training_files: list = field(default_factory=list)
    training_hashes: dict = field(default_factory=dict)

    def fit(self, data, labels, device="auto", deadline=None):
        self.training_files = sorted(data)
        self.training_hashes = {n: data[n].case.metadata["source_hash"] for n in data}
        if self.name == "thermal":
            return self
        if self.name == "healthy_residual":
            inputs, targets, weights = [], [], []
            for name, feat in sorted(data.items()):
                f = feat.case.frame
                mask = f.eligible & f.err.notna() & f.car.ne(labels[name])
                # Deterministic 5-minute thinning retains case/car balance and limits autocorrelation.
                mask &= (f.index.minute % 5 == 0) & (f.index.second == 0)
                x = behaviour_frame(feat).loc[mask]
                y = f.loc[mask, "err"]
                if not len(x):
                    continue
                inputs.append(x)
                targets.append(y)
                weights.extend([1 / len(x)] * len(x))
            if not inputs:
                raise ValueError("No valid healthy-car training observations")
            x = pd.concat(inputs)
            y = pd.concat(targets)
            nums = [c for c in x if c not in ["mode", "control"]]
            self.scaler = ColumnTransformer(
                [
                    ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), RobustScaler()), nums),
                    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["mode", "control"]),
                ]
            )
            xx = self.scaler.fit_transform(x)
            self.estimator = HuberRegressor(alpha=self.params["alpha"], epsilon=1.35, max_iter=500, tol=1e-5)
            w = np.asarray(weights)
            w *= len(w) / w.sum()
            self.estimator.fit(xx, y, sample_weight=w)
            self.feature_names = list(x.columns)
            return self
        if self.name == "tcn_mil":
            from .neural import train_neural

            self.estimator = train_neural(data, labels, self.params, device, deadline)
            self.feature_names = self.estimator["channels"]
            return self
        self.feature_names = list(LINEAR_FEATURES if self.name == "linear_ranker" else TREE_FEATURES)
        if self.params.get("quality"):
            self.feature_names += QUALITY_FEATURES[: max(0, 32 - len(self.feature_names))]
        tables = []
        groups = []
        ys = []
        for number, (name, feat) in enumerate(sorted(data.items())):
            x = feat.table.loc[feat.table.available, self.feature_names]
            tables.append(x)
            groups.extend([number] * len(x))
            ys.extend([int(c == labels[name]) for c in x.index])
        x = pd.concat(tables)
        if self.name == "linear_ranker":
            self.scaler = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), RobustScaler())
            xx = self.scaler.fit_transform(x)
            pairs, target, weight = [], [], []
            for group in sorted(set(groups)):
                idx = np.flatnonzero(np.asarray(groups) == group)
                pos = idx[np.asarray(ys)[idx] == 1]
                neg = idx[np.asarray(ys)[idx] == 0]
                if not len(pos) or not len(neg):
                    continue
                for n in neg:
                    diff = xx[pos[0]] - xx[n]
                    pairs.extend([diff, -diff])
                    target.extend([1, 0])
                    weight.extend([1 / (2 * len(neg))] * 2)
            if not pairs:
                raise ValueError("No positive/negative car pairs available")
            self.estimator = LogisticRegression(C=self.params["C"], fit_intercept=False, max_iter=1000, solver="lbfgs")
            self.estimator.fit(np.asarray(pairs), target, sample_weight=np.asarray(weight) * len(weight) / sum(weight))
        elif self.name == "catboost_ranker":
            from catboost import CatBoostRanker, Pool

            sizes = {g: groups.count(g) for g in set(groups)}
            pairs = []
            pw = []
            for g in sorted(sizes):
                idx = np.flatnonzero(np.asarray(groups) == g)
                pos = idx[np.asarray(ys)[idx] == 1]
                neg = idx[np.asarray(ys)[idx] == 0]
                if len(pos):
                    for n in neg:
                        pairs.append((int(pos[0]), int(n)))
                        pw.append(1 / max(1, len(neg)))
            pool = Pool(x, label=ys, group_id=groups, pairs=pairs, pairs_weight=pw)
            self.estimator = []
            for seed in [17, 42, 2026]:
                model = CatBoostRanker(
                    loss_function="PairLogit",
                    learning_rate=0.03,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=2,
                    **{k: v for k, v in self.params.items() if k != "quality"},
                )
                model.fit(pool)
                self.estimator.append(model)
        return self

    def predict(self, features):
        if self.name == "thermal":
            return thermal_scores(features, self.params["aggregation"])
        if self.name == "healthy_residual":
            f = features.case.frame
            pred = self.estimator.predict(self.scaler.transform(behaviour_frame(features)))
            residual = (f.err - pred).clip(lower=0).where(f.eligible)
            return pd.Series({c: aggregate(residual.loc[f.car.eq(c)], self.params["aggregation"]) for c in features.case.cars})
        if self.name == "tcn_mil":
            from .neural import predict_neural

            return predict_neural(self.estimator, features)
        x = features.table[self.feature_names]
        if self.name == "linear_ranker":
            values = self.estimator.decision_function(self.scaler.transform(x))
        else:
            values = np.mean([model.predict(x) for model in self.estimator], axis=0)
        return pd.Series(values, index=x.index).where(features.table.available)
