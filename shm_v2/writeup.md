# SHM v2: Physics-Based Fatigue Damage Prediction

## Solution

SHM v2 predicts cumulative structural fatigue damage directly from a raw, single-channel dynamic-stress signal. It reproduces the physical fatigue-assessment process rather than approximating it with a generic regression model.

The pipeline:

1. Extracts stress cycles using ASTM E1049-85 rainflow counting, assigning `1.0` to full cycles and `0.5` to residual half cycles.
2. Converts each cycle range to stress amplitude, `sigma_a = range / 2`.
3. Computes the rainflow moment:

   `S(m) = sum(count * sigma_a^m)`

4. Applies the S-N relationship and Miner's linear damage rule:

   `predicted_damage = S(m) / C`

Nested leave-one-out validation selected `m = 5.00` in all 64 folds. The final model fits only the scale parameter `C`, using the geometric-mean log estimator over all training files. Its fitted value is `728,780,380.7316327`.

## Model performance

The official SHM metric is `max(0, 1 - MAPE)`. The primary result is the leave-one-out score because every file is predicted by a model fitted without that file.

| Evaluation | MAPE | Official score |
|---|---:|---:|
| Leave-one-out cross-validation | **2.714%** | **0.9729** |
| Repeated 8-fold CV, 20 repeats | **2.858% +/- 0.059%** | **0.9714 +/- 0.0006** |
| Full-training-set fit | 2.672% | 0.9733 |
| Previous generic-feature baseline | 21.0% | 0.7900 |

SHM v2 improves the cross-validated score over the previous baseline by approximately **0.183**, while reducing average percentage error from approximately **21.0% to 2.7%**. The small difference between training and leave-one-out MAPE—about **0.042 percentage points**—indicates little validation optimism.

Additional stability checks found:

- all 64 nested leave-one-out folds selected `m = 5.00`;
- fold-specific `C` values varied by only **0.32%**;
- repeated-CV scores ranged from **0.9705 to 0.9728**;
- the search-adjusted permutation test produced `p = 0.00020`; and
- no exact duplicate training files were present.

These figures estimate performance on unseen files from the same data distribution; the final hidden-test score remains unknown because test labels are unavailable.

## What makes SHM v2 unique

SHM v2 uses the known fatigue-generation mechanism as its model architecture. Unlike a black-box regressor built from generic summary statistics, every prediction is the sum of physically meaningful cycle-damage contributions.

After fixing the validated exponent, the model has only one fitted parameter. This makes it:

- interpretable and auditable;
- resistant to overfitting;
- computationally lightweight at inference time;
- directly connected to established fatigue engineering practice; and
- explainable through its rainflow-amplitude histogram and cumulative-damage curve.

Nested and repeated cross-validation produced similar scores, fold-specific values of `C` varied by only 0.32%, and a search-adjusted permutation test gave `p = 0.00020`. No test data was used for fitting or model selection.

## SHM v2 technology stack

- **Python** — package, evaluation and command-line inference.
- **NumPy** — numerical arrays, cycle contributions and log-space calculations.
- **Pandas** — headerless signal loading, cycle tables and prediction CSVs.
- **SciPy** — statistical diagnostics used during model verification.
- **PyArrow / Parquet** — cached rainflow cycle tables, avoiding repeated cycle extraction.
- **scikit-learn** — cross-validation and diagnostic clustering experiments.
- **Streamlit** — interactive SHM upload, analysis and validated CSV download.
- **Plotly** — rainflow-amplitude and cumulative-damage visualisations.
- **Pytest** — ASTM cycle-counting, metric and end-to-end inference tests.
- **Ruff** — static code-quality checks.

## SHM v2 artefacts

- `model_metadata.json` — final fitted `m` and `C`.
- `evaluate.py` — train-only leave-one-out evaluation and metadata fitting.
- `features.py` — ASTM rainflow counting and `S(m)` calculation.
- `models.py` — analytical `D = S(m) / C` model.
- `inference.py` — reusable prediction and schema validation.
- `predict.py` — command-line batch prediction.
- `metrics.py` — official MAPE-derived score.
- `app.py` — SHM Streamlit page and visual explanations.
- `../outputs/shm_v2/evaluation.json` — validation results and per-file errors.
- `../outputs/shm_v2/fairness_overfit_audit.json` — robustness and fairness diagnostics.
