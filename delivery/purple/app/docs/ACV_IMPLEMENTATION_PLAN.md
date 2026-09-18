# ACV: accuracy-focused modelling and delivery plan

This document preserves the agreed implementation plan. Commands and deliverables below describe the intended workflow; they are not a claim that every implementation step or experiment has been completed.

## 1. Objective and findings

Build ACV first within **2–3 days**, with GPU access available. The recommended approach is **a physics-informed car-ranking pipeline**, with small learned models and a temporal neural network evaluated as challengers.

Optimize the official **linear rank-decay score**, while reporting top-1 accuracy separately:

\[
\text{score}=\frac{n-r+1}{n}
\]

For eight cars, ranks 1, 2, and 3 score 1.000, 0.875, and 0.750. Every car must appear in the submitted ranking. This follows the [ACV information kit](../PS3/03_References/ACV/ACV_Subsystem_Info_Kit.md).

### What the actual data shows

All seven workbooks were inspected, and exploratory calculations were run on the six labeled training cases.

| Case | Data rows | Columns | Dominant interval | Faulty car | Main consideration |
|---|---:|---:|---|---|---|
| 01 | 6,999 | 67 | 30 seconds | `01` | Invalid telemetry and recording gaps |
| 02 | 9,187 | 67 | 30 seconds | `02` | Subtle thermal difference |
| 03 | 8,310 | 67 | 30 seconds | `03` | Different cooling/ventilation patterns |
| 04 | 22,262 | 483 | **10 seconds** | `01` | Pressure/compressor telemetry; cars `05`–`08` entirely unavailable |
| 05 | 6,972 | 67 | 30 seconds | `04` | Outdoor readings available only for cars `01` and `08` |
| 06 | 3,263 | 67 | 30 seconds | `06` | Missing blocks and variable control settings |
| Test | 9,082 | 67 | 30 seconds | Unknown | Same basic schema as cases 01–03 |

There are **56,993 training timestamps but only six independent labeled fault cases**.

An exploratory baseline—ranking cars by their mean cooling-setpoint error relative to other cars—produced faulty-car ranks:

**`1, 1, 1, 2, 1, 1` → rank score 0.9792; top-1 accuracy 83.3%.**

These are development-data results, **not an estimate of hidden-test accuracy**. Case 05 is particularly fragile: its winning margin is approximately 0.02°C, and its faulty car ranks second on three of four individual days.

This makes careful preprocessing, temporal aggregation, and validation more valuable initially than a large neural architecture. No model can be promised to achieve the highest hidden-test accuracy before those comparisons.

## 2. Data pipeline and features

**Pipeline:** workbook → schema adapter → quality masks → operating episodes → comparative features → model scores → complete car ranking → app and CSV.

### Schema and quality handling

- Parse car identifiers from `Car <ID> - <parameter>` headers. Preserve identifiers as strings, including leading zeros; never depend on column order.
- Normalize aliases into cabin temperature, cooling target, ambient temperature, running mode, control mode, validity, and optional pressure/compressor channels.
- For case 04, map passenger-cabin temperature and target-temperature value to the shared thermal schema.
- Preserve separate masks for absent sensors, missing observations, explicit invalid readings, and unavailable cars.
- Mask invalid numerical readings, including the observed `-50` observation-area sentinel. Do not replace missing temperatures with zero.
- Apply validity rules by schema. Case 04’s operating-mode field is almost universally `Invalid` despite usable temperatures and pressures; dropping those rows would destroy the case.
- Treat Model C’s absent outdoor sensors as a sensor-layout limitation. Use the timestamp-level median of available outdoor sensors as shared ambient context.
- Exclude filenames, worksheet names, train numbers, absolute dates, and car IDs from learned fault predictors. Retain them for grouping, provenance, and output.

### Time and operating conditions

- Retain native timestamps for quality checks and event durations.
- Produce a 30-second analysis grid using numerical medians and categorical modes; preserve short-event counts separately.
- Interpolate numerical gaps only up to 60 seconds within unchanged operating states. Split episodes at longer gaps.
- Distinguish automatic, full, and half cooling from ventilation, stopped operation, and emergency ventilation.
- Maintain separate startup and steady-cooling features; define steady operation as at least five minutes after a relevant mode or setpoint change.
- Use actual elapsed time for temperature slopes and duration features.
- Generate 15-, 60-, and 180-minute summaries. Require at least 70% usable coverage for a window; retain whole-case summaries for short files.

### Feature groups

For car \(i\), define:

\[
e_i(t)=T_{\text{cabin},i}(t)-T_{\text{target},i}(t)
\]

\[
d_i(t)=e_i(t)-\operatorname{median}_{j\ne i} e_j(t)
\]

Calculate peer medians among usable cars in compatible cooling states, requiring at least two peers. When peer comparison is unavailable, retain standalone thermal evidence with a quality flag.

| Feature group | Features and purpose |
|---|---|
| Thermal burden | Mean, median, P90/P95, and positive area of setpoint error |
| Relative underperformance | Peer-adjusted error, within-case rank, and ambient-adjusted residual |
| Persistence | Fraction and longest duration above peer differences of 0.5°C, 1°C, and 2°C |
| Cooling response | Temperature slopes over 5/15 minutes and recovery after cooling activation |
| Control behaviour | Cooling duty, mode disagreement, manual-control duration, and transition counts |
| Temporal structure | Daily summaries, first-half versus second-half change, and sustained anomaly growth |
| Data quality | Usable duration, missing fraction, invalid events, and peer availability |
| Rich telemetry | Compressor duty imbalance, operating-state-conditioned pressure differences, and circuit asymmetry |

Retain absolute and peer-relative features. Avoid normalizing each car independently to zero mean: that could erase the fault signal.

Quality indicators must be ablated separately. A missing sensor or invalid event is not automatically evidence of refrigerant leakage.

## 3. Model architectures and selection

Implement the following models in order. All learned preprocessing must be fitted inside training folds.

### M0 — Thermal peer-ranking baseline

**Architecture:** valid cooling observations → setpoint errors → peer residuals → per-car aggregation → descending ranking.

Freeze the exploratory mean-peer-residual baseline as the first reference model. Evaluate two additional aggregation variants:

1. Equal weighting of eligible 60-minute block means.
2. Equal weighting of the block-mean average and block-mean P90.

Select aggregation using inner validation only. This tests whether persistent or intermittent underperformance best transfers between cases.

M0 is the production fallback and the benchmark every more complex model must beat.

### M1 — Healthy-behaviour residual model

**Architecture:** operating-condition features → robust regression predicting expected healthy setpoint error → observed-minus-expected residual → temporal aggregation → ranking.

- Fit a **Huber regressor** on the seven labeled healthy cars from training cases.
- Inputs: peer error, ambient temperature, setpoint, running/control mode, time since transition, and ambient-by-cooling interaction.
- Give each training case equal total weight so the larger rich case cannot dominate.
- Use fold-fitted numerical scaling and categorical encoding with unknown-category handling.
- Avoid using the candidate car’s contemporaneous cabin temperature as an input because it defines the target.
- Test regularization strengths `0.1, 1, 10`, with Huber epsilon `1.35`.
- Aggregate positive prediction residuals using the same alternatives as M0.

Huber regression limits the influence of outliers, making it a suitable small-data candidate. [Official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)

### M2 — Regularized linear ranker

**Architecture:** compact case-level feature vector per car → shared linear score → pairwise ranking loss.

- Use 12 predefined summaries spanning thermal burden, persistence, response, and control disagreement.
- Construct faulty-versus-healthy pairs **within each case**.
- Fit pairwise logistic regression on feature differences, with reversed pairs included and no intercept.
- Test L2 regularization through `C ∈ {0.01, 0.1, 1}`.
- Weight each case equally.
- Apply the same scoring function to every car; do not learn car-position preferences.

This directly learns which evidence should place the faulty car ahead of healthy cars without requiring a large model.

### M3 — Shallow CatBoost ranker

**Architecture:** shared case-level features → shallow boosted trees → car anomaly score.

Use `CatBoostRanker` with:

- Objective: `PairLogit`.
- Group: case ID.
- Labels: faulty `1`, healthy `0`.
- Depth: `2` or `3`.
- Iterations: `100` or `300`.
- Learning rate: `0.03`.
- L2 regularization: `10` or `30`.
- Seeds: `17, 42, 2026`.
- At most 32 predefined features, with missingness handled explicitly.

Use CPU training for this small dataset. There are only 48 case-car records; expanding timestamps into nominally independent examples would exaggerate the available supervision.

CatBoost’s pairwise objective compares objects within groups, matching the case-local ranking task. [Ranking documentation](https://catboost.ai/docs/en/concepts/loss-functions-ranking)

### M4 — GPU temporal challenger

Allocate **at most four hours of experimentation** to this model after the classical pipeline works.

**Architecture:**

`60-minute multichannel windows → shared temporal CNN → window embeddings → car-level pooling → shared MLP → eight car scores`

- Input: 120 steps at 30-second resolution.
- Channels: setpoint error, peer error, cabin-minus-ambient temperature, temperature changes, operating-state encodings, and observation masks.
- Four residual temporal-convolution blocks, 32 channels, kernel size 3, dilations `1, 2, 4, 8`, dropout `0.2`.
- Masked temporal mean/max pooling.
- Mean/max pooling across windows for each car.
- Shared `64 → 32 → 1` scoring head.
- Case-level cross-entropy over car scores, using the known faulty car.
- AdamW, learning rate `1e-3`, weight decay `1e-3`, maximum 100 epochs; early stopping uses inner held-out cases only.
- Three seeds; use synchronized window sampling across cars.

This is multiple-instance learning: the **case** carries the fault label. Individual windows are not assumed to show the fault continuously. A shared encoder also avoids learning fixed car identities.

Use a temporal CNN as the bounded neural experiment; its dilated convolutions provide temporal context with a relatively small architecture. [TCN research](https://arxiv.org/abs/1803.01271)

### Pressure/compressor branch

Case 04 needs additional interpretation: its faulty car exhibits circuit-pressure asymmetry and unequal compressor duty, while another car has greater thermal error.

- Compare pressures only under compatible compressor states.
- Extract simultaneous circuit asymmetry, peer-normalized pressure residuals, and compressor-duty imbalance.
- Use relative comparisons because pressure units and refrigerant-specific operating limits are not documented.
- Include these signals in the rich-case diagnostic report.
- Keep pressure-specific learned effects out of the production model by default: there is only one rich case, and the supplied test workbook has no pressure channels.

A pressure-augmented experiment may be reported, but improvement on that single case must not be presented as independent validation.

### Ensemble and final choice

- Compare M0 alone with M0 blended separately with each challenger.
- Convert component scores to within-case percentile ranks before blending.
- Test challenger weights `0.25, 0.50, 0.75`; select only within inner validation.
- Break selection ties by worst-case rank, then top-1 count, then lower model complexity.
- Retain a challenger only if it improves mean rank score without worsening the worst-case rank.
- Assign cars with no usable evidence a neutral percentile score of `0.5`; retain them in the ranking and label their evidence unavailable.
- Resolve exact final ties by car identifier for reproducibility.
- Show ranking scores and bootstrap stability, not unvalidated “probability of leakage.”

## 4. Validation and acceptance tests

### Validation design

Use **nested leave-one-case-out validation**:

1. Hold out one complete case as the outer evaluation case.
2. Use leave-one-case-out splits within the remaining five for model, hyperparameter, and blend selection.
3. Refit the selected pipeline on those five.
4. Predict all cars in the untouched outer case.
5. Repeat for all six cases.

Keep all windows, cars, augmentations, and derived features from a case in the same partition. Grouped splitting and nested selection prevent the most direct forms of training/validation leakage. [Grouped validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html), [nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

Also report:

- Leave-one-train-out results, grouping cases 01 and 02 together.
- Leave-one-car-model-family-out stress tests.
- Per-case faulty-car rank, official score, top-1/top-2 accuracy, and worst rank.
- First-day and truncated-file rankings as robustness checks, not extra independent test cases.
- Ranking stability from 200 synchronized one-hour block-bootstrap samples.

Freeze the feature catalogue and experiment grid before the main comparison. Because all six cases informed initial exploration, describe these results as development validation with substantial uncertainty.

The hidden test must not determine feature selection, thresholds, or ensemble weights.

### Required tests

| Area | Acceptance test |
|---|---|
| Parsing | All seven workbooks load; 67- and 483-column schemas map correctly |
| Identifiers | Leading zeros survive; shuffled columns leave predictions unchanged |
| Missing data | `None`, `Invalid`, sentinel readings, and entirely absent cars are handled distinctly |
| Timing | 10-second and 30-second inputs work; derivatives and episodes do not bridge long gaps |
| State handling | Case 04 remains usable despite its operating-mode field |
| Leakage | No case overlap between folds; fitted artifacts contain only training-case provenance |
| Invariance | Renaming car IDs consistently changes labels, not the underlying scores |
| Robustness | Missing ambient sensors, isolated spikes, and short missing blocks produce valid rankings |
| Metric | Ranks 1, 2, and 8 yield 1.000, 0.875, and 0.125; missing true car yields zero |
| Submission | Exact two-column schema; every header car appears once; filenames retain extensions |
| Reproducibility | Repeated inference from the same artifact produces identical ordering |
| Integration | App and CLI produce identical CSV content from the same input |

Use M0’s reproduced performance as the benchmark, not a promised minimum hidden-test score. Do not claim a model improvement from training-fit results alone.

## 5. Implementation, runbook, and delivery

### Implementation contract

Create one reusable Python package exposing:

- `load_case`: workbook to normalized telemetry and quality metadata.
- `extract_features`: normalized telemetry to case/car features and temporal windows.
- `rank_case`: model artifact plus telemetry to scores, evidence, and complete ranking.
- `write_predictions`: validated results to the official CSV schema.

Provide `predict.py --input <file-or-directory> --output <csv-path> --model <artifact-directory>` and make the app call the same ranking service.

Each model artifact must include preprocessing, feature definitions, model parameters, selected aggregation/blend settings, dependency versions, training-file hashes, and validation results.

Use Python 3.11, pandas, NumPy, SciPy, scikit-learn, CatBoost, PyArrow, openpyxl, Streamlit, Plotly, and pytest. Install PyTorch separately for the available GPU platform. Lock resolved dependencies.

### Planned commands

These are the interfaces specified by the plan. Consult the implementation runbook for their current implementation status and any platform-specific setup adjustments.

```bash
cd /Users/5iu5/Documents/nebulaX

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Audit and prepare training data:

```bash
python -m acv.audit --config configs/acv.yaml
python -m acv.prepare --config configs/acv.yaml
```

Run each model through the same nested evaluation:

```bash
python -m acv.evaluate --model thermal --nested
python -m acv.evaluate --model healthy_residual --nested
python -m acv.evaluate --model linear_ranker --nested
python -m acv.evaluate --model catboost_ranker --nested
python -m acv.evaluate --model tcn_mil --nested --device auto
```

Compare, test, and refit:

```bash
python -m acv.select --results outputs/acv/evaluation
python -m pytest tests/acv -q
python -m acv.train --selection outputs/acv/selection.json
```

CLI verification:

```bash
python predict.py \
  --input PS3/02_Datasets/ACV/Test \
  --output outputs/acv/cli_check/acv_predictions.csv \
  --model artifacts/acv/final
```

Launch the submission app:

```bash
streamlit run app/main.py
```

The app must let a nontechnical user upload a workbook, inspect the complete ranking and supporting temperature plots, see data-quality limitations, and download the predictions. Generate the **actual submission through the app**, as required by the [PS3 specification](../PS3/01_Problem_Statement_3_Specifications.md).

Validate the app-generated CSV before packaging:

```bash
python -m acv.validate_submission \
  --input PS3/02_Datasets/ACV/Test \
  --predictions outputs/acv/app_export/acv_predictions.csv
```

### Delivery schedule

| Time | Deliverable |
|---|---|
| Day 1 morning | Schema adapters, data audit, canonical telemetry, metric tests |
| Day 1 afternoon | Reproduced M0, feature extraction, basic upload-to-ranking app |
| Day 2 morning | M1–M3 nested comparisons and feature ablations |
| Day 2 afternoon | Bounded GPU challenger, robustness checks, model selection |
| Day 3 | Final refit, app/CLI parity, app-generated CSV, packaging, demo video |

Deliver the app, `predictions.zip` containing `acv_predictions.csv` at its top level, and a demo video of no more than three minutes. Include code, model artifacts, and a concise methodology report as supporting material.

**Defaults:** ACV only in this phase; offline whole-file inference; CPU-capable deployment; GPU used only for the temporal challenger. Build the app with a subsystem registry so Door, Rail Corrugation, and SHM can be added later. ACV contributes at most 25% of the overall four-subsystem score, so stop additional ACV experimentation when it no longer improves case-level validation.

### Confirmed delivery context

- Registered team name: **purple**.
- The GPU is on the user's other PC. GPU training must be portable to that machine; remote access is not assumed.
