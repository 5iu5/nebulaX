# ACV methodology and reproducibility

## Task and data

ACV localises one leaking car per case and must rank every car ID from the file headers. The primary score is `(n - rank + 1) / n`, averaged over cases. All eight header cars are retained even if their telemetry is unavailable. The app, prediction archive and a demo video under three minutes are compulsory deliverables under the current PS3 specification. The ACV info kit's older description of mandatory code/CLI conflicts with the newer main specification; we include the CLI and code anyway.

Six training workbooks contain 56,993 timestamps and 48 car-level records. These are six independent labeled cases, not 56,993 independent labeled examples. Cases 01–03 are model A, 04 is model B and 05–06 are model C. Cases 01 and 02 share train 0620. The held-out input has 9,082 timestamps; its answer is unknown.

## Pipeline

Headers determine sensor aliases and exact car identifiers. Numerical invalid/sentinel values, missing observations and absent sensors have separate masks. Basic-schema validity flags gate thermal evidence; the rich-schema operating-mode field does not. In case 04, operating mode is nearly always Invalid although temperatures and pressures are usable, and cars 05–08 have no telemetry. Model C has outdoor sensors only at the train ends. Shared ambient context uses available sensors' timestamp median.

Native observations are retained for the frozen benchmark and short-event counts. The analysis grid is 30 seconds, with numerical medians and categorical modes. Numerical holes of at most 60 seconds may be filled only when bounded by the same observed mode/control/target state; explicit invalid readings are never interpolated. Larger gaps split episodes. Steady cooling begins after five minutes. Slopes require the same episode at both endpoints. Window coverage must be at least 70% of the nominal 15/60/180 minutes.

The frozen baseline reproduces the exploratory native cooling peer comparison. Optimized features use peers in the same reported cooling mode and require two usable peers. Whole-case, startup/steady, daily and window summaries capture magnitude, persistence and dynamics. IDs, dates, filenames and train identifiers never enter predictors. Quality fields are diagnosed separately; unavailable observations do not establish a healthy state.

## Models

- Thermal: native mean peer error; alternative equal-weight hourly means and mean/P90 blends.
- Healthy residual: Huber regression fit only on labeled healthy cars, equal total weight per case, deterministic five-minute thinning. Fold-fitted imputation, robust scaling and categorical encoding. Positive residuals are aggregated by car.
- Linear ranker: 12 fixed thermal/persistence/control features, within-case faulty/healthy differences and reversed pairs, L2 logistic loss, no intercept.
- CatBoost: 32 fixed features, PairLogit, explicit within-case pair weights, depth 2/3, 100/300 trees, learning rate .03, L2 10/30, three seeds. Missing dependency is reported as unavailable, never silently replaced.
- Temporal MIL: shared 32-channel dilated residual CNN over 120-step windows, dilations 1/2/4/8, dropout .2, masked pooling and a shared scoring head. Supervision is at case level. A training-only pilot chooses epochs, then refits all training cases; the outer test case cannot enter the pilot. Three seeds, AdamW and a four-hour evaluation budget. Each training update uses synchronized windows for all cars and equal case weighting. At most 16 evenly spaced windows per car are used for deterministic inference.

Pressure asymmetry, operating compressor pressures and duty imbalance are shown for rich telemetry but excluded from the production ranking. Their validity cannot be established from one rich case. No refrigerant-specific absolute pressure limits are assumed.

## Validation and selection

Each outer fold holds out a complete case. Aggregation, hyperparameters and ensemble weights are chosen using case-held-out predictions within the remaining five. A challenger must improve inner mean rank score without increasing worst rank; ties prefer top-1 performance then simpler models. The global selector chooses model families using each fold's inner results, never by inspecting that outer case's label. Final parameters use grouped CV on all six development cases. Component scores are within-case percentile ranks; unavailable cars receive neutral 0.5. Exact score ties use stable car identifiers.

Leave-one-train/model-family checks use frozen selected hyperparameters and are stress tests. First-6/12/24-hour checks use the full training fit and are explicitly descriptive. The 200-sample synchronized hourly bootstrap is exact for the native thermal mean. For other models it resamples mean hourly model evidence, a stability diagnostic rather than calibrated full-case uncertainty. The initial exploration used all six labeled cases, so even nested results remain development evidence with substantial uncertainty.

## Running and submission

Run the commands in `README_ACV.md`. Predictions are generated by the Streamlit upload action through the same inference service as the CLI. The app stores the exact CSV/ZIP bytes, input hashes and model hash. Packaging validates the schema and provenance and includes no raw datasets. The reproducible inference environment is recorded in `requirements.lock`; GPU/CatBoost installation is separate where required.

## Run-specific results

The implementation runner appends actual completed evaluations, limitations and verification evidence here after testing. Hidden-test correctness is not known.

### Completed local run
Selected model: **linear_ranker**; challenger weight 0.5.
Each nested row evaluates the inner-selected baseline/challenger blend, including fallback to thermal when the challenger fails the acceptance rule. It is not the standalone challenger score.

| Experiment | Status | Nested rank score | Top-1 |
|---|---|---:|---:|
| catboost_ranker | unavailable | — | — |
| healthy_residual | complete | 0.9791666666666666 | 0.8333333333333334 |
| linear_ranker | complete | 0.9791666666666666 | 0.8333333333333334 |
| tcn_mil | complete | 0.9791666666666666 | 0.8333333333333334 |
| thermal | complete | 0.9791666666666666 | 0.8333333333333334 |

catboost_ranker: No module named 'catboost'


Standalone temporal CNN: development LOCO rank score 0.9583, top-1 66.7%. This CPU experiment did not improve the baseline and was not retained.


The model-family selector's nested development summary is:

```json
{
  "cases": 6,
  "primary_metric": 0.9791666666666666,
  "top1_accuracy": 0.8333333333333334,
  "top2_accuracy": 1.0,
  "worst_rank": 2
}
```

These scores use held-out development cases, not the hidden test labels. Classical comparisons ran on the Mac; CUDA access on the separate PC was not configured. The model artifact records the exact dependencies, training hashes and preprocessing.

#### Robustness evidence

Group stress tests keep the selected hyperparameters fixed; they are not additional unbiased model-selection estimates.

- Leave-one-model_family-out: rank score 1.0000, top-1 100.0%.
- Leave-one-train-out: rank score 1.0000, top-1 100.0%.

Case 05 remains fragile: car 04 ranks first in 59.5% of synchronized hourly-evidence bootstrap samples. This diagnostic is not a fault probability or a confidence interval for the full-case model.

Operating-state-conditioned circuit and peer pressure comparisons are in `rich_diagnostics.json`. Their source units are unspecified and they do not change the production ranking.

#### Verification

Pytest: 28 tests, 0 failures, 0 errors. The suite includes all seven workbooks, leading-zero IDs, column/ID invariance, invalid and missing data, timing gaps, the score formula, fold separation, artifact integrity, neural car permutation, and app/CLI CSV parity.

The user is recording the required live demo manually. Packaging checks that it exists and lasts no more than 180 seconds; staged files are marked incomplete until that video is supplied.
