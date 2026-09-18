# ACV fault localisation

A reproducible, case-grouped car-ranking pipeline for NebulaX PS3. The output is an inspection order, not a calibrated diagnosis probability.

## Run the app

From this directory, on a network-connected Python 3.11 machine:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e ".[dev]"
streamlit run app/main.py
```

On Windows use `py -3.11 -m venv .venv` and `.venv\Scripts\Activate.ps1`. The supplied final model and app work without CatBoost if the selected artifact is thermal/linear/Huber. CatBoost is required to run its challenger experiment. `requirements.lock` records the environment actually used locally, including CPU PyTorch; it does not claim to lock the unavailable CatBoost package. No dependency installation is attempted at inference time.

To enable the CatBoost experiment, run `python -m pip install -r requirements-catboost.txt` (or install the `boosting` extra). Record the resolved version with `python -m pip freeze > requirements.lock` after validating it on the target machine.

For the other PC, install the CUDA-enabled PyTorch build recommended by the official [PyTorch installer](https://pytorch.org/get-started/locally/), then run `scripts/run_gpu.ps1` on Windows, or the neural command below on Linux. The GPU runner never assumes a remote machine is connected. Copy the original ACV datasets to their existing repository locations on that machine. Model artifacts produced on GPU infer on CPU.

Upload the held-out workbook through the app, select **Analyze uploaded files**, review the ranking, then download **predictions.zip**. The app also saves those exact bytes and a provenance record under `outputs/acv/app_export/`. Upload all required test workbooks in one batch; a later analysis intentionally replaces that export batch.

The distributable app does not include the organizers' raw datasets. Training examples are shown only when running in the original repository. A nontechnical user can use the upload workflow without the training files.

## Train and evaluate

```bash
python -m acv.audit --include-test
python -m acv.prepare
python -m pytest tests/acv -q -p no:cacheprovider
python -m acv.evaluate --model thermal --nested
python -m acv.evaluate --model healthy_residual --nested
python -m acv.evaluate --model linear_ranker --nested
python -m acv.evaluate --model catboost_ranker --nested
python -m acv.evaluate --model tcn_mil --nested --device cuda
python -m acv.select --results outputs/acv/evaluation
python -m acv.train --selection outputs/acv/selection.json
python -m acv.robustness
python scripts/rich_diagnostics.py
```

Use `--device auto` for the neural experiment to select CUDA if available, otherwise CPU. Neural evaluation has a four-hour limit; incomplete or unavailable experiments are explicitly excluded from selection. `--quick --max-epochs 2` is a smoke test only and is never accepted as model-selection evidence. All evaluations are nested even if `--nested` is omitted. Reports contain each outer fold's training files and hashes.

Preparation writes Parquet telemetry, feature tables and trusted local caches. Training labels enter the fit/evaluation code only. Test data can be audited but is never included in `training_data()` or model fitting. Cache keys include input content, labels, parameters and source version. The feature catalogue and finite model grid are recorded in code and evaluation reports; do not change them based on the unlabeled test ranking.

Compare CLI and app exports:

```bash
python predict.py --input PS3/02_Datasets/ACV/Test --output outputs/acv/cli_check/acv_predictions.csv --model artifacts/acv/final
python -m acv.validate_submission --input PS3/02_Datasets/ACV/Test --predictions outputs/acv/app_export/acv_predictions.csv
python scripts/package_submission.py --team purple
```

The final step requires the actual app export and a recorded demo video at `outputs/acv/demo_video.mp4`. It validates source/model hashes and excludes raw datasets. The resulting `delivery/purple/` follows the organizer's folder structure.

## Results and limitations

See `outputs/acv/selection.json` for the selected model and nested development score, `outputs/acv/evaluation/` for every challenger, and `docs/ACV_METHOD.md` for the methodology and run-specific findings. There are six independent labeled cases, so the estimate has high uncertainty. The exploratory mean-thermal benchmark gives ranks 1,1,1,2,1,1 (0.9792), which is not hidden-test accuracy.

Unavailable cars retain neutral score 0.5 and appear in every ranking. Pressure features are diagnostic only because just one training case provides them. For learned models, bootstrap output measures average hourly evidence stability, not a confidence interval for the full-case estimator; the frozen thermal-mean bootstrap resamples the full-case mean exactly.

Only load model artifacts you trust: `joblib` is not a safe format for arbitrary uploads. The app only accepts telemetry XLSX uploads, never user-supplied models.
