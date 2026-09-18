#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${ACV_PYTHON:-.venv/bin/python}"
"$PYTHON" -m acv.audit
"$PYTHON" -m acv.prepare
"$PYTHON" -m pytest tests/acv -q -p no:cacheprovider
for model in thermal healthy_residual linear_ranker catboost_ranker tcn_mil; do
  status=0
  "$PYTHON" -m acv.evaluate --model "$model" --nested --device auto || status=$?
  if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
  if [[ "$status" -eq 2 ]]; then
    printf '%s\n' "$model is unavailable or exceeded its budget; consult its evaluation report."
  fi
done
"$PYTHON" -m acv.select
"$PYTHON" -m acv.train
"$PYTHON" -m acv.robustness
"$PYTHON" scripts/rich_diagnostics.py
"$PYTHON" scripts/write_results.py
printf '%s\n' 'Ready. Launch the app, upload the test workbook, then package the app-generated predictions.'
