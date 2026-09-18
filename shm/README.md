# Structural Health Monitoring subsystem

This folder contains the complete SHM cumulative-fatigue-damage solution.

## Production inference

- `inference.py` reproduces the statistical and spectral feature engineering from Notebook 5.
- `shm_final_model.joblib` is the fitted median-imputation + Gradient Boosting regression pipeline.
- `shm_final_features.csv` defines the ordered 27-feature contract.
- `predict.py` provides batch command-line inference.
- `app.py` renders SHM inside the shared Streamlit application.

Run all held-out files from the repository root:

```bash
python -m shm.predict \
  --input PS3/02_Datasets/SHM/Test \
  --output outputs/shm/shm_predictions.csv
```

Run the integrated application:

```bash
streamlit run app/main.py
```

## Training and analysis

Run the numbered notebooks in order with `shm/` as the working directory. Their dataset paths point to `../PS3/02_Datasets/SHM/`. The final model artifact was regenerated from `shm_engineered_features.csv` using the exact documented Notebook 4 recipe so it can be loaded by the repository's current scikit-learn runtime. Its held-out predictions match the originally supplied predictions to floating-point precision.
