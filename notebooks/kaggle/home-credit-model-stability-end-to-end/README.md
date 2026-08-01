# HCMS end-to-end Kaggle notebook

This private Kaggle kernel trains a compact LightGBM model directly from the
Home Credit Model Stability competition Parquet files. It uses GPU acceleration
with a CPU fallback and emits:

- `submission.csv`
- `validation_metrics.json`
- `feature_importance.csv`

Because this competition is past its deadline, Kaggle may omit the competition
files from an ordinary "Save Version" run. In that case only, the notebook builds
a deterministic small schema fixture to verify the Kaggle runtime. A code
submission still selects the competition mount first, while the local smoke run
uses the downloaded competition files.

Regenerate and validate the notebook locally:

```bash
uv run python scripts/build_hcms_kaggle_notebook.py
uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/kaggle/home-credit-model-stability-end-to-end/home-credit-model-stability-end-to-end.ipynb \
  --output /tmp/home-credit-model-stability-end-to-end.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Push and inspect it on Kaggle:

```bash
kaggle kernels push \
  -p notebooks/kaggle/home-credit-model-stability-end-to-end \
  --accelerator GPU
kaggle kernels status cyclerlol/hcms-end-to-end-lightgbm
kaggle kernels output cyclerlol/hcms-end-to-end-lightgbm \
  -p outputs/hcms/kaggle_notebook_run -o
```
