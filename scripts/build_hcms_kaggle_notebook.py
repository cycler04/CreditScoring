"""Build the self-contained Kaggle notebook for the HCMS competition.

The generated notebook uses only competition-provided Parquet files and writes
the exact ``submission.csv`` artifact expected by Kaggle code submissions.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = (
    REPO_ROOT / "notebooks" / "kaggle" / "home-credit-model-stability-end-to-end"
)
NOTEBOOK_PATH = KERNEL_DIR / "home-credit-model-stability-end-to-end.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    notebook = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        }
    )
    notebook.cells = [
        markdown(
            """
# Home Credit Model Stability — end-to-end LightGBM

This notebook is a self-contained Kaggle code-submission pipeline:

1. load the competition Parquet files;
2. create numeric application and aggregated historical features;
3. validate with a chronological `WEEK_NUM` holdout;
4. train LightGBM with GPU acceleration and an automatic CPU fallback;
5. write `submission.csv`, `validation_metrics.json`, and feature importance.

It does not depend on another notebook, a private dataset, or a pre-trained model.
The public test set contains only 10 rows; Kaggle replaces it with the hidden test
set when this notebook is used as a code submission.
"""
        ),
        code(
            """
from __future__ import annotations

import gc
import json
import platform
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
COMPETITION = "home-credit-credit-risk-model-stability"
EXPECTED_KAGGLE_DATA = Path("/kaggle/input") / COMPETITION
LOCAL_DATA = Path("datasets/raw/home-credit-model-stability")
KAGGLE_RUNTIME = Path("/kaggle/working").exists()


def discover_kaggle_data() -> Path | None:
    \"\"\"Find the competition mount without assuming Kaggle's directory alias.\"\"\"
    candidates = [EXPECTED_KAGGLE_DATA]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(
            path.parent for path in input_root.rglob("sample_submission.csv")
        )
    for candidate in candidates:
        if (
            (candidate / "sample_submission.csv").exists()
            and (candidate / "parquet_files/train/train_base.parquet").exists()
            and (candidate / "parquet_files/test/test_base.parquet").exists()
        ):
            return candidate
    return None


KAGGLE_DATA = discover_kaggle_data()
REMOTE_VALIDATION_MODE = KAGGLE_RUNTIME and KAGGLE_DATA is None


def make_remote_validation_fixture(root: Path) -> None:
    \"\"\"Create a deterministic tiny fixture when an expired competition is not mounted.\"\"\"
    rng = np.random.default_rng(SEED)
    n_train, n_test = 20_000, 10
    split_dir = {
        split: root / "parquet_files" / split for split in ("train", "test")
    }
    for directory in split_dir.values():
        directory.mkdir(parents=True, exist_ok=True)

    public_test_case_ids = np.array(
        [57543, 57549, 57551, 57552, 57569, 57630, 57631, 57632, 57633, 57634],
        dtype=np.int64,
    )
    for split, rows in (("train", n_train), ("test", n_test)):
        case_id = (
            np.arange(rows, dtype=np.int64)
            if split == "train"
            else public_test_case_ids.copy()
        )
        week = np.arange(rows, dtype=np.int32) % 92
        income = rng.lognormal(10.0, 0.7, rows).astype("float32")
        debt = rng.lognormal(9.0, 0.9, rows).astype("float32")
        base_data = {
            "case_id": case_id,
            "date_decision": (
                np.datetime64("2019-01-01") + week.astype("timedelta64[W]")
            ).astype("datetime64[D]"),
            "MONTH": (week // 4).astype("int32"),
            "WEEK_NUM": week,
        }
        if split == "train":
            logit = -3.5 + 0.000002 * debt - 0.000001 * income + 0.15 * (week > 70)
            probability = 1.0 / (1.0 + np.exp(-logit))
            base_data["target"] = (rng.random(rows) < probability).astype("int8")
        pl.DataFrame(base_data).write_parquet(split_dir[split] / f"{split}_base.parquet")

        pl.DataFrame(
            {
                "case_id": case_id,
                "amt_annuity_100A": debt,
                "amt_income_200A": income,
                "dayspastdue_300P": rng.exponential(8, rows).astype("float32"),
                "active_400L": rng.integers(0, 2, rows, dtype=np.int8),
            }
        ).write_parquet(split_dir[split] / f"{split}_static_0_0.parquet")
        pl.DataFrame(
            {
                "case_id": case_id,
                "bureau_debt_500A": (debt * rng.uniform(0, 2, rows)).astype("float32"),
                "bureau_dpd_600P": rng.exponential(12, rows).astype("float32"),
            }
        ).write_parquet(split_dir[split] / f"{split}_static_cb_0.parquet")

        repeated_id = np.repeat(case_id, 2)
        pl.DataFrame(
            {
                "case_id": repeated_id,
                "mainoccupationinc_384A": np.repeat(income, 2)
                * rng.uniform(0.7, 1.1, rows * 2).astype("float32"),
                "incometype_1044T": rng.choice(
                    ["SALARIED", "SELFEMPLOYED"], rows * 2, p=[0.85, 0.15]
                ),
            }
        ).write_parquet(split_dir[split] / f"{split}_person_1.parquet")

        bureau_id = np.repeat(case_id, 3)
        pl.DataFrame(
            {
                "case_id": bureau_id,
                "pmts_pmtsoverdue_635A": rng.exponential(100, rows * 3).astype("float32"),
                "pmts_dpdvalue_108P": rng.exponential(15, rows * 3).astype("float32"),
            }
        ).write_parquet(split_dir[split] / f"{split}_credit_bureau_b_2.parquet")

    pd.DataFrame(
        {"case_id": public_test_case_ids, "score": 0.5}
    ).to_csv(root / "sample_submission.csv", index=False)


if KAGGLE_DATA is not None:
    DATA_ROOT = KAGGLE_DATA
elif LOCAL_DATA.exists():
    DATA_ROOT = LOCAL_DATA
elif REMOTE_VALIDATION_MODE:
    DATA_ROOT = Path("/kaggle/working/hcms-remote-validation-fixture")
    make_remote_validation_fixture(DATA_ROOT)
    print(
        "Competition files are not mounted in this normal post-deadline Kaggle run; "
        "using a deterministic schema fixture to validate the remote environment."
    )
else:
    raise FileNotFoundError(
        f"Competition data not found under /kaggle/input or at {LOCAL_DATA.resolve()}"
    )

sample_submission = pd.read_csv(DATA_ROOT / "sample_submission.csv")
PUBLIC_TEST_MODE = len(sample_submission) == 10
LOCAL_RUN = DATA_ROOT == LOCAL_DATA
TRAIN_ROW_LIMIT = 250_000 if LOCAL_RUN else None
N_ESTIMATORS = 450 if (LOCAL_RUN or PUBLIC_TEST_MODE) else 1_200

print(f"Data root: {DATA_ROOT}")
print(
    f"Mode: {'remote validation fixture' if REMOTE_VALIDATION_MODE else ('local smoke' if LOCAL_RUN else 'Kaggle competition')}, "
    f"public_test={PUBLIC_TEST_MODE}, train_limit={TRAIN_ROW_LIMIT}"
)
print(
    {
        "python": platform.python_version(),
        "polars": pl.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
    }
)
"""
        ),
        markdown(
            """
## Feature construction

The model uses the application base table, numeric static features, and compact
aggregates from two historical tables. Historical rows are reduced to one row per
`case_id` before joining, which keeps memory bounded on the full competition data.
"""
        ),
        code(
            """
NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64, pl.Boolean,
}


def parquet_files(split: str, table: str) -> list[Path]:
    files = sorted((DATA_ROOT / f"parquet_files/{split}").glob(f"{split}_{table}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split}_{table} Parquet files found")
    return files


def read_base(split: str) -> pl.DataFrame:
    frame = pl.read_parquet(parquet_files(split, "base")[0])
    if split == "train" and TRAIN_ROW_LIMIT is not None:
        frame = frame.head(TRAIN_ROW_LIMIT)

    date_dtype = frame.schema["date_decision"]
    date_expr = (
        pl.col("date_decision").str.to_date(strict=False)
        if date_dtype == pl.String
        else pl.col("date_decision").cast(pl.Date, strict=False)
    )
    return frame.with_columns(
        date_expr.alias("date_decision"),
        date_expr.dt.month().cast(pl.Int8).alias("decision_month"),
        date_expr.dt.weekday().cast(pl.Int8).alias("decision_weekday"),
    )


def discover_numeric_columns(files: list[Path]) -> list[str]:
    schema = pl.read_parquet_schema(files[0])
    columns = [
        name
        for name, dtype in schema.items()
        if name != "case_id"
        and dtype in NUMERIC_DTYPES
        and name.endswith(("P", "A", "L"))
    ]
    return sorted(columns)


def read_static(
    split: str,
    table: str,
    columns: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    files = parquet_files(split, table)
    columns = discover_numeric_columns(files) if columns is None else columns
    frames = []
    for path in files:
        available = pl.read_parquet_schema(path)
        expressions = [pl.col("case_id")]
        for name in columns:
            if name in available:
                expressions.append(pl.col(name).cast(pl.Float32, strict=False))
            else:
                expressions.append(pl.lit(None, dtype=pl.Float32).alias(name))
        frames.append(pl.read_parquet(path, columns=["case_id", *[c for c in columns if c in available]]).select(expressions))

    prefix = "s0" if table == "static_0" else "scb0"
    renamed = {name: f"{prefix}__{name}" for name in columns}
    frame = (
        pl.concat(frames, how="diagonal_relaxed")
        .unique("case_id", keep="first")
        .rename(renamed)
    )
    return frame, columns


def read_person_aggregates(split: str) -> pl.DataFrame:
    files = parquet_files(split, "person_1")
    wanted = ["mainoccupationinc_384A", "incometype_1044T"]
    frames = []
    for path in files:
        available = pl.read_parquet_schema(path)
        expressions = [pl.col("case_id")]
        if wanted[0] in available:
            expressions.append(
                pl.col(wanted[0]).cast(pl.Float32, strict=False).alias(wanted[0])
            )
        else:
            expressions.append(pl.lit(None, dtype=pl.Float32).alias(wanted[0]))
        if wanted[1] in available:
            expressions.append(
                (pl.col(wanted[1]) == "SELFEMPLOYED")
                .cast(pl.Int8)
                .alias("is_self_employed")
            )
        else:
            expressions.append(pl.lit(None, dtype=pl.Int8).alias("is_self_employed"))
        frames.append(pl.read_parquet(path).select(expressions))

    return (
        pl.concat(frames, how="diagonal_relaxed")
        .group_by("case_id")
        .agg(
            pl.col("mainoccupationinc_384A").max().alias("person__max_income"),
            pl.col("is_self_employed").max().alias("person__self_employed"),
            pl.len().cast(pl.Int32).alias("person__row_count"),
        )
    )


def read_credit_bureau_aggregates(split: str) -> pl.DataFrame:
    files = parquet_files(split, "credit_bureau_b_2")
    frames = []
    for path in files:
        schema = pl.read_parquet_schema(path)
        expressions = [pl.col("case_id")]
        for name in ["pmts_pmtsoverdue_635A", "pmts_dpdvalue_108P"]:
            if name in schema:
                expressions.append(pl.col(name).cast(pl.Float32, strict=False))
            else:
                expressions.append(pl.lit(None, dtype=pl.Float32).alias(name))
        frames.append(pl.read_parquet(path).select(expressions))

    return (
        pl.concat(frames, how="diagonal_relaxed")
        .group_by("case_id")
        .agg(
            pl.col("pmts_pmtsoverdue_635A").max().alias("cbb2__max_overdue"),
            pl.col("pmts_dpdvalue_108P").max().alias("cbb2__max_dpd"),
            (pl.col("pmts_dpdvalue_108P") > 31)
            .max()
            .cast(pl.Int8)
            .alias("cbb2__dpd_over_31"),
            pl.len().cast(pl.Int32).alias("cbb2__row_count"),
        )
    )


def build_features(
    split: str,
    static_columns: list[str] | None = None,
    static_cb_columns: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    started = time.time()
    base = read_base(split)
    case_ids = base.select("case_id")

    static, static_columns = read_static(split, "static_0", static_columns)
    static_cb, static_cb_columns = read_static(
        split, "static_cb_0", static_cb_columns
    )
    person = read_person_aggregates(split)
    bureau = read_credit_bureau_aggregates(split)

    frame = (
        base.join(static.join(case_ids, on="case_id", how="inner"), on="case_id", how="left")
        .join(static_cb.join(case_ids, on="case_id", how="inner"), on="case_id", how="left")
        .join(person.join(case_ids, on="case_id", how="inner"), on="case_id", how="left")
        .join(bureau.join(case_ids, on="case_id", how="inner"), on="case_id", how="left")
    )
    print(f"{split}: {frame.shape} built in {time.time() - started:.1f}s")
    return frame, static_columns, static_cb_columns
"""
        ),
        code(
            """
train_pl, static_columns, static_cb_columns = build_features("train")
test_pl, _, _ = build_features("test", static_columns, static_cb_columns)

train_pd = train_pl.to_pandas()
test_pd = test_pl.to_pandas()
del train_pl, test_pl
gc.collect()

excluded = {"case_id", "target", "date_decision", "WEEK_NUM"}
candidate_features = [c for c in train_pd.columns if c not in excluded]
features = [
    c
    for c in candidate_features
    if c in test_pd.columns
    and train_pd[c].notna().any()
    and train_pd[c].nunique(dropna=True) > 1
]

for column in features:
    train_pd[column] = pd.to_numeric(train_pd[column], errors="coerce").astype("float32")
    test_pd[column] = pd.to_numeric(test_pd[column], errors="coerce").astype("float32")

assert features, "No usable model features were created"
print(f"Usable numeric features: {len(features)}")
"""
        ),
        markdown(
            """
## Chronological validation and GPU training

The newest 20% of application weeks form the validation set. This is a
time-ordered holdout, not an out-of-time production validation claim. LightGBM
tries the Kaggle GPU first and automatically retries on CPU if the installed
LightGBM build has no GPU support.
"""
        ),
        code(
            """
weeks = np.sort(train_pd["WEEK_NUM"].dropna().unique())
cut_index = max(1, min(len(weeks) - 1, int(len(weeks) * 0.80)))
validation_start_week = int(weeks[cut_index])
train_mask = train_pd["WEEK_NUM"] < validation_start_week
valid_mask = ~train_mask

X_train = train_pd.loc[train_mask, features]
y_train = train_pd.loc[train_mask, "target"].astype("int8")
X_valid = train_pd.loc[valid_mask, features]
y_valid = train_pd.loc[valid_mask, "target"].astype("int8")

if y_train.nunique() < 2 or y_valid.nunique() < 2:
    raise ValueError("Chronological train/validation split must contain both classes")

base_params = {
    "objective": "binary",
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "max_bin": 63,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.20,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

device_used = "gpu"
model = lgb.LGBMClassifier(**base_params, device_type="gpu")
callbacks = [lgb.early_stopping(80, verbose=True), lgb.log_evaluation(50)]

try:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=callbacks,
    )
except Exception as error:
    print(f"GPU training unavailable ({type(error).__name__}: {error}); retrying on CPU")
    device_used = "cpu"
    model = lgb.LGBMClassifier(**base_params, device_type="cpu")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=callbacks,
    )

valid_prediction = model.predict_proba(X_valid)[:, 1]
validation_auc = float(roc_auc_score(y_valid, valid_prediction))
print(
    f"Validation AUC={validation_auc:.6f}; device={device_used}; "
    f"best_iteration={model.best_iteration_}"
)
"""
        ),
        markdown(
            """
## Submission and auditable run artifacts

The final assertions enforce the competition contract: identical row order to
`sample_submission.csv`, exact `case_id,score` columns, finite probabilities,
and values within `[0, 1]`.
"""
        ),
        code(
            """
test_prediction = model.predict_proba(
    test_pd[features],
    num_iteration=model.best_iteration_,
)[:, 1]

prediction_map = pd.DataFrame(
    {
        "case_id": test_pd["case_id"].astype(sample_submission["case_id"].dtype),
        "score": test_prediction,
    }
)
submission = (
    sample_submission[["case_id"]]
    .merge(prediction_map, on="case_id", how="left", validate="one_to_one")
)

assert list(submission.columns) == ["case_id", "score"]
assert len(submission) == len(sample_submission)
assert submission["case_id"].tolist() == sample_submission["case_id"].tolist()
assert submission["score"].notna().all()
assert np.isfinite(submission["score"]).all()
assert submission["score"].between(0.0, 1.0).all()

submission.to_csv("submission.csv", index=False)

importance = (
    pd.DataFrame(
        {
            "feature": features,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    )
    .sort_values("gain", ascending=False)
)
importance.to_csv("feature_importance.csv", index=False)

metrics = {
    "competition": COMPETITION,
    "mode": (
        "remote_validation_fixture"
        if REMOTE_VALIDATION_MODE
        else ("local_smoke" if LOCAL_RUN else ("public_test" if PUBLIC_TEST_MODE else "hidden_test"))
    ),
    "device_used": device_used,
    "train_rows": int(train_mask.sum()),
    "validation_rows": int(valid_mask.sum()),
    "test_rows": int(len(test_pd)),
    "feature_count": int(len(features)),
    "validation_start_week": validation_start_week,
    "validation_auc": validation_auc,
    "best_iteration": int(model.best_iteration_),
    "submission_columns": list(submission.columns),
    "submission_rows": int(len(submission)),
    "score_min": float(submission["score"].min()),
    "score_max": float(submission["score"].max()),
}
Path("validation_metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)

print(json.dumps(metrics, indent=2, sort_keys=True))
display(submission.head())
print("Wrote submission.csv, validation_metrics.json, and feature_importance.csv")
"""
        ),
    ]

    nbf.write(notebook, NOTEBOOK_PATH)

    metadata = {
        "id": "cyclerlol/hcms-end-to-end-lightgbm",
        "title": "HCMS End-to-End LightGBM",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["home-credit-credit-risk-model-stability"],
        "model_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {NOTEBOOK_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {(KERNEL_DIR / 'kernel-metadata.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
