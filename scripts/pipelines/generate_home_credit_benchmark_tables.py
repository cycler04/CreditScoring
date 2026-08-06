#!/usr/bin/env python3
"""Build comparable HCDR and HCMS benchmark tables from saved artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import Pool
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from credit_scoring.benchmarking import (
    build_benchmark_table,
    write_benchmark_report,
)
from credit_scoring.scorecard import is_monotonic_woe
from home_credit_default_rate.pipeline import _woe_transform as hcdr_woe_transform
from home_credit_stability.pipeline import _woe_transform as hcms_woe_transform

ENSEMBLE_MEMBERS = {
    "ensemble_lightgbm_catboost": ("lightgbm", "catboost"),
    "ensemble_lightgbm_xgboost_catboost": (
        "lightgbm",
        "xgboost",
        "catboost",
    ),
    "ensemble_lightgbm_catboost_extra_trees": (
        "lightgbm",
        "catboost",
        "extra_trees",
    ),
    "ensemble_boosting": (
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
    ),
    "ensemble_all_trees": (
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    ),
}

BASE_MODELS = [
    "logistic_raw",
    "lightgbm",
    "xgboost",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "catboost",
]
INTERPRETABLE_MODELS = ["gam", "monotonic_lightgbm"]
FITTED_MODELS = [*BASE_MODELS, "logistic_woe", *INTERPRETABLE_MODELS]
METRIC_RECONSTRUCTION_TOLERANCE = 1e-5
EXPLANATION_ROWS = 100
EXPLANATION_REPEATS = 5


def _timed_explanation(callback: Callable[[], object], rows: int) -> float:
    callback()
    durations = []
    for _ in range(EXPLANATION_REPEATS):
        started = time.perf_counter()
        callback()
        durations.append(time.perf_counter() - started)
    return 1000.0 * median(durations) / rows


def _linear_explanation_time(values: object, coefficients: np.ndarray) -> float:
    sample = values[:EXPLANATION_ROWS]
    callback = (
        (lambda: sample.multiply(coefficients))
        if hasattr(sample, "multiply")
        else (lambda: np.asarray(sample) * coefficients)
    )
    return _timed_explanation(callback, len(sample))


def _blend(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        ensemble: np.mean([predictions[member] for member in members], axis=0)
        for ensemble, members in ENSEMBLE_MEMBERS.items()
    }


def _catboost_frame(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    for feature in categorical_features:
        result[feature] = result[feature].astype("string").fillna("MISSING")
    return result


def _importance_counts(dataset: str) -> dict[str, int | None]:
    directory = Path("outputs") / dataset / "models/feature_importance"
    counts: dict[str, int | None] = {}
    for model in FITTED_MODELS:
        stem = "lightgbm_C" if dataset == "hcms" and model == "lightgbm" else model
        path = directory / f"{stem}.csv"
        if not path.exists():
            counts[model] = None
            continue
        table = pd.read_csv(path)
        value_column = "importance" if "importance" in table else "importance_value"
        counts[model] = int(table[value_column].abs().gt(0.0).sum())
    for ensemble in ENSEMBLE_MEMBERS:
        counts[ensemble] = None
    return counts


def _monotonic_violations(artifact_path: Path) -> dict[str, int | None]:
    artifact = joblib.load(artifact_path)
    violations = sum(
        not is_monotonic_woe(table) for table in artifact["tables"].values()
    )
    result = {model: None for model in [*FITTED_MODELS, *ENSEMBLE_MEMBERS]}
    result["logistic_woe"] = int(violations)
    result["monotonic_lightgbm"] = 0
    return result


def _write_outputs(
    dataset: str,
    metrics: pd.DataFrame,
    labels: pd.Series,
    predictions: dict[str, np.ndarray],
    *,
    stability: dict[str, float] | None,
    stability_definition: str,
    explanation_times: dict[str, float | None],
) -> None:
    measured = metrics.loc[metrics["split"].eq("test")].drop_duplicates(
        "model", keep="last"
    )
    for row in measured.itertuples(index=False):
        scores = predictions[row.model]
        auc = float(roc_auc_score(labels, scores))
        false_positive, true_positive, _ = roc_curve(labels, scores)
        ks = float(np.max(true_positive - false_positive))
        if not np.isclose(
            auc, row.auc, atol=METRIC_RECONSTRUCTION_TOLERANCE, rtol=0.0
        ):
            raise ValueError(
                f"{dataset}/{row.model} AUC drift: {auc} != {row.auc}"
            )
        if not np.isclose(
            ks, row.ks, atol=METRIC_RECONSTRUCTION_TOLERANCE, rtol=0.0
        ):
            raise ValueError(
                f"{dataset}/{row.model} KS drift: {ks} != {row.ks}"
            )
    brier = {
        model: float(brier_score_loss(labels, scores))
        for model, scores in predictions.items()
    }
    output_dir = Path("outputs") / dataset / "models/metrics"
    table = build_benchmark_table(
        metrics,
        brier,
        active_features=_importance_counts(dataset),
        stability=stability,
        monotonic_violations=_monotonic_violations(
            Path("outputs") / dataset / "scorecard/logistic_woe.joblib"
        ),
        explanation_times=explanation_times,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "benchmark_table.csv", index=False, na_rep="N/A")
    write_benchmark_report(
        table,
        output_dir / "benchmark_table.md",
        title=(
            "Home Credit Default Risk benchmark"
            if dataset == "hcdr"
            else "Home Credit Model Stability benchmark"
        ),
        stability_definition=stability_definition,
    )
    protocol = {
        "dataset": dataset,
        "evaluation_split": (
            "stratified random test" if dataset == "hcdr" else "out-of-time test"
        ),
        "active_features": (
            "Count of non-zero persisted global-importance entries; N/A when the "
            "saved estimator has no native importance or for ensembles."
        ),
        "stability": stability_definition,
        "monotonic_violations": (
            "Counted only for models with enforced and auditable monotonicity."
        ),
        "explanation_time": (
            "Median estimator-native exact local-attribution milliseconds per row "
            "on 100 model-ready test rows, after one warm-up and five measured runs."
        ),
        "explanation_rows": EXPLANATION_ROWS,
        "explanation_repeats": EXPLANATION_REPEATS,
        "candidate_rows_are_measured": False,
        "metric_reconstruction_absolute_tolerance": (
            METRIC_RECONSTRUCTION_TOLERANCE
        ),
    }
    (output_dir / "benchmark_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(dataset, "rows", len(table), "models", len(predictions))


def build_hcdr() -> None:
    processed = Path("datasets/processed/hcdr")
    output = Path("outputs/hcdr")
    membership = pd.read_csv(processed / "split_membership.csv")
    test_ids = membership.loc[membership["split"].eq("test"), "SK_ID_CURR"]
    matrix = pd.read_parquet(processed / "feature_matrix.parquet")
    test = matrix.loc[matrix["SK_ID_CURR"].isin(test_ids)].copy()
    labels = test["TARGET"].astype("int8")
    predictions: dict[str, np.ndarray] = {}
    explanation_times: dict[str, float | None] = {
        model: None for model in [*FITTED_MODELS, *ENSEMBLE_MEMBERS]
    }
    for model_name in BASE_MODELS:
        artifact = joblib.load(output / f"models/{model_name}.joblib")
        if model_name == "catboost":
            values = _catboost_frame(
                test[artifact["features"]], artifact["categorical_features"]
            )
            pool = Pool(
                values.iloc[:EXPLANATION_ROWS],
                cat_features=artifact["categorical_features"],
            )
            explanation_times[model_name] = _timed_explanation(
                lambda: artifact["model"].get_feature_importance(
                    pool, type="ShapValues"
                ),
                pool.num_row(),
            )
        else:
            processor = artifact["preprocessor"]
            values = processor.transform(test[list(processor.feature_names_in_)])
            if artifact.get("requires_dense", False):
                if hasattr(values, "toarray"):
                    values = values.toarray()
                values = np.asarray(values, dtype="float32")
            if model_name == "logistic_raw":
                explanation_times[model_name] = _linear_explanation_time(
                    values, artifact["model"].coef_[0]
                )
            elif model_name == "lightgbm":
                sample = values[:EXPLANATION_ROWS]
                explanation_times[model_name] = _timed_explanation(
                    lambda: artifact["model"].predict(sample, pred_contrib=True),
                    len(sample),
                )
            elif model_name == "xgboost":
                sample = xgb.DMatrix(values[:EXPLANATION_ROWS])
                explanation_times[model_name] = _timed_explanation(
                    lambda: artifact["model"].get_booster().predict(
                        sample, pred_contribs=True
                    ),
                    sample.num_row(),
                )
        predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
    predictions.update(_blend(predictions))
    woe = joblib.load(output / "scorecard/logistic_woe.joblib")
    woe_values = hcdr_woe_transform(test, woe["bins"], woe["tables"])
    predictions["logistic_woe"] = woe["model"].predict_proba(woe_values)[:, 1]
    explanation_times["logistic_woe"] = _linear_explanation_time(
        woe_values, woe["model"].coef_[0]
    )
    for model_name in INTERPRETABLE_MODELS:
        artifact = joblib.load(output / f"models/{model_name}.joblib")
        if model_name == "gam":
            features = artifact["features"]
            values = test[features].replace([np.inf, -np.inf], np.nan).astype("float32")
            predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
            transformed = artifact["model"][:-1].transform(
                values.iloc[:EXPLANATION_ROWS]
            )
            explanation_times[model_name] = _linear_explanation_time(
                transformed, artifact["model"].named_steps["model"].coef_[0]
            )
        else:
            values = hcdr_woe_transform(
                test, artifact["bins"], artifact["tables"]
            )
            predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
            sample = values.iloc[:EXPLANATION_ROWS]
            explanation_times[model_name] = _timed_explanation(
                lambda: artifact["model"].predict(sample, pred_contrib=True),
                len(sample),
            )
    metrics = pd.concat(
        [
            pd.read_csv(output / "models/metrics.csv"),
            pd.read_csv(output / "models/interpretable_metrics.csv"),
        ],
        ignore_index=True,
    )
    _write_outputs(
        "hcdr",
        metrics,
        labels,
        predictions,
        stability=None,
        stability_definition=(
            "N/A because HCDR has no time column and uses a stratified random split; "
            "the valid-test gap is not treated as temporal stability."
        ),
        explanation_times=explanation_times,
    )


def build_hcms() -> None:
    processed = Path("datasets/processed/hcms")
    output = Path("outputs/hcms")
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    artifacts = {
        model: joblib.load(output / f"models/{model}.joblib")
        for model in BASE_MODELS
    }
    woe = joblib.load(output / "scorecard/logistic_woe.joblib")
    raw_features = sorted(
        {feature for artifact in artifacts.values() for feature in artifact["features"]}
        | set(woe["bins"])
    )
    week_range = summary["split_weeks"]["test"]
    test = (
        pl.scan_parquet(processed / "feature_matrix_train_C.parquet")
        .filter(
            pl.col("WEEK_NUM").is_between(week_range["min"], week_range["max"])
        )
        .select(["case_id", "WEEK_NUM", "target", *raw_features])
        .collect()
        .to_pandas()
    )
    labels = test["target"].astype("int8")
    predictions: dict[str, np.ndarray] = {}
    explanation_times: dict[str, float | None] = {
        model: None for model in [*FITTED_MODELS, *ENSEMBLE_MEMBERS]
    }
    for model_name, artifact in artifacts.items():
        features = artifact["features"]
        transformed = artifact["imputer"].transform(test[features]).astype("float32")
        names = artifact["imputer"].get_feature_names_out(features)
        values: object = pd.DataFrame(transformed, columns=names, index=test.index)
        if model_name == "logistic_raw":
            values = artifact["scaler"].transform(values)
        elif "transformed_features" in artifact:
            values = values[artifact["transformed_features"]]
        if model_name == "logistic_raw":
            explanation_times[model_name] = _linear_explanation_time(
                values, artifact["model"].coef_[0]
            )
        elif model_name == "lightgbm":
            sample = values[:EXPLANATION_ROWS]
            explanation_times[model_name] = _timed_explanation(
                lambda: artifact["model"].predict(sample, pred_contrib=True),
                len(sample),
            )
        elif model_name == "xgboost":
            sample = xgb.DMatrix(values[:EXPLANATION_ROWS])
            explanation_times[model_name] = _timed_explanation(
                lambda: artifact["model"].get_booster().predict(
                    sample, pred_contribs=True
                ),
                sample.num_row(),
            )
        elif model_name == "catboost":
            pool = Pool(values[:EXPLANATION_ROWS])
            explanation_times[model_name] = _timed_explanation(
                lambda: artifact["model"].get_feature_importance(
                    pool, type="ShapValues"
                ),
                pool.num_row(),
            )
        predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
    predictions.update(_blend(predictions))
    woe_values = hcms_woe_transform(test, woe["bins"], woe["tables"])
    predictions["logistic_woe"] = woe["model"].predict_proba(woe_values)[:, 1]
    explanation_times["logistic_woe"] = _linear_explanation_time(
        woe_values, woe["model"].coef_[0]
    )
    for model_name in INTERPRETABLE_MODELS:
        artifact = joblib.load(output / f"models/{model_name}.joblib")
        if model_name == "gam":
            features = artifact["features"]
            values = test[features].replace([np.inf, -np.inf], np.nan).astype("float32")
            predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
            transformed = artifact["model"][:-1].transform(
                values.iloc[:EXPLANATION_ROWS]
            )
            explanation_times[model_name] = _linear_explanation_time(
                transformed, artifact["model"].named_steps["model"].coef_[0]
            )
        else:
            values = hcms_woe_transform(
                test, artifact["bins"], artifact["tables"]
            )
            predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
            sample = values.iloc[:EXPLANATION_ROWS]
            explanation_times[model_name] = _timed_explanation(
                lambda: artifact["model"].predict(sample, pred_contrib=True),
                len(sample),
            )
    stability_rows = json.loads(
        (output / "stability/stability_metric.json").read_text(encoding="utf-8")
    )
    stability_rows.extend(
        json.loads(
            (output / "models/interpretable_stability.json").read_text(
                encoding="utf-8"
            )
        )
    )
    metrics = pd.concat(
        [
            pd.read_csv(output / "models/metrics.csv"),
            pd.read_csv(output / "models/interpretable_metrics.csv"),
        ],
        ignore_index=True,
    )
    _write_outputs(
        "hcms",
        metrics,
        labels,
        predictions,
        stability={row["model"]: float(row["stability"]) for row in stability_rows},
        stability_definition=(
            "the week-based metric `mean(gini) + 88 * min(0, slope) - "
            "0.5 * residual_std` over 19 out-of-time test weeks."
        ),
        explanation_times=explanation_times,
    )


def main() -> None:
    build_hcdr()
    build_hcms()


if __name__ == "__main__":
    main()
