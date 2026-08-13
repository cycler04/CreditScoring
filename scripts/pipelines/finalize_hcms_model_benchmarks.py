#!/usr/bin/env python3
"""Finalize HCMS benchmark outputs from persisted Stage C model artifacts."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from credit_scoring.visualization import (
    write_gini_by_period_plot,
    write_gini_curve,
    write_ks_curve,
    write_roc_auc_curve,
)
from home_credit_stability.data import ID_COLUMN, TARGET
from home_credit_stability.pipeline import (
    ENSEMBLE_MEMBERS,
    _equal_weight_ensembles,
    _metric_row,
    _ordered_submission,
    _woe_transform,
    _write_stability,
)
from home_credit_stability.stability import stability_metric

BASE_MODELS = [
    "lightgbm",
    "logistic_raw",
    "xgboost",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "catboost",
]


def _load_split(
    matrix_path: Path,
    features: list[str],
    week_range: dict[str, int] | None,
) -> pd.DataFrame:
    columns = [ID_COLUMN, "WEEK_NUM", *features]
    if "train" in matrix_path.name:
        columns.append(TARGET)
    scan = pl.scan_parquet(matrix_path).select(columns)
    if week_range is not None:
        scan = scan.filter(
            pl.col("WEEK_NUM").is_between(
                week_range["min"], week_range["max"]
            )
        )
    return scan.collect().to_pandas()


def main() -> None:
    processed_dir = Path("datasets/processed/hcms")
    output_dir = Path("outputs/hcms")
    models_dir = output_dir / "models"
    summary_path = output_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lightgbm_artifact = joblib.load(models_dir / "lightgbm.joblib")
    features = list(lightgbm_artifact["features"])
    imputer = lightgbm_artifact["imputer"]
    transformed_names = list(imputer.get_feature_names_out(features))
    del lightgbm_artifact

    woe_artifact = joblib.load(output_dir / "scorecard/logistic_woe.joblib")
    required_features = sorted(set(features) | set(woe_artifact["bins"]))
    predictions: dict[str, dict[str, np.ndarray]] = {
        model: {} for model in [*BASE_MODELS, "logistic_woe"]
    }
    labels: dict[str, pd.Series] = {}
    test_weeks: pd.Series | None = None
    competition_ids: pd.Series | None = None

    for split_name in ["train", "valid", "test", "competition"]:
        is_competition = split_name == "competition"
        matrix_path = processed_dir / (
            "feature_matrix_test_C.parquet"
            if is_competition
            else "feature_matrix_train_C.parquet"
        )
        week_range = None if is_competition else summary["split_weeks"][split_name]
        frame = _load_split(matrix_path, required_features, week_range)
        transformed = imputer.transform(frame[features]).astype("float32")
        values = pd.DataFrame(transformed, columns=transformed_names)
        for model_name in BASE_MODELS:
            artifact = joblib.load(models_dir / f"{model_name}.joblib")
            model_values: object = values
            if model_name == "logistic_raw":
                model_values = artifact["scaler"].transform(values)
            elif "transformed_features" in artifact:
                model_values = values[artifact["transformed_features"]]
            predictions[model_name][split_name] = artifact["model"].predict_proba(
                model_values
            )[:, 1]
            del artifact, model_values
        predictions["logistic_woe"][split_name] = woe_artifact[
            "model"
        ].predict_proba(
            _woe_transform(frame, woe_artifact["bins"], woe_artifact["tables"])
        )[:, 1]
        if is_competition:
            competition_ids = frame[ID_COLUMN].astype(int).reset_index(drop=True)
        else:
            labels[split_name] = frame[TARGET].astype("int8").reset_index(drop=True)
            if split_name == "test":
                test_weeks = frame["WEEK_NUM"].astype(int).reset_index(drop=True)
        del frame, transformed, values
        gc.collect()

    predictions.update(_equal_weight_ensembles(predictions))
    metrics = [
        _metric_row(model_name, split_name, labels[split_name], split_predictions[split_name])
        for model_name, split_predictions in predictions.items()
        for split_name in ["train", "valid", "test"]
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(models_dir / "metrics.csv", index=False)
    metrics_dir = models_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(metrics_dir / "metrics.csv", index=False)

    test_predictions = {
        model: split_predictions["test"]
        for model, split_predictions in predictions.items()
    }
    write_roc_auc_curve(
        labels["test"],
        test_predictions,
        metrics_dir / "roc_auc_curve.png",
        split_label="out-of-time test split",
    )
    write_gini_curve(
        labels["test"],
        test_predictions,
        metrics_dir / "gini_curve.png",
        split_label="out-of-time test split",
    )
    write_ks_curve(
        labels["test"],
        test_predictions,
        metrics_dir / "ks_curve.png",
        split_label="out-of-time test split",
    )

    if test_weeks is None:
        raise AssertionError("Test weeks were not loaded")
    stability_parts = []
    stability_summaries = []
    excluded_parts = []
    for model_name, scores in test_predictions.items():
        result = stability_metric(labels["test"], scores, test_weeks)
        written = _write_stability(result, model_name, output_dir / "stability")
        stability_parts.append(written["by_week"])
        stability_summaries.append(written["summary"])
        if not written["excluded"].empty:
            excluded_parts.append(written["excluded"])
    gini_by_week = pd.concat(stability_parts, ignore_index=True)
    gini_by_week.to_csv(output_dir / "stability/gini_by_week.csv", index=False)
    write_gini_by_period_plot(
        gini_by_week,
        output_dir / "stability/gini_by_week.png",
        period_column="WEEK_NUM",
        group_column="model",
        title="Out-of-time test Gini by week",
    )
    excluded = (
        pd.concat(excluded_parts, ignore_index=True)
        if excluded_parts
        else pd.DataFrame(
            columns=["model", "WEEK_NUM", "n", "n_bad", "n_good", "reason"]
        )
    )
    excluded.to_csv(output_dir / "stability/excluded_weeks.csv", index=False)
    (output_dir / "stability/stability_metric.json").write_text(
        json.dumps(stability_summaries, indent=2) + "\n",
        encoding="utf-8",
    )

    if competition_ids is None:
        raise AssertionError("Competition IDs were not loaded")
    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    sample_submission = pd.read_csv(
        "datasets/raw/home-credit-model-stability/sample_submission.csv"
    )
    for model_name, split_predictions in predictions.items():
        _ordered_submission(
            competition_ids,
            split_predictions["competition"],
            sample_submission,
        ).to_csv(submission_dir / f"{model_name}.csv", index=False)
    _ordered_submission(
        competition_ids,
        predictions["lightgbm"]["competition"],
        sample_submission,
    ).to_csv(submission_dir / "submission.csv", index=False)

    summary["devices"] = {
        **summary.get("devices", {}),
        "random_forest": "cpu",
        "extra_trees": "cpu",
        "hist_gradient_boosting": "cpu",
        "catboost": "cpu",
    }
    summary["benchmark_protocol"] = (
        "All tree models use the full set of transformed features (244 columns) "
        "and out-of-time week split; ensembles use fixed equal weights without "
        "test-week tuning."
    )
    summary["ensemble_members"] = ENSEMBLE_MEMBERS
    summary["metrics"] = metrics
    summary["stability"] = stability_summaries
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        metrics_table.loc[metrics_table["split"].eq("test")]
        .sort_values("auc", ascending=False)[["model", "auc", "gini", "ks"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
