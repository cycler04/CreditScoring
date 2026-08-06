#!/usr/bin/env python3
"""Render diagrams from completed HCDR and HCMS tabular artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from credit_scoring.visualization import (
    normalize_feature_importance,
    write_bad_rate_by_period_plot,
    write_benchmark_dashboard,
    write_cutoff_plot,
    write_feature_importance_plot,
    write_gini_by_period_plot,
    write_gini_curve,
    write_ks_curve,
    write_metrics_comparison_plot,
    write_ranked_metric_benchmark_plot,
    write_roc_auc_curve,
)
from home_credit_stability.pipeline import _equal_weight_ensembles

HCDR_ID = "SK_ID_CURR"
HCMS_ID = "case_id"
TARGET = "target"


def _woe_transform(
    frame: pd.DataFrame,
    bins: dict[str, np.ndarray],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    transformed = {}
    for feature, edges in bins.items():
        labels = (
            pd.cut(frame[feature], edges, include_lowest=True)
            .astype("string")
            .fillna("MISSING")
        )
        mapping = tables[feature].set_index("bin")["woe"]
        fallback = float(mapping.get("MISSING", 0.0))
        transformed[feature] = labels.map(mapping).fillna(fallback).astype(float)
    return pd.DataFrame(transformed, index=frame.index)


def _write_metric_curves(
    target: pd.Series,
    predictions: dict[str, np.ndarray],
    metrics_dir: Path,
    *,
    split_label: str,
) -> list[Path]:
    paths = [
        metrics_dir / "roc_auc_curve.png",
        metrics_dir / "gini_curve.png",
        metrics_dir / "ks_curve.png",
    ]
    write_roc_auc_curve(
        target, predictions, paths[0], split_label=split_label
    )
    write_gini_curve(target, predictions, paths[1], split_label=split_label)
    write_ks_curve(target, predictions, paths[2], split_label=split_label)
    return paths


def _write_saved_importance(
    table: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> list[Path]:
    table = normalize_feature_importance(table).sort_values(
        "importance_pct", ascending=False
    )
    csv_path = output_dir / f"{model_name}.csv"
    png_path = output_dir / f"{model_name}.png"
    table.to_csv(csv_path, index=False)
    write_feature_importance_plot(
        table,
        png_path,
        title=f"{model_name} — top feature importance",
    )
    return [csv_path, png_path]


def _render_hcdr_metric_curves(output_dir: Path) -> list[Path]:
    membership = pd.read_csv(
        "datasets/processed/hcdr/split_membership.csv",
        usecols=[HCDR_ID, "split"],
    )
    test_ids = membership.loc[membership["split"].eq("test"), HCDR_ID]
    matrix = pd.read_parquet("datasets/processed/hcdr/feature_matrix.parquet")
    test = matrix.loc[matrix[HCDR_ID].isin(test_ids)].copy()
    predictions = {}
    for model_name in ["logistic_raw", "lightgbm", "xgboost"]:
        artifact = joblib.load(output_dir / f"models/{model_name}.joblib")
        processor = artifact["preprocessor"]
        features = list(processor.feature_names_in_)
        values = processor.transform(test[features])
        predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
    woe_artifact = joblib.load(output_dir / "scorecard/logistic_woe.joblib")
    predictions["logistic_woe"] = woe_artifact["model"].predict_proba(
        _woe_transform(test, woe_artifact["bins"], woe_artifact["tables"])
    )[:, 1]
    importance_paths = _write_saved_importance(
        pd.DataFrame(
            {
                "feature": woe_artifact["model"].feature_names_in_,
                "coefficient": woe_artifact["model"].coef_[0],
                "importance": np.abs(woe_artifact["model"].coef_[0]),
            }
        ),
        output_dir / "models/feature_importance",
        "logistic_woe",
    )
    return importance_paths + _write_metric_curves(
        test["TARGET"].astype(int),
        predictions,
        output_dir / "models/metrics",
        split_label="test split",
    )


def _render_hcms_metric_curves(output_dir: Path) -> list[Path]:
    membership = pd.read_csv(
        "datasets/processed/hcms/split_membership.csv",
        usecols=["WEEK_NUM", "split"],
    )
    test_weeks = (
        membership.loc[membership["split"].eq("test"), "WEEK_NUM"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )
    artifacts = {
        model_name: joblib.load(output_dir / f"models/{model_name}.joblib")
        for model_name in [
            "lightgbm",
            "logistic_raw",
            "xgboost",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "catboost",
        ]
    }
    woe_artifact = joblib.load(output_dir / "scorecard/logistic_woe.joblib")
    raw_features = sorted(
        {
            feature
            for artifact in artifacts.values()
            for feature in artifact["features"]
        }
        | set(woe_artifact["bins"])
    )
    columns = [HCMS_ID, "WEEK_NUM", TARGET, *raw_features]
    test = (
        pl.scan_parquet("datasets/processed/hcms/feature_matrix.parquet")
        .filter(pl.col("WEEK_NUM").is_in(test_weeks))
        .select(columns)
        .collect()
        .to_pandas()
    )
    predictions = {}
    for model_name, artifact in artifacts.items():
        features = artifact["features"]
        transformed = artifact["imputer"].transform(test[features]).astype("float32")
        names = artifact["imputer"].get_feature_names_out(features)
        values = pd.DataFrame(transformed, columns=names, index=test.index)
        if model_name == "logistic_raw":
            values = artifact["scaler"].transform(values)
        elif "transformed_features" in artifact:
            values = values[artifact["transformed_features"]]
        predictions[model_name] = artifact["model"].predict_proba(values)[:, 1]
    ensemble_input = {
        name: {"test": scores} for name, scores in predictions.items()
    }
    predictions.update(
        {
            name: scores["test"]
            for name, scores in _equal_weight_ensembles(ensemble_input).items()
        }
    )
    predictions["logistic_woe"] = woe_artifact["model"].predict_proba(
        _woe_transform(test, woe_artifact["bins"], woe_artifact["tables"])
    )[:, 1]
    importance_dir = output_dir / "models/feature_importance"
    logistic_artifact = artifacts["logistic_raw"]
    logistic_names = logistic_artifact["imputer"].get_feature_names_out(
        logistic_artifact["features"]
    )
    importance_paths = _write_saved_importance(
        pd.DataFrame(
            {
                "feature": logistic_names,
                "coefficient": logistic_artifact["model"].coef_[0],
                "importance": np.abs(logistic_artifact["model"].coef_[0]),
            }
        ),
        importance_dir,
        "logistic_raw",
    )
    xgboost_artifact = artifacts["xgboost"]
    importance_paths.extend(
        _write_saved_importance(
            pd.DataFrame(
                {
                    "feature": xgboost_artifact["transformed_features"],
                    "importance": xgboost_artifact["model"].feature_importances_,
                }
            ),
            importance_dir,
            "xgboost",
        )
    )
    importance_paths.extend(
        _write_saved_importance(
            pd.DataFrame(
                {
                    "feature": woe_artifact["model"].feature_names_in_,
                    "coefficient": woe_artifact["model"].coef_[0],
                    "importance": np.abs(woe_artifact["model"].coef_[0]),
                }
            ),
            importance_dir,
            "logistic_woe",
        )
    )
    return importance_paths + _write_metric_curves(
        test[TARGET].astype(int),
        predictions,
        output_dir / "models/metrics",
        split_label="out-of-time test split",
    )


def _render_importance_tables(output_dir: Path) -> list[Path]:
    rendered = []
    for csv_path in sorted((output_dir / "models/feature_importance").glob("*.csv")):
        table = pd.read_csv(csv_path)
        value_column = "importance" if "importance" in table else "importance_value"
        if not {"feature", value_column}.issubset(table.columns):
            continue
        table = normalize_feature_importance(table, value_column=value_column)
        table.to_csv(csv_path, index=False)
        png_path = csv_path.with_suffix(".png")
        write_feature_importance_plot(
            table,
            png_path,
            title=f"{csv_path.stem} — top feature importance",
            value_column=value_column,
        )
        rendered.append(png_path)
    return rendered


def render_hcdr(output_dir: Path, *, with_predictions: bool = False) -> list[Path]:
    """Render HCDR diagrams that can be reconstructed without model fitting."""
    rendered = _render_importance_tables(output_dir)
    metrics = pd.read_csv(output_dir / "models/metrics.csv")
    interpretable_metrics = output_dir / "models/interpretable_metrics.csv"
    if interpretable_metrics.exists():
        metrics = pd.concat(
            [metrics, pd.read_csv(interpretable_metrics)], ignore_index=True
        )
    metrics_dir = output_dir / "models/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "metrics.csv", index=False)
    metrics_plot = metrics_dir / "metrics_comparison.png"
    write_metrics_comparison_plot(
        metrics,
        metrics_plot,
        title="Home Credit Default Risk model metrics",
    )
    rendered.append(metrics_plot)
    for metric, label in [("auc", "ROC AUC"), ("gini", "Gini"), ("ks", "KS")]:
        benchmark_plot = metrics_dir / f"{metric}_benchmark.png"
        write_ranked_metric_benchmark_plot(
            metrics,
            benchmark_plot,
            metric=metric,
            title=f"Home Credit Default Risk — {label} benchmark",
        )
        rendered.append(benchmark_plot)
    dashboard_plot = metrics_dir / "benchmark_dashboard.png"
    write_benchmark_dashboard(
        metrics,
        dashboard_plot,
        title="Home Credit Default Risk — benchmark dashboard",
    )
    rendered.append(dashboard_plot)

    cutoffs = pd.read_csv(output_dir / "scorecard/cutoffs.csv")
    cutoff_plot = output_dir / "scorecard/approval_bad_rate.png"
    write_cutoff_plot(
        cutoffs,
        cutoff_plot,
        target_column="approval_rate_target",
        actual_column="approval_rate_actual",
        bad_rate_column="approved_bad_rate",
    )
    rendered.append(cutoff_plot)
    if with_predictions:
        rendered.extend(_render_hcdr_metric_curves(output_dir))
    return rendered


def render_hcms(output_dir: Path, *, with_predictions: bool = False) -> list[Path]:
    """Render HCMS metric, EDA, importance, and stability diagrams."""
    rendered = _render_importance_tables(output_dir)
    metrics = pd.read_csv(output_dir / "models/metrics.csv")
    interpretable_metrics = output_dir / "models/interpretable_metrics.csv"
    if interpretable_metrics.exists():
        metrics = pd.concat(
            [metrics, pd.read_csv(interpretable_metrics)], ignore_index=True
        )
    metrics_dir = output_dir / "models/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "metrics.csv", index=False)
    metrics_plot = metrics_dir / "metrics_comparison.png"
    write_metrics_comparison_plot(
        metrics,
        metrics_plot,
        title="Home Credit Model Stability model metrics",
    )
    rendered.append(metrics_plot)
    for metric, label in [("auc", "ROC AUC"), ("gini", "Gini"), ("ks", "KS")]:
        benchmark_plot = metrics_dir / f"{metric}_benchmark.png"
        write_ranked_metric_benchmark_plot(
            metrics,
            benchmark_plot,
            metric=metric,
            title=f"Home Credit Model Stability — {label} benchmark",
        )
        rendered.append(benchmark_plot)

    dashboard_plot = metrics_dir / "benchmark_dashboard.png"
    write_benchmark_dashboard(
        metrics,
        dashboard_plot,
        title="Home Credit Model Stability — benchmark dashboard",
    )
    rendered.append(dashboard_plot)

    bad_rate_week = pd.read_csv(output_dir / "eda/bad_rate_by_week.csv")
    bad_rate_plot = output_dir / "eda/bad_rate_by_week.png"
    write_bad_rate_by_period_plot(
        bad_rate_week,
        bad_rate_plot,
        period_column="WEEK_NUM",
        group_column="split",
        title="Home Credit Model Stability — bad rate by week",
    )
    rendered.append(bad_rate_plot)

    for csv_name, png_name, group_column, title in [
        (
            "gini_by_week.csv",
            "gini_by_week.png",
            "model",
            "Out-of-time test Gini by week",
        ),
        (
            "stage_gini_by_week.csv",
            "stage_gini_by_week.png",
            "level",
            "Stage A/B/C LightGBM Gini by week",
        ),
    ]:
        table = pd.read_csv(output_dir / "stability" / csv_name)
        png_path = output_dir / "stability" / png_name
        write_gini_by_period_plot(
            table,
            png_path,
            period_column="WEEK_NUM",
            group_column=group_column,
            title=title,
        )
        rendered.append(png_path)

    cutoffs = pd.read_csv(output_dir / "scorecard/cutoffs.csv")
    cutoff_plot = output_dir / "scorecard/approval_bad_rate.png"
    write_cutoff_plot(
        cutoffs,
        cutoff_plot,
        target_column="approval_target",
        actual_column="approval_rate",
        bad_rate_column="approved_bad_rate",
    )
    rendered.append(cutoff_plot)
    if with_predictions:
        rendered.extend(_render_hcms_metric_curves(output_dir))
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["hcdr", "hcms", "all"],
        default="all",
    )
    parser.add_argument(
        "--with-predictions",
        action="store_true",
        help="Recreate ROC/Gini/KS from processed holdouts and saved models.",
    )
    args = parser.parse_args()
    rendered = []
    if args.dataset in {"hcdr", "all"}:
        rendered.extend(
            render_hcdr(
                Path("outputs/hcdr"),
                with_predictions=args.with_predictions,
            )
        )
    if args.dataset in {"hcms", "all"}:
        rendered.extend(
            render_hcms(
                Path("outputs/hcms"),
                with_predictions=args.with_predictions,
            )
        )
    print(f"Rendered {len(rendered)} artifact files")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
