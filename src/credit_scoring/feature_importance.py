"""Inspect feature importance from persisted model artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.pipeline import Pipeline


TABLE_COLUMNS = [
    "model",
    "rank",
    "feature",
    "importance_method",
    "native_value",
    "importance_value",
    "importance_pct",
    "direction",
    "source_artifact",
]


def _pipeline_feature_names(model: Pipeline) -> list[str]:
    """Return feature names after all pipeline transformations."""
    names = np.asarray(model.feature_names_in_, dtype=object)
    for _, transformer in model.steps[:-1]:
        if not hasattr(transformer, "get_feature_names_out"):
            raise TypeError(
                f"{type(transformer).__name__} does not expose feature names"
            )
        names = transformer.get_feature_names_out(names)
    return [str(name) for name in names]


def _importance_rows(model: Any, model_name: str) -> pd.DataFrame:
    """Extract native importance values without mixing model-family semantics."""
    if hasattr(model, "booster_"):
        features = [str(name) for name in model.booster_.feature_name()]
        values = np.asarray(
            model.booster_.feature_importance(importance_type="gain"),
            dtype=float,
        )
        native_values = values
        method = "gain"
        directions = np.full(len(values), "not_applicable", dtype=object)
    else:
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
        if not hasattr(estimator, "coef_"):
            raise TypeError(
                f"Unsupported model artifact {model_name!r}: expected a fitted "
                "LightGBM model or an estimator with coef_"
            )
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim != 2 or coefficients.shape[0] != 1:
            raise TypeError(
                f"Unsupported coefficient shape for {model_name!r}: "
                f"{coefficients.shape}"
            )
        if not np.array_equal(estimator.classes_, np.array([0, 1])):
            raise TypeError(
                f"Expected binary classes [0, 1] for {model_name!r}; "
                f"received {estimator.classes_.tolist()}"
            )
        signed_values = coefficients[0]
        features = (
            _pipeline_feature_names(model)
            if isinstance(model, Pipeline)
            else [str(name) for name in model.feature_names_in_]
        )
        values = np.abs(signed_values)
        native_values = signed_values
        is_woe_model = model_name == "logistic_woe"
        method = (
            "absolute_woe_coefficient"
            if is_woe_model
            else "absolute_coefficient"
        )
        directions = np.where(
            signed_values > 0,
            "higher_woe_higher_bad_risk" if is_woe_model else "higher_bad_risk",
            np.where(
                signed_values < 0,
                "higher_woe_lower_bad_risk" if is_woe_model else "lower_bad_risk",
                "neutral",
            ),
        )

    if len(features) != len(values):
        raise ValueError(
            f"Feature/value mismatch for {model_name!r}: "
            f"{len(features)} names and {len(values)} values"
        )

    total = float(values.sum())
    percentages = values / total if total > 0 else np.zeros_like(values)
    result = pd.DataFrame(
        {
            "model": model_name,
            "feature": features,
            "importance_method": method,
            "native_value": native_values,
            "importance_value": values,
            "importance_pct": percentages,
            "direction": directions,
        }
    ).sort_values(
        ["importance_value", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )
    result.insert(1, "rank", np.arange(1, len(result) + 1))
    return result


def build_feature_importance_table(
    artifact_paths: Sequence[Path],
) -> pd.DataFrame:
    """Load supported artifacts and return a ranked, model-aware table."""
    if not artifact_paths:
        raise FileNotFoundError("No .joblib model artifacts were provided")

    tables = []
    for artifact_path in sorted(artifact_paths):
        model = joblib.load(artifact_path)
        table = _importance_rows(model, artifact_path.stem)
        table["source_artifact"] = (
            Path(artifact_path.parent.name) / artifact_path.name
        ).as_posix()
        tables.append(table)
    return pd.concat(tables, ignore_index=True)[TABLE_COLUMNS]


def write_feature_importance_plot(
    table: pd.DataFrame,
    output_path: Path,
    top_n: int | None = None,
) -> None:
    """Draw normalized feature importance for one model."""
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1")
    models = list(table["model"].drop_duplicates())
    if not models:
        raise ValueError("Feature importance table is empty")
    if len(models) != 1:
        raise ValueError("Feature importance plot requires exactly one model")

    model_name = models[0]
    subset = (
        table if top_n is None else table.nsmallest(top_n, "rank")
    ).sort_values("importance_pct")
    figure = Figure(
        figsize=(11, max(4.5, 0.5 * len(subset))),
        layout="constrained",
    )
    axes = figure.subplots()
    colors = subset["direction"].map(
        {
            "higher_bad_risk": "#c44e52",
            "lower_bad_risk": "#4c72b0",
            "higher_woe_higher_bad_risk": "#c44e52",
            "higher_woe_lower_bad_risk": "#4c72b0",
            "neutral": "#999999",
            "not_applicable": "#55a868",
        }
    )
    axes.barh(
        subset["feature"],
        subset["importance_pct"] * 100,
        color=colors,
    )
    method = str(subset["importance_method"].iloc[0])
    axes.set(
        title=f"{model_name} — {method.replace('_', ' ')}",
        xlabel="Importance within model (%)",
        ylabel="",
    )
    axes.grid(axis="x", alpha=0.25)
    if method in {"absolute_coefficient", "absolute_woe_coefficient"}:
        value_label = "WoE" if method == "absolute_woe_coefficient" else "feature value"
        axes.legend(
            handles=[
                Patch(color="#c44e52", label=f"Higher {value_label} → higher bad risk"),
                Patch(color="#4c72b0", label=f"Higher {value_label} → lower bad risk"),
            ],
            loc="lower right",
        )

    title = (
        "All feature importance"
        if top_n is None
        else f"Top {top_n} feature importance"
    )
    figure.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)


def inspect_model_outputs(
    models_dir: Path,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, Path, list[Path]]:
    """Create the feature-importance table and one PNG per model."""
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1")
    artifact_dirs = [models_dir, models_dir.parent / "scorecard"]
    artifact_paths = sorted(
        artifact_path
        for artifact_dir in artifact_dirs
        for artifact_path in artifact_dir.glob("*.joblib")
    )
    table = build_feature_importance_table(artifact_paths)
    output_dir = models_dir / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "feature_importance_table.csv"
    table.to_csv(table_path, index=False)
    plot_paths = []
    for model_name, model_table in table.groupby("model", sort=False):
        plot_path = output_dir / f"{model_name}.png"
        write_feature_importance_plot(model_table, plot_path, top_n=top_n)
        plot_paths.append(plot_path)
    return table, table_path, plot_paths
