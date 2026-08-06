"""Reusable model, EDA, and stability diagrams for credit-risk pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve


def normalize_feature_importance(
    table: pd.DataFrame,
    *,
    value_column: str = "importance",
) -> pd.DataFrame:
    """Add within-model importance shares while preserving native values.

    ``importance_pct`` is stored as a fraction in ``[0, 1]`` so that its sum is
    one; plotting functions render it on a 0--100 percent scale.
    """
    required = {"feature", value_column}
    if not required.issubset(table.columns):
        raise ValueError(f"Importance table requires columns {sorted(required)}")
    result = table.copy()
    values = pd.to_numeric(result[value_column], errors="coerce").abs()
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Feature importance values must be finite numbers")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Feature importance values must have a positive total")
    result["importance_pct"] = values / total
    return result


def _save_figure(figure: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)


def cumulative_class_rates(
    y_true: pd.Series,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return population, bad, and good cumulative rates ordered by risk."""
    target = y_true.to_numpy()
    order = np.argsort(-scores, kind="stable")
    ordered_target = target[order]
    ordered_scores = scores[order]
    bad_count = int(ordered_target.sum())
    good_count = len(ordered_target) - bad_count
    if bad_count == 0 or good_count == 0:
        raise ValueError("Metric diagrams require both target classes")

    threshold_ends = np.flatnonzero(
        np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    )
    population_rate = (threshold_ends + 1) / len(target)
    cumulative_bad_rate = np.cumsum(ordered_target)[threshold_ends] / bad_count
    cumulative_good_rate = (
        np.cumsum(1 - ordered_target)[threshold_ends] / good_count
    )
    return (
        np.insert(population_rate, 0, 0.0),
        np.insert(cumulative_bad_rate, 0, 0.0),
        np.insert(cumulative_good_rate, 0, 0.0),
    )


def write_roc_auc_curve(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    *,
    split_label: str = "test split",
) -> None:
    """Write a combined ROC curve for several fitted models."""
    figure = Figure(figsize=(8, 6), layout="constrained")
    axes = figure.subplots()
    for model_name, scores in predictions.items():
        false_positive, true_positive, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        axes.plot(
            false_positive,
            true_positive,
            linewidth=2,
            label=f"{model_name} (AUC={auc:.3f})",
        )
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    axes.set(
        title=f"ROC curves — {split_label}",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right")
    _save_figure(figure, output_path)


def write_gini_curve(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    *,
    split_label: str = "test split",
) -> None:
    """Write cumulative-gains curves with Gini values."""
    figure = Figure(figsize=(8, 6), layout="constrained")
    axes = figure.subplots()
    for model_name, scores in predictions.items():
        population_rate, cumulative_bad_rate, _ = cumulative_class_rates(
            y_true, scores
        )
        gini = 2 * roc_auc_score(y_true, scores) - 1
        axes.plot(
            population_rate,
            cumulative_bad_rate,
            linewidth=2,
            label=f"{model_name} (Gini={gini:.3f})",
        )
    bad_rate = float(y_true.mean())
    axes.plot(
        [0, bad_rate, 1],
        [0, 1, 1],
        linestyle=":",
        color="black",
        label="Perfect",
    )
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    axes.set(
        title=f"Cumulative gains (Gini) — {split_label}",
        xlabel="Cumulative population share",
        ylabel="Cumulative bad share",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right")
    _save_figure(figure, output_path)


def write_ks_curve(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    *,
    split_label: str = "test split",
) -> None:
    """Write cumulative-distribution separation curves for KS."""
    figure = Figure(figsize=(8, 6), layout="constrained")
    axes = figure.subplots()
    for model_name, scores in predictions.items():
        population_rate, cumulative_bad_rate, cumulative_good_rate = (
            cumulative_class_rates(y_true, scores)
        )
        separation = cumulative_bad_rate - cumulative_good_rate
        ks = float(np.max(separation))
        axes.plot(
            population_rate,
            separation,
            linewidth=2,
            label=f"{model_name} (KS={ks:.3f})",
        )
    axes.axhline(0, linestyle="--", color="grey", linewidth=1)
    axes.set(
        title=f"KS separation curves — {split_label}",
        xlabel="Cumulative population share",
        ylabel="Cumulative bad share − cumulative good share",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes.grid(alpha=0.25)
    axes.legend(loc="upper right")
    _save_figure(figure, output_path)


def write_feature_importance_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    value_column: str = "importance",
    top_n: int = 20,
) -> None:
    """Write a normalized horizontal top-feature plot on a percentage scale."""
    normalized = normalize_feature_importance(table, value_column=value_column)
    subset = (
        normalized[["feature", "importance_pct"]]
        .dropna()
        .nlargest(top_n, "importance_pct")
        .sort_values("importance_pct")
    )
    if subset.empty:
        raise ValueError("Feature importance table is empty")
    figure = Figure(
        figsize=(11, max(4.5, 0.42 * len(subset))),
        layout="constrained",
    )
    axes = figure.subplots()
    axes.barh(
        subset["feature"],
        subset["importance_pct"] * 100,
        color="#4c72b0",
    )
    axes.set(title=title, xlabel="Importance within model (%)", ylabel="")
    axes.grid(axis="x", alpha=0.25)
    _save_figure(figure, output_path)


def write_bad_rate_by_period_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    period_column: str,
    title: str,
    group_column: str | None = None,
) -> None:
    """Write a bad-rate time/period diagram, optionally grouped by model or split."""
    required = {period_column, "bad_rate"}
    if not required.issubset(table.columns):
        raise ValueError(f"Bad-rate table requires columns {sorted(required)}")
    figure = Figure(figsize=(11, 5.5), layout="constrained")
    axes = figure.subplots()
    groups = (
        table.groupby(group_column, sort=False)
        if group_column and group_column in table
        else [(None, table)]
    )
    for group, part in groups:
        ordered = part.sort_values(period_column)
        axes.plot(
            ordered[period_column],
            ordered["bad_rate"] * 100,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=None if group is None else str(group),
        )
    axes.set(title=title, xlabel=period_column, ylabel="Bad rate (%)")
    axes.grid(alpha=0.25)
    if group_column and group_column in table:
        axes.legend(title=group_column)
    _save_figure(figure, output_path)


def write_eda_overview(
    frame: pd.DataFrame,
    target_column: str,
    output_path: Path,
    *,
    title: str,
    top_n: int = 15,
) -> None:
    """Write target distribution and the highest feature missing rates."""
    if target_column not in frame:
        raise ValueError(f"EDA frame is missing target column {target_column!r}")
    figure = Figure(figsize=(13, 5.5), layout="constrained")
    target_axes, missing_axes = figure.subplots(1, 2)
    distribution = frame[target_column].value_counts().sort_index()
    target_axes.bar(
        distribution.index.astype(str),
        distribution.to_numpy(),
        color=["#4c72b0", "#c44e52"][: len(distribution)],
    )
    target_axes.set(
        title="Target distribution",
        xlabel=target_column,
        ylabel="Rows",
    )
    missing = (
        frame.drop(columns=[target_column])
        .isna()
        .mean()
        .mul(100)
        .nlargest(top_n)
        .sort_values()
    )
    missing_axes.barh(missing.index, missing.to_numpy(), color="#55a868")
    missing_axes.set(
        title=f"Top {top_n} missing rates",
        xlabel="Missing (%)",
        ylabel="",
    )
    target_axes.grid(axis="y", alpha=0.25)
    missing_axes.grid(axis="x", alpha=0.25)
    figure.suptitle(title, fontsize=14)
    _save_figure(figure, output_path)


def write_gini_by_period_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    period_column: str,
    group_column: str,
    title: str,
) -> None:
    """Write per-period Gini traces for stability inspection."""
    required = {period_column, group_column, "gini"}
    if not required.issubset(table.columns):
        raise ValueError(f"Gini table requires columns {sorted(required)}")
    figure = Figure(figsize=(11, 5.5), layout="constrained")
    axes = figure.subplots()
    for group, part in table.groupby(group_column, sort=False):
        ordered = part.sort_values(period_column)
        axes.plot(
            ordered[period_column],
            ordered["gini"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=str(group),
        )
    axes.axhline(0, color="grey", linestyle="--", linewidth=1)
    axes.set(title=title, xlabel=period_column, ylabel="Gini")
    axes.grid(alpha=0.25)
    axes.legend(title=group_column)
    _save_figure(figure, output_path)


def write_cutoff_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    target_column: str,
    actual_column: str,
    bad_rate_column: str,
) -> None:
    """Compare approval targets, actual approval, and approved bad rate."""
    required = {target_column, actual_column, bad_rate_column}
    if not required.issubset(table.columns):
        raise ValueError(f"Cutoff table requires columns {sorted(required)}")
    ordered = table.sort_values(target_column)
    figure = Figure(figsize=(8, 5.5), layout="constrained")
    axes = figure.subplots()
    axes.plot(
        ordered[target_column] * 100,
        ordered[actual_column] * 100,
        marker="o",
        linewidth=2,
        label="Actual approval rate",
    )
    axes.plot(
        ordered[target_column] * 100,
        ordered[bad_rate_column] * 100,
        marker="o",
        linewidth=2,
        label="Approved bad rate",
    )
    axes.set(
        title="Approval policy outcomes — test split",
        xlabel="Target approval rate (%)",
        ylabel="Rate (%)",
    )
    axes.grid(alpha=0.25)
    axes.legend()
    _save_figure(figure, output_path)


def write_metrics_comparison_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    split: str = "test",
    title: str = "Model metrics comparison",
) -> None:
    """Write grouped AUC, Gini, and KS bars from a persisted metrics table."""
    required = {"model", "split", "auc", "gini", "ks"}
    if not required.issubset(table.columns):
        raise ValueError(f"Metrics table requires columns {sorted(required)}")
    selected = table.loc[table["split"].eq(split)].copy()
    if "level" in selected and selected["level"].nunique() > 1:
        selected = selected.loc[selected["level"].eq("C")]
    selected = selected.drop_duplicates("model", keep="last").sort_values("model")
    if selected.empty:
        raise ValueError(f"Metrics table has no rows for split {split!r}")
    positions = np.arange(len(selected))
    width = 0.24
    figure = Figure(figsize=(10, 5.5), layout="constrained")
    axes = figure.subplots()
    for offset, metric, color in [
        (-width, "auc", "#4c72b0"),
        (0.0, "gini", "#55a868"),
        (width, "ks", "#c44e52"),
    ]:
        axes.bar(
            positions + offset,
            selected[metric],
            width=width,
            label=metric.upper(),
            color=color,
        )
    axes.set_xticks(positions, selected["model"], rotation=20, ha="right")
    axes.set(title=f"{title} — {split} split", ylabel="Metric value", ylim=(0, 1))
    axes.grid(axis="y", alpha=0.25)
    axes.legend()
    _save_figure(figure, output_path)


def _benchmark_model_labels(model_names: pd.Index) -> list[str]:
    labels = [
        name.removeprefix("ensemble_").replace("_", " ").title()
        for name in model_names
    ]
    return [
        label.replace("Lightgbm", "LightGBM")
        .replace("Xgboost", "XGBoost")
        .replace("Catboost", "CatBoost")
        .replace("Ft Transformer", "FT-Transformer")
        .replace("Woe", "WoE")
        for label in labels
    ]


def write_ranked_metric_benchmark_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    metric: str,
    title: str | None = None,
) -> None:
    """Write a ranked test-metric chart with validation markers."""
    metric_labels = {"auc": "ROC AUC", "gini": "Gini", "ks": "KS"}
    if metric not in metric_labels:
        raise ValueError(f"Unsupported benchmark metric: {metric!r}")
    required = {"model", "split", metric}
    if not required.issubset(table.columns):
        raise ValueError(f"Metrics table requires columns {sorted(required)}")
    selected = table.loc[table["split"].isin(["valid", "test"])].copy()
    if "level" in selected and selected["level"].nunique() > 1:
        selected = selected.loc[selected["level"].eq("C")]
    pivot = selected.drop_duplicates(["model", "split"], keep="last").pivot(
        index="model", columns="split", values=metric
    )
    if not {"valid", "test"}.issubset(pivot.columns):
        raise ValueError(f"{metric_labels[metric]} benchmark requires valid and test")
    pivot = pivot.sort_values("test")
    labels = _benchmark_model_labels(pivot.index)
    colors = [
        "#4c72b0" if name.startswith("ensemble_") else "#9aa0a6"
        for name in pivot.index
    ]
    colors[-1] = "#dd8452"
    positions = np.arange(len(pivot))
    figure = Figure(
        figsize=(11, max(6.0, 0.48 * len(pivot) + 1.8)),
        layout="constrained",
    )
    axes = figure.subplots()
    axes.barh(positions, pivot["test"], color=colors, height=0.65)
    axes.scatter(
        pivot["valid"],
        positions,
        marker="D",
        s=34,
        color="#222222",
        zorder=3,
    )
    minimum = float(min(pivot["valid"].min(), pivot["test"].min()))
    maximum = float(max(pivot["valid"].max(), pivot["test"].max()))
    span = max(maximum - minimum, 0.01)
    label_offset = span * 0.008
    for position, value in zip(positions, pivot["test"], strict=True):
        axes.text(
            value + label_offset,
            position,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )
    metric_floor = 0.5 if metric == "auc" else 0.0
    lower = max(metric_floor, minimum - span * 0.12)
    upper = min(1.0, maximum + span * 0.14)
    axes.set_xlim(lower, upper)
    axes.set_yticks(positions, labels)
    axes.set(
        title=title or f"Model benchmark — {metric_labels[metric]}",
        xlabel=f"{metric_labels[metric]} (zoomed scale)",
        ylabel="",
    )
    axes.grid(axis="x", alpha=0.25)
    axes.legend(
        handles=[
            Patch(
                color="#dd8452",
                label=f"Best test {metric_labels[metric]}",
            ),
            Patch(color="#4c72b0", label="Ensemble"),
            Patch(color="#9aa0a6", label="Base model"),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#222222",
                markeredgecolor="#222222",
                label=f"Validation {metric_labels[metric]}",
            ),
        ],
        loc="lower right",
    )
    _save_figure(figure, output_path)


def write_auc_benchmark_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Model benchmark — ROC AUC",
) -> None:
    """Write a ranked ROC-AUC benchmark chart."""
    write_ranked_metric_benchmark_plot(
        table,
        output_path,
        metric="auc",
        title=title,
    )


def write_benchmark_dashboard(
    table: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Model benchmark dashboard — test split",
) -> None:
    """Write aligned AUC, Gini, and KS panels using one model order."""
    metrics = ("auc", "gini", "ks")
    required = {"model", "split", *metrics}
    if not required.issubset(table.columns):
        raise ValueError(f"Metrics table requires columns {sorted(required)}")
    selected = table.loc[table["split"].eq("test")].copy()
    if "level" in selected and selected["level"].nunique() > 1:
        selected = selected.loc[selected["level"].eq("C")]
    selected = selected.drop_duplicates("model", keep="last").sort_values("auc")
    if selected.empty:
        raise ValueError("Benchmark dashboard requires test rows")

    labels = _benchmark_model_labels(pd.Index(selected["model"]))
    positions = np.arange(len(selected))
    figure = Figure(
        figsize=(17, max(6.0, 0.48 * len(selected) + 1.8)),
        layout="constrained",
    )
    axes = figure.subplots(1, 3, sharey=True)
    for axis, metric, metric_label in zip(
        axes,
        metrics,
        ("ROC AUC", "Gini", "KS"),
        strict=True,
    ):
        values = selected[metric].to_numpy()
        colors = [
            "#4c72b0" if name.startswith("ensemble_") else "#9aa0a6"
            for name in selected["model"]
        ]
        colors[int(np.argmax(values))] = "#dd8452"
        axis.barh(positions, values, color=colors, height=0.65)
        minimum = float(values.min())
        maximum = float(values.max())
        span = max(maximum - minimum, 0.01)
        floor = 0.5 if metric == "auc" else 0.0
        axis.set_xlim(
            max(floor, minimum - span * 0.12),
            min(1.0, maximum + span * 0.18),
        )
        for position, value in zip(positions, values, strict=True):
            axis.text(
                value + span * 0.01,
                position,
                f"{value:.4f}",
                va="center",
                fontsize=8,
            )
        axis.set(title=metric_label, xlabel="Zoomed scale")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(positions, labels)
    figure.suptitle(title, fontsize=16)
    figure.legend(
        handles=[
            Patch(color="#dd8452", label="Best in panel"),
            Patch(color="#4c72b0", label="Ensemble"),
            Patch(color="#9aa0a6", label="Base model"),
        ],
        loc="outside lower center",
        ncols=3,
    )
    _save_figure(figure, output_path)
