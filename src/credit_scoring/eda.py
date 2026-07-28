"""EDA tables and plots for the GiveMeSomeCredit training split."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def bad_rate_by_decile(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
) -> pd.DataFrame:
    """Calculate bad rate by quantile bin for numeric features."""
    rows: list[pd.DataFrame] = []
    for feature in feature_columns:
        non_null = frame[[feature, target]].dropna()
        if non_null[feature].nunique() < 2:
            continue
        decile = pd.qcut(non_null[feature], q=10, duplicates="drop")
        summary = (
            non_null.assign(decile=decile)
            .groupby("decile", observed=True)
            .agg(
                n=(target, "size"),
                bad_rate=(target, "mean"),
                value_min=(feature, "min"),
                value_max=(feature, "max"),
            )
            .reset_index()
        )
        summary.insert(0, "feature", feature)
        summary["decile"] = summary["decile"].astype(str)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def write_eda_outputs(
    frame: pd.DataFrame,
    original_features: list[str],
    target: str,
    anomaly_findings: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write required EDA tables and compact overview plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    target_distribution = (
        frame[target]
        .value_counts(dropna=False)
        .rename_axis("target")
        .reset_index(name="count")
    )
    target_distribution["rate"] = target_distribution["count"] / len(frame)
    target_distribution.to_csv(output_dir / "target_distribution.csv", index=False)

    missing = (
        frame[original_features]
        .isna()
        .agg(["sum", "mean"])
        .T.rename(columns={"sum": "missing_count", "mean": "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    missing.to_csv(output_dir / "missing_summary.csv")

    frame[original_features].describe(percentiles=[0.01, 0.5, 0.95, 0.99]).T.to_csv(
        output_dir / "numeric_summary.csv"
    )
    anomaly_findings.to_csv(output_dir / "anomaly_findings.csv", index=False)
    deciles = bad_rate_by_decile(frame, original_features, target)
    deciles.to_csv(output_dir / "bad_rate_by_decile.csv", index=False)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(
        data=target_distribution,
        x="target",
        y="rate",
        hue="target",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Target distribution (training split)")
    axes[0].set_ylabel("Rate")
    missing.reset_index(names="feature").pipe(
        lambda table: sns.barplot(
            data=table,
            y="feature",
            x="missing_rate",
            color="#4C72B0",
            ax=axes[1],
        )
    )
    axes[1].set_title("Missing rate")
    figure.tight_layout()
    figure.savefig(output_dir / "overview.png", dpi=160)
    plt.close(figure)

    features = deciles["feature"].drop_duplicates().tolist()
    figure, axes = plt.subplots(5, 2, figsize=(14, 20), squeeze=False)
    for axis, feature in zip(axes.flat, features, strict=False):
        selected = deciles.loc[deciles["feature"].eq(feature)].reset_index(drop=True)
        sns.lineplot(
            data=selected,
            x=selected.index + 1,
            y="bad_rate",
            marker="o",
            ax=axis,
        )
        axis.set_title(feature)
        axis.set_xlabel("Quantile bin (low to high)")
        axis.set_ylabel("Bad rate")
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "bad_rate_deciles.png", dpi=160)
    plt.close(figure)
