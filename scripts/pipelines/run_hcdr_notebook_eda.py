#!/usr/bin/env python3
"""Reproduce the vendor HCDR EDA notebook as durable CSV and PNG artifacts.

The source notebook was written for an old Kaggle runtime and renders most
results only inline.  This runner keeps the notebook's analysis scope while
using the repository's current dataset and output conventions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "datasets/raw/home-credit-default-risk"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/hcdr/eda"
SOURCE_NOTEBOOK = REPOSITORY_ROOT / (
    "notebooks/top-voted/home-credit-default-risk/"
    "02-complete-eda-feature-importance/"
    "home-credit-complete-eda-feature-importance.ipynb"
)

NOTEBOOK_TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "bureau": "bureau.csv",
}

APPLICATION_CATEGORIES = [
    "NAME_TYPE_SUITE",
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "ORGANIZATION_TYPE",
]

APPLICATION_TARGET_CATEGORIES = [
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_TYPE_SUITE",
]

PREVIOUS_APPLICATION_CATEGORIES = [
    "NAME_CONTRACT_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "NAME_CASH_LOAN_PURPOSE",
    "NAME_CONTRACT_STATUS",
    "NAME_PAYMENT_TYPE",
    "CODE_REJECT_REASON",
    "NAME_TYPE_SUITE",
    "NAME_CLIENT_TYPE",
    "NAME_GOODS_CATEGORY",
    "NAME_PORTFOLIO",
    "NAME_PRODUCT_TYPE",
    "CHANNEL_TYPE",
    "NAME_SELLER_INDUSTRY",
    "NAME_YIELD_GROUP",
    "PRODUCT_COMBINATION",
    "NFLAG_INSURED_ON_APPROVAL",
]

AMOUNT_COLUMNS = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only the first N rows from each CSV for a smoke test.",
    )
    parser.add_argument(
        "--skip-feature-importance",
        action="store_true",
        help="Skip the notebook's Random Forest section.",
    )
    parser.add_argument("--random-forest-jobs", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def save_frame(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def read_csv(path: Path, sample_rows: int | None) -> pd.DataFrame:
    print(f"Reading {path.name} ...", flush=True)
    return pd.read_csv(path, nrows=sample_rows)


def missing_profile(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    profile = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": frame.dtypes.astype(str).to_numpy(),
            "missing_count": missing.to_numpy(),
            "missing_percent": (missing.to_numpy() / len(frame) * 100.0)
            if len(frame)
            else np.nan,
            "non_null_count": frame.notna().sum().to_numpy(),
        }
    )
    return profile.sort_values(
        ["missing_percent", "column"], ascending=[False, True]
    ).reset_index(drop=True)


def categorical_distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    values = frame[column].astype("string").fillna("<MISSING>")
    counts = values.value_counts(dropna=False)
    return pd.DataFrame(
        {
            "category": counts.index.astype(str),
            "count": counts.to_numpy(),
            "percent": counts.to_numpy() / len(frame) * 100.0,
        }
    )


def plot_distribution_table(
    distribution: pd.DataFrame, title: str, path: Path, *, max_categories: int = 30
) -> None:
    shown = distribution.head(max_categories).iloc[::-1]
    height = max(4.5, min(12.0, 0.32 * len(shown) + 1.8))
    plt.figure(figsize=(10, height))
    plt.barh(shown["category"], shown["percent"], color="#4c78a8")
    plt.xlabel("Percent of rows")
    plt.title(title)
    plt.grid(axis="x", alpha=0.2)
    save_figure(path)


def target_rate_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    values = frame[column].astype("string").fillna("<MISSING>")
    grouped = (
        pd.DataFrame({"category": values, "TARGET": frame["TARGET"]})
        .groupby("category", dropna=False)["TARGET"]
        .agg(count="size", bad_count="sum", bad_rate="mean")
        .reset_index()
    )
    grouped["good_count"] = grouped["count"] - grouped["bad_count"]
    grouped["population_percent"] = grouped["count"] / len(frame) * 100.0
    grouped["bad_rate_percent"] = grouped["bad_rate"] * 100.0
    return grouped.sort_values("count", ascending=False).reset_index(drop=True)


def plot_target_rates(table: pd.DataFrame, title: str, path: Path) -> None:
    shown = table.head(30).iloc[::-1]
    height = max(4.5, min(12.0, 0.32 * len(shown) + 1.8))
    plt.figure(figsize=(10, height))
    plt.barh(shown["category"], shown["bad_rate_percent"], color="#e45756")
    plt.axvline(
        table["bad_count"].sum() / table["count"].sum() * 100.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Overall bad rate",
    )
    plt.xlabel("TARGET=1 rate (%)")
    plt.title(title)
    plt.legend()
    plt.grid(axis="x", alpha=0.2)
    save_figure(path)


def write_table_profiles(
    input_dir: Path, output_dir: Path, sample_rows: int | None
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, filename in NOTEBOOK_TABLES.items():
        path = input_dir / filename
        frame = read_csv(path, sample_rows)
        save_frame(frame.head(10), output_dir / "samples" / f"{table_name}_head.csv")
        profile = missing_profile(frame)
        save_frame(profile, output_dir / "missing" / f"{table_name}.csv")
        rows.append(
            {
                "table": table_name,
                "source_file": filename,
                "rows_analyzed": len(frame),
                "columns": len(frame.columns),
                "missing_cells": int(frame.isna().sum().sum()),
                "columns_with_missing": int((frame.isna().sum() > 0).sum()),
                "memory_bytes": int(frame.memory_usage(deep=True).sum()),
                "source_bytes": path.stat().st_size,
            }
        )
        del frame, profile
        gc.collect()
    overview = pd.DataFrame(rows)
    save_frame(overview, output_dir / "dataset_overview.csv")
    return overview


def write_application_eda(
    frame: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_counts = frame["TARGET"].value_counts().sort_index()
    target = pd.DataFrame(
        {
            "TARGET": target_counts.index,
            "count": target_counts.to_numpy(),
            "percent": target_counts.to_numpy() / len(frame) * 100.0,
        }
    )
    save_frame(target, output_dir / "application" / "target_distribution.csv")
    plt.figure(figsize=(7, 4.5))
    plt.bar(target["TARGET"].astype(str), target["percent"], color=["#4c78a8", "#e45756"])
    plt.ylabel("Percent of applications")
    plt.xlabel("TARGET")
    plt.title("Application target distribution")
    save_figure(output_dir / "plots" / "application_target_distribution.png")

    numeric = frame.select_dtypes(include=np.number)
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary.insert(0, "missing_count", numeric.isna().sum())
    summary.insert(1, "missing_percent", numeric.isna().mean() * 100.0)
    summary.insert(2, "nunique", numeric.nunique(dropna=True))
    summary.index.name = "feature"
    save_frame(summary.reset_index(), output_dir / "application" / "numeric_summary.csv")

    for column in AMOUNT_COLUMNS:
        values = frame[column].dropna()
        upper = values.quantile(0.99)
        plt.figure(figsize=(9, 4.8))
        plt.hist(values.clip(upper=upper), bins=60, color="#4c78a8", alpha=0.9)
        plt.xlabel(column)
        plt.ylabel("Rows")
        plt.title(f"{column} distribution (clipped at p99={upper:,.2f})")
        save_figure(output_dir / "plots" / f"application_{slug(column)}_distribution.png")

    all_distributions: list[pd.DataFrame] = []
    all_target_rates: list[pd.DataFrame] = []
    for column in APPLICATION_CATEGORIES:
        distribution = categorical_distribution(frame, column)
        distribution.insert(0, "feature", column)
        all_distributions.append(distribution)
        plot_distribution_table(
            distribution,
            f"Application distribution: {column}",
            output_dir / "plots" / f"application_{slug(column)}_distribution.png",
        )
    for column in APPLICATION_TARGET_CATEGORIES:
        rates = target_rate_table(frame, column)
        rates.insert(0, "feature", column)
        all_target_rates.append(rates)
        plot_target_rates(
            rates,
            f"Application TARGET=1 rate: {column}",
            output_dir / "plots" / f"application_{slug(column)}_bad_rate.png",
        )

    distributions = pd.concat(all_distributions, ignore_index=True)
    target_rates = pd.concat(all_target_rates, ignore_index=True)
    save_frame(distributions, output_dir / "application" / "categorical_distributions.csv")
    save_frame(target_rates, output_dir / "application" / "categorical_bad_rates.csv")

    correlations = numeric.corr(method="pearson")
    save_frame(
        correlations.reset_index(names="feature"),
        output_dir / "application" / "pearson_correlation_matrix.csv",
    )
    target_correlations = (
        correlations["TARGET"]
        .drop("TARGET")
        .rename("pearson_correlation_with_target")
        .to_frame()
    )
    target_correlations["absolute_correlation"] = target_correlations[
        "pearson_correlation_with_target"
    ].abs()
    target_correlations = target_correlations.sort_values(
        "absolute_correlation", ascending=False
    ).reset_index(names="feature")
    save_frame(
        target_correlations,
        output_dir / "application" / "target_correlations.csv",
    )

    heatmap_features = ["TARGET", *target_correlations.head(29)["feature"].tolist()]
    heatmap = correlations.loc[heatmap_features, heatmap_features]
    plt.figure(figsize=(15, 13))
    image = plt.imshow(heatmap, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.xticks(range(len(heatmap)), heatmap.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(heatmap)), heatmap.index, fontsize=7)
    plt.colorbar(image, fraction=0.03, pad=0.02, label="Pearson correlation")
    plt.title("Top numeric features by absolute correlation with TARGET")
    save_figure(output_dir / "plots" / "application_pearson_correlation_heatmap.png")
    return target, target_correlations


def write_previous_application_eda(frame: pd.DataFrame, output_dir: Path) -> None:
    distributions: list[pd.DataFrame] = []
    for column in PREVIOUS_APPLICATION_CATEGORIES:
        distribution = categorical_distribution(frame, column)
        distribution.insert(0, "feature", column)
        distributions.append(distribution)
        plot_distribution_table(
            distribution,
            f"Previous application distribution: {column}",
            output_dir / "plots" / f"previous_{slug(column)}_distribution.png",
        )
    save_frame(
        pd.concat(distributions, ignore_index=True),
        output_dir / "previous_application" / "categorical_distributions.csv",
    )


def encode_for_random_forest(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
    for column in encoded.select_dtypes(exclude=np.number).columns:
        encoded[column] = encoded[column].astype("string").fillna("<MISSING>").astype("category").cat.codes
    return encoded.replace([np.inf, -np.inf], np.nan).fillna(-999).astype(np.float32)


def write_feature_importance(
    frame: pd.DataFrame, output_dir: Path, n_jobs: int
) -> pd.DataFrame:
    features = encode_for_random_forest(frame)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_leaf=4,
        max_features=0.5,
        random_state=2018,
        n_jobs=n_jobs,
    )
    print("Fitting notebook Random Forest feature importance ...", flush=True)
    model.fit(features, frame["TARGET"])
    importance = pd.DataFrame(
        {"feature": features.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    save_frame(importance, output_dir / "feature_importance" / "random_forest.csv")
    shown = importance.head(40).iloc[::-1]
    plt.figure(figsize=(10, 12))
    plt.barh(shown["feature"], shown["importance"], color="#59a14f")
    plt.xlabel("Impurity-based feature importance")
    plt.title("Random Forest feature importance (notebook configuration, top 40)")
    plt.grid(axis="x", alpha=0.2)
    save_figure(output_dir / "plots" / "random_forest_feature_importance.png")
    return importance


def notebook_analysis_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("cells 7-41", "load tables and inspect missing data", "dataset_overview.csv; missing/*.csv; samples/*.csv"),
            ("cells 44-48", "amount distributions", "plots/application_amt_*_distribution.png"),
            ("cells 50-76", "application categorical distributions", "application/categorical_distributions.csv; plots/application_*_distribution.png"),
            ("cells 80-92", "application outcomes by category", "application/categorical_bad_rates.csv; plots/application_*_bad_rate.png"),
            ("cells 95-133", "previous-application distributions", "previous_application/categorical_distributions.csv; plots/previous_*_distribution.png"),
            ("cell 135", "Pearson correlations", "application/pearson_correlation_matrix.csv; application/target_correlations.csv; plots/application_pearson_correlation_heatmap.png"),
            ("cells 137-140", "Random Forest feature importance", "feature_importance/random_forest.csv; plots/random_forest_feature_importance.png"),
        ],
        columns=["source_notebook_cells", "analysis", "generated_outputs"],
    )


def generated_artifact_paths(output_dir: Path) -> list[str]:
    """Inventory only files owned by this runner, not older HCDR artifacts."""
    fixed = [
        output_dir / "dataset_overview.csv",
        output_dir / "notebook_analysis_map.csv",
        output_dir / "summary.md",
    ]
    owned_directories = [
        output_dir / "application",
        output_dir / "feature_importance",
        output_dir / "missing",
        output_dir / "plots",
        output_dir / "previous_application",
        output_dir / "samples",
    ]
    paths = [path for path in fixed if path.is_file()]
    for directory in owned_directories:
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(str(path.relative_to(output_dir)) for path in set(paths))


def write_summary(
    output_dir: Path,
    overview: pd.DataFrame,
    target: pd.DataFrame,
    correlations: pd.DataFrame,
    importance: pd.DataFrame | None,
) -> None:
    bad_rate = target.loc[target["TARGET"] == 1, "percent"].iloc[0]
    top_missing_rows: list[str] = []
    for table in overview["table"]:
        missing = pd.read_csv(output_dir / "missing" / f"{table}.csv").head(3)
        rendered = ", ".join(
            f"{row.column} ({row.missing_percent:.2f}%)" for row in missing.itertuples()
        )
        top_missing_rows.append(f"- `{table}`: {rendered}")
    top_correlations = ", ".join(
        f"`{row.feature}` ({row.pearson_correlation_with_target:+.4f})"
        for row in correlations.head(10).itertuples()
    )
    feature_text = "Skipped by configuration."
    if importance is not None:
        feature_text = ", ".join(
            f"`{row.feature}` ({row.importance:.4f})"
            for row in importance.head(10).itertuples()
        )
    summary = f"""# HCDR detailed EDA from the reference notebook

Source: `{SOURCE_NOTEBOOK.relative_to(REPOSITORY_ROOT)}`.

This output reproduces the reference notebook's analytical scope with the current
official local CSV files. Static CSV/PNG artifacts replace legacy inline Plotly and
Cufflinks output. `TARGET=1` means payment difficulty in the competition data.

## Dataset overview

- Tables analyzed: {len(overview)}
- Application-train rows: {int(overview.loc[overview['table'] == 'application_train', 'rows_analyzed'].iloc[0]):,}
- Application-train bad rate: {bad_rate:.4f}%

## Highest missing percentages by table

{chr(10).join(top_missing_rows)}

## Strongest numeric Pearson correlations with TARGET

{top_correlations}

## Random Forest importance

{feature_text}

## Interpretation limits

- This is descriptive benchmark EDA, not evidence of production suitability.
- Correlation and impurity-based importance are associative, not causal.
- Random Forest importance can favor continuous or high-cardinality features.
- The reference notebook's category-by-target charts divide each target count by
  category population; this reproduction reports the equivalent value explicitly as
  `bad_rate` and preserves category support counts.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_rows is not None and args.sample_rows <= 0:
        raise ValueError("--sample-rows must be positive")
    missing_inputs = [
        str(input_dir / filename)
        for filename in NOTEBOOK_TABLES.values()
        if not (input_dir / filename).is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"Missing notebook input files: {missing_inputs}")
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(f"Missing source notebook: {SOURCE_NOTEBOOK}")

    started = datetime.now(UTC)
    analysis_map = notebook_analysis_map()
    save_frame(analysis_map, output_dir / "notebook_analysis_map.csv")
    overview = write_table_profiles(input_dir, output_dir, args.sample_rows)

    application = read_csv(input_dir / NOTEBOOK_TABLES["application_train"], args.sample_rows)
    target, correlations = write_application_eda(application, output_dir)
    importance = None
    if not args.skip_feature_importance:
        importance = write_feature_importance(
            application, output_dir, args.random_forest_jobs
        )
    del application
    gc.collect()

    previous = read_csv(input_dir / NOTEBOOK_TABLES["previous_application"], args.sample_rows)
    write_previous_application_eda(previous, output_dir)
    del previous
    gc.collect()

    write_summary(output_dir, overview, target, correlations, importance)
    finished = datetime.now(UTC)
    artifacts = generated_artifact_paths(output_dir)
    source_manifest = input_dir / "source.json"
    manifest = {
        "analysis": "HCDR detailed EDA adapted from the top-voted reference notebook",
        "source_notebook": str(SOURCE_NOTEBOOK.relative_to(REPOSITORY_ROOT)),
        "source_notebook_sha256": sha256(SOURCE_NOTEBOOK),
        "input_directory": str(input_dir),
        "input_source_manifest": str(source_manifest) if source_manifest.is_file() else None,
        "input_source_manifest_sha256": sha256(source_manifest) if source_manifest.is_file() else None,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "sample_rows": args.sample_rows,
        "feature_importance_skipped": args.skip_feature_importance,
        "random_forest": None
        if args.skip_feature_importance
        else {
            "n_estimators": 50,
            "max_depth": 8,
            "min_samples_leaf": 4,
            "max_features": 0.5,
            "random_state": 2018,
            "n_jobs": args.random_forest_jobs,
            "category_encoding": "train-only pandas categorical codes",
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(artifacts) + 1} artifacts to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
