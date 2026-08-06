"""Comparable benchmark-table helpers for persisted credit-risk experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BENCHMARK_COLUMNS = [
    "Model",
    "AUC",
    "Brier",
    "KS",
    "Active features",
    "Stability",
    "Monotonic violations",
    "Explanation time",
]

MODEL_LABELS = {
    "logistic_raw": "Logistic",
    "logistic_woe": "WoE scorecard",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "HistGradientBoosting",
    "catboost": "CatBoost",
    "ensemble_lightgbm_catboost": "LightGBM + CatBoost",
    "ensemble_lightgbm_xgboost_catboost": "LightGBM + XGBoost + CatBoost",
    "ensemble_lightgbm_catboost_extra_trees": (
        "LightGBM + CatBoost + Extra Trees"
    ),
    "ensemble_boosting": "Boosting ensemble",
    "ensemble_all_trees": "All-tree ensemble",
}

UNIMPLEMENTED_CANDIDATES = [
    "EBM (not implemented)",
    "GAM (not implemented)",
    "Monotonic LightGBM (not implemented)",
]


def build_benchmark_table(
    metrics: pd.DataFrame,
    brier_by_model: dict[str, float],
    *,
    active_features: dict[str, int | None],
    stability: dict[str, float] | None,
    monotonic_violations: dict[str, int | None],
) -> pd.DataFrame:
    """Combine measured test metrics with explicitly scoped diagnostics."""
    required = {"model", "split", "auc", "ks"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Metrics table requires columns {sorted(required)}")
    selected = metrics.loc[metrics["split"].eq("test")].copy()
    if "level" in selected and selected["level"].nunique() > 1:
        selected = selected.loc[selected["level"].eq("C")]
    selected = selected.drop_duplicates("model", keep="last")
    missing_brier = set(selected["model"]).difference(brier_by_model)
    if missing_brier:
        raise ValueError(f"Missing Brier scores for: {sorted(missing_brier)}")

    rows = []
    stability = stability or {}
    for row in selected.itertuples(index=False):
        model = row.model
        rows.append(
            {
                "Model": MODEL_LABELS.get(model, model),
                "AUC": float(row.auc),
                "Brier": float(brier_by_model[model]),
                "KS": float(row.ks),
                "Active features": active_features.get(model),
                "Stability": stability.get(model),
                "Monotonic violations": monotonic_violations.get(model),
                "Explanation time": None,
            }
        )
    table = pd.DataFrame(rows, columns=BENCHMARK_COLUMNS).sort_values(
        "AUC", ascending=False
    )
    candidates = pd.DataFrame(
        [{"Model": model} for model in UNIMPLEMENTED_CANDIDATES],
        columns=BENCHMARK_COLUMNS,
    )
    return pd.concat([table, candidates], ignore_index=True)


def _markdown_value(value: object, column: str) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if column in {"Active features", "Monotonic violations"}:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_benchmark_report(
    table: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    stability_definition: str,
) -> None:
    """Write a self-describing Markdown benchmark table."""
    header = "| " + " | ".join(BENCHMARK_COLUMNS) + " |"
    separator = "| " + " | ".join(
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    ) + " |"
    body = [
        "| "
        + " | ".join(
            _markdown_value(row[column], column) for column in BENCHMARK_COLUMNS
        )
        + " |"
        for _, row in table.iterrows()
    ]
    text = "\n".join(
        [
            f"# {title}",
            "",
            "## Metric contract",
            "",
            "- AUC, Brier, and KS use the persisted held-out test split.",
            "- Lower Brier is better; higher AUC, KS, and Stability are better.",
            "- Active features count non-zero persisted global-importance entries; "
            "ensembles and models without a native persisted importance are N/A.",
            f"- Stability: {stability_definition}",
            "- Monotonic violations are reported only for models whose monotonicity "
            "is enforced and auditable; other models are N/A.",
            "- Explanation time is N/A because no common explainer, hardware warm-up, "
            "or sample-size protocol has been benchmarked yet.",
            "- EBM, GAM, and monotonic LightGBM are candidate rows only and are not "
            "represented as measured models.",
            "- Gini is omitted because it is exactly `2 * AUC - 1` in these pipelines.",
            "",
            header,
            separator,
            *body,
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
