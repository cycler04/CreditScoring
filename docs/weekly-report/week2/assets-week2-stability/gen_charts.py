#!/usr/bin/env python3
"""Generate report charts from measured HCMS artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "outputs/hcms"
ASSET_DIR = Path(__file__).resolve().parent


def selfcheck() -> None:
    """Assert the chart inputs preserve the experiment contracts."""
    stage_metrics = pd.read_csv(OUTPUT / "models/stage_metrics.csv")
    stage_stability = pd.read_csv(OUTPUT / "stability/stage_stability.csv")
    gini = pd.read_csv(OUTPUT / "stability/gini_by_week.csv")
    membership = pd.read_csv(
        ROOT / "datasets/processed/hcms/split_membership.csv"
    )
    coefficients = pd.read_csv(OUTPUT / "scorecard/coefficients.csv")
    scorecard = pd.read_csv(OUTPUT / "scorecard/scorecard.csv")

    test_auc = (
        stage_metrics.loc[stage_metrics["split"].eq("test")]
        .set_index("level")
        .loc[list("ABC"), "auc"]
    )
    assert test_auc.is_monotonic_increasing
    assert stage_stability.set_index("level").loc[
        list("ABC"), "stability"
    ].is_monotonic_increasing
    assert membership.groupby("WEEK_NUM")["split"].nunique().max() == 1
    assert set(gini["model"]) == {
        "lightgbm",
        "xgboost",
        "logistic_raw",
        "logistic_woe",
    }
    assert gini.groupby("model")["WEEK_NUM"].nunique().eq(19).all()
    assert gini["n"].min() >= 500
    assert (coefficients["coefficient"] < 0).all()
    grouped = scorecard.groupby("feature")["points"]
    assert int(grouped.min().sum()) == 300
    assert int(grouped.max().sum()) == 850


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
        }
    )


def stage_chart() -> None:
    metrics = pd.read_csv(OUTPUT / "models/stage_metrics.csv")
    auc = (
        metrics.loc[metrics["split"].eq("test")]
        .set_index("level")
        .loc[list("ABC"), "auc"]
    )
    stability = (
        pd.read_csv(OUTPUT / "stability/stage_stability.csv")
        .set_index("level")
        .loc[list("ABC"), "stability"]
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    levels = [0, 1, 2]
    width = 0.36
    axis.bar(
        [value - width / 2 for value in levels],
        auc,
        width,
        label="Test AUC",
        color="#2563eb",
    )
    axis.bar(
        [value + width / 2 for value in levels],
        stability,
        width,
        label="Stability",
        color="#0f766e",
    )
    axis.set_xticks(levels, ["A: static", "B: + depth 1", "C: + depth 2"])
    axis.set_ylim(0.4, 0.9)
    axis.set_title("Feature depth improves both discrimination and stability")
    axis.legend(loc="upper left")
    for index, value in enumerate(auc):
        axis.text(index - width / 2, value + 0.008, f"{value:.3f}", ha="center")
    for index, value in enumerate(stability):
        axis.text(index + width / 2, value + 0.008, f"{value:.3f}", ha="center")
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "stage_auc_stability.png")
    plt.close(figure)


def gini_chart() -> None:
    frame = pd.read_csv(OUTPUT / "stability/gini_by_week.csv")
    labels = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "logistic_woe": "Logistic WoE",
        "logistic_raw": "Logistic raw",
    }
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    for model, group in frame.groupby("model"):
        axis.plot(
            group["WEEK_NUM"],
            group["gini"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=labels[model],
        )
    axis.set_title("Weekly test-period gini (weeks 73–91)")
    axis.set_xlabel("WEEK_NUM")
    axis.set_ylabel("Gini")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "gini_by_week.png")
    plt.close(figure)


def bad_rate_chart() -> None:
    frame = pd.read_csv(OUTPUT / "eda/bad_rate_by_week.csv")
    colors = {"train": "#2563eb", "valid": "#d97706", "test": "#0f766e"}
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    for split, group in frame.groupby("split", sort=False):
        axis.plot(
            group["WEEK_NUM"],
            group["bad_rate"] * 100,
            color=colors[split],
            linewidth=1.8,
            label=split,
        )
    axis.set_title("Bad rate changes materially across calendar weeks")
    axis.set_xlabel("WEEK_NUM")
    axis.set_ylabel("Bad rate (%)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "bad_rate_by_week.png")
    plt.close(figure)


def monitoring_chart() -> None:
    psi_frame = pd.read_csv(OUTPUT / "scorecard/score_psi_by_week.csv")
    cutoff_frame = pd.read_csv(OUTPUT / "scorecard/cutoffs_by_week.csv")
    cutoff_frame = cutoff_frame.loc[cutoff_frame["approval_target"].eq(0.7)]
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    axes[0].plot(
        psi_frame["WEEK_NUM"],
        psi_frame["score_psi"],
        marker="o",
        color="#7c3aed",
    )
    axes[0].axhline(0.1, color="#d97706", linestyle="--", label="PSI 0.1")
    axes[0].axhline(0.25, color="#dc2626", linestyle="--", label="PSI 0.25")
    axes[0].set_ylabel("Score PSI vs train")
    axes[0].legend()
    axes[1].plot(
        cutoff_frame["WEEK_NUM"],
        cutoff_frame["approval_rate"] * 100,
        marker="o",
        color="#0f766e",
    )
    axes[1].axhline(70, color="#111827", linestyle="--", label="70% target")
    axes[1].set_ylabel("Approval rate (%)")
    axes[1].set_xlabel("WEEK_NUM")
    axes[1].legend()
    figure.suptitle("A frozen scorecard policy drifts across test weeks")
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "scorecard_monitoring.png")
    plt.close(figure)


def main() -> None:
    selfcheck()
    _style()
    stage_chart()
    gini_chart()
    bad_rate_chart()
    monitoring_chart()
    print(f"Generated four charts in {ASSET_DIR}")


if __name__ == "__main__":
    main()
