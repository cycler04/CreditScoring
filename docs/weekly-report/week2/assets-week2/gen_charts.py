#!/usr/bin/env python3
"""Generate Week 2 charts and verify their source artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "hcdr"
PROCESSED = ROOT / "datasets" / "processed" / "hcdr"
RAW = ROOT / "datasets" / "raw" / "home-credit-default-risk"
ASSETS = Path(__file__).resolve().parent


def selfcheck() -> None:
    manifest = json.loads((RAW / "source.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 10

    final = pd.read_csv(OUTPUT / "models" / "metrics_C.csv")
    assert set(final["model"]) == {
        "logistic_raw",
        "lightgbm",
        "xgboost",
        "logistic_woe",
    }
    pivot = final.pivot(index="model", columns="split", values="auc")
    assert pivot.loc["lightgbm", "test"] >= 0.77
    assert pivot.loc["logistic_woe", "test"] >= 0.74
    assert (pivot["valid"] - pivot["test"]).abs().max() < 0.01

    scorecard = pd.read_csv(OUTPUT / "scorecard" / "scorecard.csv")
    assert scorecard["coefficient"].max() < 0
    grouped = scorecard.groupby("feature")["points"]
    assert int(grouped.min().sum()) == 300
    assert int(grouped.max().sum()) == 850

    membership = pd.read_csv(PROCESSED / "split_membership.csv")
    target = pd.read_parquet(
        PROCESSED / "feature_matrix_C.parquet",
        columns=["SK_ID_CURR", "TARGET"],
    ).dropna(subset=["TARGET"])
    rates = membership.merge(target).groupby("split")["TARGET"].mean()
    assert rates.max() - rates.min() < 0.001


def main() -> None:
    selfcheck()
    ASSETS.mkdir(parents=True, exist_ok=True)

    rows = []
    for level in "ABC":
        metrics = pd.read_csv(OUTPUT / "models" / f"metrics_{level}.csv")
        metrics = metrics.loc[metrics["split"].eq("test"), ["model", "auc"]]
        metrics["level"] = level
        rows.append(metrics)
    stage = pd.concat(rows).pivot(index="level", columns="model", values="auc")
    axes = stage.plot(marker="o", figsize=(9, 5), ylim=(0.70, 0.80), grid=True)
    axes.set(
        title="Test AUC theo tầng feature",
        xlabel="Tầng feature",
        ylabel="AUC",
    )
    axes.figure.tight_layout()
    axes.figure.savefig(ASSETS / "stage_auc.png", dpi=160)
    plt.close(axes.figure)

    cutoffs = pd.read_csv(OUTPUT / "scorecard" / "cutoffs.csv")
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(
        cutoffs["approval_rate_actual"] * 100,
        cutoffs["approved_bad_rate"] * 100,
        marker="o",
    )
    axes.set(
        title="Bad rate trong nhóm được duyệt — test split",
        xlabel="Approval rate (%)",
        ylabel="Bad rate (%)",
    )
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(ASSETS / "approval_bad_rate.png", dpi=160)
    plt.close(figure)
    print("Week 2 chart selfcheck passed")


if __name__ == "__main__":
    main()
