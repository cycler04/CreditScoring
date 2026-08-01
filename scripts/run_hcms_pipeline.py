#!/usr/bin/env python3
"""Run the Home Credit Model Stability experiment."""

from __future__ import annotations

from pathlib import Path

from home_credit_stability.pipeline import run_pipeline


def main() -> None:
    summary = run_pipeline(
        Path("datasets/raw/home-credit-model-stability"),
        Path("datasets/processed/hcms"),
        Path("outputs/hcms"),
    )
    print(
        "Completed HCMS: "
        f"{summary['train_base_rows']} cases, "
        f"{summary['distinct_weeks']} weeks, "
        f"{summary['selected_feature_count_C']} final features"
    )


if __name__ == "__main__":
    main()
