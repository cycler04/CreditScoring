#!/usr/bin/env python3
"""Run the Home Credit Default Risk experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from home_credit_default_rate.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["A", "B", "C"], default="C")
    args = parser.parse_args()
    summary = run_pipeline(
        Path("datasets/raw/home-credit-default-risk"),
        Path("datasets/processed/hcdr"),
        Path("outputs/hcdr"),
        level=args.level,
    )
    print(
        f"Completed level {summary['level']}: "
        f"{summary['train_rows']} labeled rows, "
        f"{summary['feature_count']} features"
    )


if __name__ == "__main__":
    main()
