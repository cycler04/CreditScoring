#!/usr/bin/env python3
"""Create a feature-importance table and diagram from saved models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_scoring.feature_importance import inspect_model_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect fitted .joblib files and write a combined feature-importance "
            "table and diagram."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models",
        help="Directory containing fitted .joblib model artifacts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional maximum number of features shown per model; default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, table_path, plot_paths = inspect_model_outputs(
        args.models_dir.resolve(),
        top_n=args.top_n,
    )
    display_columns = [
        "model",
        "rank",
        "feature",
        "importance_method",
        "importance_pct",
        "direction",
    ]
    display_table = (
        table
        if args.top_n is None
        else table.loc[table["rank"].le(args.top_n)]
    )
    print(display_table[display_columns].to_string(index=False))
    print(f"\nTable: {table_path}")
    for plot_path in plot_paths:
        print(f"Diagram: {plot_path}")


if __name__ == "__main__":
    main()
