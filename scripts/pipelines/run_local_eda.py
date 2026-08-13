#!/usr/bin/env python3
"""Script to run EDA analysis and export unique values table for each column in local data to CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_data_pipeline import run_local_eda


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unique values EDA table for each column in local credit data sample.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "raw" / "local_data" / "DC5xQACI- Data sample_v1.0.xlsx"),
        help="Path to input Excel dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "eda"),
        help="Directory to save EDA CSV output tables",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print(f"Running unique values EDA on input: {input_path}")
    df_raw_eda, raw_csv_path, proc_csv_path, train_csv_path = run_local_eda(
        excel_path=input_path,
        output_dir=output_dir,
    )

    # Also save a copy under datasets/processed/local_data/ for convenient access
    proc_data_dir = PROJECT_ROOT / "datasets" / "processed" / "local_data"
    proc_data_dir.mkdir(parents=True, exist_ok=True)
    raw_copy_path = proc_data_dir / "raw_column_unique_values.csv"
    df_raw_eda.to_csv(raw_copy_path, index=False, encoding="utf-8-sig")

    print("\n--- Local Data EDA Unique Values Summary ---")
    print(f"Columns analyzed: {len(df_raw_eda)}")
    print(f"Saved Raw Column Unique Values CSV to: {raw_csv_path}")
    print(f"Saved Processed Feature Unique Values CSV to: {proc_csv_path}")
    print(f"Saved Training Feature Unique Values CSV to: {train_csv_path}")
    print(f"Saved dataset copy to: {raw_copy_path}")

    print("\nTop 5 columns unique values preview:")
    cols_to_print = ["column_index", "column_name", "domain_table", "null_percentage", "unique_count", "unique_values"]
    print(df_raw_eda[cols_to_print].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
