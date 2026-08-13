#!/usr/bin/env python3
"""Script to run the local data feature extraction pipeline on DC5xQACI Excel data sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_data_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Process local credit data sample and extract competition-level features.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "raw" / "local_data" / "DC5xQACI- Data sample_v1.0.xlsx"),
        help="Path to input Excel dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "processed" / "local_data"),
        help="Directory to save processed feature outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print(f"Loading and processing raw data from: {input_path}")
    df_features, df_training, parquet_path, csv_path, train_parquet_path, train_csv_path, summary_json_path = run_pipeline(
        excel_path=input_path,
        output_dir=output_dir,
    )

    print("\n--- Pipeline Execution Summary ---")
    print(f"Rows processed: {len(df_features)}")
    print(f"Total features extracted: {len(df_features.columns)}")
    print(f"Training features selected (excl raw/constants): {len(df_training.columns) - 1 if 'user_id' in df_training.columns else len(df_training.columns)}")
    print(f"Saved Full Parquet features to: {parquet_path}")
    print(f"Saved Full CSV features to: {csv_path}")
    print(f"Saved Training Parquet features to: {train_parquet_path}")
    print(f"Saved Training CSV features to: {train_csv_path}")
    print(f"Saved Summary Metadata to: {summary_json_path}")
    
    print("\nSample Training Features Table (first 2 rows):")
    cols_to_show = list(df_training.columns)[:12]
    print(df_training[cols_to_show].head(2).to_string())


if __name__ == "__main__":
    main()
