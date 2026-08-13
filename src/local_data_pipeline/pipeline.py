"""Pipeline orchestrator for local data processing and feature engineering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .features import extract_features, select_training_features


class LocalDataPipeline:
    """Orchestrator for reading, engineering, and exporting local data features."""

    def __init__(
        self,
        excel_path: Path | str,
        output_dir: Optional[Path | str] = None,
    ) -> None:
        self.excel_path = Path(excel_path)
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else self.excel_path.parent.parent.parent / "processed" / "local_data"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw sample data and feature description sheets from Excel."""
        if not self.excel_path.is_file():
            raise FileNotFoundError(f"Excel data file not found: {self.excel_path}")

        xl = pd.ExcelFile(self.excel_path)
        
        sample_sheet = "sample" if "sample" in xl.sheet_names else xl.sheet_names[0]
        desc_sheet = "descibe" if "descibe" in xl.sheet_names else (
            "describe" if "describe" in xl.sheet_names else None
        )

        df_sample = pd.read_excel(self.excel_path, sheet_name=sample_sheet)
        df_desc = (
            pd.read_excel(self.excel_path, sheet_name=desc_sheet)
            if desc_sheet
            else pd.DataFrame()
        )
        return df_sample, df_desc

    def process(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run full feature engineering pipeline and generate summary metadata."""
        df_sample, _ = self.load_raw_data()
        df_features = extract_features(df_sample)
        numeric_cols = df_features.select_dtypes(include=["number"]).columns.tolist()
        summary_meta = {
            "input_file": str(self.excel_path),
            "num_rows": len(df_features),
            "num_features": len(df_features.columns),
            "numeric_feature_count": len(numeric_cols),
            "features_list": list(df_features.columns),
        }
        return df_features, summary_meta

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
        """Execute processing pipeline and save full & training feature artifacts to disk."""
        df_sample, df_desc = self.load_raw_data()
        df_features = extract_features(df_sample)
        df_training = select_training_features(df_features, include_id=True, drop_zero_variance=True, drop_raw_strings=True)

        # Output paths
        parquet_path = self.output_dir / "processed_features.parquet"
        csv_path = self.output_dir / "processed_features.csv"
        train_parquet_path = self.output_dir / "training_features.parquet"
        train_csv_path = self.output_dir / "training_features.csv"

        df_features.to_parquet(parquet_path, index=False)
        df_features.to_csv(csv_path, index=False)
        df_training.to_parquet(train_parquet_path, index=False)
        df_training.to_csv(train_csv_path, index=False)

        # Summary statistics
        numeric_cols = df_features.select_dtypes(include=["number"]).columns.tolist()
        summary_meta = {
            "input_file": str(self.excel_path),
            "output_parquet": str(parquet_path),
            "output_csv": str(csv_path),
            "output_training_parquet": str(train_parquet_path),
            "output_training_csv": str(train_csv_path),
            "num_rows": len(df_features),
            "num_full_features": len(df_features.columns),
            "num_training_features": len(df_training.columns) - 1 if "user_id" in df_training.columns else len(df_training.columns),
            "numeric_feature_count": len(numeric_cols),
            "training_features_list": list(df_training.columns),
            "missing_rates": df_training.isna().mean().to_dict(),
        }

        summary_json_path = self.output_dir / "feature_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_meta, f, indent=2, ensure_ascii=False)

        return df_features, df_training, parquet_path, csv_path, train_parquet_path, train_csv_path, summary_json_path


def run_pipeline(
    excel_path: Path | str,
    output_dir: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
    """Functional entry point to execute LocalDataPipeline."""
    pipeline = LocalDataPipeline(excel_path=excel_path, output_dir=output_dir)
    return pipeline.run()
