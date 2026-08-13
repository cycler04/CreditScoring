"""EDA utility functions for analyzing unique values and column statistics in local credit scoring data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .parser import clean_column_name


def generate_raw_unique_values_table(excel_path: Path | str) -> pd.DataFrame:
    """Generate detailed EDA summary of unique values for every column in the raw Excel dataset."""
    path = Path(excel_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw Excel file not found: {path}")

    xl = pd.ExcelFile(path)
    sample_sheet = "sample" if "sample" in xl.sheet_names else xl.sheet_names[0]
    desc_sheet = "descibe" if "descibe" in xl.sheet_names else (
        "describe" if "describe" in xl.sheet_names else None
    )

    df_sample = pd.read_excel(path, sheet_name=sample_sheet)
    df_desc = pd.read_excel(path, sheet_name=desc_sheet) if desc_sheet else pd.DataFrame()

    # Normalize column names
    df_sample.columns = [clean_column_name(c) for c in df_sample.columns]

    desc_map = {}
    if not df_desc.empty:
        for _, row in df_desc.iterrows():
            col_name = clean_column_name(str(row.get("column", "")))
            tbl = str(row.get("table", "")).strip()
            desc = str(row.get("describe", "")).strip()
            desc_map[col_name] = (tbl, desc)

    rows = []
    for idx, col in enumerate(df_sample.columns):
        series = df_sample[col]
        tbl, desc = desc_map.get(col, ("unknown", ""))
        non_nulls = series.dropna()
        unique_vals = non_nulls.unique()

        formatted_vals = []
        for v in unique_vals:
            if isinstance(v, (pd.Timestamp, pd.DatetimeIndex)):
                formatted_vals.append(str(v)[:10])
            else:
                formatted_vals.append(str(v))

        unique_str = " | ".join(formatted_vals) if formatted_vals else "Không có dữ liệu (100% missing)"
        total = len(series)
        null_cnt = int(series.isna().sum())
        non_null_cnt = total - null_cnt
        null_pct = round((null_cnt / total) * 100.0, 2)

        rows.append(
            {
                "column_index": idx + 1,
                "column_name": col,
                "domain_table": tbl,
                "description": desc,
                "data_type": str(series.dtype),
                "total_rows": total,
                "non_null_count": non_null_cnt,
                "null_count": null_cnt,
                "null_percentage": f"{null_pct}%",
                "unique_count": len(unique_vals),
                "unique_values": unique_str,
            }
        )

    return pd.DataFrame(rows)


def generate_processed_unique_values_table(df_features: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed EDA summary of unique values for every engineered feature in processed DataFrame."""
    rows = []
    for idx, col in enumerate(df_features.columns):
        series = df_features[col]
        non_nulls = series.dropna()
        unique_vals = non_nulls.unique()

        formatted_vals = []
        for v in unique_vals[:15]:  # Cap at 15 for broad feature sets
            if isinstance(v, (float, int)):
                formatted_vals.append(f"{v:g}")
            else:
                formatted_vals.append(str(v))

        if len(unique_vals) > 15:
            formatted_vals.append(f"... (+{len(unique_vals) - 15} more)")

        unique_str = " | ".join(formatted_vals) if formatted_vals else "Empty / All Null"
        total = len(series)
        null_cnt = int(series.isna().sum())
        non_null_cnt = total - null_cnt
        null_pct = round((null_cnt / total) * 100.0, 2)

        rows.append(
            {
                "feature_index": idx + 1,
                "feature_name": col,
                "data_type": str(series.dtype),
                "total_rows": total,
                "non_null_count": non_null_cnt,
                "null_count": null_cnt,
                "null_percentage": f"{null_pct}%",
                "unique_count": len(unique_vals),
                "unique_values": unique_str,
            }
        )

    return pd.DataFrame(rows)


def run_local_eda(
    excel_path: Path | str,
    output_dir: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, Path, Path, Path]:
    """Run full EDA analysis, generating raw, processed, and training unique value tables as CSV files."""
    path = Path(excel_path)
    out_dir = Path(output_dir) if output_dir else Path("outputs/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw_eda = generate_raw_unique_values_table(path)
    raw_csv_path = out_dir / "local_data_raw_column_unique_values.csv"
    df_raw_eda.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

    from .pipeline import LocalDataPipeline
    from .features import extract_features, select_training_features

    pipeline = LocalDataPipeline(excel_path=path)
    df_sample, _ = pipeline.load_raw_data()

    df_features = extract_features(df_sample)
    df_proc_eda = generate_processed_unique_values_table(df_features)
    proc_csv_path = out_dir / "local_data_processed_column_unique_values.csv"
    df_proc_eda.to_csv(proc_csv_path, index=False, encoding="utf-8-sig")

    df_training = select_training_features(df_features, include_id=True, drop_zero_variance=True, drop_raw_strings=True)
    df_train_eda = generate_processed_unique_values_table(df_training)
    train_csv_path = out_dir / "local_data_training_column_unique_values.csv"
    df_train_eda.to_csv(train_csv_path, index=False, encoding="utf-8-sig")

    return df_raw_eda, raw_csv_path, proc_csv_path, train_csv_path
