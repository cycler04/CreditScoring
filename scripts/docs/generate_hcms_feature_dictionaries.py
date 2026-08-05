#!/usr/bin/env python3
"""Generate raw and engineered HCMS feature dictionaries from local artifacts."""

from __future__ import annotations

import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from home_credit_stability.aggregate import LEVEL_FAMILIES, _family_files


RAW_DIR = Path("datasets/raw/home-credit-model-stability")
PROCESSED_DIR = Path("datasets/processed/hcms")
OUTPUT_DIR = Path("docs/03-competitions/home-credit-model-stability/details")
STRUCTURAL_DETAILS = {
    "case_id": "Unique identifier of the credit application/case.",
    "target": "Binary competition target available only in train_base.",
    "date_decision": "Decision date of the current credit application.",
    "MONTH": "Calendar-month index supplied in the base table.",
    "WEEK_NUM": "Consecutive week index used by the competition stability metric.",
    "num_group1": "First-level ordering/index within a case.",
    "num_group2": "Second-level ordering/index within a first-level record.",
    "score": "Predicted probability required by the competition submission.",
}


def _format_sample(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return format(value, ".12g")
    return str(value)


def _union_columns(paths: list[Path]) -> list[str]:
    columns: list[str] = []
    for path in paths:
        for name in pq.read_schema(path).names:
            if name not in columns:
                columns.append(name)
    return columns


def _first_non_null(paths: list[Path], columns: list[str]) -> dict[str, str]:
    unresolved = set(columns)
    samples: dict[str, str] = {}
    for path in paths:
        available = [name for name in unresolved if name in pq.read_schema(path).names]
        if not available:
            continue
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65_536, columns=available):
            table = pa.Table.from_batches([batch])
            for name in list(unresolved):
                if name not in table.column_names:
                    continue
                values = table[name].drop_null()
                if len(values):
                    samples[name] = _format_sample(values[0].as_py())
                    unresolved.remove(name)
            if not unresolved:
                return samples
        if not unresolved:
            break
    for name in unresolved:
        samples[name] = "<no non-null value in local train snapshot>"
    return samples


def _write_dictionary(
    output: Path,
    columns: list[str],
    descriptions: dict[str, str],
    samples: dict[str, str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "name", "details", "sample"])
        for index, name in enumerate(columns, start=1):
            writer.writerow(
                [
                    index,
                    name,
                    descriptions.get(name, STRUCTURAL_DETAILS.get(name, "Unknown")),
                    samples[name],
                ]
            )


def _engineered_details(name: str, descriptions: dict[str, str]) -> tuple[str, str]:
    if "__" not in name:
        return "base", STRUCTURAL_DETAILS.get(name, descriptions.get(name, "Base-table field."))
    family, remainder = name.split("__", 1)
    rules = [
        ("__MIN_GAP", "minimum decision-date gap"),
        ("__MAX_GAP", "maximum decision-date gap"),
        ("__NUNIQUE", "sum of per-partition distinct counts"),
        ("__MEAN", "global non-null mean reconstructed from partial sums/counts"),
        ("__MAX", "maximum"),
    ]
    if remainder == "ROW_COUNT":
        return family.lower(), "Number of raw family rows linked to the case."
    operation = "depth-0 value copied to case grain"
    raw_name = remainder
    for suffix, label in rules:
        if remainder.endswith(suffix):
            raw_name = remainder[: -len(suffix)]
            operation = label
            break
    raw_detail = descriptions.get(raw_name, "Raw feature described by its source schema.")
    return family.lower(), f"{operation} of `{raw_name}`. Source meaning: {raw_detail}"


def main() -> None:
    with (RAW_DIR / "feature_definitions.csv").open(encoding="utf-8", newline="") as stream:
        descriptions = {
            row["Variable"]: row["Description"] for row in csv.DictReader(stream)
        }
    descriptions.update(STRUCTURAL_DETAILS)

    base_path = RAW_DIR / "parquet_files/train/train_base.parquet"
    base_columns = pq.read_schema(base_path).names
    _write_dictionary(
        OUTPUT_DIR / "train_base_features.csv",
        base_columns,
        descriptions,
        _first_non_null([base_path], base_columns),
    )

    for families in LEVEL_FAMILIES.values():
        for family in families:
            paths = _family_files(RAW_DIR, "train", family)
            columns = _union_columns(paths)
            _write_dictionary(
                OUTPUT_DIR / f"{family}_features.csv",
                columns,
                descriptions,
                _first_non_null(paths, columns),
            )

    sample_path = RAW_DIR / "sample_submission.csv"
    with sample_path.open(encoding="utf-8", newline="") as stream:
        sample_rows = list(csv.DictReader(stream))
    submission_columns = ["case_id", "score"]
    _write_dictionary(
        OUTPUT_DIR / "sample_submission_features.csv",
        submission_columns,
        descriptions,
        {name: sample_rows[0][name] for name in submission_columns},
    )

    matrix_path = PROCESSED_DIR / "feature_matrix_train_C.parquet"
    matrix_columns = pq.read_schema(matrix_path).names
    stage_a = set(pq.read_schema(PROCESSED_DIR / "feature_matrix_train_A.parquet").names)
    stage_b = set(pq.read_schema(PROCESSED_DIR / "feature_matrix_train_B.parquet").names)
    with (OUTPUT_DIR / "src_engineered_features.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "name", "stage", "source_family", "details"])
        for index, name in enumerate(matrix_columns, start=1):
            if name in STRUCTURAL_DETAILS:
                stage = "base"
            elif name in stage_a:
                stage = "A"
            elif name in stage_b:
                stage = "B"
            else:
                stage = "C"
            family, detail = _engineered_details(name, descriptions)
            writer.writerow([index, name, stage, family, detail])


if __name__ == "__main__":
    main()
