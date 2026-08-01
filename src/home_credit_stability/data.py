"""Dataset discovery, identity, and lazy scan helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

COMPETITION = "home-credit-credit-risk-model-stability"
ID_COLUMN = "case_id"
TARGET = "target"


def parquet_root(raw_dir: Path) -> Path:
    """Return the competition Parquet root after validating its layout."""
    root = raw_dir / "parquet_files"
    required = [
        root / "train" / "train_base.parquet",
        root / "test" / "test_base.parquet",
        raw_dir / "feature_definitions.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required competition files: {missing}")
    return root


def base_path(raw_dir: Path, split: str) -> Path:
    """Return train/test base Parquet path."""
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    return parquet_root(raw_dir) / split / f"{split}_base.parquet"


def scan_base(raw_dir: Path, split: str) -> pl.LazyFrame:
    """Lazily scan the compact base table with normalized dtypes."""
    frame = pl.scan_parquet(base_path(raw_dir, split))
    expressions = [
        pl.col(ID_COLUMN).cast(pl.Int64),
        pl.col("date_decision").str.to_date("%Y-%m-%d"),
        pl.col("WEEK_NUM").cast(pl.Int32),
        pl.col("MONTH").cast(pl.Int32),
    ]
    if split == "train":
        expressions.append(pl.col(TARGET).cast(pl.Int8))
    return frame.with_columns(expressions)


def dataset_inventory(raw_dir: Path) -> dict[str, object]:
    """Read Parquet metadata without loading table bodies."""
    root = parquet_root(raw_dir)
    rows: list[dict[str, object]] = []
    total_bytes = 0
    total_physical_rows = 0
    for path in sorted(root.rglob("*.parquet")):
        metadata = pq.ParquetFile(path).metadata
        size = path.stat().st_size
        total_bytes += size
        total_physical_rows += metadata.num_rows
        rows.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "bytes": size,
                "rows": metadata.num_rows,
                "columns": metadata.num_columns,
            }
        )
    return {
        "competition": COMPETITION,
        "parquet_file_count": len(rows),
        "parquet_bytes": total_bytes,
        "physical_rows_across_tables": total_physical_rows,
        "files": rows,
    }


def write_source_manifest(raw_dir: Path) -> dict[str, object]:
    """Write SHA-256 fingerprints for every retained raw source file."""
    inventory = dataset_inventory(raw_dir)
    files: list[dict[str, object]] = []
    retained = sorted(
        [
            *parquet_root(raw_dir).rglob("*.parquet"),
            raw_dir / "feature_definitions.csv",
            *(
                [raw_dir / "sample_submission.csv"]
                if (raw_dir / "sample_submission.csv").exists()
                else []
            ),
        ]
    )
    for path in retained:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    manifest = {
        **{key: value for key, value in inventory.items() if key != "files"},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
    }
    (raw_dir / "source.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
