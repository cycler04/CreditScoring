"""Streaming suffix-driven aggregation for depth-0/1/2 tables."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from .data import ID_COLUMN, scan_base

LEVEL_FAMILIES = {
    "A": ["static_0", "static_cb_0"],
    "B": [
        "applprev_1",
        "credit_bureau_a_1",
        "credit_bureau_b_1",
        "debitcard_1",
        "deposit_1",
        "other_1",
        "person_1",
        "tax_registry_a_1",
        "tax_registry_b_1",
        "tax_registry_c_1",
    ],
    "C": [
        "applprev_2",
        "credit_bureau_a_2",
        "credit_bureau_b_2",
        "person_2",
    ],
}


def families_for_level(level: str) -> list[str]:
    """Return cumulative feature families for A/B/C."""
    if level not in LEVEL_FAMILIES:
        raise ValueError("level must be A, B, or C")
    levels = "ABC"[:"ABC".index(level) + 1]
    return [family for item in levels for family in LEVEL_FAMILIES[item]]


def _family_files(raw_dir: Path, split: str, family: str) -> list[Path]:
    root = raw_dir / "parquet_files" / split
    prefix = f"{split}_{family}"
    files = sorted(root.glob(f"{prefix}*.parquet"))
    # Avoid matching depth 2 when looking for a depth 1 family (and vice versa).
    files = [
        path
        for path in files
        if re.fullmatch(rf"{re.escape(prefix)}(?:_\d+)?\.parquet", path.name)
    ]
    if not files:
        raise FileNotFoundError(f"No files found for {split}/{family}")
    return files


def _scan_family(files: list[Path]) -> pl.LazyFrame:
    """Scan row-partitioned files while reconciling null-only test schemas."""
    return pl.concat(
        [pl.scan_parquet(path) for path in files],
        how="diagonal_relaxed",
    )


def _date_gap_expression(column: str) -> pl.Expr:
    event_date = (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.to_date("%Y-%m-%d", strict=False)
    )
    return (pl.col("date_decision") - event_date).dt.total_days().cast(pl.Float32)


def _selected_columns(
    schema: pl.Schema,
    *,
    max_columns: int,
) -> tuple[list[str], list[str], list[str]]:
    excluded = {ID_COLUMN, "num_group1", "num_group2", "date_decision"}
    numeric: list[str] = []
    dates: list[str] = []
    categorical: list[str] = []
    for name, dtype in schema.items():
        if name in excluded:
            continue
        if name.endswith("D") or dtype == pl.Date:
            dates.append(name)
        elif dtype.is_numeric() or dtype == pl.Boolean:
            numeric.append(name)
        elif name.endswith("M") or dtype == pl.String:
            categorical.append(name)
    # Preserve suffix semantics and schema order; cap each family before collect.
    prioritized = [
        *[name for name in numeric if name.endswith(("P", "A"))],
        *[name for name in numeric if name.endswith(("L", "T"))],
        *[name for name in numeric if not name.endswith(("P", "A", "L", "T"))],
        *dates,
        *categorical,
    ][:max_columns]
    chosen = set(prioritized)
    return (
        [name for name in numeric if name in chosen],
        [name for name in dates if name in chosen],
        [name for name in categorical if name in chosen],
    )


def aggregate_family(
    raw_dir: Path,
    processed_dir: Path,
    *,
    split: str,
    family: str,
    max_columns: int = 24,
    force: bool = False,
) -> Path:
    """Aggregate one logical family and persist one-row-per-case cache."""
    cache_dir = processed_dir / "agg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{split}_{family}.parquet"
    if output.exists() and not force:
        train_cache = cache_dir / f"train_{family}.parquet"
        if split == "train" or not train_cache.exists():
            return output
        output_columns = set(pl.scan_parquet(output).collect_schema().names())
        train_columns = set(pl.scan_parquet(train_cache).collect_schema().names())
        if output_columns == train_columns:
            return output

    files = _family_files(raw_dir, split, family)
    frame = _scan_family(files).with_columns(
        pl.col(ID_COLUMN).cast(pl.Int64, strict=False)
    )
    reference_files = (
        files
        if split == "train"
        else _family_files(raw_dir, "train", family)
    )
    schema = _scan_family(reference_files).collect_schema()
    numeric, dates, categorical = _selected_columns(
        schema, max_columns=max_columns
    )
    base_dates = scan_base(raw_dir, split).select([ID_COLUMN, "date_decision"])
    is_depth_zero = family.endswith("_0")
    prefix = family.upper()

    if is_depth_zero:
        frame = frame.join(base_dates, on=ID_COLUMN, how="left").with_columns(
            [
                *[
                    pl.col(name).cast(pl.Float32, strict=False).alias(name)
                    for name in numeric
                ],
                *[_date_gap_expression(name).alias(name) for name in dates],
            ]
        )
        expressions = [
            pl.col(name).alias(f"{prefix}__{name}") for name in [*numeric, *dates]
        ]
        expressions.extend(
            pl.col(name).cast(pl.String).alias(f"{prefix}__{name}")
            for name in categorical
        )
        result = frame.select([ID_COLUMN, *expressions]).unique(
            subset=[ID_COLUMN], keep="last"
        )
    else:
        parts_dir = cache_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_paths: list[Path] = []
        for index, path in enumerate(files):
            part_path = parts_dir / f"{split}_{family}_{index}.parquet"
            part_paths.append(part_path)
            if part_path.exists() and not force:
                continue
            part = (
                pl.scan_parquet(path)
                .with_columns(pl.col(ID_COLUMN).cast(pl.Int64, strict=False))
                .join(base_dates, on=ID_COLUMN, how="left")
                .with_columns(
                    [
                        *[
                            pl.col(name)
                            .cast(pl.Float32, strict=False)
                            .alias(name)
                            for name in numeric
                        ],
                        *[_date_gap_expression(name).alias(name) for name in dates],
                    ]
                )
            )
            partial_aggregations: list[pl.Expr] = [
                pl.len().alias("__ROW_COUNT")
            ]
            for name in numeric:
                partial_aggregations.extend(
                    [
                        pl.col(name).sum().cast(pl.Float64).alias(
                            f"{name}__SUM"
                        ),
                        pl.col(name).count().cast(pl.Int64).alias(
                            f"{name}__COUNT"
                        ),
                        pl.col(name).max().cast(pl.Float32).alias(
                            f"{name}__MAX"
                        ),
                    ]
                )
            for name in dates:
                partial_aggregations.extend(
                    [
                        pl.col(name).min().cast(pl.Float32).alias(
                            f"{name}__MIN_GAP"
                        ),
                        pl.col(name).max().cast(pl.Float32).alias(
                            f"{name}__MAX_GAP"
                        ),
                    ]
                )
            partial_aggregations.extend(
                pl.col(name).n_unique().cast(pl.Int32).alias(
                    f"{name}__NUNIQUE"
                )
                for name in categorical
            )
            part.group_by(ID_COLUMN).agg(partial_aggregations).sink_parquet(
                part_path,
                compression="zstd",
                maintain_order=False,
                mkdir=True,
            )

        partials = _scan_family(part_paths)
        final_aggregations: list[pl.Expr] = [
            pl.col("__ROW_COUNT").sum().alias(f"{prefix}__ROW_COUNT")
        ]
        for name in numeric:
            total = pl.col(f"{name}__SUM").sum()
            count = pl.col(f"{name}__COUNT").sum()
            final_aggregations.extend(
                [
                    (total / count).cast(pl.Float32).alias(
                        f"{prefix}__{name}__MEAN"
                    ),
                    pl.col(f"{name}__MAX").max().cast(pl.Float32).alias(
                        f"{prefix}__{name}__MAX"
                    ),
                ]
            )
        for name in dates:
            final_aggregations.extend(
                [
                    pl.col(f"{name}__MIN_GAP").min().cast(pl.Float32).alias(
                        f"{prefix}__{name}__MIN_GAP"
                    ),
                    pl.col(f"{name}__MAX_GAP").max().cast(pl.Float32).alias(
                        f"{prefix}__{name}__MAX_GAP"
                    ),
                ]
            )
        final_aggregations.extend(
            pl.col(f"{name}__NUNIQUE").sum().cast(pl.Float32).alias(
                f"{prefix}__{name}__NUNIQUE"
            )
            for name in categorical
        )
        result = partials.group_by(ID_COLUMN).agg(final_aggregations)

    result.sink_parquet(
        output,
        compression="zstd",
        maintain_order=False,
        mkdir=True,
    )
    check = pl.scan_parquet(output).select(
        pl.len().alias("rows"),
        pl.col(ID_COLUMN).n_unique().alias("unique_ids"),
    ).collect()
    if check["rows"][0] != check["unique_ids"][0]:
        raise AssertionError(f"{output} is not unique by {ID_COLUMN}")
    return output


def build_feature_matrix(
    raw_dir: Path,
    processed_dir: Path,
    *,
    split: str,
    level: str,
    max_columns_per_family: int = 24,
    force: bool = False,
) -> Path:
    """Join cached family aggregates to the base without dropping cases."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    output = processed_dir / f"feature_matrix_{split}_{level}.parquet"
    if output.exists() and not force:
        train_matrix = processed_dir / f"feature_matrix_train_{level}.parquet"
        if split == "train" or not train_matrix.exists():
            return output
        output_columns = set(pl.scan_parquet(output).collect_schema().names())
        train_columns = set(pl.scan_parquet(train_matrix).collect_schema().names())
        train_columns.discard("target")
        if output_columns == train_columns:
            return output
    base = scan_base(raw_dir, split)
    expected_rows = base.select(pl.len()).collect().item()
    matrix = base
    for family in families_for_level(level):
        cache = aggregate_family(
            raw_dir,
            processed_dir,
            split=split,
            family=family,
            max_columns=max_columns_per_family,
            force=force,
        )
        matrix = matrix.join(pl.scan_parquet(cache), on=ID_COLUMN, how="left")
    matrix.sink_parquet(
        output,
        compression="zstd",
        maintain_order=False,
        mkdir=True,
    )
    actual_rows = pl.scan_parquet(output).select(pl.len()).collect().item()
    if actual_rows != expected_rows:
        raise AssertionError(
            f"Feature join changed row count: {expected_rows} -> {actual_rows}"
        )
    return output
