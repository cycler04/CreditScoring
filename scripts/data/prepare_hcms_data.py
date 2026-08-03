#!/usr/bin/env python3
"""Fingerprint and inspect retained Home Credit Model Stability inputs."""

from __future__ import annotations

import json
from pathlib import Path

from home_credit_stability.data import write_source_manifest


def main() -> None:
    raw_dir = Path("datasets/raw/home-credit-model-stability")
    manifest = write_source_manifest(raw_dir)
    print(
        json.dumps(
            {
                "parquet_file_count": manifest["parquet_file_count"],
                "parquet_bytes": manifest["parquet_bytes"],
                "physical_rows_across_tables": manifest[
                    "physical_rows_across_tables"
                ],
                "source": str(raw_dir / "source.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
