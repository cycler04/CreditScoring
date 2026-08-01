#!/usr/bin/env python3
"""Extract and fingerprint the Home Credit Default Risk source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

COMPETITION = "home-credit-default-risk"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("datasets/raw/home-credit-default-risk"),
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    archive = raw_dir / f"{COMPETITION}.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing {archive}; download it with the Kaggle competition CLI"
        )

    with zipfile.ZipFile(archive) as source:
        source.extractall(raw_dir)

    files = []
    for path in sorted(raw_dir.glob("*.csv")):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if len(files) != 10:
        raise ValueError(f"Expected 10 CSV files, found {len(files)}")

    manifest = {
        "competition": COMPETITION,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "files": files,
    }
    (raw_dir / "source.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {len(files)} CSV files; wrote {raw_dir / 'source.json'}")


if __name__ == "__main__":
    main()
