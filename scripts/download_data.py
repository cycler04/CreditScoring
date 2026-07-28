#!/usr/bin/env python3
"""Download and extract GiveMeSomeCredit using the token in .env."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
COMPETITION = "GiveMeSomeCredit"
PUBLIC_MIRROR = "brycecf/give-me-some-credit-dataset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GiveMeSomeCredit into datasets/raw."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when all four expected files already exist.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("KAGGLE_API_TOKEN"):
        raise RuntimeError("Thiếu KAGGLE_API_TOKEN trong .env")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    expected_files = [
        RAW_DIR / "Data Dictionary.xls",
        RAW_DIR / "cs-test.csv",
        RAW_DIR / "cs-training.csv",
        RAW_DIR / "sampleEntry.csv",
    ]
    if all(path.exists() for path in expected_files) and not args.force:
        print("Dataset already ready: datasets/raw (4 expected files)")
        return

    api = KaggleApi()
    api.authenticate()
    source = f"competition:{COMPETITION}"
    try:
        api.competition_download_files(
            COMPETITION,
            path=str(RAW_DIR),
            force=args.force,
            quiet=False,
        )
    except HTTPError as error:
        if error.response is None or error.response.status_code != 403:
            raise
        source = f"public-dataset-mirror:{PUBLIC_MIRROR}"
        print(
            "Competition download is forbidden (rules not accepted); "
            f"using Kaggle public mirror {PUBLIC_MIRROR}."
        )
        api.dataset_download_files(
            PUBLIC_MIRROR,
            path=str(RAW_DIR),
            force=args.force,
            quiet=False,
            unzip=True,
        )

    archives = sorted(RAW_DIR.glob("*.zip"))
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(RAW_DIR)
    missing = [path for path in expected_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Download xong nhưng thiếu: {missing}")
    note = (
        "Downloaded directly from the official Kaggle competition."
        if source.startswith("competition:")
        else (
            "The public Kaggle mirror was used because competition download "
            "returned 403 for the current account."
        )
    )
    (RAW_DIR / "source.json").write_text(
        json.dumps(
            {
                "selected_source": source,
                "competition": COMPETITION,
                "public_mirror": PUBLIC_MIRROR,
                "note": note,
                "sha256": {
                    path.name: _sha256(path) for path in expected_files
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Dataset ready: datasets/raw (4 expected files)")
    print(f"Source: {source}")


if __name__ == "__main__":
    main()
