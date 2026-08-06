#!/usr/bin/env python3
"""Validate downloaded HCDR FT-Transformer Kaggle artifacts and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "experiment_config.json",
    "ft_transformer_best.pt",
    "hcdr-ft-transformer-stage-c.log",
    "kernel-metadata.json",
    "metrics.csv",
    "submission_ft_transformer.csv",
    "training_healthcheck.json",
    "training_history.csv",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(input_dir: Path) -> dict[str, object]:
    """Validate result files against the persisted HCDR split and schema."""
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FT-Transformer artifacts: {missing}")

    metadata = json.loads(
        (input_dir / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (input_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    health = json.loads(
        (input_dir / "training_healthcheck.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(input_dir / "metrics.csv")
    history = pd.read_csv(input_dir / "training_history.csv")
    submission = pd.read_csv(input_dir / "submission_ft_transformer.csv")

    membership = pd.read_csv("datasets/processed/hcdr/split_membership.csv")
    matrix = pd.read_parquet(
        "datasets/processed/hcdr/feature_matrix.parquet",
        columns=["SK_ID_CURR", "TARGET"],
    )
    sample = pd.read_csv(
        "datasets/raw/home-credit-default-risk/sample_submission.csv"
    )

    if metadata["id"] != "cyclerlol/hcdr-ft-transformer-stage-c":
        raise ValueError("Unexpected Kaggle kernel id")
    if metadata.get("machine_shape") != "NvidiaTeslaT4":
        raise ValueError("FT-Transformer run was not configured for Tesla T4")
    if metadata.get("dataset_sources") != [
        "cyclerlol/hcdr-stage-c-feature-matrix"
    ]:
        raise ValueError("Unexpected Kaggle dataset source")
    if health.get("status") != "TRAINING_HEALTHCHECK_OK":
        raise ValueError("Training health check did not pass")
    if health.get("device") != "cuda" or not health.get("gradients_finite"):
        raise ValueError("Training health check was not finite CUDA training")

    if set(metrics["model"]) != {"ft_transformer"}:
        raise ValueError("Metrics contain an unexpected model")
    if set(metrics["split"]) != {"valid", "test"}:
        raise ValueError("Metrics must contain valid and test splits")
    split_frames = {
        split: matrix.loc[
            matrix["SK_ID_CURR"].isin(
                membership.loc[membership["split"].eq(split), "SK_ID_CURR"]
            )
            & matrix["TARGET"].notna()
        ]
        for split in ("valid", "test")
    }
    for row in metrics.itertuples(index=False):
        expected = split_frames[row.split]
        if row.n != len(expected):
            raise ValueError(f"{row.split} row count mismatch")
        if not np.isclose(row.bad_rate, expected["TARGET"].mean(), atol=1e-12):
            raise ValueError(f"{row.split} bad-rate mismatch")
        if not (0.5 <= row.auc <= 1.0 and 0.0 <= row.ks <= 1.0):
            raise ValueError(f"{row.split} metric outside valid range")
        if not np.isclose(row.gini, 2.0 * row.auc - 1.0, atol=1e-12):
            raise ValueError(f"{row.split} Gini does not equal 2*AUC-1")

    valid_auc = float(
        metrics.loc[metrics["split"].eq("valid"), "auc"].iloc[0]
    )
    if not np.isclose(config["best_valid_auc"], history["valid_auc"].max()):
        raise ValueError("Config best validation AUC differs from training history")
    if not np.isclose(valid_auc, config["best_valid_auc"]):
        raise ValueError("Persisted validation metric differs from selected checkpoint")
    if config.get("device") != "cuda" or config.get("smoke_test"):
        raise ValueError("Result is not a full CUDA run")

    if list(submission.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Submission schema mismatch")
    if len(submission) != len(sample) or not submission["SK_ID_CURR"].is_unique:
        raise ValueError("Submission row or uniqueness mismatch")
    if not submission["SK_ID_CURR"].astype(int).reset_index(drop=True).equals(
        sample["SK_ID_CURR"].astype(int).reset_index(drop=True)
    ):
        raise ValueError("Submission ID order differs from sample submission")
    if not np.isfinite(submission["TARGET"]).all() or not submission[
        "TARGET"
    ].between(0.0, 1.0).all():
        raise ValueError("Submission probabilities are invalid")

    checkpoint = input_dir / "ft_transformer_best.pt"
    if checkpoint.stat().st_size < 100_000 or not zipfile.is_zipfile(checkpoint):
        raise ValueError("PyTorch checkpoint is missing or malformed")
    log_text = (input_dir / "hcdr-ft-transformer-stage-c.log").read_text(
        encoding="utf-8"
    )
    for marker in ["Tesla T4", "TRAINING_HEALTHCHECK_OK", "EARLY_STOPPING"]:
        if marker not in log_text:
            raise ValueError(f"Kernel log is missing marker: {marker}")

    test_row = metrics.loc[metrics["split"].eq("test")].iloc[0]
    return {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kernel_id": metadata["id"],
        "kernel_id_no": metadata["id_no"],
        "kernel_version": 5,
        "machine_shape": metadata["machine_shape"],
        "dataset_sources": metadata["dataset_sources"],
        "split": config["split"],
        "epochs_completed": int(history["epoch"].max()),
        "best_epoch": int(history.loc[history["valid_auc"].idxmax(), "epoch"]),
        "valid_auc": valid_auc,
        "test_auc": float(test_row["auc"]),
        "test_gini": float(test_row["gini"]),
        "test_ks": float(test_row["ks"]),
        "submission_rows": len(submission),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "files": {
            name: {
                "bytes": (input_dir / name).stat().st_size,
                "sha256": _sha256(input_dir / name),
            }
            for name in REQUIRED_FILES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/hcdr/kaggle_ft_transformer"),
    )
    args = parser.parse_args()
    manifest = validate(args.input_dir)
    manifest_path = args.input_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in [
        "kernel_id", "machine_shape", "epochs_completed", "best_epoch",
        "valid_auc", "test_auc", "test_ks", "submission_rows",
    ]}, indent=2))


if __name__ == "__main__":
    main()
