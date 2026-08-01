#!/usr/bin/env python3
"""Generate and validate HCMS submission files from saved local models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from home_credit_stability.pipeline import _woe_transform

PROCESSED = Path("datasets/processed/hcms/feature_matrix_test_C.parquet")
RAW = Path("datasets/raw/home-credit-model-stability")
OUTPUT = Path("outputs/hcms/submissions")


def _matrix(features: list[str]) -> pd.DataFrame:
    columns = ["case_id", *features]
    frame = pl.read_parquet(PROCESSED, columns=columns).to_pandas()
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(
            "float32"
        )
    return frame


def _validate(
    predictions: pd.DataFrame,
    sample: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    if predictions["case_id"].duplicated().any():
        raise ValueError(f"{model}: duplicate case_id")
    ordered = sample[["case_id"]].merge(
        predictions,
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if len(ordered) != len(sample):
        raise ValueError(f"{model}: row count does not match sample submission")
    if ordered["score"].isna().any():
        raise ValueError(f"{model}: missing prediction after sample-order merge")
    if not np.isfinite(ordered["score"]).all():
        raise ValueError(f"{model}: non-finite prediction")
    if not ordered["score"].between(0.0, 1.0).all():
        raise ValueError(f"{model}: score outside [0, 1]")
    if ordered.columns.tolist() != ["case_id", "score"]:
        raise ValueError(f"{model}: invalid submission columns")
    return ordered


def main() -> None:
    sample = pd.read_csv(RAW / "sample_submission.csv")
    if sample.columns.tolist() != ["case_id", "score"]:
        raise ValueError("Official sample submission must contain case_id,score")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, pd.DataFrame] = {}
    for model_name in ["lightgbm", "logistic_raw", "xgboost"]:
        artifact = joblib.load(OUTPUT.parent / "models" / f"{model_name}.joblib")
        frame = _matrix(artifact["features"])
        transformed = pd.DataFrame(
            artifact["imputer"].transform(frame[artifact["features"]]).astype(
                "float32"
            ),
            columns=artifact["imputer"].get_feature_names_out(
                artifact["features"]
            ),
            index=frame.index,
        )
        if model_name == "logistic_raw":
            values = artifact["scaler"].transform(transformed)
        elif model_name == "xgboost":
            values = transformed[artifact["transformed_features"]]
        else:
            values = transformed
        predictions[model_name] = pd.DataFrame(
            {
                "case_id": frame["case_id"].astype("int64"),
                "score": artifact["model"].predict_proba(values)[:, 1],
            }
        )

    woe_artifact = joblib.load(OUTPUT.parent / "scorecard/logistic_woe.joblib")
    woe_features = list(woe_artifact["bins"])
    woe_frame = _matrix(woe_features)
    transformed_woe = _woe_transform(
        woe_frame,
        woe_artifact["bins"],
        woe_artifact["tables"],
    )
    predictions["logistic_woe"] = pd.DataFrame(
        {
            "case_id": woe_frame["case_id"].astype("int64"),
            "score": woe_artifact["model"].predict_proba(transformed_woe)[:, 1],
        }
    )

    for model_name, frame in predictions.items():
        submission = _validate(frame, sample, model_name)
        path = OUTPUT / f"{model_name}.csv"
        submission.to_csv(path, index=False)
        if model_name == "lightgbm":
            submission.to_csv(OUTPUT / "submission.csv", index=False)
        print(
            f"{model_name}: {len(submission)} rows, "
            f"score=[{submission['score'].min():.8f}, "
            f"{submission['score'].max():.8f}] -> {path}"
        )


if __name__ == "__main__":
    main()
