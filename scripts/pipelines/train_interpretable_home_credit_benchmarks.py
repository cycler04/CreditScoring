#!/usr/bin/env python3
"""Train GAM and monotonic-LightGBM challengers on the saved Home Credit splits."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from home_credit_default_rate.pipeline import _metrics as hcdr_metrics
from home_credit_default_rate.pipeline import _woe_transform as hcdr_woe_transform
from home_credit_stability.pipeline import _metric_row as hcms_metrics
from home_credit_stability.pipeline import _woe_transform as hcms_woe_transform
from home_credit_stability.stability import stability_metric

RANDOM_STATE = 42
MODEL_NAMES = ["gam", "monotonic_lightgbm"]


def _clean_numeric(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].replace([np.inf, -np.inf], np.nan).astype("float32")


def _fit_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    woe: dict[str, object],
    woe_transform: object,
    output_dir: Path,
    metric_builder: object,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, float]]:
    features = list(woe["bins"])
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_dir = output_dir / "feature_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}

    gam = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "splines",
                SplineTransformer(n_knots=4, degree=3, include_bias=False),
            ),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    started = time.perf_counter()
    gam.fit(_clean_numeric(train, features), train[target])
    runtimes["gam"] = time.perf_counter() - started
    joblib.dump({"model": gam, "features": features}, output_dir / "gam.joblib")
    coefficients = np.abs(gam.named_steps["model"].coef_[0])
    width = coefficients.size // len(features)
    pd.DataFrame(
        {
            "feature": features,
            "importance": coefficients.reshape(len(features), width).sum(axis=1),
        }
    ).to_csv(importance_dir / "gam.csv", index=False)

    x_train_woe = woe_transform(train, woe["bins"], woe["tables"])
    x_valid_woe = woe_transform(valid, woe["bins"], woe["tables"])
    monotonic = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=24,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        monotone_constraints=[-1] * len(features),
        monotone_constraints_method="advanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    started = time.perf_counter()
    monotonic.fit(
        x_train_woe,
        train[target],
        eval_X=x_valid_woe,
        eval_y=valid[target],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    runtimes["monotonic_lightgbm"] = time.perf_counter() - started
    joblib.dump(
        {
            "model": monotonic,
            "features": features,
            "bins": woe["bins"],
            "tables": woe["tables"],
            "monotone_constraints": [-1] * len(features),
        },
        output_dir / "monotonic_lightgbm.joblib",
    )
    pd.DataFrame(
        {"feature": features, "importance": monotonic.feature_importances_}
    ).to_csv(importance_dir / "monotonic_lightgbm.csv", index=False)

    for model_name, model in [("gam", gam), ("monotonic_lightgbm", monotonic)]:
        for split_name, frame in [("valid", valid), ("test", test)]:
            values = (
                _clean_numeric(frame, features)
                if model_name == "gam"
                else woe_transform(frame, woe["bins"], woe["tables"])
            )
            scores = model.predict_proba(values)[:, 1]
            metrics.append(metric_builder(model_name, split_name, frame[target], scores))
            if split_name == "test":
                predictions[model_name] = scores
    return metrics, predictions, runtimes


def _load_hcdr_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_parquet("datasets/processed/hcdr/feature_matrix.parquet")
    membership = pd.read_csv("datasets/processed/hcdr/split_membership.csv")
    labeled = matrix.loc[matrix["TARGET"].notna()].merge(
        membership, on="SK_ID_CURR", how="inner", validate="one_to_one"
    )
    labeled["TARGET"] = labeled["TARGET"].astype("int8")
    return tuple(
        labeled.loc[labeled["split"].eq(name)].copy()
        for name in ["train", "valid", "test"]
    )


def train_hcdr() -> None:
    train, valid, test = _load_hcdr_splits()
    output = Path("outputs/hcdr/models")
    woe = joblib.load("outputs/hcdr/scorecard/logistic_woe.joblib")
    metrics, _, runtimes = _fit_models(
        train,
        valid,
        test,
        target="TARGET",
        woe=woe,
        woe_transform=hcdr_woe_transform,
        output_dir=output,
        metric_builder=hcdr_metrics,
    )
    pd.DataFrame(metrics).to_csv(output / "interpretable_metrics.csv", index=False)
    (output / "interpretable_training.json").write_text(
        json.dumps({"rows": {"train": len(train), "valid": len(valid), "test": len(test)}, "fit_seconds": runtimes}, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_hcms_splits(features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads(Path("outputs/hcms/run_summary.json").read_text())
    scan = pl.scan_parquet("datasets/processed/hcms/feature_matrix_train_C.parquet")
    frames = []
    for name in ["train", "valid", "test"]:
        bounds = summary["split_weeks"][name]
        frames.append(
            scan.filter(pl.col("WEEK_NUM").is_between(bounds["min"], bounds["max"]))
            .select(["case_id", "WEEK_NUM", "target", *features])
            .collect()
            .to_pandas()
        )
    return tuple(frames)


def train_hcms() -> None:
    output = Path("outputs/hcms/models")
    woe = joblib.load("outputs/hcms/scorecard/logistic_woe.joblib")
    train, valid, test = _load_hcms_splits(list(woe["bins"]))
    metrics, predictions, runtimes = _fit_models(
        train,
        valid,
        test,
        target="target",
        woe=woe,
        woe_transform=hcms_woe_transform,
        output_dir=output,
        metric_builder=hcms_metrics,
    )
    pd.DataFrame(metrics).to_csv(output / "interpretable_metrics.csv", index=False)
    stability = []
    for model, scores in predictions.items():
        result = stability_metric(test["target"], scores, test["WEEK_NUM"])
        stability.append(
            {
                "model": model,
                "stability": result.stability,
                "mean_gini": result.mean_gini,
                "slope": result.slope,
                "residual_std": result.residual_std,
                "eligible_weeks": len(result.by_week),
                "excluded_weeks": len(result.excluded_weeks),
            }
        )
    (output / "interpretable_stability.json").write_text(
        json.dumps(stability, indent=2) + "\n", encoding="utf-8"
    )
    (output / "interpretable_training.json").write_text(
        json.dumps({"rows": {"train": len(train), "valid": len(valid), "test": len(test)}, "fit_seconds": runtimes}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    train_hcdr()
    train_hcms()
    print("Trained GAM and monotonic LightGBM for HCDR and HCMS.")


if __name__ == "__main__":
    main()
