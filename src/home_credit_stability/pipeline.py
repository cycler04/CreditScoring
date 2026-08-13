"""Reproducible out-of-time modeling and scorecard experiment."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from credit_scoring.metrics import psi
from credit_scoring.scorecard import (
    bin_by_tree,
    is_monotonic_woe,
    scorecard_from_lr,
    woe_iv,
)
from credit_scoring.visualization import (
    normalize_feature_importance,
    write_bad_rate_by_period_plot,
    write_cutoff_plot,
    write_feature_importance_plot,
    write_gini_by_period_plot,
    write_gini_curve,
    write_ks_curve,
    write_roc_auc_curve,
)

from .aggregate import build_feature_matrix
from .data import ID_COLUMN, TARGET, dataset_inventory
from .split import split_by_week
from .stability import StabilityResult, stability_metric

RANDOM_STATE = 42
ENSEMBLE_MEMBERS = {
    "ensemble_lightgbm_catboost": ("lightgbm", "catboost"),
    "ensemble_lightgbm_xgboost_catboost": (
        "lightgbm",
        "xgboost",
        "catboost",
    ),
    "ensemble_lightgbm_catboost_extra_trees": (
        "lightgbm",
        "catboost",
        "extra_trees",
    ),
    "ensemble_boosting": (
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
    ),
    "ensemble_all_trees": (
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    ),
}
EXCLUDED_FEATURES = {
    ID_COLUMN,
    TARGET,
    "date_decision",
    "WEEK_NUM",
    "MONTH",
}


def _ks(y_true: pd.Series, score: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(y_true, score)
    return float(np.max(true_positive - false_positive))


def _metric_row(
    model: str,
    split: str,
    y_true: pd.Series,
    score: np.ndarray,
    *,
    protocol: str = "out_of_time",
    level: str = "C",
) -> dict[str, object]:
    auc = float(roc_auc_score(y_true, score))
    return {
        "model": model,
        "level": level,
        "protocol": protocol,
        "split": split,
        "n": len(y_true),
        "bad_rate": float(y_true.mean()),
        "auc": auc,
        "brier": float(brier_score_loss(y_true, score)),
        "gini": 2.0 * auc - 1.0,
        "ks": _ks(y_true, score),
    }


def _ordered_submission(
    case_ids: pd.Series,
    scores: np.ndarray,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Align finite model scores to the official sample-submission ID order."""
    if list(sample.columns) != [ID_COLUMN, "score"]:
        raise ValueError("HCMS sample submission must contain case_id,score")
    scored = pd.DataFrame(
        {ID_COLUMN: case_ids.astype(int).to_numpy(), "score": scores}
    )
    if scored[ID_COLUMN].duplicated().any():
        raise ValueError("HCMS competition predictions contain duplicate case_id")
    ordered = sample[[ID_COLUMN]].merge(
        scored,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if ordered["score"].isna().any() or not np.isfinite(ordered["score"]).all():
        raise ValueError("HCMS submission contains missing or non-finite scores")
    if not ordered["score"].between(0.0, 1.0).all():
        raise ValueError("HCMS submission scores must be probabilities in [0, 1]")
    return ordered


def _week_mapping(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, list[int]]]:
    split = split_by_week(frame)
    ranges = {
        name: sorted(frame.loc[split.eq(name), "WEEK_NUM"].unique().astype(int).tolist())
        for name in ["train", "valid", "test"]
    }
    return split, ranges


def _candidate_features(
    matrix_path: Path,
    train_weeks: list[int],
    *,
    per_family: int = 10,
    max_features: int = 160,
) -> list[str]:
    scan = pl.scan_parquet(matrix_path)
    schema = scan.collect_schema()
    numeric = [
        name
        for name, dtype in schema.items()
        if name not in EXCLUDED_FEATURES and (dtype.is_numeric() or dtype == pl.Boolean)
    ]
    if not numeric:
        raise ValueError(f"No numeric features in {matrix_path}")
    availability = (
        scan.filter(pl.col("WEEK_NUM").is_in(train_weeks))
        .select([pl.col(name).count().alias(name) for name in numeric])
        .collect()
        .row(0, named=True)
    )
    families: dict[str, list[str]] = {}
    for name in numeric:
        family = name.split("__", 1)[0]
        families.setdefault(family, []).append(name)
    selected: list[str] = []
    for family in sorted(families):
        available = [
            name for name in families[family] if int(availability[name]) > 0
        ]
        selected.extend(
            sorted(
                available,
                key=lambda name: (-int(availability[name]), name),
            )[:per_family]
        )
    return selected[:max_features]


def _load_matrix(path: Path, features: list[str]) -> pd.DataFrame:
    columns = [
        ID_COLUMN,
        "WEEK_NUM",
        "MONTH",
        "date_decision",
        *([TARGET] if "train" in path.name else []),
        *features,
    ]
    frame = pl.read_parquet(path, columns=columns).to_pandas()
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(
            "float32"
        )
    return frame


def _write_stability(
    result: StabilityResult,
    model: str,
    output_dir: Path,
) -> dict[str, object]:
    by_week = result.by_week.copy()
    by_week.insert(0, "model", model)
    excluded = result.excluded_weeks.copy()
    if not excluded.empty:
        excluded.insert(0, "model", model)
    return {
        "summary": {
            "model": model,
            "stability": result.stability,
            "mean_gini": result.mean_gini,
            "slope": result.slope,
            "residual_std": result.residual_std,
            "eligible_weeks": len(result.by_week),
            "excluded_weeks": len(result.excluded_weeks),
        },
        "by_week": by_week,
        "excluded": excluded,
    }


def _fit_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
        device_type="gpu",
        max_bin=63,
    )
    model.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model


def _prepare_numeric(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    competition: pd.DataFrame,
    features: list[str],
) -> tuple[SimpleImputer, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(train[features]).astype("float32")
    names = imputer.get_feature_names_out(features)
    return (
        imputer,
        pd.DataFrame(x_train, columns=names, index=train.index),
        pd.DataFrame(
            imputer.transform(valid[features]).astype("float32"),
            columns=names,
            index=valid.index,
        ),
        pd.DataFrame(
            imputer.transform(test[features]).astype("float32"),
            columns=names,
            index=test.index,
        ),
        pd.DataFrame(
            imputer.transform(competition[features]).astype("float32"),
            columns=names,
            index=competition.index,
        ),
    )


def _equal_weight_ensembles(
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Blend predefined probabilities without tuning on future test weeks."""
    if not predictions:
        raise ValueError("Base predictions cannot be empty")
    splits = tuple(next(iter(predictions.values())))
    ensembles: dict[str, dict[str, np.ndarray]] = {}
    for ensemble_name, members in ENSEMBLE_MEMBERS.items():
        missing = set(members).difference(predictions)
        if missing:
            raise ValueError(
                f"{ensemble_name} is missing base predictions: {sorted(missing)}"
            )
        ensembles[ensemble_name] = {
            split: np.mean(
                [predictions[member][split] for member in members], axis=0
            )
            for split in splits
        }
    return ensembles


def _write_importance(
    model_name: str,
    feature_names: list[str],
    importance: np.ndarray,
    output_dir: Path,
) -> None:
    table = normalize_feature_importance(pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    )).sort_values("importance", ascending=False)
    importance_dir = output_dir / "feature_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(importance_dir / f"{model_name}.csv", index=False)
    write_feature_importance_plot(
        table,
        importance_dir / f"{model_name}.png",
        title=f"{model_name} — top feature importance",
    )


def _fit_additional_tree_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    x_competition: pd.DataFrame,
    benchmark_features: list[str],
    raw_features: list[str],
    imputer: SimpleImputer,
    output_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    """Fit bounded CPU tree benchmarks on the full transformed feature set."""
    split_values = {
        "train": x_train[benchmark_features],
        "valid": x_valid[benchmark_features],
        "test": x_test[benchmark_features],
        "competition": x_competition[benchmark_features],
    }
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=100,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=100,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    devices: dict[str, str] = {}
    for model_name, model in models.items():
        model.fit(split_values["train"], y_train)
        predictions[model_name] = {
            split: model.predict_proba(values)[:, 1]
            for split, values in split_values.items()
        }
        if hasattr(model, "feature_importances_"):
            _write_importance(
                model_name,
                benchmark_features,
                np.asarray(model.feature_importances_),
                output_dir,
            )
        joblib.dump(
            {
                "features": raw_features,
                "transformed_features": benchmark_features,
                "imputer": imputer,
                "model": model,
                "device": "cpu",
            },
            output_dir / f"{model_name}.joblib",
        )
        devices[model_name] = "cpu"

    catboost_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    catboost_model.fit(
        split_values["train"],
        y_train,
        eval_set=(split_values["valid"], y_valid),
        early_stopping_rounds=60,
        verbose=False,
    )
    predictions["catboost"] = {
        split: catboost_model.predict_proba(values)[:, 1]
        for split, values in split_values.items()
    }
    _write_importance(
        "catboost",
        benchmark_features,
        np.asarray(catboost_model.feature_importances_),
        output_dir,
    )
    joblib.dump(
        {
            "features": raw_features,
            "transformed_features": benchmark_features,
            "imputer": imputer,
            "model": catboost_model,
            "device": "cpu",
        },
        output_dir / "catboost.joblib",
    )
    devices["catboost"] = "cpu"
    return predictions, devices


def _woe_transform(
    frame: pd.DataFrame,
    bins: dict[str, np.ndarray],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    for feature, edges in bins.items():
        labels = (
            pd.cut(frame[feature], edges, include_lowest=True)
            .astype("string")
            .fillna("MISSING")
        )
        mapping = tables[feature].set_index("bin")["woe"]
        fallback = float(mapping.get("MISSING", 0.0))
        result[feature] = labels.map(mapping).fillna(fallback).astype(float)
    return pd.DataFrame(result, index=frame.index)


def _weekly_iv(
    frame: pd.DataFrame,
    feature: str,
    edges: np.ndarray,
) -> pd.DataFrame:
    bins = (
        pd.cut(frame[feature], edges, include_lowest=True)
        .astype("string")
        .fillna("MISSING")
    )
    grouped = (
        pd.DataFrame(
            {
                "WEEK_NUM": frame["WEEK_NUM"].to_numpy(),
                "bin": bins,
                TARGET: frame[TARGET].to_numpy(),
            }
        )
        .groupby(["WEEK_NUM", "bin"], observed=True)[TARGET]
        .agg(total="size", bad="sum")
        .reset_index()
    )
    grouped["good"] = grouped["total"] - grouped["bad"]
    rows = []
    for week, table in grouped.groupby("WEEK_NUM", sort=True):
        n_bins = len(table)
        dist_good = (table["good"] + 0.5) / (table["good"].sum() + 0.5 * n_bins)
        dist_bad = (table["bad"] + 0.5) / (table["bad"].sum() + 0.5 * n_bins)
        iv = float(((dist_good - dist_bad) * np.log(dist_good / dist_bad)).sum())
        rows.append({"feature": feature, "WEEK_NUM": int(week), "iv": iv})
    return pd.DataFrame(rows)


def _fit_woe_scorecard(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    competition: pd.DataFrame,
    candidate_features: list[str],
    output_dir: Path,
) -> tuple[
    LogisticRegression,
    dict[str, np.ndarray],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    bins: dict[str, np.ndarray] = {}
    tables: dict[str, pd.DataFrame] = {}
    weekly_tables: list[pd.DataFrame] = []
    candidates: list[tuple[str, float]] = []
    for feature in candidate_features[:20]:
        edges = bin_by_tree(train[feature], train[TARGET])
        table, iv = woe_iv(train, feature, TARGET, bins=edges)
        weekly = _weekly_iv(train, feature, edges)
        mean_iv = float(weekly["iv"].mean())
        cv = float(weekly["iv"].std(ddof=0) / max(mean_iv, 1e-8))
        weekly["iv_cv"] = cv
        weekly_tables.append(weekly)
        if iv >= 0.02 and cv <= 1.0 and is_monotonic_woe(table):
            candidates.append((feature, iv))
            bins[feature] = edges
            tables[feature] = table
    selected = [name for name, _ in sorted(candidates, key=lambda item: -item[1])[:15]]
    if len(selected) < 2:
        raise ValueError("Fewer than two stable monotonic WoE features passed")
    bins = {name: bins[name] for name in selected}
    tables = {name: tables[name] for name in selected}
    transformed = {
        "train": _woe_transform(train, bins, tables),
        "valid": _woe_transform(valid, bins, tables),
        "test": _woe_transform(test, bins, tables),
        "competition": _woe_transform(competition, bins, tables),
    }
    while True:
        probe = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        probe.fit(transformed["train"], train[TARGET])
        wrong = [
            feature
            for feature, coefficient in zip(
                transformed["train"].columns, probe.coef_[0], strict=True
            )
            if coefficient >= 0
        ]
        if not wrong:
            break
        if len(transformed["train"].columns) - len(wrong) < 2:
            raise ValueError("Sign filtering left fewer than two WoE features")
        for key in transformed:
            transformed[key] = transformed[key].drop(columns=wrong)
        for feature in wrong:
            bins.pop(feature)
            tables.pop(feature)
    model, scorecard = scorecard_from_lr(
        transformed["train"],
        train[TARGET],
        tables,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables.values(), ignore_index=True).to_csv(
        output_dir / "woe_iv_detail.csv", index=False
    )
    pd.concat(weekly_tables, ignore_index=True).to_csv(
        output_dir / "iv_by_week.csv", index=False
    )
    scorecard.to_csv(output_dir / "scorecard.csv", index=False)
    pd.DataFrame(
        {
            "feature": transformed["train"].columns,
            "coefficient": model.coef_[0],
        }
    ).to_csv(output_dir / "coefficients.csv", index=False)
    (output_dir / "bin_edges.json").write_text(
        json.dumps(
            {name: edges.tolist() for name, edges in bins.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {"model": model, "bins": bins, "tables": tables},
        output_dir / "logistic_woe.joblib",
    )
    return model, bins, tables, transformed


def _score_from_scorecard(
    transformed: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> np.ndarray:
    points = {
        feature: table.set_index("bin")["points"]
        for feature, table in scorecard.groupby("feature")
    }
    total = np.zeros(len(transformed), dtype=float)
    # WoE values uniquely identify scorecard bins after fit.
    for feature in transformed:
        table = scorecard.loc[scorecard["feature"].eq(feature)]
        mapping = table.groupby("woe")["points"].first()
        total += transformed[feature].map(mapping).fillna(mapping.min()).to_numpy()
    return total


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    *,
    max_columns_per_family: int = 24,
) -> dict[str, object]:
    """Build all A/B/C matrices, fit baselines, and write audit artifacts."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "eda",
        "models",
        "models/metrics",
        "scorecard",
        "stability",
        "submissions",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    (output_dir / "models/feature_importance").mkdir(
        parents=True, exist_ok=True
    )

    base = pl.read_parquet(raw_dir / "parquet_files/train/train_base.parquet").to_pandas()
    split, split_weeks = _week_mapping(base)
    membership = base[[ID_COLUMN, "WEEK_NUM"]].assign(split=split)
    membership.to_csv(processed_dir / "split_membership.csv", index=False)
    bad_rate_week = (
        base.groupby("WEEK_NUM")[TARGET]
        .agg(n="size", bad_count="sum", bad_rate="mean")
        .reset_index()
    )
    bad_rate_week["split"] = bad_rate_week["WEEK_NUM"].map(
        {
            week: name
            for name, weeks in split_weeks.items()
            for week in weeks
        }
    )
    bad_rate_week.to_csv(output_dir / "eda/bad_rate_by_week.csv", index=False)
    write_bad_rate_by_period_plot(
        bad_rate_week,
        output_dir / "eda/bad_rate_by_week.png",
        period_column="WEEK_NUM",
        group_column="split",
        title="Home Credit Model Stability — bad rate by week",
    )
    inventory = dataset_inventory(raw_dir)
    pd.DataFrame(inventory["files"]).to_csv(
        output_dir / "eda/dataset_inventory.csv", index=False
    )

    stage_rows: list[dict[str, object]] = []
    stage_stability_rows: list[dict[str, object]] = []
    stage_gini_parts: list[pd.DataFrame] = []
    final: dict[str, object] | None = None
    final_frame: pd.DataFrame | None = None
    week_to_split = {
        week: name for name, weeks in split_weeks.items() for week in weeks
    }
    for level in "ABC":
        train_path = build_feature_matrix(
            raw_dir,
            processed_dir,
            split="train",
            level=level,
            max_columns_per_family=max_columns_per_family,
        )
        competition_path = build_feature_matrix(
            raw_dir,
            processed_dir,
            split="test",
            level=level,
            max_columns_per_family=max_columns_per_family,
        )
        features = _candidate_features(train_path, split_weeks["train"])
        frame = _load_matrix(train_path, features)
        competition = _load_matrix(competition_path, features)
        split_column = frame["WEEK_NUM"].map(week_to_split).astype("string")
        split_column.name = "split"
        frame = pd.concat([frame, split_column], axis=1).copy()
        if frame["split"].isna().any():
            raise AssertionError("Every feature-matrix row must map to a split")
        train = frame.loc[frame["split"].eq("train")].copy()
        valid = frame.loc[frame["split"].eq("valid")].copy()
        test = frame.loc[frame["split"].eq("test")].copy()
        imputer, x_train, x_valid, x_test, x_competition = _prepare_numeric(
            train, valid, test, competition, features
        )
        model = _fit_lightgbm(
            x_train, train[TARGET], x_valid, valid[TARGET]
        )
        stage_scores: dict[str, np.ndarray] = {}
        for name, values, labels in [
            ("train", x_train, train[TARGET]),
            ("valid", x_valid, valid[TARGET]),
            ("test", x_test, test[TARGET]),
        ]:
            scores = model.predict_proba(values)[:, 1]
            stage_scores[name] = scores
            stage_rows.append(
                _metric_row(
                    "lightgbm",
                    name,
                    labels,
                    scores,
                    level=level,
                )
            )
        stage_result = stability_metric(
            test[TARGET],
            stage_scores["test"],
            test["WEEK_NUM"],
        )
        stage_stability_rows.append(
            {
                "level": level,
                "stability": stage_result.stability,
                "mean_gini": stage_result.mean_gini,
                "slope": stage_result.slope,
                "residual_std": stage_result.residual_std,
                "eligible_weeks": len(stage_result.by_week),
                "excluded_weeks": len(stage_result.excluded_weeks),
            }
        )
        stage_by_week = stage_result.by_week.copy()
        stage_by_week.insert(0, "level", level)
        stage_gini_parts.append(stage_by_week)
        importance_table = normalize_feature_importance(pd.DataFrame(
            {"feature": x_train.columns, "importance": model.feature_importances_}
        ))
        importance_table = importance_table.sort_values(
            "importance", ascending=False
        )
        importance_path = (
            output_dir / f"models/feature_importance/lightgbm_{level}"
        )
        importance_table.to_csv(importance_path.with_suffix(".csv"), index=False)
        write_feature_importance_plot(
            importance_table,
            importance_path.with_suffix(".png"),
            title=f"lightgbm_{level} — top feature importance",
        )
        if level == "C":
            final = {
                "model": model,
                "imputer": imputer,
                "features": features,
                "x_train": x_train,
                "x_valid": x_valid,
                "x_test": x_test,
                "x_competition": x_competition,
                "train": train,
                "valid": valid,
                "test": test,
                "competition": competition,
            }
            final_frame = frame
        else:
            del (
                frame,
                competition,
                train,
                valid,
                test,
                imputer,
                x_train,
                x_valid,
                x_test,
                x_competition,
                model,
            )
            gc.collect()
    pd.DataFrame(stage_rows).to_csv(
        output_dir / "models/stage_metrics.csv", index=False
    )
    pd.DataFrame(stage_stability_rows).to_csv(
        output_dir / "stability/stage_stability.csv", index=False
    )
    pd.concat(stage_gini_parts, ignore_index=True).to_csv(
        output_dir / "stability/stage_gini_by_week.csv", index=False
    )
    write_gini_by_period_plot(
        pd.concat(stage_gini_parts, ignore_index=True),
        output_dir / "stability/stage_gini_by_week.png",
        period_column="WEEK_NUM",
        group_column="level",
        title="Stage A/B/C LightGBM Gini by week",
    )

    if final is None or final_frame is None:
        raise AssertionError("Stage C did not produce a final modeling frame")
    canonical_matrix = processed_dir / "feature_matrix.parquet"
    canonical_matrix.unlink(missing_ok=True)
    canonical_matrix.hardlink_to(
        processed_dir / "feature_matrix_train_C.parquet"
    )
    train = final["train"]
    valid = final["valid"]
    test = final["test"]
    competition = final["competition"]
    x_train = final["x_train"]
    x_valid = final["x_valid"]
    x_test = final["x_test"]
    x_competition = final["x_competition"]
    metrics = [
        row
        for row in stage_rows
        if row["level"] == "C" and row["model"] == "lightgbm"
    ]
    predictions: dict[str, dict[str, np.ndarray]] = {}
    lightgbm_model = final["model"]
    predictions["lightgbm"] = {
        "train": lightgbm_model.predict_proba(x_train)[:, 1],
        "valid": lightgbm_model.predict_proba(x_valid)[:, 1],
        "test": lightgbm_model.predict_proba(x_test)[:, 1],
        "competition": lightgbm_model.predict_proba(x_competition)[:, 1],
    }
    joblib.dump(
        {
            "features": final["features"],
            "imputer": final["imputer"],
            "model": lightgbm_model,
        },
        output_dir / "models/lightgbm.joblib",
    )

    scaler = StandardScaler(with_mean=False)
    scaled_train = scaler.fit_transform(x_train)
    logistic = LogisticRegression(
        solver="saga",
        max_iter=300,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )
    logistic.fit(scaled_train, train[TARGET])
    predictions["logistic_raw"] = {
        "train": logistic.predict_proba(scaled_train)[:, 1],
        "valid": logistic.predict_proba(scaler.transform(x_valid))[:, 1],
        "test": logistic.predict_proba(scaler.transform(x_test))[:, 1],
        "competition": logistic.predict_proba(
            scaler.transform(x_competition)
        )[:, 1],
    }
    joblib.dump(
        {
            "features": final["features"],
            "imputer": final["imputer"],
            "scaler": scaler,
            "model": logistic,
        },
        output_dir / "models/logistic_raw.joblib",
    )
    logistic_importance = normalize_feature_importance(pd.DataFrame(
        {
            "feature": x_train.columns,
            "coefficient": logistic.coef_[0],
            "importance": np.abs(logistic.coef_[0]),
        }
    )).sort_values("importance", ascending=False)
    logistic_importance.to_csv(
        output_dir / "models/feature_importance/logistic_raw.csv", index=False
    )
    write_feature_importance_plot(
        logistic_importance,
        output_dir / "models/feature_importance/logistic_raw.png",
        title="logistic_raw — absolute coefficient importance",
    )

    xgboost_features = list(x_train.columns)
    xgboost_parameters = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "max_depth": 5,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "max_bin": 64,
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 50,
    }
    xgboost_model = xgb.XGBClassifier(
        **xgboost_parameters,
        device="cuda",
    )
    xgboost_device = "cuda"
    try:
        xgboost_model.fit(
            x_train[xgboost_features],
            train[TARGET],
            eval_set=[(x_valid[xgboost_features], valid[TARGET])],
            verbose=False,
        )
    except xgb.core.XGBoostError as error:
        if "not compiled for SM" not in str(error):
            raise
        xgboost_device = "cpu_fallback_sm61_unsupported"
        xgboost_model = xgb.XGBClassifier(
            **xgboost_parameters,
            device="cpu",
            n_jobs=4,
        )
        xgboost_model.fit(
            x_train[xgboost_features],
            train[TARGET],
            eval_set=[(x_valid[xgboost_features], valid[TARGET])],
            verbose=False,
        )
    predictions["xgboost"] = {
        "train": xgboost_model.predict_proba(x_train[xgboost_features])[:, 1],
        "valid": xgboost_model.predict_proba(x_valid[xgboost_features])[:, 1],
        "test": xgboost_model.predict_proba(x_test[xgboost_features])[:, 1],
        "competition": xgboost_model.predict_proba(
            x_competition[xgboost_features]
        )[:, 1],
    }
    joblib.dump(
        {
            "features": final["features"],
            "transformed_features": xgboost_features,
            "imputer": final["imputer"],
            "model": xgboost_model,
            "device": xgboost_device,
        },
        output_dir / "models/xgboost.joblib",
    )
    xgboost_importance = normalize_feature_importance(pd.DataFrame(
        {
            "feature": xgboost_features,
            "importance": xgboost_model.feature_importances_,
        }
    )).sort_values("importance", ascending=False)
    xgboost_importance.to_csv(
        output_dir / "models/feature_importance/xgboost.csv", index=False
    )
    write_feature_importance_plot(
        xgboost_importance,
        output_dir / "models/feature_importance/xgboost.png",
        title="xgboost — top feature importance",
    )

    additional_predictions, additional_devices = _fit_additional_tree_models(
        x_train,
        train[TARGET],
        x_valid,
        valid[TARGET],
        x_test,
        x_competition,
        xgboost_features,
        final["features"],
        final["imputer"],
        output_dir / "models",
    )
    predictions.update(additional_predictions)

    importance_order = (
        pd.DataFrame(
            {
                "feature": x_train.columns,
                "importance": lightgbm_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)["feature"]
        .tolist()
    )
    original_candidates = [
        name for name in importance_order if name in final["features"]
    ]
    woe_model, bins, tables, transformed = _fit_woe_scorecard(
        train,
        valid,
        test,
        competition,
        original_candidates,
        output_dir / "scorecard",
    )
    predictions["logistic_woe"] = {
        name: woe_model.predict_proba(values)[:, 1]
        for name, values in transformed.items()
    }
    ensemble_predictions = _equal_weight_ensembles(predictions)
    predictions.update(ensemble_predictions)
    woe_importance = normalize_feature_importance(pd.DataFrame(
        {
            "feature": transformed["train"].columns,
            "coefficient": woe_model.coef_[0],
            "importance": np.abs(woe_model.coef_[0]),
        }
    )).sort_values("importance", ascending=False)
    woe_importance.to_csv(
        output_dir / "models/feature_importance/logistic_woe.csv", index=False
    )
    write_feature_importance_plot(
        woe_importance,
        output_dir / "models/feature_importance/logistic_woe.png",
        title="logistic_woe — absolute coefficient importance",
    )

    for model_name, model_predictions in predictions.items():
        if model_name != "lightgbm":
            for split_name, values, labels in [
                ("train", model_predictions["train"], train[TARGET]),
                ("valid", model_predictions["valid"], valid[TARGET]),
                ("test", model_predictions["test"], test[TARGET]),
            ]:
                metrics.append(
                    _metric_row(model_name, split_name, labels, values)
                )

    test_predictions = {
        model_name: model_predictions["test"]
        for model_name, model_predictions in predictions.items()
    }
    metrics_dir = output_dir / "models/metrics"
    write_roc_auc_curve(
        test[TARGET],
        test_predictions,
        metrics_dir / "roc_auc_curve.png",
        split_label="out-of-time test split",
    )
    write_gini_curve(
        test[TARGET],
        test_predictions,
        metrics_dir / "gini_curve.png",
        split_label="out-of-time test split",
    )
    write_ks_curve(
        test[TARGET],
        test_predictions,
        metrics_dir / "ks_curve.png",
        split_label="out-of-time test split",
    )

    stability_parts = []
    stability_summaries = []
    excluded_parts = []
    for model_name, model_predictions in predictions.items():
        result = stability_metric(
            test[TARGET],
            model_predictions["test"],
            test["WEEK_NUM"],
        )
        written = _write_stability(
            result, model_name, output_dir / "stability"
        )
        stability_parts.append(written["by_week"])
        stability_summaries.append(written["summary"])
        if not written["excluded"].empty:
            excluded_parts.append(written["excluded"])
    gini_by_week = pd.concat(stability_parts, ignore_index=True)
    gini_by_week.to_csv(
        output_dir / "stability/gini_by_week.csv", index=False
    )
    write_gini_by_period_plot(
        gini_by_week,
        output_dir / "stability/gini_by_week.png",
        period_column="WEEK_NUM",
        group_column="model",
        title="Out-of-time test Gini by week",
    )
    if excluded_parts:
        pd.concat(excluded_parts, ignore_index=True).to_csv(
            output_dir / "stability/excluded_weeks.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=["model", "WEEK_NUM", "n", "n_bad", "n_good", "reason"]
        ).to_csv(output_dir / "stability/excluded_weeks.csv", index=False)
    (output_dir / "stability/stability_metric.json").write_text(
        json.dumps(stability_summaries, indent=2) + "\n",
        encoding="utf-8",
    )

    all_indices = np.arange(len(final_frame))
    random_train, random_holdout = train_test_split(
        all_indices,
        train_size=len(train),
        test_size=len(valid) + len(test),
        stratify=final_frame[TARGET],
        random_state=RANDOM_STATE,
    )
    random_valid, random_test = train_test_split(
        random_holdout,
        train_size=len(valid),
        test_size=len(test),
        stratify=final_frame.iloc[random_holdout][TARGET],
        random_state=RANDOM_STATE,
    )
    full_x = pd.concat([x_train, x_valid, x_test]).sort_index()
    random_model = _fit_lightgbm(
        full_x.loc[random_train],
        final_frame.loc[random_train, TARGET],
        full_x.loc[random_valid],
        final_frame.loc[random_valid, TARGET],
    )
    random_score = random_model.predict_proba(full_x.loc[random_test])[:, 1]
    protocol_rows = [
        next(
            row
            for row in metrics
            if row["model"] == "lightgbm"
            and row["protocol"] == "out_of_time"
            and row["split"] == "test"
        ),
        _metric_row(
            "lightgbm",
            "test",
            final_frame.loc[random_test, TARGET],
            random_score,
            protocol="stratified_random",
        ),
    ]
    pd.DataFrame(protocol_rows).to_csv(
        output_dir / "models/split_protocol_comparison.csv", index=False
    )

    train_weeks = split_weeks["train"]
    recent_weeks = train_weeks[len(train_weeks) // 2 :]
    recent_mask = train["WEEK_NUM"].isin(recent_weeks)
    recent_model = _fit_lightgbm(
        x_train.loc[recent_mask],
        train.loc[recent_mask, TARGET],
        x_valid,
        valid[TARGET],
    )
    recent_score = recent_model.predict_proba(x_test)[:, 1]
    recent_result = stability_metric(
        test[TARGET], recent_score, test["WEEK_NUM"]
    )
    pd.DataFrame(
        [
            {
                "training_window": "all_train_weeks",
                **next(
                    item
                    for item in stability_summaries
                    if item["model"] == "lightgbm"
                ),
            },
            {
                "training_window": "recent_half_train_weeks",
                "model": "lightgbm_recent",
                "stability": recent_result.stability,
                "mean_gini": recent_result.mean_gini,
                "slope": recent_result.slope,
                "residual_std": recent_result.residual_std,
                "eligible_weeks": len(recent_result.by_week),
                "excluded_weeks": len(recent_result.excluded_weeks),
            },
        ]
    ).to_csv(output_dir / "stability/training_window_comparison.csv", index=False)

    scorecard = pd.read_csv(output_dir / "scorecard/scorecard.csv")
    train_scores = _score_from_scorecard(transformed["train"], scorecard)
    valid_scores = _score_from_scorecard(transformed["valid"], scorecard)
    test_scores = _score_from_scorecard(transformed["test"], scorecard)
    cutoff_rows = []
    cutoff_week_rows = []
    for approval_target in [0.6, 0.7, 0.8]:
        cutoff = float(np.quantile(valid_scores, 1.0 - approval_target))
        approved = test_scores >= cutoff
        cutoff_rows.append(
            {
                "approval_target": approval_target,
                "cutoff": cutoff,
                "approval_rate": float(approved.mean()),
                "approved_bad_rate": float(test.loc[approved, TARGET].mean()),
            }
        )
        for week, indices in test.groupby("WEEK_NUM").groups.items():
            positions = test.index.get_indexer(indices)
            week_approved = test_scores[positions] >= cutoff
            cutoff_week_rows.append(
                {
                    "approval_target": approval_target,
                    "WEEK_NUM": int(week),
                    "n": len(indices),
                    "approval_rate": float(week_approved.mean()),
                    "approved_bad_rate": float(
                        test.loc[indices, TARGET][week_approved].mean()
                    ),
                }
            )
    cutoff_table = pd.DataFrame(cutoff_rows)
    cutoff_table.to_csv(
        output_dir / "scorecard/cutoffs.csv", index=False
    )
    write_cutoff_plot(
        cutoff_table,
        output_dir / "scorecard/approval_bad_rate.png",
        target_column="approval_target",
        actual_column="approval_rate",
        bad_rate_column="approved_bad_rate",
    )
    pd.DataFrame(cutoff_week_rows).to_csv(
        output_dir / "scorecard/cutoffs_by_week.csv", index=False
    )
    psi_rows = []
    for week, indices in test.groupby("WEEK_NUM").groups.items():
        positions = test.index.get_indexer(indices)
        value, _ = psi(train_scores, test_scores[positions])
        psi_rows.append({"WEEK_NUM": int(week), "score_psi": value})
    pd.DataFrame(psi_rows).to_csv(
        output_dir / "scorecard/score_psi_by_week.csv", index=False
    )

    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(output_dir / "models/metrics.csv", index=False)
    metrics_table.to_csv(
        output_dir / "models/metrics/metrics.csv", index=False
    )
    sample_submission = pd.read_csv(raw_dir / "sample_submission.csv")
    for model_name, model_predictions in predictions.items():
        submission = _ordered_submission(
            competition[ID_COLUMN],
            model_predictions["competition"],
            sample_submission,
        )
        submission.to_csv(
            output_dir / f"submissions/{model_name}.csv", index=False
        )
    _ordered_submission(
        competition[ID_COLUMN],
        predictions["lightgbm"]["competition"],
        sample_submission,
    ).to_csv(output_dir / "submissions/submission.csv", index=False)
    run_summary = {
        "dataset": {
            key: value for key, value in inventory.items() if key != "files"
        },
        "train_base_rows": len(base),
        "bad_rate": float(base[TARGET].mean()),
        "distinct_weeks": int(base["WEEK_NUM"].nunique()),
        "split_weeks": {
            name: {
                "count": len(weeks),
                "min": min(weeks),
                "max": max(weeks),
            }
            for name, weeks in split_weeks.items()
        },
        "split_rows": split.value_counts().to_dict(),
        "selected_feature_count_C": len(final["features"]),
        "transformed_feature_count_C": len(x_train.columns),
        "max_columns_per_family": max_columns_per_family,
        "random_state": RANDOM_STATE,
        "devices": {
            "lightgbm": "gpu",
            "xgboost": xgboost_device,
            "logistic_raw": "cpu",
            "logistic_woe": "cpu",
            **additional_devices,
        },
        "benchmark_protocol": (
            "All tree models use the full set of transformed features (244 columns) "
            "and out-of-time week split; ensembles use fixed equal weights without "
            "test-week tuning."
        ),
        "ensemble_members": ENSEMBLE_MEMBERS,
        "stage_metrics": stage_rows,
        "stage_stability": stage_stability_rows,
        "metrics": metrics,
        "stability": stability_summaries,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_summary
