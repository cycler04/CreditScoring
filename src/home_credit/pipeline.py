"""Reproducible staged modeling pipeline for Home Credit Default Risk."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.metrics import psi
from credit_scoring.scorecard import (
    bin_by_tree,
    is_monotonic_woe,
    scorecard_from_lr,
    woe_iv,
)

from .aggregate import build_feature_matrix
from .data import ID_COLUMN, TARGET, clean_application, load_tables

RANDOM_STATE = 42


def split_application(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the documented stratified random 60/20/20 split."""
    train, holdout = train_test_split(
        frame,
        test_size=0.4,
        stratify=frame[TARGET],
        random_state=RANDOM_STATE,
    )
    valid, test = train_test_split(
        holdout,
        test_size=0.5,
        stratify=holdout[TARGET],
        random_state=RANDOM_STATE,
    )
    return train.copy(), valid.copy(), test.copy()


def _ks(y_true: pd.Series, score: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(y_true, score)
    return float(np.max(true_positive - false_positive))


def _metrics(
    model: str, split: str, y_true: pd.Series, score: np.ndarray
) -> dict[str, object]:
    auc = float(roc_auc_score(y_true, score))
    return {
        "model": model,
        "split": split,
        "n": len(y_true),
        "bad_rate": float(y_true.mean()),
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": _ks(y_true, score),
    }


def _write_eda(
    train: pd.DataFrame,
    findings: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    findings.to_csv(output_dir / "anomaly_findings.csv", index=False)
    pd.DataFrame(
        {
            "feature": train.columns,
            "dtype": train.dtypes.astype(str).to_numpy(),
            "missing_count": train.isna().sum().to_numpy(),
            "missing_rate": train.isna().mean().to_numpy(),
            "nunique": train.nunique(dropna=True).to_numpy(),
        }
    ).sort_values("missing_rate", ascending=False).to_csv(
        output_dir / "column_profile.csv", index=False
    )
    categorical = train.select_dtypes(include=["object", "string"]).columns
    rows = []
    for column in categorical:
        counts = train[column].astype("string").fillna("MISSING").value_counts()
        for value, count in counts.items():
            selected = train[column].astype("string").fillna("MISSING").eq(value)
            rows.append(
                {
                    "feature": column,
                    "level": value,
                    "count": int(count),
                    "rate": float(count / len(train)),
                    "bad_rate": float(train.loc[selected, TARGET].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "categorical_bad_rates.csv", index=False
    )


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(include=["object", "string", "category"]).columns
    numeric = frame.columns.difference(categorical)
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="MISSING",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=0.01,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _fit_raw_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    competition_test: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    x_train, y_train = train[feature_columns], train[TARGET]
    x_valid, y_valid = valid[feature_columns], valid[TARGET]
    x_test, y_test = test[feature_columns], test[TARGET]
    processor = _preprocessor(x_train)
    transformed_train = processor.fit_transform(x_train)
    transformed_valid = processor.transform(x_valid)
    transformed_test = processor.transform(x_test)
    transformed_competition = processor.transform(competition_test[feature_columns])

    models = {
        "logistic_raw": LogisticRegression(
            solver="saga",
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=32,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=6,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    rows: list[dict[str, object]] = []
    competition_scores: dict[str, np.ndarray] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if name == "lightgbm":
            model.fit(
                transformed_train,
                y_train,
                eval_X=transformed_valid,
                eval_y=y_valid,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
        elif name == "xgboost":
            model.set_params(early_stopping_rounds=100)
            model.fit(
                transformed_train,
                y_train,
                eval_set=[(transformed_valid, y_valid)],
                verbose=False,
            )
        else:
            model.fit(transformed_train, y_train)
        for split_name, values, labels in [
            ("valid", transformed_valid, y_valid),
            ("test", transformed_test, y_test),
        ]:
            rows.append(
                _metrics(
                    name,
                    split_name,
                    labels,
                    model.predict_proba(values)[:, 1],
                )
            )
        competition_scores[name] = model.predict_proba(transformed_competition)[:, 1]
        feature_names = processor.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
        else:
            importance = np.abs(np.asarray(model.coef_[0]))
        importance_dir = output_dir / "feature_importance"
        importance_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False).to_csv(
            importance_dir / f"{name}.csv", index=False
        )
        joblib.dump(
            {"preprocessor": processor, "model": model},
            output_dir / f"{name}.joblib",
        )
    return rows, competition_scores


def _woe_transform(
    frame: pd.DataFrame,
    bins: dict[str, np.ndarray],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = {}
    for feature, edges in bins.items():
        labels = (
            pd.cut(frame[feature], edges, include_lowest=True)
            .astype("string")
            .fillna("MISSING")
        )
        mapping = tables[feature].set_index("bin")["woe"]
        result[feature] = labels.map(mapping).fillna(mapping["MISSING"])
    return pd.DataFrame(result, index=frame.index, dtype=float)


def _fit_woe(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    competition_test: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[list[dict[str, object]], np.ndarray]:
    numeric = train[feature_columns].select_dtypes(include=np.number).columns
    candidates = []
    bins: dict[str, np.ndarray] = {}
    tables: dict[str, pd.DataFrame] = {}
    for feature in numeric:
        edges = bin_by_tree(train[feature], train[TARGET])
        table, iv = woe_iv(train, feature, TARGET, bins=edges)
        if iv >= 0.02 and is_monotonic_woe(table):
            empty_missing = table["bin"].eq("MISSING") & table["total"].eq(0)
            if empty_missing.any():
                table.loc[empty_missing, "woe"] = table.loc[
                    ~empty_missing, "woe"
                ].min()
            candidates.append((feature, iv))
            bins[feature] = edges
            tables[feature] = table
    selected = [name for name, _ in sorted(candidates, key=lambda x: -x[1])[:25]]
    if not selected:
        raise ValueError("No monotonic numeric feature passed IV >= 0.02")
    bins = {name: bins[name] for name in selected}
    tables = {name: tables[name] for name in selected}
    transformed_train = _woe_transform(train, bins, tables)

    while True:
        probe = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        probe.fit(transformed_train, train[TARGET])
        wrong = [
            feature
            for feature, coefficient in zip(
                transformed_train.columns, probe.coef_[0], strict=True
            )
            if coefficient >= 0
        ]
        if not wrong:
            break
        if len(transformed_train.columns) - len(wrong) < 2:
            raise ValueError("Sign filtering left fewer than two WoE features")
        transformed_train = transformed_train.drop(columns=wrong)
        for feature in wrong:
            bins.pop(feature)
            tables.pop(feature)

    model, scorecard = scorecard_from_lr(
        transformed_train, train[TARGET], tables
    )
    scorecard.to_csv(output_dir / "scorecard.csv", index=False)
    pd.concat(tables.values(), ignore_index=True).to_csv(
        output_dir / "woe_iv_detail.csv", index=False
    )
    rows = []
    for split_name, frame in [("valid", valid), ("test", test)]:
        transformed = _woe_transform(frame, bins, tables)
        rows.append(
            _metrics(
                "logistic_woe",
                split_name,
                frame[TARGET],
                model.predict_proba(transformed)[:, 1],
            )
        )
    competition_score = model.predict_proba(
        _woe_transform(competition_test, bins, tables)
    )[:, 1]
    joblib.dump(
        {"bins": bins, "tables": tables, "model": model},
        output_dir / "logistic_woe.joblib",
    )

    def credit_scores(frame: pd.DataFrame) -> pd.Series:
        values = pd.Series(0, index=frame.index, dtype="int64")
        for feature, edges in bins.items():
            labels = (
                pd.cut(frame[feature], edges, include_lowest=True)
                .astype("string")
                .fillna("MISSING")
            )
            points = scorecard.loc[
                scorecard["feature"].eq(feature), ["bin", "points"]
            ].set_index("bin")["points"]
            values += labels.map(points).fillna(points["MISSING"]).astype(int)
        return values

    train_scores = credit_scores(train)
    test_scores = credit_scores(test)
    competition_scores = credit_scores(competition_test)
    cutoff_rows = []
    for approval_rate in [0.6, 0.7, 0.8]:
        cutoff = float(test_scores.quantile(1 - approval_rate))
        approved = test_scores.ge(cutoff)
        cutoff_rows.append(
            {
                "approval_rate_target": approval_rate,
                "score_cutoff": cutoff,
                "approval_rate_actual": float(approved.mean()),
                "approved_bad_rate": float(test.loc[approved, TARGET].mean()),
            }
        )
    pd.DataFrame(cutoff_rows).to_csv(output_dir / "cutoffs.csv", index=False)
    psi_value, psi_detail = psi(train_scores, competition_scores)
    psi_detail.to_csv(output_dir / "score_psi_detail.csv", index=False)
    pd.DataFrame(
        [{"expected": "train", "actual": "application_test", "psi": psi_value}]
    ).to_csv(output_dir / "score_psi.csv", index=False)
    return rows, competition_score


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    *,
    level: str = "C",
) -> dict[str, object]:
    """Run a selected A/B/C pipeline and write all local deliverables."""
    source = load_tables(raw_dir)
    train_raw = source["application_train"]
    test_raw = source["application_test"]
    combined = pd.concat(
        [train_raw, test_raw.reindex(columns=train_raw.columns)],
        ignore_index=True,
    ).copy()
    clean, _ = clean_application(combined)
    _, findings = clean_application(train_raw)
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean.loc[clean[TARGET].notna()].to_parquet(
        processed_dir / "application-clean.parquet",
        index=False,
    )
    matrix = build_feature_matrix(
        clean,
        raw_dir,
        processed_dir / "aggregates",
        level=level,
    )
    matrix.to_parquet(processed_dir / f"feature_matrix_{level}.parquet", index=False)
    if level == "C":
        matrix.to_parquet(processed_dir / "feature_matrix.parquet", index=False)
    train_matrix = matrix[matrix[TARGET].notna()].copy()
    train_matrix[TARGET] = train_matrix[TARGET].astype("int8")
    competition_test = matrix[matrix[TARGET].isna()].copy()
    train, valid, test = split_application(train_matrix)
    membership = pd.concat(
        [
            part[[ID_COLUMN]].assign(split=name)
            for name, part in [("train", train), ("valid", valid), ("test", test)]
        ]
    ).sort_values(ID_COLUMN)
    membership.to_csv(processed_dir / "split_membership.csv", index=False)
    _write_eda(train, findings, output_dir / "eda")

    feature_columns = [
        column for column in matrix.columns if column not in {ID_COLUMN, TARGET}
    ]
    rows, competition_scores = _fit_raw_models(
        train,
        valid,
        test,
        competition_test,
        feature_columns,
        output_dir / "models",
    )
    scorecard_dir = output_dir / "scorecard"
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    woe_rows, woe_score = _fit_woe(
        train,
        valid,
        test,
        competition_test,
        feature_columns,
        scorecard_dir,
    )
    rows.extend(woe_rows)
    competition_scores["logistic_woe"] = woe_score
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "models" / "metrics.csv", index=False)
    metrics.to_csv(
        output_dir / "models" / f"metrics_{level}.csv", index=False
    )
    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for name, scores in competition_scores.items():
        pd.DataFrame(
            {ID_COLUMN: competition_test[ID_COLUMN].astype(int), TARGET: scores}
        ).to_csv(submission_dir / f"{name}.csv", index=False)
    summary = {
        "level": level,
        "random_state": RANDOM_STATE,
        "split": "stratified random 60/20/20",
        "train_rows": len(train_matrix),
        "competition_test_rows": len(competition_test),
        "feature_count": len(feature_columns),
        "metrics": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"run_summary_{level}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
