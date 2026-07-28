"""End-to-end reproducible pipeline for the weekly checklist."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import nbformat
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .data import TARGET, clean_features, find_training_csv, load_training_data
from .eda import write_eda_outputs
from .metrics import psi
from .scorecard import (
    bin_by_tree,
    is_monotonic_woe,
    scorecard_from_lr,
    woe_iv,
)

RANDOM_STATE = 42


def _split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _ks_statistic(y_true: pd.Series, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def _model_metrics(
    name: str,
    split: str,
    y_true: pd.Series,
    scores: np.ndarray,
) -> dict[str, object]:
    auc = float(roc_auc_score(y_true, scores))
    return {
        "model": name,
        "split": split,
        "n": len(y_true),
        "bad_rate": float(y_true.mean()),
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": _ks_statistic(y_true, scores),
    }


def _fit_baselines(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = train[feature_columns], train[TARGET]
    x_valid, y_valid = valid[feature_columns], valid[TARGET]
    x_test, y_test = test[feature_columns], test[TARGET]

    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            ),
        ]
    )
    logistic.fit(x_train, y_train)

    lightgbm = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_estimators=3000,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    lightgbm.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )

    rows = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in [("logistic_raw", logistic), ("lightgbm", lightgbm)]:
        for split_name, x_values, y_values in [
            ("valid", x_valid, y_valid),
            ("test", x_test, y_test),
        ]:
            scores = model.predict_proba(x_values)[:, 1]
            rows.append(_model_metrics(name, split_name, y_values, scores))
            if split_name == "test":
                predictions[name] = scores

    joblib.dump(logistic, output_dir / "logistic_raw.joblib")
    joblib.dump(lightgbm, output_dir / "lightgbm.joblib")
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": lightgbm.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    importance["gain_pct"] = importance["gain"] / importance["gain"].sum()
    importance.to_csv(output_dir / "lightgbm_feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "target": y_test.to_numpy(),
            **predictions,
        },
        index=y_test.index,
    ).sort_index().to_csv(output_dir / "test_predictions.csv", index_label="row_index")
    return pd.DataFrame(rows)


def _monotonic_table(
    train: pd.DataFrame,
    feature: str,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    for depth in range(4, 0, -1):
        edges = bin_by_tree(
            train[feature],
            train[TARGET],
            max_depth=depth,
            min_samples_leaf=0.05,
        )
        table, _ = woe_iv(train, feature, TARGET, bins=edges)
        if is_monotonic_woe(table):
            return edges, table, depth
    raise RuntimeError(f"Không tìm được monotonic bins cho {feature}")


def _map_woe(
    frame: pd.DataFrame,
    features: list[str],
    edges: dict[str, np.ndarray],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = {}
    for feature in features:
        labels = (
            pd.cut(frame[feature], edges[feature], include_lowest=True)
            .astype("string")
            .fillna("MISSING")
        )
        mapping = tables[feature].set_index("bin")["woe"]
        result[feature] = labels.map(mapping).astype(float)
    return pd.DataFrame(result, index=frame.index)


def _score_from_card(
    frame: pd.DataFrame,
    features: list[str],
    edges: dict[str, np.ndarray],
    scorecard: pd.DataFrame,
) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    for feature in features:
        labels = (
            pd.cut(frame[feature], edges[feature], include_lowest=True)
            .astype("string")
            .fillna("MISSING")
        )
        mapping = scorecard.loc[
            scorecard["feature"].eq(feature), ["bin", "points"]
        ].set_index("bin")["points"]
        total = total.add(labels.map(mapping).astype(float), fill_value=0)
    return total


def _fit_scorecard(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges: dict[str, np.ndarray] = {}
    tables: dict[str, pd.DataFrame] = {}
    binning_rows = []
    for feature in features:
        feature_edges, table, depth = _monotonic_table(train, feature)
        edges[feature] = feature_edges
        tables[feature] = table
        binning_rows.append(
            {
                "feature": feature,
                "tree_depth": depth,
                "n_bins_including_missing": len(table),
                "monotonic_woe": is_monotonic_woe(table),
                "iv": float(table["iv"].iloc[0]),
                "iv_flag": (
                    "suspicious_gt_0.5"
                    if float(table["iv"].iloc[0]) > 0.5
                    else "strong"
                    if float(table["iv"].iloc[0]) >= 0.3
                    else "medium"
                    if float(table["iv"].iloc[0]) >= 0.1
                    else "weak"
                    if float(table["iv"].iloc[0]) >= 0.02
                    else "not_predictive"
                ),
            }
        )

    woe_train = _map_woe(train, features, edges, tables)
    model, scorecard = scorecard_from_lr(
        woe_train,
        train[TARGET],
        tables,
        min_score=300,
        max_score=850,
    )
    joblib.dump(model, output_dir / "logistic_woe.joblib")
    scorecard.to_csv(output_dir / "scorecard.csv", index=False)
    pd.DataFrame(binning_rows).sort_values("iv", ascending=False).to_csv(
        output_dir / "iv_summary.csv", index=False
    )
    pd.concat(tables.values(), ignore_index=True).to_csv(
        output_dir / "woe_iv_detail.csv", index=False
    )
    with (output_dir / "bin_edges.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                feature: [
                    "-Infinity"
                    if np.isneginf(value)
                    else "Infinity"
                    if np.isposinf(value)
                    else float(value)
                    for value in values
                ]
                for feature, values in edges.items()
            },
            handle,
            indent=2,
        )

    metrics = []
    scores_by_split: dict[str, pd.Series] = {}
    for split_name, split_frame in [
        ("valid", valid),
        ("test", test),
    ]:
        woe_values = _map_woe(split_frame, features, edges, tables)
        pd_bad = model.predict_proba(woe_values)[:, 1]
        metrics.append(
            _model_metrics("logistic_woe", split_name, split_frame[TARGET], pd_bad)
        )
        scores_by_split[split_name] = _score_from_card(
            split_frame, features, edges, scorecard
        )

    approval_rows = []
    test_scores = scores_by_split["test"]
    for approval_rate in [0.6, 0.7, 0.8]:
        cutoff = float(test_scores.quantile(1 - approval_rate))
        approved = test_scores.ge(cutoff)
        approval_rows.append(
            {
                "target_approval_rate": approval_rate,
                "score_cutoff": cutoff,
                "actual_approval_rate": float(approved.mean()),
                "approved_bad_rate": float(test.loc[approved, TARGET].mean()),
            }
        )
    pd.DataFrame(approval_rows).to_csv(
        output_dir / "approval_cutoffs.csv", index=False
    )

    midpoint = len(train) // 2
    expected_scores = _score_from_card(
        train.iloc[:midpoint], features, edges, scorecard
    )
    actual_scores = _score_from_card(
        train.iloc[midpoint:], features, edges, scorecard
    )
    psi_value, psi_detail = psi(expected_scores, actual_scores, bins=10)
    psi_detail.to_csv(output_dir / "score_psi_detail.csv", index=False)

    summary = {
        "score_psi_first_vs_second_train_half": psi_value,
        "score_min_possible_from_card": int(
            scorecard.groupby("feature")["points"].min().sum()
        ),
        "score_max_possible_from_card": int(
            scorecard.groupby("feature")["points"].max().sum()
        ),
    }
    return pd.DataFrame(metrics), summary


def _write_eda_notebook(project_root: Path) -> None:
    """Create a compact, rerunnable notebook backed by generated EDA tables."""
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "CreditScoring (.venv)",
                "language": "python",
                "name": "python3",
            }
        },
        cells=[
            nbformat.v4.new_markdown_cell(
                "# GiveMeSomeCredit — EDA\n\n"
                "Notebook này đọc dữ liệu và các bảng được tạo bởi "
                "`scripts/run_pipeline.py`. EDA chỉ dùng training split để "
                "tránh nhìn vào validation/test."
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\n"
                "import pandas as pd\n\n"
                "PROJECT_ROOT = Path.cwd()\n"
                "if PROJECT_ROOT.name == 'eda':\n"
                "    PROJECT_ROOT = PROJECT_ROOT.parents[1]\n"
                "EDA_DIR = PROJECT_ROOT / 'outputs' / 'eda'\n"
                "DATASET = PROJECT_ROOT / 'datasets' / 'processed' / "
                "'cs-training-clean.csv'"
            ),
            nbformat.v4.new_markdown_cell("## Target distribution"),
            nbformat.v4.new_code_cell(
                "pd.read_csv(EDA_DIR / 'target_distribution.csv')"
            ),
            nbformat.v4.new_markdown_cell("## Missing values"),
            nbformat.v4.new_code_cell(
                "pd.read_csv(EDA_DIR / 'missing_summary.csv').head(10)"
            ),
            nbformat.v4.new_markdown_cell("## Anomalies và cách xử lý"),
            nbformat.v4.new_code_cell(
                "pd.read_csv(EDA_DIR / 'anomaly_findings.csv')"
            ),
            nbformat.v4.new_markdown_cell("## Bad rate theo decile của 10 biến"),
            nbformat.v4.new_code_cell(
                "deciles = pd.read_csv(EDA_DIR / 'bad_rate_by_decile.csv')\n"
                "deciles.groupby('feature', sort=False).head(10)"
            ),
            nbformat.v4.new_markdown_cell("## Numeric summary"),
            nbformat.v4.new_code_cell(
                "pd.read_csv(EDA_DIR / 'numeric_summary.csv')"
            ),
        ],
    )
    target = project_root / "outputs" / "eda" / "give_me_some_credit_eda.ipynb"
    nbformat.write(notebook, target)


def run_pipeline(project_root: Path) -> dict[str, object]:
    """Run data cleaning, EDA, baselines, scorecard, and stability outputs."""
    raw_csv = find_training_csv(project_root / "datasets" / "raw")
    raw = load_training_data(raw_csv)
    original_features = [column for column in raw.columns if column != TARGET]
    clean, findings = clean_features(raw)
    feature_columns = [column for column in clean.columns if column != TARGET]
    train, valid, test = _split(clean)

    processed_dir = project_root / "datasets" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_dir / "cs-training-clean.csv", index=False)
    split_membership = pd.Series(index=clean.index, dtype="string", name="split")
    split_membership.loc[train.index] = "train"
    split_membership.loc[valid.index] = "valid"
    split_membership.loc[test.index] = "test"
    split_membership.sort_index().to_csv(
        processed_dir / "split_membership.csv", index_label="row_index"
    )

    output_root = project_root / "outputs"
    write_eda_outputs(
        train,
        original_features,
        TARGET,
        findings,
        output_root / "eda",
    )
    _write_eda_notebook(project_root)
    baseline_metrics = _fit_baselines(
        train,
        valid,
        test,
        feature_columns,
        output_root / "models",
    )
    scorecard_metrics, scorecard_summary = _fit_scorecard(
        train,
        valid,
        test,
        original_features,
        output_root / "scorecard",
    )
    metrics = pd.concat([baseline_metrics, scorecard_metrics], ignore_index=True)
    metrics.to_csv(output_root / "models" / "metrics.csv", index=False)

    summary = {
        "raw_csv": str(raw_csv.relative_to(project_root)),
        "rows": len(clean),
        "original_feature_count": len(original_features),
        "model_feature_count": len(feature_columns),
        "split": {"train": len(train), "valid": len(valid), "test": len(test)},
        "validation_note": (
            "Dataset has no time column; stratified random split is used. "
            "This is not out-of-time validation."
        ),
        **scorecard_summary,
    }
    with (output_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
