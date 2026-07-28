"""Dataset loading and anomaly treatment for GiveMeSomeCredit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
ID_COLUMN = "Unnamed: 0"
DELINQUENCY_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def find_training_csv(raw_dir: Path) -> Path:
    """Find cs-training.csv under a downloaded competition directory."""
    matches = sorted(raw_dir.rglob("cs-training.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Không tìm thấy cs-training.csv trong {raw_dir}. "
            "Hãy chạy scripts/download_data.py trước."
        )
    return matches[0]


def load_training_data(csv_path: Path) -> pd.DataFrame:
    """Load training data and remove Kaggle's exported row-number column."""
    frame = pd.read_csv(csv_path)
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=ID_COLUMN)
    if TARGET not in frame.columns:
        raise ValueError(f"Thiếu target bắt buộc: {TARGET}")
    return frame


def clean_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply documented anomaly handling while retaining anomaly flags."""
    clean = frame.copy()
    findings: list[dict[str, object]] = []

    age_zero = clean["age"].eq(0)
    findings.append(
        {
            "feature": "age",
            "rule": "age == 0",
            "count": int(age_zero.sum()),
            "action": "replace with NaN; median imputation happens inside model",
        }
    )
    clean.loc[age_zero, "age"] = np.nan

    for column in DELINQUENCY_COLUMNS:
        flag_name = f"{column}SpecialCode"
        special = clean[column].isin([96, 98])
        clean[flag_name] = special.astype("int8")
        clean.loc[special, column] = np.nan
        findings.append(
            {
                "feature": column,
                "rule": "value in {96, 98}",
                "count": int(special.sum()),
                "action": f"create {flag_name}; replace source value with NaN",
            }
        )

    utilization_extreme = clean["RevolvingUtilizationOfUnsecuredLines"].gt(10)
    findings.append(
        {
            "feature": "RevolvingUtilizationOfUnsecuredLines",
            "rule": "value > 10",
            "count": int(utilization_extreme.sum()),
            "action": "retain for tree/WoE bins; robust-scale for LR",
        }
    )

    debt_extreme = clean["DebtRatio"].gt(10)
    findings.append(
        {
            "feature": "DebtRatio",
            "rule": "value > 10",
            "count": int(debt_extreme.sum()),
            "action": "retain for tree/WoE bins; robust-scale for LR",
        }
    )
    return clean, pd.DataFrame(findings)
