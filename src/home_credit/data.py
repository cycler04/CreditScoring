"""Loading and deterministic application-table cleaning for HCDR."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

TABLE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}


def resolve_table_path(raw_dir: Path, table_name: str) -> Path:
    """Resolve one canonical HCDR CSV and fail clearly when it is absent."""
    try:
        filename = TABLE_FILES[table_name]
    except KeyError as error:
        raise ValueError(f"Unknown HCDR table: {table_name}") from error
    path = raw_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing HCDR source table: {path}")
    return path


def load_tables(
    raw_dir: Path,
    table_names: Iterable[str] = ("application_train", "application_test"),
) -> dict[str, pd.DataFrame]:
    """Load selected tables.

    Auxiliary tables are intentionally opt-in: some are too large to coexist
    safely in memory and are aggregated directly from CSV by ``aggregate.py``.
    """
    tables: dict[str, pd.DataFrame] = {}
    for table_name in table_names:
        path = resolve_table_path(raw_dir, table_name)
        tables[table_name] = pd.read_csv(path, low_memory=False)
    return tables


def _bad_rate(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    if TARGET not in frame or not mask.any():
        return None
    return float(frame.loc[mask, TARGET].mean())


def clean_application(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean known sentinels and create traceable application-level features."""
    if ID_COLUMN not in frame:
        raise ValueError(f"Missing required identifier: {ID_COLUMN}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique in the application table")

    clean = frame.copy()
    findings: list[dict[str, object]] = []

    employed_sentinel = clean["DAYS_EMPLOYED"].eq(365243)
    clean["DAYS_EMPLOYED_ANOMALY"] = employed_sentinel.astype("int8")
    findings.append(
        {
            "feature": "DAYS_EMPLOYED",
            "rule": "value == 365243",
            "count": int(employed_sentinel.sum()),
            "bad_rate": _bad_rate(clean, employed_sentinel),
            "action": "create DAYS_EMPLOYED_ANOMALY; replace with NaN",
        }
    )
    clean.loc[employed_sentinel, "DAYS_EMPLOYED"] = np.nan

    for column, invalid_value in [
        ("CODE_GENDER", "XNA"),
        ("NAME_FAMILY_STATUS", "Unknown"),
    ]:
        invalid = clean[column].eq(invalid_value)
        flag = f"{column}_ANOMALY"
        clean[flag] = invalid.astype("int8")
        findings.append(
            {
                "feature": column,
                "rule": f"value == {invalid_value!r}",
                "count": int(invalid.sum()),
                "bad_rate": _bad_rate(clean, invalid),
                "action": f"create {flag}; replace with missing",
            }
        )
        clean.loc[invalid, column] = pd.NA

    # HCDR stores elapsed-day quantities as negative offsets from application.
    # Positive magnitudes are easier to interpret; the sentinel is removed first.
    day_columns = [
        column
        for column in clean.columns
        if column.startswith("DAYS_") and column != "DAYS_EMPLOYED_ANOMALY"
    ]
    for column in day_columns:
        clean[column] = clean[column].abs()

    safe_denominators = {
        "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "CREDIT_GOODS_RATIO": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
        "EMPLOYED_BIRTH_RATIO": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
    }
    for output, (numerator, denominator) in safe_denominators.items():
        denominator_values = clean[denominator].replace(0, np.nan)
        clean[output] = clean[numerator] / denominator_values

    return clean, pd.DataFrame(findings)
