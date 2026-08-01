"""Leakage-safe splits based on complete WEEK_NUM blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_week(
    frame: pd.DataFrame,
    *,
    week_column: str = "WEEK_NUM",
    train_fraction: float = 0.6,
    valid_fraction: float = 0.2,
) -> pd.Series:
    """Assign early/middle/late weeks to train/valid/test.

    Fractions are applied to the number of distinct sorted weeks, never to
    rows. Therefore a week cannot appear in more than one split.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between zero and one")
    if train_fraction + valid_fraction >= 1.0:
        raise ValueError("train_fraction + valid_fraction must be below one")
    if week_column not in frame:
        raise KeyError(f"Missing week column: {week_column}")
    if frame[week_column].isna().any():
        raise ValueError(f"{week_column} must not contain missing values")

    weeks = np.sort(frame[week_column].unique())
    if len(weeks) < 5:
        raise ValueError("At least five distinct weeks are required")
    train_end = max(1, int(np.floor(len(weeks) * train_fraction)))
    valid_end = max(train_end + 1, int(np.floor(len(weeks) * (train_fraction + valid_fraction))))
    valid_end = min(valid_end, len(weeks) - 1)
    mapping = {
        int(week): (
            "train"
            if index < train_end
            else "valid"
            if index < valid_end
            else "test"
        )
        for index, week in enumerate(weeks)
    }
    result = frame[week_column].map(mapping).astype("string")
    if result.isna().any():
        raise AssertionError("Every week must map to a split")
    return result
