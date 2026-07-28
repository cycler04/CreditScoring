"""Credit-risk validation and stability metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def psi(
    expected: Iterable[float],
    actual: Iterable[float],
    bins: int | Iterable[float] = 10,
    epsilon: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    """Calculate PSI using expected-distribution quantile bins.

    Missing values are compared in a dedicated bin. The returned table makes
    every contribution auditable.
    """
    expected_series = pd.Series(expected, dtype="float64")
    actual_series = pd.Series(actual, dtype="float64")
    expected_non_null = expected_series.dropna()

    if isinstance(bins, int):
        if bins < 2:
            raise ValueError("bins phải >= 2")
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(expected_non_null.quantile(quantiles).to_numpy())
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.asarray(list(bins), dtype=float)
        if edges.size < 2:
            raise ValueError("Cần ít nhất hai bin edges")
        edges = np.unique(edges)
        edges[0], edges[-1] = -np.inf, np.inf

    expected_bins = pd.cut(expected_series, edges, include_lowest=True)
    actual_bins = pd.cut(actual_series, edges, include_lowest=True)
    labels = list(expected_bins.cat.categories) + ["MISSING"]

    def distribution(values: pd.Series) -> pd.Series:
        counts = values.value_counts(sort=False).reindex(labels[:-1], fill_value=0)
        counts.loc["MISSING"] = int(values.isna().sum())
        return counts.astype(float) / max(len(values), 1)

    expected_pct = distribution(expected_bins)
    actual_pct = distribution(actual_bins)
    expected_safe = expected_pct.clip(lower=epsilon)
    actual_safe = actual_pct.clip(lower=epsilon)
    contributions = (actual_safe - expected_safe) * np.log(actual_safe / expected_safe)
    detail = pd.DataFrame(
        {
            "bin": [str(label) for label in labels],
            "expected_pct": expected_pct.to_numpy(),
            "actual_pct": actual_pct.to_numpy(),
            "psi_contribution": contributions.to_numpy(),
        }
    )
    return float(contributions.sum()), detail


def gini_by_period(
    frame: pd.DataFrame,
    period_col: str,
    target_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Calculate AUC and Gini for periods containing both target classes."""
    rows: list[dict[str, object]] = []
    for period, group in frame.groupby(period_col, observed=True, sort=True):
        if group[target_col].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(group[target_col], group[score_col])
        rows.append(
            {
                period_col: period,
                "n": len(group),
                "bad_rate": group[target_col].mean(),
                "auc": auc,
                "gini": np.nan if np.isnan(auc) else 2 * auc - 1,
            }
        )
    return pd.DataFrame(rows)
