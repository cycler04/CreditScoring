"""Official-style Home Credit gini-stability metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class StabilityResult:
    """Auditable decomposition of the gini-stability score."""

    stability: float
    mean_gini: float
    slope: float
    residual_std: float
    by_week: pd.DataFrame
    excluded_weeks: pd.DataFrame


def stability_metric(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    week_num: pd.Series | np.ndarray,
    *,
    min_week_rows: int = 500,
) -> StabilityResult:
    """Calculate gini stability using consecutive observed-week positions.

    Weeks are sorted by ``WEEK_NUM``. Regression x is then 0..n-1 rather than
    the raw week value, so gaps in observed week labels do not exaggerate or
    dilute the slope. Weeks below ``min_week_rows`` or containing only one
    target class are excluded and returned with an explicit reason.
    """
    if min_week_rows < 1:
        raise ValueError("min_week_rows must be positive")
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "score": np.asarray(y_score, dtype=float),
            "WEEK_NUM": np.asarray(week_num),
        }
    )
    if frame.empty:
        raise ValueError("stability_metric requires at least one row")
    if frame.isna().any().any():
        raise ValueError("target, score, and WEEK_NUM must not contain missing values")
    if not np.isfinite(frame["score"]).all():
        raise ValueError("score must contain only finite values")

    included: list[dict[str, float | int]] = []
    excluded: list[dict[str, str | int]] = []
    for week, group in frame.groupby("WEEK_NUM", sort=True, observed=True):
        n_rows = len(group)
        n_bad = int(group["target"].sum())
        n_good = int(n_rows - n_bad)
        if n_rows < min_week_rows:
            excluded.append(
                {
                    "WEEK_NUM": int(week),
                    "n": n_rows,
                    "n_bad": n_bad,
                    "n_good": n_good,
                    "reason": "fewer_than_min_week_rows",
                }
            )
            continue
        if group["target"].nunique() < 2:
            excluded.append(
                {
                    "WEEK_NUM": int(week),
                    "n": n_rows,
                    "n_bad": n_bad,
                    "n_good": n_good,
                    "reason": "single_target_class",
                }
            )
            continue
        auc = float(roc_auc_score(group["target"], group["score"]))
        included.append(
            {
                "WEEK_NUM": int(week),
                "n": n_rows,
                "n_bad": n_bad,
                "n_good": n_good,
                "bad_rate": n_bad / n_rows,
                "auc": auc,
                "gini": 2.0 * auc - 1.0,
            }
        )

    by_week = pd.DataFrame(included)
    excluded_weeks = pd.DataFrame(
        excluded,
        columns=["WEEK_NUM", "n", "n_bad", "n_good", "reason"],
    )
    if len(by_week) < 2:
        raise ValueError("stability_metric requires at least two eligible weeks")

    x = np.arange(len(by_week), dtype=float)
    gini = by_week["gini"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, gini, 1)
    residuals = gini - (slope * x + intercept)
    residual_std = float(np.std(residuals, ddof=0))
    mean_gini = float(np.mean(gini))
    stability = mean_gini + 88.0 * min(0.0, float(slope)) - 0.5 * residual_std
    by_week.insert(1, "week_index", x.astype(int))
    by_week["trend_gini"] = slope * x + intercept
    by_week["residual"] = residuals
    return StabilityResult(
        stability=float(stability),
        mean_gini=mean_gini,
        slope=float(slope),
        residual_std=residual_std,
        by_week=by_week,
        excluded_weeks=excluded_weeks,
    )


def selfcheck() -> None:
    """Verify flat, declining, and improving synthetic gini behavior."""

    def synthetic(strengths: list[float]) -> StabilityResult:
        targets: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        weeks: list[np.ndarray] = []
        for week, strength in enumerate(strengths):
            target = np.tile([0, 1], 500)
            noise = np.random.default_rng(42).normal(size=len(target))
            targets.append(target)
            scores.append(noise + strength * target)
            weeks.append(np.full(len(target), week * 2))
        return stability_metric(
            np.concatenate(targets),
            np.concatenate(scores),
            np.concatenate(weeks),
        )

    flat = synthetic([0.8, 0.8, 0.8])
    assert abs(flat.stability - flat.mean_gini) < 1e-10
    declining = synthetic([1.0, 0.8, 0.6])
    assert declining.slope < 0
    assert abs(
        declining.stability
        - (
            declining.mean_gini
            + 88.0 * declining.slope
            - 0.5 * declining.residual_std
        )
    ) < 1e-10
    improving = synthetic([0.6, 0.8, 1.0])
    assert improving.slope > 0
    assert abs(
        improving.stability
        - (improving.mean_gini - 0.5 * improving.residual_std)
    ) < 1e-10
