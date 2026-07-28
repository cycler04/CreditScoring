"""Tree binning, WoE/IV, and scorecard scaling."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def bin_by_tree(
    feature: Iterable[float],
    target: Iterable[int],
    *,
    max_depth: int = 3,
    max_leaf_nodes: int = 6,
    min_samples_leaf: float = 0.05,
) -> np.ndarray:
    """Learn numeric bin edges on non-missing observations."""
    x = pd.Series(feature, dtype="float64")
    y = pd.Series(target, index=x.index)
    valid = x.notna() & y.notna()
    if valid.sum() == 0 or x[valid].nunique() < 2:
        return np.array([-np.inf, np.inf])
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    tree.fit(x.loc[valid].to_numpy().reshape(-1, 1), y.loc[valid])
    thresholds = tree.tree_.threshold[tree.tree_.feature == 0]
    return np.array([-np.inf, *sorted(thresholds), np.inf], dtype=float)


def woe_iv(
    frame: pd.DataFrame,
    col: str,
    target: str,
    *,
    bins: Iterable[float] | None = None,
    smoothing: float = 0.5,
) -> tuple[pd.DataFrame, float]:
    """Compute WoE/IV using 1=bad and WoE=ln(%good/%bad)."""
    values = frame[col]
    if bins is not None:
        labels = (
            pd.cut(values, np.asarray(list(bins)), include_lowest=True)
            .cat.add_categories(["MISSING"])
            .fillna("MISSING")
        )
    else:
        labels = values.astype("string").fillna("MISSING")

    summary = (
        pd.DataFrame({"bin": labels, "target": frame[target]})
        .groupby("bin", observed=False, dropna=False, sort=True)["target"]
        .agg(total="size", bad="sum")
        .reset_index()
    )
    summary["bin"] = summary["bin"].astype(str)
    if not summary["bin"].eq("MISSING").any():
        summary = pd.concat(
            [
                summary,
                pd.DataFrame({"bin": ["MISSING"], "total": [0], "bad": [0]}),
            ],
            ignore_index=True,
        )
    summary["good"] = summary["total"] - summary["bad"]
    n_bins = len(summary)
    total_good = summary["good"].sum() + smoothing * n_bins
    total_bad = summary["bad"].sum() + smoothing * n_bins
    summary["dist_good"] = (summary["good"] + smoothing) / total_good
    summary["dist_bad"] = (summary["bad"] + smoothing) / total_bad
    summary["woe"] = np.log(summary["dist_good"] / summary["dist_bad"])
    summary["iv_component"] = (
        summary["dist_good"] - summary["dist_bad"]
    ) * summary["woe"]
    empty_bin = summary["total"].eq(0)
    summary.loc[empty_bin, ["woe", "iv_component"]] = 0.0
    iv = float(summary["iv_component"].sum())
    summary["iv"] = iv
    summary.insert(0, "feature", col)
    return summary, iv


def is_monotonic_woe(table: pd.DataFrame) -> bool:
    """Return whether non-missing WoE values are monotonic."""
    values = table.loc[table["bin"].ne("MISSING"), "woe"].to_numpy()
    if len(values) < 3:
        return True
    diffs = np.diff(values)
    return bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))


def scorecard_from_lr(
    woe_frame: pd.DataFrame,
    target: Iterable[int],
    woe_tables: dict[str, pd.DataFrame],
    *,
    min_score: int = 300,
    max_score: int = 850,
) -> tuple[LogisticRegression, pd.DataFrame]:
    """Fit LR on WoE values and scale bin contributions to 300-850."""
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(woe_frame, target)
    coefficients = dict(zip(woe_frame.columns, model.coef_[0], strict=True))

    rows = []
    for feature, table in woe_tables.items():
        feature_table = table.copy()
        feature_table["coefficient"] = coefficients[feature]
        feature_table["log_odds_contribution"] = (
            feature_table["woe"] * feature_table["coefficient"]
        )
        rows.append(feature_table)
    scorecard = pd.concat(rows, ignore_index=True)

    minimum = scorecard.groupby("feature")["log_odds_contribution"].min().sum()
    maximum = scorecard.groupby("feature")["log_odds_contribution"].max().sum()
    span = maximum - minimum
    if span <= 0:
        raise ValueError("Không thể scale scorecard vì contribution span bằng 0")
    # The LR predicts bad=1. A lower bad log-odds contribution must therefore
    # receive a higher conventional credit score.
    factor = (min_score - max_score) / span
    base_per_feature = (max_score - factor * minimum) / len(woe_tables)
    scorecard["points"] = (
        base_per_feature + factor * scorecard["log_odds_contribution"]
    ).round().astype(int)

    def force_boundary(kind: str, target_score: int) -> None:
        grouped = scorecard.groupby("feature")["points"]
        current = int(
            grouped.min().sum() if kind == "min" else grouped.max().sum()
        )
        delta = target_score - current
        if delta == 0:
            return
        for feature in scorecard["feature"].drop_duplicates():
            selected = scorecard["feature"].eq(feature)
            values = scorecard.loc[selected, "points"]
            extreme = values.min() if kind == "min" else values.max()
            extreme_rows = selected & scorecard["points"].eq(extreme)
            if (kind == "min" and delta > 0) or (kind == "max" and delta < 0):
                scorecard.loc[extreme_rows, "points"] += delta
            else:
                scorecard.loc[scorecard.index[extreme_rows][0], "points"] += delta
            break

    force_boundary("min", min_score)
    force_boundary("max", max_score)
    return model, scorecard
