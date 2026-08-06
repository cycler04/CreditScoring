"""Contracts for the Home Credit Model Stability pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from home_credit_stability.split import split_by_week
from home_credit_stability.stability import selfcheck, stability_metric
from home_credit_stability.pipeline import (
    ENSEMBLE_MEMBERS,
    _equal_weight_ensembles,
    _ordered_submission,
)


def _scores_for_strength(
    strength: float, n: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic scores whose discrimination rises with strength."""
    target = np.tile([0, 1], n // 2)
    noise = np.random.default_rng(42).normal(size=n)
    return target, noise + strength * target


class StabilityMetricTests(unittest.TestCase):
    def test_documented_metric_selfcheck(self) -> None:
        selfcheck()

    def _series(
        self, strengths: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        targets, scores, weeks = [], [], []
        for week, strength in enumerate(strengths):
            target, score = _scores_for_strength(strength)
            targets.append(target)
            scores.append(score)
            weeks.append(np.full(len(target), week * 2))
        return np.concatenate(targets), np.concatenate(scores), np.concatenate(weeks)

    def test_flat_gini_equals_mean(self) -> None:
        result = stability_metric(*self._series([0.8, 0.8, 0.8]))
        self.assertAlmostEqual(result.stability, result.mean_gini, places=10)

    def test_declining_gini_receives_slope_penalty(self) -> None:
        result = stability_metric(*self._series([1.0, 0.8, 0.6]))
        expected = (
            result.mean_gini
            + 88.0 * result.slope
            - 0.5 * result.residual_std
        )
        self.assertLess(result.slope, 0)
        self.assertAlmostEqual(result.stability, expected, places=10)

    def test_improving_gini_receives_no_slope_reward(self) -> None:
        result = stability_metric(*self._series([0.6, 0.8, 1.0]))
        expected = result.mean_gini - 0.5 * result.residual_std
        self.assertGreater(result.slope, 0)
        self.assertAlmostEqual(result.stability, expected, places=10)

    def test_excluded_week_is_auditable(self) -> None:
        target, score, week = self._series([0.6, 0.8])
        target = np.concatenate([target, np.zeros(500)])
        score = np.concatenate([score, np.arange(500)])
        week = np.concatenate([week, np.full(500, 99)])
        result = stability_metric(target, score, week)
        self.assertEqual(result.excluded_weeks["WEEK_NUM"].tolist(), [99])
        self.assertEqual(result.excluded_weeks["reason"].tolist(), ["single_target_class"])


class WeekSplitTests(unittest.TestCase):
    def test_splits_are_disjoint_complete_week_blocks(self) -> None:
        frame = pd.DataFrame({"WEEK_NUM": np.repeat(np.arange(10), [1, 2] * 5)})
        split = split_by_week(frame)
        membership = frame.assign(split=split)
        week_counts = membership.groupby("WEEK_NUM")["split"].nunique()
        self.assertTrue(week_counts.eq(1).all())
        self.assertEqual(set(split), {"train", "valid", "test"})
        self.assertLess(
            membership.loc[membership["split"].eq("train"), "WEEK_NUM"].max(),
            membership.loc[membership["split"].eq("valid"), "WEEK_NUM"].min(),
        )


class EnsembleTests(unittest.TestCase):
    def test_equal_weight_ensembles_average_declared_members(self) -> None:
        base_models = sorted(
            {member for members in ENSEMBLE_MEMBERS.values() for member in members}
        )
        predictions = {
            model: {
                split: np.array([index, index + 0.5], dtype=float)
                for split in ("train", "valid", "test", "competition")
            }
            for index, model in enumerate(base_models)
        }

        ensembles = _equal_weight_ensembles(predictions)

        for name, members in ENSEMBLE_MEMBERS.items():
            expected = np.mean(
                [predictions[member]["test"] for member in members], axis=0
            )
            np.testing.assert_allclose(ensembles[name]["test"], expected)

    def test_submission_is_aligned_to_official_sample_order(self) -> None:
        sample = pd.DataFrame(
            {"case_id": [2, 1], "score": [0.5, 0.5]}
        )

        result = _ordered_submission(
            pd.Series([1, 2]),
            np.array([0.1, 0.2]),
            sample,
        )

        self.assertEqual(result["case_id"].tolist(), [2, 1])
        self.assertEqual(result["score"].tolist(), [0.2, 0.1])


if __name__ == "__main__":
    unittest.main()
