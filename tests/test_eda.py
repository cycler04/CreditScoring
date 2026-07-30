"""Tests for EDA tables."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from credit_scoring.eda import summary_statistics


class SummaryStatisticsTests(unittest.TestCase):
    def test_summary_statistics_includes_range_center_and_missingness(self) -> None:
        frame = pd.DataFrame(
            {
                "feature_a": [1.0, 3.0, np.nan],
                "feature_b": [0, 2, 4],
            }
        )

        result = summary_statistics(frame, ["feature_a", "feature_b"]).set_index(
            "feature"
        )

        self.assertEqual(
            result.columns.tolist(),
            ["min", "max", "mean", "median", "missing_count", "missing_rate_pct"],
        )
        self.assertEqual(result.loc["feature_a", "min"], 1.0)
        self.assertEqual(result.loc["feature_a", "max"], 3.0)
        self.assertEqual(result.loc["feature_a", "mean"], 2.0)
        self.assertEqual(result.loc["feature_a", "median"], 2.0)
        self.assertEqual(result.loc["feature_a", "missing_count"], 1)
        self.assertEqual(result.loc["feature_a", "missing_rate_pct"], 33.33)
        self.assertEqual(result.loc["feature_b", "missing_count"], 0)


if __name__ == "__main__":
    unittest.main()
