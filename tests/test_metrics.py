"""Small deterministic tests for reusable credit-scoring utilities."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from credit_scoring.metrics import gini_by_period, psi
from credit_scoring.scorecard import bin_by_tree, woe_iv


class MetricTests(unittest.TestCase):
    def test_psi_identical_is_zero(self) -> None:
        values = np.arange(100, dtype=float)
        value, detail = psi(values, values, bins=10)
        self.assertAlmostEqual(value, 0.0, places=12)
        self.assertEqual(len(detail), 11)

    def test_gini_by_period(self) -> None:
        frame = pd.DataFrame(
            {
                "month": [1, 1, 2, 2],
                "target": [0, 1, 0, 1],
                "score": [0.1, 0.9, 0.2, 0.8],
            }
        )
        result = gini_by_period(frame, "month", "target", "score")
        self.assertTrue(result["gini"].eq(1.0).all())

    def test_binning_and_woe(self) -> None:
        frame = pd.DataFrame(
            {"x": np.arange(100, dtype=float), "target": [0] * 50 + [1] * 50}
        )
        edges = bin_by_tree(frame["x"], frame["target"], max_depth=2)
        table, iv = woe_iv(frame, "x", "target", bins=edges)
        self.assertGreaterEqual(len(table), 2)
        self.assertGreater(iv, 0)


if __name__ == "__main__":
    unittest.main()
