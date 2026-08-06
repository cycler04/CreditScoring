"""Tests for comparable model benchmark tables."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from credit_scoring.benchmarking import (
    BENCHMARK_COLUMNS,
    build_benchmark_table,
    write_benchmark_report,
)


class BenchmarkTableTests(unittest.TestCase):
    def test_table_keeps_measured_values_and_marks_candidates(self) -> None:
        metrics = pd.DataFrame(
            {
                "model": ["logistic_raw", "logistic_raw", "lightgbm"],
                "split": ["valid", "test", "test"],
                "auc": [0.70, 0.71, 0.80],
                "ks": [0.30, 0.31, 0.45],
            }
        )
        table = build_benchmark_table(
            metrics,
            {"logistic_raw": 0.08, "lightgbm": 0.07},
            active_features={"logistic_raw": 10, "lightgbm": 8},
            stability={"logistic_raw": 0.2, "lightgbm": 0.6},
            monotonic_violations={"logistic_raw": None, "lightgbm": None},
        )

        self.assertEqual(list(table.columns), BENCHMARK_COLUMNS)
        self.assertEqual(table.loc[0, "Model"], "LightGBM")
        self.assertEqual(table.loc[0, "Brier"], 0.07)
        self.assertTrue(table["Model"].str.contains("not implemented").any())

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "benchmark.md"
            write_benchmark_report(
                table,
                path,
                title="Benchmark",
                stability_definition="test definition.",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("| Model | AUC | Brier | KS |", text)
            self.assertIn("| LightGBM | 0.800000 | 0.070000 | 0.450000 | 8 |", text)
            self.assertIn("Explanation time is N/A", text)


if __name__ == "__main__":
    unittest.main()
