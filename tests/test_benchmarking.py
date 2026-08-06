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
            explanation_times={"lightgbm": 0.25},
        )

        self.assertEqual(list(table.columns), BENCHMARK_COLUMNS)
        self.assertEqual(table.loc[0, "Model"], "LightGBM + SHAP")
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
            self.assertIn(
                "| LightGBM + SHAP | 0.800000 | 0.070000 | 0.450000 | 8 |",
                text,
            )
            self.assertIn("estimator-native local-attribution latency", text)

    def test_external_metric_without_predictions_has_no_brier(self) -> None:
        metrics = pd.DataFrame(
            {
                "model": ["ft_transformer"],
                "split": ["test"],
                "auc": [0.77],
                "ks": [0.40],
            }
        )

        table = build_benchmark_table(
            metrics,
            {},
            active_features={},
            stability=None,
            monotonic_violations={},
        )

        measured = table.loc[table["Model"].eq("FT-Transformer")].iloc[0]
        self.assertEqual(measured["AUC"], 0.77)
        self.assertTrue(pd.isna(measured["Brier"]))


if __name__ == "__main__":
    unittest.main()
