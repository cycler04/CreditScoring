"""Tests for reusable cross-pipeline diagrams."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from credit_scoring.visualization import (
    normalize_feature_importance,
    write_auc_benchmark_plot,
    write_bad_rate_by_period_plot,
    write_benchmark_dashboard,
    write_cutoff_plot,
    write_eda_overview,
    write_feature_importance_plot,
    write_gini_by_period_plot,
    write_metrics_comparison_plot,
    write_ranked_metric_benchmark_plot,
)


class VisualizationTests(unittest.TestCase):
    def _assert_png(self, path: Path) -> None:
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_tabular_diagram_writers_create_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            importance = pd.DataFrame(
                {"feature": ["income", "age"], "importance": [0.7, 0.3]}
            )
            periods = pd.DataFrame(
                {
                    "week": [1, 2, 1, 2],
                    "model": ["a", "a", "b", "b"],
                    "bad_rate": [0.1, 0.2, 0.12, 0.18],
                    "gini": [0.5, 0.45, 0.52, 0.48],
                }
            )
            cutoffs = pd.DataFrame(
                {
                    "target": [0.6, 0.8],
                    "actual": [0.61, 0.79],
                    "bad_rate": [0.02, 0.04],
                }
            )
            metrics = pd.DataFrame(
                {
                    "model": ["a", "a", "ensemble_a_b", "ensemble_a_b"],
                    "split": ["valid", "test", "valid", "test"],
                    "auc": [0.79, 0.8, 0.80, 0.81],
                    "gini": [0.58, 0.6, 0.60, 0.62],
                    "ks": [0.44, 0.45, 0.45, 0.46],
                }
            )
            eda = pd.DataFrame(
                {
                    "target": [0, 0, 1, 1],
                    "income": [1.0, None, 2.0, 3.0],
                    "age": [20, 30, 40, 50],
                }
            )

            paths = {
                "importance": output_dir / "importance.png",
                "bad_rate": output_dir / "bad_rate.png",
                "gini": output_dir / "gini.png",
                "cutoff": output_dir / "cutoff.png",
                "metrics": output_dir / "metrics.png",
                "auc_benchmark": output_dir / "auc_benchmark.png",
                "gini_benchmark": output_dir / "gini_benchmark.png",
                "ks_benchmark": output_dir / "ks_benchmark.png",
                "benchmark_dashboard": output_dir / "benchmark_dashboard.png",
                "eda": output_dir / "eda.png",
            }
            write_feature_importance_plot(
                importance, paths["importance"], title="Importance"
            )
            write_bad_rate_by_period_plot(
                periods,
                paths["bad_rate"],
                period_column="week",
                group_column="model",
                title="Bad rate",
            )
            write_gini_by_period_plot(
                periods,
                paths["gini"],
                period_column="week",
                group_column="model",
                title="Gini",
            )
            write_cutoff_plot(
                cutoffs,
                paths["cutoff"],
                target_column="target",
                actual_column="actual",
                bad_rate_column="bad_rate",
            )
            write_metrics_comparison_plot(metrics, paths["metrics"])
            write_auc_benchmark_plot(metrics, paths["auc_benchmark"])
            write_ranked_metric_benchmark_plot(
                metrics,
                paths["gini_benchmark"],
                metric="gini",
            )
            write_ranked_metric_benchmark_plot(
                metrics,
                paths["ks_benchmark"],
                metric="ks",
            )
            write_benchmark_dashboard(metrics, paths["benchmark_dashboard"])
            write_eda_overview(
                eda,
                "target",
                paths["eda"],
                title="EDA",
            )

            for path in paths.values():
                self._assert_png(path)

    def test_feature_importance_is_normalized_with_native_values_preserved(
        self,
    ) -> None:
        table = pd.DataFrame(
            {"feature": ["income", "age"], "importance": [7.0, -3.0]}
        )

        normalized = normalize_feature_importance(table)

        self.assertEqual(normalized["importance"].tolist(), [7.0, -3.0])
        self.assertEqual(normalized["importance_pct"].tolist(), [0.7, 0.3])
        self.assertAlmostEqual(float(normalized["importance_pct"].sum()), 1.0)

    def test_feature_importance_requires_a_positive_total(self) -> None:
        table = pd.DataFrame(
            {"feature": ["income", "age"], "importance": [0.0, 0.0]}
        )

        with self.assertRaisesRegex(ValueError, "positive total"):
            normalize_feature_importance(table)


if __name__ == "__main__":
    unittest.main()
