"""Tests for reusable cross-pipeline diagrams."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from credit_scoring.visualization import (
    write_bad_rate_by_period_plot,
    write_cutoff_plot,
    write_eda_overview,
    write_feature_importance_plot,
    write_gini_by_period_plot,
    write_metrics_comparison_plot,
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
                    "model": ["a", "b"],
                    "split": ["test", "test"],
                    "auc": [0.8, 0.75],
                    "gini": [0.6, 0.5],
                    "ks": [0.45, 0.4],
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
            write_eda_overview(
                eda,
                "target",
                paths["eda"],
                title="EDA",
            )

            for path in paths.values():
                self._assert_png(path)


if __name__ == "__main__":
    unittest.main()
