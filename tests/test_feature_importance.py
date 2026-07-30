"""Tests for persisted-model feature importance inspection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_scoring.feature_importance import inspect_model_outputs


class FeatureImportanceTests(unittest.TestCase):
    def test_inspect_logistic_pipeline_writes_table_and_plot(self) -> None:
        features = pd.DataFrame(
            {
                "income": [20.0, np.nan, 35.0, 80.0, 90.0, 60.0],
                "late_payments": [4, 5, 3, 0, 0, 1],
            }
        )
        target = pd.Series([1, 1, 1, 0, 0, 0])
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(random_state=42)),
            ]
        ).fit(features, target)
        woe_model = LogisticRegression(random_state=42).fit(
            features.fillna(features.median()),
            target,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs_dir = Path(temporary_directory) / "outputs"
            models_dir = outputs_dir / "models"
            scorecard_dir = outputs_dir / "scorecard"
            models_dir.mkdir(parents=True)
            scorecard_dir.mkdir()
            joblib.dump(model, models_dir / "logistic.joblib")
            joblib.dump(woe_model, scorecard_dir / "logistic_woe.joblib")

            table, table_path, plot_paths = inspect_model_outputs(models_dir, top_n=2)

            self.assertTrue(table_path.is_file())
            self.assertEqual(table_path.parent.name, "feature_importance")
            self.assertEqual(
                {plot_path.name for plot_path in plot_paths},
                {"logistic.png", "logistic_woe.png"},
            )
            self.assertTrue(all(plot_path.is_file() for plot_path in plot_paths))
            self.assertEqual(
                plot_paths[0].read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            self.assertEqual(
                set(table.loc[table["model"].eq("logistic"), "feature"]),
                {"income", "late_payments", "missingindicator_income"},
            )
            self.assertEqual(
                set(table["importance_method"]),
                {"absolute_coefficient", "absolute_woe_coefficient"},
            )
            for total in table.groupby("model")["importance_pct"].sum():
                self.assertAlmostEqual(float(total), 1.0)
            self.assertEqual(
                table.loc[table["model"].eq("logistic"), "rank"].tolist(),
                [1, 2, 3],
            )

    def test_top_n_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "top_n"):
                inspect_model_outputs(Path(temporary_directory), top_n=0)


if __name__ == "__main__":
    unittest.main()
