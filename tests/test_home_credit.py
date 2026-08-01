"""Tests for Home Credit cleaning and staged aggregation contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from home_credit.aggregate import build_feature_matrix
from home_credit.data import clean_application


class HomeCreditDataTests(unittest.TestCase):
    def test_clean_application_preserves_rows_and_marks_sentinels(self) -> None:
        frame = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "TARGET": [0, 1],
                "DAYS_EMPLOYED": [-100, 365243],
                "DAYS_BIRTH": [-10000, -20000],
                "CODE_GENDER": ["F", "XNA"],
                "NAME_FAMILY_STATUS": ["Married", "Unknown"],
                "AMT_CREDIT": [100.0, 200.0],
                "AMT_INCOME_TOTAL": [50.0, 0.0],
                "AMT_ANNUITY": [10.0, 20.0],
                "AMT_GOODS_PRICE": [80.0, 100.0],
            }
        )

        clean, findings = clean_application(frame)

        self.assertEqual(len(clean), len(frame))
        self.assertEqual(clean["DAYS_EMPLOYED_ANOMALY"].tolist(), [0, 1])
        self.assertEqual(clean.loc[0, "DAYS_EMPLOYED"], 100)
        self.assertTrue(pd.isna(clean.loc[1, "DAYS_EMPLOYED"]))
        self.assertTrue(pd.isna(clean.loc[1, "CODE_GENDER"]))
        self.assertEqual(len(findings), 3)
        self.assertTrue(pd.isna(clean.loc[1, "CREDIT_INCOME_RATIO"]))

    def test_level_a_does_not_require_auxiliary_files(self) -> None:
        application = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]})
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = build_feature_matrix(
                application,
                Path(temporary_directory),
                Path(temporary_directory) / "cache",
                level="A",
            )
        pd.testing.assert_frame_equal(result, application)


if __name__ == "__main__":
    unittest.main()
