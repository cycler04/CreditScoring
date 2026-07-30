"""Tests for model output artifacts."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.pipeline import (
    _write_gini_curve,
    _write_ks_curve,
    _write_roc_auc_curve,
)


class PipelineOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        self.y_true = y_true
        self.predictions = {
            "model_a": np.array([0.1, 0.4, 0.6, 0.9]),
            "model_b": np.array([0.2, 0.3, 0.7, 0.8]),
        }

    def _assert_png_created(
        self,
        writer: Callable[[pd.Series, dict[str, np.ndarray], Path], None],
        filename: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory) / "models" / "metrics" / filename
            )
            writer(self.y_true, self.predictions, output_path)

            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_write_roc_auc_curve_creates_png(self) -> None:
        self._assert_png_created(_write_roc_auc_curve, "roc_auc_curve.png")

    def test_write_gini_curve_creates_png(self) -> None:
        self._assert_png_created(_write_gini_curve, "gini_curve.png")

    def test_write_ks_curve_creates_png(self) -> None:
        self._assert_png_created(_write_ks_curve, "ks_curve.png")


if __name__ == "__main__":
    unittest.main()
