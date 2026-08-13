"""Unit tests for local data feature extraction pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.local_data_pipeline.features import extract_features
from src.local_data_pipeline.parser import (
    clean_column_name,
    parse_currency_amount,
    parse_income_band,
    parse_internet_usage_gb,
    parse_loyalty_points,
    parse_numeric_count,
    parse_telco_monetary,
)
from src.local_data_pipeline.pipeline import LocalDataPipeline


class TestLocalDataParser(unittest.TestCase):
    """Test string and numeric parsing functions."""

    def test_clean_column_name(self) -> None:
        self.assertEqual(clean_column_name("\nuser_id "), "user_id")
        self.assertEqual(clean_column_name("age_group"), "age_group")

    def test_parse_numeric_count(self) -> None:
        self.assertEqual(parse_numeric_count("Không có"), 0.0)
        self.assertEqual(parse_numeric_count("1 đơn"), 1.0)
        self.assertEqual(parse_numeric_count("4-6 đơn"), 5.0)
        self.assertEqual(parse_numeric_count("Trên 6 đơn"), 7.0)

    def test_parse_income_band(self) -> None:
        low, high, mid = parse_income_band("Trên 18trđ đến 32trđ")
        self.assertEqual(low, 18e6)
        self.assertEqual(high, 32e6)
        self.assertEqual(mid, 25e6)

    def test_parse_loyalty_points(self) -> None:
        low, high, mid = parse_loyalty_points("150-400")
        self.assertEqual(low, 150.0)
        self.assertEqual(high, 400.0)
        self.assertEqual(mid, 275.0)

    def test_parse_telco_monetary(self) -> None:
        low, high, mid = parse_telco_monetary("1M - 2M")
        self.assertEqual(low, 1e6)
        self.assertEqual(high, 2e6)
        self.assertEqual(mid, 1.5e6)

    def test_parse_currency_amount(self) -> None:
        low, high, mid = parse_currency_amount("200-500K")
        self.assertEqual(low, 200000.0)
        self.assertEqual(high, 500000.0)
        self.assertEqual(mid, 350000.0)

    def test_parse_internet_usage_gb(self) -> None:
        low, high, mid = parse_internet_usage_gb("120 - 220GB/tháng")
        self.assertEqual(low, 120.0)
        self.assertEqual(high, 220.0)
        self.assertEqual(mid, 170.0)


class TestLocalDataFeatures(unittest.TestCase):
    """Test feature extraction and ratio engineering logic."""

    def test_extract_features(self) -> None:
        raw_df = pd.DataFrame(
            {
                "\nuser_id": ["user_1", "user_2"],
                "age_group": ["23-30", "60++"],
                "gender": ["Nam", "Nữ"],
                "city": ["Thành phố Hà Nội", "Tỉnh Đồng Nai"],
                "household_type": ["Nhà thường", "Công ty"],
                "income_band_est": ["Trên 18trđ đến 32trđ", "Trên 5trđ đến 10trđ"],
                "active_domain_count": ["4+ services", "2-3 services"],
                "tenure_group": [">5 năm", "3-5 năm"],
                "recency_group": ["<3 tháng", "1-2 năm"],
                "app_count_group": ["1 app", None],
                "app_tenure_group": ["3-6 tháng", None],
                "app_recency_days": ["<7 ngày", None],
                "loyalty_points": ["400-850", "Đến 150"],
                "loyalty_tier": ["700-1.500", "Đến 250"],
                "telco_install_date": ["2022-06-11", None],
                "telco_cancel_date": ["2024-06-17", None],
                "telco_contract_status": ["Huy", None],
                "telco_inbound_count_180d": ["Không có", None],
                "telco_outbound_count_180d": ["Không có", None],
                "telco_ticket_count_180d": ["Không có", None],
                "telco_internet_usage_group": ["Trên 350GB/tháng", None],
                "telco_internet_trend_group": ["Tăng", None],
                "telco_monetary_group": ["1M - 2M", None],
                "healthcare_last_order_date": ["1-3 tháng", None],
                "healthcare_order_count_6m": ["1 đơn", None],
                "healthcare_spend_6m": ["200-500K", None],
                "healthcare_aov_6m": ["100-250K", None],
                "healthcare_repeat_purchase_rate_6m": ["Không có", None],
                "healthcare_vaccine_visit_count_12m": ["Không có", None],
            }
        )

        out = extract_features(raw_df)

        self.assertEqual(len(out), 2)
        self.assertIn("user_id", out.columns)
        self.assertIn("age_midpoint", out.columns)
        self.assertIn("income_est_mid", out.columns)
        self.assertIn("healthcare_spend_to_income_ratio", out.columns)
        self.assertIn("telco_monetary_to_income_ratio", out.columns)
        self.assertIn("active_domains_count_calc", out.columns)
        self.assertIn("churn_risk_flag", out.columns)

        # Assert user_1 values
        self.assertEqual(out.loc[0, "is_male"], 1)
        self.assertEqual(out.loc[0, "is_metro_city"], 1)
        self.assertEqual(out.loc[0, "income_est_mid"], 25e6)
        self.assertGreater(out.loc[0, "healthcare_spend_to_income_ratio"], 0.0)


class TestLocalDataPipeline(unittest.TestCase):
    """Test LocalDataPipeline end-to-end execution."""

    def test_pipeline_run_on_sample(self) -> None:
        excel_path = Path("datasets/raw/local_data/DC5xQACI- Data sample_v1.0.xlsx")
        if not excel_path.is_file():
            self.skipTest(f"Sample file not found at {excel_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = LocalDataPipeline(excel_path=excel_path, output_dir=tmp_dir)
            df_features, df_training, parquet_path, csv_path, train_parquet_path, train_csv_path, summary_path = pipeline.run()

            self.assertEqual(len(df_features), 10)
            self.assertEqual(len(df_training), 10)
            self.assertTrue(parquet_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertTrue(train_parquet_path.is_file())
            self.assertTrue(train_csv_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertGreater(len(df_features.columns), len(df_training.columns))

    def test_run_local_eda(self) -> None:
        excel_path = Path("datasets/raw/local_data/DC5xQACI- Data sample_v1.0.xlsx")
        if not excel_path.is_file():
            self.skipTest(f"Sample file not found at {excel_path}")

        from src.local_data_pipeline import run_local_eda

        with tempfile.TemporaryDirectory() as tmp_dir:
            df_eda, raw_csv, proc_csv, train_csv = run_local_eda(excel_path=excel_path, output_dir=tmp_dir)
            self.assertEqual(len(df_eda), 41)
            self.assertTrue(raw_csv.is_file())
            self.assertTrue(proc_csv.is_file())
            self.assertTrue(train_csv.is_file())
            self.assertIn("unique_values", df_eda.columns)


if __name__ == "__main__":
    unittest.main()
