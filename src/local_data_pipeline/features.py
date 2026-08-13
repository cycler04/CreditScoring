"""Feature extraction functions for local credit scoring data sample."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parser import (
    clean_column_name,
    parse_currency_amount,
    parse_income_band,
    parse_internet_usage_gb,
    parse_loyalty_points,
    parse_numeric_count,
    parse_telco_monetary,
)


def extract_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transform raw multi-domain DataFrame into engineered feature set."""
    df = df_raw.copy()
    
    # 1. Standardize column names
    df.columns = [clean_column_name(c) for c in df.columns]
    
    out = pd.DataFrame(index=df.index)
    
    # User Identifier
    user_col = [c for c in df.columns if "user_id" in c]
    if user_col:
        out["user_id"] = df[user_col[0]].astype(str)
    else:
        out["user_id"] = [f"id_{i}" for i in range(len(df))]

    # -------------------------------------------------------------------------
    # 2. Customer Profile Features
    # -------------------------------------------------------------------------
    if "age_group" in df.columns:
        # Age group midpoints
        age_map = {
            "18-23": 20.5,
            "23-30": 26.5,
            "31-40": 35.5,
            "41-50": 45.5,
            "51-60": 55.5,
            "60++": 65.0,
        }
        out["age_group_raw"] = df["age_group"].fillna("Unknown")
        out["age_midpoint"] = df["age_group"].map(age_map).fillna(35.0).astype(float)
        out["is_senior"] = (out["age_midpoint"] >= 60.0).astype(int)
        out["is_young"] = (out["age_midpoint"] <= 23.0).astype(int)

    if "gender" in df.columns:
        out["gender_raw"] = df["gender"].fillna("Unknown")
        out["is_male"] = (df["gender"].astype(str).str.lower() == "nam").astype(int)

    if "city" in df.columns:
        out["city_raw"] = df["city"].fillna("Unknown")
        out["is_metro_city"] = (
            df["city"].astype(str).str.contains("Hà Nội|Hồ Chí Minh", regex=True, case=False).astype(int)
        )

    if "household_type" in df.columns:
        out["household_type_raw"] = df["household_type"].fillna("Unknown")
        out["is_company_household"] = (
            df["household_type"].astype(str).str.lower() == "công ty"
        ).astype(int)

    if "income_band_est" in df.columns:
        incomes = df["income_band_est"].apply(parse_income_band)
        out["income_est_min"] = [inc[0] for inc in incomes]
        out["income_est_max"] = [inc[1] for inc in incomes]
        out["income_est_mid"] = [inc[2] for inc in incomes]
        out["has_income_info"] = (out["income_est_mid"] > 0).astype(int)

    if "active_domain_count" in df.columns:
        domain_cnt_map = {
            "1 service": 1,
            "2-3 services": 2.5,
            "4+ services": 4.5,
        }
        out["active_domain_count_est"] = (
            df["active_domain_count"].map(domain_cnt_map).fillna(0.0).astype(float)
        )

    if "tenure_group" in df.columns:
        tenure_map = {
            "<1 năm": 0.5,
            "1-3 năm": 2.0,
            "3-5 năm": 4.0,
            ">5 năm": 6.5,
        }
        out["tenure_years_est"] = df["tenure_group"].map(tenure_map).fillna(0.0).astype(float)

    if "recency_group" in df.columns:
        recency_months_map = {
            "<3 tháng": 1.5,
            "3-6 tháng": 4.5,
            "6-12 tháng": 9.0,
            "1-2 năm": 18.0,
            "2-3 năm": 30.0,
            "3-5 năm": 48.0,
        }
        out["recency_months_est"] = df["recency_group"].map(recency_months_map).fillna(60.0).astype(float)

    # App usage features
    if "app_count_group" in df.columns:
        out["has_app"] = df["app_count_group"].notna().astype(int)
        out["app_count_est"] = df["app_count_group"].apply(parse_numeric_count)

    if "app_tenure_group" in df.columns:
        app_tenure_map = {
            "<3 tháng": 1.5,
            "3-6 tháng": 4.5,
            "6-12 tháng": 9.0,
            ">1 năm": 15.0,
        }
        out["app_tenure_months_est"] = (
            df["app_tenure_group"].map(app_tenure_map).fillna(0.0).astype(float)
        )

    if "app_recency_days" in df.columns:
        app_recency_map = {
            "<7 ngày": 3.5,
            "7-30 ngày": 18.5,
            "31-90 ngày": 60.5,
            ">90 ngày": 135.0,
        }
        out["app_recency_days_est"] = (
            df["app_recency_days"].map(app_recency_map).fillna(180.0).astype(float)
        )

    # Loyalty points & tier
    if "loyalty_points" in df.columns:
        l_pts = df["loyalty_points"].apply(parse_loyalty_points)
        out["loyalty_points_min"] = [lp[0] for lp in l_pts]
        out["loyalty_points_max"] = [lp[1] for lp in l_pts]
        out["loyalty_points_mid"] = [lp[2] for lp in l_pts]
        out["has_loyalty_pts"] = df["loyalty_points"].notna().astype(int)

    if "loyalty_tier" in df.columns:
        l_tier = df["loyalty_tier"].apply(parse_loyalty_points)
        out["loyalty_tier_pts_mid"] = [lt[2] for lt in l_tier]

    # -------------------------------------------------------------------------
    # 3. Telco Features
    # -------------------------------------------------------------------------
    if "telco_install_date" in df.columns:
        telco_install = pd.to_datetime(df["telco_install_date"], errors="coerce")
        out["has_telco"] = telco_install.notna().astype(int)
        
        # Fixed reference date for reproducibility (or T0 max date)
        ref_date = pd.Timestamp("2024-07-01")
        out["telco_install_age_days"] = (ref_date - telco_install).dt.days.fillna(0.0)
        out["telco_install_year"] = telco_install.dt.year.fillna(0.0).astype(int)

    if "telco_cancel_date" in df.columns:
        telco_cancel = pd.to_datetime(df["telco_cancel_date"], errors="coerce")
        out["telco_is_cancelled"] = telco_cancel.notna().astype(int)
        
        if "telco_install_date" in df.columns:
            telco_install = pd.to_datetime(df["telco_install_date"], errors="coerce")
            out["telco_contract_active_days"] = (
                (telco_cancel - telco_install).dt.days.fillna(out.get("telco_install_age_days", 0.0))
            )

    if "telco_contract_status" in df.columns:
        out["telco_contract_status_raw"] = df["telco_contract_status"].fillna("Active")

    if "telco_inbound_count_180d" in df.columns:
        out["telco_inbound_cnt_180d"] = df["telco_inbound_count_180d"].apply(parse_numeric_count)

    if "telco_outbound_count_180d" in df.columns:
        out["telco_outbound_cnt_180d"] = df["telco_outbound_count_180d"].apply(parse_numeric_count)

    if "telco_ticket_count_180d" in df.columns:
        out["telco_ticket_cnt_180d"] = df["telco_ticket_count_180d"].apply(parse_numeric_count)

    if "telco_internet_usage_group" in df.columns:
        usage = df["telco_internet_usage_group"].apply(parse_internet_usage_gb)
        out["telco_gb_usage_mid"] = [u[2] for u in usage]

    if "telco_internet_trend_group" in df.columns:
        trend_map = {"Tăng": 1.0, "Ổn định": 0.0, "Giảm": -1.0}
        out["telco_internet_trend_score"] = (
            df["telco_internet_trend_group"].map(trend_map).fillna(0.0).astype(float)
        )

    if "telco_monetary_group" in df.columns:
        monetary = df["telco_monetary_group"].apply(parse_telco_monetary)
        out["telco_monetary_mid"] = [m[2] for m in monetary]

    # -------------------------------------------------------------------------
    # 4. Healthcare / Pharmacy Features
    # -------------------------------------------------------------------------
    if "healthcare_last_order_date" in df.columns:
        out["has_healthcare"] = df["healthcare_last_order_date"].notna().astype(int)

    if "healthcare_order_count_6m" in df.columns:
        out["healthcare_order_cnt_6m"] = df["healthcare_order_count_6m"].apply(parse_numeric_count)

    if "healthcare_spend_6m" in df.columns:
        spend = df["healthcare_spend_6m"].apply(parse_currency_amount)
        out["healthcare_spend_6m_mid"] = [s[2] for s in spend]

    if "healthcare_aov_6m" in df.columns:
        aov = df["healthcare_aov_6m"].apply(parse_currency_amount)
        out["healthcare_aov_6m_mid"] = [a[2] for a in aov]

    if "healthcare_repeat_purchase_rate_6m" in df.columns:
        out["healthcare_repeat_rate_6m"] = (
            df["healthcare_repeat_purchase_rate_6m"].apply(parse_numeric_count)
        )

    if "healthcare_vaccine_visit_count_12m" in df.columns:
        out["healthcare_vaccine_visits_12m"] = (
            df["healthcare_vaccine_visit_count_12m"].apply(parse_numeric_count)
        )

    # -------------------------------------------------------------------------
    # 5. Retail Features
    # -------------------------------------------------------------------------
    if "retail_order_count_12m" in df.columns:
        out["has_retail"] = df["retail_order_count_12m"].notna().astype(int)
        out["retail_order_cnt_12m"] = df["retail_order_count_12m"].fillna(0.0)
    else:
        out["has_retail"] = 0
        out["retail_order_cnt_12m"] = 0.0

    if "retail_gmv_12m" in df.columns:
        out["retail_gmv_12m"] = df["retail_gmv_12m"].fillna(0.0)
    else:
        out["retail_gmv_12m"] = 0.0

    if "retail_aov_12m" in df.columns:
        out["retail_aov_12m"] = df["retail_aov_12m"].fillna(0.0)
    else:
        out["retail_aov_12m"] = 0.0

    # -------------------------------------------------------------------------
    # 6. Cross-Domain Ratios & Interaction Features
    # -------------------------------------------------------------------------
    eps = 1.0  # Safe division epsilon

    # 6.1 Healthcare spend to estimated income ratio
    if "healthcare_spend_6m_mid" in out.columns and "income_est_mid" in out.columns:
        # Annualized income = income_est_mid * 6 for 6 months
        out["healthcare_spend_to_income_ratio"] = (
            out["healthcare_spend_6m_mid"] / ((out["income_est_mid"] * 6.0) + eps)
        )

    # 6.2 Telco monetary payment to monthly income ratio
    if "telco_monetary_mid" in out.columns and "income_est_mid" in out.columns:
        out["telco_monetary_to_income_ratio"] = (
            out["telco_monetary_mid"] / (out["income_est_mid"] + eps)
        )

    # 6.3 Healthcare AOV to Total 6M Spend ratio
    if "healthcare_aov_6m_mid" in out.columns and "healthcare_spend_6m_mid" in out.columns:
        out["healthcare_aov_to_spend_ratio"] = (
            out["healthcare_aov_6m_mid"] / (out["healthcare_spend_6m_mid"] + eps)
        )

    # 6.4 App recency to app tenure ratio
    if "app_recency_days_est" in out.columns and "app_tenure_months_est" in out.columns:
        app_tenure_days = out["app_tenure_months_est"] * 30.0
        out["app_recency_to_tenure_ratio"] = (
            out["app_recency_days_est"] / (app_tenure_days + eps)
        )

    # 6.5 Domain breadth & engagement index
    domain_flags = []
    for col_flag in ["has_app", "has_telco", "has_healthcare", "has_retail", "has_loyalty_pts"]:
        if col_flag in out.columns:
            domain_flags.append(out[col_flag])
    
    if domain_flags:
        out["active_domains_count_calc"] = sum(domain_flags)
        out["domain_breadth_score"] = out["active_domains_count_calc"] / len(domain_flags)
    else:
        out["active_domains_count_calc"] = 0
        out["domain_breadth_score"] = 0.0

    # 6.6 Churn risk & High value customer indicators
    is_cancel = out.get("telco_is_cancelled", pd.Series(0, index=out.index))
    app_inactive = (out.get("app_recency_days_est", pd.Series(0, index=out.index)) > 90.0).astype(int)
    out["churn_risk_flag"] = ((is_cancel == 1) | (app_inactive == 1)).astype(int)

    high_inc = (out.get("income_est_mid", pd.Series(0, index=out.index)) >= 18e6).astype(int)
    high_loyalty = (out.get("loyalty_points_mid", pd.Series(0, index=out.index)) >= 800.0).astype(int)
    out["is_high_value_customer"] = ((high_inc == 1) & (high_loyalty == 1)).astype(int)

    return out


def select_training_features(
    df_features: pd.DataFrame,
    include_id: bool = True,
    drop_zero_variance: bool = True,
    drop_raw_strings: bool = True,
) -> pd.DataFrame:
    """Filter DataFrame to retain only clean, numerical/binary features valuable for model training."""
    train_df = df_features.copy()

    if drop_raw_strings:
        raw_string_cols = [c for c in train_df.columns if c.endswith("_raw")]
        train_df = train_df.drop(columns=[c for c in raw_string_cols if c in train_df.columns])

    if drop_zero_variance:
        zero_var_cols = [
            c for c in train_df.columns
            if c != "user_id" and train_df[c].nunique(dropna=False) <= 1
        ]
        train_df = train_df.drop(columns=zero_var_cols)

    if not include_id and "user_id" in train_df.columns:
        train_df = train_df.drop(columns=["user_id"])

    return train_df
