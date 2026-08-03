"""Memory-aware HCDR auxiliary-table aggregation with DuckDB."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
import pandas as pd

from .data import ID_COLUMN, resolve_table_path

QueryBuilder = Callable[[Path], str]


def _quoted(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _connection(cache_dir: Path) -> duckdb.DuckDBPyConnection:
    cache_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads = 8")
    connection.execute("SET memory_limit = '6GB'")
    connection.execute(f"SET temp_directory = '{_quoted(cache_dir)}'")
    return connection


def _csv(path: Path) -> str:
    return f"read_csv_auto('{_quoted(path)}', header=true, sample_size=100000)"


def _run(
    raw_dir: Path,
    cache_dir: Path,
    name: str,
    query: str,
) -> pd.DataFrame:
    output_path = cache_dir / f"{name}.parquet"
    if not output_path.is_file():
        connection = _connection(cache_dir / "duckdb_tmp")
        try:
            connection.execute(
                f"COPY ({query}) TO '{_quoted(output_path)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            connection.close()
    frame = pd.read_parquet(output_path)
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{name} aggregation produced duplicate {ID_COLUMN}")
    return frame


def bureau_agg(raw_dir: Path, cache_dir: Path) -> pd.DataFrame:
    path = resolve_table_path(raw_dir, "bureau")
    query = f"""
        SELECT SK_ID_CURR,
               count(*) AS BUREAU_APP_CNT,
               avg(AMT_CREDIT_SUM) AS BUREAU_AMT_CREDIT_SUM_MEAN,
               max(AMT_CREDIT_SUM) AS BUREAU_AMT_CREDIT_SUM_MAX,
               sum(AMT_CREDIT_SUM) AS BUREAU_AMT_CREDIT_SUM_SUM,
               avg(AMT_CREDIT_SUM_DEBT) AS BUREAU_AMT_DEBT_MEAN,
               sum(AMT_CREDIT_SUM_DEBT) AS BUREAU_AMT_DEBT_SUM,
               avg(DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MEAN,
               min(DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MIN,
               sum(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)
                   / count(*)::DOUBLE AS BUREAU_ACTIVE_RATIO,
               sum(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)
                   / count(*)::DOUBLE AS BUREAU_CLOSED_RATIO
        FROM {_csv(path)}
        GROUP BY SK_ID_CURR
    """
    return _run(raw_dir, cache_dir, "bureau", query)


def bureau_balance_agg(raw_dir: Path, cache_dir: Path) -> pd.DataFrame:
    balance = resolve_table_path(raw_dir, "bureau_balance")
    bureau = resolve_table_path(raw_dir, "bureau")
    query = f"""
        WITH by_loan AS (
            SELECT SK_ID_BUREAU,
                   min(MONTHS_BALANCE) AS MONTHS_MIN,
                   max(MONTHS_BALANCE) AS MONTHS_MAX,
                   count(*) AS MONTHS_SIZE,
                   avg(CASE WHEN STATUS = '0' THEN 1.0 ELSE 0.0 END)
                       AS STATUS_0_RATIO,
                   avg(CASE WHEN STATUS IN ('1','2','3','4','5')
                            THEN 1.0 ELSE 0.0 END) AS STATUS_DPD_RATIO,
                   avg(CASE WHEN STATUS = 'C' THEN 1.0 ELSE 0.0 END)
                       AS STATUS_C_RATIO
            FROM {_csv(balance)}
            GROUP BY SK_ID_BUREAU
        )
        SELECT b.SK_ID_CURR,
               avg(l.MONTHS_MIN) AS BB_MONTHS_MIN_MEAN,
               min(l.MONTHS_MIN) AS BB_MONTHS_MIN_MIN,
               max(l.MONTHS_MAX) AS BB_MONTHS_MAX_MAX,
               sum(l.MONTHS_SIZE) AS BB_MONTHS_SIZE_SUM,
               avg(l.STATUS_0_RATIO) AS BB_STATUS_0_RATIO_MEAN,
               avg(l.STATUS_DPD_RATIO) AS BB_STATUS_DPD_RATIO_MEAN,
               avg(l.STATUS_C_RATIO) AS BB_STATUS_C_RATIO_MEAN
        FROM by_loan l
        JOIN {_csv(bureau)} b USING (SK_ID_BUREAU)
        GROUP BY b.SK_ID_CURR
    """
    return _run(raw_dir, cache_dir, "bureau_balance", query)


def previous_agg(raw_dir: Path, cache_dir: Path) -> pd.DataFrame:
    path = resolve_table_path(raw_dir, "previous_application")
    query = f"""
        SELECT SK_ID_CURR,
               count(*) AS PREV_APP_CNT,
               avg(AMT_APPLICATION) AS PREV_AMT_APPLICATION_MEAN,
               max(AMT_APPLICATION) AS PREV_AMT_APPLICATION_MAX,
               min(AMT_APPLICATION) AS PREV_AMT_APPLICATION_MIN,
               avg(AMT_CREDIT) AS PREV_AMT_CREDIT_MEAN,
               max(AMT_CREDIT) AS PREV_AMT_CREDIT_MAX,
               avg(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'
                        THEN 1.0 ELSE 0.0 END) AS PREV_REFUSED_RATIO,
               avg(CASE WHEN NAME_CONTRACT_STATUS = 'Approved'
                        THEN 1.0 ELSE 0.0 END) AS PREV_APPROVED_RATIO
        FROM {_csv(path)}
        GROUP BY SK_ID_CURR
    """
    return _run(raw_dir, cache_dir, "previous", query)


def pos_agg(raw_dir: Path, cache_dir: Path) -> pd.DataFrame:
    path = resolve_table_path(raw_dir, "pos_cash_balance")
    query = f"""
        SELECT SK_ID_CURR,
               count(*) AS POS_ROW_CNT,
               count(DISTINCT SK_ID_PREV) AS POS_PREV_CNT,
               avg(SK_DPD) AS POS_DPD_MEAN,
               max(SK_DPD) AS POS_DPD_MAX,
               avg(SK_DPD_DEF) AS POS_DPD_DEF_MEAN,
               max(SK_DPD_DEF) AS POS_DPD_DEF_MAX
        FROM {_csv(path)}
        GROUP BY SK_ID_CURR
    """
    return _run(raw_dir, cache_dir, "pos", query)


def installments_agg(raw_dir: Path, cache_dir: Path) -> pd.DataFrame:
    path = resolve_table_path(raw_dir, "installments_payments")
    query = f"""
        SELECT SK_ID_CURR,
               count(*) AS INS_ROW_CNT,
               count(DISTINCT SK_ID_PREV) AS INS_PREV_CNT,
               avg(greatest(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))
                   AS INS_DPD_MEAN,
               max(greatest(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))
                   AS INS_DPD_MAX,
               avg(greatest(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT, 0))
                   AS INS_DBD_MEAN,
               avg(AMT_PAYMENT / nullif(AMT_INSTALMENT, 0))
                   AS INS_PAYMENT_PERC_MEAN,
               min(AMT_PAYMENT / nullif(AMT_INSTALMENT, 0))
                   AS INS_PAYMENT_PERC_MIN,
               avg(AMT_INSTALMENT - AMT_PAYMENT) AS INS_PAYMENT_DIFF_MEAN,
               sum(AMT_INSTALMENT - AMT_PAYMENT) AS INS_PAYMENT_DIFF_SUM
        FROM {_csv(path)}
        GROUP BY SK_ID_CURR
    """
    return _run(raw_dir, cache_dir, "installments", query)


def credit_card_agg(raw_dir: Path, cache_dir: Path) -> pd.DataFrame:
    path = resolve_table_path(raw_dir, "credit_card_balance")
    query = f"""
        SELECT SK_ID_CURR,
               count(*) AS CC_ROW_CNT,
               count(DISTINCT SK_ID_PREV) AS CC_PREV_CNT,
               avg(AMT_BALANCE) AS CC_AMT_BALANCE_MEAN,
               max(AMT_BALANCE) AS CC_AMT_BALANCE_MAX,
               avg(AMT_BALANCE / nullif(AMT_CREDIT_LIMIT_ACTUAL, 0))
                   AS CC_UTILIZATION_MEAN,
               max(AMT_BALANCE / nullif(AMT_CREDIT_LIMIT_ACTUAL, 0))
                   AS CC_UTILIZATION_MAX,
               avg(SK_DPD) AS CC_DPD_MEAN,
               max(SK_DPD) AS CC_DPD_MAX
        FROM {_csv(path)}
        GROUP BY SK_ID_CURR
    """
    return _run(raw_dir, cache_dir, "credit_card", query)


def build_feature_matrix(
    application: pd.DataFrame,
    raw_dir: Path,
    cache_dir: Path,
    *,
    level: str = "C",
) -> pd.DataFrame:
    """Build staged A/B/C feature matrices without changing application rows."""
    if level not in {"A", "B", "C"}:
        raise ValueError("level must be one of A, B, or C")
    matrix = application.copy()
    expected_rows = len(matrix)
    aggregations = []
    if level in {"B", "C"}:
        aggregations.extend([bureau_agg, previous_agg])
    if level == "C":
        aggregations.extend(
            [
                bureau_balance_agg,
                pos_agg,
                installments_agg,
                credit_card_agg,
            ]
        )

    for aggregation in aggregations:
        features = aggregation(raw_dir, cache_dir)
        matrix = matrix.merge(
            features,
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
        if len(matrix) != expected_rows:
            raise AssertionError(
                f"{aggregation.__name__} changed row count "
                f"from {expected_rows} to {len(matrix)}"
            )
    return matrix
