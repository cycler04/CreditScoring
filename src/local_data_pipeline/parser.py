"""Parser utilities for range text, currency, and date strings in local data."""

from __future__ import annotations

import re
from typing import Any, Tuple


def clean_column_name(col: str) -> str:
    """Clean column name by stripping whitespace and newlines."""
    if not isinstance(col, str):
        return str(col)
    return col.strip()


def parse_numeric_count(val: Any) -> float:
    """Parse count fields such as 'Không có', '1 đơn', '1 lần', '4-6 đơn', 'Trên 6 đơn'."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return 0.0
    s = str(val).strip()
    if not s or s.lower() in ("không có", "nan", "none", "khong co"):
        return 0.0
    
    # Check range like '4-6 đơn'
    m_range = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if m_range:
        return (float(m_range.group(1)) + float(m_range.group(2))) / 2.0
    
    # Check 'trên X'
    m_above = re.search(r"trên\s*(\d+)", s, re.IGNORECASE)
    if m_above:
        return float(m_above.group(1)) + 1.0
    
    # Extract first digit sequence
    m_num = re.search(r"(\d+)", s)
    if m_num:
        return float(m_num.group(1))
    
    return 0.0


def parse_income_band(val: Any) -> Tuple[float, float, float]:
    """Parse income band string into (min, max, mid) in VND."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return 0.0, 0.0, 0.0
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return 0.0, 0.0, 0.0

    # Pattern: 'Trên 18trđ đến 32trđ', 'Trên 5trđ đến 10trđ'
    m = re.search(r"trên\s*(\d+)\s*trđ\s*đến\s*(\d+)\s*trđ", s, re.IGNORECASE)
    if m:
        low = float(m.group(1)) * 1e6
        high = float(m.group(2)) * 1e6
        return low, high, (low + high) / 2.0
    
    m_under = re.search(r"dưới\s*(\d+)\s*trđ", s, re.IGNORECASE)
    if m_under:
        high = float(m_under.group(1)) * 1e6
        return 0.0, high, high / 2.0

    m_over = re.search(r"trên\s*(\d+)\s*trđ", s, re.IGNORECASE)
    if m_over:
        low = float(m_over.group(1)) * 1e6
        return low, low * 1.5, low * 1.25

    return 0.0, 0.0, 0.0


def parse_loyalty_points(val: Any) -> Tuple[float, float, float]:
    """Parse loyalty points string into (min, max, mid)."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return 0.0, 0.0, 0.0
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return 0.0, 0.0, 0.0

    if "Đến" in s or "dưới" in s.lower():
        m = re.search(r"(\d+[\.\d+]*)", s)
        high = float(m.group(1).replace(".", "")) if m else 150.0
        return 0.0, high, high / 2.0

    if "Trên" in s or "trên" in s.lower():
        m = re.search(r"(\d+[\.\d+]*)", s)
        low = float(m.group(1).replace(".", "")) if m else 850.0
        return low, low * 1.5, low * 1.25

    m = re.search(r"(\d+[\.\d+]*)\s*-\s*(\d+[\.\d+]*)", s)
    if m:
        low = float(m.group(1).replace(".", ""))
        high = float(m.group(2).replace(".", ""))
        return low, high, (low + high) / 2.0

    return 0.0, 0.0, 0.0


def parse_telco_monetary(val: Any) -> Tuple[float, float, float]:
    """Parse telco monetary group like '1M - 2M' into VND (min, max, mid)."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return 0.0, 0.0, 0.0
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return 0.0, 0.0, 0.0

    m = re.search(r"(\d+)\s*M\s*-\s*(\d+)\s*M", s, re.IGNORECASE)
    if m:
        low = float(m.group(1)) * 1e6
        high = float(m.group(2)) * 1e6
        return low, high, (low + high) / 2.0

    m_over = re.search(r"trên\s*(\d+)\s*M", s, re.IGNORECASE)
    if m_over:
        low = float(m_over.group(1)) * 1e6
        return low, low * 1.5, low * 1.25

    return 0.0, 0.0, 0.0


def parse_currency_amount(val: Any) -> Tuple[float, float, float]:
    """Parse spend / AOV values such as '200-500K', '500K-1,5tr', 'Trên 1,5tr', 'Đến 100K'."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return 0.0, 0.0, 0.0
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "không có"):
        return 0.0, 0.0, 0.0

    def _unit_multiplier(u: str) -> float:
        u_lower = u.lower()
        if "k" in u_lower:
            return 1e3
        if "tr" in u_lower:
            return 1e6
        return 1.0

    if "Đến" in s or "dưới" in s.lower():
        m = re.search(r"(\d+[\,\.]?\d*)\s*([Kk]|tr|TR)", s)
        if m:
            val_num = float(m.group(1).replace(",", "."))
            high = val_num * _unit_multiplier(m.group(2))
            return 0.0, high, high / 2.0

    if "Trên" in s or "trên" in s.lower():
        m = re.search(r"(\d+[\,\.]?\d*)\s*([Kk]|tr|TR)", s)
        if m:
            val_num = float(m.group(1).replace(",", "."))
            low = val_num * _unit_multiplier(m.group(2))
            return low, low * 1.5, low * 1.25

    # Range like '200-500K' or '500K-1,5tr'
    m = re.search(r"(\d+[\,\.]?\d*)\s*([Kk]|tr)?\s*-\s*(\d+[\,\.]?\d*)\s*([Kk]|tr)", s)
    if m:
        val1 = float(m.group(1).replace(",", "."))
        u1 = m.group(2) if m.group(2) else m.group(4)
        val2 = float(m.group(3).replace(",", "."))
        u2 = m.group(4)
        
        low = val1 * _unit_multiplier(u1)
        high = val2 * _unit_multiplier(u2)
        return low, high, (low + high) / 2.0

    return 0.0, 0.0, 0.0


def parse_internet_usage_gb(val: Any) -> Tuple[float, float, float]:
    """Parse internet usage strings like '120 - 220GB/tháng', 'Trên 350GB/tháng'."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return 0.0, 0.0, 0.0
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return 0.0, 0.0, 0.0

    m_over = re.search(r"trên\s*(\d+)\s*GB", s, re.IGNORECASE)
    if m_over:
        low = float(m_over.group(1))
        return low, low * 1.5, low * 1.25

    m_range = re.search(r"(\d+)\s*-\s*(\d+)\s*GB", s, re.IGNORECASE)
    if m_range:
        low = float(m_range.group(1))
        high = float(m_range.group(2))
        return low, high, (low + high) / 2.0

    return 0.0, 0.0, 0.0
