"""
Shared `exports/quotes.csv` helpers and akshare/DataFrame row mapping.

Used by fetch_stocks.py (no PostgreSQL).
"""

from __future__ import annotations

import pandas as pd

BATCH = 2000

HEADER_KEYS = {
    "日期": "trade_date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change_amount",
    "换手率": "turnover",
}


def ensure_table(cur=None) -> None:
    """Compatibility name: ensure quotes CSV exists with header."""
    from csv_io import ensure_quotes_csv_header

    ensure_quotes_csv_header()


def _coerce_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.replace("%", "").replace(",", "").strip()
        if not v:
            return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _coerce_volume(v) -> int | None:
    f = _coerce_float(v)
    if f is None:
        return None
    return int(round(f))


def dataframe_to_quote_rows(df: pd.DataFrame, symbol_override: str | None = None) -> list[tuple]:
    """
    Map a DataFrame with Chinese akshare-style headers to upsert tuples.
    If symbol_override is set, every row uses it; otherwise column `symbol` must exist.
    """
    if df is None or df.empty:
        return []
    df = df.copy()
    rename = {}
    for c in df.columns:
        key = str(c).strip()
        if key in HEADER_KEYS:
            rename[c] = HEADER_KEYS[key]
    df = df.rename(columns=rename)
    if "trade_date" not in df.columns:
        return []
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[df["trade_date"].notna()]
    if df.empty:
        return []

    rows: list[tuple] = []
    for _, r in df.iterrows():
        if symbol_override is not None:
            sym = str(symbol_override).strip()
        else:
            if "symbol" not in df.columns:
                return []
            raw = r.get("symbol")
            if pd.isna(raw):
                continue
            sym = str(raw).strip()
        if not sym or sym == "nan":
            continue
        td = r["trade_date"].date()
        rows.append(
            (
                sym,
                td,
                _coerce_float(r.get("open")),
                _coerce_float(r.get("close")),
                _coerce_float(r.get("high")),
                _coerce_float(r.get("low")),
                _coerce_volume(r.get("volume")),
                _coerce_float(r.get("amount")),
                _coerce_float(r.get("amplitude")),
                _coerce_float(r.get("pct_change")),
                _coerce_float(r.get("change_amount")),
                _coerce_float(r.get("turnover")),
            )
        )
    return rows
