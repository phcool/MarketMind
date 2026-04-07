"""
Fetch daily quotes via akshare and merge into exports/quotes.csv.

Checkpoint: checkpoint/fetch_stocks_checkpoint.json — per-symbol last trade_date (ISO).
Use --add to ignore checkpoint and fetch from DEFAULT_START_DATE.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from csv_io import upsert_quotes_rows
from stock_universe import load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_FILE = CHECKPOINT_DIR / "fetch_stocks_checkpoint.json"

DEFAULT_START_DATE = "20250101"  # used only when no checkpoint entry exists
END_DATE = date.today().strftime("%Y%m%d")

SECTORS = load_sectors()


def _pick_column(df: pd.DataFrame, *candidates: str) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _series_or_none(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    col = _pick_column(df, *candidates)
    if col is None:
        return None
    return df[col]


def dataframe_to_quote_rows(df: pd.DataFrame, symbol_override: str | None = None) -> list[tuple]:
    """
    Map akshare daily K-line dataframe to rows for exports/quotes.csv.

    Output tuple schema:
      (symbol, trade_date, open, close, high, low, volume, amount,
       amplitude, pct_change, change_amount, turnover)
    """
    if df is None or df.empty:
        return []

    date_col = _pick_column(df, "日期", "date", "trade_date")
    open_col = _pick_column(df, "开盘", "open")
    close_col = _pick_column(df, "收盘", "close")
    high_col = _pick_column(df, "最高", "high")
    low_col = _pick_column(df, "最低", "low")
    volume_col = _pick_column(df, "成交量", "volume")
    amount_col = _pick_column(df, "成交额", "amount")
    amplitude_col = _pick_column(df, "振幅", "amplitude")
    pct_change_col = _pick_column(df, "涨跌幅", "pct_change")
    change_amount_col = _pick_column(df, "涨跌额", "change_amount")
    turnover_col = _pick_column(df, "换手率", "turnover")

    required = {
        "date": date_col,
        "open": open_col,
        "close": close_col,
        "high": high_col,
        "low": low_col,
        "volume": volume_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; available columns={list(df.columns)}")

    amount_series = _series_or_none(df, "成交额", "amount")
    amplitude_series = _series_or_none(df, "振幅", "amplitude")
    pct_change_series = _series_or_none(df, "涨跌幅", "pct_change")
    change_amount_series = _series_or_none(df, "涨跌额", "change_amount")
    turnover_series = _series_or_none(df, "换手率", "turnover")
    symbol_series = _series_or_none(df, "股票代码", "代码", "symbol")

    rows: list[tuple] = []
    for i, raw_date in enumerate(df[date_col]):
        trade_date = pd.to_datetime(raw_date).date().isoformat()
        symbol = symbol_override or (
            str(symbol_series.iloc[i]).strip() if symbol_series is not None else ""
        )
        rows.append(
            (
                symbol,
                trade_date,
                df[open_col].iloc[i],
                df[close_col].iloc[i],
                df[high_col].iloc[i],
                df[low_col].iloc[i],
                df[volume_col].iloc[i],
                amount_series.iloc[i] if amount_series is not None else None,
                amplitude_series.iloc[i] if amplitude_series is not None else None,
                pct_change_series.iloc[i] if pct_change_series is not None else None,
                change_amount_series.iloc[i] if change_amount_series is not None else None,
                turnover_series.iloc[i] if turnover_series is not None else None,
            )
        )
    return rows


def load_checkpoint() -> dict[str, str]:
    """Return {symbol: latest_date_str} e.g. {'300750': '2026-03-24'}."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_FILE.exists():
        return {}
    text = CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def save_checkpoint(ckpt: dict[str, str]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fetch_stock(symbol: str, market: str, start: str, end: str) -> pd.DataFrame:
    if market == "a":
        return ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="",
        )
    return ak.stock_hk_hist(
        symbol=symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="",
    )


def fetch_stock_with_retry(
    symbol: str,
    market: str,
    start: str,
    end: str,
    *,
    max_attempts: int,
    retry_base_delay: float,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_stock(symbol, market, start, end)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            wait_seconds = retry_base_delay * (2 ** (attempt - 1))
            print(
                f"attempt {attempt}/{max_attempts} failed: {exc}; retrying after {wait_seconds:.1f}s ...",
                end=" ",
                flush=True,
            )
            time.sleep(wait_seconds)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch daily quotes via akshare into exports/quotes.csv.",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoint and fetch from DEFAULT_START_DATE",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to sleep after each symbol fetch attempt (default: 1.0).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per symbol when AkShare fetch fails (default: 3).",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=3.0,
        help="Base seconds to wait before retry; doubles after each failure (default: 3.0).",
    )
    args = parser.parse_args()

    ckpt = {} if args.add else load_checkpoint()

    for _sector, companies in SECTORS.items():
        for name, symbol, market in companies:
            if symbol in ckpt and not args.add:
                last_date = date.fromisoformat(ckpt[symbol])
                start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
            else:
                start = DEFAULT_START_DATE

            end = END_DATE
            if start > end:
                print(
                    f"  {name} ({symbol}): already up to date ({ckpt.get(symbol)}), skip.",
                )
                continue

            print(f"Fetching {name} ({symbol})  {start} → {end} ...", end=" ", flush=True)
            try:
                df = fetch_stock_with_retry(
                    symbol,
                    market,
                    start,
                    end,
                    max_attempts=max(1, args.max_attempts),
                    retry_base_delay=max(0.0, args.retry_base_delay),
                )
                if df is None or df.empty:
                    print("no data returned, skipped.")
                    continue

                rows = dataframe_to_quote_rows(df, symbol_override=symbol)
                if not rows:
                    print(
                        "no rows mapped to quotes schema (check column names), skipped.",
                        f"columns={list(df.columns)}",
                    )
                    continue

                upsert_quotes_rows(rows)

                date_col = "日期" if "日期" in df.columns else df.columns[0]
                latest = pd.to_datetime(df[date_col]).max().date().isoformat()
                ckpt[symbol] = latest
                save_checkpoint(ckpt)

                print(f"merged {len(rows)} rows, latest={latest}.")
            except Exception as exc:
                print(f"ERROR – {exc}")
            finally:
                if args.delay > 0:
                    time.sleep(args.delay)


if __name__ == "__main__":
    main()
