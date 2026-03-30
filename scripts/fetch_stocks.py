"""
Fetch daily quotes via akshare and merge into exports/quotes.csv.

Checkpoint: checkpoint/fetch_stocks_checkpoint.json — per-symbol last trade_date (ISO).
Use --add to ignore checkpoint and fetch from DEFAULT_START_DATE.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from csv_io import upsert_quotes_rows
from quotes_db import dataframe_to_quote_rows, ensure_table
from stock_universe import load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_FILE = CHECKPOINT_DIR / "fetch_stocks_checkpoint.json"

DEFAULT_START_DATE = "20250101"  # used only when no checkpoint entry exists
END_DATE = date.today().strftime("%Y%m%d")

SECTORS = load_sectors()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch daily quotes via akshare into exports/quotes.csv.",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoint and fetch from DEFAULT_START_DATE",
    )
    args = parser.parse_args()

    ckpt = {} if args.add else load_checkpoint()

    ensure_table()

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
                df = fetch_stock(symbol, market, start, end)
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


if __name__ == "__main__":
    main()
